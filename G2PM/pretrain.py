import time
import shutil

import yaml
import os.path as osp
import gc
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset

from data.pyg_data_loader import load_data, mol_graphs
from model.model import PretrainModel

from task.node import preprocess_node, pretrain_node, linear_probe_node
from task.link import preprocess_link, train_link, eval_link
from task.graph import preprocess_graph, pretrain_graph, linear_probe_graph, svm_probe_graph

# from utils.sys import set_memory_limit
from utils.args import get_pt_args
from utils.early_stop import EarlyStopping
from utils.scheduler import get_scheduler
from utils.logger import Logger
from utils.utils import seed_everything, check_path, get_num_params, to_millions

import wandb


def get_preprocess(params):
    task = params["task"]

    if task == "node":
        return preprocess_node
    elif task == "link":
        return preprocess_link
    elif task == "graph":
        return preprocess_graph
    else:
        raise ValueError("Does not support the task in preprocessing.")


def get_train(params):
    task = params["task"]

    if task == "node":
        return pretrain_node
    elif task == "graph":
        return pretrain_graph
    else:
        raise ValueError("Does not support the task in finetuning.")


def get_eval(params):
    task = params["task"]

    if task == "node":
        return linear_probe_node
    elif task == "graph":
        return linear_probe_graph if params['probe_mode'] == 'linear' else svm_probe_graph
    else:
        raise ValueError("Does not support the task in evaluation.")


def run(params):
    seed_everything(42)  # Make sure the split is the same for each run

    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['device'] = device
    print("Use Device:", device)

    # Helper function to get input and edge dimensions
    def get_input_edge_dims(g):
        if isinstance(g, Data):
            input_dim = g.x.size(1)
            edge_dim = g.edge_attr.size(1) if g.edge_attr is not None else 0
        elif isinstance(g, InMemoryDataset):
            input_dim = g._data.x_feat.size(1)
            edge_dim = g._data.e_feat.size(1) if g._data.edge_attr is not None else 0
        elif isinstance(g, dict):
            input_dim = g['train']._data.x_feat.size(1)
            edge_dim = g['train']._data.e_feat.size(1) if g['train']._data.edge_attr is not None else 0
        else:
            raise ValueError(f"Unsupported graph type: {type(g)}")
            
        if params['dataset'] in mol_graphs:
            input_dim = edge_dim = 100
            
        return input_dim, edge_dim
    
    def get_num_instances(g, params):
        if params['task'] == 'node':
            return g.num_nodes
        elif params['task'] == 'link':
            return g.num_edges
        elif params['task'] == 'graph':
            return len(g) if not isinstance(g, dict) else len(g['all'])
        else:
            raise ValueError(f"Unsupported graph type: {type(g)}")

    # Load and preprocess data
    graph_set, splits_set = load_data(params)
    for name, splits in splits_set.items():
        if splits is None:
            splits_set[name] = range(params['split_repeat'])
    data_config = params['data_config']

    # Set dimensions based on data
    if params['node_pe'] == 'none':
        params['node_pe_dim'] = 0
    # Generate a fixed train mask for pretraining, default is no mask
    params['train_mask'] = torch.rand(get_num_instances(graph, params)) < params['train_mask_ratio'] if params['train_mask_ratio'] > 0 else None

    params['input_dim'], params['edge_dim'] = {}, {}
    params['output_dim'] = {}
    params['num_tasks'] = {}
    steps_per_epoch = 0
    for name, graph in graph_set.items():
        input_dim, edge_dim = get_input_edge_dims(graph)
        params['input_dim'][name] = input_dim
        params['edge_dim'][name] = edge_dim
        num_tasks = data_config[name].get('num_tasks', None)
        params['num_tasks'][name] = num_tasks
        if num_tasks is not None:
            params['output_dim'][name] = num_tasks
        else:
            params['output_dim'][name] = graph.y.max().item() + 1
        steps_per_epoch += (get_num_instances(graph, params) + params['batch_size'] - 1) // params['batch_size']
    params['steps_per_epoch'] = steps_per_epoch

    # Get core functions
    preprocess_fn = get_preprocess(params)
    train_fn = get_train(params)
    eval_fn = get_eval(params)

    # Preprocess graph patterns
    start_time = time.time()
    pattern_set = {}
    for name, graph in graph_set.items():
        pattern_set[name] = preprocess_fn(graph, name, params)
    params['pattern_set'] = pattern_set
    preprocess_time = time.time() - start_time
    print(f"Preprocessing time: {preprocess_time:.2f}s")

    training_times = []
    inference_times = []
    logger = Logger(vq_mode=params['use_vq'])

    model = PretrainModel(params=params).to(device)
    num_params = to_millions(get_num_params(model))
    num_params_encoder = to_millions(get_num_params(model.encoder))
    print(f'The number of parameters: {num_params}M, Encoder: {num_params_encoder}M')
    # num_params_decoder = to_millions(get_num_params(model.decoder) + get_num_params(model.linear_decoder))
    # print(f'The number of parameters: {num_params}M, Encoder: {num_params_encoder}M, Decoder: {num_params_decoder}M')

    # After VQ-Pretrain, freeze tokenizer, vocab and decoder
    if not params['use_vq']:
        vq_path = osp.join(params['save_vq'], params['dataset'], f"{params['save_name']}.pt")
        model.load_state_dict(torch.load(vq_path, map_location=device))
        model.freeze_tokenizer()

    stopper = EarlyStopping(patience=params["early_stop"])

    # TODO: Sync settings for different datasets
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
        betas=(params['opt_beta1'], params['opt_beta2']), eps=params['opt_eps']
    )
    scheduler = get_scheduler(optimizer, params)

    for epoch in range(1, params['epochs'] + 1):
        start_time = time.time()
        total_loss = {'train': 0., 'val': 0., 'test': 0.}
        for name, graph in graph_set.items():
            loss = train_fn(graph, model, optimizer, name, scheduler=scheduler, params=params)
            total_loss['train'] += loss['train']
            total_loss['val'] += loss['val']
            total_loss['test'] += loss['test']
            # loss = {'train': 0, 'val': 0, 'test': 0}
        loss = total_loss
        training_time = time.time() - start_time
        training_times.append(training_time)

        if (epoch % params['eval_every'] == 0) and (params['split'] != 'pretrain'):
            if not params['use_vq']:
                start_time = time.time()
                result = []
                aggr_result = {'train': 0, 'train_std': 0, 
                                'val': 0, 'val_std': 0, 
                                'test': 0, 'test_std': 0, 
                                'metric': params['metric']}
                for name, graph in graph_set.items():
                    result.append(eval_fn(graph, model, splits_set[name], name, params=params))
                    result[-1]['dataset'] = name
                val_size, test_size = 0, 0
                for r in result:
                    aggr_result['val'] += r['val'] * r['val_size']
                    aggr_result['test'] += r['test'] * r['test_size']
                    aggr_result['val_std'] += r['val_std'] * r['val_size']
                    aggr_result['test_std'] += r['test_std'] * r['test_size']
                    val_size += r['val_size']
                    test_size += r['test_size']
                aggr_result['val'] /= val_size
                aggr_result['val_std'] /= val_size
                aggr_result['test'] /= test_size
                aggr_result['test_std'] /= test_size
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # logger, stopper have adapted to multi-graph setting
                is_stop = stopper(aggr_result)
                logger.log_pretrain(epoch, loss, aggr_result)
                if is_stop:
                    print("Early Stopping at Epoch:", epoch)
                    break

                wandb.log({
                    "training dynamics/train_loss": loss['train'],
                    "training dynamics/val_loss": loss['val'],
                    "training dynamics/test_loss": loss['test'],
                    "time/duration_training": training_time,
                    "time/duration_inference": inference_time
                })

                for r in result:
                    wandb.log({
                        "dataset": r['dataset'],
                        "training results/train_value": r['train'],
                        "training results/val_value": r['val'],
                        "training results/test_value": r['test'],
                        "training results/train_std": r['train_std'],
                        "training results/val_std": r['val_std'],
                        "training results/test_std": r['test_std'],
                    })
                
                wandb.log({
                    "Overall results/val_value": aggr_result['val'],
                    "Overall results/test_value": aggr_result['test']
                })

            else:
                loss['val'] = -loss['train']
                is_stop = stopper(loss)
                logger.log_pretrain(epoch, loss, None)
                if is_stop:
                    print("Early Stopping at Epoch:", epoch)
                    break
                wandb.log({
                    "training dynamics/train_loss": loss['train'],
                    "training dynamics/val_loss": loss['val'],
                    "training dynamics/test_loss": loss['test'],
                    "time/duration_training": training_time,
                })
        else:
            wandb.log({
                    "training dynamics/train_loss": loss['train'],
                    "training dynamics/val_loss": loss['val'],
                    "training dynamics/test_loss": loss['test'],
                    "time/duration_training": training_time,
                })

        if params['save_every'] != 0 and epoch % params['save_every'] == 0:
            # Construct checkpoint path
            checkpoint_dir = osp.join(params['save_vq'] if params['use_vq'] else params['save_path'], params['dataset'], params['save_name'])
            checkpoint_path = osp.join(checkpoint_dir, f"epoch_{epoch}.pt")
            
            # Create directory if needed
            check_path(checkpoint_dir)
            
            # Save model checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved at epoch {epoch} to {checkpoint_path}')
            if epoch == 20:
                break


    # if not params['use_vq']:
    best = logger.get_best_raw()
    wandb.log({
        "final result/train": "{:.2f} ± {:.2f}".format(best['train'], best['train_std']),
        "final result/val": "{:.2f} ± {:.2f}".format(best['val'], best['val_std']),
        "final result/test": "{:.2f} ± {:.2f}".format(best['test'], best['test_std']),
        "final result/train_mean": best['train'],
        "final result/val_mean": best['val'],
        "final result/test_mean": best['test'],
        "final result/train_std": best['train_std'],
        "final result/val_std": best['val_std'],
        "final result/test_std": best['test_std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})
    wandb.log({
        "time/training_mean": np.mean(training_time),
        "time/training_std": np.std(training_time),
        "time/training": "{:.2f} ± {:.2f}".format(np.mean(training_time), np.std(training_time)),
    })
    if not params['use_vq']:
        wandb.log({
            "time/inference_mean": np.mean(inference_time),
            "time/inference_std": np.std(inference_time),
            "time/inference": "{:.2f} ± {:.2f}".format(np.mean(inference_time), np.std(inference_time))
        })
    wandb.finish()

    if params['save_every'] != 0:
        # Construct paths
        model_dir = osp.join(params['save_path'] if not params['use_vq'] else params['save_vq'], params['dataset'])
        checkpoint_dir = osp.join(model_dir, params['save_name'])
        best_model_path = osp.join(checkpoint_dir, f"epoch_{best['epoch']}.pt")
        final_model_path = osp.join(model_dir, f"{params['save_name']}.pt")

        # Ensure save directory exists
        check_path(model_dir)

        # Copy best checkpoint to final location
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model saved at {final_model_path}, the best epoch is {best['epoch']}.")

        # Clean up checkpoint directory
        shutil.rmtree(checkpoint_dir)


def main():
    # set_memory_limit()  # 90% by default
    params = get_pt_args()

    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['pattern_path'] = osp.join(osp.dirname(__file__), '..', 'patterns')
    params['save_path'] = osp.join(osp.dirname(__file__), '..', 'model')
    params['save_vq'] = osp.join(osp.dirname(__file__), '..', 'vq')

    data_config = osp.join(osp.dirname(__file__), '..', 'config', 'data.yaml')
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    params['data_config'] = data_config
    # Current assumptions: different datasets share the same task and eval metrics, they only differ in feat dim and graph structures.
    datasets = params['dataset'].strip().split(';')
    params['task'] = data_config[datasets[0]]['task']
    params['metric'] = data_config[datasets[0]]['metric']
    # params['num_tasks'] = data_config[params['dataset']].get('num_tasks', None)

    if params["use_params"]:
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'pretrain.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
            # TODO: Unify experimental settings for different datasets
            params.update(default_params[params['task']][datasets[0]])

    if params['no_node_pe']:
        params['node_pe'] = 'none'
    if params['no_ap']:
        params['pe_encoder'] = 'none'

    if params['inference_only']:
        params['epochs'] = 1
        params['eval_every'] = 1

    wandb.init(
        project="G2PM",
        config=params,
        mode="disabled" if params["debug"] else "online"
    )
    params = dict(wandb.config)
    print(params)

    run(params)


if __name__ == "__main__":
    main()
