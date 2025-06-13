import numpy as np
import torch

metric2order = {'loss': 'min', 'acc': 'max', 'f1': 'max', 'precision': 'max', 'recall': 'max', 'auc': 'max',
                'ap': 'max', 'mcc': 'max', 'hits@20': 'max', 'hits@50': 'max', 'hits@100': 'max', 'ndcg': 'max', 'map': 'max', 'mrr': 'max', 'rmse': 'min',
                'mae': 'min'}


class Logger:
    def __init__(self, vq_mode=False):
        self.data = {}
        self.best = {}
        self.vq_mode = vq_mode

    def check_result(self, result):
        if 'metric' not in result:
            raise ValueError('Result must contain metric key')
        if result['metric'] not in metric2order:
            raise ValueError('Metric not supported')
        if result['train'] is None:
            result['train'] = 0
        if result['val'] is None:
            result['val'] = 0

        return result

    def log(self, run, epoch, loss, result):
        result = self.check_result(result)

        train_value = result['train']
        val_value = result['val']
        test_value = result['test']

        if run not in self.data:
            self.data[run] = {'train': [], 'val': [], 'test': []}

        self.data[run]['loss_train'] = loss
        self.data[run]['train'].append(train_value)
        self.data[run]['val'].append(val_value)
        self.data[run]['test'].append(test_value)
        self.data[run]['epoch'] = epoch

        if run not in self.best:
            self.best[run] = {'train': None, 'val': None, 'test': None}

        if metric2order[result['metric']] == 'max':
            if self.best[run]['val'] is None or val_value >= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['epoch'] = epoch
        else:
            if self.best[run]['val'] is None or val_value <= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['epoch'] = epoch

    def log_pretrain(self, epoch, loss, result):

        if not self.vq_mode:
            result = self.check_result(result)

            train_value = result['train']
            val_value = result['val']
            test_value = result['test']
            train_std = result['train_std']
            val_std = result['val_std']
            test_std = result['test_std']
        
        else:
            val_value = -loss['train']
            train_value, test_value, train_std, val_std, test_std = 0, 0, 0, 0, 0

        if len(self.data) == 0:
            self.data = {'loss_train': [], 'train': [], 'val': [], 'test': [], 'train_std': [], 'val_std': [], 'test_std': []}

        self.data['loss_train'].append(loss)
        self.data['train'].append(train_value)
        self.data['val'].append(val_value)
        self.data['test'].append(test_value)
        self.data['train_std'].append(train_std)
        self.data['val_std'].append(val_std)
        self.data['test_std'].append(test_std)
        self.data['epoch'] = epoch

        if len(self.best) == 0:
            self.best = {'train': None, 'val': None, 'test': None, 'train_std': None, 'val_std': None, 'test_std': None}

        if self.vq_mode or metric2order[result['metric']] == 'max':
            if self.best['val'] is None or val_value >= self.best['val']:
                self.best['train'] = train_value
                self.best['val'] = val_value
                self.best['test'] = test_value
                self.best['train_std'] = train_std
                self.best['val_std'] = val_std
                self.best['test_std'] = test_std
                self.best['epoch'] = epoch
        else:
            if self.best['val'] is None or val_value <= self.best['val']:
                self.best['train'] = train_value
                self.best['val'] = val_value
                self.best['test'] = test_value
                self.best['train_std'] = train_std
                self.best['val_std'] = val_std
                self.best['test_std'] = test_std
                self.best['epoch'] = epoch

    def get_run_raw(self):
        return self.data

    def get_best_raw(self):
        return self.best

    def get_single_run(self, run_idx):
        return self.data[run_idx]

    def get_single_best(self, run_idx):
        return self.best[run_idx]

    def get_run(self):
        train = np.mean([np.mean(self.data[run_idx]['train']) for run_idx in self.data])
        val = np.mean([np.mean(self.data[run_idx]['val']) for run_idx in self.data])
        test = np.mean([np.mean(self.data[run_idx]['test']) for run_idx in self.data])
        return {'train': train, 'val': val, 'test': test}

    def get_best(self):
        train = [self.best[run_idx]['train'] for run_idx in self.best]
        val = [self.best[run_idx]['val'] for run_idx in self.best]
        test = [self.best[run_idx]['test'] for run_idx in self.best]

        return {'train': {'mean': np.mean(train), 'std': np.std(train)},
                'val': {'mean': np.mean(val), 'std': np.std(val)},
                'test': {'mean': np.mean(test), 'std': np.std(test)}}
