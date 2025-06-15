# UGV
Universal Graph Vocabulary (UGV).

Currently, the pre-training of VQ-based graph vocabulary tokenizer and the Transformer-based graph pattern encoder are separated. To pre-train tokenizer, `--use_vq` should be activated. For example:

```bash
python G2PM/pretrain.py --dataset computers --use_vq --use_params
```

The pre-training of graph pattern encoder can be conducted after removing `--use_vq`. To train under multi-graph setting, you need to string all datasets together with ';'. For example:

```bash
python G2PM/pretrain.py --dataset computers;cora;arxiv --use_params
```

Only one dataset and its corresponding task is used for evaluation during pre-training under multi-graph setting. In our situation, the first specified dataset will be used for evaluation. For example, given instruction `--dataset computers;cora;arxiv`, computers will be the evaluation set. Given that each dataset is temporarily equipped with a pre-projection layer for feature alignment, it's not feasible to utilize unseen datasets for evaluation. 
