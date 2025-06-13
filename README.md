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

## TODO

The adaptability of multi-graph setting to graph classification task needs further improvement.

Can't use node classification datasets and graph classification datasets at the same time for now.
