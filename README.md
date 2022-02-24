# pNLP-Mixer - Unofficial PyTorch Implementation

**pNLP-Mixer: an Efficient all-MLP Architecture for Language**

<img src="figures/architecture.png">

## Requirements

* Python >= 3.6.10
* PyTorch >= 1.8.0
* All other requirements are listed in the [`requirements.txt`](./requirements.txt) file.

## Configurations

```yaml
vocab: 
  tokenizer_type: wordpiece # wordpiece, sentencepiece-unigram or sentencepiece-bpe
  tokenizer: # constructor arguments for the tokenizer
    vocab: ./wordpiece/vocab.txt 
    lowercase: false
    strip_accents: false
    clean_text: false
  vocab_path: ./vocab.npy # path of the cached vocab hashes

train:
  dataset_type: mtop # mtop, matis or imdb
  dataset_path: ./data/mtop/ # root path of the dataset
  labels: ./data/mtop/bio_labels.txt # path to a file containing all labels OR list of all labels
  tensorboard_path: ./logs/ # path where the tensorboard logs and checkponts will be stored
  log_interval_steps: 10 # training step logging interval
  epochs: 50 # number of epochs to run
  train_batch_size: 256 # batch size during training
  test_batch_size: 256 # batch size during testing
  num_workers: 32 # number of workers to use for the dataloader
  max_seq_len: &max_seq_len 64 # maximum sequence length of the model
  optimizer: 
    lr: 5e-4 
    betas: [0.9, 0.999]
    eps: 1e-8 

model: # model hyperparameters
  projection:
    num_hashes: 64
    feature_size: &feature_size 1024
    window_size: &window_size 0
  bottleneck: 
    window_size: *window_size
    feature_size: *feature_size
    hidden_dim: &hidden_dim 64
  mixer: 
    num_mixers: 2
    max_seq_len: *max_seq_len
    hidden_dim: *hidden_dim
    mlp_hidden_dim: 256
  token_cls:
    hidden_dim: *hidden_dim
    num_classes: 151

```

## Commands

```bash
python run.py -c CFG_PATH -n MODEL_NAME -m MODE -r CKPT_PATH
```

* `CFG_PATH`: path to the configurations file.
* `MODEL_NAME`: model name to be used for pytorch lightning logging.
* `MODE`: `train` or `test` (default: `train`)
* `CKPT_PATH`: (optional) checkpoint path to resume training from / use for testing

## Results

### MTOP

| Model Size         | Reported | Ours  |
| ------------------ | -------- | :---- |
| pNLP-Mixer X-Small | 76.9%    | 79.3% |
| pNLP-Mixer Base    | 80.8%    | 79.4% |
| pNLP-Mixer X-Large | 82.3%    | 82.1% |

### MultiATIS

| Model Size         | Reported | Ours  |
| ------------------ | -------- | :---- |
| pNLP-Mixer X-Small | 90.0%    | 91.3% |
| pNLP-Mixer Base    | 92.1%    | 92.8% |
| pNLP-Mixer X-Large | 91.3%    | 92.9% |

\* Note that the paper reports the performance on the MultiATIS dataset using a 8-bit quantized model, whereas our performance was measured using a 32-bit float model. 

### IMDB

| Model Size         | Reported | Ours  |
| ------------------ | -------- | :---- |
| pNLP-Mixer X-Small | 81.9%    | 81.5% |
| pNLP-Mixer Base    | 78.6%    | 82.2% |
| pNLP-Mixer X-Large | 82.9%    | 82.9% |

## Paper

```latex
@article{fusco2022pnlp,
  title={pNLP-Mixer: an Efficient all-MLP Architecture for Language},
  author={Fusco, Francesco and Pascual, Damian and Staar, Peter},
  journal={arXiv preprint arXiv:2202.04350},
  year={2022}
}
```

## Contributors

* [Tony Woo](https://github.com/tonyswoo) @ MINDsLab Inc. ([shwoo@mindslab.ai](mailto:shwoo@mindslab.ai))

Special thanks to: 

* [Hyoung-Kyu Song](https://github.com/deepkyu) @ MINDsLab Inc.
* [Kang-wook Kim](https://github.com/wookladin) @ MINDsLab Inc.

## TODO

- [ ] 8-bit quantization
