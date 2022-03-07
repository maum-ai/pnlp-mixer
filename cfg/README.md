

## Configurations in Details

We provide the configuration files used during the experiments. The properties in the `.yml` files are:

```yaml
vocab:
  tokenizer_type: wordpiece # wordpiece, sentencepiece-unigram or sentencepiece-bpe
  tokenizer: # constructor arguments for the tokenizer
    vocab: ./wordpiece/mbert_vocab.txt 
    lowercase: false
    strip_accents: false
    clean_text: false
  vocab_path: ./vocab.npy # path of the cached vocab hashes

train:
  dataset_type: mtop # mtop, matis or imdb
  dataset_path: ./mtop # root path of the dataset
  labels: ./labels/mtop_labels.txt # path to a file containing all labels OR list of all labels
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