from typing import List
import argparse
import hashlib
import numpy as np
import hashlib
from omegaconf import OmegaConf


SENTENCEPIECE_IS_CONTINUATION = lambda t: not t.startswith("▁")
WORDPIECE_IS_CONTINUATION = lambda t: t.startswith("##")
MAX_HASH_VALUE = 2 ** 31 - 1

class Projection:
    def __init__(
        self, hash_path: int, feature_size: int, window_size: int, **kwargs
    ):
        self.hash = CachedHash(hash_path)
        self.cbf = CountingBloomFilter(feature_size)
        self.feature_size = feature_size
        self.window_size = window_size

    def __call__(self, words: List[List[str]]) -> np.ndarray:
        hashed = np.array([np.array([self.hash(token) for token in word]).min(axis=-2) for word in words])
        features = self.cbf(hashed)
        if self.window_size > 0: 
            padded = np.pad(features, ((self.window_size, self.window_size), (0, 0)))
            rows = self.feature_size * np.arange(0, padded.shape[0] - 2)[:, None]
            cols = np.arange((2 * self.window_size + 1) * self.feature_size)[None]
            features = padded.flatten()[rows + cols]
        return features

class MinHash:
    def __init__(self, num_hashes: int, ngram_size: int):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.hash_fn1 = lambda data: int.from_bytes(hashlib.new('sha256', data.encode("utf8")).digest(), 'little')
        self.hash_fn2 = lambda data: int.from_bytes(hashlib.new('sha224', data.encode("utf8")).digest(), 'little')

    def __call__(self, token: str, is_cont: bool) -> np.ndarray:
        if is_cont or len(token) < self.ngram_size + 1: 
            hash1 = self.hash_fn1(token)
            hash2 = self.hash_fn2(token)
            hash = np.array([(hash1 + i * hash2) % MAX_HASH_VALUE for i in range(self.num_hashes)])
            return hash
        trigrams = []
        for index in range(len(token) - self.ngram_size + 1): 
            hash1 = self.hash_fn1(token[index:index+self.ngram_size])
            hash2 = self.hash_fn2(token[index:index+self.ngram_size])
            hash = np.array([(hash1 + i * hash2) % MAX_HASH_VALUE for i in range(self.num_hashes)])
            trigrams.append(hash)
        fingerprint = np.array(trigrams).min(axis=-2)
        return fingerprint

class CachedHash:
    def __init__(self, path: str):
        self.cached_hash = np.load(path, allow_pickle=True).item()

    def __call__(self, token: str) -> np.ndarray:
        return self.cached_hash[token]

class CountingBloomFilter: 
    def __init__(self, feature_size: int):
        self.feature_size = feature_size
        self.one_hot = np.eye(feature_size, dtype=np.float32)

    def __call__(self, words: np.ndarray) -> np.ndarray: 
        features = self.one_hot[words % self.feature_size].sum(axis=-2)
        return features

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab_file', type=str)
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-o', '--outfile', type=str, default='vocab.npy')
    return parser.parse_args()


if __name__ == '__main__': 
    args = parse_args()
    with open(args.vocab_file) as vocab_file: 
        vocabs = vocab_file.readlines()
    
    vocabs = list(map(lambda l: l.strip().split('\t')[0], vocabs))
    cfg = OmegaConf.load(args.cfg)
    is_cont = (
        WORDPIECE_IS_CONTINUATION
        if cfg.tokenizer_type == "wordpiece"
        else SENTENCEPIECE_IS_CONTINUATION
    )
    proj_cfg = cfg.model.projection
    min_hash = MinHash(proj_cfg.num_hashes)
    cache = {v: min_hash(v.replace('##', '').replace('▁', ''), is_cont(v)).astype(np.int32) for v in vocabs}
    np.save(args.outfile, cache)