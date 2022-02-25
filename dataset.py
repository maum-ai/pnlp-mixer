from projection import Projection
import json
import numpy as np
import pytorch_lightning as pl
from omegaconf.dictconfig import DictConfig
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer, SentencePieceUnigramTokenizer
from typing import Any, Dict, List

class PnlpMixerDataModule(pl.LightningDataModule): 

    def __init__(self, vocab_cfg: DictConfig, train_cfg: DictConfig, proj_cfg: DictConfig, **kwargs):
        super(PnlpMixerDataModule, self).__init__(**kwargs)
        self.vocab_cfg = vocab_cfg
        self.train_cfg = train_cfg
        self.projecion = Projection(vocab_cfg.vocab_path, proj_cfg.feature_size, proj_cfg.window_size)

        if vocab_cfg.tokenizer_type == 'wordpiece': 
            self.tokenizer = BertWordPieceTokenizer(**vocab_cfg.tokenizer)
        if vocab_cfg.tokenizer_type == 'sentencepiece_bpe': 
            self.tokenizer = SentencePieceBPETokenizer(**vocab_cfg.tokenizer)
        if vocab_cfg.tokenizer_type == 'sentencepiece_unigram': 
            self.tokenizer = SentencePieceUnigramTokenizer(**vocab_cfg.tokenizer)

    def get_dataset_cls(self): 
        if self.train_cfg.dataset_type == 'mtop': 
            return MtopDataset
        if self.train_cfg.dataset_type == 'matis': 
            return MultiAtisDataset
        if self.train_cfg.dataset_type == 'imdb': 
            return ImdbDataset

    def setup(self, stage: str = None): 
        root = Path(self.train_cfg.dataset_path)
        label_list = Path(self.train_cfg.labels).read_text().splitlines() if isinstance(self.train_cfg.labels, str) else self.train_cfg.labels
        self.label_map = {label: index for index, label in enumerate(label_list)}
        dataset_cls = self.get_dataset_cls()
        if stage in (None, 'fit'): 
            self.train_set = dataset_cls(root, 'train', self.train_cfg.max_seq_len, self.tokenizer, self.projecion, self.label_map)
            self.eval_set = dataset_cls(root, 'test', self.train_cfg.max_seq_len, self.tokenizer, self.projecion, self.label_map)
        if stage in (None, 'test'): 
            self.test_set = dataset_cls(root, 'test', self.train_cfg.max_seq_len, self.tokenizer, self.projecion, self.label_map)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.train_cfg.train_batch_size, shuffle=True, num_workers=self.train_cfg.num_workers, persistent_workers=True)#, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.train_cfg.test_batch_size, shuffle=False, num_workers=self.train_cfg.num_workers, persistent_workers=True)#, pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.train_cfg.test_batch_size, shuffle=False, num_workers=self.train_cfg.num_workers, persistent_workers=True)#, pin_memory=True)

class PnlpMixerDataset(Dataset): 
    def __init__(self, max_seq_len: int, tokenizer: Tokenizer, projection: Projection, label_map: Dict[str, int], **kwargs):
        super(PnlpMixerDataset, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_seq_len = max_seq_len
        self.label_map = label_map

    def normalize(self, text: str) -> str: 
        return text.replace('’', '\'')\
            .replace('–', '-')\
            .replace('‘', '\'')\
            .replace('´', '\'')\
            .replace('“', '"')\
            .replace('”', '"')

    def project_features(self, words: List[str]) -> np.ndarray: 
        encoded = self.tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)
        tokens = [[] for _ in range(len(words))]
        for index, token in zip(encoded.words, encoded.tokens): 
            tokens[index].append(token)
        features = self.projection(tokens)
        padded_featrues = np.pad(features, ((0, self.max_seq_len - len(words)), (0, 0)))
        return padded_featrues

    def get_words(self, fields: List[str]) -> List[str]: 
        raise NotImplementedError

    def compute_labels(self, fields: List[str]) -> np.ndarray: 
        raise NotImplementedError
    
    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index].split('\t')
        words = self.get_words(fields)
        features = self.project_features(words)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }

class ImdbDataset(PnlpMixerDataset): 
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(ImdbDataset, self).__init__(*args, **kwargs)
        self.data = []
        for file in root.glob(f'{filename}/*/*.txt'): 
            self.data.append(f'{file.read_text()}\t{file.parent.stem}')

    def __len__(self) -> int: 
        return len(self.data)

    def normalize(self, text: str) -> str: 
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]: 
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields[0]))][:self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray: 
        return np.array(self.label_map[fields[-1]])

class MultiAtisDataset(PnlpMixerDataset): 

    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(MultiAtisDataset, self).__init__(*args, **kwargs)
        self.data = []
        for file in root.glob(f'{filename}_*.tsv'): 
            self.data.extend(file.read_text().splitlines()[1:-1])

    def __len__(self): 
        return len(self.data)

    def get_words(self, fields: List[str]) -> List[str]: 
        return self.normalize(fields[1]).split(' ')

    def compute_labels(self, fields: List[str]) -> np.ndarray: 
        return np.array(self.label_map[fields[-1]])

class MtopDataset(PnlpMixerDataset): 

    def __init__(self, root: Path, filename: str, *args, **kwargs):
        super(MtopDataset, self).__init__(*args, **kwargs)
        self.data = []
        for file in root.glob(f'*/{filename}.txt'): 
            self.data.extend(file.read_text().splitlines())

    def __len__(self) -> int: 
        return len(self.data)

    def get_words(self, fields: List[str]) -> List[str]: 
        segments = json.loads(fields[-1])
        normalized_words = [self.normalize(word) for word in segments['tokens']]
        return normalized_words

    def compute_labels(self, fields: List[str]) -> np.ndarray: 
        segments = json.loads(fields[-1])
        num_words = len(segments['tokens'])
        slot_list = fields[2].split(',')
        slot = np.ones([num_words], dtype=np.long) * self.label_map['O']
        slot = np.pad(slot, (0, self.max_seq_len - num_words), constant_values=-1)
        starts = {}
        ends = {}
        for index, span in enumerate(segments['tokenSpans']): 
            starts[span['start']] = index
            ends[span['start'] + span['length']] = index + 1
        for s in slot_list: 
            if not s: 
                break
            start, end, _, val = s.split(':', maxsplit=3)
            start_index = starts[int(start)]
            end_index = ends[int(end)]
            slot[start_index] = self.label_map[f'B-{val}']
            if end_index > start_index + 1: 
                slot[start_index+1:end_index] = self.label_map[f'I-{val}']
        return slot
