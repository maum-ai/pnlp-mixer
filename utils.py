from sentencepiece import SentencePieceProcessor

from tqdm import trange, tqdm
from typing import Tuple, Dict, List

class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models.
    https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        # Get SentencePiece
        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in trange(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in tqdm(vocab.keys(), total=sp.GetPieceSize()):
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges

