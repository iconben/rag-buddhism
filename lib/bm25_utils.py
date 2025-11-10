import pickle
from typing import List, Tuple


from rank_bm25 import BM25Okapi

from .chunking import Chunk


class BM25Index:
    # Simple wrapper for BM25.
    def __init__(self):
        self.bm25 = None
        self.chunk_ids: List[str] = []

    def fit(self, chunks: List[Chunk]):
        # Fit BM25 on normalized chunk texts.
        corpus = [ch.norm_text.lower().split() for ch in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunk_ids = [ch.chunk_id for ch in chunks]

    def save(self, path: str):
        # Save index to file.
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        # Load index from file.
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = cls()
        inst.bm25 = obj["bm25"]
        inst.chunk_ids = obj["chunk_ids"]
        return inst

    def search(self, query: str, top_n: int = 50) -> List[Tuple[str, float]]:
        # Return top_n (chunk_id, score) for query.
        if self.bm25 is None:
            raise RuntimeError("BM25Index not fitted/loaded.")
        q_tokens = query.lower().split()
        scores = self.bm25.get_scores(q_tokens)
        pairs = list(zip(self.chunk_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]
