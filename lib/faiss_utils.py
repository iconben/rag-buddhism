import json
import os
from typing import List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from .chunking import Chunk

# Itt is direkt egyszerű implementációt kapsz, hogy lásd az áramlást.


class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []  # list of dicts: {chunk_id, doc_id, sc_link}

    @staticmethod
    def _embed_batch(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
        resp = client.embeddings.create(model=model, input=texts)
        vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
        return np.vstack(vecs)

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        model: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        batch_size: int = 64,
    ) -> "FaissIndex":
        client = OpenAI()
        meta = []
        all_vecs = []

        batch_texts = []
        for ch in chunks:
            batch_texts.append(ch.norm_text)
            meta.append(
                {
                    "chunk_id": ch.chunk_id,
                    "doc_id": ch.doc_id,
                    "sc_link": ch.sc_link,
                }
            )
            if len(batch_texts) == batch_size:
                V = cls._embed_batch(client, batch_texts, model)
                all_vecs.append(V)
                batch_texts = []

        if batch_texts:
            V = cls._embed_batch(client, batch_texts, model)
            all_vecs.append(V)

        X = np.vstack(all_vecs)
        dim = X.shape[1]

        faiss.normalize_L2(X)

        inst = cls(dim=dim)
        inst.index.add(X)
        inst.meta = meta
        return inst

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FaissIndex":
        index = faiss.read_index(index_path)
        dim = index.d
        inst = cls(dim=dim)
        inst.index = index
        meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        inst.meta = meta
        return inst

    def search(
        self,
        query: str,
        model: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        top_n: int = 50,
    ) -> List[Tuple[str, float]]:
        client = OpenAI()
        v = client.embeddings.create(model=model, input=[query]).data[0].embedding
        q = np.array(v, dtype="float32")[None, :]

        if q.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: Index has {self.dim}, but current model produced {q.shape[1]}. "
                "Please run 'python scripts/03_index_build.py' to rebuild the index."
            )

        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, top_n)

        res: List[Tuple[str, float]] = []
        for i, s in zip(idx[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            cid = self.meta[i]["chunk_id"]
            res.append((cid, float(s)))
        return res
