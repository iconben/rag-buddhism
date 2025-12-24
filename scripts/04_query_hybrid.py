# Quick CLI hybrid search + RAG
# Tweak n_bm25 / n_faiss / top_k to experiment

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.bm25_utils import BM25Index
from lib.faiss_utils import FaissIndex
from lib.rag_core import generate_answer, load_chunk_lookup, retrieve_hybrid

INDEX_DIR = Path("index")
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_META_PATH = INDEX_DIR / "faiss_meta.jsonl"
CHUNK_LOOKUP_PATH = INDEX_DIR / "chunk_lookup.json"


def main():
    bm25 = BM25Index.load(str(BM25_PATH))
    faiss = FaissIndex.load(str(FAISS_INDEX_PATH), str(FAISS_META_PATH))
    chunk_lookup = load_chunk_lookup(str(CHUNK_LOOKUP_PATH))

    print("Ready — type question or 'exit'.")

    while True:
        q = input("\nQ> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        contexts = retrieve_hybrid(
            q,
            bm25,
            faiss,
            chunk_lookup,
            n_bm25=50,
            n_faiss=50,
            rrf_k=60,
            top_k=5,
        )

        print("\n[Context snippets]")
        for c in contexts:
            print(f"- {c.chunk_id} ({c.doc_id}) {c.sc_link}  [score={c.score:.4f}]")
            print(f"  {c.text[:200]}...")
        print()

        answer = generate_answer(q, contexts)
        print("[Answer]")
        print(answer)


if __name__ == "__main__":
    main()
