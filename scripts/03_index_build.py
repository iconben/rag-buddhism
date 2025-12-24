# Chunks -> BM25 + FAISS index + chunk_lookup.
# Run: python scripts/03_index_build.py

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.bm25_utils import BM25Index
from lib.chunking import Chunk
from lib.faiss_utils import FaissIndex

CHUNKS_PATH = Path("chunks/chunks.json")
INDEX_DIR = Path("index")
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_META_PATH = INDEX_DIR / "faiss_meta.jsonl"
CHUNK_LOOKUP_PATH = INDEX_DIR / "chunk_lookup.json"


def load_chunks() -> list[Chunk]:
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for d in data:
        chunks.append(
            Chunk(
                chunk_id=d["chunk_id"],
                doc_id=d["doc_id"],
                segment_ids=d["segment_ids"],
                text=d["text"],
                norm_text=d["norm_text"],
                sc_link=d["sc_link"],
            )
        )
    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    # BM25
    bm = BM25Index()
    bm.fit(chunks)
    bm.save(str(BM25_PATH))
    print(f"Saved BM25 index to {BM25_PATH}")

    # FAISS
    fa = FaissIndex.build(chunks)
    fa.save(str(FAISS_INDEX_PATH), str(FAISS_META_PATH))
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}, meta to {FAISS_META_PATH}")

    # Chunk lookup (for RAG)
    lookup = {
        c.chunk_id: {
            "doc_id": c.doc_id,
            "text": c.text,
            "sc_link": c.sc_link,
        }
        for c in chunks
    }
    with CHUNK_LOOKUP_PATH.open("w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk lookup to {CHUNK_LOOKUP_PATH}")


if __name__ == "__main__":
    main()
