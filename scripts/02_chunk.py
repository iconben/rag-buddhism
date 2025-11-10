# Convert Bilara -> Chunk objects and save to data/chunks/
# Experiment with chunk size / overlap / collections here.

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.bilara_io import iter_documents
from lib.chunking import chunk_document, Chunk

BILARA_ROOT = "data"
CHUNKS_PATH = Path("chunks/chunks.json")


def main():
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for doc in iter_documents(BILARA_ROOT):
        chunks = chunk_document(doc, target_tokens=300, overlap_tokens=50)
        all_chunks.extend(chunks)

    # Save as JSON list
    serializable = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "segment_ids": c.segment_ids,
            "text": c.text,
            "norm_text": c.norm_text,
            "sc_link": c.sc_link,
        }
        for c in all_chunks
    ]

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
