import json
import os
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from .bm25_utils import BM25Index
from .chunking import Chunk
from .faiss_utils import FaissIndex
from .hybrid_search import merge_rankings


load_dotenv()


SYSTEM_PROMPT = """You are a helpful assistant answering strictly from Early Buddhist texts (SuttaCentral, English, Sujato).
Use only the provided context. If it's not enough, say you don't know.
Use brief inline citations like [doc_id#chunk_id].
At the end, list sources with SuttaCentral links.
"""


@dataclass
class RetrievedContext:
    chunk_id: str
    doc_id: str
    text: str
    sc_link: str
    score: float


def load_chunk_lookup(path: str) -> Dict[str, Dict]:
    # Simple lookup loader: returns {chunk_id: {doc_id, text, sc_link}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_user_prompt(question: str, contexts: List[RetrievedContext]) -> str:
    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"[{c.chunk_id}] ({c.doc_id}) {c.text}")
    ctx_block = "\n\n".join(ctx_lines)
    return f"Question: {question}\n\nContext:\n{ctx_block}\n\nAnswer in concise English with inline citations."


def generate_answer(
    question: str,
    contexts: List[RetrievedContext],
    model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    max_tokens: int = 500,
) -> str:
    client = OpenAI()
    prompt = build_user_prompt(question, contexts)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def retrieve_hybrid(
    query: str,
    bm25: BM25Index,
    faiss: FaissIndex,
    chunk_lookup: Dict[str, Dict],
    n_bm25: int = 50,
    n_faiss: int = 50,
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[RetrievedContext]:
    # Run BM25 and FAISS, fuse rankings, return top contexts.
    bm = bm25.search(query, top_n=n_bm25)
    ve = faiss.search(query, top_n=n_faiss)

    merged = merge_rankings(bm, ve, k=rrf_k, top_k=top_k)

    out: List[RetrievedContext] = []
    for cid, score in merged:
        meta = chunk_lookup.get(cid)
        if not meta:
            continue
        out.append(
            RetrievedContext(
                chunk_id=cid,
                doc_id=meta["doc_id"],
                text=meta["text"],
                sc_link=meta["sc_link"],
                score=score,
            )
        )
    return out
