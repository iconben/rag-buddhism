from dataclasses import dataclass
from typing import List, Iterable
import re
import unicodedata
import html

import tiktoken

from .bilara_io import Document

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    # chunk metadata
    chunk_id: str
    doc_id: str
    segment_ids: List[str]
    text: str          # display text
    norm_text: str     # normalized text
    sc_link: str


def clean(text: str) -> str:
    # Simple normalization: unicode, HTML unescape, remove bracket notes, collapse spaces.
    t = unicodedata.normalize("NFKC", text)
    t = html.unescape(t)
    t = re.sub(r"\[\d+\]", " ", t)
    t = re.sub(r"[\u200B\u00AD\s]+", " ", t).strip()
    return t.lower()


def chunk_document(doc: Document, target_tokens: int = 300, overlap_tokens: int = 50, ) -> List[Chunk]:
    # Token-based sliding-window chunking.
    items = sorted(doc.segments.items(), key=lambda kv: kv[0])

    seg_texts = []
    seg_ids = []

    for seg_id, raw in items:
        nt = clean(raw)
        if nt:
            seg_ids.append(seg_id)
            seg_texts.append(nt)

    # encode all segments
    seg_token_lens = []
    all_tokens = []
    for t in seg_texts:
        toks = enc.encode(t)
        seg_token_lens.append(len(toks))
        all_tokens.extend(toks)

    chunks: List[Chunk] = []

    # empty guard
    if not all_tokens:
        return chunks

    # sliding window over token stream
    start = 0
    while start < len(all_tokens):
        end = min(len(all_tokens), start + target_tokens)
        window = all_tokens[start:end]
        text = enc.decode(window)
        norm = clean(text)
        segment_ids = _segments_for_span(seg_ids, seg_token_lens, start, end)

        chunk_id = f"{doc.doc_id}#{len(chunks):04d}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            segment_ids=segment_ids,
            text=text,
            norm_text=norm,
            sc_link=doc.sc_link,
        ))

        if end == len(all_tokens):
            break
        start = max(0, end - overlap_tokens)

    return chunks


def _segments_for_span(seg_ids: List[str], seg_token_lens: List[int], start: int, end: int) -> List[str]:
    # Return segment IDs overlapping token span [start, end).
    res = []
    acc = 0
    for sid, ln in zip(seg_ids, seg_token_lens):
        s, e = acc, acc + ln
        if not (e <= start or s >= end):
            res.append(sid)
        acc = e
    return res
