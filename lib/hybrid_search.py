from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def merge_rankings(bm25_results: List[Tuple[str, float]], faiss_results: List[Tuple[str, float]],
    k: int = 60, top_k: int = 5) -> List[Tuple[str, float]]:
    # bm25_results / faiss_results: [(chunk_id, score), ...]
    # ranking matters, not raw scores
    # fused score(d) = sum(1 / (k + rank_i(d)))
    fused: Dict[str, float] = defaultdict(float)

    for rank, (cid, _) in enumerate(bm25_results, start=1):
        fused[cid] += 1.0 / (k + rank)

    for rank, (cid, _) in enumerate(faiss_results, start=1):
        fused[cid] += 1.0 / (k + rank)

    items = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return items[:top_k]
