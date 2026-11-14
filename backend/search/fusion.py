"""
fusion.py — Reciprocal Rank Fusion (RRF): merge ranked lists into one.
Module 4: Embeddings + RAG

THE PROBLEM:
    Vector search scores are cosines (≈ 0.2–0.8); BM25 scores are unbounded
    (3.1, 17.4, ...). You can't add or average numbers on different scales.

THE TRICK — use RANKS, not scores:
    score(doc) = Σ over each list   1 / (k + rank_in_that_list)

    A doc ranked #1 by both retrievers:  1/61 + 1/61 = 0.0328
    #1 in one list, absent from other:   1/61        = 0.0164
    #3 in one list and #4 in the other:  1/63 + 1/64 = 0.0315  ← beats a single #1

    So documents BOTH retrievers like rise to the top — exactly what hybrid
    search wants. k (default 60, from the original 2009 paper) damps the
    gap between rank 1 and rank 2 so one list can't dominate.
"""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Args:
        rankings: several lists of doc ids, each ordered best-first.
        k: damping constant.
    Returns:
        (doc_id, rrf_score) pairs, best first. Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, and dicts keep insertion order → deterministic ties
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
