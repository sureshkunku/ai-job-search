"""
bm25.py — keyword (sparse) search.
Module 4: Embeddings + RAG

WHAT BM25 does:
    Scores a document by the query words it contains, weighted so that
      - RARE words matter more ("pgvector" is a strong signal, "engineer" isn't)
      - repeating a word helps, but with diminishing returns (k1)
      - long documents don't win just by being long (b)
    It's the classic search-engine ranking function and needs no model.

WHY BM25Plus and not the classic BM25Okapi:
    Okapi's rarity weight is log((N − df + 0.5) / (df + 0.5)). For a word in
    HALF the documents that's log(1) = 0, and above half it goes negative.
    Our corpus is pre-filtered to tech jobs (Module 3 keywords), so "python"
    IS in most documents — with Okapi, searching "python" scored every job 0
    (our tests caught exactly this). BM25Plus uses log((N + 1) / df), which
    stays positive, and adds a small floor (delta) so a long description
    can't push a real match down to ~0.

WHY keep keyword search when we have embeddings:
    Embeddings are fuzzy — great for "ML engineer ≈ machine learning
    developer", weak on exact tokens. Search "FastAPI" and vector search
    may happily return Django/Flask jobs ("similar meaning"); BM25 returns
    jobs that literally say FastAPI. Hybrid search (hybrid.py) uses both.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Plus

from backend.search.types import SearchDoc, SearchFilters

# Keeps tech tokens intact: "c++", "c#", "node.js", "gpt-4o"→["gpt", "4o"] is fine.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#]*(?:\.[a-z0-9]+)*")
# Very common words carry no ranking signal — dropping them also shrinks the index.
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "of", "on", "or", "our", "the", "this", "to", "we", "will", "with", "you", "your", "who", "what", "job", "role", "team", "work"]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, docs: list[SearchDoc]) -> None:
        self.docs = docs
        self._tokens = [tokenize(d.bm25_text()) for d in docs]
        # rank_bm25 needs at least one document; an empty index just returns nothing.
        self._bm25 = BM25Plus(self._tokens) if docs else None

    def search(self, query: str, k: int = 20, filters: SearchFilters | None = None) -> list[tuple[str, float]]:
        """Top-k (doc_id, score) pairs; documents sharing no query word are left out."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        wanted = set(tokens)
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        hits: list[tuple[str, float]] = []
        for i in ranked:
            # BM25Plus's delta gives EVERY doc a small score for every query
            # word, even docs that don't contain it — so check real overlap.
            if not wanted.intersection(self._tokens[i]):
                continue
            if filters and not filters.matches(self.docs[i]):
                continue
            hits.append((self.docs[i].id, float(scores[i])))
            if len(hits) == k:
                break
        return hits
