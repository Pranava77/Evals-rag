"""Retrieval strategies: dense (cosine), keyword (term overlap), hybrid (fuse)."""
from __future__ import annotations

from itertools import zip_longest

from .store import Chunk


def keyword_top_k(query: str, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
    """BM25-lite: score by exact term overlap. Nails identifiers dense misses."""
    q_terms = set(query.lower().split())
    scored = sorted(chunks,
                    key=lambda c: len(q_terms & set(c.text.lower().split())),
                    reverse=True)
    return [c for c in scored[:k]
            if q_terms & set(c.text.lower().split())]


def rank_fuse(dense: list[Chunk], sparse: list[Chunk], k: int = 3) -> list[Chunk]:
    """Simplest fusion: interleave + dedupe. (Production: RRF — same shape.)"""
    seen, out = set(), []
    for pair in zip_longest(dense, sparse):
        for c in pair:
            if c is not None and c.id not in seen:
                seen.add(c.id)
                out.append(c)
    return out[:k]


def hybrid_top_k(qvec, query: str, store, all_chunks: list[Chunk], k: int = 3):
    dense = store.top_k(qvec, k=5)
    sparse = keyword_top_k(query, all_chunks, k=5)
    return rank_fuse(dense, sparse, k=k)
