"""Bundled mini-RAG seed (Session 3's design) so this session runs standalone."""
from __future__ import annotations

import os

from .chunkers import by_paragraph, fixed
from .embedder import FakeEmbedder
from .retrieval import hybrid_top_k
from .store import Chunk, InMemoryStore

DOCS = os.path.join(os.path.dirname(__file__), "..", "data", "docs")

PROMPT = ("Strict rule: Answer ONLY from the excerpts below. Cite the doc name. "
          "If the answer isn't in them, say you cannot answer.\n\n"
          "EXCERPTS:\n{ctx}\n\nQ: {q}")


class RagSeed:
    def __init__(self, llm, chunker=by_paragraph, prompt: str = PROMPT):
        self.llm = llm
        self.embedder = FakeEmbedder()
        self.store = InMemoryStore()
        self.chunks: list[Chunk] = []
        self.prompt = prompt
        for fn in sorted(os.listdir(DOCS)):
            text = open(os.path.join(DOCS, fn)).read()
            for i, piece in enumerate(chunker(text)):
                c = Chunk(id=f"{fn}#{i}", text=piece, doc=fn)
                self.store.add(self.embedder.embed(piece), c)
                self.chunks.append(c)

    def retrieve(self, q: str, k: int = 3):
        return hybrid_top_k(self.embedder.embed(q), q, self.store, self.chunks, k=k)

    def ask(self, q: str, k: int = 3):
        hits = self.retrieve(q, k=k)
        ctx = "\n---\n".join(f"[{c.doc}] {c.text}" for c in hits)
        return self.llm.generate(self.prompt.format(ctx=ctx, q=q)), hits


def broken_seed(llm):
    """The demo's sabotage: 120-char fixed chunking slices answers apart."""
    return RagSeed(llm, chunker=lambda t: fixed(t, 120))
