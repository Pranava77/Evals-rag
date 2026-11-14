"""Embedders. FakeEmbedder = deterministic, offline, dependency-free — good for
tests and teaching. Real options (never Claude — no embeddings endpoint):
Ollama `nomic-embed-text` (local, $0) or Gemini embeddings (free tier).
"""
from __future__ import annotations

import hashlib
import math


class FakeEmbedder:
    """Deterministic bag-of-words hash embedding. Similar word sets ->
    nearby vectors. Enough to make cosine retrieval real in tests."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OllamaEmbedder:
    """Local real embeddings via the lab stack (optional, for the live demo)."""

    def __init__(self, model: str = "nomic-embed-text",
                 base: str = "http://localhost:11434"):
        self.model, self.base = model, base

    def embed(self, text: str) -> list[float]:
        import json
        import urllib.request
        req = urllib.request.Request(
            f"{self.base}/api/embeddings",
            data=json.dumps({"model": self.model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["embedding"]
