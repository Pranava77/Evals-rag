"""Vector stores behind one two-method interface — the tier ladder is a
config change, not a rewrite.

    add(vec, chunk)  /  top_k(qvec, k) -> [Chunk]

InMemoryStore is the teaching/testing tier. Chroma/pgvector adapters sketch
the next rungs (optional deps; used in the live demo against the lab stack).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    doc: str
    parent: str = ""          # parent-document trick: full section text


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


@dataclass
class InMemoryStore:
    rows: list = field(default_factory=list)   # [(vec, Chunk)]

    def add(self, vec, chunk: Chunk) -> None:
        self.rows.append((vec, chunk))

    def top_k(self, qvec, k: int = 3):
        scored = sorted(((cos(qvec, v), c) for v, c in self.rows),
                        key=lambda p: p[0], reverse=True)
        return [c for _, c in scored[:k]]


class ChromaStore:
    """On-disk tier (optional): pip install chromadb. Same two methods."""

    def __init__(self, path: str = ".chroma"):
        import chromadb
        self.col = chromadb.PersistentClient(path).get_or_create_collection("docs")

    def add(self, vec, chunk: Chunk) -> None:
        self.col.add(ids=[chunk.id], embeddings=[vec],
                     documents=[chunk.text], metadatas=[{"doc": chunk.doc}])

    def top_k(self, qvec, k: int = 3):
        res = self.col.query(query_embeddings=[qvec], n_results=k)
        return [Chunk(id=i, text=t, doc=m["doc"]) for i, t, m in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]


class PgVectorStore:
    """SQL tier (optional): the lab stack's Postgres+pgvector. Same interface.
    Sketch — see Session 0 lab; requires `pip install psycopg[binary]`."""

    def __init__(self, dsn: str):
        import psycopg
        self.conn = psycopg.connect(dsn)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""CREATE TABLE IF NOT EXISTS chunks
                           (id text primary key, doc text, text text,
                            vec vector(64))""")
        self.conn.commit()

    def add(self, vec, chunk: Chunk) -> None:
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO chunks VALUES (%s,%s,%s,%s) "
                        "ON CONFLICT (id) DO NOTHING",
                        (chunk.id, chunk.doc, chunk.text, str(vec)))
        self.conn.commit()

    def top_k(self, qvec, k: int = 3):
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, doc, text FROM chunks "
                        "ORDER BY vec <=> %s LIMIT %s", (str(qvec), k))
            return [Chunk(id=r[0], doc=r[1], text=r[2]) for r in cur.fetchall()]
