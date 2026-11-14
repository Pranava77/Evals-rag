"""Chunking strategies — how you split decides what you find."""
from __future__ import annotations


def fixed(text: str, size: int = 400) -> list[str]:
    """Dumb baseline: cuts every `size` chars, mid-sentence and all."""
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def by_paragraph(text: str) -> list[str]:
    """Simplest semantic chunking: respect paragraph boundaries."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def parent_child(text: str, child_size: int = 200):
    """Parent-document trick: index small children, return (child, parent) pairs
    so retrieval is precise but the model gets the whole paragraph."""
    pairs = []
    for para in by_paragraph(text):
        for child in fixed(para, child_size):
            pairs.append((child, para))
    return pairs
