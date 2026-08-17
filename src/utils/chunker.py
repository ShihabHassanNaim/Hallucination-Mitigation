"""Sentence + paragraph chunking."""
from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # crude but good enough for English
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_documents(docs: list[str], chunk_size: int = 400, chunk_overlap: int = 50) -> list[tuple[str, str]]:
    """Return (doc_id, chunk) pairs."""
    out: list[tuple[str, str]] = []
    for i, d in enumerate(docs):
        text = d.strip()
        if len(text) <= chunk_size:
            out.append((f"doc-{i}", text))
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            out.append((f"doc-{i}-chunk-{start}", text[start:end]))
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
    return out
