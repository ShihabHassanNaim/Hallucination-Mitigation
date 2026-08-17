"""Atomic claim extraction (Step 5).

Three modes:
  - "rule":    regex-based sentence splitter with dedup.
  - "llm":     calls the configured Llama generator.
  - "hybrid":  rule first, fall back to LLM if < 2 claims.
"""
from __future__ import annotations

import re

from .chunker import split_sentences


def _dedup(claims: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in claims:
        key = re.sub(r"\s+", " ", c.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def extract_claims_rule(text: str) -> list[str]:
    sents = split_sentences(text)
    cleaned = []
    for s in sents:
        s = re.sub(r"^[\s\-\*\u2022]+", "", s).strip()
        if len(s) >= 5:
            cleaned.append(s)
    return _dedup(cleaned)


def extract_claims_llm(text: str, llm_generate) -> list[str]:
    """Use the generator to atomise the answer.

    The generator receives a fixed prompt and is expected to return one
    claim per line, prefixed with '- ' or just bare.
    """
    prompt = (
        "Split the following answer into atomic factual claims. "
        "Return each claim on its own line, prefixed with '- '.\n\n"
        f"Answer:\n{text}\n\nClaims:\n"
    )
    out = llm_generate(system="You extract atomic claims.", user=prompt)
    claims: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*\d\.\)\s]+", "", line)
        if line:
            claims.append(line)
    return _dedup(claims)


def extract_claims(text: str, mode: str = "rule", llm_generate=None) -> list[str]:
    if mode == "llm" and llm_generate is not None:
        return extract_claims_llm(text, llm_generate)
    if mode == "hybrid" and llm_generate is not None:
        base = extract_claims_rule(text)
        if len(base) >= 2:
            return base
        return extract_claims_llm(text, llm_generate)
    return extract_claims_rule(text)


def intrinsic_overlap(claim: str, query: str) -> float:
    """Lexical-overlap proxy for the INTRINSIC / EXTRINSIC provenance tag."""
    qs = set(re.findall(r"\w+", query.lower()))
    cs = set(re.findall(r"\w+", claim.lower()))
    if not qs or not cs:
        return 0.0
    return len(qs & cs) / len(qs | cs)
