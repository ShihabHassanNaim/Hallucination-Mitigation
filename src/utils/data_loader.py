"""Synthetic + HaluEval data loaders."""
from __future__ import annotations

import json
from pathlib import Path

SYNTHETIC_CORPUS = [
    {"id": "d0", "text": "Paris is the capital of France. France is in Europe."},
    {"id": "d1", "text": "Eric Arthur Blair, known by his pen name George Orwell, was a British writer."},
    {"id": "d2", "text": "George Orwell wrote the novel 1984, published in 1949."},
    {"id": "d3", "text": "Water boils at 100 degrees Celsius at standard atmospheric pressure."},
    {"id": "d4", "text": "Mars is the fourth planet from the Sun in the Solar System."},
    {"id": "d5", "text": "PyTorch is an open-source machine learning framework developed by Meta."},
]

SYNTHETIC_QA = [
    {"query": "What is the capital of France?", "answer": "The capital of France is Paris.", "evidence": ["d0"]},
    {"query": "Who wrote 1984?", "answer": "George Orwell wrote 1984.", "evidence": ["d1", "d2"]},
    {"query": "At what temperature does water boil?", "answer": "Water boils at 100 degrees Celsius at sea level.", "evidence": ["d3"]},
    {"query": "Which planet is fourth from the Sun?", "answer": "Mars is the fourth planet from the Sun.", "evidence": ["d4"]},
    {"query": "What is PyTorch?", "answer": "PyTorch is an open-source ML framework by Meta.", "evidence": ["d5"]},
    {"query": "Is Eric Arthur Blair British?", "answer": "No, Eric Arthur Blair was French.", "evidence": ["d1"]},  # intentional contradiction
    {"query": "French capital?", "answer": "Paris is the capital of France, which has 12 million residents.", "evidence": ["d0"]},  # fabricated number
]


def load_synthetic_corpus() -> list[dict]:
    return list(SYNTHETIC_CORPUS)


def load_synthetic_qa() -> list[dict]:
    return list(SYNTHETIC_QA)


def load_halueval(path: str | Path) -> list[dict]:
    """Load HaluEval QA as a list of dicts with keys query/answer/evidence/label."""
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("question") or obj.get("query") or ""
            a = obj.get("right_answer") or obj.get("answer") or obj.get("response") or ""
            k = obj.get("knowledge") or obj.get("evidence") or ""
            label = obj.get("label") or obj.get("hallucination") or ""
            rows.append({"query": q, "answer": a, "evidence": [k] if k else [], "label": label})
    return rows


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
