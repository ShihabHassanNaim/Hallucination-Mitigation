"""Dataset loaders for Phase 1.

Three sources, in priority order:
  1. HaluEval-QA JSON file (user supplies path)
  2. Synthetic mini-corpus generated on the fly (for smoke tests + dev)
  3. Empty fallback (raises — caller must supply data)

Each item yielded is a dict with at minimum:
  - id: str
  - question: str
  - reference_answer: Optional[str]
  - knowledge: Optional[str]            # ground-truth passage if available
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


# --- public API --------------------------------------------------------------

def load_dataset(name: str, path: str | None = None) -> List[Dict]:
    """Dispatch by name."""
    name = name.lower()
    if name == "halueval":
        if not path:
            raise ValueError("HaluEval loader requires --path to a json/jsonl file.")
        return list(_iter_halueval(Path(path)))
    if name == "synthetic":
        return list(_iter_synthetic())
    raise ValueError(f"Unknown dataset '{name}'. Choose: halueval, synthetic.")


# --- HaluEval ----------------------------------------------------------------

def _iter_halueval(path: Path) -> Iterator[Dict]:
    """HaluEval files are .json with a list of {question, answer, ...} records,
    or .jsonl with one record per line. We accept both."""
    if not path.exists():
        raise FileNotFoundError(f"HaluEval file not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                yield _normalise_halueval(rec, i)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for i, rec in enumerate(data):
            yield _normalise_halueval(rec, i)


def _normalise_halueval(rec: Dict, i: int) -> Dict:
    # HaluEval field names vary slightly across versions; cover the common ones.
    question = rec.get("question") or rec.get("query") or rec.get("input")
    reference = rec.get("right_answer") or rec.get("answer") or rec.get("ground_truth")
    knowledge = rec.get("knowledge") or rec.get("context") or rec.get("passage")
    return {
        "id": str(rec.get("id", i)),
        "question": (question or "").strip(),
        "reference_answer": reference.strip() if isinstance(reference, str) else None,
        "knowledge": knowledge.strip() if isinstance(knowledge, str) else None,
    }


# --- synthetic ---------------------------------------------------------------

def _iter_synthetic() -> Iterator[Dict]:
    """A tiny built-in corpus so the pipeline is exercisable offline.

    Each knowledge passage is short, factual, and self-contained — ideal
    for verifying that retrieval + answer grounding work end-to-end.
    """
    items = [
        {
            "id": "syn-1",
            "question": "What is the capital of France?",
            "reference_answer": "Paris",
            "knowledge": "France is a country in Western Europe. Its capital is Paris, which is also its largest city.",
        },
        {
            "id": "syn-2",
            "question": "Who wrote the novel '1984'?",
            "reference_answer": "George Orwell",
            "knowledge": "The dystopian novel '1984' was written by the British author George Orwell and published in 1949.",
        },
        {
            "id": "syn-3",
            "question": "What is the boiling point of water at sea level in Celsius?",
            "reference_answer": "100",
            "knowledge": "At standard atmospheric pressure (sea level), pure water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
        },
        {
            "id": "syn-4",
            "question": "Which planet is known as the Red Planet?",
            "reference_answer": "Mars",
            "knowledge": "Mars is often called the Red Planet because of the iron oxide (rust) on its surface, which gives it a reddish appearance.",
        },
        {
            "id": "syn-5",
            "question": "What programming language is primarily used for the PyTorch deep learning framework?",
            "reference_answer": "Python",
            "knowledge": "PyTorch is an open-source machine learning library developed by Meta AI. It is primarily used with the Python programming language, though it also has a C++ interface.",
        },
    ]
    for item in items:
        yield item


# --- corpus extraction -------------------------------------------------------

def corpus_from_dataset(items: Iterable[Dict]) -> List[str]:
    """Build a deduplicated list of retrieval passages from a dataset.

    Uses each item's 'knowledge' field when available; otherwise an empty
    string (the item will then have to rely on a separate external corpus).
    """
    seen: Dict[str, None] = {}
    for it in items:
        k = it.get("knowledge")
        if k and k not in seen:
            seen[k] = None
    return list(seen.keys())