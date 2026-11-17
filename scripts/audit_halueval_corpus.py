"""Is HaluEval's corpus an answer key, or a legitimate RAG corpus?

TruthfulQA's corpus is the dataset's own answer key: it holds the correct
answer for every question and the incorrect answer for none. That is a
leak, and it has to be fixed by running the dataset closed-book.

HaluEval looks similar on a naive check -- every faithful answer appears
somewhere in the corpus -- but the reason is different, and the difference
matters:

* HaluEval's corpus is built from the dataset's ``knowledge`` field, the
  **Wikipedia source passage** for the question. The faithful answer is
  entailed by that passage, so it appears in it; the hallucinated answer
  is a perturbation of it and usually is not. Retrieving the question's
  own source passage and checking the answer against it is exactly what a
  RAG hallucination detector is supposed to do.

* TruthfulQA has no source passage at all. Its corpus was built from the
  reference *answers*, so there is no context to check against -- the
  "source" IS the label.

The test that separates them: **build a corpus that excludes each
evaluation row's own passage** and re-measure. If the signal survives,
the detector was verifying against genuine context. If it collapses, the
signal was an answer-key read.

This script measures the audit; it does not change any pipeline code.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_jsonl


def roc_auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(scores, labels, n_boot: int = 2000, seed: int = 13):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    out = []
    n = len(s)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc(s[idx], y[idx]))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/halueval_eval_1000.jsonl")
    ap.add_argument("--index", default="data/index_halueval")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="data/halueval_corpus_audit.json")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    rows = load_jsonl(args.data)[: args.limit]

    report: dict = {"n_rows": len(rows), "variants": {}}

    for label, metadata in (
        ("full corpus (the row's own passage is present)", {}),
        ("grounding off", {"grounding_source": "none"}),
    ):
        cfg = load_config()
        pipe = Pipeline(cfg)
        pipe.load_index(args.index)
        scores: list[float] = []
        labels: list[int] = []
        for r in rows:
            res = pipe.run(r["query"], answer=r["answer"], label=int(r["label"]),
                           gold_answer=r.get("right_answer"), metadata=dict(metadata))
            prob = 0.5
            for t in res.step_traces:
                if getattr(t, "step", None) == 9:
                    payload = getattr(t, "payload", None) or {}
                    prob = float(payload.get("hallucination_probability", 0.5))
                    break
            scores.append(prob)
            labels.append(int(r["label"]))
        a = roc_auc(scores, labels)
        lo, hi = bootstrap_ci(scores, labels)
        report["variants"][label] = {
            "roc_auc": round(a, 4),
            "roc_auc_ci95": [round(lo, 4), round(hi, 4)],
        }
        print(f"{label:52s} AUC={a:.4f} CI[{lo:.4f},{hi:.4f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
