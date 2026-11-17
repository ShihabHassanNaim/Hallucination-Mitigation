"""Verify the fitted detection head through the real pipeline.

Runs the pipeline on a stratified sample of the RAGTruth matched split and
computes the detection ROC-AUC from the score the pipeline itself emits
(``1 - confidence``), with a bootstrap CI and a per-task breakdown.

This is the check that matters: it exercises the whole chain -- retrieval,
reranking, claim extraction, NLI, EEDC, fusion -- rather than the
components in isolation, so it confirms the head works where it is used.

Usage:
  python -m scripts.verify_detection_head --limit 100 --stratify
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    """Percentile bootstrap 95% CI over rows."""
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
    ap.add_argument("--data", default="data/ragtruth_eval_1000_matched.jsonl")
    ap.add_argument("--index", default="data/index_ragtruth")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--out", default="data/detection_head_verified.json")
    args = ap.parse_args()

    cfg = load_config()
    pipe = Pipeline(cfg)
    pipe.load_index(args.index)

    rows = load_jsonl(args.data)
    if args.stratify:
        by_task: dict[str, list[dict]] = defaultdict(list)
        seen: set[str] = set()
        for r in rows:
            key = r.get("control_of") or r["row_id"]
            if key in seen:
                continue
            seen.add(key)
            by_task[r.get("task_type", "?")].append(r)
        per_task = max(1, args.limit // max(1, len(by_task)))
        rows = []
        for tt, group in sorted(by_task.items()):
            rows.extend(group[:per_task])
    elif args.limit:
        rows = rows[: args.limit]

    scores: list[float] = []
    labels: list[int] = []
    task_of: list[str] = []
    for i, r in enumerate(rows):
        res = pipe.run(r["query"], answer=r["answer"], label=int(r["label"]),
                       gold_answer=r.get("right_answer"))
        d = res.to_dict()
        # The detection score is the pipeline's own hallucination
        # probability, which the fusion now emits directly.
        scores.append(float(d.get("confidence", 0.5)))
        labels.append(int(r["label"]))
        task_of.append(r.get("task_type", "?"))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    # The fusion now produces a risk probability, and ``confidence`` is
    # 1 - risk. The detection score must be the *risk*, so flip it back.
    det = [1.0 - s for s in scores]

    auc = roc_auc(det, labels)
    lo, hi = bootstrap_ci(det, labels)
    tasks = sorted(set(task_of))
    per = {}
    for tt in tasks:
        idx = [i for i, t in enumerate(task_of) if t == tt]
        sub_s = [det[i] for i in idx]
        sub_y = [labels[i] for i in idx]
        tlo, thi = bootstrap_ci(sub_s, sub_y)
        per[tt] = {
            "n": len(idx),
            "roc_auc": round(roc_auc(sub_s, sub_y), 4),
            "ci95": [round(tlo, 4), round(thi, 4)],
        }
        print(f"  {tt:10s} n={len(idx):4d} AUC={per[tt]['roc_auc']:.4f} "
              f"CI[{tlo:.4f},{thi:.4f}]")

    report = {
        "n_rows": len(rows),
        "detection_roc_auc": round(auc, 4),
        "detection_roc_auc_ci95": [round(lo, 4), round(hi, 4)],
        "per_task": per,
        "dataset": "ragtruth_matched",
    }
    print(f"\nDETECTION ROC-AUC {auc:.4f} CI[{lo:.4f},{hi:.4f}] on n={len(rows)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
