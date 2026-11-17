"""Measure the direction of the step-3 detector's contribution.

The fitted head in ``configs/default.yaml`` gives ``detector_trust`` the
sign +1 (a risk signal). That sign was decided by a tuning-half AUC of
0.5008 -- a coin flip that landed 0.0008 above the 0.5 line -- while the
all-data AUC of the same component is 0.4127, i.e. clearly below 0.5.

A sign that depends on the fourth decimal place of a near-choke AUC is
not a sign, it is noise. This script measures the direction properly, on
the full RAGTruth split, with a bootstrap interval, so the decision is
made on evidence rather than on a coin flip.

Two readings are reported:

* ``detector_trust`` -- the component as the fusion uses it, i.e.
  ``1 - hallucination_probability``.
* ``hallucination_probability`` -- the raw step-3 output.

One of the two is a trust signal and the other a risk signal, and the
relationship is a mirror, so their AUCs must sum to 1.

Usage:
  python -m scripts.check_detector_direction --limit 300
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
    ap.add_argument("--data", default="data/ragtruth_eval_1000_matched.jsonl")
    ap.add_argument("--index", default="data/index_ragtruth")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="data/detector_direction.json")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    cfg = load_config()
    pipe = Pipeline(cfg)
    pipe.load_index(args.index)

    rows = load_jsonl(args.data)[: args.limit]
    probs: list[float] = []
    trusts: list[float] = []
    labels: list[int] = []
    tasks: list[str] = []

    for i, r in enumerate(rows):
        res = pipe.run(r["query"], answer=r["answer"], label=int(r["label"]),
                       gold_answer=r.get("right_answer"))
        p = None
        for t in res.step_traces:
            if getattr(t, "step", None) == 3:
                payload = getattr(t, "payload", None) or {}
                p = payload.get("hallucination_probability")
                break
        if p is None:
            continue
        probs.append(float(p))
        trusts.append(1.0 - float(p))
        labels.append(int(r["label"]))
        tasks.append(r.get("task_type", "?"))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    y = np.array(labels, dtype=int)
    task_of = np.array(tasks)
    task_names = sorted(set(task_of.tolist()))

    report: dict = {"n_rows": len(labels), "components": {}}
    for name, col in (("hallucination_probability", np.array(probs)),
                      ("detector_trust", np.array(trusts))):
        a = roc_auc(col, y)
        lo, hi = bootstrap_ci(col, y)
        per_task = {tt: round(roc_auc(col[task_of == tt], y[task_of == tt]), 4)
                    for tt in task_names}
        report["components"][name] = {
            "auc_all": round(a, 4),
            "auc_all_ci95": [round(lo, 4), round(hi, 4)],
            "per_task_auc_all": per_task,
            "direction": ("risk" if a > 0.5 else "trust"),
        }
        print(f"{name:26s} AUC={a:.4f} CI[{lo:.4f},{hi:.4f}] "
              f"-> {'risk' if a > 0.5 else 'trust'}")
        print(f"{'':26s} per-task: " +
              "  ".join(f"{tt}={per_task[tt]:.4f}" for tt in task_names))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
