"""Measure the lexical grounding signal per task type, with CIs.

The grounding signal is "the share of the answer's content words that
occur in the source". It was measured at 0.70 AUC over the first 100 rows
and at 0.32 AUC over a later 40-row slice, which is a contradiction that
has to be resolved before anything is claimed about it.

The suspicion is that the signal does not have one sign. A faithful answer
paraphrases the source, so its wording diverges; a hallucinated answer
often copies the source's field values verbatim and adds unsupported
prose around them, so its wording overlaps *more*. That would make the
signal negative (higher coverage = more likely hallucinated) on the
structured Data2txt task and positive on the prose QA and Summary tasks.

This script measures it per task type, with a bootstrap CI, so the sign
and the strength are known per task rather than pooled and averaged into
one misleading number.

Usage:
  python -m scripts.diagnose_grounding_signal --limit 200
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.data_loader import load_jsonl


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    c = sum(1 for a in pos for b in neg if a > b)
    c += 0.5 * sum(1 for a in pos for b in neg if a == b)
    return c / (len(pos) * len(neg))


def _boot_ci(scores: list[float], labels: list[int], n_boot: int = 2000,
             seed: int = 13) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    s = np.array(scores, dtype=float)
    y = np.array(labels, dtype=int)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), size=len(s))
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(_auc(list(s[idx]), list(y[idx])))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ragtruth_eval_1000_matched.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument(
        "--stratify",
        action="store_true",
        help="take an equal number of rows from each task type instead of "
             "the first N. The evaluation file is sorted by task type, so "
             "taking a prefix of it measures one task type only.",
    )
    ap.add_argument("--out", default="data/grounding_signal.json")
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    if args.stratify:
        # One row per pair, so every task type contributes the same number
        # of *pairs* and the sample is balanced the way the evaluation is.
        by_task: dict[str, list[dict]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for r in rows:
            key = (r.get("task_type", "?"), r.get("control_of") or r["row_id"])
            if key in seen:
                continue
            seen.add(key)
            by_task[r.get("task_type", "?")].append(r)
        per_task = max(1, args.limit // max(1, len(by_task)))
        picked: list[dict] = []
        for tt, group in sorted(by_task.items()):
            picked.extend(group[:per_task])
        rows = picked
    elif args.limit:
        rows = rows[: args.limit]

    # Two variants: with and without stopwords, and a "novel content" view.
    by_task: dict[str, list[tuple[float, int]]] = defaultdict(list)
    overall: list[tuple[float, int]] = []

    for r in rows:
        src = set(_tokens(r.get("knowledge") or ""))
        ans = _tokens(r.get("answer") or "")
        if not src or not ans:
            continue
        cov = len(set(ans) & src) / len(set(ans))
        # Novel content: the content words that the source does not have.
        novel = len(set(ans) - src) / len(set(ans))
        tt = r.get("task_type", "?")
        by_task[tt].append((cov, int(r["label"])))
        by_task[tt + "::novel"].append((novel, int(r["label"])))
        overall.append((cov, int(r["label"])))

    report: dict = {"n_rows": len(rows), "per_task": {}}
    for key, pairs in sorted(by_task.items()):
        scores = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        auc = _auc(scores, labels)
        lo, hi = _boot_ci(scores, labels)
        m0 = statistics.mean([s for s, y in zip(scores, labels) if y == 0]) \
            if any(y == 0 for y in labels) else float("nan")
        m1 = statistics.mean([s for s, y in zip(scores, labels) if y == 1]) \
            if any(y == 1 for y in labels) else float("nan")
        report["per_task"][key] = {
            "n": len(scores),
            "auc_higher_means_faithful": round(auc, 4),
            "auc_ci95": [round(lo, 4), round(hi, 4)],
            "mean_faithful": round(m0, 4),
            "mean_hallucinated": round(m1, 4),
        }
        print(f"{key:22s} n={len(scores):4d} AUC={auc:.4f} "
              f"CI[{lo:.4f},{hi:.4f}] faithful={m0:.4f} halluc={m1:.4f}",
              flush=True)

    scores = [p[0] for p in overall]
    labels = [p[1] for p in overall]
    auc = _auc(scores, labels)
    lo, hi = _boot_ci(scores, labels)
    report["overall_coverage"] = {
        "n": len(scores),
        "auc_higher_means_faithful": round(auc, 4),
        "auc_ci95": [round(lo, 4), round(hi, 4)],
    }
    print(f"\nOVERALL AUC={auc:.4f} CI[{lo:.4f},{hi:.4f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
