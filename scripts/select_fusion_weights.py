"""Choose the fusion weights from measurement, on a held-out split.

The pipeline's confidence score is a weighted sum of six components. The
weights in ``configs/default.yaml`` were set by hand and never validated
against a detection metric, so it is not known whether the score they
produce can separate hallucinated from faithful answers at all.

This script runs the pipeline once per row, records *every* fusion
component for that row, and then computes the detection ROC-AUC for
candidate weight vectors without re-running the pipeline. That makes a
weight sweep cheap.

Split discipline
----------------
The rows are split into a **tuning half** and a **held-out half** before
any weight is chosen. The best weight is picked on the tuning half and
then reported on the held-out half. A weight that only works on the half it
was chosen on is overfitted, and the held-out number is what has to be
used in the paper.

Usage:
  python -m scripts.select_fusion_weights --limit 100
  python -m scripts.select_fusion_weights --limit 100 \
      --out data/fusion_weights_ragtruth.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_jsonl

# The components of the fused score, in the order step 9 combines them.
COMPONENTS = ["similarity", "retrieval_top1", "nli", "coverage", "eedc",
              "detector_trust", "grounding"]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    sum_pos = float(ranks[pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray,
                 n_boot: int = 2000, seed: int = 13) -> tuple[float, float]:
    """Percentile bootstrap CI over rows (used for the held-out report)."""
    rng = np.random.default_rng(seed)
    out = []
    n = len(scores)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        out.append(roc_auc(scores[idx], labels[idx]))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ragtruth_eval_1000_matched.jsonl")
    ap.add_argument("--index", default="data/index_ragtruth")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument(
        "--stratify",
        action="store_true",
        help="take an equal number of pairs from each task type. The "
             "evaluation file is sorted by task type, so a prefix of it "
             "measures one task type only.",
    )
    ap.add_argument("--out", default="data/fusion_weights_ragtruth.json")
    args = ap.parse_args()

    cfg = load_config()
    pipe = Pipeline(cfg)
    pipe.load_index(args.index)

    rows = load_jsonl(args.data)
    if args.stratify:
        from collections import defaultdict
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

    records: list[dict] = []
    for i, r in enumerate(rows):
        res = pipe.run(r["query"], answer=r["answer"], label=int(r["label"]),
                       gold_answer=r.get("right_answer"))
        d = res.to_dict()
        # The fusion components are exposed through the step-9 trace.
        comps: dict = {}
        for t in res.step_traces:
            if getattr(t, "step", None) == 9:
                comps = dict((t.payload or {}).get("components") or {})
                break
        if not comps:
            # Fall back: recompute the components from the pieces we have.
            from src.utils.grounding import grounding_coverage
            comps = {"grounding": grounding_coverage(
                r["answer"] or "", [e.get("text", "") for e in d["evidence"]])}
        records.append({
            "label": int(r["label"]),
            "instance": r["row_id"],
            "task_type": r.get("task_type", "?"),
            "components": comps,
            # The retrieved text, so the grounding component can be
            # recomputed against different evidence sets without re-running
            # the pipeline.
            "evidence_texts": [e.get("text", "") for e in d["evidence"]],
            "gold_text": r.get("knowledge") or "",
            "answer": r.get("answer") or "",
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    # Pad any missing component with 0.0 so the matrix is rectangular.
    X = np.array([[rec["components"].get(c, 0.0) for c in COMPONENTS]
                  for rec in records], dtype=float)
    y = np.array([rec["label"] for rec in records], dtype=int)

    # --- split into tuning and held-out halves -----------------------------
    # Split by instance so the two halves never share a pair.
    insts = sorted({rec["instance"] for rec in records})
    rng = np.random.default_rng(13)
    shuffled = list(rng.permutation(insts))
    cut = len(shuffled) // 2
    tune_insts = set(shuffled[:cut])
    is_tune = np.array([rec["instance"] in tune_insts for rec in records])

    report: dict = {
        "n_rows": len(records),
        "n_tune": int(is_tune.sum()),
        "n_holdout": int((~is_tune).sum()),
        "components": COMPONENTS,
        "component_auc": {},
        "candidates": {},
    }

    # --- candidate weight vectors -----------------------------------------
    base = {"similarity": 0.30, "retrieval": 0.20, "nli": 0.25,
            "coverage": 0.15, "eedc": 0.10, "detector": 0.15}
    candidates: dict[str, dict[str, float]] = {
        "configured (no grounding)": dict(base, grounding=0.0),
    }
    for g in (0.15, 0.25, 0.35, 0.50, 0.75, 1.00):
        candidates[f"grounding={g:.2f}"] = dict(base, grounding=g)
    # Grounding alone.
    candidates["grounding only"] = {c: 0.0 for c in base}
    candidates["grounding only"]["grounding"] = 1.0

    # Per-task-type masks. The three RAGTruth task types have different
    # base rates and different signal strengths, so a pooled AUC mixes them
    # and is reported per task type as well.
    tasks = sorted({rec["task_type"] for rec in records})
    task_of = np.array([rec["task_type"] for rec in records])

    # --- per-component AUC -------------------------------------------------
    # Which components actually carry signal, and in which direction. A
    # component with AUC below 0.5 is anti-correlated and is hurting the
    # fused score, so this has to be known before any weight is chosen.
    from src.utils.grounding import novel_content_rate

    print("\ncomponent AUC (higher = more likely HALLUCINATED when negated):")
    for j, c in enumerate(COMPONENTS):
        col = X[:, j]
        a_all = roc_auc(col, y)
        per = {tt: round(roc_auc(col[task_of == tt], y[task_of == tt]), 4)
               for tt in tasks}
        report["component_auc"][c] = {"auc_all": round(a_all, 4), "per_task": per}
        print(f"  {c:16s} all={a_all:.4f}  " +
              "  ".join(f"{tt}={per[tt]:.4f}" for tt in tasks), flush=True)

    # The grounding component recomputed against different evidence sets, to
    # see whether the choice of source text is what determines its strength.
    print("\ngrounding recomputed (AUC, higher = more likely hallucinated):")
    ground_variants: dict[str, list[float]] = {
        "top-1 (in pipeline)": [],
        "all retrieved": [],
        "gold source": [],
    }
    for rec in records:
        ev = rec["evidence_texts"]
        ground_variants["top-1 (in pipeline)"].append(
            novel_content_rate(rec["answer"], ev[:1]))
        ground_variants["all retrieved"].append(
            novel_content_rate(rec["answer"], ev))
        ground_variants["gold source"].append(
            novel_content_rate(rec["answer"], [rec["gold_text"]]))
    report["grounding_variants"] = {}
    for name, vals in ground_variants.items():
        a = roc_auc(np.array(vals), y)
        per = {tt: round(roc_auc(np.array(vals)[task_of == tt],
                                 y[task_of == tt]), 4) for tt in tasks}
        report["grounding_variants"][name] = {"auc_all": round(a, 4),
                                              "per_task": per}
        print(f"  {name:22s} all={a:.4f}  " +
              "  ".join(f"{tt}={per[tt]:.4f}" for tt in tasks), flush=True)

    idx_of = {c: i for i, c in enumerate(COMPONENTS)}
    for name, w in candidates.items():
        vec = np.zeros(len(COMPONENTS))
        for c, val in w.items():
            key = "retrieval_top1" if c == "retrieval" else c
            if key in idx_of:
                vec[idx_of[key]] = val
        total = vec.sum()
        if total <= 0:
            continue
        vec = vec / total
        scores = X @ vec
        auc_tune = roc_auc(scores[is_tune], y[is_tune])
        auc_hold = roc_auc(scores[~is_tune], y[~is_tune])
        auc_all = roc_auc(scores, y)
        lo, hi = bootstrap_ci(scores[~is_tune], y[~is_tune])
        per_task = {}
        for tt in tasks:
            m = task_of == tt
            per_task[tt] = {
                "n": int(m.sum()),
                "roc_auc_holdout": round(roc_auc(scores[m & ~is_tune],
                                                y[m & ~is_tune]), 4),
                "roc_auc_all": round(roc_auc(scores[m], y[m]), 4),
            }
        # The pooled held-out AUC is only comparable across candidates if
        # the task mix is the same in both halves; report it, but let the
        # per-task numbers decide.
        report["candidates"][name] = {
            "weights": {c: round(vec[idx_of[c]], 4) for c in COMPONENTS},
            "roc_auc_tune": round(auc_tune, 4),
            "roc_auc_holdout": round(auc_hold, 4),
            "roc_auc_holdout_ci95": [round(lo, 4), round(hi, 4)],
            "roc_auc_all": round(auc_all, 4),
            "per_task": per_task,
        }
        print(f"{name:32s} tune={auc_tune:.4f} holdout={auc_hold:.4f} "
              f"CI[{lo:.4f},{hi:.4f}] all={auc_all:.4f}", flush=True)
        print(f"{'':32s}   per-task(all): " + "  ".join(
            f"{tt}={per_task[tt]['roc_auc_all']:.4f}" for tt in tasks),
            flush=True)

    # --- pick the winner on the tuning half --------------------------------
    # Selection uses the *tuning* half only, and it uses the mean of the
    # per-task-type AUCs rather than the pooled one. The pooled number is
    # inflated or deflated by which task types happen to land in the split,
    # so choosing on it would be choosing on an artefact of the split.
    def _mean_task_auc(entry: dict, half: str) -> float:
        vals = [v[f"roc_auc_{half}"] for v in entry["per_task"].values()]
        return sum(vals) / len(vals) if vals else 0.5

    # Recompute the tuning-half per-task AUC for selection.
    for name, w in candidates.items():
        vec = np.zeros(len(COMPONENTS))
        for c, val in w.items():
            key = "retrieval_top1" if c == "retrieval" else c
            if key in idx_of:
                vec[idx_of[key]] = val
        total = vec.sum()
        if total <= 0:
            continue
        vec = vec / total
        scores = X @ vec
        tune_task = {}
        for tt in tasks:
            m = (task_of == tt) & is_tune
            tune_task[tt] = round(roc_auc(scores[m], y[m]), 4)
        report["candidates"][name]["per_task_tune"] = tune_task
        report["candidates"][name]["roc_auc_tune_mean_task"] = round(
            sum(tune_task.values()) / max(1, len(tune_task)), 4)

    best = max(report["candidates"].items(),
                key=lambda kv: kv[1]["roc_auc_tune_mean_task"])
    report["selected"] = {
        "name": best[0],
        "weights": best[1]["weights"],
        "roc_auc_tune_mean_task": best[1]["roc_auc_tune_mean_task"],
        "roc_auc_tune": best[1]["roc_auc_tune"],
        "roc_auc_holdout": best[1]["roc_auc_holdout"],
        "roc_auc_holdout_ci95": best[1]["roc_auc_holdout_ci95"],
        "per_task": best[1]["per_task"],
    }
    print(f"\nselected on tuning half (mean per-task AUC): {best[0]} -> "
          f"holdout ROC-AUC {best[1]['roc_auc_holdout']:.4f} "
          f"CI {best[1]['roc_auc_holdout_ci95']}")
    for tt in tasks:
        print(f"  {tt:10s} holdout AUC={best[1]['per_task'][tt]['roc_auc_holdout']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
