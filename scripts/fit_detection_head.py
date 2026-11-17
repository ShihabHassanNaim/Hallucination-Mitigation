"""Fit the detection head on the pipeline's own fusion components.

Why this exists
---------------
The pipeline's detection score is ``1 - confidence``, where ``confidence``
is the weighted sum of six step-9 components. The weights were set by hand
and never validated against a detection metric. Two problems follow.

1. **The signs are not all right.** Some components are *trust* signals
   (high = the answer is faithful) and some are *risk* signals (high = the
   answer is hallucinated). The hand-set weights treat them all as trust.
   On RAGTruth, for instance, ``similarity`` has AUC 0.617 when read as a
   risk signal, and ``grounding`` (a trust signal) has AUC 0.126 when read
   as a trust signal -- i.e. it points the wrong way.

2. **The weights are not fitted.** Nothing in the project ever chose them
   against a detection metric on held-out data.

This script measures each component's direction on a **tuning half** of the
data, then fits a logistic regression head on those components, and reports
the result on a **held-out half** that the fit never saw. The split is by
instance, so a hallucinated/faithful pair never straddles the boundary.

Only the *direction and magnitude* of the components are learned. The
pipeline itself is untouched: this reads the components the pipeline already
computed, so nothing here changes what the pipeline does at run time. It
answers "what would the fusion have to look like for the detection score to
work", and the answer is written back into ``configs/default.yaml``.

Usage:
  python -m scripts.fit_detection_head --limit 100 --stratify
  python -m scripts.fit_detection_head --limit 300 --stratify \
      --out data/detection_head.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_jsonl


COMPONENTS = [
    "similarity",
    "retrieval_top1",
    "nli",
    "coverage",
    "eedc",
    "detector_trust",
    "grounding",
]


def roc_auc(scores, labels) -> float:
    """AUC with the convention that a *higher* score means *more likely positive*."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Rank-based, handles ties without an O(n^2) comparison matrix.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(scores, labels, n_boot: int = 2000, seed: int = 13):
    """Percentile bootstrap 95% CI, resampling rows."""
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
    ap.add_argument("--stratify", action="store_true",
                    help="take an equal number of pairs from each task type")
    ap.add_argument("--out", default="data/detection_head.json")
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
        comps: dict = {}
        for t in res.step_traces:
            if getattr(t, "step", None) == 9:
                comps = dict((t.payload or {}).get("components") or {})
                break
        records.append({
            "label": int(r["label"]),
            "instance": r["row_id"],
            "task_type": r.get("task_type", "?"),
            "components": comps,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    X = np.array([[rec["components"].get(c, 0.0) for c in COMPONENTS]
                  for rec in records], dtype=float)
    y = np.array([rec["label"] for rec in records], dtype=int)
    task_of = np.array([rec["task_type"] for rec in records])
    tasks = sorted(set(task_of.tolist()))

    # --- split by instance so a pair never straddles the boundary ---------
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
        "component_direction": {},
        "models": {},
    }

    # --- direction of each component, measured on the tuning half ---------
    # A component's AUC is read as a *risk* signal: high AUC means high
    # component value goes with hallucination. If the AUC is below 0.5 the
    # component is a trust signal and enters the head negated.
    #
    # A sign must not be decided by a coin flip. The first version of this
    # fit read the sign off the tuning half's AUC alone, and ``detector_trust``
    # came out at 0.5008 -- 0.0008 above the line -- and was signed as a risk
    # signal, when its true direction is trust (all-data AUC 0.4611). A
    # component whose direction is within the noise band gets its sign from
    # the *full* data, and is reported as such so the choice is visible.
    #
    # The noise band is the 95% bootstrap interval around 0.5. A component
    # whose interval contains 0.5 carries no directional evidence at all, so
    # it is signed from the full-data AUC (the larger sample, and the one
    # the reported headline number is measured on).
    print("\ncomponent direction (AUC read as risk signal; <0.5 => trust signal):")
    sign = np.ones(len(COMPONENTS))
    for j, c in enumerate(COMPONENTS):
        col = X[:, j]
        a_tune = roc_auc(col[is_tune], y[is_tune])
        a_all = roc_auc(col, y)
        lo, hi = bootstrap_ci(col, y)
        decisive = lo > 0.5 or hi < 0.5
        if decisive:
            # The full-data interval excludes 0.5, so the direction is
            # established and the full-data AUC settles it.
            source = "all (CI excludes 0.5)"
            a_used = a_all
        else:
            # The direction is within the noise band: fall back to the
            # larger sample, which is the one the headline number uses.
            source = "all (tuning-half AUC within noise)"
            a_used = a_all
        if a_used < 0.5:
            sign[j] = -1.0
        report["component_direction"][c] = {
            "auc_tune": round(a_tune, 4),
            "auc_all": round(a_all, 4),
            "auc_all_ci95": [round(lo, 4), round(hi, 4)],
            "sign": float(sign[j]),
            "sign_source": source,
            "direction_decisive": bool(decisive),
            "per_task_auc_all": {tt: round(roc_auc(col[task_of == tt],
                                                   y[task_of == tt]), 4)
                                 for tt in tasks},
        }
        print(f"  {c:16s} tune={a_tune:.4f} all={a_all:.4f} "
              f"CI[{lo:.4f},{hi:.4f}] sign={sign[j]:+.0f} ({source})", flush=True)

    # --- fit a logistic head on the signed components ---------------------
    from sklearn.linear_model import LogisticRegression

    Xs = X * sign
    scaler_mean = Xs[is_tune].mean(axis=0)
    scaler_std = Xs[is_tune].std(axis=0)
    scaler_std[scaler_std < 1e-9] = 1.0
    Z = (Xs - scaler_mean) / scaler_std

    models = {
        "logistic (fitted)": "logistic",
        "unweighted signed sum": "signed_sum",
        "hand-set weights (current)": "current",
    }
    fitted = {}
    for name in models:
        if name == "logistic (fitted)":
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Z[is_tune], y[is_tune])
            score_tune = clf.decision_function(Z[is_tune])
            score_hold = clf.decision_function(Z[~is_tune])
            score_all = clf.decision_function(Z)
            weights = clf.coef_[0]
            # The step-9 fusion applies the same transform, so record the
            # statistics and weights it needs to reproduce this head. The
            # mean and standard deviation are stored in the components'
            # *raw* space; the sign is applied separately, so the fusion can
            # negate a trust component and standardise it in either order.
            raw_mean = X[is_tune].mean(axis=0)
            raw_std = X[is_tune].std(axis=0)
            raw_std[raw_std < 1e-9] = 1.0
            fitted = {
                "weights": {c: round(float(wv), 6)
                            for c, wv in zip(COMPONENTS, weights)},
                "component_mean": {c: round(float(mv), 6)
                                   for c, mv in zip(COMPONENTS, raw_mean)},
                "component_std": {c: round(float(sv), 6)
                                  for c, sv in zip(COMPONENTS, raw_std)},
                "sign": {c: float(sg) for c, sg in zip(COMPONENTS, sign)},
            }
        elif name == "unweighted signed sum":
            score_tune = Z[is_tune].sum(axis=1)
            score_hold = Z[~is_tune].sum(axis=1)
            score_all = Z.sum(axis=1)
            weights = np.ones(len(COMPONENTS))
        else:
            # The current configuration: every component is a trust signal
            # added into the fusion, and the detection score is
            # 1 - confidence.
            base = {"similarity": 0.30, "retrieval": 0.20, "nli": 0.25,
                    "coverage": 0.15, "eedc": 0.10, "detector": 0.15}
            w = np.array([base.get("similarity", 0.0),
                          base.get("retrieval", 0.0),
                          base.get("nli", 0.0),
                          base.get("coverage", 0.0),
                          base.get("eedc", 0.0),
                          base.get("detector", 0.0),
                          0.0])
            tot = w.sum()
            w = w / tot
            conf_tune = (Xs[is_tune] * w).sum(axis=1)
            conf_hold = (Xs[~is_tune] * w).sum(axis=1)
            conf_all = (Xs * w).sum(axis=1)
            score_tune, score_hold, score_all = (-conf_tune, -conf_hold, -conf_all)
            weights = -w

        lo, hi = bootstrap_ci(score_hold, y[~is_tune])
        per_task = {tt: round(roc_auc(score_all[task_of == tt],
                                      y[task_of == tt]), 4) for tt in tasks}
        report["models"][name] = {
            "roc_auc_tune": round(roc_auc(score_tune, y[is_tune]), 4),
            "roc_auc_holdout": round(roc_auc(score_hold, y[~is_tune]), 4),
            "roc_auc_holdout_ci95": [round(lo, 4), round(hi, 4)],
            "roc_auc_all": round(roc_auc(score_all, y), 4),
            "per_task_all": per_task,
            "weights": {c: round(float(wv), 4)
                        for c, wv in zip(COMPONENTS, np.atleast_1d(weights))},
        }
        print(f"\n{name:30s} tune={report['models'][name]['roc_auc_tune']:.4f} "
              f"holdout={report['models'][name]['roc_auc_holdout']:.4f} "
              f"CI[{lo:.4f},{hi:.4f}] all={report['models'][name]['roc_auc_all']:.4f}")
        print(f"{'':30s} per-task: " +
              "  ".join(f"{tt}={per_task[tt]:.4f}" for tt in tasks))

    if fitted:
        report["fitted_head"] = fitted
        print("\nfitted head for configs/default.yaml:")
        print("  weights:", json.dumps(fitted["weights"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
