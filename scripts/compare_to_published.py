"""Compare the pipeline's detection result against published work.

The comparison is deliberately strict:

* Only metrics the published work actually reports are compared. AUROC is
  never compared against an F1, and vice versa.
* A win requires this pipeline's 95% CI **lower bound** to sit strictly
  above the published value. A point estimate that beats it while the
  interval overlaps is reported as a tie, not a win.
* Every published number is read from ``data/published_anchors.json``,
  which was verified against the source PDFs.

Usage:
  python -m scripts.compare_to_published \
      --metrics data/metrics_ragtruth_fitted.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_anchors(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _per_task_auc(m: dict) -> dict:
    """Detection ROC-AUC per RAGTruth task type.

    The task type of each evaluated row is recovered by matching the row's
    query against the evaluation file. The query is the retrieval key, and
    the three RAGTruth task types have distinct prompts (a question, a
    business-overview instruction, a summarisation instruction), so the
    mapping is unambiguous.
    """
    import numpy as np

    from src.utils.data_loader import load_jsonl

    pq = m.get("per_query") or []
    if not pq:
        return {}
    try:
        rows = load_jsonl("data/ragtruth_eval_1000_matched.jsonl")
    except Exception:  # noqa: BLE001
        return {}
    q2tt: dict[str, str] = {}
    for r in rows:
        q2tt.setdefault(r.get("query", ""), r.get("task_type", "?"))

    per: dict[str, list] = {}
    for r in pq:
        tt = q2tt.get(r.get("query", ""), "?")
        per.setdefault(tt, []).append(
            (float(r.get("detector_score", 0.0)), int(r.get("label", 0))))

    def auc(pairs):
        s = np.array([x[0] for x in pairs], dtype=float)
        y = np.array([x[1] for x in pairs], dtype=int)
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        c = sum(1 for a in pos for b in neg if a > b)
        c += 0.5 * sum(1 for a in pos for b in neg if a == b)
        return float(c / (len(pos) * len(neg)))

    return {tt: {"n": len(p), "roc_auc": round(auc(p), 4)}
            for tt, p in sorted(per.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="data/metrics_ragtruth_fitted.json")
    ap.add_argument("--anchors", default="data/published_anchors.json")
    ap.add_argument("--out", default="data/comparison_to_published.json")
    args = ap.parse_args()

    m = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    agg = m.get("aggregate", {}) or {}
    a = load_anchors(args.anchors)

    our_auc = agg.get("detection_roc_auc")
    our_ci = agg.get("detection_roc_auc_ci95") or []
    our_lo = our_ci[0] if our_ci else None

    # Per-task breakdown. The published AUROC anchors are pooled averages
    # with no per-task split, so the per-task numbers are reported for
    # transparency about where the signal comes from -- they are not
    # compared against the pooled anchors.
    per_task = _per_task_auc(m)

    report: dict = {
        "source_metrics": args.metrics,
        "n_queries": agg.get("n_queries"),
        "ours": {
            "detection_roc_auc": our_auc,
            "detection_roc_auc_ci95": our_ci,
            "per_task_roc_auc": per_task,
        },
        "auroc_comparison": [],
        "f1_comparison": [],
        "not_reported": a.get("not_reported_on_ragtruth", []),
        "scope_note": (
            "The LLM-Check anchors are the 'Overall' column of its Table 4: "
            "a weighted average across the target models (Llama-2-13b, "
            "Llama-2-70b, GPT-4, Mistral-7b). It reports no per-task "
            "breakdown, so the only like-for-like comparison is our pooled "
            "number against its pooled number. Our pooled number is over "
            "1000 rows drawn equally from the three RAGTruth task types."
        ),
    }

    # --- the AUROC family: the only family we can compare against ---------
    print(f"\nOurs: detection AUROC {our_auc} CI {our_ci}\n")
    print("AUROC comparison (RAGTruth) — a win needs our CI lower bound "
          "strictly above the published value:")
    for anc in a["auroc_family"]["anchors"]:
        pub = anc["value"]
        if our_lo is None:
            verdict = "no CI"
        elif our_lo > pub:
            verdict = "WIN"
        elif (our_auc or 0) > pub:
            verdict = "tie (interval overlaps)"
        else:
            verdict = "loss"
        row = {**anc, "our_auc": our_auc, "our_ci95": our_ci,
               "our_ci_lower": our_lo, "verdict": verdict}
        report["auroc_comparison"].append(row)
        print(f"  {anc['system']:42s} pub={pub:.4f}  ours={our_auc:.4f} "
              f"CI[{our_ci[0]:.4f},{our_ci[1]:.4f}]  -> {verdict}")

    if per_task:
        print("\nOur per-task ROC-AUC (the published anchors are pooled "
              "averages with no per-task split, so these are for "
              "transparency, not a separate comparison):")
        for tt, v in per_task.items():
            print(f"  {tt:42s} n={v['n']:4d} AUC={v['roc_auc']:.4f}")

    # --- the F1 family: reported separately, never mixed with AUROC -------
    print("\nF1 comparison (RAGTruth) — reported for context. Our pipeline "
          "produces a detection AUROC, not a response-level F1 on the "
          "paper's protocol, so these are not a like-for-like test and are "
          "not scored as wins or losses.")
    report["f1_family_note"] = (
        "The RAGTruth paper reports response-level detection F1. This "
        "pipeline's headline detection metric is AUROC on a "
        "length-matched, instance-paired split of the RAGTruth test data. "
        "The two are not directly comparable; the F1 anchors are recorded "
        "for completeness only."
    )
    for anc in a["f1_family"]["anchors"]:
        report["f1_comparison"].append(anc)

    # --- the span family: a negative result -------------------------------
    print("\nSpan-level F1 (RAGTruth): this pipeline produces no span-level "
          "predictions, so there is nothing to compare. Recorded as a "
          "negative result.")
    report["span_f1_family"] = a.get("span_f1_family", {})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
