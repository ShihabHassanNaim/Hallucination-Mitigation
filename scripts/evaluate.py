"""Aggregate metrics across queries and emit the evaluation report.

Usage:
  python -m scripts.evaluate --data data/synth.jsonl --out data/metrics.json
  python -m scripts.evaluate --dataset halueval    --out data/metrics_halueval.json
  python -m scripts.evaluate --dataset truthfulqa  --out data/metrics_truthfulqa.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_jsonl, load_synthetic_qa


_METRIC_NUMERIC_KEYS = [
    "claim_precision", "claim_recall", "claim_f1",
    "retrieval_recall_at_5", "retrieval_precision_at_5",
    "retrieval_mrr", "retrieval_ndcg",
    "verification_success_rate", "end_to_end_accuracy",
    "confidence",
]


# Default data file → default output filename for each registered dataset.
# The frontend's ``src.evaluation.report.DATASET_REGISTRY`` expects the
# output filenames listed below, so keep them in sync.
_DATASET_DEFAULTS: dict[str, dict[str, str]] = {
    "halueval": {
        "data": "data/halueval_qa.jsonl",
        "out": "data/metrics_halueval.json",
    },
    "truthfulqa": {
        "data": "data/truthfulqa_qa.jsonl",
        "out": "data/metrics_truthfulqa.json",
    },
}


def _resolve_dataset_rows(args: argparse.Namespace) -> tuple[list[dict], str]:
    """Return (rows, dataset_slug_or_label) honoring --data/--synth/--dataset."""
    if args.synth:
        rows = load_synthetic_qa()
        slug = "synth"
    elif args.dataset:
        slug = args.dataset.strip().lower()
        defaults = _DATASET_DEFAULTS.get(slug)
        if defaults is None:
            raise SystemExit(
                f"unknown --dataset {args.dataset!r}; "
                f"supported: {sorted(_DATASET_DEFAULTS)} (or pass --data)"
            )
        data_path = args.data or defaults["data"]
        rows = load_jsonl(data_path)
        if not rows:
            raise SystemExit(
                f"no rows loaded from {data_path}; download/build the dataset first"
            )
        rows = rows  # keep named for clarity below
        slug = slug
    elif args.data:
        rows = load_jsonl(args.data)
        slug = Path(args.data).stem
    else:
        rows = load_synthetic_qa()
        slug = "synth"

    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]
    return rows, slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="JSONL of {query, answer, evidence:[ids], label}")
    ap.add_argument("--synth", action="store_true", help="use built-in synthetic QA")
    ap.add_argument(
        "--dataset",
        choices=sorted(_DATASET_DEFAULTS),
        help="named dataset shortcut (halueval | truthfulqa); uses the "
             "default JSONL path under data/ and writes to "
             "data/metrics_<slug>.json unless --out is overridden.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: data/metrics.json for --synth, "
             "data/metrics_<slug>.json for --dataset)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N rows (useful for smoke tests)",
    )
    ap.add_argument(
        "--index",
        default=None,
        help="path to a saved FAISS+BM25 index directory (scripts/build_index format); "
             "skips the default synthetic-corpus index.",
    )
    args = ap.parse_args()

    rows, slug = _resolve_dataset_rows(args)
    if args.out is None:
        if slug in _DATASET_DEFAULTS:
            args.out = _DATASET_DEFAULTS[slug]["out"]
        else:
            args.out = "data/metrics.json"

    cfg = load_config()
    pipe = Pipeline(cfg=cfg)
    if args.index:
        pipe.load_index(args.index)

    per_query_metrics: list[dict] = []
    t_start = time.time()
    for idx, r in enumerate(rows):
        row_t = time.time()
        try:
            res = pipe.run(r["query"])
            d = res.to_dict()
            # If gold evidence provided, compute retrieval metrics against it.
            gold = [e for e in (r.get("evidence") or []) if e]
            retrieved = [e["doc_id"] for e in d["evidence"]]
            from src.steps.step13_evaluation import (
                _recall_at_k, _precision_at_k, _mrr, _ndcg,
            )
            per_query_metrics.append({
                "query": r["query"],
                "decision": d["decision"],
                "confidence": d["confidence"],
                "retrieval_recall_at_5": _recall_at_k(retrieved, gold, 5),
                "retrieval_precision_at_5": _precision_at_k(retrieved, gold, 5),
                "retrieval_mrr": _mrr(retrieved, gold),
                "retrieval_ndcg": _ndcg(retrieved, gold),
                "claim_precision": sum(c["verdict"] == "supported" for c in d["claims"]) / max(1, len(d["claims"])),
                "verification_success_rate": 1.0 if all(c["verdict"] != "contradicted" for c in d["claims"]) else 0.0,
                "end_to_end_accuracy": 1.0 if d["decision"] == "PASS" else 0.0,
            })
        except Exception as exc:  # noqa: BLE001
            per_query_metrics.append({"query": r.get("query"), "error": str(exc)})
        # Progress line every 10 rows or on the last row.
        if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
            elapsed = time.time() - t_start
            row_dt = time.time() - row_t
            avg = elapsed / (idx + 1)
            eta = avg * (len(rows) - idx - 1)
            print(
                f"  [{idx+1}/{len(rows)}] row_dt={row_dt:.1f}s avg={avg:.1f}s eta={eta:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    # Aggregate.
    aggregate: dict[str, float] = {}
    for k in _METRIC_NUMERIC_KEYS:
        vs = [m[k] for m in per_query_metrics if k in m and isinstance(m[k], (int, float))]
        if vs:
            aggregate[f"mean_{k}"] = statistics.fmean(vs)
            aggregate[f"stdev_{k}"] = statistics.pstdev(vs) if len(vs) > 1 else 0.0

    out = {
        "n_queries": len(per_query_metrics),
        "per_query": per_query_metrics,
        "aggregate": aggregate,
        "dataset": slug,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({slug}, {len(per_query_metrics)} rows)")
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
