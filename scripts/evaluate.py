"""Aggregate metrics across queries and emit the evaluation report.

Usage:
  python -m scripts.evaluate --data data/synth.jsonl --out data/metrics.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_synthetic_qa, save_jsonl, load_jsonl


_METRIC_NUMERIC_KEYS = [
    "claim_precision", "claim_recall", "claim_f1",
    "retrieval_recall_at_5", "retrieval_precision_at_5",
    "retrieval_mrr", "retrieval_ndcg",
    "verification_success_rate", "end_to_end_accuracy",
    "confidence",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="JSONL of {query, answer, evidence:[ids], label}")
    ap.add_argument("--synth", action="store_true", help="use built-in synthetic QA")
    ap.add_argument("--out", default="data/metrics.json")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.synth:
        rows = load_synthetic_qa()
    elif args.data:
        rows = load_jsonl(args.data)
    else:
        rows = load_synthetic_qa()

    cfg = load_config()
    pipe = Pipeline(cfg=cfg)

    per_query_metrics: list[dict] = []
    for r in rows:
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
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
