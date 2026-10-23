"""Generate real labelled training data for the XGBoost detector (Step 3).

The HaluEval dataset has queries where the same question can be answered
either faithfully (label 0) or with a hallucinated answer (label 1). The
pipeline generates its own answer regardless, so the features at test time
come from the pipeline's answer, not the ground truth.

This script runs the full pipeline on each HaluEval query and labels the
extracted Step-3 features with the dataset's hallucination label. Queries
that tend to elicit hallucinations (label 1) will have systematically
different features than faithful ones (label 0), and XGBoost can learn
this pattern.

Usage:
  python -m scripts.generate_detector_data_v3 --dataset halueval --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_jsonl


_DATASET_PATHS = {
    "halueval": "data/halueval_qa.jsonl",
    "truthfulqa": "data/truthfulqa_qa.jsonl",
}


def _extract_features(result) -> dict:
    """Pull Step-3 features from the pipeline result's step_traces."""
    for trace in result.step_traces:
        if trace.name == "early_detection":
            payload = trace.payload or {}
            features = payload.get("features", {})
            return {
                "cosine": float(features.get("cosine", 0.0)),
                "claim_count": float(features.get("claim_count", 0.0)),
                "intrinsic_overlap": float(features.get("intrinsic_overlap", 0.0)),
                "answer_length": float(features.get("answer_length", 0.0)),
            }
    return {"cosine": 0.0, "claim_count": 0.0, "intrinsic_overlap": 0.0, "answer_length": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        choices=sorted(_DATASET_PATHS),
        help="Labelled dataset to mine features from.",
    )
    ap.add_argument("--data", help="Custom JSONL path (overrides --dataset).")
    ap.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of rows to process (default 500).",
    )
    ap.add_argument(
        "--out",
        default="data/detector_train.jsonl",
        help="Output JSONL path (default data/detector_train.jsonl).",
    )
    args = ap.parse_args()

    if args.data:
        rows = load_jsonl(args.data)
    else:
        rows = load_jsonl(_DATASET_PATHS[args.dataset])

    rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows.", flush=True)

    pipe = Pipeline(cfg=load_config())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    t_start = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(rows):
            query = r.get("query") or r.get("question") or ""
            if not query:
                skipped += 1
                continue

            # HaluEval: label 0 = faithful row, label 1 = hallucinated row
            # for the SAME query. Deduplicate so each unique query appears once.
            label = r.get("label", "")
            if label not in (0, 1, "0", "1"):
                skipped += 1
                continue
            label_int = int(label)

            # Skip duplicates: HaluEval has each query twice (once per label)
            # Track queries we've seen and only keep the first occurrence.
            # We do this by checking if we already wrote this query.
            existing_queries = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    try:
                        existing_queries.add(json.loads(line)["query"])
                    except Exception:
                        pass
            if query in existing_queries:
                skipped += 1
                continue

            try:
                result = pipe.run(query)
            except Exception as exc:  # noqa: BLE001
                print(f"  [row {idx}] pipeline failed: {exc}", file=sys.stderr)
                skipped += 1
                continue

            features = _extract_features(result)

            row_out = {
                "query": query,
                "label": label_int,
                "features": [
                    features["cosine"],
                    features["claim_count"],
                    features["intrinsic_overlap"],
                    features["answer_length"],
                ],
            }
            f.write(json.dumps(row_out) + "\n")
            written += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
                elapsed = time.time() - t_start
                avg = elapsed / (idx + 1)
                eta = avg * (len(rows) - idx - 1)
                print(
                    f"  [{idx+1}/{len(rows)}] avg={avg:.1f}s eta={eta:.0f}s "
                    f"written={written} skipped={skipped}",
                    flush=True,
                )

    print(f"wrote {out_path} ({written} rows, {skipped} skipped)")


if __name__ == "__main__":
    main()
