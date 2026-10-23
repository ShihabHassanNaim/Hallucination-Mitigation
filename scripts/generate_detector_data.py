"""Generate real labelled training data for the XGBoost detector (Step 3).

Runs the pipeline on HaluEval queries (which have hallucination labels),
extracts Step-3 features (cosine, claim_count, intrinsic_overlap, answer_length),
and saves them as a JSONL training set for XGBClassifier.

Usage:
  python -m scripts.generate_detector_data --dataset halueval --limit 500
  python -m scripts.generate_detector_data --dataset truthfulqa --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_jsonl, load_synthetic_qa


_DATASET_PATHS = {
    "halueval": "data/halueval_qa.jsonl",
    "truthfulqa": "data/truthfulqa_qa.jsonl",
}


def _build_pipeline() -> Pipeline:
    """Build a pipeline with all real HF models."""
    return Pipeline(cfg=load_config())


def _extract_features(result) -> dict:
    """Pull Step-3 features out of a PipelineResult.

    The pipeline exposes the per-step trace payloads via result.step_traces.
    Step 3 ('early_detection') writes its feature dict under that key.
    """
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
    # Fallback: synthesise zeros if trace not found.
    return {"cosine": 0.0, "claim_count": 0.0, "intrinsic_overlap": 0.0, "answer_length": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        choices=sorted(_DATASET_PATHS),
        help="Labelled dataset to mine features from.",
    )
    ap.add_argument("--data", help="Custom JSONL path (overrides --dataset).")
    ap.add_argument("--synth", action="store_true", help="Use the synthetic QA set.")
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

    if args.synth:
        rows = load_synthetic_qa()
    elif args.data:
        rows = load_jsonl(args.data)
    elif args.dataset:
        rows = load_jsonl(_DATASET_PATHS[args.dataset])
    else:
        rows = load_synthetic_qa()

    rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows.", flush=True)

    pipe = _build_pipeline()
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
            try:
                result = pipe.run(query)
            except Exception as exc:  # noqa: BLE001
                print(f"  [row {idx}] pipeline failed: {exc}", file=sys.stderr)
                skipped += 1
                continue

            features = _extract_features(result)

            # Resolve label. HaluEval uses 0/1 ints; TruthfulQA uses strings.
            label = r.get("label", "")
            if label in (0, 1, "0", "1"):
                label_int = int(label)
            else:
                # No reliable label — skip.
                skipped += 1
                continue

            row_out = {
                "query": query,
                "label": label_int,
                "features": [
                    features["cosine"],
                    features["claim_count"],
                    features["intrinsic_overlap"],
                    features["answer_length"],
                ],
                "decision": result.decision,
                "confidence": float(result.confidence),
            }
            f.write(json.dumps(row_out) + "\n")
            written += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
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
