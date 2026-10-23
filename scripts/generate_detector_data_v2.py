"""Generate real labelled training data for the XGBoost detector (Step 3).

For HaluEval, each row has both a right_answer and a hallucinated answer.
This script bypasses the pipeline's LLM generation and instead feeds the
actual ground-truth answer directly into Step 3's feature extractor, so
the features genuinely differ between faithful and hallucinated examples.

Produces (cosine, claim_count, intrinsic_overlap, answer_length, label)
rows suitable for XGBClassifier.

Usage:
  python -m scripts.generate_detector_data_v2 --dataset halueval --limit 500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from src.config import load_config
from src.models.embedder_bge_m3 import BGE_M3Embedder
from src.utils.data_loader import load_jsonl
from src.utils.claim_extractor import extract_claims, intrinsic_overlap


_DATASET_PATHS = {
    "halueval": "data/halueval_qa.jsonl",
    "truthfulqa": "data/truthfulqa_qa.jsonl",
}


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _compute_features(embedder: BGE_M3Embedder, query: str, answer: str) -> dict:
    """Compute the 4 Step-3 features for a (query, answer) pair."""
    import numpy as np

    # Cosine similarity via BGE-M3 embeddings.
    try:
        qv = embedder.encode([query])[0]
        av = embedder.encode([answer])[0]
        cosine = float(
            np.dot(qv, av) / (np.linalg.norm(qv) * np.linalg.norm(av) + 1e-12)
        )
    except Exception:
        cosine = 0.0

    # Claim count via rule-based sentence splitter.
    claim_texts = extract_claims(answer or "", mode="rule")
    claim_count = max(1, len(claim_texts))

    # Intrinsic overlap (Jaccard on word sets).
    overlap = intrinsic_overlap(answer or "", query)

    # Answer length in tokens.
    answer_length = len((answer or "").split())

    return {
        "cosine": cosine,
        "claim_count": float(claim_count),
        "intrinsic_overlap": float(overlap),
        "answer_length": float(answer_length),
    }


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

    # Load only the embedder (no LLM / NLI / reranker needed).
    cfg = load_config()
    embedder = BGE_M3Embedder(model_name=(cfg.models or {}).get("embedder", "BAAI/bge-m3"))
    embedder._load_hf()

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

            # HaluEval provides both right_answer and the hallucinated
            # answer. Emit one row per side so the model sees both classes.
            right = (r.get("right_answer") or "").strip()
            hallucinated = (r.get("answer") or "").strip()

            pairs = []
            if right:
                pairs.append((0, right))         # label 0 = faithful
            if hallucinated and hallucinated != right:
                pairs.append((1, hallucinated))  # label 1 = hallucinated

            if not pairs:
                skipped += 1
                continue

            for label_int, answer_text in pairs:
                try:
                    feats = _compute_features(embedder, query, answer_text)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [row {idx}] feature failed: {exc}", file=sys.stderr)
                    skipped += 1
                    continue

                row_out = {
                    "query": query,
                    "label": label_int,
                    "features": [
                        feats["cosine"],
                        feats["claim_count"],
                        feats["intrinsic_overlap"],
                        feats["answer_length"],
                    ],
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
