"""CLI: run RAG inference on a dataset.

Usage:
  python scripts/run_inference.py --dataset synthetic --index data/index --out data/preds.jsonl
  python scripts/run_inference.py --dataset halueval --data data/halueval.json --index data/index --out data/preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_loader import load_dataset
from src.pipeline import Pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Run CRISP RAG inference.")
    p.add_argument("--dataset", required=True, choices=["synthetic", "halueval"])
    p.add_argument("--data", default=None, help="Path to HaluEval file (only for halueval).")
    p.add_argument("--index", required=True, help="Directory of a saved retriever index.")
    p.add_argument("--out", required=True, help="Output JSONL path.")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of items.")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("run_inference")

    cfg = load_config(args.config)
    items = load_dataset(args.dataset, path=args.data)
    if args.limit:
        items = items[: args.limit]
    logger.info("Loaded %d items.", len(items))

    pipeline = Pipeline(cfg).load_index(args.index)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            result = pipeline.run(item["question"])
            # Spread the full RAGResult dict, then add the dataset-specific
            # id / question / reference fields on top.
            row = result.to_dict()
            row.update({
                "id": item["id"],
                "question": item["question"],
                "reference_answer": item.get("reference_answer"),
            })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(items):
                logger.info("Processed %d / %d  (%.1fs elapsed)",
                            i, len(items), time.perf_counter() - t0)
    logger.info("Wrote predictions to %s (%.1fs total).",
                out_path, time.perf_counter() - t0)


if __name__ == "__main__":
    main()