"""CLI: build a FAISS index over a corpus.

Usage:
  python scripts/build_index.py --corpus synthetic --out data/index
  python scripts/build_index.py --corpus halueval --data data/halueval.json --out data/index
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/build_index.py` to import the `src` package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_loader import corpus_from_dataset, load_dataset
from src.embeddings import Embedder


def main() -> None:
    p = argparse.ArgumentParser(description="Build a CRISP retriever index.")
    p.add_argument("--corpus", required=True, choices=["synthetic", "halueval"],
                   help="Which dataset to build the index from.")
    p.add_argument("--data", default=None,
                   help="Path to a HaluEval JSON/JSONL file (only used with --corpus halueval).")
    p.add_argument("--out", required=True, help="Output directory for the index.")
    p.add_argument("--config", default=None, help="Optional path to a YAML config.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    logger = logging.getLogger("build_index")

    if args.corpus == "halueval":
        items = load_dataset("halueval", path=args.data)
    else:
        items = load_dataset("synthetic")
    docs = corpus_from_dataset(items)
    if not docs:
        raise SystemExit(f"No documents found to index (corpus={args.corpus}).")

    embedder = Embedder(
        model_name=cfg.retrieval.embedding_model,
        mock=cfg.mock,
    )
    # Lazy import to keep top-level clean.
    from src.retriever import Retriever
    r = Retriever(embedder=embedder, top_k=cfg.retrieval.top_k).build(docs)
    r.save(args.out)
    logger.info("Saved index with %d docs to %s", len(docs), args.out)


if __name__ == "__main__":
    main()