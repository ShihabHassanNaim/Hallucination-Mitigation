"""CLI: evaluate a predictions JSONL file.

Usage:
  python scripts/evaluate.py --preds data/preds.jsonl --out data/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import aggregate, load_predictions, score_item, write_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CRISP predictions.")
    p.add_argument("--preds", required=True, help="JSONL of predictions (from run_inference.py).")
    p.add_argument("--out", required=True, help="Output JSON metrics path.")
    args = p.parse_args()

    items = load_predictions(args.preds)
    per_item = [score_item(it) for it in items]
    metrics = aggregate(per_item)
    write_metrics(metrics, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()