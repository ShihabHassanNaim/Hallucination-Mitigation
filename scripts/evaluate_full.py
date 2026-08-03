"""Step 13 CLI — full evaluation over a predictions JSONL file.

Usage
-----
    python scripts/evaluate_full.py --preds data/preds.jsonl --out data/metrics_full.json

Also writes a Markdown summary next to the JSON file when ``--md`` is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the project root importable when invoked as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation_metrics import (  # noqa: E402
    evaluate_predictions,
    load_predictions,
    render_markdown_summary,
    write_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, help="Path to preds.jsonl")
    parser.add_argument("--out", required=True, help="Path to write metrics JSON")
    parser.add_argument("--md", action="store_true",
                        help="Also write a Markdown summary next to --out")
    parser.add_argument("--k", type=int, default=5,
                        help="K for Recall@K / Precision@K / nDCG@K")
    args = parser.parse_args()

    if not os.environ.get("CRISP_MOCK"):
        # Mock mode is irrelevant for evaluation (no models run), but keep
        # the project-wide convention consistent.
        os.environ["CRISP_MOCK"] = "1"

    preds = load_predictions(args.preds)
    print(f"[evaluate_full] loaded {len(preds)} predictions from {args.preds}")

    bundle = evaluate_predictions(preds, k_for_retrieval=args.k)
    metrics = bundle.to_dict()
    write_metrics(metrics, args.out)
    print(f"[evaluate_full] wrote {args.out}")

    # Console summary (one line per section).
    for section, payload in metrics.items():
        flat = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
        print(f"  {section}: {flat}")

    if args.md:
        md_path = Path(args.out).with_suffix(".md")
        md_path.write_text(render_markdown_summary(metrics), encoding="utf-8")
        print(f"[evaluate_full] wrote {md_path}")


if __name__ == "__main__":
    main()
