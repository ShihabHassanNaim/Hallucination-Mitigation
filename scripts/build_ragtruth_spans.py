"""Extract RAGTruth's span-level hallucination annotations.

RAGTruth's ``hallucination_labels`` column is a JSON string holding a list
of annotated spans, each with ``start``/``end`` character offsets into the
``output`` column and a ``label_type``. This script turns that into a
plain JSONL keyed by the dataset's row id, which is what
``scripts/score_span_level.py`` reads.

The span offsets are relative to the ``output`` field, so the row id has to
be the same one the RAGTruth dataset builder writes into the evaluation
file. The evaluation file's ``row_id`` is RAGTruth's own ``id`` column, so
the join is exact.

Usage:
  python -m scripts.build_ragtruth_spans
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.data_loader import load_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="data/ragtruth_test.parquet")
    ap.add_argument("--eval", default="data/ragtruth_eval_1000_matched.jsonl",
                    help="the evaluation file, used to keep only the row ids "
                         "that the evaluation actually uses")
    ap.add_argument("--out", default="data/ragtruth_span_labels.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df[df["quality"] == "good"]

    wanted = {r["row_id"] for r in load_jsonl(args.eval)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_spans = 0
    with out.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            rid = str(r["id"])
            if rid not in wanted:
                continue
            raw = r["hallucination_labels"]
            spans: list[dict] = []
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except (TypeError, ValueError):
                parsed = []
            for s in parsed or []:
                try:
                    start = int(s["start"])
                    end = int(s["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if end <= start:
                    continue
                spans.append({
                    "start": start,
                    "end": end,
                    "text": str(s.get("text", "")),
                    "label_type": str(s.get("label_type", "")),
                })
            f.write(json.dumps({"row_id": rid, "spans": spans},
                               ensure_ascii=False) + "\n")
            n_rows += 1
            n_spans += len(spans)

    print(f"wrote {out}: {n_rows} rows, {n_spans} gold spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
