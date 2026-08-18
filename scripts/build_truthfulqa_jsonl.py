"""One-shot conversion of TruthfulQA CSV to labelled JSONL.

Schema per line (matches load_halueval / scripts/evaluate.py):
  {
    "question":      str,
    "query":         str,   # duplicate for evaluate.py
    "right_answer":  str,   # truthful reference
    "knowledge":     str,   # the truth context (Best Answer + Correct Answers)
    "answer":        str,   # the model answer to evaluate (truthful or hallucinated)
    "evidence":      [str], # same as knowledge in a list
    "label":         int    # 0 = truthful, 1 = hallucinated
  }

Per TruthfulQA source row we emit two output rows:
  - one truthful   (answer=Best Answer,           label=0)
  - one hallucinated (answer=Best Incorrect Answer, label=1)

This makes the dataset usable both for retrieval-vs-truth grounding and
for detection P/R/F1 / ROC-AUC.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input TruthfulQA CSV")
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()

    src = Path(args.csv)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_truth = n_hallu = n_skip = 0
    with src.open("r", encoding="utf-8", newline="") as f_in, dst.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            q = (row.get("Question") or "").strip()
            ba = (row.get("Best Answer") or "").strip()
            bia = (row.get("Best Incorrect Answer") or "").strip()
            ca = (row.get("Correct Answers") or "").strip()
            if not q or not ba or not bia:
                n_skip += 1
                continue
            extras = [x.strip() for x in ca.split(";") if x.strip() and x.strip() != ba]
            knowledge = "; ".join([ba] + extras)

            common = {
                "question": q,
                "query": q,
                "right_answer": ba,
                "knowledge": knowledge,
                "evidence": [knowledge] if knowledge else [],
            }

            f_out.write(json.dumps({**common, "answer": ba, "label": 0}) + "\n")
            n_truth += 1

            f_out.write(json.dumps({**common, "answer": bia, "label": 1}) + "\n")
            n_hallu += 1

    print(f"truthful rows: {n_truth}, hallucinated rows: {n_hallu}, skipped: {n_skip}, total: {n_truth + n_hallu}")
    print(f"wrote: {dst}")


if __name__ == "__main__":
    main()