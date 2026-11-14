"""Build a length-matched evaluation split for HaluEval.

Why this exists
---------------
HaluEval pairs a *short* correct answer (label 0, typically two or three
words — "Suffolk County") with a *long* hallucinated sentence (label 1,
typically eleven words — "Lindenhurst is a village in a suburban county on
Long Island..."). A detector can therefore separate the two classes using
``answer_length`` alone, with no understanding of the answer at all. On the
raw split, answer length alone is 94.7% accurate.

The detector is trained with length-matched faithful controls (see
``scripts/generate_detector_data_v2.py``), so it does not rely on that
shortcut. But the evaluation set contains no such controls, which means a
near-perfect ROC-AUC on the raw split measures answer verbosity rather
than hallucination detection.

This script adds, for each query, a *faithful control*: the correct answer
expanded into a full sentence using the dataset's supporting ``knowledge``
passage, matched to the hallucinated answer's word count. The control is
built with the very same helper used to build the training controls, so
the construction is identical on both sides.

The output keeps the original rows and appends the controls, giving a
balanced split in which both classes look like generated prose.

Usage:
  python -m scripts.build_length_matched_split
  python -m scripts.build_length_matched_split --limit 500
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from scripts.generate_detector_data_v2 import _length_matched_control
from src.utils.data_loader import load_jsonl


def _length_only_accuracy(rows: list[dict]) -> float:
    """Accuracy obtainable from answer word count alone."""
    lengths = [len(str(r.get("answer") or "").split()) for r in rows]
    labels = [int(r["label"]) for r in rows]
    threshold = statistics.median(lengths)
    return sum(
        1 for ln, lb in zip(lengths, labels) if (ln > threshold) == bool(lb)
    ) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        default="data/halueval_eval_1000.jsonl",
        help="the split to length-match (default: data/halueval_eval_1000.jsonl)",
    )
    ap.add_argument(
        "--out",
        default="data/halueval_eval_1000_matched.jsonl",
        help="output path for the length-matched split",
    )
    ap.add_argument("--limit", type=int, default=None, help="only use the first N rows")
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows loaded from {args.data}")

    out: list[dict] = []
    n_control = 0
    n_failed = 0

    # The two rows of a query share the question but carry different
    # answers: the label-1 row holds the long hallucinated sentence and the
    # label-0 row holds the short correct one. A control must be matched to
    # the *hallucinated* answer of the same query, so index the rows first.
    hallucinated_by_query: dict[str, str] = {}
    for r in rows:
        if int(r.get("label", 0)) == 1:
            hallucinated_by_query.setdefault(
                r.get("query") or r.get("question") or "", (r.get("answer") or "").strip()
            )

    # The training split (data/detector_train.jsonl) gives each query one
    # hallucinated row and two faithful rows — the short gold answer and a
    # length-matched control. The evaluation split mirrors that composition
    # so the detector meets the same distribution it was trained on. Keeping
    # the original short gold rows would leave the length shortcut intact.
    for r in rows:
        query = r.get("query") or r.get("question") or ""
        if int(r.get("label", 0)) == 1:
            out.append(r)
            continue

        right = (r.get("right_answer") or "").strip()
        hallucinated = hallucinated_by_query.get(query, "")
        if not right or not hallucinated or right == hallucinated:
            # Cannot build a control — fall back to the original row so the
            # query is still represented.
            out.append(r)
            n_failed += 1
            continue
        control = _length_matched_control(
            query,
            right,
            hallucinated,
            r.get("knowledge") or "",
        )
        if not control:
            out.append(r)
            n_failed += 1
            continue
        out.append({
            **r,
            "answer": control,
            "label": 0,
            "row_type": "length_matched_control",
        })
        n_control += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [int(r["label"]) for r in out]
    print(f"wrote {out_path}")
    print(f"  rows in   : {len(rows)}")
    print(f"  controls  : {n_control} added ({n_failed} could not be matched)")
    print(f"  rows out  : {len(out)}")
    print(f"  labels    : {labels.count(0)} zero / {labels.count(1)} one")
    print(f"  length-only accuracy: {_length_only_accuracy(rows):.4f} "
          f"-> {_length_only_accuracy(out):.4f}")
    print(f"  unique queries: {len({r.get('query') for r in out})}")


if __name__ == "__main__":
    main()
