"""Generate the detector's training data, free of label shortcuts.

Why this file exists
--------------------
The shipped ``data/detector_train.jsonl`` was mined with
``scripts/generate_detector_data_v2.py`` over the first 500 rows of
``data/halueval_qa.jsonl``. HaluEval stores one row per side of a pair, so
that slice covers ~250 unique queries while the generator emitted a row for
every *input* row. Two shortcuts survived into the training file:

  1. **Duplicated gold.** The short gold answer was written once as its own
     input row and again as the ``--balance`` control, so the faithful
     class carried that example with double weight.

  2. **A length-correlated overlap.** The length-matched faithful control is
     padded with words taken from the dataset's ``knowledge`` passage. Those
     words are not in the query, so the control's lexical overlap with the
     query falls below the hallucinated row's for the same query. Measured
     over the shipped file: 78% of queries show this gap, mean 0.217, and
     ``intrinsic_overlap`` correlates with ``answer_length`` at r = 0.67.

Both shortcuts let the detector reach a high training score without learning
anything about hallucination, and both are invisible in the headline metric
because the length-matched *evaluation* split removes them.

This script emits, per query, exactly one faithful row and one
hallucinated row, both length-matched, so that neither ``answer_length`` nor
``intrinsic_overlap`` can act as a label proxy. The short gold answer is
deliberately *not* emitted: keeping it would re-introduce the length
shortcut that the evaluation split was built to remove.

The same feature definitions are used at training and at inference time
(``src/steps/step03_early_detection.py``), so the detector is never served a
feature it was not fitted on.

Usage:
  python -m scripts.generate_detector_data_v4 --limit 1000
  python -m scripts.generate_detector_data_v4 --dataset truthfulqa --limit 1000
  python -m scripts.generate_detector_data_v4 --out data/detector_train_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scripts.generate_detector_data_v2 import _length_matched_control
from src.config import load_config
from src.models.embedder_bge_m3 import BGE_M3Embedder
from src.utils.data_loader import load_jsonl

_DATASET_PATHS = {
    "halueval": "data/halueval_qa.jsonl",
    "truthfulqa": "data/truthfulqa_qa.jsonl",
}

# The evaluation splits are carved out of the same source files as the
# training data, so a training query can coincide with an evaluation query
# unless it is explicitly excluded. Every query appearing in any of these
# files is dropped from the training set.
_EVAL_SPLITS_TO_EXCLUDE = (
    "data/halueval_eval_1000.jsonl",
    "data/halueval_eval_1000_matched.jsonl",
    "data/truthfulqa_eval_1000.jsonl",
)


def _excluded_queries() -> set[str]:
    """Queries that appear in any evaluation split."""
    excluded: set[str] = set()
    for path in _EVAL_SPLITS_TO_EXCLUDE:
        p = Path(path)
        if not p.exists():
            continue
        for r in load_jsonl(path):
            q = (r.get("query") or r.get("question") or "").strip()
            if q:
                excluded.add(q)
    return excluded


def _group_by_query(rows: list[dict]) -> list[tuple[str, str, str, str]]:
    """Reduce the dataset to one (query, gold, hallucinated, knowledge) tuple.

    HaluEval stores both sides of a pair as separate rows, so the same query
    appears twice. Collapsing on the query is what removes the duplicated
    gold row from the training file.
    """
    merged: dict[str, dict] = {}
    for r in rows:
        query = (r.get("query") or r.get("question") or "").strip()
        if not query:
            continue
        entry = merged.setdefault(query, {
            "query": query,
            "gold": "",
            "hallucinated": "",
            "knowledge": r.get("knowledge") or "",
        })
        label = int(r.get("label", 0))
        answer = (r.get("answer") or "").strip()
        if label == 1 and answer and not entry["hallucinated"]:
            entry["hallucinated"] = answer
        elif label == 0 and answer and not entry["gold"]:
            entry["gold"] = answer
        if r.get("right_answer"):
            entry["gold"] = entry["gold"] or r["right_answer"].strip()
        if not entry["knowledge"] and r.get("knowledge"):
            entry["knowledge"] = r["knowledge"]

    out = []
    for e in merged.values():
        if e["gold"] and e["hallucinated"] and e["gold"] != e["hallucinated"]:
            out.append((e["query"], e["gold"], e["hallucinated"], e["knowledge"]))
    return out


def _compute_features(embedder, query: str, answer: str) -> list[float]:
    """The four Step-3 features, using the same code path as inference."""
    import numpy as np

    from src.utils.claim_extractor import extract_claims, intrinsic_overlap

    cosine = 0.0
    try:
        qv = embedder.encode([query])[0]
        av = embedder.encode([answer])[0]
        cosine = float(
            np.dot(qv, av) / (np.linalg.norm(qv) * np.linalg.norm(av) + 1e-12)
        )
    except Exception:  # noqa: BLE001
        cosine = 0.0

    claim_count = float(max(1, len(extract_claims(answer or "", mode="rule"))))
    overlap = intrinsic_overlap(answer or "", query)
    answer_length = float(len((answer or "").split()))
    return [cosine, claim_count, overlap, answer_length]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(_DATASET_PATHS))
    ap.add_argument("--data", help="custom JSONL path (overrides --dataset)")
    ap.add_argument("--limit", type=int, default=1000,
                    help="number of unique queries to use")
    ap.add_argument("--out", default="data/detector_train_clean.jsonl")
    ap.add_argument("--allow-eval-overlap", action="store_true",
                    help="do not exclude queries that appear in an evaluation "
                         "split (off by default; the training and evaluation "
                         "sets are carved from the same source file)")
    ap.add_argument("--device", default=None,
                    help="device for the embedder (default: auto). Pass 'cpu' "
                         "to avoid colliding with a running GPU evaluation.")
    args = ap.parse_args()

    path = args.data or _DATASET_PATHS[args.dataset]
    rows = load_jsonl(path)
    groups = _group_by_query(rows)
    print(f"{len(rows)} rows -> {len(groups)} unique queries", flush=True)

    excluded = set()
    if not args.allow_eval_overlap:
        excluded = _excluded_queries()
        if excluded:
            before = len(groups)
            groups = [g for g in groups if g[0] not in excluded]
            print(f"excluded {before - len(groups)} queries that appear in an "
                  f"evaluation split ({len(excluded)} unique)", flush=True)

    groups = groups[: args.limit]
    if not groups:
        raise SystemExit("no usable queries found")

    cfg = load_config()
    embedder = BGE_M3Embedder(
        model_name=(cfg.models or {}).get("embedder", "BAAI/bge-m3"),
        device=args.device,
    )
    embedder._load_hf()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    n_control = 0
    n_no_control = 0
    t_start = time.time()

    with out_path.open("w", encoding="utf-8") as fh:
        for i, (query, gold, hallucinated, knowledge) in enumerate(groups):
            control = _length_matched_control(query, gold, hallucinated, knowledge)
            faithful = control or ""
            if not faithful:
                # No length-matched control could be built for this query.
                # Emitting the short gold instead would put the length
                # shortcut back in, so the query is dropped entirely.
                n_no_control += 1
                continue

            for label, answer in ((0, faithful), (1, hallucinated)):
                feats = _compute_features(embedder, query, answer)
                fh.write(json.dumps({
                    "query": query,
                    "label": label,
                    "features": feats,
                }) + "\n")
                written += 1
            n_control += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(groups):
                elapsed = time.time() - t_start
                avg = elapsed / (i + 1)
                print(f"  [{i+1}/{len(groups)}] avg={avg:.2f}s "
                      f"eta={avg * (len(groups) - i - 1):.0f}s written={written}",
                      flush=True)

    # Report the shortcut checks on what was actually written.
    rows_out = [json.loads(line) for line in
                out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lens = [r["features"][3] for r in rows_out]
    ovl = [r["features"][2] for r in rows_out]
    labels = [r["label"] for r in rows_out]
    import numpy as np
    med = lambda xs: float(np.median(xs)) if xs else 0.0
    corr = (float(np.corrcoef(ovl, lens)[0, 1])
            if len(rows_out) > 2 else 0.0)
    print(f"\nwrote {out_path} ({written} rows, {n_control} queries)")
    print(f"  queries dropped (no length-matched control): {n_no_control}")
    print(f"  labels: {labels.count(0)} zero / {labels.count(1)} one")
    print(f"  answer_length median: label0={med([l for l, y in zip(lens, labels) if y == 0]):.1f} "
          f"label1={med([l for l, y in zip(lens, labels) if y == 1]):.1f}")
    print(f"  overlap/length pearson r = {corr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
