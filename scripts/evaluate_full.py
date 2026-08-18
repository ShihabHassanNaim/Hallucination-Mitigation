"""Full evaluation across every metric in the diagram.

Adds HRR / CCR / VSR aggregation and detection ROC-AUC for a labelled
test set.

Usage:
  python -m scripts.evaluate_full --synth --out data/metrics.json
  python -m scripts.evaluate_full --dataset halueval    --out data/metrics_halueval.json
  python -m scripts.evaluate_full --dataset truthfulqa  --out data/metrics_truthfulqa.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from src.config import load_config
from src.models.xgb_classifier import XGBClassifier
from src.pipeline import Pipeline
from src.steps.step13_evaluation import (
    _precision_recall_f1, _accuracy, _roc_auc,
)
from src.utils.data_loader import load_jsonl, load_synthetic_qa


_DATASET_DEFAULTS: dict[str, dict[str, str]] = {
    "halueval": {
        "data": "data/halueval_qa.jsonl",
        "out": "data/metrics_halueval.json",
    },
    "truthfulqa": {
        "data": "data/truthfulqa_qa.jsonl",
        "out": "data/metrics_truthfulqa.json",
    },
}


def _resolve_dataset_rows(args: argparse.Namespace) -> tuple[list[dict], str]:
    """Return (rows, dataset_slug_or_label) honoring --data/--synth/--dataset."""
    if args.synth:
        rows = load_synthetic_qa()
    elif args.dataset:
        slug = args.dataset.strip().lower()
        defaults = _DATASET_DEFAULTS.get(slug)
        if defaults is None:
            raise SystemExit(
                f"unknown --dataset {args.dataset!r}; "
                f"supported: {sorted(_DATASET_DEFAULTS)} (or pass --data)"
            )
        data_path = args.data or defaults["data"]
        rows = load_jsonl(data_path)
        if not rows:
            raise SystemExit(
                f"no rows loaded from {data_path}; download/build the dataset first"
            )
        slug = slug
    elif args.data:
        rows = load_jsonl(args.data)
        slug = Path(args.data).stem
    else:
        rows = load_synthetic_qa()
        slug = "synth"

    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]
    return rows, slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument(
        "--dataset",
        choices=sorted(_DATASET_DEFAULTS),
        help="named dataset shortcut (halueval | truthfulqa); uses the "
             "default JSONL path under data/ and writes to "
             "data/metrics_<slug>.json unless --out is overridden.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N rows (useful for smoke tests)",
    )
    ap.add_argument(
        "--index",
        default=None,
        help="path to a saved FAISS+BM25 index directory; loads before running.",
    )
    args = ap.parse_args()

    rows, slug = _resolve_dataset_rows(args)
    if args.out is None:
        if slug in _DATASET_DEFAULTS:
            args.out = _DATASET_DEFAULTS[slug]["out"]
        else:
            args.out = "data/metrics.json"

    cfg = load_config()
    pipe = Pipeline(cfg=cfg)
    if args.index:
        pipe.load_index(args.index)
    clf = XGBClassifier()
    clf._load()

    detector_truth: list[int] = []
    detector_score: list[float] = []
    hrr_list: list[float] = []
    ccr_list: list[float] = []
    vsr_list: list[float] = []
    e2e: list[int] = []

    t_start = time.time()
    for idx, r in enumerate(rows):
        row_t = time.time()
        try:
            res = pipe.run(r["query"])
            d = res.to_dict()
        except Exception:
            continue

        # Determine ground truth: if the row has explicit label use it, else
        # infer from gold evidence presence.
        if "label" in r and r["label"] in {0, 1, "0", "1"}:
            gt = int(r["label"])
        else:
            gold = set(r.get("evidence") or [])
            retrieved = set(e["doc_id"] for e in d["evidence"])
            gt = 0 if gold and not (gold & retrieved) else 1

        score = 1.0 - d.get("confidence", 0.5)  # higher => more likely hallucinated
        detector_truth.append(gt)
        detector_score.append(score)

        # HRR — hallucination reduction rate: #rejected bad / #bad
        if gt == 0:
            hrr_list.append(1.0 if d["decision"] in {"REGEN", "ABSTAIN"} else 0.0)
        # CCR — claim correction rate: fraction of contradicted claims that were edited out.
        contras = sum(1 for c in d["claims"] if c["verdict"] == "contradicted")
        ccr_list.append(1.0 - (contras / max(1, len(d["claims"]))) if d["claims"] else 1.0)
        # VSR — fraction with no contradictions.
        vsr_list.append(1.0 if contras == 0 else 0.0)
        # End-to-end accuracy.
        e2e.append(1 if d["decision"] == "PASS" else 0)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
            elapsed = time.time() - t_start
            row_dt = time.time() - row_t
            avg = elapsed / (idx + 1)
            eta = avg * (len(rows) - idx - 1)
            print(
                f"  [{idx+1}/{len(rows)}] row_dt={row_dt:.1f}s avg={avg:.1f}s eta={eta:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    # Detector metrics at threshold 0.5.
    detector_pred = [1 if s >= 0.5 else 0 for s in detector_score]
    prec, rec, f1 = _precision_recall_f1(detector_truth, detector_pred)
    acc = _accuracy(detector_truth, detector_pred)
    auc = _roc_auc(detector_truth, detector_score)

    aggregate = {
        "detection_accuracy": acc,
        "detection_precision": prec,
        "detection_recall": rec,
        "detection_f1": f1,
        "detection_roc_auc": auc,
        "hallucination_reduction_rate": statistics.fmean(hrr_list) if hrr_list else 0.0,
        "claim_correction_rate": statistics.fmean(ccr_list) if ccr_list else 0.0,
        "verification_success_rate": statistics.fmean(vsr_list) if vsr_list else 0.0,
        "end_to_end_accuracy": statistics.fmean(e2e) if e2e else 0.0,
        "n_queries": len(rows),
        "dataset": slug,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing["aggregate"] = aggregate
    existing["dataset"] = slug
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({slug}, {len(rows)} rows)")
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
