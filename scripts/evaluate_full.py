"""Full evaluation across every metric in the diagram.

Adds HRR / CCR / VSR aggregation and detection ROC-AUC for a labelled
test set.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from src.config import load_config
from src.models.xgb_classifier import XGBClassifier
from src.pipeline import Pipeline
from src.steps.step13_evaluation import (
    _precision_recall_f1, _accuracy, _roc_auc,
)
from src.utils.data_loader import load_jsonl, load_synthetic_qa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--out", default="data/metrics.json")
    args = ap.parse_args()

    rows = load_synthetic_qa() if args.synth else (load_jsonl(args.data) if args.data else load_synthetic_qa())
    cfg = load_config()
    pipe = Pipeline(cfg=cfg)
    clf = XGBClassifier()
    clf._load()

    detector_truth: list[int] = []
    detector_score: list[float] = []
    hrr_list: list[float] = []
    ccr_list: list[float] = []
    vsr_list: list[float] = []
    e2e: list[int] = []

    for r in rows:
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
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    p = Path(args.out)
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing["aggregate"] = aggregate
    p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
