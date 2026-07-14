"""CLI: calibrate EEDC weights on a labelled dataset.

Each input record is a JSONL line with:
    {"nli_probs": [sup, con, nei],
     "retrieval_top1": float in [-1, 1],
     "self_consistency": float in [0, 1],
     "label": 1 if supported, 0 if not}

The script fits Platt-style weights via maximum-likelihood and writes
them to the JSON path given by --out. By default that path matches
`configs/default.yaml::eedc.weights_path`, so the pipeline picks them
up automatically on the next run.

Usage
-----
    python scripts/calibrate_eedc.py \\
        --data data/eedc_train.jsonl \\
        --out data/eedc_weights.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eedc import EEDCSignals, EEDCScorer


def load_records(path: Path) -> tuple:
    signals: list = []
    labels: list = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            probs = rec["nli_probs"]
            h_norm = _entropy_norm(probs)
            signals.append(EEDCSignals(
                nli_entropy_norm=h_norm,
                retrieval_top1=_clip01((rec.get("retrieval_top1", 0.0) + 1.0) / 2.0),
                self_consistency=_clip01(rec.get("self_consistency", 1.0)),
            ))
            labels.append(int(rec["label"]))
    return signals, labels


def _entropy_norm(probs: list) -> float:
    import math
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return min(1.0, h / math.log(len(probs)))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate EEDC Platt weights.")
    p.add_argument("--data", required=True, help="JSONL of {nli_probs, retrieval_top1, self_consistency, label}.")
    p.add_argument("--out", required=True, help="Output JSON path for the fitted weights.")
    args = p.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    if not data_path.exists():
        raise SystemExit(f"Calibration data not found: {data_path}")

    signals, labels = load_records(data_path)
    print(f"Loaded {len(signals)} calibration examples.")

    scorer = EEDCScorer()
    before = scorer.weights.as_vector()
    print(f"Pre-fit weights:  alpha={before[0]:.3f} beta={before[1]:.3f} "
          f"gamma={before[2]:.3f} delta={before[3]:.3f}")

    scorer.fit(signals, labels)

    after = scorer.weights.as_vector()
    print(f"Post-fit weights: alpha={after[0]:.3f} beta={after[1]:.3f} "
          f"gamma={after[2]:.3f} delta={after[3]:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "alpha": after[0], "beta": after[1],
        "gamma": after[2], "delta": after[3],
    }, indent=2), encoding="utf-8")
    print(f"Wrote weights to {out_path}")


if __name__ == "__main__":
    main()