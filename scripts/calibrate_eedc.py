"""Fit Platt weights for the EEDC scorer on a labelled dev set.

Usage:
  python -m scripts.calibrate_eedc --data data/dev.jsonl --out data/eedc_weights.json
Each row: {"nli_entropy": ..., "retrieval_top1": ..., "self_consistency": ..., "label": 0|1}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.models.eedc import EEDC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data/eedc_weights.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no rows")

    X = np.array([[r["nli_entropy"], 1.0 - r["retrieval_top1"], 1.0 - r["self_consistency"]] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.float64)

    # Small grid logistic regression.
    best = {"nll": float("inf"), "alpha": -1.0, "beta": -0.7, "gamma": -0.5, "delta": 1.0}
    for alpha in [-1.5, -1.0, -0.5]:
        for beta in [-1.0, -0.7, -0.3]:
            for gamma in [-0.7, -0.5, -0.3]:
                for delta in [0.5, 1.0, 1.5]:
                    phi = alpha * X[:, 0] + beta * X[:, 1] + gamma * X[:, 2] + delta
                    p = 1.0 / (1.0 + np.exp(-phi))
                    p = np.clip(p, 1e-6, 1.0 - 1e-6)
                    nll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()
                    if nll < best["nll"]:
                        best.update({"nll": float(nll), "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta})

    out = {"alpha": best["alpha"], "beta": best["beta"], "gamma": best["gamma"], "delta": best["delta"], "nll": best["nll"]}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
