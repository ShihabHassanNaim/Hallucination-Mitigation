"""Diagnose why the detector transfers poorly across datasets.

The Step-3 detector is trained on ``data/detector_train.jsonl``, which was
mined from HaluEval. Two things about that file limit how well the model
generalises:

  1. ``answer_length`` is strongly correlated with the label in HaluEval
     (gold answers are short spans, hallucinated answers are sentences).
     The detector therefore learns a length shortcut.
  2. The length-matched faithful control is padded with words from the
     dataset's ``knowledge`` passage, which lowers its ``intrinsic_overlap``
     relative to the hallucinated row of the same query. So ``overlap``
     becomes a second, partly-cosmetic shortcut.

This script quantifies both effects and measures what happens when the
detector is retrained with the offending feature removed, and when it is
retrained on length-matched rows only.

It recomputes the Step-3 features for both evaluation splits with the same
code path the pipeline uses, so the numbers are directly comparable with the
pipeline's own scores.

Usage:
  python -m scripts.diagnose_feature_transfer
  python -m scripts.diagnose_feature_transfer --out data/feature_transfer.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.utils.data_loader import load_jsonl

FEATURE_NAMES = ["cosine", "claim_count", "overlap", "answer_length"]

EVAL_SPLITS = {
    "halueval": "data/halueval_eval_1000_matched.jsonl",
    "truthfulqa": "data/truthfulqa_eval_1000.jsonl",
}


def _step3_features(query: str, answer: str, embedder=None) -> list[float]:
    """Reproduce Step 3's feature vector.

    Mirrors ``src/steps/step03_early_detection.py`` exactly: the same cosine
    between the query and answer embeddings, the same claim count, the same
    token-in-query overlap ratio and the same whitespace word count.
    """
    answer = answer or ""
    query = query or ""

    cosine = 0.0
    if embedder is not None and hasattr(embedder, "encode"):
        try:
            qv = embedder.encode([query])[0]
            av = embedder.encode([answer])[0]
            cosine = float(
                np.dot(qv, av) / (np.linalg.norm(qv) * np.linalg.norm(av) + 1e-12)
            )
        except Exception:  # noqa: BLE001
            cosine = 0.0

    claim_count = 1.0
    try:
        from src.utils.claim_extractor import extract_claims, intrinsic_overlap

        claim_count = float(max(1, len(extract_claims(answer, mode="rule"))))
        overlap = intrinsic_overlap(answer, query)
    except Exception:  # noqa: BLE001
        claim_count = 1.0
        overlap = 0.0

    return [cosine, claim_count, overlap, float(len(answer.split()))]


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank identity; ties get average ranks."""
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average the ranks of tied scores.
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _fit_predict(train_X: np.ndarray, train_y: np.ndarray,
                 test_X: np.ndarray) -> np.ndarray:
    """Fit the same XGBoost configuration the pipeline uses, then score."""
    import xgboost as xgb  # type: ignore

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(train_X, train_y)
    return model.predict_proba(test_X)[:, 1]


def _feature_report(X: np.ndarray, y: np.ndarray) -> dict:
    """Per-feature separability and the length/overlap correlation."""
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        out[f"{name}_auc"] = round(_roc_auc(X[:, i], y), 4)
    if len(X) > 2:
        out["overlap_length_pearson"] = round(
            float(np.corrcoef(X[:, 2], X[:, 3])[0, 1]), 4
        )
    out["answer_length_median_label0"] = float(np.median(X[y == 0, 3]))
    out["answer_length_median_label1"] = float(np.median(X[y == 1, 3]))
    out["overlap_median_label0"] = float(np.median(X[y == 0, 2]))
    out["overlap_median_label1"] = float(np.median(X[y == 1, 2]))
    out["n"] = int(len(y))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/detector_train.jsonl")
    ap.add_argument("--out", default="data/feature_transfer.json")
    ap.add_argument("--device", default="cpu",
                    help="device for the embedder (default cpu, to avoid "
                         "colliding with a running GPU evaluation)")
    args = ap.parse_args()

    embedder = None
    try:
        from src.config import load_config
        from src.models.embedder_bge_m3 import BGE_M3Embedder

        cfg = load_config()
        embedder = BGE_M3Embedder(
            model_name=(cfg.models or {}).get("embedder", "BAAI/bge-m3"),
            device=args.device,
        )
        embedder._load_hf()
        print(f"embedder ready on {args.device}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"embedder unavailable ({exc!r}); cosine will be 0.0",
              file=sys.stderr, flush=True)

    # --- training set -------------------------------------------------
    train_rows = load_jsonl(args.train)
    # The stored training file already carries the features the detector was
    # fitted on; reuse them so the diagnosis reflects exactly that fit.
    train_X = np.array([r["features"] for r in train_rows], dtype=float)
    train_y = np.array([int(r["label"]) for r in train_rows], dtype=int)
    print(f"train rows: {len(train_y)}", flush=True)

    # --- evaluation splits --------------------------------------------
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, path in EVAL_SPLITS.items():
        if not Path(path).exists():
            print(f"skip {name}: {path} missing", file=sys.stderr)
            continue
        rows = load_jsonl(path)
        feats, labels = [], []
        for r in rows:
            q = r.get("query") or r.get("question") or ""
            a = r.get("answer") or ""
            feats.append(_step3_features(q, a, embedder))
            labels.append(int(r["label"]))
        splits[name] = (np.array(feats, dtype=float), np.array(labels, dtype=int))
        print(f"{name}: {len(labels)} rows", flush=True)

    report: dict = {
        "train_feature_report": _feature_report(train_X, train_y),
        "splits": {},
        "experiments": {},
    }

    # --- baseline: the shipped 4-feature detector ----------------------
    for name, (X, y) in splits.items():
        report["splits"][name] = _feature_report(X, y)

    for name, (X, y) in splits.items():
        scores = _fit_predict(train_X, train_y, X)
        report["experiments"].setdefault("all_features", {})[name] = round(
            _roc_auc(scores, y), 4
        )

    # --- drop answer_length -------------------------------------------
    keep = [0, 1, 2]
    for name, (X, y) in splits.items():
        scores = _fit_predict(train_X[:, keep], train_y, X[:, keep])
        report["experiments"].setdefault("no_answer_length", {})[name] = round(
            _roc_auc(scores, y), 4
        )

    # --- drop answer_length and claim_count (claim_count is constant) --
    keep2 = [0, 2]
    for name, (X, y) in splits.items():
        scores = _fit_predict(train_X[:, keep2], train_y, X[:, keep2])
        report["experiments"].setdefault("cosine_and_overlap_only", {})[name] = round(
            _roc_auc(scores, y), 4
        )

    # --- train on length-matched rows only -----------------------------
    # A row is "length-matched" when its word count equals the word count of
    # the hallucinated row for the same query, so both classes sit at the
    # same length and `answer_length` carries no information at all.
    by_query: dict[str, list[dict]] = {}
    for r in train_rows:
        by_query.setdefault(r.get("query") or r.get("question") or "", []).append(r)
    matched_rows: list[dict] = []
    for q, group in by_query.items():
        hallucinated = [r for r in group if int(r["label"]) == 1]
        faithful = [r for r in group if int(r["label"]) == 0]
        if not hallucinated or not faithful:
            continue
        target = hallucinated[0]["features"][3]
        same = [r for r in faithful if abs(r["features"][3] - target) < 1e-6]
        if same:
            matched_rows.append(hallucinated[0])
            matched_rows.extend(same)
    if matched_rows:
        mm_X = np.array([r["features"] for r in matched_rows], dtype=float)
        mm_y = np.array([int(r["label"]) for r in matched_rows], dtype=int)
        report["experiments"]["train_rows_length_matched"] = {
            "n": len(mm_y),
            "n_queries": len(matched_rows) // 2,
        }
        for name, (X, y) in splits.items():
            scores = _fit_predict(mm_X, mm_y, X)
            report["experiments"].setdefault("trained_on_length_matched", {})[name] = \
                round(_roc_auc(scores, y), 4)
        # And length-matched training with answer_length removed.
        for name, (X, y) in splits.items():
            scores = _fit_predict(mm_X[:, keep], mm_y, X[:, keep])
            report["experiments"].setdefault(
                "length_matched_and_no_answer_length", {}
            )[name] = round(_roc_auc(scores, y), 4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["experiments"], indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
