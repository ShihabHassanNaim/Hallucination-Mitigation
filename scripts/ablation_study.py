"""Ablation study: measure the contribution of each pipeline stage.

Runs the evaluation over a dataset with individual steps disabled and
records the resulting metrics. Output: data/ablation_results.json

Usage:
  python -m scripts.ablation_study --data data/halueval_eval_split.jsonl --limit 100
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.config import load_config
from src.pipeline import Pipeline
from src.steps.step13_evaluation import (
    _accuracy, _precision_recall_f1, _roc_auc,
)
from src.utils.data_loader import load_jsonl


# Which step class each ablation key disables.
ABLATIONS = {
    "without_step03_detector": ("early_detection", 3),
    "without_step04_eedc": ("confidence_assessment", 4),
    "without_step05_claims": ("claim_extraction", 5),
    "without_step07_reranker": ("reranking", 7),
    "without_step08_nli": ("nli_verification", 8),
    "without_step09_fusion": ("confidence_fusion", 9),
    "without_step10_regen": ("regeneration", 10),
}


def _disable_step(pipe: Pipeline, step_name: str) -> None:
    """Monkey-patch a step's run() to a no-op inside this pipeline."""
    for cls in type(pipe).__mro__:
        pass
    from src.steps import step_registry

    for cls in step_registry():
        if cls.name == step_name:
            orig = cls.run

            def _noop(self, _orig=orig):
                # Preserve the metadata key so downstream steps don't crash,
                # but skip the actual work.
                return None

            cls.run = _noop
            return
    raise KeyError(f"step {step_name!r} not found in registry")


def _restore_steps() -> None:
    from src.steps import step_registry

    for cls in step_registry():
        if hasattr(cls, "_ablation_orig_run"):
            cls.run = cls._ablation_orig_run
            del cls._ablation_orig_run


def _evaluate(pipe: Pipeline, rows: list[dict]) -> dict:
    """Run the dataset through the pipeline and compute detection metrics."""
    truths: list[int] = []
    scores: list[float] = []
    decisions: list[str] = []
    hrr: list[float] = []

    for r in rows:
        query = r.get("query") or r.get("question") or ""
        if not query:
            continue
        try:
            # Verify the dataset's own answer so the detector sees the
            # correct and the hallucinated answer for each paired question.
            res = pipe.run(query, answer=r.get("answer"))
        except Exception:  # noqa: BLE001
            continue
        d = res.to_dict()

        if "label" in r and r["label"] in {0, 1, "0", "1"}:
            gt = int(r["label"])
        else:
            # Gold evidence is stored as passages while retrieved documents
            # carry index ids, so match on text.
            from scripts.evaluate import _match_gold
            gold_texts = [e for e in (r.get("evidence") or []) if e]
            retrieved_texts = [e.get("text", "") for e in d["evidence"]]
            gt = 0 if _match_gold(retrieved_texts, gold_texts) else 1

        # Detector score: prefer the XGBoost probability from step 3.
        xgb = None
        for t in res.step_traces:
            if t.name == "early_detection":
                xgb = (t.payload or {}).get("hallucination_probability")
                break
        score = float(xgb) if xgb is not None else 1.0 - d.get("confidence", 0.5)

        truths.append(gt)
        scores.append(score)
        decisions.append(d["decision"])
        if gt == 0:
            hrr.append(1.0 if d["decision"] in {"REGEN", "ABSTAIN"} else 0.0)

    if not truths:
        return {}

    pred = [1 if s >= 0.5 else 0 for s in scores]
    prec, rec, f1 = _precision_recall_f1(truths, pred)
    return {
        "detection_accuracy": _accuracy(truths, pred),
        "detection_precision": prec,
        "detection_recall": rec,
        "detection_f1": f1,
        "detection_roc_auc": _roc_auc(truths, scores),
        "hallucination_reduction_rate": statistics.fmean(hrr) if hrr else 0.0,
        "n_queries": len(truths),
    }


def _write_results(out_path: Path, results: dict[str, dict]) -> None:
    """Write ablation results atomically so a kill never truncates the file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="JSONL of evaluation rows")
    ap.add_argument("--limit", type=int, default=100, help="Rows to use")
    ap.add_argument("--out", default="data/ablation_results.json")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=sorted(ABLATIONS),
        help="Restrict to specific ablations (default: all).",
    )
    ap.add_argument(
        "--index",
        default=None,
        help="path to a saved FAISS+BM25 index directory. Defaults to the "
             "index matching the dataset name in --data; without it the "
             "pipeline silently falls back to a 6-document synthetic corpus.",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.data)[: args.limit]
    if not rows:
        raise SystemExit(f"no rows loaded from {args.data}")

    if args.index is None:
        probe = args.data.lower()
        for name, idx in (("halueval", "data/index_halueval"), ("truthfulqa", "data/index_truthfulqa")):
            if name in probe:
                args.index = idx
                break

    cfg = load_config()
    targets = args.only or list(ABLATIONS)
    results: dict[str, dict] = {}

    # --- Baseline: full pipeline ---
    print(f"Running FULL pipeline on {len(rows)} rows...", flush=True)
    pipe = Pipeline(cfg=cfg)
    if args.index:
        pipe.load_index(args.index)
        print(f"  index: {args.index} ({pipe.retriever.n_docs} docs)", flush=True)
    results["full_pipeline"] = _evaluate(pipe, rows)
    print(f"  ROC-AUC = {results['full_pipeline'].get('detection_roc_auc', 0):.3f}",
          flush=True)

    # --- Ablations ---
    for key in targets:
        step_name, _num = ABLATIONS[key]
        print(f"Running {key} (disabling '{step_name}')...", flush=True)

        # Snapshot the original run() for restoration.
        from src.steps import step_registry

        for cls in step_registry():
            if not hasattr(cls, "_ablation_orig_run"):
                cls._ablation_orig_run = cls.run

        _disable_step(pipe, step_name)
        t0 = time.time()
        results[key] = _evaluate(pipe, rows)
        _restore_steps()
        print(
            f"  ROC-AUC = {results[key].get('detection_roc_auc', 0):.3f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        # Save after every ablation so a long run keeps completed results.
        _write_results(Path(args.out), results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    # Summary table.
    print("\n=== ABLATION SUMMARY ===")
    base = results["full_pipeline"].get("detection_roc_auc", 0.0)
    print(f"{'Configuration':<32} {'ROC-AUC':>10} {'F1':>8} {'Δ ROC':>9}")
    print("-" * 62)
    print(f"{'Full pipeline':<32} {base:>10.3f} "
          f"{results['full_pipeline'].get('detection_f1', 0):>8.3f} {'—':>9}")
    for key in targets:
        v = results[key]
        name = key.replace("without_", "").replace("_", " ").title()
        delta = v.get("detection_roc_auc", 0.0) - base
        print(f"{name:<32} {v.get('detection_roc_auc', 0):>10.3f} "
              f"{v.get('detection_f1', 0):>8.3f} {delta:>+9.3f}")


if __name__ == "__main__":
    main()
