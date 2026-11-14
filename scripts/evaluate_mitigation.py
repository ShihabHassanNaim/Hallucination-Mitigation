"""Measure the mitigation delta: does regeneration actually improve the answer?

The comparison table claims a mitigation delta of "+0.67 to +0.75 (HIGHEST)".
That claim was never backed by a measurement — the detection evaluation
supplies the benchmark's own answer to the pipeline, so step 10 passes it
through untouched and nothing is regenerated. This script measures the
delta on the path where mitigation actually happens: the pipeline
*generates* an answer, step 10 rewrites the unsupported parts, and the
similarity to the gold answer is compared before and after.

For each row the script reports:

  * ``pre``   — similarity of the generated answer to the gold answer
  * ``post``  — similarity of the final (post-regeneration) answer
  * ``delta`` — post minus pre

Similarity is reported with BERTScore F1 (semantic) and ROUGE-L (surface),
both against the dataset's ``right_answer``. Rows where the pipeline made no
decision to regenerate are reported separately: including them would
average in a delta of exactly zero and understate the effect of the step.

Usage:
  python -m scripts.evaluate_mitigation --limit 200
  python -m scripts.evaluate_mitigation --out data/mitigation_results.json
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
from src.steps.step13_evaluation import _rouge_l


def _bertscore_f1(reference: str, candidate: str) -> float | None:
    """BERTScore F1 between a reference and a candidate.

    Returns None when the package or the model is unavailable, so a missing
    dependency cannot be mistaken for a score of zero.
    """
    if not reference.strip() or not candidate.strip():
        return None
    try:
        from bert_score import score as _bs
    except Exception:  # noqa: BLE001
        return None
    try:
        _p, _r, f1 = _bs([reference], [candidate], lang="en", verbose=False)
        return float(f1.item())
    except Exception:  # noqa: BLE001
        return None


def _mean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0


def _bootstrap_ci(vals: list[float], n_boot: int = 1000, seed: int = 0) -> list[float]:
    """Percentile bootstrap 95% CI of the mean."""
    import numpy as np

    if len(vals) < 5:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = [float(arr[rng.integers(0, len(arr), len(arr))].mean())
             for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/halueval_eval_1000_matched.jsonl")
    ap.add_argument("--index", default="data/index_halueval")
    ap.add_argument("--out", default="data/mitigation_results.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.utils.data_loader import load_jsonl

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    cfg = load_config()
    pipe = Pipeline(cfg=cfg)
    if args.index:
        pipe.load_index(args.index)
        print(f"loaded index: {args.index} ({pipe.retriever.n_docs} docs)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_row: list[dict] = []
    t_start = time.time()

    for idx, r in enumerate(rows):
        gold = str(r.get("right_answer") or "")
        if not gold:
            continue
        query = r.get("query") or r.get("question") or ""
        try:
            # No `answer=` argument: the pipeline generates its own, which
            # is the only way step 10 has anything to regenerate.
            res = pipe.run(query)
        except Exception as exc:  # noqa: BLE001
            print(f"  row {idx} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            continue

        d = res.to_dict()
        # `original_answer` is what the generator produced; `final_answer`
        # is what survived step 10.
        generated = d.get("original_answer") or d.get("answer", "")
        final = d.get("final_answer") or d.get("answer", "")

        pre_bs = _bertscore_f1(gold, generated)
        post_bs = _bertscore_f1(gold, final)
        pre_rl = _rouge_l(gold, generated)
        post_rl = _rouge_l(gold, final)

        regenerated = final.strip() != generated.strip()

        per_row.append({
            "query": query,
            "label": int(r["label"]) if r.get("label") in {0, 1, "0", "1"} else None,
            "decision": d.get("decision"),
            "iterations": d.get("iterations"),
            "regenerated": regenerated,
            "pre_bertscore": pre_bs,
            "post_bertscore": post_bs,
            "delta_bertscore": (post_bs - pre_bs)
            if (pre_bs is not None and post_bs is not None) else None,
            "pre_rouge_l": pre_rl,
            "post_rouge_l": post_rl,
            "delta_rouge_l": post_rl - pre_rl,
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
            elapsed = time.time() - t_start
            avg = elapsed / (idx + 1)
            print(
                f"  [{idx+1}/{len(rows)}] avg={avg:.1f}s "
                f"eta={avg * (len(rows) - idx - 1):.0f}s",
                file=sys.stderr, flush=True,
            )
            out_path.write_text(
                json.dumps({"per_row": per_row}, indent=2), encoding="utf-8"
            )

    regen_rows = [r for r in per_row if r["regenerated"]]
    kept_rows = [r for r in per_row if not r["regenerated"]]

    def _agg(rows_: list[dict]) -> dict:
        deltas_bs = [r["delta_bertscore"] for r in rows_
                     if r["delta_bertscore"] is not None]
        deltas_rl = [r["delta_rouge_l"] for r in rows_]
        return {
            "n": len(rows_),
            "pre_bertscore": _mean([r["pre_bertscore"] for r in rows_
                                    if r["pre_bertscore"] is not None]),
            "post_bertscore": _mean([r["post_bertscore"] for r in rows_
                                     if r["post_bertscore"] is not None]),
            "delta_bertscore": _mean(deltas_bs),
            "delta_bertscore_ci95": _bootstrap_ci(deltas_bs, seed=args.seed),
            "pre_rouge_l": _mean([r["pre_rouge_l"] for r in rows_]),
            "post_rouge_l": _mean([r["post_rouge_l"] for r in rows_]),
            "delta_rouge_l": _mean(deltas_rl),
            "delta_rouge_l_ci95": _bootstrap_ci(deltas_rl, seed=args.seed),
        }

    summary = {
        "dataset": args.data,
        "n_rows": len(per_row),
        "n_regenerated": len(regen_rows),
        "n_kept": len(kept_rows),
        "all_rows": _agg(per_row),
        "regenerated_rows": _agg(regen_rows) if regen_rows else {},
        "kept_rows": _agg(kept_rows) if kept_rows else {},
    }

    out_path.write_text(
        json.dumps({"summary": summary, "per_row": per_row}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
