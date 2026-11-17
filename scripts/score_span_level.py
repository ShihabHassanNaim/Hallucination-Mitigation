"""Score span-level hallucination detection on RAGTruth.

Why span level
--------------
RAGTruth is the only dataset of the three that annotates *where* in the
answer the hallucination is, not just that the answer contains one. Its
``hallucination_labels`` column lists the character offsets of every
hallucinated span. That makes span-level F1 computable, and span-level F1
is the axis where the published baselines are weakest:

  * RAGTruth's own fine-tuned Llama-2-13B baseline: span F1 52.7
  * Prompt gpt-4-turbo:                          span F1 28.3
  * Prompt gpt-3.5-turbo:                        span F1 12.8

A claim-verification pipeline is the right shape for this task by
construction: it localises the offending span and then judges it, instead
of classifying the whole response.

Matching rule
-------------
A predicted span counts as correct when it overlaps a gold span by at
least ``--iou`` (token-level IoU, the convention used by the RAGTruth
authors). A gold span counts as found when some predicted span matches it.
Both directions are reported, plus the F1 of the two.

Usage:
  python -m scripts.score_span_level --limit 100
  python -m scripts.score_span_level --limit 100 --out data/span_level.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.config import load_config
from src.pipeline import Pipeline
from src.steps.step10_regeneration import _find_span
from src.utils.data_loader import load_jsonl


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _iou(a: tuple[int, int], b: tuple[int, int], text: str) -> float:
    """Token-level IoU of two character spans of ``text``."""
    ta = set(_tokens(text[a[0]:a[1]]))
    tb = set(_tokens(text[b[0]:b[1]]))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _char_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ragtruth_eval_1000_matched.jsonl")
    ap.add_argument("--index", default="data/index_ragtruth")
    ap.add_argument("--spans",
                    default="data/ragtruth_span_labels.jsonl",
                    help="JSONL of {row_id, spans:[{start,end,text}]}")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="token-level IoU threshold for a correct match")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="minimum contradiction strength before a claim is "
                         "reported as a hallucinated span")
    ap.add_argument("--out", default="data/span_level_ragtruth.json")
    args = ap.parse_args()

    if not Path(args.spans).exists():
        raise SystemExit(
            f"{args.spans} not found. Build it with "
            f"`python -m scripts.build_ragtruth_spans` first."
        )

    span_labels = {r["row_id"]: r["spans"]
                   for r in load_jsonl(args.spans)}

    cfg = load_config()
    pipe = Pipeline(cfg)
    pipe.load_index(args.index)

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    tp = fp = fn = 0
    n_rows_with_gold = 0
    n_rows_with_pred = 0
    per_task: dict[str, list[int]] = {}
    details: list[dict] = []

    for i, r in enumerate(rows):
        gold_spans = span_labels.get(r["row_id"], [])
        answer = r["answer"] or ""
        res = pipe.run(r["query"], answer=answer, label=int(r["label"]),
                       gold_answer=r.get("right_answer"))
        claims = res.to_dict()["claims"]

        # A claim is a predicted hallucinated span when the verifier
        # contradicted it confidently.
        preds: list[tuple[int, int]] = []
        for c in claims:
            if c["verdict"] != "contradicted":
                continue
            strength = 1.0 - float(c.get("confidence") or 0.0)
            if strength < args.min_confidence:
                continue
            span = _find_span(answer, c["text"])
            if span is None:
                continue
            preds.append(span)

        golds = [(int(s["start"]), int(s["end"])) for s in gold_spans]
        if golds:
            n_rows_with_gold += 1
        if preds:
            n_rows_with_pred += 1

        used_gold: set[int] = set()
        row_tp = 0
        for p in preds:
            best_j = -1
            best_iou = 0.0
            for j, g in enumerate(golds):
                if j in used_gold:
                    continue
                v = _iou(p, g, answer)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= args.iou:
                used_gold.add(best_j)
                row_tp += 1
            else:
                # A predicted span that matches nothing is a false
                # positive, but only if it is a real span.
                pass

        row_fp = len(preds) - row_tp
        row_fn = len(golds) - row_tp
        tp += row_tp
        fp += row_fp
        fn += row_fn
        tt = r.get("task_type", "?")
        per_task.setdefault(tt, [0, 0, 0])
        per_task[tt][0] += row_tp
        per_task[tt][1] += row_fp
        per_task[tt][2] += row_fn

        details.append({
            "row_id": r["row_id"],
            "task_type": tt,
            "label": int(r["label"]),
            "n_gold_spans": len(golds),
            "n_pred_spans": len(preds),
            "tp": row_tp, "fp": row_fp, "fn": row_fn,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    p, rec, f1 = _prf(tp, fp, fn)
    report = {
        "n_rows": len(rows),
        "n_rows_with_gold_spans": n_rows_with_gold,
        "n_rows_with_predicted_spans": n_rows_with_pred,
        "iou_threshold": args.iou,
        "min_contradiction_confidence": args.min_confidence,
        "span_precision": round(p, 4),
        "span_recall": round(rec, 4),
        "span_f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "per_task": {},
        "published_span_f1": {
            "ragtruth_finetuned_llama2_13b": 52.7,
            "prompt_gpt4_turbo": 28.3,
            "prompt_gpt35_turbo": 12.8,
        },
    }
    for tt, (a, b, c) in per_task.items():
        pp, rr, ff = _prf(a, b, c)
        report["per_task"][tt] = {
            "precision": round(pp, 4), "recall": round(rr, 4),
            "f1": round(ff, 4), "tp": a, "fp": b, "fn": c,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
