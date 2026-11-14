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
from scripts.evaluate import _match_gold


_DATASET_DEFAULTS: dict[str, dict[str, str]] = {
    "halueval": {
        # Leak-free held-out split (500 unique queries, 500/500 balanced).
        # The full pool (data/halueval_qa.jsonl) contains the 200 queries the
        # detector was trained on, so it must NOT be used for evaluation.
        "data": "data/halueval_eval_1000.jsonl",
        "out": "data/metrics_halueval.json",
        # Real retrieval corpus. Without this the pipeline silently falls back
        # to a 6-document synthetic corpus and retrieval contributes nothing.
        "index": "data/index_halueval",
    },
    "truthfulqa": {
        "data": "data/truthfulqa_eval_1000.jsonl",
        "out": "data/metrics_truthfulqa.json",
        "index": "data/index_truthfulqa",
    },
}


def _bootstrap_intervals(
    truth: list[int],
    scores: list[float],
    e2e: list[int],
    groups: list[str],
    hrr: list[float] | None = None,
    ccr: list[float] | None = None,
    vsr: list[float] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap 95% CIs for the headline metrics.

    Resamples *queries* with replacement rather than rows, because each
    query contributes two rows (one hallucinated, one correct) and sampling
    rows independently would treat the pair as independent evidence.
    Returns {} when there is not enough data to resample.
    """
    n = len(truth)
    if n < 10:
        return {}

    by_query: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_query.setdefault(g, []).append(i)
    keys = list(by_query.keys())

    rng = np.random.default_rng(seed)
    accs: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    aucs: list[float] = []
    e2es: list[float] = []
    hrrs: list[float] = []
    ccrs: list[float] = []
    vsrs: list[float] = []

    for _ in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        idx: list[int] = []
        for j in picked:
            idx.extend(by_query[keys[j]])
        bt = [truth[i] for i in idx]
        bs = [scores[i] for i in idx]
        bp = [1 if s >= 0.5 else 0 for s in bs]
        accs.append(_accuracy(bt, bp))
        p, r, f = _precision_recall_f1(bt, bp)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        aucs.append(_roc_auc(bt, bs))
        be = [e2e[i] for i in idx] if e2e else []
        e2es.append((float(sum(be)) / len(be)) if be else 0.0)
        if hrr:
            hrrs.append(statistics.fmean([hrr[i] for i in idx]))
        if ccr:
            ccrs.append(statistics.fmean([ccr[i] for i in idx]))
        if vsr:
            vsrs.append(statistics.fmean([vsr[i] for i in idx]))

    def _ci(vals: list[float]) -> list[float]:
        if not vals:
            return [0.0, 0.0]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    out = {
        "detection_accuracy_ci95": _ci(accs),
        "detection_precision_ci95": _ci(precs),
        "detection_recall_ci95": _ci(recs),
        "detection_f1_ci95": _ci(f1s),
        "detection_roc_auc_ci95": _ci(aucs),
        "end_to_end_accuracy_ci95": _ci(e2es),
        "n_bootstrap": n_boot,
        "n_unique_queries": len(keys),
    }
    if hrrs:
        out["hallucination_reduction_rate_ci95"] = _ci(hrrs)
    if ccrs:
        out["claim_correction_rate_ci95"] = _ci(ccrs)
    if vsrs:
        out["verification_success_rate_ci95"] = _ci(vsrs)
    return out


def _write_checkpoint(out_path: Path, slug: str, per_query: list[dict]) -> None:
    """Persist per-row results so a crash never costs the whole run.

    The aggregate block is intentionally left untouched here: it is rebuilt
    from ``per_query`` by ``_aggregate_from_rows`` once the run finishes (or
    via ``--recompute`` on a checkpoint left by a killed run).
    """
    payload = {
        "dataset": slug,
        "n_queries": len(per_query),
        "per_query": per_query,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def _aggregate_from_rows(slug: str, per_query: list[dict], seed: int = 0) -> dict:
    """Rebuild the aggregate metric block from per-row records.

    Used both at the end of a normal run and by ``--recompute`` to recover a
    checkpoint left behind by a killed process. Keeping the aggregation in one
    place guarantees the numbers in the output file always match the CIs.
    """
    if not per_query:
        return {}

    truth = [int(r["label"]) for r in per_query]
    scores = [float(r["detector_score"]) for r in per_query]
    groups = [r.get("query") or f"row-{i}" for i, r in enumerate(per_query)]
    e2e = [int(r.get("end_to_end_accuracy", 0)) for r in per_query]
    hrr = [float(r.get("hrr", 0.0)) for r in per_query]
    ccr = [float(r.get("ccr", 0.0)) for r in per_query]
    vsr = [float(r.get("vsr", 0.0)) for r in per_query]

    detector_pred = [1 if s >= 0.5 else 0 for s in scores]
    prec, rec, f1 = _precision_recall_f1(truth, detector_pred)
    acc = _accuracy(truth, detector_pred)
    auc = _roc_auc(truth, scores)

    aggregate = {
        "detection_accuracy": acc,
        "detection_precision": prec,
        "detection_recall": rec,
        "detection_f1": f1,
        "detection_roc_auc": auc,
        "hallucination_reduction_rate": statistics.fmean(hrr),
        "claim_correction_rate": statistics.fmean(ccr),
        "verification_success_rate": statistics.fmean(vsr),
        "end_to_end_accuracy": statistics.fmean(e2e),
        "n_queries": len(per_query),
        "dataset": slug,
    }
    aggregate.update(
        _bootstrap_intervals(truth, scores, e2e, groups, hrr=hrr, ccr=ccr, vsr=vsr, seed=seed)
    )

    # Retrieval and claim metrics, averaged over the same rows. These are
    # absent from older checkpoint files, so they are only reported when the
    # rows actually carry them.
    for key in (
        "retrieval_recall_at_5",
        "retrieval_precision_at_5",
        "retrieval_mrr",
        "retrieval_ndcg",
        "claim_precision",
    ):
        vals = [r[key] for r in per_query if key in r]
        if vals:
            aggregate[key] = statistics.fmean(vals)
    return aggregate


def _resolve_dataset_rows(args: argparse.Namespace) -> tuple[list[dict], str]:
    """Return (rows, dataset_slug_or_label) honoring --data/--synth/--dataset."""
    if args.synth:
        rows = load_synthetic_qa()
        slug = "synth"
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


def _recompute(args: argparse.Namespace) -> None:
    """Rebuild the aggregate block from per_query rows already on disk."""
    out_path = Path(args.out or "data/metrics.json")
    if not out_path.exists():
        raise SystemExit(f"no such file: {out_path}")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    per_query = payload.get("per_query") or []
    if not per_query:
        raise SystemExit(f"{out_path} has no per_query rows to recompute from")
    slug = payload.get("dataset") or out_path.stem.replace("metrics_", "")
    aggregate = _aggregate_from_rows(slug, per_query)
    payload["aggregate"] = aggregate
    payload["n_queries"] = len(per_query)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"recomputed {out_path} from {len(per_query)} rows")
    print(json.dumps({"aggregate": aggregate}, indent=2))


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
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="skip the pipeline entirely: rebuild the aggregate block (metrics "
             "+ bootstrap CIs) from the per_query rows already in --out.",
    )
    args = ap.parse_args()

    if args.recompute:
        _recompute(args)
        return

    rows, slug = _resolve_dataset_rows(args)
    if args.out is None:
        if slug in _DATASET_DEFAULTS:
            args.out = _DATASET_DEFAULTS[slug]["out"]
        else:
            args.out = "data/metrics.json"

    # Default the index per dataset so retrieval never silently degrades to
    # the 6-document synthetic corpus. An explicit --index still wins.
    # When --data is used the slug is a filename stem, so match on the
    # dataset name appearing anywhere in the path (e.g. "halueval_eval_1000").
    if args.index is None:
        probe = f"{slug} {args.data or ''}".lower()
        for name, defaults in _DATASET_DEFAULTS.items():
            if name in probe and defaults.get("index"):
                args.index = defaults["index"]
                break

    cfg = load_config()
    pipe = Pipeline(cfg=cfg)
    if args.index:
        pipe.load_index(args.index)
        print(f"loaded index: {args.index} ({pipe.retriever.n_docs} docs)")
    clf = XGBClassifier()
    clf._load()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    detector_truth: list[int] = []
    detector_score: list[float] = []
    hrr_list: list[float] = []
    ccr_list: list[float] = []
    vsr_list: list[float] = []
    e2e: list[int] = []
    per_query: list[dict] = []
    query_groups: list[str] = []

    t_start = time.time()
    for idx, r in enumerate(rows):
        row_t = time.time()
        try:
            # Verify the dataset's own answer. The rows are paired: for one
            # question the dataset supplies a correct answer (label 0) and a
            # hallucinated one (label 1). Generating an answer from the
            # query alone would give both rows the same answer and the same
            # detector features, forcing the ROC-AUC to exactly 0.5.
            res = pipe.run(r["query"], answer=r.get("answer"))
            d = res.to_dict()
        except Exception:
            continue

        # Determine ground truth: if the row has explicit label use it, else
        # infer from gold evidence presence.
        if "label" in r and r["label"] in {0, 1, "0", "1"}:
            gt = int(r["label"])
        else:
            # Gold evidence is stored as passages while retrieved documents
            # carry index ids, so the two must be matched on text.
            gold_texts = [e for e in (r.get("evidence") or []) if e]
            retrieved_texts = [e.get("text", "") for e in d["evidence"]]
            gt = 0 if _match_gold(retrieved_texts, gold_texts) else 1

        # Detector score: predict hallucination from the answer under
        # judgement (d["answer"] — the dataset's own answer, since the
        # pipeline verifies it rather than generating one). The ground-truth
        # `right_answer` is deliberately never touched here — scoring it
        # would leak the label into its own feature and yield a meaningless
        # ROC-AUC of 1.0.
        # Features use the same four signals as generate_detector_data_v2.py:
        # (cosine, claim_count, overlap, answer_length).
        from src.utils.claim_extractor import extract_claims, intrinsic_overlap

        gen_answer = d.get("answer") or ""
        if gen_answer:
            try:
                embedder = pipe.embedder
                qv = embedder.encode([r["query"]])[0]
                av = embedder.encode([gen_answer])[0]
                cosine = float(
                    np.dot(qv, av)
                    / (np.linalg.norm(qv) * np.linalg.norm(av) + 1e-12)
                )
            except Exception:
                cosine = 0.0
            claim_count = max(1, len(extract_claims(gen_answer, mode="rule")))
            overlap = intrinsic_overlap(gen_answer, r["query"])
            answer_length = float(len(gen_answer.split()))
            feats = [cosine, float(claim_count), overlap, answer_length]
            try:
                score = float(pipe.early_classifier.predict_proba(feats))
            except Exception:
                score = 1.0 - d.get("confidence", 0.5)
        else:
            # No generated answer to judge — fall back to pipeline confidence
            # inverted (higher => more likely hallucinated).
            score = 1.0 - d.get("confidence", 0.5)
        detector_truth.append(gt)
        detector_score.append(score)
        query_groups.append(r.get("query") or r.get("question") or f"row-{idx}")

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

        # Per-query record — needed for the ROC curve, calibration plot,
        # and latency breakdown graphs.
        #
        # The retrieval and claim metrics are computed here too, so a single
        # pipeline pass yields the whole catalogue and no separate BASE-metrics
        # run over the same rows is needed. Gold evidence is stored as passages
        # while retrieved documents carry index ids, so the two are matched on
        # text (see scripts/evaluate.py::_match_gold).
        from src.steps.step13_evaluation import (
            _recall_at_k, _precision_at_k, _mrr, _ndcg,
        )

        gold_texts = [e for e in (r.get("evidence") or []) if e]
        retrieved_texts = [e.get("text", "") for e in d.get("evidence", [])]
        gold_hits = _match_gold(retrieved_texts, gold_texts)
        gold_ids = list(range(len(gold_texts)))
        n_supported = sum(c["verdict"] == "supported" for c in d.get("claims", []))

        per_query.append({
            "query": r.get("query") or r.get("question") or "",
            "label": gt,
            "detector_score": score,
            "confidence": float(d.get("confidence", 0.5)),
            "decision": d["decision"],
            "n_claims": len(d.get("claims", [])),
            "n_evidence": len(d.get("evidence", [])),
            "end_to_end_accuracy": 1 if d["decision"] == "PASS" else 0,
            # Row-level reduction metrics, so the bootstrap CIs for HRR/CCR/VSR
            # can be recomputed from the output file alone.
            "hrr": 1.0 if (gt == 0 and d["decision"] in {"REGEN", "ABSTAIN"}) else 0.0,
            "ccr": (1.0 - (contras / max(1, len(d["claims"])))) if d["claims"] else 1.0,
            "vsr": 1.0 if contras == 0 else 0.0,
            # Retrieval metrics against the gold passages.
            "retrieval_recall_at_5": _recall_at_k(gold_hits, gold_ids, 5),
            "retrieval_precision_at_5": _precision_at_k(gold_hits, gold_ids, 5),
            "retrieval_mrr": _mrr(gold_hits, gold_ids),
            "retrieval_ndcg": _ndcg(gold_hits, gold_ids),
            # Claim-level metrics.
            "claim_precision": n_supported / max(1, len(d.get("claims", []))),
            "step_times_ms": {
                t.name: round(float(t.duration_ms), 3)
                for t in (res.step_traces or [])
            },
        })

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
            # Checkpoint incrementally so a crash or a kill never costs the
            # whole run — the final aggregate is rebuilt from this file.
            _write_checkpoint(out_path, slug, per_query)

    # Aggregate is rebuilt from per_query so the headline numbers and the
    # bootstrap CIs are always derived from the same rows.
    aggregate = _aggregate_from_rows(slug, per_query)

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
    existing["per_query"] = per_query
    existing["n_queries"] = len(per_query)
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({slug}, {len(rows)} rows)")
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
