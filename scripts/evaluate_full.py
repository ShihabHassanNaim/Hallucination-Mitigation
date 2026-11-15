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
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from src.config import load_config
from src.models.xgb_classifier import XGBClassifier
from src.pipeline import Pipeline
from src.steps.step13_evaluation import (
    _precision_recall_f1, _accuracy, _roc_auc, _claim_level_metrics,
)
from src.types import Claim
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
    auprs: list[float] = []
    p90s: list[float] = []
    rhos: list[float] = []

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
        auprs.append(_average_precision(bt, bs))
        p90s.append(_precision_at_recall(bt, bs, 0.9))
        rhos.append(_spearman_rho(bt, bs))

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
        "detection_aupr_ci95": _ci(auprs),
        "precision_at_recall_90_ci95": _ci(p90s),
        "spearman_rho_ci95": _ci(rhos),
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


def _average_precision(y_true: list[int], y_score: list[float]) -> float:
    """Area under the precision-recall curve (average precision).

    Implemented directly rather than pulled from scikit-learn, which is not
    a dependency of this project.
    """
    if not y_true or sum(y_true) == 0:
        return 0.0
    pairs = sorted(zip(y_score, y_true), key=lambda kv: -kv[0])
    n_pos = sum(y_true)
    tp = 0
    prev_recall = 0.0
    ap = 0.0
    for rank, (_score, label) in enumerate(pairs, 1):
        if label == 1:
            tp += 1
            recall = tp / n_pos
            precision = tp / rank
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return ap


def _precision_at_recall(y_true: list[int], y_score: list[float], target: float = 0.9) -> float:
    """Best precision achievable while retaining at least ``target`` recall.

    Scans the full ranking rather than stopping at the first threshold
    crossing: precision at the crossing point is the *lowest* precision that
    satisfies the recall constraint, whereas the metric is defined as the
    best precision obtainable at that recall level. Returns 0.0 when the
    target recall is unreachable (e.g. asking for 90% recall when only one
    positive exists is not a meaningful operating point).
    """
    if not y_true or sum(y_true) == 0:
        return 0.0
    pairs = sorted(zip(y_score, y_true), key=lambda kv: -kv[0])
    n_pos = sum(y_true)
    # The smallest number of positives that satisfies "recall >= target".
    # When that is the whole positive set the constraint is vacuous -- it
    # just says "find every positive" -- and the metric degenerates. Require
    # the target to be satisfiable while still tolerating at least one miss.
    min_needed = math.ceil(target * n_pos)
    if min_needed >= n_pos:
        return 0.0
    tp = 0
    best = 0.0
    for rank, (_score, label) in enumerate(pairs, 1):
        if label == 1:
            tp += 1
        recall = tp / n_pos
        if recall >= target:
            best = max(best, tp / rank)
    return best


def _spearman_rho(y_true: list[int], y_score: list[float]) -> float:
    """Spearman rank correlation between the label and the detector score."""
    n = len(y_true)
    if n < 2:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _rank([float(t) for t in y_true])
    ry = _rank(y_score)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) ** 0.5
        * sum((b - my) ** 2 for b in ry) ** 0.5
    )
    return num / den if den else 0.0


def _detector_fingerprint() -> str:
    """A short hash of the detector's training data and configuration.

    The detector is refitted at process start from a training file on disk.
    When that file changes — a different feature set, a different set of
    training queries — every previously computed ``detector_score`` becomes
    incomparable with the new ones. Recording the fingerprint in the
    checkpoint lets ``--resume`` refuse to mix rows scored by two different
    detectors.
    """
    import hashlib

    from src.models.xgb_classifier import _CLEAN_DATA_PATH, _REAL_DATA_PATH

    parts: list[str] = []
    for path in (_CLEAN_DATA_PATH, _REAL_DATA_PATH):
        if path.exists():
            try:
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
            except Exception:  # noqa: BLE001
                parts.append(f"{path.name}:?")
        else:
            parts.append(f"{path.name}:missing")
    try:
        import xgboost as xgb  # type: ignore

        parts.append(f"xgb:{xgb.__version__}")
    except Exception:  # noqa: BLE001
        parts.append("xgb:none")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _write_checkpoint(out_path: Path, slug: str, per_query: list[dict]) -> None:
    """Persist per-row results so a crash never costs the whole run.

    The aggregate block is intentionally left untouched here: it is rebuilt
    from ``per_query`` by ``_aggregate_from_rows`` once the run finishes (or
    via ``--recompute`` on a checkpoint left by a killed run).

    ``detector_fingerprint`` records which detector produced the scores, so
    a later ``--resume`` can tell whether the rows are still comparable.
    """
    payload = {
        "dataset": slug,
        "n_queries": len(per_query),
        "detector_fingerprint": _detector_fingerprint(),
        "per_query": per_query,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def _load_checkpoint(out_path: Path) -> list[dict]:
    """Return the per-row records of an existing checkpoint, or [].

    Used by ``--resume`` so a run that was killed (or that needs to be
    restarted after a code change) does not throw away the rows it already
    finished. Rows are matched to their dataset rows by ``query`` +
    ``label`` + ``answer`` position, so a checkpoint is only reused for the
    exact row it was produced from.
    """
    if not out_path.exists():
        return []
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    rows = payload.get("per_query")
    return rows if isinstance(rows, list) else []


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
        "detection_aupr": _average_precision(truth, scores),
        "precision_at_recall_90": _precision_at_recall(truth, scores, 0.9),
        "spearman_rho": _spearman_rho(truth, scores),
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
        "claim_recall",
        "claim_f1",
        "claim_support_rate",
    ):
        # ``None`` marks a metric that is undefined for that row (claim recall
        # on a hallucinated row, for instance) — skip those rows rather than
        # averaging them in as zero.
        vals = [
            r[key] for r in per_query
            if key in r and isinstance(r[key], (int, float))
            and not isinstance(r[key], bool)
        ]
        if vals:
            aggregate[key] = statistics.fmean(vals)

    # Per-query wall-clock latency, summed from the per-step timings of the
    # same rows. The paper reports this figure, so it has to come from the
    # run that produced the other numbers rather than from a separate
    # timing estimate. Rows without timings are skipped rather than counted
    # as zero.
    latencies = [
        sum(steps.values())
        for steps in (r.get("step_times_ms") or {} for r in per_query)
        if steps
    ]
    if latencies:
        aggregate["latency_mean_s"] = statistics.fmean(latencies) / 1000.0
        aggregate["latency_median_s"] = statistics.median(latencies) / 1000.0
        aggregate["latency_min_s"] = min(latencies) / 1000.0
        aggregate["latency_max_s"] = max(latencies) / 1000.0
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
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse the rows already present in --out and only evaluate the "
             "remaining ones. Lets a killed run continue without repeating "
             "completed work.",
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
    n_failed = 0
    failures: list[str] = []

    t_start = time.time()

    # Resume support: reuse the rows an earlier run already finished so a
    # killed run — or a run restarted after a code change — does not throw
    # away completed work. A checkpoint row is reused only when it came
    # from the same dataset row (same query, same label), so a checkpoint
    # from a different split can never be silently mixed in.
    reused = 0
    if getattr(args, "resume", False):
        checkpoint = _load_checkpoint(out_path)
        # A checkpoint row is only reusable if it carries every field the
        # current code produces. A checkpoint written by an older version
        # would otherwise be mixed in silently and produce a metrics file
        # whose rows are not comparable.
        required = ("detector_score", "label", "claim_recall", "claim_f1")
        usable = [
            rec for rec in checkpoint
            if all(k in rec for k in required)
        ]
        if checkpoint and not usable:
            print(
                f"  resume: {out_path} has {len(checkpoint)} row(s) but none "
                f"carry the current fields {required}; starting from scratch.",
                file=sys.stderr, flush=True,
            )
        # Rows scored by a *different* detector are not comparable with rows
        # scored by the current one, so they must not be mixed. The
        # fingerprint covers the detector's training data and the library
        # version; a mismatch means the whole checkpoint is stale.
        fingerprint = _detector_fingerprint()
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            stored = payload.get("detector_fingerprint")
        except Exception:  # noqa: BLE001
            stored = None
        if checkpoint and stored and stored != fingerprint:
            print(
                f"  resume: {out_path} was scored by detector {stored} but the "
                f"current detector is {fingerprint}; starting from scratch.",
                file=sys.stderr, flush=True,
            )
            usable = []
        elif checkpoint and not stored:
            print(
                f"  resume: {out_path} has no detector fingerprint; starting "
                f"from scratch so rows from a different detector are not mixed.",
                file=sys.stderr, flush=True,
            )
            usable = []
        by_key: dict[tuple, dict] = {}
        for rec in usable:
            key = (rec.get("query", ""), rec.get("label"))
            by_key.setdefault(key, rec)
        for idx, r in enumerate(rows):
            gt_pre = int(r["label"]) if r.get("label") in {0, 1, "0", "1"} else None
            key = (r.get("query") or r.get("question") or "", gt_pre)
            rec = by_key.get(key)
            if rec is None:
                break
            per_query.append(rec)
            detector_truth.append(int(rec["label"]))
            detector_score.append(float(rec["detector_score"]))
            query_groups.append(rec.get("query") or f"row-{idx}")
            e2e.append(int(rec.get("end_to_end_accuracy", 0)))
            if "hrr" in rec:
                hrr_list.append(float(rec["hrr"]))
            if "ccr" in rec:
                ccr_list.append(float(rec["ccr"]))
            if "vsr" in rec:
                vsr_list.append(float(rec["vsr"]))
            reused += 1
        if reused:
            print(
                f"  resuming: reused {reused}/{len(rows)} rows from {out_path}",
                file=sys.stderr, flush=True,
            )
            _write_checkpoint(out_path, slug, per_query)

    for idx, r in enumerate(rows):
        if idx < reused:
            continue
        row_t = time.time()

        # Determine ground truth *before* running the pipeline: step 13 needs
        # the label and the gold answer to compute claim-level recall.
        if "label" in r and r["label"] in {0, 1, "0", "1"}:
            gt = int(r["label"])
            gt_known = True
        else:
            gt = None
            gt_known = False

        try:
            # Verify the dataset's own answer. The rows are paired: for one
            # question the dataset supplies a correct answer (label 0) and a
            # hallucinated one (label 1). Generating an answer from the
            # query alone would give both rows the same answer and the same
            # detector features, forcing the ROC-AUC to exactly 0.5.
            res = pipe.run(
                r["query"],
                answer=r.get("answer"),
                label=gt if gt_known else None,
                gold_answer=r.get("right_answer"),
            )
            d = res.to_dict()
        except Exception as exc:  # noqa: BLE001
            # A row that fails is dropped, but never silently: count it and
            # keep the first few messages so a systematic failure is visible
            # in the log instead of quietly shrinking the sample.
            n_failed += 1
            if len(failures) < 5:
                failures.append(f"row {idx}: {type(exc).__name__}: {exc}")
            continue

        if not gt_known:
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

        # Claim-level metrics against the row's ground truth. Uses the same
        # helper as step 13 so the two can never disagree.
        claim_metrics = _claim_level_metrics(
            [Claim(**c) for c in d.get("claims", [])],
            gt,
            str(r.get("right_answer") or ""),
        )
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
            # Claim-level metrics. Recall and F1 are None when they are
            # undefined (no gold answer, or a hallucinated row where the
            # benchmark does not say which claims are wrong).
            "claim_precision": claim_metrics["claim_precision"],
            "claim_recall": claim_metrics["claim_recall"],
            "claim_f1": claim_metrics["claim_f1"],
            "claim_support_rate": claim_metrics["claim_support_rate"],
            "step_times_ms": {
                t.name: round(float(t.duration_ms), 3)
                for t in (res.step_traces or [])
            },
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
            elapsed = time.time() - t_start
            row_dt = time.time() - row_t
            # Average over the rows this run actually evaluated, not over the
            # reused ones — otherwise a resumed run reports a near-zero
            # average and a meaningless ETA.
            n_done = idx + 1 - reused
            avg = (elapsed / n_done) if n_done > 0 else 0.0
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
    existing["n_failed"] = n_failed
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    if n_failed:
        print(
            f"WARNING: {n_failed} row(s) failed and were skipped. "
            f"First few: {failures}",
            file=sys.stderr, flush=True,
        )
    print(f"wrote {out_path} ({slug}, {len(per_query)} rows)")
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
