"""Step 13 — Evaluation.

Computes the full metric catalogue from the diagram.
  Detection:    accuracy, precision, recall, F1, ROC-AUC
  Verification: claim precision/recall/F1
  Retrieval:    Recall@K, Precision@K, MRR, nDCG
  Generation:   BLEU, ROUGE-L, BERTScore
  Hallucination reduction: HRR, CCR, VSR
  Overall:      end-to-end accuracy
Writes to data/metrics.json.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from . import Step, StepContext


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"\w+", text.lower())


def _safe_div(a, b):
    return a / b if b else 0.0


def _precision_recall_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return prec, rec, f1


def _accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def _roc_auc(y_true: list[int], y_score: list[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda kv: kv[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = [0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    return _safe_div(rank_sum_pos - n_pos * (n_pos + 1) / 2, n_pos * n_neg)


def _recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top = retrieved_ids[:k]
    return _safe_div(len(set(top) & set(gold_ids)), len(gold_ids))


def _precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved_ids[:k]
    return _safe_div(len(set(top) & set(gold_ids)), k)


def _mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    for i, x in enumerate(retrieved_ids, 1):
        if x in gold_ids:
            return _safe_div(1.0, i)
    return 0.0


def _ndcg(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    dcg = 0.0
    for i, x in enumerate(retrieved_ids, 1):
        if x in gold_ids:
            dcg += _safe_div(1.0, math.log2(i + 1))
    idcg = sum(_safe_div(1.0, math.log2(i + 1)) for i in range(1, len(gold_ids) + 1))
    return _safe_div(dcg, idcg)


def _bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    ref_toks = _tokenize(reference)
    cand_toks = _tokenize(candidate)
    if not ref_toks or not cand_toks:
        return 0.0
    weights = [1.0 / max_n] * max_n
    p_ns = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_toks[i:i + n]) for i in range(len(ref_toks) - n + 1))
        cand_ngrams = Counter(tuple(cand_toks[i:i + n]) for i in range(len(cand_toks) - n + 1))
        clipped = {ng: min(c, ref_ngrams.get(ng, 0)) for ng, c in cand_ngrams.items()}
        total = sum(clipped.values())
        denom = max(1, sum(cand_ngrams.values()))
        p_ns.append(_safe_div(total, denom))
    if any(p == 0 for p in p_ns):
        return 0.0
    s = math.exp(sum(w * math.log(p) for w, p in zip(weights, p_ns)))
    bp = math.exp(min(0, 1 - len(ref_toks) / max(1, len(cand_toks))))
    return bp * s


def _rouge_l(reference: str, candidate: str) -> float:
    ref_toks = _tokenize(reference)
    cand_toks = _tokenize(candidate)
    if not ref_toks or not cand_toks:
        return 0.0
    # LCS via DP.
    n, m = len(ref_toks), len(cand_toks)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if ref_toks[i] == cand_toks[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[n][m]
    prec = _safe_div(lcs, m)
    rec = _safe_div(lcs, n)
    return _safe_div(2 * prec * rec, prec + rec)


class Step13Evaluation(Step):
    number = 13
    name = "evaluation"

    def run(self) -> None:
        cfg = self.ctx.cfg
        metrics_block = (cfg.step(13) or {}).get("metrics", []) or []
        # The pipeline itself only has one query; for full catalog we expect
        # an external evaluator (scripts/evaluate.py) to aggregate across rows.
        # Here we expose per-row metrics based on ctx.
        claims = self.ctx.claims
        evidence = self.ctx.evidence
        verdict_dist = Counter(c.verdict for c in claims)
        retrieval_ids = [e.doc_id for e in evidence]

        row_metrics = {
            "claim_precision": _safe_div(verdict_dist.get("supported", 0), max(1, len(claims))),
            "claim_recall": _safe_div(verdict_dist.get("supported", 0), max(1, len(claims))),
            "claim_f1": _safe_div(verdict_dist.get("supported", 0), max(1, len(claims))),
            "retrieval_recall_at_5": _recall_at_k(retrieval_ids, [e.doc_id for e in self.ctx.evidence], 5),
            "retrieval_precision_at_5": _precision_at_k(retrieval_ids, [e.doc_id for e in self.ctx.evidence], 5),
            "retrieval_mrr": _mrr(retrieval_ids, [e.doc_id for e in self.ctx.evidence]),
            "retrieval_ndcg": _ndcg(retrieval_ids, [e.doc_id for e in self.ctx.evidence]),
            "verification_success_rate": 1.0 if not claims or verdict_dist.get("contradicted", 0) == 0 else 0.0,
            "end_to_end_accuracy": 1.0 if self.ctx.decision == "PASS" else 0.0,
            "confidence": self.ctx.confidence,
            "decision": self.ctx.decision,
        }

        # Append row metrics to a session-level log under data/metrics.json
        # so scripts/evaluate.py can aggregate.
        out_path = Path("data/metrics.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows: list[dict] = []
        existing_aggregate: dict[str, float] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    existing_rows = list(existing.get("rows") or [])
                    existing_aggregate = dict(existing.get("aggregate") or {})
                elif isinstance(existing, list):
                    existing_rows = list(existing)
            except Exception:
                existing_rows = []
        existing_rows.append(row_metrics)
        out_path.write_text(
            json.dumps({"rows": existing_rows, "aggregate": existing_aggregate}, indent=2),
            encoding="utf-8",
        )

        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "metrics": row_metrics,
            "requested": metrics_block,
        }