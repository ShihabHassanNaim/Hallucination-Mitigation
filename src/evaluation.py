"""Phase 1 evaluation harness.

These are **proxy** metrics designed to run without NLI models. They are NOT
a substitute for proper NLI-based faithfulness scoring — that arrives in
Phase 5. Think of them as cheap smoke-test signals:

  - exact_match          : answer string == reference (case/whitespace-normalised)
  - token_f1             : token-level F1 against reference
  - refusal_rate         : fraction of answers that say "I don't know"
  - ungrounded_token_rate: fraction of answer tokens NOT present in any retrieved doc
                           (high = model likely hallucinating)
  - avg_retrieval_score  : mean top-1 retrieval cosine score

For a real Phase 1 paper baseline, add NLI-graded faithfulness here once the
Phase 5 verifier is ready.
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


# --- tokenisation -----------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _WS.sub(" ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    return _norm(text).split()


# --- per-item metrics -------------------------------------------------------

@dataclass
class ItemScore:
    id: str
    exact_match: float
    token_f1: float
    refused: float
    ungrounded_token_rate: float
    avg_retrieval_score: float


def score_item(item: dict) -> ItemScore:
    answer = item.get("answer", "") or ""
    reference = item.get("reference_answer") or ""
    retrieved = item.get("retrieved_docs", []) or []
    retrieved_texts = [d.get("text", "") for d in retrieved]
    retrieved_scores = [d.get("score", 0.0) for d in retrieved]

    norm_answer = _norm(answer)
    norm_ref = _norm(reference)
    em = 1.0 if norm_answer and norm_answer == norm_ref else 0.0

    # Token F1
    f1 = _token_f1(_tokens(answer), _tokens(reference))

    # Refusal
    refused = 1.0 if _is_refusal(answer) else 0.0

    # Ungrounded token rate
    ungrounded = _ungrounded_token_rate(answer, retrieved_texts)

    avg_score = sum(retrieved_scores) / len(retrieved_scores) if retrieved_scores else 0.0

    return ItemScore(
        id=str(item.get("id", "")),
        exact_match=em,
        token_f1=f1,
        refused=refused,
        ungrounded_token_rate=ungrounded,
        avg_retrieval_score=avg_score,
    )


def _is_refusal(answer: str) -> bool:
    n = _norm(answer)
    triggers = [
        "i dont know",
        "i do not know",
        "im not sure",
        "i am not sure",
        "no information",
        "cannot answer",
        "cant answer",
        "insufficient information",
    ]
    return any(t in n for t in triggers)


def _token_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = {}
    for t in pred:
        common[t] = min(pred.count(t), gold.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def _ungrounded_token_rate(answer: str, retrieved_texts: Sequence[str]) -> float:
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    retrieved_norm = _norm(" ".join(retrieved_texts))
    ungrounded = sum(1 for t in ans_tokens if t not in retrieved_norm)
    return ungrounded / len(ans_tokens)


# --- aggregate --------------------------------------------------------------

def aggregate(scores: Iterable[ItemScore]) -> dict:
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return {"n": 0}

    def _mean(key: str) -> float:
        return sum(getattr(s, key) for s in scores) / n

    return {
        "n": n,
        "exact_match": round(_mean("exact_match"), 4),
        "token_f1": round(_mean("token_f1"), 4),
        "refusal_rate": round(_mean("refused"), 4),
        "ungrounded_token_rate": round(_mean("ungrounded_token_rate"), 4),
        "avg_retrieval_score": round(_mean("avg_retrieval_score"), 4),
    }


# --- file I/O ---------------------------------------------------------------

def load_predictions(path: str | Path) -> List[dict]:
    p = Path(path)
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_metrics(metrics: dict, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")