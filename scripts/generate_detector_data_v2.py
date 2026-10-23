"""Generate real labelled training data for the XGBoost detector (Step 3).

For HaluEval, each row has both a right_answer and a hallucinated answer.
This script bypasses the pipeline's LLM generation and instead feeds the
actual ground-truth answer directly into Step 3's feature extractor, so
the features genuinely differ between faithful and hallucinated examples.

Produces (cosine, claim_count, intrinsic_overlap, answer_length, label)
rows suitable for XGBClassifier.

Usage:
  python -m scripts.generate_detector_data_v2 --dataset halueval --limit 500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from src.config import load_config
from src.models.embedder_bge_m3 import BGE_M3Embedder
from src.utils.data_loader import load_jsonl
from src.utils.claim_extractor import extract_claims, intrinsic_overlap


_DATASET_PATHS = {
    "halueval": "data/halueval_qa.jsonl",
    "truthfulqa": "data/truthfulqa_qa.jsonl",
}


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _length_matched_control(
    query: str, right: str, hallucinated: str, knowledge: str = ""
) -> str:
    """Build a FAITHFUL answer that is as long as the hallucinated one.

    HaluEval gold answers are short factual spans while the hallucinated
    answers are full sentences, so a naive (gold=0, generated=1) pairing lets
    the detector separate the classes on ``answer_length`` alone. This
    constructs a third example: a faithful answer expanded to full-sentence
    length using the dataset's ``knowledge`` field (falling back to the
    query's own words). Both classes then look like generated prose, so
    cosine / overlap / claim structure must carry the signal.

    Returns "" if no credible control can be built.
    """
    import re

    hw = len(hallucinated.split())
    rw = len(right.split())
    if hw <= rw:
        return ""  # gold is already the longer side; nothing to balance

    # Context to expand with: prefer the dataset's knowledge passage, since
    # it is the *supporting evidence* and therefore faithful by construction.
    # Only fall back to query words when knowledge is missing/too short.
    ctx = (knowledge or "").strip()
    need = hw - rw
    if len(ctx.split()) < need:
        qwords = [
            w
            for w in re.findall(r"[A-Za-z0-9']+", query)
            if w.lower() not in right.lower()
        ]
        ctx = f"{ctx} {' '.join(qwords)}".strip()
    if len(ctx.split()) < need:
        return ""  # not enough faithful context to reach the target length

    # Lead with the gold fact, then the supporting context, trimmed to match.
    control = f"{right}, {' '.join(ctx.split()[:need])}"
    out = " ".join(control.split()[:hw]).strip(" ,.")

    # Reject degenerate controls: identical to gold, or too short to be a
    # sentence (which would re-introduce the length shortcut we're removing).
    if len(out.split()) < 4 or out.lower() == right.lower():
        return ""
    return out


def _compute_features(embedder: BGE_M3Embedder, query: str, answer: str) -> dict:
    """Compute the 4 Step-3 features for a (query, answer) pair."""
    import numpy as np

    # Cosine similarity via BGE-M3 embeddings.
    try:
        qv = embedder.encode([query])[0]
        av = embedder.encode([answer])[0]
        cosine = float(
            np.dot(qv, av) / (np.linalg.norm(qv) * np.linalg.norm(av) + 1e-12)
        )
    except Exception:
        cosine = 0.0

    # Claim count via rule-based sentence splitter.
    claim_texts = extract_claims(answer or "", mode="rule")
    claim_count = max(1, len(claim_texts))

    # Intrinsic overlap (Jaccard on word sets).
    overlap = intrinsic_overlap(answer or "", query)

    # Answer length in tokens.
    answer_length = len((answer or "").split())

    return {
        "cosine": cosine,
        "claim_count": float(claim_count),
        "intrinsic_overlap": float(overlap),
        "answer_length": float(answer_length),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        choices=sorted(_DATASET_PATHS),
        help="Labelled dataset to mine features from.",
    )
    ap.add_argument("--data", help="Custom JSONL path (overrides --dataset).")
    ap.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of rows to process (default 500).",
    )
    ap.add_argument(
        "--out",
        default="data/detector_train.jsonl",
        help="Output JSONL path (default data/detector_train.jsonl).",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="Also emit a length-matched FAITHFUL control per hallucinated "
             "row so `answer_length` cannot act as a label shortcut. "
             "Strongly recommended: without it the detector learns length, "
             "not hallucination.",
    )
    args = ap.parse_args()

    if args.data:
        rows = load_jsonl(args.data)
    else:
        rows = load_jsonl(_DATASET_PATHS[args.dataset])

    rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows.", flush=True)

    # Load only the embedder (no LLM / NLI / reranker needed).
    cfg = load_config()
    embedder = BGE_M3Embedder(model_name=(cfg.models or {}).get("embedder", "BAAI/bge-m3"))
    embedder._load_hf()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    t_start = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(rows):
            query = r.get("query") or r.get("question") or ""
            if not query:
                skipped += 1
                continue

            # HaluEval provides both right_answer and the hallucinated
            # answer. Emit one row per side so the model sees both classes.
            right = (r.get("right_answer") or "").strip()
            hallucinated = (r.get("answer") or "").strip()

            # CONFOUND WARNING: in HaluEval the gold answer is a short
            # factual span while the hallucinated answer is a full generated
            # sentence. Emitting that pair alone teaches the detector to read
            # `answer_length` as the label (gold ~2 words, hallucinated ~11),
            # which inflates held-out AUC while learning nothing transferable.
            #
            # `--balance` neutralises this by also emitting a length-matched
            # faithful control: the gold answer padded into a full sentence
            # with the query's own context words. Both classes then look like
            # generated prose, so cosine/overlap carry the signal instead.
            pairs = []
            if right:
                pairs.append((0, right))         # label 0 = faithful (short)
            if hallucinated and hallucinated != right:
                pairs.append((1, hallucinated))  # label 1 = hallucinated (long)
            if args.balance and right and hallucinated and hallucinated != right:
                control = _length_matched_control(
                    query, right, hallucinated, r.get("knowledge") or ""
                )
                if control and control != right:
                    pairs.append((0, control))   # label 0 = faithful (long)

            if not pairs:
                skipped += 1
                continue

            for label_int, answer_text in pairs:
                try:
                    feats = _compute_features(embedder, query, answer_text)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [row {idx}] feature failed: {exc}", file=sys.stderr)
                    skipped += 1
                    continue

                row_out = {
                    "query": query,
                    "label": label_int,
                    "features": [
                        feats["cosine"],
                        feats["claim_count"],
                        feats["intrinsic_overlap"],
                        feats["answer_length"],
                    ],
                }
                f.write(json.dumps(row_out) + "\n")
                written += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
                elapsed = time.time() - t_start
                avg = elapsed / (idx + 1)
                eta = avg * (len(rows) - idx - 1)
                print(
                    f"  [{idx+1}/{len(rows)}] avg={avg:.1f}s eta={eta:.0f}s "
                    f"written={written} skipped={skipped}",
                    flush=True,
                )

    print(f"wrote {out_path} ({written} rows, {skipped} skipped)")


if __name__ == "__main__":
    main()
