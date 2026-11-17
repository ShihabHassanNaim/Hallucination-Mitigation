"""Diagnose whether TruthfulQA's retrieval corpus leaks the label.

The corpus (``data/truthfulqa_corpus.jsonl``) was built from the dataset's
own ``knowledge`` field, which is the set of *correct* answers. If the
retrieved document for a row contains that row's own answer, then any
verification against it -- the NLI in step 8 or the grounding term in step 9
-- is measuring "is this answer the reference answer", which is the label.

Three questions are answered here:

1. Does the row's own answer appear verbatim in its retrieved document?
2. Does it appear anywhere in the corpus?
3. How many *other* rows' answers appear in the retrieved document?

The third question is the one that decides whether the corpus is usable at
all: a corpus that contains every answer is an answer key; a corpus that
contains a mix is a corpus.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.utils.data_loader import load_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="data/truthfulqa_eval_1000.jsonl")
    ap.add_argument("--corpus", default="data/truthfulqa_corpus.jsonl")
    ap.add_argument("--out", default="data/truthfulqa_leak_report.json")
    args = ap.parse_args()

    rows = load_jsonl(args.rows)
    corpus = load_jsonl(args.corpus)
    corpus_texts = [c.get("text", "") for c in corpus]
    corpus_set = {t.strip() for t in corpus_texts}

    # Every answer in the dataset, so we can ask how much of the corpus a
    # retrieved document is made of.
    answers = {}
    for r in rows:
        answers.setdefault(r.get("question", ""), set()).add(
            (r.get("answer") or "").strip()
        )

    # Which corpus doc corresponds to which question? The corpus doc's text
    # is the ';' joined knowledge list of some row.
    q_by_knowledge = {}
    for r in rows:
        k = (r.get("knowledge") or "").strip()
        if k:
            q_by_knowledge.setdefault(k, r.get("question", ""))

    # How many answers does each corpus doc contain?
    doc_answers: dict[str, set] = {}
    for t in corpus_set:
        parts = {p.strip() for p in t.split(";") if p.strip()}
        doc_answers[t] = parts

    # --- question 1 and 2: does the row's answer appear in the corpus? ----
    own_answer_in_corpus = 0
    by_label = Counter()
    for r in rows:
        a = (r.get("answer") or "").strip()
        k = (r.get("knowledge") or "").strip()
        if a and a in doc_answers.get(k, set()):
            own_answer_in_corpus += 1
            by_label[int(r.get("label", 0))] += 1
        if a and a in corpus_set:
            pass  # exact whole-document match, counted below

    exact_doc_match = sum(
        1 for r in rows
        if (r.get("answer") or "").strip() in corpus_set
    )

    # --- question 3: how much of the corpus does a retrieved doc cover? ---
    # For each corpus doc, how many *distinct* dataset answers does it hold?
    all_answers = set()
    for s in answers.values():
        all_answers |= s
    sizes = [
        len(doc_answers[t] & all_answers)
        for t in corpus_set
    ]
    sizes.sort()

    # How many of the 500 truthful answers are inside the corpus docs that
    # correspond to *other* questions? If ~0, the corpus is a per-question
    # answer key and every retrieval finds the answer.
    cross = 0
    for t in corpus_set:
        own_q = q_by_knowledge.get(t, "")
        for a in answers.get(own_q, set()):
            if a in doc_answers[t]:
                continue
        # count answers of *other* questions present in this doc
        for q, ans in answers.items():
            if q == own_q:
                continue
            if any(a in doc_answers[t] for a in ans):
                cross += 1
                break

    report = {
        "n_rows": len(rows),
        "n_corpus_docs": len(corpus_set),
        "n_distinct_answers": len(all_answers),
        "rows_whose_own_answer_is_in_their_own_knowledge_doc": own_answer_in_corpus,
        "by_label": dict(by_label),
        "rows_whose_answer_is_an_entire_corpus_doc": exact_doc_match,
        "corpus_docs_containing_another_questions_answer": cross,
        "doc_answer_count": {
            "min": sizes[0] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "max": sizes[-1] if sizes else 0,
        },
    }
    print(json.dumps(report, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
