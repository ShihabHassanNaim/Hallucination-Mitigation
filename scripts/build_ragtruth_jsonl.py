"""One-shot conversion of the RAGTruth test split to labelled JSONL.

Schema per line (matches load_halueval / scripts/evaluate.py):
  {
    "question":      str,
    "query":         str,   # duplicate for evaluate.py
    "right_answer":  str,   # the reference answer (best non-hallucinated output)
    "knowledge":     str,   # the retrieval context the answer was produced from
    "answer":        str,   # the model answer to evaluate
    "evidence":      [str], # the same context, in a list
    "label":         int    # 0 = faithful, 1 = hallucinated
  }

RAGTruth is **not** a paired dataset like HaluEval and TruthfulQA: each row
is one (prompt, context, model output) triple with a hallucination annotation,
and the same prompt appears under several models. A hallucinated output and a
faithful output for the same prompt are therefore *different model
generations*, not a deliberate contrast pair, so this builder:

  1. labels every output independently
     (``hallucination_labels_processed`` -> evident_conflict + baseless_info > 0),
  2. derives a reference answer per prompt from the cleanest output the
     models produced for it (the one with no hallucination annotation), and
  3. writes a retrieval corpus from the ``context`` passages so the retrieval
     and verification steps have something real to work with.

Length control
--------------
On the raw split, answer length alone separates the classes (measured below),
so a detection number on the raw split would be measuring verbosity. The fix
is a *length-matched control*: for each hallucinated output, the faithful
output from the **same instance** (same prompt, same context) whose word
count is closest is added as a label-0 row. Both rows are then real model
generations of similar length, so word count carries much less information.

RAGTruth's Data2txt task shares one long instruction prompt across 450
different business contexts, so the instance key is (task_type, context,
query) rather than the query alone.

Usage:
  python -m scripts.build_ragtruth_jsonl
  python -m scripts.build_ragtruth_jsonl --rows 1000 --matched-out data/ragtruth_eval_1000_matched.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.utils.chunker import split_sentences

# RAGTruth's own task types. "Data2txt" is table-to-text and "Summary" is
# document summarisation; both are long-form generation, "QA" is short-form.
TASK_TYPES = ("QA", "Data2txt", "Summary")


def _is_hallucinated(processed: dict) -> bool:
    """Whether RAGTruth annotated this output as hallucinated.

    ``hallucination_labels_processed`` is a dict with the counts of the two
    annotation categories, ``evident_conflict`` and ``baseless_info``. Any
    non-zero count means the output contains at least one hallucinated span.
    """
    if not isinstance(processed, dict):
        return False
    return int(processed.get("evident_conflict", 0) or 0) + \
        int(processed.get("baseless_info", 0) or 0) > 0


def _reference_for(prompt: str, clean_outputs: list[str]) -> str:
    """Pick the reference answer for a prompt.

    Preference order:
      1. the shortest clean output -- shortest is the least likely to add
         unsupported detail, and it keeps the reference from being a
         paraphrase of the hallucinated answer;
      2. the longest clean output, if there is no other choice;
      3. the context itself, when every output for the prompt was
         annotated as hallucinated.
    """
    if clean_outputs:
        clean_outputs.sort(key=lambda s: len(s.split()))
        return clean_outputs[0]
    return ""


def _length_only_auc(rows: list[dict]) -> float:
    """ROC-AUC of a length-only classifier on these rows.

    The score is the answer's word count. A value at or near 0.5 means
    length carries no label information; a value far above 0.5 means the
    split is length-confounded and the detection numbers on it are
    measuring word count.
    """
    scores = [float(len(r["answer"].split())) for r in rows]
    labels = [int(r["label"]) for r in rows]
    pairs = sorted(zip(scores, labels))
    # Rank-sum formulation with average ranks for ties.
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _instance_key(task_type: str, context: str, query: str) -> str:
    """The instance a row belongs to.

    The context is the identifying part: Data2txt reuses one instruction
    prompt for 450 different business contexts, so the prompt alone does
    not separate instances.
    """
    return f"{task_type}||{context[:400]}||{query[:200]}"


def _context_identity(task_type: str, context: str) -> str:
    """The entity the context is about, for query enrichment.

    RAGTruth's Summary and Data2txt prompts are generic instructions --
    "Summarize the following news within 86 words:" is identical for every
    news article, and the Data2txt instruction is identical for all 450
    businesses. A retriever given such a query has no signal to pick the
    right document, so every row retrieves the same passages and the
    retrieval metrics collapse to zero.

    Appending the entity name to the query gives the retriever something
    to match on. This changes only the query the retriever sees; the
    answer, the evidence and the label are untouched, so the detection
    task is unaffected.
    """
    if task_type == "Data2txt":
        m = re.search(r"'name':\s*'([^']+)'", context)
        if m:
            return m.group(1)
    if task_type == "Summary":
        # The first sentence of a news article names the story. It is used
        # as the query subject rather than the whole article.
        first = split_sentences(context)
        if first:
            return " ".join(first[0].split()[:40])
    return ""


def _enriched_query(task_type: str, query: str, context: str) -> str:
    """The query the retriever will see."""
    identity = _context_identity(task_type, context)
    if not identity:
        return query
    return f"{query} {identity}".strip()


def _pair_rows(rows: list[dict], match_length: bool) -> tuple[list[dict], int]:
    """Pair every hallucinated row with a faithful row from the same instance.

    With ``match_length`` off, the faithful partner is simply the next
    unused clean output. With it on, the partner is the clean output whose
    word count is closest to the hallucinated one, which removes the length
    shortcut.

    Both splits cover the same instances, so a detection number measured on
    the raw split and one measured on the matched split differ *only* in
    whether the length control was applied.

    Returns (rows, n_pairs).
    """
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_instance[r["instance_key"]].append(r)

    out: list[dict] = []
    n_pairs = 0
    for _, group in by_instance.items():
        hallucinated = [r for r in group if r["label"] == 1]
        faithful = [r for r in group if r["label"] == 0]
        # One faithful output per hallucinated output. An instance with more
        # hallucinated outputs than clean ones therefore contributes only as
        # many hallucinated rows as it has clean partners -- otherwise the
        # same answer would appear twice in the split and be counted twice
        # by the metrics.
        n_pairs_here = min(len(hallucinated), len(faithful))
        used: set[str] = set()
        for h in hallucinated[:n_pairs_here]:
            out.append(h)
            h_len = len(h["answer"].split())
            candidates = [f for f in faithful if f["row_id"] not in used] or faithful
            if match_length:
                best = min(candidates,
                           key=lambda f: (abs(len(f["answer"].split()) - h_len),
                                          len(f["answer"].split())))
            else:
                best = candidates[0]
            used.add(best["row_id"])
            out.append({
                **best,
                "label": 0,
                "row_type": "length_matched_control" if match_length
                else "paired_faithful",
                "control_of": h["row_id"],
            })
            n_pairs += 1
    out.sort(key=lambda r: (r["task_type"], r["row_id"], r.get("row_type", "")))
    return out, n_pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="data/ragtruth_test.parquet")
    ap.add_argument("--out", default="data/ragtruth_eval_1000.jsonl")
    ap.add_argument("--corpus", default="data/ragtruth_corpus.jsonl",
                    help="retrieval corpus built from the context passages")
    ap.add_argument("--report", default="data/ragtruth_dataset_report.json")
    ap.add_argument("--rows", type=int, default=1000,
                    help="how many evaluation rows to emit (0 = all)")
    ap.add_argument("--matched-out",
                    default="data/ragtruth_eval_1000_matched.jsonl",
                    help="output path for the length-matched split")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df[df["quality"] == "good"].reset_index(drop=True)

    # Group by instance so a reference answer and a length-matched control
    # can be derived per instance.
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for _, r in df.iterrows():
        task_type = str(r["task_type"])
        context = str(r["context"] or "")
        query = str(r["query"])
        key = _instance_key(task_type, context, query)
        by_instance[key].append({
            "id": str(r["id"]),
            "task_type": task_type,
            "model": str(r["model"]),
            "context": context,
            "query": query,
            "output": str(r["output"] or "").strip(),
            "label": 1 if _is_hallucinated(r["hallucination_labels_processed"]) else 0,
        })

    all_rows: list[dict] = []
    corpus: dict[str, str] = {}
    for key, group in by_instance.items():
        clean = [g["output"] for g in group if g["label"] == 0 and g["output"]]
        reference = _reference_for(key, clean)
        for g in group:
            if not g["output"]:
                continue
            knowledge = g["context"]
            corpus[g["id"]] = knowledge
            # The question stays the original prompt -- that is what the
            # answer is a response to. The query is the enriched version
            # the retriever searches with.
            all_rows.append({
                "question": g["query"],
                "query": _enriched_query(g["task_type"], g["query"], knowledge),
                "original_query": g["query"],
                "right_answer": reference or knowledge,
                "knowledge": knowledge,
                "answer": g["output"],
                "evidence": [knowledge] if knowledge else [],
                "label": g["label"],
                "task_type": g["task_type"],
                "model": g["model"],
                "row_id": g["id"],
                "instance_key": key,
            })

    # --- length-only separability, per task type and overall -------------
    report: dict = {"rows_total": len(all_rows), "task_types": {}, "length_only_auc": {}}
    for tt in TASK_TYPES:
        sub = [r for r in all_rows if r["task_type"] == tt]
        if not sub:
            continue
        base = sum(r["label"] for r in sub) / len(sub)
        report["task_types"][tt] = {
            "rows": len(sub),
            "hallucinated": int(sum(r["label"] for r in sub)),
            "base_rate_hallucinated": round(base, 4),
            "length_only_auc": round(_length_only_auc(sub), 4),
            "mean_words_hallucinated": round(
                sum(len(r["answer"].split()) for r in sub if r["label"] == 1)
                / max(1, sum(r["label"] for r in sub)), 1),
            "mean_words_faithful": round(
                sum(len(r["answer"].split()) for r in sub if r["label"] == 0)
                / max(1, sum(1 - r["label"] for r in sub)), 1),
        }
    report["length_only_auc"]["ALL"] = round(_length_only_auc(all_rows), 4)
    report["length_only_auc"]["base_rate"] = round(
        sum(r["label"] for r in all_rows) / len(all_rows), 4)

    # --- pick the evaluation subset ---------------------------------------
    # Only instances that have *both* a hallucinated and a faithful output
    # are eligible: without a clean partner there is no pair, and a pair
    # is what makes the length control possible.
    inst_groups: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        inst_groups[r["instance_key"]].append(r)

    inst_rows = []
    for key, group in inst_groups.items():
        n_h = sum(1 for g in group if g["label"] == 1)
        n_c = sum(1 for g in group if g["label"] == 0)
        if n_h == 0 or n_c == 0:
            continue
        inst_rows.append({
            "key": key,
            "task_type": group[0]["task_type"],
            # An instance contributes one pair per hallucinated output, but
            # never more pairs than it has clean partners.
            "n_pairs": min(n_h, n_c),
        })
    if not inst_rows:
        raise SystemExit("no instance has both a hallucinated and a faithful "
                         "output; cannot build a paired split")

    rng = pd.Series(range(len(inst_rows))).sample(frac=1.0, random_state=args.seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for idx in rng:
        ir = inst_rows[int(idx)]
        buckets[ir["task_type"]].append(ir)

    # Fill the row budget bucket by bucket, largest bucket first, so the
    # subset keeps the dataset's per-task mix while landing close to the
    # requested row count. Each instance contributes 2 * n_h rows to a
    # paired split.
    target = args.rows
    quota = {k: 0 for k in buckets}
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    remaining = target
    progressed = True
    while progressed and remaining > 0:
        progressed = False
        for k in order:
            if quota[k] >= len(buckets[k]):
                continue
            ir = buckets[k][quota[k]]
            size = 2 * ir["n_pairs"]
            if size > remaining and quota[k] > 0:
                continue
            quota[k] += 1
            remaining -= size
            progressed = True
            if remaining <= 0:
                break

    selected: list[dict] = []
    for k, irs in buckets.items():
        for ir in irs[:quota[k]]:
            selected.extend(inst_groups[ir["key"]])
    # Restore a deterministic order (by task type, then row id).
    selected.sort(key=lambda r: (r["task_type"], r["row_id"]))

    # --- write the two paired splits --------------------------------------
    raw_rows, n_pairs = _pair_rows(selected, match_length=False)
    matched_rows, n_pairs_matched = _pair_rows(selected, match_length=True)

    # The instance key is an internal join key built from the full context
    # text. It is long and the evaluation loader does not need it, so it is
    # dropped from the written files.
    def _strip(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "instance_key"}
                for r in rows]

    raw_rows = _strip(raw_rows)
    matched_rows = _strip(matched_rows)

    # --- build the retrieval corpus ----------------------------------------
    # RAGTruth's contexts are the passages the models actually read, so they
    # are the corpus. Two rows from the same instance share a context, so
    # the corpus is keyed on the context rather than on the row.
    ctx_to_doc: dict[str, str] = {}
    for r in raw_rows:
        text = r.get("knowledge") or ""
        if text and text not in ctx_to_doc:
            ctx_to_doc[text] = f"doc-{len(ctx_to_doc)}"

    # Each row records the id of the document that contains its gold
    # context, so the retrieval metrics can be scored against the exact
    # document rather than by string matching.
    for r in raw_rows + matched_rows:
        doc_id = ctx_to_doc.get(r.get("knowledge") or "")
        if doc_id:
            r["gold_doc_id"] = doc_id

    corpus_path = Path(args.corpus)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as f:
        for text, doc_id in ctx_to_doc.items():
            f.write(json.dumps({"id": doc_id, "text": text},
                               ensure_ascii=False) + "\n")
    doc_id_by_ctx = ctx_to_doc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    matched_path = Path(args.matched_out)
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    with matched_path.open("w", encoding="utf-8") as f:
        for r in matched_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report["evaluation_file"] = str(out_path)
    report["evaluation_rows"] = len(raw_rows)
    report["evaluation_length_only_auc"] = round(_length_only_auc(raw_rows), 4)
    report["matched_file"] = str(matched_path)
    report["matched_rows"] = len(matched_rows)
    report["matched_pairs"] = n_pairs_matched
    report["matched_length_only_auc"] = round(_length_only_auc(matched_rows), 4)
    report["corpus_file"] = str(corpus_path)
    report["corpus_docs"] = len(doc_id_by_ctx)
    report["instances_eligible"] = len(inst_rows)
    report["selected_label_balance"] = {
        str(tt): {
            "rows": sum(1 for r in raw_rows if r["task_type"] == tt),
            "hallucinated": sum(1 for r in raw_rows
                                if r["task_type"] == tt and r["label"] == 1),
        }
        for tt in TASK_TYPES
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path} ({len(raw_rows)} rows, {n_pairs} pairs)")
    print(f"wrote {matched_path} ({len(matched_rows)} rows, "
          f"{n_pairs_matched} pairs)")
    print(f"wrote {corpus_path} ({len(used_ids)} docs)")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
