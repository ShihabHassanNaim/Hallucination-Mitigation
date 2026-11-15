#!/usr/bin/env bash
# Re-run the two headline evaluations with the clean detector.
#
# Why this exists
# ---------------
# The original HaluEval and TruthfulQA runs were scored by a detector fitted
# on data/detector_train.jsonl, whose faithful answers have a median length of
# 3 words against 9 for the hallucinated ones. The detector therefore leaned
# on `answer_length` (importance 0.348) and on an `intrinsic_overlap` that was
# itself correlated with length (r = 0.67).
#
# data/detector_train_clean.jsonl replaces it: both classes are length-matched
# (median 10 words on each side), the duplicated gold row is gone, and the
# training queries are disjoint from every evaluation split. The detector now
# leans on `cosine` (importance 0.545) instead.
#
# The scores changed, so the two headline runs must be repeated. The
# checkpoint fingerprint added to evaluate_full.py makes --resume refuse to
# mix rows scored by the old detector with rows scored by the new one, so
# each output file is written from scratch.
#
# The passes share one GPU and cannot run concurrently: each needs ~5 GB of
# the 8 GB available, and two at once will OOM-kill a running evaluation.
set -u

cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"

echo "=== 1/2  HaluEval, length-matched split ==="
rm -f data/metrics_halueval.json
"$PY" -u -m scripts.evaluate_full \
    --data data/halueval_eval_1000_matched.jsonl \
    --out data/metrics_halueval.json \
    --index data/index_halueval \
    --limit 1000 \
    --resume > eval_halueval_matched.log 2>&1
echo "  exit=$? -> eval_halueval_matched.log"

echo "=== 2/2  TruthfulQA ==="
rm -f data/metrics_truthfulqa.json
"$PY" -u -m scripts.evaluate_full \
    --data data/truthfulqa_eval_1000.jsonl \
    --out data/metrics_truthfulqa.json \
    --index data/index_truthfulqa \
    --limit 1000 \
    --resume > eval_truthfulqa_1000.log 2>&1
echo "  exit=$? -> eval_truthfulqa_1000.log"

echo "=== clean-detector evaluations done ==="
