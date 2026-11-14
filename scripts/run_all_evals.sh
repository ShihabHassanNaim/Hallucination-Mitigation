#!/usr/bin/env bash
# Run the full evaluation programme sequentially.
#
# The passes share one GPU and cannot run concurrently (each needs ~5 GB of
# the 8 GB available), so they are chained. Every script checkpoints as it
# goes, so a failure only costs the run in progress.
#
#   1. HaluEval, length-matched split  -> the headline detection result
#   2. TruthfulQA                      -> an independent check, no confound
#   3. HaluEval, raw split             -> reported for transparency only
#   4. Ablation study                  -> per-step contribution
set -u

cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"

echo "=== 1/3  HaluEval, length-matched split ==="
"$PY" -u -m scripts.evaluate_full \
    --data data/halueval_eval_1000_matched.jsonl \
    --out data/metrics_halueval.json \
    --index data/index_halueval \
    --limit 1000 > eval_halueval_matched.log 2>&1
echo "  exit=$? -> eval_halueval_matched.log"

echo "=== 2/3  TruthfulQA ==="
"$PY" -u -m scripts.evaluate_full \
    --data data/truthfulqa_eval_1000.jsonl \
    --out data/metrics_truthfulqa.json \
    --index data/index_truthfulqa \
    --limit 1000 > eval_truthfulqa_1000.log 2>&1
echo "  exit=$? -> eval_truthfulqa_1000.log"

echo "=== 3/3  HaluEval, raw split (length-confounded, reported for transparency) ==="
"$PY" -u -m scripts.evaluate_full \
    --data data/halueval_eval_1000.jsonl \
    --out data/metrics_halueval_raw.json \
    --index data/index_halueval \
    --limit 1000 > eval_halueval_raw.log 2>&1
echo "  exit=$? -> eval_halueval_raw.log"

echo "=== 4/4  Ablation study (200 rows, length-matched split) ==="
"$PY" -u -m scripts.ablation_study \
    --data data/halueval_eval_1000_matched.jsonl \
    --index data/index_halueval \
    --limit 200 \
    --out data/ablation_results.json > eval_ablation.log 2>&1
echo "  exit=$? -> eval_ablation.log"

echo "=== all evaluations done ==="
