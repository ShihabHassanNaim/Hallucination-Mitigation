---
name: run-eval
description: Run the hallucination-detection evaluation pipeline on a dataset and summarize key metrics.
allowed-tools:
  - Bash(python:*)
  - Bash(ls:*)
  - Read
when_to_use: Use when the user wants to run the project evaluation pipeline (scripts/evaluate.py), check metrics on a dataset, or re-evaluate after code changes. Examples: 'run eval', 'evaluate on halueval', 'rerun evaluation', '/run-eval halueval --limit 50'.
argument-hint: "[dataset-or-file] [--limit N] [--out path.json]"
arguments:
  - dataset_or_file
  - limit
  - out_path
context: inline
---

# Run Evaluation Pipeline

Run `scripts/evaluate.py` against a dataset and surface the key hallucination-detection metrics in a compact table.

## Inputs
- `$dataset_or_file` (optional): one of `synth`, `halueval`, `truthfulqa`, or a path to a `.jsonl` file. Defaults to `synth`.
- `$limit` (optional): integer row count, e.g. `50`, for a quick smoke run. Defaults to no limit.
- `$out_path` (optional): path to write metrics JSON. Defaults to `data/metrics.json` (or the dataset's standard path).

## Goal
Produce a fresh `metrics.json` for the chosen dataset and present a readable summary of:
- claim_precision, claim_recall, claim_f1
- retrieval_recall_at_5, retrieval_precision_at_5, retrieval_mrr, retrieval_ndcg
- verification_success_rate
- end_to_end_accuracy
- confidence (mean)

Success = metrics file exists on disk AND the summary table is printed to the user.

## Steps

### 1. Resolve arguments and locate the virtualenv
Determine the actual Python interpreter (`.venv/Scripts/python.exe` on Windows) and build the CLI command:
- If `$dataset_or_file` is a registered dataset (`halueval`/`truthfulqa`) or empty: pass `--dataset <name>`.
- If it ends with `.jsonl`: pass `--data <path>`.
- If `$limit` is set: append `--limit N`.
- If `$out_path` is set: append `--out <path>`.

**Success criteria**: A fully-formed `python -m scripts.evaluate ...` command is shown to the user before execution.

### 2. Run the evaluation
Execute the command with a 10-minute timeout. Stream stdout/stderr.

**Success criteria**: Process exits 0 AND a metrics JSON file is written under `data/`.

### 3. Summarize key metrics
Read the produced metrics JSON and print a markdown table with the metric names listed in the Goal section, rounded to 4 decimals. Highlight any metric that dropped vs the previous run's file (if `data/metrics.json` already existed) by appending ` (Δ -x.xxxx)` in red.

**Success criteria**: User sees a clean metrics table; deltas vs prior run (if any) are shown.

## Rules
- Never run with `--limit 0` or negative limits.
- If `$dataset_or_file` is neither a known dataset nor a path that exists, ask the user before failing.
- Do not modify any source files — read-only against the codebase except for the metrics output.
- The `.venv` is the canonical interpreter; do not use the system Python.