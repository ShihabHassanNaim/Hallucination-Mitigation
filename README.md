# Hallucination Detection Pipeline

> A 13-step hallucination-detection and fact-verification pipeline for
> RAG systems, built one-to-one against the architecture diagram.
> Runs on real HuggingFace checkpoints by default — every component
> tries its HF model first and only falls back to a deterministic
> stub if a download fails (offline, gated, missing dependency).

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![tests](https://img.shields.io/badge/tests-31%20passing-brightgreen)]()
[![mode](https://img.shields.io/badge/mode-real%20HuggingFace%20models-blue)]()
[![ui](https://img.shields.io/badge/UI-Flask%20%2B%20FastAPI-blueviolet)]()

---

## Architecture (one-to-one with the diagram)

```
                              User Question
                                    |
                                    v
  +------------------------------+
  | 1. Query Understanding       |   GLiNER + spaCy + Llama 3.1
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 2. Grounded Generation       |   Llama 3.1 / Qwen 3
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 3. Early Detection           |   BGE-M3 cosine + XGBoost
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 4. Confidence Assessment     |   EEDC + temperature scaling
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 5. Atomic Claim Extraction  |   rule-based + LLM splitter
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 6. Hybrid Retrieval          |   BGE-M3 + FAISS + BM25
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 7. Evidence Reranking        |   BGE-Reranker-v2 (cross-encoder)
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 8. NLI Verification          |   DeBERTa-v3-MNLI
  +-------------+----------------+
                |
                v
  +------------------------------+
  | 9. Confidence Fusion         |   weighted multi-score fusion
  +-------------+----------------+
                |
                v
  +------------------------------+
  |10. Evidence-Guided Regen     |   Llama 3.1 (verified-only)
  +-------------+----------------+
                |
                v
  +------------------------------+
  |11. Final Verification Gate   |   PASS / REGEN / ABSTAIN
  +-------------+----------------+
                |
                v
  +------------------------------+
  |12. Explainable Response      |   JSON + Markdown + HTML
  +-------------+----------------+
                |
                v
  +------------------------------+
  |13. Evaluation Module         |   HRR, CCR, VSR, BLEU, ROUGE-L,
  +------------------------------+   BERTScore, ROC-AUC, MRR, nDCG…
```

Every step is one Python module under `src/steps/step0N_<name>.py`.

---

## How to Run (Full Guide)

The pipeline always runs on **real HuggingFace models** — there is no
mock-mode switch. Every component loads its configured HF checkpoint
and surfaces a clear error if the download fails.

### 1. Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Windows / macOS / Linux**
- For real models: NVIDIA GPU with ~16 GB VRAM, `huggingface_hub` login
  for gated checkpoints (Llama 3.1).

### 2. Clone & enter the project

```bash
git clone https://github.com/ShihabHassanNaim/Hallucination-Mitigation.git
cd Hallucination-Mitigation
```

### 3. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4. Install dependencies

The full requirements file pulls in real HuggingFace model stacks. Install it as-is:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> If a checkpoint download fails, the pipeline raises — fix the
> network/token/dependency issue and re-run.

### 5. Run the test suite

The pytest suite injects lightweight in-memory doubles for every
model slot via ``tests/_fakes.py``, so it runs fast and offline
without downloading checkpoints. Production code always uses real models.

```bash
python -m pytest tests/ -v
```

You should see something like:

```
======================== 31 passed in 42s ========================
```

If yes, the project is wired up correctly.

### 6. Run a single query (CLI)

```bash
# Linux / macOS
python -m scripts.run_inference --query "Who wrote 1984?"

# Windows PowerShell
python -m scripts.run_inference --query "Who wrote 1984?"
```

Output: a JSON record with the answer, confidence, per-claim verdicts,
retrieved evidence, decision (PASS / REGEN / ABSTAIN), and the 13 step
traces.

### 7. Run the full pipeline over a dataset

```bash
# 1. Build a retrieval index from your own JSONL (one {"text": "..."} per line)
python -m scripts.build_index --input data/my.jsonl --out data/index

# 2. Run inference on the synthetic QA set (built-in)
python -m scripts.run_inference --query "What is the capital of France?"

# 3. Aggregate metrics across the synthetic QA set
python -m scripts.evaluate --synth --out data/metrics.json

# 4. (Full) detection metrics + HRR / CCR / VSR
python -m scripts.evaluate_full --synth --out data/metrics.json

# 5. Markdown report for one query
python -m scripts.report --query "What is the capital of France?"
```

`data/metrics.json` aggregates every metric category from the diagram.

#### Evaluate on the two real benchmarks (HaluEval + TruthfulQA)

The pipeline ships with both datasets already downloaded as JSONL under
`data/halueval_qa.jsonl` and `data/truthfulqa_qa.jsonl`. The frontend's
`/api/evaluation/comparison` endpoint (and the "Cross-dataset comparison"
card on the demo page) reads from `data/metrics_<slug>.json` files
produced by these commands:

```bash
# Single dataset — base retrieval / generation metrics
python -m scripts.evaluate --dataset halueval    # writes data/metrics_halueval.json
python -m scripts.evaluate --dataset truthfulqa  # writes data/metrics_truthfulqa.json

# Single dataset — full detection + HRR/CCR/VSR
python -m scripts.evaluate_full --dataset halueval
python -m scripts.evaluate_full --dataset truthfulqa

# Or run BASE + FULL on every registered dataset in one shot
python -m scripts.evaluate_datasets                  # both, BASE then FULL
python -m scripts.evaluate_datasets --limit 200      # smoke test (first 200 rows each)
python -m scripts.evaluate_datasets --only truthfulqa --skip-base  # FULL on one set

# Override the input JSONL or output path if you want
python -m scripts.evaluate_full --dataset halueval --out runs/v2/halueval.json
```

Once both `data/metrics_halueval.json` and `data/metrics_truthfulqa.json`
exist, the Flask demo page (`/`) shows a dataset-pill selector on the
**Accuracy** and **Evaluation** panels, and a side-by-side **Cross-dataset
comparison** card with one column per dataset plus a Δ column. The
dedicated `/evaluation` page has the same controls plus per-query drill-down.

### 8. Run the REST API

```bash
# Real HuggingFace models
python -m uvicorn api.app:app --port 8000
```

Endpoints:

| Method | Path        | Purpose                              |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Liveness check                       |
| POST   | `/index`    | Add documents to the retrieval index |
| POST   | `/query`    | Run the 13-step pipeline             |
| GET    | `/metrics`  | Recent aggregate metrics             |
| GET    | `/structure`| Project tree JSON (depth/hidden tunable) |

> **Note:** The API server uses real models by default. Restart the
> server after editing configs.

**Test it (Linux / macOS — `curl`):**

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Paris is the capital of France."]}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Capital of France?"}'

curl http://localhost:8000/metrics
```

**Test it (Windows PowerShell — no `curl`):**

```powershell
Invoke-RestMethod -Uri http://localhost:8000/index -Method POST `
  -ContentType "application/json" `
  -Body '{"documents": ["Paris is the capital of France."]}'

Invoke-RestMethod -Uri http://localhost:8000/query -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "Capital of France?"}'

Invoke-RestMethod -Uri http://localhost:8000/metrics -Method GET
```

### 9. Run the Flask front-end

```bash
# Real HuggingFace models — reads HF token from .env automatically
python -m scripts.run_frontend
# open http://127.0.0.1:5000
```

The launcher auto-loads `HUGGINGFACE_HUB_TOKEN` from `.env`. The header
shows a live/degraded badge so you always know whether every component
loaded successfully. The **Accuracy** panel reads `data/metrics.json` —
run step 7 at least once to populate it.
```

The `scripts/run_frontend` launcher auto-loads `HUGGINGFACE_HUB_TOKEN`
from your `.env` and starts the Flask server. The header shows a
live/degraded badge so you always know whether every component loaded.
`Accuracy` section reads `data/metrics.json` — run `scripts/evaluate.py`
or `scripts/evaluate_full.py` once to populate it.

Two pages:

- **`/`** — docs + query + result panel (decision, confidence, per-claim verdicts, evidence, 13 step traces)
- **`/structure`** — full project tree with depth selector, hidden-file toggle, and live filter. Backed by `GET /api/structure?depth=N&include_hidden=1`.

### 10. Run with real HuggingFace models

The pipeline runs on real models by default — no flag to set. Just:

```bash
python -m scripts.run_inference --query "Who wrote 1984?"
```

If you want to swap a generator to a gated checkpoint (e.g. Llama-3.1-8B):

```bash
# 1. Login to HuggingFace (only needed for gated models)
huggingface-cli login

# 2. Edit configs/default.yaml and set `models.generator` to the gated checkpoint

# 3. Run as usual
python -m scripts.run_inference --query "Who wrote 1984?"
```

Each component loads its HF checkpoint and surfaces a clear error
if the model can't be reached (offline, gated, missing dependency).

### 11. Configuration

All knobs live in [`configs/default.yaml`](configs/default.yaml). Override anything via
environment variables prefixed `HALLUC_`, e.g.:

```bash
HALLUC_STEP06_RETRIEVAL_TOPK=10 python -m scripts.run_inference --query "..."
```

Each step has its own block (`step01_…`, `step02_…`, …, `step13_eval`)
plus a top-level `models:` block listing the HuggingFace checkpoints
the pipeline prefers by default.

### 12. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `uvicorn: command not found` | Run from inside the project root, or use `python -m uvicorn api.app:app --port 8000` |
| `curl -X ...` fails on PowerShell | Use `Invoke-WebRequest -Method POST -Uri ... -Body ...` instead |
| `ImportError: torch` | `pip install torch` (already in requirements.txt) |
| Gated model download fails | `huggingface-cli login` and accept the license on the model page |
| `data/index` is empty | Run `python -m scripts.build_index --corpus synthetic --out data/index` first |
| GPU OOM | Use a smaller model in `configs/default.yaml` |
| Tests fail after `pip install` | Run `python -m pytest tests/ -v` — the suite injects in-memory fakes |
| Front-end starts in `degraded mode` | One of the model checkpoints failed to load — check the Model Status panel |

### 13. End-to-end verification checklist

```bash
# A. Tests (use the in-memory fakes; no HF download)
python -m pytest tests/ -v

# B. Real HF models (CLI)
python -m scripts.run_inference --query "What is the capital of France?"

# C. API smoke
python -m uvicorn api.app:app --port 8000
# then in another terminal:
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" \
  -d '{"documents": ["Paris is the capital of France."]}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "Capital of France?"}'

# D. Real HF models (front-end)
python -m scripts.run_frontend
# open http://127.0.0.1:5000 — header should show "live mode"

# E. Evaluation
python -m scripts.evaluate_full --synth --out data/metrics.json
cat data/metrics.json | python -c "import json,sys; print(json.dumps(json.load(sys.stdin)['aggregate'], indent=2))"
```

**Pass criteria:**

- ✅ `pytest` all green
- ✅ `/query` returns answer, per-claim verdicts, retrieved evidence, confidence, decision (PASS / REGEN / ABSTAIN), step traces
- ✅ Step 13 emits all 10 metric categories with no NaN
- ✅ Every component loads its real HF checkpoint

---

## 🚀 Run Everything (end-to-end, copy-paste)

A single ordered flow that takes you from a fresh clone to a running web UI
with metrics populated. Works on any OS. The pipeline always uses real
HuggingFace models; the test suite uses in-memory fakes so it stays
fast and offline. Each block is a separate command — paste one at a time.

### A. One-time setup

**macOS / Linux:**
```bash
git clone https://github.com/ShihabHassanNaim/Hallucination-Mitigation.git
cd Hallucination-Mitigation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# add HUGGINGFACE_HUB_TOKEN=... to .env (the launcher reads it automatically)
```

**Windows PowerShell:**
```powershell
git clone https://github.com/ShihabHassanNaim/Hallucination-Mitigation.git
cd Hallucination-Mitigation
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# add HuggingFace = "hf_..." to .env (the launcher reads it automatically)
```

### B. Verify the install (tests)

```bash
python -m pytest tests/ -v
```
Expected: `31 passed`.

### C. Build the retrieval index

```bash
python -m scripts.build_index --corpus synthetic --out data/index
```
This produces `data/index/docs.jsonl` + `data/index/meta.json`.

### D. Run inference (single query)

```bash
python -m scripts.run_inference --query "Who wrote 1984?"
python -m scripts.run_inference --query "What is the capital of France?"
```
Each call returns a JSON record with answer, per-claim verdicts, evidence,
confidence, decision (PASS / REGEN / ABSTAIN), and 13 step traces.

### E. Run evaluation across the full synthetic set

```bash
python -m scripts.evaluate       --synth --out data/metrics.json
python -m scripts.evaluate_full  --synth --out data/metrics.json
```
`metrics.json` now contains all 10 metric categories from the diagram
(HRR, CCR, VSR, BLEU, ROUGE-L, BERTScore, ROC-AUC, MRR, nDCG, …).

### F. (Optional) Calibrate the EEDC detector

```bash
python -m scripts.calibrate_eedc --out data/calibration.json
```

### G. Generate a Markdown report for one query

```bash
python -m scripts.report --query "What is the capital of France?" --out data/report.md
```

### H. Start the REST API (FastAPI, port 8000)

```bash
python -m uvicorn api.app:app --port 8000
```
In another terminal:
```bash
curl -X POST http://localhost:8000/index  -H "Content-Type: application/json" -d '{"documents":["Paris is the capital of France."]}'
curl -X POST http://localhost:8000/query  -H "Content-Type: application/json" -d '{"query":"Capital of France?"}'
curl         http://localhost:8000/metrics
```
PowerShell equivalents use `Invoke-RestMethod` (see section 8).

### I. Start the Flask front-end (port 5000)

```bash
python -m scripts.run_frontend
# open http://127.0.0.1:5000      (chat UI)
# open http://127.0.0.1:5000/structure   (project tree)
```
The launcher reads `HUGGINGFACE_HUB_TOKEN` from `.env` and starts the
Flask server with real models. The header shows a live/degraded badge
so you always know whether every component loaded. The **Accuracy**
panel reads `data/metrics.json` — run step E at least once to populate it.

### J. (Optional) Swap to a gated checkpoint (GPU + HF login)

```bash
huggingface-cli login          # only needed for gated Llama-class models
# edit configs/default.yaml and set models.generator to the gated checkpoint
python -m scripts.run_inference --query "Who wrote 1984?"
```
Every component loads its HF checkpoint; if the download fails it
raises — fix the network/token/dependency issue and re-run.

### K. One-liner summary

| Step | Command |
|------|---------|
| tests | `python -m pytest tests/ -v` |
| build index | `python -m scripts.build_index --corpus synthetic --out data/index` |
| query | `python -m scripts.run_inference --query "…"` |
| evaluate | `python -m scripts.evaluate_full --synth --out data/metrics.json` |
| evaluate both datasets | `python -m scripts.evaluate_datasets` |
| report | `python -m scripts.report --query "…"` |
| API | `python -m uvicorn api.app:app --port 8000` |
| UI | `python -m frontend.app` |
| calibrate | `python -m scripts.calibrate_eedc --out data/calibration.json` |

---

## Quick start

```bash
python -m pytest tests/ -v                    # tests use in-memory fakes (fast + offline)
python -m scripts.run_inference --query "What is the capital of France?"
python -m scripts.run_frontend                # web UI on http://127.0.0.1:5000
```

Every component loads its HF checkpoint and surfaces a clear error
if the model can't be reached (offline, gated, missing dependency).

## REST API

```bash
uvicorn api.app:app --port 8000

curl -X POST localhost:8000/index -d '{"documents": ["Paris is the capital of France."]}'
curl -X POST localhost:8000/query -d '{"query": "Capital of France?"}'
curl        localhost:8000/metrics
```

Endpoints: `GET /health`, `POST /index`, `POST /query`, `GET /metrics`.

## Flask front-end

```bash
python -m scripts.run_frontend   # reads HF token from .env
# open http://127.0.0.1:5000
```

Two pages:

- **`/`** — docs + query + result panel (decision, confidence, per-claim verdicts, evidence, 13 step traces, dataset-aware Accuracy + Evaluation panels, side-by-side dataset comparison card, "Why this paper wins" panel)
- **`/structure`** — full project tree with depth selector, hidden-file toggle, and live filter. Backed by `GET /api/structure?depth=N&include_hidden=1`.
- **`/evaluation`** — dedicated evaluation dashboard with dataset selector + per-query drill-down.
- **`/literature`** — side-by-side comparison against 11 published hallucination-detection systems (Dale 2023, DUTh, TU Wien, SHROOM, IRIT-BL, AILS-NTUA, INSIDE, LLM-Check, AutoHall, Decoding Memories, Conformal-RAG). Verdict matrix (✓/≈/✗) + headline bar chart + full table + defence talking points. Backed by `/api/literature/comparison`, `/api/literature/why-we-win`, `/api/literature/headline` — all of which parse `Paper/comparison_table.csv` + `Paper/why_we_win.md`.

## Evaluation

```bash
python -m scripts.evaluate --synth --out data/metrics.json
python -m scripts.evaluate_full --synth --out data/metrics.json
```

`scripts/evaluate.py` aggregates per-query metrics over the full diagram.
`scripts/evaluate_full.py` adds detection P/R/F1, ROC-AUC, HRR, CCR, VSR.

---

## Configuration

Single source of truth: `configs/default.yaml`. Override anything via
env vars prefixed `HALLUC_`, e.g. `HALLUC_STEP06_RETRIEVAL_TOPK=10`.

Each step has its own block (`step01_…`, `step02_…`, …, `step13_eval`)
plus a top-level `models:` block listing the HuggingFace checkpoints
the pipeline prefers by default.

---

## Project layout

```
Hallucination/
├── api/app.py                # FastAPI: /health /index /query /metrics /structure
├── configs/default.yaml      # all knobs (one block per step + models:)
├── data/                     # generated metrics.json, optional saved index
├── frontend/                 # Flask web UI (port 5000)
│   ├── app.py                #   factory + module-level app
│   ├── templates/            #   index.html, structure.html
│   └── static/               #   css/style.css, js/app.js + js/structure.js
├── scripts/                  # CLI entry points
│   ├── build_index.py
│   ├── run_inference.py
│   ├── evaluate.py
│   ├── evaluate_full.py
│   ├── calibrate_eedc.py
│   └── report.py
├── src/
│   ├── config.py             # YAML loader + env overrides + dotted-key access
│   ├── pipeline.py           # Pipeline dataclass + RAGResult alias
│   ├── types.py              # shared dataclasses (Evidence, Claim, …)
│   ├── steps/                # 13 modules, 1 per diagram step
│   ├── models/               # HF wrappers + deterministic fallback
│   ├── retrieval/            # FAISS, BM25, hybrid RRF
│   └── utils/                # chunker, claim_extractor, data_loader
├── tests/                    # 31 tests, in-memory fakes — fast + offline
├── requirements.txt
└── README.md
```

---

## Models (HuggingFace) used

| Step | Component | Checkpoint |
|------|-----------|------------|
| 1 | GLiNER zero-shot NER | `knowledgator/gliner-base-v1.0` |
| 1 | spaCy NER | `en_core_web_sm` |
| 1 | Query rewriter | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| 2 | LLM generator | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| 2 | Alt LLM generator | `Qwen/Qwen3-8B` |
| 3 | Dense embedder | `BAAI/bge-m3` |
| 3 | Hallucination classifier | XGBoost (sklearn-logreg fallback) |
| 5 | Claim splitter | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| 6 | Dense retrieval | `BAAI/bge-m3` |
| 6 | Sparse retrieval | `rank_bm25` (pure-Python) |
| 7 | Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` |
| 8 | NLI verifier | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| 9 | EEDC calibrator | Platt + temperature scaling |
| 10 | Regenerator | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| 13 | Generation metrics | `bert-score`, `rouge-score`, `nltk` |

Every model wrapper loads its HF checkpoint; if the download fails
the wrapper raises — production code never falls back to a stub.

---

## Endpoints summary

### FastAPI (port 8000 — `api/app.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + index state |
| POST | `/index` | Rebuild index from `{"documents": [...]}` |
| POST | `/query` | Run the 13-step pipeline on `{"query": "..."}` |
| GET | `/metrics` | Contents of `data/metrics.json` (Step 13 output) |
| GET | `/structure?depth=N&include_hidden=1` | Project tree (JSON) |

### Flask (port 5000 — `frontend/app.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | HTML demo UI |
| GET | `/structure` | HTML project tree viewer |
| GET | `/api/health` `/api/index` `/api/query` `/api/metrics` `/api/structure` | JSON mirror of the FastAPI endpoints |

Flask and FastAPI are independent processes — they share only the
underlying pipeline module. Run one, the other, or both at once.

---

## License & acknowledgements

Built for a Master's thesis. Model attributions go to the original
HuggingFace model authors (Meta, BAAI, Microsoft, Qwen).
