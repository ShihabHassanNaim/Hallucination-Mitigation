# CRISP — Catching Hallucinated Answers in RAG Systems

> A research framework that adds fact-checking, evidence retrieval, and
> claim-level verification on top of any Retrieval-Augmented Generation
> (RAG) system, so you can trust the answers your LLM produces.

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![tests](https://img.shields.io/badge/tests-224%20passing-brightgreen)]()
[![mode](https://img.shields.io/badge/mock%20mode-runnable%20on%20laptop-orange)]()

---

## Table of contents

1. [What is this project? (for absolute beginners)](#what-is-this-project-for-absolute-beginners)
2. [What problem does it solve?](#what-problem-does-it-solve)
3. [How does it work? (the 11-step workflow)](#how-does-it-work-the-11-step-workflow)
4. [What is in this repository?](#what-is-in-this-repository)
5. [Quick start — try it in 30 seconds (mock mode)](#quick-start--try-it-in-30-seconds-mock-mode)
6. [Quick start — try it with real AI models](#quick-start--try-it-with-real-ai-models)
7. [The 8 phases, explained simply](#the-8-phases-explained-simply)
8. [Configuration reference](#configuration-reference)
9. [REST API reference](#rest-api-reference)
10. [Running the test suite](#running-the-test-suite)
11. [Project layout](#project-layout)
12. [Glossary of terms](#glossary-of-terms)
13. [License & acknowledgements](#license--acknowledgements)

---

## What is this project? (for absolute beginners)

**CRISP** is a research project that helps AI chatbots stop making things up.

When you ask an AI a question, it sometimes invents facts — like saying a
person wrote a book they never wrote, or that a city is the capital of a
country it isn't. We call this problem **hallucination**.

CRISP is built on top of a standard AI technique called **RAG**
(Retrieval-Augmented Generation). RAG works like an open-book exam: the
AI is given a small set of real documents first, and it must answer using
*only* what those documents say. That's safer than letting the AI answer
from memory.

But RAG still hallucinates — the AI can ignore the documents, misread them,
or combine facts from multiple documents incorrectly. That's where CRISP
comes in. It adds **four safety nets** on top of RAG:

1. **Split the answer into small factual claims.** ("Paris is the capital of
   France" is one claim. "It has 2 million people" is another.)
2. **Check each claim against the documents** using a fact-verification model.
3. **Fetch more documents** when a single claim needs evidence from multiple
   sources.
4. **Rewrite or regenerate** the answer when too many claims are wrong.

When CRISP finishes, every answer comes back with a per-claim report —
which claims are supported, which are not, and the exact evidence the
verdict is based on.

### Who is this for?

* **Students** learning about RAG, hallucination detection, and AI
  fact-checking.
* **Researchers** exploring NLI (Natural Language Inference), knowledge
  graphs, and adaptive retrieval.
* **Engineers** building trustworthy AI assistants who want a working
  reference implementation to start from.
* **Beginners** who want to see a complete, runnable, well-tested example
  of a hallucination-mitigation pipeline.

No GPU is required. The whole thing runs on a laptop.

---

## What problem does it solve?

There are three common ways an AI hallucinates:

| Cause                                    | Example                                              | What CRISP does               |
| ---------------------------------------- | ---------------------------------------------------- | ----------------------------- |
| **Outdated memory**                      | "The current president of X is Y" (Y retired last year) | Re-retrieve fresher evidence |
| **Wrong passage retrieved**              | Top document isn't actually about the entity asked   | Expand search or rewrite query |
| **Multi-fact claim from a single source** | "Orwell was British AND wrote 1984"                 | Multi-hop verification       |

If none of the evidence supports a claim, Phase 7 edits the span in the
answer or regenerates the whole response.

### Why bother?

In production, AI hallucinations cause legal liability, medical misdiagnosis,
financial loss, and broken user trust. A pipeline that flags every
unsupported claim — and proves *why* it was flagged — is the difference
between "AI suggestions" and "AI recommendations you can audit".

---

## How does it work? (the 11-step workflow)

This is the end-to-end flow CRISP runs for every question. Each box maps
to a file in `src/` and a phase described later in this README.

```
                              User Question
                                   |
                                   v
                  +-----------------------------------+
                  | 1. Query Understanding            |   <-- pipeline.run(query)
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | 2. Large Language Model (LLM)     |   <-- src/generator.py
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | 3. Initial Generated Response     |   <-- RAGResult.answer (Phase 1)
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | 4. Hallucination Detection        |   <-- src/detector.py  (NLI)
                  +-----------------+-----------------+
                                    |
                          Hallucination Confidence Score (EEDC, Phase 2)
                                    |
                +-------------------+-------------------+
                |                                       |
          High Confidence                       Low Confidence
                |                                       |
                v                                       v
       Return Answer                       +-----------------------------------+
                                            | 5. Claim Extraction               |  <-- src/claim_extractor.py
                                            +-----------------+-----------------+
                                                              |
                                                              v
                                            +-----------------------------------+
                                            | 6. Evidence Retrieval (RAG)       |  <-- src/{retriever,bm25,
                                            +-----------------+-----------------+      hybrid_retriever,
                                                              |                       adaptive_retriever}.py
                                                              v
                                            +-----------------------------------+
                                            | 7. Claim-Level Fact Verification  |  <-- src/detector.py
                                            +-----------------+-----------------+
                                                              |
                                                              v
                                            +-----------------------------------+
                                            | 8. Confidence Fusion Module       |  <-- src/eedc.py +
                                            +-----------------+-----------------+      src/calibration.py
                                                              |                       (Phase 6)
                                                              v
                                            +-----------------------------------+
                                            | 9. Evidence-Guided Regeneration   |  <-- src/editor.py
                                            +-----------------+-----------------+
                                                              |
                                                              v
                                            +-----------------------------------+
                                            | 10. Final Hallucination Detection |  <-- re-score inside
                                            +-----------------+-----------------+      AIC loop
                                                              |
                              +-------------------------------+-------------------------------+
                              |                                                               |
                      Verified Answer                                            Regenerate Again (max 3)
                              |                                                               |
                              v                                                               |
                  +-----------------------------------+                                     |
                  | 11. Final Verified Response        |  <-- RAGResult (Phase 8 report)   |
                  +-----------------+-----------------+                                     |
                                    |                                                       |
                                    +-------------------------------------------------------+
```

The first time an answer reaches step 11 it is **verified** and returned.
If it is still not reliable, step 9 regenerates it (capped by
`CRISP_AIC_MAX_ITER`, default `3`) and re-runs steps 4–10.

### Mapping every step to a source file

| # | Workflow step                  | Source file                                                       | Phase |
| :-: | ----------------------------- | ----------------------------------------------------------------- | :---: |
| 1 | Query Understanding           | `src/pipeline.py` → `Pipeline.run`                                |   —   |
| 2 | Large Language Model          | `src/generator.py`                                                |   1   |
| 3 | Initial Generated Response    | `RAGResult.answer`                                                |   1   |
| 4 | Hallucination Detection       | `src/detector.py`                                                 |   2   |
| — | EEDC Confidence Score         | `src/eedc.py`                                                     |   2   |
| 5 | Claim Extraction              | `src/claim_extractor.py`                                          |   3   |
| 6 | Evidence Retrieval (RAG)      | `src/retriever.py`, `src/bm25.py`, `src/hybrid_retriever.py`, `src/adaptive_retriever.py` | 4 |
| 7 | Claim-Level Fact Verification | `src/detector.py`                                                 |   2   |
| 8 | Confidence Fusion Module      | `src/eedc.py`, `src/calibration.py`                               |   6   |
| 9 | Evidence-Guided Regeneration  | `src/editor.py`                                                   |   7   |
| 10 | Final Hallucination Detection | `pipeline._detect_and_score`                                      |   7   |
| 11 | Final Verified Response       | `RAGResult`, `src/reporting.py`                                   |   8   |

---

## What is in this repository?

```
Hallucination/
+-- api/                              # FastAPI web service (Phase 5+)
|   +-- app.py                        # 3 endpoints: /health, /query, /index
+-- configs/
|   +-- default.yaml                  # all tunables (model, k, device, mock flag)
+-- src/                              # The whole pipeline lives here.
|   +-- config.py                     # pydantic settings, env-var overrides
|   +-- embeddings.py                 # BGE wrapper (or deterministic mock)
|   +-- retriever.py                  # FAISS index, save/load
|   +-- generator.py                  # HF transformers wrapper (fp16/4bit/mock)
|   +-- data_loader.py                # HaluEval loader + synthetic fallback
|   +-- pipeline.py                   # end-to-end RAG with hooks for Phases 2-8
|   +-- evaluation.py                 # hallucination proxies + coverage metrics
|   +-- detector.py                   # Phase 2: NLI hallucination detector
|   +-- eedc.py                       # Phase 2: EEDC confidence scorer
|   +-- claim_extractor.py            # Phase 3: atomic claims + provenance tags
|   +-- bm25.py                       # Phase 4: BM25 sparse retriever
|   +-- hybrid_retriever.py           # Phase 4: dense + BM25 (RRF)
|   +-- adaptive_retriever.py         # Phase 4: expand / rewrite / multi-hop controller
|   +-- calibration.py                # Phase 6: post-hoc confidence calibration
|   +-- editor.py                     # Phase 7: evidence-guided span editor
|   +-- iteration_controller.py       # Phase 7: Adaptive Iteration Controller (AIC)
|   +-- ner.py                        # Phase 5: NER tagger (mock or spaCy)
|   +-- kg_linker.py                  # Phase 5: mini knowledge-graph linker
|   +-- multi_hop.py                  # Phase 5: 2-hop retrieval planner
|   +-- reporting.py                  # Phase 8: reliability verdict + renderers
+-- scripts/                          # CLI entry points.
|   +-- build_index.py                # embed a corpus -> FAISS index on disk
|   +-- run_inference.py              # run RAG on a dataset -> JSONL predictions
|   +-- evaluate.py                   # score predictions -> JSON metrics
|   +-- calibrate_eedc.py             # fit EEDC weights + post-hoc calibrator
|   +-- report.py                     # Phase 8: render reports from preds.jsonl
+-- tests/                            # Pytest suite (224 tests, mock-friendly).
|   +-- test_pipeline.py              # Phase 1 smoke (no downloads)
|   +-- test_phase2.py                # Phase 2 NLI + EEDC
|   +-- test_phase3.py                # Phase 3 claim extraction + provenance
|   +-- test_phase4.py                # Phase 4 adaptive retrieval
|   +-- test_phase5.py                # Phase 5 multi-hop planner + integration
|   +-- test_phase6.py                # Phase 6 calibration
|   +-- test_phase7.py                # Phase 7 editor + AIC
|   +-- test_phase8.py                # Phase 8 reporting
|   +-- test_api.py                   # Phase 5+ FastAPI service
+-- data/                             # datasets and indices land here (gitignored)
+-- README.md                         # you are here
+-- requirements.txt                  # python deps
```

---

## Quick start — try it in 30 seconds (mock mode)

Mock mode means **no downloads, no GPU**. Everything uses deterministic
stubs that always behave the same way. All 224 tests pass in under a
second on a laptop.

### 1. Install (one-time setup)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **macOS / Linux?** Replace `.\.venv\Scripts\Activate.ps1` with
> `source .venv/bin/activate`. Replace `.\.venv\Scripts\python.exe` with
> `./.venv/bin/python`.

### 2. Verify the install

```powershell
$env:CRISP_MOCK = "1"
python -m pytest tests/ -v
```

You should see:

```
======================== 224 passed in 0.75s ========================
```

If yes, the project is wired up correctly. If no, copy the error message
into an issue — most failures are dependency-related.

### 3. Ask CRISP a question (one-liner demo)

```powershell
$env:CRISP_MOCK = "1"
python -c "from src.config import load_config; from src.pipeline import Pipeline; pipe = Pipeline(load_config()); pipe.build_index(['France is a country in Western Europe. Its capital is Paris.', 'The dystopian novel 1984 was written by George Orwell.']); print(pipe.run('What did George Orwell write?').answer)"
```

That single line:
1. Loads the config (`mock: true`)
2. Creates a `Pipeline`
3. Builds a tiny FAISS index in memory
4. Asks the question
5. Prints the verified answer

You can swap the documents for whatever you want. The Pipeline is a
regular Python object — read its source to learn how to extend it.

---

## Quick start — try it with real AI models

For real models you need a GPU and ~16 GB of VRAM. Uncomment the heavy
dependencies in `requirements.txt` (`torch`, `transformers`, `bitsandbytes`),
then flip `mock: false` in `configs/default.yaml`.

### 1. Build a retrieval index

```powershell
python scripts/build_index.py --corpus synthetic --out data/index
```

This embeds every document in the corpus and writes a FAISS index to
`data/index/`.

### 2. Run CRISP on a dataset

```powershell
python scripts/run_inference.py --dataset synthetic --index data/index --out data/preds.jsonl
```

Predictions are written as JSON-Lines. One line per question, with the
answer, the retrieved documents, and every per-claim verdict.

### 3. Compute hallucination + coverage metrics

```powershell
python scripts/evaluate.py --preds data/preds.jsonl --out data/metrics.json
```

`data/metrics.json` has aggregate statistics — hallucination rate,
precision/recall against the reference answer, retrieval recall@k.

### 4. Generate the reliability report

```powershell
python scripts/report.py --in data/preds.jsonl --out-dir reports/ --format html
```

Open `reports/*.html` in your browser. Each report has verdict badges
and per-claim tables.

### Try it on your own dataset

Any JSON Lines file shaped like this works as input:

```json
{"id": "q1", "question": "What did George Orwell write?", "reference_answer": "1984 and Animal Farm."}
{"id": "q2", "question": "What is the capital of France?", "reference_answer": "Paris."}
```

```powershell
python scripts/run_inference.py --dataset halueval --data data/halueval.json --index data/index --out data/preds.jsonl
```

---

## The 8 phases, explained simply

CRISP is built in **8 phases**. Each phase adds one capability. Every
phase sits behind a stable interface so later phases plug in without
breaking the earlier ones.

| Phase | One-line description                                                                  |
| :---: | ------------------------------------------------------------------------------------- |
| **1** | Retrieve top-k passages, prompt the LLM, return the answer.                          |
| **2** | Check every atomic claim against its evidence with an NLI model and score confidence. |
| **3** | Split the answer into atomic claims and tag where each one needs verification.        |
| **4** | If the top hit isn't good enough, expand k, rewrite the query, or fall back to BM25.   |
| **5** | For claims combining multiple passages, run NER → KG link → 2-hop sub-query.          |
| **6** | Re-calibrate raw confidence so 0.7 actually means 70% supported, not 95%.              |
| **7** | Decide whether to accept, edit a span, regenerate the whole answer, or stop.          |
| **8** | Roll every signal into one reliability verdict and render it.                         |

### Phase 1 — Baseline RAG (the starting point)

The starting point: **BGE embeddings → FAISS top-k → HF generator →
answer**. Faithful, well-instrumented, runs in MOCK mode without
downloading models.

* Source: `src/embeddings.py`, `src/retriever.py`, `src/generator.py`
* Output: `RAGResult.answer`

**Plain-English summary:** you ask "What is the capital of France?" — the
system embeds that question, finds the top 5 most similar documents, and
feeds them plus the question to the LLM. The LLM answers.

### Phase 2 — Per-claim NLI detection + EEDC scoring

Each claim is paired with the retrieved passage and run through a
DeBERTa-v3 NLI model. The **EEDC scorer** (Evidence-Entailment
Decomposition Confidence) turns the raw NLI probabilities, retrieval
scores, and self-consistency into a single 0–1 calibrated confidence.

* Source: `src/detector.py`, `src/eedc.py`
* Output: `ClaimVerdict.eedc_score`
* Refit weights: `python scripts/calibrate_eedc.py`

**Plain-English summary:** for every claim the LLM made, ask "is the
passage we retrieved actually evidence for this claim?" and produce a
single confidence number between 0 and 1.

### Phase 3 — Atomic claim extraction + provenance tagging

Phase 2 scores individual claims, so claim quality bounds the whole
system. Phase 3 adds `ClaimExtractor` with a `Provenance` enum:

| Provenance    | Meaning                                                              |
| :-----------: | -------------------------------------------------------------------- |
| `INTRINSIC`   | Answerable from the question alone. Skip NLI — saves cost.          |
| `EXTRINSIC`   | Needs an external passage. The default for most factual RAG claims. |
| `AGGREGATED`  | Combines 2+ passages. Triggers Phase 5 multi-hop retrieval.         |

Switch between the deterministic `mode="synthetic"` (default,
mock-friendly) and `mode="real"` (T5/PEGASUS atomiser + embedder-based
provenance).

**Plain-English summary:** instead of scoring the whole answer as one
unit, split it into tiny facts ("Paris is the capital", "It has 2M
people") and check each tiny fact separately.

### Phase 4 — Adaptive evidence retrieval

A controller inspects the top-1 retrieval score and:

* **expands `k`** when the top hit is weak or top-1/2 are too close (the
  retriever is unsure),
* **rewrites the query** with deterministic variants when even expansion
  doesn't help.

A `HybridRetriever` (dense + BM25 fused via Reciprocal Rank Fusion)
provides the underlying recall. Enable with `CRISP_ADAPTIVE=1`.

**Plain-English summary:** if the first document returned isn't a good
match, try harder — fetch more documents, or rephrase the question and
search again.

### Phase 5 — Multi-hop evidence aggregation

When Phase 3 tags a claim as `AGGREGATED` (it draws from multiple
passages), single-hop retrieval isn't enough. Phase 5 adds a small
planner that:

1. **Extracts entities** from the claim (`NER` module — mock by default,
   swap to spaCy via `CRISP_NER_BACKEND=spacy`).
2. **Resolves them in a mini knowledge graph** (`KGLinker`, with 13
   seeded entities and ~30 surface-form aliases — George Orwell, 1984,
   France, Paris, Germany, Berlin, …).
3. **Plans 2-hop sub-queries** by following priority relations (e.g.
   `PER → country_of_citizenship`, `LOC → capital`).
4. **Merges the new evidence** into the existing `RAGResult` and surfaces
   a `MultiHopTrace` (entities, hops, sub-queries, evidence, notes) that
   downstream phases can replay.

Disable with `CRISP_DISABLE_MULTIHOP=1`.

**Plain-English summary:** "Who wrote 1984?" needs only one document, but
"What country was the author of 1984 a citizen of?" needs **two** — one
to name the author, one to look up the country. Multi-hop means chaining
those lookups together.

### Phase 6 — Post-hoc confidence calibration

Raw Platt weights from Phase 2 give a 0–1 score, but the absolute value
is poorly calibrated (sigmoid outputs are over-confident when the input
is out-of-distribution). Phase 6 adds a post-hoc calibrator on top:

| Calibrator            | How it works                                                                              |
| :-------------------: | ----------------------------------------------------------------------------------------- |
| `TemperatureScaler`   | Single-parameter `T` that softens (`T > 1`) or sharpens (`T < 1`) the Platt score.       |
| `IsotonicCalibrator`  | Non-parametric Pool-Adjacent-Violators (PAV) on `(raw_score, true_label)` pairs.          |
| `CalibratedEEDC`      | Wraps an `EEDCScorer` + a calibrator behind the same `.score(...)` API.                   |

Metrics live alongside: `expected_calibration_error` (10-bin ECE),
`brier_score`, `log_loss`, `accuracy_at`. Pick a method via
`CRISP_CALIBRATION` env var (`temperature` / `isotonic` / `none`) or
`calibration.method` in `configs/default.yaml`.

**Plain-English summary:** when the model says "I'm 90% sure", is it
actually right 90% of the time? Calibration is the fix. Phase 6 adjusts
the numbers so the confidences mean what they say.

### Phase 7 — Adaptive iteration + evidence-guided editor

Instead of one-shot RAG, Phase 7 closes the loop. After Phase 2 flags
claims, the **Adaptive Iteration Controller (AIC)** picks one of four
actions:

| Action   | When                                                                                  |
| :------: | ------------------------------------------------------------------------------------- |
| `ACCEPT` | Hallucination rate ≤ `accept_rate_threshold` (default `0.05`).                       |
| `EDIT`   | A few spans are wrong; `EvidenceGuidedEditor` patches them.                           |
| `REGEN`  | Too many flags; regenerate the answer with extra evidence.                            |
| `STOP`   | Max iterations reached or rate plateaued (no improvement ≥ `min_improvement`).        |

The editor locates each flagged claim's span in the answer (exact
substring → 6-token prefix fallback) and rewrites only that span in one
of three modes:

| Mode         | Behaviour                                                    |
| :----------: | ------------------------------------------------------------ |
| `stub`       | Replace span with `[unsupported: <claim>]` (mock-friendly).  |
| `evidence`   | Replace with best-matching evidence sentence (token overlap). |
| `regenerate` | Call the `Generator` for a faithful rewrite.                 |

Toggle the whole loop with `CRISP_AIC=1` and tweak thresholds with the
`CRISP_AIC_*` env vars. Each iteration is recorded in
`RAGResult.iteration_history`.

**Plain-English summary:** if many claims are wrong, instead of giving up,
edit just the wrong sentences or regenerate the answer (and check again).

### Phase 8 — Reliability reports

The last phase stitches everything together. `ReportBuilder` consumes a
`RAGResult` and produces a `ReliabilityReport` with one of five labels:

| Label             | Hallucination rate | Mean EEDC |
| :---------------: | :----------------: | :-------: |
| `RELIABLE`        | ≤ 5 %              | ≥ 0.75    |
| `MOSTLY_RELIABLE` | ≤ 20 %             | ≥ 0.55    |
| `UNCERTAIN`       | ≤ 50 %             | any       |
| `UNRELIABLE`      | > 50 %             | any       |
| `UNVERIFIABLE`    | no claims          | —         |

Each report renders to **JSON**, **Markdown**, and styled **HTML** (with
verdict badges and per-claim tables). Batch stats land in `summary.json`.
The CLI is:

```powershell
python scripts/report.py --in data/preds.jsonl --out-dir reports/ --format html
```

**Plain-English summary:** roll every signal into one of five reliability
labels and pretty-print it as a PDF/HTML report.

---

## Configuration reference

All tunables live in `configs/default.yaml`. Every field can be
overridden from the environment by uppercasing the path and prefixing
`CRISP_`:

| YAML path                                  | Env var                       |
| ------------------------------------------ | ----------------------------- |
| `mock`                                     | `CRISP_MOCK`                  |
| `retrieval.top_k`                          | `CRISP_TOPK`                  |
| `retrieval.index_path`                     | `CRISP_RETRIEVAL_INDEX_PATH`  |
| `retrieval.mode` (`dense` / `hybrid`)      | `CRISP_RETRIEVAL_MODE`        |
| `adaptive_retrieval.enabled`               | `CRISP_ADAPTIVE`              |
| `multi_hop.enabled`                        | `CRISP_DISABLE_MULTIHOP` (inverted) |
| `multi_hop.ner_backend`                    | `CRISP_NER_BACKEND`           |
| `multi_hop.ner_model`                      | `CRISP_NER_MODEL`             |
| `multi_hop.max_entities`                   | `CRISP_MULTIHOP_ENTS`         |
| `multi_hop.max_relations_per_entity`       | `CRISP_MULTIHOP_RELS`         |
| `multi_hop.top_k_per_subquery`             | `CRISP_MULTIHOP_TOPK`         |
| `detector.enabled`                         | `CRISP_DISABLE_DETECT` (inverted) |
| `eedc.weights_path`                        | `CRISP_EEDC_WEIGHTS_PATH`     |
| `calibration.method`                       | `CRISP_CALIBRATION`           |
| `pipeline.enable_iteration_control`        | `CRISP_AIC`                   |
| `pipeline.aic_max_iterations`              | `CRISP_AIC_MAX_ITER`          |
| `pipeline.editor_mode`                     | `CRISP_EDITOR_MODE`           |

Load a non-default config from disk with:

```powershell
python scripts/run_inference.py --config path/to/my.yaml
```

---

## REST API reference

CRISP ships with a small FastAPI service that wraps the pipeline. Three
endpoints, no extra config beyond the env vars above.

### Boot the dev server

```powershell
$env:CRISP_MOCK = "1"
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload --port 8000
```

OpenAPI / Swagger docs: <http://127.0.0.1:8000/docs>

### Endpoints

| Verb   | Path       | Body                                      | What it does                                                        |
| :----: | :--------: | :---------------------------------------- | :------------------------------------------------------------------ |
| `GET`  | `/health`  | —                                         | Liveness + index status (`loaded`, `n_documents`, `mock_mode`).     |
| `POST` | `/index`   | `{"documents": ["...", "..."]}`           | Rebuild the in-memory retrieval index from a list of documents.     |
| `POST` | `/query`   | `{"query": "What is the capital of X?"}`  | Ask CRISP a question — returns the full `RAGResult` JSON.           |

### `curl` walkthrough

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/index \
     -H "Content-Type: application/json" \
     -d '{"documents": ["France is a country in Western Europe. Its capital is Paris.", "1984 was written by George Orwell."]}'

curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What did George Orwell write?"}'
```

### Sample JSON response

```json
{
  "query": "What did George Orwell write?",
  "answer": "[1] ... [q='What']",
  "retrieved_docs": [{"text": "...", "score": 0.9, "index": 0}],
  "claim_verdicts": [
    {
      "claim_id": "c1",
      "claim_text": "George Orwell wrote 1984 and was a citizen of the United Kingdom.",
      "provenance": "Provenance.AGGREGATED",
      "eedc_score": 0.55,
      "hallucinated": false,
      "multi_hop_trace": {
        "entities": [{"id": "Q1", "name": "George Orwell", "type": "PER"}],
        "sub_queries": ["George Orwell country of citizenship United Kingdom", "..."],
        "hops": [],
        "evidence": []
      }
    }
  ],
  "multi_hop_traces": [],
  "hallucination_rate": 0.0,
  "timings_ms": {"retrieve": 0.2, "generate": 0.01, "detect": 0.3, "total": 0.9},
  "_meta": {"took_ms": 0.9, "query": "What did George Orwell write?"}
}
```

### Error codes

| Status | When                                                              |
| :----: | :---------------------------------------------------------------- |
| `409`  | Index not built yet — call `POST /index` first.                  |
| `422`  | Request validation (empty `query`, missing fields, etc.).        |
| `500`  | Pipeline error — the message is echoed back, never a stack trace. |

---

## Running the test suite

```powershell
$env:CRISP_MOCK = "1"
python -m pytest tests/ -v
```

This runs **224 tests** across all phases in ~1 second with no network or
GPU. Tests are organised by phase (`test_phaseN.py`) and follow a
class-based structure:

| File                       | What it covers                                                |
| :------------------------- | :------------------------------------------------------------ |
| `tests/test_pipeline.py`   | Phase 1 smoke tests (mock pipeline wiring).                   |
| `tests/test_phase2.py`     | NLI detector + EEDC scorer + pipeline integration.           |
| `tests/test_phase3.py`     | Claim extraction + provenance tagging + pipeline integration. |
| `tests/test_phase4.py`     | BM25, hybrid retriever, adaptive controller.                  |
| `tests/test_phase5.py`     | NER, KG linker, multi-hop planner, pipeline integration.      |
| `tests/test_phase6.py`     | Temperature / isotonic calibration + CalibratedEEDC.          |
| `tests/test_phase7.py`     | Evidence-guided editor + Adaptive Iteration Controller.       |
| `tests/test_phase8.py`     | Reliability label, renderers, batch summary, CLI.             |
| `tests/test_api.py`        | FastAPI service: health, index, query endpoints.              |

### Run a single file in isolation

```powershell
python -m pytest tests/test_phase5.py -v
```

### Adding a new test

The convention is straightforward — see any `tests/test_phaseN.py` for
examples:

```python
import os; os.environ["CRISP_MOCK"] = "1"
from src.something import SomeClass


class TestSomething:
    def test_basic(self):
        s = SomeClass()
        assert s.do() == "expected"
```

---

## Glossary of terms

| Term               | Meaning                                                                                  |
| :----------------: | :--------------------------------------------------------------------------------------- |
| **RAG**            | Retrieval-Augmented Generation. Give the LLM the right passages; it gives better answers. |
| **Hallucination**  | An LLM asserting something that isn't in the retrieved evidence.                         |
| **NLI**            | Natural Language Inference. A model that classifies a `(premise, hypothesis)` pair as `entail` / `contradict` / `neutral`. |
| **EEDC**           | Evidence-Entailment Decomposition Confidence. A weighted sum of NLI prob, retrieval score, and self-consistency → a single 0–1 confidence. |
| **Atomic claim**   | A single, indivisible factual statement. One sentence, one fact.                         |
| **Provenance**     | Where a claim's truth comes from — the question itself, one passage, or several.         |
| **AGGREGATED**     | A claim composing info from multiple passages — needs Phase 5 multi-hop verification.    |
| **Multi-hop**      | Following a chain of facts through several retrieval steps to verify one claim.          |
| **Mock mode**      | A deterministic stub mode that lets the whole pipeline run without model downloads or GPU. |
| **BM25**           | A classical sparse retrieval algorithm (TF-IDF with length normalisation).               |
| **RRF**            | Reciprocal Rank Fusion. A simple way to merge ranked lists from different retrievers.    |
| **NER**            | Named Entity Recognition. Tagging spans of text as `PER` / `ORG` / `LOC` / `DATE` / …   |
| **KG**             | Knowledge Graph. A graph of `(head, relation, tail)` triples — the mini-KG has 13 entities. |
| **AIC**            | Adaptive Iteration Controller (Phase 7). Picks `ACCEPT` / `EDIT` / `REGEN` / `STOP`.      |
| **FAISS**          | Facebook AI Similarity Search — a library for fast nearest-neighbour lookup.              |
| **Embedding**      | A vector (list of numbers) representing the meaning of a word or sentence.                |
| **Top-k**          | The `k` most similar results returned by a retriever. Usually `k = 5` to `k = 20`.        |
| **Mock**           | A test double — a function that pretends to be a real model so tests run instantly.       |
| **Pipeline**       | The end-to-end code path that turns a question into a verified answer.                   |

---

## License & acknowledgements

Research use only. Models retain their original licenses
(Llama 3.1 Community License, BGE MIT, DeBERTa Apache-2.0, …).

Built around the research proposal for the **CRISP** thesis framework.

### Further reading

* **HaluEval** — the benchmark dataset of hallucinated answers we test against.
* **SelfCheckGPT** — an alternative claim-level verification idea.
* **FACTOID** — fact-verification datasets for NLI.
* **FAISS** — vector search library used in Phase 1 / Phase 4.
