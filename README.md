# CRISP — Catching Hallucinated Answers in RAG Systems

> A research framework that adds fact-checking, evidence retrieval, and
> claim-level verification on top of any Retrieval-Augmented Generation
> (RAG) system, so you can trust the answers your LLM produces.

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![tests](https://img.shields.io/badge/tests-175%20passing-brightgreen)]() [![mode](https://img.shields.io/badge/mock%20mode-runnable%20on%20laptop-orange)]()

---

## What is CRISP, in plain English?

When you ask a Large Language Model (LLM) a question, you give it a chunk of
text (the **retrieved passage**) and ask for an answer. The model often makes
things up — a problem called **hallucination**.

**CRISP** wraps a standard RAG pipeline with four safety nets:

1. **Break the answer into atomic claims.** ("Paris is the capital of France" is
   one claim. "It has 2 million people" is another.)
2. **Score how confident we are** each claim is true, using evidence from
   retrieved passages and an NLI (Natural Language Inference) model.
3. **Fetch more evidence when one passage isn't enough.** If a claim seems to
   draw from several passages at once (a *multi-hop* claim), we run a small
   knowledge-graph lookup and pull targeted sub-questions.
4. **Rewrite or regenerate** the answer if its confidence is too low (Phase 7).

The result: every answer comes back with per-claim verdicts (`supported` /
`unsupported`) and traceable evidence — so end users can see *why* an answer
is labelled reliable.

---

## At a glance

| What you want to do                           | Where to start                                  |
| --------------------------------------------- | ------------------------------------------------ |
| **See it work in 30 seconds**                 | [Quick start — Mock mode](#quick-start-mock-mode) |
| **Understand the architecture**               | [How the system is built →](#how-the-system-is-built) |
| **Build a real index**                        | [Quick start — Real run](#quick-start-real-run)  |
| **Call CRISP from your own app**              | [REST API →](#rest-api)                          |
| **Run the test suite**                        | [Running the tests →](#running-the-tests)        |
| **Tweak config (model, top-k, thresholds)**   | [`configs/default.yaml`](#configuration)         |

---

## Quick start — Mock mode

Mock mode lets you exercise the entire system **without downloading any model
weights** and without a GPU. Everything uses deterministic stubs, so all 175
tests pass in under a second on a laptop.

```powershell
# 1. Create a virtual environment and install deps.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Verify the install with the test suite.
$env:CRISP_MOCK = "1"
python -m pytest tests/ -v
```

You should see:

```
======================== 175 passed in 0.75s ========================
```

### A 30-second demo

```powershell
$env:CRISP_MOCK = "1"
python -c "from src.config import load_config; from src.pipeline import Pipeline; pipe = Pipeline(load_config()); pipe.build_index(['France is a country in Western Europe. Its capital is Paris.', 'The dystopian novel 1984 was written by George Orwell.']); print(pipe.run('What did George Orwell write?').answer)"
```

---

## Quick start — Real run

For real models you need a GPU and ~16 GB of VRAM. Uncomment the heavy
dependencies in `requirements.txt` (`torch`, `transformers`, `bitsandbytes`),
then flip `mock: false` in `configs/default.yaml`.

```powershell
# Build a retrieval index from a corpus (writes to data/index)
python scripts/build_index.py --corpus synthetic --out data/index

# Run CRISP on the synthetic demo dataset or a HaluEval json file
python scripts/run_inference.py --dataset synthetic --index data/index --out data/preds.jsonl

# Compute hallucination + coverage metrics
python scripts/evaluate.py --preds data/preds.jsonl --out data/metrics.json
```

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

## How the system is built

CRISP is built in **8 phases**, each adding one capability. Every phase
sits behind a stable interface so later phases plug in without breaking the
earlier ones.

```
Query ─┐
       ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Phase 1 — Baseline RAG                                         │
   │  embed query → retrieve top-k passages → prompt LLM → answer    │
   │  (BGE embeddings + FAISS + any HF causal LM)                    │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Phase 3 — Split answer into atomic claims                      │
   │  tag each with provenance:                                       │
   │    INTRINSIC   answerable from the question alone                │
   │    EXTRINSIC   needs one retrieved passage                       │
   │    AGGREGATED  combines 2+ passages → triggers Phase 5           │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Phase 2 — Per-claim NLI detection + EEDC confidence scoring    │
   │  did the retrieved passage actually support this claim?         │
   │  outputs: label (entail / contradict / neutral) + a 0–1 score    │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Phase 4 — Adaptive evidence retrieval                          │
   │  if the top hit isn't good enough, expand k or rewrite the query │
   │  falls back to BM25 hybrid (dense + sparse) for higher recall   │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Phase 5 — Multi-hop evidence aggregation (KG + sub-queries)    │
   │  for AGGREGATED claims: NER → knowledge-graph link → 2-hop       │
   │  expand → merge new evidence into the verdict                   │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Phase 6 — Confidence calibration (placeholder, scheduled)      │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Phase 7 — Evidence-guided regeneration (placeholder, scheduled)│
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                RAGResult (answer + per-claim verdicts + traces)
```

| Phase | What it adds                                                                 | Status |
| :---: | ----------------------------------------------------------------------------- | :----: |
|   1   | Vanilla RAG baseline (BGE + FAISS + HF generator), mock mode for laptop dev  |   ✅   |
|   2   | Per-claim NLI detection + EEDC confidence scoring with calibration           |   ✅   |
|   3   | Atomic claim extraction + provenance tagging                                 |   ✅   |
|   4   | Adaptive evidence retrieval (BM25 + dense hybrid + controller)              |   ✅   |
|   5   | Multi-hop evidence aggregation (NER + mini-KG + sub-queries)                 |   ✅   |
|   6   | Post-hoc confidence calibration (temperature / isotonic) + ECE / Brier        |   ✅   |
|   7   | Adaptive Iteration Controller + evidence-guided editor (span rewriting)       |   ✅   |
|   8   | End-to-end reliability verdict + JSON / HTML / Markdown reports               |   ✅   |

---

## Repository layout

```
Hallucination/
├── api/                              # FastAPI web service (Phase 5+)
│   └── app.py                        # 3 endpoints: /health, /query, /index
├── configs/
│   └── default.yaml                  # all tunables (model, k, device, mock flag)
├── src/
│   ├── config.py                     # pydantic settings, env-var overrides
│   ├── embeddings.py                 # BGE wrapper (or deterministic mock)
│   ├── retriever.py                  # FAISS index, save/load
│   ├── generator.py                  # HF transformers wrapper (fp16/4bit/mock)
│   ├── data_loader.py                # HaluEval loader + synthetic fallback
│   ├── pipeline.py                   # end-to-end RAG with hooks for Phases 2–8
│   ├── evaluation.py                 # hallucination proxies + coverage metrics
│   ├── detector.py                   # Phase 2: NLI hallucination detector
│   ├── eedc.py                       # Phase 2: EEDC confidence scorer
│   ├── claim_extractor.py            # Phase 3: atomic claims + provenance tags
│   ├── bm25.py                       # Phase 4: BM25 sparse retriever
│   ├── hybrid_retriever.py           # Phase 4: dense + BM25 (RRF)
│   ├── adaptive_retriever.py         # Phase 4: expand / rewrite / multi-hop controller
│   ├── ner.py                        # Phase 5: NER tagger (mock or spaCy)
│   ├── kg_linker.py                  # Phase 5: mini knowledge-graph linker
│   └── multi_hop.py                  # Phase 5: 2-hop retrieval planner
├── scripts/
│   ├── build_index.py                # embed a corpus → FAISS index on disk
│   ├── run_inference.py              # run RAG on a dataset → JSONL predictions
│   ├── evaluate.py                   # score predictions → JSON metrics
│   └── calibrate_eedc.py             # fit EEDC weights on labelled dev set
├── tests/
│   ├── test_pipeline.py              # Phase 1 smoke (no downloads)
│   ├── test_phase2.py                # Phase 2 NLI + EEDC
│   ├── test_phase3.py                # Phase 3 claim extraction + provenance
│   ├── test_phase4.py                # Phase 4 adaptive retrieval
│   ├── test_phase5.py                # Phase 5 multi-hop planner + integration
│   └── test_api.py                   # Phase 5+ FastAPI service
└── data/                             # datasets and indices land here (gitignored)
```

---

## Configuration

All tunables live in `configs/default.yaml`. Every field can be overridden
from the environment by uppercasing the path and prefixing `CRISP_`:

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

Load a non-default config from disk with
`python scripts/run_inference.py --config path/to/my.yaml`.

---

## REST API

CRISP ships with a small FastAPI service that wraps the pipeline. Three
endpoints, no extra config beyond the env vars above.

```powershell
# Boot the dev server (auto-reload on edits)
$env:CRISP_MOCK = "1"
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload --port 8000
```

OpenAPI/Swagger docs: <http://127.0.0.1:8000/docs>

| Verb   | Path       | Body                                  | What it does                                                              |
| ------ | ---------- | ------------------------------------- | -------------------------------------------------------------------------- |
| `GET`  | `/health`  | —                                     | Liveness + index status (loaded, n_documents, mock_mode)                   |
| `POST` | `/index`   | `{"documents": ["...", "..."]}`       | Rebuild the in-memory retrieval index from a list of documents             |
| `POST` | `/query`   | `{"query": "What is the capital of X?"}` | Ask CRISP a question — returns the full `RAGResult` JSON                |

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

The JSON response looks like:

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
        "hops": [...],
        "evidence": [...]
      }
    }
  ],
  "multi_hop_traces": [...],
  "hallucination_rate": 0.0,
  "timings_ms": {"retrieve": 0.2, "generate": 0.01, "detect": 0.3, "total": 0.9},
  "_meta": {"took_ms": 0.9, "query": "What did George Orwell write?"}
}
```

Errors:
- `409` — index not built yet (call `POST /index` first)
- `422` — request validation (empty `query`, missing fields, etc.)
- `500` — pipeline error (the message is echoed back, never a stack trace)

---

## Running the tests

```powershell
$env:CRISP_MOCK = "1"
python -m pytest tests/ -v
```

This runs **175 tests** across all phases in ~1 second with no network or
GPU. Tests are organised by phase (`test_phaseN.py`) and follow a
class-based structure:

```
tests/test_api.py        TestHealth / TestIndex / TestQuery / TestEnvIndexPath
tests/test_pipeline.py   Phase 1 smoke tests (mock pipeline wiring)
tests/test_phase2.py     TestNLIDetector / TestEEDCScorer / TestPipelineDetection
tests/test_phase3.py     TestProvenance / TestClaimExtractor / TestPipelineProvenance
tests/test_phase4.py     TestBM25 / TestHybridRetriever / TestAdaptiveRetriever
tests/test_phase5.py     TestNER / TestKGLinker / TestMultiHopPlanner / TestMultiHopConfig / TestPipelineMultiHop
```

Run a single file in isolation:

```powershell
python -m pytest tests/test_phase5.py -v
```

### Adding a new test

The convention is straightforward — see any `tests/test_phaseN.py` for examples:

```python
import os; os.environ["CRISP_MOCK"] = "1"
from src.something import SomeClass

class TestSomething:
    def test_basic(self):
        s = SomeClass()
        assert s.do() == "expected"
```

---

## Phase-by-phase notes

### Phase 5 — Multi-hop evidence aggregation

When Phase 3 tags a claim as `AGGREGATED` (it draws from multiple passages),
single-hop retrieval isn't enough. Phase 5 adds a small planner that:

1. **Extracts entities** from the claim (`NER` module — mock by default,
   swap to spaCy via `CRISP_NER_BACKEND=spacy`).
2. **Resolves them in a mini knowledge graph** (`KGLinker`, with 13 seeded
   entities and ~30 surface-form aliases — George Orwell, 1984, France,
   Paris, Germany, Berlin, …).
3. **Plans 2-hop sub-queries** by following priority relations
   (e.g. `PER → country_of_citizenship`, `LOC → capital`).
4. **Merges the new evidence** into the existing `RAGResult` and surfaces
   a `MultiHopTrace` (entities, hops, sub-queries, evidence, notes) that
   downstream phases can replay.

Disable with `CRISP_DISABLE_MULTIHOP=1`.

### Phase 4 — Adaptive evidence retrieval

A controller inspects the top-1 retrieval score and:

- **expands `k`** when the top hit is weak or top-1/2 are too close (the
  retriever is unsure),
- **rewrites the query** with deterministic variants when even expansion
  doesn't help.

A `HybridRetriever` (dense + BM25 fused via Reciprocal Rank Fusion)
provides the underlying recall. Enable with `CRISP_ADAPTIVE=1`.

### Phase 3 — Atomic claim extraction + provenance tagging

The Phase 2 detector scores individual claims, so claim quality bounds the
whole system. Phase 3 adds `ClaimExtractor` with a `Provenance` enum:

| Provenance    | Meaning                                                              |
| ------------- | -------------------------------------------------------------------- |
| `INTRINSIC`   | Answerable from the question alone. Skip NLI — saves cost.           |
| `EXTRINSIC`   | Needs an external passage. The default for most factual RAG claims.  |
| `AGGREGATED`  | Combines 2+ passages. Triggers Phase 5 multi-hop retrieval.          |

Switch between the deterministic `mode="synthetic"` (default, mock-friendly)
and `mode="real"` (T5/PEGASUS atomiser + embedder-based provenance).

### Phase 2 — Per-claim NLI detection + EEDC scoring

Each claim is paired with the retrieved passage and run through a
DeBERTa-v3 NLI model. The EEDC scorer turns the raw NLI probabilities,
retrieval scores, and self-consistency into a single 0–1 calibrated
confidence value. Run `python scripts/calibrate_eedc.py` on a labelled
dev set to refit weights.

### Phase 1 — Baseline RAG

The starting point: BGE embeddings → FAISS top-k → HF generator → answer.
Faithful, well-instrumented, runs in MOCK mode without downloading
models.

### Phase 6 — Post-hoc confidence calibration

Raw Platt weights from Phase 2 give a 0–1 score, but the absolute value
is poorly calibrated (Sigmoid outputs are over-confident when the input
is out-of-distribution). Phase 6 adds a post-hoc calibrator on top:

* `TemperatureScaler` — single-parameter `T` that softens (T > 1) or
  sharpens (T < 1) the Platt score. Fitted by maximising log-likelihood
  via ternary refinement.
* `IsotonicCalibrator` — non-parametric Pool-Adjacent-Violators (PAV)
  on (raw_score, true_label) pairs. More flexible; needs more data.
* `CalibratedEEDC` wraps an `EEDCScorer` + a calibrator behind the same
  `.score(...)` API, so the rest of the pipeline doesn't change.

Metrics live alongside: `expected_calibration_error` (10-bin ECE),
`brier_score`, `log_loss`, `accuracy_at`. Pick a method via
`CRISP_CALIBRATION` env var (`temperature` / `isotonic` / `none`) or
`calibration.method` in `configs/default.yaml`.

### Phase 7 — Adaptive iteration + evidence-guided editor

Instead of one-shot RAG, Phase 7 closes the loop. After Phase 2 flags
claims, the **Adaptive Iteration Controller (AIC)** picks one of four
actions:

| Action  | When                                                        |
| ------- | ----------------------------------------------------------- |
| ACCEPT  | hallucination rate ≤ `accept_rate_threshold`                |
| EDIT    | a few spans are wrong; the `EvidenceGuidedEditor` patches them |
| REGEN   | too many flags; regenerate the answer with extra evidence  |
| STOP    | max iterations reached or rate plateaued                   |

The editor locates each flagged claim's span in the answer (exact
substring → 6-token prefix fallback) and rewrites only that span in one
of three modes:

| Mode         | Behaviour                                                    |
| ------------ | ------------------------------------------------------------ |
| `stub`       | replace span with `[unsupported: <claim>]` (mock-friendly)  |
| `evidence`   | replace with best-matching evidence sentence (token overlap) |
| `regenerate` | call the `Generator` for a faithful rewrite                 |

Toggle the whole loop with `CRISP_AIC=1` and tweak thresholds with
`CRISP_AIC_*` env vars. Each iteration is recorded in
`RAGResult.iteration_history`.

### Phase 8 — Reliability reports

The last phase stitches everything together. `ReportBuilder` consumes a
`RAGResult` and produces a `ReliabilityReport` with one of five labels:

| Label             | Hallucination rate | Mean EEDC |
| ----------------- | ------------------ | --------- |
| `RELIABLE`        | ≤ 5 %              | ≥ 0.75    |
| `MOSTLY_RELIABLE` | ≤ 20 %             | ≥ 0.55    |
| `UNCERTAIN`       | ≤ 50 %             | any       |
| `UNRELIABLE`      | > 50 %             | any       |
| `UNVERIFIABLE`    | no claims          | —         |

Each report renders to **JSON**, **Markdown**, and styled **HTML** (with
verdict badges and per-claim tables). Batch stats land in
`summary.json`. The CLI is `python scripts/report.py --in data/preds.jsonl
--out-dir reports/ --format html`.

---

## Glossary

| Term               | Meaning                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| **RAG**            | Retrieval-Augmented Generation. Give the LLM the right passages; it gives better answers. |
| **Hallucination**  | An LLM asserting something that isn't in the retrieved evidence.                         |
| **NLI**            | Natural Language Inference. A model that classifies a (premise, hypothesis) pair as entail / contradict / neutral. |
| **EEDC**           | Evidence-Entailment Decomposition Confidence. A weighted sum of NLI prob, retrieval score, and self-consistency → single confidence. |
| **Atomic claim**   | A single, indivisible factual statement. One sentence, one fact.                         |
| **Provenance**     | Where a claim's truth comes from — the question itself, one passage, or several.         |
| **AGGREGATED**     | A claim composing info from multiple passages — needs Phase 5 multi-hop verification.    |
| **Multi-hop**      | Following a chain of facts through several retrieval steps to verify one claim.          |
| **Mock mode**      | A deterministic stub mode that lets the whole pipeline run without model downloads or GPU. |
| **BM25**           | A classical sparse retrieval algorithm (TF-IDF with length normalisation).                |
| **RRF**            | Reciprocal Rank Fusion. A simple way to merge ranked lists from different retrievers.    |
| **NER**            | Named Entity Recognition. Tagging spans of text as PER / ORG / LOC / DATE / etc.          |
| **KG**             | Knowledge Graph. A graph of (head, relation, tail) triples — the mini-KG has 13 entities. |

---

## License & acknowledgements

Research use only. Models retain their original licenses
(Llama 3.1 Community License, BGE MIT, DeBERTa Apache-2.0, …).

Built around the research proposal for the **CRISP** thesis framework.
#
