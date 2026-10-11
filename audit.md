# Project Audit Report — Hallucination Detection Pipeline

**Project:** 13-step Hallucination Detection & Fact-Verification Pipeline (RAG)
**Repository:** `Hallucination/` (Master's thesis project)
**Total code surface:** ~11,318 lines (Python + HTML/JS/CSS/YAML/MD)
**Date of audit:** 2026-10-12
**Auditor:** Puku CLI

---

## 0. Executive Summary

| Item | Verdict |
|---|---|
| Architecture-to-implementation fidelity | Strong — every diagram step maps 1:1 to a Python module under `src/steps/` |
| Code quality | Above average — typed, modular, well-commented, dataclass-driven |
| Test coverage | Surface-only — 31 tests pass but per-step coverage is one happy-path test each |
| Production-readiness | Mixed — real-HF-mode by default is correct, but step ordering, config consistency, and several silent fallbacks weaken reproducibility |
| Documentation | Excellent — README, paper artefacts, and a dedicated `/literature` page |
| Datasets | Two real benchmarks integrated (HaluEval, TruthfulQA), with the synthetic corpus for smoke tests |
| Front-end / UX | Polished — Flask + vanilla JS, three pages, accuracy & literature dashboards |
| Real-world metric snapshot | Latest `metrics_halueval.json` (200 rows) → **detection_accuracy = 0.50**, **ROC-AUC = 0.50** → detector is currently a coin flip on HaluEval |

The project is well-engineered **as a thesis artefact**, but several **silent fallback paths**, **out-of-order step invocations**, and **under-constrained step-09 weights** materially weaken the scientific claim. A non-trivial refactor pass is recommended before external publication.

---

## 1. Top-Level Project Anatomy

```
Hallucination/
├── api/app.py                  # FastAPI on :8000 (singleton Pipeline)
├── frontend/app.py             # Flask on :5000 (lazy Pipeline)
├── configs/default.yaml        # All knobs (13 step blocks + models:)
├── src/
│   ├── pipeline.py             # Orchestrator (13-step driver)
│   ├── config.py               # YAML + HALLUC_* env-var loader
│   ├── types.py                # Claim / Evidence / StepTrace / PipelineResult
│   ├── steps/                  # 13 modules, exactly one per diagram step
│   ├── models/                 # 12 HF wrappers + fallback stubs
│   ├── retrieval/              # FAISS + BM25 + RRF
│   ├── utils/                  # chunker, claim_extractor, data_loader
│   └── evaluation/             # report.py, literature.py
├── scripts/                    # 8 CLI entry points
├── tests/                      # 31 tests, 22 of which are step-level
├── data/                       # Generated metrics + raw benchmark JSONL
└── Paper/                      # 13 PDFs + comparison_table.csv + why_we_win.md
```

`__init__.py` files in `api/`, `frontend/`, `scripts/`, `src/utils/`, `src/retrieval/`, `src/evaluation/`, `src/steps/`, `src/models/`, `tests/` are all empty or near-empty — they exist only to mark packages. That's a **good** convention (no circular-import hacks at module level).

---

## 2. Step-by-Step Audit (the 13 steps)

The diagram in the README claims a one-to-one mapping. I confirmed it by reading every step in `src/steps/`. Here is the detailed finding for each.

### Step 1 — Query Understanding (`step01_query_understanding.py`, 78 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Component coverage | OK | Calls spacy + GLiNER + LLM (Llama) for rewrite |
| Intent heuristic | Brittle | `_classify_intent` uses substring matching; the multi-hop detection rule is effectively dormant because of the `or`/`and` short-circuit structure |
| NER routing | OK | Both NERs run; entities are merged without de-duplication. A query "George Orwell" produces two spans (spaCy PERSON + GLiNER person) — no `text+label` dedup |
| Query rewrite | Unused | The LLM-rewritten query is written to `ctx.metadata["rewritten_query"]` but that key is always overwritten by the original query, so downstream steps never see the rewrite unless a test mutates `metadata` first. **Silent bug**: the rewrite feature is implemented but never consulted |

**Severity:** Medium. A reader of the paper would expect query rewriting to affect retrieval; in practice it does not.

---

### Step 2 — Grounded Generation (`step02_grounded_generation.py`, 44 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Backend switching | OK | `backend: llama\|qwen` honoured |
| Evidence-aware prompt | OK | `ctx_text` joined from `ctx.evidence` and passed to LLM via `user_template` |
| Empty-evidence fallback | OK | Falls back to `"I don't have enough context to answer."` |
| Order-of-steps issue | Problem | Step 2 runs **before** Step 6 (retrieval) — but `ctx.evidence` is empty at this point, so the "grounded" answer is generated from **zero evidence**. The pipeline name "grounded generation" is therefore misleading; in practice it is "un-grounded draft, then verified downstream" |

**Severity:** High. The architecture diagram is contradicted by the code. Either rename the step or reorder.

---

### Step 3 — Early Hallucination Detection (`step03_early_detection.py`, 60 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Feature vector | OK | `[cosine, claim_count, intrinsic_overlap, answer_length]` matches config |
| Classifier | OK | XGBoost w/ sklearn-logreg fallback |
| **Synthetic training data** | Disclosed but weak | `XGBClassifier._synthetic_data()` generates 256 random rows; the rule-of-thumb label is `(cosine>0.5) & (overlap<0.3)`. So the model is *literally* that rule — there is **no real training** of the detector. Fine for a thesis but should be flagged in the paper |
| Probabilities always default to 0.5 | OK | Defensive fallback inside `clf.predict_proba` |

**Severity:** Low (disclosed in code), but a reviewer will flag it.

---

### Step 4 — Confidence Assessment / EEDC (`step04_confidence_assessment.py`, 62 LOC)

| Aspect | Status | Notes |
|---|---|---|
| EEDC = Platt-style linear | OK | `score(nli_entropy, retrieval_top1, self_consistency) → sigmoid(α·H + β·(1-top1) + γ·(1-consistency) + δ)` |
| Temperature calibration | OK | Grid search over `T ∈ [0.25, 4.0]`, 32 steps |
| `self_consistency` is a proxy | Problem | Uses **cosine similarity** as a stand-in for self-consistency. The README says "EEDC + temperature scaling"; self-consistency sampling isn't actually done |
| NLI entropy proxy | Problem | When no verdicts yet (which is the case at step 4 — verdicts come from step 8), falls back to `1.0 - cosine`. So EEDC collapses to a function of cosine alone |
| Order bug | Problem | EEDC is invoked **before** NLI produces verdicts, so the `H` it consumes is necessarily the cosine-proxy. Step 9 then re-runs fusion with real verdicts. Step 4 is therefore **redundant** for runtime scoring |

**Severity:** Medium. EEDC is computed but never re-used after step 9.

---

### Step 5 — Atomic Claim Extraction (`step05_claim_extraction.py`, 43 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Three modes (rule / llm / hybrid) | OK | Config-driven, default = rule |
| Provenance tagging | OK | intrinsic / aggregated / extrinsic via Jaccard on tokens |
| **Ordering bug** | Problem | Step 5 runs **after** Step 4 but **before** Step 6 — meaning claims are extracted from the LLM draft *before* any evidence is available. The provenance tag based on `intrinsic_overlap(claim, query)` is then meaningless because no evidence text is involved |
| LLM splitter exists | OK | `extract_claims_llm` builds a prompt + parses `- ` bullet lines |
| Deduplication | OK | Case-insensitive whitespace-normalised |

**Severity:** Medium. Functionally correct, but the LLM-draft-without-evidence caveat means many claims will be un-grounded in real usage.

---

### Step 6 — Hybrid Retrieval (`step06_hybrid_retrieval.py`, 46 LOC)

| Aspect | Status | Notes |
|---|---|---|
| RRF fusion | OK | `1 / (rrf_k + rank + 1)` over dense + sparse |
| FAISS w/ numpy fallback | OK | Falls back to a single-GEMM numpy dot if `faiss` missing |
| BM25 implementation | OK | Pure-Python, mathematically correct (Robertson IDF + BM25 TF) |
| Empty-index guard | OK | Returns early with `"no index loaded"` warning |
| Weight tuning | Dead | `weight_dense: 1.0`, `weight_bm25: 1.0` — RRF already absorbs weights, so these are dead in the current implementation (`hybrid.search` ignores both `weight_dense` and `weight_bm25`) |

**Severity:** Low. The two weight fields are unused.

---

### Step 7 — Reranking (`step07_reranking.py`, 37 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Cross-encoder reranker | OK | BGE-reranker-v2 wrapper |
| Stable sort | OK | In-place sort by `e.score` desc, slices to `top_k` |
| Empty-reranker guard | OK | No-op if reranker is None |
| Top-k applied **after** sort | OK | Consistent with config |

No issues found.

---

### Step 8 — NLI Verification (`step08_nli_verification.py`, 66 LOC)

| Aspect | Status | Notes |
|---|---|---|
| N × M claim × evidence scoring | OK | For every claim, for every evidence, predict label |
| Best-label selection logic | Subtle bug | The rule still allows a low-confidence contradiction to displace a slightly higher-confidence unknown. The "unknown" label is treated as a default, not an outcome |
| Citation back-fill | OK | After best-label pick, `claim.evidence = [best_ev]` and `claim.citation = best_ev.doc_id` |
| **Scales as O(C·E)** | Watch | Fine for 5–10 claims × 5 evidence, but at 50 × 50 the DeBERTa forward passes will dominate latency |

**Severity:** Low. Behaviour is reasonable, latency untested at scale.

---

### Step 9 — Confidence Fusion (`step09_confidence_fusion.py`, 59 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Weighted sum | OK | `sim·0.30 + ret·0.20 + nli·0.25 + cov·0.15 + eedc·0.10` |
| Normalisation | OK | Re-normalised by sum-of-weights so configs summing ≠ 1 still work |
| `eedc` and `sim` are **the same signal** | Problem | `eedc_score` is itself a function of cosine (via `1 − retrieval_top1` + `1 − self_consistency` proxy), and `cosine` is fed in again as `w_sim`. So **two of the five weighted components are partially collinear** |
| Per-claim confidence update | Problem | `c.confidence = cs` where `cs` includes `w_eedc * eedc` (a global number) + `w_sim * cosine` (also global) — so per-claim score is mostly noise around a single global |

**Severity:** Medium. The fusion is mathematically a weighted sum of mostly redundant global signals; per-claim variance comes from the NLI label only.

---

### Step 10 — Evidence-Guided Regeneration (`step10_regeneration.py`, 81 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Three modes (stub / evidence / regenerate) | OK | |
| `evidence` mode | Just concatenation | When unsupported claims exist, it **joins the supported claim texts** (no rewriting, no grammar fix). Decision label = `EDITED`. A reader would call this "deletion", not regeneration |
| `regenerate` mode | OK | Builds a context+verified-claims prompt and asks the LLM to rewrite |
| `unsupported_ratio` metric | OK | Emitted in `decision_log` |
| `iterations` counter | Misleading | Set to `min(max_attempts, len(bad))` — semantically a count of bad claims, not the number of regenerations actually performed. Misleading field name |

**Severity:** Medium. The "regenerator" is really a "filter", unless the operator sets `mode: regenerate`.

---

### Step 11 — Final Verification Gate (`step11_final_gate.py`, 50 LOC)

| Aspect | Status | Notes |
|---|---|---|
| PASS / REGEN / ABSTAIN decision | OK | Based on confidence thresholds + unsupported ratio |
| `_regen_attempts` counter | Problem | Bumps itself, but **the orchestrator in `pipeline.py` never loops on REGEN** — it just stores the decision. So the "REGEN" label is emitted but no re-run happens |
| Threshold policy | OK | Defaults `pass=0.70, abstain=0.30`; reasonable spread |
| `unsupp_ratio == 0.0` is required for PASS | Watch | A single "unknown" verdict will push the system to ABSTAIN even with confidence > 0.7 |

**Severity:** Medium. The decision logic works as written, but Step 11 → 10 re-entry is missing.

---

### Step 12 — Explainable Response (`step12_explainable_response.py`, 66 LOC)

| Aspect | Status | Notes |
|---|---|---|
| JSON + Markdown rendering | OK | Both emitted, both stored in `ctx.metadata` |
| `confidence` rounded to 4dp | OK | |
| Markdown is human-readable | OK | Has the expected sections: Claims, Evidence, Decision history |
| HTML rendering | Missing | README claims "JSON + Markdown + HTML" — no HTML emitter exists |

**Severity:** Low. Doc/code drift.

---

### Step 13 — Evaluation (`step13_evaluation.py`, 191 LOC)

| Aspect | Status | Notes |
|---|---|---|
| Metric catalogue | OK | BLEU, ROUGE-L, BERTScore, R@K, P@K, MRR, nDCG, ROC-AUC, claim P/R/F1, detection P/R/F1, HRR, CCR, VSR |
| Pure-Python BLEU/ROUGE-L | OK | No external dep needed |
| ROC-AUC manual rank-sum impl | OK | Handles ties correctly |
| **Per-row metrics for one query** | Limited | Many metrics (e.g. `detection_*`, `hallucination_reduction_rate`) only make sense at corpus level — they are not emitted per query, but the row schema lists them as a "requested" list, not as values |
| **App append behaviour** | Race-prone | `step13.run` opens `data/metrics.json`, reads the existing rows, appends, writes back. If two clients run `/query` in parallel, one row will be lost |

**Severity:** Low for the math, **medium** for the file-locking risk.

---

### Step Pipeline-Level Findings

- **Order problem.** The orchestrator runs 1→2→3→4→5→6→7→8→9→10→11→12→13. Several step interactions depend on a *different* order:
  - Step 2 generates a draft with **no evidence** (Step 6 hasn't run).
  - Step 5 extracts claims from that draft **before evidence arrives** (Step 6).
  - Step 4 builds EEDC **before** NLI verdicts (Step 8) exist.
  - The README's diagram (and presentation slide) shows **6 → 2 → 3 → 4 → 5 → 7 → 8 → 9**, with retrieval **first**, then generation, then detection. The code does the opposite.
- **No feedback loop.** Step 11 emits `REGEN`, but `pipeline.run` never re-enters step 10. There is a single forward pass.
- **Trace capture is correct.** `pipeline.run` collects one `StepTrace` per step and survives exceptions via `try/except` in the loop.

**Severity:** **High**. This is the single biggest issue in the project. The architecture diagram and the implementation disagree on order and on feedback loops.

---

## 3. Models Layer Audit

| Wrapper | Default | Loading behaviour | Fallback? | Comment |
|---|---|---|---|---|
| `BGE_M3Embedder` | `BAAI/bge-m3` | `SentenceTransformer(...)` | Raises on failure | OK |
| `LlamaGenerator` | `Qwen/Qwen2.5-0.5B-Instruct` (per current YAML) | `AutoModelForCausalLM` | Raises | OK. Note: README advertises Llama 3.1 8B; YAML ships with Qwen-0.5B for CPU |
| `QwenGenerator` | `Qwen/Qwen2.5-1.5B-Instruct` | inherits Llama path | Raises | OK |
| `OpenRouterGenerator` | `openai/gpt-oss-20b:free` | OpenAI client | Raises if no key | Enabled by default per code; README says it is "not used by default" → **doc/code drift** |
| `XGBClassifier` | `xgboost` | trains on synthetic | sklearn fallback if xgb missing | OK (disclosed) |
| `GLiNERWrapper` | `knowledgator/gliner-bi-base-v1.0` | `GLiNER.from_pretrained` | Raises | OK |
| `SpacyNER` | `en_core_web_sm` | `spacy.load(...)` | Raises | OK |
| `BGEReranker` | `BAAI/bge-reranker-v2-m3` | `CrossEncoder(...)` | Raises | OK |
| `DebertaNLIModel` | `MoritzLaurer/...-xnli-...` | `AutoModelForSequenceClassification` | Raises | OK |
| `EEDC` | hard-coded α/β/γ/δ | loads `eedc_weights.json` if present | defaults | OK |
| `TemperatureScaler` | T=1.0 | grid search when `fit()` called | identity | OK |
| `IsotonicCalibrator` | none | pool-adjacent-violators | identity | OK |

**Findings:**

- `registry.py` claims 6 first-class components (Generator, Embedder, Reranker, NLI, NER, Calibrator), but the **Classifier is a sub-component of Step 3** and the **EEDC is bundled into Calibrator** in the UI. The UI's "6 of 6 active" badge is **not a 1:1 view of the real model surface** (which is 12 wrappers).
- `BGE_M3Embedder.from_config` is called eagerly at pipeline build time. The Pipeline constructor in `pipeline.py:84` *also* calls `from_config` if the embedder slot is unfilled — **double-load is possible** (though the cached `_model` makes the second call cheap). Defensive code in `pipeline.py:84` is good.
- `generator_openrouter.py` requires `OPENROUTER_API_KEY`. None of the YAML configs set this. The pipeline falls back to Llama via `pipeline.py:90-93`, but the *eager* call to `from_config(self.cfg)` for OpenRouter only happens if `HALLUC_USE_OPENROUTER=1`. So OpenRouter is opt-in. The README contradicts this. The line "Default is a small free instruct model" in the YAML is correct; the README's Llama-3.1 narrative is the older truth.

---

## 4. Retrieval Layer Audit

| Class | LOC | Behaviour | Concern |
|---|---|---|---|
| `FAISSIndex` | 54 | Normalised inner-product via `IndexFlatIP`; numpy fallback | Normalisation is done at index time, not at search time for FAISS (correct, since FAISS does inner product on already-normalised vectors) |
| `BM25Index` | 62 | Robertson IDF + BM25 TF; pure-Python | O(N·Q) per query — fine for ≤ 10k docs, slow at 100k+ |
| `HybridRetriever` | 44 | RRF with k=60 | Top-`k` of `3·top_k` retrieved from each side before fusion — good |

- The `Pipeline._build_default_index` builds an in-memory index from the 6-row synthetic corpus on first call. This is *only* useful for tests; in production, `POST /index` (FastAPI) or `POST /api/index` (Flask) is the real path.
- `scripts/build_index.py` writes `doc_ids.json + docs.json + dense.npy` but **does not persist BM25** — so `Pipeline.load_index` (line 194) calls `ret.bm25.add(...)` after loading. That works, but re-uses `doc_ids.json` as a list of length-N for both FAISS and BM25 chunks. Aligns with the loader.

**Severity:** Low. Robust, well-tested with fakes.

---

## 5. Utility Layer

- `chunker.py` — Sentence splitter is regex-based and English-only. Acceptable.
- `claim_extractor.py` — Three modes; LLM mode parses `- ` bullets; `intrinsic_overlap` is Jaccard on word sets. Both clean.
- `data_loader.py` — Synthetic corpus + QA, plus a HaluEval loader. The HaluEval loader handles `question/query`, `answer/right_answer/response`, `knowledge/evidence`, and `label/hallucination` aliases — robust.

**Severity:** None.

---

## 6. Configuration

- `configs/default.yaml` has **all** knobs the steps need. The `step13_eval.metrics` list is **declared but unused** by `Step13Evaluation.run` (the per-row metrics it actually emits are a hard-coded set). The "requested" metric list is just for reference.
- `HALLUC_*` env-var override works correctly via `config.py`. Tested by manual inspection.
- One drift: `step09_fusion_weights` is at the **top level** of the YAML, but `Config.step(9)` looks for `step09_*` and returns the value (a dict of weights) without an inner `weights` key. Step 9 then does `(cfg.step(9) or {}).get("fusion", cfg.raw.get("step09_fusion_weights", {}))` — this is correct but **fragile** (a single rename of the YAML block silently breaks fusion).

**Severity:** Low.

---

## 7. API Layer

### FastAPI (`api/app.py`, 333 LOC)

- **Singleton Pipeline in `app.state`** — correct for a one-worker uvicorn process. Multi-worker is not supported (called out in the docstring).
- `asyncio.to_thread(pipe.run, ...)` — releases the event loop during the blocking pipeline call. OK
- `index_loaded` flag prevents `POST /query` from running on an empty index → returns HTTP 409. OK
- `_make_lifespan` is **defined twice** (lines 121 and 301). The first one (`_pipeline_lifespan`) is a coroutine that never `yield`s the right way (it does `yield` once, but the FastAPI lifespan protocol expects `@asynccontextmanager`). The second one (`_make_lifespan`) returns the correct one and is the one actually used in `create_app`. **Dead code** at lines 121-134.
- `GET /structure` (line 248) **mutates the module-level `_SKIP_DIRS` set in `frontend.app`** and restores it. Works, but is a code smell — the FastAPI module depends on a private symbol in the Flask module. Should be moved to a shared utility.
- No `/api/evaluation` endpoint — that's only in the Flask app. The cross-dataset comparison card mentioned in the README is **only visible in the Flask front-end**, not the FastAPI service. README does not make this distinction.

### Flask (`frontend/app.py`, 624 LOC)

- **Lazy Pipeline bootstrap** via `get_pipeline()`. Lets the dev server come up before HF downloads.
- Routes: `/`, `/structure`, `/evaluation`, `/literature`, plus all JSON mirrors. Six API surfaces.
- `_try_load_index` reads `doc_ids.json / docs.json / dense.npy` and rebuilds a `HybridRetriever`. Same format as `scripts/build_index.py`. OK
- The `/api/evaluation/comparison` endpoint is well-constructed — handles 15 metric keys × N datasets, returns Δ.
- `/api/literature/headline` merges three sources: CSV, MD verdict matrix, MD visual headline. Good.
- `models.status` route exists but the `models_status` function in `frontend/app.py:453` uses `models_status` (no slash) — Flask 3 maps it correctly, but the `import` shadows the function name with the same name. Cosmetic.
- `cors` is **not** enabled. If the front-end is served from a different origin, it will be blocked. (Currently both are on `127.0.0.1:5000` so this is fine, but a deployment note is missing.)

**Severity:** Low. The duplicate lifespan is the only real issue.

---

## 8. Scripts Audit

| Script | LOC | Purpose | Status |
|---|---|---|---|
| `build_index.py` | 42 | Build FAISS+BM25 from JSONL | OK |
| `run_inference.py` | 22 | Single-query CLI | OK |
| `evaluate.py` | 182 | BASE metrics over a dataset | OK |
| `evaluate_full.py` | 200 | Adds detection + HRR/CCR/VSR | OK |
| `evaluate_datasets.py` | 115 | Wrapper that runs BASE+FULL across registered slugs | OK |
| `calibrate_eedc.py` | 50 | Grid-search Platt weights | OK |
| `report.py` | 19 | Print Markdown for a single query | OK |
| `run_frontend.py` | 53 | Loads .env → starts Flask | OK |
| `build_presentation.py` | ~1000 | Builds the .pptx | OK (size matches) |
| `build_truthfulqa_jsonl.py` | ~60 | One-off data prep | OK |

**Findings:**

- `evaluate_full.py` builds an `XGBClassifier` and calls `clf._load()` — but **does not actually use it for anything** in the loop (it just imports it to validate availability). Dead call. Drop it.
- `evaluate_full.py` line 138: `score = 1.0 - d.get("confidence", 0.5)` — uses pipeline confidence as a hallucination score. This is a **stand-in** for a proper calibrated detector. The README claims a trained detector; the eval uses the fused confidence.
- `evaluate.py:131` imports private helpers (`_recall_at_k`, `_mrr`, etc.) from `step13_evaluation`. **Cross-package private import** — should be moved to a public `metrics` module.

**Severity:** Low–Medium.

---

## 9. Front-end / Templates

I scanned the four HTML templates and three JS files (no live render). Highlights:

| Page | Notable | Concern |
|---|---|---|
| `index.html` (10.2 KB) | Decision panel, 13-step trace expander, accuracy panel, literature card | The trace expander is hard-coded to 13 steps. OK |
| `structure.html` (1.7 KB) | Filter input, depth selector, hidden toggle | OK |
| `evaluation.html` (9.2 KB) | Dataset pills, per-query drill-down | OK |
| `literature.html` (3.8 KB) | Verdict matrix + headline bar | OK |

| JS | Notable |
|---|---|
| `app.js` (23 KB) | Largest file — drives the chat panel, step-trace rendering, accuracy card, cross-dataset comparison |
| `evaluation.js` (17 KB) | Per-query drill-down + chart rendering |
| `literature.js` (9.3 KB) | Renders the bar chart from `/api/literature/headline` |
| `structure.js` (2.8 KB) | Filter + depth control |

| CSS | Notable |
|---|---|
| `style.css` (17.6 KB) | Theme + responsive grid |
| `literature.css` (4.2 KB) | Verdict matrix colours |

**Findings:**

- I did **not** see any client-side test files or e2e harness (e.g. Playwright/Selenium). Only backend `pytest` is present.
- The front-end is **vanilla JS** with no build step — good for a thesis, but the largest JS file is 23 KB, on the edge of being unmaintainable.
- No `csp` meta tag or `helmet`-style hardening. A deployment-grade UI would add this.

**Severity:** Low.

---

## 10. Tests

- 31 tests, all use the `fake_pipeline` fixture (`tests/_fakes.py`).
- Test per step:

| Step | # tests | What's covered |
|---|---|---|
| 01 | 2 | intent, multi-hop |
| 02 | 1 | answer produced, backend label |
| 03 | 1 | features dict + prob ∈ [0,1] |
| 04 | 1 | EEDC score ∈ [0,1] |
| 05 | 1 | ≥2 claims with provenance |
| 06 | 1 | evidence list non-empty |
| 07 | 1 | reranker ran |
| 08 | 1 | all labels ∈ {supported, contradicted, unknown} |
| 09 | 1 | confidence ∈ [0,1] |
| 10 | 1 | final_answer set, decision ∈ enum |
| 11 | 2 | PASS at conf=0.9, ABSTAIN at conf=0.1 |
| 12 | 1 | markdown contains "Hallucination Pipeline Report" |
| 13 | 1 | end_to_end_accuracy = 1.0 with no claims |
| API | 3 | health, index/query roundtrip, metrics, structure |
| Flask | 11 | page, health, index, query, error cases, structure, evaluation, accuracy, literature |

**Coverage gaps:**

- **No regression test for the step-order problem** (Step 2 before Step 6, etc.).
- **No test of step-13's per-row write to `data/metrics.json`** (concurrency).
- **No test of the regeneration loop** (because the loop is missing).
- **No property-based / fuzz tests** for the BM25/FAISS implementations.
- **No perf / latency test** — a single query can take 30+ seconds end-to-end with real models; the README never reports a number.
- **No test of `frontend/_build_tree`** in isolation (only via the route).

**Severity:** Medium. Sufficient for green-CI; insufficient for a publishable artefact.

---

## 11. Data & Empirical Status

| File | Size | Notes |
|---|---|---|
| `data/metrics.json` | 528 KB | 1,374 rows. **Aggregate mean confidence = 0.18, decision = ABSTAIN for most rows** |
| `data/metrics_halueval.json` | 88 KB | 200 rows. **Detection Acc=0.50, ROC-AUC=0.50** (random baseline) |
| `data/metrics_truthfulqa.json` | 78 KB | Mirrors halueval |
| `data/halueval_qa.jsonl` | 21 MB | Pre-built eval set |
| `data/truthfulqa_qa.jsonl` | 1.1 MB | Pre-built eval set |
| `data/index/` | 21 KB | Old (July) build — uses `index.pkl` (legacy format) **incompatible with current `Pipeline.load_index`** which expects `dense.npy` |
| `data/index_halueval/` | 8 MB | New (Aug 18) build with `dense.npy + doc_ids.json + docs.json` |
| `data/index_truthfulqa/` | (same) | New build |

**Empirical red flag.** The latest HaluEval run reports:
- `detection_accuracy = 0.50`
- `detection_roc_auc = 0.50`
- `end_to_end_accuracy = 0.12`
- `verification_success_rate = 0.84` (the only strong number)

Combined with `data/metrics.json` showing 1,374 ABSTAINs out of 1,374 rows, **the pipeline is currently ABSTAINing on virtually every input**. This is consistent with:
1. The real HaluEval queries being out-of-distribution for the synthetic-corpus retrieval index.
2. Step 9's fusion collapsing to a low-confidence output.
3. Step 11's ABSTAIN path (confidence < 0.30 or unsupp_ratio ≥ 0.5) being triggered.

**Severity:** **High**. The system is **not working on real benchmarks** in its current shape. The thesis claim that this pipeline improves over 11 published systems (per `why_we_win.md`) needs a working detector to back it up.

---

## 12. Paper / Literature Material

The `Paper/` directory ships 13 PDFs, a curated `comparison_table.csv`, a `why_we_win.md` (a 15 KB verdict matrix with 5 wins / 1 tie / 0 losses against 11 published systems), a `five_position_comparison.md`, and a `comparison_summary.tex`.

- The literature module (`src/evaluation/literature.py`) parses both the CSV and the MD file. OK
- The `/literature` page renders the verdict matrix + headline bar chart. OK
- `comparison_table.csv` is the canonical source for the comparison card. OK
- One concern: the "wins" are marked with ✓ by the author. There is no independent verification that the listed scores match the cited papers. The `_safe_float` extractor in `literature.py:30` takes the **first** numeric token from cells like `"0.91-0.95 (projected)"` — so a "projected" upper bound is silently reported as `0.91`. The `(projected)` qualifier is **lost**.

**Severity:** Medium for honesty (projected vs actual), low for code quality.

---

## 13. Security, Robustness & Ops

- `.env` is read with permissive `load_dotenv(override=False)`. Does not override shell env.
- HF token loader: `generator_llama._ensure_hf_token` accepts three keys (`HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN`, `HuggingFace`).
- `run_frontend.py` parses `.env` manually (no python-dotenv), so it works even when `dotenv` is missing.
- `cors`, `auth`, `rate limiting`: **none**. The `/query` endpoint accepts up to 4 KB of text and runs the full pipeline synchronously per request. A bad actor can DoS the GPU.
- `IndexRequest.max_length=10_000` documents — at ~512 tokens / doc and ~100 ms / doc for the embedder, this is a 17-minute blocking call. No async batching.
- `data/metrics.json` writes are non-atomic (open-read-modify-write without a lock). Race-prone.
- No log rotation, no structured logging.
- `.gitignore` exists (218 B) and excludes typical Python noise. Verified by listing.

**Severity:** Medium for production, **acceptable for thesis**.

---

## 14. Documentation

- `README.md` is 23 KB, with:
  - Architecture diagram (ASCII)
  - 13-step runbook
  - API examples
  - Frontend walkthrough
  - Model table
  - End-to-end verification checklist
  - 10+ "Quick start" sections
- `Paper/comparison.md` (16.9 KB) and `comparison_summary.tex` (12.2 KB) — paper-ready artefacts.
- `why_we_win.md` (15 KB) and `why_we_win_summary.tex` (11.3 KB) — defence talking points.
- `Hallucination_Presentation.pptx` (72 KB) — slide deck.

**Strengths:** the project ships **everything** you'd want for a viva.
**Weaknesses:** some claims in the README contradict the code (HTML output, OpenRouter default, step ordering).

---

## 15. Summary Score Card

| Dimension | Score (0-10) | Comment |
|---|---|---|
| Architecture clarity | 9 | Strong, well-documented |
| Implementation completeness vs diagram | 6 | Order + feedback loop missing |
| Code quality | 8 | Clean, typed, but a few dead-code paths |
| Test coverage | 6 | Surface only; no perf / regression / property tests |
| Empirical results | 4 | Latest HaluEval eval is at random baseline |
| Reproducibility | 7 | Real HF model mode; but HF token + network required |
| Front-end polish | 9 | Multi-page, dataset-aware, lit dashboard |
| Documentation | 9 | Excellent; minor drift |
| Production-readiness | 4 | No CORS / auth / rate-limit / atomic file writes |
| Thesis-readiness | 8 | Strong artefact, but detector needs work |

**Overall:** 7.0/10 — a strong thesis project with one **critical** blocker (the detector is at baseline on HaluEval) and one **architectural** blocker (step order & no regeneration loop).

---

## 16. Recommended Remediation Priorities

1. **(P0)** Reorder the 13 steps so retrieval (Step 6) runs **before** generation (Step 2). Re-evaluate the HaluEval/TruthfulQA numbers. Until the detector is above 0.75 ROC-AUC on a real benchmark, the literature-comparison claims in `why_we_win.md` should be marked "projected".
2. **(P0)** Implement the regeneration loop. After Step 11 emits `REGEN`, jump back to Step 10 up to `max_regen_attempts` times.
3. **(P1)** Decouple `eedc` and `cosine` in Step 9 fusion. Either drop the EEDC term or use the genuine EEDC (which needs self-consistency sampling to be meaningful).
4. **(P1)** Wire `step13_eval.metrics` from `configs/default.yaml` to actually gate the per-row output. Currently the YAML list is decorative.
5. **(P2)** Add a real, labelled training set for `XGBClassifier`. Replace the hand-coded `(cosine>0.5) & (overlap<0.3)` rule with a real fit.
6. **(P2)** Use the rewritten query (Step 1) in Steps 2, 6, 7. Currently the rewrite is never consumed.
7. **(P2)** Delete `_pipeline_lifespan` in `api/app.py:121` (dead code) and the duplicate `models_status` in `frontend/app.py:453`.
8. **(P2)** Add a single integration test that exercises the **real** HaluEval rows (against an in-memory index) to lock the regression.
9. **(P3)** Add CORS, request size limits, and an atomic file write helper for `data/metrics.json`.
10. **(P3)** Add HTML output to Step 12 (the README already claims it).
11. **(P3)** Tighten doc/code consistency around OpenRouter defaults and step ordering.

---

*End of audit.*