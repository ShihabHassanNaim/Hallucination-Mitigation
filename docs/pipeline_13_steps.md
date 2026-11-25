# CRISP — The 13-Step Hallucination Detection Pipeline

A complete, code-grounded walkthrough of every step in `src/steps/`, in the
order they actually execute, with the data flowing between them.

> **Source of truth.** Every claim in this document was read out of the code
> at the file:line references given. Where the code and the older prose
> disagree, the code wins.

---

## Table of contents

1. [The one thing to understand first](#1-the-one-thing-to-understand-first)
2. [The master flow diagram](#2-the-master-flow-diagram)
3. [Data contract — the `StepContext` bag](#3-data-contract--the-stepcontext-bag)
4. [Phase A — Understand & Ground (steps 1, 6, 7, 2)](#phase-a--understand--ground-steps-1-6-7-2)
5. [Phase B — Detect (steps 3, 5, 8, 4, 9)](#phase-b--detect-steps-3-5-8-4-9)
6. [Phase C — Mitigate & Decide (steps 10, 11, 12, 13)](#phase-c--mitigate--decide-steps-10-11-12-13)
7. [The regeneration loop](#the-regeneration-loop)
8. [Step reference table](#step-reference-table)
9. [Configuration knobs](#configuration-knobs)
10. [Known defects and honest caveats](#known-defects-and-honest-caveats)

---

## 1. The one thing to understand first

**Step numbers ≠ execution order.**

The pipeline has 13 step modules named `step01_…` through `step13_…`. The
numbers are kept stable so that every trace, figure and thesis table can point
at a specific module. But the orchestrator runs them in a *different* order,
and it does so deliberately.

The registry lives at `src/steps/__init__.py:98-115`:

```python
return [
    step01_query_understanding.Step01QueryUnderstanding,
    # --- grounding first: retrieve + rerank so generation is grounded ---
    step06_hybrid_retrieval.Step06HybridRetrieval,
    step07_reranking.Step07Reranking,
    step02_grounded_generation.Step02GroundedGeneration,
    # --- then detection over the generated answer ---
    step03_early_detection.Step03EarlyDetection,
    step05_claim_extraction.Step05ClaimExtraction,
    step08_nli_verification.Step08NLIVerification,
    # EEDC consumes NLI verdict entropy, so it runs after step 8.
    step04_confidence_assessment.Step04ConfidenceAssessment,
    step09_confidence_fusion.Step09ConfidenceFusion,
    step10_regeneration.Step10Regeneration,
    step11_final_gate.Step11FinalGate,
    step12_explainable_response.Step12ExplainableResponse,
    step13_evaluation.Step13Evaluation,
]
```

### The mapping

```
STEP NUMBER :  1   2   3   4   5   6   7   8   9  10  11  12  13
EXEC  ORDER :  1 → 6 → 7 → 2 → 3 → 5 → 8 → 4 → 9 → 10 → 11 → 12 → 13
PHASE        :  └───── A ─────┘  └──────── B ────────┘  └── C ──┘
```

### Why the order was chosen

| Reordering | Reason |
|---|---|
| **6, 7 before 2** | Retrieval and reranking run *before* generation so the generator receives real retrieved context in its prompt. This was the original **P0 defect**: generation ran first, so `ctx.evidence` was always empty and every answer was ungrounded. |
| **4 after 8** | EEDC's entropy component is computed over the NLI verdicts. If step 4 ran before step 8, `nli_entropy` would always fall back to the `1 - cosine` proxy and the entropy term would carry no information. |
| **5 before 8** | The NLI verifies *claims*, so claims must exist first. |
| **3 before 5** | Step 3's feature vector is computed on the raw answer. (This ordering has a side effect — see [Known defects](#known-defects-and-honest-caveats).) |

---

## 2. The master flow diagram

```
                              ┌────────────────────────────────────┐
                              │  QUERY  (user question)           │
                              └──────────────────┬─────────────────┘
                                                 │
 ══════════════════════ PHASE A — UNDERSTAND & GROUND ═══════════════
                                                 │
   ┌─────────────────────────────────────────┐
   │ ①  QUERY UNDERSTANDING                  │   step01_query_understanding.py
   │                                         │
   │  intent      ← rule-based classifier    │   labels: factual | opinion |
   │                (step01:68)              │           multi-hop | math | other
   │                                         │
   │  entities    ← spaCy NER + GLiNER       │   knowledgator/gliner-bi-base-v1.0
   │                                         │   + en_core_web_sm
   │                                         │
   │  rewrite     ← LLM ("return ONE rewrite")│  fallback: strip trailing ? . !
   │                                         │
   └──────────────────┬──────────────────────┘
                      │  rewritten_query, intent, entities
                      ▼
   ┌─────────────────────────────────────────┐
   │ ⑥  HYBRID RETRIEVAL                     │   step06_hybrid_retrieval.py
   │                                         │
   │   dense   BAAI/bge-m3  ──► FAISS        │   top_k = 5
   │   sparse  BM25 (k1=1.5, b=0.75)         │   rrf_k = 60
   │   FUSE    Reciprocal Rank Fusion         │   src/retrieval/hybrid.py:25
   │          1/(k + rank)  over BOTH lists  │
   │                                         │
   │   ⚠ closed-book → retrieval_source:none│   → ctx.evidence = []
   └──────────────────┬──────────────────────┘
                      │  ctx.evidence[5]  (doc_id, text, score)
                      ▼
   ┌─────────────────────────────────────────┐
   │ ⑦  RERANKING                            │   step07_reranking.py
   │                                         │
   │  BAAI/bge-reranker-v2-m3 cross-encoder  │   re-scores each (query, passage)
   │  sort by score desc → truncate to top_k  │   then ctx.evidence = evidence[:5]
   └──────────────────┬──────────────────────┘
                      │  re-ranked evidence
                      ▼
   ┌─────────────────────────────────────────┐
   │ ②  GROUNDED GENERATION                  │   step02_grounded_generation.py
   │                                         │
   │  prompt = system + Context + Question   │   Qwen/Qwen2.5-0.5B-Instruct
   │                                         │   temp 0.0, max_new_tokens 256
   │  ┌───────────────────────────────────┐  │
   │  │ IF provided_answer exists →       │  │   VERBATIM pass-through.
   │  │   use it, DO NOT call the LLM     │  │   The generator is never loaded.
   │  └───────────────────────────────────┘  │
   └──────────────────┬──────────────────────┘
                      │  ctx.answer
 ══════════════════════ PHASE B — DETECT ═══════════════════════════════
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
   ┌──────────────────────┐   ┌──────────────────────┐
   │ ③  EARLY DETECTION   │   │ ⑤  CLAIM EXTRACTION  │  step05_claim_extraction.py
   │                      │   │                      │
   │  features[4]:        │   │  answer → atomic      │  mode: rule (default)
   │   cosine             │   │  claims, deduped      │  sentence split, ≥5 chars
   │   claim_count  ⚠1   │   │                      │
   │   intrinsic_overlap  │   │  provenance tag:      │  thresholds 0.7 / 0.3
   │   answer_length      │   │   intrinsic  (ov≥0.7) │
   │                      │   │   extrinsic  (ov≤0.3) │
   │  XGBoost → P(halluc) │   │   aggregated  (middle)│
   │                      │   │                      │
   │  → feature_vector    │   │  → ctx.claims         │
   │  → hallucination_    │   │                      │
   │    probability       │   │                      │
   └──────────┬───────────┘   └──────────┬───────────┘
              │                          │
              └────────────┬─────────────┘
                           ▼
   ┌─────────────────────────────────────────┐
   │ ⑧  NLI VERIFICATION                    │   step08_nli_verification.py
   │                                         │
   │  DeBERTa-v3-MNLI  (mDeBERTa-v3-xnli)   │
   │                                         │
   │  for each claim × each evidence:        │
   │    premise = _claim_premise(ev, claim)  │  ← claim-conditioned!
   │    (best-overlap sentences + neighbours)│    max_premise_chars 1800
   │    label, score = NLI(premise, claim)    │
   │                                         │
   │  pick best verdict per claim            │  supported | contradicted
   │    supported    → conf = score           │  | unknown
   │    contradicted → conf = 1 - score       │
   │    unknown      → conf = 0.3            │
   │                                         │
   │  attach best evidence to each claim     │
   └──────────────────┬──────────────────────┘
                      │  ctx.verdicts[], claim.verdict, claim.confidence
                      ▼
   ┌─────────────────────────────────────────┐
   │ ④  CONFIDENCE ASSESSMENT (EEDC)         │   step04_confidence_assessment.py
   │                                         │
   │  nli_entropy  = Shannon entropy over   │   H = -Σ p log p, normalised
   │                 {sup, con, unk} / log 3 │   to [0,1]
   │                                         │
   │  retrieval_top1 = cosine               │   (proxy)
   │  self_consistency = cosine             │   (proxy — multi-sample
   │                                         │    not implemented)
   │                                         │
   │  EEDC score → temperature calibration    │  method: temperature
   └──────────────────┬──────────────────────┘
                      │  ctx.eedc_score
                      ▼
   ┌─────────────────────────────────────────┐
   │ ⑨  CONFIDENCE FUSION                   │   step09_confidence_fusion.py
   │                                         │
   │  7 components, standardised (z-score):  │
   │    similarity      risk  +1             │
   │    retrieval_top1  risk  +1             │
   │    nli             TRUST −1             │
   │    coverage        TRUST −1             │
   │    eedc            risk  +1             │
   │    detector_trust  TRUST −1             │
   │    grounding       TRUST −1             │
   │                                         │
   │  risk = Σ wᵢ · signᵢ · zᵢ              │
   │  P(halluc) = sigmoid(risk)              │
   │  ctx.confidence = 1 − P(halluc)         │
   │                                         │
   │  ⚠ evidence-dependent terms DROPPED     │
   │    when there is no retrieved source    │
   └──────────────────┬──────────────────────┘
                      │  ctx.confidence, hallucination_probability
 ══════════════════════ PHASE C — MITIGATE & DECIDE ═══════════════════
                      ▼
   ┌─────────────────────────────────────────┐
   │ ⑩  REGENERATION  (mitigation)          │   step10_regeneration.py
   │                                         │
   │  mode = claim_edit (default)            │
   │                                         │
   │  flagged = claims with verdict==        │
   │            contradicted AND strength ≥  │
   │            min_contradiction_confidence  │
   │                                         │
   │  for each flagged claim:                │
   │    ① locate its span in the answer      │
   │    ② replacement = best sentence of     │
   │       THAT claim's evidence             │
   │    ③ guards: proportionate? relevant?   │
   │    ④ splice it in, leave rest intact    │
   │                                         │
   │  decisions: EDITED | ABSTAIN | KEEP     │
   └──────────────────┬──────────────────────┘
                      │  ctx.final_answer, ctx.iterations
                      ▼
   ┌─────────────────────────────────────────┐
   │ ⑪  FINAL GATE                          │   step11_final_gate.py
   │                                         │
   │   score ≥ 0.70 AND unsup == 0  → PASS   │
   │   score < 0.30 OR unsup ≥ 0.5 → ABSTAIN│
   │   otherwise               → REGEN      │  → loop back to ③
   │   (after max attempts: PASS if ≥0.5)   │
   └──────────────────┬──────────────────────┘
                      │  ctx.decision
                      ▼
   ┌─────────────────────────────────────────┐
   │ ⑫  EXPLAINABLE RESPONSE                │   step12_explainable_response.py
   │                                         │
   │  Markdown report + JSON blob            │  → ctx.metadata["report_markdown"]
   │  claims + verdicts + evidence           │  → ctx.metadata["report_json"]
   │  + decision history                    │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │ ⑬  EVALUATION                          │   step13_evaluation.py
   │                                         │
   │  retrieval  Recall@5 P@5 MRR nDCG       │
   │  claims     P / R / F1 / support rate   │
   │  detection  acc / P / R / F1 / ROC-AUC  │
   │  reduction  HRR, CCR, VSR               │
   │                                         │
   │  → appended to data/metrics.json        │
   └─────────────────────────────────────────┘
```

---

## 3. Data contract — the `StepContext` bag

Every step receives and mutates one shared object. Defined at
`src/steps/__init__.py:16-34`.

| Field | Type | Written by | Read by |
|---|---|---|---|
| `query` | `str` | constructor | 1, 6, 7, 12 |
| `cfg` | `Config` | constructor | every step |
| `answer` | `str` | 2 | 3, 5, 9, 10, 12, 13 |
| `claims` | `list[Claim]` | 5 (mutated by 8) | 8, 9, 10, 11, 12, 13 |
| `evidence` | `list[Evidence]` | 6 (mutated by 7) | 8, 9, 10, 12, 13 |
| `feature_vector` | `dict[str,float]` | 3 | 4, 9 |
| `verdicts` | `list[ClaimVerdict]` | 8 | 4 |
| `eedc_score` | `float` | 4 | 9 |
| `confidence` | `float` | 9 | 10, 11, 12, 13 |
| `decision` | `str` | 11 | 12, 13 |
| `iterations` | `int` | 10 | result |
| `decision_log` | `list[dict]` | 10, 11 | 12 |
| `final_answer` | `str` | 10 | 12, result |
| `metadata` | `dict` | all steps | all steps |

The `metadata` bag is the side-channel for cross-step data that does not fit
the typed fields — `rewritten_query`, `hallucination_probability`,
`provided_answer`, `row_label`, `gold_answer`, `retrieval_source`,
`grounding_source`, `regeneration_attempt`.

---

## Phase A — Understand & Ground (steps 1, 6, 7, 2)

### ① Query Understanding

**File:** `src/steps/step01_query_understanding.py:17`
**Config:** `step01_query_understanding` — `intent_labels`, `rewrite_variants: 2`

Parses the query into three things.

**Intent** (`_classify_intent`, line 68) — a rule-based classifier, not a
model. Checks in order:

| Test | Label |
|---|---|
| contains `+ - * / equals compute` | `math` |
| contains `" and "` (or `" of "` with >1 `?`) | `multi-hop` |
| starts with `do you think` / `is it good` / `should` | `opinion` |
| contains `?` or starts with `who what when where why how` | `factual` |
| otherwise | last label (`other`) |

**Entities** — merged from two NER models:
- `en_core_web_sm` (spaCy) — standard named entities.
- `knowledgator/gliner-bi-base-v1.0` (GLiNER) — passed the *same* label list,
  so it can be prompted toward the domain labels.

**Query rewrite** — asks the LLM for exactly one rewrite
(`"Return ONE rewrite, no commentary."`, 64 new tokens). If the LLM is
unavailable or returns the query unchanged, the fallback is heuristic: strip
trailing `? . !`.

**But the rewrite never reaches the rest of the pipeline.** Line 66 persists
`variants[0]` as `ctx.metadata["rewritten_query"]`, and `variants[0]` is
always the *original* query — the original is seeded as the first element at
line 38, and any rewrite is *appended* after it. So the key named
"rewritten_query" actually holds the original query, and the real rewrite
sits in the step payload as `query_variants[1]` where nothing downstream reads
it.

Consequence: steps 3, 6, 7 and 2 all read `rewritten_query` and therefore all
operate on the **original** query. The rewrite is computed, reported, and
discarded.

> This is a latent bug rather than a correctness bug — the current numbers
> were all produced with the original query, so nothing in the results is
> wrong. But the LLM rewrite costs a generator call on every row and buys
> nothing. To make it real, line 66 would need to persist `variants[-1]`
> (and the generator's `generate()` would need to return only the rewritten
> string rather than the raw prompt echo).

---

### ⑥ Hybrid Retrieval

**File:** `src/steps/step06_hybrid_retrieval.py:32`
**Fusion:** `src/retrieval/hybrid.py:25`
**Config:** `step06_retrieval` — `top_k: 5`, `rrf_k: 60.0`, `bm25_k1: 1.5`,
`bm25_b: 0.75`, `weight_dense: 1.0`, `weight_bm25: 1.0`, `index_path: data/index`

Two retrievers run over the same chunked corpus:

| Retriever | Model | Index | Params |
|---|---|---|---|
| Dense | `BAAI/bge-m3` | FAISS (flat, exact) | `top_k × 3` candidates |
| Sparse | — | BM25 | `k1=1.5`, `b=0.75` |

Both contribute to a Reciprocal Rank Fusion score:

```
rrf_score(doc) = Σ over both lists:  1 / (rrf_k + rank + 1)
```

The top-`top_k` documents by RRF score become `ctx.evidence` as
`Evidence(doc_id, text, score)`.

**The closed-book switch.** `retrieval_source: "none"` makes step 6 return
with `ctx.evidence = []` and a warning (lines 44-55). This is required for
TruthfulQA, whose corpus was built from the dataset's own `knowledge` field
— the set of *correct* answers. Retrieving there finds the answer key for
every row, so any verification against the retrieved document measures
"is this answer the reference answer", which **is the label**. The evaluation
harness overrides this per dataset via `--retrieval-source`.

**Two failure modes to know:**
- No index loaded → step 6 returns empty with `"warning": "no index loaded"`.
  Historically this silently fell back to a 6-document synthetic corpus,
  which is why early evaluations produced a 0.50 baseline. Always pass
  `--index`.
- The embedder failing → the query vector becomes a zero vector (line 74)
  rather than raising.

---

### ⑦ Reranking

**File:** `src/steps/step07_reranking.py:11`
**Config:** `step07_reranking` — `enabled: true`, `top_k: 5`

A cross-encoder (`BAAI/bge-reranker-v2-m3`) scores each `(query, passage)`
pair jointly — unlike the bi-encoder in step 6, which embeds them
independently. Each evidence's score is **overwritten** with the cross-encoder
score, the list is re-sorted descending, and truncated to `top_k`.

Disabling it (`enabled: false`) leaves step 6's RRF order intact.

---

### ② Grounded Generation

**File:** `src/steps/step02_grounded_generation.py:21`
**Config:** `step02_grounded_generation` — `backend: llama`, `max_new_tokens: 256`,
`temperature: 0.0`, `dtype: fp16`, `device: auto`

Builds the prompt from `configs/default.yaml:231-243`:

```
You are a precise assistant. Answer the user's question using ONLY the
information in the provided context. If the context does not contain the
answer, say "I don't know." Do not invent facts.

Context:
{context}

Question: {question}

Answer:
```

`{context}` is the retrieved evidence joined with blank lines (line 37).

### The provided-answer path

Lines 46-53 are the single most important branch in the whole pipeline:

```python
provided = self.ctx.metadata.get("provided_answer")
if provided is not None:
    answer = str(provided)
    backend = f"{backend} (provided)"
elif gen is None:
    answer = "I don't have enough context to answer."
else:
    answer = gen.generate(system=..., user=user)
```

When a benchmark supplies the answer, step 2 **passes it through verbatim
and never loads the generator**.

**Why this matters for the science.** A dataset that pairs a faithful answer
with a hallucinated one for the same question yields two rows differing only
in the answer. If the pipeline generated its own answer from the query, both
rows would get the same answer, the same detector features, and therefore a
ROC-AUC of exactly 0.50 — by arithmetic, not by chance. Verifying the
supplied answer is what makes the detection metric meaningful.

The `backend` string records which path was taken (`"llama (provided)"` vs
`"llama"`), and the step payload carries `provided: True/False` and
`grounded: bool(evidence)`.

---

## Phase B — Detect (steps 3, 5, 8, 4, 9)

### ③ Early Detection

**File:** `src/steps/step03_early_detection.py:18`
**Config:** `step03_early_detection` — `threshold: 0.5`, `feature_keys: [cosine, claim_count, intrinsic_overlap, answer_length]`

A gradient-boosted classifier over four cheap features:

| Feature | How it is computed |
|---|---|
| `cosine` | cosine similarity of the query and answer embeddings (BGE-M3) |
| `claim_count` | `max(1, len(ctx.claims))` — see the defect below |
| `intrinsic_overlap` | `intrinsic_overlap(answer, query)` from `src/utils/claim_extractor.py:68` — Jaccard overlap of content tokens |
| `answer_length` | `len(answer.split())` — word count |

`intrinsic_overlap` is deliberately the *same helper* the training-data
generator uses (`scripts/generate_detector_data_v2.py`). An earlier version
computed it inline with a different formula (fraction of answer tokens present
in the query), so the served model was fed a feature it had never seen.

Output: `ctx.feature_vector` and
`ctx.metadata["hallucination_probability"]` — the latter is published so
step 9 can consume it as `detector_trust`.

### The two defects here

1. **`claim_count` is structurally constant.** Line 36 reads
   `len(self.ctx.claims)`, but step 5 (claim extraction) runs *after* step 3
   in the registry. `ctx.claims` is empty at this point, so
   `max(1, 0) = 1` always. One of the four features carries no information.
2. **The step-3 probability is nearly useless as a detector.** Measured on
   the 496-row fit: `hallucination_probability` has AUC 0.5389 read as a
   risk signal — barely above chance. `detector_trust` (its mirror) has
   0.4611, and the two sum to exactly 1.0000.

---

### ⑤ Claim Extraction

**File:** `src/steps/step05_claim_extraction.py:17`
**Helper:** `src/utils/claim_extractor.py:57`
**Config:** `step05_claim_extraction` — `mode: rule`, `intrinsic_threshold: 0.7`,
`aggregated_threshold: 0.3`

Splits the answer into atomic claims and tags each one's provenance.

**Three modes:**

| Mode | Behaviour |
|---|---|
| `rule` (default) | regex sentence split, strip leading `[-*•]`, drop sentences < 5 chars, dedupe case-insensitively |
| `llm` | ask the generator to atomise; parse one claim per line |
| `hybrid` | rule first; fall back to LLM only if fewer than 2 claims were produced |

> `hybrid` was tried and reverted. It raises granularity but drops claim F1
> to 0.23 and VSR to 0.22 — short answers fragment into pieces the NLI then
> rejects. The default stays `rule`.

**Provenance tagging** (lines 32-40) — each claim's lexical overlap with the
query decides its label:

```
overlap ≥ 0.7  →  intrinsic     (the claim is about what was asked)
overlap ≤ 0.3  →  extrinsic     (the claim came from elsewhere)
in between     →  aggregated    (the claim mixes both)
```

---

### ⑧ NLI Verification

**File:** `src/steps/step08_nli_verification.py:100`
**Config:** `step08_nli_verification` — `max_evidence_chars: 4000`,
`max_premise_chars: 1800`, `device: auto`

The core fact-checker. Model:
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.

For every (claim, evidence) pair the NLI returns a label and score. The best
verdict per claim is selected.

### Claim-conditioned premise selection

This is the most important detail in the step, and it was added to fix a
real failure.

The NLI truncates its premise to 512 tokens. When a retrieved document is
longer than that, only its *opening* reaches the model — so a claim the
document supports **later** comes back `"unknown"`. The evidence was there;
the model simply never saw it. On RAGTruth, whose contexts run to 9900
characters, this made **a third of all claims undecidable** and drove the
detection score to chance.

So the premise is not the whole document. `_claim_premise` (line 30) selects
the part of the document *about the claim*:

```
1. Split the document into sentences.
2. Score each sentence by token overlap with the claim.
3. Take the best-matching sentences, plus their immediate neighbours
   (so the NLI sees a contiguous passage).
4. Join them in original order, truncated to max_premise_chars (1800).
```

If the document is already shorter than `max_premise_chars`, it is returned
unchanged (line 40). If there is no sentence boundary, a window centred on
the claim's first matching token is used instead (`_window_around_match`,
line 81).

The same claim-conditioned idea is used by step 10 to pick a replacement, so
the verifier and the rewriter agree on what "the evidence for this claim" is.

### Verdict selection and confidence

Per claim (lines 129-138):

| Verdict | Confidence | Meaning |
|---|---|---|
| `supported` | `score` | the evidence entails the claim |
| `contradicted` | `1 - score` | the evidence refutes the claim |
| `unknown` | `0.3` (fixed) | undecidable — no evidence either way |

Selection is ordered: a `supported` verdict always wins; a `contradicted`
verdict only wins if nothing was supported.

Finally, each claim is attached to its best-matching evidence document
(`claim.evidence`, `claim.citation`, lines 141-154).

---

### ④ Confidence Assessment (EEDC)

**File:** `src/steps/step04_confidence_assessment.py:15`
**Config:** `step04_confidence` — `method: temperature`, `weights_path: data/eedc_weights.json`

**Evidence-Driven Confidence Calibration.** Three inputs:

| Input | Source |
|---|---|
| `nli_entropy` | Shannon entropy over `{supported, contradicted, unknown}` verdict labels, normalised by `log 3` to land in [0, 1] |
| `retrieval_top1` | cosine similarity (proxy) |
| `self_consistency` | cosine similarity (proxy) |

Entropy is high when the verifier is split, which is the signal that the
answer is unreliable. When there are no verdicts at all, it falls back to
`1 - cosine`.

**Calibration.** The raw EEDC score is passed through a calibrator:
`method: none | temperature | isotonic`. Calibrated output → `ctx.eedc_score`.

> **Honest limitation.** `self_consistency` is currently a proxy for cosine,
> not a real multi-sample self-consistency measurement (line 40 says so
> explicitly: *"use cosine as a proxy until multi-sample"*). And
> `retrieval_top1` in the EEDC call is also cosine — so the EEDC score is
> currently less independent than its three-part description suggests.

---

### ⑨ Confidence Fusion

**File:** `src/steps/step09_confidence_fusion.py:91`
**Config:** `step09_fusion_weights` — `detection_head` (fitted), `fusion` (legacy fallback), `grounding_source`

Fuses seven components into one **hallucination risk** score, then converts
it to a probability.

### The direction problem

This step produces a **risk** — higher means more likely hallucinated. But
`ctx.confidence` remains the *trust* score (`1 - risk`), because that is the
name the rest of the project uses.

The components do **not** all point the same way:

| Component | Natural direction | Sign |
|---|---|---|
| `similarity` (cosine) | rises with hallucination risk | `+1` (risk) |
| `retrieval_top1` | rises with hallucination risk | `+1` (risk) |
| `nli` (mean supported-claim confidence) | high = supported | `−1` (trust) |
| `coverage` (share of supported claims) | high = supported | `−1` (trust) |
| `eedc` | rises with hallucination risk | `+1` (risk) |
| `detector_trust` (`1 − hallucination_probability`) | high = faithful | `−1` (trust) |
| `grounding` | high = well-grounded | `−1` (trust) |

### Why the signs matter so much

> The previous version added every component with a positive weight into a
> score it treated as uniformly trust-oriented. On RAGTruth that fusion
> scored **ROC-AUC 0.360** as a detection score — worse than a coin flip —
> because its two largest components (`nli` and `coverage`) pointed in the
> opposite direction to the truth. — `step09_confidence_fusion.py:32-39`

The fix was to fit the head: a **logistic regression on standardised
components**, fitted on a tuning half of the RAGTruth matched split and
evaluated on a held-out half. The fitted weights and signs live in
`configs/default.yaml:117-161`.

### Standardisation

Components have very different scales (EEDC entropy ~0.3, retrieval ~0.4,
NLI confidence ~0.5), so a raw-weight sum would be dominated by whichever
component has the widest range. Each component is z-scored using the stored
`component_mean` / `component_std`:

```
zᵢ = (rawᵢ − meanᵢ) / stdᵢ
```

### The score

```
risk = Σᵢ  wᵢ · signᵢ · zᵢ
P(hallucination) = sigmoid(risk)          # clipped to [−30, 30] for overflow
ctx.confidence   = 1 − P(hallucination)
```

The intercept is not fitted — the components are centred, so 0 is the
decision boundary, which keeps the score comparable across datasets.

### The grounding term

`grounding_trust` (`src/utils/grounding.py:87`) measures the share of the
answer's *content words* the source does **not** contain, then inverts it.
Function words are excluded; duplicated content words are counted once.

Grounding is measured against the **top-1 retrieved document only**. Pooling
all five documents destroys the signal — the extra documents contribute their
own vocabulary, which absorbs the answer's genuinely novel words and pushes
every answer's novel rate toward the same value.

This is a *different question* from the NLI's. The NLI asks "is this claim
entailed by the evidence" (per-claim, hard on long evidence). Grounding asks
the weaker but much more robust "how much of this answer came from the
source at all" (per-answer). On RAGTruth, whose answers are 110-160 words of
long-form generation, the claim-level terms were nearly uninformative
(detection ROC-AUC 0.478) while grounding separates the classes on its own
(AUC 0.715 / 0.776 / 0.808 for Data2txt / QA / Summary).

### Two components that get *dropped*

**Grounding** is left out entirely when `grounding_source: "none"` — for a
dataset whose retrieval corpus holds the reference answers, grounding would
measure the label rather than a property of the answer.

**Evidence-dependent terms** (`retrieval_top1`, `nli`, `coverage`) are left
out when there is no evidence at all (lines 169-190). With no source the NLI
has no premise, so every claim returns `"unknown"` and those components
collapse to constant 0.0. They are constants, not measurements — leaving
them in would shift every closed-book row by the same amount (a pure
base-rate offset with no discrimination) and would look like the score had
read a source it never saw.

### Per-claim scores

After the fused score, each claim also gets its own trust score (lines
214-227) combining the NLI confidence, its evidence's retrieval score,
support coverage, cosine, EEDC and detector trust.

---

## Phase C — Mitigate & Decide (steps 10, 11, 12, 13)

### ⑩ Regeneration (mitigation)

**File:** `src/steps/step10_regeneration.py:182`
**Config:** `step10_regeneration` — `mode: claim_edit`, `max_attempts: 2`,
`min_contradiction_confidence: 0.5`

### Four modes

| Mode | Behaviour |
|---|---|
| `stub` | no-op — passes intentionally unsupported answers through |
| `evidence` | rebuild the answer from supported claims only; if none, substitute the first evidence document |
| `claim_edit` | **default** — rewrite only the flagged claims, leave every other span byte-identical |
| `regenerate` | full LLM rewrite conditioned on verified evidence only |

### Why `evidence` mode is not the default

`evidence` mode discards the original phrasing of every claim the verifier
did not *explicitly* mark as supported. On the length-matched HaluEval split
that behaviour turned an already-correct answer into a worse one — a BERTScore
of **1.000 fell to 0.811** — so the measured mitigation delta was negative.

### `claim_edit` in detail

The pipeline's headline architectural property is **per-claim granularity**:
the verifier identifies *which* claim is unsupported, and the rewriter acts at
exactly that granularity, so the precision the detector produced is not thrown
away at the last step.

```
flagged   = claims with verdict == "contradicted"
confident = flagged with contradiction_strength ≥ 0.5
```

`_contradiction_strength` (line 386) is `1 − claim.confidence`, because
`Claim.confidence` for a contradicted claim is already `1 − contradiction_prob`
— reading it directly would invert the guard and rewrite exactly the claims
the verifier was *least* sure about.

If nothing is confidently contradicted → `KEEP`. If claims were flagged but
none confidently → `ABSTAIN` with guard `no_confident_contradiction`.

For each confident claim, three things must pass:

**① Locate the span** (`_find_span`, line 55). The claim extractor strips and
re-joins sentences, so claim text rarely matches character for character.
Fast path: verbatim substring. Slow path: match on whitespace- and
case-normalised text, then map the offset back to the original
(`_orig_offset`, line 99) and extend to the next word boundary.

**② Choose the replacement** (`_replacement_for`, line 131). It uses the
evidence attached to *that specific claim* — not the query's best document,
which is what produced the uninformative substituted passages. And it uses
only the single best-matching **sentence** of that evidence, never the whole
passage: substituting a paragraph for a short claim turned a four-word answer
into a two-hundred-word passage about an unrelated subject.

**③ Pass both guards.**

| Guard | Test | Why it exists |
|---|---|---|
| `replacement_not_proportionate` | `repl_words ≤ max(12, claim_words × 4)` | a long replacement is a passage that does not answer the question. An earlier 40-word floor let a 25-word passage replace a 3-word claim, turning "Memphis International Airport" into a paragraph about Kumasi Airport. |
| `replacement_not_relevant` | the claim's capitalised subject entity must appear in the replacement (or ≥50% token overlap if it has no entity) | the NLI has no way to say "the evidence is irrelevant", only "the premise does not entail this". Token overlap over the whole claim is the wrong measure — a valid replacement often gives a different fact about the *same* subject ("Mary Ramsey is younger than Lee Ranaldo" → "Mary Ramsey was born in 1984"), sharing only 2 of 5 content tokens while being entirely on topic. |

**Splicing.** Flagged spans are replaced from the end backwards so earlier
offsets stay valid; overlapping spans are skipped rather than producing
malformed text (lines 357-370).

**Outcome.**

| Decision | When |
|---|---|
| `EDITED` | at least one flagged claim was replaced |
| `ABSTAIN` | a claim was flagged but nothing usable could replace it — the answer is left untouched |
| `KEEP` | nothing was confidently contradicted |

The original answer is never replaced wholesale. **Degrading a correct answer
is worse than leaving a wrong one in place**, and the gate already reports the
contradiction.

A per-claim guard is not an all-or-nothing veto: if one claim has no usable
replacement, that claim is skipped and the *others* are still rewritten
(lines 320-331). An earlier version abandoned the whole answer here, which
suppressed every good replacement in the same answer and left the rewrite
firing on only 16 of 500 rows.

### The provided-answer path

Lines 208-224: when the answer was supplied by a benchmark, step 10 records
`provided_passthrough` and returns the answer untouched.

**Why:** a provided answer is being *judged*, not improved. Rewriting it
would mean the gate and the detector score a different answer than the one
the benchmark supplied. The reduction action is still recorded so the HRR/CCR
figures stay meaningful.

> **Consequence worth knowing:** the mitigation code only actually runs
> through `scripts/evaluate_mitigation.py`, which deliberately calls
> `pipe.run(query)` with **no** answer so the generator runs.

---

### ⑪ Final Gate

**File:** `src/steps/step11_final_gate.py:15`
**Config:** `step11_final_gate` — `pass_threshold: 0.70`, `abstain_threshold: 0.30`,
`max_regen_attempts: 2`

```
unsupported_ratio = (claims contradicted + unknown) / total_claims

if   confidence ≥ 0.70  AND  unsupported_ratio == 0.0   →  PASS
elif confidence < 0.30  OR   unsupported_ratio ≥ 0.5    →  ABSTAIN
else                                                      →  REGEN
     (and if attempts exhausted: PASS if confidence ≥ 0.5, else ABSTAIN)
```

Note the asymmetry: **PASS requires both** a high confidence **and** zero
unsupported claims. A high confidence with any unsupported claim falls into
the gray zone and triggers a regeneration.

The decision and its inputs are appended to `ctx.decision_log`.

---

### ⑫ Explainable Response

**File:** `src/steps/step12_explainable_response.py:17`

Renders two artefacts.

**JSON** (`ctx.metadata["report_json"]`) — query, answer, confidence,
decision, and for each claim: text, verdict, confidence, provenance,
citation, and its full evidence list. Plus the retrieved evidence and the
decision history.

**Markdown** (`ctx.metadata["report_markdown"]`) — a human-readable report:

```markdown
# Hallucination Pipeline Report

**Query:** ...
**Answer:** ...
**Confidence:** 0.87
**Decision:** `PASS`

## Claims

1. *(supported, conf=0.94)* Mary Ramsey was born in 1984.
   - provenance: `intrinsic`
   - citation: `doc_0042`

## Evidence

- **doc_0042** (score=0.712): ...

## Decision history

- step 10: {...}
- step 11: {...}
```

---

### ⑬ Evaluation

**File:** `src/steps/step13_evaluation.py:246`
**Config:** `step13_eval.metrics` — the full catalogue

Per-row metrics, appended to `data/metrics.json`:

| Family | Metrics |
|---|---|
| Retrieval | Recall@5, Precision@5, MRR, nDCG |
| Verification | claim precision / recall / F1 / support rate, VSR |
| Detection | accuracy, precision, recall, F1, ROC-AUC |
| Generation | BLEU, ROUGE-L, BERTScore (external evaluator) |
| Reduction | HRR, CCR |

### Two correctness details

**Retrieval metrics are null when there are no gold IDs** (lines 261-275).
Gold evidence IDs come from `ctx.metadata["gold_evidence_ids"]`. Scoring the
retrieved IDs against themselves would always return 1.0 and look like a
perfect retriever — so the step reports `None` instead.

**Claim-level metrics distinguish two situations** (`_claim_level_metrics`,
line 160):

- **A gold answer exists** — a claim is a true positive when the verifier
  marks it supported *and* it is consistent with the gold answer. A claim
  marked supported but contradicting the gold is a **false positive** (the
  verifier was confidently wrong). Claims the verifier rejects are false
  negatives when the answer actually contains them.
  For a hallucinated row (`label == 1`) there is no correct claim set to
  recall, so **recall is reported as `None`** — the benchmark says the answer
  is wrong but not which of its claims are wrong.
- **No gold answer** — only the support rate is well defined; P/R/F1 are
  reported as `None` rather than as three copies of the same number.

> An earlier version set `claim_precision`, `claim_recall` and `claim_f1` all
> to the support rate. That is not a recall and not an F1.

---

## The regeneration loop

When step 11 returns `REGEN`, the orchestrator loops back. `src/pipeline.py:236-277`:

```python
reverify = {
    "early_detection", "claim_extraction", "nli_verification",
    "confidence_assessment", "confidence_fusion", "regeneration", "final_gate",
}
max_attempts = int((self.cfg.step(11) or {}).get("max_regen_attempts", 2))

# ... after step 11 runs:
if step.name == "final_gate" and ctx.decision == "REGEN" and attempt < max_attempts:
    attempt += 1
    ctx.metadata["regeneration_attempt"] = attempt
    index = 0
    while index < len(registry) and registry[index].name not in reverify:
        index += 1
    continue
```

```
  ┌──────────────┐
  │  ③ → ⑤ → ⑧ → ④ → ⑨ → ⑩ → ⑪  │
  └──────────────┬───────────────┘
                 │
          ⑪ says REGEN?  ──── no ───►  ⑫ → ⑬  →  DONE
                 │ yes
                 ▼
        attempt < max_attempts?
                 │ yes
                 └──► rewind to ③, re-verify the revised answer
```

**What is re-run:** steps 3, 5, 8, 4, 9, 10, 11. Steps 1, 6, 7 and 2 are
**not** re-run — the query was already understood, the corpus already
retrieved, and regenerating the answer from scratch is not what `claim_edit`
does. The revised answer *is* re-verified, which is the point: the gate must
not pass an answer it has not checked.

**Cap:** `max_regen_attempts: 2`.

**Step 11 tracks its own attempt count** via `ctx._regen_attempts`
(line 27) — separate from the orchestrator's `attempt` counter, because the
gate needs to know it has exhausted its budget even when it is reached from a
different path.

---

## Step reference table

| # | Module | Name | Model / method | Primary output | Key config |
|---|---|---|---|---|---|
| 1 | `step01_query_understanding.py` | query_understanding | spaCy + GLiNER + LLM | intent, entities, query_variants | `intent_labels` |
| 2 | `step02_grounded_generation.py` | grounded_generation | Qwen2.5-0.5B-Instruct | `ctx.answer` | `backend`, `max_new_tokens` |
| 3 | `step03_early_detection.py` | early_detection | XGBoost | `feature_vector`, `hallucination_probability` | `threshold`, `feature_keys` |
| 4 | `step04_confidence_assessment.py` | confidence_assessment | EEDC + temperature | `ctx.eedc_score` | `method`, `weights_path` |
| 5 | `step05_claim_extraction.py` | claim_extraction | rule / LLM | `ctx.claims` | `mode`, thresholds |
| 6 | `step06_hybrid_retrieval.py` | hybrid_retrieval | BGE-M3 + FAISS + BM25 + RRF | `ctx.evidence` | `top_k`, `rrf_k`, `retrieval_source` |
| 7 | `step07_reranking.py` | reranking | BGE-reranker-v2-m3 | re-scored evidence | `enabled`, `top_k` |
| 8 | `step08_nli_verification.py` | nli_verification | mDeBERTa-v3-xnli | `ctx.verdicts`, claim verdicts | `max_premise_chars` |
| 9 | `step09_confidence_fusion.py` | confidence_fusion | fitted logistic head | `ctx.confidence` | `detection_head`, `grounding_source` |
| 10 | `step10_regeneration.py` | regeneration | claim_edit | `ctx.final_answer` | `mode`, `min_contradiction_confidence` |
| 11 | `step11_final_gate.py` | final_gate | thresholds | `ctx.decision` | `pass_threshold`, `abstain_threshold` |
| 12 | `step12_explainable_response.py` | explainable_response | — | `report_markdown`, `report_json` | — |
| 13 | `step13_evaluation.py` | evaluation | metric catalogue | row metrics → `data/metrics.json` | `metrics` |

---

## Configuration knobs

All config lives in `configs/default.yaml`, loaded by `src/config.py:111`.
Any field can be overridden by an environment variable named
`HALLUC_<UPPER__DOTTED__KEY>` (`src/config.py:22-48`) — e.g.
`HALLUC_STEP06_RETRIEVAL__TOP_K=10`.

### Models

| Key | Default |
|---|---|
| `models.embedder` | `BAAI/bge-m3` |
| `models.generator` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `models.alt_generator` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `models.reranker` | `BAAI/bge-reranker-v2-m3` |
| `models.nli` | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| `models.gliner` | `knowledgator/gliner-bi-base-v1.0` |
| `models.spacy` | `en_core_web_sm` |
| `models.classifier` | `xgboost` (or `sklearn-logreg`) |

### CLI flags that change behaviour

| Script | Flag | Effect |
|---|---|---|
| `scripts/evaluate_full.py` | `--index` | load the FAISS+BM25 index (**always pass this**) |
| | `--retrieval-source {retrieved,none}` | closed-book mode |
| | `--grounding-source {retrieved,none}` | drop the grounding term |
| | `--resume` | continue an interrupted run |
| | `--recompute` | ignore cached rows |
| | `--dataset {halueval,truthfulqa}` | dataset-specific behaviour |
| `scripts/evaluate_mitigation.py` | *(no answer passed)* | the only path that actually exercises step 10 |
| `scripts/build_index.py` | `--chunk-size`, `--chunk-overlap` | corpus chunking |
| `scripts/fit_detection_head.py` | `--data`, `--limit`, `--stratify` | refit the step-9 head |

There is **no `--model` flag**. Models are chosen via `models.*` in the YAML
plus env vars (`HALLUC_EMBEDDER_DEVICE`, `HALLUC_NLI_DEVICE`,
`HALLUC_RERANKER_DEVICE`, `HALLUC_USE_OPENROUTER`).

---

## Known defects and honest caveats

These are real, currently live, and worth stating plainly rather than
discovering later.

### 1. Step 3's `claim_count` is always 1

`step03_early_detection.py:36` reads `len(self.ctx.claims)`, but step 5 runs
later in the registry. Fix: move step 3 after step 5, or have step 3 compute
its own claim count.

### 2. Step 3's probability is nearly uninformative on its own

AUC 0.5389 as a risk signal — barely above chance. It is retained as a
fusion *component* (`detector_trust`, weight 0.3228), not as a standalone
detector.

### 3. The EEDC inputs are partly proxies

`self_consistency` and `retrieval_top1` are both cosine similarity
(`step04:40, 45`). The three-part EEDC description overstates the current
independence of its inputs.

### 4. The mitigation path is not exercised by normal benchmark runs

When the benchmark supplies the answer, step 10 short-circuits to
`provided_passthrough`. Only `scripts/evaluate_mitigation.py` runs it for
real.

### 5. `hybrid` claim extraction is not the default for a reason

It raises granularity but drops claim F1 to 0.23 and VSR to 0.22 — short
answers fragment into pieces the NLI rejects.

### 6. The headline mitigation delta is negative

Published numbers still describe the pre-rewrite code. The `claim_edit` mode
is a design fix, not a measured win; the reported delta is around −0.007.

### 7. Step 13's `bleu` / `rouge_l` / `bert_score` are not computed at step level

They are produced by the external evaluator. Step 13 reports them under
`unavailable_at_step_level` rather than silently ignoring them.

### 8. The legacy hand-set fusion weights are still in the config

`step09_fusion_weights.fusion` is kept as a fallback for configs with no
fitted head. The fitted `detection_head` takes precedence when present
(`step09:97-98`).

### 9. The LLM query rewrite is computed and then discarded

`step01:66` persists `variants[0]` as `rewritten_query`, but `variants[0]` is
always the original query — the rewrite is appended after it. So every
downstream step reads the original query, and the LLM rewrite costs a
generator call per row for no effect. The current numbers are unaffected
(they were produced with the original query), but the feature is inert.

### 10. The rewritten query is also never used for retrieval

Even if the rewrite were persisted, `step06` retrieves on
`ctx.metadata["rewritten_query"]` — the same key — so the two would be
coupled. Worth knowing before changing either.

---

## Appendix — how to read a run trace

Every step writes a payload into `ctx.metadata` under the key
`step{NN}_{name}`, and the orchestrator pops it into a `StepTrace`
(`src/pipeline.py:263-265`). `PipelineResult.step_traces` therefore carries
the full per-step record:

```python
for t in result.step_traces:
    print(t.step, t.name, f"{t.duration_ms:.0f}ms", t.payload)
```

The payload keys worth checking after a run:

| Step | Key to inspect | What it tells you |
|---|---|---|
| 1 | `rewrite_method` | `llm`, `heuristic` or `identity` — and `query_variants` to see the actual rewrite |
| 6 | `warning` / `evidence_count` | `"no index loaded"` means the index was not passed |
| 7 | `scores` | the reranker's scores for the surviving passages |
| 2 | `provided`, `grounded` | whether the answer was supplied or generated |
| 3 | `features` | the four feature values |
| 8 | `verdict_distribution` | supported / contradicted / unknown counts |
| 9 | `grounding_used`, `components` | which terms were actually in the score |
| 10 | `decision`, `guard` | `EDITED`/`ABSTAIN`/`KEEP` and why |
| 11 | `decision`, `unsupported_ratio` | the gate's reasoning |
| 13 | `unavailable_at_step_level` | metrics the step could not produce |
