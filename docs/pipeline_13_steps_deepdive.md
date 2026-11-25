# CRISP — How Each of the 13 Steps Works Internally

A step-by-step walkthrough. Every step gets:
- a **diagram** of what happens inside it
- the **exact algorithm**, in code terms
- the **inputs → outputs**
- the **file:line** where it lives
- the **config knobs** that control it

> **Reading order.** Steps are presented in *execution* order, which is not
> their number order. The mapping is in §0.

---

## §0 — Execution order (numbers ≠ order)

```
STEP NUMBER :  1   2   3   4   5   6   7   8   9  10  11  12  13
EXEC  ORDER :  1 → 6 → 7 → 2 → 3 → 5 → 8 → 4 → 9 → 10 → 11 → 12 → 13
```

| Phase | Steps | Purpose |
|---|---|---|
| **A — Understand & Ground** | 1, 6, 7, 2 | turn a question into a grounded answer |
| **B — Detect** | 3, 5, 8, 4, 9 | decide how much to trust that answer |
| **C — Mitigate & Decide** | 10, 11, 12, 13 | fix what is wrong, then report |

**Why the reordering:**
- **6, 7 before 2** — retrieval must supply the generator's context. The
  original bug had generation first, so the answer was always ungrounded.
- **4 after 8** — EEDC's entropy term is computed over NLI verdicts, so the
  verdicts must exist first.
- **5 before 8** — the NLI verifies claims, so claims must exist first.

---

# PHASE A — UNDERSTAND & GROUND

---

## Step 1 — Query Understanding

**File:** `src/steps/step01_query_understanding.py:17`
**Config:** `step01_query_understanding` → `enabled: true`, `rewrite_variants: 2`,
`intent_labels: ["factual","opinion","multi-hop","math","other"]`

### Diagram

```
                         ┌─────────────────────────┐
                         │  ctx.query  (the text)  │
                         └────────────┬────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
     │  INTENT         │     │  ENTITIES       │     │  REWRITE        │
     │                 │     │                 │     │                 │
     │ _classify_intent│     │  spaCy NER     │     │  LLM call       │
     │ (pure regex)    │     │      +         │     │ "Return ONE     │
     │                 │     │  GLiNER NER    │     │  rewrite, no    │
     │  ↓              │     │      +         │     │  commentary"     │
     │ 5 labels        │     │  merge lists   │     │      ↓          │
     │                 │     │                 │     │  fallback:      │
     │                 │     │                 │     │  strip ? . !    │
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │  metadata payload       │
                         │   intent                │
                         │   entities[]            │
                         │   query_variants[]      │
                         │   rewrite_method        │
                         └────────────┬────────────┘
                                      ▼
                    metadata["rewritten_query"] = variants[0]
```

### Inside the intent classifier

`_classify_intent` (line 68) — pure string matching, no model, in this exact
order:

```python
if any(t in ql for t in ["+", "-", "*", "/", "equals", "compute"]):
    return "math"
if " and " in ql or " of " in ql and ql.count("?") > 1:
    return "multi-hop"
if ql.startswith(("do you think", "is it good", "should")):
    return "opinion"
if "?" in ql or any(ql.startswith(w) for w in ["who","what","when","where","why","how"]):
    return "factual"
return labels[-1] if labels else "other"
```

Note the first test matches the bare characters `+ - * /`, so a hyphenated
word like "state-of-the-art" would classify as `math`.

### Inside the NER merge

Two models run over the same query and their entity lists are **concatenated**
(lines 33-36):

- **spaCy** `en_core_web_sm` — standard named entities (PERSON, ORG, GPE…)
- **GLiNER** `knowledgator/gliner-bi-base-v1.0` — called with
  `labels=labels`, i.e. the *intent* labels are reused as the NER label set,
  so GLiNER is prompted toward the domain vocabulary.

There is no deduplication between the two lists.

### Inside the rewrite

```python
variants = [q]                                  # line 38 — original is ALWAYS first
rewrite_method = "identity"
if llm is not None:
    try:
        rewritten = llm.generate(
            system="You rewrite search queries to be more precise.",
            user=f"Original: {q}\n\nReturn ONE rewrite, no commentary.",
            max_new_tokens=64)
        if rewritten and rewritten.strip() and rewritten.strip() != q:
            variants.append(rewritten.strip())   # appended AFTER the original
            rewrite_method = "llm"
    except Exception:
        pass
if rewrite_method == "identity":
    r = re.sub(r"[?\.\!]+$", "", q).strip()
    if r and r != q:
        variants.append(r)
        rewrite_method = "heuristic"
```

Then line 66:

```python
self.ctx.metadata["rewritten_query"] = variants[0] if variants else q
```

**`variants[0]` is the original query.** The rewrite lives at
`variants[1]` and is never persisted. So `rewritten_query` — the key that
steps 3, 6, 7 and 2 all read — always holds the original query. The rewrite
is computed, appears in the step payload as `query_variants`, and is
discarded.

This is **inert, not wrong**: all published numbers were produced with the
original query. But it costs an LLM call per row for no effect.

### Outputs

| Field | Where |
|---|---|
| `intent` | step payload |
| `entities` | step payload |
| `query_variants` | step payload — `[original, rewrite]` |
| `rewrite_method` | step payload — `llm` / `heuristic` / `identity` |
| `rewritten_query` | `ctx.metadata` — **= the original query** |

---

## Step 6 — Hybrid Retrieval

**File:** `src/steps/step06_hybrid_retrieval.py:32`
**Fusion:** `src/retrieval/hybrid.py:25`
**Config:** `step06_retrieval` → `top_k: 5`, `rrf_k: 60.0`, `bm25_k1: 1.5`,
`bm25_b: 0.75`, `weight_dense: 1.0`, `weight_bm25: 1.0`, `index_path: "data/index"`,
`retrieval_source: "retrieved"`

### Diagram

```
      ctx.metadata["rewritten_query"]
                    │
                    ▼
        ┌───────────────────────┐
        │  embedder.encode(query)│   BAAI/bge-m3  →  qv (dense vector)
        └───────────┬───────────┘
                    │
      query text ───┼──────► ┌───────────────────────┐
                    │        │  BM25Index.search()   │  k1=1.5, b=0.75
                    │        │  top_k × 3 candidates  │
                    │        └───────────┬───────────┘
                    ▼                    ▼
        ┌───────────────────────┐        │
        │  FAISSIndex.search()  │        │
        │  top_k × 3 candidates │        │
        └───────────┬───────────┘        │
                    │                    │
                    ▼                    ▼
        ══════════════════════════════════════════
         RECIPROCAL RANK FUSION
         for each doc in each list:
             score += 1 / (rrf_k + rank + 1)
         rrf_k = 60
        ══════════════════════════════════════════
                    │
                    ▼
         sort by rrf_score DESC
                    │
                    ▼
         take top_k (5)
                    │
                    ▼
         ctx.evidence = [Evidence(doc_id, text, score)]
```

### Inside the RRF

```python
def search(self, query, query_vec, top_k=5):
    dense_hits  = self.faiss.search(query_vec, top_k=top_k * 3)   # 15
    sparse_hits = self.bm25.search(query,   top_k=top_k * 3)     # 15

    fused = {}
    for rank, ev in enumerate(dense_hits):
        fused[ev.doc_id] = fused.get(ev.doc_id, 0.0) + 1.0/(self.rrf_k + rank + 1)
    for rank, ev in enumerate(sparse_hits):
        fused[ev.doc_id] = fused.get(ev.doc_id, 0.0) + 1.0/(self.rrf_k + rank + 1)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [Evidence(doc_id=d, text=..., score=s) for d, s in ordered[:top_k]]
```

Key properties:
- Both retrievers pull **3× top_k** candidates, so a document that only one
  retriever found still has a chance to rank.
- RRF uses **rank only, never the raw scores** — that is what makes it work
  across a cosine score (0-1) and a BM25 score (unbounded).
- `rrf_k = 60` is the standard damping constant that stops rank-1 from
  dominating.
- `weight_dense` and `weight_bm25` are in the config but the fusion code
  multiplies both by `1.0` implicitly — **the weights are not actually
  applied** in `hybrid.py`. Changing them has no effect.

### The closed-book branch

```python
retrieval_src = blk.get("retrieval_source", "retrieved")
override = self.ctx.metadata.get("retrieval_source")
if override is not None:
    retrieval_src = override
if retrieval_src == "none":
    self.ctx.evidence = []
    ... "warning": "closed-book dataset: retrieval disabled so the
                     verification chain cannot read the answer key"
    return
```

**Why it exists.** TruthfulQA's corpus was built from the dataset's own
`knowledge` field — the set of *correct* answers. Retrieval finds the
reference answer for every row, so any verification against the retrieved
document measures "is this answer the reference answer", which **is the
label**. Disabling retrieval means the verification chain has no source and
cannot read the answer key.

### The two failure modes

**No index loaded** → returns empty with `"warning": "no index loaded"`.
This is silent in the results. Historically the pipeline fell back to a
6-document synthetic corpus here, which is the root cause of the 0.50
baseline. **Always pass `--index`.**

**Embedder raises** → `qv = np.zeros(embedder.dim)` (line 74). The search
still runs, against a zero vector, and returns nonsense rather than failing.

### Outputs

| Field | Value |
|---|---|
| `ctx.evidence` | list of `Evidence(doc_id, text, score)` — the score is the RRF score |
| step payload | `top_k`, `evidence_count`, `evidence_ids` |

---

## Step 7 — Reranking

**File:** `src/steps/step07_reranking.py:11`
**Config:** `step07_reranking` → `enabled: true`, `top_k: 5`
**Model:** `BAAI/bge-reranker-v2-m3` (cross-encoder)

### Diagram

```
      ctx.evidence  (5 passages from step 6)
                    │
                    ▼
        ┌───────────────────────────────────┐
        │   reranker.rerank(query, texts)   │
        │                                   │
        │   cross-encoder scores each       │
        │   (query, passage) PAIR jointly   │
        │                                   │
        │   score_i = f(query, passage_i)    │
        └────────────────┬──────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  for e, s in zip(evidence, scores):│
        │      e.score = float(s)           │  ← OVERWRITES the RRF score
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  evidence.sort(key=lambda e: -e.score)│
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  ctx.evidence = evidence[:top_k]  │  (5)
        └───────────────────────────────────┘
```

### Why this is not just "sort again"

Step 6's dense retriever is a **bi-encoder**: query and passage are embedded
*independently*, then compared by cosine. There is no interaction between
them. Step 7 uses a **cross-encoder**: the query and the passage are fed
together into one model that reads both at once. That is strictly more
accurate but more expensive — one forward pass per candidate instead of one
embedding per document up front.

### Important details

- The reranker score **replaces** the RRF score entirely. `evidence[0].score`
  after step 7 is the cross-encoder score, not the fusion score — and step 9
  uses `evidence[0].score` as its `retrieval_top1` component.
- Truncation to `top_k` happens **after** the sort, so if `top_k` were
  smaller than the retrieved count, the lowest-scoring passages are dropped.
- `enabled: false` → step 6's RRF order is kept unchanged.
- If the reranker is missing or there is no evidence → the step returns
  without touching the evidence.

### Outputs

`ctx.evidence` — same list, new scores, new order. Step payload:
`top_k`, `evidence_count`, `scores`.

---

## Step 2 — Grounded Generation

**File:** `src/steps/step02_grounded_generation.py:21`
**Config:** `step02_grounded_generation` → `backend: "llama"`, `max_new_tokens: 256`,
`temperature: 0.0`, `dtype: "fp16"`, `device: "auto"`
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`

### Diagram

```
              ctx.evidence (5 passages)
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  ctx_text = "\n\n".join(e.text ...)   │
        └───────────────────┬───────────────────┘
                            │
      ctx.query ────────────┤
                            ▼
        ┌───────────────────────────────────────┐
        │  prompt = cfg.prompt                  │
        │                                       │
        │  SYSTEM: "You are a precise assistant.│
        │   Answer the user's question using   │
        │   ONLY the information in the         │
        │   provided context. If the context    │
        │   does not contain the answer, say    │
        │   "I don't know." Do not invent facts."│
        │                                       │
        │  USER: Context: {context}            │
        │         Question: {question}          │
        │         Answer:                      │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ╔═══════════════════════════════════════════╗
        ║   provided_answer present?               ║
        ╟───────────────────────────────────────────╢
        ║ YES → answer = provided_answer           ║
        ║       backend = "llama (provided)"       ║
        ║       ★ GENERATOR NEVER LOADED           ║
        ╟───────────────────────────────────────────╢
        ║ NO, gen is None →                        ║
        ║   "I don't have enough context to answer."║
        ╟───────────────────────────────────────────╢
        ║ NO → answer = gen.generate(system, user)  ║
        ╚═══════════════════════════════════════════╝
                            │
                            ▼
                    ctx.answer
```

### The provided-answer path — why it is load-bearing

A hallucination-detection benchmark pairs a **faithful** answer with a
**hallucinated** one for the same question. If the pipeline generated its own
answer from the query alone, both rows would receive:

- the same answer text,
- the same `cosine`, `claim_count`, `overlap`, `answer_length` features,
- the same claims, the same verdicts, the same confidence,

so the ROC-AUC would be **exactly 0.50 — by arithmetic, not by chance**.

Verifying the supplied answer is what makes the detection metric meaningful.
This is why `scripts/evaluate_full.py` passes `answer=` and why
`scripts/evaluate_mitigation.py` deliberately does not.

### Prompt construction detail

`ctx.metadata.get("rewritten_query", self.ctx.query)` — it reads the
rewritten-query key, which (per Step 1) holds the original query. The
evidence is joined with a blank line between passages.

### Outputs

| Field | Value |
|---|---|
| `ctx.answer` | the answer text |
| step payload | `backend` (`"llama"` / `"llama (provided)"`), `answer`, `evidence_count`, `grounded` (`bool(evidence)`), `provided` |

---

# PHASE B — DETECT

---

## Step 3 — Early Detection

**File:** `src/steps/step03_early_detection.py:18`
**Config:** `step03_early_detection` → `threshold: 0.5`,
`feature_keys: ["cosine","claim_count","intrinsic_overlap","answer_length"]`
**Model:** XGBoost classifier

### Diagram

```
        ctx.answer  ──────────┐
        ctx.query  ──────────┤
                              ▼
      ┌───────────────────────────────────────────────┐
      │  FEATURE 1: cosine                            │
      │    qv = embedder.encode([query])[0]           │
      │    av = embedder.encode([answer])[0]          │
      │    cosine = dot(qv,av) / (|qv|·|av| + 1e-12)  │
      ├───────────────────────────────────────────────┤
      │  FEATURE 2: claim_count                       │
      │    max(1, len(ctx.claims))                    │ ← ALWAYS 1
      │    ⚠ step 5 (claims) runs AFTER this step     │
      ├───────────────────────────────────────────────┤
      │  FEATURE 3: intrinsic_overlap                 │
      │    intrinsic_overlap(answer, query)            │ = |Q∩C| / |Q∪C|
      │    (shared helper — same formula as training)  │
      ├───────────────────────────────────────────────┤
      │  FEATURE 4: answer_length                     │
      │    len(answer.split())                        │
      └───────────────────┬───────────────────────────┘
                          ▼
                  features = [cosine, claim_count, overlap, len]
                          │
                          ▼
              ┌───────────────────────────────┐
              │  clf.predict_proba(features)  │   XGBoost
              └───────────────┬───────────────┘
                              ▼
          ctx.feature_vector  +  metadata["hallucination_probability"]
```

### The feature computation, in code

```python
qv = embedder.encode([query])[0]
av = embedder.encode([answer])[0]
cosine = float(np.dot(qv, av) / (np.linalg.norm(qv)*np.linalg.norm(av) + 1e-12))

claim_count = max(1, len(self.ctx.claims))
overlap = intrinsic_overlap(answer, query)     # shared helper
ans_len = len(answer.split())

features = [cosine, float(claim_count), overlap, float(ans_len)]
prob = float(clf.predict_proba(features))      # 0.5 on any exception
```

### The `intrinsic_overlap` helper

`src/utils/claim_extractor.py:68`:

```python
def intrinsic_overlap(claim, query):
    qs = set(re.findall(r"\w+", query.lower()))
    cs = set(re.findall(r"\w+", claim.lower()))
    if not qs or not cs: return 0.0
    return len(qs & cs) / len(qs | cs)
```

A **Jaccard** similarity over content tokens. It is deliberately the *same
helper* the training-data generator uses (`scripts/generate_detector_data_v2.py`).
An earlier version computed the overlap inline with a different formula (the
fraction of answer tokens present in the query), so the served model was fed
a feature it had never seen.

### The `claim_count` defect

`ctx.claims` is empty when step 3 runs, because step 5 populates it later in
the registry. So `claim_count = max(1, 0) = 1` on **every single row**. One
of the four features carries zero information.

### How strong is the detector on its own?

Measured on the 496-row refit: `hallucination_probability` has **AUC 0.5389**
read as a risk signal — barely above the 0.5 coin flip. Its mirror
`detector_trust = 1 − prob` has 0.4611, and the two sum to exactly 1.0000.

So the XGBoost model is retained as *one component* of the step-9 fusion
(weight 0.3228), not as a standalone detector.

### Outputs

| Field | Value |
|---|---|
| `ctx.feature_vector` | `{"cosine","claim_count","intrinsic_overlap","answer_length"}` |
| `metadata["hallucination_probability"]` | float in [0,1] — consumed by step 9 as `detector_trust` |

---

## Step 5 — Claim Extraction

**File:** `src/steps/step05_claim_extraction.py:17`
**Helper:** `src/utils/claim_extractor.py:57`
**Config:** `step05_claim_extraction` → `mode: "rule"`, `intrinsic_threshold: 0.7`,
`aggregated_threshold: 0.3`

### Diagram

```
                    ctx.answer
                        │
                        ▼
        ┌───────────────────────────────────────────┐
        │  mode == "rule"?                          │
        │                                           │
        │  split_sentences(answer)                  │  regex: (?<=[.!?])\s+(?=[A-Z])
        │      ↓                                    │
        │  strip leading [ -* •]                    │
        │      ↓                                    │
        │  drop sentences shorter than 5 chars      │
        │      ↓                                    │
        │  dedup (case-insensitive, whitespace-norm)│
        └───────────────────┬───────────────────────┘
                            │  list of claim strings
                            ▼
        ┌───────────────────────────────────────────┐
        │  PROVENANCE TAGGING (per claim)           │
        │                                           │
        │  ov = intrinsic_overlap(claim, query)     │
        │                                           │
        │  ov ≥ 0.7  →  "intrinsic"                 │  about what was asked
        │  ov ≤ 0.3  →  "extrinsic"                 │  came from elsewhere
        │  else     →  "aggregated"                 │  mixes both
        └───────────────────┬───────────────────────┘
                            ▼
                 ctx.claims = [Claim(text, provenance, overlap)]
```

### The three modes

```python
def extract_claims(text, mode="rule", llm_generate=None):
    if mode == "llm" and llm_generate is not None:
        return extract_claims_llm(text, llm_generate)
    if mode == "hybrid" and llm_generate is not None:
        base = extract_claims_rule(text)
        if len(base) >= 2: return base
        return extract_claims_llm(text, llm_generate)
    return extract_claims_rule(text)
```

| Mode | Behaviour |
|---|---|
| `rule` (default) | regex sentence split + clean + dedup |
| `llm` | ask the generator to atomise; parse one claim per line, strip `[-*\d.)]` prefixes, dedup |
| `hybrid` | rule first; fall back to LLM only if fewer than 2 claims produced |

**`hybrid` was tried and reverted.** It raises granularity but drops claim F1
to 0.23 and VSR to 0.22 — short answers fragment into pieces the NLI then
rejects as `unknown`. The default stays `rule`.

### Inside the rule splitter

`src/utils/chunker.py:7`:

```python
parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
```

The split needs a terminator **followed by whitespace and a capital letter**.
This means:

- `"...the end. next sentence"` → **not split** (lowercase `n`)
- `"...the end.Next"` → **not split** (no whitespace)
- `"...the end. Next"` → split ✓

### Inside the sentence splitter's blind spot

Sentences ending in a quotation mark, or starting with a lowercase word that
continues a sentence, are not separated. A single long sentence therefore
becomes a single "claim" — and the NLI then has to judge a whole paragraph
as one hypothesis, which it does poorly.

### Dedup

```python
def _dedup(claims):
    seen = set(); out = []
    for c in claims:
        key = re.sub(r"\s+", " ", c.lower()).strip()
        if key and key not in seen:
            seen.add(key); out.append(c)
    return out
```

Key is lowercased and whitespace-normalised, but the **original** casing of
the first occurrence is kept.

### Outputs

`ctx.claims` — list of `Claim(text, provenance, intrinsic_overlap)`. Step
payload: `mode`, `claim_count`, `claims[]`, `provenances[]`.

---

## Step 8 — NLI Verification

**File:** `src/steps/step08_nli_verification.py:100`
**Helper:** `_claim_premise` at line 30
**Config:** `step08_nli_verification` → `max_evidence_chars: 4000`,
`max_premise_chars: 1800`, `device: "auto"`
**Model:** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

This is the heart of the system. It answers, for each claim: *does the
evidence support this?*

### Diagram

```
   ctx.claims (say 3)          ctx.evidence (say 5)
        │                              │
        └──────────┬───────────────────┘
                   │
        ┌──────────▼───────────────────────────────────────┐
        │  FOR EACH claim:                                 │
        │  ┌───────────────────────────────────────────┐   │
        │  │  FOR EACH evidence:                       │   │
        │  │                                           │   │
        │  │  premise = _claim_premise(ev.text,        │   │
        │  │                     claim.text, 1800)       │   │
        │  │      ↓                                    │   │
        │  │  ┌────────────────────────────────────┐  │   │
        │  │  │  CLAIM-CONDITIONED PREMISE SELECT  │  │   │
        │  │  │                                    │  │   │
        │  │  │  if len(ev) ≤ 1800: use whole doc  │  │   │
        │  │  │  else:                             │  │   │
        │  │  │   1. split into sentences          │  │   │
        │  │  │   2. score each by token overlap   │  │   │
        │  │  │      with the claim                │  │   │
        │  │  │   3. take best + its neighbours    │  │   │
        │  │  │   4. join in original order        │  │   │
        │  │  │   5. truncate to 1800 chars        │  │   │
        │  │  └────────────────────────────────────┘  │   │
        │  │      ↓                                    │   │
        │  │  label, score = NLI(premise, claim)      │   │
        │  │      ↓                                    │   │
        │  │  ┌──────────────────────────────┐        │   │
        │  │  │ DeBERTa-v3-MNLI              │        │   │
        │  │  │  entail → "supported"         │        │   │
        │  │  │  contradict → "contradicted"  │        │   │
        │  │  │  neutral → "unknown"          │        │   │
        │  │  └──────────────────────────────┘        │   │
        │  └───────────────────────────────────────────┘   │
        │      ↓ collect all verdicts                    │
        └──────────┬───────────────────────────────────────┘
                   │
                   ▼
        ┌───────────────────────────────────────────────┐
        │  PICK BEST VERDICT PER CLAIM                  │
        │                                               │
        │  "supported"    always wins                   │
        │  "contradicted" wins only if nothing supported │
        │  otherwise          "unknown"                 │
        └───────────────────┬───────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────────┐
        │  CONFIDENCE MAPPING                           │
        │    supported    → conf = score                │
        │    contradicted → conf = 1 - score            │
        │    unknown      → conf = 0.3  (fixed)         │
        └───────────────────┬───────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────────┐
        │  ATTACH BEST EVIDENCE TO EACH CLAIM           │
        │    claim.evidence = [best_ev]                 │
        │    claim.citation = best_ev.doc_id            │
        └───────────────────────────────────────────────┘
```

### Inside the claim-conditioned premise selection

**The problem it solves.** The NLI truncates its premise to 512 tokens. When
a retrieved document is longer, only its *opening* reaches the model — so a
claim the document supports **later** comes back `"unknown"`. The evidence
was there; the model never saw it. On RAGTruth, whose contexts run to 9900
characters, this made **a third of all claims undecidable** and drove
detection to chance.

**The fix** (`_claim_premise`, line 30):

```python
if len(evidence_text) <= max_chars:
    return evidence_text          # short enough — no selection needed

sentences = split_sentences(evidence_text)
if not sentences:
    return _window_around_match(evidence_text, claim_text, max_chars)

claim_tokens = {t for t in re.findall(r"\w+", claim_text.lower())}

scored = []
for i, s in enumerate(sentences):
    tokens = set(re.findall(r"\w+", s.lower()))
    overlap = len(claim_tokens & tokens)
    scored.append((overlap, -i, i, s))
scored.sort(reverse=True)

chosen = set(); budget = max_chars
for overlap, _, i, s in scored:
    if overlap == 0 and chosen: break        # stop at the first zero-overlap sentence
    for j in (i-1, i, i+1):                   # take the sentence AND its neighbours
        if 0 <= j < len(sentences) and j not in chosen:
            chosen.add(j); budget -= len(sentences[j])
    if budget <= 0: break

picked = [sentences[i] for i in sorted(chosen)]
return " ".join(picked)[:max_chars]
```

Note the neighbour padding: the best-matching sentence is included **with
its immediate neighbours**, so the NLI sees a coherent passage rather than a
fragment. The selection stops at the first sentence with zero overlap (once
something has been chosen), so it takes a contiguous relevant region rather
than scattered matches.

`_window_around_match` (line 81) is the fallback when the document has no
sentence boundary: it centres a window on where the claim's first token
appears.

**Why it matters beyond step 8:** step 10 uses the same idea to choose a
replacement. The verifier and the rewriter therefore agree on what "the
evidence for this claim" is.

### Inside the verdict selection

```python
for ev in evidence:
    lbl, score = nli.predict(premise=premise, hypothesis=claim.text)
    verdicts.append(ClaimVerdict(claim, ev.doc_id, lbl, score))
    if lbl == "supported" and score > best_score:
        best_label, best_score, best_doc = lbl, score, ev.doc_id
    elif lbl == "contradicted" and best_label != "supported" and score > best_score:
        best_label, best_score, best_doc = lbl, score, ev.doc_id
```

**Supported always beats contradicted**, regardless of score. A contradicted
verdict only wins when nothing was supported.

### Inside the NLI model wrapper

`src/models/nli_deberta.py:54`:

```python
inputs = self._tok(premise, hypothesis, return_tensors="pt",
                   truncation=True, max_length=512).to(self.device)
with torch.no_grad():
    logits = self._model(**inputs).logits
probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
idx = int(probs.argmax())
label = self._model.config.id2label[idx].lower()
if "entail" in label:      label = "supported"
elif "contradict" in label: label = "contradicted"
else:                       label = "unknown"
return label, float(probs[idx])
```

The label mapping is **substring-based** on the model's own label names, so it
survives changes to the label set. Note `truncation=True, max_length=512` —
this is exactly the truncation that motivated claim-conditioned selection.

fp16 is applied only on CUDA (line 45): on CPU, fp16 matmul is slow and
poorly supported.

### The cost

**claims × evidence** NLI forward passes per query. With 3 claims and 5
passages that is 15 calls. This is the dominant cost of the detection path
and the reason for the 0.8–13.8 s latency band.

### Outputs

| Field | Value |
|---|---|
| `ctx.verdicts` | every (claim, evidence, label, score) tuple |
| `ctx.claims[i].verdict` | `supported` / `contradicted` / `unknown` |
| `ctx.claims[i].confidence` | score, 1−score, or 0.3 |
| `ctx.claims[i].evidence` | the single best evidence passage |
| `ctx.claims[i].citation` | that passage's `doc_id` |
| step payload | `claim_count`, `verdict_count`, `verdict_distribution` |

---

## Step 4 — Confidence Assessment (EEDC)

**File:** `src/steps/step04_confidence_assessment.py:15`
**Config:** `step04_confidence` → `method: "temperature"`, `weights_path: "data/eedc_weights.json"`

### Diagram

```
   ctx.verdicts (step 8)
          │
          ▼
   ┌────────────────────────────────────────────┐
   │  NLI ENTROPY                                │
   │                                            │
   │  counts = {supported, contradicted, unknown}│
   │  H = -Σ (c/total) · log(c/total)            │
   │  H = H / log(3)          → normalised [0,1]│
   │                                            │
   │  (if no verdicts: H = 1 - cosine)          │
   └──────────────────┬─────────────────────────┘
                      │
   ctx.feature_vector │  cosine
          │           │
          ▼           ▼
   ┌────────────────────────────────────────────┐
   │  retrieval_top1  = cosine      (proxy)     │
   │  self_consistency = cosine      (proxy)    │
   └──────────────────┬─────────────────────────┘
                      ▼
   ┌────────────────────────────────────────────┐
   │  EEDC.score(nli_entropy=H,                 │
   │             retrieval_top1=cosine,         │
   │             self_consistency=cosine)       │
   │                                            │
   │        weighted combination → raw score    │
   └──────────────────┬─────────────────────────┘
                      ▼
   ┌────────────────────────────────────────────┐
   │  CALIBRATION                               │
   │    none         → passthrough               │
   │    temperature  → learned temperature       │
   │    isotonic     → learned isotonic          │
   └──────────────────┬─────────────────────────┘
                      ▼
                ctx.eedc_score
```

### Inside the entropy

```python
counts = {"supported": 0, "contradicted": 0, "unknown": 0}
for lbl in labels: counts[lbl] += 1
total = sum(counts.values()) or 1
H = 0.0
for c in counts.values():
    p = c / total
    if p > 0: H -= p * math.log(p)
H = H / math.log(3)     # normalise to [0,1]
```

**Reading the value:**

| Entropy | Verdicts look like | Meaning |
|---|---|---|
| 0.0 | all one kind | the verifier was unanimous — very confident |
| 1.0 | evenly split three ways | the verifier had no idea |
| 0.63 | 50/50 between two kinds | genuinely split |

Normalising by `log 3` is what makes 1.0 the maximum, so the component is on
the same 0-1 scale as the others.

### The honest limitation

`self_consistency` and `retrieval_top1` are **both just cosine similarity**
(lines 40 and 45):

```python
self_consistency = cosine  # use cosine as a proxy until multi-sample
...
raw = eedc.score(nli_entropy=H, retrieval_top1=cosine, self_consistency=cosine)
```

The step's three-part description overstates the independence of its
inputs. Only the entropy term is genuinely new information from the NLI.
Real self-consistency would require sampling the generator multiple times
and measuring agreement.

### Calibration

```python
calibrated = raw
if calibrator is not None and hasattr(calibrator, "transform"):
    calibrated = float(np.asarray(calibrator.transform(np.array([raw])))[0])
```

Any failure → fall back to the raw score.

### Outputs

`ctx.eedc_score`. Step payload: `eedc_raw`, `eedc_calibrated`, `nli_entropy`,
`self_consistency`.

---

## Step 9 — Confidence Fusion

**File:** `src/steps/step09_confidence_fusion.py:91`
**Config:** `step09_fusion_weights` → `detection_head` (fitted), `fusion` (legacy), `grounding_source`

This is where everything becomes one number.

### Diagram

```
  ctx.feature_vector   ctx.claims   ctx.evidence   ctx.eedc_score
        cosine           verdicts     top-1 score      H-calibrated
           │                │            │               │
           ▼                ▼            ▼               ▼
   ┌──────────────────────────────────────────────────────────┐
   │  SEVEN RAW COMPONENTS                                     │
   │                                                          │
   │  similarity      = cosine                                │
   │  retrieval_top1  = evidence[0].score                     │
   │  nli             = mean(confidence where verdict==      │
   │                              "supported")                 │
   │  coverage        = (count supported) / (total claims)    │
   │  eedc            = ctx.eedc_score                        │
   │  detector_trust  = 1 − hallucination_probability          │
   │  grounding       = grounding_trust(answer, [top1.text])  │
   └──────────────────────────┬───────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  STANDARDISE  zᵢ = (rawᵢ − meanᵢ) / stdᵢ                │
   │  means/stds stored in configs/default.yaml               │
   └──────────────────────────┬───────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  APPLY SIGNS                                             │
   │    risk  components (+1): similarity, retrieval_top1,    │
   │                         eedc                             │
   │    trust components (−1): nli, coverage, detector_trust,  │
   │                         grounding                        │
   │                                                          │
   │  ▸ trust components are NEGATED, so a bigger contribution │
   │    to risk ALWAYS means "more likely hallucinated"        │
   └──────────────────────────┬───────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  risk = Σ  wᵢ · signᵢ · zᵢ                               │
   │                                                          │
   │  ⚠ DROP components that are undefined:                   │
   │    grounding      → if grounding_source == "none"        │
   │    retrieval_top1 → if no evidence                       │
   │    nli            → if no evidence                       │
   │    coverage       → if no evidence                       │
   └──────────────────────────┬───────────────────────────────┘
                              │
                              ▼
              P(hallucination) = sigmoid(risk)
                              │
                              ▼
              ctx.confidence = 1 − P(hallucination)
```

### The direction problem — and why the signs matter

The step produces a **risk** (higher = more likely hallucinated), but
`ctx.confidence` stays the **trust** score (`1 − risk`) because that is the
name the rest of the project uses.

| Component | Natural direction | Sign | Fitted weight |
|---|---|---|---|
| `similarity` | rises with risk | `+1` | 0.5711 |
| `retrieval_top1` | rises with risk | `+1` | 0.1188 |
| `nli` | high = supported | `−1` | 1.2497 |
| `coverage` | high = supported | `−1` | −1.0345 |
| `eedc` | rises with risk | `+1` | −0.0749 |
| `detector_trust` | high = faithful | `−1` | 0.3228 |
| `grounding` | high = well-grounded | `−1` | 0.5887 |

> The previous version added every component with a positive weight into a
> score it treated as uniformly trust-oriented. On RAGTruth that fusion
> scored **ROC-AUC 0.360** as a detection score — worse than a coin flip —
> because its two largest components (`nli` and `coverage`) pointed in the
> opposite direction to the truth. — `step09:32-39`

The fix was to **fit** the head: a logistic regression on standardised
components, fitted on a tuning half of the RAGTruth matched split and
evaluated on a held-out half. Weights and signs live in
`configs/default.yaml:117-161`.

### Inside the standardisation

Components have very different scales — EEDC entropy ~0.3, retrieval ~0.4,
NLI confidence ~0.5 — so a raw-weight sum would be dominated by whichever
component happens to have the widest range. Each is z-scored with the stored
mean and std:

```
similarity      mean 0.6186  std 0.1398
retrieval_top1  mean 0.6016  std 0.3654
nli             mean 0.5428  std 0.2672
coverage        mean 0.6095  std 0.2737
eedc            mean 0.4609  std 0.0522
detector_trust  mean 0.8103  std 0.2300
grounding       mean 0.4921  std 0.2662
```

### The grounding component

`src/utils/grounding.py:87`:

```python
def grounding_trust(answer, evidence_texts):
    return 1.0 - novel_content_rate(answer, evidence_texts)

def novel_content_rate(answer, evidence_texts):
    tokens = set(content_tokens(answer))       # content words only, deduped
    source = set()
    for t in evidence_texts: source.update(content_tokens(t))
    if not source: return 0.0
    return len(tokens - source) / len(tokens)
```

**It measures the share of the answer's content words the source does NOT
contain.** A grounded answer reuses the source's vocabulary; a hallucinated
answer introduces facts the source never had, and those facts come with
words the source never used.

**Function words are excluded** — most of an English answer is function
words that appear in any source, so keeping them would inflate every score.

**It is measured against the top-1 document only.** Pooling all five
retrieved documents destroys the signal: the extra documents contribute their
own vocabulary, which absorbs the answer's genuinely novel words and pushes
every answer's novel rate toward the same value.

**Why it belongs in the fusion.** It asks a *different question* from the
NLI. The NLI asks "is this claim entailed by the evidence" — per-claim, hard
when the evidence is long. Grounding asks "how much of this answer came from
the source at all" — per-answer, and much more robust. On RAGTruth, whose
answers are 110-160 words of long-form generation, the claim-level terms
were nearly uninformative (detection ROC-AUC 0.478) while grounding
separates the classes on its own (AUC 0.715 / 0.776 / 0.808 for Data2txt /
QA / Summary).

### The two drop rules

**Grounding** is dropped when `grounding_source == "none"` (lines 182-186).
For a dataset whose retrieval corpus holds the reference answers, grounding
would measure the label rather than a property of the answer.

**Evidence-dependent terms** (`retrieval_top1`, `nli`, `coverage`) are dropped
when there is no evidence at all (lines 169-190). With no source the NLI has
no premise, so every claim returns `"unknown"` and those components collapse
to constant 0.0. They are constants, not measurements — leaving them in
would shift every closed-book row by the same amount (a pure base-rate
offset with no discrimination) and would look like the score had read a
source it never saw.

### The final computation

```python
risk = sum(parts) if parts else 0.0
prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, risk))))
confidence = 1.0 - prob
```

The sigmoid input is clipped to [−30, 30] to avoid overflow. **The intercept
is not fitted** — the components are centred, so 0 is the decision boundary,
which keeps the score comparable across datasets.

### Per-claim scores

After the fused score, each claim also gets its own trust score combining NLI
confidence, its evidence's retrieval score, support coverage, cosine, EEDC
and detector trust.

### Outputs

| Field | Value |
|---|---|
| `ctx.confidence` | `1 − P(hallucination)`, in [0,1] |
| `metadata["hallucination_probability"]` | `P(hallucination)` — **overwrites** step 3's value with the fused one |
| step payload | `weights` (used), `components` (raw), `grounding_used`, `hallucination_probability`, `hallucination_risk`, `overall_confidence`, `per_claim_confidence` |

---

# PHASE C — MITIGATE & DECIDE

---

## Step 10 — Regeneration (mitigation)

**File:** `src/steps/step10_regeneration.py:182`
**Config:** `step10_regeneration` → `mode: "claim_edit"`, `max_attempts: 2`,
`min_contradiction_confidence: 0.5`

### Diagram

```
        ctx.claims, ctx.answer, ctx.evidence
                     │
                     ▼
        ┌──────────────────────────────────────────┐
        │  provided_answer present?                │
        │  → mode="provided_passthrough"          │
        │  → final_answer = answer (unchanged)    │
        │  → record the action, return            │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  bad = claims with verdict in            │
        │        {contradicted, unknown}           │
        │  unsupported_ratio = |bad| / |claims|    │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  mode == "claim_edit"?                   │
        └──────────────────┬───────────────────────┘
                           │
      ┌────────────────────┼────────────────────┐
      ▼                    ▼                    ▼
  ┌────────┐        ┌──────────┐         ┌──────────┐
  │ stub   │        │ evidence │         │regenerate│
  │ KEEP   │        │          │         │          │
  └────────┘        │ rebuild  │         │ full LLM │
                   │ from     │         │ rewrite  │
                   │ supported│         └──────────┘
                   │ claims  │
                   │   ↓     │
                   │ if none:│
                   │ substitute
                   │ evidence[0]
                   └──────────┘
```

### Inside `claim_edit`

```
   ┌────────────────────────────────────────────────────────────┐
   │  ① SELECT                                                   │
   │     flagged   = claims where verdict == "contradicted"      │
   │     confident = flagged where                              │
   │                 (1 − claim.confidence) ≥ 0.5                │
   │                                                            │
   │     → "contradicted" claims get confidence = 1 − prob,      │
   │       so the strength of the contradiction is                │
   │       1 − confidence, NOT confidence itself                 │
   │                                                            │
   │     if nothing confident:                                   │
   │        flagged empty → KEEP                                 │
   │        flagged present → ABSTAIN "no_confident_contradiction"│
   └────────────────────────┬───────────────────────────────────┘
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  ② LOCATE the claim's span in the original answer          │
   │     fast path: answer.find(claim_text)                     │
   │     slow path: normalise whitespace+case, find, map offset  │
   │                back to the original, extend to word end     │
   │     → if not found: SKIP this claim (never delete it)       │
   └────────────────────────┬───────────────────────────────────┘
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  ③ CHOOSE the replacement                                  │
   │     from THAT claim's own evidence (not the query's best)   │
   │     → the single best-matching SENTENCE, never the passage  │
   └────────────────────────┬───────────────────────────────────┘
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  ④ GUARDS — both must pass                                 │
   │                                                            │
   │  _replacement_is_proportionate                             │
   │    repl_words ≤ max(12, claim_words × 4)                   │
   │                                                            │
   │  _replacement_is_relevant                                  │
   │    the claim's capitalised subject entity appears in the    │
   │    replacement (or ≥50% token overlap if no entity)          │
   └────────────────────────┬───────────────────────────────────┘
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  ⑤ SPLICE                                                   │
   │     sort spans by start offset                              │
   │     walk the answer, copy untouched parts verbatim,         │
   │     substitute replacements                                │
   │     skip overlapping spans rather than malform the text     │
   └────────────────────────┬───────────────────────────────────┘
                            ▼
              EDITED / ABSTAIN / KEEP
```

### The guards in detail

**`_replacement_is_proportionate`** (line 397):

```python
claim_words = len(claim_text.split())
repl_words  = len(replacement.split())
if claim_words == 0: return False
return repl_words <= max(12, claim_words * 4)
```

A short claim is normally answered by a short span. A much longer replacement
is almost certainly a retrieved passage that does not answer the question,
and swapping it in buries the response in irrelevant text.

> An earlier version used a 40-word floor, which let a 25-word passage
> replace a 3-word claim — that is what turned "Memphis International Airport"
> into a paragraph about Kumasi Airport.

**`_replacement_is_relevant`** (line 420):

```python
entities = [w for w in re.findall(r"\b[A-Z][\w'-]*", claim_text)
            if w.lower() not in _STOPWORDS]
if entities:
    repl_lower = replacement.lower()
    return any(e.lower() in repl_lower for e in entities)

claim_tokens = {t for t in re.findall(r"\w+", claim_text.lower()) if t not in _STOPWORDS}
repl_tokens  = {t for t in re.findall(r"\w+", replacement.lower())  if t not in _STOPWORDS}
return len(claim_tokens & repl_tokens) / len(claim_tokens) >= 0.5
```

The NLI has no way to say "the evidence is irrelevant" — only "the premise
does not entail this". So the verifier can mark a correct claim as
contradicted when the retrieved evidence is about a different subject.
Swapping in a replacement about a different subject then replaces a correct
answer with a wrong one.

The check looks for the claim's **subject** — the capitalised entity it is
about. Token overlap over the whole claim is the wrong measure: a valid
replacement often gives a *different fact about the same subject* ("Mary
Ramsey is younger than Lee Ranaldo" → "Mary Ramsey was born in 1984"), which
shares only 2 of 5 content tokens while being entirely on topic.

Comparing the whole entity (not a prefix) avoids "Lee" matching "Leeds".

### The per-claim guard is not an all-or-nothing veto

If one flagged claim has no usable replacement, **that claim is skipped** and
the others are still rewritten (lines 320-331):

```python
if not replacement:
    skipped += 1
    guard_reason = guard_reason or "no_usable_replacement"
    continue
```

An earlier version abandoned the whole answer here, which meant one unusable
claim suppressed every good replacement in the same answer and left the
rewrite firing on only **16 of 500 rows**.

### Why `evidence` mode is not the default

`evidence` mode rebuilds the answer from the supported claims alone. That
discards the original phrasing of every claim the verifier did not
*explicitly* mark as supported, and when no claim is supported it substitutes
a raw retrieved passage for the answer. On the length-matched HaluEval split
that turned an already-correct answer into a worse one — a **BERTScore of
1.000 fell to 0.811** — so the measured mitigation delta was negative.

### The provided-answer path

Lines 208-224: a supplied answer is being *judged*, not improved. Rewriting
it would mean the gate and the detector score a different answer than the
one the benchmark supplied. The reduction action is still recorded so the
HRR/CCR figures stay meaningful.

**Consequence:** the mitigation code only actually runs through
`scripts/evaluate_mitigation.py`, which deliberately calls `pipe.run(query)`
with no answer.

### Outputs

| Field | Value |
|---|---|
| `ctx.final_answer` | the mitigated answer |
| `ctx.iterations` | `min(max_attempts, max(1, |bad|))` |
| `ctx.decision_log` | `{"step":10, "mode", "unsupported_ratio", "decision", "guard"?}` |

---

## Step 11 — Final Gate

**File:** `src/steps/step11_final_gate.py:15`
**Config:** `step11_final_gate` → `pass_threshold: 0.70`, `abstain_threshold: 0.30`,
`max_regen_attempts: 2`

### Diagram

```
        ctx.confidence          ctx.claims
              │                     │
              ▼                     ▼
   ┌──────────────────────────────────────────────┐
   │  unsupported_ratio =                        │
   │    (contradicted + unknown claims)          │
   │    ─────────────────────────────             │
   │              total claims                    │
   └──────────────────┬───────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
 ┌─────────┐    ┌──────────────┐  ┌──────────┐
 │ score   │    │  score       │  │  score   │
 │ ≥ 0.70  │    │  < 0.30      │  │          │
 │ AND     │    │  OR          │  │  (gray   │
 │ unsup   │    │  unsup ≥ 0.5 │  │   zone)  │
 │  == 0   │    │              │  │          │
 └────┬────┘    └──────┬───────┘  └────┬─────┘
      │                │               │
      ▼                ▼               ▼
    PASS            ABSTAIN      ┌──────────────┐
                                   │ attempts <   │
                                   │ max_attempts?│
                                   └──┬────────┬──┘
                             yes ──────┘        └──── no
                                 ▼                    ▼
                              REGEN              score ≥ 0.5?
                              (loop to ③)        PASS : ABSTAIN
```

### The decision logic

```python
score = self.ctx.confidence
unsupported = sum(1 for c in self.ctx.claims if c.verdict in {"contradicted","unknown"})
total = max(1, len(self.ctx.claims))
unsupp_ratio = unsupported / total

attempts = getattr(self.ctx, "_regen_attempts", 0)
if score >= pass_th and unsupp_ratio == 0.0:
    decision = "PASS"
elif score < abs_th or unsupp_ratio >= 0.5:
    decision = "ABSTAIN"
else:
    if attempts < max_attempts:
        self.ctx._regen_attempts = attempts + 1
        decision = "REGEN"
    else:
        decision = "PASS" if score >= 0.5 else "ABSTAIN"
```

**The asymmetry is deliberate.** PASS requires **both** a high confidence
**and** zero unsupported claims. A high confidence with any unsupported claim
falls into the gray zone and triggers a regeneration — because confidence is
a *fused* score and can be high on the strength of the retrieval and
grounding terms while individual claims still fail verification.

**Two attempt counters.** The gate tracks its own via
`ctx._regen_attempts`, separate from the orchestrator's `attempt` counter in
`src/pipeline.py:248`. The gate needs to know it has exhausted its budget
even when reached from a different path.

### Outputs

`ctx.decision` ∈ {`PASS`, `REGEN`, `ABSTAIN`}. `ctx.decision_log` gets
`{"step":11, "confidence", "unsupported_ratio", "decision", "pass_threshold",
"abstain_threshold"}`.

---

## The regeneration loop

**File:** `src/pipeline.py:236-277`

### Diagram

```
        ③ → ⑤ → ⑧ → ④ → ⑨ → ⑩ → ⑪
                              │
                              ▼
                   ⑪ returns REGEN?
                        │        │
                       no       yes
                        │        │
                        │    attempt < 2?
                        │        │        │
                        │       yes       no
                        │        │        │
                        │        ▼        ▼
                        │   ┌──────────────────────────┐
                        │   │ rewind to the first step │
                        │   │ in the `reverify` set    │
                        │   │ (③) and re-run ③→⑤→⑧→④→⑨→⑩→⑪│
                        │   └──────────────────────────┘
                        ▼
                      ⑫ → ⑬  →  DONE
```

```python
reverify = {
    "early_detection", "claim_extraction", "nli_verification",
    "confidence_assessment", "confidence_fusion", "regeneration", "final_gate",
}
max_attempts = int((self.cfg.step(11) or {}).get("max_regen_attempts", 2))

# after step 11 runs:
if step.name == "final_gate" and ctx.decision == "REGEN" and attempt < max_attempts:
    attempt += 1
    ctx.metadata["regeneration_attempt"] = attempt
    index = 0
    while index < len(registry) and registry[index].name not in reverify:
        index += 1
    continue
```

**What is re-run:** steps 3, 5, 8, 4, 9, 10, 11.

**What is NOT re-run:** steps 1, 6, 7, 2. The query was already understood,
the corpus already retrieved, and `claim_edit` does not regenerate from
scratch — it edits the existing answer.

**Why the re-run matters:** the revised answer must be **re-verified**. The
gate must not pass an answer it has not checked. A rewritten answer whose
replacement claim is itself unsupported must be caught.

**Cap:** `max_regen_attempts: 2`.

---

## Step 12 — Explainable Response

**File:** `src/steps/step12_explainable_response.py:17`

### Diagram

```
      ctx.query, ctx.answer, ctx.confidence, ctx.decision
      ctx.claims[], ctx.evidence[], ctx.decision_log[]
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │  BUILD JSON DICT                     │
        │    query, answer, confidence, decision│
        │    claims[]: text, verdict, confidence,│
        │               provenance, citation,   │
        │               evidence[]              │
        │    evidence[]: doc_id, text, score     │
        │    decision_history[]                 │
        └──────────────────┬───────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     report_json (indent=2)    report_markdown
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ # Hallucination Pipeline  │
                          │    Report                │
                          │ **Query:** ...           │
                          │ **Answer:** ...          │
                          │ **Confidence:** 0.87     │
                          │ **Decision:** `PASS`     │
                          │                          │
                          │ ## Claims                │
                          │ 1. *(supported, conf=.94)│
                          │    Mary Ramsey was born  │
                          │    in 1984.              │
                          │    - provenance:         │
                          │      `intrinsic`         │
                          │    - citation: `doc_0042`│
                          │                          │
                          │ ## Evidence              │
                          │ - **doc_0042** (score=.71)│
                          │                          │
                          │ ## Decision history       │
                          │ - step 10: {...}         │
                          │ - step 11: {...}         │
                          └──────────────────────────┘
```

### The claim rendering

```python
lines.append(
    f"{i}. *({c.verdict}, conf={c.confidence:.2f})* {c.text}  \n"
    f"   - provenance: `{c.provenance}`  \n"
    f"   - citation: `{c.citation or '-'}`"
)
```

`c.citation or "-"` — a claim with no evidence shows `-` rather than
`None`.

### Outputs

`ctx.metadata["report_markdown"]` and `ctx.metadata["report_json"]`. Both are
surfaced on `PipelineResult.report_markdown`.

---

## Step 13 — Evaluation

**File:** `src/steps/step13_evaluation.py:246`
**Config:** `step13_eval.metrics` → 19 metric names

### Diagram

```
   ctx.claims    ctx.evidence    ctx.decision    ctx.confidence
   ctx.metadata: row_label, gold_answer, gold_evidence_ids
        │             │              │               │
        └──────┬──────┴──────────────┴───────────────┘
               ▼
    ┌──────────────────────────────────────────────┐
    │  RETRIEVAL METRICS                           │
    │    gold_ids present?                         │
    │       yes → Recall@5, P@5, MRR, nDCG         │
    │       no  → all four = None                  │
    └──────────────────┬───────────────────────────┘
                       ▼
    ┌──────────────────────────────────────────────┐
    │  CLAIM-LEVEL METRICS                         │
    │    gold answer present?                      │
    │       yes → precision, recall, F1            │
    │              + support rate                  │
    │              recall = None if label == 1      │
    │       no  → all None + support rate          │
    └──────────────────┬───────────────────────────┘
                       ▼
    ┌──────────────────────────────────────────────┐
    │  OVERALL                                    │
    │    verification_success_rate =                │
    │        1.0 if no contradicted claims else 0.0│
    │    end_to_end_accuracy =                     │
    │        1.0 if decision == "PASS" else 0.0    │
    │    confidence, decision                      │
    └──────────────────┬───────────────────────────┘
                       ▼
    ┌──────────────────────────────────────────────┐
    │  APPEND to data/metrics.json                 │
    │    {"rows": [...], "aggregate": {...}}        │
    └──────────────────────────────────────────────┘
```

### Inside the claim-level metrics

`_claim_level_metrics` (line 160) distinguishes two situations, because
conflating them makes the metric look better than it is.

**A gold answer is available.** A claim is a **true positive** when the
verifier marks it supported *and* it agrees with the gold answer. A claim
marked supported but contradicting the gold is a **false positive** — the
verifier was confidently wrong. Claims the verifier rejects are false
negatives when the answer actually contains them.

```python
tp = fp = fn = 0
for c in claims:
    supported = c.verdict == "supported"
    agrees = _agrees_with_gold(c.text)
    if supported and agrees:   tp += 1
    elif supported and not agrees: fp += 1
    elif not supported and agrees: fn += 1

prec = tp / (tp + fp)
rec  = None if label == 1 else tp / (tp + fn)
f1   = None if rec is None else 2*prec*rec / (prec + rec)
```

**Why recall is `None` for a hallucinated row.** The benchmark says the
answer is wrong, but not *which of its claims* are wrong. There is no
correct-claim set to recall against, so reporting a number would be
inventing it.

**No gold answer.** Only the support rate is well defined; P/R/F1 are
reported as `None` rather than as three copies of the same number.

> An earlier version set `claim_precision`, `claim_recall` and `claim_f1` all
> to the support rate. That is not a recall and not an F1.

### Inside the retrieval metrics

Gold IDs come from `ctx.metadata["gold_evidence_ids"]`. If absent, all four
are `None`:

```python
gold_ids = list(self.ctx.metadata.get("gold_evidence_ids") or [])
if gold_ids:
    row_metrics = {"retrieval_recall_at_5": ..., ...}
else:
    row_metrics = {"retrieval_recall_at_5": None, ...}
```

Scoring the retrieved IDs against themselves would always return 1.0 and look
like a perfect retriever — so the step reports `None` instead.

### nDCG's one-credit rule

```python
credited = set()
for i, x in enumerate(retrieved_ids, 1):
    if x in gold_ids and x not in credited:
        credited.add(x)
        dcg += 1.0 / math.log2(i + 1)
```

Each gold passage may be credited only once. Without that, several retrieved
documents matching the same gold passage would each add a gain term and the
score could exceed 1.0 — which nDCG cannot do by definition.

### The config gate

The configured metric list gates what is emitted. Config names map onto row
keys via `_METRIC_KEY_ALIASES`; names with no per-row equivalent (`bleu`,
`rouge_l`, `bert_score` — computed by the external evaluator) are reported
under `unavailable_at_step_level` rather than silently ignored.

### Outputs

A row appended to `data/metrics.json`. Step payload: `metrics`, `requested`,
`unavailable_at_step_level`.

---

## Appendix — one query, end to end

A concrete trace of how a single query flows through all 13 steps.

```
QUERY: "Who is the youngest person mentioned in the passage, and when were they born?"

 ①  intent="factual", entities=["Youngest Person"?], rewrite=(computed, discarded)
 ⑥  FAISS+BGE-M3 → [d7, d2, d19, d4, d31]  (RRF scores 0.0325…0.0318)
 ⑦  bge-reranker-v2-m3 → [d2 (0.81), d7 (0.79), d19 (0.61), d4 (0.58), d31 (0.44)]
 ②  prompt = context(d2,d7,d19,d4,d31) + question → answer =
        "Mary Ramsey was born in 1984 and is younger than Lee Ranaldo."
 ③  cosine=0.64, claim_count=1 ⚠, overlap=0.42, len=14 → P(halluc)=0.38
 ⑤  2 claims:
        "Mary Ramsey was born in 1984"            (overlap 0.55 → aggregated)
        "Mary Ramsey is younger than Lee Ranaldo"  (overlap 0.50 → aggregated)
 ⑧  NLI:
        claim 1 vs d2 → entail 0.94  → supported, conf 0.94, cite d2
        claim 2 vs d2 → neutral 0.51 → unknown,  conf 0.30
      verdict_distribution: {supported: 1, contradicted: 0, unknown: 1}
 ④  H = entropy(1,0,1)/log3 = 0.631
      eedc_raw → temperature → ctx.eedc_score = 0.58
 ⑨  similarity 0.64 → z=+0.15
      retrieval_top1 0.81 → z=+0.57
      nli 0.94 → z=+1.49  (sign −1 → −1.49)
      coverage 0.50 → z=−0.40 (sign −1 → +0.40)
      eedc 0.58 → z=+2.28  (sign +1 → +2.28)
      detector_trust 0.62 → z=−0.83 (sign −1 → +0.83)
      grounding 0.78 → z=+1.08 (sign −1 → −1.08)
      risk = Σ ≈ 1.66  → P(halluc)=0.84  → confidence 0.16
 ⑩  provided_answer? YES → provided_passthrough, answer unchanged
 ⑪  confidence 0.16 < 0.30 → ABSTAIN
 ⑫  report: 2 claims, 1 supported / 1 unknown, cite d2
 ⑬  row metrics appended
```

Note the trace shows something important: **the answer was actually faithful**
(both claims come from `d2`), but the NLI returned `unknown` on claim 2, the
unsupported ratio hit 0.5, and the gate **abstained on a correct answer**.
That is the NLI's false-positive rate showing up in production — and it is
the same limitation that caps the mitigation delta.
