# CRISP — The 13 Steps, Explained Simply

A plain-English walkthrough of every step, in step-number order (1, 2, 3 …
13). Each step has:
- a short **diagram**
- a **simple explanation** of what happens inside it
- the **exact calculation**, with the real code and worked numbers
- the **input → output**
- the **file** where the code lives

---

## Quick reference — what each step computes

| Step | Name | The one thing it calculates | Model |
|---|---|---|---|
| 1 | Query understanding | intent label + entities + a query rewrite | spaCy + GLiNER + LLM |
| 2 | Grounded generation | the answer text | Qwen2.5-0.5B-Instruct |
| 3 | Early detection | cosine, claim count, Jaccard overlap, word count → XGBoost | BGE-M3 + XGBoost |
| 4 | Confidence (EEDC) | Shannon entropy → weighted sum → sigmoid → temperature | — |
| 5 | Claim extraction | sentence split + Jaccard overlap → provenance tag | — |
| 6 | Hybrid retrieval | dense + BM25 → Reciprocal Rank Fusion → top 5 | BGE-M3 + FAISS + BM25 |
| 7 | Reranking | cross-encoder score → re-sort → truncate | bge-reranker-v2-m3 |
| 8 | NLI verification | claim-conditioned premise → NLI → verdict + confidence | mDeBERTa-v3-xnli |
| 9 | Confidence fusion | 7 components → z-score → sign → weighted sum → sigmoid | fitted logistic head |
| 10 | Regeneration | find span → choose sentence → 2 guards → splice | — |
| 11 | Final gate | confidence + unsupported ratio → PASS / REGEN / ABSTAIN | — |
| 12 | Explanation | Markdown + JSON report | — |
| 13 | Evaluation | retrieval + claim + detection metrics | — |

---

## First, one important thing

The steps are **numbered 1 to 13**, but they do **not run in numeric order**.

```
Number order :  1  2  3  4  5  6  7  8  9 10 11 12 13
Running order :  1 → 6 → 7 → 2 → 3 → 5 → 8 → 4 → 9 → 10 → 11 → 12 → 13
```

**Why?**

- **Steps 6 and 7 run before step 2.** We have to find the source passages
  *before* we ask the AI to answer. If the AI answers first, it invents
  things. (This used to be a bug — the AI answered with no source at all.)
- **Step 4 runs after step 8.** Step 4 measures how confused the checker was.
  It cannot measure that until the checker has finished.
- **Step 5 runs before step 8.** The checker checks *claims*, so the claims
  have to be found first.

The numbers stay as they are so each step can be pointed at in reports and
diagrams.

---

# Step 1 — Understand the Question

**File:** `src/steps/step01_query_understanding.py`
**What it does:** looks at the user's question and works out three things
about it.

```
        The user's question
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
 What kind   Which       A better
 of question? names?     version?
   │          │           │
   ▼          ▼           ▼
 factual   "Paris",    ask the AI
 opinion   "France"     to reword
 math                        │
 multi-hop                   ▼
 other              kept, but not used
```

### 1a — What kind of question is it?

The code just looks for words and symbols. No AI is used here. The exact
function, in the exact order it runs:

```python
def _classify_intent(self, q, labels):
    ql = q.lower()
    if any(t in ql for t in ["+", "-", "*", "/", "equals", "compute"]):
        return "math"
    if " and " in ql or " of " in ql and ql.count("?") > 1:
        return "multi-hop"
    if ql.startswith(("do you think", "is it good", "should")):
        return "opinion"
    if "?" in ql or any(ql.startswith(w) for w in
                        ["who","what","when","where","why","how"]):
        return "factual"
    return labels[-1] if labels else "other"
```

| If the question contains… | it is called |
|---|---|
| `+  -  *  /  equals  compute` | math |
| ` and ` (or ` of ` with two question marks) | multi-hop |
| starts with `do you think`, `is it good`, `should` | opinion |
| has a `?`, or starts with `who what when where why how` | factual |
| none of the above | other |

**Two things to notice in the code:**

1. **The order matters.** "math" is checked first, so a question containing
   a hyphen is called math even if it is really a fact question.
2. **The bare characters `+ - * /` are matched as substrings.** A question
   like "What is state-of-the-art?" contains a hyphen, so it is classified
   as **math**. This is a genuine bug, not a subtle one.

**The `multi-hop` line has an operator-precedence quirk.** Python evaluates
`and` before `or`, so it reads as:

```python
if (" and " in ql) or (" of " in ql and ql.count("?") > 1):
```

which is almost certainly what was intended.

**The fallback** is `labels[-1]` — the *last* label in the configured list,
not a hard-coded `"other"`. With the default config that is `"other"`, but
reordering the config would change what the fallback is.

### 1b — Which names are in the question?

Two different tools look for names, and their answers are joined together:

- **spaCy** finds normal names like people, places, companies.
- **GLiNER** is asked to look for the same *kinds* of words, so it can catch
  domain-specific names.

### 1c — Rewriting the question

The AI is asked to reword the question to make it a better search query. If
the AI is not available, the fallback just removes the `?` at the end.

The exact code:

```python
variants = [q]
rewrite_method = "identity"
if llm is not None:
    try:
        rewritten = llm.generate(
            system="You rewrite search queries to be more precise.",
            user=f"Original: {q}\n\nReturn ONE rewrite, no commentary.",
            max_new_tokens=64)
        if rewritten and rewritten.strip() and rewritten.strip() != q:
            variants.append(rewritten.strip())
            rewrite_method = "llm"
    except Exception:
        pass
if rewrite_method == "identity":
    r = re.sub(r"[?\.\!]+$", "", q).strip()
    if r and r != q:
        variants.append(r)
        rewrite_method = "heuristic"
```

Three things worth noting:

- The `try/except` swallows **any** error from the generator and falls
  through silently. A failed rewrite is invisible in the results — you only
  see it as `rewrite_method: "heuristic"` in the step payload.
- The fallback regex `[?\.\!]+$` strips **trailing** punctuation only, and
  the `$` anchor means it only fires at the very end of the string.
- `max_new_tokens=64` caps the rewrite at 64 tokens.

### 1d — How the two NER models are combined

```python
entities = []
if spacy_ner is not None:
    entities.extend(spacy_ner.extract(q))
if gliner is not None:
    entities.extend(gliner.extract(q, labels=labels))
```

The two lists are **concatenated with no deduplication** — if both models
find "Ramsey", it appears twice in the entity list. The `labels` passed to
GLiNER are the *intent* labels (`factual`, `opinion`, …), not NER categories,
so GLiNER is being prompted toward those terms.

### ⚠ One small bug worth knowing

The rewritten question is **never actually used**. The code saves the
*first* version of the question in a slot called `rewritten_query`, and the
first version is always the original question. The rewrite is still produced
and still costs a little time, but nothing downstream sees it.

**This does not break anything.** All your results were produced with the
original question, so the numbers are fine. It just means one feature is
doing no work at the moment.

### Input → Output

| | |
|---|---|
| **In** | the question text |
| **Out** | the kind of question, the list of names, the query versions |

---

# Step 2 — Write the Answer

**File:** `src/steps/step02_grounded_generation.py`
**What it does:** builds a prompt containing the source passages and the
question, and asks the AI to answer using only those passages.

```
   Source passages          The question
   (from steps 6 and 7)          │
         │                      │
         └──────────┬───────────┘
                    ▼
        ┌───────────────────────────┐
        │  "Answer using ONLY the   │
        │   context. If the context  │
        │   does not have the answer,│
        │   say 'I don't know.'      │
        │   Do not invent facts."     │
        │                           │
        │  Context: [passages]      │
        │  Question: [question]      │
        └────────────┬──────────────┘
                     ▼
            ┌────────────────┐
            │  the AI writes │
            │  the answer    │
            └───────┬────────┘
                    ▼
              the answer text
```

### The exact prompt

The system and user templates come from `configs/default.yaml`:

```yaml
prompt:
  system: |
    You are a precise assistant. Answer the user's question using ONLY the
    information in the provided context. If the context does not contain the
    answer, say "I don't know." Do not invent facts.
  user_template: |
    Context:
    {context}

    Question: {question}

    Answer:
```

The context is filled in like this:

```python
evidence = self.ctx.evidence
ctx_text = "\n\n".join(e.text for e in evidence) if evidence else ""
query = self.ctx.metadata.get("rewritten_query", self.ctx.query)
user = prompt.get("user_template", "...").format(context=ctx_text, question=query)
```

Note the **blank line between passages** — `\n\n` not `\n`. That keeps the
passages visually separate so the model does not read across a boundary as
one continuous document.

If there are no passages at all, `ctx_text` is the empty string and the
prompt still goes out with an empty Context section.

### The exact branching

```python
provided = self.ctx.metadata.get("provided_answer")
if provided is not None:
    answer = str(provided)
    backend = f"{backend} (provided)"
elif gen is None:
    answer = "I don't have enough context to answer."
else:
    answer = gen.generate(system=prompt.get("system", ""), user=user)
```

The middle branch is the **no-generator fallback**: if the model could not be
loaded at all, the pipeline says it does not know rather than crashing. Note
this produces a real answer string, which then flows into every later step
as if it were a genuine answer.

```
        ┌────────────────────────────┐
        │ Did someone hand us an     │
        │ answer already?            │
        └──┬──────────────────────┬──┘
         YES                      NO
          │                       │
          ▼                       ▼
   Use it word for word     Build the prompt and
   ★ the AI is never        ask the AI
     even loaded
```

**Why the first path matters so much.**

A hallucination test set gives you a **correct** answer and a **wrong**
answer for the *same* question. If our pipeline wrote its own answer every
time, both rows would get the identical answer, the identical numbers, and
the identical verdict. The accuracy score would then be exactly 50% — not
because the system is bad, but because the two rows were literally the same
row.

So for testing, we **check the answer we were given** instead of writing our
own. That is what makes the test meaningful.

### Input → Output

| | |
|---|---|
| **In** | the source passages, the question |
| **Out** | the answer text, plus a note of *how* it was produced |

---

# Step 3 — First Quick Guess at Hallucination

**File:** `src/steps/step03_early_detection.py`
**What it does:** computes four simple numbers about the answer and asks a
small trained model to guess how likely it is to be a hallucination.

```
        The answer  +  The question
                │
                ▼
   ┌───────────────────────────────────────┐
   │  4 numbers about the answer:           │
   │                                       │
   │  1. cosine     — how similar the      │
   │                 answer is to the       │
   │                 question (0 to 1)       │
   │                                       │
   │  2. claim_count — how many separate    │
   │                 facts it makes          │
   │                                       │
   │  3. overlap    — how many words it     │
   │                 shares with the question│
   │                                       │
   │  4. length     — how many words it is  │
   └───────────────────┬───────────────────┘
                       ▼
            ┌──────────────────────┐
            │   XGBoost model      │
            │  "what's the chance  │
            │   this is wrong?"    │
            └──────────┬───────────┘
                       ▼
              a number from 0 to 1
```

### The four features, calculated in full

```python
qv = embedder.encode([query])[0]
av = embedder.encode([answer])[0]
cosine = float(np.dot(qv, av) /
               (np.linalg.norm(qv) * np.linalg.norm(av) + 1e-12))

claim_count = max(1, len(self.ctx.claims))
overlap = intrinsic_overlap(answer, query)
ans_len = len(answer.split())

features = [cosine, float(claim_count), overlap, float(ans_len)]
```

**1. `cosine`** — both texts are turned into vectors by BGE-M3, then the
angle between them is measured. The `+ 1e-12` in the denominator prevents
division by zero when one vector happens to be all zeros.

**2. `claim_count`** — see "the catch" below.

**3. `intrinsic_overlap`** — the helper in `src/utils/claim_extractor.py`:

```python
def intrinsic_overlap(claim, query):
    qs = set(re.findall(r"\w+", query.lower()))
    cs = set(re.findall(r"\w+", claim.lower()))
    if not qs or not cs:
        return 0.0
    return len(qs & cs) / len(qs | cs)
```

This is **Jaccard similarity** — the size of the shared words divided by the
size of the combined words. It is between 0 (nothing in common) and 1
(identical word sets).

Note that it does **not** strip stopwords, so "the", "is" and "of" all count
toward the overlap.

**This is deliberately the same helper the training data used.** An earlier
version computed the overlap inline with a different formula — the fraction
of *answer* tokens present in the query, rather than the Jaccard score above
— so the served model was being fed a feature it had never seen in training.

**4. `answer_length`** — just `len(answer.split())`, the number of
whitespace-separated words.

### The catch

Number 2, `claim_count`, is **always 1**. It counts the separate facts in
the answer — but those facts are not broken out until **step 5**, which runs
*after* this step. So when this step looks, the count is always zero, and
the code turns zero into one. One of the four numbers is therefore useless.

### How the probability is produced

```python
prob = 0.5
if clf is not None:
    try:
        prob = float(clf.predict_proba(features))
    except Exception:
        prob = 0.5
self.ctx.metadata["hallucination_probability"] = prob
```

If the model is missing or throws for any reason, the probability falls back
to **0.5** — a neutral "no opinion". This is the same `0.5` that later
becomes the `detector_trust` component's neutral value in step 9.

### How good is this guess on its own?

Barely better than a coin flip — about **54%** at telling the two kinds of
answer apart. That is why this number is not used as the final answer. It
gets folded into the big combined score in **step 9** instead.

Measured on the 496-row fit used for step 9:

| Read as | ROC-AUC |
|---|---|
| `hallucination_probability` as a **risk** signal (higher = worse) | **0.5389** |
| `detector_trust` = `1 − probability` as a **trust** signal | **0.4611** |

The two add to exactly 1.0000 — they are the same information read in opposite
directions, which is exactly why the sign matters in step 9.

### Input → Output

| | |
|---|---|
| **In** | the answer, the question |
| **Out** | the four numbers, and a "chance this is a hallucination" score |

---

# Step 4 — Measure How Confused the Checker Was

**File:** `src/steps/step04_confidence_assessment.py`
**Helper:** `src/models/eedc.py`
**What it does:** combines several signals into one number called the EEDC
score, which measures how shaky the whole situation looks.

```
        Step 8's verdicts
        (supported / contradicted / unknown)
                    │
                    ▼
     ┌──────────────────────────────┐
     │  How SPREAD OUT are the       │
     │  verdicts?                    │
     │                               │
     │  All the same  → 0.0  (sure)  │
     │  All mixed up  → 1.0  (lost)  │
     │  Half and half → 0.63 (split) │
     └──────────────┬───────────────┘
                    │
   cosine (from step 3) ──┐
                          ▼
              ┌───────────────────┐
              │  PHI = α·entropy   │
              │    + β·(1-ret)     │
              │    + γ·(1-self)    │
              │    + δ             │
              │                   │
              │  EEDC = sigmoid(φ)│
              └─────────┬─────────┘
                        ▼
                 temperature scaling
                 (divide logit by T)
                        ▼
                   EEDC score
```

### 4.1 — The entropy calculation, in full

The step builds a count of the three possible verdicts, then measures how
spread out they are.

```python
verdicts = self.ctx.verdicts or []
if verdicts:
    labels = [v.label for v in verdicts if hasattr(v, "label")]
    counts = {"supported": 0, "contradicted": 0, "unknown": 0}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    total = sum(counts.values()) or 1
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log(p)
    H = H / math.log(3)          # normalise to [0,1]
else:
    H = 1.0 - cosine            # pre-NLI proxy
```

**The maths.** `H` is Shannon entropy — the standard measure of disorder. For
each verdict type it works out that type's share `p = c / total`, then adds
`-p · log(p)` for each. A type that never appears contributes nothing
(`p = 0` is skipped, because `0 · log(0)` is undefined but the limit is 0).

**The normalisation.** The maximum entropy for three categories is `log(3) ≈
1.0986`, so dividing by it maps the theoretical maximum onto exactly 1.0.

**Worked examples.**

| Verdicts | H before / log 3 | H after | Reading |
|---|---|---|---|
| 3 supported | `-1·log(1) = 0` | **0.000** | totally sure |
| 2 supported, 1 unknown | `-0.667·log(0.667) - 0.333·log(0.333)` = 0.6365 | **0.579** | mostly sure |
| 1 supported, 1 unknown | `-0.5·log(0.5) - 0.5·log(0.5)` = 0.6931 | **0.631** | split |
| 1 of each | `-3·(0.333·log(0.333))` = 1.0986 | **1.000** | no idea |

Note the raw entropy is **not** the same as "how many verdicts disagree". A
claim set of 2 supported and 1 unknown (H = 0.579) is more confident than
1 supported and 1 unknown (H = 0.631), even though both contain one unclear
claim — because the first has a larger majority. The count of verdicts does
not matter, only their proportions.

**The fallback.** With no verdicts at all, `H = 1 - cosine` — a crude proxy
meaning "if we have no checker output, treat the situation as uncertain in
proportion to how dissimilar the answer is".

### 4.2 — The other two inputs

```python
feat = self.ctx.feature_vector or {}
cosine = feat.get("cosine", 0.0)
...
self_consistency = cosine  # use cosine as a proxy until multi-sample
```

Both `retrieval_top1` and `self_consistency` are **the same value** —
`cosine`. The step then calls:

```python
raw = eedc.score(nli_entropy=H, retrieval_top1=cosine, self_consistency=cosine)
```

**The honest limitation.** The step's own comment says *"until multi-sample"*,
which means a real version would sample the generator several times and
measure how much the answers agree with each other. That has not been
built. Only the entropy term is genuinely new information from the NLI; the
other two are the same number counted twice.

### 4.3 — The EEDC formula itself

`src/models/eedc.py:56`:

```python
def score(self, nli_entropy, retrieval_top1, self_consistency):
    self._maybe_load()
    phi = (
        self.alpha   * float(nli_entropy)
        + self.beta  * (1.0 - float(retrieval_top1))
        + self.gamma * (1.0 - float(self_consistency))
        + self.delta
    )
    return _sigmoid(phi)
```

**Read this as a weighted vote.** Each term contributes to `phi`, and the
sign of each weight says which direction means "more confident":

| Term | Weight (default) | Reading |
|---|---|---|
| `alpha × nli_entropy` | **−1.0** | more entropy (checker confused) → lower phi → lower score |
| `beta × (1 − retrieval_top1)` | **−0.7** | weaker top passage → lower score |
| `gamma × (1 − self_consistency)` | **−0.5** | less self-consistent → lower score |
| `delta` | **+1.0** | a constant offset, the "starting point" |

The `1 − x` on the last two terms is important: the raw values are "higher is
better", so they are inverted to make "higher is better" mean *more*
uncertainty, matching the entropy term's direction. All three uncertainty
terms then push `phi` down, and `delta` holds the base level.

**The sigmoid.**

```python
def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
```

This is the standard logistic function. It squeezes any real number into
(0, 1). The two-branch form avoids overflow — `exp(-x)` would blow up for a
very negative `x`, so the negative branch is rewritten algebraically.

**A full worked calculation.** Take the worked example from the end of this
document — 1 claim supported, 1 claim unknown, cosine = 0.64:

```
H = 0.631
1 - cosine = 1 - 0.64 = 0.36

phi = (-1.0)(0.631)
    + (-0.7)(0.36)
    + (-0.5)(0.36)
    + 1.0

    = -0.631 - 0.252 - 0.180 + 1.000
    = -0.063

EEDC = sigmoid(-0.063) = 1 / (1 + e^0.063) = 1 / 2.065 = 0.484
```

So the raw EEDC score is **0.484** — just under 0.5, meaning "mildly
suspicious". Because the entropy term is scaled by the largest weight
(−1.0) and `delta` is only +1.0, a genuinely confused checker (H near 1.0)
is enough on its own to pull the score well below 0.5.

### 4.4 — Where the weights come from

`src/models/eedc.py:41`:

```python
def _maybe_load(self):
    if self._loaded: return
    p = Path(self.weights_path)          # data/eedc_weights.json
    if p.exists():
        w = json.loads(p.read_text(encoding="utf-8"))
        self.alpha = float(w.get("alpha", self.alpha))
        self.beta  = float(w.get("beta",  self.beta))
        self.gamma = float(w.get("gamma", self.gamma))
        self.delta = float(w.get("delta", self.delta))
    self._loaded = True
```

**If `data/eedc_weights.json` exists, its values replace the defaults.** If
it does not, the defaults `alpha=-1.0, beta=-0.7, gamma=-0.5, delta=1.0`
stand.

> **Current state: the file does not exist in this repo.** So the pipeline is
> running on the hard-coded default weights. They are sensible signs, but they
> were never fitted against a labelled set — so they have not been validated
> the way the step-9 detection head has been.

**How the fitting works** (`scripts/calibrate_eedc.py`). It is a small grid
search over four parameters, choosing the combination with the lowest
negative log-likelihood:

```python
for alpha in [-1.5, -1.0, -0.5]:
    for beta  in [-1.0, -0.7, -0.3]:
        for gamma in [-0.7, -0.5, -0.3]:
            for delta in [0.5, 1.0, 1.5]:
                phi = alpha*X[:,0] + beta*X[:,1] + gamma*X[:,2] + delta
                p = 1.0 / (1.0 + np.exp(-phi))
                p = np.clip(p, 1e-6, 1.0 - 1e-6)
                nll = -(y*np.log(p) + (1.0-y)*np.log(1.0-p)).mean()
                if nll < best["nll"]:
                    best.update({"nll": nll, "alpha": alpha, ...})
```

That is **81 combinations** (3×3×3×3). The negative log-likelihood is the
standard "how wrong are the probabilities" penalty: for each row it scores
how much probability the model gave the wrong answer. The grid is coarse on
purpose — four parameters is enough to avoid overfitting a small dev set.

The input file needs one row per example:

```json
{"nli_entropy": 0.63, "retrieval_top1": 0.64, "self_consistency": 0.64, "label": 1}
```

### 4.5 — The calibration

The raw EEDC score is a probability, but probabilities from a hand-set linear
model are usually over-confident. So the step runs it through a calibrator.

```python
calibrated = raw
if calibrator is not None and hasattr(calibrator, "transform"):
    try:
        calibrated = float(np.asarray(calibrator.transform(np.array([raw])))[0])
    except Exception:
        calibrated = raw
```

Any failure falls back to the raw score.

**Temperature scaling** (`src/models/calibration.py`) works on the logit:

```python
logits = np.log(s / (1.0 - s)) / self.temperature
return 1.0 / (1.0 + np.exp(-logits))
```

The temperature `T` is fitted by the same grid-search-on-NLL approach, over
`np.linspace(0.25, 4.0, 32)`. `T > 1` softens (pulls toward 0.5); `T < 1`
sharpens. A temperature of 1.0 is the identity — no change.

**Isotonic calibration** is the alternative: it sorts the scores, then
enforces that a higher score never gets a lower label, by averaging
neighbours wherever the ordering is violated (Pool-Adjacent-Violators). It
can model any monotonic relationship but can overfit on small samples.

Which one is used is set by `step04_confidence.method` in the config:
`none`, `temperature` (default), or `isotonic`.

### 4.6 — What comes out

```python
self.ctx.eedc_score = float(calibrated)
self.ctx.metadata["step04_confidence_assessment"] = {
    "eedc_raw": raw,
    "eedc_calibrated": self.ctx.eedc_score,
    "nli_entropy": H,
    "self_consistency": self_consistency,
}
```

All four are recorded, so a run's trace shows the raw score, the calibrated
score, the entropy that drove it, and the consistency proxy. That is enough
to reconstruct the whole calculation by hand.

### 4.7 — How step 4's output is used

The EEDC score is **one of seven components** in step 9's fusion. On its own
it is not a decision — it is one input to the weighted sum.

Two things about it matter for step 9:

- It is on a **0-1 scale**, like the other components, so it can be
  z-scored against the stored mean and standard deviation before use.
- Its fitted weight in the step-9 head is **−0.0749** — very small. The
  sign is negative, meaning higher EEDC means *more* risk. But the
  magnitude is tiny compared to `nli` (1.2497) or `similarity` (0.5711).

> In fact, the small weight reflects a measured fact: the EEDC entropy
> component's own standard deviation is **0.0522**, the narrowest of all
> seven components. After standardisation it barely moves the final score.
> The entropy is real information, but it is a small nudge, not a driver.

### Input → Output

| | |
|---|---|
| **In** | the checker's verdicts, the cosine similarity |
| **Out** | the EEDC score (`ctx.eedc_score`), plus `eedc_raw`, `eedc_calibrated`, `nli_entropy`, `self_consistency` |

---

# Step 5 — Break the Answer Into Separate Facts

**File:** `src/steps/step05_claim_extraction.py`
**What it does:** splits the answer into individual factual claims, so each
one can be checked on its own.

```
        The answer
        "Mary Ramsey was born in 1984 and
         is younger than Lee Ranaldo."
                    │
                    ▼
        ┌───────────────────────┐
        │  Split into sentences │
        │  (at . ! ? followed   │
        │   by a capital)      │
        │  Remove duplicates    │
        │  Drop tiny fragments  │
        └───────────┬───────────┘
                    ▼
          ┌──────────────────┐
          │ "Mary Ramsey was   │
          │  born in 1984"    │
          │ "Mary Ramsey is    │
          │  younger than Lee  │
          │  Ranaldo"          │
          └──────────┬─────────┘
                     │
                     ▼
         ┌───────────────────┐
         │  Tag each claim:   │
         │                    │
         │  mostly from the   │
         │  question →        │
         │  "intrinsic"       │
         │                    │
         │  mostly new        │
         │  information →     │
         │  "extrinsic"       │
         │                    │
         │  a bit of both →   │
         │  "aggregated"      │
         └───────────┬────────┘
                     ▼
               the list of claims
```

### The tagging rule, calculated

For each claim, the code measures how much of it overlaps with the question,
then puts it in one of three buckets:

```python
for txt in claims_text:
    ov = intrinsic_overlap(txt, query)
    if ov   >= intr_th:      # 0.7
        prov = "intrinsic"
    elif ov <= agg_th:        # 0.3
        prov = "extrinsic"
    else:
        prov = "aggregated"
    claims.append(Claim(text=txt, provenance=prov, intrinsic_overlap=ov))
```

**The overlap measure** is the same Jaccard score from step 3:

```
ov = |query_words ∩ claim_words| / |query_words ∪ claim_words|
```

Both sides are lowercased and split on `\w+`, so punctuation disappears and
case does not matter. Each side is a **set**, so a word repeated three times
counts once.

| Overlap | Tag | Meaning |
|---|---|---|
| ≥ 0.7 | `intrinsic` | the claim is mostly made of words from the question |
| ≤ 0.3 | `extrinsic` | the claim brought in mostly new information |
| between | `aggregated` | it mixes both |

**Worked example.** Question: *"Who is the youngest person mentioned in the
passage?"* → 9 distinct words. Claim: *"Mary Ramsey is younger than Lee
Ranaldo"* → 7 distinct words.

- shared: `who, is, the, youngest, person, mentioned, in, than` → 8 words
- union: 9 + 7 − 8 = 8 distinct words (the claim's `Mary, Ramsey, Lee,
  Ranaldo` are new)
- `ov = 8/8 = 1.00`? No — the sets differ, so union = 9 + 4 = 13 and
  `ov = 8/13 = 0.615` → **aggregated**

The thresholds are configurable: `step05_claim_extraction.intrinsic_threshold`
(0.7) and `aggregated_threshold` (0.3).

**A limitation worth naming.** Because the overlap is measured against the
*whole* question, a question that names its subject makes every claim about
that subject look intrinsic. The tag is a rough provenance signal, not a
precise one.

### The exact sentence splitter

`src/utils/chunker.py:7`:

```python
def split_sentences(text):
    text = text.strip()
    if not text: return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]
```

The regex has two conditions:
- `(?<=[.!?])` — the split point must come **after** a `.`, `!` or `?`
- `\s+(?=[A-Z])` — followed by **whitespace** and then a **capital letter**

**What this means in practice:**

| Text | Split? | Why |
|---|---|---|
| `"End. Next"` | yes ✓ | `.` + space + capital |
| `"End. next"` | **no** ✗ | `n` is lowercase |
| `"End.Next"` | **no** ✗ | no whitespace |
| `"End.  Next"` | yes ✓ | `\s+` matches two spaces |
| `"Dr. Smith went"` | **no** ✗ | `S` is capital but `.` is inside "Dr." — actually splits into "Dr." and "Smith went" |

The **"Dr." case** is a real false split: the abbreviation's period is
followed by a space and a capital, so "Dr." becomes its own claim.

### The rule-mode cleaning

```python
def extract_claims_rule(text):
    sents = split_sentences(text)
    cleaned = []
    for s in sents:
        s = re.sub(r"^[\s\-\*\u2022]+", "", s).strip()   # strip list markers
        if len(s) >= 5:                                  # drop tiny fragments
            cleaned.append(s)
    return _dedup(cleaned)
```

`len(s) >= 5` is a **character** count, not a word count — so a 5-character
fragment like "Yes." (4 chars) is dropped, but "It is" survives.

### The deduplication

```python
def _dedup(claims):
    seen = set(); out = []
    for c in claims:
        key = re.sub(r"\s+", " ", c.lower()).strip()
        if key and key not in seen:
            seen.add(key); out.append(c)
    return out
```

The **key** is lowercased and whitespace-collapsed, but the **original** text
of the first occurrence is what gets kept. So "The sky is blue" and "the sky
is blue" become one claim, kept with the original capitalisation.

### Three ways to do it

| Mode | How it works |
|---|---|
| `rule` (default) | just split on sentence endings |
| `llm` | ask the AI to break the answer into one fact per line |
| `hybrid` | try the simple way first; ask the AI only if fewer than 2 claims came out |

**`hybrid` was tried and then undone.** It makes finer-grained claims but
the scores got much worse — the F1 dropped to 0.23 and the verification
success rate to 0.22. The reason is that short answers get chopped into tiny
fragments, and the checker in step 8 cannot make a decision on a fragment.
So the default stays `rule`.

### One weakness in the sentence splitter

It only splits when a sentence ending is followed by a space **and** a
capital letter. So `"...the end. next sentence"` stays as one sentence,
because `n` is lowercase.

### Input → Output

| | |
|---|---|
| **In** | the answer, the question |
| **Out** | a list of claims, each with a text and a tag |

---

# Step 6 — Find the Source Passages

**File:** `src/steps/step06_hybrid_retrieval.py` and `src/retrieval/hybrid.py`
**What it does:** looks through the document collection and pulls out the
5 passages most likely to contain the answer.

```
                  The question
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   ┌───────────────┐           ┌───────────────┐
   │  DENSE search │           │ SPARSE search │
   │               │           │               │
   │  Turn the     │           │  Look for the  │
   │  question     │           │  exact words  │
   │  into a       │           │  in the text  │
   │  number, find │           │               │
   │  similar      │           │  (BM25)       │
   │  numbers      │           │               │
   │  (FAISS)      │           │               │
   └───────┬───────┘           └───────┬───────┘
           │                           │
           ▼                           ▼
      15 passages                 15 passages
           │                           │
           └─────────────┬─────────────┘
                         ▼
        ┌────────────────────────────────┐
        │   MERGE THEM (RRF)              │
        │                                │
        │   Each list is walked from the  │
        │   top. A passage gets points   │
        │   for how high up it appears:   │
        │                                │
        │   points = 1 / (60 + rank)      │
        │                                │
        │   A passage found by BOTH       │
        │   searches gets double points.  │
        └────────────────┬───────────────┘
                         ▼
              Best 5 passages
```

### Why the scores are added by *rank*, not by value

The dense search gives a number between 0 and 1. The word search gives a
number with no fixed limit. You cannot add those directly. So both are
converted to "points by position" first, which works for any kind of search.

### The RRF formula, calculated

```python
def search(self, query, query_vec, top_k=5):
    dense_hits  = self.faiss.search(query_vec, top_k=top_k * 3)   # 15 candidates
    sparse_hits = self.bm25.search(query,   top_k=top_k * 3)     # 15 candidates

    fused = {}
    text_lookup = {}
    for rank, ev in enumerate(dense_hits):
        fused[ev.doc_id] = fused.get(ev.doc_id, 0.0) + 1.0/(self.rrf_k + rank + 1)
        text_lookup[ev.doc_id] = ev.text
    for rank, ev in enumerate(sparse_hits):
        fused[ev.doc_id] = fused.get(ev.doc_id, 0.0) + 1.0/(self.rrf_k + rank + 1)
        text_lookup[ev.doc_id] = ev.text

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [Evidence(doc_id=d, text=text_lookup[d], score=s)
            for d, s in ordered[:top_k]]
```

Each passage's points are `1 / (rrf_k + rank + 1)` with `rrf_k = 60`. The
`+1` makes the first passage rank 1, not 0.

**Worked example.** Suppose passage `d2` is rank 1 in the dense list and
rank 4 in the word list:

```
dense  points = 1/(60 + 1 + 1) = 1/62 = 0.016129
sparse points = 1/(60 + 4 + 1) = 1/65 = 0.015385
                   total         =  0.031514
```

And a passage `d19` that only the dense search found, at rank 3:

```
dense  points = 1/(60 + 3 + 1) = 1/64 = 0.015625
                   total         =  0.015625
```

So `d2` beats `d19` — **because two searches agreed on it**, not because
either search rated it higher. That is the whole point of the merge.

**Why `rrf_k = 60` matters.** It is the damping constant. Without it (or
with a small value), rank 1 would score `1/1 = 1.0` while rank 2 scored
`1/2 = 0.5` — a huge gap. With 60, the top scores are 0.0161 and 0.0164 —
nearly flat, so the *number of lists* a document appears in matters much
more than its exact position.

### Two config keys that do nothing

`weight_dense: 1.0` and `weight_bm25: 1.0` are in the config, but
`hybrid.py` never reads them — both lists are added with an implicit weight
of 1.0. **Changing them has no effect.** The code would need
`weight_dense * 1.0/(...)` to honour them.

### The BM25 parameters

| Parameter | Value | Meaning |
|---|---|---|
| `k1` | 1.5 | how quickly a term's contribution saturates as it repeats |
| `b` | 0.75 | how much the length of the passage is penalised |

These are BM25's standard defaults, applied in `src/retrieval/bm25_index.py`.

### The closed-book mode

```
   ┌──────────────────────────────────────┐
   │  Is this a closed-book question?      │
   │  (a dataset with no source documents) │
   └──┬───────────────────────────────┬───┘
    YES                              NO
     │                                │
     ▼                                ▼
  NO passages at all            normal search
  │
  └──► the checker gets no source,
       so it cannot cheat
```

**Why this is needed.** One of your test sets (TruthfulQA) was built by
taking the correct answers and using them as the document collection. So
searching it finds the correct answer for every question. If the checker
then compared the answer against the retrieved text, it would be asking
"is this the correct answer?" — which is exactly the thing it is supposed to
figure out. Turning retrieval off removes that circularity.

### Two failure modes to watch for

1. **No index loaded.** The step quietly returns nothing. This used to
   happen and was the reason for the 0.50 baseline result. Always pass
   `--index`.
2. **The embedder fails.** Instead of crashing, it searches with a vector of
   all zeros, which returns nonsense.

### Input → Output

| | |
|---|---|
| **In** | the question, the document index |
| **Out** | the 5 best passages, each with an ID, text, and score |

---

# Step 7 — Reorder the Passages by Relevance

**File:** `src/steps/step07_reranking.py`
**What it does:** re-scores the 5 passages and puts the best one first.

```
        5 passages from step 6
                 │
                 ▼
     ┌───────────────────────────┐
     │  bge-reranker-v2-m3        │
     │                           │
     │  Reads the question AND   │
     │  the passage TOGETHER     │
     │  and gives one score      │
     └────────────┬──────────────┘
                  ▼
        Replace each passage's
        score with the new one
                  │
                  ▼
        Sort highest score first
                  │
                  ▼
        Keep the best 5
```

### Why this is different from step 6

Step 6 reads the question and the document **separately**, then compares the
results — like judging a book and a question that have never met. Step 7
puts them **side by side** in the model at once, which is more accurate but
slower.

### One thing to be careful about

The new score **completely replaces** the old one. After this step, the score
on the first passage is the reranker's score, not the merged score from step
6. And step 9 uses that number.

### Input → Output

| | |
|---|---|
| **In** | the 5 passages, the question |
| **Out** | the same 5 passages, re-scored and re-ordered |

---

# Step 8 — Check Each Fact Against the Passages

**File:** `src/steps/step08_nli_verification.py`
**Model:** `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
**What it does:** this is the heart of the system. For every claim, it asks
"does this passage support this claim?" and records the answer.

```
   The claims            The passages
   (from step 5)         (from step 7)
       │                      │
       └──────────┬───────────┘
                  ▼
    ┌─────────────────────────────────────┐
    │  FOR EACH CLAIM:                    │
    │  ┌───────────────────────────────┐  │
    │  │  FOR EACH PASSAGE:            │  │
    │  │                               │  │
    │  │  Pick the part of the passage │  │
    │  │  that is about THIS claim     │  │
    │  │  (see the box below)          │  │
    │  │            │                  │  │
    │  │            ▼                  │  │
    │  │  ┌────────────────────────┐  │  │
    │  │  │  The NLI model         │  │  │
    │  │  │                        │  │  │
    │  │  │  "Does the passage     │  │  │
    │  │  │   support the claim?"  │  │  │
    │  │  │         │              │  │  │
    │  │  │         ▼              │  │  │
    │  │  │  supported /           │  │  │
    │  │  │  contradicted /        │  │  │
    │  │  │  unknown               │  │  │
    │  │  │         + a score      │  │  │
    │  │  └────────────────────────┘  │  │
    │  └───────────────────────────────┘  │
    └─────────────────┬───────────────────┘
                      ▼
         Keep the best verdict per claim
                      │
                      ▼
         Attach the best passage as a citation
```

### The "part of the passage about this claim" box

This is the most important trick in the whole step.

```
   A long passage (say 9000 characters)
   ┌────────────────────────────────────────┐
   │ Sentence 1    Sentence 2   Sentence 3  │
   │ ...           THE RELEVANT   ...       │
   │               ONE                      │
   └────────────────────────────────────────┘
                    │
                    ▼
   Pick the sentences that share the most
   words with the claim, plus their
   neighbours, join them together
                    │
                    ▼
   Send only that to the NLI model
```

**The selection algorithm, in full** (`_claim_premise`, line 30):

```python
if not evidence_text:
    return ""
if len(evidence_text) <= max_chars:          # 1800
    return evidence_text                        # short enough — use it all

sentences = split_sentences(evidence_text)
if not sentences:
    return _window_around_match(evidence_text, claim_text, max_chars)

claim_tokens = {t for t in re.findall(r"\w+", claim_text.lower())}
if not claim_tokens:
    return evidence_text[:max_chars]

scored = []
for i, s in enumerate(sentences):
    tokens = set(re.findall(r"\w+", s.lower()))
    overlap = len(claim_tokens & tokens)
    scored.append((overlap, -i, i, s))
scored.sort(reverse=True)

chosen = set()
budget = max_chars
for overlap, _, i, s in scored:
    if overlap == 0 and chosen:
        break
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(sentences) and j not in chosen:
            chosen.add(j)
            budget -= len(sentences[j])
    if budget <= 0:
        break

if not chosen:
    return _window_around_match(evidence_text, claim_text, max_chars)
picked = [sentences[i] for i in sorted(chosen)]
return " ".join(picked)[:max_chars]
```

**Step by step:**

1. **Short-circuit.** If the passage is already under 1800 characters, it is
   used whole — no selection needed.
2. **Split** the passage into sentences.
3. **Score** each sentence by how many words it shares with the claim
   (`len(claim_tokens & tokens)` — a plain set intersection count).
4. **Sort** by that score, highest first. The `-i` in the tuple means that on
   a tie, the *earlier* sentence wins — `sort` is descending, so `-i` being
   more negative for later sentences puts them last.
5. **Take** the best sentences, and for each one also take its immediate
   neighbours (i−1, i, i+1) so the model reads a contiguous passage rather
   than a fragment.
6. **Stop** at the first sentence with zero overlap (once something has been
   chosen). This means it takes a *contiguous relevant region*, not scattered
   matches from all over the document.
7. **Respect the budget** — stop adding once 1800 characters are used.
8. **Rejoin in original order** (`sorted(chosen)`), so the passage still
   reads in sequence.

**The fallback** `_window_around_match` (line 81) is used when the passage
has no sentence boundary or nothing matched:

```python
tokens = re.findall(r"\w+", query.lower())
needle = tokens[0]
pos = text.lower().find(needle)
start = max(0, pos - max_chars // 3)
return text[start:start + max_chars]
```

It centres a window on where the claim's **first word** appears, starting a
third of the budget earlier.

**Why this matters.** The NLI model can only read about 512 tokens. If you
hand it a 9000-character passage, it sees only the beginning. A fact that
the passage supports *later on* would then come back as "unknown" — the
evidence was right there, the model just never saw it.

On RAGTruth, where passages run long, this made **a third of all claims
impossible to decide**, and the whole detection score collapsed to chance.

**Why it matters for step 10 too.** The same idea is used there to pick a
replacement. So the checker and the fixer agree on what "the evidence for
this claim" is.

### The three verdicts

| Verdict | Meaning | The claim's confidence |
|---|---|---|
| `supported` | the passage backs it up | the score itself (high = good) |
| `contradicted` | the passage says otherwise | `1 - score` (high = **weak** contradiction) |
| `unknown` | cannot tell | a flat 0.3 |

Note the confusing bit about `contradicted`: its confidence is *inverted*,
because a high score there means the model was sure. Step 10 has to undo
this before it uses the number.

### The NLI model call itself

`src/models/nli_deberta.py:54`:

```python
def predict(self, premise, hypothesis):
    self._ensure_loaded()
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

Four details that matter:

- **`truncation=True, max_length=512`** — this is the truncation that
  motivated the claim-conditioned selection in the first place.
- **The label mapping is substring-based.** It checks whether the model's own
  label name contains "entail" or "contradict", so it keeps working if the
  label set changes between models.
- **`torch.no_grad()`** — no gradients are needed for inference, so this
  saves memory and time.
- **fp16 only on CUDA** (line 45 of the same file): on CPU, fp16 matmul is
  slow and poorly supported, so the device check keeps fp32 there.

### Which verdict wins

```python
if lbl == "supported" and score > best_score:
    best_label, best_score, best_doc = lbl, score, ev.doc_id
elif lbl == "contradicted" and best_label != "supported" and score > best_score:
    best_label, best_score, best_doc = lbl, score, ev.doc_id
```

`supported` always beats `contradicted`, whatever the scores are. A
contradiction only wins when nothing supported the claim.

**The confidence mapping:**

```python
if best_label == "supported":
    claim.confidence = float(best_score)
else:
    claim.confidence = 1.0 - float(best_score) if best_label == "contradicted" else 0.3
```

So for a contradicted claim, `confidence = 1 − contradiction_probability`.
A high confidence there means a **weak** contradiction. Step 10 has to
invert it again before deciding whether to rewrite.

### The cost

One model call for **every** (claim × passage) pair. Three claims and five
passages = 15 calls. This is what makes the detection path take roughly
0.8–13.8 seconds.

### Input → Output

| | |
|---|---|
| **In** | the claims, the passages |
| **Out** | a verdict and confidence for every claim, plus the supporting passage for each |

---

# Step 9 — Combine Everything Into One Score

**File:** `src/steps/step09_confidence_fusion.py`
**What it does:** takes all the signals gathered so far and combines them
into a single "how likely is this to be a hallucination" number.

```
   cosine         (step 3)  ──┐
   top passage score (step 7) ─┤
   checker confidence  (step 8) ─┤
   supported share     (step 8) ─┤
   EEDC score          (step 4) ──┤
   step-3 guess        (step 3) ──┼──►  7 components
   word overlap w/ src  (step 9) ─┘         │
                                              ▼
                              ┌───────────────────────────┐
                              │  Standardise each one     │
                              │  (put them all on the    │
                              │   same scale)            │
                              └────────────┬──────────────┘
                                           ▼
                              ┌───────────────────────────┐
                              │  Apply the sign           │
                              │                           │
                              │  Some numbers go UP when  │
                              │  the answer is bad.        │
                              │  Some go UP when it's good.│
                              │                           │
                              │  Flip the good ones so    │
                              │  everything now means      │
                              │  "more likely to be bad"   │
                              └────────────┬──────────────┘
                                           ▼
                              ┌───────────────────────────┐
                              │  Weighted sum             │
                              │  = the risk score          │
                              └────────────┬──────────────┘
                                           ▼
                              ┌───────────────────────────┐
                              │  Turn into 0 to 1          │
                              │  (sigmoid)                 │
                              └────────────┬──────────────┘
                                           ▼
                              P(hallucination)  ──►  confidence = 1 − P
```

### The sign problem — and why it caused a bug

Some of the seven numbers go **up** when the answer is bad (cosine goes
*down* actually — but the point is the direction differs per number). Others
go up when the answer is good.

| Component | High value means | Direction |
|---|---|---|
| `similarity` | more risky | risk |
| `retrieval_top1` | more risky | risk |
| `nli` | better supported | **trust** |
| `coverage` | better supported | **trust** |
| `eedc` | more risky | risk |
| `detector_trust` | more faithful | **trust** |
| `grounding` | more grounded | **trust** |

The trust ones get **flipped**, so that a bigger contribution to the final
score always means the same thing: *more likely to be a hallucination*.

**This was a real bug.** The earlier version added every number with a
positive weight into a score it treated as if all of them meant "good". The
two biggest components pointed the *opposite* way to the truth, so the whole
score came out at **ROC-AUC 0.360** — worse than guessing. The fix was to
actually **fit** the weights and the signs on real labelled data instead of
setting them by hand.

### Why standardise first

The seven numbers live on wildly different scales. The EEDC score hovers
around 0.3, the retrieval score around 0.4, the NLI confidence around 0.5.
If you just added them with weights, whichever number happens to swing the
widest would dominate the whole thing. So each is converted to "how many
standard deviations above or below its usual value".

### The word-overlap component (grounding)

```
   The answer's content words    The top passage's content words
   "Ramsey born 1984 younger"    "Ramsey 1984 born Minneapolis..."
              │                              │
              └──────────────┬───────────────┘
                             ▼
              Which answer words are NOT in the passage?
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
        trust = 1 − that share        (a number between 0 and 1)
```

`src/utils/grounding.py`:

```python
def content_tokens(text):
    return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
            if w not in _STOPWORDS and len(w) > 1]

def novel_content_rate(answer, evidence_texts):
    tokens = set(content_tokens(answer))
    if not tokens: return 0.0
    source = set()
    for t in evidence_texts or []:
        source.update(content_tokens(t))
    if not source: return 0.0
    return len(tokens - source) / len(tokens)

def grounding_trust(answer, evidence_texts):
    return 1.0 - novel_content_rate(answer, evidence_texts)
```

**Three details in that code:**

1. **Function words are excluded** by `_STOPWORDS` — a list of ~100 common
   words (`the`, `is`, `of`, `and`…). Without this, most of any English
   answer is function words that appear in every source, so every answer
   would score as well-grounded.
2. **Single characters are excluded** (`len(w) > 1`), so "a" and "I" do not
   count.
3. **Duplicates are counted once** (`set(...)`), so an answer that repeats
   one grounded phrase does not have its novelty inflated.

**Worked example.** Answer: "Ramsey was born in 1984 and is younger than
Ranaldo". Passage: "Ramsey was born in 1984 in Minneapolis and works with
Lee Ranaldo."

```
answer content words  : {ramsey, was, born, in, 1984, and, is, younger, than, ranaldo}
                         → after stopwords + len>1:
                           {ramsey, born, 1984, younger, than, ranaldo}     (6)
passage content words : {ramsey, was, born, in, 1984, minneapolis, and, works, with, lee, ranaldo}
                         → after stopwords + len>1:
                           {ramsey, born, 1984, minneapolis, works, lee, ranaldo}  (7)

novel words in answer = {younger, than}                        (2)
novel_content_rate    = 2 / 6 = 0.333
grounding_trust       = 1 - 0.333 = 0.667
```

**This asks a different question** from the checker in step 8. Step 8 asks
"is this specific fact supported?" — a hard question when the passage is
long. Grounding asks "how much of this answer came from the source at all?"
— an easier question that works on the whole answer.

**Only the top passage is used**, not all five. Using all five destroys the
signal — the other passages bring their own vocabulary, which soaks up the
answer's genuinely new words and makes every answer look the same.

**This component is the one that carries long answers.** On RAGTruth, where
answers run 110–160 words, the per-claim signals were almost useless
(ROC-AUC 0.478), while this one alone separates the two kinds of answer
well (0.715 / 0.776 / 0.808 across the three task types).

### The fusion arithmetic, step by step

```python
raw = {
    "similarity": cosine,
    "retrieval_top1": ret_top1,
    "nli": avg_nli,
    "coverage": coverage,
    "eedc": eedc,
    "detector_trust": detector_trust,
    "grounding": grounding,
}

parts = []
used_weights = {}
for c in COMPONENTS:
    w = float(weights.get(c, 0.0))
    if w == 0.0: continue
    if c == "grounding" and not grounding_used: continue
    if no_source and c in evidence_dependent: continue
    mu = float(mean.get(c, 0.0))
    sd = float(std.get(c, 1.0)) or 1.0
    if abs(sd) < 1e-9: sd = 1.0
    z = (raw[c] - mu) / sd
    sg = float(signs.get(c, -1.0 if c in TRUST_COMPONENTS else 1.0))
    parts.append(w * sg * z)
    used_weights[c] = w
risk = sum(parts) if parts else 0.0
prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, risk))))
confidence = 1.0 - prob
```

**A full worked example**, using the end-of-document trace — 1 claim
supported, 1 unknown, cosine 0.64, top passage 0.81, EEDC 0.58, detector
probability 0.38, grounding 0.78:

| Component | raw | mean | std | z = (raw−mean)/std | weight | sign | w·sign·z |
|---|---|---|---|---|---|---|---|
| similarity | 0.64 | 0.6186 | 0.1398 | +0.153 | 0.5711 | +1 | **+0.087** |
| retrieval_top1 | 0.81 | 0.6016 | 0.3654 | +0.570 | 0.1188 | +1 | **+0.068** |
| nli | 0.94 | 0.5428 | 0.2672 | +1.487 | 1.2497 | −1 | **−1.858** |
| coverage | 0.50 | 0.6095 | 0.2737 | −0.400 | −1.0345 | −1 | **−0.414** |
| eedc | 0.58 | 0.4609 | 0.0522 | +2.284 | −0.0749 | +1 | **−0.171** |
| detector_trust | 0.62 | 0.8103 | 0.2300 | −0.828 | 0.3228 | −1 | **+0.267** |
| grounding | 0.78 | 0.4921 | 0.2662 | +1.082 | 0.5887 | −1 | **−0.637** |

```
risk = 0.087 + 0.068 − 1.858 − 0.414 − 0.171 + 0.267 − 0.637
     = −2.657

P(hallucination) = sigmoid(−2.657) = 1/(1+e^2.657) = 1/14.25 = 0.0655
confidence = 1 − 0.0655 = 0.9345
```

**Two things this table shows that are easy to miss.**

First, the **weights and the signs are separate**. `coverage` has weight
−1.0345 *and* sign −1, so they multiply to a **positive** contribution
(+0.414). A negative weight on a negative sign means "more coverage is
safer", which is right — but it is not the same as a positive weight.

Second, the **largest single contributor is `nli` at −1.858**. The checker's
verdict dominates the score. Grounding is second at −0.637. The EEDC score
from step 4 contributes only −0.171, because its standard deviation
(0.0522) is tiny — after standardisation it barely moves.

> **A note on the weights.** The `sign` map and the `weights` map were both
> fitted together on the RAGTruth tuning half. The `nli` weight (1.2497) and
> `coverage` weight (−1.0345) have opposite signs *and* opposite configured
> signs, so they end up contributing in the same direction — which is
> consistent with both being trust signals.

### Two components that get dropped

Some numbers make no sense in certain runs, and including them would quietly
corrupt the score:

- **Grounding** is dropped when there is no legitimate source to compare
  against. Comparing against an answer key would be measuring the label.
- **The three passage-dependent numbers** (top score, checker confidence,
  supported share) are dropped when there are no passages at all. With no
  source, every claim comes back "unknown" and these become fixed constants,
  not measurements. Including them would shift every such row by the same
  amount — a shift that looks like a measurement but measures nothing.

### Input → Output

| | |
|---|---|
| **In** | everything gathered in steps 3, 4, 7 and 8 |
| **Out** | one confidence number (1.0 = fully trusted), plus the hallucination probability |

---

# Step 10 — Fix the Wrong Parts

**File:** `src/steps/step10_regeneration.py`
**What it does:** finds the claims the checker rejected and replaces just
those, leaving everything else exactly as it was.

```
        The answer, the claims, the passages
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Which claims were REJECTED? │
        │  (contradicted, and the      │
        │   checker was confident)      │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Find each claim's exact     │
        │  spot in the answer          │
        │  (usually works, sometimes   │
        │   not — if not found, leave   │
        │   that claim alone)          │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Pick a replacement:         │
        │  the best-matching SENTENCE  │
        │  from that claim's own       │
        │  passage — never the whole    │
        │  passage                     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  TWO GUARDS must pass:       │
        │                              │
        │  1. Right length?            │
        │     Not wildly longer than   │
        │     the claim it replaces    │
        │                              │
        │  2. Right subject?           │
        │     The claim's main name    │
        │     must appear in the       │
        │     replacement              │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Swap the claim for the      │
        │  replacement.                │
        │  EVERYTHING ELSE UNTOUCHED.  │
        └──────────────┬───────────────┘
                       ▼
              the corrected answer
```

### The three outcomes

| Result | When |
|---|---|
| `EDITED` | at least one claim was replaced |
| `ABSTAIN` | a claim was rejected but nothing usable could replace it — the answer is left alone |
| `KEEP` | nothing was confidently rejected |

### Finding the claim's exact spot

The claim text and the answer text rarely match character for character,
because the splitter strips and rejoins sentences. So there are two ways to
find it:

```python
# Fast path: the claim is a verbatim substring.
idx = answer.find(claim_text)
if idx >= 0:
    return (idx, idx + len(claim_text))

# Slow path: match on normalised text, then map the offset back.
target = _normalise(claim_text)              # lowercased, whitespace collapsed
norm_answer = _normalise(answer)
pos = norm_answer.find(target)
while pos >= 0:
    start = _orig_offset(answer, pos)
    if start is not None:
        end = start + len(claim_text)
        end = _extend_to_word_end(answer, end)
        return (start, end)
    pos = norm_answer.find(target, pos + 1)
return None
```

`_normalise` collapses runs of whitespace into one space, lowercases, and
strips the ends.

`_orig_offset` maps a position in the *normalised* text back to a position in
the *original* text, by walking the original character by character and
counting non-whitespace characters:

```python
def _orig_offset(original, norm_pos):
    seen = 0; in_ws = False
    for i, ch in enumerate(original):
        if ch.isspace():
            if not in_ws:
                if seen == norm_pos: return i
                seen += 1; in_ws = True
        else:
            in_ws = False
            if seen == norm_pos: return i
            seen += 1
    if seen == norm_pos: return len(original)
    return None
```

`_extend_to_word_end` then pushes the end forward past any trailing
punctuation, so the whole word gets replaced rather than leaving a
stray comma behind:

```python
def _extend_to_word_end(text, end):
    while end < len(text) and not text[end].isspace():
        end += 1
    return end
```

**If the claim cannot be found**, it is **skipped**, not deleted. Silently
removing content that was never identified is exactly the failure this step
is supposed to avoid.

### Choosing the replacement

```python
def _replacement_for(claim, evidence):
    if claim.evidence:
        best = max(claim.evidence, key=lambda e: e.score)
        text = (best.text or "").strip()
        if text:
            return _best_sentence(text, claim.text)
    return ""
```

Two decisions here:

- **It uses the evidence attached to *that specific claim*** — not the
  query's best document. Using the query's best document is what produced
  the uninformative substituted passages.
- **It uses only the best-matching *sentence*** of that evidence, never the
  whole passage:

```python
def _best_sentence(evidence_text, claim_text):
    sentences = split_sentences(evidence_text)
    if not sentences: return evidence_text.strip()
    claim_tokens = set(re.findall(r"\w+", claim_text.lower()))
    if not claim_tokens: return sentences[0]
    best = max(sentences,
               key=lambda s: len(claim_tokens & set(re.findall(r"\w+", s.lower())))
                            / len(claim_tokens))
    return best.strip()
```

The score is **the fraction of the claim's words that appear in the
sentence** — so the winning sentence is the one that covers the most of the
claim.

### Guard 1 — the length check

```python
def _replacement_is_proportionate(replacement, claim_text):
    claim_words = len(claim_text.split())
    repl_words  = len(replacement.split())
    if claim_words == 0: return False
    return repl_words <= max(12, claim_words * 4)
```

The limit is `max(12, 4 × claim_words)` — so it is **relative**, with a
small absolute floor of 12 words for the cases where a short claim genuinely
needs a longer sentence.

**Worked examples:**

| Claim | Words | Max allowed | Verdict |
|---|---|---|---|
| "Memphis International Airport" | 3 | max(12, 12) = 12 | a 12-word sentence passes, a 25-word one does not |
| "Ramsey was born in 1984" | 5 | max(12, 20) = 20 | a 20-word sentence passes |

> An earlier version allowed a 40-word minimum. That let a 25-word passage
> replace a 3-word claim, and turned "Memphis International Airport" into a
> paragraph about Kumasi Airport.

### Guard 2 — the subject check

```python
def _replacement_is_relevant(replacement, claim_text):
    entities = [w for w in re.findall(r"\b[A-Z][\w'-]*", claim_text)
                if w.lower() not in _STOPWORDS]
    if entities:
        repl_lower = replacement.lower()
        return any(e.lower() in repl_lower for e in entities)

    claim_tokens = {t for t in re.findall(r"\w+", claim_text.lower())
                    if t not in _STOPWORDS}
    if not claim_tokens: return True
    repl_tokens = {t for t in re.findall(r"\w+", replacement.lower())
                   if t not in _STOPWORDS}
    return len(claim_tokens & repl_tokens) / len(claim_tokens) >= 0.5
```

**The rule has two branches:**

- **If the claim has a capitalised name** (after removing stopwords like
  "The"), that name must appear somewhere in the replacement. The check is
  `e.lower() in repl_lower` — a **substring** test on the whole entity, so
  "Lee" does not match "Leeds".
- **If it has no capitalised name**, fall back to word overlap: at least
  **50%** of the claim's content words must appear in the replacement.

**Worked examples:**

| Claim | Replacement | Result |
|---|---|---|
| "Mary Ramsey is younger than Lee Ranaldo" | "Mary Ramsey was born in 1984" | **passes** — the subject "Mary" (and "Ramsey") appears, even though only 2 of 5 content words overlap |
| "Mary Ramsey is younger than Lee Ranaldo" | "The airport is near Kumasi" | **fails** — "Mary" and "Ramsey" are both absent |
| "the answer is correct" | "The result holds" | falls to the overlap branch; needs ≥50% overlap |

**Why the subject check exists.** The checker cannot say "this passage is
about something else". It can only say "this passage does not confirm this
claim". So it can wrongly reject a correct claim when the retrieved passage
is about a different subject — and then the replacement would swap a right
answer for a wrong one.

> Why not just compare all the words? Because a good replacement often gives
> a *different fact about the same subject* — "Mary Ramsey is younger than
> Lee Ranaldo" → "Mary Ramsey was born in 1984". That shares only 2 of 5
> content words but is completely on topic. Comparing the subject handles
> this.

### Putting the pieces together

```python
spans = []
skipped = 0
guard_reason = ""
for c in confident:
    span = _find_span(answer, c.text)
    if span is None:
        skipped += 1; continue
    replacement = _replacement_for(c, evidence)
    if not replacement:
        skipped += 1
        guard_reason = guard_reason or "no_usable_replacement"
        continue
    if not self._replacement_is_proportionate(replacement, c.text):
        skipped += 1
        guard_reason = guard_reason or "replacement_not_proportionate"
        continue
    if not self._replacement_is_relevant(replacement, c.text):
        skipped += 1
        guard_reason = guard_reason or "replacement_not_relevant"
        continue
    spans.append((span[0], span[1], replacement))

if not spans:
    return answer, "ABSTAIN", guard_reason or "claim_not_located"
```

**The splicing**, from the end backwards so earlier offsets stay valid:

```python
spans.sort(key=lambda s: s[0])
pieces = []
cursor = 0
for start, end, replacement in spans:
    if start < cursor:
        continue                    # overlapping span — skip it
    pieces.append(answer[cursor:start])   # the untouched part before
    pieces.append(replacement)             # the new text
    cursor = end
pieces.append(answer[cursor:])             # the untouched tail
final = " ".join(p.strip() for p in pieces if p.strip())
```

Everything outside the replaced spans is copied **verbatim**. That is the
guarantee the mode is named for.

### If one claim cannot be fixed, the others still get fixed

```
   Claim A: rejected, has a good replacement  ──►  replaced
   Claim B: rejected, no usable replacement    ──►  skipped
   Claim C: rejected, has a good replacement  ──►  replaced
```

An earlier version gave up on the whole answer when one claim was a problem.
That meant a single bad claim stopped every good fix in the same answer, and
the fix only fired on **16 of 500 rows**. Now it fires on most rows.

### Why "replace only the bad parts" and not "rewrite the whole answer"

There is a second mode, `evidence`, that rebuilds the answer using only the
claims the checker approved. That throws away the original wording of
everything it was not *explicitly* sure about, and when no claim is approved
it drops a raw passage in as the answer. On the length-matched HaluEval set,
that turned an already-correct answer into a worse one — a **BERTScore of
1.000 fell to 0.811** — which is why the measured improvement was negative.

`claim_edit` is the default because it works at the same fine-grained level
as the checker: the checker identifies *which* claim is wrong, and the fixer
corrects exactly that one.

### One important note about testing

When the answer was handed to us by the test set, this step **does not
change it**. A supplied answer is being *judged*, not improved — changing it
would mean the final score was for a different answer than the one the test
set provided.

So this step only really runs in `scripts/evaluate_mitigation.py`, which
deliberately does not supply an answer.

### Input → Output

| | |
|---|---|
| **In** | the answer, the claims with verdicts, the passages |
| **Out** | the corrected answer, plus a record of what was done and why |

---

# Step 11 — Make the Final Call

**File:** `src/steps/step11_final_gate.py`
**What it does:** decides one of three things — pass, try again, or give up.

```
     The confidence score          The claims
            │                          │
            └────────────┬─────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │  How many claims were rejected     │
        │  or unclear?                       │
        │  ─────────────                     │
        │  How many claims total?            │
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │  confidence ≥ 0.70                 │
        │  AND no rejected claims?           │
        │        → PASS  (accept it)         │
        ├────────────────────────────────────┤
        │  confidence < 0.30                 │
        │  OR half the claims rejected?      │
        │        → ABSTAIN  (refuse to answer)│
        ├────────────────────────────────────┤
        │  anything else → in between        │
        │        → REGEN  (go back and fix)   │
        └────────────────────────────────────┘
```

### The exact decision code

```python
score = self.ctx.confidence
unsupported = sum(1 for c in self.ctx.claims
                  if c.verdict in {"contradicted", "unknown"})
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

With the default thresholds (`pass_th = 0.70`, `abs_th = 0.30`,
`max_attempts = 2`), the full decision table is:

| confidence | unsupported ratio | Decision |
|---|---|---|
| ≥ 0.70 | 0.0 | **PASS** |
| ≥ 0.70 | > 0.0 but < 0.5 | **REGEN** (if attempts left), else PASS if ≥ 0.5 |
| ≥ 0.70 | ≥ 0.5 | **ABSTAIN** |
| 0.30 – 0.70 | 0.0 | **REGEN** (if attempts left), else PASS if ≥ 0.5 |
| 0.30 – 0.70 | > 0.0 but < 0.5 | **REGEN** (if attempts left), else PASS if ≥ 0.5 |
| 0.30 – 0.70 | ≥ 0.5 | **ABSTAIN** |
| < 0.30 | any | **ABSTAIN** |

Note the `max(1, len(claims))` — with zero claims the ratio is 0, not a
division error.

### The deliberate asymmetry

To **pass**, the answer needs *both* a high confidence score *and* zero
rejected claims. A high score with even one rejected claim falls into the
middle and triggers another fix attempt.

The reason: the confidence score is a *blend* of several signals. It can
look high because the passage matched well, even when an individual fact
failed the check.

### The middle case

If the answer is neither clearly good nor clearly bad, the pipeline tries to
fix it — up to **2 times**. If the attempts run out, it passes anything with
a score of 0.5 or above, and refuses anything below.

**Two separate attempt counters.** The gate keeps its own in
`ctx._regen_attempts` (set on line 35, read on line 27). The orchestrator
keeps another in `src/pipeline.py:248`. The gate needs to know it has
exhausted its budget even when it is reached from a different path.

### Input → Output

| | |
|---|---|
| **In** | the confidence score, the claims |
| **Out** | the decision (`PASS` / `REGEN` / `ABSTAIN`), plus the reasoning |

---

# Step 12 — Write the Explanation

**File:** `src/steps/step12_explainable_response.py`
**What it does:** turns everything into a human-readable report and a
machine-readable JSON file.

```
   The question, the answer, the confidence,
   the decision, the claims + verdicts,
   the passages, the decision history
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     JSON file            Markdown report
          │                   │
          ▼                   ▼
   report_json          report_markdown
```

### What the Markdown report looks like

```markdown
# Hallucination Pipeline Report

**Query:** Who is the youngest person mentioned in the passage?
**Answer:** Mary Ramsey was born in 1984 and is younger than Lee Ranaldo.
**Confidence:** 0.16
**Decision:** `ABSTAIN`

## Claims

1. *(supported, conf=0.94)* Mary Ramsey was born in 1984.
   - provenance: `aggregated`
   - citation: `doc_0042`
2. *(unknown, conf=0.30)* Mary Ramsey is younger than Lee Ranaldo.
   - provenance: `aggregated`
   - citation: `-`

## Evidence

- **doc_0042** (score=0.812): Mary Ramsey was born in 1984...

## Decision history

- step 10: {'mode': 'provided_passthrough', 'unsupported_ratio': 0.5, ...}
- step 11: {'confidence': 0.16, 'unsupported_ratio': 0.5, 'decision': 'ABSTAIN', ...}
```

### Input → Output

| | |
|---|---|
| **In** | everything on the context |
| **Out** | `report_markdown` and `report_json` |

---

# Step 13 — Measure How the Run Did

**File:** `src/steps/step13_evaluation.py`
**What it does:** calculates the scores for this row and saves them.

```
   The claims, the passages, the decision,
   the confidence, and the test set's
   correct answer + label
                │
   ┌────────────┼────────────┬──────────────┐
   ▼            ▼            ▼              ▼
 Retrieval   Claims       Overall      Saved to
 scores      scores       scores       data/metrics.json
```

### The three groups of scores

**Retrieval** — did we find the right passages?

| Score | Formula | Meaning |
|---|---|---|
| Recall@5 | `\|top5 ∩ gold\| / \|gold\|` | of the passages that should have been found, how many did we get |
| Precision@5 | `\|top5 ∩ gold\| / 5` | of the 5 we returned, how many were actually right |
| MRR | `1 / (position of first correct passage)` | how high up the list the first right passage was |
| nDCG | `DCG / IDCG` | like MRR but rewards getting more of them in the right order |

**The nDCG formula** — each correct passage is credited **once**, and its
value falls off as it appears lower down:

```python
dcg = 0.0
credited = set()
for i, x in enumerate(retrieved_ids, 1):
    if x in gold_ids and x not in credited:
        credited.add(x)
        dcg += 1.0 / math.log2(i + 1)
idcg = sum(1.0 / math.log2(i + 1) for i in range(1, len(gold_ids) + 1))
ndcg = dcg / idcg
```

The `credited` set is essential. Without it, several retrieved documents
matching the same correct passage would each add a gain term, and the score
could exceed 1.0 — which nDCG cannot do by definition.

**Worked example.** One correct passage, retrieved at position 3:

```
DCG = 1/log2(3+1) = 1/2 = 0.5
IDCG = 1/log2(1+1) = 1/1 = 1.0        (the ideal: it would be first)
nDCG = 0.5/1.0 = 0.5
```

If the test set does not say which passages were correct, all four are
reported as `null`. That is deliberate — scoring our own results against
ourselves would always give a perfect 1.0 and look like a great retriever
when it means nothing.

**Claims** — did the checker do its job?

```python
tp = fp = fn = 0
for c in claims:
    supported = c.verdict == "supported"
    agrees = _agrees_with_gold(c.text)
    if supported and agrees:        tp += 1
    elif supported and not agrees:  fp += 1
    elif not supported and agrees:  fn += 1

prec = tp / (tp + fp) if (tp + fp) else 0.0
rec  = None if label == 1 else (tp / (tp + fn) if (tp + fn) else 0.0)
f1   = None if rec is None else 2*prec*rec/(prec+rec)
```

`_agrees_with_gold` is a loose check — the claim shares **any** content token
with the correct answer:

```python
def _agrees_with_gold(text):
    toks = set(_tokenize(text))
    return bool(toks & gold_tokens)
```

| Situation | What is reported |
|---|---|
| we have the correct answer | precision, recall, F1, and the support rate |
| the row is a *hallucinated* answer | precision and F1 only — **recall is `null`** |
| we have no correct answer | the support rate only; the other three are `null` |

**Why recall is null for a hallucinated row.** The test set tells you the
answer is wrong, but not *which* of its facts are wrong. There is nothing to
recall against, so reporting a number would be making one up.

> An earlier version set precision, recall and F1 all to the same number.
> That number was the support rate — which is neither a recall nor an F1.

**Overall**

| Score | Formula | Meaning |
|---|---|---|
| verification_success_rate | `1.0` if no claim was `contradicted`, else `0.0` | did the checker find nothing wrong |
| end_to_end_accuracy | `1.0` if the decision was `PASS`, else `0.0` | did the pipeline accept the answer |

### Where the results go

Appended to `data/metrics.json`. The external evaluation script reads that
file and works out the averages, plus BLEU, ROUGE-L and BERTScore, which
need the reference answer and so are computed outside the pipeline.

### Input → Output

| | |
|---|---|
| **In** | the run's results plus the test set's correct answer and label |
| **Out** | one row of scores, appended to `data/metrics.json` |

---

# The Loop That Goes Back

When step 11 says "try again", the pipeline does not just continue. It goes
**back** to the beginning of the checking part and checks the fixed answer
all over again.

```
   ③ → ⑤ → ⑧ → ④ → ⑨ → ⑩ → ⑪
                          │
                    step 11 says REGEN
                          │
                          ▼
              ┌───────────────────────┐
              │  Go back to step ③    │
              │  and check the        │
              │  FIXED answer again   │
              │                       │
              │  Allowed: 2 times     │
              └───────────┬───────────┘
                          │
                          ▼
                   ⑫ → ⑬  →  done
```

**Which steps run again:** 3, 5, 8, 4, 9, 10, 11.

**Which do not:** 1, 6, 7, 2. The question has already been understood, the
passages have already been found, and `claim_edit` fixes the existing answer
rather than writing a new one.

**Why re-check at all?** Because a fix could introduce a new problem. The
new sentence might itself be unsupported. The gate should never pass an
answer it has not actually checked.

---

# A Worked Example — One Question Through All 13 Steps

**Question:** *"Who is the youngest person mentioned in the passage, and when
were they born?"*

```
 ①  Kind: "factual"  |  Names found  |  rewrite made but not used
 ⑥  Searched with both methods, merged → passages d7, d2, d19, d4, d31
 ⑦  Re-scored → d2 (0.81), d7 (0.79), d19 (0.61), d4 (0.58), d31 (0.44)
 ②  Prompt built with those 5 passages →
        "Mary Ramsey was born in 1984 and is younger than Lee Ranaldo."
 ③  cosine 0.64 | claim_count 1 ⚠ | overlap 0.42 | length 14
        → guess: 38% chance of hallucination
 ⑤  Two claims found:
        "Mary Ramsey was born in 1984"            → tag: aggregated
        "Mary Ramsey is younger than Lee Ranaldo"  → tag: aggregated
 ⑧  Checked each claim against each passage:
        claim 1 vs d2 → SUPPORTED (0.94), cite d2
        claim 2 vs d2 → UNKNOWN  (0.51)
        Result: 1 supported, 0 contradicted, 1 unknown
 ④  Entropy of (1 supported, 1 unknown) = 0.6309
        phi = (-1.0)(0.6309) + (-0.7)(0.36) + (-0.5)(0.36) + 1.0 = -0.0629
        → EEDC raw score 0.4843
 ⑨  All 7 components z-scored, signed and weighted:
        similarity +0.087 | retrieval_top1 +0.068 | nli -1.858
        coverage -0.414 | eedc -0.171 | detector_trust +0.267
        grounding -0.637
        → risk = -2.657
        → 6.55% chance of hallucination → confidence 0.9345
 ⑩  Answer came from the test set → left unchanged, marked "provided"
 ⑪  confidence 0.9345 ≥ 0.70 BUT unsupported ratio = 0.5 ≠ 0
        → falls to the middle → REGEN (but attempts are used up)
        → final decision: PASS if ≥ 0.5, so PASS
 ⑫  Report written: 2 claims, 1 supported, 1 unknown, cites d2
 ⑬  Scores saved
```

### What this example shows

**The answer was actually correct.** Both facts came from passage `d2`, and
the fused confidence was high (0.93) because the checker confirmed one claim
outright and the words matched the source well.

**But the gate did not pass it cleanly.** Half the claims counted as
"unclear", so the decision rule's "zero unsupported claims" condition failed
and the answer fell into the middle band. It only ended up as `PASS`
because the confidence was above 0.5 once the retry budget ran out.

That is the real limit of the system: **the checker's uncertainty is not the
same as the answer being wrong.** Being unsure is treated as being wrong.
For a system that refuses to answer, that is the safer error — but it means
the pipeline can hesitate over good answers, and it is the same uncertainty
that caps the improvement from step 10.
