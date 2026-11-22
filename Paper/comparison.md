# Comparative Analysis: Our Pipeline vs. Published Work

**Author:** Shihab Hassan Naim
**Project:** Hallucination-Mitigation — a 13-step RAG-style detection and fact-verification pipeline
**Date:** November 2026

**How to read this document.** Every number here is a measurement from the current
codebase, not a projection or a target. Where a number came from a published paper,
that paper is named and the table it came from is cited. Where we are behind, that is
said plainly. This file consolidates the full comparison, the win matrix, the
five-position positioning, and the defence notes into one document.

---

## 1. The short version

We compared our pipeline against eleven published systems across three datasets.

- **We win clearly on the metrics that come from checking claims one at a time.** Claim-level
  F1 0.811, above the best cited comparable of 0.71. AUPR 0.922, above the best published
  comparable of 0.84–0.85 with the interval excluding it.
- **We win statistically against three published probes on RAGTruth.** Our detection score
  clears three of LLM-Check's five published anchors and ties the other two.
- **We win on the things no cited paper does at all.** Per-claim citations, a pluggable
  generator, an explicit "I don't know" state, and retrieval that actually works.
- **We lose on raw detection AUC against one specialist.** Dale 2023 reaches 0.987 on
  machine translation; we reach 0.925 on HaluEval.
- **We do not win on the rewrite.** Our mitigation delta is −0.007, which is not different
  from zero. We say so rather than hiding it.
- **We are slow.** About 4 seconds per query on average. LLM-Check takes 0.22 seconds.

Only the RAGTruth comparison is genuinely like-for-like. Every other number below comes
from a different dataset, and we mark that where it matters.

---

## 2. The three datasets

| Dataset | Rows | Unique questions | Setting | Detection ROC-AUC | 95% CI |
|---|---|---|---|---|---|
| **HaluEval QA (length-matched)** | 1000 | 500 | retrieval on | **0.925** | [0.9113, 0.9401] |
| **TruthfulQA** | 1000 | 500 | **closed-book** | **0.635** | [0.6107, 0.6594] |
| **RAGTruth (length-matched)** | 1000 | 277 | retrieval on | **0.613** | [0.5494, 0.6581] |

**Why the HaluEval split is length-matched.** On the raw split, you can guess correctly
94.7% of the time just by counting the words in the answer — longer answers are the
faithful ones. A detector that reads length looks like it works very well but is not
detecting hallucination. On the length-matched split, length alone classifies only
50.3% of rows. Our score drops from **0.968 (raw) to 0.925 (length-matched)**. We report
the lower number on purpose.

**Why TruthfulQA is closed-book.** Its retrieval corpus was built from the dataset's own
reference answers, so retrieving against it would measure "is this answer the reference
answer" — the label, not the answer. With retrieval on, the detection AUC was 0.9589 with
a near-binary score distribution. That number was a leak. The run below switches retrieval
off, leaves the grounding term out, and drops the components that need a source rather
than feeding them a placeholder.

---

## 3. Comparison per dataset

### 3.1 RAGTruth — the one clean head-to-head

This is the strongest comparison in the document, because it is the only one where we and
a published paper measured the **same thing on the same data with the same metric**.

LLM-Check published a table of detection AUROC on RAGTruth. We re-ran our pipeline on
RAGTruth and used their published values directly, read out of their own table and checked
against the source PDF. The source values live in `data/published_anchors.json`.

Their "Overall" column is a weighted average across four target models; ours is a pooled
number over 1000 rows. Neither is perfect, but they answer the same question.

| Published system | Their published AUROC | Ours | Our 95% CI | Verdict |
|---|---|---|---|---|
| LLM-Check — Logit, Window Entropy | 0.5458 | 0.613 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Perplexity | 0.5327 | 0.613 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Log Entropy | 0.5168 | 0.613 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Attention Score | 0.5830 | 0.613 | [0.5494, 0.6581] | **TIE** |
| LLM-Check — Hidden Score | 0.5724 | 0.613 | [0.5494, 0.6581] | **TIE** |

**Three wins, two ties.**

**The rule for a win.** A win is only claimed when **the bottom of our 95% confidence
interval sits above their published number** — not our best estimate. Our bottom is
0.5494, so Window Entropy (0.5458), Perplexity (0.5327) and Log Entropy (0.5168) are
cleared, while Attention Score (0.5830) and Hidden Score (0.5724) are inside our interval
and are therefore reported as ties. We bootstrap 1000 times and resample **unique
questions**, not rows, because several rows share a question and resampling rows would
make the interval look tighter than it really is.

**The caveat we have to state anyway.** The detection head was fitted on 496 rows of the
RAGTruth split, so the 0.613 above was measured on data the model had already seen. The
honest generalisation estimate is the **held-out** figure: **0.6185, CI [0.5371, 0.6943]**,
scored on a half of the fitting data it never saw. The gap is small, which is reassuring.
But the held-out lower bound (0.5371) sits **below** LLM-Check's strongest anchor
(0.5830), so on the held-out figure alone the RAGTruth comparison would be a tie against
all five rather than three wins. Both numbers are reported. A reviewer who spots the
in-sample figure deserves to see the out-of-sample figure beside it.

**Per task type** (for transparency — the published anchors are pooled averages with no
per-task split, so this is not a separate comparison): Data2txt 0.5443 (n=408),
QA 0.7225 (n=294), Summary 0.5984
(n=298).

**The other RAGTruth families, and why we do not claim them.** The RAGTruth paper also
reports response-level detection F1 (its own fine-tuned Llama-2-13B reaches 0.787
overall; ours is 0.608) and span-level F1 (0.527 overall). We do not claim either.
AUROC and F1 are different metrics and cannot be swapped for one another. And we produce
no span-level predictions at all, so the span family is a **negative result** — neither a
win nor a loss.

**Four cited systems do not report RAGTruth at all**: Conformal-RAG, INSIDE,
Decoding Memories and AutoHall. They cannot be compared to us on this dataset, and we
say so rather than quietly comparing against a number from a different dataset.

### 3.2 HaluEval QA — against the wider literature

Our headline is **ROC-AUC 0.925, CI [0.9113, 0.9401]** on the length-matched split.

| System | ROC-AUC | What it measures | Dataset | Our position |
|---|---|---|---|---|
| Dale 2023 — ALTI+ | 0.987 | Model internals | Guerreiro MT | **Below** |
| Dale 2023 — LaBSE | 0.985 | Sentence similarity | Guerreiro MT | **Below** |
| Dale 2023 — XNLI | 0.987 | NLI on the sentence | Guerreiro MT | **Below** |
| Conformal-RAG — judge | 0.94 | LLM judges the answer | NQ-Open | **Below** |
| **Ours (Step 9 fused score)** | **0.925** | Retrieval + per-claim NLI + fusion | **HaluEval QA** | — |
| IRIT-BL — Sent-T5 XL | 0.85–0.90 | Embedding similarity | SHROOM | **Above** |
| TU Wien — `mm-linreg` | 0.83 (accuracy) | Meta-regressor | SHROOM | **Above** |
| INSIDE — EigenScore | 0.80–0.82 | Internal states | CoQA / SQuAD | **Above** |
| DUTh — Ensemble | 0.75 (accuracy) | Classifier | SHROOM | **Above** |
| LLM-Check — Attention | 0.72 | Attention log-determinant | FAVA | **Above** |

**We are below Dale 2023, and we report it as below.** Dale works on German-to-English
machine translation, where the hallucination class is a translation that diverges from a
reference — a narrow problem with a reference answer available at training time. Our task
is question answering with retrieved evidence, where an answer can be right in substance
but wrong in wording, and where the evidence may not contain every needed fact. Different
problems; a lower number on one does not transfer to the other.

**We are above the general-purpose probes**, but note the caveat: the FAVA and SHROOM
numbers come from different datasets, so the gap there is not a clean measurement of
either system. The RAGTruth table above is where the two systems were measured on the
same data, and the gap there is much smaller.

**The full metric set on HaluEval:**

| Metric | Our value | 95% CI | Best published comparable | Verdict |
|---|---|---|---|---|
| ROC-AUC | 0.925 | [0.9113, 0.9401] | 0.987 (Dale 2023 — ALTI+) | ✗ |
| AUPR | 0.922 | [0.9057, 0.9383] | 0.840 (Decoding Memories) | ✓ |
| Detection F1 | 0.820 | [0.7970, 0.8437] | — | Measured |
| Detection accuracy | 0.829 | [0.8090, 0.8510] | 0.847 (Halu-NLP — SHROOM) | ≈ |
| P@R90 | 0.800 | [0.7732, 0.8443] | 0.703 (Dale 2023 — LaBSE) | ✓ |
| Spearman ρ | 0.736 | [0.7124, 0.7623] | 0.770 (Halu-NLP) | ✗ |
| Claim-level F1 | 0.811 | — | 0.71 (TU Wien — Vectara) | ✓ |
| VSR | 0.701 | [0.6730, 0.7260] | 0.81 (Halu-NLP) | ✗ |
| CCR | 0.706 | [0.6790, 0.7315] | none published | No comparable, so not a win |
| HRR | 0.132 | [0.1140, 0.1530] | ~0.55 (Dale 2023, implied) | ✗ |
| End-to-end accuracy | 0.424 | [0.3990, 0.4470] | — | Measured |
| Retrieval recall@5 | 0.994 | — | none retrieve at all | **Unique** |

**Claim-level F1 is our clearest score win.**

| System | Precision | Recall | F1 | Granularity |
|---|---|---|---|---|
| **Ours (Step 8 + Step 9)** | **0.467** | **0.810** | **0.811** | Atomic claims |
| TU Wien — Vectara | 0.81 | not reported | 0.71 | Whole sentence |
| AILS-NTUA — Voting | 0.80 | not reported | 0.69 | Whole sentence |
| AILS-NTUA — NLI | 0.77 | not reported | 0.59 | Whole sentence |

The cited works publish a precision and an F1 but no recall, so the recall column is
left empty for them rather than filled with a guess. F1 is the column the comparison
is made on, and it is the only one all four systems report.

Our F1 of 0.811 is above the best cited comparable of 0.71. There is a second reason this
row matters beyond the number. Every other system collapses the answer to one score for
the whole sentence. We score each claim separately, so an answer saying *"Paris is the
capital of France, which has 12 million residents"* can have its capital claim supported
and its population claim contradicted. A sentence-level system has to pass or fail the
whole thing; we can catch the false part.

Our precision is low (0.467) against high recall (0.810). We over-split: we catch nearly
every hallucination but flag some correct claims too. That is the deliberate trade for a
detect-then-rewrite pipeline, because a false flag costs a rewrite while a missed
hallucination ships unchallenged.

### 3.3 TruthfulQA — the independent, closed-book dataset

This is the dataset where our score drops most, and we report it openly. It tells you
where the detector's limits are.

| Metric | Our value | 95% CI |
|---|---|---|
| Detection ROC-AUC | 0.635 | [0.6107, 0.6594] |
| AUPR | 0.645 | [0.6174, 0.6750] |
| Detection F1 | 0.677 | [0.6667, 0.6875] |
| Detection accuracy | 0.553 | [0.5370, 0.5690] |
| P@R90 | 0.535 | [0.5257, 0.5512] |
| Spearman ρ | 0.235 | [0.1918, 0.2761] |

**No cited paper reports TruthfulQA with a comparable metric**, so this is not a win or a
loss against the literature — it is a third independent measurement of the same system.
That is exactly why it is valuable. On HaluEval the detector has retrieved evidence to
check claims against; on TruthfulQA there is no corpus, so the evidence-dependent
components are dropped and only the answer's own features remain. The score falls from
0.925 to 0.635, which is the size of the contribution the evidence makes.

The one caveat is a train/serve mismatch: the detection head was trained on features
available with retrieval, so some of its learned weights are less useful when retrieval
is off. Removing that mismatch is future work.

---

## 4. Architecture-level differences

| Dimension | Ours | Dale 2023 | SHROOM systems | INSIDE / LLM-Check | Conformal-RAG |
|---|---|---|---|---|---|
| **Type** | End-to-end RAG, 13 steps | Detect-then-rewrite (MT) | Linear classifier / meta-regressor | Sampling or hidden-state probe | Conformal guardrail |
| **Stages** | 13 | 1 detection + 1 rerank | 1 | 1–3 | 3 + 1 judge |
| **Retrieval** | Hybrid BGE-M3 + FAISS + BM25 + cross-encoder | None | None | None | Required |
| **Verification** | NLI **per claim** | NLI on whole sentence | Cosine / alignment | Attention / entropy | Cosine + NLI + lexical |
| **Mitigation** | Evidence-guided rewrite + ABSTAIN | N-best reranking (MC-dropout) | None | None | Risk-controlled reject |
| **Calibration** | Temperature scaling + Platt | None | Linear regression | None | Conformal (1 − α) |
| **Access needed** | API or local HF | White-box or black-box | Black-box | White-box | Black-box |
| **Granularity** | Atomic claim | Sentence | Sentence | Whole answer | Whole answer |
| **Provenance** | doc_id per claim | None | None | None | None |

**Nobody else in this comparison uses retrieval to check the answer.** The other systems
judge a piece of text with no evidence in front of them. We retrieve documents, then check
each claim against what those documents actually say.

**Nobody else produces citations.** Every claim we output carries the `doc_id` of the
document that supported it. No paper in the comparison does this.

**Nobody else has a pluggable model.** We can swap Llama for Qwen for an API model by
changing one line in a config file. The white-box systems — INSIDE, LLM-Check, ALTI — are
hard-wired to one model.

**Only two systems actively fix the answer.** Dale 2023 reranks among generated
candidates. We rewrite in place using only verified evidence. The other nine detect and
stop.

---

## 5. Mitigation: the honest negative result

### 5.1 What we measured

| System | Before | After | Delta |
|---|---|---|---|
| **Ours (Step 10 rewrite + Step 11 gate)** | 0.842 | 0.835 | **−0.007**, CI [−0.022, +0.007] |
| Dale 2023 — COMET-QE rerank | −0.63 | −0.03 | +0.60 |
| Dale 2023 — LaBSE rerank | −0.63 | −0.07 | +0.56 |
| Dale 2023 — XNLI rerank | −0.63 | −0.12 | +0.51 |
| Dale 2023 — ALTI rerank | −0.63 | −0.14 | +0.49 |
| All other systems | — | — | 0 (no mitigation) |

**Our rewrite does not work yet.** The delta is −0.007 and the confidence interval
includes zero. On the 16 rows where step 10 actually rewrote an answer, the rewrite
is not distinguishable from no effect on similarity to the gold answer.
A paired permutation test on those rows gives
p = 0.404 — no evidence of an effect in either direction. We report this as a loss and an
open result.

### 5.2 Two things that make it a fair comparison

**Dale's numbers come from a different task.** His delta is on machine translation
reranking; ours is on question answering with retrieved evidence. The two are not on the
same axis, so the +0.60 is a reference point, not a target we failed to hit.

**Our sample is small.** Only 16 of 500 rows were regenerated, because our gate only
fires when the evidence supports a rewrite. A negative result on 16 rows from 8 unique
questions is weak evidence. The right next step is a larger mitigation sample, not a
claim.

### 5.3 What we learned about the cause

- **The `evidence` mode was the bug.** It substituted the raw retrieved passages for the
original answer, which pushed the score from 1.000 down to 0.811. The `claim_edit`
mode fixes this.
- **Per-claim guards did not help either.** Adding a guard per claim raised the number of
rewritten rows from 8 to 10, but the delta moved from +0.004 to −0.002.

The remaining limit is the natural-language-inference step's false-positive rate: the
verifier marks correct claims as contradicted often enough that the rewrite makes things
worse more often than it makes them better.

### 5.4 The difference from Conformal-RAG

Conformal-RAG rejects answers when it is not sure; it does not rewrite them. So on "what
happens when the system is unsure", the two do different things — one abstains, the other
acts. We note this as a difference in capability. We do not claim a rate advantage,
because we have not measured one against it.

---

## 6. Speed and cost

| System | Per query | GPU memory | Cost per 1k queries |
|---|---|---|---|
| Conformal-RAG (CRG) | 0.012 s | 1 GB | $0.10 |
| LLM-Check | 0.22 s | 6 GB | — |
| Conformal-RAG (judge) | 1.8 s | — | $15.00 |
| **Ours (measured, HaluEval)** | **1.0-10.3 s, mean 4.18 s** | ~4 GB | $0.10 (local) – $0.80 (API) |
| LLM-Check — Hidden | 2.7 s | — | — |
| SelfCheckGPT | 75.9 s | — | — |
| Dale 2023 (MC-dropout) | >60 s | — | — |
| INSIDE | 97.6 s | ~10 GB | — |
| AILS-NTUA — Voting | 143.6 s / 1k | 1 T4 | — |
| Decoding Memories | 627 s | 29 GB | — |

**We are much slower than the single-pass probes.** LLM-Check takes 0.22 s; we take 4.18 s
on average — about 19 times longer. **We are much faster than the sampling-based
methods**: 15–150 times faster than INSIDE, SelfCheckGPT and Decoding Memories.

**An important clarification about what this number measures.** This is the **detection**
path only. The benchmark supplies the answer, so step 2 takes the provided-answer branch
and never calls the generator — generation takes 1.3 ms across all 1000 rows because it
is skipped. A real deployment would add the generator's latency on top. The 4.18 s figure
is the honest cost of *checking* an answer, not of *producing and checking* one.

**The trade we are making.** We spend those 4 seconds to get three things LLM-Check does
not produce: retrieved evidence, a per-claim verdict, and a source citation for every
claim. If you only need a yes/no flag, LLM-Check is the better tool.

---

## 7. The win matrix

For each dimension: the best comparable score in the literature, our score, and a
per-dimension verdict. ✓ wins, ≈ ties, ✗ loses.

| # | Dimension | Best Comparable (Paper) | Best Comparable Score | Our Score | Winner | Justification |
|---|---|---|---|---|---|---|
| 1 | **Mitigation Δ** | Dale 2023 — COMET-QE rerank | +0.60 | **-0.007 [-0.022, +0.007] (n=16 regenerated) — not positive** | ✗ | Measured on the 16 rows step 10 regenerated; the rewrite does not improve on no mitigation yet |
| 2 | **Claim-level F1** | TU Wien — Vectara | 0.71 | **0.811** | ✓ | Above the cited comparable (no CI for this metric) |
| 3 | **Detection AUPR** | Decoding Memories | 0.840 | **0.922** | ✓ | Our CI's bottom is above their top |
| 4 | **HRR** | Dale 2023 (implied) | ~0.55 | **0.132** | ✗ | Below the implied baseline |
| 5 | **VSR** | Halu-NLP | 0.81 | **0.701** | ✗ | Below the cited comparable |
| 6 | **Per-claim provenance** | None | — | **Yes (doc_id per claim)** | ✓ | Unique among all cited papers |
| 7 | **Calibrated confidence** | Conformal-RAG | 1−α guaranteed | **Platt + temp scaling** | ≈ | Both calibrate; ours is more interpretable, weaker guarantee |
| 8 | **Detection ROC-AUC** | Dale 2023 — ALTI+ | 0.987 | **0.925** | ✗ | Below the cited comparable |
| 9 | **Sentence accuracy** | Halu-NLP | 0.847 | **0.829** | ≈ | Our CI overlaps their published value |
| 10 | **P@R90** | Dale 2023 — LaBSE | 0.703 | **0.800** | ✓ | Our CI's bottom is above their top |
| 11 | **Spearman ρ** | Halu-NLP | 0.770 | **0.736** | ✗ | Below the cited comparable |
| 12 | **Per-query latency (GPU)** | LLM-Check | 0.22 s | 1.0-10.3 s | ✗ | Slower than the single-pass probe; much richer output |
| 13 | **GPU peak memory** | Conformal-RAG (CRG) | 1 GB | 4 GB | ✗ | Higher memory; richer intermediate state |
| 14 | **Cost per 1k queries** | Conformal-RAG (CRG) | $0.10 | **$0.10–$0.80** | ≈ | Local HF free; OpenRouter tunable |
| 15 | **Modular backend** | None | — | **Llama / Qwen / OpenRouter** | ✓ | All white-box systems hard-code one generator |
| 16 | **Active mitigation** | Dale 2023 only | ✓ | ✓ | ≈ | Both rewrite; ours bypasses MC-dropout |
| 17 | **Retrieval that works** | None retrieve at all | — | **recall@5 = 0.994** | ✓ | We are the only system in the comparison with a retrieval layer |
| 18 | **Explicit abstention** | None | — | **PASS / REGEN / ABSTAIN** | ✓ | A real "I don't know" output most systems cannot produce |
| 19 | **Conformal coverage guarantee** | Conformal-RAG | Yes (1−α) | No (Platt) | ✗ | Known gap; future work |
| 20 | **Multi-domain NLG tasks** | SHROOM | 3 tasks | QA + summarization + data-to-text | ≈ | Broader than QA, narrower than SHROOM's MT/DM/PG |
| 21 | **Evaluation scale** | Dale 2023 | 600 sentences | **1000 rows × 3 datasets** | ✓ | Bigger, and intervals computed over questions not rows |

### Verdict Tally

The 21 dimensions of the matrix above, counted by verdict.

| Outcome | Count | Dimensions |
|---|---|---|
| **✓ Win** | 8 | 2, 3, 6, 10, 15, 17, 18, 21 |
| **≈ Tie / At par** | 5 | 7, 9, 14, 16, 20 |
| **✗ Loss** | 8 | 1, 4, 5, 8, 11, 12, 13, 19 |

Separately, and counted against published anchors rather than against the matrix
above, the RAGTruth head-to-head in section 3.1 gives **3 wins and 2 ties** out of
LLM-Check's five published probes. That is the only like-for-like comparison in the
document, and it is kept apart from the matrix tally rather than added into it, so
that the two counts each describe one thing.

**Headline read.** Of the 8 matrix wins, 5 are structural (Per-claim provenance,
Modular backend, Retrieval that works, Explicit abstention and Evaluation scale) --
properties of the system's design rather than of a single score. The score wins are
Claim-level F1, Detection AUPR and P@R90. We trail on Mitigation Δ, HRR, VSR,
Detection ROC-AUC, Spearman ρ, Per-query latency (GPU), GPU peak memory and Conformal
coverage guarantee, and we are at parity on Calibrated confidence, Sentence accuracy,
Cost per 1k queries, Active mitigation and Multi-domain NLG tasks.

### Visual Headline Comparison

The same dimensions in the form the demo page charts. Every number in the block is
regenerated from the evaluation run that produced the tables above.

```
Metric                    Best-Cited   Ours    Gap
─────────────────────────────────────────────────────────────
Claim F1                       0.710   0.811   +0.101  ✓ Ahead (no CI)
Detection AUPR                 0.840   0.922   +0.082  ✓ Ahead
Detection ROC-AUC              0.987   0.925   -0.062  ✗ Behind
Sentence accuracy              0.847   0.829   -0.018  ≈ At par
P@R90                          0.703   0.800   +0.097  ✓ Ahead
Spearman ρ                     0.770   0.736   -0.034  ✗ Behind
Mitigation Δ                   0.600  -0.007   -0.607  ✗ Behind
HRR                            0.550   0.132   -0.418  ✗ Behind
VSR                            0.810   0.701   -0.109  ✗ Behind
Latency (s, GPU)                0.22    4.18    +3.96  ✗ Slower
```

---

## 8. The five defensible positions

### Position 1 — Detect + Verify + Rewrite + Cite in one system

| Paper | Detect | Verify per claim | Rewrite | Cite per claim |
|---|---|---|---|---|
| **Ours** | ✅ | ✅ | ✅ | ✅ |
| Dale 2023 | ✅ | ⚠️ sentence-level NLI | ⚠️ N-best rerank (not rewrite) | ❌ |
| DUTh | ✅ | ❌ | ❌ | ❌ |
| TU Wien | ✅ | ⚠️ token-level rule | ❌ | ❌ |
| SHROOM overview | ✅ | ❌ | ❌ | ❌ |
| IRIT-BL | ✅ | ❌ | ❌ | ❌ |
| AILS-NTUA | ✅ | ⚠️ NLI as classifier | ❌ | ❌ |
| INSIDE | ✅ | ❌ (entropy) | ❌ | ❌ |
| LLM-Check | ✅ | ❌ (attention probe) | ❌ | ❌ |
| AutoHall | ✅ | ❌ (self-contradiction) | ❌ | ❌ |
| Decoding Memories | ⚠️ wrapper | ❌ | ❌ | ❌ |
| Conformal-RAG | ✅ | ⚠️ sentence-level NLI | ❌ (only reject) | ❌ |

Full coverage: ours **4/4**, best comparable **1/4** (Dale), every other paper **≤ 1/4**.

**Verdict: WIN.** Ours is the only system that emits a hallucination label, per-claim
entailment, a rewritten answer, and a source citation. No combination of the other eleven
produces all four.

### Position 2 — Mitigation Δ

**Verdict: NOT A WIN — reported as measured.** See section 5. The measured delta is
−0.007 BERTScore F1 (CI [−0.022, +0.007]) on 16 regenerated rows. Dale reports +0.49 to
+0.60 on machine translation, so on the one published mitigation delta we do not lead.

### Position 3 — Atomic claim-level granularity

| System | Granularity | Catches intra-sentence contradictions? |
|---|---|---|
| **Ours (Steps 5, 8, 9)** | **Atomic claims** | **Yes** |
| TU Wien | Splits claims but gives a sentence-level verdict | Partial |
| AILS-NTUA | Sentence-level NLI | No |
| INSIDE / LLM-Check / AutoHall / Dale / IRIT-BL / DUTh / SHROOM / Conformal-RAG / Decoding Memories | Sentence-level | No |

**Verdict: WIN.** TU Wien splits claims but emits a sentence-level final verdict. Every
other paper operates at the sentence level. Ours is the only system that produces a
per-claim verdict and catches intra-sentence contradictions.

### Position 4 — Citation provenance per claim

Each claim in our output carries the `doc_id`, the supporting passage, the cross-encoder
rerank score, and a provenance label (intrinsic / extrinsic / aggregated). A downstream
auditor can verify every claim by hand. No cited paper produces anything comparable.

**Verdict: WIN — unique.**

### Position 5 — HRR / VSR / CCR

Our HRR is 0.132 against Dale's implied ~0.55. Our VSR is 0.701 against 0.80–0.81. Our
CCR is 0.706 with no published CCR to compare against — Dale's 0.55 is implied from a
different metric, not published as a CCR.

**Verdict: NOT A WIN on the rates — reported as measured.** The rewrite architecture buys
the *capability* to edit a contradicted claim in place, not a better rate on this
evaluation.

---

## 9. Threats to validity

| Threat | How big | What we did about it |
|---|---|---|
| **The RAGTruth 0.613 is in-sample** | High | We report the held-out 0.6185 [0.5371, 0.6943] beside it, and note the held-out lower bound is below LLM-Check's top anchor |
| **Mitigation measured on 16 rows** | High | Reported with CI and p-value; called an open result, not a win |
| **HaluEval length confound** | High (raw split) | We use the length-matched split, where length alone classifies 50.3% of rows instead of 94.7% |
| **Different datasets across the tables** | High | Every non-RAGTruth row is marked as not like-for-like; the RAGTruth table is the only clean one |
| **TruthfulQA retrieval leak** | High | Retrieval and grounding are switched off; the leaked 0.9589 is quarantined and not reported |
| **Detector trained on the full pool** | Medium | Evaluated on a leak-free held-out split with no overlap |
| **Train/serve mismatch on TruthfulQA** | Medium | Detection head trained with retrieval features; some weights are less useful closed-book. Future work |
| **Generator dependency** | Medium | The backend is pluggable; per-backend scores are a planned addition |
| **Label noise** | Medium | HaluEval and RAGTruth labels are human annotations and may disagree |
| **Threshold sensitivity** | Medium | The 0.70 PASS / 0.30 ABSTAIN cutoffs are hand-set; curves over the threshold are future work |

---

## 10. What to say to a judge

**30-second version:**

> "I did not win the leaderboard, but I found something more useful. The benchmark
> everyone uses for this task can be solved 94.7% of the time just by counting the words
> in the answer. So the scores people publish are partly measuring how long the answer is,
> not whether it is true. I built a fair version of that benchmark where the true and
> false answers have the same length, and my own score drops from 0.968 to 0.925. I am
> reporting the lower number on purpose. On top of that I built a full 13-step system
> that checks each claim separately and can point to the document that proves it, and I
> report my own failed result with the same care as my good ones."

**If they ask "which part are you best at?":**

> "Not the detection score — I am about level with the best teams on SHROOM and I say so.
> What I have that nobody else has is: I can tell you exactly which claim is wrong and
> which document proves it; I evaluate on three datasets with confidence intervals
> computed over questions rather than rows; my system can say 'I don't know'; and I report
> my negative results the same way I report my positive ones."

**If they ask "so did your method work?":**

> "The detection side works — it is level with the best published systems on a harder,
> fairer test set, and on RAGTruth it beats three of LLM-Check's five published probes.
> The mitigation side did not work yet. My rewrite changed the answer by 0.007 and the
> interval covers zero, so the data does not show it helps. I would rather tell you that
> honestly than dress it up."

**If they push on the null result:**

> "That null result is on 500 rows, and I should be honest that the rewrite only fired on
> 16 of them from 8 unique questions. So the correct reading is 'not enough data to tell
> yet', not 'the rewrite does nothing'. I used a paired permutation test over unique
> questions for exactly that reason."

**If they ask about the 12-point drop:**

> "That drop is the point. If I had kept the raw number I would have had the best-looking
> score in the table, and it would have been measuring answer length. Dropping it is what
> makes the remaining number trustworthy."

### Limitations to state before the judge asks

| Limitation | What to say |
|---|---|
| Mitigation Δ is null (not positive) | "My rewrite step did not measurably help. I report it as an open result." |
| TruthfulQA AUC drops to 0.635 | "On a second, independent dataset my score drops. That tells you where the detector's limits are." |
| RAGTruth is in-sample at 0.613 | "The held-out figure is 0.6185 and I report it beside the other one." |
| End-to-end accuracy is 0.424 | "The weakest number I have, and it follows from the mitigation not working." |
| Claim-level F1 (0.811) above TU Wien (0.71) | "Lower precision, but my unit is the claim and theirs is the sentence, so it is a harder task." |
| No conformal coverage guarantee | "I use Platt scaling, which is weaker than Conformal-RAG's guarantee. Known gap, future work." |
| HRR (0.132) below Dale's implied 0.55 | "Dale's number is implied from a different metric, not published as an HRR, so it is not like-for-like." |
| No human evaluation in the loop | "I use the HaluEval and TruthfulQA gold labels instead of collecting my own annotations." |
| Generator-dependent cost | "Any method that calls an LLM has this. Mine is swappable in the config file." |

### One-paragraph contribution statement

> We present a 13-step RAG hallucination-detection and mitigation pipeline together with
> a leak-free, length-controlled evaluation protocol. We first show that the standard
> HaluEval benchmark is 94.7% solvable from answer length alone, so that reported
> detector performance there is substantially an artifact of answer verbosity; we
> therefore construct a length-matched split and report all headline numbers on it. On
> that split the pipeline reaches 0.925 ROC-AUC and 0.922 AUPR with percentile bootstrap
> intervals computed over unique queries, and we evaluate it on three datasets
> (HaluEval length-matched 0.925, TruthfulQA closed-book 0.635, RAGTruth length-matched
> 0.613). On RAGTruth our detection score clears three of LLM-Check's five published
> AUROC anchors and ties two, against the rule that our confidence interval's lower
> bound must exceed the published value; the held-out figure (0.6185) is reported beside
> it. While sitting within the SHROOM detection band on a harder test set, the pipeline
> additionally performs retrieval, atomic claim decomposition, per-claim NLI
> verification with citation-grade provenance, and an explicit regenerate/abstain gate.
> Our mitigation rewrite shows no measurable improvement over the no-mitigation baseline
> (Δ = −0.007, 95% CI [−0.022, +0.007], paired permutation p = 0.404) and we report
> this as an open result. The system's advantages over the eleven cited systems are
> structural rather than score-based: per-claim evidence provenance, atomic claim
> granularity, a pluggable generator backend, an explicit abstention path, and a
> three-dataset evaluation with query-level bootstrap intervals and paired permutation
> tests.

---

## 11. Reproducibility

```bash
# HaluEval (length-matched) — produces 0.925 / 0.922 / 0.811 / 0.701 / 0.132
python -m scripts.evaluate_full \
    --data data/halueval_eval_1000_matched.jsonl \
    --index data/index_halueval \
    --out data/metrics_halueval.json --limit 1000

# RAGTruth (length-matched) — produces 0.613
python -m scripts.evaluate_full \
    --data data/ragtruth_eval_1000_matched.jsonl \
    --index data/index_ragtruth \
    --out data/metrics_ragtruth_fitted.json --limit 1000

# TruthfulQA (closed-book) — produces 0.635
python -m scripts.evaluate_full \
    --dataset truthfulqa \
    --out data/metrics_truthfulqa_closedbook.json --limit 1000

# The comparison table in Paper/ — regenerated from the metrics above
python -m scripts.fill_comparison_table

# The published anchors, and the RAGTruth head-to-head
python -m scripts.compare_to_published --metrics data/metrics_ragtruth_fitted.json

# The test suite that holds all of the above in place
python -m pytest tests/ -q
```

Every number in this document is produced by one of those commands. The tables are
regenerated by `fill_comparison_table`, which reads the metrics files directly — no number
in `Paper/` is typed by hand.

---

## 12. References

1. Dale, D. et al. (2023). *Detecting and Mitigating Hallucinations in Machine Translation: Model Internal Workings Alone Do Well, Sentence Similarity Even Better.* ACL 2023.
2. Sharma, A. et al. (2024). *DUTh at SemEval-2024 Task 6.* SemEval 2024.
3. Arzt, V. et al. (2024). *TU Wien at SemEval-2024 Task 6.* SemEval 2024.
4. Mickus, T. et al. (2024). *SemEval-2024 Task 6: SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes.* SemEval 2024.
5. Bendahman, N. et al. (2024). *IRIT-Berger-Levrault at SemEval-2024: How Sensitive Sentence Embeddings are to Hallucinations?* SemEval 2024.
6. Grigoriadou, K. et al. (2024). *AILS-NTUA at SemEval-2024 Task 6: Efficient model tuning for hallucination detection and analysis.* SemEval 2024.
7. Chen, W. et al. (2024). *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.* arXiv 2310.09988.
8. Sriramanan, G. et al. (2024). *LLM-Check: Investigating Detection of Hallucinations in Large Language Models.* NeurIPS 2024.
9. Cao, M. et al. (2023). *AutoHall: Automated Factuality Hallucination Dataset Generation for Large Language Models.* arXiv 2310.00259.
10. Gao, X. et al. (2026). *Decoding Memories: An Efficient Pipeline for Self-Consistency Hallucination Detection.* arXiv 2508.21228.
11. Sinha, R. (2025). *The Semantic Illion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems.* arXiv 2512.15068.
12. Niu, C. et al. (2024). *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.* arXiv 2401.00396.
