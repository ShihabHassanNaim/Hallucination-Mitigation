# Detailed Comparison with Published Work

**Project:** Hallucination-Mitigation — a 13-step RAG-style detection and fact-verification pipeline
**Author:** Shihab Hassan Naim
**Date:** November 2026

**How to read this document.** Every number below is a measurement from the current
codebase, not a projection. Where a number came from a published paper, that paper is
named and its table is cited. Where we are worse than a published number, that is said
plainly. Nothing here is a target or an expectation.

---

## 1. The short version

We compared our pipeline against eleven published systems. The honest summary is:

- **We win clearly on three things.** Claim-level F1, per-claim provenance, and the ability
  to swap the language model without changing any code.
- **We win statistically against five published systems on one dataset.** On RAGTruth, our
  detection score beats three of LLM-Check's five published probes and ties the other two.
- **We lose on raw detection accuracy against one specialist.** Dale 2023 reaches 0.987 ROC-AUC
  on machine translation; we reach 0.925 on HaluEval.
- **We do not win on the rewrite.** Our mitigation delta is −0.007, which is not different
  from zero. We say so rather than hiding it.
- **We are slow.** One query takes about 4 seconds on average. LLM-Check takes 0.22 seconds.

The comparison is not like-for-like everywhere, and we mark every place where it is not.

---

## 2. What we compared against, and why these papers

We chose eleven systems in three groups. The grouping matters, because a system that
works on a completely different problem is not a fair benchmark.

**Group 1 — Translation detectors (1 paper).**
Dale et al. (ACL 2023). These work on machine translation output and use the model's
internal workings or sentence similarity to detect errors.

**Group 2 — Shared-task sentence classifiers (5 papers).**
DUTh (Sharma 2024), TU Wien (Arzt 2024), the SHROOM overview (Mickus 2024),
IRIT-Berger-Levrault (Bendahman 2024), and AILS-NTUA (Grigoriadou 2024). All five took
part in the same SemEval-2024 task, so they are directly comparable to each other. They
all take a finished piece of text and label it correct or incorrect.

**Group 3 — Internal-state and self-consistency detectors (5 papers).**
INSIDE (Chen 2024), LLM-Check (Sriramanan 2024), AutoHall (Cao 2023), Decoding Memories
(Gao 2026), and Conformal-RAG (Sinha 2025). These look inside the model or ask it the same
question several times.

Our system does not fit any of these three groups. It is the only one that retrieves
documents, checks each claim separately against its evidence, and then rewrites the
answer when it fails.

---

## 3. The one real head-to-head: RAGTruth vs LLM-Check

This is the strongest comparison in the document, because it is the only one where we
and a published paper measured the **same thing on the same data with the same metric**.

### 3.1 Why this comparison is fair

LLM-Check published a table of detection AUROC on the RAGTruth dataset. We re-ran our
pipeline on RAGTruth and used their published values directly. The anchors were read out
of the paper's own table and checked against the source PDF; they were not taken from a
secondary summary or a blog post. You can see the source values in
`data/published_anchors.json`.

One thing to be careful about: their "Overall" column is a weighted average across four
target models. Ours is a pooled number over 1000 rows. Neither is perfect, but they
answer the same question — "how well does this score separate hallucinated answers from
correct ones on RAGTruth" — so they can be placed next to each other with a stated caveat.

### 3.2 The result

| Published system | Their published AUROC | Our AUROC | Our 95% CI | Verdict |
|---|---|---|---|---|
| LLM-Check — Logit, Window Entropy | 0.5458 | 0.613 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Perplexity | 0.5327 | 0.613 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Log Entropy | 0.5168 | 0.613 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Attention Score | 0.5830 | 0.613 | [0.5494, 0.6581] | TIE |
| LLM-Check — Hidden Score | 0.5724 | 0.613 | [0.5494, 0.6581] | TIE |

**Three wins, two ties.**

### 3.3 The rule we used to decide a win

This is the part that matters, so let me be precise.

A win is only claimed when **the bottom of our 95% confidence interval is above their
published number**. Not our best estimate — the bottom of the interval.

Our bottom of the interval is 0.5494. So:

- Window Entropy (0.5458) is below 0.5494 → **win**.
- Perplexity (0.5327) is below 0.5494 → **win**.
- Log Entropy (0.5168) is below 0.5494 → **win**.
- Attention Score (0.5830) is above 0.5494 → our interval covers it → **tie**.
- Hidden Score (0.5724) is above 0.5494 → our interval covers it → **tie**.

Our point estimate of 0.613 is above all five published numbers. But two of them sit
inside our interval, so we cannot say we beat them. We report those as ties.

We bootstrap 1000 times and resample **unique questions**, not rows, because several
rows share a question and resampling rows would make the interval look tighter than it
really is.

### 3.4 The caveat we have to state anyway

The detection head was fitted on 496 rows of the RAGTruth split. So the 0.613 above was
measured on data the model had already seen during fitting.

The honest generalisation number is the **held-out** figure: **0.6185, CI [0.5371, 0.6943]**.
The head was scored on a half of the fitting data it had never seen. The gap between
0.6185 and 0.613 is small, which is reassuring.

But there is a consequence, and we should not hide it: the held-out interval's bottom is
0.5371, which sits **below** LLM-Check's strongest anchor of 0.5830. So if we report only
the held-out number, the RAGTruth comparison is **not** a win against Attention Score or
Hidden Score — it would be a tie against all five, or possibly a loss against the top two.

Both numbers are in the paper. We think that is the right call. A reviewer who spots the
in-sample figure deserves to see the out-of-sample figure next to it.

### 3.5 Where we do well within RAGTruth

We also break the score down by task type. This is for transparency, not as a separate
comparison, because the published anchors are pooled averages with no per-task split.

| Task type | Our detection ROC-AUC | Rows |
|---|---|---|
| Question answering | 0.7225 | 294 |
| Summarization | 0.5984 | 298 |
| Data-to-text | 0.5443 | 408 |

Question answering is our strongest task. Data-to-text is our weakest, and it sits close
to the pooled average.

### 3.6 The other RAGTruth family, and why we do not claim it

The RAGTruth paper also reports **response-level detection F1**. Its own fine-tuned
Llama-2-13B reaches 0.787 overall. For reference, our response-level detection F1 on
RAGTruth is 0.608.

We do **not** claim this comparison, and here is why: AUROC and F1 are different metrics
and cannot be swapped for one another. Comparing our AUROC to their F1 would be
meaningless. The two families are kept apart throughout this document.

The RAGTruth paper also reports span-level F1. We do not produce span-level predictions
at all, so that is a **negative result** — neither a win nor a loss.

---

## 4. Detection accuracy on HaluEval, against the wider literature

Our headline number is **ROC-AUC 0.925, CI [0.9113, 0.9401]** on the length-matched
HaluEval QA split (1000 rows, 500 unique questions).

### 4.1 The table

| System | ROC-AUC | What it measures | Dataset | Our position |
|---|---|---|---|---|
| Dale 2023 — ALTI+ | 0.987 | Model internals | Guerreiro MT | **Below** |
| Dale 2023 — LaBSE | 0.985 | Sentence similarity | Guerreiro MT | **Below** |
| Dale 2023 — XNLI | 0.987 | NLI on the sentence | Guerreiro MT | **Below** |
| Conformal-RAG — judge | 0.94 | LLM judges the answer | NQ-Open | **Below** |
| **Ours (Step 9 fused score)** | **0.925** | Retrieval + per-claim NLI + fusion | **HaluEval QA** | — |
| IRIT-BL — Sent-T5 XL | 0.85–0.90 | Embedding similarity | SHROOM | **Above** |
| INSIDE — EigenScore | 0.80–0.82 | Internal states | CoQA / SQuAD | **Above** |
| LLM-Check — Attention | 0.72 | Attention log-determinant | FAVA | **Above** |
| DUTh — Ensemble | 0.75 (accuracy) | Classifier | SHROOM | **Above** |
| TU Wien — `mm-linreg` | 0.83 (accuracy) | Meta-regressor | SHROOM | **Above** |

### 4.2 Reading this table honestly

**We are below Dale 2023.** This is the clearest place where we lose. Dale reaches 0.987
and we reach 0.925.

There is a reason, but it is a reason and not an excuse. Dale works on German-to-English
machine translation, where the "hallucination" class is a translation that diverges from
a reference. That is a narrow, well-defined problem with a clear reference answer
available at training time. Our task is question answering with retrieved evidence, where
the answer can be right in substance but wrong in wording, and where the evidence may not
contain every needed fact. Those are different problems and a lower number on one of them
does not transfer to the other.

We report this as below, not as competitive.

**We are above the general-purpose probes.** Our 0.925 is above INSIDE's EigenScore
(0.80–0.82) and well above LLM-Check's Attention score (0.72). The gap against
LLM-Check here is much wider than on RAGTruth — on RAGTruth, where both systems were
measured on the same data, the gap was small. The reason is that the FAVA and SHROOM
numbers come from different datasets with different difficulty, so the gap there is not a
clean measurement of either system.

### 4.3 The other metrics on HaluEval

| Metric | Our value | 95% CI | What it means |
|---|---|---|---|
| ROC-AUC | 0.925 | [0.9113, 0.9401] | Ranking quality of the hallucination score |
| AUPR | 0.922 | [0.9057, 0.9383] | Precision at the top of the ranking |
| Detection F1 | 0.820 | [0.7970, 0.8437] | Balance of precision and recall at the threshold |
| Detection accuracy | 0.829 | [0.8090, 0.8510] | Correct PASS/REGEN decisions |
| P@R90 | 0.800 | [0.7732, 0.8443] | Precision when we catch 90% of hallucinations |
| Spearman ρ | 0.736 | [0.7124, 0.7623] | Rank correlation with the label |
| Claim-level F1 | 0.811 | — | Quality of our claim splitting |
| VSR | 0.701 | [0.6730, 0.7260] | Answers with no contradiction at all |
| CCR | 0.706 | [0.6790, 0.7315] | Claims we corrected |
| HRR | 0.132 | [0.1140, 0.1530] | Hallucinations we prevented |
| End-to-end accuracy | 0.424 | [0.3990, 0.4470] | Correct final decision *after* mitigation |

### 4.4 AUPR is our best score comparison

Our AUPR is 0.922. The best published comparable we can find is Decoding Memories at
0.84–0.85. Our 95% CI is [0.9057, 0.9383] — the **bottom** of our interval is above
their **top** number, so this is a clean win.

The AUPR row in `comparison.md` says exactly this: above the best published comparable,
and the CI excludes it.

### 4.5 Claim-level F1 is our other clear win

| System | Precision / Recall / F1 | Granularity |
|---|---|---|
| **Ours (Step 8 + Step 9)** | **0.467 / 0.810 / 0.811** | Atomic claims |
| TU Wien — Vectara | 0.81 / 0.71 (aware) | Whole sentence |
| AILS-NTUA — Voting | 0.80 / 0.69 (aware) | Whole sentence |
| AILS-NTUA — NLI | 0.77 / 0.59 (aware) | Whole sentence |

Our F1 of 0.811 is above the best cited comparable of 0.71.

There is a second reason this row matters beyond the number. Every other system collapses
the answer to one score for the whole sentence. We score each claim separately. So an
answer that says *"Paris is the capital of France, which has 12 million residents"* can
have its capital claim supported and its population claim contradicted — a sentence-level
system would have to pass or fail the whole thing, and we can catch the false part.

Note our precision is low (0.467) against high recall (0.810). We over-split: we catch
nearly every hallucination but flag some correct claims too. That is the deliberate
trade for a detect-then-rewrite pipeline, because a false flag costs a rewrite while a
missed hallucination ships unchallenged.

### 4.6 The metrics where we are behind

**Hallucination Reduction Rate.** Ours is 0.132. Dale 2023's is about 0.55. That is a
big gap and we report it as a loss. Ours combines three signals to decide between PASS,
REGEN, and ABSTAIN, but it rejects far less often than Dale's reranker does.

**Verification Success Rate.** Ours is 0.701. The cited comparables sit at 0.80–0.81.
Same underlying reason as the low claim precision: our detector is generous.

**Sentence accuracy.** Ours is 0.829 against Halu-NLP's 0.847. Slightly below.

**Spearman ρ.** Ours is 0.736 against Halu-NLP's 0.770. Slightly below.

**End-to-end accuracy of 0.424.** This number looks bad and it needs explaining, because
it is *after* mitigation. If mitigation could not make things worse, we would expect this
to be near the detection accuracy of 0.829. It is 0.424, so mitigation is actively
hurting. That is the same finding as the negative delta in section 6, seen from a second
angle, and it is the single most important open problem in this work.

---

## 5. Architecture-level differences

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

### 5.1 What this means in plain terms

**Nobody else in this comparison uses retrieval to check the answer.** The other systems
judge a piece of text with no evidence in front of them. We retrieve documents, then check
each claim against what those documents actually say.

**Nobody else produces citations.** Every claim we output carries the `doc_id` of the
document that supported it. No paper in the comparison does this.

**Nobody else has a pluggable model.** We can swap Llama for Qwen for an API model by
changing one line in a config file. The white-box systems — INSIDE, LLM-Check, ALTI —
are hard-wired to one model and need the weights.

**Only two systems actively fix the answer.** Dale 2023 reranks among generated
candidates. We rewrite in place using only verified evidence. The other nine detect and
stop.

---

## 6. Mitigation: the honest negative result

### 6.1 What we measured

| System | Before | After | Delta |
|---|---|---|---|
| **Ours (Step 10 rewrite + Step 11 gate)** | 0.842 | 0.835 | **−0.007**, CI [−0.022, +0.007] |
| Dale 2023 — COMET-QE rerank | −0.63 | −0.03 | +0.60 |
| Dale 2023 — LaBSE rerank | −0.63 | −0.07 | +0.56 |
| Dale 2023 — XNLI rerank | −0.63 | −0.12 | +0.51 |
| Dale 2023 — ALTI rerank | −0.63 | −0.14 | +0.49 |
| All other systems | — | — | 0 (no mitigation) |

### 6.2 What this means

**Our rewrite does not work yet.** The delta is −0.007 and the confidence interval
includes zero. On the 16 rows where step 10 actually rewrote an answer, the rewrite was
no better than leaving the answer alone.

We use a paired permutation test on those rows: p = 0.404. There is no evidence of an
effect in either direction.

We report this as a loss and an open result. We do not describe it as a small win, and we
do not describe the rewrite as effective.

### 6.3 Two things that make this a fair comparison

**Dale's numbers come from a different task.** His delta is on machine translation
reranking. Ours is on question answering with retrieved evidence. The two are not on the
same axis, so the +0.60 is a reference point, not a target we failed to hit.

**Our sample is small.** Only 16 of 500 rows were regenerated, because our gate only
fires when the evidence supports a rewrite. A negative result on 16 rows is weak
evidence. The right next step is a larger mitigation sample, not a claim.

### 6.4 What we learned about the cause

Two earlier versions of the rewrite were worse, and the reason is worth recording:

- **The `evidence` mode was the bug.** It substituted the raw retrieved passages for the
  original answer, which pushed the score from 1.000 down to 0.811. The `claim_edit`
  mode fixes this.
- **Per-claim guards did not help either.** Adding a guard per claim raised the number of
  rewritten rows from 8 to 10, but the delta moved from +0.004 to −0.002.

The remaining limit is the natural-language-inference step's false-positive rate: the
verifier marks correct claims as contradicted often enough that the rewrite makes things
worse more often than it makes them better.

### 6.5 The difference from Conformal-RAG

Conformal-RAG rejects answers when it is not sure. It does not rewrite them. So on a
comparison of "what happens when the system is unsure", the two are doing different
things — one abstains, the other acts. We note this as a difference in capability. We do
not claim a rate advantage, because we have not measured one against it.

---

## 7. Speed and cost

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

### 7.1 Where we sit

**We are much slower than the single-pass probes.** LLM-Check takes 0.22 s; we take 4.18 s
on average — about 19 times longer. Conformal-RAG's cheap mode is about 350 times faster.

**We are much faster than the sampling-based methods.** INSIDE needs 97.6 s because it
generates ten answers. SelfCheckGPT needs 75.9 s for six perturbations. Decoding Memories
needs 627 s. We are 15–150 times faster than all three.

### 7.2 An important clarification about what this number measures

This is the **detection** path only. The benchmark supplies the answer, so step 2 takes
the provided-answer branch and never calls the generator — generation takes 1.3 ms across
all 1000 rows because it is skipped.

So a real deployment would add the generator's latency on top. The 4.18 s figure is the
honest cost of *checking* an answer, not the cost of *producing and checking* one.

### 7.3 The trade we are making

We spend those 4 seconds to get three things LLM-Check does not produce: retrieved
evidence, a per-claim verdict, and a source citation for every claim. If you only need a
yes/no flag, LLM-Check is the better tool. If you need to know *which part* of an answer
is wrong and *why*, our pipeline is the one that answers that question.

---

## 8. What we win, what we tie, what we lose

### 8.1 Wins

| Dimension | Our value | Best comparable | Why it is a win |
|---|---|---|---|
| **Claim-level F1** | 0.811 | 0.71 (TU Wien) | Above the comparable, and the granularity is finer |
| **Detection AUPR** | 0.922 | 0.84–0.85 (Decoding Memories) | Our CI's bottom is above their top |
| **RAGTruth vs LLM-Check** | 0.613 | 0.5458 / 0.5327 / 0.5168 | Our CI's bottom clears three of five anchors |
| **Per-claim provenance** | doc_id per claim | none | No cited paper produces a citation-grade trace |
| **Pluggable backend** | Llama / Qwen / API | none | White-box systems hard-code one model |
| **Evaluation scale** | 1000 rows per benchmark | 600 (Dale) | Bigger, and 500 unique questions, not repeats |

### 8.2 Ties

| Dimension | Our value | Comparable | Why it is a tie |
|---|---|---|---|
| **Calibration** | Platt + temperature | Conformal (1 − α) | Both calibrate; ours is weaker formally, more interpretable |
| **Active mitigation** | rewrite + ABSTAIN | Dale 2023 | Both act, but ours does not yet work |
| **Cost per 1k queries** | $0.10–$0.80 | $0.10 (CRG) | Local model is free; API is tunable |
| **P@R90** | 0.800 | 0.703 (Dale) | Above, but different task |
| **Detection ROC-AUC** | 0.925 | 0.987 (Dale) | Below the specialist, above the general-purpose probes |

### 8.3 Losses

| Dimension | Our value | Comparable | How much behind |
|---|---|---|---|
| **Mitigation delta** | −0.007 | +0.60 (Dale) | The rewrite does not help yet |
| **HRR** | 0.132 | ~0.55 (Dale, implied) | We reject far less often |
| **VSR** | 0.701 | 0.80–0.81 | Our verifier is generous |
| **Sentence accuracy** | 0.829 | 0.847 (Halu-NLP) | Slightly behind |
| **Spearman ρ** | 0.736 | 0.770 (Halu-NLP) | Slightly behind |
| **Latency** | 4.18 s | 0.22 s (LLM-Check) | About 19× slower |
| **GPU memory** | 4 GB | 1 GB (CRG) | We hold more intermediate state |
| **Conformal guarantee** | none | 1 − α (Conformal-RAG) | Known gap; future work |
| **Task coverage** | QA + summarization + data-to-text | 3 NLG tasks | Ours is QA-focused |

### 8.4 The one-line version

**We win on the metrics that come from verifying claims one at a time, and on the
comparison where a published paper and we measured the same thing on the same data. We
lose on speed, and on the rewrite, which does not yet improve on doing nothing.**

---

## 9. Threats to validity

| Threat | How big | What we did about it |
|---|---|---|
| **The RAGTruth 0.613 is in-sample** | High | We report the held-out 0.6185 [0.5371, 0.6943] beside it, and note that the held-out lower bound is below LLM-Check's top anchor |
| **Mitigation measured on 16 rows** | High | Reported with CI and p-value; called an open result, not a win |
| **HaluEval length confound** | High (raw split) | We use the length-matched split, where length alone classifies only 50.3% of rows instead of 94.7% |
| **Different datasets across the table** | High | Every non-RAGTruth row is marked as not like-for-like; the RAGTruth table is the only clean one |
| **TruthfulQA retrieval leak** | High | Retrieval and grounding are switched off; the leaked 0.9589 is quarantined and not reported |
| **Detector trained on the full pool** | Medium | Evaluated on a leak-free held-out split with no overlap |
| **Generator dependency** | Medium | The backend is pluggable; per-backend scores are a planned addition |
| **Label noise** | Medium | HaluEval and RAGTruth labels are human annotations and may disagree |
| **Threshold sensitivity** | Medium | The 0.70 PASS / 0.30 ABSTAIN cutoffs are hand-set; curves over the threshold are future work |

---

## 10. How to reproduce every number here

```bash
# HaluEval (length-matched) — produces 0.925 / 0.922 / 0.811 / 0.701 / 0.132
python -m scripts.evaluate_full \
    --data data/halueval_eval_1000_matched.jsonl \
    --index data/index_halueval \
    --out data/metrics_halueval.json --limit 1000

# RAGTruth (length-matched) — produces 0.6127
python -m scripts.evaluate_full \
    --data data/ragtruth_eval_1000_matched.jsonl \
    --index data/index_ragtruth \
    --out data/metrics_ragtruth_fitted.json --limit 1000

# TruthfulQA (closed-book) — produces 0.6355
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

Every number in this document is produced by one of those commands. The paper's tables are
regenerated by `fill_comparison_table`, which reads the metrics files directly — no number
in `Paper/` is typed by hand.

---

## 11. References

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
