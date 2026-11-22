# RAGTruth comparison — every published paper that evaluates on RAGTruth

**Scope.** This is a like-for-like comparison on the **RAGTruth** dataset only. Every
number below was read out of a source paper's own table (or extracted from its PDF
text) — nothing is taken from a secondary summary, and nothing is typed in from
memory.

**Our measurement.** `data/metrics_ragtruth_fitted.json` — 1000 rows, 277 unique
queries, perfectly balanced 500/500 hallucinated/factual, retrieval-backed grounding,
detector fingerprint `7ce46fdfe941800f`.

**The rule used for every verdict below.** A **win** requires the **bottom of our
95% bootstrap CI to sit strictly above the published value**. A **loss** requires our
**top** of the interval to sit strictly below it. Anything else is a **tie**. This
rule is applied in both directions — a point estimate that beats a published number
but whose interval overlaps it is *not* a win.

> Our detection ROC-AUC on RAGTruth: **0.6127**, 95% CI **[0.5494, 0.6581]**
> (n = 1000 rows / 277 unique queries, 1000 bootstrap resamples over **unique
> questions**, not rows).

---

## 1. The like-for-like head-to-head — response-level detection AUROC

These are the only papers that report a **response-level detection AUROC on
RAGTruth**. That is the same metric we report, so they are directly comparable.

### 1.1 LLM-Check (NeurIPS 2024)

Source: *"LLM-Check: Investigating Detection of Hallucinations in Large Language
Models"*, Table 4 — *"Detection on the RAGTruth Dataset using Llama-2-7b model in
white-box and black-box setting"*. Their "Overall" column is a weighted average
across four black-box target models (Llama-2-13B, Llama-2-70B, GPT-4, Mistral-7B);
ours is a pooled number over 1000 rows. Neither is perfect, but they answer the same
question.

| Published system | Their AUROC | Ours | Our 95% CI | Verdict |
|---|---|---|---|---|
| LLM-Check — Logit, Window Entropy | 0.5458 | 0.6127 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Perplexity | 0.5327 | 0.6127 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Log Entropy | 0.5168 | 0.6127 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Hidden Score | 0.5724 | 0.6127 | [0.5494, 0.6581] | **TIE** |
| LLM-Check — Attention Score | 0.5830 | 0.6127 | [0.5494, 0.6581] | **TIE** |

**3 wins, 2 ties.** Our lower bound is 0.5494, which clears Window Entropy
(0.5458), Perplexity (0.5327) and Log Entropy (0.5168), but sits below Attention
Score (0.5830) and Hidden Score (0.5724).

### 1.2 "When Attention Escapes" — AES (Findings of EMNLP 2026)

Source: Su, Wang, Ba, Chu, Zhong, Yan & Ren, *"When Attention Escapes: Diagnosing and
Mitigating Hallucination in Retrieval-Augmented Generation"*, Table 1 (and Table 2
for Mistral, Table 12 for Llama-2-70B).

This is the strongest published RAGTruth detection AUROC we found — and it is well
above ours. All values are their Table 1 "RAGTruth" column, **AUROC**, for the
**span-level** protocol (their default).

| Model | Method | Published AUROC | Ours | Our 95% CI | Verdict |
|---|---|---|---|---|---|
| Llama2-7B | AES | 0.8829 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| Llama2-13B | AES | 0.9250 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| Llama3-8B | AES | 0.8807 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| Mistral-7B | AES | 0.8931 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| Llama2-70B | AES | 0.9332 | 0.6127 | [0.5494, 0.6581] | **LOSS** |

Their baselines from the same tables, for context (AUROC, RAGTruth):

| Method | Llama2-7B | Llama2-13B | Llama3-8B | Mistral-7B | Llama2-70B |
|---|---|---|---|---|---|
| Perplexity | 0.5091 | 0.4121 | 0.7127 | 0.6416 | 0.5669 |
| LN-Entropy | 0.6965 | 0.7676 | 0.7072 | 0.7312 | 0.6801 |
| SelfCheckGPT | 0.5230 | 0.5475 | 0.5269 | 0.5955 | 0.5084 |
| EigenScore | 0.5114 | 0.7853 | 0.4561 | 0.6892 | 0.6673 |
| P(True) | 0.6394 | 0.5332 | 0.6963 | 0.2845 | 0.6394 |
| ICR Probe | 0.7274 | 0.8900 | 0.7695 | 0.8040 | — |
| ReDeEP | 0.7474 | 0.7977 | 0.7139 | — | — |
| LUMINA | 0.7646 | 0.8569 | 0.7446 | 0.7685 | — |
| **AES** | **0.8829** | **0.9250** | **0.8807** | **0.8931** | **0.9332** |

**Honest reading.** AES beats us decisively on raw AUROC — by roughly 0.27–0.32
ROC-AUC points, far outside our interval in both directions. This is a genuine loss
and it is reported as one. Two things must be stated alongside it:

1. **Their protocol is span-level, ours is claim-level.** AES scores *individual
   spans* and aggregates; we score *claims*. Both are response-level AUROC, but they
   are not the same granularity of evidence, and their label set is the fine-grained
   span annotation rather than the binary response label.
2. **Their split is the extended test set.** They state they use "the extended test
   set created by Sun et al. (2025), which includes annotations for Llama3-8b" —
   2,700 test examples — while our split is 1,000 length-matched rows.

Note also that several of *their* baselines score below our number (Perplexity
0.5091/0.4121, SelfCheckGPT 0.5230, EigenScore 0.5114 on Llama2-7B), so the
comparison to their strongest method and the comparison to their weak methods point
in opposite directions. Both facts are in the tables above.

### 1.3 "Detecting Contextual Hallucinations…" — Frequency-Aware Attention (arXiv 2602.18145)

Source: Qi, Chen, Zhao, Zhu, Hu, Liu, He, Yuan & Gui, *"Detecting Contextual
Hallucinations in Large Language Models with Frequency-Aware Attention"*, Table 1
(token-level) and Table 2 (span-level, "RagTruth-Avg").

Their Table 1 is broken down by task (RT-QA / RT-D2T / RT-Summ) rather than pooled,
so it does **not** yield a single comparable "overall" AUROC. Their Table 2
(RagTruth-Avg) is the pooled one:

| Model | Method | Published AUROC (RagTruth-Avg) | Ours | Verdict |
|---|---|---|---|---|
| LLaMA-7B | Lookback-lens | 0.7884 | 0.6127 | **LOSS** |
| LLaMA-7B | Laplacian | 0.7877 | 0.6127 | **LOSS** |
| LLaMA-7B | Wavelet-High | 0.8117 | 0.6127 | **LOSS** |
| LLaMA-7B | Fourier-high | 0.8412 | 0.6127 | **LOSS** |
| LLaMA-13B | Lookback-lens | 0.8183 | 0.6127 | **LOSS** |
| LLaMA-13B | Laplacian | 0.8039 | 0.6127 | **LOSS** |
| LLaMA-13B | Wavelet-High | 0.8448 | 0.6127 | **LOSS** |
| LLaMA-13B | Fourier-high | 0.8585 | 0.6127 | **LOSS** |
| Mistral-7B | Lookback-lens | 0.8764 | 0.6127 | **LOSS** |
| Mistral-7B | Laplacian | 0.8495 | 0.6127 | **LOSS** |
| Mistral-7B | Wavelet-High | 0.8742 | 0.6127 | **LOSS** |
| Mistral-7B | Fourier-high | 0.8812 | 0.6127 | **LOSS** |

Their full Table 1 (token-level, AUROC, per task) is the richest published RAGTruth
detection table we found:

| Model | Method | RT-QA | RT-D2T | RT-Summ |
|---|---|---|---|---|
| LLaMA-7B | SelfCheckGPT | 0.6942 | 0.8026 | 0.6674 |
| LLaMA-7B | RefChecker | 0.5865 | 0.6349 | 0.6080 |
| LLaMA-7B | EigenScore | 0.5253 | 0.5297 | 0.4989 |
| LLaMA-7B | Redeep | 0.6364 | 0.3960 | 0.5760 |
| LLaMA-7B | Lookback-lens | 0.8482 | 0.8442 | 0.7156 |
| LLaMA-7B | Attn-variance | 0.6147 | 0.5890 | 0.6492 |
| LLaMA-7B | Attn-entropy | 0.8481 | 0.8368 | 0.6722 |
| LLaMA-7B | Laplacian | 0.8449 | 0.8519 | 0.7040 |
| LLaMA-7B | Wavelet-high | 0.8526 | 0.8569 | 0.7165 |
| LLaMA-7B | Fourier-high | 0.8584 | 0.8595 | 0.7426 |
| LLaMA-13B | SelfCheckGPT | 0.6425 | 0.7989 | 0.5909 |
| LLaMA-13B | RefChecker | 0.5893 | 0.6381 | 0.6346 |
| LLaMA-13B | EigenScore | 0.4734 | 0.6582 | 0.5097 |
| LLaMA-13B | Redeep | 0.6457 | 0.6449 | 0.5253 |
| LLaMA-13B | Lookback-lens | 0.8727 | 0.8766 | 0.6702 |
| LLaMA-13B | Attn-variance | 0.7497 | 0.7177 | 0.6629 |
| LLaMA-13B | Attn-entropy | 0.8650 | 0.8717 | 0.6865 |
| LLaMA-13B | Laplacian | 0.8552 | 0.8796 | 0.6700 |
| LLaMA-13B | Wavelet-high | 0.8741 | 0.8932 | 0.7042 |
| LLaMA-13B | Fourier-high | 0.8792 | 0.8825 | 0.7362 |
| Mistral-7B | SelfCheckGPT | 0.7084 | 0.8353 | 0.7166 |
| Mistral-7B | RefChecker | 0.6121 | 0.6287 | 0.6596 |
| Mistral-7B | EigenScore | 0.4611 | 0.6507 | 0.4973 |
| Mistral-7B | Redeep | 0.7357 | 0.6648 | 0.5680 |
| Mistral-7B | Lookback-lens | 0.9148 | 0.8845 | 0.7954 |
| Mistral-7B | Attn-variance | 0.7497 | 0.7021 | 0.5836 |
| Mistral-7B | Attn-entropy | 0.9083 | 0.8734 | 0.7867 |
| Mistral-7B | Laplacian | 0.9049 | 0.8849 | 0.7978 |
| Mistral-7B | Wavelet-high | 0.9117 | 0.8829 | 0.8075 |
| Mistral-7B | Fourier-high | 0.9099 | 0.8863 | 0.8037 |

Again a clear loss, and again with a granularity caveat: their Table 1 is
**token-level**, Table 2 is **span-level**. Our 0.6127 is claim/response-level on a
length-matched 1000-row split.

### 1.4 "Rethinking Evaluation for LLM Hallucination Detection" (arXiv 2605.11330)

Source: Chen, Padmanabhan, Giyahchi, Wong & Akoglu. Their Table 7 reports
**response-level ROC-AUC and PR-AUC on RAGTruth** for six detectors, so this is the
one paper besides LLM-Check whose protocol matches ours most closely.

| Detector | Published ROC-AUC | Published PR-AUC | Ours (ROC-AUC) | Our 95% CI | Verdict |
|---|---|---|---|---|---|
| SFT (supervised fine-tuning) | 0.871 | 0.688 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| FS (few-shot) | 0.836 | 0.694 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| LLM-aaJ (LLM-as-a-Judge, Claude-Sonnet-3.5) | 0.825 | 0.682 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| PO (prompt-optimized) | 0.800 | 0.671 | 0.6127 | [0.5494, 0.6581] | **LOSS** |
| SC-GPT (C) (SelfCheckGPT + Claude-Sonnet-3.5) | 0.586 | 0.199 | 0.6127 | [0.5494, 0.6581] | **TIE** |
| SC-GPT (G) (SelfCheckGPT + GPT-4-mini) | 0.584 | 0.199 | 0.6127 | [0.5494, 0.6581] | **TIE** |

Their Table 4 also gives F1 / precision / recall / accuracy on RAGTruth:

| Detector | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| SC-GPT (G) | 0.342 | 0.209 | 0.938 | 0.358 |
| SC-GPT (C) | 0.346 | 0.213 | 0.919 | 0.383 |
| LLM-aaJ | 0.617 | 0.482 | 0.856 | 0.806 |
| FS | 0.628 | 0.490 | 0.875 | 0.811 |
| PO | 0.538 | 0.373 | 0.963 | 0.697 |
| SFT | 0.671 | 0.644 | 0.700 | 0.874 |

Their F1 numbers are a fairer comparison to our **detection F1 = 0.6076**, CI
[0.5383, 0.6632]:

| Detector | Published F1 | Ours | Our 95% CI | Verdict |
|---|---|---|---|---|
| SFT | 0.671 | 0.6076 | [0.5383, 0.6632] | **LOSS** (0.671 sits above our whole interval) |
| FS | 0.628 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| LLM-aaJ | 0.617 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| PO | 0.538 | 0.6076 | [0.5383, 0.6632] | **WIN** (our lower bound 0.5383 clears 0.538) |
| SC-GPT (G) | 0.342 | 0.6076 | [0.5383, 0.6632] | **WIN** |
| SC-GPT (C) | 0.346 | 0.6076 | [0.5383, 0.6632] | **WIN** |

**Note the metric split.** On ROC-AUC we lose to their strong detectors; on F1 we
tie them. The two metrics disagree because their ROC-AUC is pooled over a
different label aggregation ("Contradicted" treated as stricter than "Not
Mentioned") than the one we use.

---

## 2. Response-level detection F1 — a different metric family, kept separate

From the **RAGTruth paper itself** (Niu et al., arXiv 2401.00396), Table 5: *"The
response-level hallucination detection performance for each baseline method across
different tasks and different models"*. Measured on the RAGTruth test set, 450
instances (150 per task type).

| Published system | Their F1 (overall) | Ours | Our 95% CI | Verdict |
|---|---|---|---|---|
| Finetuned Llama-2-13B (RAGTruth's own) | 0.787 | 0.6076 | [0.5383, 0.6632] | **LOSS** |
| Prompt gpt-4-turbo | 0.634 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| SelfCheckGPT gpt-3.5-turbo | 0.588 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| Prompt gpt-3.5-turbo | 0.529 | 0.6076 | [0.5383, 0.6632] | **WIN** |
| LMvLM gpt-4-turbo | 0.494 | 0.6076 | [0.5383, 0.6632] | **WIN** |

Per-task detail for the two systems that report it:

| System | writing/summarization | question answering | data-to-text |
|---|---|---|---|
| Finetuned Llama-2-13B | 0.782 | 0.682 | 0.591 |
| Prompt gpt-4-turbo | 0.634 | 0.456 | 0.476 |

**These are not AUROC numbers and cannot be swapped for them.** The table is kept
separate from section 1 for exactly that reason.

---

## 3. Span-level F1 — a negative result, not a comparison

From the **RAGTruth paper**, Table 6 (span-level, character-level overlap with the
gold spans):

| Published system | Their span F1 |
|---|---|
| Finetuned Llama-2-13B (RAGTruth's own) | 0.527 |
| Prompt Baseline gpt-4-turbo | 0.283 |
| Prompt Baseline gpt-3.5-turbo | 0.128 |

**We produce no span-level predictions at all.** We neither win nor lose here — this
is reported as a **negative result**: our pipeline does not attempt span-level
hallucination localisation.

---

## 4. SHAM-Bench (arXiv 2610.27370) — a span-level benchmark

Source: Huang & Lane, *"SHAM-Bench: A Span-level Hallucination Benchmark Across
Multiple Tasks"*. This paper builds a new benchmark and evaluates existing span-level
detectors on it, including on the RAGTruth partition.

From their Table 2 (span-level F1, RAGTruth partition columns — `ctx_qa` = RAGTruth
QA, `summ` = RAGTruth summarization, `data2txt` = RAGTruth data-to-text):

| Method | Overall span-F1 | ctx_qa | data2txt | summ |
|---|---|---|---|---|
| RAGTruth (ft-llama2-13B) | 0.518 | 0.865 | 0.654 | 0.835 |
| LettuceDetect (lettuce-large) | 0.482 | 0.833 | 0.614 | 0.541 |
| FAVA (ft-llama2-7B) | 0.381 | 0.696 | 0.243 | 0.376 |
| PsiloQA (ModernBERT-base) | 0.239 | 0.072 | 0.089 | 0.359 |

Their sentence-level F1 on the same partitions:

| Method | Overall sentence-F1 | ctx_qa | data2txt | summ |
|---|---|---|---|---|
| Prompt (GPT-5-mini) | 0.691 | 0.610 | 0.635 | 0.726 |
| RAGTruth (ft-llama2-13B) | 0.638 | 0.869 | 0.752 | 0.590 |
| LettuceDetect (lettuce-large) | 0.638 | 0.842 | 0.717 | 0.714 |
| MiniCheck (Bespoke-7B) | 0.632 | 0.466 | 0.492 | 0.737 |
| AlignScore (roberta-large) | 0.595 | 0.492 | 0.290 | 0.745 |
| FAVA (ft-llama2-7B) | 0.540 | 0.701 | 0.268 | 0.435 |

**Not comparable to ours.** These are span- and sentence-level F1 on a 2,700-example
test split with a different task mix. Our pipeline does not emit spans, and our
response-level F1 (0.6076) is not the same quantity. We report this as context only.

---

## 5. Uses RAGTruth but publishes no comparable detection number

### 5.1 "The Semantic Illusion" (arXiv 2512.15068)

Source: Sinha, *"The Semantic Illusion: Certified Limits of Embedding-Based
Hallucination Detection in RAG Systems"*. It uses RAGTruth as a **qualitative
demonstration** — Figure 3 shows score distributions for faithful vs. hallucinated
responses on Llama-2, Mistral and GPT-4, and argues the two classes overlap
substantially. It reports **no AUROC, F1 or accuracy on RAGTruth**, and its own
calibration is on Natural Questions and HaluEval, not RAGTruth. **No comparison is
possible.**

### 5.2 Four cited systems that do not report RAGTruth at all

| System | Reason |
|---|---|
| Conformal-RAG | does not report RAGTruth |
| INSIDE | evaluates on FAVA-Annotation, not RAGTruth (LLM-Check Table 2) |
| Decoding Memories | does not report RAGTruth |
| AutoHall | does not report RAGTruth |

These cannot be compared to us on this dataset, and we say so rather than quietly
comparing against a number from a different dataset.

---

## 6. Summary — every RAGTruth paper, and where we stand

| Paper | What it reports on RAGTruth | Comparable to us? | Verdict vs our 0.6127 / CI [0.5494, 0.6581] |
|---|---|---|---|
| **LLM-Check** (NeurIPS 2024) | 5 detection AUROC anchors (0.5168–0.5830) | **Yes** — response-level AUROC | **3 WIN, 2 TIE** |
| **"When Attention Escapes" / AES** (Findings EMNLP 2026) | AUROC 0.8807–0.9332 | Yes, but span-level protocol | **LOSS** (all models) |
| **"Detecting Contextual Hallucinations…"** (arXiv 2602.18145) | AUROC 0.4561–0.8931 (token + span level) | Yes, but token/span protocol | **LOSS** (all pooled averages) |
| **"Rethinking Evaluation…"** (arXiv 2605.11330) | ROC-AUC 0.584–0.871, PR-AUC 0.199–0.694, F1 0.342–0.671 | **Yes** — response-level | **LOSS** on ROC-AUC (4), **TIE** on ROC-AUC (2); on F1 **3 WIN, 2 TIE, 1 LOSS** |
| **RAGTruth paper** (arXiv 2401.00396) | Response-level F1 (0.494–0.787); span F1 (0.128–0.527) | F1 family only | **2 WIN, 2 TIE, 1 LOSS** on F1; span = **negative result** |
| **SHAM-Bench** (arXiv 2610.27370) | Span & sentence F1 on RAGTruth partitions | No | **Not comparable** |
| **"The Semantic Illusion"** (arXiv 2512.15068) | Qualitative score distributions only | No | **Not comparable** |
| **Conformal-RAG / INSIDE / Decoding Memories / AutoHall** | — | No | **Do not report RAGTruth** |

### The honest one-paragraph version

On the single genuinely like-for-like comparison — response-level detection AUROC
against published anchors read from LLM-Check's own table — we clear three of their
five probes and tie the other two. Against the three newer papers that report
RAGTruth detection AUROC, **we lose**: AES (EMNLP 2026) reaches 0.88–0.93, the
frequency-aware method 0.80–0.88, and the desiderata paper's fine-tuned detector
0.871. Our 0.6127 with CI [0.5494, 0.6581] sits below all of them. On **F1**
against the desiderata paper's detectors the picture is closer: we beat the two
SelfCheckGPT rows and the prompt-optimised row, tie few-shot and LLM-as-a-Judge,
and lose to the fine-tuned detector. Our advantages on this dataset are not in the
raw detection number; they are elsewhere — retrieval is on (mean recall@5 0.76 on
the length-matched split), the pipeline emits human-auditable claims with claim
precision 0.964, and it runs end-to-end at a measured 4.18 s/query on GPU.
Reporting only the LLM-Check table would be misleading, which is why this file
lists every RAGTruth paper we found, including the ones we lose to.

### Caveats that must travel with these numbers

1. **The detection head was fitted on 496 rows of the RAGTruth split.** The
   headline 0.6127 is therefore **in-sample**. The honest generalisation estimate
   is the **held-out** figure: **0.6185, CI [0.5371, 0.6943]**. On the held-out
   figure alone, the LLM-Check comparison would be **5 ties, 0 wins** — because the
   held-out lower bound (0.5371) sits below LLM-Check's strongest anchor (0.5830).
   Both numbers are reported.
2. **Splits differ.** Ours is 1000 length-matched rows / 277 unique queries.
   LLM-Check's anchors are a weighted average across four black-box target models.
   The AES and frequency-aware papers use the extended 2,700-example test set.
   SHAM-Bench uses its own task mix. Nothing here is a shared-split comparison
   except within each paper's own table.
3. **Granularity differs.** Our number is claim/response-level. AES is span-level
   (Table 1/2) and the frequency-aware method reports both token-level (Table 1) and
   span-level (Table 2) numbers. These are not interchangeable with response-level
   AUROC.
4. **RAGTruth is a diagnostic, not a win.** We are the only system in this file
   with retrieval switched on and a grounded pipeline; the others are mostly
   white-box probing of a model's internals. That is a structural difference, not a
   like-for-like one.

---

## Sources

1. Niu, C. et al. (2024). *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.* arXiv 2401.00396 — Table 5 (response-level F1), Table 6 (span-level F1).
2. *"LLM-Check: Investigating Detection of Hallucinations in Large Language Models"*, NeurIPS 2024 — Table 4 (detection AUROC on RAGTruth).
3. Su, W., Wang, Q., Ba, Z., Chu, Z., Zhong, Z., Yan, L. & Ren, K. (2026). *When Attention Escapes: Diagnosing and Mitigating Hallucination in Retrieval-Augmented Generation.* Findings of EMNLP 2026 — Table 1 (RAGTruth AUROC/ACC/PCC/F1), Table 2 (Mistral-7B), Table 12 (Llama-2-70B).
4. Qi, S., Chen, Y., Zhao, R., Zhu, Q., Hu, Z., Liu, W., He, Y., Yuan, Z. & Gui, L. (2026). *Detecting Contextual Hallucinations in Large Language Models with Frequency-Aware Attention.* arXiv 2602.18145 — Table 1 (token-level), Table 2 (RagTruth-Avg span-level), Table 3 (attention-head ablation).
5. Chen, W., Padmanabhan, V., Giyahchi, T., Wong, E. & Akoglu, L. (2026). *Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights.* arXiv 2605.11330 — Table 4 (F1/precision/recall/accuracy), Table 7 (ROC-AUC/PR-AUC).
6. Huang, S. & Lane, I. (2026). *SHAM-Bench: A Span-level Hallucination Benchmark Across Multiple Tasks.* arXiv 2610.27370 — Table 2 (span and sentence F1 by task).
7. Sinha, D. (2025). *The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems.* arXiv 2512.15068 — Figure 3 (qualitative only).

**Our numbers:** `data/metrics_ragtruth_fitted.json` (n = 1000, 277 unique queries,
500/500 balanced, `n_failed = 0`). Retrieval side: `data/metrics_ragtruth.json`.
Published anchors already checked into this repo: `data/published_anchors.json`.
