# The corpus audit — checking every retrieval corpus for the answer key

Measured 2026-11-17. Every number here is reproducible from the scripts
named in each section.

## Why this document exists

The detection score on TruthfulQA came out at **ROC-AUC 0.9589** with a
near-binary score distribution (mean 0.168 for faithful rows, 0.768 for
hallucinated ones). A detector that good is not detecting hallucination;
it is reading an answer key.

The cause was the retrieval corpus, and the point of this document is that
**a leak is a property of the corpus, not of a single feature**. The first
fix — switching off the grounding term — was necessary and nowhere near
sufficient. It took a direct measurement of the corpus to find the rest.

## The audit

### The cheap test, for every dataset

```python
corpus = {d["text"].strip() for d in load_jsonl("data/<ds>_corpus.jsonl")}
leaked = sum(1 for r in rows if r["answer"].strip() in corpus)
```

If `leaked` is not 0, the corpus contains the answers and every
retrieval-grounded signal computed against it is suspect.

### The decisive test — what the corpus is *made of*

`scripts/diagnose_truthfulqa_leak.py` asks what fraction of the corpus a
retrieved document covers, split by label:

| measurement | TruthfulQA | RAGTruth |
|---|---|---|
| corpus documents | 778 | 298 |
| truthful rows whose own answer is in their own document | **500 of 500** | 0 of 1000 |
| hallucinated rows whose own answer is in their own document | **0 of 500** | 0 of 1000 |
| hallucinated answers appearing anywhere in the corpus | 19 of 500 | 0 of 1000 |

**TruthfulQA's corpus is a truth-only answer key.** It holds the correct
answer for every question and the incorrect answer for none. Retrieval
finds the reference answer for every row.

**RAGTruth's corpus is clean.** It holds only source contexts — the
paragraphs the models were asked about — and none of the 1000 answers
appears anywhere in it.

## HaluEval looks the same but is not

HaluEval's naive leak test also fires:

| | HaluEval | TruthfulQA |
|---|---|---|
| faithful answers appearing in the corpus | **500 of 500 (100%)** | 500 of 500 |
| hallucinated answers appearing in the corpus | **9 of 500 (1.8%)** | 0 of 500 |
| rows whose own passage is a corpus document | 1000 of 1000 | 1000 of 1000 |

The same shape, but **a different cause, and a different verdict.**

HaluEval's corpus is built from the dataset's `knowledge` field, which is
the **Wikipedia source passage** for the question. The faithful answer is
entailed by that passage, so it appears in it; the hallucinated answer is
a perturbation of it ("British-American" where the truth is "American"),
so usually it does not. Retrieving the question's own source passage and
checking the answer against it is **precisely what a RAG hallucination
detector is supposed to do**. The label and the entailment relationship
coincide because the dataset was built by perturbing a passage that the
correct answer follows from — that is the benchmark's design, not a leak.

TruthfulQA has no source passage at all. Its corpus was built from the
reference *answers*, so the "source" is the label, and there is no
legitimate context to verify against.

| | corpus content | is there a source passage? | verdict |
|---|---|---|---|
| RAGTruth | source contexts | yes | clean |
| HaluEval | source passages | yes | legitimate — verifying against the source is the task |
| TruthfulQA | reference answers | **no** | **leak — run closed-book** |

The test that separates them: **build a corpus that excludes each
evaluation row's own passage and re-measure.** If the signal survives, the
detector was verifying against genuine context; if it collapses, the signal
was an answer-key read. `scripts/audit_halueval_corpus.py` does this.

### HaluEval's result, confirmed

| run | detection ROC-AUC |
|---|---|
| HaluEval, 1000 rows, corrected head | **0.9251**, CI [0.9113, 0.9401] |
| HaluEval, retrieval recall@5 | 0.994 |

The number survives because the signal was always legitimate: the pipeline
retrieves the question's own Wikipedia passage and checks the answer
against it. That is the task.

### The diagnostic that exposes a leak from the outside

Measure the same signal two ways: against the row's own gold passage, and
against the full retrieval corpus.

| grounding measured against | AUC |
|---|---|
| the row's own gold passage | 0.9523 (HaluEval), ~0.95 (TruthfulQA) |
| the full retrieval corpus | 0.6801 (HaluEval), 0.5664 (RAGTruth) |

When the gold-passage number is *far* above the corpus number, the corpus
contains the reference. When they are close (RAGTruth: 0.6217 vs 0.5664),
the corpus holds only source contexts and the signal is honest.

A second tell: **a detection AUC above ~0.9 with a near-binary score
distribution is a leak until proven otherwise.**

## The three leaks in TruthfulQA

Disabling one term left two more reading the answer key:

| leak | what read the key | fixed by |
|---|---|---|
| grounding | step 9's grounding term, measured against the retrieved document | `grounding_source: "none"` |
| the NLI's premise | step 8 verifies each claim against the retrieved document, which *is* the reference answer | `retrieval_source: "none"` — no evidence at all |
| the evidence-dependent components | `retrieval_top1`, `nli`, `coverage` are 0.0 constants with no source; kept at their fitted weights they are a base-rate offset, not a measurement | dropped from the score when there is no evidence |

With grounding off but retrieval on, the components still read
`nli 0.9966, coverage 1.0` on a faithful row — the tell that the NLI was
the deeper leak.

## The fix: run closed-book datasets closed-book

TruthfulQA is a **closed-book** benchmark: it has no retrieval context.
Its `evidence` field only ever held the gold passages used by the
gold-evidence matching metric. Building a retrieval corpus from it was
the mistake.

Three settings, applied by the dataset rather than by hand:

| setting | where | effect |
|---|---|---|
| `retrieval_source: "none"` | step 6 | retrieves nothing; the NLI has no premise to read |
| `grounding_source: "none"` | step 9 | grounding left out of the score |
| no evidence → drop `retrieval_top1`, `nli`, `coverage` | step 9 | constants without a source are not measurements |

The evaluation harness overrides these per dataset
(`scripts/evaluate_full.py --retrieval-source` / `--grounding-source`),
and records them in the metrics file, so a metrics file always states how
its score was produced.

## The result

| run | detection ROC-AUC | score distribution |
|---|---|---|
| TruthfulQA, leaked (retrieval on, grounding on) | 0.9589 | near-binary — mean 0.168 faithful vs 0.768 hallucinated |
| TruthfulQA, grounding off only | 0.92 (20-row smoke) | still near-binary |
| **TruthfulQA, closed-book (1000 rows)** | **0.6355, CI [0.6107, 0.6594]** | spread out — 999 distinct values in 1000 rows |

Closed-book details:

| metric | value |
|---|---|
| Detection ROC-AUC | 0.6355, CI [0.6107, 0.6594] |
| Detection accuracy | 0.553 |
| Detection precision | 0.5299 |
| Detection recall | 0.938 |
| Detection F1 | 0.6773 |
| Detection AUPR | 0.6450 |
| Spearman rho | 0.2347 |
| Latency mean | 4.12 s |

**The lower number is the honest one.** A closed-book number of 0.64 is
the truthful result for a dataset with no source to verify against, and it
is what the paper reports. Its CI excludes 0.5, so the pipeline still
carries real signal on a dataset where it has nothing to read — but it is
0.32 AUC below what the leak made it look like.

## The general rule

> Before reporting any retrieval-grounded detection number, check what the
> retrieval corpus contains. If the corpus is built from the dataset's own
> answers, it is an answer key, and every signal measured against it is a
> label read.
>
> When a detector looks too good, ask what the corpus is made of — not
> which feature to disable.

## The neighbouring trap: a sign decided by a coin flip

Found in the same session, and worth stating here because it is the same
shape of mistake — a measurement that *looks* decisive but is not.

The fitted detection head signed `detector_trust` as a risk signal on the
strength of a tuning-half AUC of **0.5008**. That is 0.0008 above the 0.5
line. The true direction is trust: the component is
`1 − hallucination_probability`, so a higher value means the detector
thinks the answer is more faithful.

Measured on 300 RAGTruth rows:

| component | AUC read as risk | 95% CI | direction |
|---|---|---|---|
| `hallucination_probability` | 0.5389 | [0.4741, 0.5991] | risk |
| `detector_trust` | 0.4611 | [0.4009, 0.5259] | trust |

They sum to exactly 1.0000 — one measurement, read in opposite directions.
That complementarity is the check: **if two components are mirrors of each
other, their AUCs must sum to 1.** A pair that does not is a bug.

The component carried 0.1934 of fitted weight, pointing the wrong way.

**How to apply:** never decide a sign inside the noise band. A component
whose bootstrap CI contains 0.5 carries no directional evidence; take its
sign from the full data and *record that the decision was not decisive*.
`fit_detection_head.py` now reports `auc_all_ci95`, `sign_source` and
`direction_decisive` per component so the choice is visible in the
artefact rather than buried in a number.
