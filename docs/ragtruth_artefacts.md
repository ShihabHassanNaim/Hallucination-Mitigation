# RAGTruth evaluation — artefacts and results

Everything below was measured on 2026-11-17 with the code as it stands.
Every number is reproducible from the scripts named here.

## The dataset

Built by `scripts/build_ragtruth_jsonl.py` from the
`wandb/RAGTruth-processed` test parquet.

| file | what it is |
|---|---|
| `data/ragtruth_eval_1000.jsonl` | 1000 rows, 500 pairs, raw faithful partner |
| `data/ragtruth_eval_1000_matched.jsonl` | 1000 rows, 500 pairs, closest-length faithful partner |
| `data/ragtruth_corpus.jsonl` | 298 unique contexts (the retrieval corpus) |
| `data/index_ragtruth/` | retrieval index, built with `--chunk-size 12000 --chunk-overlap 0` |
| `data/ragtruth_dataset_report.json` | the dataset report |

The RAGTruth split is **paired within an instance**, not by query: an
instance is (task_type, context, query), because Data2txt reuses one
instruction prompt for all 450 businesses.

**Length-only separability** (ROC-AUC of the answer's word count alone):

| split | AUC |
|---|---|
| raw paired | 0.5479 |
| **length-matched** | **0.5239** |

The matched split is at chance, so a detection score on it cannot be
explained by answer length.

## The headline result

`scripts/evaluate_full.py --data data/ragtruth_eval_1000_matched.jsonl
--index data/index_ragtruth --out data/metrics_ragtruth_fitted.json`

| metric | value |
|---|---|
| **Detection ROC-AUC** | **0.6127** |
| **95% CI** | **[0.5494, 0.6581]** |
| Detection accuracy | 0.610 |
| Detection precision | 0.6113 |
| Detection recall | 0.604 |
| Detection F1 | 0.6076 |
| Detection AUPR | 0.6112 |
| Spearman rho | 0.1953, CI [0.0858, 0.2738] |
| Claim precision / recall / F1 | 0.9642 / 0.6260 / 0.7269 |
| Retrieval recall@5 | 0.902 |
| Latency mean | 10.01 s |

Per task type (the published anchors are pooled averages with no per-task
split, so these are reported for transparency, not as a separate
comparison):

| task type | n | ROC-AUC |
|---|---|---|
| Data2txt | 408 | 0.5443 |
| QA | 294 | 0.7225 |
| Summary | 298 | 0.5984 |

The CI resamples **unique instances** (277 of them), not rows, because the
rows within a pair share an instance.

### The held-out figure, reported beside it

The head was fitted on 496 rows of this split, so the number above is
measured on data the head was fitted on. Its honest generalisation
estimate is the **held-out** figure: **0.6185, CI [0.5371, 0.6943]** — the
head scored on a half of the fit data it never saw
(`data/detection_head_refit600.json`).

The gap between 0.6185 and 0.6127 is the generalisation gap, and it is
small. The held-out figure is reported next to the headline number because
the headline number's CI is what the published comparison uses, and
concealing that the head was fitted on this data would be an overclaim.

### The 99-row fit was optimistically biased

| fit | rows | tuning | held-out | held-out CI | all-data |
|---|---|---|---|---|---|
| original | 99 | 0.7867 | 0.6667 | [0.4719, 0.8337] | 0.7139 |
| refit | 300 | 0.7522 | 0.6267 | [0.5270, 0.7267] | 0.6849 |
| **in use** | **496** | 0.7093 | **0.6185** | **[0.5371, 0.6943]** | **0.6609** |

Both the tuning and the all-data AUC fall as the sample grows — the
signature of an overfit small fit. The 496-row head is the one in
`configs/default.yaml`.

## The comparison to published work

`scripts/compare_to_published.py` reads `data/published_anchors.json`, which
was verified line by line against the source PDFs checked in as
`ragtruth_paper.pdf` and `llmcheck_paper.pdf`.

**The rule:** a win requires our 95% CI *lower bound* to sit strictly above
the published value. A point estimate that beats it while the interval
overlaps is a tie, not a win.

| system | metric | published | ours | verdict |
|---|---|---|---|---|
| LLM-Check (Attention Score) | AUROC | 0.5830 | 0.6127, CI [0.5494, 0.6581] | **TIE** |
| LLM-Check (Hidden Score) | AUROC | 0.5724 | " | **TIE** |
| LLM-Check (Logit, Window Entropy) | AUROC | 0.5458 | " | **WIN** |
| LLM-Check (Logit, Perplexity) | AUROC | 0.5327 | " | **WIN** |
| LLM-Check (Logit, Log Entropy) | AUROC | 0.5168 | " | **WIN** |

**Three wins, two ties.** Our lower bound is 0.5494, which clears Window
Entropy (0.5458), Perplexity (0.5327) and Log Entropy (0.5168), but sits
below Attention Score (0.5830) and Hidden Score (0.5724), so those two are
ties rather than wins.

**On the held-out figure the tally drops to two wins and three ties.** The
held-out lower bound is 0.5371, which still clears Perplexity (0.5327) and
Log Entropy (0.5168) but no longer clears Window Entropy (0.5458), Hidden
Score (0.5724) or Attention Score (0.5830). Both tallies are reported: the
in-sample one in the paper's comparison table, the held-out one here and in
`Paper/ragtruth_comparison.md`.

**Scope caveat, stated plainly:** LLM-Check's numbers are the "Overall"
column of its Table 4 — a weighted average across the target models
(Llama-2-13b, Llama-2-70b, GPT-4, Mistral-7b). It reports no per-task
breakdown, so the only like-for-like comparison is pooled against pooled.
Ours is over 1000 rows drawn equally from the three RAGTruth task types.

### The other two datasets

Both re-measured with the corrected head and the corrected instrument
(detection score = the step-9 fusion whenever the benchmark supplies the
answer):

| dataset | n | retrieval | detection ROC-AUC | 95% CI |
|---|---|---|---|---|
| HaluEval | 1000 | on | **0.9251** | **[0.9113, 0.9401]** |
| TruthfulQA | 1000 | **off (closed-book)** | **0.6355** | **[0.6107, 0.6594]** |

HaluEval improved from 0.8589 to **0.9251** when `detector_trust`'s sign was
corrected. That is the expected direction: HaluEval is the dataset the
step-3 detector was trained on, so its `detector_trust` component is at
its strongest there — and it was previously being read backwards.

**HaluEval's corpus is a legitimate RAG corpus**: it holds the Wikipedia
source passage for each question, and checking an answer against the
retrieved source is exactly what the detector is for.

**TruthfulQA's is not**: it is closed-book, and its corpus was built from
the dataset's own reference answers, so it is an answer key. It is run
with retrieval switched off. The audit is in `docs/corpus_leak_audit.md`.

**Two families are kept apart and never mixed:**

- *AUROC family* — LLM-Check. This is the only family the detection score
  can be compared against, and the one above.
- *F1 family* — the RAGTruth paper's Table 5 (response-level detection F1:
  gpt-4-turbo 63.4 overall, SelfCheckGPT 58.8, LMvLM 49.4, the paper's own
  fine-tuned Llama-2-13B 78.7). Our metric is an AUROC on a different split,
  so these are **not** scored as wins or losses.
- *Span F1 family* — the RAGTruth paper's Table 6 (fine-tuned Llama-2-13B
  52.7 overall). Our pipeline produces **no span-level predictions**, so
  there is nothing to compare. This is a negative result.

**Systems that do not report RAGTruth** (rejected as comparables):
Conformal-RAG, INSIDE, Decoding Memories, AutoHall.

## Three bugs that had to be fixed first

The first two are documented in full in
`src/steps/step09_confidence_fusion.py`'s module docstring. The third —
the TruthfulQA corpus leak — is documented in
`docs/corpus_leak_audit.md`.

### 1. The fusion's signs were wrong

The step-9 fusion's weights were set by hand and never validated against a
detection metric. It added *every* component with a positive weight into a
score it treated as uniformly trust-oriented, but the components do not
share a direction: `nli`, `coverage`, `grounding` and `detector_trust` are
trust signals (high = less likely hallucinated), while `similarity`,
`retrieval_top1` and `eedc` are risk signals. The two largest hand-set
weights (nli 0.25, coverage 0.15) both pointed the wrong way.

As a detection score, the hand-set fusion scored **ROC-AUC 0.360 — worse
than chance**.

**The fix:** `scripts/fit_detection_head.py` measures each component's
direction on a tuning half, fits a logistic regression on standardised
components, and reports the held-out half. The fitted head lives in
`configs/default.yaml` under `step09_fusion_weights.detection_head`.

Fitted weights, and the sign each component enters with (the head in use
is the 496-row fit):

| component | sign | weight | measured AUC (risk reading) | meaning |
|---|---|---|---|---|
| nli | − | 1.2497 | 0.4504, CI [0.3988, 0.5058] | trust: claims are supported |
| coverage | − | −1.0345 | 0.4646, CI [0.4133, 0.5190] | trust: share of supported claims |
| grounding | − | 0.5887 | 0.4809, CI [0.4301, 0.5328] | trust: answer's content words came from the source |
| similarity | + | 0.5711 | 0.5986, CI [0.5442, 0.6529] | risk |
| detector_trust | − | 0.3228 | 0.4075, CI [0.3502, 0.4626] | trust: the step-3 detector's belief |
| retrieval_top1 | + | 0.1188 | 0.5859, CI [0.5312, 0.6397] | risk |
| eedc | + | −0.0749 | 0.5247, CI [0.4715, 0.5801] | risk |

Three of these CIs contain 0.5, so those components' directions are not
established by this data alone; their signs come from the full-data AUC
and the fit records that the decision was not decisive
(`direction_decisive` in `data/detection_head_refit600.json`).

Step 9 now emits a **hallucination risk**; `ctx.confidence` remains the
trust score `1 - risk`.

### 2. The evaluation used the wrong instrument

`scripts/evaluate_full.py` chose the detection score by asking "is there an
answer?" and used the **step-3 XGBoost detector** whenever there was one,
falling back to the fusion only when there was none. But a benchmark that
*supplies* the answer still has `d["answer"]` non-empty — step 2 uses the
supplied answer verbatim. So the XGBoost detector was used on every row of
every dataset and the fusion was never scored at all.

That detector was fitted on HaluEval (3–30 word answers) and returns a
near-constant 0.008–0.019 on RAGTruth's 110–160 word answers. Its ROC-AUC
on RAGTruth was 0.5134 regardless of how good the rest of the pipeline was.

**The fix:** the evaluation now asks whether step 2 *generated* the answer
(`_answer_was_provided` reads step 2's `provided` flag):

| answer | score used |
|---|---|
| generated by the pipeline | step-3 XGBoost detector |
| supplied by the benchmark | the pipeline's fused risk |

**The check to run before believing any detection number:** print the spread
of `detector_score`. A near-constant score means the wrong instrument is in
use. A healthy one spans roughly 0.28–0.50.

### 3. One sign was decided by a coin flip

The fitted head gave `detector_trust` the sign **+1** (risk signal) on the
strength of a tuning-half AUC of **0.5008** — 0.0008 above the 0.5 line. The
true direction is the opposite.

`detector_trust` is `1 − hallucination_probability`, so a *higher* value
means the step-3 detector thinks the answer is more faithful. It is a
**trust** signal. Measured on 300 RAGTruth rows
(`scripts/check_detector_direction.py`):

| component | AUC read as a risk signal | direction |
|---|---|---|
| `hallucination_probability` | 0.5389, CI [0.4741, 0.5991] | risk |
| `detector_trust` | 0.4611, CI [0.4009, 0.5259] | trust |

The two sum to exactly 1.0000, which is the check that confirms they are
one measurement read in opposite directions.

The component carried 0.1934 of fitted weight and was pointing the wrong
way. `fit_detection_head.py` now records `auc_all_ci95`, `sign_source` and
`direction_decisive` for every component, and a sign inside the noise band
is taken from the full data rather than from a coin flip. Every other
component's sign was unaffected — only `detector_trust` changed.

### 4. TruthfulQA's retrieval corpus was an answer key

TruthfulQA is closed-book: it has no source passage. Its corpus was built
from the dataset's own `knowledge` field, which is the set of *correct*
answers, so retrieval found the reference answer for every row and the
verification chain read the label. The leaked number was ROC-AUC **0.9589**
with a near-binary score.

**The fix:** TruthfulQA is now run **closed-book** — retrieval is switched
off, grounding is left out, and the components that need a source
(`retrieval_top1`, `nli`, `coverage`) are dropped from the score rather
than fed a placeholder 0.0. The full audit, including why HaluEval's
similar-looking corpus is *not* a leak, is in `docs/corpus_leak_audit.md`.

## The lexical grounding term

`src/utils/grounding.py` measures the share of the answer's **content words**
that the source does *not* contain. A grounded answer reuses the source's
vocabulary; a hallucinated answer introduces facts the source does not
have, and those facts come with words the source never used.

Grounding is measured against the **top-1 retrieved document only**.
Pooling all five retrieved documents destroys the signal: the extra
documents contribute their own vocabulary, which absorbs the answer's
genuinely novel words and pushes every answer's novel rate towards the
same value.

It answers a different question from the NLI's. The NLI asks "is this claim
entailed by the evidence" — a per-claim entailment question that is hard
when the evidence is long and the answer contains facts simply not in the
source. Grounding asks the weaker but much more robust "how much of this
answer came from the source at all", per answer rather than per claim.

Before this existed, the fusion counted the NLI's mean supported-claim
confidence and the supported-claim *rate* separately — the same evidence
twice, both at claim level — and had no answer-level grounding term at all.

## Negative results

**Span-level detection is a negative result.** Span F1 0.0455 (tp=7, fp=221,
fn=73) at the default IoU 0.5: the extracted claims are whole sentences
(~100 characters) while RAGTruth's gold spans are 10–20 characters, so the
claim granularity cannot localise. The RAGTruth paper's own fine-tuned
Llama-2-13B gets span F1 52.7. See `scripts/score_span_level.py`.

**Claim-level NLI is weak on long-form answers.** The claim-level terms
(NLI mean confidence, supported-claim rate) have near-chance AUC on
RAGTruth. This is why the answer-level grounding term is needed, and it is
why the answer-level grounding term carries the largest fitted weight.

## Reproducing

```bash
# 1. the dataset (once)
python -m scripts.build_ragtruth_jsonl

# 2. the retrieval index (once)
python -m scripts.build_index --corpus data/ragtruth_corpus.jsonl \
    --out data/index_ragtruth --chunk-size 12000 --chunk-overlap 0

# 3. fit the detection head (tuning half, reported on the held-out half)
python -m scripts.fit_detection_head --limit 100 --stratify

# 4. the full evaluation
python -m scripts.evaluate_full \
    --data data/ragtruth_eval_1000_matched.jsonl \
    --index data/index_ragtruth \
    --out data/metrics_ragtruth_fitted.json

# 5. the comparison to published work
python -m scripts.compare_to_published --metrics data/metrics_ragtruth_fitted.json
```

Step 3 writes the head into `configs/default.yaml`. **Re-running it
overwrites the fitted head**, so it must be done before step 4, and the
numbers in this file must be regenerated if it is re-run.
