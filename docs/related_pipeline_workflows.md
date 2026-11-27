# Related Work — Pipelines With a Similar Architecture

**What this file is.** A survey of published papers whose *pipeline architecture*
resembles the CRISP 13-step pipeline. For each paper: the stage-by-stage
workflow, and a precise statement of where it matches CRISP and where it
differs.

**How the facts were obtained.** Every pipeline description below was read
from the paper's own method section (arXiv HTML full text) — not from
abstracts, secondary summaries, or memory. Where a detail could not be
confirmed from the paper, that is stated explicitly rather than guessed.

**Scope note.** This file covers **architecture**, not scores. The
dataset-by-dataset score comparisons already live in
`Paper/ragtruth_comparison.md` (RAGTruth) and `Paper/halueval_papers.md`
(HaluEval). This file answers a different question: *who else built a
pipeline shaped like ours?*

**On the statistics (new in §3.10–§3.12).** These new sections record, for
every paper, the **numbers behind its claims** — how many examples it was
tested on, what score it reported, how big its model was, how much data it
was trained on, and most importantly **how sure the paper is about its own
number**.

That last one is the point. A paper that says "my method scores 0.92" is
easy to write. A paper that says "my method scores 0.92, and if you repeated
the test 1,000 times the answer stays between 0.91 and 0.94" is claiming
something stronger. §3.10.4 shows that almost nobody in this field does the
second thing.

**Plain-English summary of the new sections:**

| What the question is | What the answer is | Where |
|---|---|---|
| How many examples was each paper tested on? | Between **115** and **13,867** — a very wide spread. CRISP's effective sizes are **500 / 500 / 277 unique queries** | §3.10.1 |
| What did each paper actually report? | The numbers, with what each number refers to | §3.10.2 |
| How big were their models, and how much data were they trained on? | CRISP trained on **496 rows**; Self-RAG on **145,619**; AlignScore on **4.7 million** | §3.10.3 |
| **Do the papers say how sure they are?** | **Only 2 of 19 do.** ARES gives error bands; CRISP gives error bands. The other 17 give a single number with nothing attached | **§3.10.4** |
| Can we compare our number to theirs? | **Usually no** — the papers use different metrics, so a "winner" column would be misleading | §3.11.2 |
| What is CRISP's full statistical record? | Every headline number with its interval | §3.10.5 |

**Three things worth remembering from the new sections:**

1. **Most papers report a bare number.** Only ARES (error bands via
   prediction-powered inference) and CRISP (error bands via bootstrap
   resampling) tell you how much their result could vary. The other
   seventeen hand you a single number and expect you to take it at face
   value. SelfCheckGPT is the only one that reports how much its human
   labelers agreed with each other.
2. **CRISP is the only system here that publishes error bands for every
   headline number it reports** — on all three datasets, for every metric.
3. **The test sets are wildly different sizes** (115 examples up to
   13,867), so you cannot safely compare one paper's number to another's
   without reading §3.11 first. CRISP's own effective sample sizes are
   **500 / 500 / 277 unique queries** — *not* the largest in the survey,
   but the largest among the systems that report uncertainty.

> **A note on words.** Terms like *confidence interval*, *bootstrap*,
> *significance test* and *inter-annotator agreement* are used precisely
> throughout, because this document will be read by a thesis examiner. If
> you are new to any of them, §3.13 explains each one in plain language.

**How to read this file.**

| Section | Contents |
|---|---|
| §1 | The six architectural characteristics (C1–C6) and CRISP's implementation of each |
| §2 | The landscape grid — one row per paper, one column per characteristic |
| **§3** | **The detailed comparison tables** — stage-by-stage, per-characteristic, fusion, mitigation, abstention, verifier, head-to-head, what to borrow, the gap, the statistical record (§3.10–§3.12), and a plain-English glossary (§3.13) |
| §4 | The papers one at a time, each with its workflow diagram and match/differ analysis |
| §5 | Where CRISP stands: the gap, concrete improvements, counter-arguments |
| §6 | Reference list |

**If you only read one section, read §3.** It is the quantitative version of
the argument the whole document makes.

---

## 1. The six architectural characteristics

CRISP's pipeline is defined by six features. Each paper below is scored
against them.

| # | Characteristic | CRISP's version |
|---|---|---|
| **C1** | RAG hallucination detection (not plain-text fact-checking) | retrieval-grounded verification |
| **C2** | Multi-stage pipeline with query understanding → retrieval → rerank → generation | 13 steps, 6→7 before 2 |
| **C3** | Claim-level (atomic fact) verification, not whole-answer classification | 5 claims, NLI per claim |
| **C4** | Learned or weighted fusion of multiple confidence signals into one risk score | 7 components, fitted logistic head |
| **C5** | Post-hoc mitigation that rewrites only the unsupported parts | `claim_edit`, span-level |
| **C6** | Selective prediction — abstention / refusal | `ABSTAIN` at step 11 |

---

## 2. The landscape at a glance

**Scope note.** This grid has **17 rows** for a reason: FLARE, DRAGIN and
IRCoT share a row, because all three are the retrieval-side family that
has no verification stage and no answer-side mitigation, and the six
characteristics cannot tell them apart. The finer-grained tables in §3.1
and §3.2 split them and therefore count **19** published papers. Where a
count differs between this document and elsewhere, the two scopes are
the grid (17) and the stage tables (19); the six-characteristic counts
quoted for the thesis are always the grid's.

| Paper | C1 | C2 | C3 | C4 | C5 | C6 | Nearest to CRISP at |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| **RAGChecker** | ✓ | ✓ | ✓ | — | — | — | diagnosis |
| **FAVA** | ✓ | ✓ | — | — | ✓ | — | detection + editing |
| **FActScore** | ✓ | ✓ | ✓ | — | — | — | steps 5–6 |
| **RAGAS** | ✓ | ✓ | ✓ | — | — | — | steps 5–6 |
| **SAFE** | ✓ | ✓ | ✓ | — | — | — | steps 6→8 |
| **AlignScore** | ✓ | — | ✓ | — | — | — | step 8's verifier |
| **CRAG** | ✓ | ✓ | — | ✓ | ✓ | ✓ | retrieval gating |
| **Self-RAG** | ✓ | ✓ | ✓ | ✓ | — | — | step 9's fusion |
| **Semantic Uncertainty** | — | — | — | ✓ | — | ✓ | step 4 + step 11 |
| **Sufficient Context** | ✓ | — | — | ✓ | — | ✓ | step 11's abstention |
| **A Stitch in Time** | ✓ | ✓ | ✓ | ✓ | ✓ | — | step 10's rewriting |
| **Hybrid Retrieval** | ✓ | ✓ | — | — | — | ✓ | step 6's RRF |
| **SelfCheckGPT** | — | ✓ | ✓ | ✓ | — | — | steps 5–8 |
| **FLARE / DRAGIN / IRCoT** | ✓ | ✓ | — | — | — | — | steps 1, 6 |
| **Don't Hallucinate, Abstain** | — | — | — | ✓ | — | ✓ | step 11 |
| **ARES** | ✓ | ✓ | ✓ | ✓ | — | — | step 9's fitting |
| **CoVe** | ✓ | ✓ | ✓ | — | ✓ | — | step 10 |

> **The headline finding.** No published paper found combines all six. The
> two thinnest categories are **C5 (post-hoc span rewriting)**, with only
> FAVA and "A Stitch in Time Saves Nine" qualifying, and **C6 (selective
> prediction)**, with CRAG, Sufficient Context and the two abstention
> systems. **C4 (learned multi-signal fusion)** is the third thinnest, and
> Self-RAG and ARES are its clearest precedents. Only one surveyed pipeline
> implements all three of C4, C5 and C6: CRAG. CRISP's
> contribution is the *combination*: hybrid retrieval with reranking →
> NLI claim verification → a separately fitted multi-signal fusion head →
> span-level rewriting → answer-side abstention, in one pipeline.

---

# 3. Detailed comparison tables

This section is the quantitative companion to §4. Every table here is
derived from the same method-section readings as the per-paper sections
below, so the two halves of the document cannot disagree.

## 3.1 Stage-by-stage: what each paper actually does

Read down each column to see how many papers implement a stage at all; read
across each row to see what a given paper's pipeline is made of.

**Legend.** ✓ = the stage is present. — = absent. **?** = present but *not
confirmed from the paper itself* (only CoVe's retrieval, where the abstract
does not state that external sources are used — see §4.16).

| Paper | Query understanding | Retrieval | Rerank | Generation | Claim split | Verification | Fusion | Mitigation | Abstain |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **CRISP (ours)** | ✓ | ✓ hybrid | ✓ cross-enc | ✓ | ✓ | ✓ NLI | ✓ fitted 7-comp | ✓ span edit | ✓ |
| **RAGChecker** | — | ✓ | — | ✓ | ✓ | ✓ entail | — | — | — |
| **FAVA** | — | ✓ | — | ✓ | — | ✓ learned tag | — | ✓ | — |
| **FActScore** | — | ✓ | — | ✓ | ✓ | ✓ LM prompt | — | — | — |
| **RAGAS** | — | ✓ | — | ✓ | ✓ | ✓ LLM judge | — | — | — |
| **SAFE** | — | ✓ web | — | ✓ | ✓ | ✓ LLM judge | — | — | — |
| **AlignScore** | — | — | — | — | — | ✓ align model | — | — | — |
| **CRAG** | ✓ rewrite | ✓ | ✓ | ✓ | — | — | ✓ 1 signal | ✓ evidence | ✓ |
| **Self-RAG** | ✓ | ✓ | — | ✓ | ✓ segments | ✓ self-reflect | ✓ token prob | — | — |
| **Sufficient Context** | — | — | — | ✓ | — | — | ✓ 2 signals | — | ✓ |
| **A Stitch in Time** | ✓ | ✓ Bing | — | ✓ | ✓ concepts | ✓ QA | ✓ token prob | ✓ sentence | — |
| **Hybrid Retrieval** | ✓ expand | ✓ hybrid RRF | — | ✓ | — | — | — | — | ✓ reject |
| **Semantic Uncertainty** | — | — | — | ✓ | — | ✓ entail cluster | ✓ entropy | — | ✓ |
| **SelfCheckGPT** | — | — | — | ✓ | ✓ sentences | ✓ self-consist | ✓ 3 variants | — | — |
| **Don't Hallucinate** | — | — | — | ✓ | — | — | ✓ multi-LLM | — | ✓ |
| **ARES** | — | ✓ | — | ✓ | — | ✓ 3 judges | ✓ PPI | — | — |
| **CoVe** | ✓ | ? | — | ✓ | ✓ questions | ✓ self-check | — | ✓ final answer | — |
| **FLARE** | ✓ predict | ✓ | — | ✓ | — | ✓ token conf | — | ✓ revise | — |
| **DRAGIN** | ✓ need | ✓ | — | ✓ | — | — | — | — | — |
| **IRCoT** | ✓ CoT | ✓ | — | ✓ | ✓ | — | — | — | — |

**Counting stage coverage** (counted from the table above: 19 papers, CRISP
excluded):

| Stage | Papers implementing it | Depth of the minority |
|---|---:|---|
| Generation | 18 / 19 | Only AlignScore is a standalone verifier |
| Retrieval | 14 / 19 | Only CRAG and Hybrid Retrieval do hybrid + weighting |
| Verification | 13 / 19 | Only SAFE and RAGAS use an LLM judge; the rest use entailment |
| Claim-level split | 9 / 19 | FAVA and Self-RAG split for *editing*, the rest for *scoring* |
| Query understanding | 8 / 19 | CRAG, Self-RAG, A Stitch in Time, CoVe, FLARE, DRAGIN, IRCoT, Hybrid Retrieval |
| **Fusion** | **8 / 19** | **The thin one** — see §3.3 |
| Mitigation | 5 / 19 | Only FAVA edits at *span* granularity; the rest edit evidence or the whole answer |
| Abstention | 5 / 19 | Sufficient Context is the only one with a *fitted* gate |
| Rerank | 1 / 19 | CRAG is the only one of the 19 published papers that reranks |

**The thinnest rows are the ones CRISP contributes to** — mitigation (5/19),
abstention (5/19) and fusion (8/19). Reranking is thinner still (1/19), but
CRISP is in the same category as CRAG there rather than ahead of it. CRISP's
own step 7 is a real cross-encoder rerank (`BAAI/bge-reranker-v2-m3`), which
is why the grid above marks it "✓ cross-enc" rather than "—"; it is excluded
from the counts in this table because that table covers the 19 published
papers.

## 3.2 The six characteristics, itemised

The ✓/— grid in §2 tells you *whether* a paper has a characteristic. This
table says *what form* it takes, because "has fusion" means something very
different in Self-RAG than in Sufficient Context.

| Paper | C1 scope | C2 pipeline shape | C3 verification unit | C4 fusion mechanism | C5 mitigation target | C6 gate signal |
|---|---|---|---|---|---|---|
| **CRISP** | RAG answers | 13 steps, 6→7 before 2 | atomic claim vs premise | **logistic head, 7 z-scored features, fitted** | **claim span, re-verified** | **fused risk score** |
| **RAGChecker** | RAG answers | offline diagnostic | claim vs chunk/GT | — | — | — |
| **FAVA** | LM output | retrieve→tag→edit | tagged span + type | — | entity/relation/sentence/invented spans | — |
| **FActScore** | long-form text | retrieve→split→verify | atomic fact | — | — | — |
| **RAGAS** | RAG answers | extract→judge | statement | — | — | — |
| **SAFE** | long-form text | split→search→judge | fact | — | — | — |
| **AlignScore** | any text pair | single model | text pair | — | — | — |
| **CRAG** | RAG answers | retrieve→gate→correct→gen | — | single relevance score, 3 bands | **evidence** (knowledge strips) | retrieval evaluator |
| **Self-RAG** | RAG answers | reflect→retrieve→generate | segment | weighted token probs, `w^G` | — | — |
| **Sufficient Context** | RAG answers | classify→generate→reject | — | logistic, 2 features | — | sufficiency + self-confidence |
| **A Stitch in Time** | LM output | screen→validate→rewrite | concept | min token probability | **sentence**, instruction-based | — |
| **Hybrid Retrieval** | RAG answers | expand→fuse→ground | — | — | — | retrieval adequacy prompt |
| **Semantic Uncertainty** | LM output | sample→cluster→entropy | semantic cluster | entropy over clusters | — | semantic entropy |
| **SelfCheckGPT** | LM output | sample→score | sentence | 3 score variants, merged | — | — |
| **Don't Hallucinate** | LM output | multi-model→vote | — | cross-model vote | — | knowledge-gap vote |
| **ARES** | RAG answers | annotate→judge→PPI | context/answer pair | prediction-powered inference | — | — |
| **CoVe** | RAG answers | draft→plan→verify→answer | verification question | — | **whole answer** | — |
| **FLARE** | LM output | predict→retrieve→revise | — | — | sentence | token confidence |
| **DRAGIN** | LM output | need→retrieve→loop | — | — | — | — |
| **IRCoT** | multi-hop QA | CoT→retrieve→loop | — | — | — | — |

**The three rows that matter most for the thesis claim:**

| Claim | CRISP's position | Strongest competing position | Verdict |
|---|---|---|---|
| "We fuse multiple signals into one risk score" | 7 z-scored features, fitted logistic head, explicit sign map | Self-RAG (learned token probs), Sufficient Context (logistic, 2 features), ARES (PPI) | **CRISP is stronger on feature count and sign discipline**; all three are stronger on *training rigor* — ARES especially |
| "We mitigate by rewriting only what is unsupported" | per-claim span edit + re-verification loop | FAVA (learned span tagger), A Stitch in Time (instruction repair) | **CRISP is unique in re-verifying its own edit**; the others are stronger in *edit quality* — FAVA has a trained model, ours uses heuristics |
| "We abstain when the answer is risky" | fused risk vs thresholds 0.30/0.70 | Sufficient Context (fitted logistic gate) | **CRISP has a richer gate signal**; Sufficient Context has a *fitted* gate and a *validated* rejection gain (2–10%) |

## 3.3 The fusion comparison (C4) — the third thinnest category

Since C4 is where CRISP claims a contribution, here is every fusion
mechanism found, in full detail.

| Paper | Signals fused | How combined | Fitted or hand-set | Learned on |
|---|---|---|---|---|
| **CRISP** | similarity, retrieval_top1, nli, coverage, eedc, detector_trust, grounding | logistic head over z-scored features, explicit sign map | **fitted** (logistic regression, 496 rows) | labelled CRISP rows |
| **Self-RAG** | IsRel, IsSup, IsUse | `S = Σ_G w^G · s_t^G`, normalised token probs | learned critic + hand-set `w^G` (1.0/1.0/0.5) | 4k–20k GPT-4-labelled examples per type |
| **Sufficient Context** | context sufficiency, self-rated confidence | logistic regression → rejection | **fitted** | labelled RAG outputs |
| **ARES** | context relevance, faithfulness, answer relevance | prediction-powered inference w/ human correction | trained judges + statistical correction | synthetic + small human set |
| **A Stitch in Time** | min token probability per concept | threshold on the single minimum | threshold | — |
| **CRAG** | retrieval relevance score | 3-band threshold (correct/ambiguous/incorrect) | thresholds, per dataset | — |
| **Semantic Uncertainty** | samples → clusters | entropy over clusters | — | — |
| **SelfCheckGPT** | BERTScore, QA, n-gram | normalised merge of 3 variants | — | n-gram model trained on R + samples |
| **Don't Hallucinate** | multiple LLMs' verdicts | cooperative / competitive voting | — | — |

**What CRISP has that nobody else has:** a *separately fitted* head over
*heterogeneous* features, with the sign of every component stated explicitly
in the config:

| Component | Sign | Direction | Fitted weight | Component std |
|---|:-:|---|---:|---:|
| `similarity` | +1 | risk | 0.571052 | 0.139799 |
| `retrieval_top1` | +1 | risk | 0.118817 | 0.365418 |
| `nli` | −1 | trust | 1.249675 | 0.267181 |
| `coverage` | −1 | trust | −1.034543 | 0.273714 |
| `eedc` | +1 | risk | −0.074934 | 0.052152 |
| `detector_trust` | −1 | trust | 0.322765 | 0.230007 |
| `grounding` | −1 | trust | 0.588701 | 0.266188 |

(The sign map records *direction*; the fitted weight records *magnitude*
after z-scoring. The two are independent — which is why `eedc` has sign +1
but a negative fitted weight: it is directionally right yet carries almost
no information, `component_std` 0.052152, the smallest of the seven.)

The sign map is not cosmetic — the earlier hand-set fusion scored
ROC-AUC 0.360, i.e. *below chance*, precisely because the signs were wrong.

**What CRISP lacks that the others have:** ARES's prediction-powered inference
gives a statistical guarantee on a small human-labelled set; Sufficient
Context's gate is fitted on data it did not generate; Self-RAG's critic is
trained on 4k–20k GPT-4-labelled examples per reflection type. CRISP's head
is fitted on 496 rows from CRISP's own pipeline. **That is the honest
weakness of the fusion contribution** and it should be stated in the thesis
rather than hidden.

## 3.4 The mitigation comparison (C5) — the thinnest category

**Scope note.** Like §3.5, this table is broader than the "Mitigation" column
of §3.1: it includes **evidence-side** correction (CRAG, Hybrid Retrieval)
alongside **answer-side** repair (FAVA, A Stitch in Time, CoVe, FLARE,
CRISP). The distinction is the whole point of the table — see the "What is
changed" column.

| Paper | What is changed | Granularity | Evidence used to fix | Re-verified after edit? | Reported gain |
|---|---|---|---|:-:|---|
| **CRISP** | the **answer** | claim span | retrieved premise + NLI | **✓ loop re-runs 3→5→8→4→9→10→11** | measured Δ = −0.007 (16 rewritten rows), CI [−0.022, +0.007], p = 0.404 |
| **FAVA** | the **answer** | tagged span + 6-way type | retrieved context | ✗ | trained tagger, strong edits |
| **A Stitch in Time** | the **answer** | sentence | Bing / self-inquiry | ✗ | hallucination 47.5% → 14.5% |
| **CoVe** | the **answer** | whole answer | verification answers | ✓ implicitly | +23% F1 (MultiSpanQA), +28% FactScore |
| **CRAG** | the **evidence** | knowledge strip | web-search results | ✗ | corrects bad retrieval |
| **FLARE** | the **answer** | sentence | re-retrieved context | ✗ | — |
| **Hybrid Retrieval** | the **evidence** | doc set | — | ✗ | hallucination rate ↓ on all 4 metrics |
| **ARES** | nothing (statistics) | — | — | — | — |
| **Self-RAG** | nothing (decoding) | — | — | — | — |

**The honest reading.** CRISP's `claim_edit` is the *only* design in the set
that (a) edits at claim-span granularity, (b) preserves the rest of the
answer byte-for-byte, and (c) re-runs verification on its own output. It is
also the only one with a **measured mitigation delta that is ≈ zero**
(−0.007). The mechanism is right; the effect size is not there yet. §5.2 of
this document lists the concrete causes — a heuristic replacement source and
an NLI whose false-positive rate drives unnecessary rewrites.

## 3.5 The abstention comparison (C6)

**Scope note.** This table is deliberately *broader* than the "Abstain" column
of §3.1. §3.1 counts only systems that **refuse to produce an answer**;
this table also counts systems that **gate or repair the answer in flight**
(FLARE's per-token confidence, CRAG's three-band retrieval routing). Those
are weaker forms of the same idea, and the distinction matters for the
thesis claim — CRISP's `ABSTAIN` is a genuine refusal, not a redirect.

| Paper | Gate input | Gate form | Fitted? | Reported benefit | Risk of over-abstention |
|---|---|---|:-:|---|---|
| **CRISP** | fused risk (7 features) | 2 thresholds: pass ≥ 0.70, abstain ≤ 0.30 | head fitted; thresholds hand-set | — | unknown |
| **Sufficient Context** | sufficiency + self-confidence | logistic → reject below threshold | ✓ | +2–10% correct answers | paper reports fine-tuning to abstain *fails* to reduce hallucination |
| **Semantic Uncertainty** | semantic entropy | entropy threshold | ✓ | standard | abstains on benign phrasing variation |
| **Hybrid Retrieval** | retrieval adequacy | LLM prompt "insufficient information" | ✓ | rejection rate reported | — |
| **CRAG** | retrieval score | 3-band threshold | per-dataset thresholds | corrects bad retrieval | — |
| **Don't Hallucinate** | multi-LLM votes | vote threshold | ✓ | — | multi-model cost |
| **FLARE** | token confidence | per-token | ✓ | — | — (redirects, not refuses) |

**The key structural difference.** Sufficient Context gates on *evidence*
sufficiency; CRISP gates on *answer* risk. These fail in different places:
a system can have sufficient evidence and still produce a wrong answer (only
CRISP's step 8 catches that), and it can have insufficient evidence while the
answer happens to be correct (Sufficient Context rejects a good answer that
CRISP would keep). CRISP's step 11 sits *after* verification, so it can
distinguish these cases; Sufficient Context's gate sits *before* generation.

## 3.6 The verifier comparison (step 8 and its equivalents)

| Paper | Verifier | Type | Granularity | Truncation risk | Cost |
|---|---|---|---|---|---|
| **CRISP** | mDeBERTa-v3-base-xnli (278M) | entailment classifier | atomic claim vs premise | mitigated by claim-conditioned premise selection (1800 chars) | 1 forward pass / claim |
| **AlignScore** | AlignScore 355M | trained alignment function | text pair | same mitigation applies | 1 forward pass / pair |
| **RAGChecker** | Llama3-70B | LLM entailment | claim vs chunk | yes | very high |
| **FActScore** | InstructGPT zero-shot | LLM prompt | atomic fact | yes | high |
| **RAGAS** | LLM judge | LLM prompt | statement | yes | high |
| **SAFE** | LLM + web search | LLM reasoning | fact | yes | very high |
| **FAVA** | fine-tuned Llama2-7B tagger | generative tagger | span | no | high |
| **A Stitch in Time** | QA over retrieved evidence | QA model | concept | no | medium |
| **SelfCheckGPT** | BERTScore / QA / n-gram | self-consistency | sentence | no | N samples per answer |
| **ARES** | fine-tuned small judges | classifier | context/answer | no | low |

**CRISP's advantage:** a small off-the-shelf NLI classifier is far cheaper
and more reproducible than a 70B LLM judge, and the entailment verdict is a
*label*, which is what makes step 4's entropy over verdicts possible at all.

**CRISP's weakness:** `unknown` verdicts are frequent and ambiguous, and the
NLI's false-positive rate is the actual bottleneck on the mitigation side.
AlignScore is trained on 4.7M examples *including* fact verification, so it
is a plausible upgrade — and it is a drop-in, because the claim-conditioned
premise selection happens upstream of the verifier.

## 3.7 Head-to-head with the closest five

The five papers a reviewer is most likely to put side by side with CRISP.

| Dimension | **CRISP** | **FAVA** | **A Stitch in Time** | **Self-RAG** | **Sufficient Context** |
|---|---|---|---|---|---|
| Detection unit | atomic claim | tagged span | concept | segment | whole answer |
| Detection signal | 7 fused features | learned tag | min token prob | reflection tokens | sufficiency + confidence |
| Fusion | **7-component fitted head** | none | threshold | token-prob weights | 2-feature logistic |
| Mitigates | ✓ claim span | ✓ span + type | ✓ sentence | ✗ | ✗ |
| Re-verifies own edit | ✓ | ✗ | ✗ | ✗ | ✗ |
| Abstains | ✓ | ✗ | ✗ | ✗ | ✓ |
| Retrieval | hybrid + cross-enc | Contriever | Bing | learned | none (context given) |
| Verifier | 278M NLI | 7B fine-tuned | QA | generator | LLM/FLAMe |
| Cost | 4.18 s/query mean (3.96 median), detection path | high | high | high | high |
| **C1–C6 covered** | **6 / 6** | 3 / 6 | 5 / 6 | 4 / 6 | 3 / 6 |

### 3.7.1 RAGChecker

| Dimension | **CRISP** | **RAGChecker** |
|---|---|---|
| Purpose | defend at inference | measure offline |
| Claim unit | atomic claim (step 5) | claim (RefChecker framework) |
| Failure separation | implicit (step 6 vs step 8) | **explicit** (retrieval vs generation) |
| Fusion | 7-component fitted head | none |
| Mitigates | ✓ | ✗ |
| Metrics | ROC-AUC, AUPR, claim F1, VSR, PR@90 | Faithfulness, noise sensitivity, self-knowledge, context utilisation, overall P/R/F1 |
| Overlap with CRISP's metric set | — | **Relevant/Irrelevant Noise Sensitivity** map to CRISP's `grounding` feature |

### 3.7.2 Hybrid Retrieval (Mala et al.)

| Dimension | **CRISP** | **Hybrid Retrieval** |
|---|---|---|
| Retrieval | BGE-M3 dense + BM25 + RRF + cross-encoder rerank | dense + sparse + query expansion + **dynamic** RRF weighting |
| Weighting | **inert** (`weight_dense`/`weight_bm25` never read) | per-query specificity score |
| Reranker | bge-reranker-v2-m3 | none |
| Verification | NLI per claim | none |
| Mitigation | claim-span edit | prompt the model to refuse (evidence-side) |
| Ablation | not run | **sparse 0.724 → dense 0.768 → hybrid 0.897 MAP@3** |
| **Threat to CRISP** | — | shows hybrid retrieval alone cuts hallucination on all 4 metrics across 6 datasets |

## 3.8 What CRISP borrows from each paper

Concrete, actionable — each row is a change to CRISP's code or evaluation.

| Paper | What to adopt | Where it goes | Effort | Changes published numbers? |
|---|---|---|---|---|
| **Hybrid Retrieval** | per-query RRF weighting from specificity | `src/retrieval/hybrid.py` (currently inert) | medium | **yes** |
| **AlignScore** | swap the step-8 verifier | `src/models/nli_deberta.py` → align scorer | low | **yes** |
| **Semantic Uncertainty / SelfCheckGPT** | real multi-sample self-consistency | step 4's `self_consistency` proxy | high | **yes** |
| **DRAGIN** | apply the discarded query rewrite | step 1 (`variants[0]` → best variant) | low | **yes** |
| **RAGChecker** | report relevant vs irrelevant noise sensitivity | step 13 metrics | medium | no (new metric) |
| **FActScore** | exclude abstentions from the model-level average | step 13 aggregation | low | no (reporting change) |
| **FAVA** | six-type hallucination taxonomy | step 8 verdict labelling | medium | partly |
| **A Stitch in Time** | rewrite using the retrieved premise as the replacement source | step 10 `_replacement_for` | medium | **yes** |
| **Sufficient Context** | fit the step-11 thresholds instead of hand-setting 0.30/0.70 | step 11 config | low | **yes** |
| **ARES** | prediction-powered inference on the detection metric | step 13 + evaluation | high | **yes** |

## 3.9 The gap, stated as a table

| # | Combination | Who has it |
|---|---|---|
| 1 | RAG detection + multi-stage pipeline + claim-level verification | RAGChecker, FActScore, RAGAS, SAFE, ARES, Self-RAG |
| 2 | … + learned multi-signal fusion | Self-RAG, ARES, Sufficient Context, A Stitch in Time |
| 3 | … + post-hoc span rewriting | FAVA, A Stitch in Time |
| 4 | … + abstention | CRAG, Sufficient Context |
| 5 | **RAG detection + multi-stage pipeline + claim-level verification + learned multi-signal fusion + post-hoc span rewriting + abstention** | **nobody — CRISP is the only one** |

**The defensible thesis sentence:** *CRISP is, to the best of this survey's
knowledge, the only published RAG pipeline that combines claim-level NLI
verification, a separately fitted multi-signal fusion head, span-level
post-hoc rewriting with self-re-verification, and answer-side abstention in
a single pipeline.* Every word of that sentence is backed by §3.1–§3.5.

## 3.10 The statistical record — what each paper actually measured

**What this section is for.** §3.1–§3.9 compared *how each system is built*.
This section compares *how strong the evidence is behind each system's
claims*. When a reviewer asks "why should I believe your number more than
theirs?", they are asking about the rows here.

**Where these numbers come from.** Every value below was read from the
paper's own full text (the arXiv HTML version, not the abstract). If a paper
does not report something, the cell says **"not reported"** — that is a
finding, not a blank space.

### 3.10.1 Evaluation sets and sample sizes

**How many examples was each paper tested on?** This matters more than any
other column in this document. A score measured on 50 examples and a score
measured on 13,867 examples are not the same kind of claim.

| Paper | Datasets | Evaluation size (exact counts as stated) |
|---|---|---|
| **CRISP** | HaluEval (QA), TruthfulQA (closed-book), RAGTruth | **1,000 rows each** — HaluEval 500 unique queries, TruthfulQA 500, RAGTruth **277**; label balance 500/500; n_bootstrap = 1,000 |
| **RAGChecker** | 10 domains: ClapNQ, NovelQA, RobustQA × 7, KIWI | **4,162 queries**; per-domain 71–511 q; pairwise human label set **280 instances** |
| **FAVA** | FAVA detection set, 150-prompt source benchmark, FActScore biographies | **364 annotated passages** test (471 collected); source benchmark **300 responses** from 150 prompts; human eval **50 outputs** per system |
| **FActScore** | Human FActScore set; 13-subject automatic eval | **183 entities**; **549 prompts** (3 models × 183); **6,500 generations**; 500 biographies/subject |
| **RAGAS** | WikiEval | **50 instances** (one per Wikipedia page, 3 dimensions) |
| **SAFE** | LongFact (Concepts + Objects, 38 topics) | **2,280 prompts**; main benchmark uses **250** LongFact-Objects; human set **496 pairs / 16,011 facts** |
| **AlignScore** | **22 datasets** (SummaC, TRUE, others) | **not stated in paper** — per-dataset instance counts absent |
| **CRAG** | PopQA, Biography, PubHealth, Arc-Challenge | PopQA test **1,399 queries**; other three **not stated in paper** |
| **Self-RAG** | PopQA, TriviaQA, PubHealth, ARC-C, Biography, ASQA | PopQA long-tail **1,399**; TriviaQA **11,313**; other four **not stated in paper** |
| **Sufficient Context** | FreshQA, Musique-Ans, HotPotQA | FreshQA **452**; Musique **500**; HotPotQA **500**; gold sufficiency set **115** |
| **A Stitch in Time** | custom article generation; HotPotQA; false-premise QA | **150 topics** → **750 sentences**; Vicuna-13B 50 sentences; multi-hop QA **50**; false-premise **50** |
| **Hybrid Retrieval** | HaluBench (DROP, HaluEval, RAGTruth, FinanceBench, PubMedQA, COVIDQA) | HaluBench **13,867**; hallucination eval **300** (50/source, 25 PASS/25 FAIL); FAIL-only **125** |
| **Semantic Uncertainty** | CoQA, TriviaQA | CoQA **~8,000**; TriviaQA **8,000**; **10 samples** per question |
| **SelfCheckGPT** | WikiBio | **238 passages / 1,908 sentences**; two-annotator subset **201 sentences** |
| **Don't Hallucinate** | MMLU, K-Crosswords, HellaSwag, Propaganda, AmbigQA, ElectionQA23 | test splits 200–1,007; **67–1,094** per dataset |
| **ARES** | KILT (NQ, HotpotQA, FEVER, WoW), SuperGLUE (MultiRC, ReCoRD), AIS | WoW **707**, CNN/DM **510**; PPI set **300**; mock-system baseline **1,350** labels |
| **CoVe** | Wikidata, Wiki-Category, MultiSpanQA, longform biographies | **56** / **55** / **418** questions; biography count **not stated in paper** |
| **FLARE** | 2WikiMultihopQA, StrategyQA, ASQA, WikiAsp | **500 / 229 / 500 / 500** (capped "due to the cost of running experiments") |
| **DRAGIN** | 2WikiMultihopQA, HotpotQA, IIRC, StrategyQA | **1,000 / 1,000 / 954 / 1,000** = **3,954** |
| **IRCoT** | HotpotQA, 2WikiMultihopQA, MuSiQue, IIRC | **500** test + **100** tuning per dataset; human error set **160**; human CoT set **80** |

**What the spread tells us.** The smallest set is **115** examples
(Sufficient Context's gold set). The largest is **13,867** (HaluBench).
CRISP's **1,000** sits in the middle of that range — but it is worth
checking *what* is being counted, because the largest numbers are not
always what they look like. HaluBench's 13,867 is a pool of retrieval
documents; only **300** of those are actually scored for hallucination.
Self-RAG's 11,313 is the full TriviaQA test set, not a hallucination
benchmark. Semantic Uncertainty's 16,000 and FActScore's 6,500 are larger
than CRISP's, so **CRISP does not have the largest evaluation in the
survey** — and this document does not claim it does.

**The honest comparison is by *effective* sample size and by what is
scored:**

| Paper | Effective evaluation size | Scored for hallucination? | Reports CIs? |
|---|---:|:-:|:-:|
| Semantic Uncertainty | 16,000 questions | general QA | ✗ |
| FActScore | 6,500 generations (549 prompts) | fact-level | ✗ |
| RAGChecker | 4,162 queries | ✓ | ✗ |
| DRAGIN | 3,954 questions | QA answer quality | ✗ |
| Self-RAG | 11,313 queries | general QA | ✗ |
| Hybrid Retrieval | 13,867 pool → **300 scored** | ✓ | ✗ |
| **CRISP** | **1,000 rows / 500 unique q** (HaluEval, TruthfulQA) and **277 unique q** (RAGTruth) | ✓ | ✓ **all metrics** |
| ARES | 300 | ✓ | ✓ (PPI) |
| FAVA | 364 passages | ✓ | ✗ |
| Sufficient Context | 115 gold / 452–500 eval | ✓ | ✗ |
| RAGAS | 50 | ✓ | ✗ |

**Where CRISP stands.** It is *not* the largest evaluation in this survey.
It is **the largest evaluation among the systems that report uncertainty on
their results** — 500 unique queries against ARES's 300, and every paper
larger than CRISP publishes no interval at all, so their extra size buys a
number nobody can check. Note also that CRISP's **277 unique RAGTruth
queries** are fewer than ARES's and Hybrid Retrieval's 300, so the
per-dataset effective sizes are 500 / 500 / 277 and should always be quoted
that way rather than as "1,000".

### 3.10.2 Headline reported numbers

**What number did each paper report?** Each cell states the number *and what
it refers to*. Do not compare these numbers directly with each other —
different papers use different metrics, which is explained in §3.11.2.

| Paper | Headline reported result |
|---|---|
| **CRISP** | HaluEval ROC-AUC **0.9251** · TruthfulQA **0.6355** · RAGTruth **0.6127**; claim F1 0.8107 / 0.0 / 0.7269; mitigation Δ **−0.007** |
| **RAGChecker** | Best system E5-Mistral+GPT-4: P **62.0** / R **53.0** / F1 **52.7**; human-human upper bound overall **70.09** Pearson |
| **FAVA** | Detection **+22.1%** over ChatGPT (binary **+14.2%**); editing FActScore Δ **+4.4** (Alpaca-7B) to **+9.3** (Alpaca-13B); human eval **2.4** errors/response vs ChatGPT **1.9** |
| **FActScore** | Human FActScore: InstructGPT **42.5%**, ChatGPT **58.3%**, PerplexityAI **71.5%**; estimator error **<2%**; cross-estimator Pearson r = **0.99** |
| **RAGAS** | Agreement with human preference — faithfulness **0.95**, answer relevance **0.78**, context relevance **0.70** |
| **SAFE** | SAFE–human agreement **72.0%**; re-adjudication SAFE correct in **76%** of 100 disputed cases; F1@64 GPT-4-Turbo **95.0**; cost **$0.19/response** vs $4.00 human |
| **AlignScore** | SummaC AUC-ROC avg **87.4** (base) / **88.6** (large); TRUE AUC-ROC avg **86.3** / **87.4** |
| **CRAG** | PopQA acc RAG **52.8** → CRAG **59.8** (SelfRAG-LLaMA2-7B); PubHealth **39.0** → **75.6**; retrieval evaluator acc **84.3** vs ChatGPT **58.0** |
| **Self-RAG** | PopQA 7B **54.9** / 13B **55.8**; PubHealth **72.4** / **74.5**; ARC-C **67.3** / **73.1**; Bio FactScore **81.2** / **80.2** |
| **Sufficient Context** | Autorater F1 **0.935** / acc **0.930** (Gemini 1.5 Pro 1-shot); selective generation gains **+2–10%** correct answers; **>10%** for Gemma-27B on HotPotQA |
| **A Stitch in Time** | Detection recall **~88%**; mitigation success **57.6%**; hallucination rate **47.4% → 14.53%** |
| **Hybrid Retrieval** | MAP@3 sparse **0.724** / dense **0.768** / **hybrid 0.897**; NDCG@3 hybrid **0.915**; hallucination rate 4–22% per dataset |
| **Semantic Uncertainty** | AUROC — CoQA **0.77** (sem-entropy) vs **0.66** (distinct answers); TriviaQA **0.83** vs **0.79** |
| **SelfCheckGPT** | Passage-level AUC-PR **87.33** (combination); Pearson **69.05**, Spearman **67.77** vs human labels |
| **Don't Hallucinate** | Beats strongest baseline in **9 of 12** settings; avg **+10.1%** reliable accuracy; max abstain-accuracy gain **+19.3%**; two-stage cut incorrect-answer rate up to **56.6%** |
| **ARES** | Context/answer relevance acc — NQ **79.3% / 97.2%**, HotpotQA **92.3% / 81.3%**, FEVER **88.4% / 78.5%**; **+59.3 / +14.4** points over RAGAS |
| **CoVe** | MultiSpanQA F1 **0.39** → **0.48** factored (**+23%**); biography FactScore **55.9** → **71.4** (**+28%**) |
| **FLARE** | 2WikiMultihopQA EM **51.0** / F1 **59.7** vs no-retrieval EM **28.2** / F1 **36.8** (**+22.8 EM**); StrategyQA EM **77.3** |
| **DRAGIN** | LLaMA-2-13B-chat — 2Wiki EM **0.304** / F1 **0.3931**; HotpotQA EM **0.314** / F1 **0.4238**; StrategyQA acc **0.689** |
| **IRCoT** | HotpotQA F1 **59.1 ± 0.9**, 2Wiki **66.5 ± 1.4**, MuSiQue **30.8 ± 0.2**, IIRC **42.5 ± 2.1** (Flan-T5-XXL); factual errors cut **50%** (HotpotQA) / **40%** (2Wiki) |

### 3.10.3 Models, training data, and what was fitted

**How big was each model, and how much data was it trained on?** This is
where the training effort of the field becomes visible — and where CRISP's
own contribution looks smallest.

| Paper | Generator / verifier | Trained? | Training-data size |
|---|---|---|---|
| **CRISP** | mDeBERTa-v3-base-xnli verifier (278M); BGE-M3 + BM25 + bge-reranker-v2-m3 | **yes** | Fusion head fitted on **496 rows** (248 tune / 248 holdout) over 7 z-scored features |
| **RAGChecker** | Llama3-70B claim checker; E5-Mistral-7B retriever; GPT-4 Turbo | no | — |
| **FAVA** | fine-tuned Llama2-Chat-7B tagger; Contriever-MSMARCO | **yes** | **35,074 synthetic instances** (30,074 Wikipedia + 5,000 QA) |
| **FActScore** | InstructGPT `text-davinci-003`; GTR retriever | no | — |
| **RAGAS** | `gpt-3.5-turbo-16k`; `text-embedding-ada-002` | no | — |
| **SAFE** | GPT-3.5-Turbo + Serper/Google Search | no | — (its 496 is a *human eval* set, not training) |
| **AlignScore** | RoBERTa-base 125M / RoBERTa-large 355M | **yes** | **4.7M** examples, 15 datasets, 7 task families |
| **CRAG** | LLaMA2-7B; fine-tuned T5-large (0.77B) retrieval evaluator | **yes** | PopQA remainder of 14k (1,399 held out) |
| **Self-RAG** | Llama2-7B/13B generator + 7B critic; Contriever | **yes** | Generator **145,619** instruction pairs; critic **4k–20k per reflection type** |
| **Sufficient Context** | Gemini 1.5 Pro; FLAMe-RM-24B; TRUE-NLI T5-11B; LoRA on Llama-3.1-8B / Mistral-7B | partly (LoRA) | **2,000** instances per dataset; **400** "I don't know" replacements |
| **A Stitch in Time** | GPT-3 `text-davinci-003`; Bing Search | no | — |
| **Hybrid Retrieval** | LLaMA-3-8B-Instruct; `all-mpnet-base-v2` | no | — |
| **Semantic Uncertainty** | OPT-30B; DeBERTa-large clustering | no | — |
| **SelfCheckGPT** | GPT-3 175B generator; roberta-large BERTScore; T5-large + Longformer | no (external checkpoints) | component training sizes **not stated in paper** |
| **Don't Hallucinate** | Mistral-7B, LLaMA2-70B-chat, ChatGPT | partly (instruction-tuning) | one example per held-out question (1,000–1,094) |
| **ARES** | DeBERTa-v3-Large (304M) judges; FLAN-T5-XXL (11.3B) data generator | **yes** | synthetic count **not stated in paper**; negatives balanced to positives |
| **CoVe** | Llama 65B; FactScore checker "Llama + Retrieval + NP" | no | — |
| **FLARE** | GPT-3.5 `text-davinci-003` | no | — |
| **DRAGIN** | LLaMA2-Chat 7B/13B; Vicuna-13B-v1.5; BM25 | no | — |
| **IRCoT** | GPT-3 `code-davinci-002` 175B; Flan-T5 base→XXL (11B) | no | — |

**The training-data gap.** CRISP's head is fitted on **496 rows**. Self-RAG
trains on **145,619** instruction pairs plus 4k–20k labels *per reflection
type*; AlignScore on **4.7M**; FAVA on **35,074**; ARES on a synthetic corpus
whose size the paper does not state. **This is the single largest weakness in
CRISP's fusion contribution** and §3.3 already says so; the table makes the
gap quantitative rather than rhetorical.

### 3.10.4 Statistical rigour — the decisive comparison

**Does each paper say how sure it is?** This is the table a reviewer should
be shown, because it is the one place where CRISP is clearly ahead.

A "yes" in the table means the paper gives you *some* way of knowing how much
its number could move: a confidence interval, an error bar, a standard
deviation, a significance test, bootstrap resampling, or several random seeds.
Most papers give you one number and nothing else.

| Paper | CIs / error bars | Significance test | Bootstrap | Multiple seeds | Inter-annotator agreement | Verdict |
|---|:-:|:-:|:-:|:-:|---|---|
| **CRISP** | **yes** — 95% CI on ROC-AUC, AUPR, PR@90, VSR | **yes** — paired bootstrap p-values (mitigation Δ p = 0.404) | **yes** — 1,000 resamples over unique queries | no | not reported (no human labels) | **only paper in the set with CIs on every headline metric** |
| **RAGChecker** | no | no | no | no | 90.95% raw agreement, no κ | point estimates |
| **FAVA** | no | no | no | no | states "Cohen kappa scores" but **prints no κ**; 75.1% / 60.3% | point estimates |
| **FActScore** | no | no | no | no | 96 / 90 / 88% agreement rate, no κ | point estimates |
| **RAGAS** | no | no | no | no | ~95 / 95 / 90%, no κ | point estimates |
| **SAFE** | no | **yes** — 2 Pearson p-values (0.205, **0.031**) | no | no | not reported | partial (correlation only) |
| **AlignScore** | no | no | no | **no** — single seed 2022 | not reported | point estimates |
| **CRAG** | no | no | no | no | none (GPT-4 labels only) | point estimates |
| **Self-RAG** | no | no | no | no | 95 / 95 / 90 / 80% agreement, no κ | point estimates |
| **Sufficient Context** | no | no | no | no | not reported | point estimates |
| **A Stitch in Time** | no | no | no | no | not reported (author-annotated) | point estimates |
| **Hybrid Retrieval** | no | no | no | no | **single annotator**, no IAA possible | point estimates |
| **Semantic Uncertainty** | no | no | no | no | not reported | point estimates |
| **SelfCheckGPT** | no | no | no | no | **Cohen's κ = 0.595** (3-label) / **0.748** (2-class) | **κ only** |
| **Don't Hallucinate** | no | no | no | no | not reported | point estimates |
| **ARES** | **yes** — PPI 95% intervals, mean width 7.4 / 6.1 points | no (PPI is the guarantee) | no | no | not reported | **CIs, via PPI** |
| **CoVe** | no | no | no | no | not reported | point estimates |
| **FLARE** | no | no | no | no | not reported | point estimates |
| **DRAGIN** | no | no | no | no | none | point estimates |
| **IRCoT** | **partial** — ±std over 3 demonstration sets | no | no | no | not reported | descriptive std only |

**Three things this table shows.**

1. **Only ARES and CRISP report confidence intervals** — and they get there
   by opposite routes. ARES uses prediction-powered inference against a
   150–300-item human set; CRISP uses 1,000 bootstrap resamples over unique
   queries. ARES's intervals are *narrower in absolute terms* (6.1–7.4
   points) because PPI needs a human-labelled ground truth to condition on;
   CRISP's are the honest consequence of resampling its own evaluation set.
2. **SelfCheckGPT is the only paper reporting a formal inter-annotator
   agreement statistic** (κ = 0.595 / 0.748). FAVA says it computed "Cohen
   kappa scores" but never prints a number — a reporting omission a
   reviewer could reasonably press.
3. **Two papers use the word "significantly" with no test behind it**
   (Sufficient Context: "the training was not smooth"; Semantic
   Uncertainty: "does not significantly improve the performance"). Neither
   reports a p-value. This is the kind of detail §5.3 anticipates.

### 3.10.5 CRISP's own statistical record, in full

**For comparison, here is everything CRISP reports — with the range around
each number.** This is the table to check any CRISP figure against.

| Dataset | ROC-AUC (95% CI) | AUPR (95% CI) | Claim F1 | VSR | PR@90 (95% CI) | n rows / unique |
|---|---|---|---|---|---|---|
| **HaluEval** | **0.9251** [0.9113, 0.9401] | **0.9217** [0.9057, 0.9383] | 0.8107 (P 0.467 / R 0.810) | 0.701 [0.673, 0.726] | 0.7996 [0.7732, 0.8443] | 1,000 / 500 |
| **TruthfulQA** (closed-book) | **0.6355** [0.6107, 0.6594] | **0.6450** [0.6174, 0.6750] | 0.0 (no corpus) | 1.000 [1.000, 1.000] | 0.5351 [0.5257, 0.5512] | 1,000 / 500 |
| **RAGTruth** | **0.6127** [0.5494, 0.6581] | **0.6112** [0.5752, 0.6453] | 0.7269 (P 0.964 / R 0.626) | 0.315 [0.2455, 0.3837] | 0.5311 [0.5203, 0.5488] | 1,000 / 277 |

Source files: `data/metrics_halueval.json`,
`data/metrics_truthfulqa_closedbook.json` (the honest file — the other two
TruthfulQA files are quarantined), `data/metrics_ragtruth_fitted.json`.
`n_bootstrap = 1000` in all three.

**Fusion head** (`data/detection_head_refit600.json`): fitted on **496 rows**
(248 tune / 248 holdout), 7 components. Hold-out ROC-AUC **0.6185**
[0.5371, 0.6943]; tune 0.7093; all 0.6609.

**Mitigation** (`data/mitigation_results.json`): 500 rows / 250 unique
queries; **16 regenerated** (8 unique queries), 484 kept, 86 guarded. On the
**regenerated rows** (the only rows the rewrite actually touched):
BERTScore Δ = **−0.0069**, CI95 [−0.0217, +0.0069], **p = 0.404**; ROUGE-L
Δ = +0.0152, CI95 [−0.0135, +0.0477], p = 0.500. On **all 500 rows** the
BERTScore Δ is **−0.00022**, CI95 [−0.00074, +0.00022], p = 0.439 — the
rewrite is a no-op on the 484 kept rows, so the all-rows figure is
mechanically near zero by construction. **The quoted "−0.007" is the
regenerated-rows figure**, and the CI includes zero in both cases: the
honest statement is *no measurable effect*, not *harmful*.

**Latency** (detection path — retrieval + verification; the benchmark
supplies the answer so step 2 never calls the generator): HaluEval mean
**4.18 s** (median 3.96 s, range 0.97–10.30 s); TruthfulQA mean **4.12 s**;
RAGTruth mean **10.01 s** (median 6.65 s).

## 3.11 The comparison table with statistical information

This is the table that puts every paper with a similar workflow next to
CRISP, with the statistical information attached to each row.

**One thing to know before reading it:** because the papers use different
metrics (§3.11.2), the table shows *each paper's own metric* rather than
forcing everyone onto one scale. That means the table is good for comparing
**how well each paper was tested**, but not for declaring a winner on
accuracy.

### 3.11.1 Master statistical comparison

| Paper | Eval size | Headline metric (own scale) | Value | Uncertainty reported | Trained on | Fitted head? |
|---|---|---|---:|---|---|:-:|
| **CRISP** | 1,000 rows × 3 datasets (500 / 500 / **277** unique q) | ROC-AUC | **0.9251 / 0.6355 / 0.6127** | **95% CI, 1,000 bootstrap** | 496 rows | **yes** (holdout 0.6185) |
| **RAGChecker** | 4,162 queries | P / R / F1 | 62.0 / 53.0 / 52.7 | none | — | no |
| **FAVA** | 364 passages | detection gain vs ChatGPT | +22.1% | none | 35,074 synthetic | tagger only |
| **FActScore** | 183 entities / 6,500 gens | FActScore | 42.5 / 58.3 / 71.5% | none | — | no |
| **RAGAS** | 50 instances | human-agreement accuracy | 0.95 / 0.78 / 0.70 | none | — | no |
| **SAFE** | 250 (main) / 496 (human) | SAFE–human agreement | 72.0% | 2 Pearson p | — | no |
| **AlignScore** | not stated | SummaC AUC-ROC avg | 87.4 / 88.6 | none | 4.7M examples | model itself |
| **CRAG** | 1,399 (PopQA) | PopQA accuracy | 59.8 (vs RAG 52.8) | none | 14k minus held-out | evaluator only |
| **Self-RAG** | 1,399 / 11,313 | PopQA / PubHealth acc | 54.9–55.8 / 72.4–74.5 | none | 145,619 + 4k–20k/type | critic |
| **Sufficient Context** | 115 / 452 / 500 | autorater F1; correct-answer gain | 0.935; +2–10% | none | 2,000 per dataset | logistic gate |
| **A Stitch in Time** | 150 topics / 750 sents | hallucination rate | 47.4% → 14.53% | none | — | threshold |
| **Hybrid Retrieval** | 300 (125 FAIL) | MAP@3 | 0.897 | none | — | dynamic weights |
| **Semantic Uncertainty** | 8,000 × 2 | AUROC | 0.77 / 0.83 | none | — | entropy |
| **SelfCheckGPT** | 238 passages / 1,908 sents | passage AUC-PR | 87.33 | none | — | merge weights |
| **Don't Hallucinate** | 200–1,094 | avg reliable accuracy gain | +10.1% | none | 1 ex./question | none |
| **ARES** | 300 (PPI) / 707 / 510 | context-rel. accuracy | 79.3–92.3% | **95% PPI intervals** | synthetic, n not stated | judges + PPI |
| **CoVe** | 56 / 55 / 418 | MultiSpanQA F1 / FactScore | 0.39 → 0.48; 55.9 → 71.4 | none | — | none |
| **FLARE** | 500 / 229 / 500 / 500 | EM / F1 | 51.0 / 59.7 | none | — | none |
| **DRAGIN** | 3,954 total | EM / F1 | 0.304 / 0.393 | none | — | threshold (RIND) |
| **IRCoT** | 500 × 4 datasets | F1 | 59.1±0.9 / 66.5±1.4 | ±std over 3 demo sets | — | none |

### 3.11.2 Why there is no single score column

**Why can't we just have one "score" column?** Because the papers are
measuring different things, and adding them up would be like comparing a
temperature in Celsius to a weight in kilograms.

The metric vocabularies are disjoint, so a "winner" column would be
meaningless:

| Paper family | Metric reported | Comparable to CRISP's ROC-AUC? |
|---|---|---|
| CRISP, Semantic Uncertainty | ROC-AUC / AUROC | **yes** (Semantic Uncertainty: 0.77 CoQA, 0.83 TriviaQA — different datasets) |
| SelfCheckGPT | AUC-PR | **no** — PR curve, not ROC |
| RAGAS, RAGChecker, ARES, Hybrid Retrieval | accuracy / F1 | **no** — thresholded, and on *different* tasks |
| SAFE, FAVA, A Stitch in Time, CoVe | rate / F1 / FactScore | **no** |
| FLARE, DRAGIN, IRCoT | EM / F1 (QA) | **no** — retrieval-augmented *answer quality*, not detection |

The one clean cross-paper comparison in this set is **ARES vs RAGAS** — both
evaluate RAG faithfulness on the same datasets with the same metric, and
ARES reports a +59.3 / +14.4 point advantage. There is **no equivalent
head-to-head available for CRISP against any of the 19 papers**, because
none of them releases a per-instance risk score on a shared dataset. CRISP's
own cross-paper comparisons therefore live in `Paper/ragtruth_comparison.md`
and `Paper/halueval_papers.md`, which is where a reviewer must be pointed.

### 3.11.3 The C1–C6 grid with statistical rigour added

This is the §2 grid with two extra columns: **"CIs?"** (does the paper report
a range for its numbers?) and **"IAA?"** (does it report how much its human
labelers agreed with each other?). Together with "Eval n", these three
columns show how well-evidenced each ✓ actually is.

| Paper | C1 | C2 | C3 | C4 | C5 | C6 | CIs? | IAA? | Eval n |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---:|
| **CRISP** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | — | **500/500/277** unique q |
| **RAGChecker** | ✓ | ✓ | ✓ | — | — | — | — | 90.95% | 4,162 |
| **FAVA** | ✓ | ✓ | — | — | ✓ | — | — | 75.1% | 364 |
| **FActScore** | ✓ | ✓ | ✓ | — | — | — | — | 96/90/88% | 183 |
| **RAGAS** | ✓ | ✓ | ✓ | — | — | — | — | ~95% | 50 |
| **SAFE** | ✓ | ✓ | ✓ | — | — | — | p only | — | 250/496 |
| **AlignScore** | ✓ | — | ✓ | — | — | — | — | — | not stated |
| **CRAG** | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — | 1,399 |
| **Self-RAG** | ✓ | ✓ | ✓ | ✓ | — | — | — | 95/90/80% | 1,399–11,313 |
| **Semantic Uncertainty** | — | — | — | ✓ | — | ✓ | — | — | 16,000 |
| **Sufficient Context** | ✓ | — | — | ✓ | — | ✓ | — | — | 115–500 |
| **A Stitch in Time** | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | 750 sents |
| **Hybrid Retrieval** | ✓ | ✓ | — | — | — | ✓ | — | **1 annotator** | 300 |
| **SelfCheckGPT** | — | ✓ | ✓ | ✓ | — | — | — | **κ = 0.595/0.748** | 1,908 sents |
| **Don't Hallucinate** | — | — | — | ✓ | — | ✓ | — | — | 200–1,094 |
| **ARES** | ✓ | ✓ | ✓ | ✓ | — | — | **✓ PPI** | — | 300 |
| **CoVe** | ✓ | ✓ | ✓ | — | ✓ | — | — | — | 418 |
| **FLARE** | ✓ | ✓ | — | — | — | — | — | — | 500 |
| **DRAGIN** | ✓ | ✓ | — | — | — | — | — | — | 3,954 |
| **IRCoT** | ✓ | ✓ | — | — | — | — | ±std | — | 500 ×4 |

**The pattern.** The papers that score ✓ in C4–C6 — the columns CRISP
actually competes in — are also the papers with the weakest statistical
reporting. Sufficient Context, A Stitch in Time, Hybrid Retrieval,
SelfCheckGPT and Don't Hallucinate all report a single number with nothing
attached. The two papers that *do* say how sure they are (ARES,
SelfCheckGPT) only reach C3–C4: neither claims C5 or C6. **CRISP is the
only system that covers all six and attaches error bands to its headline
numbers** — but it also has the smallest fitted head (496 rows) and the only
negative mitigation delta. That is the honest position.

### 3.11.4 Statistical caveats a reviewer will raise, with the answers

**The seven questions a reviewer is most likely to ask about these numbers,
and the honest answer to each.**

| # | Likely objection | The honest answer |
|---|---|---|
| 1 | "Your intervals are only bootstrap; that's not a real guarantee." | Correct. Bootstrap over unique queries estimates sampling variability, not a distribution-free guarantee on unseen data. ARES's PPI intervals *are* guarantee-backed, on a smaller human-annotated set. CRISP's advantage is coverage (every headline metric), not the strength of the guarantee. |
| 2 | "Your eval set (1,000) is smaller than HaluBench (13,867) or SelfCheckGPT's 11,313 queries." | Self-RAG's 11,313 is TriviaQA *unfiltered*, not a hallucination-detection benchmark; HaluBench's 13,867 is a retrieval pool of which only 300 are used for hallucination scoring. Semantic Uncertainty (16,000) and FActScore (6,500) are also larger, and **CRISP does not claim the largest evaluation**. The accurate statement is that CRISP's effective sizes — 500 / 500 / **277** unique queries — are comparable to the largest hallucination-scored sets in the survey (ARES 300, Hybrid Retrieval 300) and that CRISP is the **largest evaluation among systems that report uncertainty**. |
| 3 | "Your n=496 fit is tiny next to Self-RAG's 145,619." | Agreed, and it is the fusion contribution's real weakness (§3.3). The mitigation is the honest one: report the hold-out number (0.6185) rather than the tune number, and treat the head as a hypothesis to be refitted, not a settled result. |
| 4 | "Hybrid Retrieval's hallucination rate comes from one annotator." | True, and the paper says so ("annotated by a single human annotator due to time constraints"). Its 4–22% hallucination rates therefore carry unquantified annotator uncertainty and should be cited as indicative, not precise. |
| 5 | "Some papers report better numbers than yours." | None of them measures the same thing. See §3.11.2 — the only clean head-to-head in the survey is ARES vs RAGAS, and it does not involve CRISP. CRISP's actual head-to-heads are in `Paper/ragtruth_comparison.md` and `Paper/halueval_papers.md`. |
| 6 | "Your mitigation delta is negative." | Yes: −0.007, CI [−0.022, +0.007], p = 0.404. The interval includes zero, so the honest statement is "no measurable effect", not "harmful". §5.2 lists the two causes. |
| 7 | "SAFE's 496 and your 496 are suspicious." | Coincidence. SAFE's 496 is a *human-labelled evaluation* set of prompt–response pairs; CRISP's 496 is a *fitting* set for the fusion head. They are unrelated quantities that happen to share a number — do not let a reviewer conflate them. |

## 3.12 The statistical-rigour comparison, condensed

**The short version of §3.10.4** — the table to reproduce in the thesis when
a reviewer asks "whose numbers should I trust, and why?"

| Rigour dimension | Papers that report it | CRISP's position |
|---|---|---|
| **Confidence intervals on headline metrics** | ARES (PPI, 95%), **CRISP** (bootstrap, 95%) | ✅ reported — 1,000 resamples over unique queries, on every metric in all three datasets |
| **Bootstrap resampling** | **CRISP only** | ✅ 1,000 resamples; resampling unit = unique query, not row (important — see the RAGTruth 1,000-rows/277-queries asymmetry) |
| **Significance testing** | SAFE (2 Pearson p-values), **CRISP** (paired bootstrap p) | ✅ p = 0.404 on the mitigation delta — a null result, stated as such |
| **Standard deviations / error bars** | IRCoT (±std over 3 demonstration sets) | ✅ — supplied as CI bounds instead |
| **Multiple random seeds** | **none** (AlignScore names one seed, 2022) | ❌ not reported |
| **Inter-annotator agreement** | SelfCheckGPT (κ = 0.595 / 0.748); RAGChecker (90.95%); FAVA (75.1% / 60.3%, κ claimed but not printed); FActScore (96/90/88%); RAGAS (~95/95/90%); Self-RAG (95/95/90/80%) | ❌ not applicable — CRISP uses dataset labels, not human annotation |
| **Human-annotated ground truth in the pipeline** | ARES (150–300 PPI labels), SAFE (16,011 facts), Semantic Uncertainty (300+300) | ❌ not reported — CRISP's labels are inherited from HaluEval / TruthfulQA / RAGTruth |
| **Multiple annotators where humans are used** | RAGChecker (10), FAVA (5 + 10), RAGAS (2), SelfCheckGPT (2) | n/a |
| **Evaluation size** | Semantic Uncertainty 16,000 · FActScore 6,500 · RAGChecker 4,162 · DRAGIN 3,954 · Self-RAG 11,313 · Hybrid Retrieval 13,867 pool | **500 / 500 / 277 unique queries** — not the largest, but the **largest among systems that report uncertainty** (ARES 300, Hybrid Retrieval 300) |

**The one-paragraph version.** Of the twenty systems surveyed, only ARES and
CRISP say how sure they are; only SelfCheckGPT reports how much its human
labelers agreed; and no paper other than CRISP re-runs the test on shuffled
data to check the result holds. CRISP's advantage is that it reports
uncertainty *everywhere*, not that any one of its guarantees is the
strongest — and what CRISP's own intervals deliver is the finding that its
mitigation effect is indistinguishable from zero.

## 3.13 Plain-English glossary of the statistical terms

This section exists so the tables above can be read by someone who is not a
statistician. Nothing here is needed to write the thesis chapter — it is
for the reader (and for the supervisor) who wants to follow the argument
without getting lost in terminology.

| Term | What it means, in one sentence | Why it matters here |
|---|---|---|
| **Point estimate** | A single number with nothing attached — "our method scores 0.92". | Most papers in this survey report only these. It hides how much the number could move if you tested on different data. |
| **Confidence interval (CI)** | A range around the number, like "0.91 to 0.94", meaning "if we repeated this test, the true value would land here 95% of the time". | Only **ARES** and **CRISP** report these. A number with a range is a stronger claim than a number alone. |
| **Error bar** | The same idea, drawn on a chart as a little line on top of the number. | No paper in the survey uses error bars on a chart. |
| **Bootstrap resampling** | Shuffle the test data many times (CRISP: 1,000 times), recompute the score each time, and use the spread of those results as the range. | CRISP's method. It is cheap, needs no assumptions, and works on any metric — but it measures *sampling* variation, not a hard guarantee. |
| **Prediction-powered inference (PPI)** | ARES's method: combine the trained scorer's predictions with a small set of human-checked answers to get a range that is *guaranteed* to be right 95% of the time. | ARES's guarantee is mathematically stronger than CRISP's. But it needs a human-labelled set, which CRISP does not have. |
| **Significance test (p-value)** | A calculation that asks "could this result have happened by chance?" A small p-value (below 0.05) means it probably did not. | SAFE reports two p-values. CRISP reports one (p = 0.404 on the mitigation effect). The rest report none. |
| **Inter-annotator agreement (κ, Cohen's kappa)** | When humans label the same thing, do they agree? κ = 1 means perfect agreement, κ = 0 means chance. | Only **SelfCheckGPT** reports it (κ = 0.595). FAVA says it computed kappa but never prints the number. |
| **Multiple random seeds** | Running the whole experiment several times with different random choices, to see how much the result moves. | **No paper in the survey does this**, including CRISP. |
| **ROC-AUC** | A score from 0 to 1 for how well a detector separates good answers from bad ones. 0.5 = a coin flip. 1.0 = perfect. | CRISP's main metric. HaluEval 0.9251, TruthfulQA 0.6355, RAGTruth 0.6127. |
| **AUPR** | Like ROC-AUC but focuses on the positive class; useful when the two classes are unbalanced. | CRISP reports it alongside ROC-AUC. |
| **F1** | The average of precision and recall — a single number summarising "how good is this detector". | CRISP's detection F1 is 0.82 (HaluEval). Claim-level F1 is a different thing. |
| **VSR (verification success rate)** | Of all the claims CRISP checked, what fraction did it successfully judge. | 0.701 on HaluEval. |
| **PR@90** | "If we only flag the 10% of answers we are most confident are bad, how many of those flags are correct?" — precision at a 90% recall operating point. | A practical number: it says what happens if you use CRISP with a strict threshold. |
| **n** | How many examples the number was measured on. | The single most important column for judging any of these papers. CRISP's is 1,000. |
| **Not stated in paper** | The paper does not report this. | Used throughout §3.10 rather than a blank, because "the paper never said" is itself a finding. |

**The one-paragraph version.** Almost every paper in this field reports a
single number for its main result. That number might be right, but the
paper gives you no way of knowing how much it would change if you ran the
experiment on different data. CRISP reports a range for every number it
claims, and the ranges are honest enough to include "no effect" for the
mitigation stage. That is a stronger scientific position than the other
seventeen papers are in — and it is the position that makes the negative
mitigation result trustworthy rather than suspicious.

---

# 4. The papers in detail

---

## 4.1 RAGChecker — the closest in *diagnosis*

> *"RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented
> Generation"* — Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi,
> Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li,
> Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei
> Liu, Yue Zhang, Zheng Zhang.
> **arXiv:2408.08067** (Aug 2024) · [github.com/amazon-science/RAGChecker](https://github.com/amazon-science/RAGChecker)

**Workflow**

```
1. CLAIM EXTRACTION
   RefChecker framework, Llama3-70B as both claim extractor and checker.
   Claims are pulled from the response, the ground-truth answer,
   and each retrieved chunk.

2. RETRIEVER MODULE
   For each retrieved chunk, test whether it entails ≥1 ground-truth claim.
   A chunk is relevant iff  ∃i such that  c_i^(gt) ∈ chunk_j

3. HALLUCINATION MODULE
   For each response claim, test whether it is entailed by any retrieved
   chunk. A claim is a hallucination iff it is in NEITHER the ground truth
   NOR any retrieved chunk.

4. METRICS
   Retrieval side : Claim Recall, Context Precision
   Generation side: Faithfulness, Relevant Noise Sensitivity,
                    Irrelevant Noise Sensitivity, Hallucination,
                    Self-Knowledge, Context Utilization
   Overall        : Precision, Recall, F1
```

**Matches CRISP:** C1, C2, C3. It decomposes the pipeline into the same
claim-level units and, crucially, **separates retrieval failure from
generation failure** — the distinction CRISP's step 6 vs step 8 implies but
does not name.

**Differs:** no reranking, no fusion head, no mitigation, no abstention. It is
a *measurement framework*, not a defence system.

**Why it matters to the thesis:** the metric vocabulary is directly
reusable. **Relevant vs Irrelevant Noise Sensitivity** and **Self-Knowledge**
map onto CRISP's fusion features — a claim supported by a *relevant* passage
is different evidence from one supported by an irrelevant one, and CRISP's
current step 9 treats them identically. It also evaluates both **BM25 and E5
dense retrievers**, which is directly comparable to CRISP's hybrid design.

**Statistics (§3.10).** Tested on **4,162 queries** across 10 domains. Best
system (E5-Mistral + GPT-4): P **62.0** / R **53.0** / F1 **52.7**, against a
human–human upper bound of **70.09** — so even the best automatic system
falls well short of human agreement. Human labelling used **280 pairwise
instances** with **10 annotators**; agreement **90.95%** (a raw percentage,
**no Cohen's κ**). **No confidence intervals, no error bars, no bootstrap, no
multiple seeds** — a single number with nothing attached.

---

## 4.2 FAVA — the closest in *detection and editing*

> *"Fine-grained Hallucination Detection and Editing for Language Models"* —
> Abhika Mishra, Akari Asai, Vidhisha Balachandran, Yizhong Wang, Graham
> Neubig, Yulia Tsvetkov, Hannaneh Hajishirzi. **COLM 2024** ·
> **arXiv:2401.06855**

**Workflow**

```
1. RETRIEVE
   Contriever-MSMARCO over English Wikipedia → top-5 documents.
   (Retrieving 5 docs beats retrieving 1 by 4.5 points.)

2. DETECT + EDIT  (one model, jointly)
   Llama2-Chat-7B, conditioned on the retrieved context AND the
   candidate output, emits tagged output:
     <entity>…</entity>   error type + proposed correction
     <mark>…</mark>       mark the erroneous span
     <delete>…</delete>    delete the erroneous span

3. ERROR TAXONOMY (six types)
   Contradictory:   Entity, Relation, Sentence
   Non-contradictory: Invented, Subjective, Unverifiable

4. TRAINING
   35,074 synthetic instances
   (30,074 from Wikipedia passages, 5,000 from QA pairs),
   built by corrupting correct text with taxonomy-specific errors.
```

**Matches CRISP:** C1, C2, C5. **This is the single closest published
analogue to CRISP's detection→edit loop**, and the only paper found that both
retrieves *and* rewrites spans.

**Two important differences.**

1. **FAVA is a fine-tuned generative tagger, not an NLI pipeline.** Its
   verification is *learned*, not entailment-based. CRISP uses an off-the-shelf
   NLI model (`mDeBERTa-v3-xnli`) — which means CRISP's verification is
   model-agnostic and swappable, while FAVA's is baked into one checkpoint.
2. **FAVA's edits are span-level tag insertions**; CRISP's are per-claim
   rewrites followed by a **re-verification pass**. CRISP checks its own
   output; FAVA does not.

**Also worth citing:** FAVA's **hallucination taxonomy** is the natural prior
art for CRISP's error typing — CRISP currently has only a binary
supported/contradicted/unknown scheme, which is strictly coarser.

**Statistics (§3.10).** Trained on **35,074 synthetic examples**. Detection
**+22.1%** over ChatGPT (binary **+14.2%**); editing gains FActScore
**+4.4** (Alpaca-7B) to **+9.3** (Alpaca-13B). Retrieval ablation: unedited
42.5 → top-1 44.2 → top-5 47.0 → reranked top-5 47.7 → entity matching
50.1. Test set **364 annotated passages**. Human check: FAVA found **2.4**
errors per response vs ChatGPT's **1.9** — it detects more, but the two
counts are not measuring the same thing. **No confidence intervals, no
significance tests, no error bars.** FAVA says it measured how well its
labelers agreed using "Cohen kappa scores" but **never prints the number**;
the figures it does give are **75.1%** (agreeing on whether a sentence has
an error) and **60.3%** (agreeing on the error type) — both modest, and
weaker than SelfCheckGPT's κ = 0.595 / 0.748.

---

## 4.3 FActScore — the closest in *claim decomposition and verification*

> *"FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long
> Form Text Generation"* — Sewon Min et al. **EMNLP 2023** ·
> **arXiv:2305.14251**

**Workflow**

```
1. RETRIEVE
   Passages from the given knowledge source (Wikipedia in the
   experiments), via GTR retrievers.

2. ATOMIC FACT DECOMPOSITION
   Split the generation into sentences, then InstructGPT
   (text-davinci-003) divides each sentence into atomic facts.

3. VERIFY each atomic fact
   Zero-shot prompt the evaluation LM with
   "<atomic-fact> True or False?"
   (or concatenate retrieved passages before the question).
   Compare the probabilities of "True" vs "False".
   NP variant: non-parametric masked-LM probability.
   Retrieve→LM + NP ensemble: supported only when BOTH agree.

4. SCORE
   FActScore(M) = E[f(M_x)]
   f = fraction of the response's atomic facts supported.
   Abstentions are excluded from the model-level average.
```

**Matches CRISP:** C1, C2, C3. **CRISP's steps 5 → 8 are the same shape**:
decompose the answer into atomic claims, then verify each against retrieved
evidence.

**Differs:** measurement only — no mitigation, no abstention, no fusion
head. Also note FActScore verifies with a **zero-shot LM prompt**, not an NLI
model, so its verifier is a *generator* rather than a classifier.

**One methodological point worth borrowing:** FActScore **excludes
abstentions** from the model-level average rather than scoring them as
failures. CRISP's step 13 currently has no such treatment.

**Statistics (§3.10).** Human set **183 entities** → 549 prompts → 6,500
generations, with **26.3–40.8 labelled facts per response**. FActScore:
InstructGPT **42.5%**, ChatGPT **58.3%**, PerplexityAI **71.5%**. The
automatic estimator tracks the human score with **<2% error**, and two
leading estimators rank the 13 subjects identically (**Pearson r = 0.99**).
Double annotation on **10%** of the data gives agreement rates **96 / 90 /
88%** — again **no Cohen's κ**. **No confidence intervals, no significance
tests, no error bars, no multiple seeds.** Note FActScore's "FS" is an
*estimated FActScore*, not an F1, and is **not comparable** to CRISP's
claim-level F1.

---

## 4.4 RAGAS — the closest in *the claim-extraction idiom*

> *"RAGAS: Automated Evaluation of Retrieval Augmented Generation"* — Shahul
> Es, James, Luis Espinosa-Anke, Steven Schockaert.
> **arXiv:2309.15217**

**Faithfulness metric workflow**

```
1. STATEMENT EXTRACTION
   The LLM splits the response into one or more statements per sentence.

2. SUPPORT JUDGEMENT
   Each statement is judged against the retrieved context for support.

3. FAITHFULNESS = supported statements / all statements

   (Two sibling metrics: answer relevance = mean cosine similarity of
   generated questions to the original question; context relevance =
   fraction of context sentences the LLM extracts as needed.)
```

**Matches CRISP:** C1, C2, C3. The claim-decomposition prompt idiom is
nearly identical to CRISP's step 5, and this is the most widely used
evaluation framework in the RAG space.

**Differs:** no reranking, no fusion, no mitigation, no abstention.

**Statistics (§3.10).** WikiEval = **50 Wikipedia pages** × 3 dimensions,
labelled by **two annotators** who settled disagreements by discussion.
Agreement with human preference: faithfulness **0.95**, answer relevance
**0.78**, context relevance **0.70** (baselines: GPT Score 0.72/0.52/0.63,
GPT Ranking 0.54/0.40/0.52). Raw agreement is quoted as "around 95% / 95% /
90%" with **no chance-corrected statistic**. **No confidence intervals, no
significance tests.** Ties are broken randomly — a small source of
variation the paper does not measure. **50 examples is the smallest
evaluation set in this survey**, which is why RAGAS is cited for its *method*
rather than its numbers.

---

## 4.5 SAFE — the closest in *long-form verification*

> *"LongFact: Long-Form Factuality in Large Language Models"* — Wei et al.
> **NeurIPS 2024** · **arXiv:2403.18802**

**Workflow**

```
1. SPLIT the response into individual facts.

2. For each fact, issue Google Search queries.

3. JUDGE whether the search evidence supports the fact,
   using multi-step reasoning.

4. AGGREGATE into an F1-style long-form factuality score that balances
   the supported-fact ratio against response length.
```

**Matches CRISP:** C1, C2, C3. Same retrieve→split→verify→aggregate loop.

**Differs:** verification is a **live web search plus LLM reasoning**, not a
fixed-corpus NLI. That makes SAFE far more expensive and not reproducible
without network access. No mitigation, no abstention.

**Statistics (§3.10).** LongFact = **2,280 prompts** across 38 topics; the
main benchmark uses a **250-prompt** random subset of LongFact-Objects. The
human comparison set is **496 prompt–response pairs containing 16,011
individual facts** — note this 496 is a *human evaluation* set and has
**nothing to do** with CRISP's 496-row fusion fit (§3.11.4, caveat 7).
SAFE agrees with **72.0%** of human annotations; on **100 sampled
disagreements**, a re-check found SAFE right in **76%** of cases versus
humans in **19%** (a 4-to-1 win). Cost **$0.19 per response** against $4.00
for human annotation ("more than 20× cheaper"). F1@64: GPT-4-Turbo **95.0**,
Gemini-Ultra **91.7**, Claude-3-Sonnet **89.4**; F1@178 drops to 66.4 / 60.3 /
51.4. The **only two p-values in the whole survey**: correlation with
Chatbot Arena ELO — precision r = 0.502, **p = 0.205** (not significant);
F1@64 r = 0.754, **p = 0.031** (significant). **No confidence intervals, no
error bars, no bootstrap, no multiple seeds**; temperature 0 was used "to
decrease variation in responses across runs". **No inter-annotator
agreement measure** is reported — the phrase does not appear in the paper.

---

## 4.6 AlignScore — the strongest drop-in for CRISP's step 8

> *"AlignScore: Evaluating Factual Consistency with a Unified Alignment
> Function"* — Zha, Yang, Li, Hu. **ACL 2023** · **arXiv:2305.16739**

**Workflow**

```
1. POOL 4.7M training examples from seven tasks:
   NLI, QA, paraphrase, fact verification, IR,
   semantic similarity, summarisation.

2. TRAIN a 355M-parameter alignment function on arbitrary text pairs.

3. USE as a verifier: score context-vs-generation alignment.
```

**Matches CRISP:** C3 — and it is the natural upgrade path for CRISP's step 8
verifier.

**Differs:** this is a **component** paper, not a pipeline. It has no
retrieval, fusion, mitigation, or abstention of its own.

**Why it matters:** CRISP's step 8 uses a general-purpose MNLI model. AlignScore
is trained on 4.7M examples *including* fact verification, so it is likely a
stronger verifier for this specific task. This is a concrete, testable
improvement — and CRISP's claim-conditioned premise selection (§3.1 of the
deep-dive doc) would still apply unchanged, since it operates on the premise
before the verifier.

**Statistics (§3.10).** **22 evaluation datasets**; **the paper does not say
how many examples each one had** — worth noting when comparing against
CRISP, where every count is published. Results: SummaC AUC-ROC avg **87.4**
(base) / **88.6** (large); TRUE AUC-ROC avg **86.3** / **87.4**; TRUE AVG-ZS
**82.5** / **83.8**. Against GPT-4 G-EVAL on Spearman: SummEval **46.6** vs
50.7, QAGS-XSum **57.2** vs 53.7, QAGS-CNNDM **73.9** vs 68.5. Trained on
**4.7M examples** from 15 datasets / 7 task families ("at most 500k samples
from each dataset"); base = RoBERTa-base **125M**, large = RoBERTa-large
**355M**; 3 epochs, batch 32, AdamW, lr 1e-5. Compute: base 2×3090 for
100 h, large 4×A5000 for 532 h. **No confidence intervals, no error bars,
no bootstrap**, and a **single random seed (2022)** — the only paper in the
survey that names its seed, but it reports no spread across seeds. **No
human-annotation counts and no inter-annotator agreement** for the
evaluation sets.

---

## 4.7 Corrective RAG (CRAG) — the closest in *confidence-gated retrieval*

> *"Corrective Retrieval Augmented Generation"* — Yan, Gu, Zhu, Ling.
> **arXiv:2401.15884**

**Workflow**

```
1. RETRIEVE documents.

2. RETRIEVAL EVALUATOR
   Fine-tuned T5-large (~0.77B), "lightweight retrieval evaluator".
   Input = question concatenated with each single document.
   Output = relevance score on a −1 to +1 scale.

3. CONFIDENCE DEGREE
   Correct    : at least one document's score > upper threshold
   Incorrect  : all documents below the lower threshold
   Ambiguous  : everything in between

4. WEB-SEARCH CORRECTION (for Incorrect / Ambiguous)
   Rewrite the query into a short keyword query ("at most three
   keywords"), query the Google Search API, take the top 5 URLs
   (preferring Wikipedia), filter them with the same evaluator.

5. KNOWLEDGE REFINEMENT — decompose, then recompose
   Split documents into "knowledge strips" (1-2 sentence passages
   stay whole; longer documents split by sentence).
   Score every strip, drop irrelevant ones, concatenate the rest.
   Settings: top-k = 5, filter threshold = −0.5.

6. GENERATE with the corrected knowledge.
```

**Measured thresholds** (empirically chosen per dataset — *do not quote as a
single global number*):

| Dataset | (upper, lower) |
|---|---|
| PopQA | (0.59, −0.99) |
| PubHealth | (0.5, −0.91) |
| Arc-Challenge | (0.5, −0.91) |
| Biography | (0.95, −0.91) |

**Matches CRISP:** C1, C2, C4, C5, C6 — the broadest coverage of any paper
here.

**The crucial architectural difference:** **CRAG edits the *evidence*;
CRISP edits the *answer*.** CRAG's decompose-then-recompose operates on
retrieved knowledge strips before generation; CRISP's `claim_edit` operates on
the generated claims after verification.

**Another difference:** CRAG gates on **a single retrieval score**. CRISP
fuses retrieval score with NLI confidence and entropy, so a well-retrieved
passage does not by itself earn trust.

**Statistics (§3.10).** PopQA test = **1,399 rare-entity queries** (of 14k
total; the rest fine-tunes the retrieval evaluator); the counts for
Biography, PubHealth and Arc-Challenge are **not stated in the paper**.
Results with SelfRAG-LLaMA2-7b: PopQA accuracy RAG **52.8** → CRAG
**59.8** (Self-RAG 54.9, Self-CRAG 61.8); PubHealth **39.0** → **75.6**;
Arc-Challenge **53.2** → 68.6; Biography FactScore **59.2** → **74.1**.
Retrieval-evaluator accuracy **84.3** vs ChatGPT 58.0, CoT 62.4, few-shot
64.7. With LLaMA2-hf-7b the gains are much smaller or negative on PubHealth
(48.9 → 59.5) and Arc-Challenge (43.4 → 53.7) — worth noting, because the
headline PopQA number does not transfer to every generator. Web-search
correction vs plain RAG: **+4.4** points on PopQA; Self-RAG + web adds
**+3.0** vs Self-CRAG's **+6.9**. Retrieval evaluator = fine-tuned
**T5-large (0.77B)**; generators LLaMA2-7B/13B. **No confidence intervals,
no error bars, no significance tests, no multiple seeds, and no human
annotation** — the only labels are GPT-4-generated Self-RAG reflection tokens,
with no agreement statistic.

---

## 4.8 Self-RAG — the closest in *fusion*

> *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-
> Reflection"* — Asai, Wu, Wang, Sil, Hajishirzi. **ICLR 2024** ·
> **arXiv:2310.11511**

**Workflow**

```
1. REFLECTION TOKENS (four groups, one sentence = one segment)
   Retrieve : {yes, no, continue}       — call the retriever now?
   IsRel    : {relevant, irrelevant}     — is the passage useful?
   IsSup    : {fully supported, partially supported, no support}
   IsUse    : {5, 4, 3, 2, 1}           — how useful is the output?

2. CRITIC TRAINING
   Critic = Llama2-7B. Generator = Llama2-7B or 13B. Both from the
   same base LM but separately trained.
   GPT-4 produces reflection-token labels via type-specific prompts
   and few-shot demonstrations; 4k–20k supervised examples per type.
   The trained critic reaches "higher than 90% agreement" with GPT-4.
   The critic is NOT needed at inference — the generator emits the
   reflection tokens itself.

3. SEGMENT-LEVEL BEAM SEARCH
   f(y_t, d, Critique) = p(y_t | x, d, y_<t) + S(Critique)
   S(Critique) = Σ_G  w^G · s_t^G     for G ∈ {IsRel, IsSup, IsUse}
   s_t^G = p_t(r̂) / Σ_i p_t(r_i)      (normalised probability of the
                                        most desirable token in group G)

   Default weights: IsRel = 1.0, IsSup = 1.0, IsUse = 0.5
   Retrieval threshold: 0.2 (most tasks), 0 for ALCE
```

**Matches CRISP:** C1, C2, C3, C4. **Architecturally the closest to CRISP's
step 9.** Self-RAG's `S(Critique)` is a learned, weighted combination of
multiple support signals into one score — the same concept as CRISP's
7-component fusion head, just computed as a token probability rather than a
fitted logistic regression.

**Differs:** the fusion is *inside* the generator (a critic emitting tokens),
not a separate head over heterogeneous features. Self-RAG has **no
abstention** and **no rewriting** — it re-ranks candidates during decoding
rather than repairing a finished answer.

**The reusable idea:** the **normalised-probability-per-group** trick
(`s_t^G`) — scaling each signal so no single one dominates by virtue of raw
scale — is the same problem CRISP's z-scoring solves in step 9, and CRISP's
`component_std` values are the direct analogue of Self-RAG's denominators.

**Statistics (§3.10).** Generator trained on **145,619** instruction–output
pairs (GPT-4 Alpaca 26,168; Stanford Alpaca 25,153; FLAN-V2 17,817; Wizard
of Wikipedia 17,367; Natural Questions 15,535; ShareGPT 13,406; FEVER 9,966;
Open Assistant 9,464; ASQA 3,897; OpenBookQA 4,699; Arc-Easy 2,147);
critic on **4k–20k** labels per reflection type (GPT-4 labels collected:
Retrieve 12,594, relevance 19,317, utility 3,831). Test sets: PopQA
long-tail **1,399**, TriviaQA-unfiltered **11,313**; the sizes of PubHealth,
ARC-C, Biography and ASQA are **not stated in the paper**. Results: PopQA
7B **54.9** / 13B **55.8**; TriviaQA 66.4 / 69.3; PubHealth 72.4 / 74.5;
ARC-C 67.3 / 73.1; Biography FactScore 81.2 / 80.2. Note **Self-RAG 13B is
*worse* than 7B on Biography FactScore** (80.2 vs 81.2) — the bigger
generator does not win on long-form factuality. The GPT-4–critic agreement is
"higher than 90%", and the authors manually checked **20 examples per
aspect** (agreement: relevance 95%, retrieval necessity 95%, support 90%,
usefulness 80%) — **no Cohen's κ**, **no confidence intervals, no error bars,
no significance tests, no multiple seeds.** "Significantly" appears in the
paper only as informal prose, never attached to a test.

---

## 4.9 Sufficient Context — the closest in *abstention*

> *"Sufficient Context: A New Lens on Retrieval Augmented Generation Systems"* —
> Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly,
> Cyrus Rashtchian. **arXiv:2411.06037**

**Workflow**

```
1. CONTEXT SUFFICIENCY CLASSIFIER
   Binary: does the context support any plausible answer?
   "there exists an answer A′ … such that A′ is a plausible answer
    to the question Q given the information in C."
   Main labeller: Gemini 1.5 Pro with one in-context example
                  → 93.0% accuracy, 0.935 F1 on 115 human-labelled
                    instances (without ground-truth answers)
   Alternative: FLAMe, a fine-tuned PaLM 24B, on 1600-token chunks
                  → 87.8% accuracy, 0.892 F1

2. ERROR STRATIFICATION
   Did the model mishandle adequate evidence, or was the evidence
   inadequate? (Plus a third "useful but not sufficient" case:
   clarifies ambiguity, or bridges part of a reasoning chain.)

3. SELECTIVE GENERATION
   Features: the binary sufficiency signal + model self-rated confidence
             (P(True) for open models, P(Correct) for proprietary).
   A logistic-regression model predicts hallucination from these.
   Answers whose risk falls below a threshold are REJECTED.

4. RESULT
   "improves the fraction of correct answers … by up to 2–10%"
   e.g. >10% for Gemma 27B on HotPotQA in high-accuracy regions;
   >5% for Gemini 1.5 Pro near 70% coverage.
```

**Matches CRISP:** C1, C4, C6. **This is the clearest published precedent for
CRISP's step 11.** It combines a sufficiency signal with a self-confidence
signal in a logistic model and rejects low-risk answers — almost exactly
CRISP's "fuse several signals → risk score → threshold" shape.

**The crucial difference:** Sufficient Context's gate is **retrieval-side**;
CRISP's is **answer-side**. Sufficient Context asks "was the evidence
enough?" CRISP asks "is this answer risky?" Both are needed — a system can
have sufficient context and still produce a bad answer, and CRISP's step 8
is what catches that case.

**A negative result worth knowing:** the paper tested fine-tuning models to
abstain and found it **did not reliably reduce hallucination** — some variants
abstained more but also produced fewer correct answers. That is a caution
against trying to fix CRISP's abstention by fine-tuning rather than by
thresholding a calibrated risk score.

**Statistics (§3.10).** Test sets: FreshQA **452**, Musique-Ans **500**,
HotPotQA **500**; the gold sufficiency set is **115** examples (query,
context, answer) — **the smallest evaluation set in this survey**. The
autorater (Gemini 1.5 Pro, 1-shot) reaches **F1 0.935 / accuracy 0.930 /
precision 0.935 / recall 0.935**; FLAMe 24B F1 0.892 / acc 0.878; TRUE-NLI
11B F1 0.818 / acc 0.826; "Contains GT" F1 0.810. Refusing to answer when the
context is insufficient raises the fraction of correct answers by **2–10%**
across Gemini, GPT and Gemma, with **>10%** for Gemma-27B on HotPotQA in
the highest-accuracy region and **>5%** for Gemini near 70% coverage. The
paper also reports that SOTA LLMs give correct answers only **35–62%** of the
time when the context is insufficient, and that the share of questions with
enough context ranges from **77.4%** (FreshQA) to **44%** (HotPotQA dev, vs
**>88%** on train). LoRA fine-tuning used **2,000 examples per dataset**
(rank 4, alpha 8, 2 epochs, batch 16, lr 1e-5) with **400** "I don't know"
replacements. **No confidence intervals, no error bars, no significance
tests, no multiple seeds**; the logistic gate used "100 iterations of random
hyperparameter search", and the authors note "the training was not smooth,
and different checkpoints led to very different results" — a candid
admission of variation that is never actually measured. **No inter-annotator
agreement statistic and no annotator count** are reported for the
115-example gold set.

---

## 4.10 A Stitch in Time Saves Nine — the closest in *mitigation*

> *"A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of
> LLMs by Validating Low-Confidence Generation"* — Neeraj Varshney, Wenlin
> Yao, Hongming Zhang, Jianshu Chen, Dong Yu. **arXiv:2307.03987**

**Workflow**

```
1. SCREEN by low-confidence tokens
   For each generated sentence, identify important concepts and take
   the MINIMUM token probability across the concept's tokens
   ("Minimum of Token Probabilities" — one uncertain token can signal
   uncertainty about the whole concept).

2. VALIDATE each uncertain concept
   Build a Yes/No validation question, retrieve supporting evidence
   (Bing Search API; the model's parametric "Self-Inquiry" as an
   alternative), and answer the question from that evidence.
   Greedy exiting: concepts are validated in ascending uncertainty
   order, stopping at the first suspected hallucination.
   A sentence is labelled hallucinated if validation fails for any concept.

3. REWRITE
   Instruction-based repair: "removing or substituting the
   hallucinated information", using the retrieved evidence.
   The repaired sentence is appended to the context so it influences
   subsequent generation and reduces hallucination propagation.

MODELS: GPT-3.5 (text-davinci-003) main, Vicuna-13B additional.
RESULTS: ~88% detection recall, 57.6% mitigation success,
         hallucination rate ~47.5% → ~14.5%.
```

**Matches CRISP:** C1, C2, C3, C4, C5. **The closest published analogue to
CRISP's mitigation stage** and to its "fix only what is unsupported"
principle.

**Differs:** CRISP gates on a *fused* score (similarity + retrieval + NLI +
entropy) rather than raw token probability, and verifies at atomic-claim
granularity via NLI rather than via question-answering. CRISP's rewriting is
also byte-preserving outside the replaced spans, whereas this paper's repair
is instruction-based at the sentence level.

**Statistics (§3.10).** Custom benchmark of **150 topics** × 5 sentences =
**750 labelled sentences** (plus concept-level labels); Vicuna-13B subset
**50 sentences**; multi-hop QA **50** HotpotQA bridge questions;
false-premise QA **50** questions plus one true-premise counterpart each.
Detection recall **~88%**; the rewrite "succeeded" on **57.6%** of correctly
detected hallucinations; **deterioration 3.06%** (false positives wrongly
changed). Hallucination rate **47.4% → 14.53%**. Sentence-level web-search
detection: accuracy **0.681**, hallucinated-class P/R **61.82/85.96**;
concept-level accuracy **0.75**, P/R **58.17/87.68**. False-premise QA:
**76%** correct with the approach vs **24%** without. Multi-hop QA: 54%
incorrect → 26% hallucinated. The only variability figures are descriptive
means±std (sentence length 18.6±5.55 words; concepts/sentence 3.27±1.63;
words/concept 1.79±1.02) and precision–recall curves reported **without
intervals**. **No confidence intervals, no significance tests, no error bars,
no multiple seeds**, and **no inter-annotator agreement statistic** — the
authors labelled the data themselves, the stated reason being that checking
each sentence "requires considerable effort". Note the asymmetry with CRISP:
this paper reports a large positive mitigation effect (**−33 points of
hallucination rate**) whereas CRISP's is **−0.007 (p = 0.404)** — the
largest single discrepancy in the survey, and §5.3 treats it as a live
objection.

---

## 4.11 Hybrid Retrieval for Hallucination Mitigation — the closest in *retrieval*

> *"Hybrid Retrieval for Hallucination Mitigation in Large Language Models:
> A Comparative Analysis"* — Chandana sree Mala, Gizem Gezici, Fosca
> Giannotti. **arXiv:2504.05324** (Feb 2025)

**Workflow**

```
1. SPARSE RETRIEVAL  (Ret_S) — BM25 keyword matching
2. DENSE RETRIEVAL   (Ret_D) — sentence-transformers/all-mpnet-base-v2,
                                  cosine similarity
3. QUERY EXPANSION    each term + its top-2 WordNet synonyms (NLTK)
                       q' = q ∪ T(q_j)
4. DYNAMIC WEIGHTING  a query-specificity score from mean tf*idf:
                       w_RetS = α·S(q'),  w_RetD = 1 − w_RetS   (α = 1)
                       specific queries favour sparse; general favour dense
5. DYNAMICALLY WEIGHTED RRF
   RRF_weighted(d_i) = Σ_Ret  w_Ret / (ε + r_Ret(d_i))
                       → top 3 documents
6. GROUND the LLM response
   top-3 documents + question → LLaMA-3-8B-Instruct, zero-shot
   (temperature 0.8, top-p 0.9), instructed to say
   "The context doesn't provide sufficient information to answer the question"
   when the context is inadequate.

BENCHMARK: HaluBench — 13,867 examples from DROP, HaluEval, RAGTruth,
            FinanceBench, PubMedQA, COVIDQA.
EVAL SET:  balanced random sample of 300 instances
            (50 per source dataset, 25 PASS / 25 FAIL each),
            human-annotated.
METRICS:  MAP@3, NDCG@3 (retrieval);
           accuracy, hallucination rate, rejection rate,
           adjusted accuracy (generation).
```

**Measured retrieval results:**

| Retriever | MAP@3 | NDCG@3 |
|---|---:|---:|
| Sparse | 0.724 | 0.732 |
| Dense | 0.768 | 0.783 |
| **Hybrid** | **0.897** | **0.915** |

**Matches CRISP:** C1, C2, C6. **The most directly comparable published
treatment of CRISP's hybrid-retrieval design** — same dense+sparse+RRF
structure.

**Three differences:**

1. **No reranker.** CRISP adds BGE-reranker-v2-m3 on top of the RRF output;
   this paper fuses the two rankings directly.
2. **CRISP's `weight_dense`/`weight_bm25` are inert** (the fusion code never
   reads them) whereas this paper's weighting is **dynamically computed
   per query** from specificity. That is a strictly better design and a
   concrete improvement path for CRISP.
3. **No claim-level NLI, no fusion head, no rewriting.**

**Why it matters — and the caution it carries.** This paper's finding is that
**better retrieval alone lowers hallucination rate.** The hybrid pipeline beat
sparse and dense on all four overall metrics across all six source datasets.
That is a genuine counterweight to the thesis's position: it means
retrieval-quality gains and detection-layer gains are not independent, and a
critic should not claim that detection-layer work matters if retrieval alone
closes the gap.

**Statistics (§3.10).** HaluBench = **13,867 samples** from six source
datasets (DROP, HaluEval, RAGTruth, FinanceBench, PubMedQA, COVIDQA); the
subset actually scored for hallucination is **300 examples** (50 per source,
balanced 25 PASS / 25 FAIL) with a FAIL-only analysis of **125** samples
(RAGTruth contributes 0 FAILs). Retrieval over the full 13,867: MAP@3 sparse
**0.724** / dense **0.768** / hybrid **0.897**; NDCG@3 **0.732 / 0.783 /
0.915**. On the 300-example set, hybrid accuracy 92.00 (HaluEval), 92.00
(PubMedQA), 88.00 (RAGTruth), 77.55 (DROP), 70.02 (COVIDQA), 62.90
(FinanceBench); hallucination rates 4.00–22.00%; rejection rates
2.00–32.00%. Dense retrieval collapses on COVIDQA (**14.00** accuracy,
**58.02%** hallucination rate) where hybrid holds **70.02** — the clearest
single illustration of why the hybrid design exists. Generator
**LLaMA-3-8B-Instruct** (temperature 0.8, top-p 0.9); dense embeddings
`all-mpnet-base-v2`; no fine-tuning. **No confidence intervals, no error
bars, no significance tests, no multiple seeds.** Critical caveat: the
300-example subset was **"annotated by a single human annotator due to time
constraints"**, so **no inter-annotator agreement is possible** and its
percentages carry wide, unmeasured annotator uncertainty. The FAIL-only
(125-sample) numeric results exist **only inside figure images** and are
therefore **not stated in the paper's text** — do not quote numbers for them.
This is the paper with the largest evaluation pool and the weakest human
verification in the survey.

---

## 4.12 Semantic Uncertainty — the closest in *entropy and abstention*

> *"Semantic Uncertainty: Linguistic Invariance for Uncertainty Estimation in
> Natural Language Generation"* — Kuhn, Gal, Farquhar. **ICLR 2023
> (Spotlight)** · **arXiv:2302.09664**

**Workflow**

```
1. SAMPLE multiple answers.
2. CLUSTER the paraphrases by entailment (semantic equivalence),
   not by string similarity.
3. Compute ENTROPY over the CLUSTERS, not over the raw samples.
4. Aggregate to a confidence.
5. ABSTAIN when confidence is low.
```

**Matches CRISP:** C4, C6. CRISP's step 4 entropy and step 11 abstention are
the direct analogues.

**The distinction that matters:** clustering *before* measuring entropy is
what makes the entropy meaningful. Two samples that are worded differently
but mean the same thing should count as one cluster, not two. **CRISP's step
4 has no such step** — it measures entropy over the three NLI verdict labels
(`supported`/`contradicted`/`unknown`), which are already semantic categories,
so the clustering problem does not arise. But the same principle would
apply if CRISP ever implemented the multi-sample self-consistency its
`self_consistency` proxy stands in for.

**Differs:** no retrieval, no claim-level NLI against evidence, no rewriting.
This is *uncertainty estimation*, not RAG verification.

**Statistics (§3.10).** CoQA dev **~8,000** questions and TriviaQA **8,000**
questions, with **10 sampled answers each** (the paper notes "using more than
10 samples does not significantly improve the performance" — an informal
claim with **no test behind it**). AUROC: CoQA **0.77** (semantic entropy) vs
**0.66** (distinct answers); TriviaQA **0.83** vs **0.79**. Average number of
distinct answer clusters — correct vs incorrect: CoQA **1.27** vs **1.77**;
TriviaQA **1.89** vs **3.89**. Model accuracy 50.6% (TriviaQA) and 82.3%
(CoQA) at Rouge-L > 0.3. Semantic entropy beats plain sampling (AUROC
**0.758** vs beam sampling 0.735). Models: **OPT** at 2.7B/6.7B/13B/30B
(headline on 30B) with **DeBERTa-large** for clustering; training-free ("no
further training or data-gathering"). Human labelling: **300 samples each**
for semantic equivalence and **200 answers per dataset** for accuracy;
automatic-vs-human label agreement CoQA **0.89**, TriviaQA **0.96**. **No
inter-annotator agreement statistic and no annotator count.** **No
confidence intervals, no error bars, no significance tests, no multiple
seeds.**

---

## 4.13 SelfCheckGPT — the closest in *sampling-based consistency*

> *"SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for
> Generative Large Language Models"* — Manakul, Liusie, Gales. **EMNLP 2023**
> · **arXiv:2303.08896**

**Workflow**

```
1. DRAW N additional stochastic responses from the same prompt.

2. SCORE each sentence of the original response R against every sample,
   using one of three variants:

   BERTScore-based: for each sentence r_i, find its most similar
                     sentence in each sample; the inconsistency score
                     is 1 − the mean of those maximum similarities.
                     Backbone: roberta-large.

   Question-answering: generate multiple-choice questions from r_i with
                     distractors, answer each using R and every sample,
                     filter by answerability, then compare:
                     S_QA(i,q) = N_n / (N_m + N_n)
                     Answered with T5-large generation and Longformer
                     answerability models.

   n-gram-based:   train an n-gram model on R plus the samples to
                   approximate the black-box LM's token probabilities;
                   score by average or max negative log-probability.
                   Best form: unigram (max). R acts as smoothing — each
                   token's count in R is increased by 1.

3. AGGREGATE
   Sentence level: one score per sentence.
   Passage level: average the sentence scores over the passage
                 (or average token-level probabilities/entropy).
   A combined score merges the normalised BERTScore, QA and n-gram scores.
```

**Matches CRISP:** C2, C3, C4 — same decompose-then-score shape.

**A correction worth noting:** the "NLI-based" variant often attributed to
SelfCheckGPT is **not** in the paper. Its three variants are **BERTScore,
question answering, and n-gram**. The QA variant uses a Longformer
answerability model, not an entailment model.

**Differs:** the evidence is *other samples*, not retrieved text — so there is
no retrieval, no grounding, and nothing to mitigate against. But it is the
paper that established **sampling consistency** as a detection signal, which
is precisely the mechanism CRISP's `self_consistency` field is a placeholder
for.

**Statistics (§3.10).** **238** generated WikiBio passages, **1,908**
manually labelled sentences (39.9% major-inaccurate, 33.1% minor-inaccurate,
27.0% accurate); **184.7 ± 36.9** tokens per passage; non-factual subset
**206** passages / **1,632** sentences. Passage-level results (AUC-PR /
avg precision / avg F1 / Pearson / Spearman): Combination **87.33 / 44.37 /
61.83 / 69.05 / 67.77**; Unigram-max **85.63 / 41.04 / 58.47 / 64.71 /
64.91**; QA **84.26 / 40.06 / 48.14 / 61.07 / 59.29**; BERTScore **81.96 /
45.96 / 44.23 / 58.18 / 55.90**; GPT-3 max(−log p) **87.51**; Random
**72.96**. Generator **GPT-3 `text-davinci-003`, 175B**; BERTScore backbone
`roberta-large`; question generation **T5-large**; answering and answerability
**Longformer**; N = **20** random samples at temperature 1.0. **The only
paper in the survey reporting how well its labelers agreed**: a
**201-sentence** subset was labelled independently by **two annotators**,
with disagreements resolved to the worse label, giving **Cohen's κ = 0.595**
(three labels, "moderate" agreement) and **κ = 0.748** (two labels,
"substantial" agreement). **No confidence intervals, no error bars, no
significance tests, no multiple seeds.** Note the combination's weights
β₁ = β₂ = 0.8 are **hand-set**.

---

## 4.14 ARES — the methodological precedent for CRISP's fitted head

> *"ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation
> Systems"* — Saad-Falcon, Khattab, Potts, Zaharia. **NAACL 2024** ·
> **arXiv:2311.09476**

**Workflow**

```
1. A SMALL LM annotator labels synthetic training data.
2. Fine-tune lightweight LM JUDGES on it.
3. The judges predict: context relevance, answer faithfulness,
   answer relevance.
4. PREDICTION-POWERED INFERENCE against a small human-labelled set
   corrects the judges' errors, with statistical guarantees.
```

**Matches CRISP:** C1, C2, C3, C4. **ARES is the methodological precedent for
CRISP's step 9.** It demonstrates the pattern CRISP follows: a small
human-annotated set plus a trained scorer yields a calibrated evaluation
signal, with the human set used to correct the scorer's systematic errors.

**Differs:** ARES is an evaluation framework with prediction-powered
inference — no mitigation, no abstention, and the "correction" is statistical
rather than an editing operation.

**Statistics (§3.10).** **ARES is one of only two papers in this survey
that reports a range around its numbers** — and the only one with a
mathematical guarantee. It uses **prediction-powered inference (PPI)**: the
PPI intervals "captured the ground truth average outcomes for context
relevance and answer relevance **more than 95% of the time**", with average
widths of **7.4 points** (context relevance) and **6.1 points** (answer
relevance); cross-domain interval widths range **6.2%–11.9%** and
mock-system PPI ranges **6.8%–9.2%**. Judge accuracy (context relevance /
answer relevance): NQ **79.3% / 97.2%**, HotpotQA **92.3% / 81.3%**, WoW
**85.7% / 96.1%**, FEVER **88.4% / 78.5%**, MultiRC **85.8% / 82.7%**, ReCoRD
**67.8% / 92.3%** — against RAGAS's 15.0–36.4% on the same tasks, an
advantage the paper states as **+59.3** and **+14.4** points (and **+59.9**
in §5.1). Kendall's τ: **0.78–1.0** across the six datasets, averaging 0.91
(context) and 0.97 (answer) on real RAG systems — the paper's success
criterion is "τ greater than 0.9". AIS: judge accuracy 62.5% (WoW) / 84.0%
(CNN/DM), "within 2.5 accuracy points of the correct scores". Data: judge =
**DeBERTa-v3-Large (304M)** fine-tuned on synthetic data from **FLAN-T5 XXL
(11.3B)**; the **exact synthetic-example count is not stated in the paper**,
though negatives are balanced to positives. Human annotation: ~150–300
examples per configuration (300 used for the PPI experiments, 1,350 for the
sampled-annotation baseline). **No inter-annotator agreement statistic**, and
**no standard deviations, error bars, significance tests, bootstrap, or
multiple seeds** — the PPI interval *is* the uncertainty reporting.

---

## 4.15 Don't Hallucinate, Abstain — multi-model abstention

> *"Don't Hallucinate, Abstain: Multi-LLM Collaboration for Hallucination
> Detection"* — Feng, Shi, Wang, Ding, Balachandran, Tsvetkov. **ACL 2024**
> · **arXiv:2402.00367**

**Workflow**

```
1. Adapt calibration and fine-tuning/prompting baselines to detect
   uncertain answers.
2. Diagnose their weaknesses: reliance on self-reflection, and reliance
   on held-out data.
3. Introduce MULTI-LLM COLLABORATION:
     COOPERATIVE  — models work jointly
     COMPETITIVE  — models probe and challenge each other
4. ABSTAIN when a knowledge gap is detected.
5. Analyse the mechanisms to expose retrieval-augmentation failures and
   multi-hop reasoning gaps.
```

**Matches CRISP:** C6.

**The difference in kind:** this paper votes across **models** on the
knowledge-gap question; CRISP fuses across **signals within one pipeline**
(similarity, retrieval, NLI, entropy, coverage, grounding). The two are
complementary rather than competing — a multi-model variant of CRISP's step 9
is an obvious extension.

**Statistics (§3.10).** Test splits: MMLU **1,000**, K-Crosswords **1,007**,
HellaSwag **1,000**, AmbigQA **1,000**, Propaganda **200**, ElectionQA23
**200**. The cooperative/competitive variants beat the strongest baseline in
**9 of 12** model–dataset settings, averaging **+10.1%** reliable accuracy,
with a best-case improvement of **+19.3%**. Average abstain-accuracy by
baseline family: Calibration **0.595**, Training **0.576**, Consistency
**0.553**, Prompting **0.475**; by model: ChatGPT **0.616**, LLaMA2 **0.537**,
Mistral **0.524**. The two-stage abstain→retrieve→abstain stage cuts the
wrong-answer rate by up to **56.6%**, and K-Crosswords knowledge-gap
localisation error rates drop **67.2–81.2%** (also stated as ≥82%).
Abstention ECE (Mistral-7B) ranges **0.1935** (Propaganda, Compete) to
**0.7281** (K-Crosswords, MoreInfo) — i.e. how well-calibrated the
abstention is varies by an order of magnitude across settings. Models:
**Mistral-7B**, **LLaMA2-70B-chat**, **ChatGPT**; no GPT-4. ElectionQA23
items were manually checked but **the number reviewed is not stated and no
agreement figure is given**. **No confidence intervals, no error bars, no
significance tests, no multiple seeds.**

---

## 4.16 The retrieval-side family (no verification stage)

These match CRISP's steps 1, 6 and 7 but have **no detection, fusion,
mitigation, or abstention**.

| Paper | Workflow | Relevance to CRISP |
|---|---|---|
| **FLARE** (Jiang et al., EMNLP 2023, [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)) | 1. Predict the next sentence → 2. Detect low-confidence tokens → 3. Use the predicted sentence as the retrieval query → 4. Retrieve and revise the sentence | Parallels CRISP's confidence-gated retrieval but at token level *during* generation. CRISP verifies *after* generation. |
| **DRAGIN** (Su et al., ACL 2024, [arXiv:2403.10081](https://arxiv.org/abs/2403.10081)) | 1. Awareness: does the generator lack the info to continue? → 2. Represent the need from the *full* context → 3. Active retrieval from that need → 4. Feed back, forming a need↔retrieval loop | CRISP's step 1 is the analogue of DRAGIN's information-need module. **DRAGIN's design is the answer to CRISP's known weakness** — the query rewrite that is computed and then discarded (see §3.8). |
| **IRCoT** (Trivedi et al., ACL 2023, [arXiv:2212.10509](https://arxiv.org/abs/2212.10509)) | 1. Sample a sentence from the current chain-of-thought → 2. Retrieve with it → 3. Retrieve again with the original question → 4. Append to the CoT → repeat | The classic precedent for reasoning-conditioned retrieval — CRISP's steps 1–3 at reasoning-step rather than query level. |
| **CoVe** (Dhuliawala et al., [arXiv:2309.11495](https://arxiv.org/abs/2309.11495)) | 1. Draft an answer → 2. Plan verification questions targeting the draft → 3. Answer each independently → 4. Produce a final answer conditioned on the results | CRISP's mitigation stage descends from this. **Caveat: the abstract does not state that CoVe retrieves external sources** — do not claim source-grounded verification for it. |

**Statistics for the four retrieval-side papers (§3.10).** These are
included because they share CRISP's *workflow shape* (query understanding →
retrieval → generation), even though they contribute no detection layer.

| Paper | Eval size | Headline numbers | Uncertainty | Statistical notes |
|---|---|---|---|---|
| **CoVe** | Wikidata **56** q (~600 gold entities each); Wiki-Category **55** q (~8 answers); MultiSpanQA **418** q; biography count **not stated** | MultiSpanQA F1 **0.39** (few-shot) → **0.46** (joint CoVe) → **0.48** (factored) = **+23%**; Wikidata precision 0.17 → 0.29 → 0.36 → 0.32, average hallucinated entities **2.95 → 0.68**; biography FactScore **55.9 → 60.8 → 63.7 → 71.4** (**+28%**) | none | No confidence intervals, no error bars, no significance tests, no multiple seeds. No human annotation is reported for CoVe itself (FactScore's correlation with human judgement is inherited). Model **Llama 65B**; 3 few-shot examples per task; FactScore checker "Llama + Retrieval + NP". The **+23% / +28%** gains are the closest published precedent for the kind of result CRISP's mitigation stage was aiming at — and CRISP's measured Δ is **−0.007 (p = 0.404)**. |
| **FLARE** | 2Wiki **500**, StrategyQA **229**, ASQA **500**, WikiAsp **500** — capped "at most 500 examples from each dataset due to the cost of running experiments" | 2Wiki EM **51.0** / F1 **59.7** vs no-retrieval EM **28.2** / F1 **36.8** (**+22.8 EM**, ~80.9% relative); vs question-decomposition baseline EM 47.8 (**+3.2**); StrategyQA EM **77.3** vs 72.9; ASQA EM 41.3 / Disambig-F1 27.1 / ROUGE-L 34.3; ASQA-hint EM **46.2** | none | No confidence intervals, no error bars, no significance tests, no multiple seeds. No parameter counts given (GPT-3.5 `text-davinci-003`), and no agreement figure for the manually written exemplars (8 ASQA, 4 WikiAsp). Threshold ablation: masking threshold 0.0 → 0.4 raises 2Wiki EM **48.8 → 51.0**. Retrieval activates for **30–60%** of sentences. |
| **DRAGIN** | 2Wiki **1,000**, HotpotQA **1,000**, IIRC **954**, StrategyQA **1,000** = **3,954** | LLaMA-2-13B-chat — 2Wiki EM **0.304** / F1 **0.3931**; HotpotQA EM **0.314** / F1 **0.4238**; StrategyQA acc **0.689**; IIRC EM **0.185** / F1 **0.2221**. LLaMA-2-7B: 2Wiki 0.220/0.2926, HotpotQA 0.232/0.3344. Vicuna-13B-v1.5: IIRC 0.2233/0.2652 | none | No confidence intervals, no error bars, no significance tests, no multiple seeds (greedy decoding used for reproducibility). RIND threshold sensitivity 0.3–1: HotpotQA EM **0.293–0.304** — the paper's substitute for an error bar, and it shows a narrow band. **No relative-improvement percentages are reported**, only absolute EM/F1. BM25 beats SGPT on every dataset. Wikipedia segmented into **100-token passages**, top-3 retrieved. |
| **IRCoT** | **500** test + **100** tuning per dataset (HotpotQA, 2Wiki, MuSiQue, IIRC); human error set **160**; human CoT set **80** | Flan-T5-XXL F1 — HotpotQA **59.1 ± 0.9**, 2Wiki **66.5 ± 1.4**, MuSiQue **30.8 ± 0.2**, IIRC **42.5 ± 2.1**; GPT-3 CoT reader — 60.7 ± 1.1 / 68.0 ± 1.5 / 36.5 ± 1.2 / 49.9 ± 1.1. Factual errors reduced **50%** (HotpotQA) / **40%** (2Wiki) vs OneR | **±std** over 3 demonstration sets | The **only paper in the survey that reports standard deviations**: "We report the mean and standard deviation of these 3 results for each experiment." The ± values are across **demonstration sets, not random seeds** — a weaker guarantee than a seed spread. No confidence intervals, no significance tests, no bootstrap, no seeds. Corpora: HotpotQA 5,233,329 paragraphs; IIRC 1,882,415; 2Wiki 430,225; MuSiQue 139,416. Notably, DSP (**+2.0 F1** on HotpotQA) and DecomP (**+2.8 F1** on 2Wiki) beat IRCoT's own extended comparison. |

---

# 5. What this survey says about CRISP's position

## 5.1 The genuine gap

No paper found combines all six characteristics. The three sparse categories,
counted from the §2 grid, are:

- **C5, post-hoc span rewriting** — only FAVA (learned tagger) and "A Stitch
  in Time" (instruction-based repair) qualify. Neither verifies its own
  output afterwards, which CRISP's regeneration loop does.
- **C6, selective prediction** — CRAG, Sufficient Context, "Don't
  Hallucinate" and Semantic Uncertainty gate or abstain; none of them
  combines the gate with a repair loop.
- **C4, learned multi-signal fusion** — Self-RAG (token-probability fusion
  inside the generator) and ARES (prediction-powered inference) are the
  precedents. Nothing does what CRISP's step 9 does: a *separately fitted
  logistic head over seven heterogeneous, standardised, correctly-signed
  features*.

Only one surveyed pipeline implements all three of C4, C5 and C6: CRAG.

**CRISP's claim should be framed as the combination**, not as any single
component.

## 5.2 Concrete improvements these papers suggest

**This table is the short version. §3.8 gives the same list with the exact
file each change would land in.**

| Suggestion | Source | Effort | Would it invalidate current numbers? |
|---|---|---|---|
| Make the EEDC weights fitted, not hard-coded | `scripts/calibrate_eedc.py` already exists | Low | **Yes** — step 9's fitted weights would need a refit |
| Use AlignScore as the step-8 verifier | AlignScore | Low | **Yes** — every detection number changes |
| Implement real multi-sample self-consistency | Semantic Uncertainty, SelfCheckGPT | High | **Yes** — adds latency and a new feature |
| Apply the query rewrite (currently discarded) | DRAGIN | Low | **Yes** — retrieval behaviour changes |
| Implement dynamic RRF weighting per query | Hybrid Retrieval (Mala et al.) | Medium | **Yes** — retrieval changes |
| Adopt FAVA's six-type hallucination taxonomy | FAVA | Medium | Partly — finer error typing, same headline numbers |
| Report noise-sensitivity split (relevant vs irrelevant chunk) | RAGChecker | Medium | No — an additional metric |
| Exclude abstentions from the model-level average | FActScore | Low | No — a reporting change |

## 5.3 The strongest counter-arguments to anticipate

These are the objections a reviewer is most likely to raise, and the papers
that support them.

1. **"Better retrieval alone fixes this."** Mala et al. (§4.11) show hybrid
   retrieval beating sparse and dense on all four overall metrics across
   six datasets. CRISP must argue that detection-layer work is *additional*,
   not substitutable.

2. **"Uncertainty is not error."** The NLI's `unknown` verdict drives
   abstention, so the pipeline refuses good answers. Sufficient Context
   (§4.9) demonstrates the same tension from the retrieval side.

3. **"Your EEDC weights were never fitted."** `data/eedc_weights.json` does
   not exist; the defaults are hand-set. The script to fix this already
   exists.

4. **"Your step-3 detector is at chance."** AUC 0.5389 as a risk signal.
   It survives only as one weighted component of seven.

5. **"Your mitigation delta is negative."** The `claim_edit` rewrite is a
   design fix whose measured benefit is ≈ −0.007. FAVA (§4.2) and "A Stitch
   in Time" (§4.10) both report large, positive mitigation results — so the
   question "why does yours not work?" is live.

6. **"RAGChecker already does the decomposition."** It separates retrieval
   from generation failure with a richer metric set (§4.1).

7. **"Your test set is small, and a bootstrap isn't a real guarantee."**
   Half right, and §3.11.4 answers it in two parts. CRISP's 1,000 rows *are*
   larger than the hallucination-scored set of every paper in the survey
   (next largest: Hybrid Retrieval 300, ARES 300) — the 13,867 figure
   belongs to a retrieval pool, not to a hallucination benchmark. But a
   bootstrap range does estimate how much the number moves with the data;
   it is not a guarantee like ARES's PPI interval. The honest claim is
   "CRISP reports uncertainty everywhere", not "CRISP's uncertainty is the
   tightest".

8. **"Your fusion head is fitted on 496 rows; Self-RAG used 145,619 and
   AlignScore 4.7M."** This is the strongest objection in the survey and it
   cannot be answered with a number. The only mitigations available are
   methodological: report the **hold-out** number (0.6185, CI [0.5371,
   0.6943]) rather than the tune number (0.7093), and treat the head as a
   hypothesis to be refitted rather than a settled contribution. §3.3
   already states this as the honest weakness of the fusion claim.

9. **"Other papers report larger effects than you."** A Stitch in Time
   reports hallucination **47.4% → 14.53%**; CoVe reports **+23% F1** and
   **+28% FactScore**; Sufficient Context reports **+2–10%** correct answers.
   CRISP's mitigation Δ is **−0.007, CI [−0.022, +0.007], p = 0.404** — i.e.
   *no measurable effect*, which is a weaker position than "a small
   positive effect" but a stronger one than "an unmeasured effect". The
   comparison is not like-for-like (§3.11.2: different metrics, different
   datasets, different annotation budgets), and CRISP's own cross-paper
   comparisons live in `Paper/ragtruth_comparison.md` and
   `Paper/halueval_papers.md`.

10. **"Only two papers in your own survey report confidence intervals."**
    That is the point of §3.10.4, and it cuts both ways: CRISP is the only
    system that covers all six characteristics *and* publishes error bands
    on every headline number — but it is also the only one whose bands
    *include "no effect"* for the mitigation claim. A survey that hid this
    would not be an asset; a survey that states it is the reason a reviewer
    believes the rest.

## 5.4 Papers that must not be miscounted

Two categories to be careful about in any related-work count:

- **Datasets are not methods.** RAGTruth and HaluEval are benchmarks, not
  pipelines. Listing them as "comparable systems" would be wrong.
- **A related-work citation is not an evaluation.** A paper that merely cites
  HaluEval, or says it *will* evaluate on it, does not belong in a
  head-to-head table. `Paper/halueval_papers.md` already applies this rule
  and finds that only 2 of 12 local papers actually ran HaluEval
  experiments.

---

# 6. Reference list

| # | Paper | Venue | arXiv | Role here |
|---|---|---|---|---|
| 1 | RAGChecker: A Fine-grained Framework for Diagnosing RAG | — | [2408.08067](https://arxiv.org/abs/2408.08067) | diagnosis |
| 2 | Fine-grained Hallucination Detection and Editing (FAVA) | COLM 2024 | [2401.06855](https://arxiv.org/abs/2401.06855) | detection + editing |
| 3 | FActScore | EMNLP 2023 | [2305.14251](https://arxiv.org/abs/2305.14251) | atomic facts |
| 4 | RAGAS | — | [2309.15217](https://arxiv.org/abs/2309.15217) | claim extraction |
| 5 | LongFact / SAFE | NeurIPS 2024 | [2403.18802](https://arxiv.org/abs/2403.18802) | long-form verification |
| 6 | AlignScore | ACL 2023 | [2305.16739](https://arxiv.org/abs/2305.16739) | verifier |
| 7 | Corrective RAG (CRAG) | — | [2401.15884](https://arxiv.org/abs/2401.15884) | retrieval gating |
| 8 | Self-RAG | ICLR 2024 | [2310.11511](https://arxiv.org/abs/2310.11511) | fusion |
| 9 | Sufficient Context | — | [2411.06037](https://arxiv.org/abs/2411.06037) | abstention |
| 10 | A Stitch in Time Saves Nine | — | [2307.03987](https://arxiv.org/abs/2307.03987) | mitigation |
| 11 | Hybrid Retrieval for Hallucination Mitigation | — | [2504.05324](https://arxiv.org/abs/2504.05324) | hybrid retrieval |
| 12 | Semantic Uncertainty | ICLR 2023 | [2302.09664](https://arxiv.org/abs/2302.09664) | entropy + abstention |
| 13 | SelfCheckGPT | EMNLP 2023 | [2303.08896](https://arxiv.org/abs/2303.08896) | sampling consistency |
| 14 | ARES | NAACL 2024 | [2311.09476](https://arxiv.org/abs/2311.09476) | fitted scorer |
| 15 | Don't Hallucinate, Abstain | ACL 2024 | [2402.00367](https://arxiv.org/abs/2402.00367) | multi-model abstention |
| 16 | FLARE | EMNLP 2023 | [2305.06983](https://arxiv.org/abs/2305.06983) | active retrieval |
| 17 | DRAGIN | ACL 2024 | [2403.10081](https://arxiv.org/abs/2403.10081) | information need |
| 18 | IRCoT | ACL 2023 | [2212.10509](https://arxiv.org/abs/2212.10509) | iterative retrieval |
| 19 | Chain-of-Verification (CoVe) | — | [2309.11495](https://arxiv.org/abs/2309.11495) | verification chain |

**Method note.** Pipeline descriptions for FAVA, RAGChecker, FActScore, CRAG,
Sufficient Context, Self-RAG, SelfCheckGPT, "A Stitch in Time" and Hybrid
Retrieval were read from each paper's own method section. Details that could
not be confirmed from the paper are marked as uncertain in the text above
rather than stated as fact.

**Method note for the statistical tables (§3.10–§3.13).** Every dataset
name, sample size, headline number, model/backbone size, training-data size
and statistical-rigour item in §3.10–§3.13 was read from the paper's own
arXiv full text (HTML versions: RAGChecker v2, FAVA v1, FActScore v2, RAGAS
v2, SAFE v3, AlignScore v1, CRAG v3, Self-RAG v2, Sufficient Context v1,
A Stitch in Time v1, Hybrid Retrieval v1, Semantic Uncertainty v3,
SelfCheckGPT v3, ARES v2, Don't Hallucinate v1, FLARE v2, DRAGIN v2, IRCoT
v2, CoVe v2). Each paper's full text was searched for *confidence interval*,
*error bar*, *standard deviation*, *significan\**, *bootstrap*, *random
seed* and *variance*; a cell reads "not reported" where none of these
appear. CRISP's own numbers were read from the repository artefacts listed
in §3.10.5, not from memory.

**Two cautions on cross-reading these tables.**

1. **Do not conflate SAFE's 496 with CRISP's 496.** SAFE's 496 is a
   human-labelled *evaluation* set of prompt–response pairs; CRISP's 496 is
   a *fitting* set for the fusion head. Unrelated quantities, coincidentally
   equal.
2. **Metric names are not comparable across papers.** SAFE's `F1@64`,
   FActScore's `FS`, SelfCheckGPT's `AUC-PR`, ARES's *accuracy*, and
   CRISP's *claim-level F1* are five different quantities. §3.11.2 lists
   which pairings are and are not legitimate.
