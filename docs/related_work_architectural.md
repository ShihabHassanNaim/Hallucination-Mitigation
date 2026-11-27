# Related Work — Systems Built Like This One

**Scope of this chapter.** This chapter reviews the published work that is
*architecturally* closest to the pipeline described in Chapter 3: systems
that also decompose a retrieval-augmented answer into claims, verify those
claims against retrieved evidence, combine several confidence signals into a
single risk score, repair what is unsupported, and refuse to answer when the
risk is too high.

It is deliberately narrower than Chapter 2 and broader than Chapter 7.
Chapter 2 organised the literature by *detection signal* — similarity,
model-aware, ensemble, self-consistency — and therefore treats any two
papers that share a detector as neighbours. This chapter organises it by
*pipeline shape*, so that papers which differ in detector but agree in
architecture are considered together. Chapter 7 then compares scores, and
only where a like-for-like comparison exists.

**Why the distinction matters.** A reader who has read Chapter 2 may
reasonably conclude that claim-level verification is well covered and that
the contribution of this project lies in the detector. It does not. The
survey below covers **nineteen papers**, and it finds that the components
this project combines are individually well studied and jointly unstudied:
of six architectural properties that together define a defensible
hallucination-defence pipeline, **no published paper implements all six**,
and the two thinnest properties are precisely the two in which this project
claims its contribution.

**How the evidence in this chapter was gathered.** Every pipeline
description, dataset name, sample size and reported figure below was read
from the paper's own full text rather than from its abstract, from a
secondary summary, or from memory. Where a paper does not report a
quantity, the text says so rather than leaving a gap or inferring a value.
The chapter is written in the same register as the rest of the thesis: where
this project is behind the published literature, that is stated directly.

---

## 2.1 A framework for describing a hallucination-defence pipeline

A pipeline that responds to hallucination in a retrieval-augmented system
can be described along six architectural dimensions. These are properties of
the *system*, not of its detector, and they can be satisfied independently.

| # | Property | The question it answers |
|---|---|---|
| **P1** | **Retrieval-grounded scope** | Does the system target hallucinations in retrieval-augmented generation, or hallucination in text generally? |
| **P2** | **Multi-stage pipeline** | Does it run a sequence of query understanding, retrieval, ranking and generation, rather than a single pass? |
| **P3** | **Claim-level verification** | Does it split the answer into atomic claims and judge each against evidence, rather than scoring the answer as a whole? |
| **P4** | **Multi-signal fusion** | Does it combine several distinct confidence signals into one risk score, and is that combination fitted or hand-set? |
| **P5** | **Post-hoc repair** | After detection, does it change the answer, and at what granularity — the whole answer, a sentence, a claim, a span? |
| **P6** | **Selective prediction** | Does it decline to answer when it judges the answer to be unreliable? |

The remaining sections use these six properties to place the literature, and
the final section states what follows for the design of this project.

---

## 2.2 The landscape

Table 2.1 scores each of the nineteen surveyed papers against the six
properties. A tick means the property is present; a dash means it is absent.

**Table 2.1 — Architectural properties of the surveyed literature**

| System | P1 | P2 | P3 | P4 | P5 | P6 | Closest to this project at |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| RAGChecker [38] | ✓ | ✓ | ✓ | — | — | — | diagnosis |
| FAVA [39] | ✓ | ✓ | — | — | ✓ | — | detection and editing |
| FActScore [40] | ✓ | ✓ | ✓ | — | — | — | claim decomposition |
| RAGAS [41] | ✓ | ✓ | ✓ | — | — | — | claim decomposition |
| SAFE [42] | ✓ | ✓ | ✓ | — | — | — | long-form verification |
| AlignScore [43] | ✓ | — | ✓ | — | — | — | the verifier |
| CRAG [44] | ✓ | ✓ | — | ✓ | ✓ | ✓ | retrieval gating |
| Self-RAG [45] | ✓ | ✓ | ✓ | ✓ | — | — | confidence fusion |
| Sufficient Context [46] | ✓ | — | — | ✓ | — | ✓ | the abstention gate |
| A Stitch in Time [47] | ✓ | ✓ | ✓ | ✓ | ✓ | — | repair |
| Hybrid Retrieval [48] | ✓ | ✓ | — | — | — | ✓ | retrieval |
| Semantic Uncertainty [49] | — | — | — | ✓ | — | ✓ | entropy |
| SelfCheckGPT [4] | — | ✓ | ✓ | ✓ | — | — | sampling consistency |
| Don't Hallucinate, Abstain [50] | — | — | — | ✓ | — | ✓ | abstention |
| ARES [51] | ✓ | ✓ | ✓ | ✓ | — | — | fusion methodology |
| Chain-of-Verification [52] | ✓ | ✓ | ✓ | — | ✓ | — | repair |
| FLARE [53] | ✓ | ✓ | — | — | — | — | retrieval |
| DRAGIN [54] | ✓ | ✓ | — | — | — | — | information need |
| IRCoT [55] | ✓ | ✓ | — | — | — | — | iterative retrieval |
| **This project** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

**The finding.** The two thinnest columns are **P4 (multi-signal fusion)**,
satisfied by eight of the nineteen systems, and **P5 (post-hoc repair)**,
satisfied by five. Both are properties this project claims to contribute.
The combination of all six is satisfied by none.

This finding is narrower than it first appears, and the qualification
matters. A survey of pipelines with similar shapes will always find that no
single paper does everything; that is a statement about the design space,
not evidence of a contribution. What makes the finding usable is that it
identifies *which* combinations are unstudied, and Table 2.2 shows that the
unstudied combinations are not arbitrary — they are the ones that are
expensive to build, because they require several independently trained
components to be calibrated against one another.

**Table 2.2 — The unstudied combinations, and what each one would cost**

| Combination | Systems realising it | Why it is hard to build | Status |
|---|---|---|---|
| P1 + P2 + P3 | RAGChecker, FActScore, RAGAS, SAFE, AlignScore | Detection alone; no fusion, repair or gate | Well covered |
| P1 + P2 + P3 + P4 | Self-RAG, ARES | Fusion present, but nothing downstream consumes the score | Well covered |
| P1 + P2 + P3 + P5 | CoVe | Repair present, but no fused risk score decides whether to repair | Covered, but no gate |
| P1 + P2 + P5 | FAVA | Span-level repair with a learned tagger, but no fusion and no gate | Covered, but no fusion |
| P1 + P2 + P3 + P4 + P5 | A Stitch in Time | Fusion is a single threshold on minimum token probability, not a fitted multi-signal head | Covered, but fusion is hand-set |
| P1 + P2 + P4 + P5 + P6 | CRAG | CRAG's fusion, repair and gate all act on the *evidence*, never on the answer | Covered, but evidence-side only |
| P1 + P4 + P6 | Sufficient Context | The gate fires on evidence sufficiency, before verification | Covered, but gate placed early |
| **P1–P6 together** | **none** | Requires a claim-level verifier, a separately fitted multi-signal head, a repair loop that re-verifies its own output, and a gate that fires after verification — four components whose outputs must be calibrated against one another | **This project** |

The last row is the only one that is not realised by any published system,
and its difficulty is the reason it is rare: each of the four components is
individually a paper-sized contribution, and combining them requires the
fusion head to be *fitted* rather than set by hand, against components that
were trained on different data with different conventions.

---

## 2.3 Where each property is realised

A tick in Table 2.1 conceals a great deal. Self-RAG and Sufficient Context
both fuse signals, but they fuse different signals for different purposes by
different mechanisms. Table 2.3 states what each property takes in each
system.

**Table 2.3 — The form each property takes**

| System | Verification unit | Fusion mechanism | Repair target | Gate signal |
|---|---|---|---|---|
| **This project** | atomic claim | logistic head over 7 standardised features, fitted | claim span, re-verified | fused risk score |
| RAGChecker [38] | claim | — | — | — |
| FAVA [39] | tagged span | — | span and error type | — |
| FActScore [40] | atomic fact | — | — | — |
| RAGAS [41] | statement | — | — | — |
| SAFE [42] | fact | — | — | — |
| AlignScore [43] | text pair | — | — | — |
| CRAG [44] | — | single relevance score, three bands | evidence | retrieval evaluator |
| Self-RAG [45] | segment | weighted token probabilities | — | — |
| Sufficient Context [46] | — | logistic regression, 2 features | — | sufficiency plus self-confidence |
| A Stitch in Time [47] | concept | minimum token probability | sentence | — |
| Hybrid Retrieval [48] | — | — | — | retrieval adequacy |
| Semantic Uncertainty [49] | semantic cluster | entropy over clusters | — | semantic entropy |
| SelfCheckGPT [4] | sentence | three score variants, merged | — | — |
| Don't Hallucinate [50] | — | cross-model vote | — | knowledge-gap vote |
| ARES [51] | context-answer pair | prediction-powered inference | — | — |
| Chain-of-Verification [52] | verification question | — | whole answer | — |
| FLARE [53] | — | — | sentence | token confidence |
| DRAGIN [54] | — | — | — | — |
| IRCoT [55] | — | — | — | — |

Three distinctions in this table are worth drawing out, because they are the
places where the nearest neighbours differ from this project in kind rather
than in degree.

**Fusion is not one thing.** Sufficient Context fits a logistic model over
two features; ARES uses prediction-powered inference to correct a trained
scorer against human labels; Self-RAG combines normalised token
probabilities inside the generator, using hand-set group weights of 1.0,
1.0 and 0.5. The project fits a separate logistic head over seven
standardised features, with the direction of every component recorded
explicitly in the configuration. That the earlier hand-set version of this
fusion scored **below chance** — a ROC-AUC of 0.360 — is the reason the
sign map is treated as part of the artefact rather than as documentation.

**Repair has three targets.** CRAG and Hybrid Retrieval repair the
*evidence*, replacing retrieved passages before generation. FAVA, A Stitch in
Time and FLARE repair the *answer*, at span, sentence and token
granularity respectively. Chain-of-Verification replaces the *whole answer*.
This project repairs at the granularity of the atomic claim and preserves
the rest of the answer unchanged.

**Gates sit at different points in the pipeline.** Sufficient Context gates
on whether the evidence was sufficient, before generation. CRAG gates on
retrieval quality. The project gates on the risk of the answer, after
verification has run. These fail in different places: a system can have
sufficient evidence and still produce a wrong answer, which only a
post-verification gate catches; and it can have insufficient evidence while
producing a correct answer, which a pre-generation gate discards needlessly.

---

## 2.4 The three closest systems

Of the nineteen, three are close enough that a reader will want them
compared directly rather than through a table.

### 2.4.1 FAVA — the closest in detection and repair

FAVA [39] is the only surveyed paper that both retrieves and rewrites spans.
It fine-tunes a Llama2-Chat-7B model to emit tagged output identifying the
erroneous span, its error type among six categories, and a proposed
correction, over 35,074 synthetic training instances built by corrupting
correct text.

Three differences matter. FAVA's verification is *learned* rather than
entailment-based, which means it is bound to one checkpoint, whereas this
project's is a general-purpose NLI model that can be replaced without
retraining. FAVA's edits are tag insertions that are not subsequently
checked. And FAVA's six-type error taxonomy is finer than the three-way
supported/contradicted/unknown scheme this project uses, which is a
limitation of the present design rather than a strength of it.

### 2.4.2 Self-RAG — the closest in fusion

Self-RAG [45] trains a critic to emit reflection tokens and combines their
normalised probabilities into a score used during beam search. It is the
clearest published precedent for the fusion stage of this project, and the
direct ancestor of two ideas this project uses: the normalised-probability
trick, in which each signal is rescaled so that no single one dominates by
virtue of its raw scale, and the practice of making the direction of each
signal explicit.

Self-RAG differs in three respects. Its fusion is internal to the generator
rather than a separate head over heterogeneous features. It has no repair
stage, since it re-ranks candidates during decoding rather than repairing a
finished answer. And it has no abstention stage. Its training scale is also
far larger: 145,619 instruction-output pairs for the generator and four to
twenty thousand labels per reflection type for the critic, against the 496
rows on which this project's fusion head is fitted. This is the single
largest quantitative gap between this project and the literature, and it is
discussed in Section 2.8.

### 2.4.3 Sufficient Context — the closest in abstention

Sufficient Context [46] classifies whether the retrieved context supports
any plausible answer, combines that binary signal with the model's own
self-rated confidence through a fitted logistic model, and rejects answers
whose predicted risk falls below a threshold. It is the clearest precedent
for the final gate of this project, and it establishes the important negative
result that fine-tuning a model to abstain does not reliably reduce
hallucination — some variants abstained more often while also producing
fewer correct answers. That result argues for thresholding a calibrated risk
score rather than for fine-tuning, which is the choice this project makes.

### 2.4.4 ARES — the closest in fitting methodology

ARES [51] is the methodological precedent for the fusion stage. It
demonstrates the pattern this project follows: a small human-annotated set
plus a trained scorer yields a calibrated evaluation signal, with the human
set used to correct the scorer's systematic errors, and the resulting
intervals carry a statistical guarantee.

ARES differs in that it is an evaluation framework with prediction-powered
inference rather than a defence system, and in that it requires human
annotation that this project does not have. Its guarantee is stronger than
this project's: prediction-powered intervals are valid at a stated
confidence level, whereas a bootstrap interval estimates sampling
variability. What ARES establishes is that the pattern is sound, and its
synthetic-example count is not stated in the paper.

---

## 2.5 The evidence behind the claims

The survey above compares architectures. It does not by itself establish
that the reported numbers can be compared, and Chapter 7 shows that for most
of the literature they cannot. This section records what each paper
actually measured, because a pipeline comparison is only as credible as the
measurement on which it rests.

**Table 2.4 — Evaluation sets and reported results**

| System | Evaluation set | Size | Headline reported result | Reports uncertainty? |
|---|---|---|---|:-:|
| **This project** | HaluEval, TruthfulQA, RAGTruth | 1,000 rows each (500 / 500 / 277 unique questions) | ROC-AUC 0.9251 / 0.6355 / 0.6127 | ✓ **bootstrap, all metrics** |
| RAGChecker [38] | 10 domains | 4,162 queries | P 62.0 / R 53.0 / F1 52.7 | ✗ |
| FAVA [39] | own set + 150-prompt benchmark | 364 passages | detection +22.1% over ChatGPT | ✗ |
| FActScore [40] | human set + 13 subjects | 183 entities, 6,500 generations | FActScore 42.5 / 58.3 / 71.5% | ✗ |
| RAGAS [41] | WikiEval | 50 instances | agreement 0.95 / 0.78 / 0.70 | ✗ |
| SAFE [42] | LongFact | 250 main, 496 human pairs | SAFE-human agreement 72.0% | p only |
| AlignScore [43] | 22 datasets | not stated | SummaC AUC-ROC 87.4 / 88.6 | ✗ |
| CRAG [44] | PopQA + 3 others | 1,399 (PopQA) | PopQA accuracy 52.8 → 59.8 | ✗ |
| Self-RAG [45] | PopQA, TriviaQA + 4 others | 1,399 / 11,313 | PopQA 54.9 / 55.8 | ✗ |
| Sufficient Context [46] | FreshQA, Musique, HotPotQA | 115 / 452 / 500 | autorater F1 0.935; +2–10% correct | ✗ |
| A Stitch in Time [47] | custom article task | 750 sentences | hallucination 47.4% → 14.53% | ✗ |
| Hybrid Retrieval [48] | HaluBench | 13,867 pool, 300 scored | MAP@3 0.897 | ✗ |
| Semantic Uncertainty [49] | CoQA, TriviaQA | 8,000 each | AUROC 0.77 / 0.83 | ✗ |
| SelfCheckGPT [4] | WikiBio | 238 passages, 1,908 sentences | passage AUC-PR 87.33 | ✗ |
| Don't Hallucinate [50] | 6 datasets | 200–1,094 | +10.1% reliable accuracy | ✗ |
| ARES [51] | KILT, SuperGLUE, AIS | 300 PPI, 707, 510 | relevance accuracy 79.3–92.3% | ✓ **PPI** |
| Chain-of-Verification [52] | Wikidata, Wiki-Cat, MultiSpanQA | 56 / 55 / 418 | F1 0.39 → 0.48; FactScore +28% | ✗ |
| FLARE [53] | 2Wiki, StrategyQA, ASQA, WikiAsp | 500 / 229 / 500 / 500 | EM 51.0 / F1 59.7 | ✗ |
| DRAGIN [54] | 2Wiki, HotpotQA, IIRC, StrategyQA | 3,954 | EM 0.304 / F1 0.3931 | ✗ |
| IRCoT [55] | HotpotQA, 2Wiki, MuSiQue, IIRC | 500 each | F1 59.1±0.9 / 66.5±1.4 | ±std only |

Three observations follow.

**The metrics do not overlap.** The surveyed papers report ROC-AUC,
AUC-PR, accuracy, F1, FActScore, MAP@3, EM, precision-recall curves and
rates of change, and these are not interchangeable. F1@64 in SAFE is a
budgeted fact-level score; FActScore in FActScore is an estimate of a
supported-fact fraction; the claim-level F1 reported here is a different
quantity again. The only clean head-to-head available anywhere in this
literature is ARES against RAGAS, which report the same metric on the same
datasets and differ by 59.3 and 14.4 percentage points. **No like-for-like
comparison between this project and any of the nineteen surveyed papers
exists**, which is why the comparisons of record are those in Chapter 7,
restricted to papers that report a comparable metric on a comparable
dataset.

**The evaluation sizes are not comparable either.** They range from 50
instances to 13,867, and the largest figures are not always measuring what
they appear to measure. Hybrid Retrieval's 13,867 is a pool of retrieval
documents, of which 300 are scored for hallucination. Self-RAG's 11,313 is
the full TriviaQA test set for a general question-answering task. This
project's effective sample sizes are 500, 500 and 277 unique questions. It
does not have the largest evaluation in this literature; Semantic
Uncertainty at 16,000 and FActScore at 6,500 are larger.

**Uncertainty is almost never reported.** Seventeen of the nineteen report a
single number with nothing attached. ARES reports prediction-powered
intervals; IRCoT reports standard deviations across three demonstration
sets, which is a weaker guarantee than a spread across random seeds. Only
SelfCheckGPT reports a formal inter-annotator agreement statistic, with
Cohen's kappa of 0.595 on three labels and 0.748 on two. FAVA states that
agreement was computed using "Cohen kappa scores" but prints no numerical
value, reporting only agreement percentages of 75.1% and 60.3%.

---

## 2.6 The literature on this project's three benchmarks

The survey in Section 2.2 organises the literature by *architecture*. A
second, independent sweep was run over the three benchmarks this project
evaluates on — HaluEval [11], TruthfulQA [12] and RAGTruth [10] — asking a
different question: **who has actually run a detection experiment on this
benchmark, on how much data, and reporting which metric?** The two sweeps
intersect only partially, and the union is what a reviewer will consult.

This section is deliberately adversarial towards this project. The three
benchmarks are the ones on which the strongest published numbers are found,
and on one of the three — RAGTruth — this project is beaten outright by
three separate papers. On HaluEval no published paper reports a comparable
metric at all, and on TruthfulQA no published paper reports the dataset with
a comparable metric, so neither of those two is a win or a loss against the
literature; they are reported here for what they are.

**Table 2.5 — Every paper that reports a comparable detection number on RAGTruth**

| Paper | What it reports | Comparable? | Verdict against our 0.6127, CI [0.5494, 0.6581] |
|---|---|---|---|
| LLM-Check [5] | 5 response-level detection AUROC anchors, 0.5168–0.5830 | **Yes** — same metric | **3 WIN, 2 TIE** |
| AES [9] | span-level AUROC 0.8807–0.9332 across five backbones | Protocol differs (span) | **LOSS** on all five |
| Frequency-Aware Attention [56] | span-level AUROC 0.7877–0.8812 (pooled); token-level 0.4561–0.9148 | Protocol differs (token/span) | **LOSS** on all pooled averages |
| Desiderata [57] | response-level ROC-AUC 0.584–0.871, PR-AUC 0.199–0.694 | **Yes** — response-level | **LOSS** on ROC-AUC (4), **TIE** (2); on F1 **3 WIN, 2 TIE, 1 LOSS** |
| RAGTruth paper [10] | response-level F1 0.494–0.787; span-level F1 0.128–0.527 | F1 family only | **2 WIN, 2 TIE, 1 LOSS** on F1; span = negative result |
| SHAM-Bench [8] | span- and sentence-level F1 on RAGTruth partitions | No | **Not comparable** |
| Semantic Illusion [18] | qualitative score distributions only | No | **Not comparable** |
| CRG [18], INSIDE [6], Decoding Memories [17], AutoHall [3] | — | No | **Do not report a comparable RAGTruth detection number** |

**Table 2.6 — Every paper that reports a comparable number on HaluEval or TruthfulQA**

| Paper | Benchmark | n evaluated | Metric | Comparable to ours? |
|---|---|---|---|---|
| Embedding & Gradient Say Wrong [58] | HaluEval | 35,000 (90% test) | accuracy | **No** — metric and size both differ |
| Decoding Memories [17] | HaluEval | 400 (QA subset, own generations, GPT-4o labels) | AUROC 61.2–84.9 for its baselines | **Closest** — but its labels are model-generated, not HaluEval's own |
| MedICO [59] | HaluEval | 1,000 triplets | F1 | **No** |
| InternalInspector I2 [60] | HaluEval | 30,000 (70% test) | accuracy | **No** |
| Markov Chain Multi-Agent Debate [61] | HaluEval | 350 | accuracy / F1 | **No** |
| Small Agent Can Also Rock! [62] | HaluEval | 100 test | accuracy / F1 | **No** |
| Teaching with Lies / HaluCheck [20] | HaluEval | 8,000 train + 2,000 test | F1 / P / accuracy | **No** |
| **This project** | TruthfulQA | 1,000 rows (500 unique queries, closed-book) | ROC-AUC **0.6355** CI [0.6107, 0.6594] | **Nothing to compare against** — no paper reports TruthfulQA detection with a comparable metric |

**Table 2.7 — Papers that are often cited as HaluEval evaluations but are not**

| Paper | Why it is wrongly suspected |
|---|---|
| Perturbation-Based Synthetic Data Generation [63] | HaluEval is **training data for a baseline**, not an eval set; evaluation is on OpenDialKG-Eval (312 test / 90 dev) and BEGIN |
| Factcheck-Bench [64] | HaluEval appears only in a related-work dataset comparison table. **0 examples evaluated** |
| GPTLODS [65] | No HaluEval anywhere in the paper; it evaluates on 1,000 manually-labelled ChatGPT facts about Greek persons |
| HaluCheck (ESWA system paper) [66] | **Paywalled; sample size and metrics could not be verified.** No number is attributed to it here |
| QuaLLM SLR [19] | A secondary source, not an evaluation; its HaluEval citation list over-claims on Factcheck-Bench and GPTLODS |

### 2.6.1 What the benchmark sweep establishes

**On RAGTruth this project loses to the three newest papers.** AES [9] reaches
AUROC 0.8807–0.9332 across Llama-2-7B/13B/70B, Llama-3-8B and Mistral-7B;
Frequency-Aware Attention [56] reaches 0.7877–0.8812 on its pooled span-level
table; the desiderata paper's supervised detector reaches ROC-AUC 0.871. This
project's 0.6127 with CI [0.5494, 0.6581] sits below all of them. Two
qualifications travel with that statement and are stated here rather than in a
footnote: AES and the frequency-aware method report at **span or token**
granularity and aggregate, whereas this project's number is at claim and
response level; and both use the extended 2,700-example RAGTruth test set
while this project uses 1,000 length-matched rows over 277 unique queries.
Neither qualification makes the loss disappear.

**The honest comparison against LLM-Check is closer than the loss above
suggests.** LLM-Check [5] publishes five response-level AUROC anchors on
RAGTruth read from its own Table 4. Against its weakest three — Logit Window
Entropy 0.5458, Perplexity 0.5327 and Log Entropy 0.5168 — this project's
lower bound of 0.5494 clears them all, giving three wins. Against Attention
Score 0.5830 and Hidden Score 0.5724 the intervals overlap, giving two ties.
No losses.

**The one place this project leads on a published anchor is the metric family
the desiderata paper's own table is weakest in.** On detection F1, this
project's 0.6076 with CI [0.5383, 0.6632] beats that paper's two SelfCheckGPT
rows (0.342 and 0.346) and its prompt-optimised detector (0.538, cleared by
the lower bound), ties few-shot (0.628) and LLM-as-a-Judge (0.617), and loses
to the supervised fine-tuned detector (0.671). The same paper is lost to on
ROC-AUC. The two metrics disagree because its ROC-AUC pools a different label
aggregation than the one this project uses, which is itself a finding about
how unstable response-level detection numbers are under label choice.

**The RAGTruth headline is in-sample and the held-out figure is worse.** The
detection head was fitted on 496 rows of the RAGTruth split, so 0.6127 is an
in-sample estimate. The held-out figure is **0.6185, CI [0.5371, 0.6943]**.
On the held-out figure alone the LLM-Check comparison becomes **two wins, three
ties and no losses** — 0.5371 still clears Perplexity (0.5327) and Log Entropy
(0.5168), but it sits below Window Entropy (0.5458), Hidden Score (0.5724) and
Attention Score (0.5830), so those three turn from wins into ties.
Both numbers are reported; the held-out one is the one that generalises.

**On HaluEval this project is not beaten by anyone on a comparable metric,
because almost nobody reports a comparable metric.** None of the external
HaluEval papers reports AUROC except Decoding Memories [17], whose 400
examples are its own regenerations labelled by GPT-4o rather than HaluEval's
own annotations. The rest report accuracy or F1 on sets ranging from 100 to
35,000 examples. On sample size this project sits in the lower-middle: 1,000
rows over 500 unique queries, against MedICO's 1,000 and Small Agent's 100
test examples at the low end and Hu et al.'s and InternalInspector's use of
the entire release at the high end. **Accuracy on a 100-example test set is
not the same claim as accuracy on 35,000**, and this project's number is not
the largest.

**On TruthfulQA there is nothing to compare against.** No paper in the
sweep reports TruthfulQA detection with a comparable metric, so this
project's ROC-AUC of 0.6355 with CI [0.6107, 0.6594] is a third independent
measurement of the same system rather than a win or a loss. Its value is
that it isolates the contribution of the evidence. On HaluEval the detector
has retrieved documents to check claims against; on TruthfulQA there is no
corpus, the evidence-dependent components are dropped, and the score falls
from 0.9251 to 0.6355. That gap is the size of the contribution the retrieved
evidence makes, and it would not be visible if TruthfulQA were not measured.
The one caveat is a train/serve mismatch: the detection head was fitted on
features available with retrieval, so some of its learned weights are less
useful when retrieval is switched off.

**One published negative result bears directly on this project's own
findings.** Sinha [18] reports that embedding-based and NLI detectors reach a
**100% false-positive rate at 95% target coverage on HaluEval**, despite an
NLI AUC of 0.81. This project's independent finding that the TruthfulQA gap
is a feature-quality limit rather than a bug — 0.80 in training against 0.60
at serving — is the same phenomenon reported by a different group. That is
citable support, not a contradiction, and Section 2.7.1 takes it up in full
because it is also the strongest published statement of the guarantee this
project does *not* provide.

---

## 2.7 Systems that are adjacent but not similar

The survey in Section 2.2 covers pipelines that respond to hallucination the
way this project does. A further set of published systems is often placed
alongside this project in reviews and defences, but differs in kind: they
either do not use retrieval at all, or they judge the answer without evidence
in front of them, or they measure a different task. They are named here so
that the comparison is complete, and the differences are stated plainly.

**Table 2.8 — Architectural differences against adjacent systems**

| Dimension | This project | Dale et al. [1] | SHROOM systems [7] | INSIDE [6] / LLM-Check [5] | CRG [18] |
|---|---|---|---|---|---|
| **Type** | End-to-end RAG, 13 steps | Detect-then-rewrite | Linear classifier / meta-regressor | Sampling or hidden-state probe | Conformal guardrail |
| **Stages** | 13 | 1 detection + 1 rerank | 1 | 1–3 | 3 signals + 1 judge |
| **Retrieval** | Hybrid BGE-M3 + FAISS + BM25 + cross-encoder | None | None | None | Required |
| **Verification** | NLI **per claim** | NLI on whole sentence | Cosine / alignment | Attention / entropy | Cosine (RAD) + NLI (SEC) + lexical (TFG) |
| **Mitigation** | Evidence-guided rewrite + ABSTAIN | N-best reranking (MC-dropout) | None | None | Risk-controlled reject |
| **Calibration** | Temperature scaling + Platt | None | Linear regression | None | Split conformal (1 − α) |
| **Access needed** | API or local HF | White-box or black-box | Black-box | White-box | Black-box |
| **Granularity** | Atomic claim | Sentence | Sentence | Whole answer | Whole answer |
| **Provenance** | `doc_id` per claim | None | None | None | None |

Three structural differences follow from this table, and they are the ones
that survive scrutiny.

**Only this system checks the answer against retrieved evidence.** Every
system to the right of Dale et al. judges a piece of text with no evidence in
front of it, or with evidence used only to score relevance. This project
retrieves documents and then checks each claim against what those documents
actually say.

**Only this system emits citations.** Every claim carries the `doc_id` of the
document that supported it. No system in the table does this, and it is what
makes a downstream human audit possible at all.

**Only this system and CRG act on the answer when it is unsure, and they do
different things.** CRG rejects answers it is not sure about; it does not
rewrite them. This project rewrites the unsupported spans and abstains only
when the evidence does not support a rewrite. The two are therefore
complementary rather than competing, and no rate advantage is claimed between
them because none has been measured.

**The two systems that reject also disagree on what they reject.** CRG
calibrates a single threshold on a scalar ensemble score and treats the whole
response as one unit, so its guarantee is a coverage guarantee on the
response. This project scores each claim separately and fuses seven signals
into a claim-level risk, so a response containing one bad claim is not
condemned wholesale — it is repaired at that claim and retained elsewhere. A
reviewer who asks which of the two is safer is asking about two different
granularities, and this project does not claim the better answer.

**Dale et al.'s mitigation number is on a different task.** Dale et al.
report a reranking delta of +0.49 to +0.60 on machine translation; this
project's rewrite produces **−0.007, CI [−0.022, +0.007]**, a paired
permutation test on the 16 rewritten rows giving **p = 0.404**. The two
deltas are on different axes, so the +0.60 is a reference point rather than a
target that was missed. This project reports the null as a loss and an open
result.

### 2.7.1 CRG — the closest work in calibration, and the most important result in this section

Sinha's Conformal RAG Guardrails [18] is the only published system anywhere
in the literature reviewed by this chapter — the nineteen-paper architectural
survey of Section 2.2 and the adjacent systems of Section 2.7 together — that
gives a **finite-sample guarantee on hallucination detection** rather than a
bootstrap estimate. It is therefore the single most important paper to read
alongside this project's own statistical claims.

CRG applies split conformal prediction to RAG hallucination detection. It
builds three nonconformity scores — **RAD**, the cosine distance between the
response embedding and the nearest retrieved context sentence using
BGE-base-en-v1.5; **SEC**, the NLI entailment score from
DeBERTa-v3-large-MNLI, defined as 1 − P(entail); and **TFG**, the percentage
of response content tokens with low lexical overlap against the retrieved
document — combines them either by simple average or by a logistic combiner
fitted on a held-out 200-example set, and then calibrates a single threshold
on a calibration split to obtain a coverage guarantee of at least 1 − α with
probability 1 − δ. A response falling below the threshold is rejected. The
paper adds an LLM-judge variant, GPT-4o-mini, as a reasoning-based baseline.

**CRG's headline result is a negative one, and it is the most important
finding in this section.** On synthetic hallucinations from Natural Questions
the method works: 595 calibration examples, 95% target coverage, 95.8%
actual coverage, **0.0% false-positive rate** (95% CI [0.0%, 1.0%]). On real
hallucinations from HaluEval (629 calibration examples) the same method
collapses: 94.5% actual coverage at **100.0% false-positive rate** (95% CI
[99.0%, 100%]). The paper's diagnosis is that the hardest hallucinations are
semantically indistinguishable from faithful responses, so the conformal
threshold is pushed low enough to reject nearly everything.

**This project sits on the wrong side of that finding, and the reason is
measurable.** This project's HaluEval result is ROC-AUC 0.9251 with CI
[0.9113, 0.9401] on 1,000 rows over 500 unique queries — well separated
classes, where a threshold exists that separates them. CRG's own NLI
component scores AUC 0.81 on HaluEval and still reaches a 100% false-positive
rate at 95% coverage. The difference between the two is that this project
performs its retrieval and its verification *per claim* against
claim-conditioned evidence, whereas CRG's RAD compares the whole response to
its nearest single context sentence. The paper's own ablation shows that
*learned weighting cannot fix this*: the logistic combiner reaches AUC 0.83
against the simple average's 0.81 and still has a 100% conformal false-positive
rate, because the failure is in the features and not in the aggregation.

**Three things follow, and the third is the most important.**

First, this project's own NLI verifier is subject to the same critique. The
finding in Chapter 6 that the rewrite produces a null result, and that the
cause is the NLI model's false-positive rate, is the same phenomenon CRG
identifies — the semantic features do not separate the tail. The two papers
reach the same conclusion from opposite directions, and this project should
cite CRG when it states that conclusion rather than presenting it as a
novel diagnosis.

Second, CRG's LLM-judge variant is the only published result in this survey
that breaks the barrier: GPT-4o-mini reaches AUC 0.94 with a 7.0%
false-positive rate at 95% coverage, and GPT-4 reaches 5.0% — at 150× the
latency (1,800 ms against 12 ms per sample) and 150× the cost ($15.00 against
$0.10 per thousand). That trade-off is a genuine and well-quantified result,
and it is the clearest published statement of the cost-safety frontier that
this project's own results sit on.

Third, and least comfortable: **the guarantee CRG provides is the guarantee
this project does not.** CRG's threshold comes with a coverage guarantee
under exchangeability. This project's risk score comes with a bootstrap
confidence interval on the *detection metric*, which estimates sampling
variability but is not a distributional guarantee. A reviewer who asks "what
is the probability that your gate rejects a correct answer?" has a question
this project cannot yet answer, and Section 2.11 states this as a threat
rather than an open question.

This project is the only system in the survey that reports a bootstrap
confidence interval for every headline metric, on three datasets, with the
resampling performed over unique questions rather than rows. It is also
the largest evaluation among the systems that report uncertainty: 500 unique
questions against ARES's 300. The two systems larger than this project that
report comparable metrics — Semantic Uncertainty and FActScore — publish no
interval, so their additional size buys a figure that cannot be checked.

The consequence is worth stating plainly, because it shapes the results
chapter. A pipeline whose mitigation stage produces no measurable benefit
can only make that statement credibly if the pipeline is capable of
measuring it. The self-re-verification loop described in Section 2.8 is
what makes a null result reportable, and the null result reported in
Chapter 6 is therefore more informative than a positive result reported by
a system that does not check its own output.

---

## 2.8 The three properties that distinguish this project

The survey found no published paper combining all six properties. Setting
aside the combination itself, three properties are individually absent from
the literature, and each is a structural property of the pipeline rather
than a quantitative advantage.

### 2.8.1 Repair that verifies its own output

The repair stage of this project rewrites an unsupported claim, and then
re-runs the verification, detection, fusion and gate stages on the result. A
rewrite that introduces a new error is caught and reverted.

No surveyed paper does this. FAVA emits a tagged correction and does not
check it. A Stitch in Time issues an instruction to repair a sentence and
does not check it. Chain-of-Verification conditions the final answer on the
results of verification questions, but replaces the answer as a whole and
does not itself constitute a repair loop.

The significance is methodological rather than quantitative. A system that
does not verify its own repair cannot detect whether the repair degraded
an answer that was previously correct, and therefore cannot report a null
result honestly. It can only report that its edits improved its score on a
metric that its own edits define. The distinction is visible in Table 2.5:
the systems that report large repair gains — A Stitch in Time at a
33-percentage-point reduction in hallucination rate, Chain-of-Verification
at +23% F1 — report no uncertainty and perform no post-repair check. This
project's repair produces a change of −0.007 with a confidence interval of
[−0.022, +0.007] and a p-value of 0.404, which is to say no measurable
effect. That null is trustworthy precisely because the mechanism exists to
produce it.

### 2.8.2 A gate placed after verification

The final gate of this pipeline scores the risk of the answer and refuses to
return it when the risk is too high. The gate is placed after the
verification stage, and it therefore has access to information that
pre-generation gates structurally do not.

Sufficient Context [46] is the closest published precedent and is a
necessary contrast. It gates on whether the retrieved context was
sufficient, before the answer exists. This project gates on whether the
answer is risky, after it exists and after every claim has been checked
against retrieved evidence. The two gates fail in different places. A
system can have sufficient evidence and still produce a wrong answer, which
a sufficiency gate passes and only a post-verification risk gate catches.
It can also have insufficient evidence while producing a correct answer,
which a sufficiency gate discards and a risk gate retains. CRAG [44] gates
on a retrieval score, which is a third position again: it is a proxy for
evidence quality rather than a judgement of the answer.

The fusion signal is also richer. Sufficient Context's gate combines two
features; this project's combines seven, and a well-retrieved passage does
not by itself earn trust, since a claim supported by an irrelevant passage
and one supported by a relevant passage are treated differently.

### 2.8.3 A fusion head with explicit directions

The seven components of this project's fusion head are standardised before
being combined, and the direction of each is recorded in the configuration
rather than being inferred from the sign of its fitted coefficient. This is
not cosmetic. An earlier hand-set fusion in which the signs were wrong
scored a ROC-AUC of 0.360 — below chance — and the correction required
re-deriving the direction of every component from its own univariate
performance.

The fitted head is trained on 496 rows, split 248 tuning and 248 hold-out,
and achieves a hold-out ROC-AUC of 0.6185 with a confidence interval of
[0.5371, 0.6943]. That interval overlaps the performance of the strongest
single component, which is the honest reading of a head fitted on this
little data. The comparison with the literature is unfavourable on scale:
Self-RAG uses 145,619 instruction-output pairs, AlignScore 4.7 million
examples, FAVA 35,074 synthetic instances, and ARES a synthetic corpus whose
size it does not state. **The mechanism is the contribution; the training
scale is not, and no claim to the contrary is made in this thesis.**

---

## 2.9 What the survey suggests for this design

The survey is useful to a design only if it identifies changes. Table 2.9
lists the changes the literature motivates, with the component of the
pipeline each would change and whether it would alter the numbers reported
in Chapter 6.

**Table 2.9 — Changes motivated by the surveyed literature**

| Source | Change | Component affected | Would alter the reported numbers? |
|---|---|---|---|
| Hybrid Retrieval [48] | compute the retrieval fusion weights per query from a specificity score, rather than fixing them | retrieval | yes |
| AlignScore [43] | replace the general-purpose NLI model with an alignment model trained on 4.7M examples including fact verification | verifier | yes |
| Semantic Uncertainty [49], SelfCheckGPT [4] | implement real multi-sample self-consistency in place of the current proxy | calibration | yes |
| DRAGIN [54] | apply the query rewrite that is currently computed and discarded | query understanding | yes |
| A Stitch in Time [47] | draw the replacement text from the retrieved premise rather than from the model | repair | yes |
| Sufficient Context [46] | fit the gate thresholds on data rather than fixing them at 0.30 and 0.70 | gate | yes |
| ARES [51] | apply prediction-powered inference to the detection metric | evaluation | yes |
| CRG [18] | calibrate the abstention threshold by split conformal prediction on a held-out calibration split, giving a finite-sample coverage guarantee on the rejection | gate | yes |
| FAVA [39] | adopt the six-type error taxonomy for verdict labelling | verifier | partly |
| RAGChecker [38] | report relevant and irrelevant noise sensitivity separately | evaluation | no, additional metric |
| FActScore [40] | exclude abstentions from the model-level average | evaluation | no, reporting change |

Two of these are already reflected in the present design: the abstain
handling follows FActScore, and the noise-sensitivity distinction between
relevant and irrelevant retrieved passages is already implemented as a
fusion feature but is not yet reported as a separate metric.

The most consequential of the outstanding changes is the replacement of the
verifier. The current NLI model's false-positive rate is the mechanism that
causes correct claims to be rewritten, which is the direct cause of the
null mitigation result in Chapter 6. This is a component-precision problem
rather than a design problem, and it is the change most likely to convert
the repair stage from a null into a measurable effect.

---

## 2.10 Positioning

The literature reviewed in Chapter 2 is organised by detector; the
literature reviewed here is organised by pipeline. Read together, they
support a specific and deliberately narrow claim.

This project does not claim to detect hallucination better than the
literature. On RAGTruth it attains a lower value than three more recent
published methods, as Chapter 7 reports, and the 496-row fusion head is
trained on two orders of magnitude less data than the nearest precedents.
The claim this chapter supports is architectural and methodological: of
nineteen published systems with a comparable pipeline shape, **none combines
claim-level verification against retrieved evidence, a separately fitted
multi-signal fusion head, repair at claim granularity, post-repair
re-verification, and answer-side abstention**; and of those nineteen, **only
two report any quantification of the uncertainty in their results**.

The first claim is a statement about the design space, and it is falsifiable
by a paper that does all six things. The second is a statement about
reporting practice, and it is falsifiable by a number in a table. Neither
claim asserts that this pipeline outperforms the literature, and Chapter 7
shows that on the metrics where a comparison is possible, it does not.

---

## 2.11 Threats to the claim

Four objections would undermine the argument of this chapter, and each is
addressed here rather than left to a reader.

**The survey may be incomplete.** Nineteen papers were found by searching
for systems with a comparable pipeline shape. A system implementing all six
properties may exist and not have been found. The claim is therefore stated
in the form "of the papers found in this survey", and the search procedure
— architecture first, then the specific combination — is described so that a
reader can repeat it.

**The combination claim may be trivial.** A critic may argue that any
sufficiently elaborate system can be made to implement all six properties,
and that the finding therefore reflects the difficulty of assembling
components rather than a scientific question. This objection has force
against the survey but less force against the specific design decisions in
Section 2.8, where each property is described in terms of where it sits in
the pipeline rather than in terms of being present at all. The gate's
*position* after verification, and the repair stage's *re-entering* the
verification loop, are design choices with consequences for what the system
can detect and report, not properties that could be satisfied by adding a
module.

**The statistical claim is the weaker of the two, and CRG [18] is the
threat to it.** This chapter argues that this project reports uncertainty
where the literature does not. CRG does not merely report uncertainty; it
provides a **finite-sample coverage guarantee** under exchangeability, and
it does so on the very task this project claims to perform. The honest
distinction is narrow and must not be overstated: a bootstrap confidence
interval on the detection metric estimates sampling variability, while a
conformal threshold bounds the probability of rejecting a correct answer.
Those are different guarantees, and the second is the stronger one. A
reviewer who knows CRG will ask why this project does not calibrate its gate
the same way. The answer given here is that a conformal guarantee would
require the calibration and test distributions to be exchangeable, and this
project's calibration set is drawn from the same length-matched split as its
test set, which makes such a guarantee a statement about the split rather
than about the method. That is a limitation of the current design, not a
defence of it, and it is the first item on the list of changes in
Table 2.9.

**The reporting claim may be an artefact of recency.** If confidence
intervals have become standard since the surveyed papers were published,
then the comparison reflects the date of the literature rather than the
quality of the systems. Two observations bear on this. ARES, which
appeared in the same period as several of the papers that report nothing,
does report intervals, so availability is not purely chronological. And the
claim is not that the older papers were negligent; it is that the field
does not currently make a like-for-like comparison possible, which is the
condition Chapter 7 has to work under.

---

## References

Reference numbers continue the sequence of Chapter 10 of the thesis. Number
[4], SelfCheckGPT, is already listed there and is not repeated.

| # | Reference |
|---|---|
| [38] | Ru et al., "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation," arXiv preprint arXiv:2408.08067, 2024. |
| [39] | Mishra, Asai, Balachandran, Wang, Neubig, Tsvetkov and Hajishirzi, "Fine-grained Hallucination Detection and Editing for Language Models," COLM 2024; arXiv preprint arXiv:2401.06855. |
| [40] | Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," EMNLP 2023; arXiv preprint arXiv:2305.14251. |
| [41] | Es, Espinosa-Anke and Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," arXiv preprint arXiv:2309.15217, 2023. |
| [42] | Wei et al., "LongFact: Long-Form Factuality in Large Language Models," NeurIPS 2024; arXiv preprint arXiv:2403.18802. |
| [43] | Zha, Yang, Li and Hu, "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function," ACL 2023; arXiv preprint arXiv:2305.16739. |
| [44] | Yan, Gu, Zhu and Ling, "Corrective Retrieval Augmented Generation," arXiv preprint arXiv:2401.15884, 2024. |
| [45] | Asai, Wu, Wang, Sil and Hajishirzi, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," ICLR 2024; arXiv preprint arXiv:2310.11511. |
| [46] | Joren, Zhang, Ferng, Juan, Taly and Rashtchian, "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems," arXiv preprint arXiv:2411.06037, 2024. |
| [47] | Varshney, Yao, Zhang, Chen and Yu, "A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation," arXiv preprint arXiv:2307.03987, 2023. |
| [48] | Mala, Gezici and Giannotti, "Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis," arXiv preprint arXiv:2504.05324, 2025. |
| [49] | Kuhn, Gal and Farquhar, "Semantic Uncertainty: Linguistic Invariance for Uncertainty Estimation in Natural Language Generation," ICLR 2023; arXiv preprint arXiv:2302.09664. |
| [50] | Feng, Shi, Wang, Ding, Balachandran and Tsvetkov, "Don't Hallucinate, Abstain: Multi-LLM Collaboration for Hallucination Detection," ACL 2024; arXiv preprint arXiv:2402.00367. |
| [51] | Saad-Falcon, Khattab, Potts and Zaharia, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems," NAACL 2024; arXiv preprint arXiv:2311.09476. |
| [52] | Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models," arXiv preprint arXiv:2309.11495, 2023. |
| [53] | Jiang et al., "Active Retrieval Augmented Generation," EMNLP 2023; arXiv preprint arXiv:2305.06983. |
| [54] | Su et al., "Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models," ACL 2024; arXiv preprint arXiv:2403.10081. |
| [55] | Trivedi et al., "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions," ACL 2023; arXiv preprint arXiv:2212.10509. |
| [56] | Qi, Chen, Zhao, Zhu, Hu, Liu, He, Yuan and Gui, "Detecting Contextual Hallucinations in LLMs with Frequency-Aware Attention," arXiv preprint arXiv:2602.18145, 2026. |
| [57] | Chen, Padmanabhan, Giyahchi, Wong and Akoglu, "Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights," arXiv preprint arXiv:2605.11330, 2026. |
| [58] | Hu, Zhang, Peng, Zhang, Wu, Chen and Zhao, "Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection," EMNLP 2024, pp. 1950–1959. |
| [59] | Zhao et al., "Medico: Towards Hallucination Detection and Correction with Multi-source Evidence Fusion," EMNLP 2024 System Demonstrations, pp. 34–45. |
| [60] | Beigi et al., "InternalInspector: Robust Confidence Estimation in LLMs through Internal States," Findings of EMNLP 2024, pp. 12847–12865. |
| [61] | Sun et al., "Towards Detecting LLMs Hallucination via Markov Chain-based Multi-agent Debate Framework," ICASSP 2025; arXiv preprint arXiv:2406.03075. |
| [62] | Cheng et al., "Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector," EMNLP 2024, pp. 14600–14615. |
| [63] | Zhang et al., "Enhancing Hallucination Detection through Perturbation-Based Synthetic Data Generation in System Responses," Findings of ACL 2024, pp. 13321–13332. |
| [64] | Wang et al., "Factcheck-Bench: Fine-Grained Evaluation Benchmark for Automatic Fact-checkers," Findings of EMNLP 2024, pp. 14199–14230. |
| [65] | Mountantonakis and Tzitzikas, "Real-Time Validation of ChatGPT Facts using RDF Knowledge Graphs (GPTLODS)," ISWC 2023, CEUR Vol-3632, paper 408. |
| [66] | Heo, Son and Park, "HaluCheck: Explainable and verifiable automation for detecting hallucinations in LLM responses," Expert Systems with Applications, vol. 272, 126712, 2025. |

---

## Source and provenance

This chapter is derived from three repository documents. The working
survey is `docs/related_pipeline_workflows.md`, which holds the per-paper
workflow descriptions, the full statistical record, the measured stage-coverage
counts, and the seven pre-answered reviewer objections from which the
arguments here are drawn. The benchmark sweep in Section 2.6 is derived from
`Paper/ragtruth_comparison.md`, which holds every published paper that
evaluates on RAGTruth with the verdicts computed against this project's
confidence intervals, and from `Paper/halueval_papers.md`, which holds every
paper that evaluates on HaluEval with the number of examples each actually
ran. The adjacent-systems comparison in Section 2.7 is derived from
`Paper/comparison.md`.

Two corrections were made to `Paper/comparison.md` while preparing Section
2.7, and both are reflected here. The system recorded there as
"Conformal-RAG" is Sinha's Conformal RAG Guardrails [18], the paper already
listed as [18] in the thesis reference list; there is no paper titled
"Conformal-RAG". Its three signals are named in the source as RAD (cosine
distance to the nearest retrieved context sentence, using BGE-base-en-v1.5),
SEC (NLI entailment from DeBERTa-v3-large-MNLI, nonconformity 1 − P(entail))
and TFG (share of response content tokens with low lexical overlap against
the retrieved document), combined by simple average or by a logistic combiner
fitted on 200 held-out examples, and calibrated by split conformal
prediction at level 1 − α. Separately, the "1 − α" conformal risk-control
system sometimes conflated with it — C-RAG [Kang et al., ICML 2024,
arXiv:2402.03181] — is a different paper: it certifies a generation
configuration against a ROUGE-L risk on AESLC, CommonGen, DART and E2E with
Llama-7B, uses no NLI signal and no judge model, and evaluates on neither
RAGTruth nor HaluEval. It is therefore not comparable to this project and is
not listed among this project's related work.

Every pipeline description, dataset name, sample size and reported figure
was read from the paper's own arXiv full text rather than from its abstract,
from a secondary summary, or from memory. Each paper's full text was
searched for *confidence interval*, *error bar*, *standard deviation*,
*significan\**, *bootstrap*, *random seed* and *variance*, and a cell reads
"not reported" where none of these appear. The numbers attributed to this
project were read from the repository artefacts `data/metrics_halueval.json`,
`data/metrics_truthfulqa_closedbook.json`, `data/metrics_ragtruth_fitted.json`,
`data/detection_head_refit600.json` and `data/mitigation_results.json`.

Three cautions carry over from the working survey and are repeated here
because they bear on how the tables should be read. SAFE's 496 is a
human-labelled *evaluation* set and the 496 rows on which this project's
fusion head is fitted are a *fitting* set; the two quantities are unrelated
and coincide only in value. The evaluation-size figures for the surveyed
papers are not mutually comparable, and Section 2.5 states which of them are
doing what. One further caution belongs with the new Section 2.6: CRG's
coverage guarantee holds under exchangeability between the calibration and
test distributions, and its own HaluEval false-positive rate of 100% at 95%
coverage is the published demonstration of what happens when that assumption
does not hold in practice. And the metric vocabularies of the surveyed papers are disjoint,
which is why no single score column is presented for them and why the
comparisons of record are those of Chapter 7.
