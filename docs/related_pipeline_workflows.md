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
> two thinnest categories are **C4 (learned multi-signal fusion)** — Self-RAG
> and ARES are the clearest precedents — and **C5 (post-hoc span rewriting)**,
> where only FAVA and "A Stitch in Time Saves Nine" qualify. CRISP's
> contribution is the *combination*: hybrid retrieval with reranking →
> NLI claim verification → a separately fitted multi-signal fusion head →
> span-level rewriting → answer-side abstention, in one pipeline.

---

# 3. The papers in detail

---

## 3.1 RAGChecker — the closest in *diagnosis*

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

---

## 3.2 FAVA — the closest in *detection and editing*

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

---

## 3.3 FActScore — the closest in *claim decomposition and verification*

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

---

## 3.4 RAGAS — the closest in *the claim-extraction idiom*

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

---

## 3.5 SAFE — the closest in *long-form verification*

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

---

## 3.6 AlignScore — the strongest drop-in for CRISP's step 8

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

---

## 3.7 Corrective RAG (CRAG) — the closest in *confidence-gated retrieval*

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

---

## 3.8 Self-RAG — the closest in *fusion*

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

---

## 3.9 Sufficient Context — the closest in *abstention*

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

---

## 3.10 A Stitch in Time Saves Nine — the closest in *mitigation*

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

---

## 3.11 Hybrid Retrieval for Hallucination Mitigation — the closest in *retrieval*

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

---

## 3.12 Semantic Uncertainty — the closest in *entropy and abstention*

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

---

## 3.13 SelfCheckGPT — the closest in *sampling-based consistency*

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

---

## 3.14 ARES — the methodological precedent for CRISP's fitted head

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

---

## 3.15 Don't Hallucinate, Abstain — multi-model abstention

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

---

## 3.16 The retrieval-side family (no verification stage)

These match CRISP's steps 1, 6 and 7 but have **no detection, fusion,
mitigation, or abstention**.

| Paper | Workflow | Relevance to CRISP |
|---|---|---|
| **FLARE** (Jiang et al., EMNLP 2023, [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)) | 1. Predict the next sentence → 2. Detect low-confidence tokens → 3. Use the predicted sentence as the retrieval query → 4. Retrieve and revise the sentence | Parallels CRISP's confidence-gated retrieval but at token level *during* generation. CRISP verifies *after* generation. |
| **DRAGIN** (Su et al., ACL 2024, [arXiv:2403.10081](https://arxiv.org/abs/2403.10081)) | 1. Awareness: does the generator lack the info to continue? → 2. Represent the need from the *full* context → 3. Active retrieval from that need → 4. Feed back, forming a need↔retrieval loop | CRISP's step 1 is the analogue of DRAGIN's information-need module. **DRAGIN's design is the answer to CRISP's known weakness** — the query rewrite that is computed and then discarded (§3.17). |
| **IRCoT** (Trivedi et al., ACL 2023, [arXiv:2212.10509](https://arxiv.org/abs/2212.10509)) | 1. Sample a sentence from the current chain-of-thought → 2. Retrieve with it → 3. Retrieve again with the original question → 4. Append to the CoT → repeat | The classic precedent for reasoning-conditioned retrieval — CRISP's steps 1–3 at reasoning-step rather than query level. |
| **CoVe** (Dhuliawala et al., [arXiv:2309.11495](https://arxiv.org/abs/2309.11495)) | 1. Draft an answer → 2. Plan verification questions targeting the draft → 3. Answer each independently → 4. Produce a final answer conditioned on the results | CRISP's mitigation stage descends from this. **Caveat: the abstract does not state that CoVe retrieves external sources** — do not claim source-grounded verification for it. |

---

# 4. What this survey says about CRISP's position

## 4.1 The genuine gap

No paper found combines all six characteristics. The two sparse categories:

- **C4, learned multi-signal fusion** — Self-RAG (token-probability fusion
  inside the generator) and ARES (prediction-powered inference) are the
  precedents. Nothing does what CRISP's step 9 does: a *separately fitted
  logistic head over seven heterogeneous, standardised, correctly-signed
  features*.
- **C5, post-hoc span rewriting** — only FAVA (learned tagger) and "A Stitch
  in Time" (instruction-based repair) qualify. Neither verifies its own
  output afterwards, which CRISP's regeneration loop does.

**CRISP's claim should be framed as the combination**, not as any single
component.

## 4.2 Concrete improvements these papers suggest

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

## 4.3 The strongest counter-arguments to anticipate

These are the objections a reviewer is most likely to raise, and the papers
that support them.

1. **"Better retrieval alone fixes this."** Mala et al. (§3.11) show hybrid
   retrieval beating sparse and dense on all four overall metrics across
   six datasets. CRISP must argue that detection-layer work is *additional*,
   not substitutable.

2. **"Uncertainty is not error."** The NLI's `unknown` verdict drives
   abstention, so the pipeline refuses good answers. Sufficient Context
   (§3.9) demonstrates the same tension from the retrieval side.

3. **"Your EEDC weights were never fitted."** `data/eedc_weights.json` does
   not exist; the defaults are hand-set. The script to fix this already
   exists.

4. **"Your step-3 detector is at chance."** AUC 0.5389 as a risk signal.
   It survives only as one weighted component of seven.

5. **"Your mitigation delta is negative."** The `claim_edit` rewrite is a
   design fix whose measured benefit is ≈ −0.007. FAVA (§3.2) and "A Stitch
   in Time" (§3.10) both report large, positive mitigation results — so the
   question "why does yours not work?" is live.

6. **"RAGChecker already does the decomposition."** It separates retrieval
   from generation failure with a richer metric set (§3.1).

## 4.4 Papers that must not be miscounted

Two categories to be careful about in any related-work count:

- **Datasets are not methods.** RAGTruth and HaluEval are benchmarks, not
  pipelines. Listing them as "comparable systems" would be wrong.
- **A related-work citation is not an evaluation.** A paper that merely cites
  HaluEval, or says it *will* evaluate on it, does not belong in a
  head-to-head table. `Paper/halueval_papers.md` already applies this rule
  and finds that only 2 of 12 local papers actually ran HaluEval
  experiments.

---

# 5. Reference list

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
