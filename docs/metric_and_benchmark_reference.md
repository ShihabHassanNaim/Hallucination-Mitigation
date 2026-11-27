# Metric and Benchmark Reference — which paper reports which metric, and on which data

**Purpose.** This document answers two questions that keep coming up in
review and defence:

1. **Which metric does each paper use?** When someone asks "what number does
   this system report?", the answer is a specific metric, and metrics are not
   interchangeable.
2. **Which benchmark data does each paper use, and how much of it?** When
   someone asks "how big was that evaluation?", the answer is a number that
   only means something alongside the metric it was measured with.

Everything in this file was read from the papers' own tables or from this
project's own artefact files. Where a paper does not report a quantity, the
cell says so rather than leaving a blank.

**How this document is organised.**

| Section | Question it answers |
|---|---|
| §1 | What each metric *means*, and why two numbers on different metrics cannot be compared |
| §2 | Which paper reports which metric — the full map |
| §3 | The metric families, and which papers are inside each |
| §4 | Which papers are directly comparable to this project, and which are not |
| §5 | Benchmark data: the datasets themselves, side by side |
| §6 | Benchmark data: every paper's evaluation set, size, split and metric |
| §7 | The comparison tables for this project's three benchmarks |
| §8 | Rules for reading any comparison in this thesis |
| §9 | Source and provenance |

---

## 1. What each metric means

A metric is a decision about what counts as "good", compressed into one
number. Two papers can both be excellent and still report numbers that cannot
be set side by side, because they answered the question differently.

**Table 1.1 — The metric vocabulary, with what it actually measures**

| Metric | What it measures | Range | Higher is | Common trap |
|---|---|---|---|---|
| **ROC-AUC / AUROC** | The probability that a random hallucinated answer is scored above a random faithful one | 0–1 | better separation | 0.5 is a coin flip. A high value with **overlapping class distributions** still produces a bad threshold. |
| **AUPR / AUC-PR** | The same separation question, but weighting the positive class | 0–1 | better | Not comparable to ROC-AUC. A PR curve looks *worse* than a ROC curve on the same data. |
| **Accuracy** | Fraction of answers classified correctly | 0–1 | more correct | Rises with class balance. A model that always says "faithful" scores 50% on a balanced set. |
| **Precision** | Of the answers flagged as hallucinated, how many really were | 0–1 | fewer false alarms | Meaningless without a recall, because a detector that flags one answer perfectly has precision 1.0. |
| **Recall** | Of the answers that really were hallucinated, how many were flagged | 0–1 | fewer misses | Meaningless without a precision. |
| **F1** | Harmonic mean of precision and recall | 0–1 | better balance | **Threshold-dependent.** Two F1 values from different thresholds are not comparable. |
| **EM (exact match)** | Whether the generated answer exactly matches the gold answer | 0–1 | more correct | Measures *answer quality*, not *detection*. A system can answer perfectly and detect badly. |
| **FactScore / FActScore** | Estimated fraction of supported facts in a long answer | 0–100 | more supported | An *estimate* from a model, not a measured count. |
| **MAP@k** | Precision at each relevant rank, averaged | 0–1 | better ranking | A retrieval metric, not a detection metric. |
| **NDCG@k** | Rank quality with graded relevance | 0–1 | better ranking | Also a retrieval metric. |
| **MRR** | Mean reciprocal of the first relevant rank | 0–1 | better ranking | Also a retrieval metric. |
| **Pearson r / Spearman ρ** | Correlation between a score and human labels | −1 to 1 | better correlation | Measures agreement with a labelling, not accuracy of a decision. |
| **Hallucination rate** | Fraction of answers containing a hallucination | 0–1 (or %) | **lower is better** | **Inverted direction.** This is the one metric in this table where a *lower* number wins, and it is the easiest to misread in a comparison table. |
| **Coverage / FPR at target coverage** | Fraction of valid outputs a detector is forced to reject | 0–1 (or %) | **lower is better** | Also inverted, and it is the metric on which CRG fails on HaluEval. |
| **Claim precision / recall / F1** | Quality of the claim *decomposition* itself | 0–1 | better decomposition | Measures the extractor, not the detector. CRISP's claim F1 0.8107 and its detection F1 0.8202 are different quantities. |

**Table 1.2 — The three that must never be mixed**

| If one side reports… | It must not be compared with… | Why |
|---|---|---|
| a **rate** (hallucination rate, FPR, abstain rate) | a **score** (ROC-AUC, F1, accuracy) | The direction is opposite. "0.14" is good on a hallucination rate and bad on a ROC-AUC. |
| **EM / F1 on QA** (FLARE, DRAGIN, IRCoT) | any **detection** metric | These measure whether the *answer* is right, not whether the system can *detect* hallucination. A system that answers every question badly and flags all of them correctly would score zero on EM and could score high on detection. |
| a **retrieval** metric (MAP@k, NDCG, MRR) | any **detection** metric | Retrieval quality and detection quality are different subsystems. Hybrid Retrieval's MAP@3 0.897 is not a detection result. |

**Table 1.3 — The two directions, side by side**

| Metric | Where it appears | Direction |
|---|---|---|
| ROC-AUC, AUPR, accuracy, F1, precision, recall, EM, FactScore, MAP, NDCG, MRR, Pearson, Spearman | almost all papers | **higher is better** |
| hallucination rate, FPR at target coverage, cost, latency, abstain-accuracy loss | A Stitch in Time, CRG, Hybrid Retrieval, this project | **lower is better** |

---

## 2. Which paper reports which metric

This is the full map. Each row gives the paper, the metric it actually
reports, the value, and what that value refers to.

**Table 2.1 — Every paper and its headline metric**

| Paper | Metric reported | Headline value | What the number refers to | Eval n |
|---|---|---:|---|---|
| **This project** | ROC-AUC | **0.9251 / 0.6355 / 0.6127** | HaluEval / TruthfulQA / RAGTruth detection | 500 / 500 / 277 unique q |
| **This project** | AUPR | 0.9217 / 0.6450 / 0.6112 | same three datasets | 500 / 500 / 277 |
| **This project** | Detection F1 | 0.8202 / 0.6773 / 0.6076 | same three datasets | 500 / 500 / 277 |
| **This project** | Claim F1 | 0.8107 / 0.0 / 0.7269 | quality of the claim decomposition | 500 / 500 / 277 |
| **This project** | Hallucination rate (mitigation) | Δ **−0.007** | change from rewriting; CI [−0.022, +0.007], p = 0.404 | 16 rewritten rows |
| **RAGChecker** | Precision / Recall / F1 | 62.0 / 53.0 / 52.7 | best system, E5-Mistral + GPT-4 | 4,162 queries |
| **RAGChecker** | Pearson | 70.09 | human–human upper bound | 280 human pairs |
| **FAVA** | Detection gain | **+22.1%** | over ChatGPT (binary +14.2%) | 364 passages |
| **FAVA** | FActScore Δ | +4.4 to +9.3 | editing quality, by model | 300-response source bench |
| **FAVA** | Errors per response | 2.4 vs 1.9 | human eval, vs ChatGPT | 5 annotators |
| **FActScore** | FActScore | 42.5 / 58.3 / 71.5% | human set, by generator | 183 entities, 6,500 gens |
| **FActScore** | Pearson r | 0.99 | cross-estimator agreement | — |
| **RAGAS** | Human-agreement accuracy | 0.95 / 0.78 / 0.70 | faithfulness / answer rel. / context rel. | 50 instances |
| **SAFE** | SAFE–human agreement | 72.0% | agreement on LongFact | 250 main, 496 human |
| **SAFE** | F1@64 | 95.0 | fact-level score under a fact budget | 496 human pairs |
| **SAFE** | Pearson p | 2 p-values | significance of agreement | — |
| **AlignScore** | AUC-ROC | 87.4 / 88.6 | SummaC average, base / large | 22 datasets, n not stated |
| **CRAG** | Accuracy | 59.8 vs 52.8 | PopQA, CRAG vs plain RAG | 1,399 (PopQA) |
| **CRAG** | Accuracy | 84.3 vs 58.0 | retrieval evaluator vs ChatGPT | 1,399 |
| **Self-RAG** | Accuracy | 54.9 / 55.8 | PopQA, 7B / 13B | 1,399 / 11,313 |
| **Self-RAG** | Accuracy | 72.4 / 74.5 | PubHealth, 7B / 13B | as above |
| **Self-RAG** | FactScore | 81.2 / 80.2 | biography task | as above |
| **Sufficient Context** | Autorater F1 / accuracy | 0.935 / 0.930 | Gemini 1.5 Pro, 1-shot | 115 / 452 / 500 |
| **Sufficient Context** | Correct-answer gain | +2–10% | selective generation, over baseline | as above |
| **A Stitch in Time** | Detection recall | ~88% | fraction of hallucinations caught | 150 topics / 750 sents |
| **A Stitch in Time** | Hallucination rate | **47.4% → 14.53%** | before and after rewriting | as above |
| **A Stitch in Time** | Mitigation success | 57.6% | rewrite judged correct | as above |
| **Hybrid Retrieval** | MAP@3 | 0.897 | sparse 0.724 / dense 0.768 / hybrid 0.897 | 300 scored (13,867 pool) |
| **Hybrid Retrieval** | NDCG@3 | 0.915 | hybrid retrieval | as above |
| **Hybrid Retrieval** | Hallucination rate | 4–22% | per dataset | as above |
| **Semantic Uncertainty** | AUROC | 0.77 / 0.83 | CoQA / TriviaQA, semantic entropy | 8,000 × 2 |
| **SelfCheckGPT** | AUC-PR | 87.33 | passage level, combined | 238 passages, 1,908 sents |
| **SelfCheckGPT** | Pearson / Spearman | 69.05 / 67.77 | correlation with human labels | as above |
| **Don't Hallucinate** | Reliable accuracy gain | **+10.1%** | average over 12 settings | 200–1,094 |
| **Don't Hallucinate** | Abstain-accuracy gain | +19.3% | maximum over settings | as above |
| **ARES** | Accuracy | 79.3–92.3% | context/answer relevance, 6 datasets | 300 PPI, 707, 510 |
| **ARES** | PPI interval | 95%, mean width 7.4 / 6.1 pts | **the only true guarantee in the survey** | 300 |
| **CoVe** | F1 | 0.39 → 0.48 | MultiSpanQA, factored | 56 / 55 / 418 |
| **CoVe** | FactScore | 55.9 → 71.4 | biographies | as above |
| **FLARE** | EM / F1 | 51.0 / 59.7 | 2WikiMultihopQA answer quality | 500 / 229 / 500 / 500 |
| **DRAGIN** | EM / F1 | 0.304 / 0.3931 | 2Wiki, LLaMA-2-13B | 3,954 total |
| **IRCoT** | F1 | 59.1±0.9 / 66.5±1.4 | HotpotQA / 2Wiki | 500 × 4 datasets |
| **LLM-Check** | AUROC | 0.5168–0.5830 | 5 detection probes, RAGTruth | 450-instance test set |
| **AES** | AUROC | 0.8807–0.9332 | span-level, 5 backbones | 2,700 extended test set |
| **Frequency-Aware Attention** | AUROC | 0.7877–0.8812 | span-level pooled (RagTruth-Avg) | 2,700 extended test set |
| **Desiderata** | ROC-AUC / PR-AUC | 0.584–0.871 / 0.199–0.694 | 6 detectors, response-level | RAGTruth |
| **CRG** | Coverage / FPR@95% | 95.8% / 0.0% (NQ); 94.5% / **100.0%** (HaluEval) | conformal, at 1 − α | 595 / 629 cal |
| **CRG** | AUC | 0.72 / 0.81 / 0.65 / 0.83 / 0.94 | RAD / SEC / TFG / learned / GPT-4o-mini | 629 cal |

---

## 3. The metric families

Grouping the papers by the metric they report shows why a single "score"
column is impossible.

**Table 3.1 — Which papers belong to which metric family**

| Metric family | Papers in it | Comparable to this project? |
|---|---|---|
| **ROC-AUC / AUROC** | This project, Semantic Uncertainty, LLM-Check, AES, Frequency-Aware Attention, Desiderata, AlignScore, CRG (judge variant) | **Yes, for LLM-Check only.** The others use a different dataset or a different granularity — see §4. |
| **AUPR / AUC-PR** | This project, SelfCheckGPT | **No.** A PR curve is not a ROC curve. SelfCheckGPT's 87.33 is a passage-level PR score on WikiBio. |
| **Accuracy** | RAGAS, CRAG, Self-RAG, ARES, HaluEval papers (Hu, InternalInspector, Markov debate, Small Agent) | **No.** Thresholded, and mostly on different tasks. |
| **F1 (detection)** | This project, RAGChecker, Sufficient Context (autorater), Desiderata, RAGTruth paper, SHAM-Bench | **Partly.** Comparable only to Desiderata's and the RAGTruth paper's response-level F1 on RAGTruth. |
| **F1 (answer quality / QA)** | FLARE, DRAGIN, IRCoT, CoVe (partly), RAGAS | **No.** These measure answer correctness. |
| **FactScore / FActScore** | FActScore, CoVe, Self-RAG (biography), FAVA (as Δ) | **No.** FActScore is an *estimate* of supported-fact fraction, not an F1. |
| **Retrieval metrics** (MAP@k, NDCG@k, MRR) | Hybrid Retrieval, this project's own retrieval side | **No**, unless the question is about retrieval. |
| **Hallucination rate** | A Stitch in Time, Hybrid Retrieval, this project's mitigation Δ | **No** against a score, but **yes** against another rate. |
| **Conformal coverage / FPR** | CRG, ARES (PPI) | **Conceptually yes**, but the guarantees differ in kind — see §4. |
| **Correlation** (Pearson, Spearman) | SelfCheckGPT, FActScore, RAGChecker, this project | **No.** Agreement with a labelling, not accuracy of a decision. |

**The single clean cross-paper comparison.** ARES and RAGAS evaluate RAG
faithfulness on the same datasets with the same metric (accuracy), and ARES
reports **+59.3 and +14.4 points** over RAGAS. That is the only head-to-head
anywhere in this literature where both sides are genuinely comparable. There
is no equivalent for this project against any of the papers above, because
none of them releases a per-instance risk score on a shared dataset. The
comparisons this thesis can make are therefore the ones in §8, restricted to
papers that report the same metric on the same dataset.

---

## 4. Which papers are directly comparable to this project, and which are not

**Table 4.1 — Comparability, with the reason in each case**

| Paper | Comparable? | The reason |
|---|---|---|
| **LLM-Check** | **YES** | Same metric (response-level detection AUROC), same dataset (RAGTruth). Only the split differs. |
| **Desiderata** | **YES** | Same metric (response-level ROC-AUC and F1), same dataset (RAGTruth). |
| **RAGTruth paper** | **YES on F1 only** | Reports response-level F1 on RAGTruth. Its span-level F1 is a negative result for this project, which emits no spans. |
| **AES** | **NO** | Reports **span-level** AUROC, aggregated to response level. This project reports claim/response-level. Different granularity of evidence. |
| **Frequency-Aware Attention** | **NO** | Reports **token-level** (Table 1) and **span-level** (Table 2) AUROC. Same reason. |
| **SHAM-Bench** | **NO** | Span- and sentence-level F1 on its own task mix, on 2,700 examples. |
| **Semantic Illusion / CRG** | **NO** | Reports conformal coverage and FPR on Natural Questions and HaluEval (calibration sets), plus qualitative RAGTruth score distributions with no AUROC, F1 or accuracy. |
| **ARES** | **NO on score, YES on practice** | Reports context/answer relevance accuracy on KILT/SuperGLUE/AIS, not RAGTruth or HaluEval detection. It *is* comparable on **method**: it is the only paper with a statistical guarantee. |
| **All HaluEval papers** | **NO** | None reports AUROC on HaluEval except Decoding Memories, whose 400 examples are its own GPT-4o-labelled regenerations, not HaluEval's own annotations. The rest report accuracy or F1. |
| **FLARE, DRAGIN, IRCoT** | **NO** | EM / F1 on question answering — answer quality, not detection. |
| **Everything in the 19-paper survey** | **NO** | Disjoint metric vocabularies; see §3. |

---

## 5. The benchmark datasets, side by side

**Table 5.1 — The three datasets this project evaluates on**

| Property | HaluEval [11] | TruthfulQA [12] | RAGTruth [10] |
|---|---|---|---|
| **Task** | Question answering with retrieved evidence | Closed-book truthfulness | RAG hallucination detection (3 sub-tasks) |
| **Sub-tasks** | QA, Dialogue, Summarization, General | Single | QA, Data-to-text, Summarization |
| **Full size** | 35,000 | 817 questions | 2,675 instances |
| **This project's rows** | 1,000 | 1,000 | 1,000 |
| **Unique questions** | 500 | 500 | **277** |
| **Class balance** | 500 / 500 (paired) | 500 / 500 (paired) | 500 / 500 (paired) |
| **Design** | Paired: same question, one faithful + one hallucinated answer, length-matched | Paired on the same question | Paired **within an instance**, not by query |
| **Retrieval** | On — hybrid BGE-M3 + BM25 | **Off** (closed book) | On — hybrid BGE-M3 + BM25 |
| **Grounding source** | Retrieved documents | **None** — the answer's own features only | Retrieved documents |
| **Corpus size** | 3.8 MB, full release | 778 docs (answer-key corpus) | 298 docs |
| **Base rate in the full release** | — | — | 35.25% hallucinated |
| **Length-only AUC (full)** | — | — | 0.6595 |
| **Length-only AUC (this split)** | **0.4945** | 0.4376 | **0.5239** (length-matched) |
| **This project's ROC-AUC** | **0.9251** | **0.6355** | **0.6127** |
| **This project's AUPR** | 0.9217 | 0.6450 | 0.6112 |
| **This project's detection F1** | 0.8202 | 0.6773 | 0.6076 |
| **Retrieval recall@5** | 0.994 | 0.000 (closed book) | 0.902 |
| **Claim precision** | 0.4674 | 0.0000 | 0.9642 |
| **Claim F1** | 0.8107 | 0.0000 | 0.7269 |
| **Latency (mean)** | 4.18 s | 4.12 s | 10.01 s |
| **Bootstrap resamples** | 1,000 | 1,000 | 1,000 |

**Table 5.2 — RAGTruth sub-task composition, and why the pairing rule is unusual**

| Sub-task | Rows in full release | Hallucinated | Base rate | Length-only AUC | Rows in this project's split | Hallucinated |
|---|---:|---:|---:|---:|---:|---:|
| QA | 875 | 160 | 18.29% | 0.7025 | 294 | 147 |
| Data-to-text | 900 | 579 | 64.33% | 0.4818 | 408 | 204 |
| Summarization | 900 | 204 | 22.67% | 0.5147 | 298 | 149 |
| **All** | **2,675** | **943** | **35.25%** | **0.6595** | **1,000** | **500** |

Two things follow. First, **the length shortcut is real in RAGTruth**: 0.6595
on the full release, and 0.7025 within the QA sub-task alone. Second, the
length-matched split drives it down to **0.5239**, which is near chance — so
the 0.6127 headline is not a length artefact. The QA sub-task's 0.7025 is
the reason the matching matters: without it, a detector could score well on
RAGTruth for the wrong reason.

**The length shortcut is not present in the other two splits.** HaluEval's
length-only AUC is **0.4945** and TruthfulQA's is **0.4376** — both at or
below chance, because HaluEval is length-matched by construction and
TruthfulQA's paired rows differ in content rather than in length. That is
worth stating plainly: **the RAGTruth matching was necessary; the HaluEval
matching was not.** It means the three headline numbers are not all measured
under the same degree of care, and a reviewer is entitled to ask why the
length-matched dataset is the one on which the project scores worst.

**Table 5.3 — The three datasets, and what each one is good for testing**

| Question you want to answer | Use | Why |
|---|---|---|
| "Can it detect hallucination when there is evidence to check against?" | **HaluEval** | Largest, most balanced, retrieval on. This project's best result (0.9251) and the one it stands on. |
| "How much of that detection is the *evidence* rather than the model?" | **TruthfulQA** | Retrieval off. The 0.9251 → 0.6355 drop **is** the contribution of the evidence. |
| "Does it work on real RAG pipelines with real noise?" | **RAGTruth** | Three sub-tasks, span-level labels, and a length confound that the matching removes. |

---

## 6. Every paper's evaluation data

**Table 6.1 — Evaluation set, size, split, and metric, for every paper**

| Paper | Dataset(s) | n evaluated | Full size | Split / subset | Metric | Comparability |
|---|---|---:|---:|---|---|---|
| **This project** | HaluEval | 1,000 rows / 500 q | 35,000 | 500/500 paired, length-matched | ROC-AUC, AUPR, F1 | — |
| **This project** | TruthfulQA | 1,000 rows / 500 q | 817 | 500/500 paired, closed book | ROC-AUC, AUPR, F1 | — |
| **This project** | RAGTruth | 1,000 rows / 277 q | 2,675 | 500/500 paired within instance, length-matched | ROC-AUC, AUPR, F1 | — |
| **RAGChecker** | 10 domains (ClapNQ, NovelQA, RobustQA ×7, KIWI) | 4,162 q | 4,162 | per-domain 71–511; separate 280-instance human set | P / R / F1 | No |
| **FAVA** | own set + 150-prompt benchmark | 364 passages | 471 collected | test split; source bench 300 responses | detection gain; FActScore Δ | No |
| **FActScore** | human set + 13 subjects | 183 entities, 6,500 gens | 6,500 | 15 human-annotated subjects | FActScore | No |
| **RAGAS** | WikiEval | 50 | 50 | full | agreement accuracy | No |
| **SAFE** | LongFact | 250 main / 496 human | 496 human pairs | human-adjudicated subset | agreement; F1@64 | No |
| **AlignScore** | 22 datasets | **not stated** | — | — | AUC-ROC | No |
| **CRAG** | PopQA, Biography, PubHealth, Arc-Challenge | 1,399 (PopQA) | 14k minus held-out | PopQA test split | accuracy | No |
| **Self-RAG** | PopQA, TriviaQA, PubHealth, ARC-C, Biography, ASQA | 1,399 / 11,313 | 11,313 | TriviaQA **unfiltered** — a general QA test set, not a hallucination set | accuracy | No |
| **Sufficient Context** | FreshQA, Musique-Ans, HotPotQA | 115 / 452 / 500 | 452 / 500 | 115 is the **gold sufficiency** set | autorater F1; correct-answer gain | No |
| **A Stitch in Time** | custom article task; HotPotQA; false-premise QA | 150 topics → 750 sentences | 750 | author-annotated | hallucination rate; detection recall | No |
| **Hybrid Retrieval** | HaluBench (DROP, HaluEval, RAGTruth, FinanceBench, PubMedQA, COVIDQA) | 300 scored (13,867 pool) | 13,867 | 300 scored for hallucination (50/source, 25 PASS/25 FAIL); **single annotator** | MAP@3; hallucination rate | No |
| **Semantic Uncertainty** | CoQA, TriviaQA | 8,000 × 2 | 8,000 | — | AUROC | No |
| **SelfCheckGPT** | WikiBio | 238 passages, 1,908 sentences | 1,908 | sentence-level | AUC-PR | No |
| **Don't Hallucinate** | MMLU, K-Crosswords, HellaSwag, Propaganda, AmbigQA, ElectionQA23 | 200–1,007 (test splits) | 1,094 | one held-out example per question; 67–1,094 per dataset | reliable accuracy gain | No |
| **ARES** | KILT (NQ, HotpotQA, FEVER, WoW), SuperGLUE (MultiRC, ReCoRD), AIS | 300 PPI / 707 / 510 | 707 / 510 | PPI set of 300 for the guarantee | relevance accuracy | No (score), yes (practice) |
| **CoVe** | Wikidata, Wiki-Category, MultiSpanQA, biographies | 56 / 55 / 418 | 418 | — | F1; FactScore | No |
| **FLARE** | 2WikiMultihopQA, StrategyQA, ASQA, WikiAsp | 500 / 229 / 500 / 500 | 500 | "at most 500 per dataset due to cost" | EM / F1 | No |
| **DRAGIN** | 2Wiki, HotpotQA, IIRC, StrategyQA | 3,954 | 3,954 | — | EM / F1 | No |
| **IRCoT** | HotpotQA, 2Wiki, MuSiQue, IIRC | 500 × 4 | 500 | — | F1 ± std | No |
| **LLM-Check** | RAGTruth | 450 test instances | 450 | 150 per sub-task | AUROC | **YES** |
| **AES** | RAGTruth | 2,700 | 2,675+ | **extended** test set (adds Llama3-8B) | span AUROC | No (granularity) |
| **Frequency-Aware Attention** | RAGTruth | 2,700 | 2,675+ | extended test set | token / span AUROC | No (granularity) |
| **Desiderata** | RAGTruth + own RAG benchmark | not stated in paper | — | — | ROC-AUC, PR-AUC, F1 | **YES** |
| **CRG** | Natural Questions, HaluEval | 595 / 629 **calibration** | — | 60/40 calibration–test split | coverage; FPR@95%; AUC | No |
| **Embedding & Gradient Say Wrong** | HaluEval | 35,000 | 35,000 | 10% train / 90% test **per task** | accuracy | No |
| **Decoding Memories** | HaluEval | 400 | 10,000 QA | QA subset only, random; **own** generations, GPT-4o labels | AUROC | Closest, but labels differ |
| **MedICO** | HaluEval | 1,000 | 10,000 QA | random `<query, right, hallucinated>` triplets | F1 | No |
| **InternalInspector I2** | HaluEval | 30,000 | 30,000 | 30% train / 70% test | accuracy | No |
| **Markov Chain Debate** | HaluEval | 350 | 30,000 | QA 150 / Summ 50 / Dialogue 150 | accuracy / F1 | No |
| **Small Agent Can Also Rock!** | HaluEval | 100 test (900 train) | 10,000 QA | QA only | accuracy / F1 | No |
| **Teaching with Lies** | HaluEval | 8,000 train + 2,000 test | 10,000 | train + test corpus | F1 / precision / accuracy | No |
| **SHAM-Bench** | RAGTruth partitions | 2,700 | — | own task mix | span & sentence F1 | No |

**Table 6.2 — Evaluation size, ranked**

| Rank | Paper | n | What it is |
|---:|---|---:|---|
| 1 | Hu et al. | 35,000 | 90% test of the full HaluEval release, accuracy |
| 2 | InternalInspector I2 | 30,000 | 70% test of full HaluEval subsets, accuracy |
| 3 | Semantic Uncertainty | 16,000 | 8,000 × 2 datasets, AUROC |
| 4 | Hybrid Retrieval | 13,867 | a retrieval **pool**; only 300 scored |
| 5 | Self-RAG | 11,313 | TriviaQA **unfiltered** |
| 6 | FActScore | 6,500 | generations across 13 subjects |
| 7 | RAGChecker | 4,162 | queries over 10 domains |
| 8 | DRAGIN | 3,954 | across 4 datasets |
| 9 | SelfCheckGPT | 1,908 sents | WikiBio sentences |
| 10 | Don't Hallucinate | 1,007 | largest test split |
| 11 | **This project** | **1,000 rows** | 500 / 500 / 277 unique questions |
| 12 | MedICO | 1,000 | HaluEval triplets |
| 13 | Sufficient Context | 500 | HotPotQA |
| 14 | FLARE / IRCoT | 500 | per dataset |
| 15 | CoVe | 418 | MultiSpanQA |
| 16 | FAVA | 364 | annotated passages |
| 17 | Markov Chain Debate | 350 | HaluEval |
| 18 | ARES | 300 | PPI set |
| 19 | Hybrid Retrieval | 300 | scored for hallucination |
| 20 | RAGAS | 50 | WikiEval |
| 21 | Sufficient Context | 115 | gold sufficiency set |

**Read this ranking carefully — it is a trap.** The largest numbers are not
comparable to each other. Hu et al.'s 35,000 is 90% of a test split
measured with **accuracy**. Semantic Uncertainty's 16,000 is measured with
**AUROC** on a different task entirely. Hybrid Retrieval's 13,867 is a
retrieval pool of which 300 rows were scored. This project's 1,000 rows sit
in the lower-middle of the ranking, and it is **not** the largest evaluation
in the literature. What it *is* the largest of: **the largest evaluation among
systems that report uncertainty on their results** — 500 unique queries
against ARES's 300 and Hybrid Retrieval's 300, both of which report point
estimates only.

---

## 7. The comparison tables for this project's three benchmarks

### 7.1 RAGTruth — response-level detection AUROC

This is the only genuinely like-for-like comparison in the whole literature.
Every row reports response-level detection AUROC on RAGTruth. The win rule
is fixed: **a win requires this project's 95% CI lower bound to sit strictly
above the published value.**

**Table 7.1 — AUROC family (LLM-Check anchors)**

| Published system | Their AUROC | Ours | Our 95% CI | Verdict |
|---|---:|---:|---|---|
| LLM-Check — Logit, Window Entropy | 0.5458 | 0.6127 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Perplexity | 0.5327 | 0.6127 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Logit, Log Entropy | 0.5168 | 0.6127 | [0.5494, 0.6581] | **WIN** |
| LLM-Check — Hidden Score | 0.5724 | 0.6127 | [0.5494, 0.6581] | **TIE** |
| LLM-Check — Attention Score | 0.5830 | 0.6127 | [0.5494, 0.6581] | **TIE** |

**3 wins, 2 ties, 0 losses.** The lower bound 0.5494 clears three of the
five anchors and sits below the other two.

**Table 7.2 — AUROC family (the three newer papers — all losses)**

| Paper | Their AUROC | Protocol | Ours | Verdict |
|---|---:|---|---|---|
| AES — Llama2-7B | 0.8829 | span-level | 0.6127 | **LOSS** |
| AES — Llama2-13B | 0.9250 | span-level | 0.6127 | **LOSS** |
| AES — Llama3-8B | 0.8807 | span-level | 0.6127 | **LOSS** |
| AES — Mistral-7B | 0.8931 | span-level | 0.6127 | **LOSS** |
| AES — Llama2-70B | 0.9332 | span-level | 0.6127 | **LOSS** |
| Freq-Aware — LLaMA-7B (Fourier-high) | 0.8412 | span-level pooled | 0.6127 | **LOSS** |
| Freq-Aware — LLaMA-13B (Fourier-high) | 0.8585 | span-level pooled | 0.6127 | **LOSS** |
| Freq-Aware — Mistral-7B (Fourier-high) | 0.8812 | span-level pooled | 0.6127 | **LOSS** |
| Desiderata — SFT | 0.871 | response-level | 0.6127 | **LOSS** |
| Desiderata — Few-shot | 0.836 | response-level | 0.6127 | **LOSS** |
| Desiderata — LLM-as-a-Judge | 0.825 | response-level | 0.6127 | **LOSS** |
| Desiderata — Prompt-optimized | 0.800 | response-level | 0.6127 | **LOSS** |

**Table 7.3 — Desiderata's own detectors, ROC-AUC and PR-AUC**

| Detector | ROC-AUC | PR-AUC | Verdict on ROC-AUC | Verdict on PR-AUC vs our 0.6112 |
|---|---:|---:|---|---|
| SFT | 0.871 | 0.688 | **LOSS** | **LOSS** |
| Few-shot | 0.836 | 0.694 | **LOSS** | **LOSS** |
| LLM-as-a-Judge | 0.825 | 0.682 | **LOSS** | **LOSS** |
| Prompt-optimized | 0.800 | 0.671 | **LOSS** | **LOSS** |
| SC-GPT (Claude) | 0.586 | 0.199 | **TIE** | **LOSS** |
| SC-GPT (GPT-4-mini) | 0.584 | 0.199 | **TIE** | **LOSS** |

**Table 7.4 — F1 family (response-level detection F1)**

| Published system | Their F1 | Ours | Our 95% CI | Verdict |
|---|---:|---:|---|---|
| RAGTruth paper — Finetuned Llama-2-13B | 0.787 | 0.6076 | [0.5383, 0.6632] | **LOSS** |
| RAGTruth paper — Prompt gpt-4-turbo | 0.634 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| RAGTruth paper — SelfCheckGPT gpt-3.5 | 0.588 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| RAGTruth paper — Prompt gpt-3.5 | 0.529 | 0.6076 | [0.5383, 0.6632] | **WIN** |
| RAGTruth paper — LMvLM gpt-4-turbo | 0.494 | 0.6076 | [0.5383, 0.6632] | **WIN** |
| Desiderata — SFT | 0.671 | 0.6076 | [0.5383, 0.6632] | **LOSS** |
| Desiderata — Few-shot | 0.628 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| Desiderata — LLM-as-a-Judge | 0.617 | 0.6076 | [0.5383, 0.6632] | **TIE** |
| Desiderata — Prompt-optimized | 0.538 | 0.6076 | [0.5383, 0.6632] | **WIN** |
| Desiderata — SC-GPT (GPT-4-mini) | 0.342 | 0.6076 | [0.5383, 0.6632] | **WIN** |
| Desiderata — SC-GPT (Claude) | 0.346 | 0.6076 | [0.5383, 0.6632] | **WIN** |

**Table 7.5 — Span-level F1: a negative result, not a comparison**

| Published system | Their span F1 | This project |
|---|---:|---|
| RAGTruth paper — Finetuned Llama-2-13B | 0.527 | **no span-level predictions** |
| RAGTruth paper — Prompt gpt-4-turbo | 0.283 | **no span-level predictions** |
| RAGTruth paper — Prompt gpt-3.5 | 0.128 | **no span-level predictions** |

This project neither wins nor loses here. The pipeline does not attempt
span-level hallucination localisation, and that is stated as a limitation
rather than hidden.

### 7.2 HaluEval

**Table 7.6 — Every paper that reports a number on HaluEval**

| Paper | n evaluated | Metric | Their headline | Comparable to our ROC-AUC 0.9251? |
|---|---:|---|---|---|
| **This project** | 1,000 rows / 500 q | ROC-AUC | **0.9251** CI [0.9113, 0.9401] | — |
| Decoding Memories | 400 | AUROC | 61.2–84.9 for its baselines | **Closest** — but its labels are GPT-4o-generated, not HaluEval's own |
| Embedding & Gradient Say Wrong | 35,000 | accuracy | 97.19 / 77.10 / 95.08 / 83.01 | **No** — metric and size both differ |
| InternalInspector I2 | 30,000 | accuracy | 0.691 / 0.648 / 0.671 | **No** |
| MedICO | 1,000 | F1 | 0.946 / 0.925 | **No** |
| Markov Chain Debate | 350 | accuracy / F1 | 70.67 / 73.81 | **No** |
| Small Agent Can Also Rock! | 100 test | accuracy / F1 | 71.00 / 72.38 | **No** |
| Teaching with Lies / HaluCheck | 8,000 + 2,000 test | F1 / precision / accuracy | 0.753 / 0.782 / 0.857 | **No** |
| CRG | 629 **calibration** | FPR @ 95% coverage | **100.0%** (CI [99.0%, 100%]) | **No** — but it is a negative result about the same data |

### 7.3 TruthfulQA

**Table 7.7 — Nothing to compare against**

| System | Dataset | n | Metric | Result |
|---|---|---:|---|---|
| **This project** | TruthfulQA, closed book | 1,000 rows / 500 q | ROC-AUC | **0.6355** CI [0.6107, 0.6594] |
| *every published paper* | — | — | — | **none reports TruthfulQA detection with a comparable metric** |

This is not a win and not a loss. It is a third independent measurement, and
its value is that it isolates the contribution of the evidence: with
retrieval on the score is 0.9251, with retrieval off it is 0.6355, and that
gap is the size of what the retrieved documents contribute.

---

## 8. Rules for reading any comparison in this thesis

**Table 8.1 — The seven rules, and what each one prevents**

| Rule | Prevents |
|---|---|
| **1. Never compare across metric families.** A ROC-AUC is not set against an F1, an accuracy, a FactScore or a hallucination rate. | The most common error in a review, and the easiest to make by accident when the tables are adjacent. |
| **2. State the direction.** "0.14" is good on a hallucination rate and bad on a ROC-AUC. | A reviewer reading a summary table out of context. |
| **3. A win needs CI separation, not a point-estimate overlap.** | The overclaim bug this project already had once: comparing point estimates and calling an overlap a win. |
| **4. Report the held-out number where the head was fitted on the same split.** | The RAGTruth headline 0.6127 is in-sample; the held-out figure is 0.6185, CI [0.5371, 0.6943], and on held-out alone the LLM-Check comparison becomes 2 wins, 3 ties and 0 losses. |
| **5. Granularity must match.** Span-level, token-level, sentence-level and response-level AUROC are not the same quantity. | Understating or overstating the AES and Frequency-Aware losses. |
| **6. Report n evaluated alongside the full dataset size.** | "accuracy on 100 examples" and "accuracy on 35,000" are different claims. |
| **7. Say when a comparison is not possible.** | The temptation to compare against a number from a different dataset rather than write "not comparable". |

**The two numbers most likely to be misquoted in this thesis.** The
mitigation Δ **−0.007** is the *regenerated-rows* figure (16 rows, CI
[−0.0217, +0.0069], p = 0.404). The all-500-rows figure is **−0.00022**,
which is near zero by construction because the rewrite is a no-op on the 484
kept rows. And this project's HaluEval evaluation is **not** the largest in
the literature — Semantic Uncertainty's 16,000 and FActScore's 6,500 are
larger. It is the largest *among systems that report uncertainty*.

---

## 9. Source and provenance

Every published number in this document comes from one of three places, and
each was read from the source rather than from a secondary summary.

| Source | What it supplied |
|---|---|
| `docs/related_pipeline_workflows.md` §3.10–§3.12 | The per-paper metric and evaluation-size record for the 19-paper architectural survey, and the statistical-rigour columns. |
| `Paper/ragtruth_comparison.md` | Every paper that evaluates on RAGTruth, with the AUROC and F1 anchors and the verdicts in §7.1. |
| `Paper/halueval_papers.md` | Every paper that evaluates on HaluEval, with n evaluated, and the five papers that must not be listed. |
| `Paper/comparison.md` | The adjacent-systems architecture table and the mitigation Δ. |
| `data/published_anchors.json` | The RAGTruth AUROC and F1 anchors, with their source tables and the win rule. |
| `data/metrics_halueval.json`, `data/metrics_truthfulqa_closedbook.json`, `data/metrics_ragtruth_fitted.json` | This project's own headline metrics, CIs, retrieval figures and latencies. |
| `data/ragtruth_dataset_report.json` | The RAGTruth sub-task composition, base rates and length-only AUCs in Table 5.2. |
| `data/halueval_eval_1000_matched.jsonl`, `data/truthfulqa_eval_1000.jsonl` | The HaluEval and TruthfulQA length-only AUCs in Table 5.1, computed with the same `_length_only_auc` function as the RAGTruth figures. |

**Four cautions that travel with these tables.**

1. **SAFE's 496 and this project's 496 are unrelated quantities that
   coincide.** SAFE's is a human *evaluation* set; this project's is a
   *fitting* set for the fusion head. Never write them in the same sentence
   without saying which is which.
2. **FActScore's "FS" is an estimated FActScore, not an F1.** It is an
   estimate of a supported-fact fraction produced by a model, and it is not
   comparable to this project's claim-level F1.
3. **Hybrid Retrieval's FAIL-only figures exist only inside images in the
   paper.** They are not stated in its text and must not be quoted.
4. **TruthfulQA's honest evaluation file is `metrics_truthfulqa_closedbook.json`.**
   `metrics_truthfulqa.json` and `metrics_truthfulqa_fitted.json` hold
   markers, not measurements, and are quarantined.

**One correction recorded here for the record.** `Paper/comparison.md` labels
a system "Conformal-RAG". No paper has that title — it is Sinha's Conformal
RAG Guardrails, already reference [18], whose three signals are RAD (cosine),
SEC (NLI) and TFG (lexical), calibrated by split conformal prediction at
1 − α. It is not to be confused with C-RAG (Kang et al., ICML 2024,
arXiv:2402.03181), which certifies a generation configuration against a
ROUGE-L risk on AESLC, CommonGen, DART and E2E, uses no NLI and no judge
model, and evaluates on neither RAGTruth nor HaluEval. It is not comparable
and is not listed among this project's related work.
