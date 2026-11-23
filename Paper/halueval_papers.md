# HaluEval — every paper that evaluates on it, and how much data each uses

**Our measurement.** `data/metrics_halueval.json` — 1000 rows / 500 unique
queries, 500/500 balanced, length-matched split.

> ROC-AUC **0.9251**, CI [0.9113, 0.9401] · AUPR 0.9217 CI [0.9057, 0.9383]
> · Detection F1 0.8202 CI [0.7970, 0.8437] · Accuracy 0.829
> · Claim F1 0.8107 (precision 0.4674, recall 0.8103)

**Two rules used throughout this file:**

1. **A paper counts as "using HaluEval" only if it ran its own detection
   experiments on HaluEval data.** A related-work citation, a dataset it merely
   *trained* on, or a benchmark it says it *will* evaluate on in future work does
   not count.
2. **Sample size is reported as the number of examples actually evaluated**, with
   the full dataset size alongside, because the gap between them is often large.

---

## 1. Papers in this repository

| Paper | Evaluates on HaluEval? | n evaluated | Subset / split | Metric |
|---|---|---|---|---|
| **Decoding Memories** (arXiv 2508.21228) | **YES** | **400** | QA subset only, random sample; own generations, GPT-4o labels | AUROC |
| **The Semantic Illusion** (arXiv 2512.15068) | **YES** | **629** calibration (+200 judge subset) | "HaluEval (ChatGPT)", 60/40 cal-test; subset not specified | conformal FPR @95% coverage, AUC, F1 |
| AutoHall (arXiv 2310.00259) | NO — cited only | — | — | — |
| AILS-NTUA (arXiv 2404.01210) | NO — cited only | — | — | — |
| TU Wien (SemEval-2024 Task 6) | NO — cited only | — | — | — |
| SHROOM shared task (organizers) | NO — cited / contrasted against | — | — | — |
| IJSART (2026) | NO — **proposed only, future work** | — | — | — |
| QuaLLM SLR (2026) | NO — secondary source | — | — | — |
| SHAM-Bench (arXiv 2610.27370) | NO — cited / excluded by design | — | — | — |
| INSIDE (ICLR 2024) | NO | — | — | — |
| LLM-Check (NeurIPS 2024) | NO | — | — | — |
| Dale et al. (ACL 2023) | NO | — | — | — |

**Only 2 of the 12 local papers actually ran HaluEval experiments**, and both
used small derived subsets (400 and 629) rather than any full HaluEval split.

## 2. Papers outside this repository that do evaluate on HaluEval

Verified from the papers' own tables.

| Paper | Venue | Subset | n evaluated | Full size | Metric / headline |
|---|---|---|---|---|---|
| **Embedding and Gradient Say Wrong** (Hu, Zhang, Peng, Zhang, Wu, Chen, Zhao) | EMNLP 2024 Main — [2024.emnlp-main.116](https://aclanthology.org/2024.emnlp-main.116/) | QA + Dialogue + Summary + General | **35,000** (10% train / 90% test per task) | 35,000 | **Accuracy**: LLaMA-2-7B QA 97.19, Dialogue 77.10, Summary 95.08, General 83.01 |
| **MedICO** (Zhao et al.) | EMNLP 2024 Demo — [2024.emnlp-demo.4](https://aclanthology.org/2024.emnlp-demo.4/) | QA | **1,000** random `<query, right, hallucinated>` triplets | 10,000 QA | **F1**: ENSB Llama3-8B 0.946; Qwen2-7B 0.925 |
| **Small Agent Can Also Rock!** (Cheng et al.) | EMNLP 2024 Main — [2024.emnlp-main.809](https://aclanthology.org/2024.emnlp-main.809/) | HaluEval-QA | **1,000** → 900 train / **100 test** | 10,000 QA | **Acc / F1**: HaluAgent-13B 71.00 / 72.38 |
| **InternalInspector I2** (Beigi et al.) | Findings EMNLP 2024 — [2024.findings-emnlp.751](https://aclanthology.org/2024.findings-emnlp.751/) | QA + Dialogue + Summarization | **full subsets**, 30% train / 70% test | 30,000 | **Accuracy**: LLaMA-2-7B QA 0.691, Dialogue 0.648, Summ 0.671 |
| **Markov Chain Multi-Agent Debate** (Sun et al.) | ICASSP 2025 — [arXiv:2406.03075](https://arxiv.org/abs/2406.03075) | QA + Summ + Dialogue | **350** (QA 150, Sum 50, Dialogue 150) | 30,000 | **Acc / F1**: QA 70.67 / 73.81; Sum 70.00 / 68.09; Dialog 76.00 / 70.97 |
| **Teaching with Lies / HaluCheck** (Pandit et al.) | arXiv preprint — [arXiv:2505.17558](https://arxiv.org/abs/2505.17558) | train + test corpus | **8,000 train + 2,000 test** | 10,000 | **F1 / P / Acc**: HaluCheck-3B 0.753 / 0.782 / 0.857 |

## 3. Papers that must NOT be listed as HaluEval evaluations

These were checked and **do not** have a HaluEval evaluation.

| Paper | Venue | Why it was wrongly suspected |
|---|---|---|
| **Enhancing Hallucination Detection through Perturbation-Based Synthetic Data Generation** (Zhang et al.) | Findings ACL 2024 — [2024.findings-acl.789](https://aclanthology.org/2024.findings-acl.789/) | HaluEval is **training data for a baseline**, not an eval set. Evaluation is on OpenDialKG-Eval (312 test / 90 dev) and BEGIN. |
| **Factcheck-Bench** (Wang et al.) | Findings EMNLP 2024 — [2024.findings-emnlp.830](https://aclanthology.org/2024.findings-emnlp.830/) | HaluEval appears only in a related-work dataset comparison table. **0 examples evaluated.** |
| **GPTLODS** (Mountantonakis & Tzitzikas) | ISWC 2023 — [CEUR Vol-3632](https://ceur-ws.org/Vol-3632/ISWC2023_paper_408.pdf) | **No HaluEval anywhere in the paper.** Evaluates on 1,000 manually-labelled ChatGPT facts about Greek persons. |
| **HaluCheck (ESWA system paper)** (Heo, Son, Park) | Expert Systems with Applications 272:126712 (2025) | **Paywalled; sample size and metrics could not be verified.** Do not put a number in a table for this one. |

> **Note on the QuaLLM review.** Its HaluEval citation list over-claims in two
> places: Factcheck-Bench and GPT LODS are cited as HaluEval evaluations but
> neither evaluates on it. The review is still useful as a lead generator, but
> its claims need re-verification before use.

## 4. Two traps in this table

**Metric mismatch — none of the external papers report AUROC on HaluEval.**
Hu, InternalInspector and Markov chain report **accuracy**; MedICO and
Teaching-with-Lies report **F1**; Small Agent reports both. Our headline
number is **ROC-AUC 0.9251**. It can only be set beside another AUROC — which
among all the HaluEval papers here means only Decoding Memories (400 QA
examples, AUROC 61–84 for its baselines).

**Sample-size mismatch — the spread is 350 to 35,000.** Our 1000 rows /
500 unique queries sits in the lower-middle. MedICO's 1,000 and Small Agent's
100 test examples are far from the full 10,000-question QA subset; Hu and
InternalInspector use the entire release. On HaluEval, **accuracy on a
100-example test set is not the same claim as accuracy on 35,000.**

**Protocol note.** Decoding Memories regenerates its own responses from the
HaluEval QA questions and labels them with GPT-4o — it does not use HaluEval's
own annotations. Our split uses HaluEval's own labels.

## 5. A useful agreement

"The Semantic Illusion" reports that embedding-based and NLI detectors reach
**100% false-positive rate at 95% target coverage on HaluEval**, despite an
NLI AUC of 0.81 — a *negative* result for the detector families. Our own
finding that the TruthfulQA gap is a feature-quality limit rather than a bug
(0.80 in training vs 0.60 at serving) is the same phenomenon, reported
independently. This is citable support rather than a contradiction.

---

## Sources

- Hu et al. *Embedding and Gradient Say Wrong.* EMNLP 2024 — <https://aclanthology.org/2024.emnlp-main.116/>
- Zhao et al. *MedICO.* EMNLP 2024 Demo — <https://aclanthology.org/2024.emnlp-demo.4/>
- Cheng et al. *Small Agent Can Also Rock!* EMNLP 2024 — <https://aclanthology.org/2024.emnlp-main.809/>
- Beigi et al. *InternalInspector I2.* Findings EMNLP 2024 — <https://aclanthology.org/2024.findings-emnlp.751/>
- Zhang et al. *Enhancing Hallucination Detection through Perturbation-Based Synthetic Data Generation.* Findings ACL 2024 — <https://aclanthology.org/2024.findings-acl.789/>
- Wang et al. *Factcheck-Bench.* Findings EMNLP 2024 — <https://aclanthology.org/2024.findings-emnlp.830/>
- Sun et al. *Markov Chain-based Multi-agent Debate.* ICASSP 2025 — <https://arxiv.org/abs/2406.03075>
- Pandit et al. *Teaching with Lies.* arXiv:2505.17558 — <https://arxiv.org/abs/2505.17558>
- Mountantonakis & Tzitzikas. *GPTLODS.* ISWC 2023 — <https://ceur-ws.org/Vol-3632/ISWC2023_paper_408.pdf>
- Heo, Son & Park. *HaluCheck.* ESWA 272:126712 (2025) — paywalled, **not verified**
- Bestetti, Zhang & Stefanidis. *QuaLLM SLR.* EDBT/ICDT 2026 Workshops — <https://ceur-ws.org/Vol-4192/QuaLLM_KG-paper3.pdf>

**Our numbers:** `data/metrics_halueval.json` (n = 1000, 500 unique queries,
500/500 balanced).
