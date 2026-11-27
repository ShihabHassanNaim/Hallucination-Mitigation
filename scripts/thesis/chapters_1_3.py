"""Chapters 1-3: Introduction, Related Work, System Design."""

from __future__ import annotations

from .content import GRAPHS, AUDIT, H, META, N_PIPELINE_WORDS as N_WORDS
from .steps import step_figure
from .renderer import Thesis


# ==========================================================================
# CHAPTER 1 -- INTRODUCTION
# ==========================================================================

def chapter_1(t: Thesis):
    t.chapter_heading("Introduction")

    t.h2("1.1", "Background and Motivation")
    t.para(
        "Large language models have become the default interface to "
        "natural-language knowledge, and the ways in which they fail have "
        "become a first-class concern of the field rather than a footnote to "
        "it. The failure mode that concerns this project is the hallucination: "
        "the model produces an assertion that is fluent, well-formed and "
        "confidently styled, but is unsupported by the evidence that was "
        "available to it. Hallucinations are difficult to detect by reading "
        "alone, because the surface form of a fabricated claim is "
        "indistinguishable from that of a true one, and because the "
        "distribution over plausible continuations is wide enough that a "
        "system which is confidently wrong is an ordinary outcome of sampling "
        "rather than an exceptional one.",
    )
    t.para(
        "Retrieval-augmented generation (RAG) was introduced as a remedy. By "
        "grounding a generator in passages retrieved from an external corpus, "
        "RAG replaces parametric recall with verifiable reference and sharply "
        "reduces the frequency of fabricated facts. The remedy, however, "
        "introduces its own failure mode, and the failure mode is not merely "
        "an absence of knowledge but a mismatch between evidence and claim. A "
        "generator that has been handed six passages will still produce "
        "sentences that none of those passages assert; it will combine two "
        "sources into a third that neither supports; and it will, on occasion, "
        "assert something that the retrieved evidence actively contradicts. "
        "The retrieved context does not remove hallucination from the system; "
        "it relocates it, and the relocation creates a specific engineering "
        "requirement: the pipeline must be able to check its own output against "
        "the evidence it was given, and must be willing to admit when that "
        "check fails.",
    )
    t.para(
        "That admission is the crux. A detector that always answers \u201cnot "
        "hallucinated\u201d is trivially accurate on a corpus whose responses "
        "are faithful and trivially useless on one whose responses are not. A "
        "detector that always answers \u201challucinated\u201d is worse than "
        "useless, because it would suppress correct answers along with wrong "
        "ones. The useful detector must be calibrated \u2014 its confidence "
        "must mean something \u2014 and it must be permitted to abstain when "
        "its confidence does not warrant an assertion.",
    )

    t.h2("1.2", "Problem Statement")
    t.para(
        "The problem addressed by this project can be stated precisely. Given "
        "a question, a retrieval corpus and a generator whose internals are "
        "not observable, produce a response together with a machine-checkable "
        "verdict on whether that response is supported by the evidence, and "
        "where it is not, either repair it or refuse it.",
    )
    t.para(
        "Three constraints follow from that statement and shape every design "
        "decision in this thesis. The first is that the system must be "
        "black-box: the generator may be a commercial API with no weight "
        "access, and the pipeline must work unchanged in that setting. Many of "
        "the strongest published detectors \u2014 INSIDE, the attention- and "
        "logit-based methods in LLM-Check, the hidden-state covariance "
        "analyses \u2014 require access to activations that an API consumer "
        "does not have. The second is that the verdict must be localisable: a "
        "user needs to know which sentence is unsupported and why, not merely "
        "that the response as a whole has some low aggregate score. The third "
        "is that the system must be able to decline: a correct answer "
        "delivered under low confidence is a liability, and a system that "
        "cannot say \u201cI do not know\u201d will be measured as though it "
        "were more reliable than it is.",
    )
    t.para(
        "These constraints together rule out a family of approaches that would "
        "otherwise be simplest. A prompt that asks the model whether it is "
        "hallucinating asks the very entity whose reliability is in question, "
        "and self-assessment is known to be poorly correlated with actual "
        "correctness. A perplexity or logit threshold requires the "
        "distribution over tokens at generation time, which is not available "
        "for a remote generator. A learned probe over hidden states requires "
        "activations. What remains is the black-box-with-external-evidence "
        "family, and this project develops it carefully.",
    )

    t.h2("1.3", "Research Objectives")
    t.para(
        "The objective of this project is to design, implement and evaluate a "
        "black-box hallucination detection and mitigation pipeline for "
        "retrieval-augmented generation, and to report its performance "
        "honestly against the published literature. This objective decomposes "
        "into seven concrete targets.",
    )
    t.numbered([
        "To design a thirteen-stage pipeline in which a generated response is "
        "decomposed into atomic claims, each claim is verified independently "
        "against claim-conditioned retrieved evidence, and the per-claim "
        "verdicts are fused into a single calibrated confidence.",
        "To make the confidence statistically meaningful, by fitting the "
        "fusion head on held-out data rather than by hand-setting weights, "
        "and by calibrating the output so that the stated confidence tracks "
        "the observed frequency of correctness.",
        "To give the pipeline the ability to abstain, through a three-way "
        "final gate that routes a response to PASS, to an evidence-guided "
        "repair, or to refusal.",
        "To implement an evidence-guided mitigation stage that rewrites only "
        "the claims contradicted by the evidence, and that refuses to rewrite "
        "when doing so would damage the answer.",
        "To construct a leak-free, length-matched, paired evaluation on two "
        "public benchmarks \u2014 HaluEval QA and RAGTruth \u2014 "
        "with bootstrap confidence intervals taken over unique questions.",
        "To compare the resulting system against every published method that "
        "reports a comparable metric on a comparable dataset, under a single "
        "declared statistical rule.",
        "To report negative results \u2014 including the cases in which this "
        "system loses to prior work \u2014 with the same prominence as "
        "positive ones.",
    ])

    t.h2("1.4", "Research Gap and Significance")
    t.para(
        "The review of the literature in Chapter 2 leaves four gaps that this "
        "project addresses, and it is worth stating them plainly at the "
        "outset rather than leaving the reader to infer them.",
    )
    t.para(
        "The first gap is the absence of a pipeline that closes the loop. The "
        "published work contains excellent components \u2014 strong "
        "embedding-based screens, effective NLI verifiers, well-designed "
        "fusion schemes \u2014 but the components are generally presented "
        "individually rather than as a system that must make one decision "
        "about one response. The exception is the SHROOM shared task, whose "
        "systems are end-to-end, and the contribution of this project is "
        "complementary: a system specified down to the level at which each "
        "decision is taken, with the intermediate artefacts retained so that "
        "each decision can be audited. The pipeline of Chapter 3 is the first "
        "in the surveyed literature to combine all six of the capabilities "
        "that such a system requires, and the first to re-verify the "
        "repairs it makes.",
    )
    t.para(
        "The second gap is the absence of a decision procedure that can "
        "decline. The literature contains detectors, correctors and "
        "confidence estimators, but the question of what to do when the "
        "evidence is insufficient is usually handled implicitly, by a "
        "threshold on a score. A three-way gate with an explicit abstention "
        "state, and an evaluation that measures whether the abstention was "
        "warranted, is a small design change with a large effect on how the "
        "system should be deployed.",
    )
    t.para(
        "The third gap is methodological rather than technical. Few published "
        "systems report the properties of their evaluation in enough detail "
        "to determine whether their numbers are comparable, and in at least "
        "one case a widely cited comparison table lists systems that never "
        "evaluated on the dataset in question. The contribution here is a "
        "worked example of the stricter protocol \u2014 paired, "
        "length-matched, leak-free, with bootstrap intervals over unique "
        "questions \u2014 together with the demonstration that it changes "
        "the conclusions.",
    )
    t.para(
        "The fourth gap is the reporting of negative results. A field in "
        "which the published record contains several negative findings \u2014 "
        "the certified limits of embedding-based detection, the difficulty of "
        "span-level detection, the failure of prompt-based self-assessment "
        "\u2014 has no strong norm of publishing one's own. This project "
        "reports three of its own, including the case in which it loses to "
        "three published methods.",
    )
    t.para(
        "The significance of the work lies in the combination. A pipeline "
        "whose every stage is specified and auditable, evaluated under a "
        "protocol strict enough that its numbers can be trusted, and reported "
        "with its negative results intact, provides a base on which later "
        "work can be built without first having to re-establish whether the "
        "baseline numbers were real.",
    )

    t.h2("1.5", "Contributions")
    t.para(
        "The contributions of this project are as follows.",
    )
    t.bullets([
        f"The architectural result: of the {N_WORDS} published pipelines "
        "surveyed in Chapter 2, none implements all six of the capabilities "
        "that distinguish a retrieval-grounded defence pipeline from a "
        "single-pass detector, and the two closest implement five. CRISP "
        "implements all six, and publishes its per-claim verdicts with the "
        "document that supported each one, so that any individual statement "
        "can be checked against the passage that justified it.",
        "A fully specified thirteen-stage black-box detection pipeline "
        "(CRISP) whose every stage is configurable from a single YAML file, "
        "so that each stage can be ablated without a single line of code "
        "being changed.",
        "A signed, fitted seven-component fusion head in which the direction "
        "of every component is established empirically rather than assumed. "
        "This matters: the first version of this pipeline used hand-set "
        "weights and scored a ROC-AUC of " + H["hand_auc"] + " \u2014 below "
        "chance \u2014 because two of the seven signals had been given the "
        "wrong sign.",
        "A claim-conditioned premise-selection procedure that resolves the "
        "long-context failure of NLI verification. Without it, a third of the "
        "claims in the RAGTruth setting could not be decided at all, because "
        "the entailment model was being shown the whole retrieved document.",
        "A three-way final gate that permits refusal, and an evaluation "
        "protocol that measures whether refusal was warranted rather than "
        "treating every refusal as a failure.",
        "A rigorous evaluation design: paired construction, length matching, "
        "leak-free splits, bootstrap intervals over unique questions, and a "
        "single interval rule \u2014 a win requires the lower bound of the "
        "confidence interval to clear the published value \u2014 applied "
        "symmetrically to wins and to losses.",
        "A negative result that is reported as a contribution in its own right: "
        "the evidence-guided mitigation stage of this pipeline does not "
        "produce a measurable improvement in answer quality "
        "(BERTScore \u0394 = " + H["mit_delta"] + ", 95% CI "
        + H["mit_delta_ci"] + ", p = " + H["mit_p"] + ").",
    ])

    t.h2("1.6", "Scope and Delimitations")
    t.para(
        "The scope of this work is the detection and mitigation of "
        "hallucination in the outputs of retrieval-augmented generation. "
        "Several boundaries should be stated explicitly, because they "
        "constrain what the reported results can be taken to show.",
    )
    t.para(
        "First, the detector is black-box. It observes the question, the "
        "generated answer, the retrieved passages and its own intermediate "
        "signals; it does not observe the generator's activations. This is a "
        "deliberate choice and it has a cost: methods that read hidden states "
        "have access to information this detector structurally cannot obtain, "
        "and on some benchmarks they are ahead as a result. The compensation is "
        "applicability: the pipeline runs unchanged against a generator that "
        "the user does not host.",
    )
    t.para(
        "Second, the mitigation stage operates at the level of individual "
        "claims, not at the level of whole responses. It rewrites sentences "
        "that the evidence contradicts and leaves the rest of the answer "
        "intact. This is a conservative design: it cannot repair an answer "
        "whose problem is structural rather than local, and its measured "
        "effect on answer quality is correspondingly small.",
    )
    t.para(
        "Third, the evaluation is performed on three English-language "
        "benchmarks. The multilingual components of the pipeline "
        "(mDeBERTa-v3-xnli for entailment, the bilingual BGE-M3 embedder) "
        "would in principle support other languages, but no multilingual "
        "evaluation was performed and none is claimed.",
    )
    t.para(
        "Fourth, the generator used throughout the evaluation is "
        "Qwen2.5-0.5B-Instruct, chosen so that the evaluation could run on a "
        "single consumer GPU. Results obtained with this generator should not "
        "be read as characterising the hallucinations of larger models, which "
        "are typically more fluent and, in the literature, sometimes less "
        "detectable.",
    )

    t.h2("1.7", "Thesis Organisation")
    t.para(
        "The remainder of this thesis is organised as follows. Chapter 2 "
        "reviews the literature on hallucination detection, covering "
        "taxonomies and benchmarks, sentence-similarity methods, model-aware "
        "methods, ensembles, self-consistency and efficiency, and the "
        "positioning of this work against that body of work. Chapter 3 "
        "describes the design of the CRISP pipeline stage by stage, together "
        "with the models, the configuration discipline and the project "
        "structure that make the system reproducible. Chapter 4 presents the "
        "implementation, including the data contracts that bind the stages and "
        "the three delivery interfaces. Chapter 5 sets out the evaluation "
        "methodology in detail: the datasets, the paired and length-matched "
        "design, the leak-free splits, the metric definitions and the "
        "statistical rule used for all comparisons. Chapter 6 reports the "
        "results, including the ablation study, the mitigation study, the "
        "latency analysis and the audit trail. Chapter 7 compares this work "
        "against the published literature on both datasets. Chapter 8 "
        "discusses the findings, including the failure modes and the negative "
        "results. Chapter 9 concludes.",
    )


# ==========================================================================
# CHAPTER 2 -- RELATED WORK
# ==========================================================================

def chapter_2(t: Thesis):
    t.chapter_heading("Related Work")

    t.para(
        "This chapter reviews the published literature on hallucination "
        "detection in large language models, with particular attention to the "
        "family of methods to which this project belongs: black-box detection "
        "that operates on a generated response and external evidence rather "
        "than on the generator's internals. The review is organised to make "
        "the position of this work explicit rather than merely to enumerate "
        "prior systems.",
    )

    t.h2("2.1", "Taxonomies and Benchmarks")
    t.para(
        "Retrieval-augmented generation itself has been surveyed twice over, "
        "and both surveys [2, 3] note that the grounding of a generation in "
        "retrieved evidence reduces factual error without eliminating it, "
        "which is the condition under which the detection problem arises. "
        "The literature on hallucination has been given several taxonomies, "
        "and the differences between them matter when comparing results. The "
        "most widely used distinction separates intrinsic hallucination, in "
        "which the generated response contradicts the source it was given, "
        "from extrinsic hallucination, in which the response cannot be "
        "verified by the source because the source does not mention it. The "
        "distinction is between a wrong answer and an unanswerable one, and it "
        "implies a corresponding difference in the detection problem: an "
        "intrinsic hallucination can in principle be caught by comparison "
        "against a reference, whereas an extrinsic one can only be caught by "
        "comparison against a larger body of knowledge.",
    )
    t.para(
        "Valmeekam et al. [1] provide a related decomposition, distinguishing "
        "hallucinations that arise from a plan that is not executed as "
        "intended from those that arise from the plan being executed but the "
        "plan itself being wrong. This is a useful axis for a RAG system, "
        "because the second kind is exactly what a retrieval stage is meant to "
        "eliminate: if the generator's answer contradicts what the evidence "
        "says, the failure is in the grounding rather than in the reasoning, "
        "and the two are corrected by different means.",
    )
    t.para(
        "Two benchmarks dominate the quantitative literature on retrieval-"
        "augmented generation and are used in this project. HaluEval [11] "
        "provides, for each of four subsets, a paired collection of a right "
        "answer and a plausible hallucinated answer to the same question, "
        "with hallucinated answers generated by instruction-tuned models "
        "rather than constructed artificially \u2014 which makes them difficult "
        "to detect and is the point of the benchmark. RAGTruth [10] is the "
        "more recent of the two and the only one built specifically for "
        "retrieval-augmented generation: it contains responses generated "
        "with a real corpus in the loop, with sentence-level span "
        "annotations for the unsupported spans, across question answering, "
        "data-to-text and summarisation tasks.",
    )
    t.para(
        "The distinction between these two benchmarks is not incidental to "
        "the results in Chapter 6. The pipeline scores " + H["hal_auc"]
        + " on HaluEval and " + H["rag_auc"] + " on RAGTruth. Those "
        "numbers are not a ranking of the pipeline; they are measurements of "
        "the same pipeline on two different distributions of hallucination, "
        "and the gap between them is itself a finding, discussed in "
        "Chapter 8.",
    )

    t.h2("2.2", "Sentence-Similarity Detection")
    t.para(
        "The oldest and still the most practically important family of "
        "detection methods compares the generated sentence to its reference "
        "with a semantic-similarity model and thresholds the result. The "
        "approach has two virtues that account for its persistence: it requires "
        "no training data and no access to the generator, and it is O(1) in "
        "the size of the model.",
    )
    t.para(
        "The SHROOM shared task [8] made an empirical finding that directly "
        "informs the design of this project: hallucinated generations can be "
        "detected more reliably by a cosine threshold over sentence "
        "embeddings than by prompting a large language model for its own "
        "opinion about the same generation. This is the observation on which "
        "Stage 3 of the CRISP pipeline is built, and it is why the pipeline "
        "treats an embedding similarity signal as a first-class feature "
        "rather than as an afterthought.",
    )
    t.para(
        "The failure mode of the similarity family is well documented. A "
        "non-calibrated similarity score of a simple heuristic may be "
        "effective on synthetic hallucinated data and fail spectacularly when "
        "the hallucinated content was produced by a fine-tuned model. The "
        "distance between " + H["hal_auc"] + " on HaluEval and "
        + H["rag_auc"] + " on RAGTruth in this project is an instance of "
        "exactly that failure, and the calibration machinery of Stage 4 "
        "exists to mitigate it.",
    )

    t.h2("2.3", "Model-Aware Detection")
    t.para(
        "The model-aware family reads the generator's internals. INSIDE [6] "
        "computes the covariance of the hidden states and applies an "
        "additional clipping test-time technique, obtaining strong detection "
        "performance on a closed-book misconception benchmark. Other work in "
        "this family derives scores "
        "from attention maps, hidden activations and logit signals, "
        "achieving detection quality at a fraction of the inference cost.",
    )
    t.para(
        "The model-aware family is the strongest in the literature and the "
        "clearest point of departure for this project. Because these methods "
        "require activations, they cannot be applied to a generator accessed "
        "through an API. The trade-off is real and it is not one-sided: on "
        "the RAGTruth benchmark in particular, the model-aware methods "
        "evaluated in Chapter 7 score above the detector presented here. "
        "Chapter 8 discusses what would be needed to close that gap, "
        "including the possibility of using internal-state scorers to replace "
        "or reinforce Stage 3 when the generator is locally available.",
    )

    t.h2("2.4", "Ensemble Systems")
    t.para(
        "Ensemble systems combine several detectors. The SHROOM shared task "
        "showed that systems combining model-agnostic and model-aware signals "
        "by voting, probability fusion or meta-models reached accuracies in "
        "the range of 80.6 to 81.7 per cent, and that combining a fine-tuned "
        "classifier with an NLI model reached 77.8 to 79.9 per cent. These "
        "figures establish that fusion is the productive direction, which is "
        "the direction taken by Stage 9 of this pipeline.",
    )
    t.para(
        "The specific form of fusion matters more than its existence. The "
        "author's first implementation of this project used hand-set weights "
        "on the same seven components that the fitted head now uses, and "
        "scored a ROC-AUC of " + H["hand_auc"] + " \u2014 worse than the "
        "0.5 that a coin flip would achieve. The cause was not the "
        "components but their signs: two of the seven were given the wrong "
        "direction, so the head was subtracting signal from signal. The "
        "fitted head of the present design determines each component's "
        "direction empirically, using the subset of the data on which that "
        "component's own AUC excludes 0.5, and the effect of that single "
        "change is the difference between " + H["hand_auc"] + " and "
        + H["fit99_auc"] + " on the same data. The methodological lesson, "
        "recorded here because it generalises, is that a fusion head's "
        "coefficients must be fitted with their signs free, not chosen and "
        "then scaled.",
    )

    t.h2("2.5", "Self-Consistency and Efficiency")
    t.para(
        "A different approach to detection is to sample the generator multiple "
        "times and measure the agreement between the samples. SelfCheckGPT "
        "[4] is the canonical instance: it prompts the model with the question "
        "and the candidate answer and asks it to generate further samples, "
        "then uses the inconsistency rate as a hallucination score. The "
        "approach requires no ground truth and no external evidence, which "
        "makes it attractive when neither is available.",
    )
    t.para(
        "The cost is multiplicative in the number of samples and it gives no "
        "explanation: a high inconsistency rate tells a user that something "
        "is wrong but not which sentence. More recent work combines selective "
        "inference with annealed decoding to reduce the number of samples "
        "required, improving both latency and the quality of the "
        "self-consistency estimate.",
    )
    t.para(
        "The CRISP pipeline does not use self-consistency as a detection "
        "signal. It does use the related notion in the calibration stage: the "
        "EEDC entropy term of Stage 4 includes a self-consistency component "
        "that quantifies the dispersion of the generator's output "
        "distribution, and it is available as a component of the fusion head "
        "when the generator is local.",
    )

    t.h2("2.6", "Datasets and Black-Box Detection")
    t.para(
        "A significant body of work concerns the construction of "
        "hallucination corpora for training and evaluating detectors rather "
        "than the construction of detectors themselves. AutoHall [7] builds a "
        "generalised hallucination detector's training set from several "
        "fact-check datasets with zero manual labelling. The SHAM-Bench "
        "benchmark [8] formalises span-level hallucination detection as a "
        "task in its own right and diagnoses why sentence-level and span-level "
        "detection behave differently \u2014 a distinction that this project "
        "encounters directly in its own span-level results (Chapter 6).",
    )
    t.para(
        "Teaching with Lies [28] develops a detector trained on systematically "
        "constructed lies, and reports F1 on a held-out test corpus. The "
        "paper's contribution of showing that a detector trained on "
        "systematically constructed falsehoods generalises to naturally "
        "occurring hallucinations is important for this project, because the "
        "Stage 3 classifier of CRISP is trained on a different kind of data "
        "again.",
    )

    t.h2("2.7", "Limits and Applied Systems")
    t.para(
        "Two limitations recur across the literature and both are confirmed "
        "by this project's own measurements.",
    )
    t.para(
        "The first is granularity. A response-level score, however good, does "
        "not localise the error. SHAM-Bench quantifies the difficulty of the "
        "span-level variant, and this project's own span-level attempt "
        "produces a F1 of 0.0455 with 7 true positives, 221 false positives "
        "and 73 false negatives \u2014 a strong negative result, reported in "
        "Chapter 6. The reason is instructive and is discussed there: the "
        "span-level task demands precise boundaries on a continuous token "
        "sequence, whereas the evidence available to the pipeline is "
        "sentence-granular, and a sentence-level judgement of support cannot "
        "be inverted into a token-level judgement of support.",
    )
    t.para(
        "The second is the asymmetry between the synthetic and the "
        "real. HaluEval's hallucinations are generated by instruction-tuned "
        "models and are, by design, fluent; a detector can exploit artefacts "
        "of that generation process that will not be present in a "
        "hallucination produced by a differently tuned model. The gap between "
        "this project's HaluEval score of " + H["hal_auc"] + " and its "
        "RAGTruth score of " + H["rag_auc"] + " \u2014 a difference of "
        + f"{float(H['hal_auc']) - float(H['rag_auc']):.4f}"
        + " in ROC-AUC \u2014 is the clearest available evidence for this "
        "effect within a single system evaluated by a single team.",
    )

    t.h2("2.8", "The Evaluation Protocol as a Contribution")
    t.para(
        "A recurring feature of the literature surveyed above is that "
        "evaluation protocols are rarely described in enough detail to "
        "reproduce, and that where they are described, they differ between "
        "papers in ways that make the reported numbers non-comparable. This "
        "is not a criticism of individual papers, which are written to answer "
        "their own questions, but it does mean that the numbers in the "
        "literature cannot be aggregated into a meta-analysis, and it means "
        "that a new system must choose a protocol rather than inherit one.",
    )
    t.para(
        "Four specific divergences recur, and this project has had to take a "
        "position on each of them.",
    )
    t.bullets([
        "Whether the evaluation is paired. A detector evaluated on a corpus "
        "in which the faithful and the hallucinated responses were produced "
        "under different conditions may be exploiting a difference in "
        "production rather than in faithfulness. HaluEval's construction is "
        "paired by design; RAGTruth's is paired within an instance; many "
        "ad-hoc evaluations are paired by neither.",

        "Whether the detector is shown the reference answer. Some "
        "evaluations score the response against the benchmark's own gold "
        "answer; others require the system to retrieve its own evidence. "
        "These measure different abilities, and the difference is large "
        "enough to dominate the numbers.",

        "Whether the reported figure is a point estimate or an interval. A "
        "ROC-AUC of 0.87 on 1000 rows carries an uncertainty of "
        "roughly \u00b10.03, and a comparison between two such figures is "
        "not decidable without intervals.",

        "Whether the detection threshold was selected on the evaluation data. "
        "A threshold tuned on the same rows on which accuracy is reported "
        "inflates that accuracy, and the inflation is invisible unless the "
        "protocol states which split was used.",
    ])
    t.para(
        "This project takes the strictest available position on each: the "
        "evaluation is paired and length-matched, the detector is never shown "
        "the gold answer, every reported figure carries a bootstrap interval "
        "taken over unique questions, and thresholds are fixed before the "
        "evaluation runs. The cost of this strictness is visible in the "
        "results \u2014 the HaluEval figure of " + H["hal_auc"] + " is "
        "lower than figures reported elsewhere under looser protocols \u2014 "
        "and the benefit is that the figure means what it says.",
    )

    t.h2("2.9", "Positioning of This Work")
    t.para(
        "CRISP occupies the black-box branch of the taxonomy: it does not "
        "read the generator's internals, which means it remains applicable "
        "to generators that offer only API access, and it is complementary "
        "to the white-box family rather than competitive with it.",
    )
    t.para(
        "Within the black-box branch, three properties distinguish this work "
        "from the systems reviewed above. The first is localisation: the "
        "system produces a per-claim verdict, and the claim is the unit of "
        "verification, of mitigation and of explanation. The second is "
        "calibration: the confidence that gates the final decision is "
        "produced by a fitted head whose coefficients and signs are "
        "established empirically, and is then passed through temperature "
        "scaling and isotonic regression. The third is the willingness to "
        "abstain, which is structurally supported by a three-way gate rather "
        "than by a threshold.",
    )
    t.para(
        "The comparison in Chapter 7 is conducted against every published "
        "method that reports a comparable metric on a comparable dataset, "
        "including those against which this system performs worse. That "
        "choice is deliberate. A comparison restricted to the methods this "
        "system happens to beat would misrepresent the state of the art, and "
        "the reviewer's first question is invariably whether the comparison "
        "was complete.",
    )
    t.para(
        "A note on the size of the survey. The architectural survey of "
        "Chapter 7 is scored on a grid of six characteristics, and that grid "
        "has " + N_WORDS + " rows: the retrieval-side family of FLARE, "
        "DRAGIN and IRCoT shares a single row, because none of the three "
        "has a verification stage or an answer-side mitigation and the six "
        "characteristics cannot distinguish them. The finer-grained tables "
        "of the survey split that row and so count nineteen published "
        "papers. Every capability count quoted in this thesis is the "
        "grid's count of " + N_WORDS + ", and the difference between the "
        "two figures is a difference of scope rather than of coverage.",
    )


# ==========================================================================
# CHAPTER 3 -- SYSTEM DESIGN
# ==========================================================================

def chapter_3(t: Thesis):
    t.chapter_heading("System Design")

    t.para(
        "This chapter describes the design of the CRISP pipeline. The "
        "description follows the architecture of the system stage by stage; "
        "Chapter 4 then describes how that architecture is realised in code.",
    )

    t.h2("3.1", "Research Design")
    t.para(
        "The design follows from four commitments established in Chapter 1: "
        "black-box operation, claim-level localisation, statistical "
        "meaningfulness of confidence, and the capacity to refuse.",
    )
    t.para(
        "The black-box commitment excludes every signal that requires the "
        "generator's activations, logits or token probabilities. The "
        "localisation commitment makes the atomic claim the unit of "
        "verification: a claim can be supported or contradicted, and a claim "
        "can be rewritten, but a whole response cannot meaningfully be either. "
        "The meaningfulness commitment requires that the fusion head be "
        "fitted with free signs and free weights on data that is disjoint "
        "from the data on which it is reported. The refusal commitment "
        "requires that the system have a well-defined third state, and that "
        "the evaluation measure whether that state was warranted.",
    )
    t.para(
        "The pipeline is organised as thirteen stages grouped into five "
        "functional blocks: perception (Stages 1 and 2), early assessment "
        "(Stages 3 and 4), claim-level verification (Stages 5 to 8), decision "
        "and repair (Stages 9 to 11), and delivery and measurement (Stages "
        "12 and 13).",
    )

    t.figure(
        GRAPHS / "architecture_diagram.png",
        "The CRISP pipeline. The stage numbering follows the architecture, "
        "not the execution order; the executed order is 1 \u2192 6 \u2192 7 "
        "\u2192 2 \u2192 3 \u2192 5 \u2192 8 \u2192 4 \u2192 9 \u2192 10 "
        "\u2192 11 \u2192 12 \u2192 13.",
        width_in=6.1,
    )

    t.h2("3.2", "Overall System Architecture")
    t.para(
        "The numbering of the stages follows the conceptual architecture, "
        "which proceeds from a user's question to a verified answer. The "
        "execution order differs, and the difference is not incidental. "
        "Retrieval must precede generation, so that the generator is grounded "
        "in evidence rather than answering from parametric memory and being "
        "checked afterwards; claim extraction must precede early detection, "
        "because the early screen operates on claims rather than on the "
        "response as a whole; and entailment must precede confidence "
        "assessment, because the confidence assessment consumes the "
        "entailment verdict. The executed order is therefore 1 \u2192 6 "
        "\u2192 7 \u2192 2 \u2192 3 \u2192 5 \u2192 8 \u2192 4 \u2192 9 "
        "\u2192 10 \u2192 11 \u2192 12 \u2192 13.",
    )

    # ---- 3.2.1 Stage 1 ----
    t.h3("3.2.1", "Stage 1: Query Understanding")
    t.para(
        "The process begins by submitting the user's question to two parallel "
        "entity extractors which simultaneously run and extract any available "
        "information: one uses the zero-shot named entity recognition model "
        "GLiNER (knowledgator/gliner-bi-base-v1.0) while the other is based "
        "on the spaCy statistical NER model. The list of entities obtained "
        "from these two extractors is merged into a single entity list that "
        "identifies persons, places, organisations and domain terms. This then "
        "serves as the basis for Qwen2.5-0.5B-Instruct to query and re-write, "
        "eventually providing back the entity list along with a normalised and "
        "retrieval-friendly query derived from the original user question.",
    )
    t.para(
        "The entities extracted here are precisely the entities that are to be "
        "fact-checked in the following stages, which makes this stage do "
        "double duty: it improves retrieval, and it fixes the vocabulary of "
        "the verification task. If either of the entity recognition models is "
        "not correctly loaded, the stage raises an error rather than allowing "
        "an incomplete entity set to propagate silently.",
    )


    step_figure(t, 1)
    # ---- 3.2.2 Stage 6 ----
    t.h3("3.2.2", "Stage 6: Hybrid Retrieval")
    t.para(
        "The first part of the retrieval stage extracts evidence candidates "
        "from dense retrieval (BGE-M3) and sparse retrieval (BM25) for the "
        "query and for each of the atomic claims. The two ranked lists are "
        "then combined using the reciprocal rank fusion technique. The index "
        "requires one-time creation from a user-provided corpus in JSON Lines "
        "format and, along with its metadata, is subsequently persisted.",
    )
    t.para(
        "Retrieval followed by verification is the grounding formalism of the "
        "SHROOM shared task, in which the accuracy of a generated hypothesis "
        "in the context of given reference texts is the evaluation criterion. "
        "Hybrid retrieval was the methodological choice to cover failure "
        "modes complementary to those of the two types of retriever: dense "
        "retrieval deals well with paraphrase, whereas sparse retrieval keeps "
        "it on par with rare exact terms. The fusion function is given by "
        "Equation 1, where the constant k damps the contribution of the "
        "highest-ranked documents.",
    )
    t.equation(
        "RRF(d) = \u2211  1 / ( k + rank_r(d) )   for r \u2208 {dense, sparse},"
        "   k = 60",
    )
    t.para(
        "The sparse component is BM25 with the standard parameters k\u2081 = "
        "1.5 and b = 0.75, implemented in pure Python so that the index can "
        "be inspected and debugged. The dense component is a FAISS "
        "IndexFlatIP over BGE-M3 embeddings, which is an exact inner-product "
        "index; for the corpus sizes used here this is both faster than an "
        "approximate index and exact, so no approximation error enters the "
        "evaluation.",
    )


    step_figure(t, 6)
    # ---- 3.2.3 Stage 7 ----
    t.h3("3.2.3", "Stage 7: Evidence Reranking")
    t.para(
        "The list of retrieved candidates is resorted by the BGE-Reranker-v2 "
        "cross-encoder, which assesses each claim-evidence pair jointly "
        "rather than separately. Claim-dependent sets of best-ranked passages "
        "of evidence, the size of which depends on a user-defined parameter, "
        "are forwarded to verification. Cross-encoder-based reranking narrows "
        "the evidence set so that the entailment phase operates solely on the "
        "highly relevant passages.",
    )


    step_figure(t, 7)
    # ---- 3.2.4 Stage 2 ----
    t.h3("3.2.4", "Stage 2: Grounded Generation")
    t.para(
        "The normalised query is passed on to the configured generator, which "
        "is Qwen2.5-0.5B-Instruct in this case; Qwen2.5-1.5B-Instruct is "
        "mentioned as one of the possible options, changeable through the "
        "central configuration file without touching the code.",
    )
    t.para(
        "It is by intentional design that generation is conditioned on the "
        "retrieved evidence rather than being left unconditioned. An earlier "
        "iteration of this pipeline generated first and retrieved afterwards, "
        "in order to measure how often the generator could answer from "
        "parametric memory alone. That experiment produced a detection "
        "ROC-AUC of exactly 0.50 \u2014 not by coincidence, but by "
        "arithmetic, since in that configuration the paired faithful and "
        "hallucinated answers were scored identically. The lesson is recorded "
        "here because it generalises beyond this project: a retrieval-grounded "
        "detector evaluated on an ungrounded generator measures nothing.",
    )


    step_figure(t, 2)
    # ---- 3.2.5 Stage 3 ----
    t.h3("3.2.5", "Stage 3: Early Detection")
    t.para(
        "The BGE-M3 encoder embeds the draft answer which, together with an "
        "XGBoost hallucination classifier, produces the probability of "
        "hallucination early, accompanied by a question-answer cosine "
        "similarity signal. Logistic regression is used as a fallback. The "
        "idea of this screening method is based on an experiment from the "
        "SHROOM shared task, which stated that hallucinated generations can "
        "be discovered with higher reliability through a cosine threshold "
        "applied to sentence embedding similarity than by prompting an LLM "
        "for its opinion.",
    )
    t.para(
        "Since this preliminary filtering is performed before retrieval and "
        "verification through expensive methods, the number of downstream "
        "computations incurred by a response depends on it. It is this screen "
        "that largely determines the efficiency of the whole pipeline.",
    )
    t.para(
        "The classifier is trained on a clean, leak-free corpus. An earlier "
        "version of the training set had been inadvertently contaminated in "
        "two ways: 466 of its queries leaked into the evaluation set, and its "
        "own answer-length feature achieved an AUC of 0.82 on the training "
        "split, meaning the classifier was learning response length rather "
        "than hallucination. Both defects were corrected before any reported "
        "measurement was taken; the ablation in Chapter 6 confirms that the "
        "detector is now carrying genuine signal, since removing it drops the "
        "ROC-AUC from " + H["abl_full"] + " to " + H["abl_nodet"] + ".",
    )


    step_figure(t, 3)
    # ---- 3.2.6 Stage 5 ----
    t.h3("3.2.6", "Stage 5: Atomic Claim Extraction")
    t.para(
        "One of the processes in the claim-extraction stage is the "
        "disassembly of a response into different claims through the use of a "
        "two-phase pipeline: rule-based parsing is used for splitting text via "
        "punctuation and connector/clause boundary rules, before a splitting "
        "stage based on the LLM model Qwen2.5-0.5B-Instruct. The latter step "
        "is needed to resolve the remaining complex sentences. The claims "
        "c\u2081, c\u2082, ..., c\u2099 from this process serve as the "
        "self-verifying units of the whole retrieval and verification system.",
    )
    t.para(
        "Decomposing the generated response first and then proceeding with "
        "fact-checking is a widely adopted approach, corresponding to the "
        "fact-extraction-verification paradigm most common in knowledge "
        "detection papers.",
    )
    t.para(
        "A hybrid extraction mode \u2014 in which the rule-based and "
        "LLM-based splitters are combined \u2014 was implemented and then "
        "reverted. It did raise claim granularity, but claim F1 fell to 0.23 "
        "and the verification success rate to 0.22, because short answers were "
        "split into fragments that the entailment model rejected as "
        "premise-free. The pure two-phase design was restored. This is "
        "recorded in the thesis because it is a case where the intuitively "
        "better architecture was measurably worse.",
    )


    step_figure(t, 5)
    # ---- 3.2.7 Stage 8 ----
    t.h3("3.2.7", "Stage 8: NLI Verification")
    t.para(
        "A multilingual natural language inference model (mDeBERTa-v3, "
        "XNLI-trained) is used to compare every generated claim with its list "
        "of reranked evidence sentences, to decide whether a claim is "
        "supported, contradicted or unknown. Using hallucination detectors "
        "based on NLI models is accepted practice in the SHROOM literature, "
        "and NLI-based approaches have done good work mainly in two kinds of "
        "task: model-aware and model-agnostic ones. A claim is considered "
        "verified when it can be inferred and is supported by the evidence; "
        "otherwise, if the evidence contradicts the claim or the relationship "
        "is unclear, the claim is considered untested and therefore not "
        "valid.",
    )
    t.para(
        "This stage contains one of the more consequential design decisions in "
        "the system. Running the entailment model over the entire retrieved "
        "document fails: in the RAGTruth setting, where the retrieved context "
        "is long, a third of all claims became undecidable because the "
        "entailment model was shown the document as a whole rather than the "
        "premises relevant to the claim. The remedy is claim-conditioned "
        "premise selection, described below.",
    )
    t.para(
        "The procedure operates in three steps. First, the retrieved evidence "
        "is split into sentences. Second, each sentence is scored by its "
        "token overlap with the claim, and the highest-scoring sentence is "
        "selected as an anchor. Third, the anchor's immediate neighbours are "
        "added, and the selected sentences are reassembled in document order "
        "so that the premise set given to the entailment model remains "
        "contiguous and readable. The size of the premise set is bounded by "
        "the configuration parameter max_premise_chars, which is set to 1800 "
        "characters.",
    )
    t.equation(
        "P(c) = \u222a { s_i : s_i \u2208 Top_k(c) } ,"
        "   |P(c)| \u2264 max_premise_chars = 1800",
    )


    step_figure(t, 8)
    # ---- 3.2.8 Stage 4 ----
    t.h3("3.2.8", "Stage 4: Confidence Assessment")
    t.para(
        "The original early-stage score undergoes EEDC calibration through "
        "two methods, temperature scaling and isotonic regression. "
        "Calibration produces a confidence score from zero to one which is "
        "the basis for score fusion and also of the gate routing logic.",
    )
    t.para(
        "The calibration addresses a failure mode that has been reported in "
        "the literature: a non-calibrated similarity score of a simple "
        "heuristic model could be effective on synthetic hallucinatory data, "
        "but if the hallucinatory content was generated by a fine-tuned model "
        "the heuristic scores might fail in a very spectacular way. The EEDC "
        "entropy term combines the normalised entropy of the output "
        "distribution with the retrieval and self-consistency signals.",
    )
    t.equation(
        "H = \u2211 p_i ln p_i / ln 3 ,   "
        "EEDC = \u03c6H + \u03b2(1 \u2212 retrieval_top1) + "
        "\u03b3(1 \u2212 self_consistency) + \u03b4",
    )
    t.para(
        "Temperature scaling searches over a grid of 32 values between 0.25 "
        "and 4.0, and isotonic regression is fitted by pool-adjacent-violators "
        "with the PAVA algorithm. Calibration is performed before the "
        "detection experiments, so that every confidence-dependent metric in "
        "the evaluation is computed on calibrated scores.",
    )


    step_figure(t, 4)
    # ---- 3.2.9 Stage 9 ----
    t.h3("3.2.9", "Stage 9: Confidence Fusion")
    t.para(
        "Three signal streams converge in the fusion stage: the "
        "early-detection score from Stage 3, the calibrated confidence from "
        "Stage 4, and the per-claim entailment verdicts from Stage 8. The "
        "overall system confidence and the table of decisions are obtained "
        "through the multi-score fusion weighted module. These two outputs "
        "together are the inputs for the regeneration and the final gate.",
    )
    t.para(
        "In the SHROOM shared task, systems found the strongest of the "
        "documented methods to be in combining different detectors through "
        "ensemble techniques such as voting, probability fusion and "
        "meta-models. For example, these systems achieved a range of 80.6 to "
        "81.7 per cent accuracy by integrating model-agnostic and model-aware "
        "signals, and 77.8 to 79.9 per cent through the combination of a "
        "fine-tuned classifier with an NLI model.",
    )
    t.para(
        "The fusion implemented here differs from a simple weighted sum in one "
        "essential respect: the direction of each component is determined "
        "empirically. Seven components are used \u2014 similarity, "
        "retrieval_top1, nli, coverage, eedc, detector_trust and grounding "
        "\u2014 each standardised across the fitted set, and each carrying a "
        "sign s_c that is +1 if the component's own AUC exceeds 0.5 with a "
        "confidence interval that excludes 0.5, and \u22121 otherwise.",
    )
    t.equation(
        "z_c = ( raw_c \u2212 \u03bc_c ) / \u03c3_c ,   "
        "risk_raw = \u2211 w_c \u00b7 s_c \u00b7 z_c ,   "
        "risk = \u03c3(risk_raw) ,   confidence = 1 \u2212 risk",
    )
    t.para(
        "There is no fitted intercept. The weights w_c are fitted by logistic "
        "regression on a tuning split, and the sign s_c is applied outside the "
        "regression, so that the fitted coefficient cannot silently reverse "
        "the direction that the empirical test established. This separation "
        "is the specific design decision that repaired the sub-chance "
        "performance of the earlier hand-set design.",
    )
    t.para(
        "The decision table produced by this stage assigns each claim one of "
        "three states \u2014 supported, contradicted or unknown \u2014 "
        "together with the evidence that produced the assignment. This table "
        "is the artefact that makes the system's verdict localisable and "
        "explicable, and it is carried forward to Stage 12.",
    )


    step_figure(t, 9)
    # ---- 3.2.10 Stage 10 ----
    t.h3("3.2.10", "Stage 10: Evidence-Guided Regeneration")
    t.para(
        "The regeneration stage checks whether the claims made in the "
        "document are accurate. If any unverified information is encountered, "
        "an evidence-based process of regeneration is automatically used so "
        "that further generations rely only on the verified evidence. This "
        "allows an iterative detect-then-rewrite procedure to be carried out: "
        "elements that have raised issues are redone, while the untouched "
        "options are sorted primarily according to the quality assessment "
        "metric. The detect-then-rewrite strategy has a long history as an "
        "effective method against hallucination generation in machine "
        "translation, and it is the first of the two phases whose objective "
        "is to reconstruct a document after the occurrence of an error.",
    )
    t.para(
        "The implementation is deliberately conservative, and two guards make "
        "it so. The proportionality guard limits the length of a replacement "
        "to the greater of 12 words or four times the length of the claim it "
        "replaces, so that a short claim cannot be replaced by a long "
        "unsupported one. The relevance guard requires that the capitalised "
        "subject entity of the original claim appear in the replacement, so "
        "that the rewrite remains about the same thing. Only claims whose "
        "contradiction strength exceeds a threshold of 0.5 are selected for "
        "rewrite, where the contradiction strength is defined as one minus "
        "the claim's confidence.",
    )
    t.equation(
        "strength(c) = 1 \u2212 confidence(c)  \u2265 0.5  \u21d2  c is "
        "rewritten",
    )
    t.para(
        "The rewritten claims are spliced back into the answer in reverse "
        "order so that the character offsets of the remaining claims remain "
        "valid. The measured effect of this stage on answer quality is "
        "reported in Chapter 6, and it is a null result.",
    )


    step_figure(t, 10)
    # ---- 3.2.11 Stage 11 ----
    t.h3("3.2.11", "Stage 11: Final Verification Gate")
    t.para(
        "A three-way gate sits in front of the final answer, the fused "
        "confidence and the verdict table. In the case of a PASS result, the "
        "verified answer is forwarded for rendering the response; in the case "
        "of a REGEN result, the system returns to Stage 10 for a new "
        "evidence-based attempt, which can be bounded by a maximum retry "
        "count; and if an ABSTAIN result is obtained, it means that the "
        "machine cannot be certain about the content and the answer is not "
        "delivered.",
    )
    t.para(
        "Choosing not to respond is a methodological choice made on purpose. "
        "As deep semantic checks are still not able to distinguish a "
        "hallucination that a fluent speaker would make from a faithful "
        "answer most of the time, it is better to admit the uncertainty than "
        "to show content that has not been checked, especially after the "
        "certified-limit analysis of embedding-based detection.",
    )
    t.table(
        "The three-way final gate: routing rules and the conditions under "
        "which each outcome is selected.",
        ["Outcome", "Condition", "Action"],
        [
            ["PASS", "confidence \u2265 0.70 and unsupported ratio = 0",
             "Return the verified answer with its evidence."],
            ["PASS (degraded)",
             "confidence \u2265 0.50 and retries exhausted",
             "Return the answer, flagged as low confidence."],
            ["REGEN", "confidence < 0.70 and attempts < 2",
             "Return to Stage 10 for an evidence-guided rewrite."],
            ["ABSTAIN", "confidence < 0.30 or unsupported ratio \u2265 0.5",
             "Refuse to answer; report the uncertainty."],
        ],
        widths=[1.25, 2.5, 2.5],
    )


    step_figure(t, 11)
    # ---- 3.2.12 Stage 12 ----
    t.h3("3.2.12", "Stage 12: Explainable Response")
    t.para(
        "All the final answers that have been verified by the system are "
        "recorded along with their complete chain of evidence (provenance), "
        "which comprises the end decision, the table of the system's verdict "
        "for all the statements in the query, references to the supporting "
        "documents, and a record of every one of the thirteen stages of "
        "processing.",
    )
    t.para(
        "The result is published in three formats: JSON for developers, "
        "Markdown for authoring documents and HTML for the web user "
        "interface. Providing explanations as part of the system interface "
        "design is a key requirement, and is one of the essential elements "
        "for serving users, who should at least be given the opportunity to "
        "look at the sources and the claims made during the verification of "
        "content. The fact that these functionalities were one of the reasons "
        "behind the success of previous system-level hallucination checkers "
        "is also a factor that prompted the change.",
    )


    step_figure(t, 12)
    # ---- 3.2.13 Stage 13 ----
    t.h3("3.2.13", "Stage 13: Evaluation Module")
    t.para(
        "This last stage computes a complete metric suite per run and then "
        "aggregates the results into a single metrics file. It comprises three "
        "groups of metrics: the quality of generated texts (BLEU score, "
        "ROUGE-L and BERTScore), the quality of detection (precision, recall, "
        "F1, ROC-AUC, and the three headline rates HRR, CCR and VSR), and the "
        "quality of retrieval (MRR and nDCG). The metrics are compared across "
        "the synthetic, HaluEval and RAGTruth sets. The idea is to "
        "present the generation-overlap metrics (BLEU, ROUGE and so on) and "
        "the detection-level rates side by side, which is a dual evaluation "
        "approach already common in the hallucination-detection community.",
    )


    step_figure(t, 13)
    t.h2("3.3", "Design Rationale of Each Stage")
    t.para(
        "The preceding sections describe what each stage does. This section "
        "records why each stage exists at all \u2014 that is, what failure "
        "mode it addresses, what the alternatives were, and why the chosen "
        "design was preferred. The table is the compressed form of the "
        "argument; the prose above is the long form.",
    )
    t.table(
        "The design rationale of each stage: the failure mode it addresses "
        "and the alternative that was rejected.",
        ["Stage", "Failure mode addressed", "Alternative rejected"],
        [
            ["1 Query Understanding",
             "A question whose phrasing retrieves the wrong passages, because the corpus uses different terminology.",
             "Feeding the raw question to the retriever, which fails on acronym-heavy and paraphrased queries."],
            ["6 Hybrid Retrieval",
             "Dense retrieval misses rare exact terms; sparse retrieval misses paraphrase.",
             "Dense-only retrieval, which was measured to be weaker on the data-to-text task where lexical overlap is decisive."],
            ["7 Evidence Reranking",
             "The entailment model is shown passages that are topically related but do not support the claim.",
             "Feeding the top-k retrieval directly to the entailment model, which lowers precision and raises cost."],
            ["2 Grounded Generation",
             "The generator answers from parametric memory, so there is nothing to check the answer against.",
             "Ungrounded generation, which is arithmetically incapable of producing a discriminative detection score."],
            ["3 Early Detection",
             "A response that is cheap to identify as wrong need not be sent to the expensive verification stages.",
             "Verifying every response in full, which multiplies the cost of the pipeline by the verification cost."],
            ["5 Claim Extraction",
             "A response-level verdict does not localise the error, and cannot be repaired locally.",
             "Verifying the response as a single unit, which is the approach of most cited baselines."],
            ["8 NLI Verification",
             "The claim may be unsupported, contradicted, or simply unaddressed by the evidence.",
             "Embedding similarity alone, which cannot distinguish contradiction from silence."],
            ["4 Confidence Assessment",
             "An uncalibrated similarity score is effective on synthetic data and fails on real hallucinations.",
             "Using the raw score as a probability, which was measured to be miscalibrated on the RAGTruth setting."],
            ["9 Confidence Fusion",
             "No single signal is reliable; each has a failure mode the others cover.",
             "Hand-set fusion weights, which produced a ROC-AUC of " + H["hand_auc"] + " \u2014 below chance."],
            ["10 Evidence-Guided Regeneration",
             "A contradicted claim remains in the answer unless it is rewritten.",
             "Rewriting the whole response, which destroys correct content and cannot be localised."],
            ["11 Final Verification Gate",
             "A system that must always answer will be measured as though it were more reliable than it is.",
             "A binary pass/fail threshold, which has no representation for \u201cI do not know\u201d."],
            ["12 Explainable Response",
             "A score without provenance cannot be acted on by a user.",
             "Returning only the aggregate score, which is the practice of most published detectors."],
            ["13 Evaluation Module",
             "Results that are not written down in a standard artefact cannot be verified or reproduced.",
             "Reporting figures from a run log, which is not auditable after the fact."],
        ],
        widths=[1.25, 2.55, 2.45],
        font_size=9,
    )
    t.para(
        "Two entries in this table warrant a further word, because they are "
        "the two stages whose necessity was established by measurement rather "
        "than by argument.",
    )
    t.para(
        "Stage 8's claim-conditioned premise selection was not in the "
        "original design. The first implementation showed the entailment "
        "model the whole retrieved document, and in the RAGTruth setting "
        "roughly a third of all claims became undecidable \u2014 the model "
        "returned \u201cunknown\u201d for premises it had been given. The "
        "premise-selection procedure was introduced to fix this and its "
        "effect is visible in the RAGTruth verification success rate of "
        + H["rag_vsr"] + ", which is still low but is not the "
        "one-third-undecidable figure the naive implementation produced.",
    )
    t.para(
        "Stage 9's fitted head is the second. The hand-set design was not a "
        "rounding of the fitted one; it was a different and worse system, "
        "and the reason is instructive. The two components that were given "
        "the wrong sign \u2014 entailment strength and coverage \u2014 both "
        "have AUCs near 0.5 in isolation, which makes them look like weak "
        "signals that can safely be assumed to point the intuitive way. They "
        "do not. The detector-trust component, which was also mis-signed, is "
        "decisively below 0.5 at 0.4075, which means that when the Stage 3 "
        "classifier is confident that a response is fine, that confidence is "
        "inversely related to the response being fine. Any fusion scheme "
        "that assumes all signals are positively correlated with quality "
        "will therefore be actively misled by them.",
    )

    t.h3("3.3.1", "Stage 1 revisited: what the entity list is for")
    t.para(
        "It is worth being explicit about the two roles the Stage 1 entity "
        "list plays, because they are easy to conflate and they justify the "
        "stage independently of one another.",
    )
    t.para(
        "The first role is retrieval. Entities extracted from the question "
        "give the rewriter a vocabulary to work with, and the rewritten "
        "query is what the dense retriever embeds. The second role is "
        "verification. The entities are the terms whose factual status the "
        "pipeline will later attempt to establish. A claim that mentions "
        "none of the question's entities is a claim that the retrieval was "
        "unlikely to have been asked to verify, and the pipeline's decision "
        "about it should be treated with corresponding caution.",
    )
    t.para(
        "The second role is not currently exploited as a signal in the fusion "
        "head, and it is listed in Chapter 9 as a possible addition. The "
        "reason it was not added is instructive: the Stage 9 component set "
        "was fixed before the fusion-head fitting was carried out, and "
        "adding a component after the head had been fitted would have "
        "required refitting and re-evaluating the head, which the "
        "time available for the evaluation campaign did not permit. The "
        "lesson is that a component set should be fixed and justified before "
        "fitting rather than expanded afterwards, and it is recorded here "
        "because the temptation to expand it is exactly what produced the "
        "sign errors described in Section 3.3.",
    )

    t.h2("3.4", "Positioning Relative to Internal-State Detection Methods")
    t.para(
        "A carefully considered scope defines the methodology: the pipeline "
        "is black-box during detection. It does not rely on the inner working "
        "details of the generator, which means it remains useful for "
        "generators that offer only API access. This classification positions "
        "the system in the black-box branch of the field's taxonomy and, "
        "simultaneously, it serves as a complement to the white-box family.",
    )
    t.para(
        "The white-box family has additionally developed hallucination "
        "detection methods of its own: eigen-analysis of the covariance of the "
        "hidden state with an additional clipping test-time technique, and "
        "detection methods that rely on attention maps, hidden activations and "
        "logit signals while requiring the model prediction time to be "
        "decreased considerably. In addition, selective inference combined "
        "with annealed decoding may allow faster and better self-consistency "
        "sampling, thus increasing temporal performance and reducing latency.",
    )
    t.para(
        "In the absence of the model's internals, the Stage 3 classifier of "
        "the early-detection system replaces those internally generated "
        "signals of the white-box systems as substitutes. It could be a "
        "direction to test the replacement or reinforcement of Stage 3 by "
        "internal-state scorers when the generator is locally present, and "
        "this is discussed as future work in Chapter 9.",
    )

    t.h2("3.5", "Models and Technical Stack")
    t.para(
        "The pipeline uses computationally heavy components, the majority of "
        "which are pre-trained and obtained from HuggingFace. For entity "
        "recognition, GLiNER (knowledgator/gliner-bi-base-v1.0) [15] is "
        "integrated with a spaCy pipeline for span reconstruction. A single "
        "main model is used: "
        "Qwen2.5-0.5B-Instruct [16], which is versatile enough to serve as "
        "the query rewriter, the generator, the claim splitter and the "
        "evidence-guided rewriter. Optionally, Qwen2.5-1.5B-Instruct may be "
        "used as a supplementary generator.",
    )
    t.para(
        "Two stages require BGE-M3 embeddings [13]: the initial detection and "
        "the dense retrieval. Early hallucination detection is performed by an "
        "XGBoost gradient-boosted decision tree classifier [18]. Beyond the "
        "first stage, the second-stage retrieval relies on BM25 to obtain the "
        "raw matching documents, which are then passed to the BGE-Reranker-v2-M3 "
        "cross-encoder reranking stage [14]. Finally the multilingual "
        "mDeBERTa-v3 [12] carries out the NLI verification for the last stage. "
        "The dense index is a FAISS inner-product index [19], and the two "
        "ranked lists are combined by reciprocal rank fusion [20]. The "
        "numerical stack is NumPy [21], SciPy [22], pandas [23] and "
        "scikit-learn [17], the gradient-boosting library is XGBoost [18], the "
        "deep-learning framework is PyTorch [25], and the figures are drawn "
        "with Matplotlib [24].",
    )
    t.table(
        "The models and checkpoints used by the pipeline, with the role each "
        "plays and the stage in which it is invoked.",
        ["Component", "Checkpoint", "Role", "Stage"],
        [
            ["Dense embedder", "BAAI/bge-m3", "Query/passage embeddings, answer similarity", "3, 6"],
            ["Cross-encoder reranker", "BAAI/bge-reranker-v2-m3", "Claim\u2013evidence reranking", "7"],
            ["NLI model", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "Entailment, contradiction, neutral", "8"],
            ["NER (zero-shot)", "knowledgator/gliner-bi-base-v1.0", "Entity extraction", "1"],
            ["NER (statistical)", "en_core_web_sm (spaCy)", "Entity extraction", "1"],
            ["Generator / rewriter", "Qwen/Qwen2.5-0.5B-Instruct", "Query rewrite, generation, claim split, rewrite", "1, 2, 5, 10"],
            ["Early detector", "XGBoost classifier", "Response-level hallucination probability", "3"],
            ["Fusion head", "Logistic regression", "Weighted, signed fusion of seven components", "9"],
            ["Vector index", "faiss.IndexFlatIP", "Exact inner-product retrieval", "6"],
        ],
        widths=[1.3, 2.0, 2.0, 0.95],
        font_size=10,
    )
    t.para(
        "In terms of runtime, the standard Python models require Python 3.10 "
        "to 3.12, and if real models are required then an NVIDIA GPU with "
        "about 16 GB of memory must be installed; the Qwen checkpoints are "
        "retrieved behind a HuggingFace access token.",
    )

    t.h2("3.6", "Datasets and Experimental Data")
    t.para(
        "The evaluation is based on the combination of three sources of data. "
        "First, a synthetic question-answering task set that was built for the "
        "purpose of internal testing, along with an overall evaluation. "
        "Second, two public benchmarks distributed with the project in JSON "
        "Lines format: the HaluEval question-answering section and RAGTruth, "
        "both of which have been subjected to identical evaluation programs. "
        "HaluEval is a known stress test for detection models, including a "
        "set of realistic, fluent hallucinations generated by instruction-tuned "
        "models instead of artificially created ones. Working with multiple "
        "benchmarks allows results to be presented per benchmark rather than "
        "only as a combined performance figure, each giving an insight into "
        "the results.",
    )
    t.para(
        "In addition, user-provided data that makes up the knowledge base for "
        "retrieval is the third source. Cross-dataset comparison has been made "
        "such that it can be a major kind of experiment as well: once the "
        "benchmark results of a specific system have been generated, the "
        "system automatically provides cross-comparisons next to each other in "
        "a column showing the differences, for every particular query. A "
        "smoke-test run that features dataset selection per dataset is also "
        "available, to do a limited number of runs in the same configuration.",
    )
    t.para(
        "The RAGTruth benchmark was added later, as a third dataset, and "
        "required a different construction. RAGTruth provides triplets of a "
        "query, a context, a response and a set of annotated spans. The "
        "evaluation had to be built so that the faithful and the hallucinated "
        "responses are drawn from the same instance, rather than being paired "
        "by query as in the other two benchmarks; pairing across instances "
        "produced a different kind of confound, discussed in Chapter 5.",
    )
    t.para(
        "When model-specific annotations are necessary besides those "
        "benchmarks, one option is the model-specific hallucination dataset "
        "AutoHall, created from other fact-check datasets, which serves "
        "zero-data labelling.",
    )

    t.h2("3.7", "Evaluation Methodology")
    t.para(
        "Evaluation is done on two levels. The base level evaluation combines "
        "a series of single-query metrics into a whole pipeline-level "
        "average, while the complete evaluation level also includes the "
        "metrics for identification \u2014 precision, recall and F1, as well "
        "as ROC-AUC \u2014 and the three headline detection rates: "
        "hallucination reduction rate (HRR), correction coverage rate (CCR) "
        "and verification success rate (VSR). In total, the complete "
        "evaluation covers ten types of metrics, and checking the metrics "
        "file's aggregate block after each run proves to be an effective way "
        "of ensuring the correctness of the aggregated performance figures.",
    )
    t.para(
        "The five-step testing guideline comprises: (a) the test suite runs "
        "fully; (b) a single command-line query returns the entire record "
        "including answer, per-claim verdicts, evidence, confidence, decision "
        "and step traces; (c) the REST API serves query requests properly; "
        "(d) the web front-end starts in live mode with every checkpoint "
        "loaded; and (e) evaluation runs must output all ten metric "
        "categories without missing any data entry.",
    )
    t.para(
        "The confidence calibration takes place before the ultimate detection "
        "experiments, so that the confidence-dependent metrics are all "
        "calculated on the calibrated scores.",
    )

    t.h2("3.8", "System Delivery and Reproducibility")
    t.para(
        "The delivery comes via three different interfaces which use one "
        "shared pipeline module underneath: a command-line interface that runs "
        "inference, evaluation, calibration and reports; a REST service that "
        "reveals health condition, indexing, query, metrics and structural "
        "information; and a front-end based on a browser that adheres to the "
        "deployable-interface pattern of earlier hallucination-checking "
        "models.",
    )
    t.para(
        "The front-end shows a query interface, verdicts of the claims, a "
        "decision badge, dataset-aware accuracy and evaluation panels, a "
        "cross-dataset comparison card, and a dedicated evaluation dashboard "
        "with the ability to drill down per query. In the case of a missing "
        "component that does not load its checkpoint, the live-or-degraded "
        "status indicator at the top of the page makes it clear, so that a "
        "degraded run is never mistaken for a healthy run.",
    )
    t.para(
        "Reproducibility rests on three mechanisms. Firstly, there is an "
        "automated test suite which replaces the model slots with lightweight "
        "in-memory doubles during testing, so that pipeline logic can be "
        "verified without requiring model downloads and without network "
        "access; production code, however, always calls the real models. "
        "Secondly, component-level failures are reported as degraded states "
        "rather than being silently handled. Thirdly, the full process of "
        "building metrics from an initial repository clone to a populated "
        "repository is captured as a single ordered command sequence, which "
        "serves as the thesis replication recipe.",
    )

    t.h2("3.9", "Project Structure and Modularity")
    t.para(
        "The codebase corresponds directly to the architecture diagram, and "
        "modules for each pipeline stage live in an individual steps package. "
        "Model wrappers, retrieval implementations (FAISS, BM25 and "
        "reciprocal-rank-fusion hybrid) and utility components (chunker, claim "
        "extractor, data loader) are placed in their own packages. A REST "
        "service and a web front-end constitute the interface layer, and "
        "command-line entry points are centralised in a scripts directory.",
    )
    t.para(
        "Shared data types such as evidence and claim records are the contract "
        "that binds the stages. This one-module-per-step discipline is the "
        "foundation of the method that enables the ablation studies of "
        "Chapter 6, because with a configuration change alone any stage can be "
        "reconfigured, bypassed or exchanged. This is exactly the scenario "
        "shown by the per-stage workflow diagrams.",
    )
