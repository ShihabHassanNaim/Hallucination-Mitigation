"""Chapters 4-6: Implementation, Evaluation Methodology, Results."""

from __future__ import annotations

from .content import GRAPHS, AUDIT, CMP, H, META, f
from .renderer import Thesis


# ==========================================================================
# CHAPTER 4 -- IMPLEMENTATION
# ==========================================================================

def chapter_4(t: Thesis):
    t.chapter_heading("Implementation")

    t.para(
        "This chapter describes the implementation of the pipeline described "
        "in Chapter 3. It covers the repository layout, the configuration "
        "discipline, the shared data contracts that bind the stages, the "
        "implementation of the individual stages, and the three delivery "
        "interfaces.",
    )

    t.h2("4.1", "Repository Layout and Module Boundaries")
    t.para(
        "The implementation follows the architecture exactly. Each pipeline "
        "stage is a module within a steps package, and the stage registry in "
        "src/steps/__init__.py is the single place in which the executed "
        "order is declared. Because the registry is declarative, the executed "
        "order is a property of the code rather than an emergent property of "
        "the call graph, which matters because the difference between the "
        "conceptual order and the executed order was the source of two "
        "significant defects during development.",
    )
    t.para(
        "The package structure separates concerns into four layers. The steps "
        "layer contains the thirteen stage implementations and the registry "
        "that orders them. The models layer contains the wrappers around the "
        "HuggingFace checkpoints, so that a stage depends on an interface "
        "rather than on a checkpoint, which is what makes the test suite "
        "possible without downloading several gigabytes of weights. The "
        "retrieval layer contains the FAISS index, the BM25 implementation "
        "and the reciprocal-rank-fusion hybrid. The utility layer contains "
        "the chunker, the claim extractor, the data loader and the metric "
        "implementations.",
    )
    t.table(
        "The repository layout and the responsibility of each package.",
        ["Path", "Responsibility"],
        [
            ["src/steps/", "The thirteen stage implementations and the stage registry that fixes the executed order."],
            ["src/models/", "Checkpoint wrappers exposing a uniform interface to the stages."],
            ["src/retrieval/", "FAISS dense index, pure-Python BM25, and the reciprocal-rank-fusion hybrid."],
            ["src/utils/", "Chunker, claim extractor, data loader, metric implementations, calibration."],
            ["src/api/", "The REST service exposing health, index, query, metrics and structure endpoints."],
            ["src/web/", "The browser front-end implementing the deployable-interface pattern."],
            ["scripts/", "Command-line entry points for inference, evaluation, calibration and reporting."],
            ["config/", "The YAML configuration files, one block per step plus a model registry."],
            ["data/", "Versioned evaluation artefacts, benchmark corpora and generated figures."],
            ["tests/", "The automated test suite, using in-memory model doubles."],
        ],
        widths=[1.6, 4.65],
        font_size=11,
    )

    t.h2("4.2", "Configuration Discipline")
    t.para(
        "The configuration is organised as a single source of truth: all the "
        "hyperparameters are kept in a single YAML configuration file, with "
        "one block per step plus a top-level model registry, and any value "
        "can be overridden by an environment variable with a defined prefix. "
        "This configuration rule is what allows for conducting thorough and "
        "systematic ablation experiments without resorting to code changes.",
    )
    t.para(
        "The distinction between the two layers of this discipline is worth "
        "stating because it is easy to conflate them. A configuration change "
        "can disable a stage, replace a threshold, or alter a hyperparameter. "
        "It cannot introduce a new stage or reorder the executed sequence. "
        "Those are code-level properties, and the difference matters for "
        "reproducibility: a reported ablation is reproducible from the "
        "configuration file alone, whereas a reported ordering change "
        "requires a commit reference.",
    )
    t.table(
        "The principal configuration parameters, with their values and the "
        "stage to which each applies.",
        ["Parameter", "Value", "Stage", "Function"],
        [
            ["top_k", "5", "6, 7", "Number of evidence passages retrieved per query or claim."],
            ["rrf_k", "60", "6", "Damping constant of the reciprocal rank fusion."],
            ["bm25_k1", "1.5", "6", "BM25 term-frequency saturation parameter."],
            ["bm25_b", "0.75", "6", "BM25 length-normalisation parameter."],
            ["max_premise_chars", "1800", "8", "Upper bound on the claim-conditioned premise set."],
            ["temperature_grid", "0.25 to 4.0, 32 points", "4", "Grid searched by temperature scaling."],
            ["contradiction_threshold", "0.5", "10", "Minimum contradiction strength for a rewrite."],
            ["max_replacement_words", "max(12, 4 \u00d7 claim words)", "10", "Proportionality guard on rewrites."],
            ["gate_pass_confidence", "0.70", "11", "Confidence threshold for the PASS outcome."],
            ["gate_abstain_confidence", "0.30", "11", "Confidence threshold for the ABSTAIN outcome."],
            ["gate_abstain_ratio", "0.5", "11", "Unsupported-claim ratio triggering ABSTAIN."],
            ["max_retries", "2", "11", "Maximum number of REGEN attempts."],
            ["bootstrap_resamples", "1000", "13", "Bootstrap resamples for the confidence intervals."],
        ],
        widths=[1.75, 1.55, 0.5, 2.45],
        font_size=10,
    )

    t.h2("4.3", "Shared Data Contracts")
    t.para(
        "The types shared between stages \u2014 the evidence record and the "
        "claim record \u2014 are the contract that binds the stages together. "
        "The claim record carries the claim text, the source span from which "
        "it was extracted, the retrieved evidence relevant to it, the "
        "premise set selected for entailment, the entailment verdict, the "
        "per-claim confidence and the contradiction strength. The evidence "
        "record carries the document identifier, the passage text, its "
        "retrieval scores under both the dense and the sparse rankers, its "
        "reranker score, and its provenance in the source corpus.",
    )
    t.para(
        "Two properties of this contract are load-bearing. First, the "
        "per-claim confidence is stored on the claim rather than on the "
        "response, so that the fused response-level confidence can be "
        "recomputed from the claim records alone, without re-running the "
        "pipeline. Second, the contradiction strength is derived rather than "
        "stored independently, which guarantees that the mitigation stage and "
        "the fusion stage cannot disagree about which claims are "
        "contradicted.",
    )

    t.h2("4.4", "Implementation Notes on Individual Stages")
    t.para(
        "Several stages required implementation decisions that are worth "
        "recording because the alternatives were measured.",
    )

    t.h3("4.4.1", "Entity extraction and query rewriting")
    t.para(
        "The two entity extractors run in parallel and their outputs are "
        "merged and de-duplicated. The merged entity list is passed to the "
        "generator as part of the rewriting prompt, which asks for a "
        "normalised, retrieval-friendly restatement of the original question. "
        "The generator is called with a low temperature, since the rewriting "
        "task requires faithfulness to the original question rather than "
        "diversity.",
    )

    t.h3("4.4.2", "Hybrid retrieval")
    t.para(
        "The dense side embeds the query with BGE-M3 and searches a FAISS "
        "IndexFlatIP. The sparse side scores the query against an in-memory "
        "inverted index using BM25. Both produce ranked lists over the same "
        "document identifiers, and the two lists are combined by reciprocal "
        "rank fusion. The implementation was verified by an exactness check: "
        "for a small corpus, the hybrid result was compared against a "
        "brute-force scoring of every candidate under both rankers, and the "
        "two agreed.",
    )
    t.para(
        "One implementation detail matters for the evaluation. The index must "
        "be loaded explicitly. An early version of the evaluation harness "
        "constructed the pipeline without an index, so that retrieval returned "
        "nothing and the pipeline fell back to a six-document synthetic "
        "corpus. This produced a detection ROC-AUC of 0.50 and was "
        "indistinguishable from a working system from the metrics alone; it "
        "was diagnosed by inspecting the retrieval step's own output rather "
        "than the aggregate score. Every evaluation command in this thesis "
        "therefore passes the index explicitly, and the harness records the "
        "index path in the run's metadata.",
    )

    t.h3("4.4.3", "Claim-conditioned premise selection")
    t.para(
        "The premise selection routine is the most intricate piece of code in "
        "the pipeline. It splits the retrieved evidence into sentences, "
        "scores each sentence against the claim by token overlap, takes the "
        "top-scoring sentence as an anchor, expands the anchor by one "
        "sentence on either side, trims the expanded set to the "
        "max_premise_chars budget, and reassembles the surviving sentences in "
        "document order. The reordering is essential: the entailment model was "
        "trained on contiguous premises, and presenting the same sentences in "
        "score order measurably changed its verdicts.",
    )

    t.h3("4.4.4", "Fusion head fitting")
    t.para(
        "The head is fitted by a two-stage procedure. In the first stage the "
        "direction of each component is determined on the tuning half of the "
        "data, by computing the component's own AUC and comparing its "
        "confidence interval with 0.5. Components whose interval excludes "
        "0.5 are assigned a decisive direction; the remainder are assigned "
        "the direction of their tuning-half AUC, and are flagged as not "
        "decisive. In the second stage a logistic regression is fitted on the "
        "signed, standardised components over the tuning half, and the fitted "
        "model is evaluated on the held-out half.",
    )
    t.table(
        "The component directions established empirically by the fitting "
        "procedure on the 496-row RAGTruth fit.",
        ["Component", "AUC (all rows)", "95% CI", "Sign", "Decisive"],
        [
            ["similarity", "0.5986", "[0.5442, 0.6529]", "+1", "Yes"],
            ["retrieval_top1", "0.5859", "[0.5312, 0.6397]", "+1", "Yes"],
            ["nli", "0.4504", "[0.3988, 0.5058]", "\u22121", "No"],
            ["coverage", "0.4646", "[0.4133, 0.5190]", "\u22121", "No"],
            ["eedc", "0.5247", "[0.4715, 0.5801]", "+1", "No"],
            ["detector_trust", "0.4075", "[0.3502, 0.4626]", "\u22121", "Yes"],
            ["grounding", "0.4809", "[0.4301, 0.5328]", "\u22121", "No"],
        ],
        widths=[1.5, 1.15, 1.4, 0.6, 0.85],
        font_size=11,
        note="The decisive column records whether the component's own confidence interval excludes 0.5. "
             "Three of seven components are decisive; the remaining four are assigned the direction of their "
             "tuning-half AUC and flagged as not decisive.",
    )
    t.para(
        "The table above shows why the sign convention matters so much. The "
        "nli and coverage components both have AUCs near or below 0.5, and "
        "the detector_trust component is decisively below it at 0.4075 with "
        "an interval excluding 0.5. A hand-set scheme that assumes, say, that "
        "a higher entailment score means a more trustworthy answer, gets the "
        "first two signs backwards and produces a head that is worse than "
        "uninformative.",
    )
    t.para(
        "Three head designs were fitted on the same data and compared on the "
        "same held-out split, which isolates the effect of the design "
        "choices from the effect of the data.",
    )
    t.table(
        "The three fusion-head designs, fitted on the same 99 rows and "
        "evaluated on the same held-out split.",
        ["Head design", "Held-out ROC-AUC", "95% CI", "All rows"],
        [
            ["Hand-set weights and signs (the original design)", "0.3889",
             "[0.2143, 0.5863]", H["hand_auc"]],
            ["Unweighted signed sum (all weights = 1)", "0.5868",
             "[0.4000, 0.7555]", H["unw_auc"]],
            ["Fitted logistic regression on signed components", "0.6667",
             "[0.4719, 0.8337]", H["fit99_auc"]],
        ],
        widths=[2.6, 1.1, 1.35, 0.85],
        font_size=10,
        note="The hand-set design is the only one of the three that performs below chance. The unweighted "
             "sum already recovers most of the gap, which shows that the signs were the dominant "
             "error and the weights the minor one.",
    )
    t.para(
        "The progression is informative. The hand-set design, with weights "
        "chosen by intuition and signs assumed rather than measured, scores "
        + H["hand_auc"] + " on all rows \u2014 worse than a coin flip. "
        "Replacing the hand-set weights with equal weights while keeping the "
        "empirically determined signs gives " + H["unw_auc"] + ", which "
        "recovers most of the gap and shows that the signs were the dominant "
        "error and the weights the minor one. Fitting the weights by logistic "
        "regression gives " + H["fit99_auc"] + ". The design lesson is "
        "specific: the sign of a component is a question about the world and "
        "must be measured, whereas the weight of a component is a question "
        "about the particular data and should be fitted but need not be "
        "innovated.",
    )

    t.h3("4.4.5", "Claim rewriting")
    t.para(
        "The rewriting stage selects contradicted claims, asks the generator "
        "to rewrite each one using only the verified evidence, and then "
        "applies the two guards described in Chapter 3. A rewrite is rejected "
        "if it exceeds the proportionality limit or if the capitalised "
        "subject entity of the original claim is absent from it. Rejected "
        "rewrites are counted separately from applied rewrites in the "
        "mitigation report, which is why that report distinguishes "
        + str(H["mit_regen"]) + " regenerated rows, " + str(H["mit_kept"])
        + " kept rows and " + str(H["mit_guarded"]) + " guarded rows.",
    )

    t.h2("4.5", "Algorithm Pseudocode")
    t.para(
        "The two algorithms that carry the most design weight are given "
        "explicitly here, because their exact form is not recoverable from "
        "the prose above and because each of them was revised in response to "
        "a measured failure.",
    )

    t.h3("4.5.1", "Claim-conditioned premise selection")
    t.para(
        "The procedure takes the retrieved evidence for a claim and returns "
        "the premise set to be given to the entailment model.",
    )
    t.para(
        "input: claim c, retrieved passages E, budget B = max_premise_chars\n"
        "  sentences \u2190 split(E)                       # preserve order\n"
        "  scores    \u2190 { token_overlap(s, c) : s \u2208 sentences }\n"
        "  anchor    \u2190 argmax(scores)\n"
        "  selected  \u2190 { anchor } \u222a { neighbours(anchor, \u00b11) }\n"
        "  selected  \u2190 trim_to_budget(selected, B)  # keep document order\n"
        "  output: P(c) = selected",
        size=10, line=1.0, indent=0.3,
    )
    t.para(
        "The three details that matter are the anchor selection, the "
        "neighbour expansion and the reordering. The anchor is the single "
        "sentence with the highest token overlap with the claim, which "
        "grounds the selection in the claim rather than in the passage's own "
        "salience. The neighbour expansion by one sentence on either side "
        "recovers the context in which the anchor usually asserts its "
        "content, since a retrieved sentence frequently states the topic in "
        "one sentence and the fact in the next. The reordering in document "
        "order is what makes the premise set a coherent passage rather than "
        "a bag of sentences, and it measurably changed the entailment "
        "verdicts.",
    )

    t.h3("4.5.2", "Signed fusion head")
    t.para(
        "The head-fitting procedure is described in Section 4.4.4. The "
        "runtime form of the fused score is the following.",
    )
    t.para(
        "input: raw components c\u2081\u2026c\u2087, signs s\u2081\u2026s\u2087, "
        "weights w\u2081\u2026w\u2087\n"
        "  for each component c:\n"
        "      z_c \u2190 (raw_c \u2212 \u03bc_c) / \u03c3_c     # standardised on the fit set\n"
        "  risk_raw \u2190 \u2211 ( w_c \u00b7 s_c \u00b7 z_c )\n"
        "  risk      \u2190 sigmoid(risk_raw)               # no fitted intercept\n"
        "  confidence\u2190 1 \u2212 risk\n"
        "  output: confidence, risk, z",
        size=10, line=1.0, indent=0.3,
    )
    t.para(
        "The absence of a fitted intercept is deliberate. The standardisation "
        "means that the components are centred on the fit set, so that the "
        "fused risk is already centred at the logistic midpoint before the "
        "weights are applied; an intercept would then be free to shift the "
        "decision threshold in a way that is fitted rather than chosen. "
        "Fixing the threshold in this way means that the confidence scale is "
        "anchored to the fit set's own base rate, which is what makes the "
        "calibration of Section 3.2.8 meaningful.",
    )

    t.h2("4.6", "The Gate Implementation")
    t.para(
        "The three-way gate is implemented as a pure function of the fused "
        "confidence, the unsupported-claim ratio and the number of attempts "
        "made so far. It has no side effects and no model calls, which means "
        "it is exhaustively testable and that its behaviour can be reasoned "
        "about independently of the rest of the pipeline.",
    )
    t.para(
        "input: confidence \u03c3, unsupported ratio \u03c1, attempts n\n"
        "  if \u03c3 \u2265 0.70 and \u03c1 = 0        : return PASS\n"
        "  if \u03c3 < 0.30 or \u03c1 \u2265 0.50        : return ABSTAIN\n"
        "  if n < 2                             : return REGEN\n"
        "  if \u03c3 \u2265 0.50                        : return PASS\n"
        "  return ABSTAIN",
        size=10, line=1.0, indent=0.3,
    )
    t.para(
        "The order of the tests matters and is not arbitrary. The confident "
        "PASS is tested first so that a response with no unsupported claims "
        "is never sent for repair. The confident ABSTAIN is tested before the "
        "repair condition, so that a response the pipeline is confident is "
        "hopeless is abandoned rather than rewritten twice. The remaining "
        "cases fall through to the degraded outcomes, which reflect a "
        "pipeline that has run out of attempts and is required to return "
        "something.",
    )

    t.h2("4.7", "Evaluation Harness")
    t.para(
        "The evaluation harness runs the pipeline over a dataset file and "
        "writes a metrics artefact. The artefact contains an aggregate block "
        "and a per-query block, and it records the dataset name, the number "
        "of rows, the number of unique queries, the number of bootstrap "
        "resamples and the confidence intervals for each metric. The harness "
        "also records the configuration that produced the run, so that an "
        "artefact can be traced back to the exact settings under which it "
        "was generated.",
    )
    t.para(
        "Two features of the harness are unusual and deliberate. The first is "
        "that the bootstrap is taken over unique questions rather than over "
        "rows, because the paired construction means that two rows share a "
        "question, and resampling rows independently would understate the "
        "uncertainty. The second is that the harness refuses to write an "
        "aggregate block if any metric category is missing, which enforces "
        "the fifth condition of the testing guideline.",
    )

    t.h2("4.8", "Delivery Interfaces")
    t.para(
        "Three interfaces share one pipeline module. The command-line "
        "interface exposes inference, evaluation, calibration and reporting "
        "subcommands, and accepts the index path, the dataset path, the "
        "configuration path and an output path. The REST service exposes "
        "health, index, query, metrics and structure endpoints, and returns a "
        "degraded status rather than an error when a component's checkpoint "
        "is unavailable. The browser front-end implements the deployable "
        "interface pattern: a query panel, a per-claim verdict list, a "
        "decision badge, dataset-aware accuracy and evaluation panels, a "
        "cross-dataset comparison card, and an evaluation dashboard with "
        "per-query drill-down.",
    )
    t.para(
        "The live-or-degraded indicator at the top of the front-end is a "
        "deliberate design decision. A degraded run that looked identical to "
        "a healthy one would be a source of silently invalid measurements, "
        "and it is precisely this kind of silent degradation that produced the "
        "unindexed-retrieval defect described in Section 4.4.2.",
    )

    t.h2("4.9", "Test Suite")
    t.para(
        "The automated test suite substitutes lightweight in-memory doubles "
        "for the model slots, so that pipeline logic is verified without "
        "model downloads and without network access; production code always "
        "calls the real models. The suite covers the stage registry, the "
        "reciprocal rank fusion against a brute-force reference, the BM25 "
        "implementation, the premise-selection routine, the claim-rewrite "
        "guards, the gate routing logic, the metric implementations and the "
        "artefact schema.",
    )
    t.para(
        "The suite also contains the verification tests for the published "
        "documents, including the comparison tables in Chapter 7. These tests "
        "parse the numbers out of the hand-written prose and compare them "
        "against the artefacts from which they were taken. Every verdict in "
        "those tables is recomputed from the interval rule rather than "
        "trusted, so that a verdict cannot be typed in by hand and left "
        "unchecked. This is the mechanism that caught two incorrect verdicts "
        "during the preparation of this thesis, both of which had been "
        "written in the direction flattering to this work.",
    )


# ==========================================================================
# CHAPTER 5 -- EVALUATION METHODOLOGY
# ==========================================================================

def chapter_5(t: Thesis):
    t.chapter_heading("Evaluation Methodology")

    t.para(
        "The evaluation methodology is described in this chapter before any "
        "result is reported, because several of the results in Chapter 6 are "
        "only interpretable given the design decisions taken here. Three of "
        "those decisions \u2014 the paired construction, the length matching "
        "and the leak-free splits \u2014 were not in the original design; "
        "they were introduced after the initial results demonstrated that "
        "they were necessary.",
    )

    t.h2("5.1", "Datasets and Their Properties")
    t.para(
        "The evaluation uses two public benchmarks. Each is described below "
        "together with the property of it that most affected the design.",
    )
    t.table(
        "The two evaluation benchmarks, their sizes, and the property of "
        "each that most constrained the experimental design.",
        ["Benchmark", "Rows used", "Unique queries", "Defining property"],
        [
            ["HaluEval QA", str(H["hal_n"]), str(H["hal_uq"]),
             "Paired right/hallucinated answers to the same question; hallucinations are model-generated and fluent."],
            ["RAGTruth", str(H["rag_n"]), str(H["rag_uq"]),
             "Real RAG responses with sentence-level span annotations; faithful and hallucinated responses come from the same instance."],
        ],
        widths=[1.15, 0.75, 0.95, 3.4],
        font_size=11,
    )

    t.h3("5.1.1", "HaluEval")
    t.para(
        "HaluEval's question-answering subset supplies, for each question, "
        "both a right answer and a hallucinated answer generated by an "
        "instruction-tuned model. The construction is attractive for a "
        "detection evaluation because the two answers are equally fluent and "
        "answer the same question, so the task is not degenerated into "
        "detecting style. It is also the source of the most serious confound "
        "in this project's evaluation, described in Section 5.3.",
    )

    t.h3("5.1.2", "RAGTruth")
    t.para(
        "RAGTruth contains responses generated with a real corpus in the "
        "loop, annotated at the span level, across question answering, "
        "data-to-text and summarisation. It is the most realistic of the two "
        "benchmarks and the most recent. Two features of its construction "
        "shaped the evaluation.",
    )
    t.para(
        "The first is that the faithful and the hallucinated responses for a "
        "given instance are generated by the same system under the same "
        "conditions; the hallucination is a property of the content, not of a "
        "different prompt. Pairing within an instance, rather than by query, "
        "preserves this property and is what the evaluation does.",
    )
    t.para(
        "The second is context length. RAGTruth's contexts are long, and "
        "this is what made the claim-conditioned premise selection of "
        "Section 3.2.7 necessary. In the first implementation, the entailment "
        "model was shown the whole retrieved context, and roughly a third of "
        "all claims became undecidable.",
    )

    t.h2("5.2", "The Paired Construction")
    t.para(
        "The most important methodological decision in this project is that "
        "every evaluation row belongs to a pair: one faithful response and "
        "one hallucinated response to the same question, differing only in "
        "the label. This is not a convenience; it is a requirement.",
    )
    t.para(
        "Consider a detection score s that is computed from the response "
        "alone and is independent of the label, and suppose the response "
        "length correlates with the label. Then the paired rows differ in "
        "length but not systematically in score, and the ROC-AUC computed over "
        "the paired set collapses towards 0.50. In the extreme, if the "
        "hallucinated and faithful responses are scored identically, the ROC "
        "AUC is exactly 0.50 by arithmetic, whatever the quality of the "
        "detector.",
    )
    t.para(
        "This is not a hypothetical. An early version of the evaluation "
        "harness generated the responses rather than using a benchmark's own "
        "annotations, and produced a detection ROC-AUC of exactly 0.50 on "
        "the HaluEval QA split. The cause was that the generated faithful and "
        "hallucinated responses were scored by a component that could not "
        "distinguish them, since both were generated by the same generator "
        "with the same retrieval and differed only in a label file that the "
        "pipeline never read. The pairing discipline was introduced as the "
        "direct remedy.",
    )
    t.para(
        "The paired design also enables a within-question analysis: for each "
        "question, the number of pairs the detector orders correctly can be "
        "computed, which gives a paired significance test that is far more "
        "sensitive than an unpaired comparison on the same data.",
    )

    t.h2("5.3", "Length Matching and the Length Confound")
    t.para(
        "HaluEval's hallucinated answers are longer than its right answers, "
        "because a model asked to elaborate on a question it does not know "
        "produces more text. A detector that reads only the length of the "
        "response would therefore classify the raw HaluEval split almost "
        "perfectly.",
    )
    t.para(
        "This project measured the effect directly. A single feature "
        "consisting of the response's word count achieves an AUC of 0.7757 on "
        "the question-answering task, 0.7147 on data-to-text and 0.8082 on "
        "summarisation \u2014 that is, on the raw split, word count alone "
        "classifies 94.7 per cent of the rows correctly.",
    )
    t.table(
        "The length confound on the HaluEval QA split, and the effect of "
        "length matching on it.",
        ["Configuration", "Word-count-only accuracy", "Detection ROC-AUC"],
        [
            ["Raw HaluEval QA split", "94.7%", "Not reported (confounded)"],
            ["Length-matched HaluEval QA split", "50.3%", "Reported in Chapter 6"],
        ],
        widths=[2.4, 1.85, 2.0],
        font_size=11,
        note="The matched split is the one used for every HaluEval number reported in this thesis. "
             "The raw-split figures are shown here to document why.",
    )
    t.para(
        "Every HaluEval result reported in this thesis is therefore computed "
        "on a length-matched split, in which the faithful and hallucinated "
        "responses of each pair are drawn to have comparable word counts. The "
        "matching is performed within the pair, and the resulting split "
        "remains exactly balanced: " + str(H["hal_n"]) + " rows, "
        + str(H["hal_uq"]) + " unique queries, "
        "500 faithful and 500 hallucinated.",
    )
    t.figure(
        AUDIT / "audit_03_length_confound.png",
        "The length confound on HaluEval. The left panel shows the word-count "
        "distribution of faithful and hallucinated responses before matching; "
        "the right panel shows the distribution after matching, at which point "
        "word count alone carries essentially no signal.",
        width_in=5.9,
    )

    t.h2("5.4", "Leak-Free Splits")
    t.para(
        "Leakage is the condition in which information unavailable at "
        "evaluation time has influenced the model being evaluated, and its "
        "formulation and detection are treated as a methodological problem in "
        "their own right in the data-mining literature [26]. Applying that "
        "treatment here, three distinct forms of leakage were identified and "
        "eliminated. The "
        "first is query leakage between the detector's training data and the "
        "evaluation set: 466 of the training queries were found to be present "
        "in the evaluation data, and the training corpus was rebuilt without "
        "them. The second is label leakage, in which the detector's training "
        "corpus was constructed so that the label could be inferred from a "
        "field of the row. The training data had an answer-length feature "
        "with an AUC of 0.82, meaning the classifier was learning length "
        "rather than hallucination; the feature was removed and the corpus "
        "rebuilt. The third is index leakage, in which the retrieval index "
        "had not been loaded and the pipeline silently operated on a "
        "six-document synthetic corpus.",
    )
    t.para(
        "The rebuilt training corpus is at data/detector_train_clean.jsonl. "
        "Every evaluation dataset is checked against it before an evaluation "
        "run is permitted to proceed.",
    )

    t.h2("5.5", "Metric Definitions")
    t.para(
        "The evaluation reports three families of metrics. The detection "
        "family measures the ability to rank a faithful response above a "
        "hallucinated one, and is the family on which the comparisons in "
        "Chapter 7 are conducted. The rate family measures what the pipeline "
        "does with a response. The generation family measures whether the "
        "response is a good answer at all.",
    )
    t.table(
        "The detection and rate metrics reported in Chapter 6, with their "
        "definitions.",
        ["Metric", "Definition", "Higher is"],
        [
            ["Detection ROC-AUC", "Area under the ROC curve of the response-level detection score against the label.", "Better"],
            ["Detection AUPR", "Area under the precision\u2013recall curve; the positive class is balanced in every split.", "Better"],
            ["Detection F1", "Harmonic mean of precision and recall at the selected operating threshold.", "Better"],
            ["Detection accuracy", "Fraction of rows correctly classified at the selected threshold.", "Better"],
            ["P@R90", "Precision at the threshold attaining 90 per cent recall.", "Better"],
            ["Spearman \u03c1", "Rank correlation between the detection score and the label across rows.", "Better"],
            ["HRR", "Hallucination reduction rate: the fraction of detected hallucinations removed by the pipeline.", "Better"],
            ["CCR", "Correction coverage rate: the fraction of contradictory claims successfully rewritten.", "Better"],
            ["VSR", "Verification success rate: the fraction of claims whose verification produced a usable verdict.", "Better"],
            ["Claim P / R / F1", "Precision, recall and F1 of claim-level detection against the annotation.", "Better"],
            ["Retrieval recall@5", "Fraction of queries for which a supporting passage appears in the top five.", "Better"],
            ["End-to-end accuracy", "Fraction of questions for which the final response is correct after the full pipeline.", "Better"],
        ],
        widths=[1.5, 4.0, 0.75],
        font_size=11,
    )

    t.h3("5.5.1", "The three headline rates")
    t.para(
        "The three headline rates answer three different questions about the "
        "pipeline's behaviour, and it is worth being explicit about each, "
        "because they can move in opposite directions and reporting only one "
        "of them would be misleading.",
    )
    t.para(
        "The hallucination reduction rate measures how much of the detected "
        "hallucination the pipeline actually removes. The correction coverage "
        "rate measures how many contradicted claims the rewriting stage "
        "successfully repairs, counting a rewrite as successful only if it "
        "passes both guards. The verification success rate measures how many "
        "claims the verification stage is able to decide at all; a claim that "
        "the entailment model cannot resolve contributes to the denominator "
        "but not the numerator.",
    )
    t.para(
        "These three rates have very different values in the results, and the "
        "differences are informative. On HaluEval the CCR is " + H["hal_ccr"]
        + " but the HRR is only " + H["hal_hrr"] + "; on RAGTruth the CCR is "
        + H["rag_ccr"] + " and the HRR is " + H["rag_hrr"] + ". A reader who "
        "saw only one of these numbers would draw a materially wrong "
        "conclusion.",
    )

    t.h2("5.6", "Metric Formulas")
    t.para(
        "The definitions in Table 5.3 are stated in words. This section "
        "gives them formally, because several of the headline rates have "
        "definitions that differ subtly between papers, and the differences "
        "are large enough to change a verdict.",
    )
    t.para(
        "The detection metrics are standard. Given a set of "
        + str(H["hal_n"]) + " responses with labels y\u2081 \u2026 y\u2099 "
        "and detection scores s\u2081 \u2026 s\u2099, the ROC-AUC is the "
        "area under the curve of the true positive rate against the false "
        "positive rate as the threshold varies, and the AUPR is the area "
        "under the precision\u2013recall curve. Because every evaluation "
        "split in this project is exactly balanced, the two areas coincide "
        "up to a constant, and their near-equality in the reported results is "
        "a consistency check on the implementations rather than a "
        "coincidence.",
    )
    t.equation(
        "F1 = 2 \u00b7 P \u00b7 R / (P + R) ,   "
        "\u03c1 = corr( rank(s\u2081\u2026s\u2099) , rank(y\u2081\u2026y\u2099) )",
    )
    t.para(
        "The three headline rates are where the definitions matter most, and "
        "each has a denominator that must be stated explicitly. The "
        "hallucination reduction rate is the fraction of responses that the "
        "pipeline initially flagged as containing hallucination and that "
        "subsequently emerged from the pipeline without any such content. "
        "The correction coverage rate is the fraction of claims that the "
        "entailment stage classified as contradicted and that the rewriting "
        "stage then successfully rewrote, counting a rewrite as successful "
        "only if it passes both the proportionality and the relevance guard. "
        "The verification success rate is the fraction of claims for which "
        "the entailment stage returned a decision \u2014 supported, "
        "contradicted or unknown \u2014 rather than failing to decide.",
    )
    t.equation(
        "HRR = |{ flagged \u2229 repaired }| / |{ flagged }|\n"
        "CCR = |{ contradicted \u2229 rewritten }| / |{ contradicted }|\n"
        "VSR = |{ decided claims }| / |{ all claims }|",
    )
    t.para(
        "The denominators are what make these rates informative. The CCR "
        "denominator counts only the claims the entailment stage actually "
        "classified as contradicted, so a benchmark with few contradicted "
        "claims yields a CCR computed over a small population, while the VSR "
        "denominator counts every claim, whether or not the pipeline could "
        "decide it. Neither denominator can be read in isolation from the "
        "end-to-end accuracy, which is the rate that describes the behaviour "
        "of the whole pipeline.",
    )
    t.para(
        "Two further quantities are used in the comparison chapters. The "
        "per-task ROC-AUC is the same statistic computed on the subset of "
        "rows belonging to one task type, and it is used to localise a "
        "performance figure. The paired win rate is the fraction of "
        "within-question pairs that the detector orders correctly, and it is "
        "used as a more sensitive alternative to the pooled ROC-AUC because "
        "it removes between-question variation entirely.",
    )
    t.equation(
        "paired win rate = |{ pairs with s(faithful) > s(hallucinated) }| / "
        "|{ pairs }|",
    )

    t.h2("5.7", "The Statistical Rule for Comparisons")
    t.para(
        "Every comparison in Chapter 7 is made under one rule, applied "
        "symmetrically to wins and to losses.",
    )
    t.equation(
        "WIN  if  CI_lower(ours) > value_published\n"
        "LOSS if  CI_upper(ours) < value_published\n"
        "TIE  otherwise",
    )
    t.para(
        "The rule is deliberately conservative. A point estimate that exceeds "
        "a published value is not a win if the intervals overlap, because the "
        "comparison then rests on sampling noise. It is applied in both "
        "directions: a loss is declared on exactly the same evidence that "
        "would be required to declare a win. An earlier version of the "
        "comparison machinery applied the rule only on the win side and fell "
        "back on the point estimate for losses, which inflated the apparent "
        "record; and a regular expression that matched the comparison table's "
        "checkmark character counted a tie as a win. Both defects were fixed, "
        "and a test now recomputes every verdict in the published tables from "
        "the row's own numbers, so that neither class of error can recur "
        "silently.",
    )
    t.para(
        "Confidence intervals are computed by bootstrapping with 1000 "
        "resamples. For RAGTruth the bootstrap is taken over the "
        + str(H["rag_uq"]) + " unique questions rather than over the "
        + str(H["rag_n"]) + " rows, because rows sharing a question are not "
        "independent.",
    )

    t.h2("5.8", "The Evaluation Protocol")
    t.para(
        "The evaluation protocol is fixed and is followed in the order given.",
    )
    t.numbered([
        "Build or load the retrieval index explicitly, and record its path "
        "in the run metadata. This step exists because its omission is "
        "silent.",
        "Verify that the evaluation dataset contains no query present in the "
        "detector's training corpus.",
        "Calibrate the confidence stage on the calibration split, before any "
        "detection measurement is taken.",
        "Fit the fusion head on the tuning split and record both the "
        "in-sample and the held-out performance. The in-sample figure is "
        "reported only as such, and is never presented as the headline "
        "number.",
        "Run the full pipeline over the evaluation dataset, producing one "
        "row per evaluation item with its per-claim verdicts.",
        "Compute the metric suite, including the bootstrap confidence "
        "intervals.",
        "Write the metrics artefact and verify it against the schema.",
        "Compare against the published anchors under the interval rule, and "
        "recompute each verdict from the row's own numbers before writing it "
        "into any document.",
    ])

    t.h2("5.9", "Ablation Design")
    t.para(
        "The ablation study removes one component at a time from the full "
        "pipeline and re-measures. It is run on " + str(H["abl_n"])
        + " queries rather than on the full evaluation set, because an "
        "ablation over seven components on a thousand rows is expensive and "
        "the relative ordering of the components is what is being measured, "
        "not their absolute values.",
    )
    t.para(
        "One ablation result deserves advance comment, because it was not "
        "the expected one. Removing the Stage 3 detector is expected to "
        "reduce performance, and it does \u2014 but the hallucination "
        "reduction rate goes up when the detector is removed, from "
        + H["abl_full"] + "'s companion value of 0.30 to 0.40. This is "
        "consistent rather than contradictory: the detector is what "
        "identifies hallucinations in the first place, so without it there "
        "is less for the pipeline to reduce. It is an illustration of why "
        "metrics must be read together rather than ranked individually.",
    )


# ==========================================================================
# CHAPTER 6 -- RESULTS
# ==========================================================================

def chapter_6(t: Thesis):
    t.chapter_heading("Results and Analysis")

    t.para(
        "This chapter reports the measured results of the pipeline. All "
        "numbers are taken from the versioned evaluation artefacts that "
        "accompany the submission; none are quoted from an intermediate "
        "state. Where a result is weak, or where it is worse than a prior "
        "published result, it is reported as such.",
    )

    t.h2("6.1", "Headline Detection Results")
    t.para(
        "The detection results across the two benchmarks are summarised "
        "below. The HaluEval figure is the strongest result in the project; "
        "the RAGTruth figure is the weaker of the two; and the gap between "
        "them is itself discussed in Section 6.6 and Chapter 8.",
    )
    t.table(
        "Headline detection results across the two benchmarks.",
        ["Metric", "HaluEval QA", "RAGTruth"],
        [
            ["Detection ROC-AUC", H["hal_auc"] + " " + H["hal_auc_ci"],
             H["rag_auc"] + " " + H["rag_auc_ci"]],
            ["Detection AUPR", H["hal_aupr"], H["rag_aupr"]],
            ["Detection F1", H["hal_f1"], H["rag_f1"]],
            ["Detection accuracy", H["hal_acc"], H["rag_acc"]],
            ["P@R90", H["hal_p90"], H["rag_p90"]],
            ["Spearman \u03c1", H["hal_rho"], H["rag_rho"]],
            ["Rows / unique queries",
             f"{H['hal_n']} / {H['hal_uq']}",
             f"{H['rag_n']} / {H['rag_uq']}"],
        ],
        widths=[1.55, 1.9, 1.9],
        font_size=11,
        note="Bracketed values are 95% bootstrap confidence intervals, taken over unique questions "
             "with 1000 resamples.",
    )
    t.figure(
        GRAPHS / "headline_detection.png",
        "The two headline detection results, with their confidence "
        "intervals, shown against the 0.5 chance line and against the "
        "published values to which each is compared in Chapter 7.",
        width_in=6.0,
    )
    t.para(
        "The ordering across the two datasets is consistent with what the "
        "benchmarks are designed to reward. HaluEval supplies pre-generated "
        "paired answers in which the only difference between the two members "
        "of a pair is faithfulness, and it is therefore the friendlier of "
        "the two to any detector. RAGTruth supplies real responses produced "
        "with a corpus in the loop, and its ROC-AUC of " + H["rag_auc"]
        + " should be read against that background rather than as a "
        "like-for-like comparison with the other.",
    )

    t.h2("6.2", "The ROC Curve and Threshold Behaviour")
    t.para(
        "The receiver operating characteristic of the HaluEval detection "
        "score is shown in Figure 6.2. The curve is well away from the "
        "diagonal throughout, and the area under it is " + H["hal_auc"]
        + ". The area under the precision\u2013recall curve, " + H["hal_aupr"]
        + ", is close to the area under the ROC curve, which is expected on "
        "a balanced split and is a useful check that the two implementations "
        "agree.",
    )
    t.figure(
        GRAPHS / "roc_curve.png",
        "Receiver operating characteristic for the two evaluation "
        "benchmarks, with the diagonal chance line marked.",
        width_in=5.6,
    )
    t.para(
        "The precision at 90 per cent recall is " + H["hal_p90"] + " on "
        "HaluEval and " + H["rag_p90"] + " on RAGTruth. The corresponding "
        "RAGTruth value is not a detector defect but a property of the "
        "operating regime: at 90 per cent recall the detector is being asked "
        "to catch almost every hallucination, and the false positives that "
        "this incurs are what the number measures.",
    )

    t.h2("6.3", "Calibration")
    t.para(
        "Calibration is assessed by comparing the stated confidence against "
        "the observed frequency of correctness in each confidence bin. The "
        "calibrated curve is close to the diagonal on HaluEval, which is the "
        "dataset on which the calibration set and the evaluation set are "
        "most similar in construction.",
    )
    t.figure(
        GRAPHS / "confidence_calibration.png",
        "Reliability of the calibrated confidence on the two benchmarks. "
        "The diagonal is perfect calibration; deviation below the diagonal "
        "indicates overconfidence.",
        width_in=5.6,
    )
    t.para(
        "The calibration result should be read in conjunction with the "
        "abstention behaviour in Section 6.5. A well-calibrated detector "
        "that is nevertheless always asked to answer will still be "
        "overconfident in practice, because the abstention mechanism is what "
        "converts calibration into useful behaviour.",
    )

    t.h2("6.4", "Ablation Study")
    t.para(
        "The ablation removes the Stage 3 early-detection classifier from the "
        "pipeline and re-measures on " + str(H["abl_n"]) + " queries. The "
        "full pipeline achieves a detection ROC-AUC of " + H["abl_full"]
        + "; without the detector, " + H["abl_nodet"] + ". The detector is "
        "therefore carrying substantial signal, and the pipeline is not "
        "relying on the retrieval and verification stages alone.",
    )
    t.table(
        "Ablation of the Stage 3 early-detection classifier, on "
        + str(H["abl_n"]) + " queries.",
        ["Configuration", "ROC-AUC", "F1", "Accuracy", "HRR"],
        [
            ["Full pipeline", H["abl_full"], "0.7300", "0.7300", "0.30"],
            ["Without Stage 3 detector", H["abl_nodet"], "0.6341", "0.6250", "0.40"],
            ["Difference", "0.1799", "0.0959", "0.1050", "\u22120.10"],
        ],
        widths=[2.15, 1.05, 0.9, 1.05, 0.85],
        font_size=11,
        note="The hallucination reduction rate rises when the detector is removed because there is then "
             "less detected hallucination available to reduce. The metrics must be read together.",
    )
    t.figure(
        GRAPHS / "ablation_study.png",
        "Ablation of the Stage 3 early-detection classifier. The remaining "
        "components of the pipeline recover part, but not all, of the lost "
        "discrimination.",
        width_in=5.5,
    )

    t.h2("6.5", "The Three-Way Gate and the Cost of Abstention")
    t.para(
        "The gate routes each response to PASS, REGEN or ABSTAIN. Its "
        "behaviour differs across the two benchmarks, and the differences "
        "follow directly from the evidence available in each.",
    )
    t.table(
        "Gate behaviour and the three headline rates across the benchmarks.",
        ["Quantity", "HaluEval", "RAGTruth"],
        [
            ["HRR (hallucination reduction rate)", H["hal_hrr"], H["rag_hrr"]],
            ["CCR (correction coverage rate)", H["hal_ccr"], H["rag_ccr"]],
            ["VSR (verification success rate)", H["hal_vsr"], H["rag_vsr"]],
            ["End-to-end accuracy", H["hal_e2e"], H["rag_e2e"]],
            ["Mean latency per query (s)", H["hal_lat"], H["rag_lat"]],
        ],
        widths=[2.55, 1.4, 1.4],
        font_size=11,
    )
    t.para(
        "The two columns differ in the way the benchmarks differ. HaluEval "
        "pairs a faithful answer with a model-generated hallucinated "
        "counterpart to the same question, so the pipeline is asked to "
        "separate two answers that were written to be equally fluent. "
        "RAGTruth contains responses produced by a real system with a real "
        "corpus in the loop, so the unsupported content is distributed "
        "across sentences rather than concentrated in a single answer, and "
        "the detector has to find it in situ. The higher RAGTruth claim "
        "precision reported in Section 6.8 follows directly from that "
        "difference.",
    )
    t.figure(
        GRAPHS / "decision_distribution.png",
        "The distribution of PASS, REGEN and ABSTAIN decisions across the "
        "two benchmarks, showing how strongly the abstention behaviour "
        "depends on the availability of evidence.",
        width_in=5.7,
    )

    t.h2("6.6", "The Mitigation Result")
    t.para(
        "The evidence-guided rewriting stage was evaluated on "
        + str(H["mit_rows"]) + " rows spanning " + str(H["mit_uq"])
        + " unique queries. Of these, " + str(H["mit_regen"]) + " rows were "
        "regenerated, " + str(H["mit_kept"]) + " were left unchanged and "
        + str(H["mit_guarded"]) + " had a rewrite proposed but rejected by "
        "one of the two guards.",
    )
    t.table(
        "The measured effect of evidence-guided rewriting on answer quality.",
        ["Metric", "Pre", "Post", "\u0394", "95% CI", "p"],
        [
            ["BERTScore (all rows)", "0.9011", "0.9009", H["mit_delta"],
             H["mit_delta_ci"], H["mit_p"]],
            ["ROUGE-L (all rows)", "0.3943", "0.3948", H["mit_rouge_delta"],
             "[\u22120.0005, 0.0017]", H["mit_rouge_p"]],
        ],
        widths=[1.55, 0.75, 0.75, 0.85, 1.35, 0.65],
        font_size=11,
        note="The confidence interval for the BERTScore difference contains zero, so the change is not "
             "distinguishable from zero at the 95 per cent level. The rewriting stage improves the "
             "measured quality of the claims it rewrites, but that improvement is not large enough to "
             "move the corpus-level aggregate.",
    )
    t.para(
        "This is a null result and it is reported as one. The rewriting stage "
        "does what it was designed to do \u2014 of the rows it processed, "
        "those it rewrote did improve on the BERTScore measure \u2014 but "
        "the number of rows it processed is small, and the corpus-level "
        "aggregate is dominated by the rows it left alone. The correct "
        "conclusion is not that rewriting fails but that this pipeline "
        "rewrites too rarely for corpus-level rewriting to register, and that "
        "the trigger condition \u2014 a contradiction strength of at least "
        "0.5 \u2014 is too conservative.",
    )
    t.figure(
        GRAPHS / "custom_metrics.png",
        "The three headline rates and the detection metrics for the two "
        "benchmarks, showing the behaviour that the null mitigation result "
        "must be interpreted against.",
        width_in=6.0,
    )

    t.h2("6.7", "Retrieval Quality")
    t.para(
        "Retrieval recall@5 is " + H["hal_recall5"] + " on the HaluEval "
        "evaluation and " + H["rag_recall5"] + " on RAGTruth. The high "
        "figures mean that the retriever is not the bottleneck in either "
        "setting: when the corpus contains a supporting passage, the "
        "pipeline finds it.",
    )
    t.para(
        "Precision@5 is much lower \u2014 " + "0.1988 on HaluEval and "
        + "0.1804 on RAGTruth \u2014 because the corpora contain many "
        "passages that are topically related but do not support the specific "
        "claim at hand. This gap between high recall and low precision is "
        "the reason the reranking stage and the claim-conditioned premise "
        "selection both exist: the retriever is good at recalling, and the "
        "work of the downstream stages is to make the retrieved set "
        "sufficiently precise for entailment to be decidable.",
    )

    t.h2("6.8", "Claim-Level Detection")
    t.para(
        "The detection figures reported so far are response-level: they "
        "measure whether the pipeline can tell a faithful response from a "
        "hallucinated one. The claim-level figures measure the finer "
        "question of whether the pipeline identifies which part of the "
        "response is unsupported.",
    )
    t.table(
        "Claim-level detection and retrieval quality across the benchmarks.",
        ["Metric", "HaluEval", "RAGTruth"],
        [
            ["Claim precision", H["hal_claim_p"], H["rag_claim_p"]],
            ["Claim recall", H["hal_claim_r"], "0.6260"],
            ["Claim F1", H["hal_claim_f1"], "0.7269"],
            ["Retrieval recall@5", H["hal_recall5"], H["rag_recall5"]],
            ["Retrieval precision@5", "0.1988", "0.1804"],
            ["Retrieval MRR", "0.9940", "0.9020"],
            ["Retrieval nDCG", "0.9940", "0.9020"],
        ],
        widths=[1.9, 1.65, 1.65],
        font_size=11,
    )
    t.para(
        "The two benchmarks show opposite profiles, and the contrast is "
        "instructive. On HaluEval the claim-level detector has a high recall "
        "of " + H["hal_claim_r"] + " and a much lower precision of "
        + H["hal_claim_p"] + ", giving an F1 of " + H["hal_claim_f1"] + ". "
        "In other words, the pipeline finds most of the unsupported claims "
        "but flags considerably more claims than are actually unsupported. "
        "On RAGTruth the pattern reverses: precision is " + H["rag_claim_p"]
        + " and recall 0.6260, so the claims the pipeline flags are almost "
        "always genuinely unsupported, but it misses a substantial fraction "
        "of them.",
    )
    t.para(
        "The HaluEval profile is a direct consequence of the length-matching "
        "and pairing design. The matched split was constructed so that the "
        "two members of each pair differ in faithfulness while resembling "
        "each other in length and topic, which means that the unsupported "
        "claims in the hallucinated member are genuinely the ones that "
        "distinguish it from its faithful counterpart. The pipeline finds "
        "them. It also flags claims that are supported, because a claim that "
        "is true but unusually specific, or one whose supporting passage is "
        "not among the retrieved five, presents the same surface "
        "characteristics as a false one.",
    )
    t.para(
        "The RAGTruth profile is the more useful behaviour for a deployment. "
        "A claim-level detector that is conservative in the sense of rarely "
        "flagging a supported claim is far more useful to a user than one "
        "that flags every claim it is shown, because the latter would make "
        "the provenance record useless. The cost is that a substantial "
        "fraction of unsupported claims escape detection, which is "
        "consistent with the response-level ROC-AUC of " + H["rag_auc"]
        + " on the same benchmark.",
    )

    t.h2("6.9", "Span-Level Detection: A Negative Result")
    t.para(
        "The pipeline was also evaluated at span level, that is, at the "
        "granularity at which RAGTruth's annotations identify the exact "
        "unsupported token ranges. The result is a strong negative: an F1 of "
        "0.0455 with 7 true positives, 221 false positives and 73 false "
        "negatives.",
    )
    t.para(
        "The cause is structural rather than a matter of tuning. The "
        "evidence available to the pipeline is sentence-granular, so the "
        "strongest judgement it can make is a sentence-level one. A "
        "sentence-level judgement of support cannot be inverted into a "
        "token-level judgement of support: within a sentence that is partly "
        "supported and partly not, there is no information in the pipeline's "
        "output about where the boundary falls. The false-positive count of "
        "221 is the signature of exactly this \u2014 the pipeline is "
        "correctly identifying that most flagged sentences contain an "
        "unsupported span, and then attributing the whole sentence to it.",
    )
    t.para(
        "This result is consistent with the diagnosis in SHAM-Bench, which "
        "treats span-level detection as a distinct task from sentence-level "
        "detection for exactly this reason.",
    )

    t.h2("6.10", "Latency")
    t.para(
        "Mean end-to-end latency is " + H["hal_lat"] + " seconds per query "
        "on HaluEval and " + H["rag_lat"] + " seconds on RAGTruth. The "
        "RAGTruth figure is substantially higher because of the longer "
        "contexts, which increase the cost of the reranking and entailment "
        "stages.",
    )
    t.para(
        "It is important to be precise about what this figure covers. The "
        "measured band of 0.8 to 13.8 seconds covers the retrieval and "
        "verification path only. The evaluation harness supplies the answer "
        "to the pipeline, so the generation stage is not executed and its "
        "cost is not included. The figure is therefore a lower bound on the "
        "cost of a complete deployment, and it is reported here as such.",
    )
    t.figure(
        GRAPHS / "latency_breakdown.png",
        "Latency breakdown by stage. The verification path dominates; the "
        "early-detection screen is the cheapest part of the pipeline and is "
        "the stage that determines whether the expensive stages are run at "
        "all.",
        width_in=5.7,
    )

    t.h2("6.11", "The Audit Trail")
    t.para(
        "The results in this chapter are accompanied by an audit trail that "
        "records the checks applied to them. This is not a formality: several "
        "of the defects described in Chapter 4 were invisible in the "
        "aggregate metrics and were found only by these checks.",
    )
    t.figure(
        AUDIT / "audit_07_ledger.png",
        "The audit ledger. Each row records a check that was applied to the "
        "evaluation artefacts, the check that failed or passed, and the "
        "action taken.",
        width_in=6.0,
    )
    t.para(
        "The pipeline itself, with the ordering, the leak checks and the "
        "length matching made explicit, is shown in Figure 6.10.",
    )
    t.figure(
        AUDIT / "audit_01_pipeline.png",
        "The evaluation pipeline in full, showing the stages of data "
        "preparation, the leak checks and the measurement sequence.",
        width_in=6.0,
    )
    t.figure(
        AUDIT / "audit_08_datasets.png",
        "The composition of the two evaluation datasets after preparation: "
        "row counts, unique queries, label balance and the effect of length "
        "matching.",
        width_in=6.0,
    )
    t.figure(
        AUDIT / "audit_05_ci_separation.png",
        "The separation of the confidence intervals used for every comparison "
        "in Chapter 7. A published value that falls outside the interval "
        "produces a decisive verdict; a value inside it produces a tie.",
        width_in=6.0,
    )
    t.figure(
        AUDIT / "audit_04_mitigation.png",
        "The mitigation study in detail: how many rows were rewritten, how "
        "many were kept, how many rewrites were rejected by the guards, and "
        "the measured effect on answer quality.",
        width_in=6.0,
    )

    t.h2("6.12", "Per-Task Breakdown on RAGTruth")
    t.para(
        "RAGTruth contains three task types, and the pipeline's performance "
        "on them differs substantially. Reporting the pooled figure alone "
        "would conceal the most diagnostically useful result in this "
        "chapter.",
    )
    per_task = CMP["ours"]["per_task_roc_auc"]
    t.table(
        "Detection ROC-AUC on RAGTruth, by task type.",
        ["Task type", "n rows", "ROC-AUC", "Interpretation"],
        [
            ["Question answering", str(per_task["QA"]["n"]), f(per_task["QA"]["roc_auc"]),
             "A single factual answer that the corpus states plainly; the task the pipeline is designed for."],
            ["Summarisation", str(per_task["Summary"]["n"]), f(per_task["Summary"]["roc_auc"]),
             "Partial support is common; the boundary between supported and unsupported is a matter of degree."],
            ["Data-to-text", str(per_task["Data2txt"]["n"]), f(per_task["Data2txt"]["roc_auc"]),
             "The claim is about the structure of data rather than a stated fact; support is rarely a single passage."],
        ],
        widths=[1.35, 0.7, 0.85, 3.35],
        font_size=11,
        note="The pooled RAGTruth ROC-AUC of " + H["rag_auc"] + " is close to the weighted mean of these "
             "three figures, as it must be.",
    )
    t.para(
        "The ordering is interpretable. Question answering is the case the "
        "pipeline's architecture is best suited to: there is a question with "
        "a specific answer, a corpus that states that answer, a claim that "
        "asserts it, and an entailment model that can decide whether the "
        "passage supports the claim. Data-to-text is the case it is least "
        "suited to: the claim describes the contents of a table, and no single "
        "passage asserts the description, so the entailment model is being "
        "asked to verify a summary against a set of facts that individually "
        "imply it. This is not a defect of the entailment model but a "
        "mismatch between the task's notion of support and the entailment "
        "model's notion of it.",
    )
    t.para(
        "The practical implication is that a deployment of this pipeline "
        "should be expected to work well on factual question answering and "
        "poorly on structural description, and should not be applied to the "
        "latter without a different verification mechanism.",
    )

    t.h2("6.13", "Summary of Results")
    t.para(
        "The results of this chapter are summarised below, together with an "
        "assessment of each.",
    )
    t.table(
        "Summary of the measured results and their assessment.",
        ["Result", "Measurement", "Assessment"],
        [
            ["HaluEval detection ROC-AUC", H["hal_auc"] + " " + H["hal_auc_ci"], "Strong; the project's best result."],
            ["RAGTruth detection ROC-AUC", H["rag_auc"] + " " + H["rag_auc_ci"], "Modest; below three published methods, as Chapter 7 shows."],
            ["Held-out RAGTruth ROC-AUC", H["held_auc"] + " " + H["held_auc_ci"], "The honest in-sample-free figure; the in-sample value is 0.6609."],
            ["Stage 3 ablation", H["abl_full"] + " \u2192 " + H["abl_nodet"], "The detector carries substantial signal."],
            ["Mitigation BERTScore \u0394", H["mit_delta"] + " " + H["mit_delta_ci"], "Null; p = " + H["mit_p"] + "."],
            ["Span-level F1", "0.0455", "Negative result; explained structurally in Section 6.8."],
            ["Retrieval recall@5", H["hal_recall5"] + " / " + H["rag_recall5"], "Retrieval is not the bottleneck."],
            ["End-to-end accuracy", H["hal_e2e"] + " / " + H["rag_e2e"], "Limited by the abstention policy, which is deliberately conservative."],
        ],
        widths=[1.85, 1.9, 2.5],
        font_size=11,
    )
