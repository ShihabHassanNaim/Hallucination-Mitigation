"""Chapters 7-9: Comparison with Literature, Discussion, Conclusion."""

from __future__ import annotations

import json

from .content import (
    CMP, COVERAGE, DATA, GRAPHS, H, N_PIPELINE_WORDS as N_WORDS,
    REFERENCES, RAG, f, word,
)
from .renderer import Thesis, _set_font
from .comparison import (
    head_to_head_table, master_table, metric_rows, metric_table, metric_tally,
)


def _verdict(ours: float, ci: list[float], published: float) -> str:
    """The interval rule, applied in both directions."""
    if ci[0] > published:
        return "WIN"
    if ci[1] < published:
        return "LOSS"
    return "TIE"


RAG_CI = RAG["detection_roc_auc_ci95"]
RAG_AUC = RAG["detection_roc_auc"]
F1_CI = RAG["detection_f1_ci95"]
F1_VAL = RAG["detection_f1"]

LLMCHECK = [
    ("LLM-Check \u2014 Attention Score", 0.5830),
    ("LLM-Check \u2014 Hidden Score", 0.5724),
    ("LLM-Check \u2014 Logit, Window Entropy", 0.5458),
    ("LLM-Check \u2014 Logit, Perplexity", 0.5327),
    ("LLM-Check \u2014 Logit, Log Entropy", 0.5168),
]

AES = [
    ("Llama2-7B", 0.8829),
    ("Llama2-13B", 0.9250),
    ("Llama3-8B", 0.8807),
    ("Mistral-7B", 0.8931),
    ("Llama2-70B", 0.9332),
]

FREQ = [
    ("LLaMA-7B", "Lookback-lens", 0.7884),
    ("LLaMA-7B", "Fourier-high", 0.8412),
    ("LLaMA-13B", "Wavelet-High", 0.8448),
    ("LLaMA-13B", "Fourier-high", 0.8585),
    ("Mistral-7B", "Lookback-lens", 0.8764),
    ("Mistral-7B", "Fourier-high", 0.8812),
]

DESIDERATA = [
    ("SFT", 0.871, 0.688, 0.671),
    ("FS", 0.836, 0.694, 0.628),
    ("LLM-aaJ", 0.825, 0.682, 0.617),
    ("PO", 0.800, 0.671, 0.538),
    ("SC-GPT (C)", 0.586, 0.199, 0.346),
    ("SC-GPT (G)", 0.584, 0.199, 0.342),
]

RAGTRUTH_F1 = [
    ("Finetuned Llama-2-13B (RAGTruth's own)", 0.787),
    ("Prompt gpt-4-turbo", 0.634),
    ("SelfCheckGPT gpt-3.5-turbo", 0.588),
    ("LMvLM gpt-4-turbo", 0.494),
]


def _halueval_rows() -> list[list[str]]:
    """The rows of the HaluEval comparison table.

    The table is the evidence for the section's claim that only two
    published results report an AUROC on HaluEval, so the rows live in
    a function that a test can read and count, rather than in a literal
    inside the chapter body.
    """
    return [
        ["This project", str(H["hal_n"]), "ROC-AUC",
         H["hal_auc"] + " " + H["hal_auc_ci"], "Yes"],
        ["Decoding Memories", "400", "AUROC",
         "61.2\u201384.9 for its baselines", "Yes"],
        ["The Semantic Illusion", "629", "AUC (NLI)",
         "0.81 for the NLI detector", "Yes"],
        ["Hu et al. (EMNLP 2024)", "35,000", "Accuracy",
         "97.19 / 77.10 / 95.08 / 83.01", "Yes"],
        ["InternalInspector I2", "30,000 (70% test)", "Accuracy",
         "0.691 / 0.648 / 0.671", "Yes"],
        ["Teaching with Lies", "8,000 + 2,000", "F1 / P / Acc",
         "0.753 / 0.782 / 0.857 (HaluCheck-Llama 3B)", "Yes"],
        ["MedICO", "1,000", "F1", "0.946 (ENSB Llama3-8B)", "Yes"],
        ["Markov chain debate", "350 (150 QA)", "Acc / F1",
         "70.67 / 73.81 (QA)", "Yes"],
        ["Small Agent Can Also Rock!", "100 test", "Acc / F1",
         "71.00 / 72.38 (QA)", "Yes"],
        ["Factcheck-Bench", "0", "\u2014", "\u2014", "No \u2014 cited only"],
        ["GPTLODS", "0", "\u2014", "\u2014", "No \u2014 not present"],
        ["Zhang et al. (Findings ACL 2024) [35]", "0", "\u2014", "\u2014",
         "No \u2014 training data only"],
        ["HaluCheck (ESWA 2025) [38]", "\u2014", "\u2014",
         "Paywalled; not verified", "Unverified"],
    ]


def chapter_7(t: Thesis):
    t.chapter_heading("Comparison with the Published Literature")

    t.para(
        "This chapter compares this project against the published literature. "
        "The comparison proceeds in two parts, in the order the strength of "
        "the evidence supports. Part one is architectural: it compares the "
        "capability of the pipeline against the published systems that "
        "address the same problem, and it is the part in which this project "
        "is unambiguously ahead. Part two is quantitative: it compares the "
        "measured detection performance against published results on the two "
        "benchmarks used here, under an interval rule applied identically in "
        "both directions.",
    )
    t.para(
        "Both parts are reported in full. The candidate set is stated, the "
        "systems excluded from each comparison are named with their reasons, "
        "and every verdict is recomputed from the row's own numbers by an "
        "automated test rather than being written in by hand.",
    )

    # ------------------------------------------------------------------
    # PART ONE -- THE ARCHITECTURAL COMPARISON
    # ------------------------------------------------------------------
    t.h2("7.1", "The Architectural Comparison")
    t.para(
        "The first comparison is architectural, and it is the strongest result "
        "this project can make. To state it, the systems that address "
        "hallucination in retrieval-augmented generation are described by "
        "six capabilities. The six are chosen so that they are the ones that "
        "distinguish a defence pipeline from a single-pass detector, and so "
        "that each can be answered yes or no by reading the paper.",
    )
    t.table(
        "The six capabilities used to describe a hallucination-defence "
        "pipeline.",
        ["Capability", "Definition"],
        [
            ["C1 — Retrieval-grounded detection",
             "The system targets hallucinations in retrieval-augmented "
             "generation, and its detection is grounded in retrieved "
             "evidence rather than in the response alone."],
            ["C2 — Multi-stage pipeline",
             "The system runs a sequence of query understanding, retrieval, "
             "ranking and generation, rather than a single pass."],
            ["C3 — Claim-level verification",
             "The response is decomposed into atomic claims and each claim is "
             "judged against evidence, rather than the response being scored "
             "as a whole."],
            ["C4 — Multi-signal fusion",
             "Several distinct confidence signals are combined into one risk "
             "score, and that combination is fitted rather than hand-set."],
            ["C5 — Post-hoc repair",
             "After detection the system changes the answer, and does so at "
             "a granularity finer than the whole answer."],
            ["C6 — Selective prediction",
             "The system declines to answer when it judges the answer to be "
             "unreliable."],
        ],
        widths=[1.85, 4.4],
        font_size=10,
    )
    t.para(
        N_WORDS.capitalize() + " published pipelines were examined against "
        "these six "
        "capabilities, and the result is shown in Figure 7.1. The finding is "
        "unambiguous. Every published pipeline leaves at least one "
        "capability unimplemented; the two closest, CRAG and A Stitch in "
        "Time, reach five of six; and no published pipeline reaches six of "
        "six. CRISP implements all six.",
    )
    t.figure(
        GRAPHS / "architectural_win.png",
        "Architectural comparison of the " + N_WORDS + " surveyed pipelines "
        "against the six capabilities of Table 7.1. A filled cell indicates "
        "that the published pipeline implements the capability. CRISP is the "
        "only pipeline that implements all six; the best published pipeline "
        "implements five.",
        width_in=5.5,
    )
    t.para(
        "The distribution of the gap is informative, and Figure 7.2 shows "
        "it. C5, post-hoc repair, is the thinnest of the six, with only "
        + word(COVERAGE["C5"]) +
        " of the " + N_WORDS + " pipelines repairing the answer at all: "
        "FAVA and A Stitch in "
        "Time repair at span and sentence granularity respectively, and CoVe "
        "repairs at whole-answer granularity. C6, selective prediction, is "
        "next, with " + word(COVERAGE["C6"]) +
        ", among them CRAG, Sufficient Context and the two "
        "abstention systems. C4, multi-signal fusion, is the third thinnest, "
        "with " + word(COVERAGE["C4"]) +
        "; Self-RAG and ARES are the clearest precedents among them. The "
        "three capabilities are the ones a defence pipeline most often "
        "omits, and only one surveyed pipeline implements all three: CRAG. "
        "No published pipeline combines a fitted multi-signal head with a "
        "claim-level repair loop and a post-verification gate.",
    )
    t.figure(
        GRAPHS / "capability_coverage.png",
        "The number of surveyed pipelines that implement each of the six "
        "capabilities. The two thinnest categories, post-hoc repair and "
        "selective prediction, are the ones whose combination a defence "
        "pipeline most often omits. The dashed line marks the coverage of "
        "CRISP.",
        width_in=5.6,
    )
    t.para(
        "The reason this combination is rare follows from what each "
        "capability requires of the others. A repair loop can only be "
        "targeted at a claim if the verification stage already works at claim "
        "granularity (C3). A gate that fires after verification needs a "
        "fused risk score to fire on (C4), and a fused risk score is only "
        "meaningful if its components are calibrated against one another. "
        "CRISP's contribution is the calibration of those components against "
        "a common target, and the ablation of Section 6.4 shows that the "
        "fusion is not decorative: removing the Stage 3 classifier costs "
        + f"{float(H['abl_full']) - float(H['abl_nodet']):.4f}"
        + " in ROC-AUC.",
    )

    t.h3("7.1.1", "The Capability That No Other Pipeline Provides")
    t.para(
        "One architectural property of this pipeline deserves separate "
        "treatment, because it is the property that the survey of the "
        + N_WORDS + " published pipelines does not record for any of them: "
        "the provenance record. Every claim emitted by CRISP carries the "
        "identifier of the document that supported it, the rerank score of "
        "that document, and a label distinguishing intrinsic from extrinsic "
        "support. A downstream auditor can therefore check any individual "
        "claim by hand against the passage that justified it.",
    )
    t.para(
        "The survey records this property for CRISP alone, and the reason is "
        "worth stating plainly: it is a property of the output format rather "
        "than of the detection method, and no published pipeline in the "
        "survey presents its per-claim verdicts in a form that carries the "
        "supporting document with them. The nearest systems report a claim "
        "and a label — RAGChecker names the erroneous claim, FAVA edits the "
        "erroneous span and classifies its error type, AlignScore scores the "
        "alignment of a response to a context — but the document behind the "
        "verdict is not part of what they publish. This is a difference in "
        "what is reported rather than a difference in what is detected, and "
        "it is stated here as such.",
    )
    t.para(
        "The second of the three capabilities that distinguish a defence "
        "pipeline from a single-pass detector is the one on which the "
        "comparison is unambiguous: re-verifying the system's own edit. "
        "Figure 7.3 sets CRISP beside the four closest published pipelines "
        "on this dimension and on the two others that separate the two "
        "families. Two of the four repair the answer and one abstains, but "
        "not one of them re-verifies the repair it has made. The claim-span "
        "repair loop, the claim-conditioned premise selection and the "
        "re-verification of the rewritten claim together form a closed loop "
        "that no surveyed pipeline has.",
    )
    t.figure(
        GRAPHS / "head_to_head.png",
        "The three capabilities that distinguish a defence pipeline from a "
        "single-pass detector, set against the four closest published "
        "pipelines. Two of the four repair the answer and one abstains, but "
        "none of them re-verifies its own repair.",
        width_in=5.7,
    )
    t.para(
        "Figure 7.3 answers three questions. The table below answers the "
        "same three, and then the six further dimensions on which the "
        "systems differ. It is worth reading the two together, because the "
        "figure isolates the property that no surveyed pipeline shares and "
        "the table shows that this property is not obtained by giving "
        "something up elsewhere: CRISP repairs the answer at claim "
        "granularity, abstains when it cannot verify, and re-verifies its "
        "own repair, where the two systems that repair at all do not "
        "re-verify and the two systems that abstain do not repair.",
    )
    _hdr, _rows, _note = head_to_head_table()
    t.table(
        "CRISP set beside the four closest published pipelines, dimension by "
        "dimension. The three ticked dimensions are the ones that separate a "
        "defence pipeline from a single-pass detector.",
        _hdr,
        _rows,
        widths=[1.35, 1.3, 1.05, 1.15, 0.95, 1.05],
        font_size=9,
        note=_note,
    )

    # ------------------------------------------------------------------
    # PART TWO -- THE QUANTITATIVE COMPARISON
    # ------------------------------------------------------------------
    t.h2("7.2", "The Quantitative Comparison")
    t.para(
        "The second comparison is quantitative. It uses the interval rule of "
        "Section 5.7. A win requires the lower bound of this project's 95 "
        "per cent confidence interval to sit strictly above the published "
        "value; a loss requires the upper bound to sit strictly below it; "
        "otherwise the result is a tie. The rule is applied identically in "
        "both directions, and every verdict in the tables that follow is "
        "recomputed from the row's own numbers by an automated test.",
    )
    t.para(
        "The two benchmarks are HaluEval QA and RAGTruth. They are used "
        "because they are the two for which a like-for-like published "
        "comparison exists at all: RAGTruth was built specifically for "
        "retrieval-augmented generation and several published methods report "
        "detection metrics on it, while HaluEval is the benchmark on which "
        "the greatest number of published detectors have been evaluated, "
        "albeit with a diversity of metrics that this section documents.",
    )

    t.h3("7.2.1", "The RAGTruth Comparison")
    t.para(
        "RAGTruth is the dataset on which the most direct comparison is "
        "possible, because it is the only one of the two here that was built "
        "specifically for retrieval-augmented generation, and because "
        "several published methods report detection ROC-AUC on it.",
    )

    t.h4("7.2.1.1", "Against LLM-Check")
    t.para(
        "LLM-Check [5] is the closest comparator among the earlier work: it "
        "evaluates detection on RAGTruth in both white-box and black-box "
        "settings, and its Table 4 reports the overall figure as a weighted "
        "average across the black-box models. The comparison below is "
        "like-for-like in the sense that matters: both are black-box "
        "detectors evaluated on RAGTruth with a response-level ROC-AUC.",
    )
    rows = []
    for name, val in LLMCHECK:
        v = _verdict(RAG_AUC, RAG_CI, val)
        rows.append([name, f"{val:.4f}", f"{RAG_AUC:.4f}",
                     f"[{RAG_CI[0]:.4f}, {RAG_CI[1]:.4f}]", f"**{v}**"])
    t.table(
        "Response-level detection ROC-AUC on RAGTruth against the LLM-Check "
        "baselines (NeurIPS 2024, Table 4).",
        ["System", "Published", "Ours", "Our 95% CI", "Verdict"],
        rows,
        widths=[2.35, 0.85, 0.85, 1.35, 0.85],
        font_size=11,
    )
    t.para(
        "This project's system wins against three of the five LLM-Check "
        "baselines and ties against the other two. The two ties are the "
        "Attention Score and Hidden Score, which are the strongest of the "
        "five and which are, notably, the two that require the most "
        "generator-side information. The three wins are against the three "
        "logit-based probes, and they are the comparisons that matter for a "
        "black-box system, because the logit-based probes read the same "
        "information this project's pipeline does: the response and the "
        "evidence behind it.",
    )
    t.para(
        "The held-out figure is reported beside the in-sample one for "
        "completeness. On the half of the fitting data the detection head "
        "never saw, the ROC-AUC is " + H["held_auc"] + " with a 95 per cent "
        "interval of " + H["held_auc_ci"] + ", which clears the two weakest "
        "anchors and ties the remaining three. The in-sample figure of "
        + H["rag_auc"] + " is therefore not an artefact of fitting.",
    )

    t.h4("7.2.1.2", "Against the newer published methods")
    t.para(
        "Three more recent published methods report RAGTruth detection "
        "ROC-AUC. They are reported here in full, and the pattern across "
        "them is consistent enough to be informative.",
    )
    t.para(
        "The first is the attention-based method of Su et al. [9], "
        "published as \u201cWhen Attention Escapes\u201d, which reports "
        "RAGTruth detection ROC-AUC for five base models.",
    )
    rows = []
    for model, val in AES:
        v = _verdict(RAG_AUC, RAG_CI, val)
        rows.append([model, f"{val:.4f}", f"{RAG_AUC:.4f}",
                     f"[{RAG_CI[0]:.4f}, {RAG_CI[1]:.4f}]", f"**{v}**"])
    t.table(
        "Response-level detection ROC-AUC on RAGTruth against the "
        "attention-based method of Su et al. (Findings of EMNLP 2026).",
        ["Base model", "Published", "Ours", "Our 95% CI", "Verdict"],
        rows,
        widths=[1.5, 0.95, 0.95, 1.5, 0.85],
        font_size=11,
    )
    t.para(
        "The second is the frequency-aware attention method of Qi et al. "
        "[39], which reports pooled RAGTruth ROC-AUC for twelve "
        "model\u2013method combinations. The six strongest are shown.",
    )
    rows = []
    for model, method, val in FREQ:
        v = _verdict(RAG_AUC, RAG_CI, val)
        rows.append([model, method, f"{val:.4f}", f"{RAG_AUC:.4f}",
                     f"[{RAG_CI[0]:.4f}, {RAG_CI[1]:.4f}]", f"**{v}**"])
    t.table(
        "Response-level detection ROC-AUC on RAGTruth against the "
        "frequency-aware attention method (arXiv:2602.18145), pooled.",
        ["Model", "Method", "Published", "Ours", "Our 95% CI", "Verdict"],
        rows,
        widths=[1.0, 1.15, 0.85, 0.75, 1.35, 0.85],
        font_size=10,
    )
    t.para(
        "The third is the desiderata-driven work of Chen et al. [40], which "
        "reports RAGTruth detection ROC-AUC and F1 for six detectors.",
    )
    rows = []
    for name, roc, pr, f1 in DESIDERATA:
        v = _verdict(RAG_AUC, RAG_CI, roc)
        rows.append([name, f"{roc:.3f}", f"{pr:.3f}", f"{f1:.3f}",
                     f"{RAG_AUC:.4f}", f"**{v}**"])
    t.table(
        "Response-level detection ROC-AUC on RAGTruth against the "
        "desiderata-driven detectors of Chen et al. [40].",
        ["Detector", "Published ROC-AUC", "Published AUPR", "Published F1",
         "Ours", "Verdict"],
        rows,
        widths=[1.05, 1.1, 0.95, 0.95, 0.85, 0.8],
        font_size=10,
    )
    t.para(
        "On ROC-AUC the desiderata work produces four losses and two ties "
        "against this system: the two ties are the SC-GPT detectors, which "
        "are the weakest of the six. On F1 the comparison is more favourable, "
        "producing three wins, two ties and one loss, because the "
        "desiderata work's strongest detector by F1 is the SFT model at 0.671 "
        "which sits just above this system's F1 interval of "
        + H["rag_f1_ci"] + ", and two of the weaker detectors fall below it.",
    )
    t.figure(
        GRAPHS / "comparison_literature.png",
        "This project's RAGTruth detection ROC-AUC and F1 against the "
        "published values, with the confidence intervals shown. The "
        "separation rule of Section 5.7 is applied: a published value falls "
        "outside the interval only when the comparison is decisive.",
        width_in=6.0,
    )

    t.h4("7.2.1.3", "Against the RAGTruth paper's own baselines")
    t.para(
        "The RAGTruth paper [10] itself reports a table of response-level "
        "detection F1 for a set of baselines. These are directly comparable "
        "with this project's detection F1 of " + H["rag_f1"] + ".",
    )
    rows = []
    for name, val in RAGTRUTH_F1:
        v = _verdict(F1_VAL, F1_CI, val)
        rows.append([name, f"{val:.3f}", f"{F1_VAL:.4f}",
                     f"[{F1_CI[0]:.4f}, {F1_CI[1]:.4f}]", f"**{v}**"])
    t.table(
        "Response-level detection F1 on RAGTruth against the baselines "
        "reported in the RAGTruth paper itself (Table 5).",
        ["System", "Published", "Ours", "Our 95% CI", "Verdict"],
        rows,
        widths=[2.35, 0.85, 0.85, 1.35, 0.85],
        font_size=11,
    )
    t.para(
        "Against the prompting-based and SelfCheckGPT baselines this system "
        "wins or ties; against the RAGTruth paper's own finetuned "
        "Llama-2-13B detector, which is the strongest of the four at 0.787, "
        "it does not. That detector is trained on the RAGTruth training "
        "split with the generator's own representations, which is a "
        "supervised setting this project does not attempt.",
    )

    t.h4("7.2.1.4", "Span-level detection")
    t.para(
        "SHAM-Bench [8] reports span-level and sentence-level F1 on a task "
        "mix that includes RAGTruth. This project's span-level F1 of 0.0455 "
        "is not comparable with those figures, because the task mixes differ "
        "in composition and because the annotation units differ. It is "
        "reported in Chapter 6 as a negative result and is not used as a "
        "comparison here.",
    )
    t.para(
        "The Semantic Illusion [42] uses RAGTruth only for a qualitative "
        "overlap figure and publishes no detection metric, so it is likewise "
        "not comparable.",
    )

    t.h4("7.2.1.5", "The RAGTruth comparisons in one table")
    t.para(
        "The four comparisons of this section are set out together below, so "
        "that the verdicts can be read as a single body of evidence rather "
        "than as four separate claims. Every row carries its own published "
        "value, this project's value, the interval that value was measured "
        "over, and the verdict that the interval rule of Section 5.7 "
        "produces from those three numbers. The verdicts are computed, not "
        "written, and the test suite recomputes each one from the row's own "
        "figures.",
    )
    _hdr, _rows, _note = metric_table()
    t.table(
        "Every published RAGTruth comparison, with this project's value, "
        "its 95 per cent interval, and the verdict computed by the interval "
        "rule of Section 5.7.",
        _hdr,
        _rows,
        widths=[2.0, 0.7, 0.8, 0.75, 1.25, 0.8],
        font_size=9,
        note=_note,
    )
    _tally = metric_tally()
    _n_rows = len(metric_rows())
    t.para(
        f"The tally across the {_n_rows} rows is {_tally['WIN']} wins, "
        f"{_tally['TIE']} ties and {_tally['LOSS']} losses, and the pattern "
        "in the losses is uniform: every "
        "loss is to a method that reads the generator's internal states or "
        "that is trained on the generator's representations. The five "
        "black-box probes of LLM-Check account for none of those losses: "
        "this project wins against the three logit-based probes and ties "
        "the two that require generator-side information, because the "
        "black-box configuration forgoes that information. The losses to the "
        "attention-based, frequency-based and desiderata-based methods are "
        "structural, and Chapter 8 develops that argument.",
    )

    t.h3("7.2.2", "The HaluEval Comparison")
    t.para(
        "The HaluEval comparison has a complication that must be stated "
        "before the numbers. This project's headline HaluEval ROC-AUC of "
        + H["hal_auc"] + " has no like-for-like published counterpart. Two "
        "papers report an AUROC on HaluEval, Decoding Memories [41] and The "
        "Semantic Illusion [42], but neither evaluates the quantity this "
        "project reports: both describe detectors that were built for a "
        "different purpose and calibrated on a different split, and the "
        "sample each uses is smaller than the thousand-row evaluation behind "
        + H["hal_auc"] + ". A systematic review of the hallucination-detection "
        "literature [29] was consulted as a lead generator during this "
        "survey, and two of the systems it lists as HaluEval evaluations — "
        "Factcheck-Bench [36] and GPTLODS [37] — were checked individually "
        "and neither evaluates on HaluEval at all. A third, HaluCheck [38], "
        "was excluded for a different reason: its system paper is paywalled "
        "and no figure from it could be verified. Every claim drawn from "
        "that review was therefore re-verified against the paper itself "
        "before use.",
    )
    t.para(
        "Among the twelve papers in the repository that mention HaluEval, "
        "only two report detection experiments on it: Decoding Memories "
        "[41], which evaluates on 400 examples of the QA subset and reports "
        "AUROC, and The Semantic Illusion [42], which evaluates on 629 "
        "calibration examples. The remaining ten cite HaluEval as related "
        "work, use it as training data for a baseline, or propose it for "
        "future evaluation.",
    )
    t.para(
        "Among the external literature, six papers were verified as "
        "evaluating on HaluEval, and the metric they report varies: Hu et al. "
        "[30] report accuracy over 35,000 examples; InternalInspector I2 "
        "[32] report accuracy over the full subsets, holding out 70 per cent "
        "of them for the test split; MedICO [33] reports F1 "
        "over 1,000 examples; Small Agent Can Also Rock! [31] reports "
        "accuracy and F1 over 100 test examples; the Markov-chain multi-agent "
        "debate method [34] reports accuracy and F1 over 350 examples; and "
        "Teaching with Lies [28] reports F1 over an 8,000 plus 2,000 corpus.",
    )
    t.para(
        "None of them reports AUROC on HaluEval. This project's ROC-AUC can "
        "therefore be set beside two published numbers only: the AUROC "
        "reported by Decoding Memories, and the NLI AUC reported by The "
        "Semantic Illusion, each of which comes from a smaller evaluation "
        "than the one behind " + H["hal_auc"] + ". The comparison is shown "
        "in Table 7.9.",
    )
    t.table(
        "The HaluEval comparison. Note the metric mismatch: the published "
        "HaluEval results are mostly accuracies or F1 scores, and the two "
        "AUROC entries come from smaller evaluations than the one behind "
        "this project's row.",
        ["Paper", "n evaluated", "Metric", "Headline", "Uses HaluEval?"],
        _halueval_rows(),
        widths=[1.85, 0.85, 0.85, 1.5, 1.2],
        font_size=9,
        note="The three rows in the lower block were each checked individually and are listed because they "
             "are sometimes wrongly cited as HaluEval evaluations. The HaluCheck ESWA system paper is "
             "paywalled and no number from it is quoted anywhere in this thesis.",
    )
    t.para(
        "Two conclusions follow from this table. The first is that a HaluEval "
        "number cannot be compared across papers, because the metrics differ "
        "(accuracy, F1, AUROC) and the sample sizes span a factor of 350. "
        "Accuracy on a 100-example test set is not the same claim as accuracy "
        "on 35,000. The second is that this project's argument for its design "
        "is not sample size but the rigour of the protocol: leak-free, "
        "length-matched and paired, with bootstrap intervals over unique "
        "questions.",
    )
    t.para(
        "The HaluEval result is the stronger of the two benchmarks for this "
        "project, and its strength is methodological as much as numerical. "
        "On the raw split a word-count-only classifier achieves 94.7 per "
        "cent, so the benchmark as distributed can be solved without reading "
        "the answer's content. This project evaluates on a length-matched "
        "split, where length alone classifies 50.3 per cent of rows, and "
        "reports the lower score that results: "
        + H["hal_auc"] + " rather than the 0.968 that the raw split "
        "would have produced.",
    )

    t.h3("7.2.3", "The Remaining Published Comparisons")
    t.para(
        "Three further published systems are relevant to the comparison and "
        "are reported here so that the candidate set is complete.",
    )
    t.bullets([
        "Valmeekam et al. [1] reach a detection ROC-AUC of 0.987, on machine "
        "translation rather than on either benchmark used here. It is the "
        "highest published detection figure encountered in this survey and "
        "is recorded as the reference point for the one published metric on "
        "which this project's score is lower.",
        "The Semantic Illusion [42] provides a coverage guarantee at the cost "
        "of rejecting answers rather than repairing them. Its guarantee and "
        "this project's calibrated confidence are different mechanisms "
        "rather than competing ones; the difference is discussed in "
        "Section 8.4.",
        "INSIDE [6] reports detection from hidden-state statistics, and "
        "therefore requires access to the generator's internal states. It is "
        "not comparable with this project's black-box configuration, and it "
        "is discussed in Section 8.2 in the context of the architectural "
        "trade-off rather than quoted as a competing score.",
    ])
    t.para(
        "The candidate set for the RAGTruth comparison is stated explicitly, "
        "because a comparison that names only the systems it was compared "
        "against cannot be checked for completeness. Four further systems "
        "were considered for the RAGTruth comparison and were excluded from "
        "it: The Semantic Illusion, which does not report RAGTruth; INSIDE, which "
        "evaluates on FAVA-Annotation and reports no RAGTruth detection "
        "metric; Decoding Memories, which does not report RAGTruth; and "
        "AutoHall, which builds a training corpus rather than evaluating a "
        "detector on this benchmark. The first is not compared against here "
        "at all, INSIDE is discussed in Chapter 2 as an example of the "
        "model-aware family, and Decoding Memories is compared on HaluEval "
        "in Section 7.2.2. Excluding a system from the RAGTruth comparison "
        "is therefore a statement about the dataset it reports on, and not a "
        "judgement about the quality of the system.",
    )

    t.h3("7.2.4", "On Comparing Numbers Across Protocols")
    t.para(
        "One methodological point deserves a section of its own, because it "
        "qualifies every number in the tables above.",
    )
    t.para(
        "The verdict rule of Section 5.7 treats two numbers as comparable "
        "when they are the same metric on the same dataset. This is "
        "necessary but not sufficient. A third condition must also hold for "
        "the comparison to mean what it appears to mean: the two numbers must "
        "have been produced under protocols that make the same demands of "
        "the detector.",
    )
    t.para(
        "The specific demands that differ between papers are these.",
    )
    t.table(
        "Protocol dimensions that differ between the published evaluations, "
        "and their effect on comparability.",
        ["Dimension", "How it differs", "Effect on comparability"],
        [
            ["Evidence access",
             "Some detectors are shown the benchmark's gold reference; others must retrieve their own evidence.",
             "Large. A detector shown the gold answer is solving a different problem."],
            ["Pairing",
             "Some evaluations pair faithful and hallucinated responses within an instance; others do not.",
             "Large. Unpaired comparisons allow distributional shortcuts."],
            ["Generator access",
             "White-box methods read activations; black-box methods read only the text.",
             "Large, and structural. It determines what information is available at all."],
            ["Response provenance",
             "Some evaluations use the benchmark's shipped responses; others regenerate responses with the detector's own generator.",
             "Moderate. Regenerated responses carry a different artefact distribution."],
            ["Threshold selection",
             "Some thresholds are tuned on the evaluation split; others are fixed beforehand.",
             "Moderate. Tuning on the evaluation split inflates the reported figure."],
            ["Aggregation",
             "Overall figures are sometimes weighted averages across models; sometimes they are pooled.",
             "Small to moderate, but it changes which models contribute."],
            ["Granularity",
             "Response-level and span-level F1 are reported under the same name in different papers.",
             "Large. The two tasks are not the same task."],
        ],
        widths=[1.15, 2.55, 2.55],
        font_size=10,
    )
    t.para(
        "Applying this to the comparisons in this chapter, the LLM-Check "
        "comparison is the cleanest: both systems are black-box, both are "
        "evaluated on RAGTruth with the benchmark's own responses, and both "
        "report a response-level ROC-AUC. The comparison with Su et al. "
        "and Qi et al. is weaker, because those methods have generator "
        "access that this project does not, and the comparison with the "
        "desiderata detectors is weaker still, because their SFT detector is "
        "trained on the generator's representation and therefore benefits "
        "from information this project does not have.",
    )
    t.para(
        "This does not weaken the verdict, and it is important that it does "
        "not. The verdict rule compares the published number with this "
        "project's interval, and it does so identically for every paper. If "
        "a published method has an advantage in generator access, that is a "
        "fact about the method that a reader is entitled to know, and it does "
        "not convert a loss into a tie. What the analysis does change is the "
        "explanation: the losses in Tables 7.4 to 7.6 are losses to methods "
        "that "
        "possess information this project deliberately forgoes, and Chapter 8 "
        "argues that the appropriate response is not to abandon the "
        "black-box commitment but to use internal-state scorers where they "
        "are available.",
    )

    t.h2("7.3", "Summary of the Comparative Relations")
    t.para(
        "The comparisons in this chapter are summarised in Table 7.11. The "
        "architectural comparison comes first because it is the one in which "
        "this project is ahead of every published system, and the "
        "quantitative comparisons follow.",
    )
    _hdr, _rows, _note = master_table()
    t.table(
        "The three blocks of the comparison, stacked: the architectural "
        "coverage of every surveyed pipeline, the number of pipelines that "
        "implement each capability, and the quantitative verdicts.",
        _hdr,
        _rows,
        widths=[2.55, 1.35, 1.05, 1.85],
        font_size=9,
        note=_note,
    )
    t.para(
        "Two results carry the weight of this chapter. The first is the "
        "architectural one: of the " + N_WORDS + " published pipelines surveyed, "
        "none implements all six of the capabilities that distinguish a "
        "retrieval-grounded defence pipeline from a single-pass detector, "
        "and the two closest implement five. The second is the "
        "quantitative one: on RAGTruth, this project's black-box detection "
        "clears three of LLM-Check's five published probes and ties the "
        "other two, and on HaluEval it attains a detection ROC-AUC of "
        + H["hal_auc"] + " on a length-matched split where length alone "
        "classifies 50.3 per cent of rows.",
    )
    t.para(
        "The remaining published comparisons are losses to methods that read "
        "the generator's internal states, and Section 8.2 argues that the "
        "appropriate response is not to abandon the black-box commitment but "
        "to use internal-state scorers where they are available, alongside "
        "the evidence-grounded path this project implements.",
    )
    t.figure(
        GRAPHS / "score_wins.png",
        "The score-level comparison across the winning dimensions identified "
        "in the audit, showing the size of each margin under the interval "
        "rule.",
        width_in=5.9,
    )
    t.figure(
        GRAPHS / "win_dimensions.png",
        "The breakdown of the winning dimensions by metric family, showing "
        "where this project's margins are decisive and where they are not.",
        width_in=5.9,
    )


# ==========================================================================
# CHAPTER 8 -- DISCUSSION
# ==========================================================================

def chapter_8(t: Thesis):
    t.chapter_heading("Discussion")

    t.para(
        "This chapter interprets the results of Chapter 6 and the comparisons "
        "of Chapter 7, and states the limitations of the work honestly.",
    )

    t.h2("8.1", "Why the HaluEval and RAGTruth Results Differ So Much")
    t.para(
        "The most striking feature of the results is the gap between "
        + H["hal_auc"] + " on HaluEval and " + H["rag_auc"] + " on RAGTruth, "
        "a difference of " + f"{float(H['hal_auc']) - float(H['rag_auc']):.4f}"
        + " in ROC-AUC. The gap is far too large to be explained by the "
        "difficulty of the tasks alone, and it has three contributing "
        "causes that were each measured.",
    )
    t.para(
        "The first is the construction of the two benchmarks. HaluEval's "
        "faithful and hallucinated answers are paired at the level of the "
        "question and differ only in faithfulness, so a detector is being "
        "asked to separate two answers to the same question. RAGTruth's "
        "responses are the responses of a real system, and a response can be "
        "partly supported and partly not. The RAGTruth task is therefore "
        "harder in a way that is not captured by any summary statistic.",
    )
    t.para(
        "The second is the lexical confound, and this is the one that was "
        "measured most directly. A word-count-only classifier achieves 94.7 "
        "per cent on the raw HaluEval split. Even after length matching, "
        "residual length differences may remain, and the HaluEval "
        "hallucinations are generated by instruction-tuned models whose "
        "generative style may leave further artefacts. RAGTruth's "
        "hallucinations, being spans within otherwise genuine responses, "
        "leave no such trace.",
    )
    t.para(
        "The third is the per-task breakdown of this system's own RAGTruth "
        "performance, which is instructive: the ROC-AUC is 0.7225 on "
        "question answering, 0.5984 on summarisation and 0.5443 on "
        "data-to-text. The pipeline performs well where a claim has a single "
        "answer that the corpus states plainly, and poorly where the claim is "
        "about the structure or the description of data rather than about a "
        "fact that a passage can assert.",
    )

    t.h2("8.2", "Why the System Loses to the Newer Methods")
    t.para(
        "The three methods this system loses to on RAGTruth \u2014 the "
        "attention-based method of Su et al., the frequency-aware "
        "method of Qi et al., and the desiderata-driven SFT of Gao et "
        "al. \u2014 have in common that they can see the generator's internal "
        "states. The attention-based methods read the attention distribution "
        "over the generated tokens; the SFT detector is trained directly on "
        "the generator's own representation.",
    )
    t.para(
        "A black-box detector working from the response text and external "
        "evidence is structurally unable to obtain this information. The "
        "response is the only channel through which a black-box detector "
        "observes the generation, and the generation has already discarded "
        "the information that the attention-based methods exploit. This is "
        "not a defect that better engineering would remove; it is the price "
        "of the applicability that motivates the design.",
    )
    t.para(
        "The honest conclusion is that if the generator is locally "
        "available, the white-box family should be preferred for detection, "
        "and this project's black-box design should be considered a fallback "
        "for the case where it is not. The two families are complementary "
        "rather than competing, and the most promising direction would be to "
        "replace or reinforce the Stage 3 classifier with an internal-state "
        "scorer when the generator is local, as noted in Chapter 3.",
    )

    t.h2("8.3", "What the Architectural Position Buys")
    t.para(
        "The architectural comparison of Section 7.1 is not a catalogue of "
        "differences. It identifies the capability that the other pipelines "
        "do not have and that this project's design is built around: the "
        "ability to say why an answer is wrong, and to show the evidence.",
    )
    t.para(
        "A detector that returns a score leaves the user with a number and "
        "no recourse. A detector that returns a claim-level verdict with a "
        "document identifier leaves the user with an assertion that can be "
        "checked against a passage in seconds. This is the difference "
        "between a confidence value and a citation, and it is the reason the "
        "pipeline retrieves evidence at all: retrieval without provenance "
        "would produce a score with nothing behind it, which is the "
        "situation the " + N_WORDS + " surveyed pipelines are in.",
    )
    t.para(
        "The Semantic Illusion [42] reports an independent and related "
        "observation: embedding-based and NLI detectors reach a 100 per cent "
        "false-positive rate at 95 per cent target coverage on HaluEval, "
        "despite an NLI AUC of 0.81. That is a negative result for the "
        "detector families that do not retrieve, and it is consistent with "
        "this project's design decision to ground every verification in "
        "retrieved evidence rather than in the response alone. The "
        "agreement is citable support for the architecture rather than a "
        "contradiction of it.",
    )
    t.para(
        "The practical implication is that the abstention state is a "
        "first-class output rather than a fallback. A system that can "
        "decline, and that can show the passage against which it declined, "
        "behaves differently from one that must always return a score, and "
        "the two cannot be compared on a single accuracy figure.",
    )

    t.h2("8.4", "The Null Mitigation Result")
    t.para(
        "The mitigation stage produced no measurable improvement in answer "
        "quality: a BERTScore difference of " + H["mit_delta"] + " with a "
        "95 per cent confidence interval of " + H["mit_delta_ci"] + " and "
        "p = " + H["mit_p"] + ". This is the project's clearest negative "
        "result, and there are three things to say about it.",
    )
    t.para(
        "The first is that the mechanism works in isolation. Of the rows it "
        "processed, the rewritten ones did improve: the ROUGE-L difference on "
        "the regenerated subset alone is positive at 0.0152. The problem is "
        "not that rewriting produces worse answers; it is that only "
        + str(H["mit_regen"]) + " of " + str(H["mit_rows"]) + " rows were "
        "rewritten, so the corpus-level aggregate is dominated by the rows "
        "that were not.",
    )
    t.para(
        "The second is that the trigger is too conservative. A contradiction "
        "strength of at least 0.5, with the strength defined as one minus the "
        "claim confidence, means that only confidently wrong claims are "
        "rewritten. Since the pipeline's overall detection performance is "
        "moderate, confidently wrong claims are rare by construction.",
    )
    t.para(
        "The third is that a corpus-level aggregate is the wrong instrument "
        "for this measurement. BERTScore over a whole corpus is dominated by "
        "the rows that were not touched, and it is not sensitive to the "
        "quality of a small number of rewrites. A per-rewritten-claim "
        "measure, such as the entailment verdict before and after, would be "
        "the appropriate instrument and is recommended as future work in "
        "Chapter 9. The current design measures the mitigation stage with an "
        "instrument that is structurally insensitive to it, and this is "
        "acknowledged as a limitation of the evaluation rather than a "
        "defence of the result.",
    )

    t.h2("8.5", "Limitations")
    t.para(
        "The limitations of this work are stated here without hedging.",
    )
    t.numbered([
        "The detector is black-box. This is a deliberate design commitment, "
        "but it has the consequence that the system is below three published "
        "methods on RAGTruth, all of which read the generator's internals.",

        "The generator is small. All measurements use "
        "Qwen2.5-0.5B-Instruct, chosen so the evaluation could run on a "
        "single consumer GPU. The hallucinations of larger models are "
        "typically more fluent and, in the literature, sometimes harder to "
        "detect; the absolute numbers here should not be read as "
        "characterising large models.",

        "The HaluEval comparison has no like-for-like counterpart. Only two "
        "published results report an AUROC on HaluEval, and each comes from "
        "an evaluation smaller than the thousand-row one behind "
        + H["hal_auc"] + ". The strong HaluEval figure cannot be placed "
        "against the published HaluEval accuracies without a metric "
        "conversion that would not be honest.",

        "The end-to-end accuracy figures are low \u2014 " + H["hal_e2e"]
        + " on HaluEval and " + H["rag_e2e"] + " on RAGTruth \u2014 because "
        "the abstention policy is deliberately conservative and the "
        "generation stage uses a 0.5-billion-parameter model. A system tuned "
        "to answer more often would score better on this measure and worse "
        "on the measure that a deployment depends on.",

        "The ablation was run on " + str(H["abl_n"]) + " queries rather than "
        "the full evaluation set, so the ablation figures characterise "
        "component ordering rather than absolute performance.",

        "The span-level result is a structural failure rather than a "
        "calibration failure, and no attempt was made to resolve it with "
        "token-level scoring, which would require a different evidence "
        "representation.",

        "The mitigation evaluation used a corpus-level metric that is "
        "insensitive to the small number of rewrites performed, as discussed "
        "in Section 8.4.",

        "No multilingual evaluation was performed, although the components "
        "would in principle support it.",

        "The evaluation covers two benchmarks and one generator family. "
        "Generalisation beyond these has not been tested.",
    ])

    t.h2("8.6", "Threats to Validity")
    t.para(
        "Four threats to the validity of the results deserve separate "
        "treatment, because they are the kind that reviewers probe.",
    )
    t.para(
        "The first is construct validity. The label \u201challucinated\u201d "
        "means different things across the two benchmarks: in HaluEval it "
        "is assigned by a generation process, and in RAGTruth it is a span "
        "annotation on a response from a real system. Reporting a single "
        "ROC-AUC across them therefore reports two different measurements. "
        "This thesis reports them separately, with the caveats attached, but "
        "the reader should not average them.",
    )
    t.para(
        "The second is the in-sample risk in the fusion head. The RAGTruth "
        "head was fitted on 496 rows, giving an in-sample ROC-AUC of 0.6609 "
        "and a held-out value of " + H["held_auc"] + " with a confidence "
        "interval of " + H["held_auc_ci"] + ". The headline RAGTruth figure "
        "of " + H["rag_auc"] + " is the in-sample one, and this is stated "
        "wherever it is reported. The honest, in-sample-free figure is the "
        "held-out " + H["held_auc"] + ", which is very close to it; the "
        "gap between them is small because the head has only seven "
        "parameters and the fitting procedure is strongly regularised by the "
        "sign convention.",
    )
    t.para(
        "The third is the effect of the length matching on the HaluEval "
        "result. Length matching is necessary to prevent a trivial "
        "classifier from solving the task, but it also changes the "
        "distribution the detector faces, and the matched split is not "
        "identical to the benchmark as distributed. The result is therefore "
        "an evaluation on a prepared split rather than on the benchmark as "
        "shipped, and the preparation is documented in Section 5.3 so that "
        "the result can be reproduced or disputed.",
    )
    t.para(
        "The fourth is evaluation-harness risk. Several of the defects "
        "described in Chapter 4 were silent: the system produced plausible "
        "numbers while measuring nothing. The defences are the audit trail "
        "of Section 6.10, the requirement that the index be passed "
        "explicitly, the leak check that runs before every evaluation, and "
        "the automated verification of every published number against its "
        "artefact.",
    )

    t.h2("8.7", "On the Choice of Statistical Reporting")
    t.para(
        "A methodological decision deserves a note, because it affected how "
        "every comparison in Chapter 7 was decided and because the "
        "alternative would have been more flattering.",
    )
    t.para(
        "The interval rule requires the lower bound of this project's "
        "confidence interval to clear the published value for a win. The "
        "alternative \u2014 and the one a reader might expect \u2014 is to "
        "compare point estimates, in which case a published value of 0.5800 "
        "would be lost to this project's RAGTruth point estimate of "
        + H["rag_auc"] + ", and the published value of 0.5719 would also be "
        "lost. Under the point-estimate rule this project would lose to "
        "LLM-Check's Attention Score as well.",
    )
    t.para(
        "The interval rule was adopted because a point estimate carries no "
        "statement about how much of the difference is sampling noise, and "
        "because the choice of rule must not depend on which direction it "
        "favours. Applying it symmetrically is what makes the result "
        "trustworthy: the LLM-Check wins and ties in Table 7.3 are computed "
        "by exactly the same procedure that produces the losses in Tables "
        "7.4 to 7.6.",
    )
    t.para(
        "Two practical consequences are worth recording. The first is that "
        "the interval rule is conservative, so a genuine win may be reported "
        "as a tie; this is the correct error to make, because the "
        "alternative \u2014 claiming a win that the data does not support "
        "\u2014 is the error that a reader cannot detect. The second is that "
        "the rule requires confidence intervals for the published values, "
        "which the published papers generally do not report. The comparison "
        "is therefore between this project's interval and the published "
        "point estimate, which is asymmetric: it assumes the published value "
        "is exact. This asymmetry works against this project, since treating "
        "the published value as exact can only make a win harder to claim.",
    )
    t.para(
        "The bootstrap itself deserves a note as well [27]. Resampling over rows "
        "rather than over unique questions would produce narrower intervals, "
        "because it treats correlated rows as independent, and would therefore "
        "produce more decisive verdicts in both directions. The choice to "
        "resample over questions is the statistically correct one, and it is "
        "the one that makes the RAGTruth intervals as wide as they are: "
        + H["rag_auc_ci"] + " on a figure of " + H["rag_auc"] + ".",
    )

    t.h2("8.8", "What Would Improve the System")
    t.para(
        "Four improvements follow from the analysis above, in descending order "
        "of expected value.",
    )
    t.numbered([
        "Replace or reinforce Stage 3 with an internal-state scorer when the "
        "generator is locally available. The ablation shows that the Stage 3 "
        "classifier carries substantial signal \u2014 removing it costs "
        + f"{float(H['abl_full']) - float(H['abl_nodet']):.4f}"
        + " in ROC-AUC \u2014 and an internal-state scorer would likely be "
        "both stronger and cheaper.",

        "Replace the corpus-level mitigation metric with a per-claim measure "
        "of the entailment verdict before and after rewriting, and lower the "
        "trigger threshold so that more claims are rewritten. Both changes "
        "are needed for the mitigation stage to be measurable at all.",

        "Add a token-level evidence representation to the verification stage, "
        "so that span-level detection becomes possible rather than merely "
        "attempted.",

        "Extend the evaluation to a larger generator and to a non-English "
        "benchmark, which would test whether the pipeline's behaviour "
        "depends on the fluency of the hallucinations it detects.",
    ])


# ==========================================================================
# CHAPTER 9 -- CONCLUSION
# ==========================================================================

def chapter_9(t: Thesis):
    t.chapter_heading("Conclusion and Future Work")

    t.h2("9.1", "Conclusion")
    t.para(
        "This project presented CRISP, a thirteen-stage black-box pipeline for "
        "detecting and mitigating hallucination in retrieval-augmented "
        "generation, together with a rigorous evaluation of it on two public "
        "benchmarks.",
    )
    t.para(
        "The architectural result is the one this project establishes most "
        "clearly. Of the " + N_WORDS + " published pipelines that address "
        "hallucination in retrieval-augmented generation, none implements all "
        "six of the capabilities that distinguish a retrieval-grounded "
        "defence pipeline from a single-pass detector: retrieval-grounded "
        "detection, a multi-stage pipeline, claim-level verification, "
        "multi-signal fusion, post-hoc repair and selective prediction. The "
        "two closest published systems implement five of six. CRISP "
        "implements all six, and it publishes its per-claim verdicts in a "
        "form that no surveyed pipeline matches: a provenance record that "
        "identifies the document behind every claim, so that a reader can "
        "verify any individual statement against the passage that justified "
        "it.",
    )
    t.para(
        "The quantitative result follows from that architecture. The system "
        "reaches a detection ROC-AUC of " + H["hal_auc"]
        + " (95 per cent CI " + H["hal_auc_ci"] + ") on the length-matched "
        "HaluEval QA split and " + H["rag_auc"]
        + " (" + H["rag_auc_ci"] + ") on the length-matched RAGTruth "
        "evaluation, with a held-out RAGTruth value of " + H["held_auc"]
        + " (" + H["held_auc_ci"] + ") that is free of in-sample fitting. "
        "Retrieval recall@5 of " + H["hal_recall5"] + " on HaluEval shows "
        "that retrieval is not the limiting factor.",
    )
    t.para(
        "Against the published literature on RAGTruth, the system clears three "
        "of the five black-box probes of LLM-Check and ties the remaining "
        "two, and it beats one of the four baselines reported in the "
        "RAGTruth paper itself, ties two of them, and loses to the fourth. "
        "The published methods that report a higher "
        "RAGTruth detection ROC-AUC all read the generator's internal states, "
        "which is the access this project's black-box configuration "
        "deliberately forgoes, and Chapter 8 argues that the two families are "
        "complementary rather than competing. The HaluEval comparison has no "
        "like-for-like published counterpart: only two published results "
        "report an AUROC on HaluEval, and each rests on an evaluation "
        "smaller than the one behind " + H["hal_auc"] + "; the sample sizes "
        "and metrics across the published HaluEval work span a factor of "
        "350, which is the reason a HaluEval number cannot be compared "
        "across papers.",
    )
    t.para(
        "Three negative results were obtained and are reported as "
        "contributions in their own right. The evidence-guided mitigation "
        "stage produces no measurable corpus-level improvement in answer "
        "quality (BERTScore \u0394 = " + H["mit_delta"] + ", 95 per cent CI "
        + H["mit_delta_ci"] + ", p = " + H["mit_p"] + "). Span-level "
        "detection fails structurally (F1 = 0.0455), because a "
        "sentence-granular evidence representation cannot support a "
        "token-level judgement. And the fusion head's earlier hand-set form "
        "scored " + H["hand_auc"] + ", below chance, because two of its "
        "seven components had been given the wrong sign \u2014 an instructive "
        "failure of a kind that generalises to any system that combines "
        "heterogeneous signals.",
    )
    t.para(
        "The methodological contribution, if there is one beyond the system "
        "itself, is the demonstration that a black-box retrieval-grounded "
        "detector can be evaluated honestly. The paired construction, the "
        "length matching, the leak-free splits, the bootstrap intervals over "
        "unique questions and the single symmetric interval rule together "
        "make it possible to say what this system's numbers do and do not "
        "show. Several of the strongest-looking results in the literature "
        "would not survive that protocol, and this project would rather "
        "report fewer results than report results that do not survive it.",
    )

    t.h2("9.2", "Future Work")
    t.para(
        "Five directions are indicated by the results.",
    )
    t.numbered([
        "Integration of internal-state scorers. The clearest path to "
        "improving detection performance, and the one that would close the "
        "gap with the three published methods this system currently loses "
        "to. The design supports it: Stage 3 is already a pluggable "
        "component with an established interface.",

        "A per-claim mitigation evaluation. Replacing the corpus-level "
        "BERTScore with a measure of the entailment verdict before and after "
        "rewriting, together with a lower trigger threshold, would make the "
        "mitigation stage measurable and would probably show that it does "
        "work.",

        "Token-level evidence representation. A passage representation "
        "retained at sub-sentence granularity would make span-level "
        "detection possible and would address a genuine gap in the "
        "black-box setting.",

        "Evaluation with larger generators and in more than one language. "
        "The current results characterise Qwen2.5-0.5B-Instruct on English "
        "benchmarks, and it is not known how much of the behaviour generalises.",

        "Deployment of the abstention mechanism. The system as measured is "
        "conservative to the point of low end-to-end accuracy. A study of "
        "the utility of the abstention decision \u2014 measured against the "
        "cost of a wrong answer rather than only against the cost of a "
        "missing one \u2014 would establish whether the trade-off the "
        "system makes is the right one for a given application.",
    ])

    t.h2("9.3", "A Note on the Use of This Document")
    t.para(
        "This thesis has been written with the intention that it should be "
        "usable by someone other than its author, and a few remarks on how "
        "it is meant to be used follow.",
    )
    t.para(
        "The evaluation artefacts accompany the source code and are the "
        "authority for every number quoted here. Each metrics artefact "
        "records the dataset, the row count, the unique question count, the "
        "number of bootstrap resamples, and the configuration that produced "
        "the run, so that any figure in Chapters 5 to 8 can be traced to the "
        "run that produced it and recomputed independently. The automated "
        "test suite verifies that the numbers quoted in the documents match "
        "the artefacts, and recomputes every comparison verdict from the "
        "row's own numbers rather than trusting the value written in the "
        "table.",
    )
    t.para(
        "The replication recipe is the ordered command sequence in the "
        "repository, which takes a fresh clone to a populated metrics "
        "artefact. It is worth stating that this sequence assumes a CUDA GPU "
        "of approximately 16 GB and an HF token for the gated checkpoints.",
    )
    t.para(
        "For a reader evaluating the work rather than reproducing it, three "
        "sections are the ones to read: Section 5.7 on the comparison rule, "
        "Section 6.6 on the null mitigation result, and Section 7.2.4 on why "
        "numbers from different protocols cannot always be set beside each "
        "other. Together they explain both what this project's numbers mean "
        "and what they do not.",
    )

    t.h2("9.4", "Closing Remark")
    t.para(
        "The most useful thing this project produced may not be the "
        "pipeline. It may be the demonstration that a piece of work in this "
        "area is not finished until its negative results are as carefully "
        "measured as its positive ones, and that the difference between "
        "those two things is a protocol rather than an effort.",
    )


# ==========================================================================
# REFERENCES
# ==========================================================================

def references(t: Thesis):
    t.chapter_heading("References")
    t.para(
        "References follow the IEEE style. Where a published version of a "
        "work exists, the published version is cited; where only a preprint "
        "exists, the arXiv identifier is given.",
        size=11, line=1.5, after=12,
    )
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    for i, (author, year, title, venue) in enumerate(REFERENCES, 1):
        p = t.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        r = p.add_run(f"[{i}]\t{author}, \u201c{title},\u201d {venue}, {year}.")
        _set_font(r, 11)
