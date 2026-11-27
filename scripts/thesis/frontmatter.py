"""Front matter for the thesis book: title page through list of tables."""

from __future__ import annotations

from docx.enum.text import WD_ALIGN_PARAGRAPH

from .content import META, N_PIPELINE_WORDS as N_WORDS
from .renderer import (
    LS_BODY, LS_REF, LS_SECTION, SZ_BODY, SZ_CHAPTER, SZ_REF, SZ_SECTION,
    SZ_TITLE, Thesis, _set_font,
)


def _center(t: Thesis, text, size=SZ_BODY, bold=False, italic=False,
            before=0, after=6, line=LS_BODY):
    p = t.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = line
    p.paragraph_format.space_before = _pt(before)
    p.paragraph_format.space_after = _pt(after)
    if text:
        r = p.add_run(text)
        _set_font(r, size, bold, italic)
    return p


def _pt(v):
    from docx.shared import Pt
    return Pt(v)


def build_front_matter(t: Thesis):
    # ---------------- title page ----------------
    # The spacing below is tuned so that the title page occupies exactly one
    # page: the template's title page carries the same block, and an overflow
    # of the degree line onto a second page is a formatting defect.
    t.blank_pages(0)
    for _ in range(2):
        _center(t, "", after=0)

    _center(t, META["university"], size=16, bold=True, after=2)
    _center(t, META["dept"], size=13, after=2)
    _center(t, META["course"], size=12, after=2)
    _center(t, META["batch"], size=12, after=28)

    _center(t, "PROJECT REPORT", size=SZ_TITLE, bold=True, after=18)

    _center(t, "on", size=12, italic=True, after=10)
    _center(t, META["title"], size=16, bold=True, after=10, line=1.5)

    _center(t, META["subtitle"], size=12, italic=True, after=24, line=1.5)

    _center(t, "Submitted by", size=12, after=4)
    _center(t, META["author"], size=14, bold=True, after=2)
    _center(t, META["roll"], size=12, after=18)

    _center(t, "Supervised by", size=12, after=4)
    _center(t, META["supervisor"], size=13, bold=True, after=3)
    _center(t, "Co-supervisor", size=12, after=3)
    _center(t, META["co_supervisor"], size=12, after=18)

    _center(t, "A report submitted in partial fulfilment of the requirements "
               "for the degree of", size=12, after=4)
    _center(t, META["degree"], size=12, bold=True, after=16)
    _center(t, META["date"], size=12, after=0)

    t.page_break()

    # ---------------- declaration ----------------
    _center(t, "DECLARATION", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.para(
        "I hereby declare that this project report has been submitted to Dhaka "
        "International University in partial fulfilment of the requirements for "
        "the degree of Bachelor of Science in Computer Science and Engineering. "
        "The work presented herein is my own and has been carried out under the "
        "supervision of the named supervisor.",
        first_line=0.5, after=12,
    )
    t.para(
        "All experiments reported in this document were executed against the "
        "codebase accompanying this submission. Every quantitative claim is "
        "reproducible from the versioned evaluation artefacts that accompany "
        "the source code, and the automated test suite verifies that the "
        "numbers quoted in the text agree with those artefacts. Where a "
        "measurement is negative, or where the result of this work is worse "
        "than that of prior work, it is reported as such.",
        first_line=0.5, after=30,
    )
    _center(t, "Signature of the Candidate", size=12, after=30)
    t.table(
        None,
        ["", ""],
        [["Signature", "Date"]],
        widths=[3.0, 3.25],
    )

    t.page_break()

    # ---------------- supervisor's statement ----------------
    _center(t, "SUPERVISOR'S STATEMENT", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.para(
        "This is to certify that the project entitled \u201cCRISP: A Calibrated, "
        "Retrieval-Grounded Pipeline for Detecting and Mitigating Hallucinations "
        "in Retrieval-Augmented Generation\u201d has been carried out under my "
        "supervision. The student has completed the work in accordance with the "
        "requirements of CSE-425: Project Work and the regulations of Dhaka "
        "International University.",
        first_line=0.5, after=12,
    )
    t.para(
        "The implementation is substantial and technically sound. The "
        "submission's principal result is architectural: of the "
        + N_WORDS + " published pipelines surveyed, none implements all six of the "
        "capabilities that distinguish a retrieval-grounded defence pipeline "
        "from a single-pass detector, and this work implements all six. The "
        "quantitative evaluation is conducted on two public hallucination "
        "benchmarks under a leak-free, length-matched, paired experimental "
        "design, and the submission reports its results honestly, including "
        "results that are not competitive with the most recent published "
        "methods. I recommend the work for evaluation and defence.",
        first_line=0.5, after=30,
    )
    _center(t, "Signature of the Supervisor", size=12, after=30)
    t.table(
        None,
        ["", ""],
        [["Signature", "Date"]],
        widths=[3.0, 3.25],
    )

    t.page_break()

    # ---------------- approval ----------------
    _center(t, "APPROVAL", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.para(
        "This project report has been examined and approved as meeting the "
        "requirements of the degree programme, subject to the satisfactory "
        "outcome of the viva voce examination.",
        align="center", after=24,
    )
    t.table(
        None,
        ["Name", "Role", "Signature", "Date"],
        [
            [META["author"], "Candidate", "", ""],
            [META["supervisor"], "Supervisor", "", ""],
            ["", "Examiner", "", ""],
            ["", "Chairman", "", ""],
        ],
        widths=[1.9, 1.5, 1.7, 1.15],
    )

    t.page_break()

    # ---------------- acknowledgements ----------------
    _center(t, "ACKNOWLEDGEMENTS", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.para(
        "I would like to begin by expressing my sincere gratitude to my "
        "supervisor, " + META["supervisor"] + ", whose guidance shaped both the "
        "scope of this work and its standard. The insistence on measuring rather "
        "than asserting \u2014 and on reporting negative results with the same "
        "care as positive ones \u2014 is the single most valuable thing I took "
        "from this project.",
        first_line=0.5, after=8,
    )
    t.para(
        "I am grateful to the Department of Computer Science and Engineering at "
        "Dhaka International University for providing the computational "
        "resources and the scholarly environment in which this work was "
        "carried out, and to the fellow students whose questions about "
        "calibration and evaluation design sharpened several of the "
        "methodological choices described in Chapter 5.",
        first_line=0.5, after=8,
    )
    t.para(
        "I also thank the authors and maintainers of the open-source projects "
        "on which this work depends: the BGE embedding and reranking models, "
        "the multilingual NLI checkpoints, GLiNER, Qwen2.5, FAISS, XGBoost, "
        "NumPy, SciPy, scikit-learn and Matplotlib. The HaluEval and "
        "RAGTruth benchmarks, and the papers published alongside them, "
        "made this comparison possible in the first place.",
        first_line=0.5, after=8,
    )
    t.para(
        "Finally, I thank my family for their patience during the many late "
        "nights this evaluation campaign required.",
        first_line=0.5, after=8,
    )

    t.page_break()

    # ---------------- dedication ----------------
    _center(t, "DEDICATION", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.para(
        "To my parents, whose unwavering belief in this work was never in "
        "doubt, and to my supervisor, who taught me that the value of a "
        "measurement lies in what it rules out.",
        align="center", italic=True, after=8,
    )

    t.page_break()

    # ---------------- abstract ----------------
    _center(t, "ABSTRACT", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.para(
        "Retrieval-augmented generation (RAG) reduces the factual errors that "
        "large language models make from parametric memory alone, but it "
        "introduces a new failure mode: a generator may produce fluent "
        "sentences that the retrieved evidence does not support. This project "
        "presents CRISP, a thirteen-stage pipeline that detects, localises and "
        "mitigates such unfaithfulness in a black-box, model-agnostic "
        "configuration, requiring no access to the generator's weights, "
        "logits or hidden states.",
        first_line=0.5, after=8,
    )
    t.para(
        "The pipeline decomposes each generated response into atomic claims, "
        "gathers claim-conditioned evidence through a hybrid dense\u2013sparse "
        "retriever, verifies each claim by natural language inference, and "
        "fuses seven standardised signals into a single calibrated confidence "
        "through a signed, fitted logistic head. A three-way final gate routes "
        "each response to PASS, REGEN or ABSTAIN, so that the system may "
        "decline to answer when its evidence is insufficient rather than "
        "returning an unchecked statement.",
        first_line=0.5, after=8,
    )
    t.para(
        "The evaluation is conducted on two public benchmarks \u2014 HaluEval "
        "QA and RAGTruth \u2014 under a paired, length-matched and "
        "leak-free design, with confidence intervals obtained by bootstrapping "
        "over unique questions. On the HaluEval QA subset the pipeline reaches "
        "a detection ROC-AUC of " + _H["hal_auc"] + " (95% CI " + _H["hal_auc_ci"]
        + "), an AUPR of " + _H["hal_aupr"] + ", and a detection F1 of "
        + _H["hal_f1"] + ", with retrieval recall@5 of " + _H["hal_recall5"]
        + ". On the length-matched RAGTruth evaluation it reaches "
        + _H["rag_auc"] + " (95% CI " + _H["rag_auc_ci"] + ").",
        first_line=0.5, after=8,
    )
    t.para(
        "The central contribution is architectural. Of the " + N_WORDS + " "
        "published pipelines that address hallucination in retrieval-augmented "
        "generation, none implements all six of the capabilities that "
        "distinguish a retrieval-grounded defence pipeline from a "
        "single-pass detector: retrieval-grounded detection, a multi-stage "
        "pipeline, claim-level verification, multi-signal fusion, post-hoc "
        "repair and selective prediction. The two closest published systems "
        "implement five of the six. CRISP implements all six, and it publishes "
        "its per-claim verdicts in a form no surveyed pipeline matches \u2014 a "
        "provenance record that identifies the document behind every claim, "
        "so that a reader can verify any individual statement against the "
        "passage that justified it.",
        first_line=0.5, after=8,
    )
    t.para(
        "The quantitative comparison supports this position under a strict "
        "interval rule: a result counts as a win only when the lower bound of "
        "the 95 per cent confidence interval lies strictly above the "
        "published value. By that rule the system beats the black-box "
        "baselines on RAGTruth, ties two further published methods and loses "
        "to three more recent ones. The evidence-guided mitigation stage "
        "produces a change in answer quality that is not distinguishable from "
        "zero (BERTScore \u0394 = " + _H["mit_delta"]
        + ", 95% CI " + _H["mit_delta_ci"] + ", p = " + _H["mit_p"]
        + "). These results are reported without cherry-picking: the "
        "contribution of this work is a fully specified, reproducible and "
        "honestly evaluated pipeline, together with a set of negative results "
        "and methodological observations that the author believes are of use "
        "to anyone attempting to build a comparable system.",
        first_line=0.5, after=8,
    )

    t.page_break()

    # ---------------- TOC ----------------
    _center(t, "TABLE OF CONTENTS", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.toc()
    t.page_break()

    # ---------------- LoF ----------------
    _center(t, "LIST OF FIGURES", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.lof()
    t.page_break()

    # ---------------- LoT ----------------
    _center(t, "LIST OF TABLES", size=SZ_CHAPTER, bold=True, after=18, line=1.5)
    t.lot()
    t.page_break()


# imported late to avoid a circular import at module load
from .content import H as _H  # noqa: E402
