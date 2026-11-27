"""The thirteen per-stage flow diagrams, and where they belong in the book.

The figures themselves are drawn by ``scripts/plot_step_diagrams.py`` into
``data/graphs/``. This module holds the *book* side of the same thirteen:
which figure goes in which subsection, and what its caption says.

A caption states the finding and names the inputs and the outputs; it
carries no title and no annotation of its own, because the plot carries
none either. Each caption is written to be read without the surrounding
prose.

The stage numbers here are the *stage* numbers, which are not the
execution order. The execution order is 1 → 6 → 7 → 2 → 3 → 5 → 8 → 4 →
9 → 10 → 11 → 12 → 13.
"""
from __future__ import annotations

from .content import GRAPHS

# stage number -> (slug, width in inches, caption)
STEP_FIGURES: dict[int, tuple[str, float, str]] = {
    1: (
        "step01_query_understanding",
        5.6,
        "The flow of Stage 1. The stage works out what kind of question "
        "was asked, pulls out the named things the question mentions, and "
        "rewrites it into a better search query. It writes the question's "
        "type, the things it names, and the rewritten query to the "
        "context.",
    ),
    6: (
        "step06_hybrid_retrieval",
        5.6,
        "The flow of Stage 6. The stage searches the collection twice, once "
        "by meaning and once by exact word, combines the two ranked lists "
        "into one, and writes the best passages to the context as the "
        "evidence to check the answer against.",
    ),
    7: (
        "step07_reranking",
        5.6,
        "The flow of Stage 7. The stage reads each passage and the question "
        "together and scores how well they match, sorts the passages by "
        "that score, and keeps only the best ones, leaving a reordered "
        "and scored evidence set in the context.",
    ),
    2: (
        "step02_grounded_generation",
        5.6,
        "The flow of Stage 2. The stage fills a prompt with the question and "
        "the passages found for it, asks the language model to write the "
        "answer, and writes that draft answer to the context for the "
        "later stages to verify.",
    ),
    3: (
        "step03_early_detection",
        5.6,
        "The flow of Stage 3. The stage measures four simple signs of a weak "
        "answer, turns them into one suspicion score, and writes that "
        "score to the context as its early screen, taken before the "
        "costlier checks that follow.",
    ),
    5: (
        "step05_claim_extraction",
        5.6,
        "The flow of Stage 5. The stage breaks the answer into the separate "
        "statements it makes, notes for each one whether it came from the "
        "question, and writes the statements with those provenance labels "
        "to the context so that each can be checked on its own.",
    ),
    8: (
        "step08_nli_verification",
        5.6,
        "The flow of Stage 8. The stage picks the part of the evidence that is "
        "about each statement, asks whether that evidence supports the "
        "statement, keeps the evidence used, and writes a verdict for "
        "every statement to the context: supported, contradicted, or "
        "cannot tell.",
    ),
    4: (
        "step04_confidence_assessment",
        5.6,
        "The flow of Stage 4. The stage measures how unsure the verification "
        "model was, combines that with the other signals into one score, "
        "and adjusts it so that a stated probability means what it says, "
        "leaving a calibrated confidence score in the context.",
    ),
    9: (
        "step09_confidence_fusion",
        5.6,
        "The flow of Stage 9. The stage puts the seven signals on a common "
        "scale, makes sure each one counts in the direction that was "
        "measured rather than the one that was assumed, and adds them up "
        "with their fitted weights to give one trust score.",
    ),
    10: (
        "step10_regeneration",
        5.6,
        "The flow of Stage 10. The stage chooses the statements that failed "
        "the check, finds where each one sits in the answer, and replaces "
        "it only if the new wording is usable, the right length and still "
        "on topic. It then writes the corrected answer and what was done "
        "to it to the context.",
    ),
    11: (
        "step11_final_gate",
        5.6,
        "The flow of Stage 11. The stage compares the trust score and the "
        "share of unsupported statements against the thresholds set for "
        "the system, and writes the decision to the context: hand the "
        "answer over, try again, or decline to answer.",
    ),
    12: (
        "step12_explainable_response",
        5.6,
        "The flow of Stage 12. The stage renders the answer, the statements, "
        "the verdicts, the evidence and the log of decisions into a "
        "Markdown report for a person to read and a JSON report for "
        "another program to use.",
    ),
    13: (
        "step13_evaluation",
        5.6,
        "The flow of Stage 13. The stage scores how well the right passages "
        "were found and how well the statements were judged, adds up the "
        "results for the whole run, and writes them to the metrics file.",
    ),
}


# The one-line statement of what each stage is for. It is the same
# sentence the diagram opens with, and it is repeated here so that the
# reader meets the purpose of the stage in the prose as well as in the
# figure. A reader new to the pipeline gets the point of the stage
# before meeting the detail.
STAGE_PURPOSE = {
    1: "Work out what the question is asking, and what it is about.",
    6: "Find the passages in the document collection that answer the "
       "question.",
    7: "Put the most relevant passages first, and drop the rest.",
    2: "Answer the question using only the passages that were found.",
    3: "Take a quick first look at the answer, before spending time on "
       "the checks that follow.",
    5: "Break the answer into the separate statements it makes, so each "
       "can be checked on its own.",
    8: "Check each statement against the evidence, and say whether the "
       "evidence supports it.",
    4: "Turn the raw signals into a confidence score that can be "
       "trusted, and compared.",
    9: "Bring every signal together into one judgement about how much to "
       "trust the answer.",
    10: "Fix the statements the evidence does not support, rather than "
        "throwing the whole answer away.",
    11: "Decide whether to hand the answer over, try again, or decline "
        "to answer.",
    12: "Show the answer together with the evidence behind it, so a "
        "reader can check it.",
    13: "Measure how well the whole pipeline did.",
}


def step_figure(t, stage: int):
    """Insert the flow diagram for ``stage`` into the thesis.

    The figure is placed immediately after the subsection heading's
    prose, so the reader meets the diagram while the stage is being
    described and not after the description has closed.
    """
    slug, width, caption = STEP_FIGURES[stage]
    t.para(
        "In short, the purpose of this stage is to "
        + STAGE_PURPOSE[stage][0].lower() + STAGE_PURPOSE[stage][1:].rstrip(".")
        + ". Figure {n} shows what goes in, what the stage does with it, "
        "and what comes out.".format(n=t.next_figure()),
    )
    return t.figure(GRAPHS / f"{slug}.png", caption, width_in=width)
