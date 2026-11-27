"""One flow diagram per pipeline stage.

The master's architecture figure (Figure 3.1) shows the whole pipeline at
once. These thirteen figures are the zoom: each one shows a single
stage's inputs, the operations that stage performs, and what it writes
out. Read together they are the architecture figure at the resolution
the prose in §3.2 discusses it.

Every diagram is drawn from one declarative spec, so all thirteen share
a single visual language and a change to the language changes all of
them. The specs are transcribed from the stage sources in ``src/steps/``
and from ``docs/pipeline_13_steps_deepdive.md``; each carries the file
it describes.

**Audience.** The reader of the book may be meeting the pipeline for the
first time, so the labels are plain language: the boxes say what the
stage does, not what the internal identifiers are called. The internal
name is kept in the small grey line under each operation, where it is
useful for someone tracing the code but does not have to be understood
to follow the diagram. Each figure opens with a one-line statement of
what the stage is for.

Like the other figures in this set, they carry no title and no
annotation: the caption states the finding and the plot shows only the
data.

Usage:
    python scripts/plot_step_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "graphs"

# -- the shared visual language ------------------------------------------
INPUT_BG = "#f1f5f9"        # a context field entering the stage
OP_BG = "#eff6ff"           # an operation the stage performs
OUT_BG = "#ecfdf5"          # something the stage writes out
DEC_BG = "#fef3c7"          # a decision the stage makes
INPUT_EDGE = "#94a3b8"
OP_EDGE = "#2563eb"
OUT_EDGE = "#16a34a"
DEC_EDGE = "#d97706"

TEXT = "#0f172a"
MUTED = "#475569"
FAINT = "#94a3b8"
ARROW = "#64748b"
RULE = "#cbd5e1"

# The colour of a node by its kind.
KIND_STYLE = {
    "input": (INPUT_BG, INPUT_EDGE),
    "op": (OP_BG, OP_EDGE),
    "out": (OUT_BG, OUT_EDGE),
    "dec": (DEC_BG, DEC_EDGE),
}

# The three column headings. A reader who has not seen the diagram
# before needs to be told which way to read it.
COL_HEADS = ("What goes in", "What the stage does", "What comes out")


# --------------------------------------------------------------------------
# The thirteen stage specs.
#
# Each is: the stage number and slug, the source file it is transcribed
# from, a one-line statement of what the stage is for, and the nodes of
# the flow. A node is (kind, label, technical) where the label is the
# plain-language description and the technical name is the internal
# identifier, shown in small grey type under the label.
#
# The layout is three columns -- input, operations, output -- with the
# decision column appended when the stage makes one.
# --------------------------------------------------------------------------
SPECS = [
    {
        "n": 1,
        "slug": "step01_query_understanding",
        "file": "src/steps/step01_query_understanding.py",
        "purpose": "Work out what the question is asking, and what it is about.",
        "cols": [
            [("input", "The user's question", "")],
            [
                ("op", "Work out what kind of question it is",
                 "factual · opinion · multi-hop · math · other"),
                ("op", "Pull out the named things it mentions",
                 "GLiNER + spaCy, merged entity list"),
                ("op", "Rewrite it into a better search query",
                 "Qwen2.5-0.5B, falling back to a rule"),
            ],
            [("out", "The question's type, the things it names,\nand a rewritten search query", "")],
        ],
    },
    {
        "n": 6,
        "slug": "step06_hybrid_retrieval",
        "file": "src/steps/step06_hybrid_retrieval.py",
        "purpose": "Find the passages in the document collection that answer the question.",
        "cols": [
            [("input", "The rewritten search query", "")],
            [
                ("op", "Search by meaning", "BGE-M3, FAISS IndexFlatIP"),
                ("op", "Search by exact words",
                 "BM25, k1 = 1.5, b = 0.75"),
                ("op", "Combine both lists into one",
                 "RRF, k = 60, top-k = 5"),
            ],
            [("out", "The best passages to check the answer against", "")],
        ],
    },
    {
        "n": 7,
        "slug": "step07_reranking",
        "file": "src/steps/step07_reranking.py",
        "purpose": "Put the most relevant passages first, and drop the rest.",
        "cols": [
            [("input", "The retrieved passages", "")],
            [
                ("op", "Read each passage and the question together\nand score how well they match",
                 "BGE-Reranker-v2-M3 cross-encoder"),
                ("op", "Sort the passages by that score", "descending by rerank score"),
                ("op", "Keep only the best ones", "truncated to top-k = 5"),
            ],
            [("out", "The same passages, best first, each with a score", "")],
        ],
    },
    {
        "n": 2,
        "slug": "step02_grounded_generation",
        "file": "src/steps/step02_grounded_generation.py",
        "purpose": "Answer the question using only the passages that were found.",
        "cols": [
            [("input", "The question and the passages found for it", "")],
            [
                ("op", "Fill in the prompt with the question and the passages",
                 "context template, evidence-conditioned"),
                ("op", "Ask the language model to write the answer",
                 "Qwen2.5-0.5B-Instruct"),
            ],
            [("out", "The draft answer, written from the evidence", "")],
        ],
    },
    {
        "n": 3,
        "slug": "step03_early_detection",
        "file": "src/steps/step03_early_detection.py",
        "purpose": "Take a quick first look at the answer, before spending time on the checks that follow.",
        "cols": [
            [("input", "The draft answer and the question", "")],
            [
                ("op", "Measure four simple signs of a weak answer",
                 "cosine · claim count · intrinsic overlap · length"),
                ("op", "Turn those signs into one suspicion score",
                 "XGBoost classifier"),
            ],
            [("out", "How likely the answer is to be wrong", "")],
        ],
    },
    {
        "n": 5,
        "slug": "step05_claim_extraction",
        "file": "src/steps/step05_claim_extraction.py",
        "purpose": "Break the answer into the separate statements it makes, so each can be checked on its own.",
        "cols": [
            [("input", "The draft answer", "")],
            [
                ("op", "Split the answer at sentence and clause boundaries",
                 "rule mode: punctuation and connector rules"),
                ("op", "Note whether each statement came from the question",
                 "provenance by overlap against the query"),
            ],
            [("out", "The individual statements, each with where it came from", "")],
        ],
    },
    {
        "n": 8,
        "slug": "step08_nli_verification",
        "file": "src/steps/step08_nli_verification.py",
        "purpose": "Check each statement against the evidence, and say whether the evidence supports it.",
        "cols": [
            [("input", "The statements and the passages found for them", "")],
            [
                ("op", "Pick the part of the evidence that is about\nthat statement, so the check stays focused",
                 "token-overlap anchor, contiguous span, max 1800 chars"),
                ("op", "Ask: does this evidence support that statement?",
                 "DeBERTa-v3-MNLI, 278M parameters"),
                ("op", "Keep the evidence used, for each statement",
                 "best supporting premise per claim"),
            ],
            [("out", "A verdict per statement: supported, contradicted,\nor cannot tell", "")],
        ],
    },
    {
        "n": 4,
        "slug": "step04_confidence_assessment",
        "file": "src/steps/step04_confidence_assessment.py",
        "purpose": "Turn the raw signals into a confidence score that can be trusted, and compared.",
        "cols": [
            [("input", "The verdicts and the early suspicion score", "")],
            [
                ("op", "Measure how unsure the verification model was",
                 "normalised NLI entropy, H = sum p ln p / ln 3"),
                ("op", "Combine that with the other signals into one score",
                 "EEDC = aH + b(1 - retrieval_top1)\n+ c(1 - self_consistency) + d"),
                ("op", "Adjust it so the number means what it says",
                 "temperature scaling, 32-point grid, 0.25 to 4.0"),
            ],
            [("out", "A calibrated confidence score", "")],
        ],
    },
    {
        "n": 9,
        "slug": "step09_confidence_fusion",
        "file": "src/steps/step09_confidence_fusion.py",
        "purpose": "Bring every signal together into one judgement about how much to trust the answer.",
        "cols": [
            [("input", "The seven signals collected so far", "")],
            [
                ("op", "Put them on a common scale",
                 "z = (raw - mean) / std, fitted on the tuning split"),
                ("op", "Make sure each one counts in the right direction",
                 "sign from AUC against 0.5, applied outside the fit"),
                ("op", "Add them up with their fitted weights",
                 "logistic regression, sigmoid, confidence = 1 - risk"),
            ],
            [("out", "One trust score, from 0 (do not trust) to 1", "")],
        ],
    },
    {
        "n": 10,
        "slug": "step10_regeneration",
        "file": "src/steps/step10_regeneration.py",
        "purpose": "Fix the statements the evidence does not support, rather than throwing the whole answer away.",
        "cols": [
            [("input", "The statements and the passages found for them", "")],
            [
                ("op", "Choose the statements that failed the check",
                 "strength = 1 - confidence, threshold 0.5"),
                ("op", "Find where each one sits in the answer",
                 "character offsets held across the edit"),
                ("op", "Replace it, but only if the new wording is\nusable, the right length and still on topic",
                 "guards: 12-word or 4x length cap, subject must match"),
            ],
            [("out", "The corrected answer, and what was done to it", "")],
        ],
    },
    {
        "n": 11,
        "slug": "step11_final_gate",
        "file": "src/steps/step11_final_gate.py",
        "purpose": "Decide whether to hand the answer over, try again, or decline to answer.",
        "cols": [
            [("input", "The trust score and the share of unsupported statements", "")],
            [
                ("op", "Compare them against the thresholds set for this system",
                 "pass at 0.70, abstain at 0.30, at most 2 retries"),
            ],
            [("out", "The decision: hand it over, try again, or decline", "")],
        ],
    },
    {
        "n": 12,
        "slug": "step12_explainable_response",
        "file": "src/steps/step12_explainable_response.py",
        "purpose": "Show the answer together with the evidence behind it, so a reader can check it.",
        "cols": [
            [("input", "The answer, the statements, the verdicts,\nthe evidence and the log of decisions", "")],
            [
                ("op", "Lay it out for a person to read", "Markdown report"),
                ("op", "Lay it out for another program to use", "JSON report"),
            ],
            [("out", "The report, in both forms", "")],
        ],
    },
    {
        "n": 13,
        "slug": "step13_evaluation",
        "file": "src/steps/step13_evaluation.py",
        "purpose": "Measure how well the whole pipeline did.",
        "cols": [
            [("input", "The statements, the evidence, the correct answers\nand the decision that was taken", "")],
            [
                ("op", "Score how well the right passages were found",
                 "recall@5 · P@5 · MRR · nDCG"),
                ("op", "Score how well the statements were judged",
                 "precision · recall · F1"),
                ("op", "Add up the results for the whole run", "aggregate, per row and overall"),
            ],
            [("out", "The measurements, row by row and in total", "")],
        ],
    },
]



def _extent_in(fig, text) -> tuple[float, float]:
    """A text's vertical extent, in inches, measured by the renderer."""
    fig.canvas.draw()
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
    # The axes span the figure exactly, so one data unit is one inch.
    return bbox.y0 / fig.dpi, bbox.y1 / fig.dpi


def _bottom_in(fig, text) -> float:
    """The lower edge of a text, in data units."""
    return _extent_in(fig, text)[0]


def _top_in(fig, text) -> float:
    """The upper edge of a text, in data units."""
    return _extent_in(fig, text)[1]


def _wrap(label: str, width: int) -> str:
    """Wrap a label to a character width, respecting explicit newlines.

    A box is a fixed physical width, so a long line has to be broken or it
    will run past the edge of the box. The explicit newlines in the spec
    are the author's line breaks and are kept.
    """
    out = []
    for para in label.split("\n"):
        if not para:
            out.append("")
            continue
        words = para.split(" ")
        line = ""
        for word in words:
            if not line:
                line = word
            elif len(line) + 1 + len(word) <= width:
                line += " " + word
            else:
                out.append(line)
                line = word
        out.append(line)
    return "\n".join(out)


def _box(fig, ax, x, y, w, h, label, kind, technical=None, fontsize=9.0):
    """Draw one node.

    The label is the plain-language description. The technical name, when
    there is one, goes underneath in small grey type: it is there for the
    reader who wants to find the code, and it can be ignored by the
    reader who only wants to follow the flow.
    """
    bg, edge = KIND_STYLE[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=bg, edgecolor=edge, linewidth=1.0,
    ))
    # The wrap width is chosen so that the widest line fits the box with
    # a margin on each side, at the size the label is drawn.
    label = _wrap(label, int(w * 9.6))
    if technical:
        # The label sits above the technical name, so the two are never
        # confused for one another, and a hairline separates them. The
        # rule's position is taken from the measured extent of the two
        # texts rather than from an estimate, because an estimate that
        # is a little small puts the rule through the last line of the
        # label.
        tech = _wrap(technical, int(w * 13.0))

        # The label is anchored to the top of the box. The technical name
        # is drawn immediately below it, separated by a hairline. The
        # rule's position is taken from the measured extent of the label
        # rather than from an estimate, because an estimate that is a
        # little small puts the rule through the last line of the label.
        label_text = ax.text(
            x + w / 2, y + h - 0.09, label, ha="center", va="top",
            fontsize=fontsize, color=TEXT, linespacing=1.35,
        )
        # The renderer knows the height of the text it drew.
        label_bottom = _bottom_in(fig, label_text)
        rule_y = label_bottom - 0.10

        ax.plot([x + w * 0.16, x + w * 0.84], [rule_y, rule_y],
                color=RULE, linewidth=0.7)
        ax.text(x + w / 2, rule_y - 0.07, tech, ha="center", va="top",
                fontsize=fontsize - 2.2, color=FAINT, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=TEXT, linespacing=1.4)


def _arrow(ax, p0, p1, label=None):
    """Draw an arrow from one box edge to another."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9,
        linewidth=1.0, color=ARROW, shrinkA=1.5, shrinkB=1.5,
        connectionstyle="arc3,rad=0",
    ))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[0] + 1) / 2 + 0.16, label,
                ha="center", va="bottom", fontsize=7.4, color=MUTED)


def _draw(spec) -> None:
    """Draw one stage's flow diagram.

    The stage's operations are a *sequence*, not parallel branches: the
    intent classifier runs, then the entity extraction, then the rewrite.
    So the middle column is a vertical chain with arrows down it, and the
    input enters at the top and the output leaves at the bottom.
    """
    cols = spec["cols"]
    n_cols = len(cols)

    # The middle column is the operation chain; the outer two are the
    # single input and the single output.
    ops = cols[1]
    n_ops = len(ops)

    # Everything below is in inches, and the axes are set up so that one
    # data unit is one inch on both sides. Working in inches keeps the
    # boxes the same physical size no matter how long the chain is, so
    # the thirteen diagrams read as one set.
    box_w = 2.55          # inches
    col_gap = 0.46        # inches between columns
    vgap = 0.34           # inches between operations
    margin = 0.14
    head_h = 0.86         # inches for the purpose line and the column heads

    # The box height is computed from the text that will actually be
    # drawn: the label is wrapped first, and then the number of lines it
    # needs is counted. A box that is too short is a box whose text runs
    # over its edge, so this is measured rather than guessed.
    def _node_height(node):
        n_label = len(_wrap(node[1], int(box_w * 9.6)).split("\n"))
        n_tech = 0
        if len(node) > 2 and node[2]:
            n_tech = len(
                _wrap(node[2], int(box_w * 13.0)).split("\n")
            )
        # The label is anchored to the top of the box, and the technical
        # name is drawn below a hairline that follows the label's last
        # line, so the height has to hold the label's lines, the gap for
        # the rule, and the technical name's lines. The two blocks are
        # measured in inches: a line of label type is taller than a line
        # of the smaller technical type.
        label_h = n_label * 0.172
        tech_h = n_tech * 0.134
        if n_tech:
            return 0.20 + label_h + 0.19 + tech_h
        return 0.20 + label_h

    heights = [
        _node_height(node)
        for col in cols
        for node in col
    ]
    box_h = max(heights)

    chain_h = n_ops * box_h + (n_ops - 1) * vgap
    fig_w = n_cols * box_w + (n_cols - 1) * col_gap + 2 * margin
    fig_h = chain_h + head_h + 2 * margin

    # The axes span the figure exactly, so one data unit is one inch.
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    fontsize = 9.0

    # The purpose line, at the top, states what the stage is for.
    ax.text(margin, fig_h - margin - 0.15, spec["purpose"],
            ha="left", va="center", fontsize=10.2, color=TEXT)

    # The column heads, under a rule, tell the reader how to read the
    # diagram without having to be told. The rule is drawn below the
    # heads, and the chain starts below the rule, so nothing overlaps.
    head_y = fig_h - margin - 0.53
    rule_y = head_y - 0.20
    ax.plot([margin, fig_w - margin], [rule_y, rule_y],
            color=RULE, linewidth=0.9)
    for i, head in enumerate(COL_HEADS):
        cx = margin + i * (box_w + col_gap) + box_w / 2
        ax.text(cx, head_y, head, ha="center", va="center",
                fontsize=8.6, color=MUTED, style="italic")

    # The chain occupies the space below the rule, centred in it.
    chain_bottom = margin
    chain_top = fig_h - margin - 0.86
    top = chain_top
    op_pos = []
    for ri, node in enumerate(ops):
        kind, label = node[0], node[1]
        technical = node[2] if len(node) > 2 else None
        y = top - ri * (box_h + vgap) - box_h
        x = margin + box_w + col_gap
        _box(fig, ax, x, y, box_w, box_h, label, kind, technical, fontsize=fontsize)
        op_pos.append((x, y, box_w, box_h))

    # The input and output are centred on the chain.
    cy = chain_bottom + chain_h / 2
    in_pos = (margin, cy - box_h / 2, box_w, box_h)
    out_x = margin + 2 * (box_w + col_gap)
    out_pos = (out_x, cy - box_h / 2, box_w, box_h)
    _box(fig, ax, in_pos[0], in_pos[1], in_pos[2], in_pos[3], cols[0][0][1],
         "input", cols[0][0][2] if len(cols[0][0]) > 2 else None,
         fontsize=fontsize)
    _box(fig, ax, out_pos[0], out_pos[1], out_pos[2], out_pos[3], cols[2][0][1],
         "out", cols[2][0][2] if len(cols[2][0]) > 2 else None,
         fontsize=fontsize)

    # The chain runs downward.
    for i in range(len(op_pos) - 1):
        x, y, w, h = op_pos[i]
        nx, ny, nw, nh = op_pos[i + 1]
        _arrow(ax, (x + w / 2, y), (nx + nw / 2, ny + nh))

    # The input enters the first operation; the last operation leaves
    # into the output.
    x, y, w, h = op_pos[0]
    _arrow(ax, (in_pos[0] + in_pos[2], cy), (x, y + h / 2))
    x, y, w, h = op_pos[-1]
    _arrow(ax, (x + w, y + h / 2), (out_pos[0], cy))

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"step{spec['n']:02d}_{spec['slug'].split('_', 1)[1]}.png"
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def main() -> int:
    for spec in SPECS:
        _draw(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
