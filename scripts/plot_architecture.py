"""Journal-ready pipeline architecture diagram.

The other charts plot measurements. This one plots *structure*: the 13
pipeline steps in the order they actually execute, the data that flows
between them, and the three-way decision the gate can return.

Two things keep it honest:

  * The step order is not typed in. It is read from
    ``src/steps/__init__.py`` -- the same registry the orchestrator
    imports -- so if the pipeline is reordered the diagram follows. This
    matters because the *numbering* is not the *execution* order:
    retrieval (6) and reranking (7) run before generation (2) so the
    generator receives real retrieved context in its prompt. A diagram
    drawn in numerical order would be wrong.
  * The measured numbers on the diagram (detection ROC-AUC per dataset)
    are read from the metrics files, not written into the source.

The diagram is drawn with matplotlib patches rather than a graphviz
``.dot`` file so it renders identically on a machine without graphviz
installed, which is the common case for a journal submission pipeline.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


OUT_DIR = Path("data/graphs")
REGISTRY_PATH = Path("src/steps/__init__.py")

# The colours follow the ones already used by the other charts, so a
# reader who has seen the detection charts recognises the datasets.
C_STAGE = "#2563eb"     # the pipeline's own stages
C_DETECT = "#0891b2"    # the detection / verification chain
C_GATE = "#dc2626"      # the decision points
C_TEXT = "#111827"
C_MUTED = "#64748b"
C_EDGE = "#94a3b8"
C_BAND = "#f1f5f9"

# Which group each step belongs to. The grouping is by role, not by
# number, and the boxes are laid out left to right in execution order.
GROUP_OF = {
    "query_understanding": "understand",
    "hybrid_retrieval": "ground",
    "reranking": "ground",
    "grounded_generation": "ground",
    "early_detection": "detect",
    "claim_extraction": "detect",
    "nli_verification": "detect",
    "confidence_assessment": "detect",
    "confidence_fusion": "detect",
    "regeneration": "gate",
    "final_gate": "gate",
    "explainable_response": "output",
    "evaluation": "output",
}

GROUP_LABEL = {
    "understand": "Query understanding",
    "ground": "Evidence grounding",
    "detect": "Detection & claim-level verification",
    "gate": "Mitigation & decision",
    "output": "Output & evaluation",
}

GROUP_COLOR = {
    "understand": "#6366f1",
    "ground": "#2563eb",
    "detect": "#0891b2",
    "gate": "#dc2626",
    "output": "#16a34a",
}

# A short label for each step. The step's own docstring gives the full
# title; this is the version that fits in a box. Thirteen boxes across the
# canvas is tight, so each line is kept short and the model names go on
# a third line where they are needed.
SHORT_LABEL = {
    "query_understanding": "Query\nunderstanding",
    "hybrid_retrieval": "Hybrid retrieval\nBGE-M3 + FAISS\n+ BM25",
    "reranking": "Reranking\nBGE-reranker-v2",
    "grounded_generation": "Grounded\ngeneration",
    "early_detection": "Early detection\nXGBoost",
    "claim_extraction": "Claim\nextraction",
    "nli_verification": "NLI verification\nDeBERTa-v3-MNLI",
    "confidence_assessment": "Confidence\nassessment\n(EEDC)",
    "confidence_fusion": "Confidence\nfusion",
    "regeneration": "Evidence-guided\nregeneration",
    "final_gate": "Final gate\nPASS / REGEN\n/ ABSTAIN",
    "explainable_response": "Explainable\nresponse",
    "evaluation": "Evaluation",
}


def _read_execution_order() -> list[tuple[int, str]]:
    """Read (step_number, step_name) in execution order from the registry.

    The registry is parsed rather than executed: importing it would pull in
    torch and the model zoo, which is not something a plotting script
    should need. The AST gives the list of class names in the order they
    appear, and the module names give the step names.

    Execution order differs from the numbering on purpose: retrieval (6)
    and reranking (7) run *before* generation (2) so the generator gets
    real retrieved context in its prompt.
    """
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)

    # The registry function returns a list of class references. Find the
    # list literal it returns.
    classes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.List):
            for elt in node.value.elts:
                if isinstance(elt, ast.Attribute) and isinstance(elt.value, ast.Name):
                    classes.append(elt.value.id)  # e.g. step06_hybrid_retrieval

    if not classes:
        return []

    out: list[tuple[int, str]] = []
    for mod in classes:
        m = re.match(r"step(\d+)_(\w+)$", mod)
        if not m:
            continue
        number = int(m.group(1))
        name = m.group(2)
        out.append((number, name))
    return out


def _dataset_aucs() -> list[tuple[str, str, float]]:
    """(label, short label, detection ROC-AUC) for each evaluated dataset.

    Read from the metrics files so the numbers on the diagram cannot
    drift from the measurements.
    """
    sources = [
        ("HaluEval QA (length-matched)", "data/metrics_halueval.json"),
        ("TruthfulQA (closed-book)", "data/metrics_truthfulqa_closedbook.json"),
        ("RAGTruth (length-matched)", "data/metrics_ragtruth_fitted.json"),
    ]
    out: list[tuple[str, str, float]] = []
    for label, path in sources:
        p = Path(path)
        if not p.exists():
            continue
        try:
            agg = json.loads(p.read_text(encoding="utf-8")).get("aggregate", {})
        except Exception:  # noqa: BLE001
            continue
        auc = agg.get("detection_roc_auc")
        if auc is None:
            continue
        short = label.split(" (")[0]
        out.append((label, short, float(auc)))
    return out


def _box(ax, x, y, w, h, text, face, edge, fontsize=8.2, weight="normal"):
    """Draw one rounded stage box."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.25, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=C_TEXT, zorder=4,
            linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color=C_EDGE, style="-|>", lw=1.2,
           rad=0.0, ls="-", zorder=2):
    """Draw an arrow between two points."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)


def plot_architecture() -> None:
    """Draw the pipeline architecture diagram."""
    order = _read_execution_order()
    if not order:
        print("  [skip] 12 architecture (could not read the step registry)")
        return

    aucs = _dataset_aucs()

    # The 13 boxes need room for their labels, and the results strip needs
    # its own band, so the canvas is tall enough for three horizontal
    # bands: stages, decisions, measured results.
    fig = plt.figure(figsize=(16.5, 9.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- header ----------------------------------------------------------
    ax.text(0.5, 0.975,
            "Hallucination-Mitigation: a 13-step retrieval-grounded "
            "detection and fact-verification pipeline",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color=C_TEXT)
    ax.text(0.5, 0.935,
            "Steps are drawn in execution order, not numerical order: "
            "retrieval (6) and reranking (7) run before generation (2) so "
            "the generator is grounded.",
            ha="center", va="top", fontsize=10, color=C_MUTED)

    # ---- the stage row ---------------------------------------------------
    n = len(order)
    left, right = 0.030, 0.970
    gap = 0.008
    bw = (right - left - gap * (n - 1)) / n
    top, bh = 0.800, 0.150

    xs = []
    for i, (number, name) in enumerate(order):
        x = left + i * (bw + gap)
        xs.append(x)
        group = GROUP_OF.get(name, "detect")
        face = "#ffffff"
        edge = GROUP_COLOR[group]
        _box(ax, x, top - bh, bw, bh,
             SHORT_LABEL.get(name, f"{number} · {name.replace('_', ' ')}"),
             face, edge, fontsize=6.6)
        # The step number, printed above its box.
        ax.text(x + bw / 2, top + 0.012, f"step {number:02d}",
                ha="center", va="bottom", fontsize=7.0, color=C_MUTED,
                fontweight="bold")

    # Arrows between consecutive stages.
    for i in range(n - 1):
        _arrow(ax, xs[i] + bw, top - bh / 2, xs[i + 1], top - bh / 2,
               color=C_EDGE, lw=1.2)

    # ---- group bands -----------------------------------------------------
    band_top = top + 0.042
    band_h = 0.026
    i = 0
    while i < n:
        g = GROUP_OF.get(order[i][1], "detect")
        j = i
        while j + 1 < n and GROUP_OF.get(order[j + 1][1], "detect") == g:
            j += 1
        x1 = xs[i] - 0.004
        x2 = xs[j] + bw + 0.004
        ax.add_patch(FancyBboxPatch(
            (x1, band_top), x2 - x1, band_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0, facecolor=GROUP_COLOR[g], alpha=0.16, zorder=1))
        ax.text((x1 + x2) / 2, band_top + band_h / 2, GROUP_LABEL[g].upper(),
                ha="center", va="center", fontsize=7.4, fontweight="bold",
                color=GROUP_COLOR[g], zorder=2)
        i = j + 1

    # ---- the feedback loop from the gate back to regeneration ----------
    gate_x = xs[[nm for _, nm in order].index("final_gate")]
    regen_x = xs[[nm for _, nm in order].index("regeneration")]
    y_loop = top - bh - 0.062
    # Down from the gate, left along the loop, up into regeneration. The
    # middle segment is a plain line: giving it an arrowhead pointed the
    # arrow at nothing.
    _arrow(ax, gate_x + bw / 2, top - bh, gate_x + bw / 2, y_loop,
           color=C_GATE, lw=1.2, style="-|>")
    ax.plot([regen_x + bw / 2, gate_x + bw / 2], [y_loop, y_loop],
            color=C_GATE, linewidth=1.2, zorder=2, solid_capstyle="butt")
    _arrow(ax, regen_x + bw / 2, y_loop, regen_x + bw / 2, top - bh,
           color=C_GATE, lw=1.2, style="-|>")
    # The label sits above the loop line, on its own row, so it cannot
    # sit on top of the line it describes.
    ax.text((gate_x + regen_x) / 2 + bw / 2, y_loop + 0.014,
            "REGEN: re-verify the rewritten answer (≤ 2 attempts)",
            ha="center", va="bottom", fontsize=8.4, color=C_GATE,
            fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none"))

    # ---- the three gate outcomes ----------------------------------------
    y_dec = 0.435
    ax.text(0.030, y_dec + 0.105,
            "THE GATE CAN RETURN THREE OUTCOMES",
            ha="left", va="center", fontsize=8.4, fontweight="bold",
            color=C_MUTED)
    outcomes = [
        ("PASS", "#16a34a", "answer is supported by the evidence"),
        ("REGEN", "#f59e0b", "rewrite the answer with retrieved evidence"),
        ("ABSTAIN", "#dc2626", "\"I don't know\" — an explicit refusal"),
    ]
    # The three boxes plus their gaps must fit inside the same margins as
    # the stage row, so the width is solved for rather than fixed.
    ogap = 0.026
    ox = 0.030
    obw = (0.940 - ogap * 2) / 3
    for name, color, note in outcomes:
        _box(ax, ox, y_dec, obw, 0.085,
             f"{name}\n{note}", "#ffffff", color, fontsize=9.6,
             weight="bold")
        ox += obw + ogap

    # ---- measured results strip ------------------------------------------
    y_strip = 0.255
    ax.add_patch(FancyBboxPatch(
        (0.030, y_strip - 0.075), 0.940, 0.150,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.0, edgecolor=C_EDGE, facecolor=C_BAND, zorder=1))
    ax.text(0.052, y_strip + 0.050, "MEASURED DETECTION ROC-AUC",
            ha="left", va="top", fontsize=8.6, fontweight="bold",
            color=C_MUTED, zorder=3)

    if aucs:
        sx = 0.052
        for label, short, auc in aucs:
            ax.text(sx, y_strip - 0.022, f"{auc:.3f}", ha="left", va="center",
                    fontsize=17, fontweight="bold", color="#1d4ed8", zorder=3)
            ax.text(sx + 0.092, y_strip - 0.022, short, ha="left", va="center",
                    fontsize=9.6, color=C_TEXT, zorder=3)
            sx += 0.305
    ax.text(0.948, y_strip + 0.050,
            "1,000 rows per dataset · 95% bootstrap CI over questions",
            ha="right", va="top", fontsize=8.4, color=C_MUTED, zorder=3)

    # ---- caption ---------------------------------------------------------
    ax.text(0.5, 0.030,
            "Every claim is verified individually against retrieved "
            "evidence (step 8), so the pipeline reports per-claim "
            "provenance and an explicit abstention state rather than a "
            "single answer-level score.",
            ha="center", va="bottom", fontsize=9.2, color=C_MUTED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "architecture_diagram.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] 12 architecture_diagram.png "
          f"({len(order)} steps, {len(aucs)} datasets)")


def main() -> None:
    print(f"Writing the architecture diagram to {OUT_DIR}/")
    plot_architecture()


if __name__ == "__main__":
    main()
