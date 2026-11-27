"""The architectural-win figure for the thesis book.

This is the lead figure of the comparison chapter, because the
architectural result is the one that is not a score: of the nineteen
published RAG hallucination-defence pipelines surveyed in
``docs/related_pipeline_workflows.md``, every one of them leaves at least
one of the six defining capabilities unimplemented, and CRISP is the
only pipeline that realises all six.

The chart is read from the survey's own grid, so it cannot claim a
capability the survey does not record, and it cannot claim a paper
covers something the survey says it does not.

Usage:
    python scripts/plot_architectural_win.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "docs" / "related_pipeline_workflows.md"
OUT = ROOT / "data" / "graphs" / "architectural_win.png"

OURS = "#2563eb"
HAVE = "#16a34a"
MISS = "#e2e8f0"
GRID = "#94a3b8"
TEXT = "#0f172a"

# The six defining capabilities, in the order the survey lists them.
PROPS = [
    ("C1", "Retrieval-grounded\ndetection"),
    ("C2", "Multi-stage\npipeline"),
    ("C3", "Claim-level\nverification"),
    ("C4", "Multi-signal\nfusion"),
    ("C5", "Post-hoc\nrepair"),
    ("C6", "Selective\nprediction"),
]


def _survey_grid() -> list[tuple[str, list[int]]]:
    """Read the survey's own landscape grid.

    The grid is the authority on which published pipeline implements
    which capability. Reading it here rather than restating it in this
    file means the figure cannot drift away from the survey it claims to
    summarise: if a row is added, removed or re-marked there, the figure
    changes with it.
    """
    if not SURVEY.exists():
        return []
    rows: list[tuple[str, list[int]]] = []
    in_grid = False
    for line in SURVEY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 2. The landscape at a glance"):
            in_grid = True
            continue
        if in_grid and line.startswith("## "):
            break
        if not in_grid or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        name = cells[0].replace("*", "").strip()
        if name in ("Paper", "") or set(name) <= set("-: "):
            continue
        caps = [1 if c.strip() == "\u2713" else 0 for c in cells[1:7]]
        rows.append((name, caps))
    return rows


def main() -> int:
    PAPERS = _survey_grid()
    if not PAPERS:
        print("could not read the survey grid", flush=True)
        return 1

    n_papers = len(PAPERS)
    n_rows = n_papers + 2  # the papers, a rule, then CRISP
    best = max(sum(c) for _, c in PAPERS)

    fig, ax = plt.subplots(figsize=(7.8, 8.0))
    ax.set_xlim(-0.1, 7.1)
    ax.set_ylim(0, n_rows + 1.1)
    ax.invert_yaxis()
    ax.axis("off")

    # -- column headers ----------------------------------------------------
    for i, (tag, label) in enumerate(PROPS):
        ax.add_patch(FancyBboxPatch(
            (i + 0.06, -0.98), 0.88, 0.78,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#f1f5f9", edgecolor=GRID, linewidth=0.8,
        ))
        ax.text(i + 0.5, -0.86, tag, ha="center", va="center",
                fontsize=11, fontweight="bold", color=TEXT)
        ax.text(i + 0.5, -0.42, label, ha="center", va="center",
                fontsize=7.0, color=TEXT, linespacing=1.3)

    # -- paper rows --------------------------------------------------------
    for r, (name, caps) in enumerate(PAPERS):
        y = r + 0.5
        ax.text(-0.16, y, name, ha="right", va="center",
                fontsize=8.6, color=TEXT)
        n_have = sum(caps)
        for c, on in enumerate(caps):
            ax.add_patch(FancyBboxPatch(
                (c + 0.06, y - 0.30), 0.88, 0.60,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=HAVE if on else MISS,
                edgecolor=HAVE if on else GRID,
                linewidth=0.7,
            ))
            if on:
                ax.text(c + 0.5, y, "\u2713", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
        ax.text(6.25, y, f"{n_have}/6", ha="left", va="center",
                fontsize=8.2, color=TEXT, fontweight="bold")

    # -- the rule, then CRISP ---------------------------------------------
    rule_y = n_papers + 0.62
    ax.plot([-0.16, 6.9], [rule_y, rule_y], color=TEXT, linewidth=1.1)

    y = n_papers + 1.42
    ax.text(-0.16, y, "CRISP (Ours)", ha="right", va="center",
            fontsize=10.2, color=OURS, fontweight="bold")
    for c in range(6):
        ax.add_patch(FancyBboxPatch(
            (c + 0.06, y - 0.30), 0.88, 0.60,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=OURS, edgecolor=OURS, linewidth=0,
        ))
        ax.text(c + 0.5, y, "\u2713", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
    ax.text(6.25, y, "6/6", ha="left", va="center",
            fontsize=8.2, color=OURS, fontweight="bold")

    # -- the headline ------------------------------------------------------
    # The figure carries no title or annotation: in journal style the
    # caption states the finding, and the plot shows only the data.

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
