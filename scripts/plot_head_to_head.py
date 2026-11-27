"""The head-to-head with the closest published pipelines.

The architectural claim in Chapter 7 is not only that no published
pipeline covers all six capabilities, but that the three capabilities
which distinguish a defence pipeline from a single-pass detector --
claiming-level repair, re-verifying its own edit, and abstaining --
are absent from the nearest competitors.

This figure shows that, read from the survey's own head-to-head table
with the five closest pipelines. The table is the authority, so a
change to the survey changes the figure.

Like the other figures in this set, it carries no title and no
annotation: the caption states the finding and the plot shows only the
data.

Usage:
    python scripts/plot_head_to_head.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "docs" / "related_pipeline_workflows.md"
OUT = ROOT / "data" / "graphs" / "head_to_head.png"

OURS = "#2563eb"
HAVE = "#16a34a"
MISS = "#e2e8f0"
GRID = "#94a3b8"
TEXT = "#0f172a"

# The dimensions of the head-to-head table that separate a defence
# pipeline from a single-pass detector. Each is answered yes or no by
# the survey, so each can be drawn as a cell.
#
# Only the dimensions the survey answers with a tick are used. "Detection
# unit" and "Fusion" are free-text cells describing what the system
# does, not a yes-or-no capability, and a cell that is neither would
# read as "does not implement it" when it actually means "does it
# differently".
ROWS = [
    ("Repairs the\nanswer", "Mitigates"),
    ("Re-verifies\nits own edit", "Re-verifies own edit"),
    ("Abstains", "Abstains"),
]


def _head_to_head() -> tuple[list[str], dict[str, list[bool]]]:
    """Read the survey's head-to-head table.

    Returns the competitor names and, for each dimension, which of them
    implement it. The survey's own cells are the authority: a cell that
    is not a tick is a cell that does not implement the dimension.
    """
    if not SURVEY.exists():
        return [], {}
    lines = SURVEY.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## 3.7 Head-to-head"):
            start = i
            break
    if start is None:
        return [], {}

    header: list[str] | None = None
    rows: dict[str, list[str]] = {}
    for line in lines[start + 1:]:
        if line.startswith("### "):
            break
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            if cells and cells[0] == "Dimension":
                header = cells[1:]
            continue
        if len(cells) < 2:
            continue
        rows[cells[0]] = cells[1:]

    if not header:
        return [], {}

    names = [h.replace("*", "").strip() for h in header]
    out: dict[str, list[bool]] = {}
    for label, key in ROWS:
        cells = rows.get(key)
        if cells is None:
            continue
        out[label] = [c.strip().startswith("✓") for c in cells[:len(names)]]
    return names, out


def main() -> int:
    names, grid = _head_to_head()
    if not names or not grid:
        print("could not read the head-to-head table", flush=True)
        return 1

    # CRISP is the first column of the survey's table.
    ours = names[0]
    competitors = names[1:]
    dims = [label for label, _ in ROWS if label in grid]

    n_rows = len(dims)
    n_cols = len(competitors)
    height = 1.35 + n_rows * 0.72

    fig, ax = plt.subplots(figsize=(1.55 * n_cols + 2.9, height))
    ax.set_xlim(-0.1, n_cols + 1.35)
    ax.set_ylim(0, n_rows + 1.9)
    ax.invert_yaxis()
    ax.axis("off")

    # -- column headers ---------------------------------------------------
    for c, name in enumerate(competitors):
        ax.add_patch(FancyBboxPatch(
            (c + 0.06, -1.34), 0.88, 0.86,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#f1f5f9", edgecolor=GRID, linewidth=0.8,
        ))
        ax.text(c + 0.5, -0.91, name, ha="center", va="center",
                fontsize=8.4, color=TEXT, linespacing=1.3)

    # -- the rows ---------------------------------------------------------
    for r, label in enumerate(dims):
        y = r + 0.5
        ax.text(-0.18, y, label, ha="right", va="center",
                fontsize=8.8, color=TEXT, linespacing=1.3)
        flags = grid[label]
        for c in range(n_cols):
            on = flags[c + 1] if c + 1 < len(flags) else False
            ax.add_patch(FancyBboxPatch(
                (c + 0.06, y - 0.30), 0.88, 0.60,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=HAVE if on else MISS,
                edgecolor=HAVE if on else GRID,
                linewidth=0.7,
            ))
            if on:
                ax.text(c + 0.5, y, "✓", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")

    # -- the rule, then CRISP ---------------------------------------------
    rule_y = n_rows + 0.62
    ax.plot([-0.18, n_cols + 1.3], [rule_y, rule_y], color=TEXT, linewidth=1.1)

    y = n_rows + 1.42
    ax.text(-0.18, y, f"{ours} (Ours)", ha="right", va="center",
            fontsize=10.2, color=OURS, fontweight="bold")
    for c in range(n_cols):
        ax.add_patch(FancyBboxPatch(
            (c + 0.06, y - 0.30), 0.88, 0.60,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=OURS, edgecolor=OURS, linewidth=0,
        ))
        ax.text(c + 0.5, y, "✓", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
