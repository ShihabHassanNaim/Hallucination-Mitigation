"""The capability-coverage profile of the surveyed pipelines.

The architectural comparison in Chapter 7 rests on two claims: that no
published pipeline implements all six capabilities, and that the gap is
distributed unevenly -- that post-hoc repair and selective prediction
are the thin categories, so the combination is rare for a structural
reason rather than an accident.

This figure shows the second claim directly: how many of the surveyed
pipelines implement each capability. It is read from the survey's own
grid, so it cannot show a distribution the survey does not record.

Like the architectural-win figure, it carries no title and no
annotation: in journal style the caption states the finding and the
plot shows only the data.

Usage:
    python scripts/plot_capability_coverage.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_architectural_win import PROPS, _survey_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "graphs" / "capability_coverage.png"

OURS = "#2563eb"
BAR = "#94a3b8"
GRID = "#e2e8f0"
TEXT = "#0f172a"


def main() -> int:
    rows = _survey_grid()
    if not rows:
        print("could not read the survey grid", flush=True)
        return 1

    n_papers = len(rows)
    counts = [sum(caps[i] for _, caps in rows) for i in range(len(PROPS))]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = np.arange(len(PROPS))
    width = 0.62

    bars = ax.bar(xs, counts, width, color=BAR, edgecolor="white",
                  linewidth=1.0)
    # The count goes above the bar, and the CRISP row is marked at the
    # full height so that the reader can see the gap the figure is about.
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, str(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=TEXT)
    ax.plot(xs, [n_papers] * len(PROPS), linestyle="--", linewidth=1.3,
            color=OURS, zorder=3)
    ax.text(xs[0] - 0.42, n_papers + 0.7, f"CRISP ({n_papers} of {n_papers})",
            ha="left", va="bottom", fontsize=9, fontweight="bold",
            color=OURS)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{tag}\n{label}" for tag, label in PROPS],
                       fontsize=8.2, linespacing=1.35)
    ax.set_ylabel("Published pipelines implementing the capability",
                  fontsize=9.5, labelpad=8)
    ax.set_ylim(0, n_papers + 3.2)
    ax.set_yticks(range(0, n_papers + 1, max(1, n_papers // 5)))
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
