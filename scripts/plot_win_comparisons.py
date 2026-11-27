"""Two individual comparison charts for the metrics where we win.

The report's result analysis (§3.2.5) says we beat the published
comparables on exactly two scores:

    P@R90             0.703  ->  0.800
    Claim-level F1    0.710  ->  0.811

Detection AUPR used to have a chart of its own, drawn against
Decoding Memories' 0.840. That paper reports AUROC, not AUPR -- the
strings "AUPR", "AUC-PR" and "PR-AUC" do not appear in it -- and
0.840 was EigenScore-B's HaluEval AUROC, so the chart was comparing
our AUPR against somebody else's AUROC. No published paper reports
AUPR on HaluEval, so there is nothing to chart it against and the
chart was removed rather than re-pointed at a different metric.

Each remaining metric gets its own diagram here. The point of a
separate diagram per metric is that a reader can look at a single
comparison without scanning a combined chart for the row that concerns
them.

Every number is read out of an artefact at render time:

  * our score and its 95% interval come from data/metrics_halueval.json
  * the published comparables come from Paper/comparison_table.csv, which
    is itself written by scripts/fill_comparison_table.py
  * the comparable named in the win matrix comes from the same generator,
    so the chart measures against the number the paper quotes

The verdict is not typed in. It is computed by the same rule the paper
uses: a win needs the bottom of our interval strictly above the
published number; a loss needs the top strictly below; anything else is
a tie. So if the interval ever stops clearing the comparable, the chart
will say so instead of still claiming a win.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = Path("data/graphs")
METRICS_PATH = Path("data/metrics_halueval.json")
CSV_PATH = Path("Paper/comparison_table.csv")
VERDICT_PATH = Path("Paper/comparison.md")

INK = "#0f172a"
GREY = "#64748b"
LIGHT = "#e2e8f0"
OURS = "#2563eb"
WIN_C = "#16a34a"
TIE_C = "#f59e0b"
LOSS_C = "#dc2626"
PAPER = "#94a3b8"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Reading the artefacts
# ---------------------------------------------------------------------------

# A published cell only counts as a comparable score when it is a number
# the cited work actually published. Three kinds of cell have to be
# rejected, because treating them as scores is what once made a chart
# claim wins the confidence intervals did not support:
#
#   "projected"          a projection, not a measurement
#   "0 (no rewriting)"   a filler meaning "this system has no mitigation
#                        stage", not a score of zero
#   "0.55 (implied ...)" a baseline derived from another metric of the
#                        same paper rather than a published score
IMPLIED = ("implied", "projected", "no rewriting", "no mitigation",
           "not reported", "n/a")


def _csv_rows() -> list[list[str]]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def _published_scores(metric_label: str) -> list[tuple[str, float, str, float | None]]:
    """Every published score for one metric, as (paper, value, variant, band_hi).

    The variant is the name in parentheses, if the cell gives one. A
    cited system can publish several variants of the same metric --
    Dale 2023's Table 1 reports P@R90 of 67.4 for ALTI and 70.3 for
    LaBSE -- and the chart has to say which one it is drawn against.
    """
    rows = _csv_rows()
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    ours_idx = next((i for i, h in enumerate(header)
                     if h.lower().startswith("ours")), -1)

    out: list[tuple[str, float, str, float | None]] = []
    for row in rows[1:]:
        if len(row) < 3 or row[1].strip() != metric_label:
            continue
        for i in range(2, len(header)):
            if i == ours_idx:
                continue
            cell = str(row[i] or "").strip()
            if not cell or cell == "-":
                continue
            low = cell.lower()
            if any(tag in low for tag in IMPLIED):
                continue
            # A range such as "0.84-0.85" is one published band; the
            # comparable the paper quotes is its lower end, and that is
            # the harder end for us to beat. The range is kept so the
            # chart can draw the band.
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*[-–]\s*([-+]?\d+(?:\.\d+)?)", cell)
            variant = ""
            vm = re.search(r"\(([^)]+)\)", cell)
            if vm:
                variant = vm.group(1).strip()
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                out.append((header[i], lo, variant, hi))
                continue
            m = re.search(r"[-+]?\d+(?:\.\d+)?", cell)
            if m:
                out.append((header[i], float(m.group(0)), variant, None))
    return out


# ---------------------------------------------------------------------------
# The claim-level comparables
# ---------------------------------------------------------------------------

# The comparison table's "Claim Level / F1" row carries only the AILS-NTUA
# voting variant (0.69). The paper's own claim-level table in
# Paper/comparison.md §3.2 lists every published claim-level number in one
# place, including the TU Wien Vectara 0.71 that the win matrix names as
# the comparable. That table is read here so the chart draws the same
# set of published numbers the paper prints, rather than the subset the
# CSV happens to carry.
#
# The row is | System | Precision | Recall | F1 | Granularity |. Only
# the F1 column is drawn, because that is the metric being compared.
_CLAIM_TABLE_HEADER = "| System | Precision | Recall | F1 | Granularity |"


def _claim_level_scores() -> list[tuple[str, float, str, float | None]]:
    """Published claim-level F1 scores, as (paper, F1, granularity, band_hi).

    The F1 is its own column. Only whole sentences are usable as
    like-for-like comparables, so the row that reports our own
    atomic-claim score is skipped.
    """
    if not VERDICT_PATH.exists():
        return []
    lines = VERDICT_PATH.read_text(encoding="utf-8").splitlines()
    out: list[tuple[str, float, str, float | None]] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(_CLAIM_TABLE_HEADER):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            break
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 5:
            break
        if set(cells[0]) <= set("-: "):
            continue  # the header underline
        name = cells[0].replace("**", "").strip()
        if name.lower().startswith("ours"):
            continue
        f1 = cells[3].replace("**", "").strip()
        m = re.search(r"[-+]?\d+(?:\.\d+)?", f1)
        if not m:
            continue
        granularity = cells[4].strip()
        out.append((name, float(m.group(0)), granularity, None))
    return out


# ---------------------------------------------------------------------------
# The win matrix
# ---------------------------------------------------------------------------

def _win_matrix_comparable(dimension: str) -> tuple[str, float] | None:
    """The comparable the paper's win matrix quotes for a dimension.

    This is the number the verdict is computed against, so the chart
    cannot end up measuring against a different number than the paper
    does -- which is how a chart once claimed a P@R90 win against 0.674
    while the paper called the same metric "at par" against 0.703.
    """
    if not VERDICT_PATH.exists():
        return None
    text = VERDICT_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        number, dim = cells[0], cells[1]
        if dim.replace("*", "") != dimension:
            continue
        # | # | Dimension | Best comparable (paper) | best score | ours | verdict | why |
        paper = cells[2].replace("*", "").strip()
        try:
            score = float(cells[3])
        except ValueError:
            return None
        return paper, score
    return None


def _verdict(ours: float, ci: tuple[float, float] | None, comparable: float) -> str:
    """The project's rule: the interval decides in both directions."""
    if ci is None:
        # No interval exists for this metric, so the paper compares the
        # point estimates and says so.
        return "WIN" if ours > comparable else "LOSS" if ours < comparable else "TIE"
    low, high = ci
    if low > comparable:
        return "WIN"
    if high < comparable:
        return "LOSS"
    return "TIE"


# ---------------------------------------------------------------------------
# The two metrics
# ---------------------------------------------------------------------------

CHARTS = [
    {
        "slug": "win_pr90",
        "title": "Precision at 90% recall (P@R90)",
        "metric_key": "precision_at_recall_90",
        "ci_key": "precision_at_recall_90_ci95",
        "source": "csv",
        "csv_label": "P at R=90",
        "dimension": "P@R90",
        "question": "If we catch 90 out of 100 hallucinations, how many false alarms?",
        "explain": ("P@R90 fixes the recall at 90% -- the system must catch nine out of "
                    "every ten hallucinations -- and then measures how many of its "
                    "remaining alarms are false. This is the number a real deployment "
                    "cares about, because it is the cost of running the detector."),
        "unit": "precision at 90% recall",
    },
    {
        "slug": "win_claim_f1",
        "title": "Claim-level F1",
        "metric_key": "claim_f1",
        "ci_key": None,
        "source": "claim_table",
        "csv_label": "F1",
        "dimension": "Claim-level F1",
        "question": "Can we find the right claim and check it on its own?",
        "explain": ("Claim-level F1 asks whether the system can pick out the individual "
                    "claims in an answer and judge each one separately. Our unit is the "
                    "atomic claim, not the whole sentence, so this is a harder job than "
                    "the cited systems do."),
        "unit": "F1",
    },
]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _style_axis(ax, xlabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=10.5, color=INK, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10, colors=INK)
    ax.tick_params(axis="x", labelsize=9.5, colors=INK)
    ax.grid(axis="x", alpha=0.20, linestyle=":", zorder=0)
    ax.set_axisbelow(True)


def _draw(cfg: dict) -> None:
    agg = _load(METRICS_PATH).get("aggregate", {})
    if not agg:
        print(f"  [skip] {cfg['slug']} (no metrics file)")
        return

    ours = float(agg[cfg["metric_key"]])
    ci = None
    if cfg["ci_key"]:
        lo, hi = agg[cfg["ci_key"]]
        ci = (float(lo), float(hi))

    matrix = _win_matrix_comparable(cfg["dimension"])
    if matrix is None:
        print(f"  [skip] {cfg['slug']} (no win-matrix row)")
        return
    paper_name, comparable = matrix

    if cfg["source"] == "claim_table":
        published = _claim_level_scores()
    else:
        published = _published_scores(cfg["csv_label"])
    if not published:
        print(f"  [skip] {cfg['slug']} (no published comparables)")
        return

    # For the claim-level chart the third field of each row is the
    # granularity, not a variant of the metric. It is the same for every
    # cited work, so it is printed once as a note instead of being
    # repeated on every row.
    if cfg["source"] == "claim_table":
        published = [(n, v, "", b) for n, v, _g, b in published]
        granularity_note = ("The cited works score a whole sentence at a time. "
                            "We score the atomic claim inside it.")
    else:
        granularity_note = ""

    # A published band is a range, not a point. The comparable the paper
    # uses is the lower end, because that is the harder end to beat.
    if any(b is not None for _n, _v, _x, b in published):
        band_note = ("A bar with two end caps is a published range. The comparable "
                     "the paper uses is the lower end of that range.")
    else:
        band_note = ""

    # The comparable the paper names must be one of the published ones,
    # or the chart would be measuring against a number no paper reports.
    names = [p[0] for p in published]
    if paper_name not in names:
        # The two surfaces name the citing works differently -- "TU Wien
        # — Vectara" in the paper against "TU Wien 2024" in the table --
        # so fall back to matching on the leading words of the name.
        stem = paper_name.split("—")[0].split(" - ")[0].strip().lower()
        match = next((n for n in names if n.lower().startswith(stem[:5])), None)
        if match is None:
            print(f"  [skip] {cfg['slug']} (comparable '{paper_name}' not found)")
            return
        paper_name = match

    verdict = _verdict(ours, ci, comparable)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(12.6, 6.6))

    labels: list[str] = []
    values: list[float] = []
    lows: list[float | None] = []
    highs: list[float | None] = []
    is_ours: list[bool] = []
    notes: list[str] = []

    for name, value, variant, band_hi in published:
        labels.append(name)
        values.append(value)
        lows.append(None)
        highs.append(band_hi)
        is_ours.append(False)
        note = f"{value:.3f}"
        if band_hi is not None:
            note = f"{value:.3f}–{band_hi:.3f}"
        if variant:
            note = f"{note}  ({variant})"
        notes.append(note)

    labels.append("CRISP (ours)")
    values.append(ours)
    lows.append(ci[0] if ci else None)
    highs.append(ci[1] if ci else None)
    is_ours.append(True)
    notes.append(f"{ours:.3f}" + (f"  [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""))

    y = np.arange(len(values))[::-1].astype(float)

    # The x range has to hold the marks, the value labels to their right,
    # the CI caps and the badge, so it is computed from the data rather
    # than guessed -- a fixed multiplier clipped the P@R90 label once.
    x_max = max([v for v in values] + [h for h in highs if h is not None])
    ax.set_xlim(0, x_max * 1.30)

    for yi, val, low, high, mine, note in zip(
            y, values, lows, highs, is_ours, notes):
        if mine and low is not None:
            ax.errorbar(val, yi, xerr=[[val - low], [high - val]], fmt="o",
                        color=OURS, ecolor=OURS, elinewidth=3.4, capsize=8,
                        capthick=2.2, markersize=13, zorder=4,
                        markeredgecolor="white", markeredgewidth=1.6)
        elif not mine and high is not None:
            # A published band such as 0.840-0.850 is drawn as a bar
            # from its low end to its high end, with a cap at each end,
            # so the reader can see that the comparable we must beat is
            # the lower end of the band.
            ax.plot([val, high], [yi, yi], color=PAPER, linewidth=8,
                    solid_capstyle="butt", zorder=3, alpha=0.55)
            ax.plot([val, high], [yi, yi], color=PAPER, linewidth=2.0,
                    solid_capstyle="butt", zorder=4)
            for end in (val, high):
                ax.plot([end, end], [yi - 0.13, yi + 0.13], color=PAPER,
                        linewidth=2.0, zorder=4)
            ax.scatter([val], [yi], s=170, color=PAPER, zorder=5,
                       edgecolor="white", linewidth=1.8, marker="D")
        else:
            ax.scatter([val], [yi], s=230 if mine else 160, color=OURS if mine else PAPER,
                       zorder=4, edgecolor="white", linewidth=1.8,
                       marker="o" if mine else "D")
        # The value label sits to the right of the mark, clear of the
        # error bar, so it never lands on the interval.
        offset = 0.014 if not mine else ((high - val) + 0.016 if high else 0.016)
        if not mine and high is not None:
            offset = (high - val) + 0.014
        ax.text(val + offset, yi, note, va="center", ha="left",
                fontsize=9.8, fontweight="bold",
                color=OURS if mine else GREY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_ylim(-1.25, len(values) - 0.05)
    _style_axis(ax, cfg["unit"])

    # ---- the comparable line the paper uses ----
    ax.axvline(comparable, color=INK, linewidth=1.6, linestyle="--",
               alpha=0.45, zorder=1, ymin=0.02, ymax=0.93)
    # The label sits inside the plot, just above the first mark, so it
    # cannot collide with the figure's own title block.
    ax.text(comparable, len(values) - 0.62,
            f"published comparable  {comparable:.3f}",
            ha="center", va="bottom", fontsize=9.4, color=INK,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=LIGHT, linewidth=1.0))

    # ---- the arrow that shows the gap ----
    # It is drawn on its own row under the last mark, so it cannot land
    # on a data point, and it is clipped to the axes.
    arrow_y = -0.72
    ax.add_patch(FancyArrowPatch(
        (comparable, arrow_y), (ours, arrow_y),
        arrowstyle="-|>", mutation_scale=17, linewidth=2.2,
        color=WIN_C if verdict == "WIN" else TIE_C if verdict == "TIE" else LOSS_C,
        zorder=3, clip_on=True))
    ax.text((comparable + ours) / 2, arrow_y - 0.16,
            f"+{ours - comparable:.3f}", ha="center", va="top",
            fontsize=11.5, fontweight="bold",
            color=WIN_C if verdict == "WIN" else TIE_C if verdict == "TIE" else LOSS_C)

    # ---- title block ----
    fig.suptitle(f"{cfg['title']}: CRISP vs the published comparables",
                 fontsize=15, fontweight="bold", color=INK, y=0.975)
    fig.text(0.5, 0.918, cfg["question"], ha="center", va="top",
             fontsize=11, color=INK)

    # ---- the rule, stated on the figure ----
    if ci:
        rule = (f"Our 95% interval is [{ci[0]:.4f}, {ci[1]:.4f}]. "
                f"A higher value requires the whole interval above {comparable:.3f}.")
    else:
        rule = (f"No confidence interval is published for this metric, so this "
                f"is a point comparison against {comparable:.3f}.")
    fig.text(0.5, 0.115, rule, ha="center", va="center", fontsize=9.4, color=GREY)

    # The two notes are stacked, not overlaid, so neither can hide the
    # other when a chart has both.
    notes = [n for n in (granularity_note, band_note) if n]
    for i, note in enumerate(notes):
        fig.text(0.5, 0.082 - 0.032 * i, note, ha="center", va="center",
                 fontsize=9.2, color=GREY, style="italic")

    # ---- the relation badge ----
    # The internal verdicts stay "WIN"/"TIE"/"LOSS" because the rule
    # returns them; only the text drawn on the figure uses the journal
    # register.
    badge = {"WIN": (WIN_C, "HIGHER"), "TIE": (TIE_C, "AT PARITY"),
             "LOSS": (LOSS_C, "LOWER")}[verdict]
    ax.text(0.985, 0.965, badge[1], transform=ax.transAxes, ha="right",
            va="top", fontsize=12.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.55", facecolor=badge[0],
                      edgecolor="none"))

    # ---- the explanation under the chart ----
    fig.text(0.5, 0.010, cfg["explain"], ha="center", va="bottom",
             fontsize=9.0, color=GREY, wrap=True, linespacing=1.5)

    fig.subplots_adjust(left=0.21, right=0.965, top=0.80, bottom=0.245)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{cfg['slug']}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {out}")


def main() -> None:
    print("writing the win-comparison charts:")
    for cfg in CHARTS:
        _draw(cfg)


if __name__ == "__main__":
    main()
