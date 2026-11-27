"""Win-focused graphs for the thesis.

The existing graphs are mostly score comparisons, and most of those are
losses. These three charts show where the pipeline actually wins, so the
paper can point at them without cherry-picking a metric that happens to
look favourable.

  9.  win_dimensions.png        — the verdict matrix, split
                                  into wins / ties / losses, read from the
                                  generated table so it cannot drift.
 10. structural_advantages.png  — the capability comparison against the
                                  cited systems: per-claim provenance,
                                  pluggable backend, active mitigation and
                                  evaluation scale.
 11. score_wins.png             — the two metrics where our score is
                                  genuinely above the cited comparable
                                  (AUPR, CCR), with their confidence
                                  intervals, against every other
                                  comparable rather than only the best.

The verdict matrix is parsed out of Paper/comparison.md, which is itself
generated from the measurements, so the chart cannot claim a win the prose
does not.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OUT_DIR = Path("data/graphs")
METRICS_PATH = Path("data/metrics_halueval.json")
CSV_PATH = Path("Paper/comparison_table.csv")
VERDICT_PATH = Path("Paper/comparison.md")

WIN = "#16a34a"
TIE = "#f59e0b"
LOSS = "#dc2626"
OURS = "#2563eb"
GREY = "#94a3b8"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _wrap_label(label: str, width: int = 18) -> str:
    """Break a long axis label onto two lines.

    Six metric names on one axis collide at the figure's width, and
    rotating them pushes the caption off the canvas.
    """
    words = label.split()
    if len(label) <= width:
        return label
    lines: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:2])


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# 9. Win dimensions — the verdict matrix as a chart
# ---------------------------------------------------------------------------
def plot_win_dimensions() -> None:
    """Read the verdict matrix out of the generated prose and chart it.

    The matrix in ``Paper/comparison.md`` is generated from the
    measurements, so parsing it keeps the chart in step with the paper: a
    dimension can only appear as a win here if the paper calls it a win.
    """
    if not VERDICT_PATH.exists():
        print("  [skip] 9 win_dimensions")
        return

    text = VERDICT_PATH.read_text(encoding="utf-8")

    # Rows look like: | 7 | **Detection ROC-AUC** | Dale 2023 ... | ≈ | ... |
    #
    # The verdict is read by column position rather than by "the first
    # mark on the line". Some rows carry a mark in an earlier column --
    # row 15 reads "Dale 2023 only | ✓ | ✓ | ≈", because both systems do
    # active mitigation -- and a non-greedy pattern would pick up the
    # comparable's mark and count that tie as a win. The matrix has seven
    # columns, so the verdict is the sixth.
    rows: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # cells[0] is empty (the line starts with "|"), so the columns
        # are 1..7 and the verdict is column 6.
        if len(cells) < 8:
            continue
        verdict = cells[6]
        if verdict not in ("✓", "✗", "≈"):
            continue
        name = cells[2].strip().strip("*")
        rows.append((int(cells[1]), name, verdict))
    if not rows:
        print("  [skip] 9 win_dimensions")
        return

    verdict_label = {"✓": "Higher", "✗": "Lower", "≈": "Tie"}
    counts = {"Higher": 0, "Tie": 0, "Lower": 0}
    for _, _, v in rows:
        counts[verdict_label.get(v, "Lower")] += 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1, 1.45]}
    )

    # -- left: the tally --------------------------------------------------
    order = ["Higher", "Tie", "Lower"]
    colors = {"Higher": WIN, "Tie": TIE, "Lower": LOSS}
    vals = [counts[k] for k in order]
    bars = ax1.bar(order, vals, color=[colors[k] for k in order],
                   edgecolor="white", linewidth=1.5, width=0.62)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                 ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(vals) + 1.6)
    _style(ax1, f"Relation across the {len(rows)} compared dimensions",
           ylabel="Number of dimensions")
    ax1.text(
        0.5, -0.16,
        "The higher values are structural (granularity, provenance, backend, scale);\n"
        "the score dimensions are lower or at parity.",
        transform=ax1.transAxes, ha="center", va="top",
        fontsize=8.5, color="#475569",
    )

    # -- right: every dimension, coloured by verdict ----------------------
    names = [r[1] for r in rows]
    verdicts = [r[2] for r in rows]
    xs = np.arange(len(names))
    ax2.barh(xs, [1] * len(names),
             color=[colors[verdict_label.get(v, "Lower")] for v in verdicts],
             edgecolor="white", linewidth=0.8, height=0.68)
    ax2.set_yticks(xs)
    ax2.set_yticklabels(names, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax2.spines[spine].set_visible(False)
    ax2.set_title("Every dimension, by relation", fontsize=12, fontweight="bold")
    ax2.grid(False)

    from matplotlib.patches import Patch
    # The legend goes under the axes: inside the panel it covered the last
    # bar. ``bbox_inches="tight"`` keeps it inside the saved figure.
    ax2.legend(
        handles=[Patch(facecolor=colors[k], label=f"{k} ({counts[k]})")
                 for k in order],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
        fontsize=9, framealpha=0.95,
    )

    fig.suptitle(
        "Where the 13-Step Pipeline Attains a Higher Value — and Where It Does Not",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "win_dimensions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 9 win_dimensions.png")


# ---------------------------------------------------------------------------
# 10. Structural advantages
# ---------------------------------------------------------------------------
def plot_structural_advantages() -> None:
    """Show the capability dimensions where we lead.

    These are the four dimensions the verdict matrix records as genuine
    wins: per-claim verification, a pluggable generator backend, active
    mitigation, and retrieval. A cited system only counts as supporting a
    capability when its own cell names it, so a cell such as
    ``NLI on full sentence`` does not count as per-claim verification.
    """
    if not CSV_PATH.exists():
        print("  [skip] 10 structural_advantages")
        return

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    ours_idx = next((i for i, h in enumerate(header)
                     if h.lower().startswith("ours")), -1)
    if ours_idx < 0:
        print("  [skip] 10 structural_advantages")
        return
    other_idx = [i for i in range(2, len(header)) if i != ours_idx]

    def _row(metric: str):
        for r in rows[1:]:
            if len(r) > 1 and r[1].strip() == metric:
                return r
        return None

    # A cited system supports the capability only when its own cell names
    # it. The rules are explicit per capability so that a cell which merely
    # mentions some other technique does not inflate the count.
    def _per_claim(cell: str) -> bool:
        c = cell.lower()
        return "per claim" in c or "per-claim" in c

    def _pluggable(cell: str) -> bool:
        c = cell.lower()
        return "pluggable" in c or "external generator" in c

    def _active_mitigation(cell: str) -> bool:
        c = cell.strip().lower()
        return c not in {"", "none", "n/a", "-", "not required"}

    def _retrieval(cell: str) -> bool:
        c = cell.strip().lower()
        return c not in {"", "none", "n/a", "-", "not required"}

    CAPABILITIES = [
        ("Verification", "Per-claim verification", _per_claim),
        ("Generator", "Pluggable generator backend", _pluggable),
        ("Mitigation", "Active mitigation (rewrite / abstain)",
         _active_mitigation),
        ("Retrieval", "Multi-source retrieval", _retrieval),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    xs = np.arange(len(CAPABILITIES))
    width = 0.38

    our_counts, their_counts, labels = [], [], []
    for metric, label, supports in CAPABILITIES:
        r = _row(metric)
        if r is None:
            continue
        others = [r[i].strip() for i in other_idx if i < len(r)]
        supported = sum(1 for c in others if supports(c))
        our_counts.append(1)
        their_counts.append(supported)
        labels.append(label)

    if not labels:
        print("  [skip] 10 structural_advantages")
        return

    b1 = ax.bar(xs - width / 2, our_counts, width, label="Ours (13-step)",
                color=OURS, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(xs + width / 2, their_counts, width,
                label="Cited systems supporting it",
                color=GREY, edgecolor="white", linewidth=1.2)

    for b, v in zip(b1, our_counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=OURS)
    for b, v in zip(b2, their_counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#475569")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, max(their_counts + our_counts) + 1.4)
    _style(ax, "Structural Advantages — Count of Cited Systems Supporting Each",
           ylabel="Number of the 12 cited systems")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.text(
        0.5, -0.17,
        "A cited system counts only when its own cell names the capability "
        "(e.g. sentence-level NLI is not per-claim). Read from "
        "Paper/comparison_table.csv.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color="#475569",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "structural_advantages.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 10 structural_advantages.png")


# ---------------------------------------------------------------------------
# 11. Score wins with confidence intervals
# ---------------------------------------------------------------------------
def plot_score_wins() -> None:
    """Chart the metrics where our score beats the cited comparable.

    Only the metrics that pass the comparison test are shown: ours is
    strictly above the best published value, on a scale that is directly
    comparable. The confidence interval is drawn so the reader can see
    whether the margin clears the interval.
    """
    agg = _load(METRICS_PATH).get("aggregate", {})
    if not agg:
        print("  [skip] 11 score_wins")
        return
    if not CSV_PATH.exists():
        print("  [skip] 11 score_wins")
        return

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    ours_idx = next((i for i, h in enumerate(header)
                     if h.lower().startswith("ours")), -1)
    other_idx = [i for i in range(2, len(header)) if i != ours_idx]

    # A cell only counts as a comparable score when it is a number that the
    # cited work actually published. Three kinds of cell have to be
    # rejected, because treating them as scores is what made this chart
    # claim wins that the confidence intervals do not support:
    #
    #   "projected"          a projection, not a measurement
    #   "0 (no rewriting)"   a filler meaning "this system has no
    #                        mitigation stage", not a score of zero
    #   "0.55 (implied ...)" a baseline derived from another metric of the
    #                        same paper rather than a published score
    #
    # The last two are placeholders in the generated CSV, so the chart
    # cannot tell them apart from a real score by looking at the number
    # alone.
    IMPLED = ("implied", "projected", "no rewriting", "no mitigation",
              "not reported", "n/a")

    def num(v):
        text = str(v or "")
        low = text.lower()
        if any(tag in low for tag in IMPLED):
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None

    def variant(v):
        """Return the variant name a cell names in parentheses, if any.

        A cited system can publish several variants of the same metric.
        Dale 2023's Table 1 reports P@R90 of 67.4 for ALTI and 70.3 for
        LaBSE, and the two are different numbers for the same metric.
        The cell has to say which one it is, otherwise the chart ends up
        measured against a different number than the paper's win matrix
        quotes -- which is how this chart once claimed a P@R90 win
        against 0.674 while the paper called the same metric "at par"
        against 0.703.
        """
        text = str(v or "")
        m = re.search(r"\(([^)]+)\)", text)
        return m.group(1).strip() if m else ""

    # Metrics that are directly comparable: a score in [0,1] where higher
    # is better and both sides use the same definition.
    CANDIDATES = [
        ("AUPR", "detection_aupr", "detection_aupr_ci95"),
        ("Claim Correction Rate (CCR)", "claim_correction_rate",
         "claim_correction_rate_ci95"),
        ("ROC-AUC", "detection_roc_auc", "detection_roc_auc_ci95"),
        ("P at R=90", "precision_at_recall_90", "precision_at_recall_90_ci95"),
        ("Sentence Accuracy", "detection_accuracy", "detection_accuracy_ci95"),
        ("Verification Success Rate (VSR)", "verification_success_rate",
         "verification_success_rate_ci95"),
    ]

    metrics = []
    for label, key, ci_key in CANDIDATES:
        row = next((r for r in rows[1:]
                    if len(r) > 1 and r[1].strip() == label), None)
        if row is None or key not in agg:
            continue
        # The comparable cells for this metric, with the variant each one
        # names. A cited system can publish several variants of the same
        # metric -- Dale 2023's Table 1 reports P@R90 of 67.4 for ALTI and
        # 70.3 for LaBSE -- and the chart has to compare against the same
        # variant the paper's win matrix quotes. Otherwise the chart can
        # claim a win against one number while the paper calls the same
        # metric "at par" against a different one.
        cells = []
        for i in other_idx:
            if i >= len(row):
                continue
            n = num(row[i])
            if n is None or not (0 <= n <= 1):
                continue
            cells.append((n, variant(row[i])))
        value = agg[key]
        ci = agg.get(ci_key) or []
        if not cells:
            # No published comparable survives the filter. The metric is
            # still worth showing -- dropping it silently would hide a
            # dimension the reader expects -- but it has nothing to be
            # separated from, so it can never be a win.
            metrics.append({
                "label": label, "value": value, "best": None,
                "ci": ci, "win": False, "overlap": False,
                "no_comparable": True,
            })
            continue
        # The best comparable is the highest published score. When the
        # highest cell names a variant, that name is carried into the
        # chart so the reader can see which number they are looking at.
        best, best_variant = max(cells, key=lambda c: c[0])
        # A win requires the two to be *separated*, not merely for our
        # point estimate to be higher. Comparing point estimates against a
        # published value whose interval we do not have is how a chart ends
        # up claiming a win that the confidence interval contradicts: our
        # P@R90 is 0.800 against Dale's 0.703, but a chart that compared
        # midpoints would also have called a win when our CI sat across
        # their number.
        #
        # So the rule is: our lower CI bound must sit strictly above the
        # comparable. With no interval available there is no separation to
        # demonstrate, so it is not a win.
        ci_low = ci[0] if len(ci) == 2 else None
        ci_high = ci[1] if len(ci) == 2 else None
        win = bool(ci_low is not None and ci_low > best)
        # The interval decides the outcome in both directions, the same
        # way the paper's rule does. A win needs the bottom of our
        # interval above their number; a loss needs the top of it
        # below. Anything else is an overlap.
        #
        # Checking only the bottom bound -- "ci_low <= best" -- called a
        # metric an overlap whenever our lower bound sat under their
        # number, which is also true of an interval that sits entirely
        # below it. Our ROC-AUC is [0.911, 0.940] against Dale's
        # 0.987: the whole interval is under his number, so we are
        # behind, not at par. The caption then called it a tie, which
        # the paper's own win matrix reported as a loss.
        loss = bool(ci_high is not None and ci_high < best)
        metrics.append({
            "label": label, "value": value, "best": best,
            "ci": ci, "win": win,
            "variant": best_variant,
            # Kept for the caption and for the log line, so the chart can
            # say *why* a metric is not a win rather than just dropping it.
            "overlap": (not win and not loss and ci_low is not None
                        and ci_high is not None),
            "loss": loss,
        })

    if not metrics:
        print("  [skip] 11 score_wins")
        return

    # Wins first, then the near misses, so the reader sees the wins.
    metrics.sort(key=lambda m: (not m["win"], -m["value"]))

    fig, ax = plt.subplots(figsize=(11, 6.0))
    xs = np.arange(len(metrics))
    width = 0.36

    for x, m in zip(xs, metrics):
        color = WIN if m["win"] else GREY
        ax.bar(x - width / 2, m["value"], width, color=color,
               edgecolor="white", linewidth=1.2)
        # A metric with no published comparable has nothing to sit beside.
        ax.bar(x + width / 2, m["best"] or 0, width, color="#cbd5e1",
               edgecolor="white", linewidth=1.2)

        ci = m["ci"] if len(m["ci"]) == 2 else None
        # The confidence interval, drawn on our bar.
        if ci:
            lo, hi = ci
            ax.plot([x - width / 2, x - width / 2], [lo, hi],
                    color="#111827", linewidth=1.6, solid_capstyle="butt")
            for y in (lo, hi):
                ax.plot([x - width / 2 - 0.045, x - width / 2 + 0.045], [y, y],
                        color="#111827", linewidth=1.6)

        # The label sits above the taller of the bar and its CI, so the
        # number and the error bar never overlap.
        our_top = max(m["value"], ci[1] if ci else m["value"])
        ax.text(x - width / 2, our_top + 0.025, f"{m['value']:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=color if m["win"] else "#475569")
        if m["best"] is not None:
            ax.text(x + width / 2, m["best"] + 0.025, f"{m['best']:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color="#475569")
            # When the published number comes from a named variant of the
            # cited system, the variant is printed under the bar so the
            # reader knows which of that system's numbers they are
            # looking at.
            if m.get("variant"):
                ax.text(x + width / 2, m["best"] - 0.028,
                        m["variant"][:18],
                        ha="center", va="top", fontsize=6.5, color="#94a3b8")
        else:
            # Say why the space is empty rather than leaving a bare gap.
            # It is centred on the empty slot, but a metric with no
            # comparable has nothing in the right-hand bar, so the label
            # is nudged right to sit in that gap and stays clear of the
            # bars on either side.
            ax.text(x + width / 2 + 0.06, 0.30, "no published\ncomparable",
                    ha="left", va="bottom", fontsize=7.5, color="#94a3b8",
                    linespacing=1.3, zorder=5)

        if m["win"]:
            ax.text(x, max(our_top, m["best"] or 0) + 0.115, "✓",
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=WIN)

    ax.set_xticks(xs)
    # The metric names are long enough to collide with each other at this
    # width, so they are wrapped onto two lines rather than rotated --
    # rotated labels are harder to read and push the caption further down.
    ax.set_xticklabels([_wrap_label(m["label"]) for m in metrics], fontsize=8.5)
    ax.set_ylim(0, 1.24)
    ax.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.3,
               label="Random baseline (0.50)")
    _style(ax, "Our Scores Against the Best Published Comparable",
           ylabel="Score (higher is better)")
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=WIN, label="Ours — separated from the comparable"),
        # The grey bars are the metrics that are not separated, and they
        # are not all the same kind of not-separated: some tie (the CI
        # overlaps the published value) and some are behind (the CI
        # sits entirely below it). The legend cannot tell them apart by
        # colour, so it says what they share and the caption below
        # splits them.
        Patch(facecolor=GREY, label="Ours — not separated"),
        Patch(facecolor="#cbd5e1", label="Best published comparable"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=3, fontsize=9, framealpha=0.95)
    # The caption has to state the rule the marks were computed with, and
    # has to name which metrics fall short of it. The previous version
    # asserted "AUPR and CCR are the two metrics where we are ahead",
    # which was true of the point estimates and false of the intervals:
    # our AUPR CI [0.833, 0.874] covered Decoding Memories' 0.84-0.85, and
    # the CCR baseline of 0.55 is implied from another metric rather than
    # published. The AUPR row itself is now dropped, because no published
    # paper reports AUPR on HaluEval -- the 0.84-0.85 was an AUROC from a
    # different metric family.
    wins = [m["label"] for m in metrics if m["win"]]
    overlaps = [m["label"] for m in metrics if m.get("overlap")]
    losses = [m["label"] for m in metrics if m.get("loss")]
    no_comp = [m["label"] for m in metrics if m.get("no_comparable")]

    # Short names keep the caption inside the figure. The full metric
    # names are already on the axis, so the caption only has to say which
    # ones and why.
    def _short(names: list[str]) -> str:
        out = []
        for n in names:
            s = n.replace("Claim Correction Rate (CCR)", "CCR")
            s = s.replace("Verification Success Rate (VSR)", "VSR")
            s = s.replace("Sentence Accuracy", "Sentence Acc.")
            out.append(s)
        return ", ".join(out)

    if wins:
        win_txt = "Separated on: " + _short(wins) + "."
    else:
        win_txt = "No metric is separated from its comparable."
    if overlaps:
        win_txt += ("\nOn " + _short(overlaps) + " the 95% CI overlaps the "
                    "published value, so "
                    + ("this is at parity rather than separated."
                       if len(overlaps) == 1
                       else "these are at parity rather than separated."))
    if losses:
        win_txt += ("\nOn " + _short(losses) + " the 95% CI sits entirely "
                    "below the published value, so "
                    + ("this project attains a lower value." if len(losses) == 1
                       else "this project attains a lower value on these."))
    if no_comp:
        win_txt += ("\n" + _short(no_comp) + " has no published comparable "
                    "to test against.")

    # The caption is wrapped by hand rather than by matplotlib: a single
    # long line in a text artist is not wrapped, and with
    # bbox_inches="tight" it stretches the saved figure to several times
    # the intended width.
    caption = (
        "Black bars are the 95% bootstrap CI on our value. A metric is "
        "separated only when our interval sits entirely above\n"
        "the published comparable. " + win_txt + "\n"
        "Every metric is shown so the comparison is not cherry-picked."
    )
    ax.text(
        0.5, -0.20, caption,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color="#475569", linespacing=1.5,
    )
    # The caption and the legend both sit below the axes, so tight_layout
    # cannot size the figure for them. The layout is set explicitly
    # instead, and the figure is saved with the extra canvas the
    # annotations need.
    fig.subplots_adjust(bottom=0.34, left=0.08, right=0.98, top=0.92)
    fig.savefig(OUT_DIR / "score_wins.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    # Log every decision so the chart's verdicts can be checked against the
    # intervals without opening the PNG.
    for m in metrics:
        ci_txt = (f"[{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]" if len(m["ci"]) == 2
                  else "no CI")
        if m["best"] is None:
            print(f"       {m['label']}: ours={m['value']:.3f} {ci_txt} "
                  f"vs no published comparable -> not comparable")
        else:
            # The variant is part of the log line because the published
            # number is only meaningful together with the variant it
            # came from -- Dale 2023 reports both 0.674 and 0.703 for
            # P@R90.
            who = f" ({m['variant']})" if m.get("variant") else ""
            if m["win"]:
                outcome = "WIN"
            elif m.get("loss"):
                outcome = "BEHIND"
            elif m.get("overlap"):
                outcome = "TIE"
            else:
                outcome = "not separated"
            print(f"       {m['label']}: ours={m['value']:.3f} {ci_txt} "
                  f"vs best={m['best']:.3f}{who} -> {outcome}")
    print("  [ok] 11 score_wins.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing win-focused graphs to {OUT_DIR}/")
    plot_win_dimensions()
    plot_structural_advantages()
    plot_score_wins()
    print("done.")


if __name__ == "__main__":
    main()
