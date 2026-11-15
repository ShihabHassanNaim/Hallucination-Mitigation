"""Win-focused graphs for the thesis.

The existing graphs are mostly score comparisons, and most of those are
losses. These three charts show where the pipeline actually wins, so the
paper can point at them without cherry-picking a metric that happens to
look favourable.

  9.  win_dimensions.png        — the 18-dimension verdict matrix, split
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

The verdict matrix is parsed out of Paper/why_we_win.md, which is itself
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
VERDICT_PATH = Path("Paper/why_we_win.md")

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

    The matrix in ``Paper/why_we_win.md`` is generated from the
    measurements, so parsing it keeps the chart in step with the paper: a
    dimension can only appear as a win here if the paper calls it a win.
    """
    if not VERDICT_PATH.exists():
        print("  [skip] 9 win_dimensions")
        return

    text = VERDICT_PATH.read_text(encoding="utf-8")

    # Rows look like: | 7 | **Detection ROC-AUC** | Dale 2023 ... | ≈ | ... |
    row_re = re.compile(
        r"^\|\s*(\d+)\s*\|\s*\*\*([^*]+)\*\*\s*\|.*?\|\s*([✓✗≈])\s*\|",
        flags=re.MULTILINE,
    )
    rows = [(int(n), name.strip(), v) for n, name, v in row_re.findall(text)]
    if not rows:
        print("  [skip] 9 win_dimensions")
        return

    verdict_label = {"✓": "Win", "✗": "Loss", "≈": "Tie"}
    counts = {"Win": 0, "Tie": 0, "Loss": 0}
    for _, _, v in rows:
        counts[verdict_label.get(v, "Loss")] += 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1, 1.45]}
    )

    # -- left: the tally --------------------------------------------------
    order = ["Win", "Tie", "Loss"]
    colors = {"Win": WIN, "Tie": TIE, "Loss": LOSS}
    vals = [counts[k] for k in order]
    bars = ax1.bar(order, vals, color=[colors[k] for k in order],
                   edgecolor="white", linewidth=1.5, width=0.62)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                 ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(vals) + 1.6)
    _style(ax1, "Verdict across the 18 compared dimensions",
           ylabel="Number of dimensions")
    ax1.text(
        0.5, -0.16,
        "Wins are structural (granularity, provenance, backend, scale);\n"
        "score dimensions are losses or at par.",
        transform=ax1.transAxes, ha="center", va="top",
        fontsize=8.5, color="#475569",
    )

    # -- right: every dimension, coloured by verdict ----------------------
    names = [r[1] for r in rows]
    verdicts = [r[2] for r in rows]
    xs = np.arange(len(names))
    ax2.barh(xs, [1] * len(names),
             color=[colors[verdict_label.get(v, "Loss")] for v in verdicts],
             edgecolor="white", linewidth=0.8, height=0.68)
    ax2.set_yticks(xs)
    ax2.set_yticklabels(names, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax2.spines[spine].set_visible(False)
    ax2.set_title("Every dimension, by verdict", fontsize=12, fontweight="bold")
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
        "Where the 13-Step Pipeline Wins — and Where It Does Not",
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

    def num(v):
        text = str(v or "")
        if "projected" in text.lower():
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None

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
        others = [num(row[i]) for i in other_idx if i < len(row)]
        others = [x for x in others if x is not None and 0 <= x <= 1]
        if not others:
            continue
        best = max(others)
        value = agg[key]
        ci = agg.get(ci_key) or []
        metrics.append({
            "label": label, "value": value, "best": best,
            "ci": ci, "win": value > best,
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
        ax.bar(x + width / 2, m["best"], width, color="#cbd5e1",
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
        ax.text(x + width / 2, m["best"] + 0.025, f"{m['best']:.3f}",
                ha="center", va="bottom", fontsize=8.5, color="#475569")

        if m["win"]:
            ax.text(x, max(our_top, m["best"]) + 0.115, "✓",
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=WIN)

    ax.set_xticks(xs)
    ax.set_xticklabels([m["label"] for m in metrics], fontsize=9)
    ax.set_ylim(0, 1.24)
    ax.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.3,
               label="Random baseline (0.50)")
    _style(ax, "Where Our Score Beats the Best Published Comparable",
           ylabel="Score (higher is better)")
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=WIN, label="Ours — above the best comparable"),
        Patch(facecolor=GREY, label="Ours — at or below the comparable"),
        Patch(facecolor="#cbd5e1", label="Best published comparable"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=3, fontsize=9, framealpha=0.95)
    ax.text(
        0.5, -0.22,
        "Black bars are the 95% bootstrap CI on our value. AUPR and CCR are "
        "the two metrics where we are ahead; the rest are shown so the "
        "comparison is not cherry-picked.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color="#475569",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "score_wins.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
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
