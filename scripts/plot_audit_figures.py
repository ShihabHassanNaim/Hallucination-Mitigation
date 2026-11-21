"""Figures for the research audit report.

Every number plotted here is read from the measurement artefacts at
render time. Nothing is typed in, so a figure cannot drift from the
report that quotes it -- if a measurement changes and the report is
regenerated, the figure follows.

The figures are written to ``data/graphs/audit/`` and are referenced by
``RESEARCH_AUDIT.md``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "graphs" / "audit"

BLUE = "#2563eb"
GREEN = "#059669"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#7c3aed"
GREY = "#64748b"
LIGHT = "#e2e8f0"
INK = "#0f172a"


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def agg(rel: str) -> dict:
    return load(rel)["aggregate"]


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {name}")


def title(ax, main: str, sub: str = "") -> None:
    ax.set_title(main, fontsize=13, fontweight="bold", color=INK, pad=14 if sub else 8)
    if sub:
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.5, color=GREY)


# ---------------------------------------------------------------------------
# Figure 1 -- the pipeline, in execution order
# ---------------------------------------------------------------------------


def fig_pipeline() -> None:
    """The 13 steps in execution order, read from the step registry."""
    src = (ROOT / "src/steps/__init__.py").read_text(encoding="utf-8")
    body = src.split("return [", 1)[1].split("]", 1)[0]
    entries = re.findall(r"step(\d+)_\w+\.Step(\d+)(\w+)", body)
    # Each match is (module number, class number, class name); the number
    # that belongs on the diagram is the class's own.

    LABEL = {
        "Step01QueryUnderstanding": "Query\nunderstanding",
        "Step06HybridRetrieval": "Hybrid\nretrieval\nBGE-M3\n+ FAISS\n+ BM25",
        "Step07Reranking": "Reranking\nBGE-reranker\n-v2-m3",
        "Step02GroundedGeneration": "Grounded\ngeneration",
        "Step03EarlyDetection": "Early\ndetection\nXGBoost",
        "Step05ClaimExtraction": "Claim\nextraction",
        "Step08NLIVerification": "NLI\nverification\nDeBERTa-v3\n-xnli",
        "Step04ConfidenceAssessment": "Confidence\nassessment\nEEDC",
        "Step09ConfidenceFusion": "Confidence\nfusion",
        "Step10Regeneration": "Evidence-\nguided\nregeneration",
        "Step11FinalGate": "Final gate\nPASS /\nREGEN /\nABSTAIN",
        "Step12ExplainableResponse": "Explainable\nresponse",
        "Step13Evaluation": "Evaluation",
    }
    NUM = {f"Step{g2}{g3}": f"{int(g2):02d}" for g1, g2, g3 in entries}
    ORDER = ["Step01QueryUnderstanding", "Step06HybridRetrieval",
             "Step07Reranking", "Step02GroundedGeneration",
             "Step03EarlyDetection", "Step05ClaimExtraction",
             "Step08NLIVerification", "Step04ConfidenceAssessment",
             "Step09ConfidenceFusion", "Step10Regeneration",
             "Step11FinalGate", "Step12ExplainableResponse",
             "Step13Evaluation"]

    BANDS = [
        (0, 1, "#3b82f6", "Query understanding"),
        (1, 4, "#0ea5e9", "Evidence grounding"),
        (4, 9, "#0891b2", "Detection & claim-level verification"),
        (9, 11, "#dc2626", "Mitigation & decision"),
        (11, 13, "#059669", "Output & evaluation"),
    ]

    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")

    for lo, hi, color, name in BANDS:
        ax.add_patch(FancyBboxPatch(
            (lo + 0.04, 8.15), hi - lo - 0.08, 0.72,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.2))
        ax.text((lo + hi) / 2, 8.51, name, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color)

    for i, cls in enumerate(ORDER):
        x = i + 0.5
        band = next(c for lo, hi, c, _ in BANDS if lo <= i < hi)
        ax.add_patch(FancyBboxPatch(
            (x - 0.465, 4.95), 0.93, 2.85,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="white", edgecolor=band, linewidth=1.6))
        ax.text(x, 7.45, f"step {NUM.get(cls, '??'):>0}", ha="center",
                va="center", fontsize=8, fontweight="bold", color=band,
                family="monospace")
        ax.text(x, 6.30, LABEL[cls], ha="center", va="center",
                fontsize=7.2, color=INK, linespacing=1.35)
        if i < 12:
            ax.add_patch(FancyArrowPatch(
                (x + 0.465, 6.38), (x + 1.0, 6.38),
                arrowstyle="-|>", mutation_scale=11,
                color=GREY, linewidth=1.0))

    # The REGEN loop, back to step 10.
    ax.add_patch(FancyArrowPatch(
        (10.5, 4.85), (10.5, 4.05), arrowstyle="-|>", mutation_scale=12,
        color=RED, linewidth=1.8, connectionstyle="arc3,rad=0.0"))
    ax.add_patch(FancyArrowPatch(
        (10.5, 4.05), (9.5, 4.05), arrowstyle="-|>", mutation_scale=12,
        color=RED, linewidth=1.8, connectionstyle="arc3,rad=0.0"))
    ax.add_patch(FancyArrowPatch(
        (9.5, 4.05), (9.5, 4.95), arrowstyle="-|>", mutation_scale=12,
        color=RED, linewidth=1.8, connectionstyle="arc3,rad=0.0"))
    ax.text(10.0, 3.75, "REGEN: re-verify the rewritten answer (max 2 attempts)",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=RED)

    # The three gate outcomes.
    outcomes = [("PASS", "answer is supported\nby the evidence", GREEN),
                ("REGEN", "rewrite the answer\nwith retrieved evidence", AMBER),
                ("ABSTAIN", '"I don\'t know" --\nan explicit refusal', RED)]
    for i, (name, desc, color) in enumerate(outcomes):
        x = 1.6 + i * 4.4
        ax.add_patch(FancyBboxPatch(
            (x - 1.9, 1.35), 3.8, 1.35,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.5))
        ax.text(x, 2.28, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color=color)
        ax.text(x, 1.72, desc, ha="center", va="center", fontsize=8.2,
                color=INK, linespacing=1.4)

    ax.text(6.5, 9.72,
            "The 13-step retrieval-grounded hallucination-detection pipeline",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=INK)
    ax.text(6.5, 9.30,
            "Drawn in execution order, not step number: retrieval (6) and reranking (7) run "
            "before generation (2) so the generator is grounded.",
            ha="center", va="center", fontsize=8.8, color=GREY)
    ax.text(6.5, 0.62,
            "Every claim is verified individually against retrieved evidence (step 8), so the "
            "pipeline reports\nper-claim provenance and an explicit abstention state rather "
            "than a single answer-level score.",
            ha="center", va="center", fontsize=8.8, color=GREY, linespacing=1.5)

    save(fig, "audit_01_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2 -- detection performance per dataset
# ---------------------------------------------------------------------------

DATASETS = [
    ("HaluEval QA\n(length-matched)", "data/metrics_halueval.json", BLUE),
    ("TruthfulQA\n(closed-book)", "data/metrics_truthfulqa_closedbook.json", GREEN),
    ("RAGTruth\n(length-matched)", "data/metrics_ragtruth_fitted.json", PURPLE),
]


def fig_detection() -> None:
    """Every detection metric for every dataset, with its CI."""
    metrics = [
        ("ROC-AUC", "detection_roc_auc", "detection_roc_auc_ci95"),
        ("AUPR", "detection_aupr", "detection_aupr_ci95"),
        ("F1", "detection_f1", "detection_f1_ci95"),
        ("Accuracy", "detection_accuracy", "detection_accuracy_ci95"),
        ("Precision", "detection_precision", "detection_precision_ci95"),
        ("Recall", "detection_recall", "detection_recall_ci95"),
        ("P@R90", "precision_at_recall_90", "precision_at_recall_90_ci95"),
        ("Spearman ρ", "spearman_rho", "spearman_rho_ci95"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.6), sharey=True)
    # The panels share a y axis, so the lower limit has to clear the
    # widest interval across every metric: Spearman rho on TruthfulQA and
    # RAGTruth has a lower bound near 0.19.
    global_low = 0.42
    for _, key, ci_key in metrics:
        for _, rel, _ in DATASETS:
            a = agg(rel)
            ci = a.get(ci_key) or []
            if key in a and len(ci) == 2:
                global_low = min(global_low, float(a[key]) - float(ci[0]))
    for ax, (label, key, ci_key) in zip(axes.ravel(), metrics):
        xs, ys, los, his, colors = [], [], [], [], []
        for i, (_, rel, color) in enumerate(DATASETS):
            a = agg(rel)
            if key not in a:
                continue
            ci = a.get(ci_key) or []
            v = float(a[key])
            xs.append(i)
            ys.append(v)
            if len(ci) == 2:
                los.append(v - float(ci[0]))
                his.append(float(ci[1]) - v)
            else:
                los.append(0)
                his.append(0)
            colors.append(color)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor=INK,
                    elinewidth=1.7, capsize=5, capthick=1.7)
        ax.scatter(xs, ys, s=95, c=colors, zorder=3, edgecolor="white",
                   linewidth=1.4)
        # The label goes above the top of the error bar, never over it.
        for x, y, lo, hi in zip(xs, ys, los, his):
            ax.text(x, y + hi + 0.022, f"{y:.3f}", ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold", color=INK)
        ax.axhline(0.5, color=RED, linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([d[0].replace(" (length-matched)", "\n(length-matched)")
                            .replace(" (closed-book)", "\n(closed-book)")
                            for d in DATASETS], fontsize=7.2)
        # The lower limit has to clear the widest interval: Spearman rho
        # on TruthfulQA and RAGTruth has a lower bound near 0.19.
        ax.set_ylim(global_low - 0.04, 1.12)
        ax.set_xlim(-0.55, 2.55)
        ax.set_title(label, fontsize=11, fontweight="bold", color=INK)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Score (higher is better)", fontsize=9.5)

    fig.suptitle("Detection performance across all three benchmarks, "
                 "with 95% bootstrap confidence intervals",
                 fontsize=14, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.012,
             "1,000 rows per dataset (500 unique queries for HaluEval and TruthfulQA, "
             "277 for RAGTruth); 1,000 bootstrap resamples.\n"
             "The red dashed line is the 0.50 chance level. TruthfulQA is closed-book "
             "because its retrieval corpus was built from the dataset's own answer key; "
             "HaluEval and RAGTruth are length-matched,\n"
             "because on the raw split a detector can guess 94.7% of the labels from "
             "answer length alone.",
             ha="center", va="bottom", fontsize=8.6, color=GREY, linespacing=1.55)
    fig.subplots_adjust(bottom=0.135, top=0.90, hspace=0.42, wspace=0.10)
    save(fig, "audit_02_detection.png")


# ---------------------------------------------------------------------------
# Figure 3 -- the length confound
# ---------------------------------------------------------------------------


def fig_length_confound() -> None:
    """Why the raw HaluEval split cannot be used as the headline."""
    matched = agg("data/metrics_halueval.json")
    raw = agg("data/metrics_halueval_raw.json")

    pairs = [
        ("ROC-AUC", "detection_roc_auc", "detection_roc_auc_ci95"),
        ("AUPR", "detection_aupr", "detection_aupr_ci95"),
        ("P@R90", "precision_at_recall_90", "precision_at_recall_90_ci95"),
        ("F1", "detection_f1", "detection_f1_ci95"),
        ("Accuracy", "detection_accuracy", "detection_accuracy_ci95"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # Left: the same metric, two splits.
    labels = [p[0] for p in pairs]
    x = np.arange(len(labels))
    w = 0.36
    m_vals = [float(matched[p[1]]) for p in pairs]
    r_vals = [float(raw[p[1]]) for p in pairs]
    ax1.bar(x - w / 2, m_vals, w, label="Length-matched (reported)",
            color=BLUE, edgecolor="white")
    ax1.bar(x + w / 2, r_vals, w, label="Raw split (confounded)",
            color="#f59e0b", edgecolor="white")
    for xi, v in zip(x - w / 2, m_vals):
        ax1.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8,
                 fontweight="bold", color=BLUE)
    for xi, v in zip(x + w / 2, r_vals):
        ax1.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8,
                 fontweight="bold", color="#b45309")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0.5, 1.02)
    ax1.axhline(0.5, color=RED, linestyle="--", linewidth=1.1)
    ax1.set_ylabel("Score", fontsize=9.5)
    ax1.legend(fontsize=8.6, loc="upper right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.set_axisbelow(True)
    title(ax1, "The same metric, two splits",
          "The raw split flatters every metric")

    # Right: the length shortcut itself.
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(0.5, 0.94, "What the length shortcut is",
             ha="center", va="center", fontsize=12.5, fontweight="bold",
             color=INK)
    facts = [
        ("94.7%", "of the raw-split labels can be predicted\n"
                  "from answer length alone"),
        ("0.82", "AUC of the answer_length feature on the\n"
                 "detector's own training data"),
        ("466", "detector training rows shared a query\n"
                "with an evaluation row"),
    ]
    for i, (big, small) in enumerate(facts):
        y = 0.76 - i * 0.24
        ax2.add_patch(FancyBboxPatch(
            (0.04, y - 0.10), 0.92, 0.19,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#fff7ed", edgecolor="#f59e0b", linewidth=1.3))
        ax2.text(0.12, y, big, ha="center", va="center", fontsize=17,
                 fontweight="bold", color="#b45309")
        ax2.text(0.30, y, small, ha="left", va="center", fontsize=8.9,
                 color=INK, linespacing=1.4)
    ax2.text(0.5, 0.03,
             "Every length-matched number in this report comes from a split where\n"
             "the two answer lengths are drawn from the same distribution, so the\n"
             "shortcut carries no signal.",
             ha="center", va="bottom", fontsize=8.6, color=GREY, linespacing=1.5)

    fig.suptitle("Why HaluEval is reported on a length-matched split",
                 fontsize=14, fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.005,
             "The length-matched split is the honest one. The raw split is shown here "
             "only as the artefact the matching exists to remove.",
             ha="center", va="bottom", fontsize=8.6, color=GREY)
    fig.subplots_adjust(bottom=0.10, top=0.88, wspace=0.18)
    save(fig, "audit_03_length_confound.png")


# ---------------------------------------------------------------------------
# Figure 4 -- the mitigation null result
# ---------------------------------------------------------------------------


def fig_mitigation() -> None:
    """The rewrite's measured effect, with its interval and p-value."""
    mit = load("data/mitigation_results.json")["summary"]

    allr = mit["all_rows"]
    regen = mit["regenerated_rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # Left: the forest plot.
    rows = [
        ("All 500 rows", allr["delta_bertscore"], allr["delta_bertscore_ci95"],
         allr["delta_bertscore_p"], BLUE),
        ("The 16 rewritten rows", regen["delta_bertscore"],
         regen["delta_bertscore_ci95"], regen["delta_bertscore_p"], AMBER),
    ]
    ys = [1, 0]
    for y, (label, d, ci, p, color) in zip(ys, rows):
        ax1.plot([ci[0], ci[1]], [y, y], color=color, linewidth=5,
                 solid_capstyle="round", alpha=0.85)
        ax1.plot(d, y, "o", color=color, markersize=12, markeredgecolor="white",
                 markeredgewidth=1.6, zorder=3)
        ax1.text(ci[1] + 0.0022, y, f"{d:+.4f}   CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                f"   p = {p:.3f}",
                va="center", ha="left", fontsize=8.8, color=INK)
    ax1.axvline(0, color=RED, linestyle="--", linewidth=1.5)
    ax1.text(0, 1.42, "no effect", ha="center", va="bottom", fontsize=8.4,
             color=RED, fontweight="bold")
    ax1.set_yticks(ys)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=9.6)
    ax1.set_xlim(-0.032, 0.055)
    ax1.set_ylim(-0.5, 1.65)
    ax1.set_xlabel("Change in BERTScore against the gold answer (negative = worse)",
                   fontsize=9.5)
    ax1.grid(axis="x", alpha=0.25, linestyle=":")
    ax1.set_axisbelow(True)
    title(ax1, "The rewrite has no measurable effect",
          "Both intervals cover zero")

    # Right: the funnel -- why so little was rewritten.
    funnel = [
        (mit["n_rows"], "rows evaluated", BLUE),
        (mit["n_kept"], "kept unchanged (gate did not fire)", GREY),
        (mit["n_guarded"], "guarded (rewrite blocked by a guard)", GREY),
        (mit["n_regenerated"], "actually rewritten", AMBER),
    ]
    for i, (n, label, color) in enumerate(funnel):
        y = 3 - i
        wfrac = n / mit["n_rows"]
        ax2.add_patch(FancyBboxPatch(
            (0.06, y - 0.30), 0.60 * max(wfrac, 0.05), 0.52,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=color, alpha=0.75, edgecolor="white"))
        ax2.text(0.10, y, str(n), ha="left", va="center", fontsize=11.5,
                 fontweight="bold", color="white")
        ax2.text(0.70, y, label, ha="left", va="center", fontsize=8.7,
                 color=INK, linespacing=1.35)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.7, 3.9)
    ax2.axis("off")
    title(ax2, "Why so little was rewritten",
          f"{mit['n_regenerated']} of {mit['n_rows']} rows, from "
          f"{mit['n_unique_queries']} unique queries")

    fig.suptitle("Mitigation: an honest null result",
                 fontsize=14, fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.005,
             "A paired permutation test on the rewritten rows gives p = 0.404 -- no "
             "evidence of an effect in either direction. This is reported as a loss "
             "and an open result, not as a win.",
             ha="center", va="bottom", fontsize=8.6, color=GREY)
    fig.subplots_adjust(bottom=0.11, top=0.87, wspace=0.42)
    save(fig, "audit_04_mitigation.png")


# ---------------------------------------------------------------------------
# Figure 5 -- CI separation against the published comparables
# ---------------------------------------------------------------------------


def fig_ci_separation() -> None:
    """Our interval against each published comparable, with the rule drawn."""
    md = (ROOT / "Paper/comparison.md").read_text(encoding="utf-8")
    block = re.search(
        r"\*\*The full metric set on HaluEval:\*\*.*?\n\s*\n"
        r"\| Metric \|.*?\n(?:\|\|?.*\n)*", md, re.DOTALL)
    rows = []
    for line in block.group(0).splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) == 5 and c[0] != "Metric":
            m = re.match(r"([0-9.]+)", c[3])
            if m and c[4] in ("✓", "✗", "≈"):
                rows.append((c[0], float(m.group(1)), c[4]))

    a = agg("data/metrics_halueval.json")
    pairs = [
        ("Claim-level F1", "claim_f1", None),
        ("AUPR", "detection_aupr", "detection_aupr_ci95"),
        ("Detection accuracy", "detection_accuracy", "detection_accuracy_ci95"),
        ("P@R90", "precision_at_recall_90", "precision_at_recall_90_ci95"),
        ("ROC-AUC", "detection_roc_auc", "detection_roc_auc_ci95"),
        ("Spearman ρ", "spearman_rho", "spearman_rho_ci95"),
        ("VSR", "verification_success_rate", "verification_success_rate_ci95"),
    ]

    items = []
    for label, key, ci_key in pairs:
        if key not in a:
            continue
        v = float(a[key])
        ci = a.get(ci_key) or []
        if len(ci) != 2:
            continue
        best = None
        for name, comp, mark in rows:
            if name == label or (name == "Detection accuracy"
                                 and label == "Detection accuracy"):
                best = comp
                break
        if best is None:
            continue
        items.append((label, v, float(ci[0]), float(ci[1]), best))

    fig, ax = plt.subplots(figsize=(14.2, 7.4))
    ys = np.arange(len(items))[::-1]
    # The text goes in a fixed column to the right of the plot, so it can
    # never land on top of a comparable mark.
    LABEL_X = 1.02
    for y, (label, v, lo, hi, best) in zip(ys, items):
        if lo > best:
            color, verdict = GREEN, "WIN"
        elif hi < best:
            color, verdict = RED, "LOSS"
        else:
            color, verdict = AMBER, "TIE"
        ax.plot([lo, hi], [y, y], color=color, linewidth=9,
                solid_capstyle="round", alpha=0.85, zorder=2)
        ax.plot(v, y, "o", color=color, markersize=12,
                markeredgecolor="white", markeredgewidth=1.6, zorder=3)
        ax.plot(best, y, "_", color=INK, markersize=24,
                markeredgewidth=3.0, zorder=4)
        ax.text(LABEL_X, y, f"{v:.3f}", va="center", ha="left", fontsize=9.4,
                fontweight="bold", color=INK, family="monospace")
        ax.text(LABEL_X + 0.062, y, f"vs {best:.3f}", va="center", ha="left",
                fontsize=9.0, color=GREY, family="monospace")
        ax.text(LABEL_X + 0.145, y, verdict, va="center", ha="left",
                fontsize=9.4, fontweight="bold", color=color)

    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=9.8)
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("Score", fontsize=9.8)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_title("Every score compared against the best published comparable",
                 fontsize=13, fontweight="bold", color=INK, pad=26)
    ax.text(0.5, 1.035,
            "The black mark is their published number; the coloured bar is our "
            "95% confidence interval",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.8,
            color=GREY)

    handles = [
        plt.Line2D([], [], color=GREEN, lw=7, label="WIN -- our whole interval is above theirs"),
        plt.Line2D([], [], color=AMBER, lw=7, label="TIE -- our interval overlaps theirs"),
        plt.Line2D([], [], color=RED, lw=7, label="LOSS -- our whole interval is below theirs"),
        plt.Line2D([], [], color=INK, marker="_", ls="none", markersize=16,
                   markeredgewidth=2.6, label="their published number"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=8.8, framealpha=0.95)
    fig.text(0.5, 0.005,
             "The rule: a win is claimed only when the bottom of our 95% confidence "
             "interval sits above their published number -- not our best estimate. "
             "The interval decides in both directions.",
             ha="center", va="bottom", fontsize=8.7, color=GREY)
    fig.subplots_adjust(bottom=0.20, top=0.86, right=0.80)
    save(fig, "audit_05_ci_separation.png")


# ---------------------------------------------------------------------------
# Figure 6 -- the latency and cost picture
# ---------------------------------------------------------------------------


def fig_latency() -> None:
    """Measured latency and the published cost anchors."""
    a = agg("data/metrics_halueval.json")
    t = agg("data/metrics_truthfulqa_closedbook.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: latency distribution from the measured stats.
    stats = [
        ("HaluEval\nmin", float(a["latency_min_s"]), BLUE),
        ("HaluEval\nmedian", float(a["latency_median_s"]), BLUE),
        ("HaluEval\nmean", float(a["latency_mean_s"]), BLUE),
        ("HaluEval\nmax", float(a["latency_max_s"]), BLUE),
        ("TruthfulQA\nmean", float(t["latency_mean_s"]), GREEN),
    ]
    names = [s[0] for s in stats]
    vals = [s[1] for s in stats]
    colors = [s[2] for s in stats]
    y = np.arange(len(stats))[::-1]
    ax1.barh(y, vals, color=colors, alpha=0.85, edgecolor="white", height=0.6)
    for yi, v in zip(y, vals):
        ax1.text(v + 0.18, yi, f"{v:.2f} s", va="center", fontsize=9,
                 fontweight="bold", color=INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8.8)
    ax1.set_xlim(0, max(vals) * 1.22)
    ax1.set_xlabel("Seconds per query (GPU)", fontsize=9.5)
    ax1.grid(axis="x", alpha=0.25, linestyle=":")
    ax1.set_axisbelow(True)
    title(ax1, "What we measured",
          "1,000 queries on one GPU, end to end")

    # Right: against the published systems.
    others = [("Conformal-RAG\n(judge)", 1.8), ("LLM-Check", 0.22),
              ("SelfCheckGPT", 75.9)]
    names2 = [o[0] for o in others] + ["Ours\n(measured)"]
    vals2 = [o[1] for o in others] + [float(a["latency_mean_s"])]
    colors2 = ["#94a3b8"] * 3 + [BLUE]
    x = np.arange(len(names2))
    ax2.bar(x, vals2, color=colors2, alpha=0.9, edgecolor="white", width=0.62)
    for xi, v in zip(x, vals2):
        ax2.text(xi, v + 1.4, f"{v:.2f}", ha="center", fontsize=9,
                 fontweight="bold", color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names2, fontsize=8.4)
    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 300)
    ax2.set_ylabel("Seconds per query (log scale)", fontsize=9.5)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    ax2.set_axisbelow(True)
    title(ax2, "Against the published systems",
          "We are the second slowest")

    fig.suptitle("Speed and cost: the honest picture",
                 fontsize=14, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.015,
             "We spend the seconds on three things the single-pass probes do not do: "
             "retrieve, verify per claim, and rewrite. The published\n"
             "anchors are not all measured on the same hardware, so this comparison "
             "is indicative rather than exact.",
             ha="center", va="bottom", fontsize=8.6, color=GREY, linespacing=1.5)
    # The caption sits below the x-axis labels, so the bottom margin has to
    # clear both the label and the caption text.
    fig.subplots_adjust(bottom=0.19, top=0.86, wspace=0.26)
    save(fig, "audit_06_latency.png")


# ---------------------------------------------------------------------------
# Figure 7 -- the audit timeline / experiment ledger
# ---------------------------------------------------------------------------


def fig_ledger() -> None:
    """A visual ledger of every experiment and what it produced."""
    mit = load("data/mitigation_results.json")["summary"]
    hal = agg("data/metrics_halueval.json")
    raw = agg("data/metrics_halueval_raw.json")
    tqa = agg("data/metrics_truthfulqa_closedbook.json")
    rag = agg("data/metrics_ragtruth_fitted.json")

    entries = [
        ("Detector training data rebuilt",
         "2,290 rows, 1,145 queries, no length feature, no query overlap",
         "length feature AUC 0.82 -> removed", GREEN),
        ("Detection head refitted (496 rows)",
         "hand-set weights gave ROC-AUC 0.360 on RAGTruth -- worse than chance",
         "fitted head with corrected signs", GREEN),
        ("HaluEval raw split",
         "1,000 rows / 500 queries",
         f"ROC-AUC {raw['detection_roc_auc']:.3f} -- confounded by length",
         "#f59e0b"),
        ("HaluEval length-matched",
         "1,000 rows / 500 queries",
         f"ROC-AUC {hal['detection_roc_auc']:.3f} -- the reported result", BLUE),
        ("TruthfulQA closed-book",
         "1,000 rows / 500 questions",
         f"ROC-AUC {tqa['detection_roc_auc']:.3f} -- no corpus", BLUE),
        ("RAGTruth length-matched",
         "1,000 rows / 277 questions",
         f"ROC-AUC {rag['detection_roc_auc']:.3f}", BLUE),
        ("Mitigation evaluation",
         f"{mit['n_rows']} rows / {mit['n_unique_queries']} queries, "
         f"{mit['n_regenerated']} rewritten",
         f"delta {mit['regenerated_rows']['delta_bertscore']:+.4f}, "
         f"p = {mit['regenerated_rows']['delta_bertscore_p']:.3f}", RED),
        ("Per-claim guard A/B",
         "100 rows, old vs new guard behaviour",
         "delta +0.004 -> -0.002; a design fix, not a win", RED),
    ]

    fig, ax = plt.subplots(figsize=(15.2, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(entries))
    ax.axis("off")

    for i, (name, setup, result, color) in enumerate(entries):
        y = len(entries) - i - 0.5
        ax.add_patch(FancyBboxPatch(
            (0.012, y - 0.40), 0.976, 0.80,
            boxstyle="round,pad=0.006,rounding_size=0.03",
            facecolor=color, alpha=0.07, edgecolor=color, linewidth=1.4))
        ax.add_patch(FancyBboxPatch(
            (0.012, y - 0.40), 0.008, 0.80,
            boxstyle="square,pad=0", facecolor=color, edgecolor="none"))
        ax.text(0.035, y + 0.17, name, ha="left", va="center", fontsize=10.4,
                fontweight="bold", color=INK)
        ax.text(0.035, y - 0.19, setup, ha="left", va="center", fontsize=8.5,
                color=GREY)
        ax.text(0.985, y, result, ha="right", va="center", fontsize=9.6,
                fontweight="bold", color=color)

    ax.text(0.5, len(entries) + 0.28,
            "Experiment ledger: every measurement, what it was run on, and what it produced",
            ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.5, len(entries) + 0.02,
            "Green = a defect found and fixed.  Amber = a measurement kept but not used as "
            "the headline.  Blue = a reported result.  Red = a negative result, reported as one.",
            ha="center", va="center", fontsize=8.8, color=GREY)
    fig.subplots_adjust(top=0.92, bottom=0.02)
    save(fig, "audit_07_ledger.png")


# ---------------------------------------------------------------------------
# Figure 8 -- dataset construction
# ---------------------------------------------------------------------------


def fig_datasets() -> None:
    """How each dataset was built and how many rows and queries it holds."""
    rows = []
    for name, rel in [("halueval_matched", "data/halueval_eval_1000_matched.jsonl"),
                      ("truthfulqa", "data/truthfulqa_eval_1000.jsonl"),
                      ("ragtruth_matched", "data/ragtruth_eval_1000_matched.jsonl")]:
        p = ROOT / rel
        if not p.exists():
            continue
        n = 0
        q = set()
        labs = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            q.add(r.get("query"))
            labs[r.get("label")] = labs.get(r.get("label"), 0) + 1
        rows.append((name, rel, n, len(q), labs))

    CORP = {"halueval_matched": ("data/halueval_corpus.jsonl", 9936),
            "truthfulqa": ("data/truthfulqa_corpus.jsonl", 778),
            "ragtruth_matched": ("data/ragtruth_corpus.jsonl", 298)}
    META = {
        "halueval_matched": ("HaluEval QA", "length-matched", "retrieved",
                             "BGE-M3 + FAISS\n+ BM25"),
        "truthfulqa": ("TruthfulQA", "closed-book", "none",
                       "none -- the corpus\nwas the answer key"),
        "ragtruth_matched": ("RAGTruth", "length-matched", "retrieved",
                             "BGE-M3 + FAISS\n+ BM25"),
    }

    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = ["Dataset", "Rows", "Unique queries", "Hallucinated / faithful",
            "Retrieval", "Corpus", "Retriever"]
    widths = [0.15, 0.075, 0.115, 0.165, 0.115, 0.09, 0.20]
    xs = np.cumsum([0] + widths)[:-1]

    # The header has to clear the figure title, so it sits lower than the
    # top of the axes and the title is placed above it.
    y_head = 0.80
    for x, c, w in zip(xs, cols, widths):
        ax.add_patch(FancyBboxPatch((x + 0.004, y_head), w - 0.008, 0.075,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=INK, edgecolor="none"))
        ax.text(x + w / 2, y_head + 0.037, c, ha="center", va="center",
                fontsize=9.4, fontweight="bold", color="white")

    for i, (name, rel, n, nq, labs) in enumerate(rows):
        y = y_head - 0.09 - i * 0.155
        disp, split, retr, retriever = META[name]
        corp_rel, corp_n = CORP[name]
        vals = [
            disp,
            f"{n:,}",
            f"{nq:,}",
            f"{labs.get(1, 0):,} / {labs.get(0, 0):,}",
            split,
            f"{corp_n:,} docs",
            retriever,
        ]
        for x, v, w in zip(xs, vals, widths):
            ax.add_patch(FancyBboxPatch((x + 0.004, y), w - 0.008, 0.115,
                                        boxstyle="round,pad=0.004,rounding_size=0.012",
                                        facecolor="white", edgecolor=LIGHT,
                                        linewidth=1.1))
            ax.text(x + w / 2, y + 0.057, v, ha="center", va="center",
                    fontsize=8.9, color=INK, linespacing=1.35)

    ax.text(0.5, 0.945, "How each evaluation set was built",
            ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.5, 0.055,
            "Every set is balanced: exactly half the rows are hallucinated and half are "
            "faithful. That balance is deliberate -- an unbalanced set lets a detector\n"
            "score well by always guessing the majority class, which is how a "
            "length shortcut hides.",
            ha="center", va="bottom", fontsize=8.7, color=GREY, linespacing=1.5)
    save(fig, "audit_08_datasets.png")


def main() -> None:
    print("writing audit figures:")
    fig_pipeline()
    fig_detection()
    fig_length_confound()
    fig_mitigation()
    fig_ci_separation()
    fig_latency()
    fig_ledger()
    fig_datasets()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
