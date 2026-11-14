"""Generate the 8 result graphs for the thesis.

Outputs to data/graphs/:
  1. detection_metrics.png        — Acc/P/R/F1/ROC-AUC per dataset
  2. custom_metrics.png           — HRR / CCR / VSR per dataset (unique metrics)
  3. comparison_literature.png    — Ours vs best published system, per metric
  4. decision_distribution.png    — PASS / REGEN / ABSTAIN breakdown
  5. roc_curve.png                — ROC curve per dataset
  6. ablation_study.png           — contribution of each pipeline stage
  7. latency_breakdown.png        — per-step wall-clock cost (waterfall)
  8. confidence_calibration.png   — reliability curve of the confidence score

Usage:
  python -m scripts.plot_results
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OUT_DIR = Path("data/graphs")

DATASETS = [
    ("halueval", "HaluEval QA (length-matched)", "data/metrics_halueval.json", "#2563eb"),
    ("truthfulqa", "TruthfulQA", "data/metrics_truthfulqa.json", "#059669"),
    ("halueval_raw", "HaluEval QA (raw)", "data/metrics_halueval_raw.json", "#f59e0b"),
    ("synth", "Synthetic", "data/metrics.json", "#a855f7"),
]

PRIMARY = ("halueval", "truthfulqa")


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _rows_of(d: dict) -> list[dict]:
    return d.get("per_query") or d.get("rows") or []


def _color(slug: str) -> str:
    return next(c for s, _, _, c in DATASETS if s == slug)


def _label(slug: str) -> str:
    return next(l for s, l, _, _ in DATASETS if s == slug)


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
# 1. Detection metrics
# ---------------------------------------------------------------------------
def plot_detection_metrics() -> None:
    keys = [
        ("detection_accuracy", "Accuracy"),
        ("detection_precision", "Precision"),
        ("detection_recall", "Recall"),
        ("detection_f1", "F1"),
        ("detection_roc_auc", "ROC-AUC"),
    ]
    data = {}
    for slug, lbl, path, _ in DATASETS:
        agg = _load(path).get("aggregate", {})
        if any(k in agg for k, _ in keys):
            data[slug] = agg

    if not data:
        print("  [skip] 1 detection_metrics")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(keys))
    width = 0.8 / len(data)

    for i, (slug, agg) in enumerate(data.items()):
        vals = [agg.get(k, 0.0) for k, _ in keys]
        bars = ax.bar(xs + i * width - 0.4 + width / 2, vals, width,
                      label=_label(slug), color=_color(slug),
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.4,
               label="Random baseline (0.50)")
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in keys])
    ax.set_ylim(0, 1.15)
    _style(ax, "Detection Performance — 13-Step Pipeline", ylabel="Score")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "detection_metrics.png", dpi=160)
    plt.close(fig)
    print("  [ok] 1 detection_metrics.png")


# ---------------------------------------------------------------------------
# 2. Custom metrics (HRR / CCR / VSR)
# ---------------------------------------------------------------------------
def plot_custom_metrics() -> None:
    keys = [
        ("hallucination_reduction_rate", "HRR\n(Hallucination Reduction)"),
        ("claim_correction_rate", "CCR\n(Claim Correction)"),
        ("verification_success_rate", "VSR\n(Verification Success)"),
    ]
    data = {}
    for slug, lbl, path, _ in DATASETS:
        agg = _load(path).get("aggregate", {})
        if any(k in agg for k, _ in keys):
            data[slug] = agg

    if not data:
        print("  [skip] 2 custom_metrics")
        return

    fig, ax = plt.subplots(figsize=(10, 5.8))
    xs = np.arange(len(keys))
    width = 0.8 / len(data)

    for i, (slug, agg) in enumerate(data.items()):
        vals = [agg.get(k, 0.0) for k, _ in keys]
        bars = ax.bar(xs + i * width - 0.4 + width / 2, vals, width,
                      label=_label(slug), color=_color(slug),
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in keys], fontsize=9.5)
    ax.set_ylim(0, 1.15)
    _style(ax, "Pipeline-Specific Metrics\n(unique to this 13-step system)",
           ylabel="Score")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "custom_metrics.png", dpi=160)
    plt.close(fig)
    print("  [ok] 2 custom_metrics.png")


# ---------------------------------------------------------------------------
# 3. Literature comparison
# ---------------------------------------------------------------------------
def plot_literature_comparison() -> None:
    csv_path = Path("Paper/comparison_table.csv")
    if not csv_path.exists():
        print("  [skip] 3 comparison_literature")
        return

    import csv as _csv

    def num(v):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(v or ""))
        return float(m.group(0)) if m else None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(_csv.reader(f))
    if len(rows) < 2:
        print("  [skip] 3 comparison_literature")
        return

    header = [h.strip() for h in rows[0]]
    ours_idx = next((i for i, h in enumerate(header) if h.lower().startswith("ours")), -1)
    if ours_idx < 0:
        print("  [skip] 3 comparison_literature")
        return

    other_idx = [i for i in range(2, len(header)) if i != ours_idx]
    labels, ours_vals, best_vals = [], [], []
    for r in rows[1:]:
        if len(r) < 3:
            continue
        metric = (r[1] if len(r) > 1 else "").strip()
        if not metric:
            continue
        o = num(r[ours_idx]) if ours_idx < len(r) else None
        if o is None or o > 1.5:
            continue
        others = [num(r[i]) for i in other_idx if i < len(r)]
        others = [x for x in others if x is not None and x <= 1.5]
        if not others:
            continue
        labels.append(metric)
        ours_vals.append(o)
        best_vals.append(max(others))

    if not labels:
        print("  [skip] 3 comparison_literature")
        return

    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(labels))))
    xs = np.arange(len(labels))
    ax.barh(xs - 0.2, ours_vals, 0.4, label="Ours (13-step)", color="#2563eb",
            edgecolor="white")
    ax.barh(xs + 0.2, best_vals, 0.4, label="Best published", color="#94a3b8",
            edgecolor="white")

    for x, o, b in zip(xs, ours_vals, best_vals):
        ax.text(o + 0.012, x - 0.2, f"{o:.2f}", va="center", fontsize=8,
                fontweight="bold", color="#2563eb")
        ax.text(b + 0.012, x + 0.2, f"{b:.2f}", va="center", fontsize=8,
                color="#475569")
        if o >= b:
            ax.text(max(o, b) + 0.06, x, "✓", va="center", fontsize=11,
                    fontweight="bold", color="#16a34a")

    ax.set_yticks(xs)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.invert_yaxis()
    _style(ax, "Ours vs Best Published System — per metric", xlabel="Score")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_literature.png", dpi=160)
    plt.close(fig)
    print("  [ok] 3 comparison_literature.png")


# ---------------------------------------------------------------------------
# 4. Decision distribution
# ---------------------------------------------------------------------------
def plot_decision_distribution() -> None:
    counts = {}
    for slug, lbl, path, _ in DATASETS:
        rows = _rows_of(_load(path))
        if not rows:
            continue
        c = Counter((r.get("decision") or "UNKNOWN").upper() for r in rows)
        counts[slug] = c

    if not counts:
        print("  [skip] 4 decision_distribution")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    slugs = list(counts)
    totals = [sum(counts[s].values()) for s in slugs]
    xlabels = [f"{_label(s)}\n(n={t})" for s, t in zip(slugs, totals)]

    bottom = np.zeros(len(slugs))
    palette = {"PASS": "#16a34a", "REGEN": "#f59e0b", "ABSTAIN": "#dc2626"}
    for dec in ("PASS", "REGEN", "ABSTAIN"):
        vals = np.array([counts[s].get(dec, 0) / max(1, t) * 100
                         for s, t in zip(slugs, totals)])
        bars = ax.bar(xlabels, vals, bottom=bottom, label=dec,
                      color=palette.get(dec, "#6b7280"), edgecolor="white",
                      linewidth=1.2)
        for b, v, base in zip(bars, vals, bottom):
            if v > 3:
                ax.text(b.get_x() + b.get_width() / 2, base + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9.5, fontweight="bold",
                        color="white")
        bottom += vals

    ax.set_ylim(0, 100)
    _style(ax, "Final Gate Decision Distribution (Step 11)", ylabel="% of queries")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decision_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 4 decision_distribution.png")


# ---------------------------------------------------------------------------
# 5. ROC curves
# ---------------------------------------------------------------------------
def plot_roc_curves() -> None:
    from sklearn.metrics import roc_curve, auc as _auc

    data = {}
    for slug, lbl, path, _ in DATASETS:
        rows = _rows_of(_load(path))
        if not rows:
            continue
        ys, ss = [], []
        for r in rows:
            if "label" not in r:
                continue
            ys.append(int(r["label"]))
            # label == 1 means the answer was hallucinated, so the score must
            # be the hallucination probability (higher = more likely
            # hallucinated). detector_score is that probability; confidence is
            # a trust score and would invert the curve.
            ss.append(float(r.get("detector_score", r.get("confidence", 0.5))))
        if len(set(ys)) > 1:
            data[slug] = (ys, ss)

    if not data:
        print("  [skip] 5 roc_curve")
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for slug, (ys, ss) in data.items():
        fpr, tpr, _ = roc_curve(ys, ss)
        a = _auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.4, color=_color(slug),
                label=f"{_label(slug)} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1.5,
            label="Random (AUC = 0.50)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    _style(ax, "ROC Curves — Hallucination Detection",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)
    print("  [ok] 5 roc_curve.png")


# ---------------------------------------------------------------------------
# 6. Ablation study
# ---------------------------------------------------------------------------
def plot_ablation_study() -> None:
    path = Path("data/ablation_results.json")
    if not path.exists():
        print("  [skip] 6 ablation_study (no data/ablation_results.json)")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        print("  [skip] 6 ablation_study")
        return
    if not data:
        print("  [skip] 6 ablation_study")
        return

    full = data.get("full_pipeline", {})
    ablations = {k: v for k, v in data.items() if k != "full_pipeline"}
    if not full or not ablations:
        print("  [skip] 6 ablation_study")
        return

    metric = "detection_roc_auc"
    mlabel = "Detection ROC-AUC"

    rows = [("Full pipeline", full.get(metric, 0.0))]
    for k, v in ablations.items():
        name = k.replace("without_step", "Without Step ").replace("_", " ").title()
        rows.append((name, v.get(metric, 0.0)))

    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = ["#16a34a" if r[0] == "Full pipeline" else "#f97316" for r in rows]
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=colors,
                  edgecolor="white", linewidth=1.2)
    for b, (_, v) in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    base = full.get(metric, 0.0)
    ax.axhline(base, color="#2563eb", linestyle="--", linewidth=1.3,
               label=f"Full pipeline ({base:.3f})")
    ax.set_ylim(0, max(1.0, max(r[1] for r in rows) * 1.2))
    _style(ax, "Ablation Study — contribution of each pipeline stage",
           ylabel=mlabel)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_study.png", dpi=160)
    plt.close(fig)
    print("  [ok] 6 ablation_study.png")


# ---------------------------------------------------------------------------
# 7. Latency breakdown (waterfall)
# ---------------------------------------------------------------------------
def plot_latency_breakdown() -> None:
    traces_by_ds: dict[str, list[dict]] = {}
    for slug, lbl, path, _ in DATASETS:
        rows = _rows_of(_load(path))
        if not rows:
            continue
        step_times: dict[str, list[float]] = {}
        for r in rows:
            # scripts/evaluate_full.py stores {"step_name": duration_ms}; the
            # older scripts/evaluate.py stores a list of {"name","duration_ms"}.
            # Accept both so the figure works whichever file it reads.
            timings = r.get("step_times_ms")
            if isinstance(timings, dict):
                for name, ms in timings.items():
                    step_times.setdefault(name, []).append(float(ms))
                continue
            for t in r.get("step_traces", []) or []:
                step_times.setdefault(t.get("name", "step"), []).append(
                    float(t.get("duration_ms", 0.0)))
        if step_times:
            traces_by_ds[slug] = step_times

    if not traces_by_ds:
        print("  [skip] 7 latency_breakdown (no per-step timings in metrics)")
        return

    slug = next((s for s in PRIMARY if s in traces_by_ds), list(traces_by_ds)[0])
    step_times = traces_by_ds[slug]

    STEP_NAMES = {
        "query_understanding": "1. Query Understanding",
        "grounded_generation": "2. Grounded Generation",
        "early_detection": "3. Early Detection",
        "confidence_assessment": "4. Confidence (EEDC)",
        "claim_extraction": "5. Claim Extraction",
        "hybrid_retrieval": "6. Hybrid Retrieval",
        "reranking": "7. Reranking",
        "nli_verification": "8. NLI Verification",
        "confidence_fusion": "9. Confidence Fusion",
        "regeneration": "10. Regeneration",
        "final_gate": "11. Final Gate",
        "explainable_response": "12. Explainable Response",
        "evaluation": "13. Evaluation",
    }

    ordered = sorted(step_times.items(), key=lambda kv: -np.mean(kv[1]))
    names = [STEP_NAMES.get(k, k.replace("_", " ").title()) for k, _ in ordered]
    means = [np.mean(v) for _, v in ordered]
    total = sum(means)

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.YlOrRd(np.linspace(0.25, 0.95, len(means)))
    bars = ax.bar(names, means, color=cmap, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, means):
        pct = v / total * 100 if total else 0
        ax.text(b.get_x() + b.get_width() / 2, v + max(means) * 0.012,
                f"{v:.0f} ms\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_ylim(0, max(means) * 1.28 if max(means) else 1)
    _style(ax, f"Per-Step Latency Breakdown — {_label(slug)}",
           ylabel="Mean duration (ms)")
    plt.setp(ax.get_xticklabels(), rotation=32, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_breakdown.png", dpi=160)
    plt.close(fig)
    print("  [ok] 7 latency_breakdown.png")


# ---------------------------------------------------------------------------
# 8. Confidence calibration
# ---------------------------------------------------------------------------
def plot_confidence_calibration() -> None:
    data = {}
    for slug, lbl, path, _ in DATASETS:
        rows = _rows_of(_load(path))
        if not rows:
            continue
        pts = [(float(r.get("confidence", 0.0)), int(r["label"]))
               for r in rows if "label" in r]
        if len(set(l for _, l in pts)) > 1:
            data[slug] = pts

    if not data:
        print("  [skip] 8 confidence_calibration")
        return

    NBINS = 8
    fig, ax = plt.subplots(figsize=(8, 7))

    for slug, pts in data.items():
        bins: list[list[tuple[float, int]]] = [[] for _ in range(NBINS)]
        for conf, lbl in pts:
            idx = min(NBINS - 1, int(conf * NBINS))
            bins[idx].append((conf, lbl))

        xs, ys, ns = [], [], []
        for i, b in enumerate(bins):
            if not b:
                continue
            xs.append(np.mean([c for c, _ in b]))
            ys.append(np.mean([l for _, l in b]))
            ns.append(len(b))
        if xs:
            ax.plot(xs, ys, "o-", linewidth=2.2, markersize=7, color=_color(slug),
                    label=f"{_label(slug)} (n={sum(ns)})")

    ax.plot([0, 1], [0, 1], "--", color="#6b7280", linewidth=1.6,
            label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], step="mid", alpha=0.04, color="#6b7280")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Confidence Calibration (reliability diagram)",
           xlabel="Predicted confidence (mean confidence in bin)",
           ylabel="Observed accuracy (fraction hallucinated)")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confidence_calibration.png", dpi=160)
    plt.close(fig)
    print("  [ok] 8 confidence_calibration.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing 8 graphs to {OUT_DIR}/")
    plot_detection_metrics()
    plot_custom_metrics()
    plot_literature_comparison()
    plot_decision_distribution()
    plot_roc_curves()
    plot_ablation_study()
    plot_latency_breakdown()
    plot_confidence_calibration()
    print("done.")


if __name__ == "__main__":
    main()
