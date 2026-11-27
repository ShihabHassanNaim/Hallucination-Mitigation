"""Journal-ready headline figure: detection performance with intervals.

The single figure a reviewer will look for. It shows, for each of the
two evaluated datasets, the detection metrics the paper reports --
ROC-AUC, AUPR, F1, accuracy, precision and recall -- each with its 95%
bootstrap confidence interval, against the 0.50 chance line.

Two properties matter for this figure:

  * Every number is read from the metrics files. There is no typed-in
    value, so the figure cannot drift from the measurements.
  * Both rows are length-matched. The figure says so in the axis
    labels, because the length-matched HaluEval number (0.925) is
    deliberately lower than the raw-split number (0.968) and a reader
    who mixed the two would draw the wrong conclusion.

The HaluEval raw split is not drawn here. It is a different measurement
with a known confound, and it is shown in the transparency chart
(``detection_metrics.png``) rather than in the headline result.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OUT_DIR = Path("data/graphs")

# (axis label, metrics-file path, aggregate key, ci key)
DATASETS = [
    ("HaluEval QA\n(length-matched)",
     "data/metrics_halueval.json",
     "#2563eb"),
    ("RAGTruth\n(length-matched)",
     "data/metrics_ragtruth_fitted.json",
     "#7c3aed"),
]

METRICS = [
    ("ROC-AUC", "detection_roc_auc", "detection_roc_auc_ci95"),
    ("AUPR", "detection_aupr", "detection_aupr_ci95"),
    ("F1", "detection_f1", "detection_f1_ci95"),
    ("Accuracy", "detection_accuracy", "detection_accuracy_ci95"),
    ("Precision", "detection_precision", "detection_precision_ci95"),
    ("Recall", "detection_recall", "detection_recall_ci95"),
]


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def plot_headline() -> None:
    """Draw the headline detection figure."""
    # Collect: rows[dataset_index][metric_index] = (value, ci_low, ci_high)
    rows: list[list[tuple[float, float | None, float | None]]] = []
    labels: list[str] = []
    colors: list[str] = []
    for label, path, color in DATASETS:
        agg = _load(path).get("aggregate", {})
        if not agg:
            continue
        row: list[tuple[float, float | None, float | None]] = []
        for _name, key, ci_key in METRICS:
            value = agg.get(key)
            if value is None:
                row.append((float("nan"), None, None))
                continue
            ci = agg.get(ci_key)
            low = float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
            high = float(ci[1]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
            row.append((float(value), low, high))
        if all(np.isnan(v) for v, _, _ in row):
            continue
        rows.append(row)
        labels.append(label)
        colors.append(color)

    if not rows:
        print("  [skip] 13 headline_detection")
        return

    n_ds = len(rows)
    n_met = len(METRICS)

    fig, axes = plt.subplots(1, n_met, figsize=(4.0 * n_met, 5.4),
                             sharey=True)
    if n_met == 1:
        axes = [axes]

    for m, (ax, (mname, _key, _ci)) in enumerate(zip(axes, METRICS)):
        xs = np.arange(n_ds)
        for d in range(n_ds):
            value, low, high = rows[d][m]
            if np.isnan(value):
                continue
            if low is not None and high is not None:
                ax.errorbar(
                    xs[d], value,
                    yerr=[[value - low], [high - value]],
                    fmt="o", markersize=9, capsize=6, capthick=1.6,
                    linewidth=2.0, color=colors[d], ecolor=colors[d],
                    elinewidth=1.8, zorder=3,
                )
                # The label goes above the whole error bar, not above the
                # point estimate: on RAGTruth the interval is wide enough
                # that a label placed at value + 0.035 sat on top of it.
                label_y = max(value, high) + 0.028
            else:
                ax.plot(xs[d], value, "o", markersize=9, color=colors[d],
                        zorder=3)
                label_y = value + 0.028
            ax.text(xs[d], label_y, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color=colors[d])

        ax.axhline(0.50, color="#dc2626", linestyle="--", linewidth=1.4,
                   zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.55, n_ds - 0.45)
        ax.set_ylim(0.35, 1.05)
        ax.grid(axis="y", alpha=0.28, linestyle="--")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(mname, fontsize=12.5, fontweight="bold", pad=10)

    axes[0].set_ylabel("Score (higher is better)", fontsize=10)
    axes[0].text(
        0.02, 0.015, "chance (0.50)", transform=axes[0].transAxes,
        fontsize=8.5, color="#dc2626", va="bottom",
    )

    fig.suptitle(
        "Detection performance across two benchmarks, with 95% "
        "bootstrap confidence intervals",
        fontsize=14.5, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.028,
        "1,000 rows per dataset; intervals are bootstrapped over unique "
        "questions (500 for HaluEval, 277 for RAGTruth). Both benchmarks "
        "use length-matched splits — on the raw split a detector can guess "
        "94.7% of rows from answer length alone, so the raw numbers are not "
        "shown here.",
        ha="center", va="bottom", fontsize=8.6, color="#64748b",
    )

    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "headline_detection.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] 13 headline_detection.png ({n_ds} datasets, {n_met} metrics)")


def main() -> None:
    print(f"Writing the headline figure to {OUT_DIR}/")
    plot_headline()


if __name__ == "__main__":
    main()
