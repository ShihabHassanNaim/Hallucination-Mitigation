"""Fill the "Ours" column of ``Paper/comparison_table.csv`` with measured values.

The comparison table originally carried hand-written projections in the
"Ours" cells (e.g. ``"0.91-0.95 (projected)"``). Those were estimates made
before the evaluation could actually measure detection, and the plotting
script rendered them as if they were measurements.

This script replaces every projected cell with the corresponding number
computed from the evaluation output, so the table is always regenerated
from the data and can never drift from it.

Which measurement fills which cell:

  * ROC-AUC, AUPR, P at R=90, Sentence Accuracy, Spearman rho
      -> ``data/metrics_halueval.json`` (length-matched HaluEval)
  * Claim F1, Claim P/R
      -> the claim-level metrics of the same file. These are computed on
      the faithful rows only: on a hallucinated row the benchmark does not
      say which of the answer's claims are wrong, so claim recall is
      undefined there and is excluded from the average.

Numbers are written with their bootstrap confidence interval, e.g.
``0.789 [0.732, 0.841]``.

Usage:
  python -m scripts.fill_comparison_table
  python -m scripts.fill_comparison_table --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

CSV_PATH = Path("Paper/comparison_table.csv")
METRICS_PATH = Path("data/metrics_halueval.json")

# Metric row label in the CSV -> (aggregate key, CI key)
DETECTION_ROWS = {
    "ROC-AUC": ("detection_roc_auc", "detection_roc_auc_ci95"),
    "AUPR": ("detection_aupr", "detection_aupr_ci95"),
    "P at R=90": ("precision_at_recall_90", "precision_at_recall_90_ci95"),
    "Sentence Accuracy": ("detection_accuracy", "detection_accuracy_ci95"),
    "Spearman rho": ("spearman_rho", "spearman_rho_ci95"),
}

# Metric row label in the CSV -> aggregate key for the claim-level metrics.
CLAIM_ROWS = {
    "F1": "claim_f1",
    "P/R": ("claim_precision", "claim_recall"),
}

_PROJECTED = re.compile(r"projected", re.IGNORECASE)


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _load_aggregate(path: Path) -> dict:
    """Return the aggregate block, computing it from the rows if absent.

    A metrics file written by an in-progress run carries only ``per_query``
    (the aggregate is written at the end), so the aggregate is rebuilt from
    the rows rather than being required to be present.
    """
    payload = _load_metrics(path)
    if not payload:
        return {}
    agg = payload.get("aggregate")
    if isinstance(agg, dict) and agg:
        return agg
    rows = payload.get("per_query") or payload.get("rows") or []
    if not rows:
        return {}
    from scripts.evaluate_full import _aggregate_from_rows

    slug = payload.get("dataset") or "unknown"
    return _aggregate_from_rows(slug, rows)


def _fmt(value: float, ci: list[float] | None) -> str:
    """Format a value with its CI when one is available."""
    if value is None:
        return "n/a"
    if ci:
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{value:.3f}"


def _find_ours_column(header: list[str]) -> int:
    for i, h in enumerate(header):
        if h.strip().lower().startswith("ours"):
            return i
    raise SystemExit(
        "comparison_table.csv has no 'Ours' column; refusing to guess."
    )


def build_replacements(agg: dict) -> dict[str, str]:
    """Return {csv metric label: new cell text} for the measurable rows."""
    out: dict[str, str] = {}

    for label, (key, ci_key) in DETECTION_ROWS.items():
        if key not in agg:
            continue
        out[label] = _fmt(agg[key], agg.get(ci_key))

    for label, key in CLAIM_ROWS.items():
        if isinstance(key, tuple):
            p, r = key
            if p in agg and r in agg:
                out[label] = f"{agg[p]:.3f} / {agg[r]:.3f}"
        elif key in agg:
            out[label] = _fmt(agg[key], None)

    return out


def _safe(text: str) -> str:
    """Render a string for the console, replacing non-cp1252 characters.

    The Windows console uses cp1252, which cannot encode characters like the
    Greek rho used in the paper's tables.
    """
    return text.encode("ascii", errors="replace").decode("ascii")


def _fill_comparison_md(
    path: Path, agg: dict, dry_run: bool = False
) -> list[tuple[str, str, str]]:
    """Replace the projected "Ours" rows in ``Paper/comparison.md``.

    The prose table and the summary table below it both carried the same
    hand-written estimates as the CSV. Each replacement is anchored on the
    row's own text, so an unrelated number is never touched.
    """
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        """Formatted value, or None when the metric was not measured."""
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    replacements: list[tuple[str, str, str]] = []

    # Each substitution is described by the metrics it needs. A row whose
    # metrics have not been measured is left alone rather than filled in
    # with a guess.
    row_specs = [
        (
            "3.1 detection row",
            (
                "| **Ours (RAG fused score, Step 9)** | **0.91–0.95** "
                "(projected from synthetic) | **0.85–0.90** | **0.55–0.70** | "
                "Black-box + retrieval | HaluEval QA |"
            ),
            lambda v: (
                f"| **Ours (RAG fused score, Step 9)** | "
                f"**{v('detection_roc_auc')}** | **{v('detection_aupr')}** | "
                f"**{v('precision_at_recall_90')}** | "
                "Black-box + retrieval | HaluEval QA (length-matched) |"
            ),
            ("detection_roc_auc", "detection_aupr", "precision_at_recall_90"),
        ),
        (
            "3.2 claim row",
            (
                "| **Ours (Step 8 + Step 9)** | **0.78 / 0.74 / 0.76** "
                "(per-claim NLI) | Atomic claims | HaluEval |"
            ),
            lambda v: (
                f"| **Ours (Step 8 + Step 9)** | **{v('claim_precision')} / "
                f"{v('claim_recall')} / {v('claim_f1')}** (per-claim NLI) | "
                "Atomic claims | HaluEval |"
            ),
            ("claim_precision", "claim_recall", "claim_f1"),
        ),
        (
            "summary: ROC-AUC",
            "| **Detection ROC-AUC** | 0.987 (Dale 2023 ALTI+) | **0.91–0.95** |",
            lambda v: (
                f"| **Detection ROC-AUC** | 0.987 (Dale 2023 ALTI+) | "
                f"**{v('detection_roc_auc')}** |"
            ),
            ("detection_roc_auc",),
        ),
        (
            "summary: sentence accuracy",
            "| **Sentence accuracy** | 0.847 (Halu-NLP, SHROOM) | **0.80–0.86** |",
            lambda v: (
                f"| **Sentence accuracy** | 0.847 (Halu-NLP, SHROOM) | "
                f"**{v('detection_accuracy')}** |"
            ),
            ("detection_accuracy",),
        ),
        (
            "summary: Spearman rho",
            "| **Spearman ρ** | 0.770 (Halu-NLP) | **0.65–0.72** |",
            lambda v: (
                f"| **Spearman ρ** | 0.770 (Halu-NLP) | "
                f"**{v('spearman_rho')}** |"
            ),
            ("spearman_rho",),
        ),
        (
            "summary: claim-level F1",
            "| **Claim-level F1** | 0.71 (TU Wien Vectara) | **0.76** |",
            lambda v: (
                f"| **Claim-level F1** | 0.71 (TU Wien Vectara) | "
                f"**{v('claim_f1')}** |"
            ),
            ("claim_f1",),
        ),
    ]

    for label, old, build, needed in row_specs:
        if old not in text:
            continue
        values = {k: _v(k) for k in needed}
        if any(val is None for val in values.values()):
            # The metric was not measured — leave the row untouched rather
            # than substituting a number that does not exist.
            continue
        new = build(lambda k: values[k])
        text = text.replace(old, new)
        replacements.append((label, old, new))

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the proposed changes without writing the file",
    )
    args = parser.parse_args(argv)

    metrics = _load_metrics(args.metrics)
    agg = _load_aggregate(args.metrics)
    if not agg:
        print(
            f"No metrics in {args.metrics}. Run the evaluation first.",
            file=sys.stderr,
        )
        return 1

    replacements = build_replacements(agg)
    if not replacements:
        print("No measurable metrics found in the aggregate.", file=sys.stderr)
        return 1

    with args.csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print(f"{args.csv} is empty.", file=sys.stderr)
        return 1

    header = rows[0]
    ours_idx = _find_ours_column(header)

    changed: list[tuple[str, str, str]] = []
    for row in rows[1:]:
        if len(row) <= ours_idx:
            continue
        label = (row[1] if len(row) > 1 else "").strip()
        if label not in replacements:
            continue
        new_cell = replacements[label]
        old_cell = row[ours_idx]
        if old_cell.strip() != new_cell:
            changed.append((label, old_cell, new_cell))
            row[ours_idx] = new_cell

    if not changed:
        print("Nothing to update — the table already matches the metrics.")
        return 0

    for label, old, new in changed:
        print(f"  {label}: {old!r} -> {new!r}")

    # The prose version of the same comparison carries the same estimates.
    md_changes = _fill_comparison_md(Path("Paper/comparison.md"), agg, dry_run=args.dry_run)
    for label, old, new in md_changes:
        print(f"  comparison.md {label}: {_safe(old)} -> {_safe(new)}")

    if args.dry_run:
        total = len(changed) + len(md_changes)
        print(f"\n[dry-run] {total} cell(s) would change")
        return 0

    if changed:
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"\nUpdated {len(changed)} cell(s) in {args.csv}")
    if md_changes:
        print(f"Updated {len(md_changes)} row(s) in Paper/comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
