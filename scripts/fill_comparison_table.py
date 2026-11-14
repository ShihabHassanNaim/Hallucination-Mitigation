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

    if args.dry_run:
        print(f"\n[dry-run] {len(changed)} cell(s) would change in {args.csv}")
        return 0

    with args.csv.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nUpdated {len(changed)} cell(s) in {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
