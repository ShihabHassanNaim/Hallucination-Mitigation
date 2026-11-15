"""The win-focused charts must agree with the measurements.

``scripts/plot_wins.py`` reads the verdict matrix out of the generated
prose and the capability counts out of the comparison table. Both charts
make a claim about where the pipeline is ahead, so both are checked
against the underlying files here.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from scripts import plot_wins


VERDICT_PATH = Path("Paper/why_we_win.md")
CSV_PATH = Path("Paper/comparison_table.csv")
METRICS_PATH = Path("data/metrics_halueval.json")


@pytest.fixture(scope="module")
def verdict_rows():
    if not VERDICT_PATH.exists():
        pytest.skip("verdict matrix not present")
    text = VERDICT_PATH.read_text(encoding="utf-8")
    row_re = re.compile(
        r"^\|\s*(\d+)\s*\|\s*\*\*([^*]+)\*\*\s*\|.*?\|\s*([✓✗≈])\s*\|",
        flags=re.MULTILINE,
    )
    rows = [(int(n), name.strip(), v) for n, name, v in row_re.findall(text)]
    if not rows:
        pytest.skip("verdict matrix has no rows")
    return rows


def test_the_verdict_matrix_parses(verdict_rows):
    assert len(verdict_rows) == 18, (
        f"the matrix has {len(verdict_rows)} rows, not 18; the chart title "
        "and the paper both say 18"
    )


def test_the_matrix_covers_every_dimension(verdict_rows):
    numbers = [n for n, _, _ in verdict_rows]
    assert numbers == list(range(1, len(verdict_rows) + 1)), (
        "the matrix rows must be numbered consecutively from 1"
    )


def test_every_verdict_is_a_known_symbol(verdict_rows):
    for _, name, verdict in verdict_rows:
        assert verdict in {"✓", "✗", "≈"}, (
            f"{name} has an unreadable verdict {verdict!r}"
        )


def test_the_structural_advantages_do_not_contradict_the_matrix(verdict_rows):
    """The capability chart must not claim a cited system supports a
    capability the verdict matrix records as unique to us.

    Per-claim verification is one of the four structural wins. If a cited
    system supported it, the matrix would not call it a win — so the
    counting rule must return zero for that row.
    """
    if not CSV_PATH.exists():
        pytest.skip("comparison table not present")
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    ours = next(i for i, h in enumerate(header)
                if h.lower().startswith("ours"))
    other_idx = [i for i in range(2, len(header)) if i != ours]

    verification = next(r for r in rows[1:]
                        if len(r) > 1 and r[1].strip() == "Verification")
    supported = sum(
        1 for i in other_idx
        if i < len(verification)
        and ("per claim" in verification[i].lower()
             or "per-claim" in verification[i].lower())
    )
    assert supported == 0, (
        f"{supported} cited systems are counted as supporting per-claim "
        "verification, but the verdict matrix records per-claim provenance "
        "as unique to us"
    )


def test_the_score_wins_are_the_two_genuine_wins():
    """AUPR and CCR are the only metrics above their best comparable."""
    if not METRICS_PATH.exists() or not CSV_PATH.exists():
        pytest.skip("metrics or comparison table not present")
    agg = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["aggregate"]
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    ours = next(i for i, h in enumerate(header)
                if h.lower().startswith("ours"))
    other_idx = [i for i in range(2, len(header)) if i != ours]

    def num(v):
        text = str(v or "")
        if "projected" in text.lower():
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None

    wins = []
    for label, key in (("AUPR", "detection_aupr"),
                       ("Claim Correction Rate (CCR)", "claim_correction_rate"),
                       ("ROC-AUC", "detection_roc_auc"),
                       ("P at R=90", "precision_at_recall_90"),
                       ("Sentence Accuracy", "detection_accuracy"),
                       ("Verification Success Rate (VSR)",
                        "verification_success_rate")):
        row = next((r for r in rows[1:]
                    if len(r) > 1 and r[1].strip() == label), None)
        if row is None or key not in agg:
            continue
        others = [num(row[i]) for i in other_idx if i < len(row)]
        others = [x for x in others if x is not None and 0 <= x <= 1]
        if others and agg[key] > max(others):
            wins.append(label)

    assert wins == ["AUPR", "Claim Correction Rate (CCR)"], (
        f"the score wins changed to {wins}"
    )


def test_the_output_directory_is_the_graphs_directory():
    assert plot_wins.OUT_DIR == Path("data/graphs")
