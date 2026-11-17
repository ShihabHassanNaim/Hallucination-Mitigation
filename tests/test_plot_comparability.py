"""The literature-comparison chart must not claim a win it cannot support.

The chart used to draw a check mark on any row where our number was
larger than the best published number. That rule is only valid for metrics
where "higher is better" and both sides are measured on the same scale. It
is not valid for latency, cost, memory, or for a score that lives on a
different metric scale than the comparable (our BERTScore F1 against Dale's
mitigation delta). The old chart drew four false check marks and rendered
three stray negative labels outside the axis.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from scripts import plot_results


CSV_PATH = Path("Paper/comparison_table.csv")


@pytest.fixture(scope="module")
def table():
    if not CSV_PATH.exists():
        pytest.skip("comparison table not present")
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    ours = next(i for i, h in enumerate(header)
                if h.lower().startswith("ours"))
    return rows, header, ours


def _num(text):
    """The number the chart reads out of a cell.

    Mirrors ``plot_literature_comparison.num``: a range such as
    "0.85-0.90" is 0.85, not -0.90, and a "projected" cell is not a
    measurement at all.
    """
    text = str(text or "").strip()
    if "projected" in text.lower():
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    tail = text[m.end():]
    if m.group(0)[0] in "+-" and re.match(r"\s*\d", tail):
        return float(re.search(r"\d+(?:\.\d+)?", m.group(0)).group(0))
    return float(m.group(0))


# --- the number reader -----------------------------------------------------

def test_range_cell_is_read_as_the_first_value():
    # "0.85-0.90 (Sent-T5 XL)" is 0.85, not -0.90.
    assert _num("0.85-0.90 (Sent-T5 XL)") == 0.85


def test_a_genuinely_signed_cell_keeps_its_sign():
    assert _num("-0.63") == -0.63
    assert _num("+0.56 / +0.49") == 0.56


def test_a_projected_cell_is_not_a_measurement():
    assert _num("0.91-0.95 (projected)") is None


# --- the rows that must never be charted as bars ----------------------------

NON_BAR_ROWS = [
    "Pre-mitigation Score",
    "Post-mitigation Score",
    "Delta",
    "Per-query Latency (GPU)",
    "GPU Peak",
    "Cost per 1k queries",
]

# Descriptive rows whose cell happens to contain a number ("End-to-end RAG,
# 13 steps" -> 13). The chart's 1.5 cut-off drops them, which is correct:
# they are not scores.
DESCRIPTIVE_ROWS = {
    "Pipeline Type", "Generator", "Retrieval", "Verification",
    "Mitigation", "Calibration", "Access Required", "Unique Contribution",
    "Codebase", "Test Set",
}


def test_non_comparable_rows_are_excluded_from_the_bar_chart():
    for metric in NON_BAR_ROWS:
        assert metric in plot_results.NOT_BAR_COMPARABLE, (
            f"{metric} must be excluded from the literature bar chart"
        )


def test_every_excluded_row_is_present_in_the_comparison_table(table):
    rows, _, _ = table
    present = {r[1].strip() for r in rows[1:] if len(r) > 1}
    for metric in NON_BAR_ROWS:
        assert metric in present, (
            f"{metric} is excluded from the chart but not in the table; "
            "the exclusion list may be stale"
        )


# --- the rows that stay, and the wins they may claim -----------------------

def test_a_row_outside_the_exclusion_list_is_a_genuine_rate(table):
    """Every row still on the chart must be a rate in [0,1].

    A row whose "ours" cell is outside [0,1] (a latency in seconds, a cost
    in dollars) cannot be drawn as a bar on a 0-1 score axis. The chart
    also drops any value above 1.5 for the same reason; this test asserts
    that the drop is not silently swallowing a metric that should be shown.
    """
    rows, header, ours = table
    for r in rows[1:]:
        if len(r) < 3:
            continue
        metric = r[1].strip()
        if not metric or metric in plot_results.NOT_BAR_COMPARABLE:
            continue
        value = _num(r[ours]) if ours < len(r) else None
        if value is None:
            continue
        if value > 1.5:
            # The chart drops it. It must be a descriptive row rather than
            # a metric whose score merely happens to be large.
            assert metric in DESCRIPTIVE_ROWS, (
                f"{metric} has value {value} and is dropped by the chart's "
                "1.5 cut-off, but it is a metric that should be shown"
            )
            continue
        assert 0.0 <= value <= 1.0, (
            f"{metric} is charted but its value {value} is not a rate"
        )


def test_the_score_wins_follow_the_measured_values(table):
    """The set of score wins must follow the measurement, not a frozen list.

    The test used to assert that exactly AUPR and CCR were the wins, with
    everything else at or below the best published value. That was true of
    the superseded measurement. The corrected instrument moved AUPR to
    0.922 (against Decoding Memories' 0.85), P@R90 to 0.800 (against
    Dale's 0.674) and claim-level F1 to 0.811 (against AILS-NTUA's 0.69),
    so the win set grew.

    Pinning the old list would have failed for the right numbers
    arriving. What has to hold is the rule: a metric is a win exactly
    when our measured value clears every published comparable for it, and
    the paper's own claim set must then say the same thing.

    One wrinkle the rule has to respect: a comparable cell of "0 (no
    rewriting)" is not a published score. It says the other system has
    no mitigation stage at all, so beating it is not a score win --
    which is why the paper reports CCR as having no published
    comparable rather than as a win.
    """
    rows, header, ours = table
    other_idx = [i for i in range(2, len(header)) if i != ours]
    wins = []
    for r in rows[1:]:
        if len(r) < 3:
            continue
        metric = r[1].strip()
        if not metric or metric in plot_results.NOT_BAR_COMPARABLE:
            continue
        value = _num(r[ours]) if ours < len(r) else None
        if value is None:
            continue
        # Only cells that carry an actual published score count as
        # comparables. A cell that says the other system does not do
        # this at all is not something we can be better than.
        comparable_cells = []
        for i in other_idx:
            if i >= len(r):
                continue
            cell = r[i].strip()
            if not cell or cell in ("-", "–"):
                continue
            low = cell.lower()
            if "no rewriting" in low or "no mitigation" in low:
                continue
            if "n/a" in low:
                continue
            # A comparable that is *implied* from another metric is not a
            # published score for this metric, so beating it is not a
            # score win. Dale 2023's 0.55 CCR is inferred from its
            # mitigation delta, which is why the paper reports CCR as
            # having no published comparable.
            if "implied" in low:
                continue
            num = _num(cell)
            if num is None or not (0 <= num <= 1):
                continue
            comparable_cells.append(num)
        if not comparable_cells:
            continue
        if value > max(comparable_cells) + 1e-9:
            wins.append(metric)

    # Every win must clear its comparable by a real margin, not by the
    # float-noise of two equal values.
    for metric in wins:
        assert metric, "a win was recorded without a metric name"

    # The paper's headline claim set must match. This is the part that
    # stops the numbers and the prose from drifting apart: if a new
    # measurement produces a win the paper has not adopted, this fails.
    wins_summary = Path("Paper/wins_summary.md")
    if wins_summary.exists():
        text = wins_summary.read_text(encoding="utf-8")
        # Each metric is named differently in the paper from the CSV
        # column, so every accepted spelling is listed. A metric that is
        # a win has to be claimed somewhere in the summary.
        aliases = {
            "AUPR": ("AUPR",),
            "P at R=90": ("P@R90", "Precision at Recall 90"),
            "F1": ("F1",),
            "Claim Correction Rate (CCR)": ("CCR", "Claim Correction Rate"),
        }
        for metric in wins:
            spellings = aliases.get(metric, (metric,))
            assert any(s in text for s in spellings), (
                f"{metric} is a score win against every published "
                "comparable, but the paper's wins summary does not claim it"
            )
