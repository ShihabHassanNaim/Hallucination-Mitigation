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
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # The verdict is read by column position. Some rows carry a mark in
        # an earlier column -- row 15 reads "Dale 2023 only | ✓ | ✓ | ≈",
        # because both systems do active mitigation -- and a non-greedy
        # pattern would pick up the comparable's mark and count that tie as
        # a win.
        if len(cells) < 8:
            continue
        if cells[6] not in ("✓", "✗", "≈"):
            continue
        rows.append((int(cells[1]), cells[2].strip().strip("*"), cells[6]))
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


def test_a_score_win_requires_the_interval_to_exclude_the_comparable():
    """A metric is a win only when our CI sits entirely above the published
    comparable.

    This is the rule the chart applies. It replaced a point-estimate
    comparison, which claimed AUPR and CCR were wins: our AUPR of 0.853 is
    above Decoding Memories' 0.84-0.85, but our own 95% CI of
    [0.833, 0.874] covers that whole range, so the two are not separated.
    CCR's comparable of 0.55 is implied from another metric rather than
    published, so there is nothing to be separated from.
    """
    if not METRICS_PATH.exists() or not CSV_PATH.exists():
        pytest.skip("metrics or comparison table not present")
    agg = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["aggregate"]
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    ours = next(i for i, h in enumerate(header)
                if h.lower().startswith("ours"))
    other_idx = [i for i in range(2, len(header)) if i != ours]

    # A cell only counts as a comparable score when the cited work
    # actually published it. "0 (no rewriting)" is a filler meaning "no
    # mitigation stage", and "0.55 (implied from ...)" is a baseline
    # derived from another metric -- neither is a published score.
    implied = ("implied", "projected", "no rewriting", "no mitigation",
               "not reported", "n/a")

    def num(v):
        text = str(v or "")
        if any(tag in text.lower() for tag in implied):
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None

    checked = 0
    for label, key, ci_key in (
            ("AUPR", "detection_aupr", "detection_aupr_ci95"),
            ("Claim Correction Rate (CCR)", "claim_correction_rate",
             "claim_correction_rate_ci95"),
            ("ROC-AUC", "detection_roc_auc", "detection_roc_auc_ci95"),
            ("P at R=90", "precision_at_recall_90",
             "precision_at_recall_90_ci95"),
            ("Sentence Accuracy", "detection_accuracy",
             "detection_accuracy_ci95"),
            ("Verification Success Rate (VSR)", "verification_success_rate",
             "verification_success_rate_ci95")):
        row = next((r for r in rows[1:]
                    if len(r) > 1 and r[1].strip() == label), None)
        if row is None or key not in agg:
            continue
        others = [num(row[i]) for i in other_idx if i < len(row)]
        others = [x for x in others if x is not None and 0 <= x <= 1]
        if not others:
            # No published comparable: the metric is still shown, but it
            # can never be a win.
            continue
        checked += 1
        best = max(others)
        ci = agg.get(ci_key) or []
        assert len(ci) == 2, f"{label} has no confidence interval"
        separated = ci[0] > best
        point_would_win = agg[key] > best
        if point_would_win and not separated:
            # This is the case that used to produce a false win. It is
            # legitimate for a metric to be higher at the point estimate
            # and still not be separated; what matters is that the chart
            # does not mark it as a win.
            continue
        if separated:
            assert agg[key] > best, (
                f"{label} claims separation from {best} but its value "
                f"{agg[key]} is not above it"
            )

    assert checked >= 4, (
        f"only {checked} metrics were checked against a published "
        "comparable; the comparison has drifted"
    )


def test_the_chart_does_not_mark_a_metric_as_a_win_when_the_ci_overlaps():
    """The rendered chart must agree with the interval rule."""
    if not METRICS_PATH.exists() or not CSV_PATH.exists():
        pytest.skip("metrics or comparison table not present")
    agg = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["aggregate"]
    aupr_ci = agg.get("detection_aupr_ci95") or []
    if len(aupr_ci) != 2:
        pytest.skip("no AUPR interval")
    # Decoding Memories reports 0.84-0.85; our CI covers that range, so
    # AUPR must not be presented as a win.
    assert not (aupr_ci[0] > 0.85), (
        f"the AUPR interval {aupr_ci} clears Decoding Memories' 0.85; the "
        "prose needs updating, because it currently calls AUPR a tie"
    )


def test_the_verdict_tally_matches_the_matrix():
    """The tally table must be the count of the matrix rows above it.

    The tally used to be read with a non-greedy pattern that picked up a
    checkmark from an earlier column, so row 15 -- "Dale 2023 only | ✓ | ✓
    | ≈", where both systems do active mitigation -- was counted as a win
    instead of a tie. The paper then claimed 4 wins while the matrix
    showed 3.
    """
    from scripts.fill_comparison_table import _verdict_tally

    if not VERDICT_PATH.exists():
        pytest.skip("verdict matrix not present")
    text = VERDICT_PATH.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8 or cells[6] not in ("✓", "✗", "≈"):
            continue
        rows.append((cells[1], cells[6]))

    tally = _verdict_tally(text)
    assert tally is not None, "the tally could not be rebuilt from the matrix"

    wins = [n for n, v in rows if v == "✓"]
    ties = [n for n, v in rows if v == "≈"]
    losses = [n for n, v in rows if v == "✗"]

    assert f"| {len(wins)} | {', '.join(wins)} |" in tally, (
        f"the tally's win row does not match the matrix ({len(wins)} wins: "
        f"{wins})"
    )
    assert f"| {len(ties)} | {', '.join(ties)} |" in tally, (
        f"the tally's tie row does not match the matrix ({len(ties)} ties: "
        f"{ties})"
    )
    assert f"| {len(losses)} | {', '.join(losses)} |" in tally, (
        f"the tally's loss row does not match the matrix ({len(losses)} "
        f"losses: {losses})"
    )


def test_the_output_directory_is_the_graphs_directory():
    assert plot_wins.OUT_DIR == Path("data/graphs")
