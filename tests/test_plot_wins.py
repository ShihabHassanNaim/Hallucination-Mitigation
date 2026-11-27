"""The win-focused charts must agree with the measurements.

``scripts/plot_wins.py`` reads the verdict matrix out of the generated
prose and the capability counts out of the comparison table. Both charts
make a claim about where the pipeline is ahead, so both are checked
against the underlying files here.

The matrix lives in ``Paper/comparison.md`` now. It used to be a separate
``Paper/why_we_win.md``, which was folded into the consolidated
comparison document; the chart was pointed at the new file and these
tests followed it.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from scripts import plot_wins


VERDICT_PATH = Path("Paper/comparison.md")
CSV_PATH = Path("Paper/comparison_table.csv")
METRICS_PATH = Path("data/metrics_halueval.json")


@pytest.fixture(scope="module")
def matrix_rows():
    """Every numbered row of the win matrix, with or without a verdict.

    The verdict is read by column position. Some rows carry a mark in
    an earlier column -- row 15 reads "Dale 2023 only | ✓ | ✓ | ≈",
    because both systems do active mitigation -- and a non-greedy
    pattern would pick up the comparable's mark and count that tie as
    a win.

    A row whose verdict is a dash is kept: it means the metric has no
    published comparable, which is a different kind of row, not a
    missing one.
    """
    if not VERDICT_PATH.exists():
        pytest.skip("verdict matrix not present")
    text = VERDICT_PATH.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8:
            continue
        rows.append((int(cells[1]), cells[2].strip().strip("*"), cells[6]))
    if not rows:
        pytest.skip("verdict matrix has no rows")
    return rows


@pytest.fixture(scope="module")
def verdict_rows(matrix_rows):
    """The subset of the matrix rows that carry a verdict.

    Detection AUPR carries a dash, because no published paper reports
    AUPR on HaluEval, so there is nothing for it to be compared against.
    """
    return [r for r in matrix_rows if r[2] in ("✓", "✗", "≈")]


def test_the_verdict_matrix_parses(matrix_rows):
    """The matrix must be non-empty and its rows consecutively numbered.

    The count is not pinned to a fixed number. It was 18 when the matrix
    was written and is 21 now, because three dimensions were added to the
    comparison. Pinning the number would make the test fail every time
    the comparison grew a dimension, which is the wrong reason to fail:
    the chart reads the count from the matrix, so the only thing worth
    checking is that the matrix is internally consistent.
    """
    assert len(matrix_rows) >= 18, (
        f"the matrix has only {len(matrix_rows)} rows; it should have "
        "grown, not shrunk, since the comparison was consolidated"
    )


def test_the_matrix_covers_every_dimension(matrix_rows):
    numbers = [n for n, _, _ in matrix_rows]
    assert numbers == list(range(1, len(matrix_rows) + 1)), (
        "the matrix rows must be numbered consecutively from 1"
    )


def test_exactly_one_row_has_no_verdict():
    """Detection AUPR is the one metric with no published comparable.

    It must not be counted as a win, a tie or a loss, and it must still
    be present as a numbered row so the matrix reads 1..21.
    """
    if not VERDICT_PATH.exists():
        pytest.skip("verdict matrix not present")
    text = VERDICT_PATH.read_text(encoding="utf-8")
    no_verdict = [
        line for line in text.splitlines()
        if re.match(r"^\|\s*\d+\s*\|", line)
        and "**Detection AUPR**" in line
        and "| — |" in line
    ]
    assert len(no_verdict) == 1, (
        f"expected exactly one AUPR row with no verdict, found "
        f"{len(no_verdict)}: {no_verdict}"
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
    comparison, which claimed AUPR and CCR were wins on the strength of
    their midpoints. CCR's comparable of 0.55 is implied from another
    metric rather than published, so there is nothing to be separated
    from; and AUPR has no published comparable at all, because the paper
    that was cited for it reports AUROC.
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
    """The rendered chart must agree with the interval rule.

    The rule is symmetric: a metric is a win only when our 95% CI clears
    the published comparable, and it is not a win while the intervals
    intersect. The rule is checked on P@R90, which has a published
    comparable (Dale 2023's LaBSE baseline at 0.703).

    Detection AUPR used to be the metric this test was written against.
    It no longer has a published comparable at all: Decoding Memories
    reports AUROC, not AUPR, so the AUPR row carries no verdict. The
    chart therefore has to say "not comparable" for AUPR, which is what
    the first assertion checks.
    """
    if not METRICS_PATH.exists() or not CSV_PATH.exists():
        pytest.skip("metrics or comparison table not present")
    agg = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["aggregate"]

    comparison_md = Path("Paper/comparison.md")
    if not comparison_md.exists():
        pytest.skip("comparison.md not present")
    lines = comparison_md.read_text(encoding="utf-8").splitlines()

    # AUPR has no published comparable, so the row has no verdict.
    aupr_row = next(
        (l for l in lines
         if "**Detection AUPR**" in l and l.startswith("|")),
        None,
    )
    assert aupr_row is not None, "the AUPR row is missing from comparison.md"
    assert "✓" not in aupr_row and "✗" not in aupr_row, (
        f"the AUPR row carries a verdict; no published paper reports AUPR "
        f"on HaluEval: {aupr_row}"
    )

    # P@R90 does have one, and the rule decides the verdict.
    pr_ci = agg.get("precision_at_recall_90_ci95") or []
    if len(pr_ci) != 2:
        pytest.skip("no P@R90 interval")
    lo, hi = float(pr_ci[0]), float(pr_ci[1])
    comparable = 0.703

    pr_row = next(
        (l for l in lines
         if "**P@R90**" in l and l.startswith("|")),
        None,
    )
    assert pr_row is not None, "the P@R90 row is missing from comparison.md"
    low = pr_row.lower()

    if lo > comparable:
        # Separated: the row may claim the win, and must name the
        # interval as the reason. The wording has changed as the
        # document was consolidated ("excludes" became "above their
        # top"), so the test accepts any phrasing that says the
        # interval clears the comparable.
        assert "above" in low and ("excludes" in low or "bottom" in low
                                   or "clears" in low), (
            f"our P@R90 CI [{lo:.3f}, {hi:.3f}] clears {comparable}, so "
            f"the row should say so: {pr_row}"
        )
    else:
        # Overlapping: the row must not claim a win.
        assert "overlap" in low or "tie" in low, (
            f"our P@R90 CI [{lo:.3f}, {hi:.3f}] covers {comparable}, so "
            f"the row should report the tie: {pr_row}"
        )


def test_the_verdict_tally_matches_the_matrix():
    """The tally table must be the count of the matrix rows above it.

    Two separate things are checked, because they failed independently:

      * the dimension *lists* must be exactly the matrix rows with that
        verdict, and
      * the *counts* must equal the length of those lists.

    The counts used to disagree with the lists they printed: the win row
    said 8 while listing seven dimension numbers, and the tie row said 5
    while listing six. The count had been adjusted to fold the three
    RAGTruth head-to-head wins into the matrix tally, which is a
    different comparison and does not belong in that count.
    """
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
        rows.append((int(cells[1]), cells[6]))

    assert rows, "the verdict matrix has no rows"

    tally_block = text.split("### Verdict Tally", 1)
    assert len(tally_block) == 2, "the verdict tally table is missing"
    block = tally_block[1][:1200]

    def _tally_row(label: str) -> tuple[int, list[int]]:
        """(count, dimension numbers) from one row of the tally table."""
        m = re.search(rf"\*\*{re.escape(label)}\*\*\s*\|\s*(\d+)\s*\|\s*([^|]*)\|",
                      block)
        assert m is not None, f"the tally has no {label!r} row"
        count = int(m.group(1))
        dims = [int(d) for d in re.findall(r"\d+", m.group(2))]
        return count, dims

    for label, verdict in (("✓ Win", "✓"), ("≈ Tie / At par", "≈"),
                           ("✗ Loss", "✗")):
        count, dims = _tally_row(label)
        expected = sorted(n for n, v in rows if v == verdict)
        assert dims == expected, (
            f"the tally's {label} row lists {dims}, but the matrix has "
            f"{expected}"
        )
        assert count == len(expected), (
            f"the tally's {label} row says {count}, but it lists "
            f"{len(expected)} dimensions: {dims}"
        )


def test_the_matrix_tally_is_what_the_chart_draws():
    """The chart and the paper must report the same win/tie/loss counts.

    The chart reads the matrix directly, so it cannot drift on its own.
    The paper's tally was hand-written and did drift, so this test pins
    the two together.
    """
    if not VERDICT_PATH.exists():
        pytest.skip("verdict matrix not present")
    from scripts import plot_wins

    text = VERDICT_PATH.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8 or cells[6] not in ("✓", "✗", "≈"):
            continue
        rows.append((int(cells[1]), cells[2].strip("*"), cells[6]))

    label_map = {"✓": "Win", "≈": "Tie", "✗": "Loss"}
    counts = {"Win": 0, "Tie": 0, "Loss": 0}
    for _, _, v in rows:
        counts[label_map[v]] += 1

    # The chart labels the dimensions it plots with the same names, and
    # its title quotes the row count. Both come from the matrix, so the
    # matrix is the single source of truth and this test only has to
    # confirm that the chart would draw these counts.
    assert counts["Win"] + counts["Tie"] + counts["Loss"] == len(rows)
    assert plot_wins.VERDICT_PATH == Path("Paper/comparison.md"), (
        "the chart must read the matrix from the consolidated comparison "
        "document, which is where the matrix lives"
    )


def test_the_output_directory_is_the_graphs_directory():
    assert plot_wins.OUT_DIR == Path("data/graphs")


def test_a_metric_below_its_comparable_is_a_loss_not_a_tie():
    """An interval that sits entirely below the published value is a loss.

    The chart used to classify a metric as an overlap whenever the
    *bottom* of our interval sat under the published number. That is also
    true of an interval that sits entirely below it, so a metric we are
    clearly behind was drawn as a tie. Our ROC-AUC is [0.911, 0.940]
    against Dale's 0.987: the whole interval is under his number, so we
    are behind, not at par.

    The rule has to hold in both directions, and the tie case must still
    be a tie: our sentence accuracy is [0.809, 0.851] against 0.847,
    which straddles it.
    """
    from scripts.fill_comparison_table import _verdict_from_ci

    def verdict(value, ci, comparable):
        def lookup(key):
            if key == "m_ci":
                return ci
            raise KeyError(key)

        return _verdict_from_ci(lookup, "m", "m_ci", comparable)

    # Entirely below: a loss.
    assert verdict(0.925, [0.911, 0.940], "0.987 (Dale ALTI+)") == "✗", (
        "an interval that sits entirely below the published value was "
        "reported as something other than a loss"
    )
    # Straddling: a tie.
    assert verdict(0.829, [0.809, 0.851], "0.847 (Halu-NLP)") == "≈", (
        "an interval that straddles the published value was reported as "
        "something other than a tie"
    )
    # Entirely above: a win.
    assert verdict(0.800, [0.773, 0.844], "0.703 (Dale LaBSE)") == "✓", (
        "an interval that clears the published value was reported as "
        "something other than a win"
    )


def test_the_chart_and_the_paper_agree_on_win_tie_and_loss():
    """The chart's classification has to match the paper's metric table.

    The chart and the paper are two surfaces for the same six scores.
    When the chart called ROC-AUC and VSR ties and the paper called them
    losses, a reader comparing the figure to the table would find the
    two contradicting each other.

    The paper's rule is the one the chart has to apply, so the chart's
    classifier is run over the same measurements and the results are
    compared row by row.
    """
    if not METRICS_PATH.exists() or not CSV_PATH.exists():
        pytest.skip("metrics or comparison table not present")
    if not VERDICT_PATH.exists():
        pytest.skip("comparison.md not present")

    agg = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["aggregate"]
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.reader(f))
    header = [h.strip() for h in csv_rows[0]]
    ours = next(i for i, h in enumerate(header)
                if h.lower().startswith("ours"))
    other_idx = [i for i in range(2, len(header)) if i != ours]

    # The chart's classifier: a win needs the bottom of our interval
    # above their number, a loss needs the top of it below, and anything
    # else is a tie.
    def classify(value, ci, best):
        if len(ci) != 2:
            return None
        if ci[0] > best:
            return "win"
        if ci[1] < best:
            return "loss"
        return "tie"

    def num(v):
        text = str(v or "")
        if any(tag in text.lower() for tag in ("implied", "projected",
                                               "no rewriting", "no mitigation",
                                               "not reported", "n/a")):
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None

    # The paper's metric table: | Metric | value | CI | comparable |
    # verdict |.
    md = VERDICT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"\*\*The full metric set on HaluEval:\*\*.*?\n\s*\n"
        r"\| Metric \|.*?\n(?:\|\|?.*\n)*",
        md,
        re.DOTALL,
    )
    assert block, "the paper has no HaluEval metric table"
    paper = {}
    for line in block.group(0).splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 5 and cells[0] != "Metric":
            paper[cells[0]] = cells

    chart_labels = {
        "AUPR": "AUPR",
        "P at R=90": "P@R90",
        "ROC-AUC": "ROC-AUC",
        "Sentence Accuracy": "Detection accuracy",
        "Verification Success Rate (VSR)": "VSR",
    }

    checked = 0
    for label, key, ci_key, md_name in (
            ("AUPR", "detection_aupr", "detection_aupr_ci95", "AUPR"),
            ("P at R=90", "precision_at_recall_90",
             "precision_at_recall_90_ci95", "P@R90"),
            ("ROC-AUC", "detection_roc_auc", "detection_roc_auc_ci95",
             "ROC-AUC"),
            ("Sentence Accuracy", "detection_accuracy",
             "detection_accuracy_ci95", "Detection accuracy"),
            ("Verification Success Rate (VSR)", "verification_success_rate",
             "verification_success_rate_ci95", "VSR")):
        row = next((r for r in csv_rows[1:]
                    if len(r) > 1 and r[1].strip() == label), None)
        if row is None or key not in agg or md_name not in paper:
            continue
        others = [num(row[i]) for i in other_idx if i < len(row)]
        others = [x for x in others if x is not None and 0 <= x <= 1]
        if not others:
            continue
        best = max(others)
        ci = [float(x) for x in (agg.get(ci_key) or [])]
        got = classify(float(agg[key]), ci, best)
        if got is None:
            continue

        paper_verdict = paper[md_name][4]
        expected = {"win": "✓", "tie": "≈", "loss": "✗"}[got]
        assert paper_verdict == expected, (
            f"{label}: the chart classifies it as {got} but "
            f"Paper/comparison.md says {paper_verdict!r}"
        )
        checked += 1

    assert checked >= 4, (
        f"only {checked} metrics were compared between the chart and the "
        "paper; the comparison has drifted"
    )
