"""The three win-comparison charts must agree with the measurements.

``scripts/plot_win_comparisons.py`` draws one diagram per metric where
the paper claims a win: Detection AUPR, P@R90 and claim-level F1.

Each chart carries a verdict badge, so each one can overclaim. These
tests check that the numbers the chart would draw are the numbers the
artefacts hold, and that the verdict it computes is the verdict the
paper prints. The rule is the project's one: a win needs the bottom of
our 95% interval strictly above the published comparable, a loss needs
the top strictly below, and anything else is a tie.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import plot_win_comparisons as pwc

METRICS_PATH = Path("data/metrics_halueval.json")
COMPARISON_PATH = Path("Paper/comparison.md")
CSV_PATH = Path("Paper/comparison_table.csv")
GRAPH_DIR = Path("data/graphs")


# ---------------------------------------------------------------------------
# The three charts exist and are configured against real metrics
# ---------------------------------------------------------------------------

SLUGS = ["win_aupr", "win_pr90", "win_claim_f1"]


@pytest.fixture(scope="module")
def agg():
    if not METRICS_PATH.exists():
        pytest.skip("metrics file not present")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))["aggregate"]


def test_there_are_three_charts_one_per_metric():
    assert len(pwc.CHARTS) == 3, "there must be one diagram per winning metric"
    assert [c["slug"] for c in pwc.CHARTS] == SLUGS


@pytest.mark.parametrize("slug", SLUGS)
def test_each_chart_has_been_rendered(slug):
    path = GRAPH_DIR / f"{slug}.png"
    assert path.exists(), f"{slug}.png was not generated"
    assert path.stat().st_size > 5000, f"{slug}.png is suspiciously small"


@pytest.mark.parametrize("cfg", pwc.CHARTS, ids=[c["slug"] for c in pwc.CHARTS])
def test_each_chart_reads_a_metric_that_exists(cfg, agg):
    """A chart pointing at a missing metric would silently skip itself."""
    assert cfg["metric_key"] in agg, (
        f"{cfg['slug']} reads {cfg['metric_key']}, which is not in the "
        "metrics file"
    )
    if cfg["ci_key"]:
        assert cfg["ci_key"] in agg, (
            f"{cfg['slug']} reads the interval {cfg['ci_key']}, which is not "
            "in the metrics file"
        )


@pytest.mark.parametrize("cfg", pwc.CHARTS, ids=[c["slug"] for c in pwc.CHARTS])
def test_each_chart_has_a_win_matrix_row(cfg):
    """The comparable is taken from the win matrix, so the row must exist."""
    row = pwc._win_matrix_comparable(cfg["dimension"])
    assert row is not None, (
        f"{cfg['slug']} has no win-matrix row for {cfg['dimension']!r}"
    )
    _name, score = row
    assert 0 < score < 1, f"{cfg['slug']} has an implausible comparable {score}"


# ---------------------------------------------------------------------------
# The verdict is computed, not typed in
# ---------------------------------------------------------------------------

def test_the_verdict_rule_works_in_both_directions():
    """A win needs the whole interval above; a loss needs it all below."""
    assert pwc._verdict(0.922, (0.9057, 0.9383), 0.840) == "WIN"
    assert pwc._verdict(0.925, (0.9113, 0.9401), 0.987) == "LOSS"
    # Straddling is a tie, not a win.
    assert pwc._verdict(0.829, (0.8090, 0.8510), 0.847) == "TIE"
    # A point estimate above the comparable is not a win while the
    # interval covers it.
    assert pwc._verdict(0.853, (0.833, 0.874), 0.840) == "TIE"


def test_a_metric_with_no_interval_is_compared_on_the_point_estimate():
    """Claim-level F1 has no published interval, so it falls back to points.

    The fallback must still be a comparison, not a free pass: a value
    below the comparable is a loss.
    """
    assert pwc._verdict(0.811, None, 0.710) == "WIN"
    assert pwc._verdict(0.600, None, 0.710) == "LOSS"
    assert pwc._verdict(0.710, None, 0.710) == "TIE"


# ---------------------------------------------------------------------------
# The charts agree with the artefacts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", pwc.CHARTS, ids=[c["slug"] for c in pwc.CHARTS])
def test_the_chart_computes_a_win_for_each_of_the_three_metrics(cfg, agg):
    """Each of these three metrics must actually come out as a win.

    The paper claims three score wins. If a re-measurement stopped
    clearing the comparable, this test fails rather than letting a chart
    keep saying WIN.
    """
    ci = None
    if cfg["ci_key"]:
        lo, hi = agg[cfg["ci_key"]]
        ci = (float(lo), float(hi))
    ours = float(agg[cfg["metric_key"]])
    _name, comparable = pwc._win_matrix_comparable(cfg["dimension"])
    verdict = pwc._verdict(ours, ci, comparable)
    assert verdict == "WIN", (
        f"{cfg['slug']} is no longer a win: ours {ours} with CI {ci} "
        f"against comparable {comparable} gives {verdict}"
    )


@pytest.mark.parametrize("cfg", pwc.CHARTS, ids=[c["slug"] for c in pwc.CHARTS])
def test_the_gap_the_chart_draws_is_the_real_gap(cfg, agg):
    """The +N.NNN the arrow shows must be ours minus the comparable."""
    ours = float(agg[cfg["metric_key"]])
    _name, comparable = pwc._win_matrix_comparable(cfg["dimension"])
    gap = ours - comparable
    assert gap > 0, f"{cfg['slug']} draws a gap but is not ahead"
    # The gap is printed to three decimals on the chart.
    assert f"+{gap:.3f}" in {
        "+0.082", "+0.097", "+0.101",
    }, f"{cfg['slug']} has an unexpected gap {gap:.3f}"


def test_the_published_comparables_are_read_from_the_artefacts():
    """The AUPR chart must draw the comparables the CSV actually holds."""
    published = pwc._published_scores("AUPR")
    names = [p[0] for p in published]
    values = {p[0]: p[1] for p in published}
    assert any("Decoding Memories" in n for n in names), (
        "the AUPR chart has no Decoding Memories comparable"
    )
    dm = next(v for n, v in values.items() if "Decoding Memories" in n)
    assert dm == 0.840, f"the Decoding Memories AUPR changed to {dm}"
    # The band is kept so the chart can show that 0.840 is the lower end.
    band = next(p[3] for p in published if "Decoding Memories" in p[0])
    assert band == 0.850, f"the Decoding Memories AUPR band changed to {band}"


def test_the_pr90_chart_reads_the_dale_comparable():
    published = pwc._published_scores("P at R=90")
    assert published, "the P@R90 chart has no published comparables"
    name, value, variant, _band = published[0]
    assert "Dale" in name, f"the P@R90 comparable is {name}, not Dale 2023"
    assert value == 0.703, f"Dale's P@R90 changed to {value}"
    assert variant == "LaBSE", (
        f"the P@R90 variant is {variant!r}; the paper quotes Dale's LaBSE "
        "number"
    )


def test_the_claim_f1_chart_reads_the_papers_claim_table():
    """The claim-level comparables come from the paper's own table."""
    scores = pwc._claim_level_scores()
    assert scores, "the claim-level chart has no published comparables"
    by_name = {n: v for n, v, _g, _b in scores}
    assert "TU Wien — Vectara" in by_name, (
        f"the claim-level table has no TU Wien row: {list(by_name)}"
    )
    assert by_name["TU Wien — Vectara"] == 0.710
    assert by_name["AILS-NTUA — Voting"] == 0.690
    assert by_name["AILS-NTUA — NLI"] == 0.590
    # Our own row is atomic-claim, so it is not a comparable.
    assert not any(n.lower().startswith("ours") for n in by_name)


def test_a_cell_that_is_not_a_published_score_is_rejected():
    """Implied, projected and filler cells are not comparable scores."""
    for cell in ("0.55 (implied from Δ)", "0.60 (projected)",
                 "0 (no rewriting)", "n/a", "not reported"):
        assert any(tag in cell.lower() for tag in pwc.IMPLIED), (
            f"{cell!r} should have been rejected as a published score"
        )
    # The mitigation rows carry exactly those cells, so the rejection
    # rule has to actually reject them.
    rows = pwc._csv_rows()
    ours_idx = next(i for i, h in enumerate(rows[0])
                     if h.strip().lower().startswith("ours"))
    for row in rows[1:]:
        if len(row) < 3 or row[1].strip() != "Hallucination Reduction Rate (HRR)":
            continue
        for i in range(2, len(rows[0])):
            if i == ours_idx:
                continue
            cell = str(row[i] or "").strip()
            if cell in ("-", ""):
                continue
            assert any(tag in cell.lower() for tag in pwc.IMPLIED), (
                f"the HRR row's cell {cell!r} is not a published score but "
                "is not rejected either"
            )
            return
    pytest.fail("the HRR row was not found in the comparison table")


# ---------------------------------------------------------------------------
# The chart cannot measure against a number no paper reports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", pwc.CHARTS, ids=[c["slug"] for c in pwc.CHARTS])
def test_the_named_comparable_is_one_of_the_published_ones(cfg):
    """The comparable the paper names must exist among the published rows.

    Otherwise the chart would be measuring against a number that no cited
    work reports, which is how a chart once claimed a P@R90 win against
    0.674 while the paper called the same metric "at par" against 0.703.
    """
    name, _score = pwc._win_matrix_comparable(cfg["dimension"])
    if cfg["source"] == "claim_table":
        published = pwc._claim_level_scores()
    else:
        published = pwc._published_scores(cfg["csv_label"])
    names = [p[0] for p in published]
    assert name in names or any(
        n.lower().startswith(name.split("—")[0].strip().lower()[:5])
        for n in names
    ), f"{cfg['slug']} names {name!r}, which is not among {names}"


# ---------------------------------------------------------------------------
# The paper's claim-level table is honest about what it reports
# ---------------------------------------------------------------------------

def test_the_claim_level_table_gives_each_cited_work_its_own_f1():
    """The paper's claim-level table must not present an F1 as a recall.

    The table used to have a single "P / R / F1" column, but the cited
    rows carry two numbers -- their precision and their F1 -- and the
    third column was a heading that promised a number nobody published.
    """
    if not COMPARISON_PATH.exists():
        pytest.skip("comparison.md not present")
    text = COMPARISON_PATH.read_text(encoding="utf-8")
    assert "| System | Precision | Recall | F1 | Granularity |" in text, (
        "the claim-level table does not have one column per published number"
    )
    # The recall of a cited work must be marked as unpublished rather
    # than filled with a guess. Only the claim-level table's rows are
    # checked -- the architecture table has rows for the same papers
    # whose second column is a system type, not a recall.
    for line in text.splitlines():
        if not (line.startswith("| TU Wien") or line.startswith("| AILS-NTUA")):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        if not re.fullmatch(r"[\d.]+", cells[1]):
            continue  # not a claim-level row
        assert cells[2] == "not reported", (
            f"{cells[0]}'s recall is {cells[2]!r}; the cited works do "
            "not publish a recall, so it must not be filled in"
        )


def test_the_comparison_table_does_not_label_an_f1_as_a_recall():
    """The CSV's claim-level P/R row must not carry an F1 under that label.

    The row held "0.80 / 0.69" for AILS-NTUA, where 0.69 is that work's
    F1 -- the same number its F1 row already carries -- so the second
    number was a recall slot filled with an F1.
    """
    if not CSV_PATH.exists():
        pytest.skip("comparison table not present")
    rows = pwc._csv_rows()
    header = [h.strip() for h in rows[0]]
    ails_idx = next((i for i, h in enumerate(header)
                     if h.startswith("AILS-NTUA")), None)
    assert ails_idx is not None, "the comparison table has no AILS-NTUA column"
    row = next((r for r in rows[1:]
                if len(r) > 1 and r[1].strip() == "P/R"), None)
    assert row is not None, "the claim-level P/R row is missing"
    ails = row[ails_idx] if ails_idx < len(row) else ""
    assert "not reported" in ails.lower(), (
        f"the AILS-NTUA P/R cell is {ails!r}; its F1 must not sit in the "
        "recall slot"
    )
