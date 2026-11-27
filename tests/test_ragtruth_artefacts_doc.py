"""The RAGTruth artefacts document must agree with the artefacts it describes.

``docs/ragtruth_artefacts.md`` is a hand-written companion to
``data/metrics_ragtruth_fitted.json`` and
``data/detection_head_refit600.json``. It is not written by the paper
generator, so nothing keeps it in step with the run it describes: the
headline ROC-AUC, its interval, the per-task breakdown and the
LLM-Check verdicts all drifted at some point without a test noticing.

Two things are checked here.

The numbers are read out of the artefacts and required to appear in the
document, so a re-measurement that moves the headline fails the test
rather than leaving the prose stale.

The verdicts are *recomputed* from the interval rule the paper uses -- a
win needs the lower bound strictly above the published value -- so the
document cannot claim a win the interval does not support. The document
used to mark all five LLM-Check anchors as wins against a figure whose
lower bound was below three of them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "ragtruth_artefacts.md"
METRICS = ROOT / "data" / "metrics_ragtruth_fitted.json"
HEAD = ROOT / "data" / "detection_head_refit600.json"
ANCHORS = ROOT / "data" / "published_anchors.json"

# The LLM-Check anchors, in the order the document's table lists them.
LLM_CHECK = [
    ("Attention Score", 0.5830),
    ("Hidden Score", 0.5724),
    ("Logit, Window Entropy", 0.5458),
    ("Logit, Perplexity", 0.5327),
    ("Logit, Log Entropy", 0.5168),
]


def verdict(ci, published):
    """The interval rule, applied in both directions."""
    lower, upper = ci
    if lower > published:
        return "WIN"
    if upper < published:
        return "LOSS"
    return "TIE"


@pytest.fixture(scope="module")
def doc():
    if not DOC.exists():
        pytest.skip("the artefacts document is not present")
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agg():
    return json.loads(METRICS.read_text(encoding="utf-8"))["aggregate"]


@pytest.fixture(scope="module")
def holdout():
    fitted = json.loads(HEAD.read_text(encoding="utf-8"))["models"]["logistic (fitted)"]
    return fitted["roc_auc_holdout"], tuple(fitted["roc_auc_holdout_ci95"])


# ---------------------------------------------------------------------------
# The headline numbers come from the metrics file
# ---------------------------------------------------------------------------

def test_the_headline_roc_auc_is_the_one_in_the_metrics_file(doc, agg):
    value = agg["detection_roc_auc"]
    assert f"{value:.4f}" in doc, (
        f"the document does not quote the headline ROC-AUC {value:.4f}"
    )


def test_the_headline_interval_is_the_one_in_the_metrics_file(doc, agg):
    low, high = agg["detection_roc_auc_ci95"]
    assert f"[{low:.4f}, {high:.4f}]" in doc


@pytest.mark.parametrize(
    "key",
    [
        "detection_accuracy",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "detection_aupr",
        "spearman_rho",
        "claim_precision",
        "claim_recall",
        "claim_f1",
        "retrieval_recall_at_5",
    ],
)
def test_every_headline_metric_is_quoted(doc, agg, key):
    value = agg[key]
    assert f"{value:.4f}" in doc or f"{value:.3f}" in doc, (
        f"the document does not quote {key} ({value})"
    )


def test_the_per_task_breakdown_matches_the_metrics_file(doc, agg):
    """The per-task ROC-AUCs are computed the same way the paper computes them."""
    import scripts.fill_comparison_table as fill

    per_task = fill._ragtruth_per_task(METRICS)
    assert per_task, "the per-task breakdown could not be computed"
    for task, values in per_task.items():
        assert f"{values['roc_auc']:.4f}" in doc, (
            f"the document does not quote the {task} ROC-AUC"
        )
        # The row count is a column of the table, so it is matched as a
        # table cell rather than as "(n=408)".
        assert re.search(
            rf"^\|\s*{task}\s*\|\s*{values['n']}\s*\|", doc, re.MULTILINE
        ), f"the document's {task} row does not carry n={values['n']}"


def test_the_held_out_figure_is_quoted(doc, holdout):
    auc, ci = holdout
    assert f"{auc:.4f}" in doc
    assert f"[{ci[0]:.4f}, {ci[1]:.4f}]" in doc


# ---------------------------------------------------------------------------
# The verdicts are recomputed, not asserted
# ---------------------------------------------------------------------------

def _verdict_rows(doc):
    """The document's LLM-Check table, keyed by the anchor's short name.

    The table's system column reads "LLM-Check (Attention Score)", so the
    "LLM-Check (" prefix is stripped to get the same short name the
    anchors are listed under.
    """
    rows = {}
    for line in doc.splitlines():
        if not line.startswith("| LLM-Check ("):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # | system | metric | published | ours | verdict |
        short = cells[0].removeprefix("LLM-Check (").rstrip(")")
        rows[short] = (float(cells[2]), cells[4])
    return rows


def test_the_llm_check_table_has_every_anchor(doc):
    rows = _verdict_rows(doc)
    assert len(rows) == len(LLM_CHECK), (
        f"the document's LLM-Check table has {len(rows)} rows, expected "
        f"{len(LLM_CHECK)}"
    )


@pytest.mark.parametrize("name,published", LLM_CHECK)
def test_each_llm_check_verdict_follows_the_interval(doc, agg, name, published):
    """A win needs the lower bound strictly above the published value."""
    rows = _verdict_rows(doc)
    assert name in rows, f"the document's LLM-Check table has no {name} row"
    doc_published, doc_verdict = rows[name]
    assert doc_published == pytest.approx(published)

    ci = tuple(agg["detection_roc_auc_ci95"])
    want = verdict(ci, published)
    got = doc_verdict.strip("*").upper()
    assert got == want, (
        f"{name}: the document says {doc_verdict!r} but the interval rule "
        f"gives {want} (our CI {ci} against {published})"
    )


def test_the_held_out_tally_is_stated_correctly(doc, holdout):
    """On the held-out figure the tally is two wins and three ties.

    The held-out lower bound is 0.5371. It clears Perplexity (0.5327)
    and Log Entropy (0.5168), and not the other three. The document
    used to claim the held-out comparison was a tie against all five,
    which is what you get if you only compare the lower bound with the
    strongest anchor.
    """
    _auc, ci = holdout
    counts = {"WIN": 0, "TIE": 0, "LOSS": 0}
    for _name, published in LLM_CHECK:
        counts[verdict(ci, published)] += 1
    assert (counts["WIN"], counts["TIE"], counts["LOSS"]) == (2, 3, 0)

    assert "two wins and three ties" in doc, (
        "the document does not state the held-out tally"
    )
    assert "tie against all five" not in doc
    assert "5 ties" not in doc


def test_the_in_sample_tally_is_stated_correctly(doc, agg):
    """The headline figure gives three wins and two ties."""
    ci = tuple(agg["detection_roc_auc_ci95"])
    counts = {"WIN": 0, "TIE": 0, "LOSS": 0}
    for _name, published in LLM_CHECK:
        counts[verdict(ci, published)] += 1
    assert (counts["WIN"], counts["TIE"], counts["LOSS"]) == (3, 2, 0)
    assert "Three wins, two ties" in doc


def test_no_verdict_is_claimed_for_an_anchor_the_interval_does_not_clear(doc, agg):
    """No table line may call a win where the interval straddles."""
    ci = tuple(agg["detection_roc_auc_ci95"])
    for line in doc.splitlines():
        if not line.startswith("| LLM-Check ("):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        published = float(cells[2])
        mark = cells[4].strip("*").upper()
        assert mark == verdict(ci, published), (
            f"{cells[0]}: the row says {cells[4]!r} but the rule gives "
            f"{verdict(ci, published)}"
        )


# ---------------------------------------------------------------------------
# The stage-coverage counts in the survey are derived from its own grid
# ---------------------------------------------------------------------------

SURVEY = ROOT / "docs" / "related_pipeline_workflows.md"


@pytest.fixture(scope="module")
def survey():
    if not SURVEY.exists():
        pytest.skip("the survey is not present")
    return SURVEY.read_text(encoding="utf-8")


def _grid(survey):
    """The §3.1 stage grid, as a header and a list of rows."""
    lines = survey.splitlines()
    start = next(
        i for i, l in enumerate(lines)
        if l.startswith("| Paper | Query understanding")
    )
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    rows = [
        [c.strip() for c in l.strip("|").split("|")]
        for l in lines[start + 2:end]
    ]
    return header, rows


def test_the_survey_grid_marks_crsp_as_reranking(survey):
    """CRISP's step 7 is a real cross-encoder rerank.

    The config has ``step07_reranking.enabled: true`` and the model is
    ``BAAI/bge-reranker-v2-m3``, so the grid has to say so.
    """
    header, rows = _grid(survey)
    col = header.index("Rerank")
    crisp = next(r for r in rows if "CRISP" in r[0])
    assert crisp[col].startswith("✓"), (
        f"the survey's grid marks CRISP's rerank as {crisp[col]!r}; the "
        "pipeline does rerank (step07_reranking.enabled: true)"
    )


def test_the_survey_rerank_count_matches_its_own_grid(survey):
    """The counting table's Rerank row must follow from the grid.

    The counting table covers the 19 published papers, so CRISP is
    excluded from it. Exactly one of those 19 reranks, so the row must
    read 1 / 19 -- the count that excludes CRISP is 1, and counting
    CRISP as well would be 2.
    """
    header, rows = _grid(survey)
    col = header.index("Rerank")
    others = [r for r in rows if "CRISP" not in r[0]]
    assert len(others) == 19, (
        f"the grid has {len(others)} published papers, the counting table "
        "says 19"
    )
    count = sum(1 for r in others if r[col].startswith("✓"))

    m = re.search(
        r"^\|\s*Rerank\s*\|\s*(\d+)\s*/\s*19\s*\|(.+)\|\s*$",
        survey,
        re.MULTILINE,
    )
    assert m, "the survey has no Rerank row in the counting table"
    stated = int(m.group(1))
    assert stated == count, (
        f"the counting table says {stated} of 19 papers rerank, but the "
        f"grid marks {count}"
    )
    # The depth column must not name a paper the grid says does not
    # rerank.
    for name, _published in LLM_CHECK:
        assert name not in m.group(2)
    assert "Hybrid Retrieval" not in m.group(2), (
        "the Rerank row names Hybrid Retrieval, but the grid marks it as "
        "not reranking"
    )
    assert "CRISP" not in m.group(2), (
        "the Rerank row names CRISP, but that row counts the 19 published "
        "papers and CRISP is excluded from them"
    )
