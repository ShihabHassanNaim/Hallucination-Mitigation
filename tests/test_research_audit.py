"""Check that RESEARCH_AUDIT.md agrees with the artefacts.

The audit report quotes a lot of numbers. This test reads each of them back
out of the source file it claims to come from, so a stale number in the
report fails the suite instead of reaching a reader.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "RESEARCH_AUDIT.md"
FIGDIR = ROOT / "data" / "graphs" / "audit"


def _text() -> str:
    return REPORT.read_text(encoding="utf-8")


def _flat() -> str:
    """The report with the typographic minus normalised to a plain hyphen.

    The report writes negative numbers with a Unicode minus sign (U+2212),
    which is right for prose but would make a machine check miss them.
    """
    return _text().replace("\u2212", "-")


def _json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _agg(rel: str) -> dict:
    return _json(rel)["aggregate"]


# ---------------------------------------------------------------------------
# The report exists and embeds every figure it claims to
# ---------------------------------------------------------------------------

FIGURES = [
    "audit_01_pipeline.png",
    "audit_02_detection.png",
    "audit_03_length_confound.png",
    "audit_04_mitigation.png",
    "audit_05_ci_separation.png",
    "audit_06_latency.png",
    "audit_07_ledger.png",
    "audit_08_datasets.png",
]


def test_the_report_exists_and_is_substantial():
    assert REPORT.exists(), "RESEARCH_AUDIT.md is missing"
    assert len(_text()) > 30000, "the audit report is suspiciously short"


@pytest.mark.parametrize("name", FIGURES)
def test_every_figure_is_embedded_and_exists(name):
    assert (FIGDIR / name).exists(), f"{name} was not generated"
    assert name in _text(), f"{name} is not embedded in the report"


# ---------------------------------------------------------------------------
# Headline numbers are quoted exactly
# ---------------------------------------------------------------------------

HEADLINES = [
    # (metric file, metric key, decimals the report quotes it at)
    ("data/metrics_halueval.json", "detection_roc_auc", 6),
    ("data/metrics_halueval.json", "detection_aupr", 6),
    ("data/metrics_halueval.json", "detection_f1", 6),
    ("data/metrics_halueval.json", "detection_accuracy", 3),
    ("data/metrics_halueval.json", "precision_at_recall_90", 6),
    ("data/metrics_halueval.json", "spearman_rho", 6),
    ("data/metrics_halueval.json", "end_to_end_accuracy", 3),
    ("data/metrics_halueval.json", "hallucination_reduction_rate", 3),
    ("data/metrics_halueval.json", "claim_correction_rate", 3),
    ("data/metrics_halueval.json", "verification_success_rate", 3),
    ("data/metrics_halueval.json", "claim_f1", 6),
    ("data/metrics_halueval.json", "latency_mean_s", 3),
    ("data/metrics_halueval.json", "latency_median_s", 3),
    ("data/metrics_halueval.json", "latency_min_s", 3),
    ("data/metrics_halueval.json", "latency_max_s", 3),
    ("data/metrics_truthfulqa_closedbook.json", "detection_roc_auc", 6),
    ("data/metrics_truthfulqa_closedbook.json", "detection_accuracy", 3),
    ("data/metrics_truthfulqa_closedbook.json", "detection_recall", 3),
    ("data/metrics_truthfulqa_closedbook.json", "end_to_end_accuracy", 3),
    ("data/metrics_truthfulqa_closedbook.json", "latency_mean_s", 3),
    ("data/metrics_truthfulqa_closedbook.json", "latency_median_s", 3),
    ("data/metrics_truthfulqa_closedbook.json", "latency_min_s", 3),
    ("data/metrics_truthfulqa_closedbook.json", "latency_max_s", 3),
    ("data/metrics_ragtruth_fitted.json", "detection_roc_auc", 6),
    ("data/metrics_ragtruth_fitted.json", "detection_accuracy", 3),
    ("data/metrics_ragtruth_fitted.json", "end_to_end_accuracy", 3),
    ("data/metrics_ragtruth_fitted.json", "latency_mean_s", 3),
    ("data/metrics_ragtruth_fitted.json", "latency_median_s", 3),
    ("data/metrics_ragtruth_fitted.json", "latency_min_s", 3),
    ("data/metrics_ragtruth_fitted.json", "latency_max_s", 3),
    ("data/metrics_halueval_raw.json", "detection_roc_auc", 6),
    ("data/metrics_halueval_raw.json", "detection_precision", 6),
    ("data/metrics_halueval_raw.json", "precision_at_recall_90", 6),
    ("data/metrics_halueval_raw.json", "spearman_rho", 6),
    ("data/metrics_halueval_raw.json", "detection_recall", 3),
    ("data/metrics_halueval_raw.json", "end_to_end_accuracy", 3),
]


@pytest.mark.parametrize("rel,key,dp", HEADLINES)
def test_the_report_quotes_the_headline_numbers_exactly(rel, key, dp):
    value = _agg(rel)[key]
    quoted = f"{value:.{dp}f}"
    assert quoted in _flat(), (
        f"{rel}:{key} is {value!r} but the report does not quote {quoted}"
    )


def test_the_report_quotes_every_confidence_interval_it_claims():
    """Every CI the report prints for a headline metric must be the artefact's."""
    md = _text()
    for rel, key in [
        ("data/metrics_halueval.json", "detection_roc_auc_ci95"),
        ("data/metrics_halueval.json", "detection_aupr_ci95"),
        ("data/metrics_halueval.json", "detection_accuracy_ci95"),
        ("data/metrics_halueval.json", "precision_at_recall_90_ci95"),
        ("data/metrics_halueval.json", "spearman_rho_ci95"),
        ("data/metrics_truthfulqa_closedbook.json", "detection_roc_auc_ci95"),
        ("data/metrics_ragtruth_fitted.json", "detection_roc_auc_ci95"),
    ]:
        low, high = _agg(rel)[key]
        assert f"[{low:.4f}, {high:.4f}]" in _flat(), (
            f"{rel}:{key} interval [{low:.4f}, {high:.4f}] is not in the report"
        )


# ---------------------------------------------------------------------------
# The three dataset descriptions
# ---------------------------------------------------------------------------

def test_the_dataset_table_matches_the_eval_files():
    md = _text()
    for rel, rows, queries in [
        ("data/halueval_eval_1000_matched.jsonl", 1000, 500),
        ("data/truthfulqa_eval_1000.jsonl", 1000, 500),
        ("data/ragtruth_eval_1000_matched.jsonl", 1000, 277),
    ]:
        path = ROOT / rel
        assert path.exists(), rel
        n, q = 0, set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n += 1
            q.add(json.loads(line)["query"])
        assert n == rows, f"{rel} has {n} rows, the report says {rows}"
        assert len(q) == queries, f"{rel} has {len(q)} queries, report says {queries}"
        assert rel in md, f"the report does not name {rel}"


def test_the_detector_training_data_numbers_are_current():
    md = _text()
    clean = ROOT / "data/detector_train_clean.jsonl"
    n, q = 0, set()
    for line in clean.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        q.add(json.loads(line)["query"])
    assert (n, len(q)) == (2290, 1145)
    assert "2,290" in md and "1,145" in md
    assert "data/detector_train_clean.jsonl" in md


# ---------------------------------------------------------------------------
# The negative results are quoted correctly -- these matter most
# ---------------------------------------------------------------------------

def test_the_mitigation_null_result_is_quoted_correctly():
    md = _text()
    regen = _json("data/mitigation_results.json")["summary"]["regenerated_rows"]
    assert regen["n"] == 16
    assert regen["n_unique_queries"] == 8
    assert f"{regen['delta_bertscore']:.6f}" == "-0.006940"
    assert "-0.006940" in _flat()
    assert f"{regen['delta_bertscore_p']:.3f}" == "0.404"
    assert "0.404" in md
    low, high = regen["delta_bertscore_ci95"]
    # The report writes a signed interval, so the upper bound carries a "+".
    assert f"[{low:.4f}, {high:.4f}]" == "[-0.0217, 0.0069]"
    assert "[-0.0217, +0.0069]" in _flat()
    # The report must call it a null, not a win.
    assert "p = 0.404" in md or "p = 0.404," in md


def test_the_mitigation_row_counts_are_quoted():
    md = _text()
    s = _json("data/mitigation_results.json")["summary"]
    assert s["n_rows"] == 500
    assert s["n_kept"] == 484
    assert s["n_guarded"] == 86
    assert s["n_regenerated"] == 16
    for n in ("500", "484", "86", "16"):
        assert n in md


def test_the_span_level_negative_result_is_quoted():
    md = _text()
    span = _json("data/span_level_ragtruth.json")
    assert f"{span['span_f1']:.4f}" == "0.0455"
    assert "0.0455" in md
    assert "0.527" in md, "the published span F1 must be shown next to ours"


# ---------------------------------------------------------------------------
# The win rule is stated correctly and the tally matches the paper
# ---------------------------------------------------------------------------

def test_the_report_states_the_two_sided_interval_rule():
    md = _text()
    assert "bottom of our 95% confidence interval" in md
    assert "upper" in md.lower(), "the rule must also cover the loss direction"


def test_the_win_matrix_tally_matches_the_paper():
    md = _text()
    paper = (ROOT / "Paper" / "comparison.md").read_text(encoding="utf-8")
    tally = re.search(
        r"\|\s*\*\*✓ Win\*\*\s*\|\s*(\d+)\s*\|.*?"
        r"\|\s*\*\*≈ Tie / At par\*\*\s*\|\s*(\d+)\s*\|.*?"
        r"\|\s*\*\*✗ Loss\*\*\s*\|\s*(\d+)\s*\|",
        paper,
        re.S,
    )
    assert tally, "the paper's verdict tally could not be parsed"
    wins, ties, losses = (int(g) for g in tally.groups())
    assert (wins, ties, losses) == (7, 5, 8)
    assert "7 wins / 5 ties / 8 losses" in md


def test_the_ragtruth_head_to_head_is_quoted_correctly():
    md = _text()
    anchors = _json("data/published_anchors.json")["auroc_family"]["anchors"]
    values = sorted(a["value"] for a in anchors)
    assert values == [0.5168, 0.5327, 0.5458, 0.5724, 0.583]
    for v in values:
        assert f"{v:.4f}" in md, f"the published anchor {v} is not in the report"
    assert "3 wins" in md and "2 ties" in md


def test_the_held_out_figure_is_reported_beside_the_in_sample_one():
    md = _text()
    head = _json("data/detection_head_refit600.json")
    assert head["n_rows"] == 496
    model = head["models"]["logistic (fitted)"]
    assert f"{model['roc_auc_holdout']:.4f}" == "0.6185"
    assert "0.6185" in md
    low, high = model["roc_auc_holdout_ci95"]
    assert f"[{low:.4f}, {high:.4f}]" == "[0.5371, 0.6943]"
    assert "[0.5371, 0.6943]" in md


# ---------------------------------------------------------------------------
# Section 3.2.5 -- the result analysis against other papers
# ---------------------------------------------------------------------------

def _section_325() -> str:
    """The text of the result-analysis section, from its heading to the next one."""
    md = _text()
    start = md.index("#### 3.2.5 Result analysis")
    end = md.index("### 3.3 Claim-level granularity", start)
    return md[start:end]


def test_the_result_analysis_section_exists():
    sec = _section_325()
    assert len(sec) > 3000, "the result-analysis section is too short to be useful"
    assert "3.2.5" in _text()


def test_the_result_analysis_quotes_the_winning_scores_exactly():
    sec = _section_325()
    hal = _agg("data/metrics_halueval.json")
    # P@R90 0.800 vs Dale's 0.703.
    assert f"{hal['precision_at_recall_90']:.3f}" == "0.800"
    assert "0.800" in sec and "0.703" in sec
    low, high = hal["precision_at_recall_90_ci95"]
    assert f"[{low:.4f}, {high:.4f}]" == "[0.7732, 0.8443]"
    assert "[0.7732, 0.8443]" in sec
    # Claim-level F1 0.811 vs TU Wien's 0.710.
    assert f"{hal['claim_f1']:.3f}" == "0.811"
    assert "0.811" in sec and "0.710" in sec
    assert "0.467" in sec and "0.810" in sec


def test_detection_aupr_is_reported_as_a_measurement_not_a_win():
    """No published paper reports AUPR on HaluEval.

    Decoding Memories was previously cited here at 0.840, but its
    Table 1 reports AUROC -- "AUPR", "AUC-PR" and "PR-AUC" do not
    appear in that paper -- and the 0.840 was EigenScore-B's HaluEval
    AUROC. Comparing our AUPR against somebody else's AUROC is
    comparing two different quantities, so the row reports the
    measurement and says there is no comparable.
    """
    sec = _section_325()
    hal = _agg("data/metrics_halueval.json")
    assert f"{hal['detection_aupr']:.3f}" == "0.922"
    assert "0.922" in sec
    low, high = hal["detection_aupr_ci95"]
    assert f"[{low:.4f}, {high:.4f}]" == "[0.9057, 0.9383]"
    assert "[0.9057, 0.9383]" in sec
    # The row must name the metric, say there is no comparable, and say
    # why -- not carry a verdict.
    assert "AUPR" in sec
    assert "no published comparable" in sec.lower() or \
        "no published paper reports AUPR" in sec
    # The old AUROC number must not be back as a comparable for AUPR. It
    # may still appear in the paragraph that explains why the comparison
    # was removed, so the check looks for it in a table cell rather than
    # anywhere in the section.
    for line in sec.splitlines():
        if not line.startswith("|"):
            continue
        assert "0.840" not in line, (
            f"a table row still quotes Decoding Memories' 0.840 against "
            f"our AUPR: {line}"
        )


def test_the_result_analysis_states_the_gaps_it_claims():
    sec = _section_325()
    hal = _agg("data/metrics_halueval.json")
    # The P@R90 gap and the claim-F1 gap are stated as differences.
    assert "0.097" in sec, "the P@R90 gap must be stated"
    assert f"{hal['precision_at_recall_90'] - 0.703:.3f}" == "0.097"
    assert "0.101" in sec, "the claim-F1 gap must be stated"
    assert f"{hal['claim_f1'] - 0.710:.3f}" == "0.101"


def test_the_result_analysis_ties_are_quoted_with_their_intervals():
    sec = _section_325()
    hal = _agg("data/metrics_halueval.json")
    # Sentence accuracy: the tie must show the interval straddling 0.847.
    assert "0.847" in sec
    assert "0.829" in sec
    low, high = hal["detection_accuracy_ci95"]
    assert f"[{low:.4f}, {high:.4f}]" == "[0.8090, 0.8510]"
    assert "[0.8090, 0.8510]" in sec
    # The section must say the interval straddles the comparable.
    assert "straddle" in sec.lower()


def test_the_result_analysis_ragtruth_ties_are_quoted():
    sec = _section_325()
    anchors = _json("data/published_anchors.json")["auroc_family"]["anchors"]
    for a in anchors:
        if a["value"] in (0.583, 0.5724):
            assert f"{a['value']:.4f}" in sec, f"the tie anchor {a['value']} is missing"


def test_the_result_analysis_tally_matches_the_paper():
    sec = _section_325()
    paper = (ROOT / "Paper" / "comparison.md").read_text(encoding="utf-8")
    tally = re.search(
        r"\|\s*\*\*✓ Win\*\*\s*\|\s*(\d+)\s*\|.*?"
        r"\|\s*\*\*≈ Tie / At par\*\*\s*\|\s*(\d+)\s*\|.*?"
        r"\|\s*\*\*✗ Loss\*\*\s*\|\s*(\d+)\s*\|",
        paper,
        re.S,
    )
    assert tally
    wins, ties, losses = (int(g) for g in tally.groups())
    assert (wins, ties, losses) == (7, 5, 8)
    assert "7 wins, 5 ties" in sec or "7 wins / 5 ties / 8 losses" in sec
    assert "3 wins" in sec and "2 ties" in sec


def test_the_result_analysis_explains_why_the_section_matters():
    """The user asked for a brief explanation of why this matters."""
    sec = _section_325().lower()
    assert "why these" in sec or "why the structural wins matter" in sec
    assert "why ties are important" in sec
    # The plain-English framing of the three verdicts must be present.
    assert "whole 95% interval sits" in sec
    assert "overlaps" in sec


def test_the_result_analysis_does_not_hide_the_losses():
    """A win-focused section may still not pretend the losses do not exist."""
    sec = _section_325()
    assert "loss" in sec.lower()
    assert "8 losses" in sec or "and 8 losses" in sec
    # The concrete losses must be named, not implied away.
    assert "0.987" in sec, "the ROC-AUC loss must be named"
    assert "19" in sec, "the latency loss must be named"


def test_the_result_analysis_is_in_the_table_of_contents():
    md = _text()
    assert "[Result analysis: where we win, and where we tie]" in md
    assert "#325-result-analysis-where-we-win-and-where-we-tie" in md


# ---------------------------------------------------------------------------
# The two win-comparison diagrams
# ---------------------------------------------------------------------------

WIN_FIGURES = ["win_pr90.png", "win_claim_f1.png"]


@pytest.mark.parametrize("name", WIN_FIGURES)
def test_the_win_diagrams_are_embedded_and_exist(name):
    """Each of the two wins has its own diagram, and the report shows it."""
    path = ROOT / "data" / "graphs" / name
    assert path.exists(), f"{name} was not generated"
    assert name in _text(), f"{name} is not embedded in the report"


def test_there_is_no_diagram_for_a_metric_with_no_published_comparable():
    """Detection AUPR has no published comparable, so it has no diagram.

    A diagram for it would have to draw our AUPR against somebody
    else's AUROC, which is the comparison that was removed.
    """
    path = ROOT / "data" / "graphs" / "win_aupr.png"
    assert not path.exists(), (
        "win_aupr.png still exists; no published paper reports AUPR on "
        "HaluEval, so there is nothing for it to be drawn against"
    )
    assert "win_aupr.png" not in _text()


def test_the_two_diagrams_are_in_the_score_wins_subsection():
    """They belong with the score wins, not somewhere else in the report."""
    md = _text()
    start = md.index("##### 3.2.5.1 The wins")
    end = md.index("##### 3.2.5.2", start)
    sec = md[start:end]
    for name in WIN_FIGURES:
        assert name in sec, f"{name} is not in the score-wins subsection"


def test_each_diagram_caption_explains_how_to_read_it():
    """A figure nobody can read is not evidence."""
    md = _text()
    start = md.index("##### 3.2.5.1 The wins")
    end = md.index("##### 3.2.5.2", start)
    sec = md[start:end]
    assert sec.count("*How to read it.*") == 2, (
        "each of the two diagrams needs a plain-English reading guide"
    )


def test_the_diagram_captions_quote_the_real_numbers():
    """The alt text must not drift from the measurements."""
    md = _text()
    hal = _agg("data/metrics_halueval.json")
    # P@R90.
    assert f"{hal['precision_at_recall_90']:.3f}" == "0.800"
    assert "[0.7732, 0.8443]" in md
    # Claim-level F1.
    assert f"{hal['claim_f1']:.3f}" == "0.811"
    assert "0.811" in md
    # The published comparables the diagrams draw.
    for comparable in ("0.703", "0.710"):
        assert comparable in md, f"the comparable {comparable} is not quoted"


def test_the_claim_level_diagram_says_the_metric_has_no_interval():
    """The figure must not imply an interval it does not have."""
    md = _text()
    start = md.index("##### 3.2.5.1 The wins")
    end = md.index("##### 3.2.5.2", start)
    sec = md[start:end]
    claim = sec[sec.index("win_claim_f1.png"):]
    assert "no confidence interval" in claim.lower()
    assert "point estimates" in claim.lower()


def test_the_diagrams_are_regenerated_by_a_script_the_report_names():
    """The figures must be reproducible, not hand-made."""
    md = _text()
    assert "python -m scripts.plot_win_comparisons" in md, (
        "the reproduction section does not name the script that draws "
        "the three win diagrams"
    )


# ---------------------------------------------------------------------------
# The length confound, the leak and the bad fusion
# ---------------------------------------------------------------------------

def test_the_length_confound_facts_are_quoted():
    md = _text()
    assert "94.7%" in md
    assert "50.3%" in md
    leak = _json("data/truthfulqa_leak_report.json")
    assert leak["rows_whose_own_answer_is_in_their_own_knowledge_doc"] == 500
    assert "466" in md, "the leaked training rows must be quoted"


def test_the_truthfulqa_leak_number_is_quoted():
    md = _text()
    assert "0.9589" in md, "the leaked AUC must be named so it can be recognised"


def test_the_bad_fusion_score_is_quoted():
    md = _text()
    head = _json("data/detection_head.json")
    bad = head["models"]["hand-set weights (current)"]["roc_auc_all"]
    assert f"{bad:.3f}" == "0.360"
    assert "0.360" in md


def test_the_component_table_matches_the_fitted_head():
    md = _text()
    head = _json("data/detection_head_refit600.json")
    for comp, w in head["fitted_head"]["weights"].items():
        assert f"{w:.6f}" in _flat(), f"the weight for {comp} ({w}) is not in the report"


def test_the_latency_numbers_match_the_metrics_files():
    md = _text()
    for rel, mean, median, lo, hi in [
        ("data/metrics_halueval.json", 4.180, 3.962, 0.966, 10.305),
        ("data/metrics_truthfulqa_closedbook.json", 4.123, 2.991, 0.765, 36.347),
        ("data/metrics_ragtruth_fitted.json", 10.010, 6.646, 1.172, 73.148),
    ]:
        a = _agg(rel)
        assert f"{a['latency_mean_s']:.3f}" == f"{mean:.3f}"
        assert f"{a['latency_median_s']:.3f}" == f"{median:.3f}"
        assert f"{a['latency_min_s']:.3f}" == f"{lo:.3f}"
        assert f"{a['latency_max_s']:.3f}" == f"{hi:.3f}"
        assert f"{mean:.3f}" in md


def test_the_codebase_stats_are_current():
    md = _text()
    py_files = list((ROOT / "src").rglob("*.py")) \
        + list((ROOT / "scripts").rglob("*.py")) \
        + list((ROOT / "tests").rglob("*.py"))
    loc = sum(len(f.read_text(encoding="utf-8", errors="replace").splitlines())
              for f in py_files)
    # The report's counts are a snapshot; they must not be *below* reality.
    assert str(len(py_files)) in md or "Python files" in md
    assert loc >= 24310, "the codebase shrank -- the report's LOC figure is stale"
    assert "24,310" in md


def test_the_report_names_the_reproduction_commands():
    md = _text()
    for cmd in [
        "python -m scripts.evaluate_full",
        "python -m scripts.fill_comparison_table",
        "python -m scripts.compare_to_published",
        "python -m scripts.plot_audit_figures",
        "python -m pytest tests/ -q",
    ]:
        assert cmd in md, f"the reproduction section is missing `{cmd}`"


def test_the_report_does_not_claim_the_mitigation_works():
    """The report must never describe the mitigation as a win.

    The phrase "the rewrite works" does appear, but only inside a quotation
    of the opposite claim ("not 'the rewrite works'"), so the check looks
    for the affirmative sentence and not the bare phrase.
    """
    flat = _flat().lower()
    for bad in [
        "our mitigation works",
        "the rewrite works.",
        "mitigation is a win",
        "we win on mitigation",
        "our rewrite works",
        "mitigation improves",
    ]:
        assert bad not in flat, f"the report claims `{bad}`"
    # And it must state the null result plainly.
    assert "negative result" in flat
    assert "no measurable effect" in flat
