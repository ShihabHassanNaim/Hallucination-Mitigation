"""Tests for the results-summary generator's mitigation section.

``data/RESULTS.md`` is generated, so the mitigation verdict it prints is
derived from the measurement. The rule it must follow is the same one
everywhere else in the project: a metric is a win only when the *interval*
separates it, not when the point estimate happens to land on the right
side of zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import summarize_results


def _mitigation_file(tmp_path: Path, regen: dict, **summary_extra) -> Path:
    summary = {
        "n_rows": 100,
        "n_regenerated": regen.get("n", 20),
        "n_kept": 80,
        "regenerated_rows": regen,
        "kept_rows": {"n": 80, "delta_bertscore": 0.0},
    }
    summary.update(summary_extra)
    path = tmp_path / "mitigation.json"
    path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    return path


def _metrics_file(tmp_path: Path) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({
            "dataset": "halueval",
            "aggregate": {"detection_roc_auc": 0.843, "detection_aupr": 0.853},
        }),
        encoding="utf-8",
    )
    return path


def _generate(tmp_path: Path, monkeypatch, mit_path: Path) -> str:
    """Run the generator against the temporary files and return the text."""
    metrics = _metrics_file(tmp_path)
    out = tmp_path / "RESULTS.md"
    monkeypatch.setattr(summarize_results, "MITIGATION_PATH", str(mit_path))
    monkeypatch.setattr(
        summarize_results, "DATASETS", [("halueval", "HaluEval", str(metrics))]
    )
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_results", "--out", str(out)],
    )
    summarize_results.main()
    return out.read_text(encoding="utf-8")


def test_a_positive_delta_with_an_interval_above_zero_is_reported_as_an_improvement(
    tmp_path, monkeypatch
):
    mit = _mitigation_file(tmp_path, {
        "n": 20,
        "pre_bertscore": 0.60,
        "post_bertscore": 0.66,
        "delta_bertscore": 0.06,
        "delta_bertscore_ci95": [0.02, 0.10],
    })
    text = _generate(tmp_path, monkeypatch, mit)
    assert "The delta is positive and its 95% CI excludes zero" in text


def test_a_positive_delta_whose_interval_covers_zero_is_not_reported_as_an_improvement(
    tmp_path, monkeypatch
):
    mit = _mitigation_file(tmp_path, {
        "n": 20,
        "pre_bertscore": 0.60,
        "post_bertscore": 0.62,
        "delta_bertscore": 0.02,
        "delta_bertscore_ci95": [-0.01, 0.05],
    })
    text = _generate(tmp_path, monkeypatch, mit)
    assert "not distinguishable from zero" in text
    assert "The delta is positive" not in text


def test_a_negative_delta_is_reported_as_not_positive(tmp_path, monkeypatch):
    mit = _mitigation_file(tmp_path, {
        "n": 20,
        "pre_bertscore": 0.86,
        "post_bertscore": 0.83,
        "delta_bertscore": -0.035,
        "delta_bertscore_ci95": [-0.067, -0.006],
    })
    text = _generate(tmp_path, monkeypatch, mit)
    assert "**This delta is not positive.**" in text


def test_the_unique_query_count_and_p_value_are_surfaced(tmp_path, monkeypatch):
    mit = _mitigation_file(
        tmp_path,
        {
            "n": 20,
            "n_unique_queries": 18,
            "pre_bertscore": 0.86,
            "post_bertscore": 0.83,
            "delta_bertscore": -0.035,
            "delta_bertscore_ci95": [-0.067, -0.006],
            "delta_bertscore_p": 0.019,
        },
        n_unique_queries=18,
        n_guarded=4,
    )
    text = _generate(tmp_path, monkeypatch, mit)
    assert "| Unique queries | 18 |" in text
    assert "| Rows flagged but guarded (answer left intact) | 4 |" in text
    assert "paired permutation p" in text
    assert "not independent" in text


def test_the_mitigation_section_is_absent_without_a_measurement(tmp_path, monkeypatch):
    """An unmeasured delta must not appear at all, rather than appear
    with a made-up number."""
    text = _generate(tmp_path, monkeypatch, tmp_path / "missing.json")
    assert "## Mitigation Delta" not in text


def test_a_partial_run_states_its_sample_size(tmp_path, monkeypatch):
    """A run over a prefix of the split must say so.

    The mitigation evaluation can be run over fewer rows than the file
    holds. A reader who only sees "Rows evaluated: 500" could assume the
    whole 1000-row split was measured.
    """
    mit = _mitigation_file(
        tmp_path,
        {
            "n": 107,
            "pre_bertscore": 0.86,
            "post_bertscore": 0.83,
            "delta_bertscore": -0.035,
            "delta_bertscore_ci95": [-0.067, -0.006],
        },
        n_rows=500,
        n_rows_in_file=1000,
    )
    text = _generate(tmp_path, monkeypatch, mit)
    assert "| Rows in the evaluation file | 1000 |" in text
    assert "| Rows evaluated | 500 |" in text
    assert "the first 500 rows" in text
