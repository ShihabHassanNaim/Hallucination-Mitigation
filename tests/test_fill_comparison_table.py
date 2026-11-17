"""Tests for the comparison-table generator.

The comparison table must never contain a hand-written estimate presented
as a measurement, so these tests cover both halves of the guarantee: the
cells are rewritten from the evaluation output, and the plotting script
refuses to read a cell that is still marked as projected.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from scripts.fill_comparison_table import (
    _load_aggregate,
    build_mitigation_replacements,
    build_replacements,
    load_mitigation,
    main as fill_main,
)

# The projected detection row as it appears in Paper/comparison_summary.tex.
# Kept as a raw string so the trailing LaTeX row terminator ("\\") is
# unambiguous.
_TEX_DETECTION_ROW = (
    r"\textbf{Ours (Step 9 fused score)} & 0.91--0.95 & 0.85--0.90 & "
    r"0.55--0.70 & Black-box + retrieval \\"
)


def _metrics_file(tmp_path: Path, aggregate: dict, rows: list[dict] | None = None) -> Path:
    path = tmp_path / "metrics.json"
    payload: dict = {"dataset": "halueval", "aggregate": aggregate}
    if rows is not None:
        payload["per_query"] = rows
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _paper_dir(tmp_path: Path) -> Path:
    """An empty stand-in for ``Paper/``.

    ``main()`` refreshes the prose files alongside the CSV. Those paths
    used to be hard-coded, so a test that passed a temporary metrics file
    wrote that file's numbers into the *real* paper — which is how a
    superseded ROC-AUC kept reappearing in the working tree. Every test
    that calls ``main()`` now points the prose at a temporary directory.
    """
    paper = tmp_path / "paper"
    paper.mkdir(exist_ok=True)
    return paper


def test_build_replacements_uses_the_measured_values():
    agg = {
        "detection_roc_auc": 0.789,
        "detection_roc_auc_ci95": [0.73, 0.84],
        "detection_aupr": 0.80,
        "precision_at_recall_90": 0.63,
        "detection_accuracy": 0.715,
        "spearman_rho": 0.52,
        "claim_precision": 0.41,
        "claim_recall": 0.64,
        "claim_f1": 0.64,
    }
    out = build_replacements(agg)
    assert out["ROC-AUC"] == "0.789 [0.730, 0.840]"
    assert out["AUPR"] == "0.800"
    assert out["P at R=90"] == "0.630"
    assert out["Sentence Accuracy"] == "0.715"
    assert out["Spearman rho"] == "0.520"
    assert out["F1"] == "0.640"
    assert out["P/R"] == "0.410 / 0.640"


def test_build_replacements_skips_metrics_that_are_absent():
    out = build_replacements({"detection_roc_auc": 0.79})
    assert "ROC-AUC" in out
    assert "AUPR" not in out


def test_load_aggregate_rebuilds_from_rows_when_absent(tmp_path):
    """A file written by a run still in progress has no aggregate block."""
    from tests.test_evaluation_aggregate import _rows

    path = _metrics_file(tmp_path, {}, rows=_rows(30))
    agg = _load_aggregate(path)
    assert agg["detection_roc_auc"] == pytest.approx(1.0)


def test_load_aggregate_prefers_the_existing_block(tmp_path):
    path = _metrics_file(tmp_path, {"detection_roc_auc": 0.42})
    agg = _load_aggregate(path)
    assert agg["detection_roc_auc"] == 0.42


def test_fill_rewrites_the_projected_cells(tmp_path, monkeypatch):
    csv_path = tmp_path / "comparison_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Metric", "Ours (Hallucination-Mitigation)", "Other"])
        w.writerow(["Detection", "ROC-AUC", "0.91-0.95 (projected)", "0.987"])
        w.writerow(["Detection", "AUPR", "0.85-0.90 (projected)", "-"])

    metrics = _metrics_file(tmp_path, {
        "detection_roc_auc": 0.789,
        "detection_roc_auc_ci95": [0.73, 0.84],
        "detection_aupr": 0.80,
    })

    rc = fill_main([
        "--csv", str(csv_path),
        "--metrics", str(metrics),
        "--paper-dir", str(_paper_dir(tmp_path)),
    ])
    assert rc == 0

    rows = list(csv.reader(csv_path.open("r", encoding="utf-8", newline="")))
    ours = {r[1]: r[2] for r in rows[1:]}
    assert ours["ROC-AUC"] == "0.789 [0.730, 0.840]"
    assert ours["AUPR"] == "0.800"
    # No cell may still claim to be a projection.
    assert not any("projected" in v.lower() for v in ours.values())


def test_fill_cannot_write_to_the_real_paper(tmp_path):
    """A run with temporary metrics must not touch the real paper.

    The prose paths used to be hard-coded, so any test that called
    ``main()`` with a temporary metrics file wrote *that file's* numbers
    into ``Paper/``. In practice that meant a test fixture's ROC-AUC of
    0.789 silently overwrote the measured 0.843 in the working tree, and
    the paper reverted every time the suite ran.

    This test pins the guarantee: the prose files are written relative to
    ``--paper-dir``, so pointing it at a temporary directory keeps every
    write inside that directory.
    """
    csv_path = tmp_path / "comparison_table.csv"
    csv_path.write_text(
        "Category,Metric,Ours (Hallucination-Mitigation),Other\n"
        "Detection,ROC-AUC,\"0.91-0.95 (projected)\",0.987\n",
        encoding="utf-8",
    )
    paper = _paper_dir(tmp_path)
    # The fill functions only rewrite a file that already exists, so the
    # temporary paper directory needs a copy of the row under test.
    (paper / "why_we_win.md").write_text(
        "| 7 | **Detection ROC-AUC** | Dale 2023 — ALTI+ | "
        "0.987 (MT-only) | **0.843** | ≈ | below |\n",
        encoding="utf-8",
    )

    metrics = _metrics_file(tmp_path, {
        "detection_roc_auc": 0.111,
        "detection_roc_auc_ci95": [0.10, 0.12],
    })

    rc = fill_main([
        "--csv", str(csv_path),
        "--metrics", str(metrics),
        "--paper-dir", str(paper),
    ])
    assert rc == 0

    # The temporary copy was refreshed with the fixture's number.
    assert "0.111" in (paper / "why_we_win.md").read_text(encoding="utf-8")

    # Every file the run wrote lives under the temporary paper directory.
    written = {p.resolve() for p in paper.rglob("*") if p.is_file()}
    assert written, "the run wrote nothing into the temporary paper dir"
    for path in written:
        assert paper.resolve() in path.parents, (
            f"{path} was written outside the temporary paper directory"
        )

    # And the real paper still holds the measured value.
    real = Path("Paper/why_we_win.md")
    if real.exists():
        text = real.read_text(encoding="utf-8")
        assert "0.111" not in text, (
            "the fixture's ROC-AUC reached the real paper; the prose paths "
            "are still hard-coded somewhere"
        )


def test_fill_dry_run_leaves_the_file_untouched(tmp_path, monkeypatch):
    csv_path = tmp_path / "comparison_table.csv"
    original = (
        "Category,Metric,Ours (Hallucination-Mitigation),Other\n"
        "Detection,ROC-AUC,\"0.91-0.95 (projected)\",0.987\n"
    )
    csv_path.write_text(original, encoding="utf-8")

    metrics = _metrics_file(tmp_path, {"detection_roc_auc": 0.789})
    rc = fill_main([
        "--csv", str(csv_path),
        "--metrics", str(metrics),
        "--paper-dir", str(_paper_dir(tmp_path)),
        "--dry-run",
    ])
    assert rc == 0
    assert csv_path.read_text(encoding="utf-8") == original


def test_fill_reports_an_error_when_there_are_no_metrics(tmp_path, capsys):
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    csv_path = tmp_path / "comparison_table.csv"
    csv_path.write_text("Category,Metric,Ours\n", encoding="utf-8")

    rc = fill_main([
        "--csv", str(csv_path),
        "--metrics", str(metrics),
        "--paper-dir", str(_paper_dir(tmp_path)),
    ])
    assert rc == 1


def test_plot_results_refuses_a_projected_cell():
    """A projection must never be read as a measurement by the plotter."""
    from scripts.plot_results import plot_literature_comparison  # noqa: F401

    # The number parser is local to the plotting function, so exercise the
    # same rule directly: a cell containing "projected" yields no number.
    import re

    def num(v):
        text = str(v or "")
        if "projected" in text.lower():
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None

    assert num("0.91-0.95 (projected)") is None
    assert num("0.789 [0.730, 0.840]") == pytest.approx(0.789)


def test_fill_comparison_tex_replaces_the_projected_rows(tmp_path):
    """The LaTeX summary carries the same estimates and must be regenerated."""
    from scripts.fill_comparison_table import _fill_comparison_tex

    tex = tmp_path / "comparison_summary.tex"
    tex.write_text(
        _TEX_DETECTION_ROW
        + "\n"
        + "\\caption{Headline performance vs.\\ state-of-the-art. Numbers are projected\n"
        + "from the implemented pipeline's synthetic + HaluEval runs.}\n",
        encoding="utf-8",
    )

    agg = {
        "detection_roc_auc": 0.78,
        "detection_aupr": 0.79,
        "precision_at_recall_90": 0.63,
    }
    changes = _fill_comparison_tex(tex, agg)
    assert changes, "expected the projected rows to be replaced"

    out = tex.read_text(encoding="utf-8")
    assert "0.91--0.95" not in out
    assert "0.780" in out
    assert "projected" not in out


def test_fill_comparison_tex_leaves_rows_alone_when_a_metric_is_missing(tmp_path):
    """A missing metric must not be substituted with a guess."""
    from scripts.fill_comparison_table import _fill_comparison_tex

    tex = tmp_path / "comparison_summary.tex"
    original = _TEX_DETECTION_ROW + "\n"
    tex.write_text(original, encoding="utf-8")

    # Only ROC-AUC is measured; AUPR and P@R=90 are not.
    changes = _fill_comparison_tex(tex, {"detection_roc_auc": 0.78})
    assert changes == []
    assert tex.read_text(encoding="utf-8") == original


def test_fill_comparison_tex_dry_run_does_not_write(tmp_path):
    from scripts.fill_comparison_table import _fill_comparison_tex

    tex = tmp_path / "comparison_summary.tex"
    original = _TEX_DETECTION_ROW + "\n"
    tex.write_text(original, encoding="utf-8")

    _fill_comparison_tex(
        tex,
        {
            "detection_roc_auc": 0.78,
            "detection_aupr": 0.79,
            "precision_at_recall_90": 0.63,
        },
        dry_run=True,
    )
    assert tex.read_text(encoding="utf-8") == original


def test_fill_comparison_md_leaves_rows_alone_when_a_metric_is_missing(tmp_path):
    from scripts.fill_comparison_table import _fill_comparison_md

    md = tmp_path / "comparison.md"
    original = (
        "| **Ours (RAG fused score, Step 9)** | **0.91–0.95** "
        "(projected from synthetic) | **0.85–0.90** | **0.55–0.70** | "
        "Black-box + retrieval | HaluEval QA |\n"
    )
    md.write_text(original, encoding="utf-8")

    changes = _fill_comparison_md(md, {"detection_roc_auc": 0.78})
    assert changes == []
    assert md.read_text(encoding="utf-8") == original


# -- mitigation rows


def _mitigation_file(tmp_path: Path, summary: dict) -> Path:
    path = tmp_path / "mitigation.json"
    path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    return path


def test_load_mitigation_returns_the_summary(tmp_path):
    summary = {
        "n_rows": 10,
        "n_regenerated": 4,
        "regenerated_rows": {"delta_bertscore": 0.043},
    }
    path = _mitigation_file(tmp_path, summary)
    assert load_mitigation(path) == summary


def test_load_mitigation_returns_empty_when_the_file_is_missing(tmp_path):
    assert load_mitigation(tmp_path / "nope.json") == {}


def test_mitigation_replacements_use_the_regenerated_rows():
    """The delta must come from the rows step 10 actually rewrote."""
    mit = {
        "n_regenerated": 12,
        "regenerated_rows": {
            "n": 12,
            "pre_bertscore": 0.51,
            "post_bertscore": 0.553,
            "delta_bertscore": 0.043,
            "delta_bertscore_ci95": [0.012, 0.074],
        },
        "kept_rows": {
            "n": 30,
            "delta_bertscore": 0.0,
        },
    }
    out = build_mitigation_replacements(mit)
    assert out["Delta"].startswith("+0.043 [+0.012, +0.074]")
    assert "n=12" in out["Delta"]
    assert out["Pre-mitigation Score"] == "0.510 (BERTScore F1)"
    assert out["Post-mitigation Score"] == "0.553 (BERTScore F1)"


def test_mitigation_replacements_are_empty_without_a_measurement():
    assert build_mitigation_replacements({}) == {}
    assert build_mitigation_replacements({"regenerated_rows": {}}) == {}


def test_a_negative_delta_is_not_presented_as_a_win():
    """A negative delta must not keep the 'highest' claim in the cell."""
    mit = {
        "n_regenerated": 5,
        "regenerated_rows": {
            "n": 5,
            "pre_bertscore": 0.6,
            "post_bertscore": 0.55,
            "delta_bertscore": -0.05,
        },
    }
    out = build_mitigation_replacements(mit)
    assert out["Delta"].startswith("-0.050")
    assert "not positive" in out["Delta"]


def test_a_positive_delta_whose_interval_covers_zero_is_not_a_win():
    """A positive point estimate is not evidence of an improvement.

    The point estimate can land above zero by chance. Only an interval
    that sits entirely above zero separates the rewrite from no effect, so
    a delta whose CI covers zero must not keep the "highest" claim.
    """
    mit = {
        "n_regenerated": 20,
        "regenerated_rows": {
            "n": 20,
            "pre_bertscore": 0.60,
            "post_bertscore": 0.62,
            "delta_bertscore": 0.02,
            "delta_bertscore_ci95": [-0.01, 0.05],
        },
    }
    out = build_mitigation_replacements(mit)
    assert out["Delta"].startswith("+0.020")
    assert "not positive" in out["Delta"], (
        f"a CI covering zero was presented as a win: {out['Delta']!r}"
    )


def test_a_positive_delta_with_an_interval_above_zero_is_a_win():
    """The counterpart: a CI that clears zero *is* a win."""
    mit = {
        "n_regenerated": 20,
        "regenerated_rows": {
            "n": 20,
            "pre_bertscore": 0.60,
            "post_bertscore": 0.66,
            "delta_bertscore": 0.06,
            "delta_bertscore_ci95": [0.02, 0.10],
        },
    }
    out = build_mitigation_replacements(mit)
    assert out["Delta"].startswith("+0.060")
    assert "not positive" not in out["Delta"]


def test_a_positive_delta_without_an_interval_is_not_a_win():
    """Without an interval there is no evidence of an effect, so the
    conservative reading applies."""
    mit = {
        "n_regenerated": 20,
        "regenerated_rows": {
            "n": 20,
            "pre_bertscore": 0.60,
            "post_bertscore": 0.66,
            "delta_bertscore": 0.06,
        },
    }
    out = build_mitigation_replacements(mit)
    assert "not positive" in out["Delta"]


def test_the_five_position_prose_sentence_tracks_the_measurement(tmp_path, monkeypatch):
    """The sentence that states the mitigation result must not drift away
    from the number it describes.

    It was hand-written, so a new evaluation left it holding the previous
    run's delta while the table beside it was refreshed.
    """
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "five_position_comparison.md"
    paper.write_text(
        "*Positions 2 and 5 are not wins and are reported as measured. "
        "The mitigation \u0394 is -0.035 BERTScore F1 (95% CI [-0.067, "
        "-0.006]) on the 18 of 60 rows where step 10 regenerated, and the "
        "HRR / VSR / CCR rates are below the cited comparables. The rewrite "
        "path is implemented and exercised, but on this sample it did not "
        "improve similarity to the gold answer; improving it is future "
        "work. The defensible claims are the architectural ones \u2014 "
        "Positions 1, 3 and 4.\n",
        encoding="utf-8",
    )
    mit = tmp_path / "mit.json"
    mit.write_text(
        json.dumps({
            "summary": {
                "n_rows": 1000,
                "n_regenerated": 214,
                "regenerated_rows": {
                    "n": 214,
                    "pre_bertscore": 0.80,
                    "post_bertscore": 0.81,
                    "delta_bertscore": 0.012,
                    "delta_bertscore_ci95": [-0.01, 0.034],
                },
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_five_position_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "-0.035" not in out, "the stale delta survived the fill"
    assert "+0.012" in out
    assert "[-0.010, +0.034]" in out
    assert "214 of 1000 rows" in out


def test_the_readme_mitigation_paragraph_tracks_the_measurement(tmp_path, monkeypatch):
    """The README states the mitigation result in prose, so it must not be
    hand-written — otherwise a new evaluation leaves it stale.

    The paragraph is wrapped across lines, so the match has to tolerate
    newlines.
    """
    import scripts.fill_comparison_table as fill

    readme = tmp_path / "README.md"
    readme.write_text(
        "**The measured mitigation \u0394 is negative.** On the 18 rows where "
        "step 10\nactually regenerated (out of 60 evaluated on the "
        "length-matched HaluEval\nsplit), BERTScore F1 moved from 0.863 to "
        "0.828: a \u0394 of **\u22120.035** (95% CI\n[\u22120.067, "
        "\u22120.006]). The rewrite therefore did *not* improve similarity "
        "to\nthe gold answer on this sample. Every prose table, chart and "
        "verdict in\n`Paper/` reports that number rather than the \"+0.67 to "
        "+0.75\" an earlier\ndraft projected, and the win/tie/loss verdict "
        "for mitigation is a loss.\nImproving the rewrite path is future "
        "work; the negative result is reported\nas a finding.\n",
        encoding="utf-8",
    )
    mit = tmp_path / "mit.json"
    mit.write_text(
        json.dumps({
            "summary": {
                "n_rows": 1000,
                "n_regenerated": 214,
                "regenerated_rows": {
                    "n": 214,
                    "pre_bertscore": 0.800,
                    "post_bertscore": 0.812,
                    "delta_bertscore": 0.012,
                    "delta_bertscore_ci95": [-0.010, 0.034],
                },
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_readme_md(readme, {}, dry_run=False)

    out = readme.read_text(encoding="utf-8")
    assert "\u22120.035" not in out, "the stale delta survived the fill"
    assert "+0.012" in out
    assert "[-0.010, +0.034]" in out
    assert "214 rows" in out
    # The projected figure must never come back.
    assert "0.67 to +0.75" not in out.split("rather than the")[-1].split(".")[0] \
        or "an earlier" in out


def test_the_mitigation_flag_is_honoured_by_the_prose(tmp_path, monkeypatch):
    """``--mitigation`` must select the measurement the prose reports.

    The prose read the module-level path instead of the one the command
    line was given, so the table and the sentence beside it could report
    two different measurements. A test that passed a temporary file caught
    nothing because it only checked the table.
    """
    paper = tmp_path / "Paper"
    paper.mkdir()
    for name in ("comparison.md", "comparison_summary.tex",
                 "five_position_comparison.md", "why_we_win.md",
                 "why_we_win_summary.tex"):
        shutil.copy(Path("Paper") / name, paper / name)
    shutil.copy("Paper/comparison_table.csv", paper / "comparison_table.csv")
    readme = tmp_path / "README.md"
    shutil.copy("README.md", readme)

    mit = tmp_path / "mit.json"
    mit.write_text(
        json.dumps({
            "summary": {
                "n_rows": 1000,
                "n_regenerated": 214,
                "regenerated_rows": {
                    "n": 214,
                    "pre_bertscore": 0.812,
                    "post_bertscore": 0.819,
                    "delta_bertscore": 0.007,
                    "delta_bertscore_ci95": [-0.009, 0.023],
                },
                "kept_rows": {"n": 786, "delta_bertscore": 0.0},
            }
        }),
        encoding="utf-8",
    )

    rc = fill_main([
        "--csv", str(paper / "comparison_table.csv"),
        "--metrics", "data/metrics_halueval.json",
        "--paper-dir", str(paper),
        "--mitigation", str(mit),
        "--readme", str(readme),
    ])
    assert rc == 0

    text = readme.read_text(encoding="utf-8")
    assert "+0.007" in text, (
        "the README did not report the measurement the flag selected"
    )
    assert "214 rows" in text

    fp = (paper / "five_position_comparison.md").read_text(encoding="utf-8")
    assert "+0.007" in fp, (
        "the five-position prose did not report the selected measurement"
    )


def test_fill_rewrites_the_mitigation_cells(tmp_path, monkeypatch):
    csv_path = tmp_path / "comparison_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Metric", "Ours (Hallucination-Mitigation)", "Other"])
        w.writerow(["Mitigation", "Pre-mitigation Score", "-0.63 (COMET)", "-0.63"])
        w.writerow(["Mitigation", "Post-mitigation Score", "+0.04 to +0.12 (BERTScore F1)", "-0.07"])
        w.writerow(["Mitigation", "Delta", "+0.67 to +0.75 (HIGHEST)", "+0.60"])

    metrics = _metrics_file(tmp_path, {"detection_roc_auc": 0.79})
    mitigation = _mitigation_file(tmp_path, {
        "n_regenerated": 7,
        "regenerated_rows": {
            "n": 7,
            "pre_bertscore": 0.50,
            "post_bertscore": 0.57,
            "delta_bertscore": 0.07,
            "delta_bertscore_ci95": [0.02, 0.12],
        },
    })

    rc = fill_main([
        "--csv", str(csv_path),
        "--metrics", str(metrics),
        "--paper-dir", str(_paper_dir(tmp_path)),
        "--mitigation", str(mitigation),
    ])
    assert rc == 0

    rows = list(csv.reader(csv_path.open("r", encoding="utf-8", newline="")))
    ours = {r[1]: r[2] for r in rows[1:]}
    assert ours["Pre-mitigation Score"] == "0.500 (BERTScore F1)"
    assert ours["Post-mitigation Score"] == "0.570 (BERTScore F1)"
    assert ours["Delta"].startswith("+0.070")
    # The unbacked claims must be gone.
    assert "0.67" not in ours["Delta"]
    assert "HIGHEST" not in ours["Delta"]


def test_mitigation_cells_are_left_alone_without_a_measurement(tmp_path, monkeypatch):
    csv_path = tmp_path / "comparison_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Metric", "Ours (Hallucination-Mitigation)", "Other"])
        w.writerow(["Mitigation", "Delta", "+0.67 to +0.75 (HIGHEST)", "+0.60"])

    metrics = _metrics_file(tmp_path, {"detection_roc_auc": 0.79})
    missing = tmp_path / "no_mitigation.json"

    rc = fill_main([
        "--csv", str(csv_path),
        "--metrics", str(metrics),
        "--paper-dir", str(_paper_dir(tmp_path)),
        "--mitigation", str(missing),
    ])
    assert rc == 0

    rows = list(csv.reader(csv_path.open("r", encoding="utf-8", newline="")))
    ours = {r[1]: r[2] for r in rows[1:]}
    assert ours["Delta"] == "+0.67 to +0.75 (HIGHEST)"


# -- refreshing a prose row


def test_prose_row_is_refreshed_when_the_number_changes(tmp_path):
    """A prose row must follow the metrics, not freeze at its first value.

    An earlier version anchored each replacement on the exact old string,
    so the row could only be filled once: a second run with new metrics
    found no match and left the previous number in place.
    """
    from scripts.fill_comparison_table import _fill_comparison_md

    md = tmp_path / "comparison.md"
    md.write_text(
        "| **Ours (RAG fused score, Step 9)** | **0.802** | **0.813** | "
        "**0.652** | Black-box + retrieval | HaluEval QA (length-matched) |\n",
        encoding="utf-8",
    )

    _fill_comparison_md(md, {
        "detection_roc_auc": 0.843,
        "detection_aupr": 0.853,
        "precision_at_recall_90": 0.658,
    })
    after_first = md.read_text(encoding="utf-8")
    assert "0.843" in after_first
    assert "0.802" not in after_first

    # A second run with different metrics must update the row again.
    _fill_comparison_md(md, {
        "detection_roc_auc": 0.901,
        "detection_aupr": 0.911,
        "precision_at_recall_90": 0.700,
    })
    after_second = md.read_text(encoding="utf-8")
    assert "0.901" in after_second
    assert "0.843" not in after_second


def test_prose_fill_is_idempotent(tmp_path):
    """Re-filling with the same metrics must not change the file again."""
    from scripts.fill_comparison_table import _fill_comparison_md

    md = tmp_path / "comparison.md"
    md.write_text(
        "| **Ours (RAG fused score, Step 9)** | **0.802** | **0.813** | "
        "**0.652** | Black-box + retrieval | HaluEval QA (length-matched) |\n",
        encoding="utf-8",
    )

    agg = {
        "detection_roc_auc": 0.843,
        "detection_aupr": 0.853,
        "precision_at_recall_90": 0.658,
    }
    _fill_comparison_md(md, agg)
    first = md.read_text(encoding="utf-8")

    changes = _fill_comparison_md(md, agg)
    assert changes == []
    assert md.read_text(encoding="utf-8") == first


def test_prose_rows_are_refreshed_even_when_the_csv_is_current(tmp_path):
    """The prose files are refreshed even if the CSV needs no change.

    ``main`` used to return as soon as the CSV matched the metrics, so a
    prose row that had drifted out of date was never corrected.
    """
    csv_path = tmp_path / "comparison_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Metric", "Ours (Hallucination-Mitigation)", "Other"])
        w.writerow(["Detection", "ROC-AUC", "0.789 [0.730, 0.840]", "0.987"])

    metrics = _metrics_file(tmp_path, {
        "detection_roc_auc": 0.789,
        "detection_roc_auc_ci95": [0.73, 0.84],
        "detection_aupr": 0.80,
    })

    prose = tmp_path / "comparison.md"
    prose.write_text(
        "| **Ours (RAG fused score, Step 9)** | **0.802** | **0.813** | "
        "**0.652** | Black-box + retrieval | HaluEval QA (length-matched) |\n",
        encoding="utf-8",
    )

    rc = fill_main([
        "--csv", str(csv_path),
        "--metrics", str(metrics),
        "--paper-dir", str(_paper_dir(tmp_path)),
    ])
    assert rc == 0

    # The prose row is written to Paper/comparison.md, not the tmp_path
    # copy, so assert on the change count instead: the CSV was already
    # current, yet the run must not have bailed out early.
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8", newline="")))
    assert rows[1][2] == "0.789 [0.730, 0.840]"


def test_five_position_file_is_refreshed(tmp_path):
    """The five-position write-up was never wired to the generator."""
    from scripts.fill_comparison_table import _fill_five_position_md

    md = tmp_path / "five_position_comparison.md"
    md.write_text(
        "| **Ours (Step 11 gate)** | **0.857** (1 \u2212 ABSTAIN-failure rate) |\n"
        "| **Ours (per-claim NLI)** | **0.85+** |\n"
        "| **Ours (Step 10 regen)** | **0.917** |\n",
        encoding="utf-8",
    )

    _fill_five_position_md(md, {
        "hallucination_reduction_rate": 0.172,
        "verification_success_rate": 0.608,
        "claim_correction_rate": 0.616,
    })

    out = md.read_text(encoding="utf-8")
    assert "0.172" in out
    assert "0.608" in out
    assert "0.616" in out
    assert "0.857" not in out
    assert "0.85+" not in out
    assert "0.917" not in out



# ---------------------------------------------------------------------------
# Verdict cells
#
# The prose tables carry a win/tie/loss verdict next to every number. A
# verdict that is not derived from the measurement is exactly the kind of
# claim the generator exists to prevent, so the verdicts are computed from
# the same numbers the cells quote.
# ---------------------------------------------------------------------------


def test_above_is_false_when_the_value_is_below_the_comparable():
    from scripts.fill_comparison_table import _above

    values = {"detection_accuracy": "0.745"}
    assert not _above(lambda k: values[k], "detection_accuracy", 0.847)


def test_above_flips_when_the_measurement_improves():
    from scripts.fill_comparison_table import _above

    values = {"detection_accuracy": "0.900"}
    assert _above(lambda k: values[k], "detection_accuracy", 0.847)


def test_above_is_false_for_a_missing_measurement():
    from scripts.fill_comparison_table import _above

    assert not _above(lambda k: (_ for _ in ()).throw(KeyError(k)),
                      "detection_accuracy", 0.847)


def test_delta_positive_only_for_a_positive_measurement():
    from scripts.fill_comparison_table import _delta_positive

    positive = {"delta": "+0.035 [0.006, 0.067] (n=18 regenerated)"}
    negative = {"delta": "-0.035 [-0.067, -0.006] (n=18 regenerated)"}
    assert _delta_positive(lambda k: positive[k])
    assert not _delta_positive(lambda k: negative[k])


def test_delta_positive_is_false_for_a_missing_measurement():
    from scripts.fill_comparison_table import _delta_positive

    assert not _delta_positive(lambda k: (_ for _ in ()).throw(KeyError(k)))


def test_why_we_win_verdict_rows_are_rewritten(tmp_path):
    """The win/tie/loss cells are recomputed, not left hand-written."""
    from scripts.fill_comparison_table import _fill_why_we_win_md

    md = tmp_path / "why_we_win.md"
    md.write_text(
        "| 2 | **Claim-level F1** | TU Wien 2024 \u2014 Vectara | 0.71 | "
        "**0.76** | \u2713 | Above the cited comparable |\n"
        "| 3 | **Hallucination Reduction Rate (HRR)** | Dale 2023 "
        "(N-best rerank) | ~0.55 (implied) | **0.71** | \u2713 | "
        "Above the implied baseline |\n"
        "| 4 | **Verification Success Rate (VSR)** | AILS-NTUA (Voting) | "
        "0.80 (acc) | **0.85+** | \u2713 | Above the cited comparables |\n"
        "| 8 | **Sentence accuracy** | Halu-NLP (SHROOM winner) | 0.847 | "
        "**0.83** | \u2713 | Above the SHROOM winner |\n",
        encoding="utf-8",
    )

    _fill_why_we_win_md(md, {
        "claim_f1": 0.653,
        "hallucination_reduction_rate": 0.172,
        "verification_success_rate": 0.608,
        "detection_accuracy": 0.745,
    })

    out = md.read_text(encoding="utf-8")
    # The measured values are in place.
    assert "**0.653**" in out
    assert "**0.172**" in out
    assert "**0.608**" in out
    assert "**0.745**" in out
    # The stale values are gone.
    assert "0.76" not in out
    assert "0.85+" not in out
    # Every row that lost the comparison now carries a loss verdict.
    assert "Above the cited comparable |" not in out
    assert "Above the implied baseline |" not in out
    assert "Above the cited comparables |" not in out
    assert "Above the SHROOM winner |" not in out
    assert "Below the cited comparable" in out
    assert "Below the implied baseline" in out
    assert "Below the cited comparables" in out
    assert "Below the SHROOM winner" in out


def test_why_we_win_verdict_rows_are_idempotent(tmp_path):
    """A second pass must not append a second verdict column."""
    from scripts.fill_comparison_table import _fill_why_we_win_md

    md = tmp_path / "why_we_win.md"
    md.write_text(
        "| 2 | **Claim-level F1** | TU Wien 2024 \u2014 Vectara | 0.71 | "
        "**0.76** | \u2713 | Above the cited comparable |\n",
        encoding="utf-8",
    )

    agg = {"claim_f1": 0.653}
    _fill_why_we_win_md(md, agg)
    once = md.read_text(encoding="utf-8")
    _fill_why_we_win_md(md, agg)
    assert md.read_text(encoding="utf-8") == once


def test_why_we_win_tex_verdict_rows_are_rewritten(tmp_path):
    """The LaTeX verdict matrix is recomputed from the measurements."""
    from scripts.fill_comparison_table import _fill_why_we_win_tex

    tex = tmp_path / "why_we_win_summary.tex"
    tex.write_text(
        r"Claim-level F1               & 0.71 (TU Wien Vectara) & "
        r"\textbf{0.76} & \textbf{Win} & Per-claim NLI \\"
        "\n"
        r"Verification Success Rate    & 0.81 (Halu-NLP) & "
        r"\textbf{0.85+} & \textbf{Win} & Atomic granularity \\"
        "\n",
        encoding="utf-8",
    )

    _fill_why_we_win_tex(tex, {
        "claim_f1": 0.653,
        "verification_success_rate": 0.608,
    })

    out = tex.read_text(encoding="utf-8")
    assert "0.653" in out
    assert "0.608" in out
    assert "0.85+" not in out
    assert "Loss" in out
    # No stray column is left behind by the row rewrite.
    assert "\\ \\ \\ &" not in out


def test_verdict_tally_is_computed_from_the_rows():
    """The tally summarises the matrix, so it is computed from it."""
    from scripts.fill_comparison_table import _verdict_tally

    matrix = (
        "| 1 | **Mitigation** | x | y | z | \u2717 | lost |\n"
        "| 2 | **Claim F1** | x | y | z | \u2713 | won |\n"
        "| 3 | **HRR** | x | y | z | \u2248 | tied |\n"
        "| 4 | **VSR** | x | y | z | \u2717 | lost |\n"
    )
    tally = _verdict_tally(matrix)
    assert tally is not None
    assert "**\u2713 Win** | 1 | 2 |" in tally
    assert "**\u2248 Tie / At par** | 1 | 3 |" in tally
    assert "**\u2717 Loss** | 2 | 1, 4 |" in tally


def test_verdict_tally_is_none_without_a_matrix():
    from scripts.fill_comparison_table import _verdict_tally

    assert _verdict_tally("no table here\n") is None


def test_why_we_win_tally_is_refreshed(tmp_path):
    """A stale tally is rewritten to agree with the rows above it."""
    from scripts.fill_comparison_table import _fill_why_we_win_md

    md = tmp_path / "why_we_win.md"
    md.write_text(
        "| 2 | **Claim-level F1** | TU Wien 2024 \u2014 Vectara | 0.71 | "
        "**0.76** | \u2713 | Above the cited comparable |\n"
        "| 3 | **Hallucination Reduction Rate (HRR)** | Dale 2023 "
        "(N-best rerank) | ~0.55 (implied) | **0.71** | \u2713 | "
        "Above the implied baseline |\n"
        "\n"
        "| Outcome | Count | Dimensions |\n"
        "|---|---|---|\n"
        "| **\u2713 Win** | 6 | 1, 2, 3, 4, 5, 6 |\n"
        "| **\u2248 Tie / At par** | 5 | 7, 8, 9, 10, 11 |\n"
        "| **\u2717 Loss** | 6 | 12, 13, 14, 15, 16, 17 |\n",
        encoding="utf-8",
    )

    _fill_why_we_win_md(md, {
        "claim_f1": 0.653,
        "hallucination_reduction_rate": 0.172,
    })

    out = md.read_text(encoding="utf-8")
    assert "| **\u2713 Win** | 0 |  |" in out
    assert "| **\u2717 Loss** | 2 | 2, 3 |" in out


# ---------------------------------------------------------------------------
# The AUPR and CCR rows must not claim a win the interval does not support.
# ---------------------------------------------------------------------------

_PAPER_FILES = (
    Path("Paper/comparison.md"),
    Path("Paper/comparison_summary.tex"),
    Path("Paper/why_we_win.md"),
    Path("Paper/why_we_win_summary.tex"),
    Path("Paper/five_position_comparison.md"),
    Path("README.md"),
)


def test_the_aupr_row_verdict_follows_the_interval():
    """The AUPR verdict must follow the interval, not the point estimate.

    The row used to say "Above the general-purpose probes" on the strength
    of a point estimate whose own 95% CI covered the published range -- a
    claim of a lead that the interval did not support. The verdict is
    derived from the CI: it may only say "above ... 95% CI excludes it"
    when our lower bound clears the comparable, and it must say
    "overlaps" when the intervals intersect.

    The test does not hard-code which of the two applies. The measured
    AUPR moved from 0.853 (CI [0.833, 0.874], which covered the published
    0.84-0.85 range) to 0.922 (CI [0.906, 0.938], which clears it), and a
    test that pinned the old conclusion would have failed for the right
    number arriving.
    """
    md = Path("Paper/comparison.md")
    if not md.exists():
        pytest.skip("comparison.md not present")
    metrics = Path("data/metrics_halueval.json")
    if not metrics.exists():
        pytest.skip("no evaluation output to read the interval from")
    agg = json.loads(metrics.read_text(encoding="utf-8")).get("aggregate") or {}

    text = md.read_text(encoding="utf-8")
    row = next((l for l in text.splitlines()
                if l.startswith("| **Detection AUPR** |")), None)
    assert row is not None, "the AUPR row is missing from comparison.md"
    low = row.lower()

    ci = agg.get("detection_aupr_ci95")
    value = agg.get("detection_aupr")
    if not ci or len(ci) != 2 or value is None:
        pytest.skip("the run has no AUPR interval to check the row against")
    lo, hi = float(ci[0]), float(ci[1])
    comparable = 0.84

    if lo > comparable:
        # Separated: the row may claim the lead, and must say the interval
        # is what secures it.
        assert "above" in low and "excludes" in low, (
            f"our AUPR CI [{lo:.3f}, {hi:.3f}] clears {comparable}, so the "
            f"row should say so: {row}"
        )
    elif hi < comparable:
        assert "below" in low, (
            f"our AUPR CI [{lo:.3f}, {hi:.3f}] sits under {comparable}, so "
            f"the row should say so: {row}"
        )
    else:
        assert "tie" in low or "overlap" in low, (
            f"our AUPR CI [{lo:.3f}, {hi:.3f}] covers {comparable}, so the "
            f"row should report the overlap: {row}"
        )

    # A tie may never be dressed up as a lead, whatever the interval says.
    for overclaim in ("above the general-purpose probes", "leads the field"):
        assert overclaim not in low, (
            f"the AUPR row still claims a lead over the field: {row}"
        )


def test_the_ccr_row_says_there_is_no_published_comparable():
    """CCR's 0.55 baseline is implied, not published.

    The other cited systems have no mitigation stage at all, so their
    cells are "0 (no rewriting)". Comparing our measured CCR against a
    baseline nobody published is not a comparison, and the row must say so
    rather than claim to be above it.
    """
    md = Path("Paper/comparison.md")
    if not md.exists():
        pytest.skip("comparison.md not present")
    text = md.read_text(encoding="utf-8")
    row = next((l for l in text.splitlines() if l.startswith("| **CCR** |")),
               None)
    assert row is not None, "the CCR row is missing from comparison.md"
    low = row.lower()
    assert "no published comparable" in low or "none published" in low, (
        f"the CCR row does not say that no comparable is published: {row}"
    )
    assert "above the implied baseline" not in low, (
        f"the CCR row still claims to be above an implied baseline: {row}"
    )


def test_the_paper_does_not_claim_aupr_or_ccr_are_wins():
    """No file may state that AUPR or CCR is a win."""
    banned = (
        "aupr and ccr are the two metrics where we are ahead",
        "aupr and ccr are the two metrics where we are ahead",
    )
    for path in _PAPER_FILES:
        if not path.exists():
            continue
        low = path.read_text(encoding="utf-8").lower()
        for phrase in banned:
            assert phrase not in low, (
                f"{path} still asserts that AUPR and CCR are wins"
            )


def _write_mitigation(path: Path, delta: float, ci, n_regen: int = 16,
                      n_rows: int = 500) -> Path:
    """Write a mitigation measurement and return its path."""
    path.write_text(
        json.dumps({
            "summary": {
                "n_rows": n_rows,
                "n_regenerated": n_regen,
                "regenerated_rows": {
                    "n": n_regen,
                    "pre_bertscore": 0.842,
                    "post_bertscore": 0.835,
                    "delta_bertscore": delta,
                    "delta_bertscore_ci95": ci,
                },
            }
        }),
        encoding="utf-8",
    )
    return path


def test_the_why_we_win_qa_prose_tracks_the_measurement(tmp_path, monkeypatch):
    """The two Q&A paragraphs that quote the delta must not drift.

    They were hand-written, so a new evaluation left them holding the
    previous run's delta while the table beside them was refreshed.
    """
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "why_we_win.md"
    paper.write_text(
        "2. **\"How is your work different from Dale 2023?\"** — Dale reports "
        "+0.49 to +0.60 on machine translation; our measured \u0394 is -0.035 "
        "on the length-matched HaluEval split, so we do not claim to lead on "
        "mitigation \u0394.\n\n"
        "5. **\"What's the strongest result?\"** — Detection ROC-AUC of "
        "**0.843**. The mitigation \u0394 is *not* a win: measured on the rows "
        "where step 10 regenerated, it is -0.035 (CI [-0.067, -0.006]), so it "
        "is reported as an open result.\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_why_we_win_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "-0.035" not in out, "the stale delta survived the fill"
    assert "-0.007" in out
    assert "[-0.022, +0.007]" in out


def test_the_comparison_observation_tracks_the_measurement(tmp_path, monkeypatch):
    """The observation paragraph in comparison.md quotes the delta, the
    interval and the number of regenerated rows."""
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "comparison.md"
    paper.write_text(
        "**Observation.** The pipeline is the only system that produces a "
        "**rewritten answer** rather than just a label. Measured on the rows "
        "where step 10 actually regenerated (n=18 of 60), the \u0394 over the "
        "no-mitigation baseline is **-0.035 BERTScore F1** (95% CI [-0.067, "
        "-0.006]) \u2014 i.e. regeneration did not improve similarity to the "
        "gold answer on this sample. The rewrite step is therefore not yet a "
        "demonstrated win over the no-mitigation baseline, and the \u0394 is "
        "reported here as measured rather than as the projected figure an "
        "earlier draft carried.\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_comparison_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "-0.035" not in out, "the stale delta survived the fill"
    assert "-0.007" in out
    assert "n=16 of 500" in out, (
        f"the regenerated-row count was not refreshed: {out!r}"
    )


def test_the_five_position_heading_and_prose_track_the_measurement(
    tmp_path, monkeypatch,
):
    """The Position 2 heading and its two prose blocks quote the delta."""
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "five_position_comparison.md"
    paper.write_text(
        "## Position 2 \u2014 Mitigation \u0394 (Measured: -0.035 vs. +0.60 Dale)\n\n"
        "**NOT A WIN \u2014 reported as measured.** On the length-matched "
        "HaluEval split, the mitigation \u0394 over the no-mitigation baseline "
        "is **-0.035 BERTScore F1** (95% CI [-0.067, -0.006], measured on the "
        "18 of 60 rows where step 10 actually regenerated). Dale 2023 reports "
        "+0.49 to +0.60 on machine translation, so on the one published "
        "mitigation \u0394 we do not lead. The step that rewrites "
        "contradicted claims is implemented and exercised, but on this sample "
        "it did not improve similarity to the gold answer; that is an open "
        "result, and it is reported here rather than asserted as a win.\n\n"
        "> **Paragraph 2 \u2014 Position 2 (\u0394 reported as measured).** "
        "*Our pipeline implements evidence-grounded regeneration and measures "
        "its effect: on the length-matched HaluEval split, the mitigation "
        "\u0394 over the no-mitigation baseline is -0.035 BERTScore F1 (95% CI "
        "[-0.067, -0.006]), measured on the 18 of 60 rows where step 10 "
        "actually regenerated.*\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_five_position_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "-0.035" not in out, "the stale delta survived the fill"
    assert "-0.007" in out
    assert "16 of 500" in out


def test_the_prose_says_the_result_is_not_separable_when_the_interval_covers_zero(
    tmp_path, monkeypatch,
):
    """A delta whose interval covers zero must not be described as a
    measured failure either: the measurement does not separate it from
    no effect at all."""
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "five_position_comparison.md"
    paper.write_text(
        "## Position 2 \u2014 Mitigation \u0394 (Measured: -0.035 vs. +0.60 Dale)\n\n"
        "**NOT A WIN \u2014 reported as measured.** On the length-matched "
        "HaluEval split, the mitigation \u0394 over the no-mitigation baseline "
        "is **-0.035 BERTScore F1** (95% CI [-0.067, -0.006], measured on the "
        "18 of 60 rows where step 10 actually regenerated). Dale 2023 reports "
        "+0.49 to +0.60 on machine translation, so on the one published "
        "mitigation \u0394 we do not lead. The step that rewrites "
        "contradicted claims is implemented and exercised, but on this sample "
        "it did not improve similarity to the gold answer; that is an open "
        "result, and it is reported here rather than asserted as a win.\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_five_position_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "not distinguishable from no effect" in out, (
        f"an interval covering zero was described as a measured failure: {out!r}"
    )
    assert "it did not improve similarity" not in out, (
        f"an interval covering zero was described as a measured failure: {out!r}"
    )


def test_no_paper_file_holds_a_stale_mitigation_delta():
    """No generated file may still carry the previous run's delta.

    The tokens below are the ones the 500-row run replaced. If any of
    them appears in a generated file, a surface was missed.
    """
    stale = ("-0.035", "[-0.067, -0.006]", "n=18 regenerated", "18 of 60")
    for path in _PAPER_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in stale:
            assert token not in text, (
                f"{path} still holds the stale mitigation figure {token!r}"
            )


# ---------------------------------------------------------------------------
# The generated surfaces must not drift away from the measurement.
# ---------------------------------------------------------------------------

# Every file the generator writes, including the defence-notes summary that
# was previously hand-written and froze at whatever numbers were current
# when it was last edited.
_DRIFT_SURFACES = (
    "comparison.md",
    "comparison_summary.tex",
    "why_we_win.md",
    "why_we_win_summary.tex",
    "five_position_comparison.md",
    "wins_summary.md",
    "comparison_detailed.md",
)


@pytest.mark.parametrize("filename", _DRIFT_SURFACES)
def test_the_paper_surfaces_match_the_current_measurement(filename, tmp_path):
    """Regenerating the surfaces must be a no-op on the committed files.

    The generator is the only thing that keeps the paper's numbers equal to
    the evaluation's numbers. This test runs it against the current metrics
    into a temporary directory and requires the result to match what is
    committed. A difference means the paper holds a number the measurement
    no longer supports -- the exact failure that let a superseded ROC-AUC
    of 0.843 and a claim-level F1 of 0.653 sit in the paper while the
    measurement said 0.9251 and 0.8107.

    The test is skipped when the metrics files are absent, because a fresh
    checkout has no evaluation output to regenerate from.
    """
    metrics = Path("data/metrics_halueval.json")
    if not metrics.exists():
        pytest.skip("no evaluation output to regenerate from")

    committed = Path("Paper") / filename
    if not committed.exists():
        pytest.skip(f"{committed} is not present")

    # A copy of the real Paper/ directory, so the generator starts from the
    # committed text and the diff isolates the measurement-driven changes.
    staged = tmp_path / "Paper"
    shutil.copytree("Paper", staged)

    # The generator reads the CSV back to fill the cells it derives from
    # it, so the staged run needs the committed CSV as its input.
    staged_csv = tmp_path / "comparison_table.csv"
    committed_csv = Path("Paper/comparison_table.csv")
    if committed_csv.exists():
        shutil.copy2(committed_csv, staged_csv)
    else:
        staged_csv.write_text("", encoding="utf-8")

    rc = fill_main([
        "--csv", str(staged_csv),
        "--metrics", str(metrics),
        "--mitigation", "data/mitigation_results.json",
        "--paper-dir", str(staged),
        "--readme", str(tmp_path / "README.md"),
    ])
    assert rc == 0

    regenerated = (staged / filename).read_text(encoding="utf-8")
    assert regenerated == committed.read_text(encoding="utf-8"), (
        f"Paper/{filename} does not match the current measurement. "
        "Run `python -m scripts.fill_comparison_table` and commit the result."
    )
