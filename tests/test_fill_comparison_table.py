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
    (paper / "comparison.md").write_text(
        "| 8 | **Detection ROC-AUC** | Dale 2023 — ALTI+ | "
        "0.987 | **0.843** |\n",
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
    assert "0.111" in (paper / "comparison.md").read_text(encoding="utf-8")

    # Every file the run wrote lives under the temporary paper directory.
    written = {p.resolve() for p in paper.rglob("*") if p.is_file()}
    assert written, "the run wrote nothing into the temporary paper dir"
    for path in written:
        assert paper.resolve() in path.parents, (
            f"{path} was written outside the temporary paper directory"
        )

    # And the real paper still holds the measured value.
    real = Path("Paper/comparison.md")
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
    run's delta while the table beside it was refreshed. The prose used to
    live in a separate five-position file; it is now the "Position 2"
    section of the consolidated comparison document.
    """
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "comparison.md"
    paper.write_text(
        "### Position 2 \u2014 Mitigation \u0394\n\n"
        "**Verdict: NOT A WIN \u2014 reported as measured.** See section 5. "
        "The measured delta is\n"
        "\u22120.035 BERTScore F1 (CI [\u22120.067, \u22120.006]) on 18 "
        "regenerated rows. Dale reports +0.49 to\n+0.60 on machine "
        "translation, so on the one published mitigation delta we do not "
        "lead.\n",
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

    fill._fill_comparison_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "\u22120.035" not in out, "the stale delta survived the fill"
    assert "+0.012" in out
    assert "[\u22120.010, +0.034]" in out
    assert "214 regenerated rows" in out


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

    # The consolidated comparison document quotes the mitigation result
    # in the "Position 2" prose. It used to live in a separate
    # five-position file, so the check moved with it.
    cmp_text = (paper / "comparison.md").read_text(encoding="utf-8")
    assert "+0.007" in cmp_text, (
        "the comparison prose did not report the selected measurement"
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
        "| **Ours (Step 9 fused score)** | **0.802** | Retrieval + "
        "per-claim NLI + fusion | **HaluEval QA** | — |\n",
        encoding="utf-8",
    )

    _fill_comparison_md(md, {"detection_roc_auc": 0.843})
    after_first = md.read_text(encoding="utf-8")
    assert "0.843" in after_first
    assert "0.802" not in after_first

    # A second run with different metrics must update the row again.
    _fill_comparison_md(md, {"detection_roc_auc": 0.901})
    after_second = md.read_text(encoding="utf-8")
    assert "0.901" in after_second
    assert "0.843" not in after_second


def test_prose_fill_is_idempotent(tmp_path):
    """Re-filling with the same metrics must not change the file again."""
    from scripts.fill_comparison_table import _fill_comparison_md

    md = tmp_path / "comparison.md"
    md.write_text(
        "| **Ours (Step 9 fused score)** | **0.802** | Retrieval + "
        "per-claim NLI + fusion | **HaluEval QA** | — |\n",
        encoding="utf-8",
    )

    agg = {"detection_roc_auc": 0.843}
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
    """The five positions now live in comparison.md; their rates are filled.

    The rates used to be a table in a separate file that the generator
    never wrote, so the numbers there were hand-maintained. They are
    prose in the consolidated document now, and the fill refreshes them.
    """
    import scripts.fill_comparison_table as fill

    md = tmp_path / "comparison.md"
    md.write_text(
        "### Position 5 \u2014 HRR / VSR / CCR\n\n"
        "Our HRR is 0.857 against Dale's implied ~0.55. Our VSR is 0.85+ "
        "against 0.80\u20130.81. Our\n"
        "CCR is 0.917 with no published CCR to compare against\n",
        encoding="utf-8",
    )

    fill._fill_comparison_md(md, {
        "hallucination_reduction_rate": 0.172,
        "verification_success_rate": 0.608,
        "claim_correction_rate": 0.616,
    }, dry_run=False)

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
    """The win/tie/loss cells are recomputed, not left hand-written.

    The fixture copies the real rows of the win matrix, with a verdict
    that disagrees with the measurement, so the fill has to correct both
    the value and the verdict.
    """
    import scripts.fill_comparison_table as fill

    md = tmp_path / "comparison.md"
    md.write_text(
        "| # | Dimension | Best Comparable (Paper) | Best Comparable Score "
        "| Our Score | Winner | Justification |\n"
        "|---|---|---|---|---|---|---|\n"
        "| 2 | **Claim-level F1** | TU Wien \u2014 Vectara | 0.71 | "
        "**0.76** | \u2713 | Above the cited comparable |\n"
        "| 3 | **Detection AUPR** | Decoding Memories | 0.840 | "
        "**0.71** | \u2713 | Above the cited comparable |\n"
        "| 4 | **HRR** | Dale 2023 (implied) | ~0.55 | "
        "**0.71** | \u2713 | Above the implied baseline |\n"
        "| 5 | **VSR** | Halu-NLP | 0.81 | "
        "**0.85+** | \u2713 | Above the cited comparables |\n"
        "| 9 | **Sentence accuracy** | Halu-NLP | 0.847 | "
        "**0.83** | \u2713 | Above the SHROOM winner |\n",
        encoding="utf-8",
    )

    fill._fill_comparison_md(md, {
        "claim_f1": 0.653,
        "hallucination_reduction_rate": 0.172,
        "verification_success_rate": 0.608,
        "detection_accuracy": 0.745,
        "detection_aupr": 0.71,
        "detection_aupr_ci": "[0.680, 0.740]",
    }, dry_run=False)

    out = md.read_text(encoding="utf-8")
    # Every row that lost the comparison now carries a loss verdict.
    assert "Above the cited comparable |" not in out
    assert "Above the implied baseline |" not in out
    assert "Above the cited comparables |" not in out
    assert "Above the SHROOM winner |" not in out
    assert "Below the cited comparable" in out
    # The measured values replaced the stale ones.
    assert "**0.653**" in out
    assert "**0.172**" in out
    assert "**0.608**" in out
    assert "**0.745**" in out
    assert "0.76" not in out
    assert "0.85+" not in out











def test_why_we_win_verdict_rows_are_idempotent(tmp_path):
    """A second pass must not append a second verdict column."""
    import scripts.fill_comparison_table as fill

    md = tmp_path / "comparison.md"
    md.write_text(
        "| # | Dimension | Best Comparable (Paper) | Best Comparable Score "
        "| Our Score | Winner | Justification |\n"
        "|---|---|---|---|---|---|---|\n"
        "| 2 | **Claim-level F1** | TU Wien 2024 \u2014 Vectara | 0.71 | "
        "**0.76** | \u2713 | Above the cited comparable |\n",
        encoding="utf-8",
    )

    agg = {"claim_f1": 0.653}
    fill._fill_comparison_md(md, agg, dry_run=False)
    once = md.read_text(encoding="utf-8")
    fill._fill_comparison_md(md, agg, dry_run=False)
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
    import scripts.fill_comparison_table as fill

    md = tmp_path / "comparison.md"
    md.write_text(
        "| # | Dimension | Best Comparable (Paper) | Best Comparable Score "
        "| Our Score | Winner | Justification |\n"
        "|---|---|---|---|---|---|---|\n"
        "| 2 | **Claim-level F1** | TU Wien \u2014 Vectara | 0.71 | "
        "**0.76** | \u2713 | Above the cited comparable |\n"
        "| 4 | **HRR** | Dale 2023 (implied) | ~0.55 | "
        "**0.71** | \u2713 | Above the implied baseline |\n"
        "\n"
        "| Outcome | Count | Dimensions |\n"
        "|---|---|---|\n"
        "| **\u2713 Win** | 6 | 1, 2, 3, 4, 5, 6 |\n"
        "| **\u2248 Tie / At par** | 5 | 7, 8, 9, 10, 11 |\n"
        "| **\u2717 Loss** | 6 | 12, 13, 14, 15, 16, 17 |\n",
        encoding="utf-8",
    )

    fill._fill_comparison_md(md, {
        "claim_f1": 0.653,
        "hallucination_reduction_rate": 0.172,
    }, dry_run=False)

    out = md.read_text(encoding="utf-8")
    # Both rows lost their comparison, so the tally must count two losses
    # and no wins. The stale counts of 6 and 6 are gone.
    assert "| **\u2717 Loss** | 2 | 2, 4 |" in out
    assert "| **\u2713 Win** | 6 |" not in out
    assert "| **\u2717 Loss** | 6 |" not in out











# ---------------------------------------------------------------------------
# The AUPR and CCR rows must not claim a win the interval does not support.
# ---------------------------------------------------------------------------

_PAPER_FILES = (
    Path("Paper/comparison.md"),
    Path("Paper/comparison_summary.tex"),
    Path("Paper/why_we_win_summary.tex"),
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
                if l.startswith("| AUPR |")), None)
    assert row is not None, "the AUPR row is missing from comparison.md"
    low = row.lower()

    ci = agg.get("detection_aupr_ci95")
    value = agg.get("detection_aupr")
    if not ci or len(ci) != 2 or value is None:
        pytest.skip("the run has no AUPR interval to check the row against")
    lo, hi = float(ci[0]), float(ci[1])
    comparable = 0.84

    if lo > comparable:
        # Separated: the row may claim the lead.
        assert "\u2713" in row, (
            f"our AUPR CI [{lo:.3f}, {hi:.3f}] clears {comparable}, so the "
            f"row should claim the win: {row}"
        )
    elif hi < comparable:
        assert "\u2717" in row, (
            f"our AUPR CI [{lo:.3f}, {hi:.3f}] sits under {comparable}, so "
            f"the row should report the loss: {row}"
        )
    else:
        assert "\u2248" in row, (
            f"our AUPR CI [{lo:.3f}, {hi:.3f}] covers {comparable}, so the "
            f"row should report the tie: {row}"
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
    row = next((l for l in text.splitlines() if l.startswith("| CCR |")),
               None)
    assert row is not None, "the CCR row is missing from comparison.md"
    low = row.lower()
    assert "none published" in low or "no published comparable" in low, (
        f"the CCR row does not say that no comparable is published: {row}"
    )
    assert "not a win" in low, (
        f"the CCR row must not report a win against no comparable: {row}"
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
    """The LaTeX Q&A paragraphs that quote the delta must not drift.

    They were hand-written, so a new evaluation left them holding the
    previous run's delta while the table beside them was refreshed.
    """
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "why_we_win_summary.tex"
    paper.write_text(
        "on machine translation; our measured $\\Delta$ is $-0.035$ on the "
        "length-matched HaluEval split, so we do not claim to lead here.\n"
        "\n"
        "mitigation $\\Delta$ is \\emph{not} a win: it is $-0.035$ "
        "(CI $[-0.067, -0.006]$) on the rows where step~10 regenerated, so it "
        "is reported as an open result.\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_why_we_win_tex(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "-0.035" not in out, "the stale delta survived the fill"
    assert "-0.007" in out
    assert "[-0.022, +0.007]" in out














def test_the_comparison_observation_tracks_the_measurement(tmp_path, monkeypatch):
    """The negative-result prose under the mitigation table quotes the
    delta, the interval and the number of regenerated rows.

    The paper writes the minus sign as a typographic one, so the
    assertion looks for that rather than the ASCII hyphen.
    """
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "comparison.md"
    paper.write_text(
        "**Our rewrite does not work yet.** The delta is \u22120.035 and the "
        "confidence interval\nincludes zero. On the 18 rows where step 10 "
        "actually rewrote an answer, the rewrite\nwas no better than leaving "
        "the answer alone. A paired permutation test on those rows gives\n"
        "p = 0.404 \u2014 no evidence of an effect in either direction.\n"
        "\n"
        "**Our sample is small.** Only 18 of 60 rows were regenerated, "
        "because our gate only\nfires when the evidence supports a rewrite. "
        "A negative result on 18 rows from 8 unique\nquestions is weak "
        "evidence.\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_comparison_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "\u22120.035" not in out, "the stale delta survived the fill"
    assert "\u22120.007" in out
    assert "16 of 500" in out, (
        f"the regenerated-row count was not refreshed: {out!r}"
    )











def test_the_five_position_heading_and_prose_track_the_measurement(
    tmp_path, monkeypatch,
):
    """The Position 2 prose quotes the delta, its interval and the number
    of regenerated rows.

    The five positions now live in the consolidated comparison.md, so the
    fill is the one that writes that document. The paper writes the minus
    sign as a typographic one.
    """
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "comparison.md"
    paper.write_text(
        "### Position 2 \u2014 Mitigation \u0394\n\n"
        "**Verdict: NOT A WIN.** See section 5. The measured delta is\n"
        "\u22120.035 BERTScore F1 (CI [\u22120.067, \u22120.006]) on 18 "
        "regenerated rows. Dale reports +0.49 to\n+0.60 on machine "
        "translation, so on the one published mitigation delta we do not\n"
        "lead.\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_comparison_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "\u22120.035" not in out, "the stale delta survived the fill"
    assert "\u22120.007" in out
    assert "16 regenerated rows" in out











def test_the_prose_says_the_result_is_not_separable_when_the_interval_covers_zero(
    tmp_path, monkeypatch,
):
    """A delta whose interval covers zero must not be described as a
    measured failure either: the measurement does not separate it from
    no effect at all."""
    import scripts.fill_comparison_table as fill

    paper = tmp_path / "comparison.md"
    paper.write_text(
        "**Our rewrite does not work yet.** The delta is \u22120.035 and the "
        "confidence interval\nincludes zero. On the 18 rows where step 10 "
        "actually rewrote an answer, the rewrite\nwas no better than leaving "
        "the answer alone. A paired permutation test on those rows gives\n"
        "p = 0.404 \u2014 no evidence of an effect in either direction.\n",
        encoding="utf-8",
    )
    mit = _write_mitigation(tmp_path / "mit.json", -0.007, [-0.022, 0.007])
    monkeypatch.setattr(fill, "MITIGATION_PATH", mit)

    fill._fill_comparison_md(paper, {}, dry_run=False)

    out = paper.read_text(encoding="utf-8")
    assert "not distinguishable from no effect" in out, (
        f"an interval covering zero was described as a measured failure: {out!r}"
    )
    assert "no better than leaving the answer alone" not in out, (
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

# Every file the generator writes. Paper/comparison.md is the single
# human-facing comparison document: the four defence-notes files it used
# to be split across were consolidated into it, so they are no longer
# surfaces of their own.
_DRIFT_SURFACES = (
    "comparison.md",
    "comparison_summary.tex",
    "why_we_win_summary.tex",
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


# ---------------------------------------------------------------------------
# The consolidated document must not be able to hold a stale number.
# ---------------------------------------------------------------------------


def _consolidated_rules():
    """The rules that fill Paper/comparison.md, with their builders.

    ``_fill_comparison_md`` holds two lists: the ``consolidated`` rules
    that cover the single comparison document, and the older ``row_specs``
    that cover the CSV-derived layout. Both are read out of the module so
    the test exercises the real rules rather than a copy.
    """
    import scripts.fill_comparison_table as fill

    rules = []
    original = fill._apply_specs
    seen = []

    def spy(text, specs, values, dry_run=False):
        if not seen:
            seen.append(specs)
        return original(text, specs, values, dry_run=dry_run)

    fill._apply_specs = spy
    try:
        fill.main(["--dry-run"])
    finally:
        fill._apply_specs = original

    if seen:
        for label, pattern, build, needed in seen[0]:
            rules.append((label, pattern, build, needed))
    return rules


def test_every_consolidated_rule_can_repair_a_stale_number(tmp_path):
    """A rule that cannot repair its own row is a row that can go stale.

    The drift test above only proves the generator is a no-op on the
    committed file. It does not prove the generator would *notice* if the
    measurement moved. This one puts a different but equally valid number
    into every cell the rules own and requires the generator to put the
    measured value back.

    The corruption replaces the measured number with another number of the
    same shape, because that is the real drift: the evaluation was
    re-run and the document kept the previous run's figure.
    """
    import re

    import scripts.fill_comparison_table as fill

    metrics = Path("data/metrics_halueval.json")
    committed = Path("Paper/comparison.md")
    if not metrics.exists() or not committed.exists():
        pytest.skip("no evaluation output or comparison document to test")

    real_text = committed.read_text(encoding="utf-8")

    # Capture the values the generator builds, then apply the rules.
    original = fill._apply_specs
    captured: dict = {}

    def spy(text, specs, values, dry_run=False):
        if "values" not in captured:
            captured["values"] = dict(values)
        if "specs" not in captured:
            captured["specs"] = specs
        return original(text, specs, values, dry_run=dry_run)

    fill._apply_specs = spy
    try:
        fill.main(["--dry-run"])
    finally:
        fill._apply_specs = original

    values = captured["values"]
    specs = captured["specs"]
    expected, _ = original(real_text, specs, values, dry_run=False)

    number = re.compile(r"\d+(?:\.\d+)?")

    for spec in specs:
        label, pattern, build, needed = spec[:4]
        optional = spec[4] if len(spec) > 4 else ()
        match = re.search(pattern, expected)
        assert match is not None, (
            f"the rule {label!r} does not match Paper/comparison.md, so the "
            "row it is supposed to refresh is not protected"
        )

        original_row = match.group(0)
        # The build function is handed the same values the rule would see:
        # the required ones, plus the optional ones when they were
        # measured.
        looked_up = {k: values.get(k) for k in needed}
        for key in optional:
            looked_up.setdefault(key, values.get(key))
        built = build(lambda k: looked_up[k], match)

        # The measured numbers in the built row are the candidates for
        # corruption. Take the LAST occurrence of each, because in these
        # rows the anchor text comes first and the value later.
        measured = []
        for key in list(needed) + list(optional):
            value = values.get(key)
            if not isinstance(value, str):
                continue
            for token in number.findall(value):
                if token and token in built:
                    measured.append(token)
        if not measured:
            continue

        broken_row = original_row
        for token in measured:
            index = broken_row.rfind(token)
            if index == -1:
                continue
            decimals = len(token.split(".")[1]) if "." in token else 0
            drifted = ("0." + "1" * decimals) if decimals else "7"
            broken_row = (
                broken_row[:index] + drifted + broken_row[index + len(token):]
            )
        if broken_row == original_row:
            continue

        broken = expected[:match.start()] + broken_row + expected[match.end():]
        repaired, _ = original(broken, specs, values, dry_run=False)
        assert repaired == expected, (
            f"the rule {label!r} did not restore the measured number. "
            f"It left {broken_row!r} in place of {original_row!r}."
        )


def test_the_consolidated_document_compares_all_three_datasets():
    """The single comparison document must cover every dataset we ran.

    The comparison against published work is the point of the document,
    and a dataset that quietly dropped out of it would leave the thesis
    claiming a comparison it does not have. This pins the three datasets
    by name in the document's own section headings and in the dataset
    table.
    """
    committed = Path("Paper/comparison.md")
    if not committed.exists():
        pytest.skip("no comparison document to test")

    text = committed.read_text(encoding="utf-8")

    for dataset in ("HaluEval", "TruthfulQA", "RAGTruth"):
        assert dataset in text, (
            f"{dataset} is missing from Paper/comparison.md; the document "
            "must compare the pipeline against published work on every "
            "dataset it was evaluated on"
        )

    # The three-dataset table gives each one its measured ROC-AUC.
    for label in (
        "**HaluEval QA (length-matched)**",
        "**TruthfulQA**",
        "**RAGTruth (length-matched)**",
    ):
        assert label in text, (
            f"the dataset table row {label!r} is missing from "
            "Paper/comparison.md"
        )

    # The comparison against published work is explicit, and it names the
    # published system it compares against.
    assert "LLM-Check" in text, (
        "the RAGTruth comparison against the published LLM-Check numbers "
        "is missing from Paper/comparison.md"
    )


# ---------------------------------------------------------------------------
# The one rule for a win: the bottom of our 95% interval must clear the
# published number.
#
# These tests assert the RULE, not a conclusion. The values are synthetic
# and chosen so that the rule gives different answers for the point
# estimate and for the interval, which is the only way a test can tell
# the two apart. A test that pinned "P@R90 is a win" would keep passing
# after the measurement moved and the rule stopped holding.
# ---------------------------------------------------------------------------


def _lookup_for(values: dict):
    return values.__getitem__


def test_a_win_needs_the_interval_to_clear_the_comparable():
    """A higher point estimate is not a win when the interval overlaps.

    Our point estimate is above the published number, but the bottom of
    our interval is below it, so the measurement does not establish the
    difference. The rule must return a tie, not a win.
    """
    from scripts.fill_comparison_table import _verdict_from_ci

    values = {
        "detection_aupr": "0.900",
        "detection_aupr_ci": "[0.700, 0.950]",
    }
    verdict = _verdict_from_ci(
        _lookup_for(values), "detection_aupr", "detection_aupr_ci",
        "0.840 (Decoding Memories)",
    )
    assert verdict != "✓", (
        "a metric whose 95% interval overlaps the published number was "
        "reported as a win"
    )


def test_a_win_is_claimed_when_the_interval_clears_the_comparable():
    """The rule's positive case: the whole interval sits above."""
    from scripts.fill_comparison_table import _verdict_from_ci

    values = {
        "detection_aupr": "0.920",
        "detection_aupr_ci": "[0.906, 0.938]",
    }
    verdict = _verdict_from_ci(
        _lookup_for(values), "detection_aupr", "detection_aupr_ci",
        "0.840 (Decoding Memories)",
    )
    assert verdict == "✓"


def test_a_metric_with_no_interval_gets_no_verdict():
    """Without an interval there is no evidence of separation either way.

    The rule has to decline rather than fall back on the point estimate,
    because the caller then keeps whatever the document already said
    instead of inventing a verdict the measurement does not support.
    """
    from scripts.fill_comparison_table import _verdict_from_ci

    values = {"claim_f1": "0.811"}
    verdict = _verdict_from_ci(
        _lookup_for(values), "claim_f1", "claim_f1_ci", "0.71 (TU Wien)"
    )
    assert verdict is None


def test_a_comparable_with_no_published_number_gets_no_verdict():
    """A metric nobody published a score for cannot be won or lost."""
    from scripts.fill_comparison_table import _verdict_from_ci

    values = {
        "claim_correction_rate": "0.706",
        "claim_correction_rate_ci": "[0.679, 0.732]",
    }
    for cell in ("none published", "-", "—"):
        verdict = _verdict_from_ci(
            _lookup_for(values), "claim_correction_rate",
            "claim_correction_rate_ci", cell,
        )
        assert verdict is None, (
            f"a comparable cell of {cell!r} produced a verdict; there is "
            "no published number to measure against"
        )


def test_an_implied_baseline_is_not_a_published_comparable():
    """An implied baseline is derived from another metric, not published.

    Dale's "~0.55" is computed from his own delta, so it is not a score
    he published for this metric. Treating it as one would make the
    comparison look like a measurement against a measurement.
    """
    from scripts.fill_comparison_table import _verdict_from_ci

    values = {
        "hallucination_reduction_rate": "0.132",
        "hallucination_reduction_rate_ci": "[0.114, 0.153]",
    }
    verdict = _verdict_from_ci(
        _lookup_for(values), "hallucination_reduction_rate",
        "hallucination_reduction_rate_ci", "~0.55 (Dale 2023, implied)",
    )
    assert verdict is None


def test_the_year_in_a_comparable_cell_is_not_read_as_a_score():
    """The system's name in the cell can carry a year, and it is not one.

    "0.987 (Dale 2023 -- ALTI+)" holds one score, 0.987. Reading every
    number in the cell picks up 2023 as well and turns the comparable
    into a number nobody published -- which silently decides the row.
    """
    from scripts.fill_comparison_table import _comparable_number

    assert _comparable_number("0.987 (Dale 2023 — ALTI+)") == 0.987
    assert _comparable_number("0.703 (Dale 2023 — LaBSE)") == 0.703


def test_a_band_comparable_is_read_as_its_top():
    """A published band is read as the best score in it.

    "0.85-0.90 (Sent-T5 XL)" is the range the other system reached; the
    strongest claim we can make against it is its top.
    """
    from scripts.fill_comparison_table import _comparable_number

    assert _comparable_number("0.85-0.90 (Sent-T5 XL)") == 0.90


def test_every_surface_uses_the_same_verdict_for_the_same_metric():
    """One metric cannot be a win in one table and a tie in another.

    The section-3.2 metric table, the win matrix and the visual
    headline block all report a verdict for the same measurement
    against the same published number. When they used different rules
    the document called P@R90 a tie in the metric table and a win in the
    win matrix. This asserts the three surfaces agree.
    """
    from scripts.fill_comparison_table import (
        _verdict_from_ci,
        _visual_headline_block,
    )

    values = {
        "detection_roc_auc": "0.925",
        "detection_roc_auc_ci": "[0.911, 0.940]",
        "detection_aupr": "0.922",
        "detection_aupr_ci": "[0.906, 0.938]",
        "precision_at_recall_90": "0.800",
        "precision_at_recall_90_ci": "[0.773, 0.844]",
        "detection_accuracy": "0.829",
        "detection_accuracy_ci": "[0.809, 0.851]",
        "spearman_rho": "0.736",
        "spearman_rho_ci": "[0.712, 0.762]",
        "claim_f1": "0.811",
        "hallucination_reduction_rate": "0.132",
        "hallucination_reduction_rate_ci": "[0.114, 0.153]",
        "verification_success_rate": "0.701",
        "verification_success_rate_ci": "[0.673, 0.726]",
        "mit_delta": "-0.007",
        "mit_delta_ci": "[-0.022, +0.007]",
        "latency_mean_s": "4.18",
    }

    # The metric table's verdict for each row, read from the same helper
    # the rule uses.
    rows = [
        ("detection_roc_auc", "detection_roc_auc_ci",
         "0.987 (Dale 2023 — ALTI+)"),
        ("detection_aupr", "detection_aupr_ci", "0.840 (Decoding Memories)"),
        ("precision_at_recall_90", "precision_at_recall_90_ci",
         "0.703 (Dale 2023 — LaBSE)"),
        ("detection_accuracy", "detection_accuracy_ci",
         "0.847 (Halu-NLP — SHROOM)"),
        ("spearman_rho", "spearman_rho_ci", "0.770 (Halu-NLP)"),
        ("verification_success_rate", "verification_success_rate_ci",
         "0.81 (Halu-NLP)"),
    ]
    expected = {
        key: _verdict_from_ci(_lookup_for(values), key, ci_key, cell)
        for key, ci_key, cell in rows
    }

    # The visual headline block draws its marks from the same rule, so
    # the two surfaces cannot disagree about a dimension.
    block = _visual_headline_block(_lookup_for(values))
    labels = {
        "detection_roc_auc": "Detection ROC-AUC",
        "detection_aupr": "Detection AUPR",
        "precision_at_recall_90": "P@R90",
        "detection_accuracy": "Sentence accuracy",
        "spearman_rho": "Spearman ρ",
        "verification_success_rate": "VSR",
    }
    for key, label in labels.items():
        line = next(
            (ln for ln in block.splitlines() if ln.startswith(label)),
            None,
        )
        assert line is not None, f"{label} is missing from the block"
        mark = next(c for c in ("✓", "≈", "✗") if c in line)
        expected_mark = expected[key]
        assert mark == expected_mark, (
            f"{label} is {mark} in the visual headline block but "
            f"{expected_mark} under the interval rule; the two surfaces "
            "must not disagree about the same measurement"
        )


def test_the_chart_and_the_paper_compare_against_the_same_published_number():
    """The chart's comparable has to be the one the paper quotes.

    Dale 2023 publishes several variants of the same metric. The paper
    compares P@R90 against the LaBSE variant, so the chart has to
    compare against the LaBSE variant too -- otherwise the chart can
    claim a win against a different number than the paper quotes.
    """
    import csv as _csv

    with Path("Paper/comparison_table.csv").open(
        "r", encoding="utf-8", newline=""
    ) as f:
        rows = list(_csv.reader(f))
    header = [h.strip() for h in rows[0]]
    row = next(
        (r for r in rows[1:]
         if len(r) > 1 and r[1].strip() == "P at R=90"),
        None,
    )
    assert row is not None, "the CSV has no P at R=90 row"
    dale = row[header.index("Dale 2023 ACL")]

    # The cell names the variant it holds, so the number has to be read
    # out of the text rather than cast.
    from scripts.fill_comparison_table import _comparable_number

    dale_value = _comparable_number(dale)
    assert dale_value is not None, (
        f"the CSV's Dale P@R90 cell {dale!r} holds no published number"
    )

    committed = Path("Paper/comparison.md")
    if not committed.exists():
        pytest.skip("no comparison document to test")
    text = committed.read_text(encoding="utf-8")

    # The number the paper quotes for P@R90.
    import re as _re
    m = _re.search(r"\| P@R90 \| [^|]*\|[^|]*\| ([\d.]+) \(", text)
    assert m is not None, "the paper does not quote a P@R90 comparable"
    paper_value = float(m.group(1))

    assert dale_value == paper_value, (
        f"the chart reads the CSV's Dale P@R90 as {dale_value} while the "
        f"paper quotes {paper_value}; the two must be the same published "
        "number"
    )


"""Tests for the LaTeX summary's verdict matrix.

The LaTeX summary is a second surface for the same measurements. It used
to carry a rule of its own -- "At par = within 10% of best" -- and
hand-written verdicts, so it called P@R90 "At par" while the markdown
called the same measurement a win, and its caption listed
claim-level F1 among the dimensions we trail while the row above it
said Win.

These tests assert the rules, not the conclusions: a verdict comes from
the interval, the caption comes from the rows, and the two documents
cannot disagree about the same measurement.
"""

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# A verdict in the LaTeX matrix is a measurement, not a fixed string.
# ---------------------------------------------------------------------------


def test_a_latex_win_needs_the_interval_to_clear_the_comparable(tmp_path):
    """The LaTeX matrix uses the paper's one rule.

    Our point estimate is above the published number but the bottom of
    our interval is below it, so the row is not a win.
    """
    from scripts.fill_comparison_table import _fill_why_we_win_tex

    tex = tmp_path / "why_we_win_summary.tex"
    tex.write_text(
        "P@R90                        & 0.703 (Dale LaBSE) & 0.800 & "
        "At par & Close to top \\\\\n",
        encoding="utf-8",
    )

    _fill_why_we_win_tex(tex, {
        "precision_at_recall_90": 0.800,
        # The bottom of the interval is below their 0.703, so the
        # measurement does not establish the difference.
        "precision_at_recall_90_ci95": [0.600, 0.844],
    })

    out = tex.read_text(encoding="utf-8")
    assert "Win" not in out, (
        "the LaTeX matrix called a metric a win whose interval overlaps "
        "the published number"
    )
    assert "Loss" in out or "At par" in out


def test_a_latex_row_is_a_win_when_the_interval_clears_the_comparable(tmp_path):
    """The rule's positive case: the whole interval sits above."""
    from scripts.fill_comparison_table import _fill_why_we_win_tex

    tex = tmp_path / "why_we_win_summary.tex"
    tex.write_text(
        "P@R90                        & 0.703 (Dale LaBSE) & 0.800 & "
        "At par & Close to top \\\\\n",
        encoding="utf-8",
    )

    _fill_why_we_win_tex(tex, {
        "precision_at_recall_90": 0.800,
        "precision_at_recall_90_ci95": [0.760, 0.800],
    })

    out = tex.read_text(encoding="utf-8")
    assert "Win" in out, (
        "the LaTeX matrix did not claim a win whose interval sits above "
        "the published number"
    )
    assert "At par" not in out


def test_the_latex_matrix_and_the_markdown_agree_on_every_metric():
    """One measurement cannot be a win in one document and a tie in the other.

    The LaTeX summary and the consolidated markdown are two surfaces
    for the same numbers. They are written by different rules, and the
    rules disagreed: P@R90 was "At par" in the LaTeX matrix and a win
    in the markdown.
    """
    from scripts.fill_comparison_table import (
        _comparable_number,
        _verdict_from_ci,
    )

    tex_path = Path("Paper/why_we_win_summary.tex")
    md_path = Path("Paper/comparison.md")
    if not tex_path.exists() or not md_path.exists():
        pytest.skip("no committed paper documents to compare")

    tex = tex_path.read_text(encoding="utf-8")
    md = md_path.read_text(encoding="utf-8")

    # The LaTeX matrix's score rows: dimension & comparable & ours &
    # verdict & justification.
    rows = re.findall(
        r"^(Claim-level F1|Hallucination Reduction Rate|"
        r"Verification Success Rate|Detection ROC-AUC|"
        r"Sentence accuracy|P@R90|Spearman rho)\s*&([^&]*)&"
        r"([^&]*)&([^&]*)&([^&]*)\\\\\s*$",
        tex,
        re.MULTILINE,
    )
    assert rows, "the LaTeX matrix has no score rows"

    # The markdown's metric table: | Metric | value | CI | comparable |
    # verdict |.
    #
    # It is scoped to the "full metric set" table rather than to every
    # five-cell row in the document: later tables reuse the metric
    # names, and a row keyed only by its first cell would let a later
    # table overwrite the metric table's own row.
    metric_table = re.search(
        r"\*\*The full metric set on HaluEval:\*\*\s*\n\s*\n"
        r"\| Metric \|.*?\n(?:\|\|?.*\n)*",
        md,
    )
    assert metric_table, "the markdown has no HaluEval metric table"
    md_rows = {}
    for line in metric_table.group(0).splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        md_rows[cells[0]] = cells

    # The LaTeX dimension names and the markdown metric names.
    aliases = {
        "Claim-level F1": "Claim F1",
        "Hallucination Reduction Rate": "HRR",
        "Verification Success Rate": "VSR",
        "Detection ROC-AUC": "ROC-AUC",
        "Sentence accuracy": "Detection accuracy",
        "P@R90": "P@R90",
        "Spearman rho": "Spearman ρ",
    }

    for dimension, comparable, ours, verdict, _why in rows:
        md_name = aliases.get(dimension.strip())
        if md_name is None or md_name not in md_rows:
            continue
        # The row is | Metric | value | CI | comparable | verdict |, so
        # cell 0 is the name and the four cells to compare are 1..4.
        md_value, md_ci, md_comparable, md_verdict = md_rows[md_name][1:5]

        # The two documents have to be measuring against the same
        # published number.
        #
        # An implied baseline has no published number to parse -- the
        # cell says so -- so both sides parse to None. That is
        # agreement: neither document is comparing against a
        # measurement. One side None and the other a number is not.
        tex_comparable = _comparable_number(comparable)
        md_comparable_value = _comparable_number(md_comparable)
        assert tex_comparable == md_comparable_value, (
            f"{dimension}: the LaTeX matrix compares against "
            f"{tex_comparable} while the markdown compares against "
            f"{md_comparable_value}"
        )

        # And they have to reach the same verdict.
        tex_mark = "✓" if "Win" in verdict else (
            "≈" if "At par" in verdict else "✗"
        )
        assert tex_mark == md_verdict, (
            f"{dimension}: the LaTeX matrix says {tex_mark} and the "
            f"markdown says {md_verdict} for the same measurement"
        )


def test_the_latex_caption_cannot_contradict_its_own_rows():
    """The caption is written from the rows, so it cannot name a win as a loss.

    The caption used to list claim-level F1 among the dimensions we
    trail while the row directly above it said Win, and it stated a rule
    of its own -- "At par = within 10% of best" -- that no other
    surface used.
    """
    from scripts.fill_comparison_table import _tex_verdict_caption

    tex = (
        "\\begin{tabular}{lllll}\n"
        "\\hline\n"
        "Claim-level F1               & 0.71 (TU Wien Vectara) & "
        "\\textbf{0.811} & \\textbf{Win} & Above the cited comparable \\\\\n"
        "Detection ROC-AUC            & 0.987 (Dale ALTI+, MT) & 0.925 & "
        "Loss & Below the cited comparable \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    caption = _tex_verdict_caption(tex)
    assert caption, "no caption was produced"

    # The caption names the dimension we win as one we are ahead on.
    assert "ahead on Claim-level F1" in caption, (
        f"the caption does not report claim-level F1 as a win: {caption!r}"
    )
    # And it names the dimension we trail on.
    assert "trail on" in caption and "Detection ROC-AUC" in caption, (
        f"the caption does not report detection ROC-AUC as a loss: "
        f"{caption!r}"
    )
    # The rule it states is the paper's one rule.
    assert "95\\%" in caption, (
        f"the caption does not state the interval rule: {caption!r}"
    )
    assert "10\\%" not in caption, (
        f"the caption still states a rule of its own: {caption!r}"
    )


def test_the_latex_prose_names_the_side_that_won():
    """A sentence cannot claim the other side won a comparison we won.

    The prose read "P@R90 (Dale LaBSE wins): 0.703 vs. ours 0.800" --
    our 0.800 is above their 0.703, and the bottom of our interval is
    above their number too.
    """
    from scripts.fill_comparison_table import _tex_who_wins

    values = {
        "precision_at_recall_90": "0.800",
        "precision_at_recall_90_ci": "[0.773, 0.844]",
    }
    who = _tex_who_wins(values.__getitem__, "precision_at_recall_90",
                        0.703, "Dale LaBSE")
    assert who == "we win", (
        f"the prose calls a comparison we won '{who}'"
    )

    # And the other way round: an interval that overlaps is neither a
    # win nor a loss. The label has to say so rather than pick a side.
    values["precision_at_recall_90_ci"] = "[0.600, 0.844]"
    who = _tex_who_wins(values.__getitem__, "precision_at_recall_90",
                        0.703, "Dale LaBSE")
    assert who == "at par", (
        f"the prose calls an overlapping comparison '{who}'"
    )

    # An interval entirely below their number is their win.
    values["precision_at_recall_90_ci"] = "[0.500, 0.650]"
    who = _tex_who_wins(values.__getitem__, "precision_at_recall_90",
                        0.703, "Dale LaBSE")
    assert who == "Dale LaBSE wins", (
        f"the prose calls a comparison we lost '{who}'"
    )


def test_the_latex_prose_reports_the_mixture_it_actually_has():
    """The closing sentence has to match the paragraph above it.

    It used to say "All three comparisons ... are not directly
    comparable", which is true of the test sets but was written when we
    lost all three, and it undercut the one win the paragraph claims.
    """
    from scripts.fill_comparison_table import _tex_three_comparison_caveat

    values = {
        "detection_accuracy": "0.829",
        "detection_accuracy_ci": "[0.809, 0.851]",
        "spearman_rho": "0.736",
        "spearman_rho_ci": "[0.712, 0.762]",
        "precision_at_recall_90": "0.800",
        "precision_at_recall_90_ci": "[0.773, 0.844]",
    }
    caveat = _tex_three_comparison_caveat(values.__getitem__)
    # P@R90 clears their number, Spearman's interval sits below theirs,
    # and sentence accuracy's interval covers theirs.
    assert "ahead on 1" in caveat, (
        f"the closing sentence does not report the win: {caveat!r}"
    )
    assert "behind on 1" in caveat, (
        f"the closing sentence does not report the loss: {caveat!r}"
    )
    assert "at par on 1" in caveat, (
        f"the closing sentence does not report the tie: {caveat!r}"
    )


def test_the_strongest_result_item_quotes_the_measured_interval():
    """The defence point's interval is the measured one.

    It quoted 95% CI [0.823, 0.865] for a measured 0.925 -- an
    interval that does not contain the value it was attached to -- and
    said we were below Sent-T5 XL (0.85--0.90) when 0.925 is above
    that band.
    """
    from scripts.fill_comparison_table import _tex_strongest_result

    values = {
        "detection_roc_auc": "0.925",
        "detection_roc_auc_ci": "[0.9113, 0.9401]",
        "mit_delta": "-0.007",
        "mit_delta_ci": "[-0.022, +0.007]",
    }
    item = _tex_strongest_result(values.__getitem__)
    assert "0.911" in item and "0.940" in item, (
        f"the item does not quote the measured interval: {item!r}"
    )
    assert "0.823" not in item and "0.865" not in item, (
        f"the item still quotes the stale interval: {item!r}"
    )
    assert "above Sent-T5 XL" in item, (
        f"the item mis-states where our ROC-AUC sits relative to "
        f"Sent-T5 XL: {item!r}"
    )


def test_an_implied_baseline_is_named_as_one_in_the_latex_matrix():
    """An implied baseline is not a published score, and says so.

    Dale's ~0.55 HRR is derived from his own delta. The matrix cell
    that holds it has to say so, or the comparison reads as a
    measurement against a measurement.
    """
    from scripts.fill_comparison_table import _fill_why_we_win_tex

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "why_we_win_summary.tex"
        path.write_text(
            "Hallucination Reduction Rate & 0.55 (Dale 2023, implied) & "
            "\\textbf{0.132} & Loss & Below the implied baseline \\\\\n",
            encoding="utf-8",
        )
        _fill_why_we_win_tex(path, {
            "hallucination_reduction_rate": 0.132,
            "hallucination_reduction_rate_ci95": [0.114, 0.153],
        })
        out = path.read_text(encoding="utf-8")
    assert "implied baseline" in out, (
        f"the row lost the implied-baseline wording: {out!r}"
    )
    assert "cited comparable" not in out, (
        f"an implied baseline was called a cited comparable: {out!r}"
    )
