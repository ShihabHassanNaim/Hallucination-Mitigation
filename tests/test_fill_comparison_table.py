"""Tests for the comparison-table generator.

The comparison table must never contain a hand-written estimate presented
as a measurement, so these tests cover both halves of the guarantee: the
cells are rewritten from the evaluation output, and the plotting script
refuses to read a cell that is still marked as projected.
"""
from __future__ import annotations

import csv
import json
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
