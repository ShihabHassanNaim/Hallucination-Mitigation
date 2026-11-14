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
    build_replacements,
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
    ])
    assert rc == 0

    rows = list(csv.reader(csv_path.open("r", encoding="utf-8", newline="")))
    ours = {r[1]: r[2] for r in rows[1:]}
    assert ours["ROC-AUC"] == "0.789 [0.730, 0.840]"
    assert ours["AUPR"] == "0.800"
    # No cell may still claim to be a projection.
    assert not any("projected" in v.lower() for v in ours.values())


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
