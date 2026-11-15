"""Regression tests for the measured latency and the mitigation delta.

Two things went wrong before these tests existed:

  1. The reported latency band ("0.5-1.5 s (GPU)") came from a separate
     timing estimate rather than from the run that produced the other
     numbers. The measured value is 0.8-13.8 s/query on the
     length-matched HaluEval split, and the "faster than every sampling
     baseline" claim that rested on the old band does not hold.

  2. The mitigation table in the LaTeX summary still carried the
     projected "+0.67 to +0.75" and a caption claiming a win, while the
     measured delta is -0.035.
"""
from scripts.fill_comparison_table import (
    _latency_band,
    _latency_mean,
    _fill_why_we_win_tex,
    _fill_comparison_md,
)

LATENCY_AGG = {
    "latency_min_s": 0.837,
    "latency_max_s": 13.832,
    "latency_mean_s": 4.841,
}


def _lookup(agg):
    return lambda key: agg.get(key)


def test_latency_band_is_the_measured_range():
    assert _latency_band(_lookup(LATENCY_AGG)) == "0.8-13.8 s"


def test_latency_band_is_empty_without_a_measurement():
    assert _latency_band(_lookup({})) == ""


def test_latency_mean_is_zero_without_a_measurement():
    assert _latency_mean(_lookup({})) == 0.0
    assert _latency_mean(_lookup(LATENCY_AGG)) == 4.841


def test_latency_row_is_rewritten_from_the_measurement(tmp_path):
    """The stale band is replaced by the measured one."""
    md = tmp_path / "comparison.md"
    md.write_text(
        "| **Per-query latency (GPU)** | 0.22 s (LLM-Check) | **0.5-1.5 s** | "
        "Slower than LLM-Check, faster than every sampling baseline |\n",
        encoding="utf-8",
    )

    _fill_comparison_md(md, dict(LATENCY_AGG))

    out = md.read_text(encoding="utf-8")
    assert "0.5-1.5 s" not in out
    assert "0.8-13.8 s" in out
    # The claim that rested on the old band must not survive.
    assert "faster than every sampling baseline" not in out


def test_why_we_win_tex_latency_row_is_rewritten(tmp_path):
    """The LaTeX latency table carries the measured band."""
    tex = tmp_path / "why_we_win_summary.tex"
    tex.write_text(
        "\\textbf{Ours} & 0.5--1.5 s & Platt + temperature \\\\\n",
        encoding="utf-8",
    )

    _fill_why_we_win_tex(tex, dict(LATENCY_AGG))

    out = tex.read_text(encoding="utf-8")
    assert "0.5--1.5 s" not in out
    assert "0.8--13.8 s" in out


def test_why_we_win_tex_mitigation_row_uses_the_measured_delta(tmp_path):
    """The mitigation row reports the measured delta, not the projection."""
    tex = tmp_path / "why_we_win_summary.tex"
    tex.write_text(
        "\\textbf{Ours (Step 10 regen + Step 11 gate)} & $-0.63$ & "
        "$+0.04$ to $+0.12$ & \\textbf{$+0.67$ to $+0.75$} \\\\\n",
        encoding="utf-8",
    )

    _fill_why_we_win_tex(tex, dict(LATENCY_AGG))

    out = tex.read_text(encoding="utf-8")
    assert "+0.67" not in out
    assert "-0.035" in out
    # The row must still end with a proper LaTeX row terminator.
    assert out.rstrip().endswith(chr(92) * 2)
