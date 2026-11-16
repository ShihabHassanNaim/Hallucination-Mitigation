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


def test_the_reported_band_is_the_detection_path_not_a_generating_run():
    """The benchmark supplies the answer, so step 2 never calls the generator.

    The paper reported the measured latency band (0.8-13.8 s) and then
    explained it "because it runs a generator call and a full claim-level
    verification pass". The second half was false: on the length-matched
    HaluEval split every row takes the provided-answer branch in
    ``step02_grounded_generation``, so the band covers retrieval,
    reranking and verification but no generation.

    This test pins the mechanism so the claim cannot be restated wrongly,
    and it fails loudly if the evaluation is ever changed to generate.
    """
    import json
    from pathlib import Path

    metrics = Path("data/metrics_halueval.json")
    if not metrics.exists():
        import pytest
        pytest.skip("headline metrics file is not present")

    rows = json.loads(metrics.read_text(encoding="utf-8")).get("per_query", [])
    if not rows:
        import pytest
        pytest.skip("headline metrics file has no per-query rows")

    gen = [r["step_times_ms"]["grounded_generation"] for r in rows]
    assert max(gen) < 5.0, (
        f"grounded_generation peaks at {max(gen):.3f} ms, which means the "
        "evaluation now generates its own answers. The reported latency band "
        "and every sentence that describes it as the detection path must be "
        "re-measured before they can be reused."
    )


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


def test_why_we_win_tex_mitigation_row_uses_the_measured_delta(tmp_path, monkeypatch):
    """The mitigation row reports the measured delta, not the projection.

    The measurement is injected rather than read from
    ``data/mitigation_results.json``: that file is rewritten by the
    mitigation evaluation, and while a run is in progress it holds only a
    partial set of rows. A test that read it would silently assert a
    different number depending on whether an evaluation was running.
    """
    monkeypatch.setattr(
        "scripts.fill_comparison_table.MITIGATION_PATH",
        _mitigation_fixture(tmp_path),
    )

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


def _mitigation_fixture(tmp_path, delta: float = -0.035, ci=None):
    """Write a mitigation measurement with a known delta and return its path."""
    import json
    from pathlib import Path

    path = Path(tmp_path) / "mitigation.json"
    path.write_text(
        json.dumps({
            "summary": {
                "n_rows": 60,
                "n_regenerated": 18,
                "n_kept": 42,
                "regenerated_rows": {
                    "n": 18,
                    "pre_bertscore": 0.8632,
                    "post_bertscore": 0.8279,
                    "delta_bertscore": delta,
                    "delta_bertscore_ci95": ci,
                },
                "kept_rows": {"n": 42, "delta_bertscore": 0.0},
            }
        }),
        encoding="utf-8",
    )
    return path
