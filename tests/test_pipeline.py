"""End-to-end pipeline smoke test.

Uses the fake_pipeline fixture so the test exercises the full 13-step
wiring without paying for HuggingFace model downloads. Production code
always uses real models; the fakes live in tests/_fakes.py.
"""
def test_pipeline_endpoint_to_end(fake_pipeline):
    res = fake_pipeline.run("What is the capital of France?")
    assert res.query == "What is the capital of France?"
    assert res.answer
    assert res.decision in {"PASS", "REGEN", "ABSTAIN"}
    assert 0.0 <= res.confidence <= 1.0
    # One pass runs the 13 steps. A REGEN verdict re-runs the verification
    # chain, so the trace count is a multiple of 13 in that case.
    assert len(res.step_traces) >= 13
    assert len(res.step_traces) % 13 == 0
    assert res.report_markdown


def test_pipeline_regeneration_loop_is_bounded(fake_pipeline):
    """A REGEN verdict must not loop forever."""
    res = fake_pipeline.run("What is the capital of France?")
    max_attempts = int((fake_pipeline.cfg.step(11) or {}).get("max_regen_attempts", 2))
    assert len(res.step_traces) <= 13 * (max_attempts + 1)