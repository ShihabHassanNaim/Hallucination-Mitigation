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
    assert len(res.step_traces) == 13
    assert res.report_markdown