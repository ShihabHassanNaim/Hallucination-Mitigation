def test_step12_explainable_response():
    from src.config import load_config
    from src.steps import StepContext
    from src.steps.step12_explainable_response import Step12ExplainableResponse

    cfg = load_config()
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.answer = "Paris."
    ctx.claims = []
    Step12ExplainableResponse(ctx).run()
    assert "Hallucination Pipeline Report" in ctx.metadata["report_markdown"]
    assert "confidence" in ctx.metadata["report_json"].lower()