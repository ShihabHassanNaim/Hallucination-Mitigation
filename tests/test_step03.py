def test_step03_early_detection(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step03_early_detection import Step03EarlyDetection

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.answer = "Paris is the capital of France."
    ctx.metadata["pipeline"] = fake_pipeline
    Step03EarlyDetection(ctx).run()
    payload = ctx.metadata["step03_early_detection"]
    assert "features" in payload
    assert 0.0 <= payload["hallucination_probability"] <= 1.0