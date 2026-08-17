def test_step02_grounded_generation(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step02_grounded_generation import Step02GroundedGeneration

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.metadata["pipeline"] = fake_pipeline
    Step02GroundedGeneration(ctx).run()
    assert ctx.answer
    payload = ctx.metadata["step02_grounded_generation"]
    assert payload["backend"] in {"llama", "qwen"}