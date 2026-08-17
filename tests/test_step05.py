def test_step05_claim_extraction(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step05_claim_extraction import Step05ClaimExtraction

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.answer = "Paris is the capital of France. It is in Europe."
    Step05ClaimExtraction(ctx).run()
    assert len(ctx.claims) >= 2
    assert all(c.text for c in ctx.claims)
    assert all(c.provenance in {"intrinsic", "extrinsic", "aggregated"} for c in ctx.claims)