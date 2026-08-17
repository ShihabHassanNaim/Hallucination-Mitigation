def test_step09_confidence_fusion(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step05_claim_extraction import Step05ClaimExtraction
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval
    from src.steps.step09_confidence_fusion import Step09ConfidenceFusion

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.answer = "Paris is the capital of France."
    ctx.metadata["pipeline"] = fake_pipeline
    Step05ClaimExtraction(ctx).run()
    Step06HybridRetrieval(ctx).run()
    Step09ConfidenceFusion(ctx).run()
    assert 0.0 <= ctx.confidence <= 1.0