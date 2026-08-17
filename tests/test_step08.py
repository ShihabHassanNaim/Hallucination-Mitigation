def test_step08_nli_verification(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step05_claim_extraction import Step05ClaimExtraction
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval
    from src.steps.step08_nli_verification import Step08NLIVerification

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.answer = "Paris is the capital of France. It is in Mars."
    ctx.metadata["pipeline"] = fake_pipeline
    Step06HybridRetrieval(ctx).run()
    Step05ClaimExtraction(ctx).run()
    Step08NLIVerification(ctx).run()
    labels = [c.verdict for c in ctx.claims]
    assert all(l in {"supported", "contradicted", "unknown"} for l in labels)