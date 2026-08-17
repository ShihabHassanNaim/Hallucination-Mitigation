def test_step07_reranking(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval
    from src.steps.step07_reranking import Step07Reranking

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.metadata["pipeline"] = fake_pipeline
    Step06HybridRetrieval(ctx).run()
    Step07Reranking(ctx).run()
    assert ctx.evidence
    payload = ctx.metadata["step07_reranking"]
    assert payload["evidence_count"] >= 1