def test_step06_hybrid_retrieval(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.metadata["pipeline"] = fake_pipeline
    Step06HybridRetrieval(ctx).run()
    assert ctx.evidence
    assert all(e.doc_id for e in ctx.evidence)