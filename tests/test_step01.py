def test_step01_query_understanding(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step01_query_understanding import Step01QueryUnderstanding
    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.metadata["pipeline"] = fake_pipeline
    Step01QueryUnderstanding(ctx).run()
    payload = ctx.metadata["step01_query_understanding"]
    assert payload["intent"] == "factual"
    assert payload["query_variants"]


def test_step01_multi_hop(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step01_query_understanding import Step01QueryUnderstanding
    cfg = fake_pipeline.cfg
    ctx = StepContext(query="Who wrote 1984 and when did they die?", cfg=cfg)
    ctx.metadata["pipeline"] = fake_pipeline
    Step01QueryUnderstanding(ctx).run()
    payload = ctx.metadata["step01_query_understanding"]
    assert payload["intent"] in {"multi-hop", "factual"}