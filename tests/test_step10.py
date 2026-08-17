def test_step10_regeneration(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step05_claim_extraction import Step05ClaimExtraction
    from src.steps.step10_regeneration import Step10Regeneration

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="What is the capital of France?", cfg=cfg)
    ctx.answer = "Paris is the capital of France. Foo is bar."
    ctx.metadata["pipeline"] = fake_pipeline
    Step05ClaimExtraction(ctx).run()
    Step10Regeneration(ctx).run()
    assert ctx.final_answer
    assert ctx.metadata["step10_regeneration"]["decision"] in {"KEEP", "EDITED", "EVIDENCE_REPLACE", "ABSTAIN", "REGENERATED"}