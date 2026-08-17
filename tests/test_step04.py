def test_step04_confidence_assessment(fake_pipeline):
    from src.steps import StepContext
    from src.steps.step04_confidence_assessment import Step04ConfidenceAssessment

    cfg = fake_pipeline.cfg
    ctx = StepContext(query="x", cfg=cfg)
    ctx.feature_vector = {"cosine": 0.9}
    ctx.metadata["pipeline"] = fake_pipeline
    Step04ConfidenceAssessment(ctx).run()
    assert 0.0 <= ctx.eedc_score <= 1.0