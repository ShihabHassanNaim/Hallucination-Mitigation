def test_step13_evaluation():
    from src.config import load_config
    from src.steps import StepContext
    from src.steps.step13_evaluation import Step13Evaluation

    cfg = load_config()
    ctx = StepContext(query="x", cfg=cfg)
    ctx.confidence = 0.7
    ctx.decision = "PASS"
    ctx.claims = []
    Step13Evaluation(ctx).run()
    m = ctx.metadata["step13_evaluation"]["metrics"]
    assert m["end_to_end_accuracy"] == 1.0
    assert m["verification_success_rate"] == 1.0