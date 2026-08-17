def test_step11_final_gate():
    from src.config import load_config
    from src.steps import StepContext
    from src.steps.step11_final_gate import Step11FinalGate

    cfg = load_config()
    ctx = StepContext(query="x", cfg=cfg)
    ctx.confidence = 0.9
    ctx.claims = []
    Step11FinalGate(ctx).run()
    assert ctx.decision == "PASS"

    ctx.confidence = 0.1
    ctx.claims = []
    Step11FinalGate(ctx).run()
    assert ctx.decision == "ABSTAIN"