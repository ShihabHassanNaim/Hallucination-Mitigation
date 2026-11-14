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


def test_ndcg_never_exceeds_one():
    """A gold passage matched by several retrieved documents counts once.

    Without deduplication the gain is credited repeatedly and the score
    can exceed 1.0, which nDCG cannot do by definition.
    """
    from src.steps.step13_evaluation import _ndcg

    assert _ndcg([0, 0], [0]) == 1.0
    assert _ndcg([0, 0, 0], [0]) == 1.0
    assert _ndcg([0], [0]) == 1.0
    assert _ndcg([1], [0]) == 0.0
    assert _ndcg([0, 1], [0, 1]) == 1.0


def test_ndcg_penalises_a_lower_rank():
    from src.steps.step13_evaluation import _ndcg

    first = _ndcg([0], [0])
    second = _ndcg([1, 0], [0])
    assert second < first


def test_retrieval_metrics_are_null_without_gold_evidence():
    """Scoring retrieved ids against themselves would always return 1.0."""
    from src.config import load_config
    from src.steps import StepContext
    from src.steps.step13_evaluation import Step13Evaluation

    cfg = load_config()
    ctx = StepContext(query="x", cfg=cfg)
    ctx.claims = []
    Step13Evaluation(ctx).run()
    m = ctx.metadata["step13_evaluation"]["metrics"]
    assert m["retrieval_recall_at_5"] is None
    assert m["retrieval_mrr"] is None