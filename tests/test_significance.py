"""Tests for the paired significance testing.

The literature comparison cannot carry a significance test: none of the
cited papers publish per-query scores, so their numbers are point
estimates only. These tests cover the tests that *are* possible -- the
pipeline's own components, compared on identical queries.
"""
import random

import pytest

from scripts.significance import (
    auc_difference_ci,
    compare_variants,
    roc_auc,
)


def _records(pairs):
    """pairs: list of (query, label, score)."""
    return [
        {"query": q, "label": lab, "score": s, "decision": "PASS"}
        for q, lab, s in pairs
    ]


def _separated_records(seed=0, n=200):
    """Two variants on the same queries: one informative, one not.

    Variant A scores the hallucinated rows above the faithful ones, so its
    AUC is ~1.0. Variant B scores everything identically, so its AUC is
    exactly 0.5. The paired difference is therefore unambiguous.
    """
    rng = random.Random(seed)
    a, b = [], []
    for i in range(n):
        q = f"q{i}"
        for lab in (0, 1):
            # Hallucinated (label 1) gets the higher score.
            a.append((q, lab, 0.5 + 0.4 * lab + rng.uniform(-0.05, 0.05)))
            b.append((q, lab, 0.5))  # constant score -> AUC 0.5
    return _records(a), _records(b)


def test_roc_auc_matches_a_known_value():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_roc_auc_handles_ties_as_half_credit():
    # All scores identical -> every pair is a tie -> AUC 0.5.
    assert roc_auc([0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0]) == 0.5


def test_roc_auc_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.5])


def test_roc_auc_returns_half_for_a_single_class():
    assert roc_auc([1, 1], [0.2, 0.8]) == 0.5


def test_a_better_variant_gives_a_separated_paired_difference():
    a, b = _separated_records()
    res = auc_difference_ci(a, b, n_bootstrap=200, seed=1)

    assert res["difference"] > 0
    assert res["significant"] is True
    assert res["ci95"][0] > 0, (
        "a strictly better variant must have a CI that excludes zero"
    )
    assert res["p_value"] < 0.05


def test_an_identical_variant_is_not_significant():
    a, _ = _separated_records()
    res = auc_difference_ci(a, a, n_bootstrap=200, seed=2)

    assert res["difference"] == pytest.approx(0.0)
    assert res["significant"] is False
    assert res["ci95"][0] <= 0 <= res["ci95"][1]


def test_the_test_only_uses_queries_shared_by_both_variants():
    a = _records([(f"q{i}", lab, 0.9 if lab else 0.1) for i in range(50) for lab in (0, 1)])
    b = _records([(f"q{i}", lab, 0.5) for i in range(25, 50) for lab in (0, 1)])
    res = auc_difference_ci(a, b, n_bootstrap=50, seed=3)

    # 25 shared queries, not 50 + 25.
    assert res["n_queries"] == 25


def test_disjoint_query_sets_are_rejected():
    a = _records([("q1", 0, 0.1), ("q1", 1, 0.9)])
    b = _records([("q2", 0, 0.1), ("q2", 1, 0.9)])
    with pytest.raises(ValueError, match="no queries"):
        auc_difference_ci(a, b)


def test_load_ablation_explains_how_to_get_per_query_data(tmp_path):
    from scripts.significance import load_ablation

    p = tmp_path / "abl.json"
    p.write_text(
        '{"full_pipeline": {"detection_roc_auc": 0.8},'
        ' "without_step03_detector": {"detection_roc_auc": 0.6}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="per_query"):
        load_ablation(p)


def test_compare_variants_uses_the_baseline_and_variant_by_name():
    a, b = _separated_records()
    results = {
        "full_pipeline": {"per_query": a},
        "without_step03_detector": {"per_query": b},
    }
    res = compare_variants(results, n_bootstrap=100, seed=4)
    assert res["difference"] > 0
    assert res["auc_a"] > res["auc_b"]
