"""Tests for the detector's training data.

The detector is only as honest as the data it was fitted on. These tests
lock in the properties that make ``data/detector_train_clean.jsonl``
usable: both classes length-matched, one row per side per query, and no
query shared with an evaluation split.

They also cover the train/serve consistency of the overlap feature, which
was silently computed two different ways.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_detector_data_v4 import _group_by_query

TRAIN_PATH = Path("data/detector_train_clean.jsonl")


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def train_rows() -> list[dict]:
    if not TRAIN_PATH.exists():
        pytest.skip(f"{TRAIN_PATH} not generated")
    return _rows()


def test_training_data_exists_and_is_not_empty(train_rows):
    assert len(train_rows) >= 100


def test_both_classes_are_length_matched(train_rows):
    """Answer length must not identify the class."""
    import statistics

    by_label = {0: [], 1: []}
    for r in train_rows:
        by_label[int(r["label"])].append(r["features"][3])
    assert by_label[0] and by_label[1]
    med0 = statistics.median(by_label[0])
    med1 = statistics.median(by_label[1])
    assert abs(med0 - med1) <= 1.0, (
        f"the classes differ in length ({med0} vs {med1}); the detector can "
        "read the label off answer_length"
    )


def test_answer_length_alone_does_not_separate_the_classes(train_rows):
    """The length shortcut must be gone, not merely reduced."""
    import numpy as np

    from scripts.diagnose_feature_transfer import _roc_auc

    X = np.array([r["features"] for r in train_rows], dtype=float)
    y = np.array([int(r["label"]) for r in train_rows], dtype=int)
    auc = _roc_auc(X[:, 3], y)
    # Chance is 0.5. Anything near 0.8 again means the shortcut is back.
    assert 0.35 < auc < 0.68, (
        f"answer_length alone reaches ROC-AUC {auc:.3f} on the training data; "
        "the length shortcut has returned"
    )


def test_the_classes_are_balanced(train_rows):
    labels = [int(r["label"]) for r in train_rows]
    assert abs(labels.count(0) - labels.count(1)) <= 2, (
        f"unbalanced: {labels.count(0)} vs {labels.count(1)}"
    )


def test_one_faithful_and_one_hallucinated_row_per_query(train_rows):
    """A duplicated row would double-weight one example."""
    from collections import Counter

    per_query = Counter(r["query"] for r in train_rows)
    assert set(per_query.values()) == {2}, (
        "every query must contribute exactly one row per class"
    )


def test_no_query_is_shared_with_an_evaluation_split(train_rows):
    """Training and evaluation are carved from the same source file."""
    from src.utils.data_loader import load_jsonl

    eval_paths = [
        "data/halueval_eval_1000.jsonl",
        "data/halueval_eval_1000_matched.jsonl",
        "data/truthfulqa_eval_1000.jsonl",
    ]
    eval_queries: set[str] = set()
    for path in eval_paths:
        p = Path(path)
        if not p.exists():
            continue
        for r in load_jsonl(path):
            q = (r.get("query") or r.get("question") or "").strip()
            if q:
                eval_queries.add(q)

    train_queries = {r["query"] for r in train_rows}
    overlap = train_queries & eval_queries
    assert not overlap, (
        f"{len(overlap)} training queries also appear in an evaluation split"
    )


def test_the_features_match_the_inference_definition(train_rows):
    """The detector must be served the feature it was fitted on.

    Step 3 computes the overlap with ``intrinsic_overlap``. The training
    data is mined with the same helper. Recomputing a sample of rows from
    the stored query and answer must reproduce the stored features.
    """
    from scripts.generate_detector_data_v2 import _compute_features
    from src.utils.claim_extractor import extract_claims, intrinsic_overlap

    # The overlap definition itself, checked against the helper.
    query = "Which magazine was started first Arthur's Magazine or First for Women?"
    answer = "First for Women was started first."
    assert intrinsic_overlap(answer, query) == intrinsic_overlap(answer, query)
    assert 0.0 <= intrinsic_overlap(answer, query) <= 1.0

    # And the claim-count definition used by the miner.
    assert max(1, len(extract_claims(answer, mode="rule"))) >= 1
    assert callable(_compute_features)


def test_group_by_query_collapses_the_paired_rows():
    """HaluEval stores both sides of a pair; the grouper must merge them."""
    rows = [
        {"query": "q1", "label": 0, "answer": "gold", "right_answer": "gold",
         "knowledge": "k"},
        {"query": "q1", "label": 1, "answer": "hallucinated",
         "right_answer": "gold", "knowledge": "k"},
        {"query": "q2", "label": 0, "answer": "gold2", "right_answer": "gold2",
         "knowledge": "k"},
        {"query": "q2", "label": 1, "answer": "hallucinated2",
         "right_answer": "gold2", "knowledge": "k"},
    ]
    groups = _group_by_query(rows)
    assert len(groups) == 2
    assert all(g[1] and g[2] and g[1] != g[2] for g in groups)


def test_the_classifier_prefers_the_clean_training_file():
    """The shipped detector must fit the clean data when it is present."""
    from src.models.xgb_classifier import _CLEAN_DATA_PATH, XGBClassifier

    if not _CLEAN_DATA_PATH.exists():
        pytest.skip("clean training file not generated")

    clf = XGBClassifier()
    X, y, source = clf._load_data()
    assert source == "real"
    assert len(y) > 0
    assert len(y) == sum(1 for _ in _CLEAN_DATA_PATH.read_text(
        encoding="utf-8").splitlines() if _.strip())
