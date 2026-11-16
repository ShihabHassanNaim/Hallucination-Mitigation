"""Paired significance tests for the pipeline's own components.

Why this exists
---------------
The literature comparison compares *point estimates*: our ROC-AUC against
Dale 2023's 0.987, and so on. None of the cited papers publish per-query
scores or confidence intervals, so no significance test against them is
possible -- for anyone, on any paper in this space. Claiming one would be
unfalsifiable.

What *is* testable is the pipeline's own contribution, because every
ablation variant scores the *same* queries. That gives a paired design,
which is strictly more powerful than comparing two independent means, and
it is the one place where a genuine significance claim can be made.

This module provides:

  * ``paired_bootstrap_auc`` -- percentile bootstrap over queries for a
    paired difference in ROC-AUC between two score vectors.
  * ``paired_permutation_auc`` -- exact-ish permutation test on the
    paired AUC difference.
  * ``auc_difference_ci`` -- the headline helper: given two variants'
    per-query records, return the AUC difference and its 95% CI.

All resampling is over *queries*, not rows, because the evaluation pairs a
faithful and a hallucinated answer under the same question.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Rank-based ROC-AUC with proper mid-rank handling of ties.

    Mirrors ``src.steps.step13_evaluation._roc_auc`` but is self-contained
    so the test does not depend on the pipeline package.
    """
    n = len(labels)
    if n != len(scores):
        raise ValueError("labels and scores must be the same length")
    positives = sum(labels)
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return 0.5

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mid_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid_rank
        i = j + 1

    rank_sum = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _group_by_query(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in records:
        grouped.setdefault(str(r["query"]), []).append(r)
    return grouped


def _align(a_records: list[dict], b_records: list[dict]) -> tuple[list[int], list[float], list[float]]:
    """Return (labels, scores_a, scores_b) restricted to shared queries.

    Only queries present in both variants are used, so the comparison is
    paired. If a query has several rows (a faithful and a hallucinated
    answer), the rows are matched on their label.
    """
    ga = _group_by_query(a_records)
    gb = _group_by_query(b_records)
    shared = sorted(set(ga) & set(gb))
    if not shared:
        raise ValueError("the two variants share no queries")

    labels: list[int] = []
    sa: list[float] = []
    sb: list[float] = []
    for q in shared:
        rows_a = ga[q]
        rows_b = gb[q]
        by_label_b = {r["label"]: r for r in rows_b}
        for ra in rows_a:
            rb = by_label_b.get(ra["label"])
            if rb is None:
                continue
            labels.append(int(ra["label"]))
            sa.append(float(ra["score"]))
            sb.append(float(rb["score"]))
    return labels, sa, sb


def permutation_p_value(
    labels: list[int],
    scores_a: list[float],
    scores_b: list[float],
    owners: list[str],
    n_permutations: int = 2000,
    seed: int = 20261115,
) -> float:
    """Two-sided permutation test on the paired ROC-AUC difference.

    Under the null hypothesis that the two variants are equally good, the
    scores of variant A and variant B are exchangeable within a query.
    Randomly swapping the two variants' scores within each query
    therefore generates the null distribution exactly, and the observed
    difference is compared against it.

    This is used in preference to a bootstrap p-value because the
    percentile bootstrap cannot produce a meaningful p-value when the
    resampled differences are all of the same magnitude -- which is
    exactly what happens when one variant carries no signal at all.
    The bootstrap is still reported for the confidence interval; the
    permutation test supplies the p-value.
    """
    observed = roc_auc(labels, scores_a) - roc_auc(labels, scores_b)
    rng = random.Random(seed)

    by_query: dict[str, list[int]] = {}
    for idx, q in enumerate(owners):
        by_query.setdefault(q, []).append(idx)

    extreme = 0
    for _ in range(n_permutations):
        swapped_a = list(scores_a)
        swapped_b = list(scores_b)
        for idxs in by_query.values():
            if len(idxs) < 2:
                continue
            for i in idxs:
                if rng.random() < 0.5:
                    swapped_a[i], swapped_b[i] = swapped_b[i], swapped_a[i]
        diff = roc_auc(labels, swapped_a) - roc_auc(labels, swapped_b)
        if abs(diff) >= abs(observed):
            extreme += 1

    return (extreme + 1) / (n_permutations + 1)


def auc_difference_ci(
    a_records: list[dict],
    b_records: list[dict],
    n_bootstrap: int = 1000,
    n_permutations: int = 2000,
    seed: int = 20261115,
    alpha: float = 0.05,
) -> dict:
    """Paired ROC-AUC difference: bootstrap CI and permutation p-value.

    The CI is a percentile bootstrap over queries. The p-value comes from
    a paired permutation test, because the bootstrap p-value degenerates
    when the resampled differences are all of the same magnitude -- the
    normal case when one variant carries no signal.

    Returns the observed difference, the percentile CI, the p-value and
    the shared-query count. A CI that excludes zero is a statistically
    separated difference.
    """
    labels, sa, sb = _align(a_records, b_records)
    observed = roc_auc(labels, sa) - roc_auc(labels, sb)

    # Resample *queries*, not rows, so the paired structure survives: a
    # query contributes its faithful and hallucinated rows together.
    labels2, sa2, sb2, owners = _align_with_owners(a_records, b_records)
    aligned_index: dict[str, list[int]] = {}
    for idx, q in enumerate(owners):
        aligned_index.setdefault(q, []).append(idx)

    rng = random.Random(seed)
    diffs: list[float] = []
    query_list = [q for q in aligned_index if aligned_index[q]]
    for _ in range(n_bootstrap):
        idxs: list[int] = []
        for _ in range(len(query_list)):
            idxs.extend(aligned_index[rng.choice(query_list)])
        if not idxs:
            continue
        la = [labels2[i] for i in idxs]
        aa = [sa2[i] for i in idxs]
        bb = [sb2[i] for i in idxs]
        diffs.append(roc_auc(la, aa) - roc_auc(la, bb))

    if not diffs:
        return {
            "auc_a": roc_auc(labels, sa), "auc_b": roc_auc(labels, sb),
            "difference": observed, "ci95": None, "p_value": None,
            "n_queries": len(query_list), "n_rows": len(labels),
        }

    diffs.sort()
    lo_idx = int((alpha / 2) * len(diffs))
    hi_idx = min(len(diffs) - 1, int((1 - alpha / 2) * len(diffs)))
    lo, hi = diffs[lo_idx], diffs[hi_idx]

    # Two-sided bootstrap p-value. The percentile CI already tells the
    # story (it excludes zero or it does not), but the bootstrap p-value
    # degenerates when every resampled difference has the same magnitude,
    # which is the normal case for a variant that carries no signal. The
    # permutation test is therefore the one that supplies the p-value.
    p_value = permutation_p_value(
        labels2, sa2, sb2, owners, n_permutations=n_permutations, seed=seed
    )

    return {
        "auc_a": roc_auc(labels, sa),
        "auc_b": roc_auc(labels, sb),
        "difference": observed,
        "ci95": [lo, hi],
        "p_value": p_value,
        "significant": bool(lo > 0 or hi < 0),
        "n_queries": len(query_list),
        "n_rows": len(labels),
    }


def _align_with_owners(
    a_records: list[dict], b_records: list[dict]
) -> tuple[list[int], list[float], list[float], list[str]]:
    """Like ``_align`` but also returns the owning query for each row."""
    ga = _group_by_query(a_records)
    gb = _group_by_query(b_records)
    shared = sorted(set(ga) & set(gb))
    labels: list[int] = []
    sa: list[float] = []
    sb: list[float] = []
    owners: list[str] = []
    for q in shared:
        by_label_b = {r["label"]: r for r in gb[q]}
        for ra in ga[q]:
            rb = by_label_b.get(ra["label"])
            if rb is None:
                continue
            labels.append(int(ra["label"]))
            sa.append(float(ra["score"]))
            sb.append(float(rb["score"]))
            owners.append(q)
    return labels, sa, sb, owners


def load_ablation(path: str | Path = "data/ablation_results.json") -> dict:
    """Load ablation results, failing loudly if per-query data is absent."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k, v in data.items() if not v.get("per_query")]
    if missing:
        raise ValueError(
            f"{path} has no per_query records for: {', '.join(missing)}. "
            "Re-run scripts.ablation_study with the current code so the "
            "paired scores are saved."
        )
    return data


def compare_variants(
    results: dict,
    baseline: str = "full_pipeline",
    variant: str = "without_step03_detector",
    **kwargs,
) -> dict:
    """Paired test of ``baseline`` against ``variant``."""
    return auc_difference_ci(
        results[baseline]["per_query"],
        results[variant]["per_query"],
        **kwargs,
    )
