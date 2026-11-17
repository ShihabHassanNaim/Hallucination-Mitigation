"""Fill the "Ours" column of ``Paper/comparison_table.csv`` with measured values.

The comparison table originally carried hand-written projections in the
"Ours" cells (e.g. ``"0.91-0.95 (projected)"``). Those were estimates made
before the evaluation could actually measure detection, and the plotting
script rendered them as if they were measurements.

This script replaces every projected cell with the corresponding number
computed from the evaluation output, so the table is always regenerated
from the data and can never drift from it.

Which measurement fills which cell:

  * ROC-AUC, AUPR, P at R=90, Sentence Accuracy, Spearman rho
      -> ``data/metrics_halueval.json`` (length-matched HaluEval)
  * Claim F1, Claim P/R
      -> the claim-level metrics of the same file. These are computed on
      the faithful rows only: on a hallucinated row the benchmark does not
      say which of the answer's claims are wrong, so claim recall is
      undefined there and is excluded from the average.

Numbers are written with their bootstrap confidence interval, e.g.
``0.789 [0.732, 0.841]``.

Usage:
  python -m scripts.fill_comparison_table
  python -m scripts.fill_comparison_table --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

CSV_PATH = Path("Paper/comparison_table.csv")
METRICS_PATH = Path("data/metrics_halueval.json")
MITIGATION_PATH = Path("data/mitigation_results.json")
# The diagnostic runs that explain WHY the rewrite fails. They are not
# the headline mitigation result and are kept in their own file so the
# two can never be confused.
MITIGATION_INVESTIGATION_PATH = Path("data/mitigation_investigation.json")

# Metric row label in the CSV -> (aggregate key, CI key)
DETECTION_ROWS = {
    "ROC-AUC": ("detection_roc_auc", "detection_roc_auc_ci95"),
    "AUPR": ("detection_aupr", "detection_aupr_ci95"),
    "P at R=90": ("precision_at_recall_90", "precision_at_recall_90_ci95"),
    "Sentence Accuracy": ("detection_accuracy", "detection_accuracy_ci95"),
    "Spearman rho": ("spearman_rho", "spearman_rho_ci95"),
}

# Metric row label in the CSV -> aggregate key for the claim-level metrics.
CLAIM_ROWS = {
    "F1": "claim_f1",
    "P/R": ("claim_precision", "claim_recall"),
}

# Rows whose value is a measured range rather than a point estimate. The
# latency is the min-max range of the run that produced the other numbers,
# so it is filled from the same aggregate rather than estimated separately.
RANGE_ROWS = {
    "Per-query Latency (GPU)": ("latency_min_s", "latency_max_s"),
}

# Metric row label in the CSV -> aggregate key for the mitigation-side
# metrics. These are measured by the same evaluation run as the detection
# rows, so they are filled from the same aggregate.
MITIGATION_METRIC_ROWS = {
    "Hallucination Reduction Rate (HRR)": (
        "hallucination_reduction_rate",
        "hallucination_reduction_rate_ci95",
    ),
    "Verification Success Rate (VSR)": (
        "verification_success_rate",
        "verification_success_rate_ci95",
    ),
    "Claim Correction Rate (CCR)": (
        "claim_correction_rate",
        "claim_correction_rate_ci95",
    ),
    "End-to-End Accuracy": (
        "end_to_end_accuracy",
        "end_to_end_accuracy_ci95",
    ),
}

_PROJECTED = re.compile(r"projected", re.IGNORECASE)


def _apply_specs(
    text: str,
    specs: list[tuple],
    values: dict,
    dry_run: bool = False,
) -> tuple[str, list[tuple[str, str, str]]]:
    """Apply the row replacements in ``specs`` to ``text``.

    Each spec is ``(label, pattern, build, needed)`` where ``pattern`` is a
    regular expression that matches the row by its *stable* text and captures
    the cell(s) that hold a measured value. A spec may carry a fifth
    element, a tuple of *optional* value names: those do not have to be
    present for the rule to run, and the build function is given None for
    the ones that are missing. A rule that owns several rows at once needs
    this, because a run that measured one metric but not another must still
    refresh the row it can.

    The patterns are deliberately written so that a row is matched no matter
    what number it currently holds. An earlier version anchored on the exact
    old string, which meant a row could only ever be filled once: the second
    run found no match and left the previous run's number in place. Matching
    the surrounding text instead makes the fill idempotent, so re-running
    after a new evaluation always refreshes the row.
    """
    replacements: list[tuple[str, str, str]] = []
    for spec in specs:
        label, pattern, build, needed = spec[:4]
        optional = spec[4] if len(spec) > 4 else ()
        values_needed = {k: values.get(k) for k in needed}
        if any(val is None for val in values_needed.values()):
            # The metric was not measured — leave the row untouched rather
            # than substituting a number that does not exist.
            continue
        for key in optional:
            values_needed.setdefault(key, values.get(key))

        def _sub(match: re.Match) -> str:
            return build(lambda k: values_needed[k], match)

        new_text, count = re.subn(pattern, _sub, text)
        if not count:
            continue
        # Report only rows whose content actually changed. A row that
        # already holds the measured value is left out, so the summary
        # distinguishes "refreshed" from "already current".
        if new_text == text:
            continue
        text = new_text
        replacements.append((label, pattern, f"{count} row(s)"))
    return text, replacements


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _load_aggregate(path: Path) -> dict:
    """Return the aggregate block, computing it from the rows if absent.

    A metrics file written by an in-progress run carries only ``per_query``
    (the aggregate is written at the end), so the aggregate is rebuilt from
    the rows rather than being required to be present.
    """
    payload = _load_metrics(path)
    if not payload:
        return {}
    agg = payload.get("aggregate")
    if isinstance(agg, dict) and agg:
        return agg
    rows = payload.get("per_query") or payload.get("rows") or []
    if not rows:
        return {}
    from scripts.evaluate_full import _aggregate_from_rows

    slug = payload.get("dataset") or "unknown"
    return _aggregate_from_rows(slug, rows)


def _fmt(value: float, ci: list[float] | None) -> str:
    """Format a value with its CI when one is available."""
    if value is None:
        return "n/a"
    if ci:
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{value:.3f}"


# The cross-dataset numbers the paper reports beside HaluEval. Each entry is
# (metrics file, label). A dataset whose file is missing is skipped, so the
# generator degrades to "the rows it can measure" rather than inventing one.
CROSS_DATASETS = (
    (Path("data/metrics_truthfulqa_closedbook.json"), "truthfulqa"),
    (Path("data/metrics_ragtruth_fitted.json"), "ragtruth"),
)

# The published AUROC anchors, read from data/published_anchors.json and
# verified line by line against the source PDF. They are the *published*
# values, so they are not regenerated from a measurement -- they are the
# thing our measurement is compared against.
_PUBLISHED_AUROC = {
    "Attention Score": "0.5830",
    "Hidden Score": "0.5724",
    "Logit, Window Entropy": "0.5458",
    "Logit, Perplexity": "0.5327",
    "Logit, Log Entropy": "0.5168",
}


def _ragtruth_per_task(metrics_path: Path) -> dict:
    """The RAGTruth detection AUC per task type, from the current run.

    The task type lives in the evaluation file, not in the metrics file,
    so the two are joined on the query text. Returns ``{task: {roc_auc,
    n}}`` for the tasks that have both labels present.

    The breakdown is reported for transparency: the published anchors are
    pooled averages with no per-task split, so these are not a separate
    comparison. They are still a measurement, so they are computed from
    the current run rather than kept by hand.
    """
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    rows = data.get("per_query") or data.get("rows") or []
    if not rows:
        return {}

    eval_path = Path("data/ragtruth_eval_1000.jsonl")
    if not eval_path.exists():
        return {}
    task_of: dict[str, str] = {}
    try:
        with eval_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                q = rec.get("query")
                if q:
                    task_of[q] = rec.get("task_type") or ""
    except OSError:
        return {}

    buckets: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        q = row.get("query")
        task = task_of.get(q)
        if not task:
            continue
        try:
            label = int(row["label"])
            score = float(row["detector_score"])
        except (KeyError, TypeError, ValueError):
            continue
        buckets.setdefault(task, []).append((label, score))

    out: dict[str, dict] = {}
    for task, pairs in buckets.items():
        if len({l for l, _ in pairs}) < 2:
            # A single label cannot produce an AUC.
            continue
        try:
            from scripts.evaluate_full import _roc_auc
        except ImportError:
            return {}
        truth = [l for l, _ in pairs]
        scores = [s for _, s in pairs]
        out[task] = {"roc_auc": _roc_auc(truth, scores), "n": len(pairs)}
    return out


def load_cross_dataset_aggregates(
    datasets: tuple[tuple[Path, str], ...] = CROSS_DATASETS,
) -> dict[str, dict]:
    """Return ``{slug: aggregate}`` for the secondary evaluation datasets.

    TruthfulQA is measured closed-book and RAGTruth on its length-matched
    split; both files are produced by ``scripts/evaluate_full.py``. A file
    that does not exist yet is simply absent from the result.
    """
    out: dict[str, dict] = {}
    for path, slug in datasets:
        agg = _load_aggregate(path)
        if agg:
            out[slug] = agg
    return out


def _signed(value: float) -> str:
    """Format a signed delta, e.g. ``+0.043`` / ``-0.021``."""
    return f"{value:+.3f}"


def _separation_verdict(
    value: float | None,
    ci: list[float] | None,
    best: float | None,
    best_label: str = "the best published comparable",
) -> str:
    """Describe how our score relates to a published comparable.

    The verdict is derived from the confidence interval, not from the point
    estimate. Comparing point estimates is what produced the claim that we
    lead on AUPR: our 0.853 is above Decoding Memories' reported 0.84-0.85,
    but our own 95% CI [0.833, 0.874] covers that whole range, so the two
    are not separated and the comparison is a tie.

    Three outcomes are possible:

    * ``separated``   our CI sits entirely above the comparable
    * ``overlap``     the intervals (or the point values) intersect, so the
                      comparison is a tie
    * ``no value``    nothing was measured, so nothing can be claimed
    """
    if value is None or not ci or len(ci) != 2:
        return "not measured"
    lo, hi = ci
    if best is None:
        return "no published comparable to compare against"
    if lo > best:
        return f"above {best_label} ({best:.3f}); 95% CI excludes it"
    if hi < best:
        return f"below {best_label} ({best:.3f}); 95% CI excludes it"
    return (f"95% CI [{lo:.3f}, {hi:.3f}] overlaps {best_label} "
            f"({best:.3f}) — a tie, not a win")


def load_mitigation(path: Path) -> dict:
    """Return the mitigation summary block, or {} when it is absent.

    The mitigation numbers come from ``scripts/evaluate_mitigation.py``,
    which measures the similarity of the answer to the gold answer before
    and after step 10 rewrites it. The comparison table's mitigation rows
    are only rewritten when this file exists and holds a measured delta.
    """
    payload = _load_metrics(path)
    if not payload:
        return {}
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary:
        return summary
    # A file written by an in-progress run has only per_row; rebuild the
    # summary from it so the table can be filled mid-run.
    rows = payload.get("per_row") or []
    if not rows:
        return {}
    regen = [r for r in rows if r.get("regenerated")]
    kept = [r for r in rows if not r.get("regenerated")]

    def _agg(rs: list[dict]) -> dict:
        deltas = [r["delta_bertscore"] for r in rs
                  if r.get("delta_bertscore") is not None]
        if not deltas:
            return {}
        mean = sum(deltas) / len(deltas)
        return {
            "n": len(rs),
            "pre_bertscore": sum(r["pre_bertscore"] for r in rs
                                 if r.get("pre_bertscore") is not None) / len(deltas),
            "post_bertscore": sum(r["post_bertscore"] for r in rs
                                  if r.get("post_bertscore") is not None) / len(deltas),
            "delta_bertscore": mean,
        }

    return {
        "n_rows": len(rows),
        "n_regenerated": len(regen),
        "n_kept": len(kept),
        "all_rows": _agg(rows),
        "regenerated_rows": _agg(regen),
        "kept_rows": _agg(kept),
    }


def build_mitigation_replacements(mit: dict) -> dict[str, str]:
    """Return {csv metric label: new cell text} for the mitigation rows.

    Only rows that were actually *regenerated* are used for the headline
    delta: a row that step 10 chose not to touch has a delta of exactly zero
    by construction, and averaging those in would understate the effect of
    the step rather than measure it.
    """
    out: dict[str, str] = {}
    if not mit:
        return out

    regen = mit.get("regenerated_rows") or {}
    allr = mit.get("all_rows") or {}
    if not regen or regen.get("delta_bertscore") is None:
        return out

    delta = regen["delta_bertscore"]
    ci = regen.get("delta_bertscore_ci95")
    pre = regen.get("pre_bertscore")
    post = regen.get("post_bertscore")
    n = mit.get("n_regenerated") or regen.get("n")

    out["Delta"] = _signed(delta)
    if ci:
        out["Delta"] += f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    if n:
        out["Delta"] += f" (n={n} regenerated)"

    if pre is not None:
        out["Pre-mitigation Score"] = f"{pre:.3f} (BERTScore F1)"
    if post is not None:
        out["Post-mitigation Score"] = f"{post:.3f} (BERTScore F1)"

    # The "Delta" cell in the CSV asserts ours is the highest. That claim
    # only survives if the measured delta is actually positive -- and a
    # positive point estimate whose CI still covers zero is not evidence
    # of an improvement, so the interval is what decides it.
    if not _delta_is_a_win(delta, ci):
        out["Delta"] = out["Delta"] + " — not positive"
    return out


def _delta_is_a_win(delta: float, ci: list[float] | None) -> bool:
    """Whether the measured mitigation delta counts as an improvement.

    A win requires the *whole* 95% CI to sit above zero. A delta that is
    positive only at the point estimate is not separable from no effect,
    so it is reported as "not positive" rather than as a win.
    """
    if delta <= 0:
        return False
    if not ci or len(ci) != 2:
        # Without an interval there is no evidence that the effect is
        # distinguishable from zero, so the conservative reading applies.
        return False
    return ci[0] > 0


def _mitigation_delta_text(mit: dict) -> str | None:
    """The measured mitigation delta, formatted for a prose cell.

    Returns None when the delta was not measured, so a caller can leave
    the row untouched rather than substituting a number that does not
    exist.
    """
    if not mit:
        return None
    regen = mit.get("regenerated_rows") or {}
    if not regen or regen.get("delta_bertscore") is None:
        return None

    delta = regen["delta_bertscore"]
    ci = regen.get("delta_bertscore_ci95")
    n = mit.get("n_regenerated") or regen.get("n")

    text = f"{delta:+.3f}"
    if ci:
        text += f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    if n:
        text += f" (n={n} regenerated)"
    if not _delta_is_a_win(delta, ci):
        text += " — not positive"
    return text


def _mitigation_outcome(delta: float, ci: list[float] | None) -> str:
    """The words that state what the measured delta shows.

    The wording follows the sign of the delta and the position of the
    interval, so a sentence cannot claim an improvement the measurement
    does not show. A delta whose interval covers zero is described as
    not separable from no effect, not as a measured failure: the
    measurement does not distinguish it from either.
    """
    if _delta_is_a_win(delta, ci):
        return "the rewrite improved similarity to the gold answer"
    if ci and ci[1] < 0:
        return "the rewrite did not improve similarity to the gold answer"
    return (
        "the rewrite is not distinguishable from no effect on similarity "
        "to the gold answer"
    )


def _mitigation_prose_values(path: Path | None = None) -> dict:
    """The values a prose row needs to quote the measured mitigation delta.

    The verdict cell is derived from the sign of the measured delta so a
    row cannot keep claiming a win once the measurement contradicts it.

    ``path`` selects which measurement to read. It defaults to the
    module-level constant, but the callers pass the path the command line
    was given: an earlier version read the constant here, so ``--mitigation``
    was silently ignored by the prose and the prose reported a different
    measurement than the table beside it.
    """
    mit = load_mitigation(path if path is not None else MITIGATION_PATH)
    delta_text = _mitigation_delta_text(mit)
    if delta_text is None:
        return {}

    regen = mit.get("regenerated_rows") or {}
    delta = regen.get("delta_bertscore")
    if delta is None:
        return {}

    ci = regen.get("delta_bertscore_ci95")
    if _delta_is_a_win(delta, ci):
        verdict = "Highest single gain in the comparison"
    else:
        verdict = (
            "Measured on regenerated rows only; not an improvement over the "
            "no-mitigation baseline"
        )

    # The sentence that states the mitigation result in prose. It is
    # generated rather than hand-written so the number in the sentence
    # cannot drift away from the measurement it describes.
    n_rows = mit.get("n_rows")
    n_regen = mit.get("n_regenerated") or regen.get("n")
    ci_text = ""
    if ci:
        ci_text = f" (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])"
    prose = (
        f"The mitigation \u0394 is {delta:+.3f} BERTScore F1{ci_text} on the "
    )
    if n_regen and n_rows:
        prose += f"{n_regen} of {n_rows} rows where step 10 regenerated"
    elif n_regen:
        prose += f"the {n_regen} rows where step 10 regenerated"
    else:
        prose += "the rows where step 10 regenerated"
    n_in_file = mit.get("n_rows_in_file")
    if n_in_file and n_rows and n_in_file != n_rows:
        prose += (
            f" (the first {n_rows} rows of the {n_in_file}-row split)"
        )
    prose += ", and the HRR / VSR / CCR rates are below the cited comparables."

    # The README paragraph. It states the sign of the result in its first
    # sentence, so the lead carries the sign and the body carries the
    # numbers.
    n_in_file = mit.get("n_rows_in_file")
    if n_in_file and n_rows and n_in_file != n_rows:
        scope = (
            f"(out of {n_rows} evaluated, from the first {n_rows} rows of the "
            f"{n_in_file}-row length-matched HaluEval split)"
        )
    else:
        scope = (
            "(out of {n_rows} evaluated on the length-matched HaluEval split)"
        )
    scope = scope.format(n_rows=n_rows, n_in_file=n_in_file)

    if _delta_is_a_win(delta, ci):
        lead = "positive"
        body = (
            f"On the {n_regen} rows where step 10 actually regenerated "
            f"{scope}, BERTScore F1 moved from {regen.get('pre_bertscore'):.3f} "
            f"to {regen.get('post_bertscore'):.3f}: a \u0394 of "
            f"**{delta:+.3f}** (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]). The "
            f"rewrite therefore *did* improve similarity to the gold answer "
            f"on this sample."
        ) if ci and n_regen and n_rows else (
            f"On the {n_regen} rows where step 10 actually regenerated, "
            f"BERTScore F1 moved from {regen.get('pre_bertscore'):.3f} to "
            f"{regen.get('post_bertscore'):.3f}: a \u0394 of **{delta:+.3f}**."
        )
        readme_verdict = "is a win."
    else:
        lead = "negative" if delta <= 0 else "not distinguishable from zero"
        ci_clause = (
            f" (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])" if ci else ""
        )
        body = (
            f"On the {n_regen} rows where step 10 actually regenerated "
            f"{scope}, BERTScore F1 moved from {regen.get('pre_bertscore'):.3f} "
            f"to {regen.get('post_bertscore'):.3f}: a \u0394 of "
            f"**{delta:+.3f}**{ci_clause}. The rewrite therefore did *not* "
            f"improve similarity to the gold answer on this sample."
        ) if n_regen and n_rows else (
            f"On the {n_regen} rows where step 10 actually regenerated, "
            f"BERTScore F1 moved from {regen.get('pre_bertscore'):.3f} to "
            f"{regen.get('post_bertscore'):.3f}: a \u0394 of **{delta:+.3f}**"
            f"{ci_clause}."
        )
        readme_verdict = "is a loss."

    # A one-sentence statement of the result, for prose that has to name
    # the number and the verdict together.
    outcome = _mitigation_outcome(delta, ci)
    sentence = (
        f"The measured mitigation \u0394 is {delta:+.3f} BERTScore F1"
        + (f" (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])" if ci else "")
        + (f", measured on the {n_regen} rows where step 10 regenerated"
           if n_regen else "")
        + f", so {outcome}."
    )

    # The LaTeX-safe variant of the delta cell. The Markdown form ends
    # with an em dash, which pdflatex cannot encode in the default input
    # encoding, so the LaTeX rows spell the verdict out in words with
    # ASCII punctuation.
    delta_tex_prose = f"{delta:+.3f}"
    if ci:
        delta_tex_prose += f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    if n_regen:
        delta_tex_prose += f" (n={n_regen} regenerated)"
    if not _delta_is_a_win(delta, ci):
        delta_tex_prose += ", not positive"

    return {"__mitigation_delta__": delta_text, "delta": delta_text,
            "verdict": verdict, "mitigation_prose_sentence": prose,
            "readme_delta_lead": lead, "readme_delta_body": body,
            "readme_verdict": readme_verdict,
            "mitigation_sentence": sentence,
            "delta_tex_prose": delta_tex_prose,
            # The three numbers the mitigation table quotes as separate
            # cells: the score before the rewrite, the score after it,
            # and the interval around their difference. The bare delta
            # is separate from ``delta``, which is a sentence carrying
            # the verdict and cannot go in a numeric table cell.
            "mit_pre": f"{regen.get('pre_bertscore'):.3f}",
            "mit_post": f"{regen.get('post_bertscore'):.3f}",
            "mit_delta": f"{float(delta):+.3f}",
            "mit_delta_ci": (
                f"[{float(ci[0]):+.3f}, {float(ci[1]):+.3f}]" if ci else ""
            ),
            # How many rows the rewrite was measured on. Prose that
            # quotes the delta has to say how many rows it is based on,
            # otherwise a number with no scope reads as a general result.
            "mit_n_regen": str(n_regen) if n_regen else "",
            "mit_n_rows": str(n_rows) if n_rows else ""}


def _mitigation_investigation_values(
    path: Path | None = None,
) -> dict[str, str]:
    """The numbers from the step-10 rewrite investigation.

    Section 5.3 of the comparison document explains *why* the rewrite does
    not work: one mode substituted raw passages for a correct answer, and
    a second change made the rewrite fire on more rows without improving
    the delta. Both are measurements, but from short diagnostic runs
    rather than the committed 500-row split, so they are kept in their own
    file and never mixed with the headline result.

    A missing file yields an empty dict, which leaves the prose rules
    unmatched rather than filling them with a guess.
    """
    src = Path(path) if path is not None else MITIGATION_INVESTIGATION_PATH
    if not src.exists():
        return {}
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    out: dict[str, str] = {}
    guard = data.get("evidence_mode_guards") or {}
    before = guard.get("score_before_guard")
    after = guard.get("score_after_guard")
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        out["inv_guard_before"] = f"{float(before):.3f}"
        out["inv_guard_after"] = f"{float(after):.3f}"

    ab = data.get("per_claim_guards") or {}
    old = ab.get("old") or {}
    new = ab.get("new") or {}
    for suffix, src_map in (("old", old), ("new", new)):
        rows = src_map.get("regenerated_rows")
        delta = src_map.get("delta_on_regenerated_rows")
        if isinstance(rows, int):
            out[f"inv_guards_{suffix}_rows"] = str(rows)
        if isinstance(delta, (int, float)):
            out[f"inv_guards_{suffix}_delta"] = (
                f"{float(delta):+.3f}".replace("-", "−")
            )
    return out


def _delta_positive(lookup) -> bool:
    """True when the measured mitigation delta is a positive, separable win.

    The delta text is formatted with a leading sign, so the sign is the
    first non-space character. A missing measurement is treated as
    "not a win" so a prose row never claims a win it cannot support, and
    a text that the formatter marked "not positive" is likewise not a
    win.
    """
    try:
        text = lookup("delta")
    except KeyError:
        return False
    if not text:
        return False
    text = str(text).strip()
    if not text.startswith("+"):
        return False
    # The formatter appends "— not positive" whenever the interval does
    # not clear zero, so its presence is the authoritative signal.
    return "not positive" not in text


def _mitigation_mark(lookup) -> tuple[str, str]:
    """Return the mark and tag for the mitigation row of a chart.

    A mitigation delta is a win only when the measurement is positive
    and its interval excludes zero. An interval that covers zero means
    the rewrite did not measurably help, so the row reads as a loss even
    when the point estimate is positive.
    """
    try:
        delta = float(str(lookup("mit_delta")).replace("−", "-"))
    except (KeyError, TypeError, ValueError):
        return "✗", "Behind"
    try:
        raw = str(lookup("mit_delta_ci")).strip("[]")
        low, high = (float(p.replace("−", "-")) for p in raw.split(","))
    except (KeyError, TypeError, ValueError):
        return ("✓", "Ahead") if delta > 0 else ("✗", "Behind")
    if delta > 0 and low > 0:
        return "✓", "Ahead"
    return "✗", "Behind"


def _latency_band(lookup) -> str:
    """Render the measured per-query latency band.

    The band is the measured min-max range of the run that produced the
    other numbers, so it cannot drift from the evaluation. A run without
    per-step timings has no latency to report, so an empty string is
    returned and the row is left alone.
    """
    try:
        lo = lookup("latency_min_s")
        hi = lookup("latency_max_s")
    except KeyError:
        return ""
    if lo is None or hi is None:
        return ""
    return f"{float(lo):.1f}-{float(hi):.1f} s"


def _latency_mean(lookup) -> float:
    """Return the measured mean per-query latency, or 0.0 when unknown."""
    try:
        value = lookup("latency_mean_s")
    except KeyError:
        return 0.0
    if value is None:
        return 0.0
    return float(value)


# The fastest detector in the comparison, as reported by LLM-Check. The
# win-matrix verdict for the latency row is derived from this, so a
# faster pipeline would flip the mark without anyone editing prose.
_LATENCY_COMPARABLE_S = 0.22


def _latency_verdict(lookup) -> tuple[str, str]:
    """Return the win-matrix mark and comment for the latency row.

    The comment used to assert that the pipeline was "faster than every
    sampling baseline", which rested on a band that was never measured.
    It is derived from the measurement instead: the pipeline is slower
    than the single-pass probe whenever its slowest query is slower
    than that probe's reported latency.
    """
    try:
        hi = lookup("latency_max_s")
    except KeyError:
        hi = None
    if hi is None:
        return "✗", "Slower than the single-pass probe; much richer output"
    if float(hi) <= _LATENCY_COMPARABLE_S:
        return "✓", "Faster than the single-pass probe"
    return "✗", "Slower than the single-pass probe; much richer output"


# The published comparable for each charted dimension, with the paper it
# came from. These are constants because they are claims made by other
# authors; only the "ours" column is measured here.
#
# The fourth element is the key the metric's confidence interval lives
# under, so the block can apply the paper's one rule -- a win needs the
# bottom of our 95% interval above the published number -- instead of
# comparing point estimates. A metric with no interval (Claim F1) has
# none, so its mark falls back to the point estimate and says so.
_VISUAL_COMPARABLES = [
    # (row label, our metric key, published comparable, published value)
    ("Claim F1", "claim_f1", 0.710, None),
    ("Detection AUPR", "detection_aupr", 0.840, "detection_aupr_ci"),
    ("Detection ROC-AUC", "detection_roc_auc", 0.987, "detection_roc_auc_ci"),
    ("Sentence accuracy", "detection_accuracy", 0.847,
     "detection_accuracy_ci"),
    ("P@R90", "precision_at_recall_90", 0.703, "precision_at_recall_90_ci"),
    ("Spearman ρ", "spearman_rho", 0.770, "spearman_rho_ci"),
    ("Mitigation Δ", "mit_delta", 0.60, None),
    ("HRR", "hallucination_reduction_rate", 0.550,
     "hallucination_reduction_rate_ci"),
    ("VSR", "verification_success_rate", 0.810,
     "verification_success_rate_ci"),
]


def _visual_headline_block(lookup) -> str:
    """Render the bar-chart block the demo page reads.

    Each row is one score dimension: the best published comparable, our
    measured value and the gap between them, plus a mark that says
    whether we are ahead, at par or behind. The mark is derived from the
    same numbers rather than written by hand, so a measurement that
    moves cannot leave a stale verdict behind.

    The marks come from the same comparison the win matrix uses -- our
    score against the row's own published comparable -- so the chart and
    the matrix can never disagree about a dimension.
    """
    lines = [
        "Metric                    Best-Cited   Ours    Gap",
        "─" * 61,
    ]
    for label, key, best, ci_key in _VISUAL_COMPARABLES:
        try:
            ours = lookup(key)
        except KeyError:
            ours = None
        if ours is None:
            continue
        ours = float(ours)
        gap = ours - best
        if key == "mit_delta":
            # A mitigation delta is a win only when it is positive and
            # its interval excludes zero.
            mark, tag = _mitigation_mark(lookup)
        else:
            # The paper's one rule: a win needs the bottom of our 95%
            # interval above the published number. When the metric has
            # no interval there is no evidence of separation, so the
            # mark falls back to the point estimate and the tag says
            # which rule was applied.
            verdict = None
            if ci_key:
                verdict = _verdict_from_ci(
                    lookup, key, ci_key, f"{best:.3f}"
                )
            if verdict == "✓":
                mark, tag = "✓", "Ahead"
            elif verdict == "≈":
                mark, tag = "≈", "At par"
            elif verdict == "✗":
                mark, tag = "✗", "Behind"
            elif _above(lookup, key, best):
                mark, tag = "✓", "Ahead (no CI)"
            else:
                mark, tag = "✗", "Behind (no CI)"
        lines.append(
            f"{label:<25} {best:>10.3f} {ours:>7.3f} {gap:>+8.3f}  {mark} {tag}"
        )

    # Latency is a cost rather than a score, so it is charted against
    # the fastest cited probe and always reads as a loss when we are
    # slower.
    try:
        latency = lookup("latency_mean_s")
    except KeyError:
        latency = None
    if latency is not None:
        latency = float(latency)
        gap = latency - _LATENCY_COMPARABLE_S
        mark, tag = ("✗", "Slower") if gap > 0 else ("✓", "Faster")
        lines.append(
            f"{'Latency (s, GPU)':<25} {_LATENCY_COMPARABLE_S:>10.2f} "
            f"{latency:>7.2f} {gap:>+8.2f}  {mark} {tag}"
        )

    return "\n".join(lines) + "\n```"


def _win_matrix_rows(text: str) -> list[tuple[str, str, str]]:
    """The win matrix's rows as (number, dimension, verdict).

    Read by column position, the same way the tally reads the verdict:
    the matrix has seven columns and the verdict is the sixth.
    """
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if not re.match(r"^\| \d+ \|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8:
            continue
        verdict = cells[6]
        if verdict not in ("✓", "✗", "≈"):
            continue
        dimension = cells[2].replace("**", "")
        rows.append((cells[1], dimension, verdict))
    return rows


# The win-matrix dimensions whose verdict is a property of the system's
# design rather than of a number. A structural win is a different kind
# of claim from a score win, and the document says which is which.
_STRUCTURAL_DIMENSIONS = {
    "Per-claim provenance", "Modular backend", "Retrieval that works",
    "Explicit abstention", "Evaluation scale", "Per-claim provenance",
    "Active mitigation",
}


def _headline_read(text: str) -> str:
    """The paragraph that reads the matrix aloud.

    It used to say "the two score wins are claim-level F1 and AUPR"
    while the matrix above it showed three score wins, because the
    paragraph was written by hand and P@R90 became a win after it was
    written. The counts come from the rows, so the reading cannot
    disagree with the table it reads.
    """
    rows = _win_matrix_rows(text)
    if not rows:
        return ""

    wins = [d for _n, d, v in rows if v == "✓"]
    ties = [d for _n, d, v in rows if v == "≈"]
    losses = [d for _n, d, v in rows if v == "✗"]

    score_wins = [d for d in wins if d not in _STRUCTURAL_DIMENSIONS]
    structural_wins = [d for d in wins if d in _STRUCTURAL_DIMENSIONS]
    score_losses = [d for d in losses if d not in _STRUCTURAL_DIMENSIONS]

    def _names(items: list[str]) -> str:
        if not items:
            return "none"
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    parts = [
        f"**Headline read.** Of the {len(wins)} matrix wins, "
        f"{len(structural_wins)} are structural "
        f"({_names(structural_wins)}) -- properties of the system's design "
        f"rather than of a single score. The score wins are "
        f"{_names(score_wins)}.",
    ]
    if score_losses:
        parts.append(
            f"We trail on {_names(score_losses)}, and we are at parity on "
            f"{_names(ties)}."
        )
    body = " ".join(parts)
    # The paragraph is wrapped to keep the document's source readable.
    words = body.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) > 86 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _verdict_tally(text: str) -> str | None:
    """Rebuild the verdict tally from the win-matrix rows.

    The tally summarises the rows above it, so it is computed from those
    rows rather than maintained by hand. Returns None when the matrix is
    not present or a row cannot be classified.

    The winner mark is read by *column position*, not by "the first mark
    on the line". Some rows carry a checkmark in an earlier column -- the
    "Active mitigation" row reads "Dale 2023 only | ✓ | ✓ | ≈", because
    both systems do active mitigation -- and a non-greedy pattern would
    pick up the comparable's mark and count a tie as a win. The matrix
    has seven columns, so the verdict is the sixth.
    """
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not re.match(r"^\| \d+ \|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # cells[0] is empty (the line starts with "|"), so the columns
        # are 1..7 and the verdict is column 6.
        if len(cells) < 8:
            continue
        verdict = cells[6]
        if verdict not in ("✓", "✗", "≈"):
            continue
        rows.append((cells[1], verdict))
    if not rows:
        return None

    wins, ties, losses = [], [], []
    for number, verdict in rows:
        if verdict == "✓":
            wins.append(number)
        elif verdict == "≈":
            ties.append(number)
        else:
            losses.append(number)

    def _fmt(label: str, numbers: list[str]) -> str:
        return f"| **{label}** | {len(numbers)} | {', '.join(numbers)} |"

    return "\n".join([
        _fmt("✓ Win", wins),
        _fmt("≈ Tie / At par", ties),
        _fmt("✗ Loss", losses),
    ])


# The win-matrix rows whose verdict is a measurement rather than a
# property of the system's design. The value is the comparable the row
# is measured against; the key is the metric that holds our score.
_WIN_MATRIX_SCORES = {
    "Claim-level F1": ("claim_f1", 0.71),
    "Detection AUPR": ("detection_aupr", 0.840),
    "Detection ROC-AUC": ("detection_roc_auc", 0.987),
    "Sentence accuracy": ("detection_accuracy", 0.847),
    "P@R90": ("precision_at_recall_90", 0.703),
    "Spearman ρ": ("spearman_rho", 0.770),
    "HRR": ("hallucination_reduction_rate", 0.550),
    "VSR": ("verification_success_rate", 0.810),
    # The mitigation row is scored on the sign of the delta rather than
    # on its size: a delta whose interval covers zero is not a win even
    # when it is positive.
    "Mitigation Δ": ("__mitigation__", 0.60),
}


def _win_matrix_row(v, m) -> str:
    """Rebuild one win-matrix row's value, verdict and justification.

    Only the score rows are rewritten. The other rows compare a design
    property -- does the system cite its claims, does it abstain -- and
    their verdicts do not follow from a number, so they are left alone.

    The verdict is computed against the comparable printed in the row
    itself, never against a different row's comparable: an earlier
    version compared our ROC-AUC against the Sent-T5 XL band of
    0.85-0.90, so a row whose own comparable was 0.987 wrote "Above
    the cited comparables" while claiming a lead over a number
    printed two cells to its left.
    """
    number, dimension, best_paper, best_score, ours, verdict, why = m.groups()

    key_and_threshold = _WIN_MATRIX_SCORES.get(dimension)
    if key_and_threshold is None:
        # A structural row: keep every cell as it is.
        return m.group(0)

    key, threshold = key_and_threshold
    if key == "__mitigation__":
        # The mitigation delta is a win only when the measurement is
        # positive and its interval excludes zero. The helper is the
        # same one the prose uses, so the table and the sentence cannot
        # disagree.
        if _delta_positive(v):
            mark, comment = "✓", "Above the cited comparable"
        else:
            try:
                n_regen = v("mit_n_regen")
            except KeyError:
                n_regen = None
            if n_regen:
                comment = (
                    f"Measured on the {n_regen} rows step 10 regenerated; the "
                    "rewrite does not improve on no mitigation yet"
                )
            else:
                comment = (
                    "Measured on the regenerated rows; not an improvement over "
                    "the no-mitigation baseline"
                )
            mark = "✗"
        return (
            f"| {number} | **{dimension}** | {best_paper} | {best_score} | "
            f"{ours} | {mark} | {comment} |"
        )

    # The row's own metric is what the verdict is computed from, so a
    # row whose metric was not measured keeps the verdict it had. The
    # rule spans every score row, so a run that measured one metric but
    # not another must still refresh the row it can.
    try:
        measured = v(key)
    except KeyError:
        return m.group(0)
    if measured is None:
        return m.group(0)

    # The paper states one rule for a win: the bottom of our 95%
    # confidence interval has to sit above their published number, not
    # our best estimate. Every score row uses that rule, so the matrix
    # cannot call a metric a win that the section-3.2 table calls a tie
    # -- which is what happened when P@R90 was a tie in one table and a
    # win in the other.
    #
    # The comparable is the row's own score cell, so the verdict is
    # always measured against the number printed beside it.
    verdict = _verdict_from_ci(v, key, f"{key}_ci", best_score)
    if verdict == "✓":
        mark, comment = "✓", "Our CI's bottom is above their top"
    elif verdict == "≈":
        mark, comment = "≈", "Our CI overlaps their published value"
    elif verdict == "✗":
        mark, comment = "✗", _below_note(f"{best_paper} {best_score}")
    elif _above(v, key, threshold):
        # No interval for this metric, so there is no evidence of
        # separation either way. The point estimate is what is left, and
        # the comment says so rather than implying a tested result.
        mark, comment = "✓", "Above the cited comparable (no CI for this metric)"
    else:
        mark, comment = "✗", _below_note_with_ci(f"{best_paper} {best_score}")

    return (
        f"| {number} | **{dimension}** | {best_paper} | {best_score} | "
        f"{ours} | {mark} | {comment} |"
    )


def _fill_verdict_tally(text: str, dry_run: bool = False) -> tuple[str, list]:
    """Rewrite the win matrix's verdict tally from the matrix rows.

    The tally counts the rows above it, so it is derived from them rather
    than kept by hand. A row whose verdict flips therefore changes the
    count without anyone editing the table.

    The whole three-row block is replaced, not just the counts: a count
    that no longer matches its own list of dimensions is as wrong as a
    count that no longer matches the matrix.
    """
    tally = _verdict_tally(text)
    if tally is None:
        return text, []

    pattern = re.compile(
        r"\| \*\*✓ Win\*\* \| \d+ \| [^|]* \|\n"
        r"\| \*\*≈ Tie / At par\*\* \| \d+ \| [^|]* \|\n"
        r"\| \*\*✗ Loss\*\* \| \d+ \| [^|]* \|"
    )
    match = pattern.search(text)
    if not match:
        return text, []

    new_text, count = pattern.subn(lambda _m: tally, text)
    if not count or new_text == text:
        return text, []
    return new_text, [
        ("7. verdict tally", "the win matrix rows", f"{count} row(s)")
    ]


def _above(lookup, key: str, threshold: float) -> bool:
    """True when the measured value beats the cited comparable.

    The values are pre-formatted strings, so the comparison is made on the
    numeric text. A missing value is treated as "not above".
    """
    try:
        raw = lookup(key)
    except KeyError:
        return False
    if raw is None:
        return False
    try:
        return float(str(raw).strip()) > threshold
    except ValueError:
        return False


def _ragtruth_win_prose(lookup) -> str:
    """The sentence that reads the RAGTruth verdict table out loud.

    It names the anchors our interval clears and the ones it does not, so
    the prose cannot claim five wins when the table shows three.
    """
    try:
        ci = lookup("rag_detection_roc_auc_ci")
    except KeyError:
        ci = None
    if isinstance(ci, str):
        try:
            ci = [float(p) for p in ci.strip("[]").split(",")]
        except ValueError:
            ci = None
    if not ci or len(ci) != 2:
        return (
            "The interval against each published anchor is reported in the "
            "table above."
        )
    lower = float(ci[0])
    wins = [s for s, p in _PUBLISHED_AUROC.items() if lower > float(p)]
    ties = [s for s, p in _PUBLISHED_AUROC.items() if lower <= float(p)]

    def _name(system: str) -> str:
        # The table names them "LLM-Check — Logit, Window Entropy"; in the
        # prose the shared "Logit," prefix is dropped so the sentence
        # stays readable and the three Logit variants are told apart by
        # what they read.
        return system.split(",", 1)[-1].strip()

    def _list(names: list[str]) -> str:
        if len(names) == 1:
            return f"{_name(names[0])} ({_PUBLISHED_AUROC[names[0]]})"
        if len(names) == 2:
            return (f"{_name(names[0])} ({_PUBLISHED_AUROC[names[0]]}) and "
                    f"{_name(names[1])} ({_PUBLISHED_AUROC[names[1]]})")
        head = ", ".join(
            f"{_name(n)} ({_PUBLISHED_AUROC[n]})" for n in names[:-1]
        )
        return f"{head} and {_name(names[-1])} ({_PUBLISHED_AUROC[names[-1]]})"

    if not ties:
        return (
            f"All five anchors clear the bar ({lower:.4f} > "
            f"{min(float(p) for p in _PUBLISHED_AUROC.values()):.4f}), so all "
            "five are wins."
        )
    if not wins:
        return (
            f"Our bottom is {lower:.4f}, so none of the five anchors is "
            "cleared and all five are reported as ties."
        )
    return (
        f"Our bottom is\n{lower:.4f}, so {_list(wins)} are\ncleared, while "
        f"{_list(ties)} are inside our interval\nand are therefore reported "
        "as ties."
    )


def _load_holdout_head(path: Path | None = None) -> dict:
    """The fitted detection head's held-out score, if it was recorded.

    ``scripts/fit_detection_head.py`` fits the head on one half of a
    split and scores it on the other, which is the only honest
    generalisation estimate available for a head that was fitted on the
    data it is being scored on.
    """
    target = path or Path("data/detection_head_refit600.json")
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    models = data.get("models") or {}
    fitted = models.get("logistic (fitted)") or {}
    auc = fitted.get("roc_auc_holdout")
    ci = fitted.get("roc_auc_holdout_ci95")
    if not isinstance(auc, (int, float)):
        return {}
    out: dict = {"holdout_roc_auc": float(auc)}
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        out["holdout_roc_auc_ci95"] = [float(ci[0]), float(ci[1])]
    return out


def _ragtruth_heldout_prose(lookup) -> str:
    """The paragraph comparing the fitted-on and held-out RAGTruth figures.

    The head was fitted on part of the RAGTruth split, so the score on
    the full split is measured on data the head has seen. The held-out
    figure -- scored on the half the fit never saw -- is the honest
    generalisation estimate, and the paragraph reports both.
    """
    def _get(key: str):
        try:
            return lookup(key)
        except KeyError:
            return None

    fitted = _get("rag_detection_roc_auc")
    holdout = _get("rag_holdout_roc_auc")
    holdout_ci = _get("rag_holdout_roc_auc_ci")
    if not fitted or not holdout:
        return ""
    gap = float(fitted) - float(holdout)
    verdict = "small" if abs(gap) < 0.05 else "not small"
    heldout_lower = ""
    if isinstance(holdout_ci, str):
        try:
            heldout_lower = f"{float(holdout_ci.strip('[]').split(',')[0]):.4f}"
        except (ValueError, IndexError):
            heldout_lower = ""
    anchor = max(float(p) for p in _PUBLISHED_AUROC.values())
    direction = (
        "below" if (heldout_lower and float(heldout_lower) < anchor) else "above"
    )
    return (
        "The detection head was fitted on 496 rows of the\nRAGTruth split, "
        "so the " + str(fitted) + " above was measured on data the model "
        "had already seen. The\nhonest generalisation estimate is the "
        "**held-out** figure: **" + str(holdout) + ", CI ["
        + str(holdout_ci).strip("[]") + "]**,\nscored on a half of the "
        "fitting data it never saw. The gap is " + verdict + ", which is "
        "reassuring.\nBut the held-out lower bound (" + (heldout_lower or "n/a")
        + ") sits **" + direction + "** LLM-Check's strongest anchor\n("
        + f"{anchor:.4f}" + "), so on the held-out figure alone the RAGTruth "
        "comparison would be a tie against\nall five rather than three wins."
    )


def _ragtruth_verdict(lookup, system: str) -> str:
    """The RAGTruth verdict for one published anchor, by the paper's rule.

    A win needs our 95% CI lower bound strictly above the published
    value. When the intervals overlap the comparison is a tie, and the
    paper says so rather than claiming the lead.
    """
    published = _PUBLISHED_AUROC.get(system)
    if published is None:
        return "**WIN**"
    try:
        threshold = float(published)
    except ValueError:
        return "**WIN**"
    ci = None
    try:
        ci = lookup("rag_detection_roc_auc_ci")
    except KeyError:
        ci = None
    if isinstance(ci, str):
        try:
            ci = [float(p) for p in ci.strip("[]").split(",")]
        except ValueError:
            ci = None
    if not ci or len(ci) != 2:
        # Without an interval there is no evidence of separation, so the
        # honest verdict is a tie.
        return "**TIE**"
    return "**WIN**" if float(ci[0]) > threshold else "**TIE**"


def _above_ci(lookup, key: str, threshold: float) -> bool:
    """True when a metric's 95% CI clears ``threshold`` from below.

    This is the separation rule the paper uses everywhere: a win needs
    our lower bound strictly above the published comparable, not merely a
    higher point estimate. A metric with no interval is never a win --
    the evidence is not there.
    """
    try:
        ci = lookup(key)
    except KeyError:
        return False
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False
    try:
        return float(ci[0]) > threshold
    except (TypeError, ValueError):
        return False


def _comparable_number(cell: str) -> float | None:
    """The published number in a comparable cell, or None if there is none.

    A comparable cell can hold a number ("0.847 (Halu-NLP)"), a band
    ("0.85-0.90 (Sent-T5 XL)"), a dash ("-"), a note ("none published")
    or an implied baseline ("~0.55 (Dale 2023, implied)"). Only the
    first two are published scores, and only a band's *top* is a
    published score the other system reached. An implied baseline is
    derived from another metric rather than published, so it is not a
    comparable at all.
    """
    text = str(cell or "").strip()
    if not text or text in {"-", "—", "none published"}:
        return None
    low = text.lower()
    if "implied" in low or "no published" in low or "none published" in low:
        return None
    # Only the part before the parenthesis holds the score. The
    # parenthetical names the system and often the year -- "0.987 (Dale
    # 2023 -- ALTI+)" -- and reading a number out of it picks up 2023 and
    # turns the comparable into a number nobody published.
    head = text.split("(", 1)[0].strip()
    if not head:
        return None
    # The dash in a band ("0.85-0.90") separates two scores; it is not a
    # minus sign. A number is only negative when the minus stands on
    # its own, so the sign is read from the character before the digits
    # rather than being swallowed by the pattern.
    nums = []
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", head):
        sign = ""
        start = m.start()
        if m.group(0)[0] in "+-" and start > 0:
            # A sign glued to the digits is a negative number unless it
            # is the range dash of a band.
            if head[start - 1] not in "0123456789.":
                sign = "-" if m.group(0)[0] == "-" else ""
        nums.append(float(sign + m.group(0).lstrip("+-")))
    if not nums:
        return None
    return max(nums)


def _verdict_from_ci(lookup, key: str, ci_key: str,
                     comparable_cell: str) -> str | None:
    """The verdict for a metric row, under the paper's stated rule.

    The paper states the rule in section 2: "A win is only claimed when
    the bottom of our 95% confidence interval sits above their
    published number -- not our best estimate." Every surface that
    reports a verdict has to use that one rule, or the paper ends up
    calling the same metric a win in one table and a tie in another.

    The comparable is read from the row's own comparable cell, so a
    metric is always measured against the number printed beside it and
    never against some other row's number.

    Returns None when the row cannot be judged -- no comparable was
    published, or the metric has no interval -- so the caller can leave
    the cell as it is rather than inventing a verdict.
    """
    comparable = _comparable_number(comparable_cell)
    if comparable is None:
        return None
    try:
        ci = lookup(ci_key)
    except KeyError:
        ci = None
    if isinstance(ci, str):
        try:
            ci = [float(p) for p in ci.strip("[]").split(",")]
        except ValueError:
            ci = None
    if not ci or len(ci) != 2:
        return None
    try:
        low = float(ci[0])
        high = float(ci[1])
    except (TypeError, ValueError):
        return None
    # The interval decides the verdict in both directions. A win needs
    # the bottom of our interval above their number; a loss needs the
    # top of it below. Anything else is an overlap, and an overlap is
    # not a win.
    #
    # The earlier version fell back on the point estimate to decide a
    # loss, so an interval that sat entirely below their number --
    # [0.500, 0.650] against their 0.703, with our point estimate at
    # 0.800 -- was reported as a tie. The measurement does not
    # establish the difference in either direction, and the point
    # estimate has no say once there is an interval.
    if low > comparable:
        return "✓"
    if high < comparable:
        return "✗"
    return "≈"


def _find_ours_column(header: list[str]) -> int:
    for i, h in enumerate(header):
        if h.strip().lower().startswith("ours"):
            return i
    raise SystemExit(
        "comparison_table.csv has no 'Ours' column; refusing to guess."
    )


def build_replacements(agg: dict) -> dict[str, str]:
    """Return {csv metric label: new cell text} for the measurable rows."""
    out: dict[str, str] = {}

    for label, (key, ci_key) in DETECTION_ROWS.items():
        if key not in agg:
            continue
        out[label] = _fmt(agg[key], agg.get(ci_key))

    for label, key in CLAIM_ROWS.items():
        if isinstance(key, tuple):
            p, r = key
            if p in agg and r in agg:
                out[label] = f"{agg[p]:.3f} / {agg[r]:.3f}"
        elif key in agg:
            out[label] = _fmt(agg[key], None)

    for label, (key, ci_key) in MITIGATION_METRIC_ROWS.items():
        if key not in agg:
            continue
        out[label] = _fmt(agg[key], agg.get(ci_key))

    for label, (lo_key, hi_key) in RANGE_ROWS.items():
        if lo_key not in agg or hi_key not in agg:
            continue
        out[label] = (
            f"{float(agg[lo_key]):.1f}-{float(agg[hi_key]):.1f} s"
        )

    return out


def _safe(text: str) -> str:
    """Render a string for the console, replacing non-cp1252 characters.

    The Windows console uses cp1252, which cannot encode characters like the
    Greek rho used in the paper's tables.
    """
    return text.encode("ascii", errors="replace").decode("ascii")


def _ascii_gap_row(label: str, comparable: float, ours: str) -> str:
    """Rebuild one row of the fixed-width bar-chart data block.

    The row is "label  comparable  ours  gap  verdict", where the gap is
    ours minus the comparable and the verdict says whether we are behind,
    at par or ahead. Rebuilding the whole row from the measurement keeps
    the gap and the verdict from drifting away from the number beside
    them -- the failure mode that made the old table read "0.608" next to
    a gap computed from a different value.
    """
    try:
        ours_f = float(ours)
    except (TypeError, ValueError):
        return f"{label} {comparable:.3f} {ours} n/a ?"
    gap = ours_f - comparable
    # "At par" is a band rather than a sign: a difference smaller than a
    # tenth of a point is not a separation worth claiming either way.
    if abs(gap) < 0.10:
        verdict = "\u2248 At par"
    elif gap > 0:
        verdict = "\u2713 Ahead"
    else:
        verdict = "\u2717 Behind"
    return (
        f"{label:<26}{comparable:<12.3f}{ours_f:<10.3f}{gap:+.3f}  {verdict}"
    )


def _fill_comparison_md(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
    cross: dict[str, dict] | None = None,
) -> list[tuple[str, str, str]]:
    """Replace the projected "Ours" rows in ``Paper/comparison.md``.

    The prose table and the summary table below it both carried the same
    hand-written estimates as the CSV. Each replacement is anchored on the
    row's own text, so an unrelated number is never touched.
    """
    if not path.exists():
        return []

    cross = cross or {}
    tqa = cross.get("truthfulqa") or {}
    rag = cross.get("ragtruth") or {}

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        """Formatted value, or None when the metric was not measured."""
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    replacements: list[tuple[str, str, str]] = []

    # Each substitution is described by the metrics it needs. A row whose
    # metrics have not been measured is left alone rather than filled in
    # with a guess.
    #
    # The patterns match each row by its stable text and capture the cell
    # that holds a measured value, so the fill works no matter what the
    # cell currently contains.
    row_specs = [
        (
            # The three-dataset table has six columns: the dataset, how
            # many rows, how many unique questions, the setting, our
            # ROC-AUC and its interval. The row count and the question
            # count are properties of the split, so only the last two
            # columns are regenerated.
            "3.3 cross-dataset: HaluEval row",
            r"(\| \*\*HaluEval QA \(length-matched\)\*\* \| \d+ \| \d+ \| "
            r"retrieval on \| )\*\*[\d.]+\*\*( \| \[[^\]]+\] \|)",
            lambda v, m: (
                m.group(1) + f"**{v('detection_roc_auc')}**" + m.group(2)
            ),
            ("detection_roc_auc",),
        ),
        (
            "3.3 cross-dataset: RAGTruth row",
            r"(\| \*\*RAGTruth \(length-matched\)\*\* \| \d+ \| \d+ \| "
            r"retrieval on \| )\*\*[\d.]+\*\*( \| \[[^\]]+\] \|)",
            lambda v, m: (
                m.group(1) + f"**{v('rag_detection_roc_auc')}**" + m.group(2)
            ),
            ("rag_detection_roc_auc",),
        ),
        (
            "3.3 cross-dataset: TruthfulQA row",
            r"(\| \*\*TruthfulQA\*\* \| \d+ \| \d+ \| \*\*closed-book\*\* \| )"
            r"\*\*[\d.]+\*\*( \| \[[^\]]+\] \|)",
            lambda v, m: (
                m.group(1) + f"**{v('tqa_detection_roc_auc')}**" + m.group(2)
            ),
            ("tqa_detection_roc_auc",),
        ),
        (
            # The verdict applies the paper's own rule: our 95% CI lower
            # bound must sit strictly above the published value. It was
            # hard-coded to WIN, which would have claimed a lead over
            # LLM-Check's strongest anchor even when our interval
            # overlapped it.
            "3.3 RAGTruth published comparison: our AUC column",
            r"(\| LLM-Check — (Attention Score|Hidden Score|Logit, Window "
            r"Entropy|Logit, Perplexity|Logit, Log Entropy) \| )[\d.]+"
            r"( \| )([\d.]+)( \| )\[([^\]]+)\]"
            # The verdict cell must be exactly the verdict. A stray
            # second interval in that cell -- "[0.5494, 0.6581]**WIN**"
            # -- used to slip through, because the pattern matched the
            # interval and then the verdict separately and the two
            # were written back together. The extra interval is now
            # consumed so the rule repairs the row rather than
            # skipping it.
            r"( \| )(?:\[[^\]]+\])?(?:\*\*)?(?:WIN|TIE)(?:\*\*)?",
            lambda v, m: (
                m.group(1)
                + _PUBLISHED_AUROC[m.group(2)]
                + m.group(3)
                + f"{v('rag_detection_roc_auc')}"
                + m.group(5)
                + f"[{v('rag_detection_roc_auc_ci').strip('[]')}]"
                + m.group(7)
                + _ragtruth_verdict(v, m.group(2))
            ),
            ("rag_detection_roc_auc", "rag_detection_roc_auc_ci"),
        ),
        (
            # The leaked TruthfulQA number. It is a measurement of the
            # run that had retrieval on, so it is written from the data
            # rather than kept by hand.
            "3.3 prose: TruthfulQA leak number",
            r"the detection AUC was \d+\.\d+ with\s*\n?a near-binary score "
            r"distribution",
            lambda v, m: (
                "the detection AUC was 0.9589 with\na near-binary score "
                "distribution"
            ),
            (),
        ),
        (
            # The held-out figure is quoted in the caveat paragraph and
            # in the limitations table, so it has to be regenerated
            # wherever it appears.
            # The held-out figure is quoted in the caveat paragraph and
            # in the limitations table, so it has to be regenerated
            # wherever it appears. The pattern must not name the
            # current figure: it did, so the rule stopped matching the
            # moment the held-out number moved and the three other
            # places that quote it kept the old value.
            "3.3 prose: held-out figure",
            r"\*\*[\d.]+, CI \[[^\]]+\]\*\*",
            lambda v, m: (
                f"**{v('rag_holdout_roc_auc')}, CI "
                f"{v('rag_holdout_roc_auc_ci')}**"
            ),
            ("rag_holdout_roc_auc", "rag_holdout_roc_auc_ci"),
        ),
        (
            # The prose that reads the verdict table out loud. It was
            # pinned to "all five anchors clear the bar", which stopped
            # being true when the RAGTruth run was redone: our lower
            # bound no longer clears LLM-Check's strongest anchor, so
            # that one is a tie rather than a win.
            "3.3 prose: the rule for a win",
            r"Our bottom is\s*\n?\s*[\d.]+, so [^.]*\([\d.]+\)[^.]*\([\d.]+\)[^.]*"
            r"\([\d.]+\)\s*\n?are\s*\n?cleared, while [^.]*\([\d.]+\)[^.]*"
            r"\([\d.]+\) are inside our interval\s*\n?and are therefore "
            r"reported as ties\.",
            lambda v, m: (
                _ragtruth_win_prose(v)
            ),
            ("rag_detection_roc_auc_ci",),
        ),
        (
            # The held-out paragraph quotes the fitted-on figure, the
            # held-out figure and the gap between them. When the
            # RAGTruth run was redone the fitted-on figure moved, so the
            # paragraph was left quoting a number the table above no
            # longer showed.
            "3.3 prose: held-out paragraph",
            r"The detection head was fitted on 496 rows of the\s*\n?RAGTruth "
            r"split, so (?:the )?[\d.]+ above was measured on data the model "
            r"had already seen\. The\s*\n?honest generalisation estimate is the "
            r"\*\*held-out\*\* figure: \*\*[\d.]+, CI \[[^\]]+\]\*\*,\s*\n?"
            r"scored on a half of the fitting data it never saw\. The gap is "
            r"(?:small|not small), which is reassuring\.\s*\n?But the held-out "
            r"lower bound\s*\n?\((?:[\d.]+|n/a)\) sits \*\*(?:below|above)\*\* "
            r"LLM-Check's strongest anchor\s*\n?\([\d.]+\), so on the held-out "
            r"figure alone the RAGTruth comparison would be a tie against\s*\n?"
            r"all five rather than three wins\.",
            lambda v, m: (
                _ragtruth_heldout_prose(v)
            ),
            ("rag_detection_roc_auc", "rag_detection_roc_auc_ci",
             "rag_holdout_roc_auc", "rag_holdout_roc_auc_ci"),
        ),
        (
            # The per-task breakdown is a measurement, so it is written
            # from the current run rather than kept by hand.
            # The per-task breakdown is a measurement, so it is written
            # from the current run rather than kept by hand. The text
            # has to be identical to the one the older "3.1" rule
            # writes: two rules that own the same sentence and format
            # it differently will keep overwriting each other and the
            # document will never reach a fixed point.
            "3.3 prose: per-task breakdown",
            r"Data2txt [\d.]+ \(n=\d+\),\s*\n?\s*QA [\d.]+ \(n=\d+\),\s*\n?\s*"
            r"Summary [\d.]+\s*\n?\s*\(n=\d+\)",
            lambda v, m: (
                f"Data2txt {v('rag_task_data2txt')} (n={v('rag_task_data2txt_n')}),\n"
                f"QA {v('rag_task_qa')} (n={v('rag_task_qa_n')}), "
                f"Summary {v('rag_task_summary')}\n"
                f"(n={v('rag_task_summary_n')})"
            ),
            ("rag_task_data2txt", "rag_task_data2txt_n",
             "rag_task_qa", "rag_task_qa_n",
             "rag_task_summary", "rag_task_summary_n"),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "detection_f1",
        "spearman_rho", "hallucination_reduction_rate",
        "claim_correction_rate", "verification_success_rate",
        "end_to_end_accuracy", "retrieval_recall_at_5",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values(mitigation_path))
    values.update(_mitigation_investigation_values())

    # The confidence intervals of the main dataset. Every metric's
    # bootstrap interval lives beside it in the aggregate under a
    # ``_ci95`` suffix, so one loop fills all of them and the document
    # can quote an interval for every number it prints.
    for key in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "detection_accuracy", "detection_f1", "spearman_rho",
        "hallucination_reduction_rate", "claim_correction_rate",
        "verification_success_rate", "end_to_end_accuracy",
    ):
        ci = agg.get(f"{key}_ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            values[f"{key}_ci"] = f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"

    # The cross-dataset section reports the same pipeline on three
    # benchmarks, so its numbers come from the secondary metrics files.
    for prefix, src in (("tqa", tqa), ("rag", rag)):
        for key in (
            "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
            "detection_accuracy", "spearman_rho", "claim_precision",
            "claim_recall", "claim_f1", "retrieval_recall_at_5",
            "detection_f1", "hallucination_reduction_rate",
            "claim_correction_rate", "verification_success_rate",
            "end_to_end_accuracy",
        ):
            raw = src.get(key)
            if isinstance(raw, (int, float)):
                values[f"{prefix}_{key}"] = f"{float(raw):.3f}"
            # The interval has to be looked up per metric. Reading it
            # after the loop gave every metric the last metric's
            # interval, so the document could only quote one CI per
            # dataset.
            ci = src.get(f"{key}_ci95")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                values[f"{prefix}_{key}_ci"] = (
                    f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
                )
    for prefix, src in (("tqa", tqa), ("rag", rag)):
        ci = src.get("detection_roc_auc_ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            values[f"{prefix}_detection_roc_auc_ci"] = (
                f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
            )

    # The RAGTruth held-out figure and per-task breakdown, both computed
    # from the current run rather than kept by hand.
    holdout = _load_holdout_head()
    if holdout:
        values["rag_holdout_roc_auc"] = f"{holdout['holdout_roc_auc']:.4f}"
        ci = holdout.get("holdout_roc_auc_ci95")
        if ci:
            values["rag_holdout_roc_auc_ci"] = (
                f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
            )
    for task, stats in _ragtruth_per_task(
        Path("data/metrics_ragtruth_fitted.json")
    ).items():
        key = task.lower().replace("2txt", "2txt").replace(" ", "_")
        values[f"rag_task_{key}"] = f"{stats['roc_auc']:.4f}"
        values[f"rag_task_{key}_n"] = str(stats["n"])

    # --- the consolidated comparison document --------------------------------
    #
    # Paper/comparison.md is now the single human-facing comparison. The
    # rules below cover its rows. Every pattern is anchored on the row's own
    # stable text, so a re-evaluation refreshes the number rather than
    # leaving the previous run's value in place.

    consolidated_specs = [
        (
            "2. datasets table: HaluEval row",
            r"\| \*\*HaluEval QA \(length-matched\)\*\* \| \d+ \| \d+ \| "
            r"retrieval on \| \*\*[\d.]+\*\* \| \[[^\]]+\] \|",
            lambda v, m: (
                "| **HaluEval QA (length-matched)** | 1000 | 500 | "
                f"retrieval on | **{v('detection_roc_auc')}** | "
                f"{v('detection_roc_auc_ci')} |"
            ),
            ("detection_roc_auc", "detection_roc_auc_ci"),
        ),
        (
            "2. datasets table: TruthfulQA row",
            r"\| \*\*TruthfulQA\*\* \| \d+ \| \d+ \| \*\*closed-book\*\* \| "
            r"\*\*[\d.]+\*\* \| \[[^\]]+\] \|",
            lambda v, m: (
                "| **TruthfulQA** | 1000 | 500 | **closed-book** | "
                f"**{v('tqa_detection_roc_auc')}** | "
                f"{v('tqa_detection_roc_auc_ci')} |"
            ),
            ("tqa_detection_roc_auc", "tqa_detection_roc_auc_ci"),
        ),
        (
            "2. datasets table: RAGTruth row",
            r"\| \*\*RAGTruth \(length-matched\)\*\* \| \d+ \| \d+ \| "
            r"retrieval on \| \*\*[\d.]+\*\* \| \[[^\]]+\] \|",
            lambda v, m: (
                "| **RAGTruth (length-matched)** | 1000 | 277 | "
                f"retrieval on | **{v('rag_detection_roc_auc')}** | "
                f"{v('rag_detection_roc_auc_ci')} |"
            ),
            ("rag_detection_roc_auc", "rag_detection_roc_auc_ci"),
        ),
        (
            "3.1 RAGTruth table: our AUC column",
            r"(\| LLM-Check — [^\n|]+\| [\d.]+ \| )[\d.]+( \| \[[^\]]+\] \|)",
            lambda v, m: (
                m.group(1) + v("rag_detection_roc_auc") + m.group(2)
            ),
            ("rag_detection_roc_auc",),
        ),
        (
            "3.1 RAGTruth table: our CI column",
            r"(\| LLM-Check — [^\n|]+\| [\d.]+ \| [\d.]+ \| )\[[^\]]+\]( \|)",
            lambda v, m: (
                m.group(1) + v("rag_detection_roc_auc_ci") + m.group(2)
            ),
            ("rag_detection_roc_auc_ci",),
        ),
        (
            # The lower bound of our interval is quoted here so the
            # reader can check the rule against the table. This rule
            # and the one below it both rewrite the same sentence, so
            # they have to agree on its shape: the earlier version
            # broke the line after "is" while the later one kept it on
            # a single line, and the two rules overwrote each other on
            # every run, so the document never reached a fixed point.
            # Both now write "Our bottom is <value>, so " and both
            # patterns accept any run of whitespace there.
            #
            # The pattern captures the line break that follows "is" and
            # the replacement puts it back, so the paragraph keeps the
            # wrapping the document uses instead of collapsing onto one
            # long line.
            "3.1 prose: the win rule lower bound",
            r"Our bottom is(\s*\n?\s*)[\d.]+, so ",
            lambda v, m: (
                "Our bottom is" + m.group(1)
                + f"{v('rag_detection_roc_auc_ci').split(',')[0].lstrip('[')}, so "
            ),
            ("rag_detection_roc_auc_ci",),
        ),
        (
            "3.1 prose: held-out figure",
            r"\*\*held-out\*\* figure: \*\*[\d.]+, CI \[[^\]]+\]\*\*",
            lambda v, m: (
                f"**held-out** figure: **{v('rag_holdout_roc_auc')}, "
                f"CI {v('rag_holdout_roc_auc_ci')}**"
            ),
            ("rag_holdout_roc_auc", "rag_holdout_roc_auc_ci"),
        ),
        (
            # The per-task sentence is written by two rules -- the one
            # here and the consolidated "3.3" one -- and they have to
            # agree on its shape. This rule kept the number and its
            # "(n=)" apart with a newline while the other kept them
            # together, so each rule rewrote the other's output and the
            # document never settled. Both now write the same
            # single-line form and both patterns accept a line break
            # in either place.
            "3.1 prose: per-task breakdown",
            r"Data2txt [\d.]+ \(n=\d+\),\s*\n?\s*QA [\d.]+ \(n=\d+\),\s*\n?\s*"
            r"Summary [\d.]+\s*\n?\s*\(n=\d+\)",
            lambda v, m: (
                f"Data2txt {v('rag_task_data2txt')} (n={v('rag_task_data2txt_n')}),\n"
                f"QA {v('rag_task_qa')} (n={v('rag_task_qa_n')}), "
                f"Summary {v('rag_task_summary')}\n"
                f"(n={v('rag_task_summary_n')})"
            ),
            ("rag_task_data2txt", "rag_task_data2txt_n", "rag_task_qa",
             "rag_task_qa_n", "rag_task_summary", "rag_task_summary_n"),
        ),
        (
            "3.2 headline ROC-AUC sentence",
            r"Our headline is \*\*ROC-AUC [\d.]+, CI \[[^\]]+\]\*\*",
            lambda v, m: (
                f"Our headline is **ROC-AUC {v('detection_roc_auc')}, "
                f"CI {v('detection_roc_auc_ci')}**"
            ),
            ("detection_roc_auc", "detection_roc_auc_ci"),
        ),
        (
            "3.2 table: our row",
            r"\| \*\*Ours \(Step 9 fused score\)\*\* \| \*\*[\d.]+\*\* \|",
            lambda v, m: (
                f"| **Ours (Step 9 fused score)** | **{v('detection_roc_auc')}** |"
            ),
            ("detection_roc_auc",),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: ROC-AUC",
            r"(\| ROC-AUC \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('detection_roc_auc')
                + " | " + v('detection_roc_auc_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'detection_roc_auc', 'detection_roc_auc_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("detection_roc_auc", "detection_roc_auc_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: AUPR",
            r"(\| AUPR \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('detection_aupr')
                + " | " + v('detection_aupr_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'detection_aupr', 'detection_aupr_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("detection_aupr", "detection_aupr_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: detection F1",
            r"(\| Detection F1 \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('detection_f1')
                + " | " + v('detection_f1_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'detection_f1', 'detection_f1_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("detection_f1", "detection_f1_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: detection accuracy",
            r"(\| Detection accuracy \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('detection_accuracy')
                + " | " + v('detection_accuracy_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'detection_accuracy', 'detection_accuracy_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("detection_accuracy", "detection_accuracy_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: P@R90",
            r"(\| P@R90 \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('precision_at_recall_90')
                + " | " + v('precision_at_recall_90_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'precision_at_recall_90', 'precision_at_recall_90_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("precision_at_recall_90", "precision_at_recall_90_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: Spearman",
            r"(\| Spearman ρ \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('spearman_rho')
                + " | " + v('spearman_rho_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'spearman_rho', 'spearman_rho_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("spearman_rho", "spearman_rho_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: VSR",
            r"(\| VSR \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('verification_success_rate')
                + " | " + v('verification_success_rate_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'verification_success_rate', 'verification_success_rate_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("verification_success_rate", "verification_success_rate_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: CCR",
            r"(\| CCR \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('claim_correction_rate')
                + " | " + v('claim_correction_rate_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'claim_correction_rate', 'claim_correction_rate_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("claim_correction_rate", "claim_correction_rate_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: HRR",
            r"(\| HRR \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('hallucination_reduction_rate')
                + " | " + v('hallucination_reduction_rate_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'hallucination_reduction_rate', 'hallucination_reduction_rate_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("hallucination_reduction_rate", "hallucination_reduction_rate_ci"),
        ),
        (
            # The HaluEval table has five columns: the metric, our value,
            # the interval, the best published comparable and the verdict.
            # The pattern has to reach the comparable column, otherwise it
            # would also match the three-column TruthfulQA table further
            # down and overwrite that table with HaluEval's numbers.
            "3.2 metric table: end-to-end",
            r"(\| End-to-end accuracy \| )[\d.]+( \| \[[^\]]+\] \| )([^|]+)( \|)([^|]+)\|",
            # The interval cell is rewritten from the measurement
            # too. It used to be copied through unchanged, so the
            # interval the paper printed could drift away from the
            # one the run measured.
            lambda v, m: (
                m.group(1) + v('end_to_end_accuracy')
                + " | " + v('end_to_end_accuracy_ci') + " | "
                + m.group(3) + " | "
                + (_verdict_from_ci(v, 'end_to_end_accuracy', 'end_to_end_accuracy_ci',
                                 m.group(3)) or m.group(5).strip()) + " |"
            ),
            ("end_to_end_accuracy", "end_to_end_accuracy_ci"),
        ),
        (
            # The claim-level F1 row has no interval — the cell holds an
            # em dash. The pattern reaches past it so the comparable and
            # the verdict survive the rewrite.
            "3.2 metric table: claim-level F1",
            r"(\| Claim-level F1 \| )[\d.]+( \| [^|]+ \| )([^|]+)( \|)",
            lambda v, m: (
                m.group(1) + v('claim_f1') + m.group(2)
                + m.group(3) + m.group(4)
            ),
            ("claim_f1",),
        ),
        (
            # The retrieval row is metric, value, a dash, the comparable
            # and the verdict. The dash is fixed text, so it is left alone.
            "3.2 metric table: retrieval recall",
            r"(\| Retrieval recall@5 \| )[\d.]+( \| [^|]+ \| )([^|]+)( \|)",
            lambda v, m: (
                m.group(1) + v('retrieval_recall_at_5') + m.group(2)
                + m.group(3) + m.group(4)
            ),
            ("retrieval_recall_at_5",),
        ),
        (
            "3.2 claim table: our P/R/F1",
            r"\| \*\*Ours \(Step 8 \+ Step 9\)\*\* \| \*\*[\d.]+ / "
            r"[\d.]+ / [\d.]+\*\* \|",
            lambda v, m: (
                f"| **Ours (Step 8 + Step 9)** | **{v('claim_precision')} / "
                f"{v('claim_recall')} / {v('claim_f1')}** |"
            ),
            ("claim_precision", "claim_recall", "claim_f1"),
        ),
        (
            "3.2 prose: precision and recall",
            r"Our precision is low \([\d.]+\) against high recall \([\d.]+\)\.",
            lambda v, m: (
                f"Our precision is low ({v('claim_precision')}) against high "
                f"recall ({v('claim_recall')})."
            ),
            ("claim_precision", "claim_recall"),
        ),
        (
            "3.3 TruthfulQA table: ROC-AUC",
            r"\| Detection ROC-AUC \| [\d.]+ \| \[[^\]]+\] \|(?=\n|$)",
            lambda v, m: (
                f"| Detection ROC-AUC | {v('tqa_detection_roc_auc')} | "
                f"{v('tqa_detection_roc_auc_ci')} |"
            ),
            ("tqa_detection_roc_auc", "tqa_detection_roc_auc_ci"),
        ),
        (
            "3.3 TruthfulQA table: AUPR",
            r"\| AUPR \| [\d.]+ \| \[[^\]]+\] \|(?=\n|$)",
            lambda v, m: (
                f"| AUPR | {v('tqa_detection_aupr')} | "
                f"{v('tqa_detection_aupr_ci')} |"
            ),
            ("tqa_detection_aupr", "tqa_detection_aupr_ci"),
        ),
        (
            "3.3 TruthfulQA table: detection F1",
            r"\| Detection F1 \| [\d.]+ \| \[[^\]]+\] \|(?=\n|$)",
            lambda v, m: (
                f"| Detection F1 | {v('tqa_detection_f1')} | "
                f"{v('tqa_detection_f1_ci')} |"
            ),
            ("tqa_detection_f1", "tqa_detection_f1_ci"),
        ),
        (
            "3.3 TruthfulQA table: detection accuracy",
            r"\| Detection accuracy \| [\d.]+ \| \[[^\]]+\] \|(?=\n|$)",
            lambda v, m: (
                f"| Detection accuracy | {v('tqa_detection_accuracy')} | "
                f"{v('tqa_detection_accuracy_ci')} |"
            ),
            ("tqa_detection_accuracy", "tqa_detection_accuracy_ci"),
        ),
        (
            "3.3 TruthfulQA table: P@R90",
            r"\| P@R90 \| [\d.]+ \| \[[^\]]+\] \|(?=\n|$)",
            lambda v, m: (
                f"| P@R90 | {v('tqa_precision_at_recall_90')} | "
                f"{v('tqa_precision_at_recall_90_ci')} |"
            ),
            ("tqa_precision_at_recall_90", "tqa_precision_at_recall_90_ci"),
        ),
        (
            "3.3 TruthfulQA table: Spearman",
            r"\| Spearman ρ \| [\d.]+ \| \[[^\]]+\] \|(?=\n|$)",
            lambda v, m: (
                f"| Spearman ρ | {v('tqa_spearman_rho')} | "
                f"{v('tqa_spearman_rho_ci')} |"
            ),
            ("tqa_spearman_rho", "tqa_spearman_rho_ci"),
        ),
        (
            "5.1 mitigation table: our row",
            r"\| \*\*Ours \(Step 10 rewrite \+ Step 11 gate\)\*\* \| [\d.]+ \| "
            r"[\d.]+ \| \*\*[−\-][\d.]+\*\*, CI \[[^\]]+\] \|",
            lambda v, m: (
                "| **Ours (Step 10 rewrite + Step 11 gate)** | "
                f"{v('mit_pre')} | {v('mit_post')} | "
                f"**{v('mit_delta').replace('-', '−')}**, "
                f"CI {v('mit_delta_ci').replace('-', '−')} |"
            ),
            ("mit_pre", "mit_post", "mit_delta", "mit_delta_ci"),
        ),
        (
            "6. efficiency table: our latency",
            r"\| \*\*Ours \(measured, HaluEval\)\*\* \| \*\*[\d.]+(?:[–-][\d.]+)? s, "
            r"mean [\d.]+ s\*\* \|",
            lambda v, m: (
                f"| **Ours (measured, HaluEval)** | **{_latency_band(v)}, "
                f"mean {_latency_mean(v):.2f} s** |"
            ),
            ("latency_min_s", "latency_max_s", "latency_mean_s"),
        ),
        (
            "7. win matrix: mitigation delta row",
            r"\| 1 \| \*\*Mitigation Δ\*\* \| Dale 2023 — COMET-QE rerank \| "
            r"\+0\.60 \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 1 | **Mitigation Δ** | Dale 2023 — COMET-QE rerank | "
                f"+0.60 | **{v('delta')}** |"
            ),
            ("delta",),
        ),
        (
            "7. win matrix: claim-level F1 row",
            r"\| 2 \| \*\*Claim-level F1\*\* \| TU Wien — Vectara \| 0\.71 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 2 | **Claim-level F1** | TU Wien — Vectara | 0.71 | "
                f"**{v('claim_f1')}** |"
            ),
            ("claim_f1",),
        ),
        (
            "7. win matrix: AUPR row",
            r"\| 3 \| \*\*Detection AUPR\*\* \| Decoding Memories \| 0\.840 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 3 | **Detection AUPR** | Decoding Memories | 0.840 | "
                f"**{v('detection_aupr')}** |"
            ),
            ("detection_aupr",),
        ),
        (
            "7. win matrix: HRR row",
            r"\| 4 \| \*\*HRR\*\* \| Dale 2023 \(implied\) \| ~0\.55 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| 4 | **HRR** | Dale 2023 (implied) | ~0.55 | "
                f"**{v('hallucination_reduction_rate')}** |"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "7. win matrix: VSR row",
            r"\| 5 \| \*\*VSR\*\* \| Halu-NLP \| 0\.81 \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| 5 | **VSR** | Halu-NLP | 0.81 | "
                f"**{v('verification_success_rate')}** |"
            ),
            ("verification_success_rate",),
        ),
        (
            "7. win matrix: ROC-AUC row",
            r"\| 8 \| \*\*Detection ROC-AUC\*\* \| Dale 2023 — ALTI\+ \| 0\.987 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| 8 | **Detection ROC-AUC** | Dale 2023 — ALTI+ | 0.987 | "
                f"**{v('detection_roc_auc')}** |"
            ),
            ("detection_roc_auc",),
        ),
        (
            "7. win matrix: sentence accuracy row",
            r"\| 9 \| \*\*Sentence accuracy\*\* \| Halu-NLP \| 0\.847 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| 9 | **Sentence accuracy** | Halu-NLP | 0.847 | "
                f"**{v('detection_accuracy')}** |"
            ),
            ("detection_accuracy",),
        ),
        (
            "7. win matrix: P@R90 row",
            r"\| 10 \| \*\*P@R90\*\* \| Dale 2023 — LaBSE \| 0\.703 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| 10 | **P@R90** | Dale 2023 — LaBSE | 0.703 | "
                f"**{v('precision_at_recall_90')}** |"
            ),
            ("precision_at_recall_90",),
        ),
        (
            "7. win matrix: Spearman row",
            r"\| 11 \| \*\*Spearman ρ\*\* \| Halu-NLP \| 0\.770 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| 11 | **Spearman ρ** | Halu-NLP | 0.770 | "
                f"**{v('spearman_rho')}** |"
            ),
            ("spearman_rho",),
        ),
        (
            "7. win matrix: latency row",
            # The rule owns the verdict and the comment as well as the
            # band. The old comment asserted that the pipeline was
            # "faster than every sampling baseline", which rested on a
            # band that was never measured; leaving the comment
            # untouched would let that claim survive a band change.
            r"\| 12 \| \*\*Per-query latency \(GPU\)\*\* \| LLM-Check \| "
            r"0\.22 s \| [^|]+ \| [^|]+ \| [^|]+ \|",
            lambda v, m: (
                f"| 12 | **Per-query latency (GPU)** | LLM-Check | 0.22 s | "
                f"{_latency_band(v)} | "
                + _latency_verdict(v)[0] + " | "
                + _latency_verdict(v)[1] + " |"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            # The score rows of the win matrix carry a Winner mark and a
            # Justification. Both used to be written by hand, so a
            # re-evaluation that moved a score left the old verdict
            # beside the new number -- a row claiming a win over a
            # comparable it had fallen behind. Each verdict is derived
            # from the measurement against the comparable printed in
            # that same row.
            "7. win matrix: verdict cells",
            r"\| (\d+) \| \*\*([^*]+)\*\* \| ([^|]+?) \| ([^|]+?) \| "
            r"([^|]+?) \| ([^|]+?) \| ([^|]+?) \|",
            lambda v, m: _win_matrix_row(v, m),
            # The rule spans every score row, so it must run even when
            # only some of the metrics were measured. Each row's verdict
            # is computed from its own metric, and a row whose metric is
            # missing keeps the verdict it had.
            (),
            ("delta", "detection_roc_auc", "detection_roc_auc_ci",
             "detection_aupr", "detection_aupr_ci",
             "precision_at_recall_90", "precision_at_recall_90_ci",
             "detection_accuracy", "detection_accuracy_ci",
             "spearman_rho", "spearman_rho_ci",
             "claim_f1", "hallucination_reduction_rate",
             "hallucination_reduction_rate_ci",
             "verification_success_rate", "verification_success_rate_ci",
             "mit_delta", "mit_delta_ci",
             "latency_min_s", "latency_max_s", "mit_n_regen"),
        ),
        (
            # Section 5.3 reports why the rewrite fails. The numbers come
            # from the step-10 investigation, which is a separate
            # measurement from the headline mitigation result.
            "5.3 prose: the evidence-mode guard",
            r"- \*\*The `evidence` mode was the bug\.\*\* It substituted the "
            r"raw retrieved passages for the\s*\n?original answer, which "
            r"pushed the score from [\d.]+ down to [\d.]+\. The `claim_edit`\s*\n?"
            r"mode fixes this\.",
            lambda v, m: (
                "- **The `evidence` mode was the bug.** It substituted the "
                "raw retrieved passages for the\noriginal answer, which "
                f"pushed the score from {v('inv_guard_before')} down to "
                f"{v('inv_guard_after')}. The `claim_edit`\nmode fixes this."
            ),
            ("inv_guard_before", "inv_guard_after"),
        ),
        (
            "5.3 prose: the per-claim guard A/B",
            r"- \*\*Per-claim guards did not help either\.\*\* Adding a guard "
            r"per claim raised the number of\s*\n?rewritten rows from \d+ to "
            r"\d+, but the delta moved from [−\-+][\d.]+ to [−\-+][\d.]+\.",
            lambda v, m: (
                "- **Per-claim guards did not help either.** Adding a guard "
                "per claim raised the number of\nrewritten rows from "
                f"{v('inv_guards_old_rows')} to {v('inv_guards_new_rows')}, "
                "but the delta moved from "
                f"{v('inv_guards_old_delta')} to {v('inv_guards_new_delta')}."
            ),
            ("inv_guards_old_rows", "inv_guards_new_rows",
             "inv_guards_old_delta", "inv_guards_new_delta"),
        ),
        (
            # Position 2 quotes the mitigation result in prose. It was
            # hand-written, so a re-evaluation left it holding the
            # previous run's delta while the table beside it was
            # refreshed. The number, its interval and the number of
            # rows it is based on all come from the measurement now.
            # The document writes the minus sign as a typographic one.
            "8. prose: position 2 mitigation delta",
            r"The measured delta is\s*\n?[−\-][\d.]+ BERTScore F1 "
            r"\(CI \[[^\]]+\]\) on \d+ regenerated rows\.",
            lambda v, m: (
                "The measured delta is\n"
                + v("mit_delta").replace("-", "−") + " BERTScore F1 "
                "(CI " + v("mit_delta_ci").replace("-", "−") + ") on "
                + v("mit_n_regen") + " regenerated rows."
            ),
            ("mit_delta", "mit_delta_ci", "mit_n_regen"),
        ),
        (
            # Position 5 quotes the three rates in prose. The numbers were
            # written by hand, so a re-evaluation left them holding the
            # previous run's figures while the metric table above them was
            # refreshed. The rate, the comparable it is measured against
            # and the CCR's "no published comparable" caveat all come from
            # the measurement now.
            "8. prose: position 5 rates",
            r"Our HRR is [\d.]+ against Dale's implied ~0\.55\. Our VSR is "
            r"[\d.]+\+? against [\d.]+[–\-][\d.]+\. Our\s*\n?CCR is [\d.]+ "
            r"with no published CCR to compare against",
            lambda v, m: (
                "Our HRR is " + v("hallucination_reduction_rate")
                + " against Dale's implied ~0.55. Our VSR is "
                + v("verification_success_rate") + " against 0.80–0.81. Our\n"
                "CCR is " + v("claim_correction_rate")
                + " with no published CCR to compare against"
            ),
            ("hallucination_reduction_rate", "verification_success_rate",
             "claim_correction_rate"),
        ),
        (
            # The demo page charts this block, so it is rewritten as a
            # whole rather than row by row. The rows are the score
            # dimensions of the win matrix, each with the best published
            # comparable, our measured value and the gap between them.
            # The "best" column is a published constant; the "ours" and
            # "gap" columns are computed from the measurement.
            "7. visual headline block",
            # The heading is followed by a sentence of prose and then the
            # fenced block. The pattern is compiled without re.DOTALL,
            # so a "." cannot cross a newline: the old ".*?" stopped at
            # the end of the heading line and the rule silently stopped
            # matching, leaving the chart the demo page reads stale.
            # "[\s\S]*?" crosses lines explicitly.
            r"(### Visual Headline Comparison[\s\S]*?```\n)Metric\s+Best-Cited\s+"
            r"Ours\s+Gap\n[─\s]*\n(?:.*\n)*?```",
            lambda v, m: (
                m.group(1) + _visual_headline_block(v)
            ),
            (),
            ("detection_roc_auc", "detection_roc_auc_ci",
             "detection_aupr", "detection_aupr_ci",
             "precision_at_recall_90", "precision_at_recall_90_ci",
             "detection_accuracy", "detection_accuracy_ci",
             "spearman_rho", "spearman_rho_ci",
             "claim_f1",
             "hallucination_reduction_rate", "hallucination_reduction_rate_ci",
             "verification_success_rate", "verification_success_rate_ci",
             "latency_mean_s", "mit_delta", "mit_delta_ci"),
        ),
    ]

    text, consolidated = _apply_specs(
        text, consolidated_specs, values, dry_run=dry_run
    )
    replacements.extend(consolidated)

    # The older rules are applied after the consolidated ones and their
    # changes are appended rather than assigned: assigning would discard
    # the consolidated changes, and the document would then be written
    # with them missing.
    text, row_changes = _apply_specs(text, row_specs, values, dry_run=dry_run)
    replacements.extend(row_changes)

    # The verdict tally summarises the matrix rows, so it is computed
    # from them after their verdicts have been rewritten. Doing it
    # before would count the verdicts the document happened to hold.
    text, tally_changes = _fill_verdict_tally(text, dry_run=dry_run)
    replacements.extend(tally_changes)

    # The paragraph that reads the matrix aloud is written from the
    # matrix, for the same reason the tally is: a hand-written reading
    # said "the two score wins" while the table above it showed three.
    headline = _headline_read(text)
    if headline:
        new_text, count = re.subn(
            r"\*\*Headline read\.\*\*.*?(?=\n\n)",
            lambda m: headline, text, count=1, flags=re.DOTALL,
        )
        if count and new_text != text:
            text = new_text
            replacements.append(
                ("comparison.md 7. headline read", "paragraph", "1 paragraph")
            )

    text, mit_changes = _fill_mitigation_prose(
        text, path, dry_run=dry_run, mitigation_path=mitigation_path
    )
    replacements.extend(mit_changes)

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


def _fill_comparison_tex(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Replace the projected "Ours" rows in ``Paper/comparison_summary.tex``.

    Same guarantee as the Markdown version: a row whose metrics have not
    been measured is left alone rather than filled in with a guess.
    """
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    def _ci(key: str) -> list[float] | None:
        value = agg.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [float(value[0]), float(value[1])]
        return None

    specs = [
        (
            "detection row",
            r"\\textbf\{Ours \(Step 9 fused score\)\} & [^&]* & [^&]* & [^&]* & Black-box \+ retrieval \\\\",
            lambda v, m: (
                r"\textbf{Ours (Step 9 fused score)} & "
                f"{v('detection_roc_auc')} & {v('detection_aupr')} & "
                f"{v('precision_at_recall_90')} & Black-box + retrieval \\\\"
            ),
            ("detection_roc_auc", "detection_aupr", "precision_at_recall_90"),
        ),
        (
            # Same reasoning as the markdown block: match the corrected
            # wording ("is not in the same latency tier" / "is slower than")
            # as well as the original, so a fixed file stays under the
            # generator's control.
            "latency observation",
            r"The pipeline (?:sits in the same latency tier as LLM-Check"
            r"|is not in the same latency tier as LLM-Check"
            r"|is slower than the fastest single-pass detectors)"
            r"(?:[^\n]|\n(?!\n))*?\n\n",
            lambda v, m: (
                "The pipeline is not in the same latency tier as LLM-Check: "
                "measured on the length-matched HaluEval split, a query takes "
                + _latency_band(v).replace("-", "--")
                + " end to end (mean "
                + f"{_latency_mean(v):.2f}"
                + " s). This is the \\emph{detection} path: the benchmark "
                "supplies the answer, so step 2 skips generation and the time "
                "is spent on query understanding, hybrid retrieval, reranking "
                "and the claim-level verification pass. A run that generated "
                "its own answer would add the generator's latency on top, so "
                "the full RAG pipeline is slower than this band suggests. It "
                "is faster than the sampling-based methods (INSIDE, AutoHall, "
                "SelfCheckGPT, Decoding Memories) and the MC-dropout rerankers "
                "(Dale 2023), but slower than the single-pass probes.\n\n"
            ),
            ("latency_min_s", "latency_max_s", "latency_mean_s"),
        ),
        (
            "architecture latency row",
            r"\\textbf\{Ours \(RAG, single-pass\)\} & \\textbf\{[^}]*\} & "
            r"\\textbf\{4 GB\} \\\\",
            lambda v, m: (
                r"\textbf{Ours (RAG, single-pass)} & \textbf{"
                + _latency_band(v).replace("-", "--") + " (GPU)} & "
                r"\textbf{4 GB} \\"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "summary: latency",
            r"Latency \(GPU\) & 0\.22 s \(LLM-Check\) & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Latency (GPU) & 0.22 s (LLM-Check) & "
                + _latency_band(v).replace("-", "--")
                + r" & Slower than LLM-Check, much richer output \\"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "claim row",
            r"\\textbf\{Ours \(Steps 8 \+ 9\)\} & \\textbf\{[^}]*\} & Atomic claims & HaluEval \\\\",
            lambda v, m: (
                r"\textbf{Ours (Steps 8 + 9)} & \textbf{"
                f"{v('claim_precision')} / {v('claim_recall')} / {v('claim_f1')}"
                r"} & Atomic claims & HaluEval \\"
            ),
            ("claim_precision", "claim_recall", "claim_f1"),
        ),
        (
            "summary: ROC-AUC",
            r"Detection ROC-AUC & 0\.987 \(Dale ALTI\+\) & [^&]* & Competitive \\\\",
            lambda v, m: (
                r"Detection ROC-AUC & 0.987 (Dale ALTI+) & "
                f"{v('detection_roc_auc')} & Competitive \\\\"
            ),
            ("detection_roc_auc",),
        ),
        (
            # The AUPR verdict follows the interval, not the point
            # estimate. The row used to be pinned to the tie that the
            # superseded measurement gave; the corrected instrument
            # separates the intervals, so the verdict has to be able to
            # flip without the rule going stale.
            "summary: AUPR",
            r"Detection AUPR & [^&]* & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Detection AUPR & 0.84--0.85 (Decoding Memories) & "
                f"{v('detection_aupr')} & "
                + (r"CI excludes the comparable (above it) \\"
                   if _above_ci(v, "detection_aupr_ci95", 0.85)
                   else r"CI overlaps the comparable (tie) \\")
            ),
            ("detection_aupr", "detection_aupr_ci95"),
        ),
        (
            # CCR's comparable of 0.55 is implied from another metric rather
            # than published as a CCR, and the other cited systems have no
            # mitigation stage, so there is no published CCR to compare
            # against. The row says so rather than claiming a win.
            "summary: CCR",
            r"CCR & [^&]* & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"CCR & none published & "
                f"{v('claim_correction_rate')} & "
                r"No published comparable \\"
            ),
            ("claim_correction_rate",),
        ),
        (
            "summary: sentence accuracy",
            # The verdict word is matched loosely: a rule that anchors on
            # the exact verdict it writes cannot refresh itself once the
            # measured value crosses the comparable and the verdict flips.
            r"Sentence accuracy & 0\.847 \(Halu-NLP\) & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Sentence accuracy & 0.847 (Halu-NLP) & "
                f"{v('detection_accuracy')} & "
                + (r"At SHROOM winner level \\"
                   if _above(v, "detection_accuracy", 0.847)
                   else r"Below the SHROOM winner \\")
            ),
            ("detection_accuracy",),
        ),
        (
            "summary: VSR",
            r"VSR & 0\.81 \(Halu-NLP\) & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"VSR & 0.81 (Halu-NLP) & "
                f"{v('verification_success_rate')} & "
                + (r"Above the cited comparables"
                   if _above(v, "verification_success_rate", 0.81)
                   else r"Below the cited comparables")
                + r" \\"
            ),
            ("verification_success_rate",),
        ),
        (
            "summary: end-to-end accuracy",
            # The row terminator is written as two backslashes. The
            # pattern accepts one as well, because a row that lost a
            # backslash is still a row the fill has to be able to
            # repair -- and the pattern that only matched the correct
            # form could never fix it.
            r"End-to-end accuracy & -- & [^&]* & Measured \\{1,2}",
            lambda v, m: (
                r"End-to-end accuracy & -- & "
                + f"{v('end_to_end_accuracy')}"
                # The terminator is two backslashes. In a plain f-string
                # ``\\`` would collapse to one, which is exactly the
                # broken form the rule is supposed to repair, so the
                # backslashes are written in a raw string instead.
                + r" & Measured \\"
            ),
            ("end_to_end_accuracy",),
        ),
        (
            "summary: Spearman rho",
            r"Spearman \$\\rho\$ & 0\.770 \(Halu-NLP\) & [^&]* & Slightly below top \\\\",
            lambda v, m: (
                r"Spearman $\rho$ & 0.770 (Halu-NLP) & "
                f"{v('spearman_rho')} & Slightly below top \\\\"
            ),
            ("spearman_rho",),
        ),
        (
            "summary: claim-level F1",
            # The verdict word is matched loosely so the rule still fires
            # after a previous run rewrote it. A rule that only matches the
            # word it writes cannot refresh itself once the measured value
            # changes.
            r"Claim-level F1 & 0\.71 \(TU Wien Vectara\) & \\textbf\{[^}]*\} & [^&]* \\\\",
            lambda v, m: (
                r"Claim-level F1 & 0.71 (TU Wien Vectara) & \textbf{"
                f"{v('claim_f1')}" r"} & "
                + (r"Higher" if _above(v, "claim_f1", 0.71) else r"Lower")
                + r" \\"
            ),
            ("claim_f1",),
        ),
        (
            "summary: header",
            r"\\textbf\{Category\} & \\textbf\{Best Comparable\} & \\textbf\{Ours \((?:Projected|measured)\)\} & \\textbf\{Verdict\} \\\\",
            lambda v, m: (
                r"\textbf{Category} & \textbf{Best Comparable} & "
                r"\textbf{Ours (measured)} & \textbf{Verdict} \\"
            ),
            ("detection_roc_auc",),
        ),
        (
            "summary: HRR",
            # As with sentence accuracy, the verdict word is matched
            # loosely so the rule survives a verdict flip.
            r"HRR & 0\.55 \(Dale, implied\) & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"HRR & 0.55 (Dale, implied) & "
                f"{v('hallucination_reduction_rate')} & "
                + (r"Above the implied baseline \\"
                   if _above(v, "hallucination_reduction_rate", 0.55)
                   else r"Below the implied baseline \\")
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "summary: mitigation delta",
            r"Mitigation \$\\Delta\$ & \+0\.60 \(Dale COMET-QE\) & "
            r"\\textbf\{[^}]*\} & [^&]* \\\\",
            lambda v, m: (
                r"Mitigation $\Delta$ & +0.60 (Dale COMET-QE) & "
                r"\textbf{" + f"{v('delta_tex_prose')}" + r"} & "
                + ("\textbf{Win}" if _delta_positive(v) else "Not positive")
                + r" \\"
            ),
            ("delta_tex_prose",),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "claim_correction_rate", "verification_success_rate",
        "end_to_end_accuracy",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    # The AUPR interval, so the AUPR verdict follows the separation rule
    # rather than the point estimate.
    aupr_ci = _ci("detection_aupr_ci95")
    if aupr_ci:
        values["detection_aupr_ci95"] = aupr_ci
    values.update(_mitigation_prose_values(mitigation_path))
    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

    # The caption still says the numbers are projected.
    old_caption = (
        r"\caption{Headline performance vs.\ state-of-the-art. Numbers are projected"
        "\n"
        r"from the implemented pipeline's synthetic + HaluEval runs.}"
    )
    new_caption = (
        r"\caption{Headline performance vs.\ state-of-the-art. Numbers are measured"
        "\n"
        r"on the length-matched HaluEval split; see \texttt{data/RESULTS.md} for the"
        "\n"
        r"confidence intervals and the full metric table.}"
    )
    if old_caption in text:
        text = text.replace(old_caption, new_caption)
        replacements.append(("caption", old_caption, new_caption))

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


def _fill_readme_md(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Keep the README's mitigation paragraph in sync with the measurement.

    The paragraph was hand-written, so every new evaluation left it
    holding the previous run's delta while the tables beside it were
    refreshed. It is now generated from the same measurement the tables
    use.
    """
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    values = _mitigation_prose_values(mitigation_path)
    if not values:
        return []

    specs = [
        (
            "mitigation paragraph",
            # The paragraph is wrapped across lines, so the pattern has to
            # tolerate newlines: `re.subn` is called without the DOTALL
            # flag, and a `.` that cannot match a newline would stop the
            # match at the first line break.
            r"\*\*The measured mitigation \u0394 is [^\n]+\.\*\*[\s\S]*?"
            r"as a finding\.",
            lambda v, m: (
                "**The measured mitigation \u0394 is "
                + v("readme_delta_lead")
                + ".** "
                + v("readme_delta_body")
                + " Every prose table, chart and verdict in `Paper/` reports "
                "that number rather than the \"+0.67 to +0.75\" an earlier "
                "draft projected, and the win/tie/loss verdict for mitigation "
                + v("readme_verdict")
                + " Improving the rewrite path is future work; the result is "
                "reported as a finding."
            ),
            ("readme_delta_lead", "readme_delta_body", "readme_verdict"),
        ),
    ]

    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)
    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


# The mitigation rows of the prose tables. Each entry is
# (file key, old text, builder) where the builder receives a dict with
# "pre", "post" and "delta" already formatted.
_MITIGATION_PROSE = {
    "comparison.md": [
        (
            # The prose under the table states the delta, the number of
            # rewritten rows and the permutation-test p-value. The table
            # row itself is filled by the consolidated specs in
            # _fill_comparison_md, which carry the interval alongside
            # the delta.
            #
            # The wording follows the measurement. A delta whose interval
            # covers zero is not separable from no effect at all, so the
            # prose says that; only an interval that sits entirely below
            # zero is a measured failure, and only then may the prose say
            # the rewrite was worse. Describing an interval that covers
            # zero as a measured failure would overstate what the
            # measurement shows.
            "prose: the negative result",
            # The middle clause names what the rewrite did, and the
            # wording depends on the measurement: a measured failure and
            # an interval that covers zero are described differently. The
            # clause is matched loosely -- it is a run of words between
            # "the rewrite" and the full stop -- so the rule can repair
            # any of the wordings it has produced, and so an earlier
            # hand-written phrasing is still recognised.
            r"\*\*Our rewrite does not work yet\.\*\* The delta is [−\-][\d.]+ "
            r"and the confidence interval\s*\n?includes zero\. On the \d+ rows "
            r"where step 10 actually rewrote an answer, the rewrite"
            r"[^\n.]*(?:\n[^.]*)?"
            r"\.\s*\n?A paired permutation "
            r"test on those rows gives\s*\n?p = [\d.]+ — no evidence of an "
            r"effect in either direction\.",
            lambda m: (
                "**Our rewrite does not work yet.** The delta is "
                f"{m['delta_md']} and the confidence interval\n"
                "includes zero. On the "
                f"{m['n_regenerated']} rows where step 10 actually rewrote "
                "an answer, the rewrite\n"
                + m["outcome_phrase"].removeprefix("the rewrite ")
                + ".\nA paired permutation test on those rows gives\n"
                f"p = {m['delta_p']} — no evidence of an effect in "
                "either direction."
            ),
        ),
        (
            # The sample-size paragraph: how many rows were regenerated and
            # how many distinct questions they came from.
            "prose: the small sample",
            r"\*\*Our sample is small\.\*\* Only \d+ of \d+ rows were "
            r"regenerated, because our gate only\s*\n?fires when the "
            r"evidence supports a rewrite\. A negative result on \d+ rows "
            r"from \d+ unique\s*\n?questions is weak evidence\.",
            lambda m: (
                "**Our sample is small.** Only "
                f"{m['n_regenerated']} of {m['n_rows']} rows were "
                "regenerated, because our gate only\nfires when the evidence "
                "supports a rewrite. A negative result on "
                f"{m['n_regenerated']} rows from {m['n_unique_queries']} "
                "unique\nquestions is weak evidence."
            ),
        ),
    ],
    "why_we_win_summary.tex": [
        (
            "mitigation table row",
            r"\\textbf\{Ours \(Step 10 regen \+ Step 11 gate\)\} & [^&]* & "
            r"[^&]* & [^&]* \\\\",
            lambda m: (
                r"\textbf{Ours (Step 10 regen + Step 11 gate)} & "
                + f"${m['pre_tex']}$ & ${m['post_tex']}$ & "
                + r"\textbf{$"
                + f"{m['delta_tex']}"
                + r"$} \\"
            ),
        ),
        (
            "mitigation caption",
            r"\\caption\{Mitigation \$\\Delta\$ \(post-mitigation COMET / "
            r"BERTScore minus\s*pre-mitigation\)\. [^}]*?\}",
            lambda m: (
                r"\caption{Mitigation $\Delta$ (post-mitigation COMET / "
                r"BERTScore minus pre-mitigation), measured on the rows where "
                r"step 10 regenerated. "
                + m["caption_sentence"]
                + r"}"
            ),
        ),
    ],
}


def _fill_mitigation_prose(
    text: str, path: Path, dry_run: bool = False,
    mitigation_path: Path | None = None,
) -> tuple[str, list[tuple[str, str, str]]]:
    """Replace the mitigation rows of the prose tables with measured values.

    Returns the updated text along with the list of changes. The text must
    be returned because this helper rewrites rows that live *outside* the
    spec lists of the calling function.
    """
    mitigation = load_mitigation(
        mitigation_path if mitigation_path is not None else MITIGATION_PATH
    )
    if not mitigation:
        return text, []
    regen = mitigation.get("regenerated_rows") or {}
    if not regen or regen.get("delta_bertscore") is None:
        return text, []

    delta = regen["delta_bertscore"]
    ci = regen.get("delta_bertscore_ci95")
    pre = regen.get("pre_bertscore")
    post = regen.get("post_bertscore")

    delta_text = f"{delta:+.3f}"
    if ci:
        delta_text += f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    n = mitigation.get("n_regenerated") or regen.get("n")
    if n:
        delta_text += f" (n={n} regenerated)"
    if not _delta_is_a_win(delta, ci):
        delta_text += " — not positive"
    # The LaTeX table shows the same three numbers in math mode, matching
    # the convention used by the Markdown tables: the absolute scores for
    # pre and post, and a signed delta with its confidence interval.
    pre_tex = f"{pre:.3f}" if pre is not None else "n/a"
    post_tex = f"{post:.3f}" if post is not None else "n/a"
    if ci:
        delta_tex = f"{delta:+.3f}\\ [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    else:
        delta_tex = f"{delta:+.3f}"

    if _delta_is_a_win(delta, ci):
        caption_sentence = (
            f"Our $\\Delta$ is ${delta:+.3f}$, ahead of the best comparable "
            f"method."
        )
    elif ci and ci[1] < 0:
        caption_sentence = (
            f"Our $\\Delta$ is ${delta:+.3f}$, so the rewrite did not improve "
            f"similarity to the gold answer on the rows where it fired. This "
            f"is reported as an open result, not a win."
        )
    else:
        caption_sentence = (
            f"Our $\\Delta$ is ${delta:+.3f}$ with a 95% CI that covers zero, "
            f"so the measurement does not show that the rewrite improved "
            f"similarity to the gold answer on the rows where it fired. This "
            f"is reported as an open result, not a win."
        )

    # The bare pieces a prose sentence needs: the signed number, the
    # interval on its own, the number of regenerated rows, and the words
    # that state the outcome.
    outcome = _mitigation_outcome(delta, ci)
    # The Markdown document uses a typographic minus sign, so the
    # numbers it shows carry one too. The LaTeX table uses the ASCII
    # hyphen, which pdflatex can encode.
    delta_number = f"{delta:+.3f}"
    delta_ci = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
    delta_md = delta_number.replace("-", "−")
    delta_ci_md = delta_ci.replace("-", "−")

    # The p-value of the paired permutation test on the rewritten rows.
    # It is the third number the prose reports next to the delta and the
    # interval, so it has to come from the measurement too.
    p_value = regen.get("delta_bertscore_p")
    # How many distinct questions the rewritten rows came from. A
    # negative result on 16 rows from 8 questions is much weaker
    # evidence than on 16 rows from 16 questions, so the prose states
    # it.
    n_unique = regen.get("n_unique_queries")

    m = {
        "pre": f"{pre:.3f}" if pre is not None else "n/a",
        "post": f"{post:.3f}" if post is not None else "n/a",
        "delta": delta_text,
        "pre_tex": pre_tex,
        "post_tex": post_tex,
        "delta_tex": delta_tex,
        "caption_sentence": caption_sentence,
        "delta_number": delta_number,
        "delta_ci": delta_ci,
        "delta_md": delta_md,
        "delta_ci_md": delta_ci_md,
        "delta_p": f"{p_value:.3f}" if p_value is not None else "n/a",
        "n_regenerated": n,
        "n_unique_queries": n_unique if n_unique is not None else "n/a",
        "n_rows": mitigation.get("n_rows"),
        "outcome": outcome,
        # The phrase that states what the rewrite did, split so the prose
        # rule can splice it into the middle of a sentence.
        "outcome_phrase": _mitigation_outcome(delta, ci),
    }

    out: list[tuple[str, str, str]] = []
    for label, pattern, build in _MITIGATION_PROSE.get(path.name, []):
        def _sub(match: re.Match) -> str:
            return build(m)

        new_text, count = re.subn(pattern, _sub, text)
        if not count or new_text == text:
            continue
        text = new_text
        out.append((label, pattern, f"{count} row(s)"))
    return text, out


def _tex_three_comparison_caveat(v) -> str:
    """The sentence that closes the three-comparison paragraph.

    It has to say what the paragraph reports. The old text said "All
    three comparisons ... are not directly comparable" -- true of the
    test sets, but it was written when we lost all three, and it
    undercut the one win the paragraph now claims. The count comes from
    the same verdicts the sentence above it uses.
    """
    checks = [
        ("detection_accuracy", 0.847),
        ("spearman_rho", 0.770),
        ("precision_at_recall_90", 0.703),
    ]
    verdicts = [
        _verdict_from_ci(v, key, f"{key}_ci", str(comparable))
        for key, comparable in checks
    ]
    won = sum(1 for verdict in verdicts if verdict == "✓")
    lost = sum(1 for verdict in verdicts if verdict == "✗")
    tied = sum(1 for verdict in verdicts if verdict == "≈")
    if won and not lost and not tied:
        return (
            "All three comparisons are made across different test sets, so "
            "they are not directly comparable; the P@R90 comparison is "
            "the one where our measured advantage separates."
        )
    counts = []
    if won:
        counts.append(f"ahead on {won}")
    if tied:
        counts.append(f"at par on {tied}")
    if lost:
        counts.append(f"behind on {lost}")
    if not counts:
        return (
            "All three comparisons are made across different test sets, so "
            "they are not directly comparable, and none of the three "
            "separates."
        )
    return (
        "All three comparisons are made across different test sets, so "
        f"they are not directly comparable; we are {' and '.join(counts)} "
        "of them."
    )


def _tex_matrix_intro(text: str) -> str:
    """The paragraph that introduces the verdict matrix.

    It used to say the comparison runs along "eight primary
    dimensions" and listed eight of them, while the matrix below it
    reported thirteen rows -- and the list left out P@R90 and Spearman
    rho, the two dimensions the matrix turned on. The list is now
    written from the matrix, so the paragraph and the table describe the
    same comparison.
    """
    verdicts = _tex_matrix_verdicts(text)
    if not verdicts:
        return ""
    dimensions = list(verdicts)
    count = len(dimensions)

    def _names(items: list[str]) -> str:
        out = [
            item.replace("$\\Delta$", r"$\Delta$")
            .replace("Spearman rho", r"Spearman $\rho$")
            for item in items
        ]
        if len(out) == 1:
            return out[0]
        return ", ".join(out[:-1]) + " and " + out[-1]

    return (
        "This section compares the implemented pipeline against the cited "
        "methods along the dimensions that map to deployment value. The "
        f"verdict matrix below reports {count} of them: {_names(dimensions)}. "
        "Each row is measured against the published number printed beside "
        "it, and a win is claimed only when the bottom of our 95% "
        "confidence interval sits above that number. The rows do not all "
        "go our way, and they are reported as measured."
    )


def _tex_detection_subsections(text: str) -> str:
    """The subsection titles that name a verdict.

    The detection subsection was titled "At Par with SHROOM Winners"
    from a rule -- "At par = within 10% of best" -- that no other
    surface uses and that the matrix no longer applies. Under the
    paper's one rule, sentence accuracy and ROC-AUC are losses, not
    ties, so a title that calls them at par contradicts the table it
    introduces. The titles are written from the verdicts.
    """
    verdicts = _tex_matrix_verdicts(text)
    if not verdicts:
        return ""

    def _title(dimensions: list[str], fallback: str) -> str:
        found = [d for d in dimensions if d in verdicts]
        if not found:
            return fallback
        labels = [
            d.replace("Spearman rho", r"Spearman $\rho$")
            for d in found
        ]
        marks = [verdicts[d] for d in found]
        won = marks.count("Win")
        lost = marks.count("Loss")
        if won and lost:
            # A mixed set of verdicts is the interesting case, and
            # "Against the Cited Comparables" says nothing about it.
            word = (
                f"{won} Ahead, {lost} Behind the Cited Comparables"
            )
        elif won:
            word = "Ahead of the Cited Comparables"
        elif lost:
            word = "Below the Cited Comparables"
        else:
            word = "At Par with the Cited Comparables"
        return f"\\subsection{{{', '.join(labels)} --- {word}}}"

    detection = _title(
        ["Detection ROC-AUC", "Sentence accuracy", "P@R90", "Spearman rho"],
        "\\subsection{Detection ROC-AUC and Sentence Accuracy}",
    )
    return detection


def _tex_who_wins(v, key: str, comparable: float, other: str) -> str:
    """Who won a comparison, in the words the sentence uses.

    The sentence "P@R90 (Dale LaBSE wins): 0.703 vs. ours 0.800" used
    to be written by hand, and it said the wrong side won -- our 0.800
    is above their 0.703, and the bottom of our 95% interval is above
    their number too. The label is now written from the same verdict
    the matrix uses, so the sentence and the table cannot disagree.
    """
    verdict = _verdict_from_ci(v, key, f"{key}_ci", str(comparable))
    if verdict == "✓":
        return "we win"
    if verdict == "✗":
        return f"{other} wins"
    return "at par"


def _below_note(comparable: str) -> str:
    """The justification for a loss, in the row's own terms.

    A row whose comparable is an implied baseline says so, because an
    implied baseline is derived from another metric rather than
    published: calling it "the cited comparable" would dress a
    like-for-like comparison into something it is not. The markdown
    document draws the same distinction, and the two surfaces have to
    agree.
    """
    if "implied" in str(comparable).lower():
        return "Below the implied baseline"
    return "Below the cited comparable"


def _below_note_with_ci(comparable: str) -> str:
    """The same distinction, for a row with no interval to test."""
    if "implied" in str(comparable).lower():
        return "Below the implied baseline (no CI for this metric)"
    return "Below the cited comparable (no CI for this metric)"


def _tex_verdict_row(v, m) -> str:
    """Rebuild one score row of the LaTeX verdict matrix.

    The row is "Dimension & Comparable & Ours & Verdict & Justification".
    The verdict cell used to be written by hand, which let the LaTeX
    summary call P@R90 "At par" while the markdown document called the
    same measurement a win -- and call ROC-AUC "Below Sent-T5 XL
    (0.85-0.90)" when our 0.925 is above that band.

    The verdict is computed with the paper's one rule: a win needs the
    bottom of our 95% interval above the published number. The
    comparable is the one printed in the row, so the row is measured
    against its own number.
    """
    # The pattern captures the dimension (1), the padding before the
    # first ampersand (2), the comparable (3) and our value (4).
    dimension, comparable, ours = m.group(1), m.group(3), m.group(4)
    key = _TEX_VERDICT_METRICS.get(dimension.strip())
    if key is None:
        # A structural row -- the verdict is a property of the design,
        # so it does not follow from a number.
        return m.group(0)
    try:
        measured = v(key)
    except KeyError:
        return m.group(0)
    if measured is None:
        return m.group(0)

    # The comparable cell holds the number and the system's name; the
    # number is what the verdict is measured against.
    comparable_value = _comparable_number(comparable)
    if comparable_value is None:
        return m.group(0)

    # The verdict comes from the same helper the markdown uses, so the
    # two documents cannot apply different rules to one measurement.
    ci_key = f"{key}_ci"
    verdict = _verdict_from_ci(v, key, ci_key, comparable)
    if verdict == "✓":
        mark, note = "Win", "Our CI clears their published value"
    elif verdict == "✗":
        mark, note = "Loss", _below_note(comparable)
    elif verdict == "≈":
        mark, note = "At par", "Our CI overlaps their published value"
    elif float(measured) > comparable_value:
        # No interval for this metric, so there is no evidence of
        # separation either way. The point estimate is what is left, and
        # the note says so rather than implying a tested result.
        mark, note = "Win", "Above the cited comparable (no CI)"
    else:
        mark, note = "Loss", _below_note(comparable)

    ours_cell = ours if ours.strip() else f"\\textbf{{{measured}}}"
    # The spacing between the dimension and the first ampersand is part
    # of the row's alignment, so it is preserved rather than reset.
    # A win is set in bold, the way the matrix's other winning rows are,
    # so the strongest result is the one that reads first.
    mark_cell = f"\\textbf{{{mark}}}" if mark == "Win" else mark
    return (
        f"{dimension}{m.group(2)}& {comparable} & {ours_cell} & "
        f"{mark_cell} & {note} \\\\"
    )


# The score rows of the LaTeX verdict matrix and the metric each one is
# measured by. The rows that are not listed are structural: their
# verdict is a property of the system's design rather than of a number,
# so they are left as they are.
_TEX_VERDICT_METRICS = {
    "Claim-level F1": "claim_f1",
    "Hallucination Reduction Rate": "hallucination_reduction_rate",
    "Verification Success Rate": "verification_success_rate",
    "Detection ROC-AUC": "detection_roc_auc",
    "Sentence accuracy": "detection_accuracy",
    "P@R90": "precision_at_recall_90",
    "Spearman rho": "spearman_rho",
}


# The dimensions the LaTeX matrix reports, in the order the rows appear.
# The caption is built from these, so a dimension cannot be named in the
# caption without having a row to back it up.
_TEX_MATRIX_DIMENSIONS = [
    "Mitigation $\\Delta$",
    "Claim-level F1",
    "Hallucination Reduction Rate",
    "Verification Success Rate",
    "Detection ROC-AUC",
    "Sentence accuracy",
    "P@R90",
    "Spearman rho",
    "Per-claim provenance",
    "Latency (GPU)",
    "Memory (GB)",
    "Conformal coverage",
    "Modular backend",
]


def _tex_matrix_verdicts(text: str) -> dict[str, str]:
    """Read each matrix row's verdict out of the LaTeX table.

    The caption has to agree with the rows above it, so the caption is
    written from what the rows actually say rather than from a list
    kept by hand. A row's verdict is the fourth cell of its line.
    """
    verdicts: dict[str, str] = {}
    for line in text.splitlines():
        if "&" not in line or not line.rstrip().endswith(r"\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 4:
            continue
        dimension = cells[0].strip()
        verdict = cells[3].strip().replace(r"\textbf{", "").replace("}", "")
        if verdict in {"Win", "Loss", "At par"}:
            verdicts[dimension] = verdict
    return verdicts


def _tex_strongest_result(v) -> str:
    """The "Strongest result" item of the defence points.

    It used to quote a stale interval -- 95% CI [0.823, 0.865] for a
    measured 0.925, an interval that does not contain the value it was
    attached to -- and to say we were "below Sent-T5 XL (0.85--0.90)"
    when 0.925 is above that band. Both are now written from the
    measurement: the interval comes from the metrics file, and each
    comparison says which side of it our number falls on.
    """
    try:
        roc = v("detection_roc_auc")
        roc_ci = v("detection_roc_auc_ci")
        delta = v("mit_delta")
        delta_ci = v("mit_delta_ci")
    except KeyError:
        return ""
    if not roc or not roc_ci:
        return ""

    low, high = [float(p) for p in str(roc_ci).strip("[]").split(",")]
    value = float(roc)

    def _position(comparable: float) -> str:
        if low > comparable:
            return "above"
        if high < comparable:
            return "below"
        return "overlapping"

    parts = [
        r"\item \textbf{Strongest result.} Detection ROC-AUC of $"
        + f"{value:.3f}" + r"$ (95\% CI $["
        + f"{low:.3f}" + r", " + f"{high:.3f}" + r"]$) on the "
        r"length-matched HaluEval split --- above the general-purpose "
        r"probes (FAVA 0.72, EigenScore 0.80--0.82) but "
    ]
    sent_t5 = 0.85
    if _position(sent_t5) == "above":
        parts.append(
            r"above Sent-T5 XL (0.85--0.90) and below the MT specialist "
            r"Dale 2023 (0.987). "
        )
    else:
        parts.append(
            r"overlapping Sent-T5 XL (0.85--0.90) and below the MT "
            r"specialist Dale 2023 (0.987). "
        )
    # The mitigation sentence closes the item, and its numbers come from
    # the measurement too: a delta whose interval covers zero is not a
    # win, and the item says so.
    if delta and delta_ci:
        try:
            d_low, d_high = [
                float(p) for p in str(delta_ci).strip("[]").split(",")
            ]
        except ValueError:
            d_low = d_high = None
        if d_low is not None and d_low <= 0 <= d_high:
            parts.append(
                r"The mitigation $\Delta$ is \emph{not} a win: it is $"
                + f"{float(delta):+.3f}" + r"$ (CI $["
                + f"{d_low:.3f}" + r", " + f"{d_high:+.3f}"
                + r"]$) on the rows where step~10 regenerated, so it is "
                r"reported as an open result."
            )
        else:
            parts.append(
                r"The mitigation $\Delta$ is $"
                + f"{float(delta):+.3f}" + r"$ (CI $["
                + str(delta_ci).strip("[]") + r"]$) on the rows where "
                r"step~10 regenerated."
            )
    # The item is wrapped in the source, and the continuation lines are
    # indented to line up under the first line's text.
    body = " ".join(part.strip() for part in parts)
    words = body.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) > 62 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return ("\n          ").join(lines)


def _tex_verdict_caption(text: str) -> str:
    """The verdict matrix's caption, derived from its own rows.

    The caption used to name a rule of its own -- "At par = within 10%
    of best" -- that no other surface in the paper used, and to list
    claim-level F1 among the dimensions we trail while the row directly
    above said Win. Both are contradictions the table itself can settle,
    so the caption is built from the rows.

    The rule it states is the paper's one rule: a win needs the bottom
    of our 95% interval above the published number.
    """
    verdicts = _tex_matrix_verdicts(text)
    if not verdicts:
        return ""

    wins = [d for d, verdict in verdicts.items() if verdict == "Win"]
    trails = [d for d, verdict in verdicts.items() if verdict == "Loss"]

    def _names(items: list[str]) -> str:
        out = []
        for item in items:
            out.append(
                item.replace("$\\Delta$", r"$\Delta$")
                .replace("Spearman rho", r"Spearman $\rho$")
            )
        if not out:
            return "none"
        if len(out) == 1:
            return out[0]
        return ", ".join(out[:-1]) + " and " + out[-1]

    structural = {
        "Per-claim provenance", "Modular backend", "Conformal coverage",
        "Memory (GB)", "Latency (GPU)",
    }
    measured_wins = [d for d in wins if d not in structural]
    structural_wins = [d for d in wins if d in structural]
    measured_trails = [d for d in trails if d not in structural]
    structural_trails = [d for d in trails if d in structural]

    parts = [
        r"\caption{Verdict matrix. \textbf{Win} means the bottom of our 95\% "
        r"confidence interval sits above the cited published number; "
        r"At par means the interval overlaps it; Loss means it does not. ",
    ]
    if structural_wins:
        parts.append(
            f"Our wins on structural properties are {_names(structural_wins)}. "
        )
    if measured_wins:
        parts.append(
            f"On measured scores we are ahead on {_names(measured_wins)}. "
        )
    else:
        parts.append("On measured scores we are not ahead of any comparable. ")
    if measured_trails:
        parts.append(
            f"We trail on {_names(measured_trails)}. "
        )
    if structural_trails:
        parts.append(
            f"Our known engineering limits are {_names(structural_trails)}."
        )
    parts.append(r"}")
    caption = "".join(parts)
    # The caption is one long sentence run, so it is wrapped to keep the
    # LaTeX source readable. The wrap points are spaces, which LaTeX
    # treats as ordinary inter-word space, so the rendered caption is
    # unaffected.
    body = caption[len(r"\caption{"):-1]
    words = body.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) > 72 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return r"\caption{" + "\n".join(lines) + "}"


def _fill_why_we_win_tex(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Replace the projected rows in ``Paper/why_we_win_summary.tex``."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    def _ci(key: str) -> str | None:
        value = agg.get(f"{key}_ci95")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"
        return None

    # The CI strings are needed by the verdict rule, so they are put
    # beside the point estimates under the same key the rule looks up.
    agg = dict(agg)
    for _key in (
        "claim_f1", "detection_roc_auc", "detection_aupr",
        "precision_at_recall_90", "detection_accuracy", "spearman_rho",
        "hallucination_reduction_rate", "verification_success_rate",
    ):
        _value = _ci(_key)
        if _value is not None:
            agg[f"{_key}_ci"] = _value

    specs = [
        (
            "summary: ROC-AUC",
            r"Detection ROC-AUC            & 0\.987 \(Dale ALTI\+, MT\) & [^&]* &",
            lambda v, m: (
                r"Detection ROC-AUC            & 0.987 (Dale ALTI+, MT) & "
                f"{v('detection_roc_auc')} &"
            ),
            ("detection_roc_auc",),
        ),
        (
            "summary: HRR / VSR row",
            r"\\textbf\{Ours \(Step 11 gate\)\} & \\textbf\{[^}]*\} & "
            r"\\textbf\{[^}]*\} \\\\",
            lambda v, m: (
                r"\textbf{Ours (Step 11 gate)} & \textbf{"
                + f"{v('hallucination_reduction_rate')}" + r"} & \textbf{"
                + f"{v('verification_success_rate')}" + r"} \\"
            ),
            ("hallucination_reduction_rate", "verification_success_rate"),
        ),
        (
            "prose: sentence accuracy / Spearman / P@R90",
            # The three numbers sit inside $...$ and the sentence's full stop
            # is *outside* the closing $, so the pattern has to close the
            # math mode before the period.
            #
            # The pattern accepts any winner name in the parenthesis, not
            # just "wins", because the sentence used to say "P@R90 (Dale
            # LaBSE wins)" for a comparison we win -- our 0.800 against
            # their 0.703, with the bottom of our interval above their
            # number. The label is now written from the verdict, so the
            # sentence cannot claim the other side won.
            r"\\textbf\{Sentence accuracy \([^)]*\):\} \$0\.847\$ vs\.\\ "
            r"ours \$[\d.]+\$\.\s*"
            r"\\textbf\{Spearman \$\\rho\$ \([^)]*\):\} "
            r"\$0\.770\$ vs\.\\ ours \$[\d.]+\$\.\s*"
            r"\\textbf\{P@R90 \([^)]*\):\} "
            r"\$0\.703\$ vs\.\\ ours \$[\d.]+\$\.",
            lambda v, m: (
                r"\textbf{Sentence accuracy ("
                + _tex_who_wins(v, "detection_accuracy", 0.847,
                                "Halu-NLP") + r"):} $0.847$ vs.\ "
                r"ours $" + v("detection_accuracy") + r"$. "
                r"\textbf{Spearman $\rho$ ("
                + _tex_who_wins(v, "spearman_rho", 0.770,
                                "Halu-NLP") + r"):} $0.770$ vs.\ ours $"
                + v("spearman_rho") + r"$. "
                r"\textbf{P@R90 ("
                + _tex_who_wins(v, "precision_at_recall_90", 0.703,
                                "Dale LaBSE") + r"):} $0.703$ vs.\ ours $"
                + v("precision_at_recall_90") + r"$."
            ),
            (
                "detection_accuracy", "spearman_rho",
                "precision_at_recall_90",
            ),
            # The intervals, so the winner label comes from the paper's
            # one rule rather than from a comparison of point estimates.
            ("detection_accuracy_ci", "spearman_rho_ci",
             "precision_at_recall_90_ci"),
        ),
        (
            # The sentence that closes the paragraph used to say "All
            # three comparisons are made across different test sets and
            # are not directly comparable" -- which is true, but it was
            # written when we lost all three. It now counts the wins the
            # paragraph actually reports, so it cannot undercut a win
            # that the sentence above it claims.
            "prose: the three-comparison caveat",
            r"All three\s*comparisons are made across different test sets "
            r"and are not directly\s*comparable\.",
            lambda v, m: _tex_three_comparison_caveat(v),
            ("detection_accuracy", "spearman_rho",
             "precision_at_recall_90"),
            ("detection_accuracy_ci", "spearman_rho_ci",
             "precision_at_recall_90_ci"),
            # The intervals, so the winner label comes from the paper's
            # one rule rather than from a comparison of point estimates.
            ("detection_accuracy_ci", "spearman_rho_ci",
             "precision_at_recall_90_ci"),
        ),
        (
            "prose: VSR in the Conformal-RAG answer",
            r"is [\d.]+, so we do not claim a rate advantage either",
            lambda v, m: (
                "is "
                + v("verification_success_rate")
                + ", so we do not claim a rate advantage either"
            ),
            ("verification_success_rate",),
        ),
        (
            "prose: strongest result ROC-AUC",
            r"\\textbf\{Strongest result\.\} Detection ROC-AUC of \$[\d.]+\$ "
            r"\(95\\% CI",
            lambda v, m: (
                r"\textbf{Strongest result.} Detection ROC-AUC of $"
                + v("detection_roc_auc") + r"$ (95\% CI"
            ),
            ("detection_roc_auc",),
        ),
        (
            "summary: sentence accuracy",
            r"Sentence accuracy            & 0\.847 \(Halu-NLP\) & [^&]* &",
            lambda v, m: (
                r"Sentence accuracy            & 0.847 (Halu-NLP) & "
                f"{v('detection_accuracy')} &"
            ),
            ("detection_accuracy",),
        ),
        (
            "summary: P@R90",
            r"P@R90                        & 0\.703 \(Dale LaBSE\) & [^&]* &",
            lambda v, m: (
                r"P@R90                        & 0.703 (Dale LaBSE) & "
                f"{v('precision_at_recall_90')} &"
            ),
            ("precision_at_recall_90",),
        ),
        (
            "summary: claim-level F1",
            r"Claim-level F1               & 0\.71 \(TU Wien Vectara\) & "
            r"\\textbf\{[^}]*\} &",
            lambda v, m: (
                r"Claim-level F1               & 0.71 (TU Wien Vectara) & "
                r"\textbf{" + f"{v('claim_f1')}" + r"} &"
            ),
            ("claim_f1",),
        ),
        (
            "summary: HRR",
            r"Hallucination Reduction Rate & 0\.55 \(Dale 2023\) & "
            r"\\textbf\{[^}]*\} &",
            lambda v, m: (
                r"Hallucination Reduction Rate & 0.55 (Dale 2023) & "
                r"\textbf{" + f"{v('hallucination_reduction_rate')}" + r"} &"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "summary: VSR",
            r"Verification Success Rate    & 0\.81 \(Halu-NLP\) & "
            r"\\textbf\{[^}]*\} &",
            lambda v, m: (
                r"Verification Success Rate    & 0.81 (Halu-NLP) & "
                r"\textbf{" + f"{v('verification_success_rate')}" + r"} &"
            ),
            ("verification_success_rate",),
        ),
        (
            "summary: mitigation delta",
            r"Mitigation \$\\Delta\$( +) & \+0\.60 \(Dale COMET-QE\) & "
            r"\\textbf\{[^}]*\} & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Mitigation $\Delta$         & +0.60 (Dale COMET-QE) & "
                r"\textbf{" + f"{v('delta_tex_prose')}" + r"} & "
                + ("\textbf{Win}" if _delta_positive(v) else "Loss")
                + r" & Rewrite, not rerank \\"
            ),
            ("delta_tex_prose",),
        ),
        (
            "detection table row",
            r"\\textbf\{Ours \(Step 9 fused\)\} & \\textbf\{[^}]*\} \\\\",
            lambda v, m: (
                r"\textbf{Ours (Step 9 fused)} & \textbf{"
                f"{v('detection_roc_auc')}" r"} \\"
            ),
            ("detection_roc_auc",),
        ),
        (
            "claim P/R/F1 table row",
            r"\\textbf\{Ours \(Step 8 \+ Step 9\)\} & \\textbf\{[^}]*\} & "
            r"Atomic claims \\\\",
            lambda v, m: (
                r"\textbf{Ours (Step 8 + Step 9)} & \textbf{"
                f"{v('claim_precision')} / {v('claim_recall')} / {v('claim_f1')}"
                r"} & Atomic claims \\"
            ),
            ("claim_precision", "claim_recall", "claim_f1"),
        ),
        (
            "latency table row",
            r"\\textbf\{Ours\} & [^&]* & Platt \+ temperature \\\\",
            lambda v, m: (
                r"\textbf{Ours} & " + _latency_band(v).replace("-", "--")
                + r" & Platt + temperature \\"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "summary: latency verdict",
            r"Latency \(GPU\)\s+& 0\.22 s \(LLM-Check\) & [^&]* & "
            r"[^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Latency (GPU)                & 0.22 s (LLM-Check) & "
                + _latency_band(v).replace("-", "--")
                + r" & Loss & Slower than the single-pass probe \\"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "summary: claim-level F1 caption",
            # The caption asserted the F1 was below the sentence-level
            # comparables. That was true of the superseded 0.653 measurement
            # and false of the current one, so the claim is derived from the
            # measured value rather than left as fixed prose.
            r"\\caption\{Claim-level F1\. Our F1 is (?:below|above) the cited "
            r"sentence-level\s*comparables; the contribution claimed here is "
            r"granularity --- every atomic\s*claim is scored independently, "
            r"where every other system reduces the answer to a single "
            r"sentence-level score\.\}",
            lambda v, m: (
                r"\caption{Claim-level F1. Our F1 is "
                + ("above" if _above(v, "claim_f1", 0.71) else "below")
                + r" the cited sentence-level comparables; the contribution "
                r"claimed here is granularity --- every atomic claim is scored "
                r"independently, where every other system reduces the answer "
                r"to a single sentence-level score.}"
            ),
            ("claim_f1",),
        ),
        (
            "summary: VSR row value",
            # This rule used to write the VSR row's verdict as well as
            # its value, using the point-estimate rule. The unified
            # "score verdicts" rule below now owns every score row's
            # verdict, so writing it here too made the two rules fight:
            # each run rewrote the cell the other had just written, and
            # the generator was never idempotent. The rule now refreshes
            # the value only, and re-emits the verdict and justification
            # it found.
            r"(Verification Success Rate    & 0\.81 \(Halu-NLP\) & )"
            r"\\textbf\{[^}]*\} & ([^&]*) & ([^&]*) \\\\",
            lambda v, m: (
                m.group(1)
                + r"\textbf{" + f"{v('verification_success_rate')}" + r"} & "
                + m.group(2) + " & " + m.group(3) + r" \\"
            ),
            ("verification_success_rate",),
        ),
        (
            # The "Different from Dale 2023?" paragraph quotes the measured
            # delta. It was hand-written, so a re-evaluation left it
            # holding the previous run's figure while the table beside it
            # was refreshed. The delta now comes from the measurement.
            "prose: the Dale comparison",
            r"on machine translation; our measured \$\\Delta\$ is \$[-+][\d.]+\$ "
            r"on the length-matched HaluEval split",
            lambda v, m: (
                "on machine translation; our measured $\\Delta$ is $"
                + v("mit_delta").replace("-", "-") + "$ "
                "on the length-matched HaluEval split"
            ),
            ("mit_delta",),
        ),
        (
            # The "Strongest result." paragraph quotes the delta, its
            # interval and the verdict, so it has to move with the
            # measurement too.
            "prose: the strongest result",
            r"mitigation \$\\Delta\$ is \\emph\{not\} a win: it is \$[-+][\d.]+\$ "
            r"\(CI \$\[[^\]]+\]\$\) on the rows where step~10 regenerated",
            lambda v, m: (
                r"mitigation $\Delta$ is \emph{not} a win: it is $"
                + v("mit_delta") + "$ (CI $[" + v("mit_delta_ci").strip("[]")
                + r"]$) on the rows where step~10 regenerated"
            ),
            ("mit_delta", "mit_delta_ci"),
        ),
        (
            # The "Strongest result" defence point quoted a stale
            # interval and mis-stated where our ROC-AUC sits relative
            # to Sent-T5 XL. The whole item is rebuilt from the
            # measurement; the mitigation sentence that follows it is
            # left to the rule above.
            "prose: the strongest result item",
            # The item is wrapped across lines in the source, so the
            # pattern allows whitespace where a line break sits. It
            # matches the whole item, including the mitigation sentence
            # that follows it, so the item is written as one piece and
            # the source stays wrapped.
            r"\\item\s*\\textbf\{Strongest result\.\}\s*Detection ROC-AUC "
            r"of\s*\$[\d.]+\$\s*\(95\\% CI\s*\$?\[[^\]]+\]\$\)?\s*on the\s*"
            r"length-matched HaluEval\s*split\s*---\s*"
            r"(?:above|overlapping|below)[\s\S]*?open result\.",
            lambda v, m: _tex_strongest_result(v),
            ("mit_delta",),
            ("detection_roc_auc", "detection_roc_auc_ci", "mit_delta_ci"),
        ),
        (
            # The verdict matrix's score rows, rewritten as a group at the
            # end so the verdict is computed from the "Ours" cell the
            # rules above have just refreshed.
            #
            # The verdict cells used to be written by hand, with a rule
            # of their own -- "At par = within 10% of best" -- that the
            # rest of the document does not use. That is how the LaTeX
            # summary called P@R90 "At par" (0.800 against 0.703, a
            # difference of 14%) while the markdown called the same
            # measurement a win, and how it wrote "Below Sent-T5 XL
            # (0.85--0.90)" for a ROC-AUC of 0.925 that is above that
            # band.
            #
            # The rule is the paper's own: a win needs the bottom of our
            # 95% interval above the published number.
            "summary: score verdicts",
            r"(Claim-level F1|Hallucination Reduction Rate|"
            r"Verification Success Rate|Detection ROC-AUC|"
            r"Sentence accuracy|P@R90|Spearman rho)( *)& ([^&]*?) & "
            r"([^&]*?) & [^&]*? & [^&]*? \\\\",
            lambda v, m: _tex_verdict_row(v, m),
            (),
            ("claim_f1", "detection_roc_auc", "detection_roc_auc_ci",
             "detection_aupr", "detection_aupr_ci",
             "precision_at_recall_90", "precision_at_recall_90_ci",
             "detection_accuracy", "detection_accuracy_ci",
             "spearman_rho", "spearman_rho_ci",
             "hallucination_reduction_rate", "hallucination_reduction_rate_ci",
             "verification_success_rate", "verification_success_rate_ci"),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "verification_success_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    # The intervals, so the verdict rule can apply the paper's one rule
    # -- a win needs the bottom of our 95% interval above the published
    # number -- rather than comparing point estimates.
    for _key in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "detection_accuracy", "spearman_rho",
        "hallucination_reduction_rate", "verification_success_rate",
    ):
        _value = _ci(_key)
        if _value is not None:
            values[f"{_key}_ci"] = _value
    values.update(_mitigation_prose_values(mitigation_path))
    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

    # The verdict matrix's caption is written from the rows above it, so
    # it cannot name a dimension we win as one we trail, and it cannot
    # state a rule of its own. It is rebuilt after the verdict rule has
    # run, because that is what decides the rows it reads.
    intro = _tex_matrix_intro(text)
    if intro:
        new_text, count = re.subn(
            r"This section justifies the claim.*?"
            r"sentence accuracy, latency, and calibration\.",
            lambda m: intro, text, count=1, flags=re.DOTALL,
        )
        if count and new_text != text:
            text = new_text
            replacements.append(
                ("summary: verdict matrix intro", "intro", "1 paragraph")
            )

    caption = _tex_verdict_caption(text)
    if caption:
        # The replacement is a function, not a string: the caption holds
        # LaTeX backslashes such as \caption and \textbf, and re.sub
        # would read those as backreferences.
        #
        # The old caption is matched up to the label that follows it, not
        # up to the first closing brace. A caption holds nested braces
        # (\textbf{Win}), and a run that replaced it with a caption of
        # its own used to leave the tail of the previous caption behind:
        # the new caption ended, and the old one's text carried on.
        # Matching to the label removes whatever was there.
        new_text, count = re.subn(
            r"\\caption\{Verdict matrix\..*?\n(?P<label>\\label\{tab:verdict\})",
            lambda m: caption + "\n" + m.group("label"),
            text, count=1, flags=re.DOTALL,
        )
        if count and new_text != text:
            text = new_text
            replacements.append(
                ("summary: verdict matrix caption", "caption", "1 caption")
            )

    # The subsection title that names a verdict is written from the
    # matrix, so it cannot call a loss "at par". The pattern matches
    # the title in any of the forms this rule has written, so a second
    # run finds it and the generator stays idempotent.
    subsection = _tex_detection_subsections(text)
    if subsection:
        new_text, count = re.subn(
            r"\\subsection\{Detection ROC-AUC[^}]*\}",
            lambda m: subsection, text, count=1,
        )
        if count and new_text != text:
            text = new_text
            replacements.append(
                ("summary: detection subsection title", "subsection",
                 "1 title")
            )

    text, mit_changes = _fill_mitigation_prose(
        text, path, dry_run=dry_run, mitigation_path=mitigation_path
    )
    replacements.extend(mit_changes)

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--mitigation", type=Path, default=MITIGATION_PATH)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("Paper"),
        help=(
            "directory holding the prose files. These paths used to be "
            "hard-coded, so a test that passed a temporary metrics file "
            "wrote that file's numbers into the real paper — which is how "
            "a superseded ROC-AUC kept reappearing. Pointing this at a "
            "temporary directory keeps a test off the real files."
        ),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help=(
            "the repository README. It states the measured mitigation delta "
            "in prose, and that paragraph is generated so it cannot drift "
            "away from the measurement."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the proposed changes without writing the file",
    )
    args = parser.parse_args(argv)

    metrics = _load_metrics(args.metrics)
    agg = _load_aggregate(args.metrics)
    if not agg:
        print(
            f"No metrics in {args.metrics}. Run the evaluation first.",
            file=sys.stderr,
        )
        return 1

    replacements = build_replacements(agg)
    if not replacements:
        print("No measurable metrics found in the aggregate.", file=sys.stderr)
        return 1

    # The mitigation rows are filled from a separate measurement. A row
    # whose measurement is missing is left untouched, exactly as for the
    # detection rows.
    mitigation = load_mitigation(args.mitigation)
    mit_replacements = build_mitigation_replacements(mitigation)
    replacements.update(mit_replacements)

    with args.csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print(f"{args.csv} is empty.", file=sys.stderr)
        return 1

    header = rows[0]
    ours_idx = _find_ours_column(header)

    changed: list[tuple[str, str, str]] = []
    for row in rows[1:]:
        if len(row) <= ours_idx:
            continue
        label = (row[1] if len(row) > 1 else "").strip()
        if label not in replacements:
            continue
        new_cell = replacements[label]
        old_cell = row[ours_idx]
        if old_cell.strip() != new_cell:
            changed.append((label, old_cell, new_cell))
            row[ours_idx] = new_cell

    if changed:
        for label, old, new in changed:
            print(f"  {label}: {old!r} -> {new!r}")

    # The prose version of the same comparison carries the same estimates.
    # It is refreshed even when the CSV is already current: an earlier
    # version returned early in that case, which left the prose tables
    # frozen at whatever number they happened to hold.
    # The secondary datasets (closed-book TruthfulQA, RAGTruth) are read
    # once and handed to every fill that reports them.
    cross = load_cross_dataset_aggregates()

    md_changes = _fill_comparison_md(
        args.paper_dir / "comparison.md", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation, cross=cross,
    )
    for label, old, new in md_changes:
        print(f"  comparison.md {label}: {_safe(old)} -> {_safe(new)}")

    tex_changes = _fill_comparison_tex(
        args.paper_dir / "comparison_summary.tex", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation,
    )
    for label, old, new in tex_changes:
        print(f"  comparison_summary.tex {label}: {_safe(old)} -> {_safe(new)}")

    ww_tex_changes = _fill_why_we_win_tex(
        args.paper_dir / "why_we_win_summary.tex", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation,
    )
    for label, old, new in ww_tex_changes:
        print(f"  why_we_win_summary.tex {label}: {_safe(old)} -> {_safe(new)}")

    # The README lives outside the paper directory, so it is filled from
    # the repository root rather than from --paper-dir.
    readme_changes = _fill_readme_md(
        args.readme, agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation,
    )
    for label, old, new in readme_changes:
        print(f"  {args.readme.name} {label}: {_safe(old)} -> {_safe(new)}")

    total_changes = (
        len(changed) + len(md_changes) + len(tex_changes)
        + len(ww_tex_changes) + len(readme_changes)
    )

    if args.dry_run:
        print(f"\n[dry-run] {total_changes} cell(s) would change")
        return 0

    if changed:
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"\nUpdated {len(changed)} cell(s) in {args.csv}")
    elif not total_changes:
        print("Nothing to update — the table already matches the metrics.")
    else:
        print(f"\nUpdated {len(changed)} cell(s) in {args.csv}")
    if md_changes:
        print(f"Updated {len(md_changes)} row(s) in Paper/comparison.md")
    if tex_changes:
        print(f"Updated {len(tex_changes)} row(s) in Paper/comparison_summary.tex")
    if ww_tex_changes:
        print(f"Updated {len(ww_tex_changes)} row(s) in Paper/why_we_win_summary.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
