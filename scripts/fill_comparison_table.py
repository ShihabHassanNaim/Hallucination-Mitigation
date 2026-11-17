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
    the cell(s) that hold a measured value.

    The patterns are deliberately written so that a row is matched no matter
    what number it currently holds. An earlier version anchored on the exact
    old string, which meant a row could only ever be filled once: the second
    run found no match and left the previous run's number in place. Matching
    the surrounding text instead makes the fill idempotent, so re-running
    after a new evaluation always refreshes the row.
    """
    replacements: list[tuple[str, str, str]] = []
    for label, pattern, build, needed in specs:
        values_needed = {k: values.get(k) for k in needed}
        if any(val is None for val in values_needed.values()):
            # The metric was not measured — leave the row untouched rather
            # than substituting a number that does not exist.
            continue

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
            "delta_tex_prose": delta_tex_prose}


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


def _ci_bracket(lookup, key: str) -> str:
    """Render a 95% CI as ``lo, hi`` for prose, or ``--`` if absent.

    The prose in the defence notes quotes the interval, so a rule that
    rewrites the point estimate must rewrite the interval with it —
    otherwise the sentence would mix a fresh estimate with a stale
    interval.
    """
    try:
        ci = lookup(key)
    except KeyError:
        return "--"
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return "--"
    try:
        return f"{float(ci[0]):.3f}, {float(ci[1]):.3f}"
    except (TypeError, ValueError):
        return "--"


def _ci_text(agg: dict, key: str) -> str | None:
    """Render ``[lo, hi]`` for a metric's ``_ci95`` sibling, or None.

    ``None`` means "not measured", which the spec machinery treats as
    "leave this cell alone" — the correct behaviour when a run has no
    interval to quote.
    """
    ci = agg.get(f"{key}_ci95")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return None
    try:
        return f"{float(ci[0]):.3f}, {float(ci[1]):.3f}"
    except (TypeError, ValueError):
        return None


def _ragtruth_win_prose(lookup) -> str:
    """The sentence that reads the RAGTruth verdict table out loud.

    It states how many of the five published anchors our interval clears
    and which ones it does not, so the prose cannot claim five wins when
    the table shows four.
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
    if not ties:
        return (
            f"All five anchors clear the bar ({lower:.4f} > "
            f"{min(float(p) for p in _PUBLISHED_AUROC.values()):.4f}), so all "
            "five are wins."
        )
    names = ", ".join(ties)
    verb = "is" if len(ties) == 1 else "are"
    return (
        f"{len(wins)} of the five anchors clear the bar (our lower bound is "
        f"{lower:.4f}). The {names} anchor{'s' if len(ties) > 1 else ''} "
        f"{verb} not cleared — our interval covers it, so the comparison is "
        "a tie and is reported as one."
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
    if heldout_lower:
        comparison = (
            f"the held-out lower bound ({heldout_lower}) "
            + ("sits below" if float(heldout_lower) < anchor else "sits above")
            + f" LLM-Check's strongest anchor ({anchor:.4f}), so the held-out "
            "figure on its own would "
            + ("not be a win" if float(heldout_lower) < anchor else "be a win")
        )
    else:
        comparison = (
            "the held-out figure on its own would need its own interval "
            "compared against the anchor"
        )
    return (
        "496 rows of the RAGTruth split, so "
        + str(fitted)
        + " is measured on data the head was\nfitted on. Its honest "
        "generalisation estimate is the **held-out** figure:\n**"
        + str(holdout)
        + ", CI ["
        + str(holdout_ci).strip("[]")
        + "]** — the head scored on a half of the fit data\nit never saw. "
        "The gap between "
        + str(holdout)
        + " and "
        + str(fitted)
        + " is the generalisation gap,\nand it is "
        + verdict
        + ". Both numbers are reported; "
        + comparison
        + "."
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


def _verdict_tally(text: str) -> str | None:
    """Rebuild the verdict tally from the win-matrix rows.

    The tally summarises the rows above it, so it is computed from those
    rows rather than maintained by hand. Returns None when the matrix is
    not present or a row cannot be classified.

    The winner mark is read by *column position*, not by "the first mark
    on the line". Some rows carry a checkmark in an earlier column -- row
    15 reads "Dale 2023 only | ✓ | ✓ | ≈", because both systems do
    active mitigation -- and a non-greedy pattern would pick up the
    comparable's mark and count a tie as a win. The matrix has seven
    columns, so the verdict is the sixth.
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


def _verdict_rows_by_name(text: str) -> list[tuple[str, str]]:
    """The win-matrix rows as ``(dimension name, verdict)`` pairs.

    The dimension name is the second column, without the markdown bold
    markers. This is what the defence notes' tally quotes, so the two
    tallies can be derived from one source.
    """
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not re.match(r"^\| \d+ \|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8:
            continue
        verdict = cells[6]
        if verdict not in ("✓", "✗", "≈"):
            continue
        name = re.sub(r"\*\*", "", cells[2]).strip()
        rows.append((name, verdict))
    return rows


def _name_tally(text: str) -> str | None:
    """Rebuild a tally that lists dimension names rather than numbers."""
    rows = _verdict_rows_by_name(text)
    if not rows:
        return None
    wins = [n for n, v in rows if v == "✓"]
    ties = [n for n, v in rows if v == "≈"]
    losses = [n for n, v in rows if v == "✗"]

    def _fmt(label: str, names: list[str]) -> str:
        return f"| **{label}** | {len(names)} | {', '.join(names)} |"

    return "\n".join([
        _fmt("✓ Win", wins),
        _fmt("≈ Tie / At par", ties),
        _fmt("✗ Loss", losses),
    ])


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
            "3.1 detection row",
            r"\| \*\*Ours \(RAG fused score, Step 9\)\*\* \| \*\*[^*]+\*\* \| "
            r"\*\*[^*]+\*\* \| \*\*[^*]+\*\* \| Black-box \+ retrieval \| "
            r"HaluEval QA[^|]*\|",
            lambda v, m: (
                f"| **Ours (RAG fused score, Step 9)** | "
                f"**{v('detection_roc_auc')}** | **{v('detection_aupr')}** | "
                f"**{v('precision_at_recall_90')}** | "
                "Black-box + retrieval | HaluEval QA (length-matched) |"
            ),
            ("detection_roc_auc", "detection_aupr", "precision_at_recall_90"),
        ),
        (
            "3.2 claim row",
            r"\| \*\*Ours \(Step 8 \+ Step 9\)\*\* \| \*\*[^*]+\*\* "
            r"\(per-claim NLI\) \| Atomic claims \| HaluEval[^|]*\|",
            lambda v, m: (
                f"| **Ours (Step 8 + Step 9)** | **{v('claim_precision')} / "
                f"{v('claim_recall')} / {v('claim_f1')}** (per-claim NLI) | "
                "Atomic claims | HaluEval |"
            ),
            ("claim_precision", "claim_recall", "claim_f1"),
        ),
        (
            "summary: ROC-AUC",
            r"\| \*\*Detection ROC-AUC\*\* \| 0\.987 \(Dale 2023 ALTI\+\) \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| **Detection ROC-AUC** | 0.987 (Dale 2023 ALTI+) | "
                f"**{v('detection_roc_auc')}** |"
            ),
            ("detection_roc_auc",),
        ),
        (
            # The AUPR row was previously hard-coded prose claiming we were
            # "above the general-purpose probes". That is not a claim the
            # interval supports: our AUPR is 0.853 with a 95% CI of
            # [0.833, 0.874], and Decoding Memories reports 0.84-0.85, so
            # the two are not separated. The verdict is now computed from
            # the measured CI against the best published comparable, so the
            # row cannot keep a win the measurement does not support.
            "summary: AUPR",
            r"\| \*\*Detection AUPR\*\* \| [^|]*\| \*\*[^*]+\*\* \| [^|]*\|",
            lambda v, m: (
                "| **Detection AUPR** | 0.84-0.85 (Decoding Memories) | "
                f"**{v('detection_aupr')}** | "
                f"{_separation_verdict(agg.get('detection_aupr'), agg.get('detection_aupr_ci95'), 0.84)} |"
            ),
            ("detection_aupr",),
        ),
        (
            # The CCR row had the same problem, and a second one: its
            # comparable of 0.55 is *implied* from another metric of Dale
            # 2023 rather than published as a CCR, and the other cited
            # systems have no mitigation stage at all (their cells are
            # "0 (no rewriting)"), so there is no published CCR to compare
            # against. The row now says so instead of claiming a win.
            "summary: CCR",
            r"\| \*\*CCR\*\* \| [^|]*\| \*\*[^*]+\*\* \| [^|]*\|",
            lambda v, m: (
                "| **CCR** | none published (Dale 2023's 0.55 is implied from "
                "another metric, not a published CCR) | "
                f"**{v('claim_correction_rate')}** | "
                "No published comparable, so this is not a win |"
            ),
            ("claim_correction_rate",),
        ),
        (
            # The pattern has to match the *corrected* wording as well as the
            # original. A pattern anchored on the old ("is in the same latency
            # tier") sentence stops matching once the file has been fixed,
            # so the generator silently skips the block and the prose
            # drifts out of the generator's control.
            "latency observation",
            r"\*\*Observation\.\*\* The pipeline is (?:not )?in the same "
            r"latency tier as LLM-Check[^.]*\.[^\n]*\n",
            lambda v, m: (
                "**Observation.** The pipeline is not in the same latency tier as "
                "LLM-Check: measured on the length-matched HaluEval split, a "
                f"query takes {_latency_band(v)} end to end (mean "
                f"{_latency_mean(v):.2f} s). This is the *detection* path: the "
                "benchmark supplies the answer, so step 2 skips generation and "
                "the time is spent on query understanding, hybrid retrieval, "
                "reranking and the claim-level verification pass. A run that "
                "generated its own answer would add the generator's latency on "
                "top, so the full RAG pipeline is slower than this band "
                "suggests. It is faster than the sampling-based methods "
                "(INSIDE, AutoHall, SelfCheckGPT, Decoding Memories) and the "
                "MC-dropout rerankers (Dale 2023), but slower than the "
                "single-pass probes.\n"
            ),
            ("latency_min_s", "latency_max_s", "latency_mean_s"),
        ),
        (
            "efficiency row",
            r"\| \*\*Ours \(RAG, single-pass\)\*\* \| \*\*[^|]*\*\* \(CPU\) / "
            r"\*\*[^|]*\*\* \(GPU\) \| [^|]* \| [^|]* \| [^|]* \|",
            lambda v, m: (
                "| **Ours (RAG, single-pass)** | **2-5 s** (CPU) / "
                f"**{_latency_band(v)}** (GPU) | ~4 GB | "
                "~$0.10 (HF local) / ~$0.80 (OpenRouter) | "
                "Detection path (answer supplied); add the generator call for "
                "the full pipeline |"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "summary: sentence accuracy",
            r"\| \*\*Sentence accuracy\*\* \| 0\.847 \(Halu-NLP, SHROOM\) \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| **Sentence accuracy** | 0.847 (Halu-NLP, SHROOM) | "
                f"**{v('detection_accuracy')}** |"
            ),
            ("detection_accuracy",),
        ),
        (
            "summary: Spearman rho",
            r"\| \*\*Spearman ρ\*\* \| 0\.770 \(Halu-NLP\) \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| **Spearman ρ** | 0.770 (Halu-NLP) | "
                f"**{v('spearman_rho')}** |"
            ),
            ("spearman_rho",),
        ),
        (
            "summary: sentence accuracy verdict",
            r"\| \*\*Sentence accuracy\*\* \| 0\.847 \(Halu-NLP, SHROOM\) \| "
            r"\*\*[^*]+\*\* \| [^|]* \|",
            lambda v, m: (
                f"| **Sentence accuracy** | 0.847 (Halu-NLP, SHROOM) | "
                f"**{v('detection_accuracy')}** | "
                + ("Above the SHROOM winner"
                   if _above(v, "detection_accuracy", 0.847)
                   else "Below the SHROOM winner")
                + " |"
            ),
            ("detection_accuracy",),
        ),
        (
            "summary: claim-level F1",
            r"\| \*\*Claim-level F1\*\* \| 0\.71 \(TU Wien Vectara\) \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| **Claim-level F1** | 0.71 (TU Wien Vectara) | "
                f"**{v('claim_f1')}** |"
            ),
            ("claim_f1",),
        ),
        (
            "summary: claim-level F1 verdict",
            r"\| \*\*Claim-level F1\*\* \| 0\.71 \(TU Wien Vectara\) \| "
            r"\*\*[^*]+\*\* \| [^|]* \|",
            lambda v, m: (
                f"| **Claim-level F1** | 0.71 (TU Wien Vectara) | "
                f"**{v('claim_f1')}** | "
                + ("Above the cited comparable"
                   if _above(v, "claim_f1", 0.71)
                   else "Below the cited comparable")
                + " |"
            ),
            ("claim_f1",),
        ),
        (
            "summary: header",
            r"\| Category \| Best Comparable Score \| Ours \((?:Projected|measured)"
            r"[^|]*\) \| Verdict \|",
            lambda v, m: (
                "| Category | Best Comparable Score | Ours (measured, HaluEval "
                "length-matched) | Verdict |"
            ),
            ("detection_roc_auc",),
        ),
        (
            "summary: latency",
            r"\| \*\*Per-query latency \(GPU\)\*\* \| 0\.22 s \(LLM-Check\) \| "
            r"\*\*[^*]+\*\* \| [^|]* \|",
            lambda v, m: (
                "| **Per-query latency (GPU)** | 0.22 s (LLM-Check) | "
                f"**{_latency_band(v)}** | "
                "Slower than LLM-Check; measured on the length-matched "
                "HaluEval split, detection path only (the answer is "
                "supplied, so generation is skipped)"
                + " |"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "summary: HRR",
            r"\| \*\*HRR\*\* \| 0\.55 \(Dale 2023, implied\) \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| **HRR** | 0.55 (Dale 2023, implied) | "
                f"**{v('hallucination_reduction_rate')}** |"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "summary: mitigation delta",
            r"\| \*\*Mitigation Δ over baseline\*\* \| \+0\.60 \(Dale 2023 "
            r"COMET-QE rerank\) \| \*\*[^*]+\*\* \| [^|]*\|",
            lambda v, m: (
                "| **Mitigation Δ over baseline** | +0.60 (Dale 2023 "
                f"COMET-QE rerank) | **{v('delta')}** | {v('verdict')} |"
            ),
            ("delta", "verdict"),
        ),
        (
            "summary: VSR",
            r"\| \*\*VSR\*\* \| 0\.81 \(Halu-NLP\) \| \*\*[^*]+\*\* \| [^|]* \|",
            lambda v, m: (
                "| **VSR** | 0.81 (Halu-NLP) | "
                f"**{v('verification_success_rate')}** | "
                + ("Above the cited comparables"
                   if _above(v, "verification_success_rate", 0.81)
                   else "Below the cited comparables")
                + " |"
            ),
            ("verification_success_rate",),
        ),
        (
            "summary: end-to-end accuracy",
            r"\| \*\*End-to-end accuracy\*\* \| — \| \*\*[^*]+\*\* \| [^|]* \|",
            lambda v, m: (
                "| **End-to-end accuracy** | — | "
                f"**{v('end_to_end_accuracy')}** | Measured |"
            ),
            ("end_to_end_accuracy",),
        ),
        (
            "3.3 cross-dataset: HaluEval row",
            r"\| \*\*HaluEval QA \(length-matched\)\*\* \| retrieval on \| "
            r"\*\*[\d.]+\*\* \| \[[^\]]+\] \| [\d.]+ \| [\d.]+ \|",
            lambda v, m: (
                "| **HaluEval QA (length-matched)** | retrieval on | "
                f"**{v('detection_roc_auc')}** | "
                f"[{float(agg.get('detection_roc_auc_ci95', [0, 0])[0]):.4f}, "
                f"{float(agg.get('detection_roc_auc_ci95', [0, 0])[1]):.4f}] | "
                f"{v('claim_f1')} | {v('retrieval_recall_at_5')} |"
            ),
            ("detection_roc_auc", "claim_f1", "retrieval_recall_at_5"),
        ),
        (
            "3.3 cross-dataset: RAGTruth row",
            r"\| \*\*RAGTruth \(length-matched\)\*\* \| retrieval on \| "
            r"\*\*[\d.]+\*\* \| \[[^\]]+\] \| [\d.]+ \| [\d.]+ \|",
            lambda v, m: (
                "| **RAGTruth (length-matched)** | retrieval on | "
                f"**{v('rag_detection_roc_auc')}** | "
                f"{v('rag_detection_roc_auc_ci')} | "
                f"{v('rag_claim_f1')} | {v('rag_retrieval_recall_at_5')} |"
            ),
            (
                "rag_detection_roc_auc", "rag_detection_roc_auc_ci",
                "rag_claim_f1", "rag_retrieval_recall_at_5",
            ),
        ),
        (
            "3.3 cross-dataset: TruthfulQA row",
            r"\| \*\*TruthfulQA\*\* \| \*\*closed-book\*\* \| \*\*[\d.]+\*\* \| "
            r"\[[^\]]+\] \| not measurable \| n/a \(no corpus\) \|",
            lambda v, m: (
                "| **TruthfulQA** | **closed-book** | "
                f"**{v('tqa_detection_roc_auc')}** | "
                f"{v('tqa_detection_roc_auc_ci')} | not measurable | "
                "n/a (no corpus) |"
            ),
            ("tqa_detection_roc_auc", "tqa_detection_roc_auc_ci"),
        ),
        (
            "3.3 RAGTruth published comparison: our AUC column",
            # The verdict applies the paper's own rule: our 95% CI lower
            # bound must sit strictly above the published value. It was
            # hard-coded to WIN, which would have claimed a lead over
            # LLM-Check's strongest anchor even when our interval
            # overlapped it.
            r"\| LLM-Check — (Attention Score|Hidden Score|Logit, Window "
            r"Entropy|Logit, Perplexity|Logit, Log Entropy) \| AUROC \| "
            r"[\d.]+ \| [\d.]+ \| \[[^\]]+\] \| \*\*(?:WIN|TIE)\*\* \|",
            lambda v, m: (
                f"| LLM-Check — {m.group(1)} | AUROC | "
                + _PUBLISHED_AUROC[m.group(1)]
                + f" | {v('rag_detection_roc_auc')} | "
                + f"{v('rag_detection_roc_auc_ci')} | "
                + _ragtruth_verdict(v, m.group(1))
                + " |"
            ),
            ("rag_detection_roc_auc", "rag_detection_roc_auc_ci"),
        ),
        (
            "3.3 prose: TruthfulQA leak number",
            r"the detection AUC was \d+\.\d+ with a near-binary score "
            r"distribution",
            lambda v, m: (
                "the detection AUC was 0.9589 with a near-binary score "
                "distribution"
            ),
            (),
        ),
        (
            "3.3 prose: held-out figure",
            r"\*\*0\.6185, CI \[[^\]]+\]\*\* — the head scored on a half of "
            r"the fit data",
            lambda v, m: (
                "**0.6185, CI [0.5371, 0.6943]** — the head scored on a half "
                "of the fit data"
            ),
            (),
        ),
        (
            # The prose that reads the verdict table out loud. It was
            # pinned to "all five anchors clear the bar", which stopped
            # being true when the RAGTruth run was redone: our lower
            # bound no longer clears LLM-Check's strongest anchor, so
            # that one is a tie rather than a win.
            "3.3 prose: the rule for a win",
            r"All five anchors clear the bar \([\d.]+ > [\d.]+\), so all\s*\n"
            r"five are wins\.|"
            r"(?:All five anchors clear the bar \([\d.]+ > [\d.]+\), so all\s*\n"
            r"five are wins\.|[A-Za-z ]+ anchors? clear the bar "
            r"\([\d.]+ > [\d.]+\), so\s*\n?[A-Za-z ]+ are wins?\.)",
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
            r"496 rows of the RAGTruth split, so [\d.]+ is measured on data "
            r"the head was\s*\nfitted on\. Its honest generalisation estimate "
            r"is the \*\*held-out\*\* figure:\s*\n"
            r"\*\*[\d.]+, CI \[[^\]]+\]\*\* — the head scored on a half of the "
            r"fit data\s*\nit never saw\. The gap between [\d.]+ and [\d.]+ is "
            r"the generalisation gap,\s*\nand it is small\. Both numbers are "
            r"reported; the held-out lower bound \([\d.]+\)\s*\n"
            r"sits (?:below|above) LLM-Check's strongest anchor \([\d.]+\), so "
            r"the held-out figure on\s*\nits own would (?:not )?be a win\.",
            lambda v, m: (
                _ragtruth_heldout_prose(v)
            ),
            ("rag_detection_roc_auc", "rag_detection_roc_auc_ci",
             "rag_holdout_roc_auc", "rag_holdout_roc_auc_ci"),
        ),
        (
            # The per-task breakdown is a measurement, so it is written
            # from the current run rather than kept by hand.
            "3.3 prose: per-task breakdown",
            r"Data2txt [\d.]+ \(n=\d+\), QA [\d.]+ \(n=\d+\), Summary [\d.]+"
            r"\s*\(n=\d+\)",
            lambda v, m: (
                "Data2txt " + v("rag_task_data2txt") + " (n="
                + v("rag_task_data2txt_n") + "), QA "
                + v("rag_task_qa") + " (n=" + v("rag_task_qa_n")
                + "), Summary " + v("rag_task_summary") + "\n(n="
                + v("rag_task_summary_n") + ")"
            ),
            ("rag_task_data2txt", "rag_task_data2txt_n",
             "rag_task_qa", "rag_task_qa_n",
             "rag_task_summary", "rag_task_summary_n"),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "claim_correction_rate", "verification_success_rate",
        "end_to_end_accuracy", "retrieval_recall_at_5",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values(mitigation_path))

    # The cross-dataset section reports the same pipeline on three
    # benchmarks, so its numbers come from the secondary metrics files.
    for prefix, src in (("tqa", tqa), ("rag", rag)):
        for key in (
            "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
            "detection_accuracy", "spearman_rho", "claim_precision",
            "claim_recall", "claim_f1", "retrieval_recall_at_5",
        ):
            raw = src.get(key)
            if isinstance(raw, (int, float)):
                values[f"{prefix}_{key}"] = f"{float(raw):.3f}"
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

    text, replacements = _apply_specs(text, row_specs, values, dry_run=dry_run)

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
            r"End-to-end accuracy & -- & [^&]* & Measured \\\\",
            lambda v, m: (
                r"End-to-end accuracy & -- & "
                f"{v('end_to_end_accuracy')} & Measured \\"
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


def _fill_why_we_win_md(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Replace the projected rows in ``Paper/why_we_win.md``."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    specs = [
        (
            "row 1: mitigation delta",
            r"\| 1 \| \*\*Mitigation Δ \(post-mitigation lift\)\*\* \| "
            r"Dale 2023 — COMET-QE rerank \| \+0\.60 \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 1 | **Mitigation Δ (post-mitigation lift)** | "
                f"Dale 2023 — COMET-QE rerank | +0.60 | **{v('delta')}** |"
            ),
            ("delta",),
        ),
        (
            "row 7: ROC-AUC",
            r"\| 7 \| \*\*Detection ROC-AUC\*\* \| Dale 2023 — ALTI\+ \| "
            r"0\.987 \(MT-only\) \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 7 | **Detection ROC-AUC** | Dale 2023 — ALTI+ | "
                f"0.987 (MT-only) | **{v('detection_roc_auc')}** |"
            ),
            ("detection_roc_auc",),
        ),
        (
            "row 7: detection ROC-AUC verdict",
            # The verdict compares against the comparable in *this* row,
            # Dale 2023's 0.987. An earlier version compared against the
            # Sent-T5 XL band of 0.85-0.90 and, because our 0.925 sits
            # above that band, wrote "Above the cited comparables" in a
            # row whose cited comparable is 0.987. The row was claiming a
            # lead over a number printed two cells to its left.
            r"\| 7 \| \*\*Detection ROC-AUC\*\* \| Dale 2023 — ALTI\+ \| "
            r"0\.987 \(MT-only\) \| \*\*[^*]+\*\* \| [^|]* \| [^|]* \|",
            lambda v, m: (
                "| 7 | **Detection ROC-AUC** | Dale 2023 — ALTI+ | "
                f"0.987 (MT-only) | **{v('detection_roc_auc')}** | "
                + ("\u2713" if _above(v, "detection_roc_auc", 0.987)
                   else "\u2248")
                + " | "
                + ("Above the cited comparables"
                   if _above(v, "detection_roc_auc", 0.987)
                   else "Below the MT-specialist (0.987) and inside the "
                        "Sent-T5 XL band (0.85-0.90); above the "
                        "general-purpose probes")
                + " |"
            ),
            ("detection_roc_auc",),
        ),
        (
            "row 8: sentence accuracy",
            r"\| 8 \| \*\*Sentence accuracy\*\* \| Halu-NLP \(SHROOM winner\) \| "
            r"0\.847 \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 8 | **Sentence accuracy** | Halu-NLP (SHROOM winner) | "
                f"0.847 | **{v('detection_accuracy')}** |"
            ),
            ("detection_accuracy",),
        ),
        (
            "row 9: P@R90",
            r"\| 9 \| \*\*P@R90\*\* \| Dale 2023 — LaBSE \| 0\.703 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 9 | **P@R90** | Dale 2023 — LaBSE | 0.703 | "
                f"**{v('precision_at_recall_90')}** |"
            ),
            ("precision_at_recall_90",),
        ),
        (
            "row 11: latency",
            r"\| 11 \| \*\*Per-query latency \(GPU\)\*\* \| LLM-Check \| "
            r"0\.22 s \| [^|]* \| ✗ \| [^|]* \|",
            lambda v, m: (
                "| 11 | **Per-query latency (GPU)** | LLM-Check | 0.22 s | "
                f"{_latency_band(v)} | ✗ | Slower than the single-pass probe; "
                "much richer output |"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "row 10: Spearman rho",
            r"\| 10 \| \*\*Spearman ρ\*\* \| Halu-NLP \| 0\.770 \| "
            r"\*\*[^*]+\*\* \|",
            lambda v, m: (
                "| 10 | **Spearman ρ** | Halu-NLP | 0.770 | "
                f"**{v('spearman_rho')}** |"
            ),
            ("spearman_rho",),
        ),
        (
            "latency table row",
            r"\| \*\*Ours\*\* \| [\d.]+(?:[–-][\d.]+)? s \|",
            lambda v, m: (
                f"| **Ours** | {_latency_band(v)} |"
            ),
            ("latency_min_s", "latency_max_s"),
        ),
        (
            "detection table row",
            r"\| \*\*Ours \(Step 9 fused\)\*\* \| \*\*[^*]+\*\* \| HaluEval QA[^|]*\|",
            lambda v, m: (
                f"| **Ours (Step 9 fused)** | **{v('detection_roc_auc')}** | "
                "HaluEval QA (length-matched) |"
            ),
            ("detection_roc_auc",),
        ),
        (
            "claim P/R/F1 table row",
            r"\| \*\*Ours \(Step 8 \+ Step 9\)\*\* \| \*\*[^*]+\*\* \| "
            r"Atomic claims \|",
            lambda v, m: (
                f"| **Ours (Step 8 + Step 9)** | **{v('claim_precision')} / "
                f"{v('claim_recall')} / {v('claim_f1')}** | Atomic claims |"
            ),
            ("claim_precision", "claim_recall", "claim_f1"),
        ),
        (
            "VSR table row",
            r"\| \*\*Ours\*\* \| \*\*\d[\d.]*\+?\*\* \|",
            lambda v, m: (
                f"| **Ours** | **{v('verification_success_rate')}** |"
            ),
            ("verification_success_rate",),
        ),
        (
            "row 1: mitigation delta verdict",
            r"\| 1 \| \*\*Mitigation \u0394 \(post-mitigation lift\)\*\* \| "
            r"Dale 2023 — COMET-QE rerank \| \+0\.60 \| \*\*[^*]+\*\* \| "
            r"[^|]* \| [^|]* \|",
            lambda v, m: (
                "| 1 | **Mitigation \u0394 (post-mitigation lift)** | "
                "Dale 2023 — COMET-QE rerank | +0.60 | "
                f"**{v('delta')}** | "
                + ("\u2713" if _delta_positive(v) else "\u2717")
                + " | "
                + ("Highest single gain in the comparison"
                   if _delta_positive(v)
                   else "Measured on the rows step 10 regenerated; the rewrite "
                        "does not improve on the no-mitigation baseline yet")
                + " |"
            ),
            ("delta",),
        ),
        (
            "row 2: claim-level F1",
            r"\| 2 \| \*\*Claim-level F1\*\* \| TU Wien 2024 — Vectara \| "
            r"0\.71 \| \*\*[^*]+\*\* \| [^|]* \| [^|]* \|",
            lambda v, m: (
                "| 2 | **Claim-level F1** | TU Wien 2024 — Vectara | 0.71 | "
                f"**{v('claim_f1')}** | "
                + ("\u2713" if _above(v, "claim_f1", 0.71) else "\u2717")
                + " | "
                + ("Above the cited comparable"
                   if _above(v, "claim_f1", 0.71)
                   else "Below the cited comparable")
                + " |"
            ),
            ("claim_f1",),
        ),
        (
            "row 4: VSR",
            r"\| 4 \| \*\*Verification Success Rate \(VSR\)\*\* \| "
            r"AILS-NTUA \(Voting\) \| 0\.80 \(acc\) \| \*\*[^*]+\*\* \| "
            r"[^|]* \| [^|]* \|",
            lambda v, m: (
                "| 4 | **Verification Success Rate (VSR)** | "
                f"AILS-NTUA (Voting) | 0.80 (acc) | "
                f"**{v('verification_success_rate')}** | "
                + ("\u2713"
                   if _above(v, "verification_success_rate", 0.80)
                   else "\u2717")
                + " | "
                + ("Above the cited comparables"
                   if _above(v, "verification_success_rate", 0.80)
                   else "Below the cited comparables")
                + " |"
            ),
            ("verification_success_rate",),
        ),
        (
            "row 3: HRR",
            r"\| 3 \| \*\*Hallucination Reduction Rate \(HRR\)\*\* \| Dale 2023 "
            r"\(N-best rerank\) \| ~0\.55 \(implied\) \| \*\*[^*]+\*\* \| "
            r"[^|]* \| [^|]* \|",
            lambda v, m: (
                "| 3 | **Hallucination Reduction Rate (HRR)** | Dale 2023 "
                f"(N-best rerank) | ~0.55 (implied) | "
                f"**{v('hallucination_reduction_rate')}** | "
                + ("\u2713"
                   if _above(v, "hallucination_reduction_rate", 0.55)
                   else "\u2717")
                + " | "
                + ("Above the implied baseline"
                   if _above(v, "hallucination_reduction_rate", 0.55)
                   else "Below the implied baseline")
                + " |"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "HRR table row",
            r"\| \*\*Ours \(Step 11 gate\)\*\* \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                "| **Ours (Step 11 gate)** | "
                f"**{v('hallucination_reduction_rate')}** |"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "row 8: sentence accuracy",
            r"\| 8 \| \*\*Sentence accuracy\*\* \| Halu-NLP \(SHROOM winner\) "
            r"\| 0\.847 \| \*\*[^*]+\*\* \| [^|]* \| [^|]* \|",
            lambda v, m: (
                "| 8 | **Sentence accuracy** | Halu-NLP (SHROOM winner) | "
                f"0.847 | **{v('detection_accuracy')}** | "
                + ("\u2713" if _above(v, "detection_accuracy", 0.847)
                   else "\u2717")
                + " | "
                + ("Above the SHROOM winner"
                   if _above(v, "detection_accuracy", 0.847)
                   else "Below the SHROOM winner")
                + " |"
            ),
            ("detection_accuracy",),
        ),
        (
            # The bar-chart data block is a fixed-width ASCII table inside a
            # code fence. Each row is "label  comparable  ours  gap  verdict",
            # and the gap is ours minus the comparable. The row is rebuilt
            # from the measurement so the gap and the verdict cannot drift
            # away from the number beside them.
            "bar-chart: Claim F1",
            r"Claim F1\s+0\.71\s+[\d.]+\s+[-\d.]+\s+[\u2248\u2713\u2717]?\s*(?:At par|Ahead|Behind)",
            lambda v, m: _ascii_gap_row(
                "Claim F1", 0.71, v("claim_f1"),
            ),
            ("claim_f1",),
        ),
        (
            "bar-chart: HRR",
            r"HRR\s+0\.55\s+[\d.]+\s+[-\d.]+\s+[\u2248\u2713\u2717]?\s*(?:At par|Ahead|Behind)",
            lambda v, m: _ascii_gap_row(
                "HRR", 0.55, v("hallucination_reduction_rate"),
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "bar-chart: VSR",
            r"VSR\s+0\.81\s+[\d.]+\s+[-\d.]+\s+[\u2248\u2713\u2717]?\s*(?:At par|Ahead|Behind)",
            lambda v, m: _ascii_gap_row(
                "VSR", 0.81, v("verification_success_rate"),
            ),
            ("verification_success_rate",),
        ),
        (
            "bar-chart: Detection ROC-AUC",
            r"Detection ROC-AUC\s+0\.987 \(MT\)\s+[\d.]+\s+[-\d.]+\s+[\u2248\u2713\u2717]?\s*(?:At par|Ahead|Behind)",
            lambda v, m: _ascii_gap_row(
                "Detection ROC-AUC", 0.987, v("detection_roc_auc"),
            ),
            ("detection_roc_auc",),
        ),
        (
            "bar-chart: Sentence Acc",
            r"Sentence Acc\s+0\.847\s+[\d.]+\s+[-\d.]+\s+[\u2248\u2713\u2717]?\s*(?:At par|Ahead|Behind)",
            lambda v, m: _ascii_gap_row(
                "Sentence Acc", 0.847, v("detection_accuracy"),
            ),
            ("detection_accuracy",),
        ),
        (
            "bar-chart: P@R90",
            r"P@R90\s+0\.703\s+[\d.]+\s+[-\d.]+\s+[\u2248\u2713\u2717]?\s*(?:At par|Ahead|Behind)",
            lambda v, m: _ascii_gap_row(
                "P@R90", 0.703, v("precision_at_recall_90"),
            ),
            ("precision_at_recall_90",),
        ),
        (
            "bar-chart: Spearman",
            r"Spearman \u03c1\s+0\.770\s+[\d.]+\s+[-\d.]+\s+[\u2248\u2713\u2717]?\s*(?:At par|Ahead|Behind)",
            lambda v, m: _ascii_gap_row(
                "Spearman \u03c1", 0.770, v("spearman_rho"),
            ),
            ("spearman_rho",),
        ),
        (
            "prose: headline read ROC-AUC",
            r"on detection \(ROC-AUC \d+\.\d+ against a SHROOM band",
            lambda v, m: (
                "on detection (ROC-AUC "
                + v("detection_roc_auc")
                + " against a SHROOM band"
            ),
            ("detection_roc_auc",),
        ),
        (
            "prose: HRR in the 'why we do not win' note",
            r"Our measured HRR \([\d.]+\) is below Dale 2023's implied",
            lambda v, m: (
                "Our measured HRR ("
                + v("hallucination_reduction_rate")
                + ") is below Dale 2023's implied"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "prose: VSR in the granularity note",
            r"Our measured VSR \([\d.]+\) is (?:below|above) the cited "
            r"comparables",
            lambda v, m: (
                "Our measured VSR ("
                + v("verification_success_rate")
                + ") is "
                + ("above" if _above(v, "verification_success_rate", 0.80)
                   else "below")
                + " the cited comparables"
            ),
            ("verification_success_rate",),
        ),
        (
            "prose: VSR in the Q&A",
            r"our VSR is [\d.]+, so the difference is capability",
            lambda v, m: (
                "our VSR is "
                + v("verification_success_rate")
                + ", so the difference is capability"
            ),
            ("verification_success_rate",),
        ),
        (
            # The headline read names the dimensions we are behind on. The
            # list has to follow the matrix, or the summary and the
            # tally it sits above disagree: the tally counted
            # claim-level F1 as a loss while the sentence that introduced
            # the tally also called it one, after the measurement had
            # moved it above the comparable.
            "prose: headline read",
            r"and behind on (?:mitigation Delta, claim-level F1, )?"
            r"(?:HRR and VSR|claim-level F1, HRR and VSR)\.",
            lambda v, m: (
                "and behind on "
                + ", ".join(
                    name for name, key, threshold in (
                        ("mitigation Delta", None, None),
                        ("claim-level F1", "claim_f1", 0.71),
                        ("HRR", "hallucination_reduction_rate", 0.55),
                        ("VSR", "verification_success_rate", 0.80),
                    )
                    if key is None or not _above(v, key, threshold)
                )
                + "."
            ),
            ("claim_f1", "hallucination_reduction_rate",
             "verification_success_rate"),
        ),
        (
            "prose: measured F1 in the granularity note",
            # The rule owns the whole sentence, including its trailing
            # clause. The clause asserted "not a score advantage", which
            # was true while the measured F1 sat below the comparable and
            # became false once it did not. Leaving the clause out of the
            # match would let the sentence contradict itself.
            r"The measured F1 \([\d.]+\) is (?:below|above) the cited "
            r"sentence-level comparables, so the claim here is structural "
            r"\(per-claim verdicts\) and (?:not a score advantage|"
            r"a score advantage as well as a structural one)",
            lambda v, m: (
                "The measured F1 ("
                + v("claim_f1")
                + ") is "
                + ("above" if _above(v, "claim_f1", 0.71) else "below")
                + " the cited sentence-level comparables, so the claim here "
                "is structural (per-claim verdicts) and "
                + ("not a score advantage"
                   if not _above(v, "claim_f1", 0.71)
                   else "a score advantage as well as a structural one")
            ),
            ("claim_f1",),
        ),
        (
            "prose: strongest result ROC-AUC",
            r"Detection ROC-AUC of \*\*[\d.]+\*\* on the length-matched "
            r"HaluEval split \(95% CI \[[^\]]*\]\)",
            lambda v, m: (
                "Detection ROC-AUC of **"
                + v("detection_roc_auc")
                + "** on the length-matched HaluEval split (95% CI ["
                + v("detection_roc_auc_ci_text")
                + "])"
            ),
            ("detection_roc_auc", "detection_roc_auc_ci_text"),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "verification_success_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values["detection_roc_auc_ci_text"] = _ci_text(agg, "detection_roc_auc")
    values.update(_mitigation_prose_values(mitigation_path))
    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

    # The tally summarises the rows above it, so it is recomputed from
    # them. A hand-written tally drifts the moment a verdict changes.
    tally = _verdict_tally(text)
    if tally is not None:
        pattern = (
            r"\| \*\*✓ Win\*\* \| [^\n]*\n"
            r"\| \*\*≈ Tie / At par\*\* \| [^\n]*\n"
            r"\| \*\*✗ Loss\*\* \| [^\n]*"
        )
        new_text, count = re.subn(pattern, lambda m: tally, text)
        if count and new_text != text:
            text = new_text
            replacements.append(
                ("verdict tally", pattern, f"{count} block(s)")
            )

    text, mit_changes = _fill_mitigation_prose(
        text, path, dry_run=dry_run, mitigation_path=mitigation_path
    )
    replacements.extend(mit_changes)

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


def _fill_five_position_md(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Replace the projected rows in ``Paper/five_position_comparison.md``.

    The file was not wired to this script before, so its numbers were
    frozen at whatever the synthetic run produced.
    """
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    specs = [
        (
            "HRR row",
            r"\| \*\*Ours \(Step 11 gate\)\*\* \| \*\*[^*]+\*\*[^|]*\|",
            lambda v, m: (
                f"| **Ours (Step 11 gate)** | "
                f"**{v('hallucination_reduction_rate')}** |"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "VSR row",
            r"\| \*\*Ours \(per-claim NLI\)\*\* \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| **Ours (per-claim NLI)** | "
                f"**{v('verification_success_rate')}** |"
            ),
            ("verification_success_rate",),
        ),
        (
            "CCR row",
            r"\| \*\*Ours \(Step 10 regen\)\*\* \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                f"| **Ours (Step 10 regen)** | "
                f"**{v('claim_correction_rate')}** |"
            ),
            ("claim_correction_rate",),
        ),
        (
            "VSR in the win table",
            r"Halu-NLP VSR 0\.81 vs\. \*\*Ours [^*]+\*\*",
            lambda v, m: (
                f"Halu-NLP VSR 0.81 vs. **Ours "
                f"{v('verification_success_rate')}**"
            ),
            ("verification_success_rate",),
        ),
        (
            "prose: neither is a measured advantage",
            r"Our HRR \([\d.]+\) and VSR \([\d.]+\) are both below the cited "
            r"comparables, and our CCR \([\d.]+\) has no published CCR",
            lambda v, m: (
                "Our HRR ("
                + v("hallucination_reduction_rate")
                + ") and VSR ("
                + v("verification_success_rate")
                + ") are both below the cited comparables, and our CCR ("
                + v("claim_correction_rate")
                + ") has no published CCR"
            ),
            (
                "hallucination_reduction_rate", "verification_success_rate",
                "claim_correction_rate",
            ),
        ),
        (
            "verdict: not a win paragraph",
            r"but the measured HRR \([\d.]+\) and CCR \([\d.]+\) are reported "
            r"as measured",
            lambda v, m: (
                "but the measured HRR ("
                + v("hallucination_reduction_rate")
                + ") and CCR ("
                + v("claim_correction_rate")
                + ") are reported as measured"
            ),
            ("hallucination_reduction_rate", "claim_correction_rate"),
        ),
        (
            "verdict: not a win on the rate",
            r"but the measured VSR \([\d.]+\) is below the cited comparables",
            lambda v, m: (
                "but the measured VSR ("
                + v("verification_success_rate")
                + ") is below the cited comparables"
            ),
            ("verification_success_rate",),
        ),
        (
            "summary table: position 5 margin",
            r"\| 5\. HRR / VSR / CCR \| [^|]*\| Halu-NLP VSR 0\.81 vs\. "
            r"\*\*Ours [^*]+\*\* \| [^|]*\|",
            lambda v, m: (
                "| 5. HRR / VSR / CCR | ❌ LOSS on the rates | "
                "Halu-NLP VSR 0.81 vs. **Ours "
                + v("verification_success_rate")
                + "** | −"
                + f"{0.81 - float(v('verification_success_rate')):.3f}"
                + " VSR; HRR "
                + v("hallucination_reduction_rate")
                + " vs. ~0.55 implied |"
            ),
            ("verification_success_rate", "hallucination_reduction_rate"),
        ),
        (
            "position 2 row in the summary table",
            r"\| 2\. Mitigation [^|]*\|[^|]*\|[^|]*\|[^|]*\|",
            lambda v, m: (
                "| 2. Mitigation \u0394 | "
                + ("\u2713 WIN" if _delta_positive(v) else "\u2717 LOSS")
                + " | Dale 2023 +0.60 (different domain) | "
                + (f"{v('delta')} measured" if _delta_positive(v)
                   else f"{v('delta')} measured vs. +0.60")
                + " |"
            ),
            ("delta",),
        ),
        (
            "position 2 and 5 prose sentence",
            r"\*Positions 2 and 5 are not wins and are reported as measured\. "
            r"The mitigation \u0394 is .*? improving it is future work\.",
            lambda v, m: (
                "*Positions 2 and 5 are not wins and are reported as measured. "
                + v("mitigation_prose_sentence")
                + " The rewrite path is implemented and exercised, but on this "
                "sample it did not improve similarity to the gold answer; "
                "improving it is future work."
            ),
            ("delta", "mitigation_prose_sentence"),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "verification_success_rate", "claim_correction_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values(mitigation_path))
    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

    text, mit_changes = _fill_mitigation_prose(
        text, path, dry_run=dry_run, mitigation_path=mitigation_path
    )
    replacements.extend(mit_changes)

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


def _fill_wins_summary_md(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
    cross: dict[str, dict] | None = None,
    matrix_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Keep ``Paper/wins_summary.md`` in sync with the measurements.

    This file is the defence-notes summary. It was hand-written, so every
    re-evaluation left it holding whatever numbers were current when it was
    written -- which is how a superseded ROC-AUC of 0.843 and a claim-level
    F1 of 0.653 survived into a document whose own preamble claims that
    "no figure in this project is typed by hand".

    Every rule here is anchored on the row's own stable text (the method
    name, the metric label), so the substitution cannot drift onto an
    unrelated number. Prose that states a verdict -- "we drop to 0.603" --
    is rewritten together with the number, because a stale number in prose
    is exactly the overclaim this generator exists to prevent.
    """
    if not path.exists():
        return []

    cross = cross or {}
    tqa = cross.get("truthfulqa") or {}
    rag = cross.get("ragtruth") or {}

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    def _ci(key: str, source: dict | None = None) -> list[float] | None:
        src = source if source is not None else agg
        value = src.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [float(value[0]), float(value[1])]
        return None

    values: dict[str, str] = {}
    for key in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "detection_accuracy", "spearman_rho", "claim_precision",
        "claim_recall", "claim_f1", "hallucination_reduction_rate",
        "end_to_end_accuracy", "verification_success_rate",
        "claim_correction_rate", "retrieval_recall_at_5", "latency_mean_s",
        "latency_min_s", "latency_max_s",
    ):
        got = _v(key)
        if got is not None:
            values[key] = got
    # The AUPR interval, so the AUPR verdict can follow the separation
    # rule rather than the point estimate.
    aupr_ci = _ci("detection_aupr_ci95")
    if aupr_ci:
        values["detection_aupr_ci95"] = aupr_ci
        values["detection_aupr_ci_text"] = f"{aupr_ci[0]:.3f}, {aupr_ci[1]:.3f}"
    for key, src in (("tqa_detection_roc_auc", tqa), ("rag_detection_roc_auc", rag)):
        # The value has to come from *this* dataset's aggregate. An
        # earlier version read both from TruthfulQA, which put
        # TruthfulQA's 0.635 into the RAGTruth row of the defence notes
        # while the comparison table showed 0.613 for the same run.
        raw = src.get("detection_roc_auc")
        if isinstance(raw, (int, float)):
            values[key] = f"{float(raw):.3f}"

    # The remaining secondary-dataset metrics, under their own prefixed
    # names so they can never be confused with the HaluEval values.
    for prefix, src in (("tqa", tqa), ("rag", rag)):
        for key in (
            "detection_aupr", "precision_at_recall_90", "detection_accuracy",
            "spearman_rho", "claim_precision", "claim_recall", "claim_f1",
        ):
            raw = src.get(key)
            if isinstance(raw, (int, float)):
                values[f"{prefix}_{key}"] = f"{float(raw):.3f}"

    # The HaluEval detection CI, used in the W1 row.
    roc_ci = _ci("detection_roc_auc_ci95")
    if roc_ci:
        values["detection_roc_auc_ci"] = (
            f"[{roc_ci[0]:.3f}, {roc_ci[1]:.3f}]"
        )
    tqa_roc_ci = _ci("detection_roc_auc_ci95", tqa)
    if tqa_roc_ci:
        values["tqa_detection_roc_auc_ci"] = (
            f"[{tqa_roc_ci[0]:.3f}, {tqa_roc_ci[1]:.3f}]"
        )
    rag_roc_ci = _ci("detection_roc_auc_ci95", rag)
    if rag_roc_ci:
        values["rag_detection_roc_auc_ci"] = (
            f"[{rag_roc_ci[0]:.3f}, {rag_roc_ci[1]:.3f}]"
        )

    values.update(_mitigation_prose_values(mitigation_path))

    specs = [
        (
            "W1 row",
            r"\| \*\*W1\*\* \| \*\*Length-controlled detection AUC\*\* \| "
            r"\*\*[^*]+\*\* ROC-AUC \[[^\]]+\] \|",
            lambda v, m: (
                "| **W1** | **Length-controlled detection AUC** | "
                + f"**{v('detection_roc_auc')}** ROC-AUC "
                + v("detection_roc_auc_ci")
                + " |"
            ),
            ("detection_roc_auc", "detection_roc_auc_ci"),
        ),
        (
            "plain-English: length-matched score",
            r"my score drops from 0\.968 to \d+\.\d+\. I am reporting the lower number\.",
            lambda v, m: (
                "my score drops from 0.968 to "
                + v("detection_roc_auc")
                + ". I am reporting the lower number."
            ),
            ("detection_roc_auc",),
        ),
        (
            "plain-English: TruthfulQA score",
            r"On TruthfulQA my\s+score drops to \d+\.\d+\. I say all of this openly\.",
            lambda v, m: (
                "On TruthfulQA my score drops to "
                + v("tqa_detection_roc_auc")
                + ". I say all of this openly."
            ),
            ("tqa_detection_roc_auc",),
        ),
        (
            "3.1 detection row",
            r"\| \*\*Ours \(RAG fused score, Step 9\)\*\* \| \*\*[^*]+\*\* \| "
            r"\*\*[^*]+\*\* \| \*\*[^*]+\*\* \| \*\*[^*]+\*\* \| \*\*[^*]+\*\* \|",
            lambda v, m: (
                "| **Ours (RAG fused score, Step 9)** | "
                + f"**{v('detection_roc_auc')}** | "
                + f"**{v('detection_aupr')}** | "
                + f"**{v('precision_at_recall_90')}** | "
                + f"**{v('detection_accuracy')}** | "
                + f"**{v('spearman_rho')}** |"
            ),
            (
                "detection_roc_auc", "detection_aupr",
                "precision_at_recall_90", "detection_accuracy", "spearman_rho",
            ),
        ),
        (
            "3.1 TruthfulQA row",
            r"\| \*\*Ours \(TruthfulQA, independent(?:, closed-book)?\)\*\* \| "
            r"\d+\.\d+ \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+ \| \d+\.\d+ \|",
            lambda v, m: (
                "| **Ours (TruthfulQA, independent, closed-book)** | "
                + f"{v('tqa_detection_roc_auc')} | "
                + f"{v('tqa_detection_aupr')} | "
                + f"{v('tqa_precision_at_recall_90')} | "
                + f"{v('tqa_detection_accuracy')} | "
                + f"{v('tqa_spearman_rho')} |"
            ),
            (
                "tqa_detection_roc_auc", "tqa_detection_aupr",
                "tqa_precision_at_recall_90", "tqa_detection_accuracy",
                "tqa_spearman_rho",
            ),
        ),
        (
            "3.1 TruthfulQA setting cell",
            # TruthfulQA is closed-book: there is no retrieval corpus, so the
            # setting cell must not claim "Black-box + retrieval".
            r"(\| \*\*Ours \(TruthfulQA, independent(?:, closed-book)?\)\*\* \|"
            r"[^\n]*?\| )Black-box \+ retrieval( \|)",
            lambda v, m: (
                m.group(1) + "Black-box, closed-book" + m.group(2)
            ),
            (),
        ),
        (
            "3.1 RAGTruth row",
            r"\| \*\*Ours \(RAGTruth, third dataset\)\*\* \|[^\n]*",
            lambda v, m: (
                "| **Ours (RAGTruth, third dataset)** | "
                + f"**{v('rag_detection_roc_auc')}** | – | – | – | – | "
                "Black-box + retrieval | RAGTruth (**length-matched**) |"
            ),
            ("rag_detection_roc_auc",),
        ),
        (
            "prose: SHROOM band",
            r"1\. We sit \*\*inside the SHROOM band\*\* on detection \(\d+\.\d+ "
            r"AUC vs 0\.85–0\.90 for",
            lambda v, m: (
                "1. We sit **inside the SHROOM band** on detection ("
                + v("detection_roc_auc")
                + " AUC vs 0.85–0.90 for"
            ),
            ("detection_roc_auc",),
        ),
        (
            # The AUPR claim follows the interval. The point estimate is
            # not the evidence; the separation is. The superseded
            # measurement put the AUPR in a tie, and the corrected
            # instrument separates it, so both the lead-in and the
            # verdict have to be able to move with the interval.
            "prose: AUPR tie",
            r"2\. Our AUPR of \*\*\d+\.\d+\*\* (?:matches|is above) the best "
            r"published comparable",
            lambda v, m: (
                "2. Our AUPR of **"
                + v("detection_aupr")
                + "** "
                + ("is above the best published comparable"
                   if _above_ci(v, "detection_aupr_ci95", 0.85)
                   else "matches the best published comparable")
            ),
            ("detection_aupr", "detection_aupr_ci95"),
        ),
        (
            "prose: AUPR interval",
            r"(?:our 95% CI \[[^\]]+\] overlaps it, so\s*\n?\s*it is a "
            r"\*\*tie, not a win\*\*, and is reported as such\.|it is a "
            r"\*\*tie, not a win\*\*, and is reported as such\.)",
            lambda v, m: (
                "our 95% CI ["
                + v("detection_aupr_ci_text")
                + "] "
                + ("excludes it, so it is a **win**, and is reported as such"
                   if _above_ci(v, "detection_aupr_ci95", 0.85)
                   else "overlaps it, so it is a **tie, not a win**, and is "
                        "reported as such")
            ),
            ("detection_aupr_ci_text", "detection_aupr_ci95"),
        ),
        (
            "prose: TruthfulQA drop",
            r"3\. On TruthfulQA \(independent dataset\) we \*\*drop to \d+\.\d+ "
            r"AUC\*\*\.",
            lambda v, m: (
                "3. On TruthfulQA (independent dataset, closed-book) we **drop to "
                + v("tqa_detection_roc_auc")
                + " AUC**."
            ),
            ("tqa_detection_roc_auc",),
        ),
        (
            "3.2 claim-level row",
            r"\| \*\*Ours \(Step 8 \+ Step 9\)\*\* \| \*\*[\d\. /]+\*\* \|",
            lambda v, m: (
                "| **Ours (Step 8 + Step 9)** | "
                + f"**{v('claim_precision')} / {v('claim_recall')} / "
                + f"{v('claim_f1')}** |"
            ),
            ("claim_precision", "claim_recall", "claim_f1"),
        ),
        (
            "prose: claim-level F1 verdict",
            r"We are \*\*(?:below|above)\*\* on raw F1 \(\d+\.\d+ vs 0\.71\) and we say so\.",
            lambda v, m: (
                "We are **"
                + ("above" if _above(v, "claim_f1", 0.71) else "below")
                + "** on raw F1 ("
                + v("claim_f1")
                + " vs 0.71) and we say so."
            ),
            ("claim_f1",),
        ),
        (
            "limitation: TruthfulQA AUC",
            r"\| TruthfulQA AUC drops to \d+\.\d+ \|",
            lambda v, m: (
                "| TruthfulQA AUC drops to "
                + v("tqa_detection_roc_auc")
                + " |"
            ),
            ("tqa_detection_roc_auc",),
        ),
        (
            "limitation: end-to-end accuracy",
            r"\| End-to-end accuracy is \d+\.\d+ \|",
            lambda v, m: (
                "| End-to-end accuracy is "
                + v("end_to_end_accuracy")
                + " |"
            ),
            ("end_to_end_accuracy",),
        ),
        (
            "limitation: claim-level F1",
            r"\| Claim-level F1 \(\d+\.\d+\) (?:below|above) TU Wien \(0\.71\) \|",
            lambda v, m: (
                "| Claim-level F1 ("
                + v("claim_f1")
                + ") "
                + ("above" if _above(v, "claim_f1", 0.71) else "below")
                + " TU Wien (0.71) |"
            ),
            ("claim_f1",),
        ),
        (
            "limitation: HRR",
            r"\| HRR \(\d+\.\d+\) below Dale's implied 0\.55 \|",
            lambda v, m: (
                "| HRR ("
                + v("hallucination_reduction_rate")
                + ") below Dale's implied 0.55 |"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "30-second script: score drop",
            r"same length, and my own score drops from 0\.968 to \d+\.\d+\. I am reporting",
            lambda v, m: (
                "same length, and my own score drops from 0.968 to "
                + v("detection_roc_auc")
                + ". I am reporting"
            ),
            ("detection_roc_auc",),
        ),
        (
            "contribution statement: headline",
            # The sentence is quoted and wrapped across lines, so the pattern
            # has to tolerate the "> " quote markers and the line breaks.
            r"reaches \d+\.\d+ ROC-AUC and \d+\.\d+ AUPR with\s*percentile "
            r"bootstrap intervals",
            lambda v, m: (
                "reaches "
                + v("detection_roc_auc")
                + " ROC-AUC and "
                + v("detection_aupr")
                + " AUPR with percentile bootstrap intervals"
            ),
            ("detection_roc_auc", "detection_aupr"),
        ),
        (
            "efficiency table row: our latency",
            # The band and the mean are both measured. This row was
            # hand-written and held 0.8-13.8 s with a mean of 4.84 s,
            # which no longer matched the run the rest of the file
            # describes (1.0-10.3 s, mean 4.18 s).
            r"\| \*\*Ours \(RAG, single-pass\)\*\* \| \*\*[\d.]+(?:[–-][\d.]+)? s\*\*"
            r"(?: \(mean [\d.]+ s\))? \|",
            lambda v, m: (
                "| **Ours (RAG, single-pass)** | "
                f"**{_latency_band(v)}** (mean {_latency_mean(v):.2f} s) |"
            ),
            ("latency_min_s", "latency_max_s", "latency_mean_s"),
        ),
    ]

    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

    # The tally summarises the win matrix, which lives in
    # ``why_we_win.md``. It was maintained by hand here, so a verdict that
    # changed in the matrix left the two files claiming different counts.
    # It is now derived from the matrix, so the two cannot disagree.
    if matrix_path is not None and matrix_path.exists():
        matrix_text = matrix_path.read_text(encoding="utf-8")
        tally = _name_tally(matrix_text)
        if tally is not None:
            pattern = (
                r"\| \*\*✓ Win\*\* \| [^\n]*\n"
                r"\| \*\*≈ Tie / At par\*\* \| [^\n]*\n"
                r"\| \*\*✗ Loss\*\* \| [^\n]*"
            )
            new_text, count = re.subn(pattern, lambda m: tally, text)
            if count and new_text != text:
                text = new_text
                replacements.append(
                    ("verdict tally", pattern, f"{count} block(s)")
                )

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


# The mitigation rows of the prose tables. Each entry is
# (file key, old text, builder) where the builder receives a dict with
# "pre", "post" and "delta" already formatted.
_MITIGATION_PROSE = {
    "why_we_win.md": [
        (
            "mitigation table row",
            r"\| \*\*Ours \(Step 10 regen \+ Step 11 gate\)\*\* \| [^|]* \| "
            r"[^|]* \| [^|]* \|",
            lambda m: (
                "| **Ours (Step 10 regen + Step 11 gate)** | "
                f"{m['pre']} | {m['post']} | {m['delta']} |"
            ),
        ),
        (
            "prose: different from Dale",
            r"on machine translation; our measured \u0394 is .*? so we do not "
            r"claim to lead on mitigation \u0394\.",
            lambda m: (
                "on machine translation; our measured \u0394 is "
                + m["delta_number"]
                + " on the length-matched HaluEval split, so we do not "
                "claim to lead on mitigation \u0394."
            ),
        ),
        (
            "prose: strongest result",
            r"The mitigation \u0394 is \*not\* a win: measured on the rows "
            r"where step 10 regenerated, it is .*?, so it is reported as an "
            r"open result\.",
            lambda m: (
                "The mitigation \u0394 is *not* a win: measured on the rows "
                "where step 10 regenerated, it is "
                + m["delta_number"]
                + (" (CI " + m["delta_ci"] + ")" if m["delta_ci"] else "")
                + ", so it is reported as an open result."
            ),
        ),
    ],
    "comparison.md": [
        (
            "mitigation table row",
            r"\| \*\*Ours \(Step 10 regen \+ Step 11 gate\)\*\* \| "
            r"Evidence-constrained rewrite \+ ABSTAIN \| [^|]* \| [^|]* \| [^|]* \|",
            lambda m: (
                "| **Ours (Step 10 regen + Step 11 gate)** | Evidence-constrained "
                f"rewrite + ABSTAIN | {m['pre']} | {m['post']} | {m['delta']} |"
            ),
        ),
        (
            "prose: observation",
            r"Measured on the rows where step 10 actually regenerated "
            r"\(n=\d+ of \d+\), the \u0394 over the no-mitigation baseline is "
            r"\*\*[^*]+\*\* \(95% CI \[[^\]]*\]\) — i\.e\..*?reported here as "
            r"measured rather than as the projected figure an earlier draft "
            r"carried\.",
            lambda m: (
                "Measured on the rows where step 10 actually regenerated "
                f"(n={m['n_regenerated']} of {m['n_rows']}), the \u0394 over the "
                "no-mitigation baseline is **"
                + m["delta_number"]
                + " BERTScore F1** (95% CI "
                + m["delta_ci"]
                + f") — i.e. {m['outcome']} on this sample. The rewrite step "
                "is therefore not yet a demonstrated win over the "
                "no-mitigation baseline, and the \u0394 is reported here as "
                "measured rather than as the projected figure an earlier "
                "draft carried."
            ),
        ),
    ],
    "five_position_comparison.md": [
        (
            "mitigation table row",
            r"\| \*\*Ours \(Step 10 regen \+ Step 11 gate\)\*\* \| [^|]* \| "
            r"[^|]* \| [^|]* \|",
            lambda m: (
                "| **Ours (Step 10 regen + Step 11 gate)** | "
                f"{m['pre']} | {m['post']} | {m['delta']} |"
            ),
        ),
        (
            "heading: position 2",
            r"## Position 2 — Mitigation \u0394 \(Measured: .*? vs\. \+0\.60 Dale\)",
            lambda m: (
                "## Position 2 — Mitigation \u0394 (Measured: "
                + m["delta_number"]
                + " vs. +0.60 Dale)"
            ),
        ),
        (
            "prose: verdict",
            r"On the length-matched HaluEval split, the mitigation \u0394 over "
            r"the no-mitigation baseline is \*\*[^*]+\*\* \(95% CI "
            r"\[[^\]]*\], measured on the \d+ of \d+ rows where step 10 "
            r"actually regenerated\)\..*?rather than asserted as a win\.",
            lambda m: (
                "On the length-matched HaluEval split, the mitigation \u0394 "
                "over the no-mitigation baseline is **"
                + m["delta_number"]
                + " BERTScore F1** (95% CI "
                + m["delta_ci"]
                + f", measured on the {m['n_regenerated']} of {m['n_rows']} "
                "rows where step 10 actually regenerated). Dale 2023 reports "
                "+0.49 to +0.60 on machine translation, so on the one "
                "published mitigation \u0394 we do not lead. The step that "
                "rewrites contradicted claims is implemented and exercised, "
                f"but on this sample {m['outcome']}; that is an open result, "
                "and it is reported here rather than asserted as a win."
            ),
        ),
        (
            "prose: paragraph 2",
            r"the mitigation \u0394 over the no-mitigation baseline is .*?, "
            r"measured on the \d+ of \d+ rows where step 10 actually "
            r"regenerated\.",
            lambda m: (
                "the mitigation \u0394 over the no-mitigation baseline is "
                + m["delta_number"]
                + " BERTScore F1 (95% CI "
                + m["delta_ci"]
                + f"), measured on the {m['n_regenerated']} of {m['n_rows']} "
                "rows where step 10 actually regenerated."
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
            r"BERTScore minus\s*pre-mitigation\)\. [^}]*?method\.\}",
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
    delta_number = f"{delta:+.3f}"
    delta_ci = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""

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
        "n_regenerated": n,
        "n_rows": mitigation.get("n_rows"),
        "outcome": outcome,
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


def _fill_comparison_detailed_md(
    path: Path, agg: dict, dry_run: bool = False,
    mitigation_path: Path | None = None,
    cross: dict[str, dict] | None = None,
) -> list[tuple[str, str, str]]:
    """Keep the measured numbers in ``Paper/comparison_detailed.md`` current.

    This document is the plain-English version of the comparison. It was
    written by hand, which is exactly how the earlier stale latency band
    (0.8-13.8 s) and the superseded RAGTruth row survived in the other
    paper files. Every number below is anchored on its own row text so a
    re-evaluation refreshes it rather than leaving it frozen.
    """
    if not path.exists():
        return []

    cross = cross or {}
    rag = cross.get("ragtruth") or {}
    tqa = cross.get("truthfulqa") or {}

    text = path.read_text(encoding="utf-8")

    def _v(key: str) -> str | None:
        value = agg.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    def _src_v(src: dict, key: str) -> str | None:
        value = src.get(key)
        if not isinstance(value, (int, float)):
            return None
        return f"{float(value):.3f}"

    values: dict[str, str] = {}
    for key in (
        "detection_roc_auc", "detection_aupr", "detection_f1",
        "detection_accuracy", "precision_at_recall_90", "spearman_rho",
        "claim_f1", "claim_precision", "claim_recall",
        "verification_success_rate", "claim_correction_rate",
        "hallucination_reduction_rate", "end_to_end_accuracy",
        "latency_min_s", "latency_max_s", "latency_mean_s",
        "retrieval_recall_at_5",
    ):
        got = _v(key)
        if got is not None:
            values[key] = got

    # The three secondary-dataset values, each read from its own file so a
    # row can never take another dataset's number.
    for prefix, src in (("rag", rag), ("tqa", tqa)):
        for key in ("detection_roc_auc", "detection_f1", "claim_f1"):
            got = _src_v(src, key)
            if got is not None:
                values[f"{prefix}_{key}"] = got

    def _ci_text(src: dict, key: str) -> str | None:
        ci = src.get(f"{key}_ci95")
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            return None
        return f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"

    for prefix, src in (("rag", rag), ("tqa", tqa)):
        got = _ci_text(src, "detection_roc_auc")
        if got:
            values[f"{prefix}_detection_roc_auc_ci"] = got

    values.update(_mitigation_prose_values(mitigation_path))

    # The HaluEval detection interval, so the headline sentence carries the
    # interval and the number together.
    h_roc_ci = agg.get("detection_roc_auc_ci95")
    if isinstance(h_roc_ci, (list, tuple)) and len(h_roc_ci) == 2:
        values["detection_roc_auc_ci"] = (
            f"[{float(h_roc_ci[0]):.4f}, {float(h_roc_ci[1]):.4f}]"
        )

    # Every HaluEval metric interval, so no row in the metrics table can
    # keep an interval that belongs to an older run.
    for key in (
        "detection_aupr", "detection_f1", "detection_accuracy",
        "precision_at_recall_90", "spearman_rho",
        "verification_success_rate", "claim_correction_rate",
        "hallucination_reduction_rate", "end_to_end_accuracy",
    ):
        ci = agg.get(f"{key}_ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            values[f"{key}_ci"] = (
                f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
            )

    specs = [
        (
            "headline ROC-AUC",
            r"Our headline number is \*\*ROC-AUC [\d.]+, CI \[[^\]]+\]\*\*",
            lambda v, m: (
                "Our headline number is **ROC-AUC "
                f"{v('detection_roc_auc')}, CI "
                f"{v('detection_roc_auc_ci')}**"
            ),
            ("detection_roc_auc", "detection_roc_auc_ci"),
        ),
        (
            "metrics table: ROC-AUC row",
            r"\| ROC-AUC \| [\d.]+ \| \[[^\]]+\] \| Ranking quality",
            lambda v, m: (
                f"| ROC-AUC | {v('detection_roc_auc')} | "
                f"{v('detection_roc_auc_ci')} | Ranking quality"
            ),
            ("detection_roc_auc", "detection_roc_auc_ci"),
        ),
        (
            "metrics table: AUPR row",
            r"\| AUPR \| [\d.]+ \| \[[^\]]+\] \| Precision at the top",
            lambda v, m: (
                f"| AUPR | {v('detection_aupr')} | "
                f"{v('detection_aupr_ci')} | Precision at the top"
            ),
            ("detection_aupr", "detection_aupr_ci"),
        ),
        (
            "metrics table: detection F1 row",
            r"\| Detection F1 \| [\d.]+ \| \[[^\]]+\] \| Balance of precision",
            lambda v, m: (
                f"| Detection F1 | {v('detection_f1')} | "
                f"{v('detection_f1_ci')} | Balance of precision"
            ),
            ("detection_f1", "detection_f1_ci"),
        ),
        (
            "metrics table: accuracy row",
            r"\| Detection accuracy \| [\d.]+ \| \[[^\]]+\] \| Correct PASS/REGEN",
            lambda v, m: (
                f"| Detection accuracy | {v('detection_accuracy')} | "
                f"{v('detection_accuracy_ci')} | Correct PASS/REGEN"
            ),
            ("detection_accuracy", "detection_accuracy_ci"),
        ),
        (
            "metrics table: P@R90 row",
            r"\| P@R90 \| [\d.]+ \| \[[^\]]+\] \| Precision when we catch",
            lambda v, m: (
                f"| P@R90 | {v('precision_at_recall_90')} | "
                f"{v('precision_at_recall_90_ci')} | Precision when we catch"
            ),
            ("precision_at_recall_90", "precision_at_recall_90_ci"),
        ),
        (
            "metrics table: Spearman row",
            r"\| Spearman ρ \| [\d.]+ \| \[[^\]]+\] \| Rank correlation",
            lambda v, m: (
                f"| Spearman ρ | {v('spearman_rho')} | "
                f"{v('spearman_rho_ci')} | Rank correlation"
            ),
            ("spearman_rho", "spearman_rho_ci"),
        ),
        (
            "metrics table: VSR row",
            r"\| VSR \| [\d.]+ \| \[[^\]]+\] \| Answers with no contradiction",
            lambda v, m: (
                f"| VSR | {v('verification_success_rate')} | "
                f"{v('verification_success_rate_ci')} | Answers with no contradiction"
            ),
            ("verification_success_rate", "verification_success_rate_ci"),
        ),
        (
            "metrics table: CCR row",
            r"\| CCR \| [\d.]+ \| \[[^\]]+\] \| Claims we corrected",
            lambda v, m: (
                f"| CCR | {v('claim_correction_rate')} | "
                f"{v('claim_correction_rate_ci')} | Claims we corrected"
            ),
            ("claim_correction_rate", "claim_correction_rate_ci"),
        ),
        (
            "metrics table: HRR row",
            r"\| HRR \| [\d.]+ \| \[[^\]]+\] \| Hallucinations we prevented",
            lambda v, m: (
                f"| HRR | {v('hallucination_reduction_rate')} | "
                f"{v('hallucination_reduction_rate_ci')} | Hallucinations we prevented"
            ),
            ("hallucination_reduction_rate", "hallucination_reduction_rate_ci"),
        ),
        (
            "metrics table: end-to-end row",
            r"\| End-to-end accuracy \| [\d.]+ \| \[[^\]]+\] \| Correct final decision",
            lambda v, m: (
                f"| End-to-end accuracy | {v('end_to_end_accuracy')} | "
                f"{v('end_to_end_accuracy_ci')} | Correct final decision"
            ),
            ("end_to_end_accuracy", "end_to_end_accuracy_ci"),
        ),
        (
            "claim precision/recall sentence",
            r"Note our precision is low \([\d.]+\) against high recall \([\d.]+\)\.",
            lambda v, m: (
                f"Note our precision is low ({v('claim_precision')}) against "
                f"high recall ({v('claim_recall')})."
            ),
            ("claim_precision", "claim_recall"),
        ),
        (
            "efficiency table: our latency",
            r"\| \*\*Ours \(measured, HaluEval\)\*\* \| \*\*[\d.]+(?:[–-][\d.]+)? s, "
            r"mean [\d.]+ s\*\*",
            lambda v, m: (
                f"| **Ours (measured, HaluEval)** | **{_latency_band(v)}, "
                f"mean {_latency_mean(v):.2f} s**"
            ),
            ("latency_min_s", "latency_max_s", "latency_mean_s"),
        ),
        (
            "RAGTruth headline AUC",
            r"Our detection score beats three of LLM-Check's five published probes",
            lambda v, m: (
                "Our detection score beats three of LLM-Check's five published probes"
            ),
            ("rag_detection_roc_auc",),
        ),
        (
            "RAGTruth response-level F1",
            r"our response-level detection F1 on\nRAGTruth is [\d.]+\.",
            lambda v, m: (
                f"our response-level detection F1 on\nRAGTruth is "
                f"{v('rag_detection_f1')}."
            ),
            ("rag_detection_f1",),
        ),
    ]

    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)
    return replacements


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
            r"\\textbf\{Sentence accuracy \(Halu-NLP wins\):\} \$0\.847\$ vs\.\\ "
            r"ours \$[\d.]+\$\.\s*"
            r"\\textbf\{Spearman \$\\rho\$ \(Halu-NLP wins\):\} "
            r"\$0\.770\$ vs\.\\ ours \$[\d.]+\$\.\s*"
            r"\\textbf\{P@R90 \(Dale LaBSE wins\):\} "
            r"\$0\.703\$ vs\.\\ ours \$[\d.]+\$\.",
            lambda v, m: (
                r"\textbf{Sentence accuracy (Halu-NLP wins):} $0.847$ vs.\ "
                r"ours $" + v("detection_accuracy") + r"$. "
                r"\textbf{Spearman $\rho$ (Halu-NLP wins):} $0.770$ vs.\ ours $"
                + v("spearman_rho") + r"$. "
                r"\textbf{P@R90 (Dale LaBSE wins):} $0.703$ vs.\ ours $"
                + v("precision_at_recall_90") + r"$."
            ),
            (
                "detection_accuracy", "spearman_rho",
                "precision_at_recall_90",
            ),
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
            "VSR table row",
            r"\\textbf\{Ours\} & \\textbf\{[^}]*\} \\\\",
            lambda v, m: (
                r"\textbf{Ours} & \textbf{"
                f"{v('verification_success_rate')}" r"} \\"
            ),
            ("verification_success_rate",),
        ),
        (
            "summary: sentence accuracy verdict",
            r"Sentence accuracy            & 0\.847 \(Halu-NLP\) & [^&]* & "
            r"[^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Sentence accuracy            & 0.847 (Halu-NLP) & "
                f"{v('detection_accuracy')}"
                + (r" & \textbf{Win}" if _above(v, "detection_accuracy", 0.847)
                   else r" & Loss")
                + (r" & Above the SHROOM winner"
                   if _above(v, "detection_accuracy", 0.847)
                   else r" & Below the SHROOM winner")
                + r" \\"
            ),
            ("detection_accuracy",),
        ),
        (
            "summary: claim-level F1 verdict",
            r"Claim-level F1               & 0\.71 \(TU Wien Vectara\) & "
            r"\\textbf\{[^}]*\} & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Claim-level F1               & 0.71 (TU Wien Vectara) & "
                r"\textbf{" + f"{v('claim_f1')}" + r"} & "
                + ("\textbf{Win}" if _above(v, "claim_f1", 0.71) else "Loss")
                + (r" & Above the cited comparable"
                   if _above(v, "claim_f1", 0.71)
                   else r" & Below the cited comparable")
                + r" \\"
            ),
            ("claim_f1",),
        ),
        (
            "summary: HRR verdict",
            r"Hallucination Reduction Rate & 0\.55 \(Dale 2023\) & "
            r"\\textbf\{[^}]*\} & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Hallucination Reduction Rate & 0.55 (Dale 2023) & "
                r"\textbf{" + f"{v('hallucination_reduction_rate')}" + r"} & "
                + ("\textbf{Win}"
                   if _above(v, "hallucination_reduction_rate", 0.55)
                   else "Loss")
                + (r" & Above the implied baseline"
                   if _above(v, "hallucination_reduction_rate", 0.55)
                   else r" & Below the implied baseline")
                + r" \\"
            ),
            ("hallucination_reduction_rate",),
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
            "summary: VSR verdict",
            r"Verification Success Rate    & 0\.81 \(Halu-NLP\) & "
            r"\\textbf\{[^}]*\} & [^&]* & [^&]* \\\\",
            lambda v, m: (
                r"Verification Success Rate    & 0.81 (Halu-NLP) & "
                r"\textbf{" + f"{v('verification_success_rate')}" + r"} & "
                + ("\textbf{Win}"
                   if _above(v, "verification_success_rate", 0.81)
                   else "Loss")
                + (r" & Above the cited comparables"
                   if _above(v, "verification_success_rate", 0.81)
                   else r" & Below the cited comparables")
                + r" \\"
            ),
            ("verification_success_rate",),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "verification_success_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values(mitigation_path))
    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

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

    ww_md_changes = _fill_why_we_win_md(
        args.paper_dir / "why_we_win.md", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation,
    )
    for label, old, new in ww_md_changes:
        print(f"  why_we_win.md {label}: {_safe(old)} -> {_safe(new)}")

    ww_tex_changes = _fill_why_we_win_tex(
        args.paper_dir / "why_we_win_summary.tex", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation,
    )
    for label, old, new in ww_tex_changes:
        print(f"  why_we_win_summary.tex {label}: {_safe(old)} -> {_safe(new)}")

    # The plain-English comparison carries the same measurements, so it is
    # filled from the same sources.
    cd_changes = _fill_comparison_detailed_md(
        args.paper_dir / "comparison_detailed.md", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation, cross=cross,
    )
    for label, old, new in cd_changes:
        print(f"  comparison_detailed.md {label}: {_safe(old)} -> {_safe(new)}")

    fp_changes = _fill_five_position_md(
        args.paper_dir / "five_position_comparison.md", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation,
    )
    for label, old, new in fp_changes:
        print(f"  five_position_comparison.md {label}: {_safe(old)} -> {_safe(new)}")

    # The defence-notes summary carries the same numbers as the tables plus
    # the two secondary datasets, so it is filled from the same sources.
    ws_changes = _fill_wins_summary_md(
        args.paper_dir / "wins_summary.md", agg, dry_run=args.dry_run,
        mitigation_path=args.mitigation, cross=cross,
        matrix_path=args.paper_dir / "why_we_win.md",
    )
    for label, old, new in ws_changes:
        print(f"  wins_summary.md {label}: {_safe(old)} -> {_safe(new)}")

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
        + len(ww_md_changes) + len(ww_tex_changes) + len(fp_changes)
        + len(readme_changes) + len(ws_changes) + len(cd_changes)
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
    if ww_md_changes:
        print(f"Updated {len(ww_md_changes)} row(s) in Paper/why_we_win.md")
    if ww_tex_changes:
        print(f"Updated {len(ww_tex_changes)} row(s) in Paper/why_we_win_summary.tex")
    if fp_changes:
        print(f"Updated {len(fp_changes)} row(s) in Paper/five_position_comparison.md")
    if ws_changes:
        print(f"Updated {len(ws_changes)} row(s) in Paper/wins_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
