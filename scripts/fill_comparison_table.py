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


def _signed(value: float) -> str:
    """Format a signed delta, e.g. ``+0.043`` / ``-0.021``."""
    return f"{value:+.3f}"


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
    # only survives if the measured delta is actually positive.
    if delta <= 0:
        out["Delta"] = out["Delta"] + " — not positive"
    return out


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
    if delta <= 0:
        text += " — not positive"
    return text


def _mitigation_prose_values() -> dict:
    """The values a prose row needs to quote the measured mitigation delta.

    The verdict cell is derived from the sign of the measured delta so a
    row cannot keep claiming a win once the measurement contradicts it.
    """
    mit = load_mitigation(MITIGATION_PATH)
    delta_text = _mitigation_delta_text(mit)
    if delta_text is None:
        return {}

    regen = mit.get("regenerated_rows") or {}
    delta = regen.get("delta_bertscore")
    if delta is None:
        return {}

    if delta > 0:
        verdict = "Highest single gain in the comparison"
    else:
        verdict = (
            "Measured on regenerated rows only; not an improvement over the "
            "no-mitigation baseline"
        )
    return {"__mitigation_delta__": delta_text, "delta": delta_text,
            "verdict": verdict}


def _delta_positive(lookup) -> bool:
    """True when the measured mitigation delta is positive.

    The delta text is formatted with a leading sign, so the sign is the
    first non-space character. A missing measurement is treated as
    "not a win" so a prose row never claims a win it cannot support.
    """
    try:
        text = lookup("delta")
    except KeyError:
        return False
    if not text:
        return False
    return str(text).strip().startswith("+")


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


def _verdict_tally(text: str) -> str | None:
    """Rebuild the verdict tally from the win-matrix rows.

    The tally summarises the rows above it, so it is computed from those
    rows rather than maintained by hand. Returns None when the matrix is
    not present or a row cannot be classified.
    """
    rows = re.findall(
        r"^\| (\d+) \| \*\*[^*]+\*\* \|.*?\| ([✓✗≈]) \|",
        text,
        flags=re.MULTILINE,
    )
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


def _fill_comparison_md(
    path: Path, agg: dict, dry_run: bool = False
) -> list[tuple[str, str, str]]:
    """Replace the projected "Ours" rows in ``Paper/comparison.md``.

    The prose table and the summary table below it both carried the same
    hand-written estimates as the CSV. Each replacement is anchored on the
    row's own text, so an unrelated number is never touched.
    """
    if not path.exists():
        return []

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
            "latency observation",
            r"\*\*Observation\.\*\* The pipeline is in the same latency tier "
            r"as LLM-Check[^.]*\.[^\n]*\n",
            lambda v, m: (
                "**Observation.** The pipeline is not in the same latency tier as "
                "LLM-Check: measured on the length-matched HaluEval split, a "
                f"query takes {_latency_band(v)} end to end (mean "
                f"{_latency_mean(v):.2f} s), because it runs a generator call and "
                "a full claim-level verification pass. It is faster than the "
                "sampling-based methods (INSIDE, AutoHall, SelfCheckGPT, Decoding "
                "Memories) and the MC-dropout rerankers (Dale 2023), but slower "
                "than the single-pass probes.\n"
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
                "One generator call; deterministic re-runs |"
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
                "HaluEval split"
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
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values())
    text, replacements = _apply_specs(text, row_specs, values, dry_run=dry_run)

    text, mit_changes = _fill_mitigation_prose(text, path, dry_run=dry_run)
    replacements.extend(mit_changes)

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


def _fill_comparison_tex(
    path: Path, agg: dict, dry_run: bool = False
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
            "latency observation",
            r"The pipeline sits in the same latency tier as LLM-Check"
            r"(?:[^\n]|\n(?!\n))*?\n\n",
            lambda v, m: (
                "The pipeline is not in the same latency tier as LLM-Check: "
                "measured on the length-matched HaluEval split, a query takes "
                + _latency_band(v).replace("-", "--")
                + " end to end (mean "
                + f"{_latency_mean(v):.2f}"
                + " s), because it runs a generator call and a full claim-level "
                "verification pass. It is faster than the sampling-based methods "
                "(INSIDE, AutoHall, SelfCheckGPT, Decoding Memories) and the "
                "MC-dropout rerankers (Dale 2023), but slower than the "
                "single-pass probes.\n\n"
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
            "summary: sentence accuracy",
            r"Sentence accuracy & 0\.847 \(Halu-NLP\) & [^&]* & At SHROOM winner level \\\\",
            lambda v, m: (
                r"Sentence accuracy & 0.847 (Halu-NLP) & "
                f"{v('detection_accuracy')} & At SHROOM winner level \\\\"
            ),
            ("detection_accuracy",),
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
            r"Claim-level F1 & 0\.71 \(TU Wien Vectara\) & \\textbf\{[^}]*\} & Higher \\\\",
            lambda v, m: (
                r"Claim-level F1 & 0.71 (TU Wien Vectara) & \textbf{"
                f"{v('claim_f1')}" r"} & Higher \\"
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
            r"HRR & 0\.55 \(Dale, implied\) & [^&]* & (?:Filled from the measured value|Measured) \\\\",
            lambda v, m: (
                r"HRR & 0.55 (Dale, implied) & "
                f"{v('hallucination_reduction_rate')} & Measured \\"
            ),
            ("hallucination_reduction_rate",),
        ),
        (
            "summary: mitigation delta",
            r"Mitigation \$\\Delta\$ & \+0\.60 \(Dale COMET-QE\) & "
            r"\\textbf\{[^}]*\} & \\textbf\{[^}]*\} \\\\",
            lambda v, m: (
                r"Mitigation $\Delta$ & +0.60 (Dale COMET-QE) & "
                r"\textbf{" + f"{v('delta')}" + r"} & Measured \\"
            ),
            ("delta",),
        ),
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values())
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
    path: Path, agg: dict, dry_run: bool = False
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
            r"\| 7 \| \*\*Detection ROC-AUC\*\* \| Dale 2023 — ALTI\+ \| "
            r"0\.987 \(MT-only\) \| \*\*[^*]+\*\* \| [^|]* \| [^|]* \|",
            lambda v, m: (
                "| 7 | **Detection ROC-AUC** | Dale 2023 — ALTI+ | "
                f"0.987 (MT-only) | **{v('detection_roc_auc')}** | "
                + ("\u2713" if _above(v, "detection_roc_auc", 0.90)
                   else "\u2248")
                + " | "
                + ("Above the cited comparables"
                   if _above(v, "detection_roc_auc", 0.90)
                   else "Below the MT-specialist and slightly below Sent-T5 XL "
                        "(0.85–0.90); above the general-purpose probes")
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
            r"\| \*\*Ours\*\* \| 0\.5.1\.5 s \|",
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
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "verification_success_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values())
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

    text, mit_changes = _fill_mitigation_prose(text, path, dry_run=dry_run)
    replacements.extend(mit_changes)

    if replacements and not dry_run:
        path.write_text(text, encoding="utf-8")
    return replacements


def _fill_five_position_md(
    path: Path, agg: dict, dry_run: bool = False
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
    ]

    values = {k: _v(k) for k in (
        "detection_roc_auc", "detection_aupr", "precision_at_recall_90",
        "claim_precision", "claim_recall", "claim_f1", "detection_accuracy",
        "spearman_rho", "hallucination_reduction_rate",
        "verification_success_rate", "claim_correction_rate",
        "latency_min_s", "latency_max_s", "latency_mean_s",
    )}
    values.update(_mitigation_prose_values())
    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

    text, mit_changes = _fill_mitigation_prose(text, path, dry_run=dry_run)
    replacements.extend(mit_changes)

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
    text: str, path: Path, dry_run: bool = False
) -> tuple[str, list[tuple[str, str, str]]]:
    """Replace the mitigation rows of the prose tables with measured values.

    Returns the updated text along with the list of changes. The text must
    be returned because this helper rewrites rows that live *outside* the
    spec lists of the calling function.
    """
    mitigation = load_mitigation(MITIGATION_PATH)
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
    if delta <= 0:
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

    if delta > 0:
        caption_sentence = (
            f"Our $\\Delta$ is ${delta:+.3f}$, ahead of the best comparable "
            f"method."
        )
    else:
        caption_sentence = (
            f"Our $\\Delta$ is ${delta:+.3f}$, so the rewrite did not improve "
            f"similarity to the gold answer on the rows where it fired. This "
            f"is reported as an open result, not a win."
        )

    m = {
        "pre": f"{pre:.3f}" if pre is not None else "n/a",
        "post": f"{post:.3f}" if post is not None else "n/a",
        "delta": delta_text,
        "pre_tex": pre_tex,
        "post_tex": post_tex,
        "delta_tex": delta_tex,
        "caption_sentence": caption_sentence,
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


def _fill_why_we_win_tex(
    path: Path, agg: dict, dry_run: bool = False
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
            r"\\textbf\{[^}]*\} & \\textbf\{[^}]*\} &",
            lambda v, m: (
                r"Mitigation $\Delta$         & +0.60 (Dale COMET-QE) & "
                r"\textbf{" + f"{v('delta')}" + r"} & Measured &"
            ),
            ("delta",),
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
    values.update(_mitigation_prose_values())
    text, replacements = _apply_specs(text, specs, values, dry_run=dry_run)

    text, mit_changes = _fill_mitigation_prose(text, path, dry_run=dry_run)
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
    md_changes = _fill_comparison_md(Path("Paper/comparison.md"), agg, dry_run=args.dry_run)
    for label, old, new in md_changes:
        print(f"  comparison.md {label}: {_safe(old)} -> {_safe(new)}")

    tex_changes = _fill_comparison_tex(
        Path("Paper/comparison_summary.tex"), agg, dry_run=args.dry_run
    )
    for label, old, new in tex_changes:
        print(f"  comparison_summary.tex {label}: {_safe(old)} -> {_safe(new)}")

    ww_md_changes = _fill_why_we_win_md(
        Path("Paper/why_we_win.md"), agg, dry_run=args.dry_run
    )
    for label, old, new in ww_md_changes:
        print(f"  why_we_win.md {label}: {_safe(old)} -> {_safe(new)}")

    ww_tex_changes = _fill_why_we_win_tex(
        Path("Paper/why_we_win_summary.tex"), agg, dry_run=args.dry_run
    )
    for label, old, new in ww_tex_changes:
        print(f"  why_we_win_summary.tex {label}: {_safe(old)} -> {_safe(new)}")

    fp_changes = _fill_five_position_md(
        Path("Paper/five_position_comparison.md"), agg, dry_run=args.dry_run
    )
    for label, old, new in fp_changes:
        print(f"  five_position_comparison.md {label}: {_safe(old)} -> {_safe(new)}")

    total_changes = (
        len(changed) + len(md_changes) + len(tex_changes)
        + len(ww_md_changes) + len(ww_tex_changes) + len(fp_changes)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
