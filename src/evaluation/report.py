"""Read-only evaluation report loader.

Loads ``data/metrics.json`` (produced by ``scripts/evaluate.py`` and
optionally enriched by ``scripts/evaluate_full.py``) and returns a
JSON-serialisable report that the frontend can render directly.

The module is dependency-free (stdlib only) so it can be imported from
the Flask app, the FastAPI app, or tests without dragging in torch /
transformers.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
#
# These mirror the numeric columns produced by ``scripts/evaluate.py``
# (BASE) and the scalar keys emitted by ``scripts/evaluate_full.py``
# (FULL). They are used purely to know which keys to aggregate / display.

BASE_METRICS: tuple[str, ...] = (
    "claim_precision",
    "claim_recall",
    "claim_f1",
    "retrieval_recall_at_5",
    "retrieval_precision_at_5",
    "retrieval_mrr",
    "retrieval_ndcg",
    "verification_success_rate",
    "end_to_end_accuracy",
    "confidence",
)

FULL_METRICS: tuple[str, ...] = (
    "detection_accuracy",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "detection_roc_auc",
    "hallucination_reduction_rate",
    "claim_correction_rate",
    "n_queries",
)

# Repo-root-relative default. Resolved at call time so tests can
# monkeypatch chdir() before invoking load_report().
_DEFAULT_METRICS_PATH = Path("data/metrics.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_metrics_path(explicit: Optional[Path]) -> Path:
    """Return the absolute path to metrics.json.

    Falls back to the cwd-relative path; the caller is responsible for
    chdir-ing to the repo root (Flask does this implicitly, tests use
    ``monkeypatch.chdir``).
    """
    if explicit is not None:
        return Path(explicit)
    return _DEFAULT_METRICS_PATH


def _is_number(v: Any) -> bool:
    """True iff v is a real number (bool is excluded — too surprising)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _detect_source(existing_aggregate: Dict[str, Any]) -> str:
    """Guess which script produced the file based on the aggregate keys.

    Returns one of ``"evaluate_full"``, ``"evaluate"``, ``"unknown"``.
    NOTE: an empty/{} aggregate is reported as ``"unknown"`` — the
    caller may have aggregated the rows on the fly themselves.
    """
    if not existing_aggregate:
        return "unknown"
    if any(k.startswith("detection_") for k in existing_aggregate):
        return "evaluate_full"
    if any(k.startswith("mean_") for k in existing_aggregate):
        return "evaluate"
    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean / stdev for every numeric column in BASE_METRICS.

    Defensive by design:
      * None / non-numeric values are skipped per column.
      * ``stdev_*`` keys are only emitted when at least 2 numeric values
        exist for that column (otherwise the stdev is undefined).
      * Columns with zero numeric values are silently omitted.

    Returns a flat dict like ``{"mean_confidence": 0.7, "stdev_confidence": 0.1, ...}``.
    """
    out: Dict[str, float] = {}
    if not rows:
        return out

    for key in BASE_METRICS:
        values = [r[key] for r in rows if key in r and _is_number(r[key])]
        if not values:
            continue
        out[f"mean_{key}"] = float(statistics.fmean(values))
        if len(values) >= 2:
            try:
                out[f"stdev_{key}"] = float(statistics.stdev(values))
            except statistics.StatisticsError:  # pragma: no cover
                pass

    return out


def load_report(metrics_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the evaluation report from disk.

    Returns a dict with the schema ::

        {
            "aggregate":     dict[str, float],
            "per_query":     list[dict],
            "n_queries":     int,
            "source_file":   str,
            "last_modified": str | None,   # ISO-8601, or None if missing
            "source":        "evaluate_full" | "evaluate" | "unknown" | "missing",
        }

    Behaviour:
      * Missing file → returns empty fields with ``source="missing"``.
      * Malformed JSON → returns empty fields with ``source="unknown"``.
      * Always starts with on-the-fly aggregates computed from ``rows``.
      * Any pre-existing ``aggregate`` block from ``evaluate_full.py``
        is merged in, with **precomputed values taking precedence**
        over the on-the-fly ones (so the user can re-run the heavier
        detection-ROC eval without losing the cheaper retrieval stats).
    """
    path = _resolve_metrics_path(metrics_path)

    # Missing file path.
    if not path.exists():
        return {
            "aggregate": {},
            "per_query": [],
            "n_queries": 0,
            "source_file": str(path),
            "last_modified": None,
            "source": "missing",
        }

    # Load JSON.
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — never crash the UI on bad JSON
        return {
            "aggregate": {},
            "per_query": [],
            "n_queries": 0,
            "source_file": str(path),
            "last_modified": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "source": "unknown",
        }

    # Normalise shape.
    if not isinstance(raw, dict):
        raw = {"rows": [], "aggregate": {}}
    rows = raw.get("rows") or []
    existing_agg = raw.get("aggregate") or {}
    if not isinstance(rows, list):
        rows = []
    if not isinstance(existing_agg, dict):
        existing_agg = {}

    # Compute aggregate on the fly from the rows, then layer in any
    # precomputed block (precomputed wins).
    aggregate = compute_aggregate(rows)
    for k, v in existing_agg.items():
        if _is_number(v) or k == "n_queries":
            aggregate[k] = v

    # Last-modified timestamp.
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:  # pragma: no cover
        mtime = None

    return {
        "aggregate": aggregate,
        "per_query": rows,
        "n_queries": len(rows),
        "source_file": str(path),
        "last_modified": mtime,
        "source": _detect_source(existing_agg),
    }


__all__ = ["BASE_METRICS", "FULL_METRICS", "compute_aggregate", "load_report"]
