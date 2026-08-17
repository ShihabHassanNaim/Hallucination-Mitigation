"""Runtime model registry.

Provides a /models/status view of the 6 first-class components of the
hallucination pipeline. Each component is probed by:

1. Checking whether its required Python module is importable.
2. Reading the configured model name out of the project config.

The result is a JSON-serialisable dict shaped like::

    {
        "mode": "live",
        "summary": "6 of 6 components active",
        "components": [
            {
                "name": "Generator",
                "category": "generation",
                "backend": "hf",
                "configured_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "active": True,
                "reason": "loaded",
            },
            ...
        ],
    }

The module is intentionally dependency-free (no transformers/torch needed
to *introspect* — only to actually run). It is safe to import from tests
and from the FastAPI app at startup.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------
#
# Each component knows:
#   * ``name``         — the human label shown in the UI.
#   * ``category``     — coarse bucket (generation / embedding / ..).
#   * ``backend``      — short tag for the UI ("hf", "sklearn").
#   * ``module``       — the Python module that must be importable for the
#                        real backend to be available.
#   * ``config_keys``  — tuple of config keys (under ``models.``) we try
#                        in order to surface the configured model name.
#
# Order in this list == order in the UI table.

_COMPONENTS: List[Dict[str, Any]] = [
    {
        "name": "Generator",
        "category": "generation",
        "backend": "hf",
        "module": "transformers",
        "config_keys": ("generator", "alt_generator"),
    },
    {
        "name": "Embedder",
        "category": "embedding",
        "backend": "hf",
        "module": "sentence_transformers",
        "config_keys": ("embedder",),
    },
    {
        "name": "Reranker",
        "category": "reranking",
        "backend": "hf",
        "module": "sentence_transformers",
        "config_keys": ("reranker",),
    },
    {
        "name": "NLI",
        "category": "nli",
        "backend": "hf",
        "module": "transformers",
        "config_keys": ("nli",),
    },
    {
        "name": "NER",
        "category": "ner",
        "backend": "gliner",
        "module": "gliner",
        "config_keys": ("gliner", "spacy"),
    },
    {
        "name": "Calibrator",
        "category": "calibration",
        "backend": "sklearn",
        "module": "sklearn",
        "config_keys": ("classifier",),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_available(mod_name: str) -> bool:
    """Return True iff ``mod_name`` can be imported."""
    if not mod_name:
        return True
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:  # pragma: no cover — best-effort probe
        return False


def _resolve_configured_model(spec: Dict[str, Any]) -> str:
    """Pick the first configured model name from the active config.

    Falls back to ``"unknown"`` so the field is always present in the
    JSON response. We tolerate a missing config file (e.g. running tests
    without a YAML on disk) by returning ``unknown``.
    """
    try:
        from src.config import load_config  # noqa: WPS433 — lazy import

        cfg = load_config()
    except Exception:  # pragma: no cover — config is optional here
        return "unknown"

    models = {}
    try:
        models = cfg.models or {}
    except Exception:  # pragma: no cover
        models = {}

    for key in spec["config_keys"]:
        val = models.get(key)
        if val:
            return str(val)
    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _probe_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Probe a single component. Returns a JSON-ready dict."""
    if spec["module"] and not _module_available(spec["module"]):
        return {
            "name": spec["name"],
            "category": spec["category"],
            "backend": spec["backend"],
            "configured_model": _resolve_configured_model(spec),
            "active": False,
            "reason": f"module '{spec['module']}' not installed",
        }

    return {
        "name": spec["name"],
        "category": spec["category"],
        "backend": spec["backend"],
        "configured_model": _resolve_configured_model(spec),
        "active": True,
        "reason": "loaded",
    }


def build_status_report() -> Dict[str, Any]:
    """Build the full status report.

    Returns a JSON-serialisable dict with the shape documented at the
    top of this module. Used by both the FastAPI endpoint and tests.
    """
    components = [_probe_one(spec) for spec in _COMPONENTS]

    active = sum(1 for c in components if c["active"])
    total = len(components)
    inactive = total - active

    if inactive == 0:
        mode = "live"
        summary = f"{active} of {total} components active"
    elif inactive == total:
        mode = "degraded"
        summary = f"0 of {total} components active"
    else:
        mode = "degraded"
        summary = f"{inactive} of {total} components unavailable"

    return {
        "mode": mode,
        "summary": summary,
        "components": components,
    }


__all__ = ["build_status_report"]