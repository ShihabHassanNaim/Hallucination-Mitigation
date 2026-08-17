"""Shared pytest fixtures.

The test suite uses lightweight in-memory fakes (see ``_fakes.py``) so
it can run fast and offline without downloading 6 HuggingFace
checkpoints. The production code is always real-model — there is no
mock-mode flag in production. Tests get their fast path via explicit
injection, not via a switch.
"""
from __future__ import annotations

import pytest

from tests._fakes import make_fake_pipeline


@pytest.fixture(autouse=True)
def _disable_openrouter(monkeypatch):
    # Make sure tests never accidentally hit OpenRouter.
    monkeypatch.delenv("HALLUC_USE_OPENROUTER", raising=False)
    yield


@pytest.fixture(scope="session")
def fake_pipeline():
    """A Pipeline with every model slot pre-filled with fakes."""
    return make_fake_pipeline()