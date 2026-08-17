"""OpenRouter-backed generator.

Uses the OpenAI-compatible API at https://openrouter.ai/api/v1. The
default model is one of OpenRouter's free offerings — set
``models.openrouter`` in configs/default.yaml to switch.

Reads the API key from env var ``OPENROUTER_API_KEY`` (loaded from
``../.env`` via python-dotenv). If the key is missing or the API call
fails, ``generate()`` raises so the caller can decide how to handle it.
"""
from __future__ import annotations

import logging

log = logging.getLogger("openrouter")

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"


class OpenRouterGenerator:
    """LLM generator backed by OpenRouter."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    # ----- Network path -----
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        # Imported lazily so the test suite doesn't require openai installed.
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        self._ensure_client()
        cap = max_new_tokens or self.max_new_tokens
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=cap,
            temperature=float(self.temperature or 0.0),
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("OpenRouter returned empty content")
        return content

    def probe(self) -> bool:
        """Quick connectivity check. Returns True if the API responded."""
        try:
            self._ensure_client()
            resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model_name,
                messages=[{"role": "user", "content": "Reply with the single word READY."}],
                max_tokens=8,
                temperature=0.0,
            )
            return bool((resp.choices[0].message.content or "").strip())
        except Exception as exc:  # noqa: BLE001
            log.warning("openrouter_probe failed: %s", exc)
            return False


def _load_dotenv_quiet() -> None:
    """Best-effort load of .env into os.environ. Silently skipped on failure."""
    try:
        from dotenv import load_dotenv  # type: ignore
        from pathlib import Path

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except Exception:
        pass


def from_config(cfg) -> OpenRouterGenerator:
    import os

    _load_dotenv_quiet()
    models = cfg.models or {}
    openrouter_model = (
        models.get("openrouter")
        or models.get("generator")
        or DEFAULT_OPENROUTER_MODEL
    )
    blk = cfg.step(2) or {}
    key = os.environ.get("OPENROUTER_API_KEY") or None
    return OpenRouterGenerator(
        model_name=openrouter_model,
        api_key=key,
        max_new_tokens=int(blk.get("max_new_tokens", 256)),
        temperature=float(blk.get("temperature", 0.0)),
    )