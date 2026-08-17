"""Configuration loader.

Reads configs/default.yaml and layers environment overrides on top.
Every key can be overridden via HALLUC_<UPPER_DOTTED_KEY>, e.g.

    HALLUC_STEP06_RETRIEVAL_TOPK=10
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _env_overrides() -> dict[str, str]:
    return {k.removeprefix("HALLUC_"): v for k, v in os.environ.items() if k.startswith("HALLUC_")}


def _apply_env_overrides(cfg: dict[str, Any], overrides: dict[str, str]) -> dict[str, Any]:
    """Recursively apply HALLUC_A_B_C=value into cfg["a"]["b"]["c"]."""
    for key, raw in overrides.items():
        parts = key.lower().split("__")
        cursor = cfg
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                cursor = {}
        leaf = parts[-1]
        coerced: Any = raw
        if raw.lower() in {"true", "false"}:
            coerced = raw.lower() == "true"
        else:
            try:
                coerced = int(raw)
            except ValueError:
                try:
                    coerced = float(raw)
                except ValueError:
                    pass
        cursor[leaf] = coerced
    return cfg


@dataclass
class _Namespace:
    """Read-only attribute view over a dict (for ``cfg.retrieval.mode`` lookups)."""

    raw: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.raw:
            v = self.raw[name]
            if isinstance(v, dict):
                return _Namespace(raw=v)
            return v
        return None


@dataclass
class Config:
    """A flat wrapper around the loaded YAML mapping."""

    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def models(self) -> dict[str, str]:
        return self.raw.get("models", {})

    @property
    def prompt(self) -> dict[str, str]:
        return self.raw.get("prompt", {})

    @property
    def retrieval(self) -> _Namespace:
        return _Namespace(raw=self.raw.get("retrieval", {}))

    def __getattr__(self, name: str) -> Any:
        """Allow ``cfg.<name>`` for any top-level key, returning a Namespace."""
        if name.startswith("_"):
            raise AttributeError(name)
        v = self.raw.get(name)
        if isinstance(v, dict):
            return _Namespace(raw=v)
        return v

    def step(self, n: int) -> dict[str, Any]:
        """Return the config block for step_n_<name>."""
        prefix = f"step{n:02d}_"
        for k, v in self.raw.items():
            if k.startswith(prefix):
                return v  # type: ignore[return-value]
        return {}

    def as_dict(self) -> dict[str, Any]:
        return self.raw


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load default config, apply env overrides."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _apply_env_overrides(cfg, _env_overrides())
    return Config(raw=cfg)
