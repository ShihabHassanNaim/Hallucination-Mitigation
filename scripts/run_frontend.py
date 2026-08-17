"""Launch the Flask front-end.

This is a thin wrapper around ``frontend.app`` that:

  1. Reads the HF token out of ``.env`` if present (so the user does
     not have to set ``HUGGINGFACE_HUB_TOKEN`` manually).
  2. Starts the Flask dev server on port 5000.

Every component loads its real HuggingFace checkpoint. Loading failures
are surfaced as errors.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_hf_token_from_env() -> None:
    """Best-effort: read HF token from .env if not already set in env."""
    if os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"):
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key.lower() in {"huggingface", "hf_token", "huggingface_hub_token"} and val:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = val
                os.environ["HF_TOKEN"] = val
                return
    except Exception:  # noqa: BLE001 — never crash the launcher
        return


def main() -> None:
    _load_hf_token_from_env()
    # Make sure src/ is importable when this script is run directly.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    port = int(os.environ.get("PORT", "5000"))
    debug = bool(int(os.environ.get("DEBUG", "0")))
    from frontend.app import app  # noqa: WPS433 — late import after path bootstrap
    print(f"[run_frontend] starting Flask on http://127.0.0.1:{port}/")
    print(f"[run_frontend] HF token loaded: {bool(os.environ.get('HUGGINGFACE_HUB_TOKEN'))}")
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()