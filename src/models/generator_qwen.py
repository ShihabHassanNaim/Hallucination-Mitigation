"""Qwen generator (alt backend).

Thin wrapper that picks the Qwen model name from cfg.
"""
from __future__ import annotations

from .generator_llama import LlamaGenerator


class QwenGenerator(LlamaGenerator):
    """Subclass of LlamaGenerator with a Qwen checkpoint."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        dtype: str = "fp16",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        super().__init__(
            model_name=model_name,
            dtype=dtype,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


def from_config(cfg) -> QwenGenerator:
    blk = cfg.step(2)
    name = (cfg.models or {}).get("alt_generator", "Qwen/Qwen2.5-1.5B-Instruct")
    gen = QwenGenerator(
        model_name=name,
        dtype=blk.get("dtype", "fp16"),
        device=blk.get("device", "auto"),
        max_new_tokens=int(blk.get("max_new_tokens", 256)),
        temperature=float(blk.get("temperature", 0.0)),
    )
    gen._load_hf()
    return gen