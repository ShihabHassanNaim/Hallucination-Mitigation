"""Llama 3.1 generator (HF transformers).

Default backend for Steps 2 and 10. Loads the configured checkpoint on init.
"""
from __future__ import annotations


class LlamaGenerator:
    """Text generator wrapping a Llama-3.1 chat model."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        dtype: str = "fp16",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tok = None
        self._model = None

    # ----- HF path -----
    def _load_hf(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        kwargs: dict = {}
        if self.dtype == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.dtype == "4bit":
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                kwargs["torch_dtype"] = torch.float16

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tok = AutoTokenizer.from_pretrained(self.model_name)
        mdl = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if device == "cpu":
            mdl = mdl.to("cpu")
        self._tok = tok
        self._model = mdl

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tok is None:
            self._load_hf()

    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        self._ensure_loaded()
        import torch  # type: ignore

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # type: ignore[union-attr]
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)  # type: ignore[union-attr]
        cap = max_new_tokens or self.max_new_tokens
        with torch.no_grad():
            out = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=cap,
                do_sample=self.temperature > 0.0,
                temperature=max(self.temperature, 1e-3),
            )
        text = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)  # type: ignore[union-attr]
        return text.strip()


def from_config(cfg) -> LlamaGenerator:
    _load_dotenv_quiet()
    _ensure_hf_token()
    blk = cfg.step(2)
    name = (cfg.models or {}).get("generator", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    gen = LlamaGenerator(
        model_name=name,
        dtype=blk.get("dtype", "fp16"),
        device=blk.get("device", "auto"),
        max_new_tokens=int(blk.get("max_new_tokens", 256)),
        temperature=float(blk.get("temperature", 0.0)),
    )
    gen._load_hf()
    return gen


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


def _ensure_hf_token() -> None:
    """Make sure HUGGINGFACE_HUB_TOKEN is set if the user put it in .env.

    Supports common .env spellings: HUGGINGFACE_HUB_TOKEN, HF_TOKEN, HuggingFace.
    """
    import os

    if os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"):
        return
    for k, v in os.environ.items():
        if k.lower() in {"huggingface", "hf"} and v:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = v
            os.environ["HF_TOKEN"] = v
            return