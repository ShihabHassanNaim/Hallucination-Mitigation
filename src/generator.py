"""Text generator.

Real mode wraps HuggingFace transformers (chat-completion style). Mock mode
returns a deterministic extractive answer so the pipeline can be tested
without GPUs or model downloads.

Phase 7 (evidence-guided regeneration) will subclass or compose this with
an editor that rewrites only flagged spans.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Generator:
    """Generate an answer given a system prompt + user prompt."""

    def __init__(self, model_name: str, dtype: str = "fp16",
                 max_new_tokens: int = 256, temperature: float = 0.0,
                 device: str = "auto", mock: bool = False):
        self.model_name = model_name
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_preference = device
        self.mock = mock

        self._tokenizer = None
        self._model = None

    # ----- public API ---------------------------------------------------------

    def generate(self, system_prompt: str, user_prompt: str,
                 stop: Optional[List[str]] = None) -> str:
        if self.mock:
            return self._mock_generate(system_prompt, user_prompt)

        self._ensure_loaded()
        return self._hf_generate(system_prompt, user_prompt, stop)

    # ----- lazy real-mode loading --------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()

        kwargs = {}
        if self.dtype == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.dtype == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "4bit requested but bitsandbytes isn't installed. "
                    "Install bitsandbytes or set generator.dtype=fp16."
                ) from e

        logger.info("Loading generator: %s (dtype=%s, device=%s)",
                    self.model_name, self.dtype, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if device == "cpu":
            self._model.to("cpu")
        # If CUDA, .to("cuda") is implicit via device_map='auto' on bnb; else explicit:
        elif device == "cuda" and self.dtype != "4bit":
            self._model.to("cuda")
        self._model.eval()

    def _resolve_device(self) -> str:
        if self.device_preference in ("cpu", "cuda"):
            return self.device_preference
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ----- real-mode generation ----------------------------------------------

    def _hf_generate(self, system_prompt: str, user_prompt: str,
                     stop: Optional[List[str]]) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prefer the tokenizer's chat template if available (Llama-3.1, Mistral, etc.)
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_ids = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:  # pragma: no cover - extremely old tokenizers
            prompt = f"{system_prompt}\n\n{user_prompt}"
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(self._model.device)
        do_sample = self.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text.strip()

    # ----- mock generation ---------------------------------------------------

    @staticmethod
    def _mock_generate(system_prompt: str, user_prompt: str) -> str:
        """Deterministic extractive answer built from the first context sentence.

        Splits the user prompt on 'Context:' and 'Question:' to find the
        context block, then returns its first sentence. Keeps the pipeline
        testable on a laptop without GPUs.
        """
        try:
            after_context = user_prompt.split("Context:", 1)[1]
            context_block, _, rest = after_context.partition("Question:")
            sentences = [s.strip() for s in context_block.replace("\n", " ").split(".") if s.strip()]
            if sentences:
                answer = sentences[0] + "."
            else:
                answer = "I don't know."
        except Exception:
            answer = "I don't know."
        # Echo back the question word for traceability in tests.
        question_words = rest.strip().split()
        if question_words:
            answer = f"{answer} [q='{question_words[0]}']"
        return answer