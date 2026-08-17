"""Pipeline orchestrator.

Walks the 13 steps in order, threading a StepContext through each one.
Records per-step traces for explainability (Step 12 uses them).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .models import (
    BGE_M3Embedder,
    DebertaNLIModel,
    EEDC,
    LlamaGenerator,
    BGEReranker,
    GLiNERWrapper,
    SpacyNER,
    TemperatureScaler,
    XGBClassifier,
)
from .models.generator_openrouter import OpenRouterGenerator  # noqa: F401
from .steps import StepContext, step_registry
from .types import PipelineResult, StepTrace
from .retrieval.hybrid import HybridRetriever
from .utils.data_loader import load_synthetic_corpus
from .utils.chunker import chunk_documents


# Backward-compatible alias for the FastAPI service that expects
# ``RAGResult`` (the original CRISP name).
RAGResult = PipelineResult

log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    import os
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


@dataclass
class Pipeline:
    """The full 13-step pipeline."""

    cfg: Config | None = None
    config: Config | None = None  # alias for FastAPI compatibility
    # Injectable model slots. Leave None to auto-load from cfg.
    embedder: Any = None
    retriever: Any = None
    nli: Any = None
    generator: Any = None
    generator_qwen: Any = None
    reranker: Any = None
    gliner_ner: Any = None
    spacy_ner: Any = None
    early_classifier: Any = None
    eedc: Any = None
    calibrator: Any = None

    def __post_init__(self) -> None:
        # ``config`` is the legacy/fastapi-friendly name; cfg is the new one.
        if self.config is not None and self.cfg is None:
            self.cfg = self.config
        if self.cfg is None:
            self.cfg = load_config()
        if self.config is None:
            self.config = self.cfg

        # Pick LLM backend. Order of precedence (highest first):
        #   1. If HALLUC_USE_OPENROUTER=1        -> OpenRouter
        #   2. Else (default)                    -> HuggingFace (Llama-class)
        use_openrouter = _env_flag("HALLUC_USE_OPENROUTER", False)

        # Auto-load if not injected.
        if self.embedder is None:
            self.embedder = BGE_M3Embedder.from_config(self.cfg) if hasattr(BGE_M3Embedder, "from_config") else self._auto_embedder()
        if self.generator is None:
            if use_openrouter:
                try:
                    from .models.generator_openrouter import from_config as or_from_config
                    self.generator = or_from_config(self.cfg)
                except Exception as exc:  # noqa: BLE001
                    log.warning("OpenRouter generator unavailable: %s; falling back to local HF", exc)
                    from .models.generator_llama import from_config as llama_from_config
                    self.generator = llama_from_config(self.cfg)
            else:
                from .models.generator_llama import from_config as llama_from_config
                self.generator = llama_from_config(self.cfg)
        if self.generator_qwen is None:
            try:
                from .models.generator_qwen import from_config as qwen_from_config
                self.generator_qwen = qwen_from_config(self.cfg)
            except Exception:
                self.generator_qwen = None
        if self.nli is None:
            from .models.nli_deberta import from_config as nli_from_config
            self.nli = nli_from_config(self.cfg)
        if self.reranker is None:
            from .models.reranker_bge import from_config as reranker_from_config
            self.reranker = reranker_from_config(self.cfg)
        if self.gliner_ner is None:
            from .models.ner_gliner import from_config as gliner_from_config
            self.gliner_ner = gliner_from_config(self.cfg)
        if self.spacy_ner is None:
            from .models.ner_spacy import from_config as spacy_from_config
            self.spacy_ner = spacy_from_config(self.cfg)
        if self.early_classifier is None:
            from .models.xgb_classifier import from_config as xgb_from_config
            self.early_classifier = xgb_from_config(self.cfg)
        if self.eedc is None:
            from .models.eedc import from_config as eedc_from_config
            self.eedc = eedc_from_config(self.cfg)
        if self.calibrator is None:
            from .models.calibration import from_config as calibration_from_config
            self.calibrator = calibration_from_config(self.cfg)
        # Auto-build a retriever on the synthetic corpus if none provided.
        if self.retriever is None and self.embedder is not None:
            try:
                self.retriever = self._build_default_index()
            except Exception as exc:  # noqa: BLE001
                log.warning("default index build failed: %s", exc)
                self.retriever = None

    def _auto_embedder(self) -> BGE_M3Embedder:
        from .models.embedder_bge_m3 import from_config
        return from_config(self.cfg)

    def _build_default_index(self) -> HybridRetriever:
        corpus = load_synthetic_corpus()
        chunks = chunk_documents([d["text"] for d in corpus], chunk_size=400, chunk_overlap=50)
        doc_ids = [c[0] for c in chunks]
        docs = [c[1] for c in chunks]
        vecs = self.embedder.encode(docs)
        ret = HybridRetriever(dim=vecs.shape[1])
        ret.add(doc_ids, docs, vecs)
        return ret

    def run(self, query: str) -> PipelineResult:
        ctx = StepContext(query=query, cfg=self.cfg)
        ctx.metadata["pipeline"] = self
        traces: list[StepTrace] = []

        for cls in step_registry():
            step = cls(ctx)
            log.info("Running step %02d %s", step.number, step.name)
            started = time.time()
            try:
                step.run()
            except Exception as exc:  # noqa: BLE001
                log.exception("Step %02d failed: %s", step.number, exc)
                ctx.metadata["pipeline_error"] = f"step{step.number:02d}:{exc}"
                duration_ms = (time.time() - started) * 1000.0
                traces.append(step.trace({"error": str(exc)}, duration_ms))
                break
            duration_ms = (time.time() - started) * 1000.0
            key = f"step{step.number:02d}_{step.name}"
            payload = ctx.metadata.pop(key, {})
            traces.append(step.trace(payload, duration_ms))

        return PipelineResult(
            query=ctx.query,
            answer=ctx.final_answer or ctx.answer,
            claims=ctx.claims,
            evidence=ctx.evidence,
            confidence=ctx.confidence,
            decision=ctx.decision,
            iterations=ctx.iterations,
            step_traces=traces,
            report_markdown=ctx.metadata.get("report_markdown", ""),
        )

    # ----- FastAPI compatibility helpers --------------------------------

    def build_index(self, documents: list[str]) -> None:
        """Rebuild the in-memory index from a list of documents."""
        if self.embedder is None:
            raise RuntimeError("embedder not loaded")
        chunks = chunk_documents(list(documents), chunk_size=400, chunk_overlap=50)
        doc_ids = [c[0] for c in chunks]
        chunk_texts = [c[1] for c in chunks]
        vecs = self.embedder.encode(chunk_texts)
        ret = HybridRetriever(dim=vecs.shape[1])
        ret.add(doc_ids, chunk_texts, vecs)
        self.retriever = ret

    def load_index(self, index_dir: str) -> None:
        """Load a previously-saved index from disk (scripts/build_index.py format)."""
        import json
        import numpy as np

        root = Path(index_dir)
        doc_ids = json.loads((root / "doc_ids.json").read_text(encoding="utf-8"))
        docs = json.loads((root / "docs.json").read_text(encoding="utf-8"))
        vecs = np.load(root / "dense.npy")
        ret = HybridRetriever(dim=vecs.shape[1])
        ret.faiss.add(doc_ids, docs, vecs)
        ret.bm25.add(doc_ids, docs)
        self.retriever = ret
