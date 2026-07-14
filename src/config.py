"""Centralised configuration for CRISP.

Loads configs/default.yaml, then overlays CRISP_* environment variables and
optional --config overrides. Pydantic validation catches typos at startup
rather than mid-pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# --- pydantic models ---------------------------------------------------------

class RetrievalConfig(BaseModel):
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    top_k: int = Field(5, ge=1, le=100)
    index_path: str = "data/index"
    normalize_embeddings: bool = True
    # Phase 4 — adaptive retrieval mode. "dense" keeps the Phase 1
    # dense-only behaviour; "hybrid" fuses dense + BM25 with RRF and
    # runs the adaptive controller (expand / rewrite / multi-hop).
    mode: Literal["dense", "hybrid"] = "dense"
    # BM25 knobs (only used when mode == "hybrid").
    bm25_k1: float = Field(1.5, ge=0.0, le=5.0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)
    # Hybrid fusion knobs.
    rrf_k: float = Field(60.0, ge=1.0, le=200.0)
    weight_dense: float = Field(1.0, ge=0.0, le=5.0)
    weight_bm25: float = Field(1.0, ge=0.0, le=5.0)


class AdaptiveRetrievalConfig(BaseModel):
    """Phase 4 adaptive controller policy knobs."""
    enabled: bool = False
    initial_k: int = Field(5, ge=1, le=50)
    max_k: int = Field(20, ge=1, le=100)
    expansion_score_threshold: float = Field(0.20, ge=-1.0)
    expansion_gap_threshold: float = Field(0.05, ge=-1.0)
    rewrite_variants: int = Field(2, ge=0, le=5)
    multi_hop_max_entities: int = Field(3, ge=0, le=10)
    multi_hop_min_entity_len: int = Field(3, ge=1, le=20)


class MultiHopConfig(BaseModel):
    """Phase 5 multi-hop evidence aggregation policy.

    Only kicks in for claims tagged ``Provenance.AGGREGATED`` by Phase 3.
    The planner runs NER -> KG link -> sub-query -> merge.
    """
    enabled: bool = True
    # NER backend: "mock" (laptop-runnable, default) or "spacy".
    ner_backend: Literal["mock", "spacy"] = "mock"
    ner_model: str = "en_core_web_sm"
    # KG: max entities to consider per claim.
    max_entities: int = Field(3, ge=1, le=10)
    # KG: how many priority relations to try per entity.
    max_relations_per_entity: int = Field(2, ge=0, le=10)
    # Retriever: top_k for each sub-query.
    top_k_per_subquery: int = Field(3, ge=1, le=20)


class GeneratorConfig(BaseModel):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: Literal["fp32", "fp16", "bf16", "4bit"] = "fp16"
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    device: Literal["cpu", "cuda", "auto"] = "auto"


class PromptConfig(BaseModel):
    system: str
    user_template: str


class DetectorConfig(BaseModel):
    """Phase 2 NLI detector settings."""
    model_name: str = "microsoft/deberta-v3-large"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    max_evidence_chars: int = 4000


class ClaimExtractorConfig(BaseModel):
    """Phase 3 atomic claim extraction + provenance tagging."""
    mode: Literal["synthetic", "real"] = "synthetic"
    intrinsic_threshold: float = Field(0.7, ge=0.0, le=1.0)
    aggregated_threshold: float = Field(0.3, ge=0.0, le=1.0)


class EEDCConfig(BaseModel):
    """Phase 6 EEDC confidence scorer settings.

    Default weights give an uncalibrated but monotonic scorer. Run
    `scripts/calibrate_eedc.py` on FEVER-dev to fit better weights and
    write them back to this file.
    """
    alpha: float = -1.0
    beta: float = -0.7
    gamma: float = -0.5
    delta: float = 1.0
    weights_path: str = "data/eedc_weights.json"


class PipelineConfig(BaseModel):
    max_iterations: int = Field(1, ge=1, le=10)
    log_level: str = "INFO"
    # Phase 2: enable/disable per-claim verification + EEDC.
    enable_detection: bool = True


class AppConfig(BaseModel):
    mock: bool = True
    retrieval: RetrievalConfig = RetrievalConfig()
    adaptive_retrieval: AdaptiveRetrievalConfig = AdaptiveRetrievalConfig()
    # Phase 5 — multi-hop evidence aggregation planner.
    multi_hop: MultiHopConfig = MultiHopConfig()
    generator: GeneratorConfig = GeneratorConfig()
    detector: DetectorConfig = DetectorConfig()
    claim_extractor: ClaimExtractorConfig = ClaimExtractorConfig()
    eedc: EEDCConfig = EEDCConfig()
    prompt: PromptConfig = PromptConfig(
        system="You are a helpful assistant.",
        user_template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    )
    pipeline: PipelineConfig = PipelineConfig()


# --- loader ------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    """Map CRISP_* env vars onto nested config fields.

    Only a few common knobs are wired here; anything exotic should be edited
    in configs/default.yaml directly.
    """
    data = cfg.model_dump()

    def _set(d: dict, dotted: str, value: object) -> None:
        node = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value

    env_map = {
        "CRISP_MOCK": ("mock", bool),
        "CRISP_TOPK": ("retrieval.top_k", int),
        "CRISP_INDEX_PATH": ("retrieval.index_path", str),
        "CRISP_GENERATOR": ("generator.model_name", str),
        "CRISP_DTYPE": ("generator.dtype", str),
        "CRISP_DEVICE": ("generator.device", str),
        "CRISP_DETECTOR": ("detector.model_name", str),
        "CRISP_CLAIM_MODE": ("claim_extractor.mode", str),
        "CRISP_CLAIM_INTRINSIC": ("claim_extractor.intrinsic_threshold", float),
        "CRISP_CLAIM_AGGREGATED": ("claim_extractor.aggregated_threshold", float),
        # Phase 4 — adaptive retrieval
        "CRISP_RETRIEVAL_MODE": ("retrieval.mode", str),
        "CRISP_BM25_K1": ("retrieval.bm25_k1", float),
        "CRISP_BM25_B": ("retrieval.bm25_b", float),
        "CRISP_RRF_K": ("retrieval.rrf_k", float),
        "CRISP_W_DENSE": ("retrieval.weight_dense", float),
        "CRISP_W_BM25": ("retrieval.weight_bm25", float),
        "CRISP_ADAPTIVE": ("adaptive_retrieval.enabled", bool),
        "CRISP_ADAPTIVE_K_INIT": ("adaptive_retrieval.initial_k", int),
        "CRISP_ADAPTIVE_K_MAX": ("adaptive_retrieval.max_k", int),
        "CRISP_ADAPTIVE_SCORE": ("adaptive_retrieval.expansion_score_threshold", float),
        "CRISP_ADAPTIVE_GAP": ("adaptive_retrieval.expansion_gap_threshold", float),
        "CRISP_ADAPTIVE_REWRITES": ("adaptive_retrieval.rewrite_variants", int),
        "CRISP_ADAPTIVE_MULTIHOP": ("adaptive_retrieval.multi_hop_max_entities", int),
        # Phase 5 — multi-hop evidence aggregation
        "CRISP_DISABLE_MULTIHOP": ("multi_hop.enabled", bool),
        "CRISP_NER_BACKEND": ("multi_hop.ner_backend", str),
        "CRISP_NER_MODEL": ("multi_hop.ner_model", str),
        "CRISP_MULTIHOP_ENTS": ("multi_hop.max_entities", int),
        "CRISP_MULTIHOP_RELS": ("multi_hop.max_relations_per_entity", int),
        "CRISP_MULTIHOP_TOPK": ("multi_hop.top_k_per_subquery", int),
        "CRISP_MAX_NEW": ("generator.max_new_tokens", int),
        "CRISP_TEMP": ("generator.temperature", float),
        "CRISP_MAX_ITER": ("pipeline.max_iterations", int),
        "CRISP_LOG_LEVEL": ("pipeline.log_level", str),
        "CRISP_EEDC_WEIGHTS": ("eedc.weights_path", str),
        "CRISP_DISABLE_DETECT": ("pipeline.enable_detection", bool),
    }

    raw = os.environ
    for var, (dotted, caster) in env_map.items():
        if var not in raw:
            continue
        val = raw[var]
        if caster is bool:
            val = val.lower() in ("1", "true", "yes", "on")
            # CRISP_DISABLE_*=1 means "set the corresponding ``enabled`` to False".
            if var in {"CRISP_DISABLE_DETECT", "CRISP_DISABLE_MULTIHOP"}:
                val = not val
        else:
            val = caster(val)
        _set(data, dotted, val)

    return AppConfig(**data)


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load config from YAML, validate, then apply env overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = AppConfig(**raw)
    return _apply_env_overrides(cfg)