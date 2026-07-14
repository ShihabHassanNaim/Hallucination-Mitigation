"""CRISP pipeline.

Phase 1: vanilla RAG — query → retrieve → prompt → generate.
Phase 2: per-claim NLI verification + EEDC confidence scoring.

Latency is recorded per-stage in `RAGResult.timings_ms` so you can profile
bottlenecks without code changes. In Phase 7 the Adaptive Iteration
Controller will plug into the same `iterations` field.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .claim_extractor import Claim, ClaimExtractor, Provenance, attach_evidence, extract_claims
from .config import AppConfig, load_config
from .detector import NLIPrediction
from .eedc import EEDCScorer, EEDCWeights, signals_from_prediction
from .embeddings import Embedder
from .generator import Generator
from .retriever import Hit, Retriever
from .adaptive_retriever import AdaptiveConfig, AdaptiveRetriever, RetrievalTrace
from .hybrid_retriever import HybridConfig, HybridRetriever
from .multi_hop import MultiHopPlanner, MultiHopTrace
from .ner import NER
from .kg_linker import KGLinker

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerdict:
    """Per-claim verdict produced by Phase 2."""
    claim: Claim
    evidence_text: str
    evidence_score: float
    nli: NLIPrediction
    eedc_score: float                # P(supported | signals)
    hallucinated: bool               # eedc_score < threshold
    # Phase 5 — multi-hop evidence trace for aggregated claims.
    # ``None`` for INTRINSIC / EXTRINSIC claims or when multi-hop is off.
    multi_hop_trace: Optional[dict] = None


@dataclass
class RAGResult:
    """Single-pipeline call result.

    Fields used by later phases
    ---------------------------
    - retrieved_docs (Phase 4 / 5)
    - prompt         (Phase 7 — editor rewrites only flagged spans)
    - answer         (Phase 2 / 7 — detection + regeneration target)
    - confidence     (Phase 6 — EEDC replaces the placeholder 1.0)
    - claim_verdicts (Phase 7 — drives per-claim EDIT/REGEN decisions)
    - iterations     (Phase 7 — Adaptive Iteration Controller)
    """
    query: str
    answer: str
    retrieved_docs: List[Hit] = field(default_factory=list)
    prompt: str = ""
    confidence: float = 1.0
    iterations: int = 1
    timings_ms: dict = field(default_factory=dict)
    # Phase 2 outputs
    claim_verdicts: List[ClaimVerdict] = field(default_factory=list)
    hallucination_rate: float = 0.0   # fraction of claims with EEDC < threshold
    # Phase 4 — adaptive retrieval trace. ``None`` for dense / Phase 1-3
    # runs; populated when ``adaptive_retrieval.enabled`` is true.
    retrieval_trace: Optional[dict] = None
    # Phase 5 — per-claim multi-hop traces. Populated for any claim tagged
    # ``Provenance.AGGREGATED`` when ``multi_hop.enabled`` is true. Each
    # entry is a ``MultiHopTrace.to_dict()`` payload keyed by claim id.
    multi_hop_traces: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_docs": [
                {"text": h.text, "score": h.score, "index": h.index}
                for h in self.retrieved_docs
            ],
            "prompt": self.prompt,
            "confidence": self.confidence,
            "iterations": self.iterations,
            "timings_ms": self.timings_ms,
            "claim_verdicts": [
                {
                    "claim_id": v.claim.id,
                    "claim_text": v.claim.text,
                    "provenance": str(v.claim.provenance),
                    "evidence_text": v.evidence_text,
                    "evidence_score": v.evidence_score,
                    "nli_label": v.nli.label,
                    "nli_probs": v.nli.probs,
                    "nli_entropy": v.nli.entropy,
                    "eedc_score": v.eedc_score,
                    "hallucinated": v.hallucinated,
                    "multi_hop_trace": v.multi_hop_trace,
                }
                for v in self.claim_verdicts
            ],
            "hallucination_rate": self.hallucination_rate,
            "retrieval_trace": self.retrieval_trace,
            "multi_hop_traces": list(self.multi_hop_traces),
        }


# EEDC threshold below which we label a claim as hallucinated. Tuned on
# FEVER dev in the calibration script; default of 0.5 is the natural
# decision boundary of a sigmoid.
EEDC_HALLUCINATION_THRESHOLD = 0.5


class Pipeline:
    """RAG pipeline with optional Phase 2 detection."""

    def __init__(self, config: Optional[AppConfig] = None,
                 retriever: Optional[Retriever] = None,
                 generator: Optional[Generator] = None,
                 embedder: Optional[Embedder] = None,
                 detector=None,
                 claim_extractor: Optional[ClaimExtractor] = None,
                 eedc_scorer: Optional[EEDCScorer] = None,
                 adaptive_retriever: Optional[AdaptiveRetriever] = None,
                 multi_hop_planner: Optional[MultiHopPlanner] = None):
        self.config = config or load_config()
        self.embedder = embedder or Embedder(
            model_name=self.config.retrieval.embedding_model,
            mock=self.config.mock,
        )
        self.generator = generator or Generator(
            model_name=self.config.generator.model_name,
            dtype=self.config.generator.dtype,
            max_new_tokens=self.config.generator.max_new_tokens,
            temperature=self.config.generator.temperature,
            device=self.config.generator.device,
            mock=self.config.mock,
        )
        # Phase 4 — pick dense vs hybrid at construction time. Either way
        # the Pipeline exposes ``self.retriever`` as a retriever-compatible
        # object; the adaptive wrapper (if enabled) sits on top.
        self.retriever = retriever or self._build_base_retriever()

        # Phase 3 — atomic claim extraction + provenance tagging.
        self.claim_extractor = claim_extractor or ClaimExtractor(
            mode=self.config.claim_extractor.mode,
            embedder=self.embedder if self.config.claim_extractor.mode == "real" else None,
            intrinsic_threshold=self.config.claim_extractor.intrinsic_threshold,
            aggregated_threshold=self.config.claim_extractor.aggregated_threshold,
        )

        # Phase 4 — adaptive controller wraps the base retriever.
        self.adaptive_retriever = adaptive_retriever
        if self.adaptive_retriever is None and self.config.adaptive_retrieval.enabled:
            self.adaptive_retriever = AdaptiveRetriever(
                base=self.retriever,
                config=AdaptiveConfig(
                    initial_k=self.config.adaptive_retrieval.initial_k,
                    max_k=self.config.adaptive_retrieval.max_k,
                    expansion_score_threshold=self.config.adaptive_retrieval.expansion_score_threshold,
                    expansion_gap_threshold=self.config.adaptive_retrieval.expansion_gap_threshold,
                    rewrite_variants=self.config.adaptive_retrieval.rewrite_variants,
                    multi_hop_max_entities=self.config.adaptive_retrieval.multi_hop_max_entities,
                    multi_hop_min_entity_len=self.config.adaptive_retrieval.multi_hop_min_entity_len,
                ),
            )

        # Phase 2 modules — lazy.
        self._detector = detector
        self.eedc_scorer = eedc_scorer or self._build_default_scorer()

        # Phase 5 — multi-hop planner (NER + KG link + sub-query merge).
        # Auto-built from config unless the caller injected one.
        self.multi_hop_planner = multi_hop_planner
        if self.multi_hop_planner is None and self.config.multi_hop.enabled:
            self.multi_hop_planner = self._build_default_multi_hop_planner()

    # ----- index wiring ------------------------------------------------------

    def _build_base_retriever(self):
        """Construct the dense or hybrid base retriever from config."""
        dense = Retriever(
            embedder=self.embedder,
            top_k=self.config.retrieval.top_k,
        )
        if self.config.retrieval.mode == "hybrid":
            from .bm25 import BM25Index
            bm25 = BM25Index(
                k1=self.config.retrieval.bm25_k1,
                b=self.config.retrieval.bm25_b,
            )
            return HybridRetriever(
                dense=dense,
                bm25=bm25,
                config=HybridConfig(
                    rrf_k=self.config.retrieval.rrf_k,
                    weight_dense=self.config.retrieval.weight_dense,
                    weight_bm25=self.config.retrieval.weight_bm25,
                ),
            )
        return dense

    def build_index(self, documents: List[str]) -> "Pipeline":
        logger.info("Building index over %d documents (mode=%s, mock=%s).",
                    len(documents), self.config.retrieval.mode, self.config.mock)
        if self.adaptive_retriever is not None:
            self.adaptive_retriever.build(documents)
            # AdaptiveRetriever delegates to base.build(); mirror that so
            # callers can still call ``Pipeline.retriever`` directly.
            self.retriever = self.adaptive_retriever.base
        else:
            self.retriever.build(documents)
        return self

    def save_index(self, out_dir: str) -> None:
        self.retriever.save(out_dir)

    def load_index(self, in_dir: str) -> "Pipeline":
        if isinstance(self.retriever, HybridRetriever):
            self.retriever = HybridRetriever.load(in_dir, self.retriever.dense)
            if self.adaptive_retriever is not None:
                self.adaptive_retriever.base = self.retriever
        else:
            self.retriever = Retriever.load(in_dir, self.embedder)
            if self.adaptive_retriever is not None:
                self.adaptive_retriever.base = self.retriever
        return self

    # ----- single query ------------------------------------------------------

    def run(self, query: str) -> RAGResult:
        t0 = time.perf_counter()

        # 1. retrieve (Phase 4: adaptive; otherwise pass-through)
        t = time.perf_counter()
        retrieval_trace: Optional[dict] = None
        aggregated_text: Optional[str] = None
        if self.adaptive_retriever is not None:
            # First pass: just the question. After we extract claims,
            # we'll know whether any are AGGREGATED; in that case we'd
            # re-query with that claim text. To keep Phase 4 single-pass
            # we surface a trace here; Phase 7 (AIC) can drive the loop.
            hits = self.adaptive_retriever.retrieve(query)
            retrieval_trace = (
                self.adaptive_retriever.last_trace.to_dict()
                if self.adaptive_retriever.last_trace is not None
                else None
            )
        else:
            hits = self.retriever.retrieve(query)
        t_retrieve = (time.perf_counter() - t) * 1000

        # 2. build prompt
        t = time.perf_counter()
        context = "\n\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        user_prompt = self.config.prompt.user_template.format(
            context=context or "(no context retrieved)",
            question=query,
        )
        t_prompt = (time.perf_counter() - t) * 1000

        # 3. generate
        t = time.perf_counter()
        answer = self.generator.generate(
            system_prompt=self.config.prompt.system,
            user_prompt=user_prompt,
        )
        t_generate = (time.perf_counter() - t) * 1000

        # 4. Phase 2 — claim extraction + NLI verification + EEDC scoring.
        t_detect = 0.0
        claim_verdicts: List[ClaimVerdict] = []
        confidence = 1.0
        hallucination_rate = 0.0
        # Phase 5 — default empty; populated if multi-hop runs.
        multi_hop_traces: List[dict] = []

        if self.config.pipeline.enable_detection:
            t = time.perf_counter()
            claim_verdicts, confidence, hallucination_rate = self._detect_and_score(
                answer=answer, hits=hits, question=query,
            )
            t_detect = (time.perf_counter() - t) * 1000

            # Phase 5 — for any claim tagged AGGREGATED, run the multi-hop
            # planner to gather 2-hop evidence. Attach the trace to the
            # ClaimVerdict and surface a summary in RAGResult.
            multi_hop_traces: List[dict] = []
            if self.multi_hop_planner is not None:
                retriever = self.adaptive_retriever or self.retriever
                for v in claim_verdicts:
                    if v.claim.provenance != Provenance.AGGREGATED:
                        continue
                    trace = self.multi_hop_planner.execute(
                        v.claim.text, retriever=retriever,
                        seed_evidence=hits,
                    )
                    payload = trace.to_dict()
                    payload["claim_id"] = v.claim.id
                    v.multi_hop_trace = payload
                    multi_hop_traces.append(payload)

                    # Phase 4 backward-compat: keep the legacy hook so the
                    # adaptive retriever's trace also reflects a multi-hop
                    # attempt. Useful when running with adaptive_retrieval
                    # enabled + multi_hop disabled.
                    if self.adaptive_retriever is not None:
                        self.adaptive_retriever.retrieve(
                            query=query,
                            aggregated_claim_text=v.claim.text,
                        )
                        if self.adaptive_retriever.last_trace is not None:
                            retrieval_trace = (
                                self.adaptive_retriever.last_trace.to_dict()
                            )

        t_total = (time.perf_counter() - t0) * 1000

        return RAGResult(
            query=query,
            answer=answer,
            retrieved_docs=hits,
            prompt=user_prompt,
            confidence=confidence,
            iterations=1,            # Phase 7 (AIC) will adapt this.
            claim_verdicts=claim_verdicts,
            hallucination_rate=hallucination_rate,
            retrieval_trace=retrieval_trace,
            multi_hop_traces=multi_hop_traces,
            timings_ms={
                "retrieve": round(t_retrieve, 2),
                "prompt": round(t_prompt, 2),
                "generate": round(t_generate, 2),
                "detect": round(t_detect, 2),
                "total": round(t_total, 2),
            },
        )

    # ----- Phase 2 internals -------------------------------------------------

    def _detect_and_score(self, answer: str,
                          hits: List[Hit], question: str) -> tuple:
        """Run claim extraction → NLI → EEDC; return verdicts + summary stats.

        Phase 3: claim provenance is now set by ``ClaimExtractor`` rather than
        the default EXTRINSIC fallback.
        """
        detector = self._get_detector()
        claims = self.claim_extractor.extract(answer, hits=hits, question=question)
        if not claims:
            return [], 1.0, 0.0

        pairs = attach_evidence(claims, hits)
        # Build batch inputs.
        claims_for_nli = []
        evidences_for_nli = []
        for c, h in pairs:
            claims_for_nli.append(c.text)
            evidences_for_nli.append(h.text if h is not None else "")

        # Drop empties (no evidence at all) by handling separately.
        nli_preds: List[Optional[NLIPrediction]] = [None] * len(pairs)
        non_empty_idx = [i for i, e in enumerate(evidences_for_nli) if e]
        if non_empty_idx:
            preds = detector.verify_batch(
                [claims_for_nli[i] for i in non_empty_idx],
                [evidences_for_nli[i] for i in non_empty_idx],
            )
            for slot, pred in zip(non_empty_idx, preds):
                nli_preds[slot] = pred

        # Score and aggregate.
        verdicts: List[ClaimVerdict] = []
        phi_values: List[float] = []
        for (claim, hit), pred in zip(pairs, nli_preds):
            if pred is None:
                # No evidence -> maximally uncertain.
                pred = NLIPrediction(
                    claim=claim.text, evidence="",
                    label="NEI", probs=[0.0, 0.0, 1.0],
                )
            top1 = (hit.score + 1.0) / 2.0 if hit is not None else 0.0  # [-1,1] -> [0,1]
            signals = signals_from_prediction(pred, top1)
            phi = self.eedc_scorer.score(signals)
            phi_values.append(phi)
            verdicts.append(ClaimVerdict(
                claim=claim,
                evidence_text=pred.evidence,
                evidence_score=top1,
                nli=pred,
                eedc_score=phi,
                hallucinated=phi < EEDC_HALLUCINATION_THRESHOLD,
            ))

        confidence = sum(phi_values) / len(phi_values) if phi_values else 1.0
        hallucinated_count = sum(1 for v in verdicts if v.hallucinated)
        hallucination_rate = hallucinated_count / len(verdicts) if verdicts else 0.0
        return verdicts, confidence, hallucination_rate

    def _get_detector(self):
        if self._detector is not None:
            return self._detector
        # Lazy import so this module doesn't pull in transformers unless used.
        from .detector import NLIDetector
        self._detector = NLIDetector(
            model_name=self.config.detector.model_name,
            mock=self.config.mock,
            device=self.config.detector.device,
            max_evidence_chars=self.config.detector.max_evidence_chars,
        )
        return self._detector

    def _build_default_scorer(self) -> EEDCScorer:
        """Try to load calibrated weights from disk; fall back to defaults."""
        weights_path = Path(self.config.eedc.weights_path)
        if weights_path.exists():
            try:
                payload = json.loads(weights_path.read_text(encoding="utf-8"))
                w = EEDCWeights(**payload)
                logger.info("Loaded EEDC weights from %s", weights_path)
                return EEDCScorer(weights=w)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to load EEDC weights (%s); using defaults.", e)
        return EEDCScorer(
            weights=EEDCWeights(
                alpha=self.config.eedc.alpha,
                beta=self.config.eedc.beta,
                gamma=self.config.eedc.gamma,
                delta=self.config.eedc.delta,
            )
        )

    def _build_default_multi_hop_planner(self) -> MultiHopPlanner:
        """Construct the NER + KG + planner pipeline from config."""
        cfg = self.config.multi_hop
        ner = NER(
            backend=cfg.ner_backend,
            model_name=cfg.ner_model,
            mock=self.config.mock or cfg.ner_backend != "spacy",
        )
        linker = KGLinker()
        return MultiHopPlanner(
            ner=ner,
            linker=linker,
            max_entities=cfg.max_entities,
            max_relations_per_entity=cfg.max_relations_per_entity,
            top_k_per_subquery=cfg.top_k_per_subquery,
        )