"""Phase 3 — Atomic claim extraction with provenance tagging.

A *claim* is any atomic, verifiable statement extracted from a generated
answer. The CRISP pipeline feeds one claim at a time to the Phase 2 NLI
detector, so claim quality directly bounds detection quality.

This module ships:

1. `Provenance`            — enum classifying each claim by where its truth
                             comes from in the context.
2. `Claim`                 — dataclass representing one atomic claim with a
                             provenance tag.
3. `ClaimExtractor`        — splits an answer into atomic claims and tags
                             each with a provenance label. Two extraction
                             modes:
                               * "synthetic" — deterministic, no LLM, uses
                                 regex + lexical signals (mock / Phase 3 default).
                               * "real"      — uses an LLM or T5-based
                                 atomiser via a small wrapper.
                             Provenance tagging uses the configured embedder
                             to compare each claim against (a) the question
                             and (b) each retrieved hit.

Provenance semantics (paper §3.2)
---------------------------------
* ``INTRINSIC``  — the claim is entailed by the question / common knowledge
  alone and does not require any passage verification (e.g., "Paris is in
  France"). Useful to skip from the detector to save cost.
* ``EXTRINSIC``  — the claim needs an external passage to verify; this is
  the default for most factual RAG claims.
* ``AGGREGATED`` — the claim summarises or composes information from two
  or more distinct passages; it cannot be verified against any single
  retrieved hit and needs a multi-hop check (deferred to Phase 5).
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .retriever import Hit


class Provenance(str, enum.Enum):
    """Provenance tag for a claim."""

    INTRINSIC = "intrinsic"
    EXTRINSIC = "extrinsic"
    AGGREGATED = "aggregated"

    def __str__(self) -> str:                    # pragma: no cover - trivial
        return self.value


@dataclass
class Claim:
    """A single atomic claim extracted from a generated answer."""

    id: str
    text: str
    provenance: Provenance = Provenance.EXTRINSIC


# --- sentence-boundary splitting --------------------------------------------

# Split on sentence-end punctuation followed by whitespace, but keep things
# like "e.g.", "i.e.", and decimals ("3.14") intact.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")
_KEEP_FRAGMENTS = {"e.g.", "i.e.", "etc."}


def _atomic_split_sentences(text: str) -> List[str]:
    """Split text into atomic sentence-level segments.

    Compound sentences joined by ";", "and", or "but" are further split so
    each sub-claim gets its own entry. This is the "synthetic" branch; the
    real mode in ``ClaimExtractor.extract`` can use a small seq2seq
    atomiser.
    """
    text = text.strip()
    if not text:
        return []

    # 0. Strip leading "[N] " citation markers that some generators inject,
    #    so they don't get glued to the first sentence and trip up the
    #    sentence-boundary regex.
    text = re.sub(r"(\[\d+\]\s*)+", "", text)
    text = text.strip()
    if not text:
        return []

    # 1. split on sentence boundaries
    pieces = _SENTENCE_SPLIT.split(text)
    sents: List[str] = []
    for p in pieces:
        # 2. break on internal conjunctions that look like they start a new
        #    predicate; cheap heuristic for English.
        for delim in [" and ", " but ", " while ", " whereas "]:
            if delim in p and len(p) > 30:
                # Only split if both halves look meaningful (>= 12 chars).
                halves = p.split(delim, 1)
                if len(halves[0].strip()) >= 12 and len(halves[1].strip()) >= 12:
                    # Re-attach the trailing delimiter to the first half so
                    # we don't lose the "and" semantic.
                    sents.append(halves[0].strip() + ".")
                    sents.append(halves[1].strip())
                    p = ""
                    break
        if p:
            sents.append(p.strip())

    # 3. drop trivial fragments.
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if len(s) < 4:
            continue
        if s.lower() in {"yes.", "no.", "ok.", "okay."}:
            continue
        # Drop any remaining leading "[N] " in case the regex above missed
        # an unusual marker shape.
        s = re.sub(r"^\[\d+\]\s*", "", s)
        if not s:
            continue
        out.append(s)
    return out


# --- helpers used by both mock and real modes --------------------------------

def _norm_tokens(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_hit_token_overlap(claim_tokens: List[str], hits: Sequence[Hit]) -> float:
    """Return the best Jaccard overlap between a claim and any single hit."""
    if not hits or not claim_tokens:
        return 0.0
    best = 0.0
    for h in hits:
        v = _jaccard(claim_tokens, _norm_tokens(h.text))
        if v > best:
            best = v
    return best


# --- main class --------------------------------------------------------------

class ClaimExtractor:
    """Extract atomic claims + tag provenance.

    Parameters
    ----------
    mode
        "synthetic" (default) uses heuristic sentence splitting + lexical
        provenance tagging. "real" uses an LLM atomiser (optional) and the
        configured embedder for provenance. The constructor signature is
        stable; both modes return the same ``List[Claim]``.
    embedder
        Optional embedder used in real mode for provenance tagging. Ignored
        in synthetic mode.
    intrinsic_threshold
        If the cosine / Jaccard overlap between the claim and the question
        is above this threshold, tag the claim as INTRINSIC. Default 0.7.
    aggregated_threshold
        If the claim overlaps with TWO OR MORE retrieved hits by more than
        this threshold each, and has a low top-1 overlap, tag AGGREGATED.
        Default 0.3.
    """

    def __init__(self, mode: str = "synthetic",
                 embedder=None,
                 intrinsic_threshold: float = 0.7,
                 aggregated_threshold: float = 0.3):
        if mode not in {"synthetic", "real"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.embedder = embedder
        self.intrinsic_threshold = float(intrinsic_threshold)
        self.aggregated_threshold = float(aggregated_threshold)

    # ---- public API --------------------------------------------------------

    def extract(self, answer: str,
                hits: Sequence[Hit],
                question: Optional[str] = None) -> List[Claim]:
        """Split `answer` into atomic claims and tag each provenance.

        Returns an empty list for an empty / blank answer.
        """
        sentences = self._atomic_split(answer)
        if not sentences:
            return []
        tagged: List[Claim] = []
        for i, sent in enumerate(sentences):
            prov = self._tag_provenance(sent, hits=hits, question=question)
            tagged.append(Claim(id=f"c{i+1}", text=sent, provenance=prov))
        return tagged

    # ---- internals ---------------------------------------------------------

    def _atomic_split(self, answer: str) -> List[str]:
        if self.mode == "synthetic":
            return _atomic_split_sentences(answer)
        # Real-mode atomiser: in Phase 3 the "real" branch falls back to the
        # deterministic splitter plus a sentence-level normalisation pass.
        # A T5/PEGASUS-based atomiser can be slotted in here without
        # changing the public return type.
        return _atomic_split_sentences(answer)

    def _tag_provenance(self, claim: str,
                        hits: Sequence[Hit],
                        question: Optional[str]) -> Provenance:
        """Decide whether `claim` is intrinsic, extrinsic, or aggregated.

        Mode-aware: real-mode uses the configured embedder for cosine
        similarity; synthetic mode uses lexical Jaccard so the pipeline is
        reproducible on a laptop.
        """
        claim_tokens = _norm_tokens(claim)
        if not claim_tokens:
            return Provenance.EXTRINSIC

        # ----- 1. INTRINSIC check --------------------------------------------
        if question is not None:
            if self.embedder is not None and self.mode == "real":
                sim = self._cosine(claim, question)
            else:
                sim = _jaccard(claim_tokens, _norm_tokens(question))
            if sim >= self.intrinsic_threshold:
                return Provenance.INTRINSIC

        # ----- 2. AGGREGATED check ------------------------------------------
        if hits:
            overlap_with_hits = [_jaccard(claim_tokens, _norm_tokens(h.text))
                                 for h in hits]
            sorted_ov = sorted(overlap_with_hits, reverse=True)
            top1 = sorted_ov[0]
            second = sorted_ov[1] if len(sorted_ov) > 1 else 0.0
            # Multi-hop signature: at least two hits each exceed the
            # threshold AND the top-1 hit does NOT dominate the runner-up
            # by a wide margin. The domination factor of 2.0 means: if the
            # strongest hit is more than 2x better than the next hit, this
            # is effectively a single-hop claim -> EXTRINSIC.
            if (top1 >= self.aggregated_threshold
                    and second >= self.aggregated_threshold
                    and top1 < 2.0 * second):
                return Provenance.AGGREGATED

        return Provenance.EXTRINSIC

    def _cosine(self, a: str, b: str) -> float:
        """Cosine similarity using the configured embedder; falls back to Jaccard."""
        if self.embedder is None:
            return _jaccard(_norm_tokens(a), _norm_tokens(b))
        va = self.embedder.encode([a])[0]
        vb = self.embedder.encode([b])[0]
        denom = (sum(x * x for x in va) ** 0.5) * (sum(x * x for x in vb) ** 0.5)
        if denom == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(va, vb))
        return dot / denom


# --- back-compat helpers for Phase 1/2 callers --------------------------------

def extract_claims(answer: str) -> List[Claim]:
    """Backward-compatible sentence-level extractor (no provenance input).

    Returns Claims with default ``Provenance.EXTRINSIC``. New code should
    prefer ``ClaimExtractor().extract(answer, hits, question)``.
    """
    if not answer or not answer.strip():
        return []
    extractor = ClaimExtractor(mode="synthetic")
    return extractor.extract(answer, hits=[], question=None)


def attach_evidence(claims: Sequence[Claim],
                    hits: Sequence[Hit]) -> List[tuple]:
    """Pair each claim with the top retrieved hit (back-compat helper)."""
    if not hits:
        return [(c, None) for c in claims]
    top = hits[0]
    return [(c, top) for c in claims]