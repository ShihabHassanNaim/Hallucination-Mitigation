"""Phase 7 — Evidence-guided answer editor.

What ``EvidenceGuidedEditor`` does
----------------------------------
After Phase 2 has flagged low-EEDC (``hallucinated=True``) claims,
naively regenerating the whole answer is wasteful: most of it is fine and
the LLM may forget good parts while fixing the bad ones. The editor
rewrites *only the spans of the answer that contain flagged claims* and
leaves the rest untouched.

Three modes are supported:

  * ``stub``       (default, mock-friendly): replace flagged spans with a
                    bracket ``[unsupported: <claim text>]`` token. No LLM
                    call. Useful for tests and offline eval.
  * ``evidence``   : replace flagged spans with the best-matching evidence
                    sentence from the top-k hits (sentence-level extractive
                    fallback). Also runs without an LLM.
  * ``regenerate`` : delegate to a :class:`Generator` to rewrite the spans
                    given the supporting evidence (real LLM call). When
                    ``Generator.mock=True`` this degrades to ``evidence``
                    behaviour so tests stay deterministic.

A :class:`EditResult` is returned with the new answer, a list of edits
(``before``/``after``/``claim_id``/``reason``), and a counter of how many
spans were rewritten.
"""
from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .claim_extractor import Claim
from .generator import Generator
from .retriever import Hit

logger = logging.getLogger(__name__)


class EditorMode(str, enum.Enum):
    """How aggressively to rewrite flagged spans."""

    STUB = "stub"            # placeholder only
    EVIDENCE = "evidence"    # replace with best evidence sentence
    REGENERATE = "regenerate"  # delegate to Generator

    def __str__(self) -> str:                # pragma: no cover
        return self.value


# ---------------------------------------------------------------------------
# Span discovery
# ---------------------------------------------------------------------------


def _split_into_sentences(text: str) -> List[str]:
    """Tiny sentence splitter used to locate flagged spans.

    Heavier splitting is delegated to ``ClaimExtractor._atomic_split_sentences``
    in real mode; this helper is intentionally cheap so the editor stays
    mock-friendly and runs without any model.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _span_for_claim(answer: str, claim: Claim) -> Tuple[int, int]:
    """Locate ``claim.text`` (or a prefix of it) inside ``answer``.

    Falls back to character-match against the first 6 tokens of the claim
    when an exact substring isn't found (handles paraphrasing). Returns
    ``(-1, -1)`` if no span can be located.
    """
    if not claim.text or not answer:
        return -1, -1
    needle = claim.text.strip()
    idx = answer.find(needle)
    if idx >= 0:
        return idx, idx + len(needle)

    # Try a ~6-token prefix.
    tokens = needle.split()
    if len(tokens) >= 4:
        prefix = " ".join(tokens[:6])
        idx = answer.lower().find(prefix.lower())
        if idx >= 0:
            return idx, idx + len(prefix)

    # Last-resort: try a 4-token prefix.
    if len(tokens) >= 3:
        prefix = " ".join(tokens[:4])
        idx = answer.lower().find(prefix.lower())
        if idx >= 0:
            return idx, idx + len(prefix)

    return -1, -1


# ---------------------------------------------------------------------------
# Evidence sentence selection
# ---------------------------------------------------------------------------


def _split_evidence_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _best_evidence_sentence(claim: Claim, hits: Sequence[Hit]) -> str:
    """Return the single evidence sentence with the highest token overlap.

    Cheap and deterministic so it works in mock mode. Real deployments can
    swap in a cross-encoder reranker.
    """
    if not hits:
        return "I don't have enough evidence to support this claim."
    claim_tokens = set(re.findall(r"\w+", claim.text.lower()))
    if not claim_tokens:
        return hits[0].text

    best_score = -1
    best_sentence = hits[0].text
    for hit in hits:
        for sent in _split_evidence_sentences(hit.text):
            sent_tokens = set(re.findall(r"\w+", sent.lower()))
            if not sent_tokens:
                continue
            overlap = len(claim_tokens & sent_tokens)
            # Normalise by the smaller side so short sentences aren't penalised.
            score = overlap / max(1, min(len(claim_tokens), len(sent_tokens)))
            if score > best_score:
                best_score = score
                best_sentence = sent
    return best_sentence


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


@dataclass
class EditRecord:
    """One span rewrite performed by the editor."""

    claim_id: str
    claim_text: str
    span_start: int
    span_end: int
    original: str
    replacement: str
    reason: str             # "hallucinated" | "low_confidence" | "no_evidence"

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "span": [self.span_start, self.span_end],
            "original": self.original,
            "replacement": self.replacement,
            "reason": self.reason,
        }


@dataclass
class EditResult:
    """What :meth:`EvidenceGuidedEditor.edit` returns."""

    original_answer: str
    edited_answer: str
    edits: List[EditRecord] = field(default_factory=list)
    mode: str = EditorMode.STUB.value

    @property
    def num_edits(self) -> int:
        return len(self.edits)

    @property
    def changed(self) -> bool:
        return self.num_edits > 0

    def to_dict(self) -> dict:
        return {
            "original_answer": self.original_answer,
            "edited_answer": self.edited_answer,
            "mode": self.mode,
            "num_edits": self.num_edits,
            "edits": [e.to_dict() for e in self.edits],
        }


class EvidenceGuidedEditor:
    """Rewrite only the spans of an answer that contain flagged claims.

    Parameters
    ----------
    mode : EditorMode
        How to rewrite. ``stub`` is the safe default.
    generator : Generator | None
        Required only when ``mode == REGENERATE``. In mock mode the editor
        automatically degrades to ``evidence`` behaviour so the pipeline
        stays testable without an LLM.
    """

    def __init__(self,
                 mode: EditorMode | str = EditorMode.STUB,
                 generator: Optional[Generator] = None):
        self.mode = EditorMode(mode) if not isinstance(mode, EditorMode) else mode
        self.generator = generator

    # ----- public API --------------------------------------------------------

    def edit(self, answer: str,
             flagged_claims: Iterable[tuple],
             hits: Sequence[Hit]) -> EditResult:
        """Rewrite flagged spans.

        ``flagged_claims`` is an iterable of ``(Claim, ClaimVerdict)`` tuples
        (or anything with ``.claim``/``reason`` attributes — we only need
        the claim text + a reason string for the audit log). Claims whose
        span cannot be located are skipped with a logger.debug line.
        """
        flagged = list(flagged_claims)
        if not flagged or not answer:
            return EditResult(original_answer=answer, edited_answer=answer,
                              edits=[], mode=self.mode.value)

        # Process edits in *reverse* order so character offsets stay valid.
        edits: List[EditRecord] = []
        edited = answer
        for verdict, _span_start, _span_end in reversed(
            [(self._unwrap(v), *self._resolve_span(edited, self._unwrap(v)[0])) for v in flagged]
        ):
            claim, reason = verdict
            start, end = _span_start, _span_end
            if start < 0 or end < 0 or start >= end:
                logger.debug("Could not locate span for claim %s; skipping.", claim.id)
                continue
            original = edited[start:end]
            replacement = self._replacement_for(claim, hits, reason)
            edits.append(EditRecord(
                claim_id=claim.id,
                claim_text=claim.text,
                span_start=start,
                span_end=end,
                original=original,
                replacement=replacement,
                reason=reason,
            ))
            edited = edited[:start] + replacement + edited[end:]

        edits.reverse()  # restore original order for readability
        return EditResult(
            original_answer=answer,
            edited_answer=edited,
            edits=edits,
            mode=self.mode.value,
        )

    # ----- helpers -----------------------------------------------------------

    @staticmethod
    def _unwrap(verdict) -> Tuple[Claim, str]:
        """Accept either (Claim, reason_str) or ClaimVerdict with .claim."""
        if isinstance(verdict, tuple) and len(verdict) == 2:
            claim, reason = verdict
            return claim, str(reason)
        # Has .claim and a hallucinated flag -> derive reason.
        claim = getattr(verdict, "claim", verdict)
        reason = "hallucinated" if getattr(verdict, "hallucinated", False) else "low_confidence"
        return claim, reason

    @staticmethod
    def _resolve_span(answer: str, claim: Claim) -> Tuple[int, int]:
        return _span_for_claim(answer, claim)

    def _replacement_for(self, claim: Claim, hits: Sequence[Hit],
                         reason: str) -> str:
        if self.mode == EditorMode.STUB:
            return f"[unsupported: {claim.text}]"
        if self.mode == EditorMode.EVIDENCE:
            return _best_evidence_sentence(claim, hits)
        # REGENERATE — use the LLM if available; degrade otherwise.
        if self.generator is None or self.generator.mock:
            logger.debug("REGENERATE mode but no real generator available; "
                         "falling back to evidence sentence.")
            return _best_evidence_sentence(claim, hits)
        # Real LLM call: ask for a faithful rewrite of the claim using the
        # strongest evidence sentence as context.
        evidence = _best_evidence_sentence(claim, hits)
        system = ("You are a precise editor. Rewrite the following claim so it "
                  "is fully supported by the given evidence sentence. Keep the "
                  "rewriting short and faithful. If the evidence doesn't "
                  "support the claim, answer 'I don't know.'")
        user = f"Evidence: {evidence}\nClaim: {claim.text}\nRewrite:"
        try:
            return self.generator.generate(system_prompt=system, user_prompt=user)
        except Exception as e:  # pragma: no cover - LLM failure path
            logger.warning("Generator failed (%s); falling back to evidence.", e)
            return _best_evidence_sentence(claim, hits)