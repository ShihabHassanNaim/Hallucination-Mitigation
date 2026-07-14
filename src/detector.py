"""Phase 2 — NLI-based claim verification.

A "claim" here is any atomic statement we want to check against a passage of
evidence. The detector returns one of three labels:
    SUP — the passage supports the claim
    CON — the passage contradicts the claim
    NEI — not enough information

It also returns the full probability distribution so Phase 6's EEDC scorer
can use predictive entropy as one of its confidence signals.

Real mode uses a cross-encoder NLI model (default: DeBERTa-v3-large fine-tuned
on FEVER/VitaminC/SciFact, as recommended in the research proposal). Mock
mode uses a deterministic token-overlap heuristic so the pipeline structure
is testable on a laptop without GPUs or downloads.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


# Label vocabulary — keep these constants stable; downstream code depends on them.
LABELS: Tuple[str, ...] = ("SUP", "CON", "NEI")
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


@dataclass
class NLIPrediction:
    """Result of verifying a single claim against a single evidence passage."""
    claim: str
    evidence: str
    label: str           # one of LABELS
    probs: List[float]   # len 3, ordering matches LABELS, sums to 1.0

    @property
    def sup_prob(self) -> float:
        return self.probs[LABEL2ID["SUP"]]

    @property
    def con_prob(self) -> float:
        return self.probs[LABEL2ID["CON"]]

    @property
    def nei_prob(self) -> float:
        return self.probs[LABEL2ID["NEI"]]

    @property
    def entropy(self) -> float:
        """Shannon entropy of the label distribution in nats.

        Lower entropy = the model is confident. Maximum is log(3) ~ 1.099.
        """
        entropy = 0.0
        for p in self.probs:
            if p > 0.0:
                entropy -= p * math.log(p)
        return entropy


class NLIDetector:
    """Cross-encoder NLI for fact verification."""

    def __init__(self, model_name: str = "microsoft/deberta-v3-large",
                 mock: bool = False, device: str = "auto",
                 max_evidence_chars: int = 4000):
        self.model_name = model_name
        self.mock = mock
        self.max_evidence_chars = max_evidence_chars

        self._tokenizer = None
        self._model = None
        self._device = self._resolve_device(device)

    # --- public API ---------------------------------------------------------

    def verify(self, claim: str, evidence: str) -> NLIPrediction:
        """Verify a single (claim, evidence) pair."""
        if self.mock:
            return self._mock_verify(claim, evidence)
        return self._hf_verify(claim, evidence)

    def verify_batch(self, claims: Sequence[str],
                     evidences: Sequence[str]) -> List[NLIPrediction]:
        """Verify multiple pairs. Real-mode batches them in one forward pass."""
        if len(claims) != len(evidences):
            raise ValueError("claims and evidences must have the same length.")
        if not claims:
            return []

        if self.mock:
            return [self._mock_verify(c, e) for c, e in zip(claims, evidences)]

        # Real mode: encode everything in one batched forward pass.
        return self._hf_verify_batch(list(claims), list(evidences))

    # --- device resolution --------------------------------------------------

    def _resolve_device(self, preference: str) -> str:
        if preference in ("cpu", "cuda"):
            return preference
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # --- lazy model loading -------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None or self.mock:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model: %s (device=%s)", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device).eval()

    # --- real mode ----------------------------------------------------------

    def _hf_verify(self, claim: str, evidence: str) -> NLIPrediction:
        self._ensure_loaded()
        return self._hf_verify_batch([claim], [evidence])[0]

    def _hf_verify_batch(self, claims: List[str],
                         evidences: List[str]) -> List[NLIPrediction]:
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()
        # Truncate very long evidence — most NLI checkpoints cap at 512 tokens.
        clipped = [e[: self.max_evidence_chars] for e in evidences]

        enc = self._tokenizer(  # type: ignore[union-attr]
            claims, clipped,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            logits = self._model(**enc).logits  # [n, num_labels]

        # Many NLI checkpoints (including DeBERTa-MNLI) use 3 labels in this order:
        #   0 = contradiction, 1 = neutral/NEI, 2 = entailment/SUP
        # We map them to (CON, NEI, SUP) by reordering.
        num_labels = logits.shape[-1]
        if num_labels == 3:
            # Source order: CONTRADICTION, NEUTRAL, ENTAILMENT
            # Target order: SUP, CON, NEI -> indices [2, 0, 1]
            reorder = [2, 0, 1]
            logits = logits[:, reorder]
        elif num_labels == 2:  # pragma: no cover - unusual NLI heads
            # Some binary models: NOT_ENTAILED, ENTAILED -> treat NEI=0.5
            ent = F.softmax(logits, dim=-1)
            sup = ent[:, 1]
            con = ent[:, 0] * 0.5
            nei = ent[:, 0] * 0.5
            probs = torch.stack([sup, con, nei], dim=-1)
            return [
                NLIPrediction(
                    claim=c, evidence=e,
                    label=LABELS[int(p.argmax().item())],
                    probs=p.tolist(),
                )
                for c, e, p in zip(claims, clipped, probs)
            ]
        # else: assume model already emits in (SUP, CON, NEI) order.

        probs = F.softmax(logits, dim=-1).cpu().tolist()
        return [
            NLIPrediction(
                claim=c, evidence=e,
                label=LABELS[int(p.index(max(p)))],
                probs=p,
            )
            for c, e, p in zip(claims, clipped, probs)
        ]

    # --- mock mode ----------------------------------------------------------

    @staticmethod
    def _mock_verify(claim: str, evidence: str) -> NLIPrediction:
        """Deterministic mock that mirrors the real interface.

        Strategy (NOT a real NLI model, just enough structure for tests):
          - Tokenise both strings.
          - Compute Jaccard overlap.
          - Compute a negation flag (does the evidence contain 'not' near a
            claim token?).
          - Emit probabilities in (SUP, CON, NEI) order.
        """
        c_tokens = _norm_tokens(claim)
        e_tokens = _norm_tokens(evidence)
        if not c_tokens or not e_tokens:
            probs = [0.0, 0.0, 1.0]
            label = "NEI"
            return NLIPrediction(claim=claim, evidence=evidence, label=label, probs=probs)

        overlap = _jaccard(c_tokens, e_tokens)
        negated = _has_negation_conflict(c_tokens, evidence)

        # Base confidence: high overlap + no negation -> SUP.
        if overlap > 0.5 and not negated:
            sup = 0.85
            con = 0.05
            nei = 1.0 - sup - con
            label = "SUP"
        elif overlap > 0.5 and negated:
            # Evidence contradicts claim.
            sup = 0.10
            con = 0.75
            nei = 1.0 - sup - con
            label = "CON"
        elif overlap > 0.2:
            # Some overlap, ambiguous.
            sup = 0.40
            con = 0.15
            nei = 1.0 - sup - con
            label = "SUP"  # weak entailment
        else:
            # Little overlap -> not enough info.
            sup = 0.10
            con = 0.05
            nei = 1.0 - sup - con
            label = "NEI"

        return NLIPrediction(
            claim=claim, evidence=evidence,
            label=label, probs=[sup, con, nei],
        )


# --- helpers used by the mock ------------------------------------------------

_NEGATION_TOKENS = {"not", "no", "never", "without", "isnt", "arent", "wasnt", "werent", "cant", "cannot", "doesnt", "didnt"}


def _norm_tokens(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _has_negation_conflict(claim_tokens: Sequence[str], evidence: str) -> bool:
    """Heuristic: does the evidence negate something the claim asserts?

    Example: claim = "Water boils at 50 degrees", evidence mentions "does not
    boil at 50". Crude but deterministic, which is all we need for the mock.
    """
    e_lower = evidence.lower()
    if not any(t in _NEGATION_TOKENS for t in e_lower.split()):
        return False
    # Look for numbers/quantities shared between claim and evidence
    claim_nums = {t for t in claim_tokens if any(ch.isdigit() for ch in t)}
    if not claim_nums:
        return False
    for n in claim_nums:
        if n in e_lower and any(neg in e_lower for neg in _NEGATION_TOKENS):
            return True
    return False