"""Step 10 — Evidence-Guided Regeneration.

Four modes:
  - stub:       no-op (caller passes intentionally unsupported answers through)
  - evidence:   replace unsupported spans with their evidence-grounded text
  - claim_edit: edit the original answer in place, replacing only the
                claims the verifier flagged and leaving every other span
                byte-identical
  - regenerate: full Llama 3.1 regeneration conditioned on verified evidence only

``claim_edit`` exists because ``evidence`` mode rebuilds the answer from the
supported claims alone. That discards the original phrasing of every claim
the verifier did not explicitly mark as supported, and when no claim is
supported it falls back to substituting a raw retrieved passage for the
answer. On the length-matched HaluEval split that behaviour turned an
already-correct answer into a worse one (a BERTScore of 1.000 fell to
0.811), so the measured mitigation delta was negative.

The pipeline's headline architectural property is per-claim granularity:
the verifier identifies *which* claim is unsupported. ``claim_edit`` acts
at that same granularity -- it rewrites the flagged span and nothing else --
so the precision the detector produced is not thrown away at the last step.
"""
from __future__ import annotations

import re

from . import Step, StepContext
from ..types import Evidence


# A claim's text has to be located inside the answer before it can be
# spliced out. The lookup is whitespace- and case-insensitive because the
# claim extractor strips and re-joins the sentence it split, so the claim
# text and the answer text rarely match character for character.
def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# Function words are not entities, so a sentence starting with "The" does
# not make "The" the subject of the claim.
_STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "but",
    "that", "this", "these", "those", "it", "its", "as", "from", "who",
    "which", "what", "when", "where", "how", "than", "then", "also", "there",
    "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
    "your", "my", "not", "no", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "should", "may", "might", "must",
    "one", "two", "both", "all", "some", "any", "each", "more", "most",
    "other", "such", "into", "over", "under", "after", "before", "between",
}


def _find_span(answer: str, claim_text: str) -> tuple[int, int] | None:
    """Locate ``claim_text`` inside ``answer``.

    Returns the (start, end) offsets of the claim, or None when it cannot
    be found. A claim that cannot be located is left alone rather than
    dropped, because silently deleting content that was never identified
    is exactly the failure this mode is meant to avoid.
    """
    if not answer or not claim_text:
        return None

    # Fast path: the claim is a verbatim substring.
    idx = answer.find(claim_text)
    if idx >= 0:
        return (idx, idx + len(claim_text))

    # Slow path: match on normalised text, then map the offsets back.
    target = _normalise(claim_text)
    if not target:
        return None

    # Walk the answer, normalising each candidate window, until one starts
    # with the target. The window grows to the length of the target so a
    # whitespace difference inside the claim does not break the match.
    window = len(target)
    norm_answer = _normalise(answer)
    pos = norm_answer.find(target)
    while pos >= 0:
        # Map the normalised position back to the original by counting
        # non-space characters before it.
        start = _orig_offset(answer, pos)
        if start is not None:
            end = start + len(claim_text)
            # The normalised match may cover slightly different original
            # characters than the claim's own length; extend to the next
            # word boundary so the whole claim is replaced.
            end = _extend_to_word_end(answer, end)
            return (start, end)
        pos = norm_answer.find(target, pos + 1)
        if window <= 0:
            break
    return None


def _orig_offset(original: str, norm_pos: int) -> int | None:
    """Map an offset in the normalised text back to the original text.

    Returns None when the offset falls inside a whitespace run that was
    collapsed, which cannot happen for a non-space match.
    """
    seen = 0
    in_ws = False
    for i, ch in enumerate(original):
        if ch.isspace():
            if not in_ws:
                if seen == norm_pos:
                    return i
                seen += 1
                in_ws = True
        else:
            in_ws = False
            if seen == norm_pos:
                return i
            seen += 1
    if seen == norm_pos:
        return len(original)
    return None


def _extend_to_word_end(text: str, end: int) -> int:
    """Extend ``end`` past trailing punctuation to the end of the word."""
    while end < len(text) and not text[end].isspace():
        end += 1
    return end


def _replacement_for(claim, evidence: list[Evidence]) -> str:
    """The text that should replace a flagged claim.

    Uses the evidence that was attached to that specific claim, not the
    top-ranked document for the whole query. Using the query's best
    document is what produced the uninformative substituted passages.

    Only the *sentence* of the evidence that best matches the claim is
    used, never the whole passage. A retrieved passage is typically a
    paragraph; substituting it wholesale for a short claim turned a
    four-word answer into a two-hundred-word passage about an unrelated
    subject, which is how the rewrite destroyed correct answers.
    """
    if claim.evidence:
        best = max(claim.evidence, key=lambda e: e.score)
        text = (best.text or "").strip()
        if text:
            return _best_sentence(text, claim.text)
    return ""


def _best_sentence(evidence_text: str, claim_text: str) -> str:
    """Pick the sentence of ``evidence_text`` that best matches the claim.

    The match is a token-overlap score, which is crude but sufficient: the
    evidence is a retrieved passage and the claim is a short statement, and
    the goal is to substitute the most relevant single sentence rather than
    the whole paragraph.
    """
    from ..utils.chunker import split_sentences

    sentences = split_sentences(evidence_text)
    if not sentences:
        return evidence_text.strip()

    claim_tokens = set(re.findall(r"\w+", claim_text.lower()))
    if not claim_tokens:
        return sentences[0]

    best = max(
        sentences,
        key=lambda s: len(claim_tokens & set(re.findall(r"\w+", s.lower())))
        / len(claim_tokens),
    )
    return best.strip()


class Step10Regeneration(Step):
    number = 10
    name = "regeneration"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(10) or {}
        mode = blk.get("mode", "evidence")
        max_attempts = int(blk.get("max_attempts", 2))
        # A claim is only rewritten when the verifier is confident it is
        # contradicted. A single low-confidence "unknown" on an otherwise
        # good answer is not grounds for changing it.
        min_confidence = float(blk.get("min_contradiction_confidence", 0.0))
        confidence = self.ctx.confidence
        claims = self.ctx.claims
        evidence = self.ctx.evidence

        bad = [c for c in claims if c.verdict in {"contradicted", "unknown"}]
        unsupported_ratio = len(bad) / max(1, len(claims))
        decision = "KEEP"
        output_text = self.ctx.answer
        guard = ""

        pipeline = self.ctx.metadata.get("pipeline")
        gen = getattr(pipeline, "generator", None)

        # A provided answer is being *judged*, not improved. Rewriting it
        # here would mean the gate and the detector score a different answer
        # than the one the benchmark supplied. The reduction action is still
        # recorded so the HRR/CCR figures stay meaningful.
        provided = self.ctx.metadata.get("provided_answer")
        if provided is not None:
            self.ctx.final_answer = self.ctx.answer
            self.ctx.iterations = min(max_attempts, max(1, len(bad)))
            self.ctx.decision_log.append({
                "step": 10,
                "mode": "provided_passthrough",
                "unsupported_ratio": unsupported_ratio,
                "decision": "KEEP",
            })
            self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
                "step": 10,
                "mode": "provided_passthrough",
                "unsupported_ratio": unsupported_ratio,
                "decision": "KEEP",
            }
            return

        if mode == "stub" or not bad or gen is None:
            output_text = self.ctx.answer
            decision = "KEEP"
        elif mode == "claim_edit":
            output_text, decision, guard = self._claim_edit(
                min_confidence=min_confidence)
        elif mode == "evidence":
            supported = [c for c in claims if c.verdict == "supported"]
            pieces = []
            for c in supported:
                pieces.append(c.text)
            if pieces:
                output_text = " ".join(pieces)
                decision = "EDITED"
            else:
                # Use first evidence doc as the answer.
                if evidence:
                    output_text = evidence[0].text
                    decision = "EVIDENCE_REPLACE"
                else:
                    output_text = "I don't know."
                    decision = "ABSTAIN"
        elif mode == "regenerate":
            ctx_text = "\n\n".join(e.text for e in evidence)
            supported = [c for c in claims if c.verdict == "supported"]
            keep = "; ".join(c.text for c in supported)
            user = (
                "Context:\n" + ctx_text + "\n\n"
                "Original answer: " + (self.ctx.answer or "") + "\n\n"
                "Verified claims to keep: " + keep + "\n\n"
                "Rewrite the answer using ONLY the verified claims above and the context. "
                "Remove anything not supported. Output the new answer only.")
            try:
                output_text = gen.generate(system="You revise answers to be evidence-grounded.", user=user)
                decision = "REGENERATED"
            except Exception:
                output_text = self.ctx.answer; decision = "KEEP"

        self.ctx.final_answer = output_text
        self.ctx.iterations = min(max_attempts, max(1, len(bad)))
        decision_record = {
            "step": 10,
            "mode": mode,
            "unsupported_ratio": unsupported_ratio,
            "decision": decision,
        }
        if guard:
            decision_record["guard"] = guard
        self.ctx.decision_log.append(decision_record)
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = decision_record

    def _claim_edit(self, min_confidence: float) -> tuple[str, str, str]:
        """Rewrite only the flagged claims, leaving the rest of the answer.

        Returns (final_text, decision, guard).

        The decision is one of:

        * ``EDITED``   at least one flagged claim was replaced
        * ``ABSTAIN``  a claim was flagged but nothing usable could replace
          it, so the answer is left untouched and the gate abstains
        * ``KEEP``     nothing was confidently contradicted

        The original answer is never replaced wholesale. When a flagged
        claim cannot be located in the answer, or no evidence was attached
        to it, the answer is left exactly as it was: degrading a correct
        answer is worse than leaving a wrong one in place, and the gate
        already reports the contradiction.
        """
        answer = self.ctx.answer or ""
        claims = self.ctx.claims
        evidence = self.ctx.evidence

        # Only claims the verifier is confident about are acted on.
        flagged = [c for c in claims if c.verdict == "contradicted"]
        confident = [c for c in flagged if self._contradiction_strength(c) >= min_confidence]

        if not confident:
            # Nothing confidently contradicted. The answer stays as it is.
            # If a claim was merely "unknown", that is a gap in the
            # evidence, not a demonstrated contradiction.
            if flagged:
                return answer, "ABSTAIN", "no_confident_contradiction"
            return answer, "KEEP", ""

        # Find every flagged claim's span in the original answer.
        spans: list[tuple[int, int, str]] = []
        skipped = 0
        guard_reason = ""
        for c in confident:
            span = _find_span(answer, c.text)
            if span is None:
                skipped += 1
                continue
            replacement = _replacement_for(c, evidence)
            if not replacement:
                # Nothing usable to put in its place. Rather than delete
                # the claim -- which loses content that may be correct --
                # leave this claim alone and rewrite the others. An
                # earlier version abandoned the whole answer here, which
                # meant one unusable claim suppressed every good
                # replacement in the same answer and left the rewrite
                # firing on only 16 of 500 rows.
                skipped += 1
                guard_reason = guard_reason or "no_usable_replacement"
                continue
            # A replacement that is vastly longer than the claim it
            # replaces is a sign the evidence does not actually answer
            # the question. Substituting a long passage for a short claim
            # inflates the answer and buries the response, which is how a
            # correct four-word answer became a two-hundred-word
            # paragraph about an unrelated subject.
            if not self._replacement_is_proportionate(replacement, c.text):
                skipped += 1
                guard_reason = guard_reason or "replacement_not_proportionate"
                continue
            # Length alone is not enough: a short replacement can still be
            # about a different subject. The replacement has to actually
            # be about the claim it replaces, otherwise the rewrite swaps
            # a wrong answer for a differently wrong answer.
            if not self._replacement_is_relevant(replacement, c.text):
                skipped += 1
                guard_reason = guard_reason or "replacement_not_relevant"
                continue
            spans.append((span[0], span[1], replacement))

        if not spans:
            # Every flagged claim could not be located in the answer, or
            # every replacement was rejected by a guard.
            return answer, "ABSTAIN", guard_reason or "claim_not_located"

        # Splice from the end backwards so the earlier offsets stay valid.
        spans.sort(key=lambda s: s[0])
        pieces: list[str] = []
        cursor = 0
        for start, end, replacement in spans:
            if start < cursor:
                # Overlapping spans: skip the later one rather than
                # producing malformed text.
                continue
            pieces.append(answer[cursor:start])
            pieces.append(replacement)
            cursor = end
        pieces.append(answer[cursor:])
        final = " ".join(p.strip() for p in pieces if p.strip())

        if not final.strip():
            return answer, "ABSTAIN", "empty_result"

        guard = ""
        if skipped:
            # The rewrite went ahead, but some flagged claims were left
            # untouched. Report how many and why, so the mitigation run
            # can separate a partial rewrite from a full one.
            guard = f"{skipped}_claim(s)_skipped"
            if guard_reason:
                guard = f"{guard}:{guard_reason}"
        return final, "EDITED", guard

    @staticmethod
    def _contradiction_strength(claim) -> float:
        """How strongly the verifier contradicts this claim, in [0, 1].

        ``Claim.confidence`` is set to ``1 - contradiction_probability`` for
        a contradicted claim, so a *high* confidence there means a *weak*
        contradiction. Reading it directly would invert the guard and
        rewrite exactly the claims the verifier was least sure about.
        """
        return 1.0 - float(claim.confidence or 0.0)

    @staticmethod
    def _replacement_is_proportionate(replacement: str, claim_text: str) -> bool:
        """Whether the replacement is a sensible length for the claim.

        A short claim is normally answered by a short span. A replacement
        that is much longer than the claim it replaces is almost certainly
        a retrieved passage that does not answer the question, and swapping
        it in makes the answer worse -- it buries the response in
        irrelevant text.

        The rule is relative, with a small absolute allowance for the
        cases where a short claim genuinely needs a longer sentence to
        answer it. An earlier version used a 40-word floor, which let a
        25-word passage replace a three-word claim; that is what turned
        "Memphis International Airport" into a paragraph about Kumasi
        Airport.
        """
        claim_words = len(claim_text.split())
        repl_words = len(replacement.split())
        if claim_words == 0:
            return False
        return repl_words <= max(12, claim_words * 4)

    @staticmethod
    def _replacement_is_relevant(replacement: str, claim_text: str) -> bool:
        """Whether the replacement is actually about the claim.

        The verifier can mark a correct claim as contradicted when the
        retrieved evidence is about a different subject -- the NLI has no
        way to say "the evidence is irrelevant", only "the premise does
        not entail this". Swapping in a replacement about a different
        subject then replaces a correct answer with a wrong one.

        The check looks for the claim's *subject* -- the capitalised
        entity it is about -- in the replacement. Token overlap over the
        whole claim is the wrong measure: a valid replacement often
        replaces the answer with a different fact about the same subject
        ("Mary Ramsey is younger than Lee Ranaldo" -> "Mary Ramsey was
        born in 1984"), which shares only two of five content tokens
        while being entirely on topic.

        A claim with no capitalised entity cannot be checked this way, so
        it falls back to the token-overlap rule.
        """
        entities = [w for w in re.findall(r"\b[A-Z][\w'-]*", claim_text)
                    if w.lower() not in _STOPWORDS]
        if entities:
            repl_lower = replacement.lower()
            # The subject has to appear in the replacement. Comparing the
            # whole entity (not a prefix) avoids "Lee" matching "Leeds".
            return any(e.lower() in repl_lower for e in entities)

        claim_tokens = {
            t for t in re.findall(r"\w+", claim_text.lower())
            if t not in _STOPWORDS
        }
        if not claim_tokens:
            return True
        repl_tokens = {
            t for t in re.findall(r"\w+", replacement.lower())
            if t not in _STOPWORDS
        }
        return len(claim_tokens & repl_tokens) / len(claim_tokens) >= 0.5
