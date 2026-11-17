"""Lexical grounding: how much of the answer came from the source.

What this measures
------------------
The share of the answer's *content words* that the source does **not**
contain. A grounded answer reuses the source's vocabulary; a hallucinated
answer introduces facts the source does not have, and those facts come
with words the source never used. So a high novel-content rate is evidence
of hallucination.

This is a different question from the one the NLI asks. The NLI judges
"is this claim entailed by the evidence", which is a per-claim entailment
question that is hard when the evidence is long and the answer contains
facts that are simply not in the source. Lexical grounding asks the weaker
but much more robust question "how much of this answer came from the
source at all", and it answers it per answer rather than per claim.

Measured on RAGTruth (100 pairs per task type, length-matched split):

| task type | AUC (higher novel rate = more likely hallucinated) | 95% CI |
|---|---|---|
| Data2txt | 0.715 | [0.568, 0.835] |
| QA        | 0.776 | [0.678, 0.860] |
| Summary   | 0.808 | [0.723, 0.887] |

Why it belongs in the fusion
----------------------------
Before this existed, the fused score mixed the NLI's mean supported-claim
confidence with the share of supported claims -- the same evidence counted
twice -- and had no answer-level grounding term at all. On RAGTruth, whose
answers are 110-160 words of long-form generation, the claim-level terms
were nearly uninformative (detection ROC-AUC 0.478, CI [0.426, 0.542])
while this term separates the classes on its own.

Direction
---------
``novel_content_rate`` is high for hallucinated answers, so the fusion
weight is applied to ``1 - novel_content_rate`` as a *trust* signal.
"""
from __future__ import annotations

import re

# Function words carry no information about grounding. Keeping them would
# inflate the score of any answer, because most of an English answer is
# function words that appear in any source.
_STOPWORDS = frozenset("""
a an the is was were are be been being am of in on at to for by with and or
but nor so yet if then than that this these those there here it its as
from who whom whose which what when where why how all any both each few
more most other some such no not only own same too very can will just
should now i me my we our you your he him his she her they them their
about above after again against below between during into through under
until up down out off over again further once because while
""".split())


def content_tokens(text: str) -> list[str]:
    """The content words of ``text``, in order, lowercased."""
    return [
        w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
        if w not in _STOPWORDS and len(w) > 1
    ]


def novel_content_rate(answer: str, evidence_texts: list[str]) -> float:
    """The share of the answer's content words the source does not contain.

    Returns a value in [0, 1]. Higher means more of the answer is novel,
    i.e. more likely to be hallucinated.

    Duplicated content words are counted once, so an answer that repeats
    one grounded phrase does not have its novel rate inflated.
    """
    tokens = set(content_tokens(answer))
    if not tokens:
        return 0.0
    source: set[str] = set()
    for t in evidence_texts or []:
        source.update(content_tokens(t))
    if not source:
        # No evidence to compare against: nothing is known to be novel.
        return 0.0
    return len(tokens - source) / len(tokens)


def grounding_trust(answer: str, evidence_texts: list[str]) -> float:
    """The trust signal: 1.0 when the answer is fully grounded, 0.0 when
    none of its content words appear in the source."""
    return 1.0 - novel_content_rate(answer, evidence_texts)
