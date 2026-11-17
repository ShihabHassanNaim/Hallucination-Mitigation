"""Step 9 — Confidence Fusion.

Combines similarity, retrieval, NLI, coverage, lexical grounding, EEDC and
the early detector into a single **hallucination risk** score.

Direction
---------
This step produces a risk: **higher means more likely to be hallucinated**.
That is the direction the downstream gate (step 11) and the detection metric
both need. The step's output field, ``ctx.confidence``, remains the *trust*
score -- ``1 - risk`` -- because that is the name the rest of the project
has always used for it.

Why the signs are what they are
-------------------------------
Each component has a natural direction, and the directions are not all the
same:

* ``nli`` and ``coverage`` are high when the claims are *supported*, so
  they are **trust** signals: high value means less likely hallucinated.
* ``grounding`` is high when the answer's content words came from the
  source, so it is also a **trust** signal.
* ``detector_trust`` is ``1 - hallucination_probability`` from step 3, so
  a *high* value means the detector thinks the answer is faithful. It is
  therefore a **trust** signal too. (Measured: the detector's raw
  probability has AUC 0.5389 read as a risk signal and ``detector_trust``
  has 0.4611 — the two mirror each other exactly. See
  ``scripts/check_detector_direction.py``.)
* ``similarity``, ``retrieval_top1`` and ``eedc`` rise with hallucination
  risk on this data, so they are **risk** signals.

The previous version of this step added every component with a positive
weight into a score it treated as uniformly trust-oriented. On RAGTruth
that fusion scored **ROC-AUC 0.360** as a detection score -- worse than a
coin flip -- because its two largest components (``nli`` and ``coverage``)
pointed in the opposite direction to the truth. The weights and signs below
were fitted on a tuning half of the RAGTruth matched split and evaluated
on a held-out half (see ``scripts/fit_detection_head.py``): held-out
detection ROC-AUC **0.667, CI [0.472, 0.834]**.

The fit is a logistic regression on standardised components. The
standardisation statistics are stored in ``configs/default.yaml`` next to
the weights, because the components have very different scales (the EEDC
entropy is ~0.3, the retrieval score ~0.4, the NLI confidence ~0.5) and a
raw-weight sum would be dominated by whichever component happens to have the
widest range.

The grounding term
------------------
The lexical grounding of the answer in the source is new. The old fusion
counted the NLI's mean supported-claim confidence and the supported-claim
*rate* separately -- the same evidence twice, both at claim level -- and
had no answer-level grounding term at all. On long-form answers, where
the claim-level NLI is close to uninformative, the whole score collapsed
towards the base rate. Grounding asks a different question: how much of
this answer came from the source at all.

Grounding is measured against the **top-1 retrieved document only**.
Pooling all five retrieved documents destroys the signal: the extra
documents contribute their own vocabulary, which absorbs the answer's
genuinely novel words and pushes every answer's novel rate towards the
same value.
"""
from __future__ import annotations

from . import Step, StepContext
from ..utils.grounding import grounding_trust


# The order the weights are stored in.
COMPONENTS = [
    "similarity",
    "retrieval_top1",
    "nli",
    "coverage",
    "eedc",
    "detector_trust",
    "grounding",
]

# Which components are trust signals (high value = less likely
# hallucinated). Everything else is a risk signal. Fitted on the RAGTruth
# tuning half; see scripts/fit_detection_head.py.
TRUST_COMPONENTS = {"nli", "coverage", "grounding", "detector_trust"}


class Step09ConfidenceFusion(Step):
    number = 9
    name = "confidence_fusion"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(9) or {}
        head = blk.get("detection_head", {}) or {}
        # The fitted head takes precedence; the hand-set weights are the
        # fallback when a config has no head block.
        weights = head.get("weights") or blk.get("fusion") \
            or cfg.raw.get("step09_fusion_weights", {})
        mean = head.get("component_mean", {}) or {}
        std = head.get("component_std", {}) or {}
        signs = head.get("sign", {}) or {}

        claims = self.ctx.claims
        evidence = self.ctx.evidence
        feat = self.ctx.feature_vector or {}
        cosine = feat.get("cosine", 0.0)
        if claims:
            nli_scores = [c.confidence if c.verdict == "supported" else 0.0 for c in claims]
            avg_nli = sum(nli_scores) / len(nli_scores)
        else:
            avg_nli = 0.0
        ret_top1 = evidence[0].score if evidence else 0.0
        coverage = (sum(1 for c in claims if c.verdict == "supported") / len(claims)) if claims else 0.0
        eedc = self.ctx.eedc_score or 0.0

        # The lexical grounding of the whole answer in the top-1 source.
        #
        # Grounding is only meaningful when the retrieved source is a real
        # context. On a closed-book dataset whose retrieval corpus is built
        # from the reference answers, the "source" *is* the answer key, and
        # grounding then measures "is this answer the reference answer" --
        # the label. The dataset declares this explicitly with
        # ``grounding_source: none``, and grounding is then left out of the
        # score rather than silently reading the answer key.
        grounding_src = blk.get("grounding_source", "retrieved")
        # The run may override it (the evaluation script sets this per
        # dataset, because whether the retrieved source is a legitimate
        # context is a property of the dataset, not of the pipeline).
        override = self.ctx.metadata.get("grounding_source")
        if override is not None:
            grounding_src = override
        if grounding_src == "none" or not evidence:
            grounding = 0.5
            grounding_used = False
        else:
            grounding = grounding_trust(self.ctx.answer or "", [evidence[0].text])
            grounding_used = True

        # The early detector (step 3) emits a *hallucination* probability.
        halluc_prob = self.ctx.metadata.get("hallucination_probability")
        if halluc_prob is None:
            detector_trust = 0.5
        else:
            detector_trust = 1.0 - float(halluc_prob)

        raw = {
            "similarity": cosine,
            "retrieval_top1": ret_top1,
            "nli": avg_nli,
            "coverage": coverage,
            "eedc": eedc,
            "detector_trust": detector_trust,
            "grounding": grounding,
        }

        # Components that are *undefined* when there is no retrieved source.
        #
        # With no evidence the NLI has no premise, so every claim comes back
        # "unknown" and ``nli``/``coverage`` collapse to 0.0; ``retrieval_top1``
        # has no top-1 document to score. These are constants, not
        # measurements. They carry no information about the label, and
        # leaving them in with their fitted weights would shift every
        # closed-book row by the same amount -- a pure base-rate offset
        # with no discrimination, and a misleading one because the score
        # would look like it had read a source it never saw.
        #
        # So on a closed-book run they are left out of the score entirely,
        # the same treatment the grounding term gets.
        no_source = not evidence
        evidence_dependent = {"retrieval_top1", "nli", "coverage"}

        # --- the risk score ------------------------------------------------
        # A logistic head on standardised components. Each trust component
        # is negated first, so a larger contribution to ``risk`` always
        # means *more likely to be hallucinated*.
        parts = []
        used_weights = {}
        for c in COMPONENTS:
            w = float(weights.get(c, 0.0))
            if w == 0.0:
                continue
            if c == "grounding" and not grounding_used:
                # No legitimate source to ground against, so the term is
                # left out of the score entirely rather than contributing a
                # meaningless or leaking value.
                continue
            if no_source and c in evidence_dependent:
                # No retrieved source at all, so this component is a
                # constant rather than a measurement. Left out.
                continue
            mu = float(mean.get(c, 0.0))
            sd = float(std.get(c, 1.0)) or 1.0
            if abs(sd) < 1e-9:
                sd = 1.0
            z = (raw[c] - mu) / sd
            # The sign comes from the config when a fitted head is present;
            # otherwise fall back to the known trust components.
            sg = float(signs.get(c, -1.0 if c in TRUST_COMPONENTS else 1.0))
            parts.append(w * sg * z)
            used_weights[c] = w
        risk = sum(parts) if parts else 0.0

        # Sigmoid so the risk is a probability in [0, 1]. The intercept is
        # not fitted -- the components are centred, so 0 is the decision
        # boundary -- which keeps the score comparable across datasets.
        import math
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, risk))))

        # The trust score is what the rest of the project has always called
        # ``ctx.confidence``: 1.0 for a fully trusted answer.
        confidence = 1.0 - prob

        # Per-claim scores, in trust terms, as before.
        w_nli = float(weights.get("nli", 0.25))
        w_cov = float(weights.get("coverage", 0.15))
        w_ret = float(weights.get("retrieval", 0.20))
        w_sim = float(weights.get("similarity", 0.30))
        w_eedc = float(weights.get("eedc", 0.10))
        w_det = float(weights.get("detector", 0.15))
        for c in claims:
            cs = w_nli * (c.confidence if c.verdict == "supported" else 0.0)
            cs += w_ret * (c.evidence[0].score if c.evidence else 0.0)
            cs += w_cov * (1.0 if c.verdict == "supported" else 0.0)
            cs += w_sim * cosine
            cs += w_eedc * eedc
            cs += w_det * detector_trust
            c.confidence = float(cs)

        self.ctx.confidence = float(confidence)
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "weights": used_weights,
            "components": raw,
            "grounding_used": grounding_used,
            "hallucination_probability": prob,
            "hallucination_risk": risk,
            "overall_confidence": confidence,
            "per_claim_confidence": [c.confidence for c in claims],
        }
