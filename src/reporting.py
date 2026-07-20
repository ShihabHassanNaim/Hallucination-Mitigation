"""Phase 8 — End-to-end verification & reporting.

What this module does
---------------------
Phases 2–7 each produce a piece of evidence per claim. Phase 8 stitches
them together into a single reliability verdict for the answer and
serialises the whole trace to JSON / HTML / Markdown so it can be
inspected offline.

Three concerns live here:

1. :class:`ReliabilityLabel` — enum with five ordinal labels:
   ``RELIABLE``, ``MOSTLY_RELIABLE``, ``UNCERTAIN``,
   ``UNRELIABLE``, ``UNVERIFIABLE``. Determined from
   hallucination_rate + mean EEDC + fraction of supported claims.

2. :class:`ReportBuilder` — given a :class:`RAGResult`, computes the
   reliability verdict and returns a :class:`ReliabilityReport`
   dataclass that knows how to render itself to JSON / HTML / Markdown.

3. :func:`summarise_batch` — convenience for evaluating multiple
   ``RAGResult`` objects at once (used by ``scripts/evaluate.py``).
"""
from __future__ import annotations

import enum
import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .pipeline import RAGResult


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class ReliabilityLabel(str, enum.Enum):
    RELIABLE = "reliable"
    MOSTLY_RELIABLE = "mostly_reliable"
    UNCERTAIN = "uncertain"
    UNRELIABLE = "unreliable"
    UNVERIFIABLE = "unverifiable"

    def __str__(self) -> str:                # pragma: no cover
        return self.value


def _verdict(hallucination_rate: float, mean_eedc: float,
             num_claims: int) -> ReliabilityLabel:
    if num_claims == 0:
        return ReliabilityLabel.UNVERIFIABLE
    if hallucination_rate <= 0.05 and mean_eedc >= 0.75:
        return ReliabilityLabel.RELIABLE
    if hallucination_rate <= 0.20 and mean_eedc >= 0.55:
        return ReliabilityLabel.MOSTLY_RELIABLE
    if hallucination_rate <= 0.50:
        return ReliabilityLabel.UNCERTAIN
    return ReliabilityLabel.UNRELIABLE


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class ReliabilityReport:
    """Aggregate verdict + summary stats for a single :class:`RAGResult`."""

    query: str
    answer: str
    label: ReliabilityLabel
    confidence: float
    hallucination_rate: float
    mean_eedc: float
    num_claims: int
    num_flagged: int
    iterations: int
    timings_ms: dict
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "mean_eedc": round(self.mean_eedc, 4),
            "num_claims": self.num_claims,
            "num_flagged": self.num_flagged,
            "iterations": self.iterations,
            "timings_ms": dict(self.timings_ms),
            "extra": dict(self.extra),
        }

    # ----- renderers --------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        e = self.extra
        lines = [
            f"# CRISP Reliability Report",
            "",
            f"- **Query:** {self.query}",
            f"- **Verdict:** **{self.label.value.upper()}**",
            f"- **Confidence:** {self.confidence:.3f}",
            f"- **Hallucination rate:** {self.hallucination_rate:.1%}",
            f"- **Mean EEDC:** {self.mean_eedc:.3f}",
            f"- **Claims:** {self.num_claims} ({self.num_flagged} flagged)",
            f"- **Iterations:** {self.iterations}",
            "",
            "## Answer",
            "",
            self.answer,
            "",
            "## Latency (ms)",
            "",
            "| stage | ms |",
            "|---|---|",
        ]
        for stage, ms in self.timings_ms.items():
            lines.append(f"| {stage} | {ms} |")
        if e.get("claim_verdicts"):
            lines += ["", "## Per-claim verdicts", "",
                      "| # | claim | eedc | hallucinated | nli |",
                      "|---|---|---|---|---|"]
            for v in e["claim_verdicts"]:
                lines.append(
                    f"| {v['claim_id']} | {v['claim_text']} | {v['eedc_score']:.2f} "
                    f"| {'✅' if v['hallucinated'] else '❌'} | {v['nli_label']} |"
                )
        return "\n".join(lines) + "\n"

    def to_html(self) -> str:
        e = self.extra
        body_rows = []
        for v in e.get("claim_verdicts", []):
            flagged = "🚩" if v["hallucinated"] else "✅"
            body_rows.append(
                f"<tr><td>{html.escape(str(v['claim_id']))}</td>"
                f"<td>{html.escape(v['claim_text'])}</td>"
                f"<td>{v['eedc_score']:.3f}</td>"
                f"<td>{html.escape(v['nli_label'])}</td>"
                f"<td>{flagged}</td></tr>"
            )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>CRISP report — {html.escape(self.query[:80])}</title>
<style>
  body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:2rem;color:#222;max-width:900px}}
  h1{{border-bottom:2px solid #444;padding-bottom:.3rem}}
  .verdict{{display:inline-block;padding:.3rem .7rem;border-radius:.4rem;font-weight:600}}
  .reliable{{background:#d4edda;color:#155724}}
  .mostly_reliable{{background:#d1ecf1;color:#0c5460}}
  .uncertain{{background:#fff3cd;color:#856404}}
  .unreliable{{background:#f8d7da;color:#721c24}}
  .unverifiable{{background:#e2e3e5;color:#383d41}}
  table{{border-collapse:collapse;width:100%;margin-top:1rem}}
  th,td{{border:1px solid #ddd;padding:.5rem;text-align:left;font-size:.9rem}}
  th{{background:#f5f5f5}}
  pre{{background:#f8f8f8;padding:.8rem;border-radius:.3rem;overflow:auto}}
  .meta{{color:#666;font-size:.9rem}}
</style></head><body>
<h1>CRISP Reliability Report</h1>
<p class="meta">Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}</p>
<p><strong>Query:</strong> {html.escape(self.query)}</p>
<p><span class="verdict {self.label.value}">{self.label.value.upper()}</span>
&nbsp; <strong>Confidence:</strong> {self.confidence:.3f}
&nbsp; <strong>Hallucination rate:</strong> {self.hallucination_rate:.1%}
&nbsp; <strong>Iterations:</strong> {self.iterations}</p>
<h2>Answer</h2>
<pre>{html.escape(self.answer)}</pre>
<h2>Per-claim verdicts</h2>
<table><thead><tr><th>#</th><th>claim</th><th>EEDC</th><th>NLI</th><th>flag</th></tr></thead>
<tbody>{''.join(body_rows) or '<tr><td colspan="5"><em>no claims</em></td></tr>'}</tbody></table>
</body></html>
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ReportBuilder:
    """Turn a :class:`RAGResult` into a :class:`ReliabilityReport`."""

    def __init__(self, eedc_threshold: float = 0.5):
        self.eedc_threshold = eedc_threshold

    def build(self, result: RAGResult) -> ReliabilityReport:
        verdicts = result.claim_verdicts or []
        scores = [v.eedc_score for v in verdicts]
        flagged = [v for v in verdicts if v.hallucinated]
        mean_eedc = (sum(scores) / len(scores)) if scores else 1.0
        label = _verdict(result.hallucination_rate, mean_eedc, len(verdicts))
        return ReliabilityReport(
            query=result.query,
            answer=result.answer,
            label=label,
            confidence=result.confidence,
            hallucination_rate=result.hallucination_rate,
            mean_eedc=mean_eedc,
            num_claims=len(verdicts),
            num_flagged=len(flagged),
            iterations=result.iterations,
            timings_ms=result.timings_ms,
            extra={
                "claim_verdicts": [
                    {
                        "claim_id": v.claim.id,
                        "claim_text": v.claim.text,
                        "eedc_score": v.eedc_score,
                        "nli_label": v.nli.label,
                        "hallucinated": v.hallucinated,
                    }
                    for v in verdicts
                ],
                "retrieval_trace": result.retrieval_trace,
                "multi_hop_traces": result.multi_hop_traces,
                "iteration_history": result.iteration_history,
                "edit_result": result.edit_result,
            },
        )

    def build_batch(self, results: Iterable[RAGResult]) -> List[ReliabilityReport]:
        return [self.build(r) for r in results]


# ---------------------------------------------------------------------------
# Aggregate batch summary
# ---------------------------------------------------------------------------


@dataclass
class BatchSummary:
    n: int
    label_counts: dict
    mean_confidence: float
    mean_hallucination_rate: float
    mean_iterations: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "label_counts": dict(self.label_counts),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_hallucination_rate": round(self.mean_hallucination_rate, 4),
            "mean_iterations": round(self.mean_iterations, 4),
        }


def summarise_batch(reports: Sequence[ReliabilityReport]) -> BatchSummary:
    if not reports:
        return BatchSummary(n=0, label_counts={}, mean_confidence=0.0,
                            mean_hallucination_rate=0.0, mean_iterations=0.0)
    counts: dict = {}
    for r in reports:
        counts[r.label.value] = counts.get(r.label.value, 0) + 1
    return BatchSummary(
        n=len(reports),
        label_counts=counts,
        mean_confidence=sum(r.confidence for r in reports) / len(reports),
        mean_hallucination_rate=sum(r.hallucination_rate for r in reports) / len(reports),
        mean_iterations=sum(r.iterations for r in reports) / len(reports),
    )


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_report(report: ReliabilityReport, out_path) -> str:
    """Dispatch on extension: ``.json`` / ``.html`` / ``.md`` else JSON."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".html":
        p.write_text(report.to_html(), encoding="utf-8")
    elif suffix in (".md", ".markdown"):
        p.write_text(report.to_markdown(), encoding="utf-8")
    else:
        p.write_text(report.to_json(), encoding="utf-8")
    return str(p)