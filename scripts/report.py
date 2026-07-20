"""CLI: turn an inference trace (``data/preds.jsonl``) into reports.

Usage
-----
    # JSONL of CRISP pipeline results -> JSON / HTML reports
    python scripts/report.py \\
        --in data/preds.jsonl \\
        --out-dir reports/

The script reads each line of ``--in`` (must be ``RAGResult.to_dict()``),
re-builds a ``RAGResult``, runs ``ReportBuilder`` to compute the
reliability verdict, and writes one report per query plus a
``summary.json`` with batch statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import RAGResult
from src.reporting import (
    BatchSummary,
    ReportBuilder,
    summarise_batch,
    write_report,
)


def _load_results(path: Path):
    """Yield (raw_dict, RAGResult) pairs."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            result = RAGResult(
                query=payload.get("query", ""),
                answer=payload.get("answer", ""),
                retrieved_docs=[],
                prompt=payload.get("prompt", ""),
                confidence=payload.get("confidence", 1.0),
                iterations=payload.get("iterations", 1),
                timings_ms=payload.get("timings_ms", {}),
                hallucination_rate=payload.get("hallucination_rate", 0.0),
                retrieval_trace=payload.get("retrieval_trace"),
                multi_hop_traces=payload.get("multi_hop_traces", []),
                iteration_history=payload.get("iteration_history", []),
                edit_result=payload.get("edit_result"),
            )
            yield payload, result


def main() -> None:
    p = argparse.ArgumentParser(description="Build CRISP reliability reports.")
    p.add_argument("--in", dest="in_path", required=True,
                   help="Input JSONL of RAGResult payloads (e.g. data/preds.jsonl).")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write per-query reports + summary.json.")
    p.add_argument("--format", choices=["json", "html", "md"], default="json",
                   help="Per-report format (default: json).")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = ReportBuilder()
    reports = []
    for i, (_raw, result) in enumerate(_load_results(in_path)):
        rep = builder.build(result)
        reports.append(rep)
        ext = "." + args.format
        # Sanitise the query for filename use.
        slug = "".join(c if c.isalnum() else "_" for c in result.query[:40]) or f"q{i:03d}"
        write_report(rep, out_dir / f"{i:03d}_{slug}{ext}")
        print(f"[{i}] {rep.label.value:>16}  conf={rep.confidence:.3f}  "
              f"halluc={rep.hallucination_rate:.1%}  q={result.query[:50]!r}")

    summary = summarise_batch(reports)
    (out_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(reports)} report(s) and summary to {out_dir}")


if __name__ == "__main__":
    main()
