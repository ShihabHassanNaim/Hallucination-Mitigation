"""Measure the claim-verdict distribution per label on the RAGTruth split.

A fast diagnostic: the detection score is built from the claim verdicts, so
the verdict distribution is what has to move for detection to work. This
script runs the pipeline over the first N rows and reports the verdict mix
for faithful and hallucinated rows separately, plus the contradiction rate
that the detection score is built from.

Usage:
  python -m scripts.measure_claim_verdicts --limit 100
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.data_loader import load_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ragtruth_eval_1000_matched.jsonl")
    ap.add_argument("--index", default="data/index_ragtruth")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="data/ragtruth_verdicts.json")
    args = ap.parse_args()

    cfg = load_config()
    pipe = Pipeline(cfg)
    pipe.load_index(args.index)

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    mix = {0: collections.Counter(), 1: collections.Counter()}
    n_claims = {0: 0, 1: 0}
    conf = {0: [], 1: []}
    per_task: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)

    for i, r in enumerate(rows):
        res = pipe.run(r["query"], answer=r["answer"], label=int(r["label"]),
                       gold_answer=r.get("right_answer"))
        d = res.to_dict()
        lab = int(r["label"])
        for c in d["claims"]:
            mix[lab][c["verdict"]] += 1
            per_task[r.get("task_type", "?")][c["verdict"]] += 1
        n_claims[lab] += len(d["claims"])
        conf[lab].append(float(d["confidence"]))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    report: dict = {"n_rows": len(rows), "per_label": {}}
    for lab in (0, 1):
        tot = max(1, sum(mix[lab].values()))
        report["per_label"][str(lab)] = {
            "n_claims": n_claims[lab],
            "verdict_mix": {k: round(v / tot, 4) for k, v in mix[lab].items()},
            "contradiction_rate": round(mix[lab]["contradicted"] / tot, 4),
            "mean_confidence": round(sum(conf[lab]) / max(1, len(conf[lab])), 4),
        }
    report["per_task_verdict_mix"] = {
        tt: {k: round(v / max(1, sum(c.values())), 4) for k, v in c.items()}
        for tt, c in per_task.items()
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
