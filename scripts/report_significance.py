"""Report the pipeline's own contributions as paired significance tests.

This is the one place in the thesis where a statistical win can be
claimed, and the reason is structural rather than a matter of luck:
every ablation variant scores the *same* queries, so the comparison is
paired and a significance test is valid.

The literature comparison cannot carry a significance test -- none of the
cited papers publish per-query scores, so their numbers are point
estimates only. That is a limitation of the field, not of this work.

Usage:
  python -m scripts.report_significance --ablation data/ablation_paired.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.significance import compare_variants, load_ablation


def _fmt(res: dict) -> str:
    ci = res.get("ci95")
    ci_txt = (
        f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
    )
    sig = "significant" if res.get("significant") else "not significant"
    return (
        f"  AUC {res['auc_a']:.3f} vs {res['auc_b']:.3f}   "
        f"diff {res['difference']:+.3f}   "
        f"95% CI {ci_txt}   "
        f"p = {res['p_value']:.4f}   ({sig})   "
        f"[n = {res['n_queries']} queries]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ablation",
        default="data/ablation_paired.json",
        help="ablation results containing per_query records",
    )
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    results = load_ablation(args.ablation)

    print("Paired significance tests — pipeline components vs. full pipeline")
    print("=" * 78)
    print()
    print("Each variant disables one stage and is scored on the *same*")
    print("queries as the full pipeline, so the test is paired.")
    print()

    report: dict[str, dict] = {}
    for variant in results:
        if variant == "full_pipeline":
            continue
        res = compare_variants(results, "full_pipeline", variant)
        report[variant] = res
        print(f"{variant}")
        print(_fmt(res))
        print()

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
