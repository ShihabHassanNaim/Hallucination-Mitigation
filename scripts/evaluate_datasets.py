"""Run the BASE + FULL evaluation across every registered dataset.

This is a thin wrapper that invokes :mod:`scripts.evaluate` and
:mod:`scripts.evaluate_full` once per dataset slug, so the resulting
files land in the paths the Flask app's ``DATASET_REGISTRY`` already
expects (``data/metrics_halueval.json``, ``data/metrics_truthfulqa.json``).

Usage::

    python -m scripts.evaluate_datasets                 # all registered datasets
    python -m scripts.evaluate_datasets --only halueval
    python -m scripts.evaluate_datasets --limit 200     # cap rows per dataset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# Mirror of src/evaluation/report.py:DATASET_REGISTRY paths so the runner
# doesn't need a full import (keeps it cheap + scriptable).
DATASET_SLUGS = ("halueval", "truthfulqa")


def _run_step(label: str, cmd: list[str]) -> int:
    """Run a subprocess step with live stdout streaming."""
    import subprocess

    print(f"\n=== {label} :: {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    # Stream stdout/stderr line-by-line so progress is visible in real time.
    rc = 0
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        rc = proc.returncode or 0
    print(f"=== {label} done in {time.time() - t0:.1f}s (exit={rc}) ===\n", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="+",
        choices=DATASET_SLUGS,
        help="restrict to one or more dataset slugs (default: all)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap rows per dataset (forwarded to the underlying scripts)",
    )
    ap.add_argument(
        "--skip-base",
        action="store_true",
        help="skip the BASE metrics pass (only run evaluate_full)",
    )
    ap.add_argument(
        "--skip-full",
        action="store_true",
        help="skip the FULL metrics pass (only run evaluate)",
    )
    ap.add_argument(
        "--index",
        default=None,
        help="path to a saved FAISS+BM25 index directory forwarded to both scripts",
    )
    args = ap.parse_args()

    targets = args.only or list(DATASET_SLUGS)
    overall_rc = 0

    for slug in targets:
        for step_name, script, enabled in (
            ("BASE", "scripts.evaluate",       not args.skip_base),
            ("FULL", "scripts.evaluate_full",  not args.skip_full),
        ):
            if not enabled:
                continue
            cmd = [sys.executable, "-u", "-m", script, "--dataset", slug]
            if args.index:
                cmd.extend(["--index", args.index])
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            rc = _run_step(f"{slug}/{step_name}", cmd)
            if rc != 0:
                overall_rc = rc
                print(
                    f"WARN: {slug} {step_name} step exited with {rc}; "
                    f"continuing so the other dataset still produces output.",
                    flush=True,
                )

    # Final summary.
    print("\n=== Summary ===")
    for slug in targets:
        out = Path(f"data/metrics_{slug}.json")
        if out.exists():
            print(f"  {slug:>10}  -> {out}  ({out.stat().st_size // 1024} KB)")
        else:
            print(f"  {slug:>10}  -> MISSING")
    return 1 if overall_rc else 0


if __name__ == "__main__":
    raise SystemExit(main())