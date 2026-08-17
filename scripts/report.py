"""Print the markdown report for a single query."""
from __future__ import annotations

import argparse

from src.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    pipe = Pipeline()
    res = pipe.run(args.query)
    print(res.report_markdown or "")


if __name__ == "__main__":
    main()
