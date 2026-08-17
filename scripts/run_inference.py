"""Run a single query through the 13-step pipeline and print the report."""
from __future__ import annotations

import argparse
import json

from src.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the 13-step pipeline on a query.")
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    pipe = Pipeline()
    result = pipe.run(args.query)
    out = result.to_dict()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
