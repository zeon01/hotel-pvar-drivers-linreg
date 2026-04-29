"""Generate the synthetic price-variance dataset and persist it."""

from __future__ import annotations

import argparse
import logging
import sys

from pvar_linreg.data import write_synthetic


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = write_synthetic(n_rows=args.rows, seed=args.seed)
    print(f"synthetic dataset at: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
