"""Generate the synthetic price-variance dataset and persist it.

Usage::

    uv run python scripts/generate_data.py --rows 200000 --seed 42

Default sample size is 200k rows (configurable). Full DGP yields ~5.5M rows over 5,000
properties x 365 days x 3 channels.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 4: write_synthetic(n_rows={args.rows}, seed={args.seed})")


if __name__ == "__main__":
    sys.exit(main())
