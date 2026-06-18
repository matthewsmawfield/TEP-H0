#!/usr/bin/env python3
"""
extract_rsp_period.py
=====================

Standalone utility to extract the converged pulsation period from a
MESA/RSP history file.

Usage:
    python scripts/utils/extract_rsp_period.py LOGS/history.data

If no direct period column is found, the script prints the available
columns and advises the user to run a time-series peak-finding script.

Author: Matthew Lukin Smawfield
Date: June 2026
License: CC-BY-4.0
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path when run standalone
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.stellar_validation_core import read_mesa_history, extract_period_from_history


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract converged period from MESA/RSP history.data"
    )
    ap.add_argument(
        "history_file",
        help="Path to MESA LOGS/history.data (or equivalent)",
    )
    ap.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to write the period (as plain text)",
    )
    args = ap.parse_args()

    hist_path = Path(args.history_file)
    if not hist_path.exists():
        print(f"ERROR: File not found: {hist_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading MESA history: {hist_path}")
    hist = read_mesa_history(hist_path)
    print(f"  Rows: {len(hist)}, Columns: {len(hist.columns)}")

    try:
        period = extract_period_from_history(hist)
        print(f"  Extracted period: {period:.6f} days")
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(str(period))
            print(f"  Written to: {out_path}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("\nAvailable columns:", file=sys.stderr)
        for col in hist.columns:
            print(f"  {col}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
