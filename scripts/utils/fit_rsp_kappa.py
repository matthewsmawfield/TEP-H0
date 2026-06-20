#!/usr/bin/env python3
"""
fit_rsp_kappa.py
================

Standalone script to fit the Observable Response Coefficient kappa_Cep
back from a synthetic TEP period-transport grid.

This is the numerical closure test described in Paper 11, Appendix C:
starting from the headline kappa_Cep, we generate a grid of DeltaMu
values via the scalar-boundary transport law, then perform a
least-squares fit through the origin to recover kappa_Cep.

Usage:
    python scripts/utils/fit_rsp_kappa.py results/outputs/step_33_stellar_validation_grid.csv

If no CSV path is supplied, the script auto-generates a fresh grid
using the canonical MESA baseline period (5.5 d) and fits that.

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

from scripts.utils.stellar_validation_core import (
    KAPPA_CEP,
    SIGMA_REF,
    C_KM_S,
    ALPHA_CLOCK,
    generate_transport_grid,
    fit_kappa_from_grid,
    save_validation_json,
    P_MESA_CANONICAL_DAYS,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit kappa_Cep back from a TEP period-transport grid"
    )
    ap.add_argument(
        "grid_csv",
        nargs="?",
        default=None,
        help=(
            "Path to transport-grid CSV (e.g. results/outputs/step_33_stellar_validation_grid.csv). "
            "If omitted, a fresh grid is generated using P_MESA_CANONICAL_DAYS."
        ),
    )
    ap.add_argument(
        "--mesa-period", "-p",
        type=float,
        default=None,
        help="Override the matter-frame baseline period (days).",
    )
    ap.add_argument(
        "--output-json", "-j",
        default=None,
        help="Path to write the closure-test JSON summary.",
    )
    args = ap.parse_args()

    if args.grid_csv is not None:
        import pandas as pd
        grid_path = Path(args.grid_csv)
        if not grid_path.exists():
            print(f"ERROR: Grid CSV not found: {grid_path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(grid_path)
        P_mesa = df["P_mesa_days"].iloc[0]
    else:
        P_mesa = args.mesa_period if args.mesa_period is not None else P_MESA_CANONICAL_DAYS
        df = generate_transport_grid(P_mesa)
        grid_path = _PROJECT_ROOT / "results" / "outputs" / "step_33_stellar_validation_grid.csv"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(grid_path, index=False)
        print(f"Generated fresh grid: {grid_path}")

    kappa_hat, rms, max_diff = fit_kappa_from_grid(df)

    print("=" * 60)
    print("  RSP/TEP kappa_Cep Closure Test")
    print("=" * 60)
    print(f"  P_MESA           : {P_mesa:.4f} d")
    print(f"  sigma_ref        : {SIGMA_REF:.2f} km/s")
    print(f"  c                : {C_KM_S:.3f} km/s")
    print(f"  Fitted kappa_hat : {kappa_hat:.6e} mag")
    print(f"  Expected kappa   : {KAPPA_CEP:.6e} mag")
    print(f"  RMS residual     : {rms:.6e} mag")
    print(f"  Max |dmu_transport - dmu_direct| : {max_diff:.6e} mag")
    print("=" * 60)

    if max_diff < 1e-9:
        print("  PASS: Transport and direct formulae agree to numerical precision.")
    else:
        print("  FAIL: Discrepancy exceeds 1e-9 mag.")

    if args.output_json:
        save_validation_json(
            args.output_json,
            kappa_hat=kappa_hat,
            rms_residual=rms,
            max_abs_diff=max_diff,
            alpha_clock=ALPHA_CLOCK,
            P_mesa_days=P_mesa,
        )
        print(f"  Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
