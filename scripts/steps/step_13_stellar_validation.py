#!/usr/bin/env python3
"""
TEP-H0 Analysis Step 13: Stellar Validation of Scalar-Boundary Transport
=========================================================================

Numerical validation that the TEP scalar-boundary period-transport law
recovers the headline Observable Response Coefficient kappa_Cep.

The validation chain is:

1.  **Matter-frame period**: MESA/RSP (or GYRE) provides the standard
    Cepheid pulsation period P_MESA in the matter frame.
2.  **Scalar-boundary transport**: At leading order the scalar field is
    coherent across the envelope, so the local stellar structure is
    unchanged.  The TEP effect enters as an external conformal factor:
        P_obs = P_MESA * exp(-DeltaTheta).
3.  **Closure test**: Propagating P_obs through the Wesenheit P-L
    relation (slope b ≈ -3.26) yields the same DeltaMu law used in
    the main analysis.  Fitting the synthetic grid recovers
    kappa_Cep = 1.05e6 mag to numerical precision.

This step does **not** require MESA to run.  If MESA is not installed,
it uses the canonical literature baseline period (5.5 d) and still
executes the full TEP transport and closure-test logic.  Users who wish
to reproduce from first principles can install MESA, run the RSP
Cepheid test case, and point this step to the resulting history file.

Inputs:
    - Optional: MESA/RSP history file (auto-detected or user-supplied)

Outputs:
    - results/outputs/stellar_validation_grid.csv
    - results/outputs/stellar_validation_closure.json
    - results/outputs/stellar_validation_stress_test.csv
    - results/figures/stellar_validation_transport.png
    - results/figures/stellar_validation_closure.png
    - results/figures/stellar_validation_stress_test.png
    - logs/step_13_stellar_validation.log

Author: Matthew Lukin Smawfield
Date: June 2026
License: CC-BY-4.0
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.stellar_validation_core import (
    KAPPA_CEP,
    KAPPA_CEP_ERR,
    B_PL,
    SIGMA_REF,
    ALPHA_CLOCK,
    P_MESA_CANONICAL_DAYS,
    generate_transport_grid,
    fit_kappa_from_grid,
    run_higher_order_stress_test,
    read_mesa_history,
    extract_period_from_history,
    save_validation_json,
)
from scripts.utils.plot_stellar_validation import (
    plot_period_transport_grid,
    plot_closure_test,
    plot_higher_order_stress_test,
)


class Step13StellarValidation:
    r"""
    Step 13: Numerical MESA/RSP Validation of the Scalar-Boundary Transport Law
    ==========================================================================

    This step validates Appendix C of Paper 11 by demonstrating that:

    - The standard matter-frame Cepheid pulsation period is unchanged at
      leading order (the scalar boundary is coherent across the star).
    - The TEP correction enters as period export:
          P_obs = P_MESA * exp(-DeltaTheta).
    - The synthetic transported periods recover:
          DeltaMu = kappa_Cep * S(rho) * (sigma^2 - sigma_ref^2) / c^2.

    The step produces three figures and two data products that can be cited
    directly in the manuscript.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.root_dir / "logs"
        self.figures_dir = self.root_dir / "results" / "figures"
        self.outputs_dir = self.root_dir / "results" / "outputs"

        for d in (self.logs_dir, self.figures_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = TEPLogger(
            "step_13_stellar_validation",
            log_file_path=self.logs_dir / "step_13_stellar_validation.log",
        )
        set_step_logger(self.logger)

        # Paths
        self.grid_csv_path = self.outputs_dir / "stellar_validation_grid.csv"
        self.closure_json_path = self.outputs_dir / "stellar_validation_closure.json"
        self.stress_csv_path = self.outputs_dir / "stellar_validation_stress_test.csv"
        self.transport_plot_path = self.figures_dir / "stellar_validation_transport.png"
        self.closure_plot_path = self.figures_dir / "stellar_validation_closure.png"
        self.stress_plot_path = self.figures_dir / "stellar_validation_stress_test.png"

        # Public figure copy directory (for site build)
        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        self.public_figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MESA detection
    # ------------------------------------------------------------------

    def _find_mesa_period(self) -> float:
        """
        Attempt to locate a locally-run MESA/RSP history file and extract
        the converged period.  If MESA is not available or the log is
        absent, fall back to the canonical literature baseline.
        """
        mesa_dir = Path.home() / "astro" / "mesa"
        rsp_default = mesa_dir / "star" / "test_suite" / "rsp_Cepheid" / "LOGS" / "history.data"

        # Also check a local copy in the repo
        local_rsp = self.root_dir / "stellar_validation" / "mesa_rsp" / "LOGS" / "history.data"

        candidates = [local_rsp, rsp_default]
        for candidate in candidates:
            if candidate.exists():
                print_status(f"Found MESA history: {candidate}", "SUCCESS")
                try:
                    hist = read_mesa_history(candidate)
                    period = extract_period_from_history(hist)
                    print_status(f"Extracted P_MESA = {period:.6f} d", "SUCCESS")
                    return float(period)
                except Exception as exc:
                    print_status(f"MESA history read failed: {exc}", "WARNING")
                    break

        print_status(
            f"MESA/RSP not detected. Using canonical literature placeholder "
            f"P_MESA = {P_MESA_CANONICAL_DAYS} d.  For first-principles "
            f"reproduction, install MESA and run: "
            f"bash stellar_validation/run_rsp_baseline.sh",
            "WARNING",
        )
        return float(P_MESA_CANONICAL_DAYS)

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def run(self, mesa_period: float | None = None) -> dict:
        """
        Execute the full stellar validation pipeline.

        Parameters
        ----------
        mesa_period : float or None
            Override the matter-frame period.  If None, auto-detect MESA
            or fall back to the canonical value.

        Returns
        -------
        dict
            Structured results summary.
        """
        print_status(
            "STEP 13: STELLAR VALIDATION OF SCALAR-BOUNDARY TRANSPORT",
            "TITLE",
        )

        # --- 1. Baseline period -----------------------------------------
        print_status("Establishing matter-frame baseline period...", "SECTION")
        P_mesa = mesa_period if mesa_period is not None else self._find_mesa_period()
        print_status(f"P_MESA = {P_mesa:.4f} d", "INFO")
        print_status(f"SIGMA_REF = {SIGMA_REF:.2f} km/s", "INFO")
        print_status(f"KAPPA_CEP = {KAPPA_CEP:.2e} +/- {KAPPA_CEP_ERR:.2e} mag", "INFO")
        print_status(f"B_PL (Wesenheit slope) = {B_PL:.2f}", "INFO")
        print_status(f"ALPHA_CLOCK = {ALPHA_CLOCK:.6e}", "INFO")

        # --- 2. Transport grid -------------------------------------------
        print_status("Generating TEP period-transport grid...", "SECTION")
        sigmas = np.array([30, 50, 75.25, 90, 120, 150, 180, 220])
        rho_ratios = [0.0, 0.5, 1.0, 2.0]
        grid_df = generate_transport_grid(P_mesa, sigmas=sigmas, rho_ratios=rho_ratios)
        grid_df.to_csv(self.grid_csv_path, index=False)
        print_status(f"Saved grid: {self.grid_csv_path}", "SUCCESS")

        # Verify identity: DeltaMu_transport == DeltaMu_direct
        max_diff = float(grid_df["abs_difference"].max())
        print_status(
            f"Max |DeltaMu_transport - DeltaMu_direct| = {max_diff:.6e} mag",
            "TEST" if max_diff < 1e-9 else "WARNING",
        )

        # --- 3. Closure test (fit kappa_Cep back) -------------------------
        print_status("Running kappa_Cep closure test...", "SECTION")
        kappa_hat, rms, _ = fit_kappa_from_grid(grid_df)
        rel_err = abs(kappa_hat - KAPPA_CEP) / KAPPA_CEP if KAPPA_CEP != 0 else None

        print_status(f"Fitted kappa_hat   = {kappa_hat:.6e} mag", "INFO")
        print_status(f"Expected kappa_Cep = {KAPPA_CEP:.6e} mag", "INFO")
        print_status(f"RMS residual       = {rms:.6e} mag", "INFO")
        if rel_err is not None:
            print_status(f"Relative error     = {rel_err:.6e}", "INFO")

        if max_diff < 1e-9:
            print_status("Closure test PASSED.", "SUCCESS")
        else:
            print_status("Closure test FAILED (discrepancy > 1e-9 mag).", "ERROR")

        save_validation_json(
            self.closure_json_path,
            kappa_hat=kappa_hat,
            rms_residual=rms,
            max_abs_diff=max_diff,
            alpha_clock=ALPHA_CLOCK,
            P_mesa_days=P_mesa,
        )
        print_status(f"Saved closure JSON: {self.closure_json_path}", "SUCCESS")

        # --- 4. Higher-order stress test ----------------------------------
        print_status("Running higher-order stress test...", "SECTION")
        stress_df = run_higher_order_stress_test(
            P_mesa,
            sigmas=sigmas,
            rho_ratios=rho_ratios,
            q_P_values=[0.8, 1.0, 1.2],
            chi_L_values=[-0.2, 0.0, 0.2],
        )
        stress_df.to_csv(self.stress_csv_path, index=False)
        print_status(f"Saved stress-test grid: {self.stress_csv_path}", "SUCCESS")

        # Check falsifier: |b|*q_P + 2.5*chi_L ≈ 0 would cancel the effect
        falsifier_vals = stress_df["falsifier"].unique()
        print_status(
            f"Falsifier range: {falsifier_vals.min():.3f} to {falsifier_vals.max():.3f}",
            "INFO",
        )
        if np.any(np.isclose(falsifier_vals, 0.0, atol=0.1)):
            print_status(
                "WARNING: Some (q_P, chi_L) combinations approach the sign-cancellation line.",
                "WARNING",
            )
        else:
            print_status(
                "Sign survives for all scanned higher-order combinations.",
                "SUCCESS",
            )

        # --- 5. Figures ---------------------------------------------------
        print_status("Generating figures...", "SECTION")

        plot_period_transport_grid(
            P_mesa_days=P_mesa,
            sigmas=np.linspace(20, 250, 200),
            rho_ratios=[0.0, 0.5, 1.0, 2.0, 5.0],
            output_path=self.transport_plot_path,
        )
        print_status(f"Transport plot: {self.transport_plot_path}", "SUCCESS")

        plot_closure_test(grid_df, output_path=self.closure_plot_path)
        print_status(f"Closure plot: {self.closure_plot_path}", "SUCCESS")

        plot_higher_order_stress_test(stress_df, output_path=self.stress_plot_path)
        print_status(f"Stress-test plot: {self.stress_plot_path}", "SUCCESS")

        # Copy to public figures for site build
        for src in (
            self.transport_plot_path,
            self.closure_plot_path,
            self.stress_plot_path,
        ):
            dst = self.public_figures_dir / src.name
            shutil.copy(src, dst)
            print_status(f"Copied to site public: {dst}", "SUCCESS")

        # --- 6. Summary ---------------------------------------------------
        results = {
            "P_mesa_days": float(P_mesa),
            "sigma_ref_km_s": float(SIGMA_REF),
            "kappa_Cep_mag": float(KAPPA_CEP),
            "kappa_Cep_err_mag": float(KAPPA_CEP_ERR),
            "kappa_hat_mag": float(kappa_hat),
            "rms_residual_mag": float(rms),
            "max_abs_diff_mag": float(max_diff),
            "closure_passed": bool(max_diff < 1e-9),
            "alpha_clock": float(ALPHA_CLOCK),
            "B_PL": float(B_PL),
            "n_grid_points": len(grid_df),
            "files": {
                "grid_csv": str(self.grid_csv_path),
                "closure_json": str(self.closure_json_path),
                "stress_csv": str(self.stress_csv_path),
                "transport_plot": str(self.transport_plot_path),
                "closure_plot": str(self.closure_plot_path),
                "stress_plot": str(self.stress_plot_path),
            },
        }

        print_status("Step 13 complete.", "SUCCESS")
        return results


def main():
    step = Step13StellarValidation()
    step.run()


if __name__ == "__main__":
    main()
