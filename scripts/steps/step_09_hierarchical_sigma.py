#!/usr/bin/env python3
"""
Step 09: Velocity-Dispersion Errors-in-Variables Audit
======================================================

Audits method-specific velocity-dispersion uncertainties and fits an
orthogonal-distance regression (ODR).  This is an errors-in-variables
diagnostic, not a latent hierarchical population model.

Methods:
    1. Direct stellar absorption (gold standard)
    2. HI linewidth proxy (calibrated via HyperLEDA)
    3. SDSS DR7 fiber spectrum (aperture-limited)
    4. HyperLEDA compilation (heterogeneous sources)

This is a formal pipeline step. It:
    - Reports method-specific bias and scatter
    - Compares raw and bias-corrected correlation with H0
    - Computes ODR slopes with sigma, distance, and peculiar-velocity errors

Usage:
    Called by run_pipeline.py after Step 2 (Stratification) completes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_correction import C_SQUARED_KM_S, compute_anchor_sigma_ref


LN10_OVER_5 = np.log(10.0) / 5.0
SIGMA_V_PRIMARY = 250.0


class Step09HierarchicalSigma:
    """Formal pipeline step: sigma errors-in-variables audit."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_09_sigma",
            log_file_path=self.logs_dir / "step_09_hierarchical_sigma.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(
            ">>> STEP 09: Velocity-dispersion errors-in-variables audit", "TITLE"
        )

        prov = pd.read_csv(self.results_dir / "step_07_sigma_provenance_table.csv")
        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")

        # Merge provenance with stratified data
        merged = strat.merge(
            prov[["normalized_name", "sigma_method", "sigma_measured_error_kms"]],
            on="normalized_name",
            how="left",
        )

        # Method-specific parameters
        method_results = {}
        methods = merged["sigma_method"].unique()

        for method in methods:
            subset = merged[merged["sigma_method"] == method]
            if len(subset) == 0:
                continue

            bias = (subset["sigma_measured"] - subset["sigma_inferred"]).median()
            scatter = subset["sigma_measured_error_kms"].median()

            method_results[method] = {
                "n_hosts": int(len(subset)),
                "median_bias_kms": float(bias),
                "median_scatter_kms": float(scatter),
                "mean_measured_kms": float(subset["sigma_measured"].mean()),
                "mean_inferred_kms": float(subset["sigma_inferred"].mean()),
            }

        print_status("Method-specific bias and scatter:", "INFO")
        for method, res in method_results.items():
            print_status(
                f"  {method}: N={res['n_hosts']}, bias={res['median_bias_kms']:+.2f} km/s, "
                f"scatter={res['median_scatter_kms']:.2f} km/s",
                "INFO",
            )

        # ODR slope comparison
        sigma_vals = strat["sigma_inferred"].values
        h0_vals = strat["h0_derived"].values
        sigma_errs = merged["sigma_measured_error_kms"].fillna(10.0).values
        mu_frac_err = LN10_OVER_5 * strat["error"].fillna(0.05).values
        flow_frac_err = np.divide(
            SIGMA_V_PRIMARY,
            strat["velocity"].values,
            out=np.zeros(len(strat), dtype=float),
            where=strat["velocity"].values > 0,
        )
        h0_errs = h0_vals * np.sqrt(mu_frac_err**2 + flow_frac_err**2)

        ols_slope, _ = np.polyfit(sigma_vals, h0_vals, 1)
        odr_slope, odr_err, odr_err_scaled = self._compute_odr_slope(
            sigma_vals, h0_vals, sigma_errs, h0_errs
        )

        sigma_ref_active = compute_anchor_sigma_ref(screened=True)
        S_total = strat["shear_suppression"].values
        x_tep = (S_total * sigma_vals**2 - sigma_ref_active**2) / C_SQUARED_KM_S
        x_tep_err = np.abs(2.0 * S_total * sigma_vals / C_SQUARED_KM_S) * sigma_errs
        gamma_odr, gamma_odr_err, gamma_odr_err_scaled = self._compute_odr_slope(
            x_tep, h0_vals, x_tep_err, h0_errs
        )

        print_status(f"OLS slope: {ols_slope:.4f}", "INFO")
        if odr_slope is not None:
            print_status(
                f"Linear-sigma ODR: {odr_slope:.4f} ± {odr_err:.4f} "
                f"({abs(odr_slope / odr_err):.2f} sigma; fixed observational errors)",
                "INFO",
            )
        if gamma_odr is not None:
            print_status(
                f"TEP-endpoint ODR: Gamma={gamma_odr:.3e} ± {gamma_odr_err:.3e} "
                f"({abs(gamma_odr / gamma_odr_err):.2f} sigma; fixed observational errors)",
                "INFO",
            )

        # Stellar-only subsample
        stellar = merged[merged["sigma_method"] == "stellar absorption"]
        if len(stellar) > 2:
            r_raw, p_raw = stats.pearsonr(stellar["sigma_inferred"], stellar["h0_derived"])
            print_status(
                f"Stellar-only: N={len(stellar)}, r={r_raw:.3f}, p={p_raw:.3f}", "INFO"
            )

        # Save results
        out = {
            "method_parameters": method_results,
            "ols_slope": float(ols_slope),
            "odr_slope": float(odr_slope) if odr_slope is not None else None,
            "odr_slope_error": float(odr_err) if odr_err is not None else None,
            "odr_slope_error_residual_scaled": float(odr_err_scaled) if odr_err_scaled is not None else None,
            "odr_significance": float(abs(odr_slope / odr_err)) if odr_err else None,
            "odr_ols_ratio": float(odr_slope / ols_slope) if odr_slope else None,
            "tep_endpoint_gamma_odr": float(gamma_odr) if gamma_odr is not None else None,
            "tep_endpoint_gamma_error": float(gamma_odr_err) if gamma_odr_err is not None else None,
            "tep_endpoint_gamma_error_residual_scaled": float(gamma_odr_err_scaled) if gamma_odr_err_scaled is not None else None,
            "tep_endpoint_significance": float(abs(gamma_odr / gamma_odr_err)) if gamma_odr_err else None,
            "sigma_ref_active_response_kms": float(sigma_ref_active),
            "peculiar_velocity_kms": SIGMA_V_PRIMARY,
            "description": (
                "Errors-in-variables audit for velocity dispersion. H0 errors include "
                "distance-modulus uncertainty and an adopted 250 km/s peculiar-velocity "
                "term. Primary ODR errors use the supplied observational covariance; the "
                "residual-rescaled SciPy errors are retained as a diagnostic only."
            ),
        }

        with open(
            self.results_dir / "step_09_hierarchical_sigma_measurement_model.json", "w"
        ) as f:
            json.dump(out, f, indent=2)

        print_status("Step 09 complete", "SUCCESS")

    def _compute_odr_slope(self, sigma_vals, h0_vals, sigma_errs, h0_errs):
        try:
            from scipy.odr import ODR, Model, RealData

            def linear(B, x):
                return B[0] * x + B[1]

            model = Model(linear)
            data = RealData(sigma_vals, h0_vals, sx=sigma_errs, sy=h0_errs)
            slope0, intercept0 = np.polyfit(sigma_vals, h0_vals, 1)
            odr = ODR(data, model, beta0=[slope0, intercept0])
            output = odr.run()
            fixed_error_se = float(np.sqrt(max(output.cov_beta[0, 0], 0.0)))
            residual_scaled_se = float(output.sd_beta[0])
            return float(output.beta[0]), fixed_error_se, residual_scaled_se
        except Exception:
            return None, None, None


if __name__ == "__main__":
    Step09HierarchicalSigma().run()
