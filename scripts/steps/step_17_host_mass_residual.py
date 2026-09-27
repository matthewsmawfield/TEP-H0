#!/usr/bin/env python3
"""
Step 16: Host-Mass Residual Test (TEP-Specific Bias Isolation)
===============================================================

TEP predicts that the σ–H0 correlation is a CLOCK-RATE effect specific to
periodic indicators (Cepheids). If σ also tracks a shared astrophysical
systematic (e.g., host stellar mass, metallicity, dust), then regressing
out host mass M_* should:

    - WEAKEN the σ–H0 correlation in Cepheids (some signal is shared systematic)
    - WEAKEN or ELIMINATE the σ–H0 correlation in TRGB (all signal is shared systematic)

This step performs:
1. Host-mass partial correlation on Cepheid H0 vs σ
2. Host-mass partial correlation on TRGB H0 vs σ
3. Comparison of residual trends

If TEP is real, the Cepheid residual should still show a significant σ trend
after M_* correction, while the TRGB residual should be null.

Usage:
    Called by run_pipeline.py after Step 7b (TRGB Comparison).
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

from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table


class Step16HostMassResidual:
    """Formal pipeline step: isolate TEP-specific signal from shared systematics."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_16_mass_residual",
            log_file_path=self.logs_dir / "step_17_host_mass_residual.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 16: Host-mass residual test (TEP-specific bias isolation)", "TITLE"
        )

        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")
        trgb = pd.read_csv(self.results_dir / "step_15_trgb_hosts_data.csv")

        # Compute TEP screened coordinate for Cepheid hosts
        C_KM_S = 299792.458
        sigma_ref = 89.26
        if "shear_suppression" in strat.columns and "sigma_inferred" in strat.columns:
            strat["X_tep"] = (strat["shear_suppression"] * strat["sigma_inferred"]**2 - sigma_ref**2) / (C_KM_S**2)
        else:
            strat["X_tep"] = (strat["sigma_inferred"]**2 - sigma_ref**2) / (C_KM_S**2)

        if "host_logmass" in strat.columns:
            strat["step_10"] = (strat["host_logmass"] >= 10.0).astype(float)

        # --- 1. Cepheid Host-Mass Partial Correlation ---
        print_status("Cepheid: Host-mass partial correlation", "SECTION")

        # Raw correlation
        r_raw, p_raw = stats.pearsonr(strat["sigma_inferred"], strat["h0_derived"])
        r_x_raw, p_x_raw = stats.pearsonr(strat["X_tep"], strat["h0_derived"])
        print_status(f"Raw sigma vs H0: r={r_raw:.3f}, p={p_raw:.4f}", "INFO")
        print_status(f"Raw X_tep vs H0: r={r_x_raw:.3f}, p={p_x_raw:.4f}", "INFO")

        # Partial correlation controlling for host_logmass
        if "host_logmass" in strat.columns and strat["host_logmass"].notna().sum() > 5:
            valid = strat.dropna(subset=["host_logmass", "sigma_inferred", "X_tep", "h0_derived"]).copy()

            # Residuals after regressing out logmass
            slope_m, intercept_m, _, _, _ = stats.linregress(
                valid["host_logmass"], valid["h0_derived"]
            )
            h0_residual = valid["h0_derived"] - (slope_m * valid["host_logmass"] + intercept_m)

            r_part, p_part = stats.pearsonr(valid["sigma_inferred"], h0_residual)
            r_x_part, p_x_part = stats.pearsonr(valid["X_tep"], h0_residual)
            print_status(
                f"Continuous Mass-residual: sigma vs H0: r={r_part:.3f}, p={p_part:.4f} (N={len(valid)})",
                "INFO",
            )
            print_status(
                f"Continuous Mass-residual: X_tep vs H0: r={r_x_part:.3f}, p={p_x_part:.4f} (N={len(valid)})",
                "INFO",
            )

            # Residuals after regressing out discrete mass step (logM >= 10.0)
            slope_s10, intercept_s10, _, _, _ = stats.linregress(
                valid["step_10"], valid["h0_derived"]
            )
            h0_res_step = valid["h0_derived"] - (slope_s10 * valid["step_10"] + intercept_s10)
            r_part_step, p_part_step = stats.pearsonr(valid["sigma_inferred"], h0_res_step)
            r_x_part_step, p_x_part_step = stats.pearsonr(valid["X_tep"], h0_res_step)
            print_status(
                f"Mass-step residual: sigma vs H0: r={r_part_step:.3f}, p={p_part_step:.4f}",
                "INFO",
            )
            print_status(
                f"Mass-step residual: X_tep vs H0: r={r_x_part_step:.3f}, p={p_x_part_step:.4f}",
                "INFO",
            )

            # Host-mass decorrelation checks
            r_sig_M, p_sig_M = stats.pearsonr(valid["sigma_inferred"], valid["host_logmass"])
            r_x_M, p_x_M = stats.pearsonr(valid["X_tep"], valid["host_logmass"])
            r_x_step, p_x_step = stats.pearsonr(valid["X_tep"], valid["step_10"])
            print_status(f"Host-mass correlation: r(sigma, logM*) = {r_sig_M:.3f} (p={p_sig_M:.2e})", "INFO")
            print_status(f"Host-mass decorrelation: r(X_tep, logM*) = {r_x_M:.3f} (p={p_x_M:.4f})", "INFO")
            print_status(f"Mass-step decorrelation: r(X_tep, step_10) = {r_x_step:.3f} (p={p_x_step:.4f})", "INFO")

            # Also test sigma residual after logmass
            slope_s, intercept_s, _, _, _ = stats.linregress(
                valid["host_logmass"], valid["sigma_inferred"]
            )
            sigma_residual = valid["sigma_inferred"] - (
                slope_s * valid["host_logmass"] + intercept_s
            )
            r_both, p_both = stats.pearsonr(sigma_residual, h0_residual)
            print_status(
                f"Both-residual: r={r_both:.3f}, p={p_both:.4f}", "INFO"
            )
        else:
            r_part = p_part = r_both = p_both = r_x_part = p_x_part = np.nan
            r_sig_M = p_sig_M = r_x_M = p_x_M = r_x_step = p_x_step = np.nan
            r_part_step = p_part_step = r_x_part_step = p_x_part_step = np.nan
            print_status("host_logmass missing or insufficient; skipping", "INFO")

        # --- 2. TRGB Host-Mass Partial Correlation ---
        print_status("TRGB: Host-mass partial correlation", "SECTION")

        # Merge TRGB with stratified to get host_logmass
        trgb["match"] = trgb["galaxy"].str.replace(" ", "").str.upper()
        strat["match"] = strat["normalized_name"].str.replace(" ", "").str.upper()
        merged = pd.merge(
            trgb, strat[["match", "host_logmass", "shear_suppression"]], on="match", how="inner"
        )
        if "shear_suppression" in merged.columns and "sigma_inferred" in merged.columns:
            merged["X_tep"] = (merged["shear_suppression"] * merged["sigma_inferred"]**2 - sigma_ref**2) / (C_KM_S**2)
        else:
            merged["X_tep"] = (merged["sigma_inferred"]**2 - sigma_ref**2) / (C_KM_S**2)

        if len(merged) > 3 and "host_logmass" in merged.columns:
            r_trgb_raw, p_trgb_raw = stats.pearsonr(
                merged["sigma_inferred"], merged["h0_trgb"]
            )
            r_trgb_x, p_trgb_x = stats.pearsonr(
                merged["X_tep"], merged["h0_trgb"]
            )
            print_status(f"TRGB raw sigma vs H0: r={r_trgb_raw:.3f}, p={p_trgb_raw:.4f}", "INFO")
            print_status(f"TRGB raw X_tep vs H0: r={r_trgb_x:.3f}, p={p_trgb_x:.4f}", "INFO")

            valid_trgb = merged.dropna(subset=["host_logmass", "sigma_inferred", "X_tep", "h0_trgb"])
            if len(valid_trgb) > 3:
                slope_tm, intercept_tm, _, _, _ = stats.linregress(
                    valid_trgb["host_logmass"], valid_trgb["h0_trgb"]
                )
                h0_trgb_residual = valid_trgb["h0_trgb"] - (
                    slope_tm * valid_trgb["host_logmass"] + intercept_tm
                )
                r_trgb_part, p_trgb_part = stats.pearsonr(
                    valid_trgb["sigma_inferred"], h0_trgb_residual
                )
                r_trgb_x_part, p_trgb_x_part = stats.pearsonr(
                    valid_trgb["X_tep"], h0_trgb_residual
                )
                print_status(
                    f"TRGB mass-residual sigma: r={r_trgb_part:.3f}, p={p_trgb_part:.4f} (N={len(valid_trgb)})",
                    "INFO",
                )
                print_status(
                    f"TRGB mass-residual X_tep: r={r_trgb_x_part:.3f}, p={p_trgb_x_part:.4f} (N={len(valid_trgb)})",
                    "INFO",
                )
            else:
                r_trgb_part = p_trgb_part = r_trgb_x_part = p_trgb_x_part = np.nan
        else:
            r_trgb_raw = p_trgb_raw = r_trgb_part = p_trgb_part = np.nan
            r_trgb_x = p_trgb_x = r_trgb_x_part = p_trgb_x_part = np.nan
            print_status("TRGB mass data insufficient; skipping", "INFO")

        # --- 3. Summary ---
        print_status("--- SUMMARY ---", "SECTION")
        headers = ["Channel", "Raw r", "Raw p", "Mass-residual r", "Mass-residual p"]
        rows = [
            ["Cepheid (sigma)", f"{r_raw:.3f}", f"{p_raw:.4f}",
             f"{r_part:.3f}" if np.isfinite(r_part) else "N/A",
             f"{p_part:.4f}" if np.isfinite(p_part) else "N/A"],
            ["Cepheid (X_tep)", f"{r_x_raw:.3f}", f"{p_x_raw:.4f}",
             f"{r_x_part:.3f}" if np.isfinite(r_x_part) else "N/A",
             f"{p_x_part:.4f}" if np.isfinite(p_x_part) else "N/A"],
            ["TRGB (sigma)", f"{r_trgb_raw:.3f}" if np.isfinite(r_trgb_raw) else "N/A",
             f"{p_trgb_raw:.4f}" if np.isfinite(p_trgb_raw) else "N/A",
             f"{r_trgb_part:.3f}" if np.isfinite(r_trgb_part) else "N/A",
             f"{p_trgb_part:.4f}" if np.isfinite(p_trgb_part) else "N/A"],
            ["TRGB (X_tep)", f"{r_trgb_x:.3f}" if np.isfinite(r_trgb_x) else "N/A",
             f"{p_trgb_x:.4f}" if np.isfinite(p_trgb_x) else "N/A",
             f"{r_trgb_x_part:.3f}" if np.isfinite(r_trgb_x_part) else "N/A",
             f"{p_trgb_x_part:.4f}" if np.isfinite(p_trgb_x_part) else "N/A"],
        ]
        print_table(headers, rows)

        # --- 4. Save Results ---
        results = {
            "cepheid": {
                "raw_r": float(r_raw),
                "raw_p": float(p_raw),
                "mass_residual_r": float(r_part) if np.isfinite(r_part) else None,
                "mass_residual_p": float(p_part) if np.isfinite(p_part) else None,
                "both_residual_r": float(r_both) if np.isfinite(r_both) else None,
                "both_residual_p": float(p_both) if np.isfinite(p_both) else None,
                "x_tep_raw_r": float(r_x_raw),
                "x_tep_raw_p": float(p_x_raw),
                "x_tep_mass_residual_r": float(r_x_part) if np.isfinite(r_x_part) else None,
                "x_tep_mass_residual_p": float(p_x_part) if np.isfinite(p_x_part) else None,
                "x_tep_step_residual_r": float(r_x_part_step) if np.isfinite(r_x_part_step) else None,
                "x_tep_step_residual_p": float(p_x_part_step) if np.isfinite(p_x_part_step) else None,
            },
            "trgb": {
                "raw_r": float(r_trgb_raw) if np.isfinite(r_trgb_raw) else None,
                "raw_p": float(p_trgb_raw) if np.isfinite(p_trgb_raw) else None,
                "mass_residual_r": float(r_trgb_part) if np.isfinite(r_trgb_part) else None,
                "mass_residual_p": float(p_trgb_part) if np.isfinite(p_trgb_part) else None,
                "x_tep_raw_r": float(r_trgb_x) if np.isfinite(r_trgb_x) else None,
                "x_tep_raw_p": float(p_trgb_x) if np.isfinite(p_trgb_x) else None,
                "x_tep_mass_residual_r": float(r_trgb_x_part) if np.isfinite(r_trgb_x_part) else None,
                "x_tep_mass_residual_p": float(p_trgb_x_part) if np.isfinite(p_trgb_x_part) else None,
            },
            "decorrelation_metrics": {
                "r_sigma_logM": float(r_sig_M) if np.isfinite(r_sig_M) else None,
                "p_sigma_logM": float(p_sig_M) if np.isfinite(p_sig_M) else None,
                "r_x_tep_logM": float(r_x_M) if np.isfinite(r_x_M) else None,
                "p_x_tep_logM": float(p_x_M) if np.isfinite(p_x_M) else None,
                "r_x_tep_step10": float(r_x_step) if np.isfinite(r_x_step) else None,
                "p_x_tep_step10": float(p_x_step) if np.isfinite(p_x_step) else None,
            },
            "tep_prediction": (
                "If TEP is real: Cepheid mass-residual r should remain significant; "
                "TRGB mass-residual r should collapse toward zero."
            ),
        }

        with open(self.results_dir / "step_17_host_mass_residual_test.json", "w") as f:
            json.dump(results, f, indent=2)

        print_status("Saved results to step_17_host_mass_residual_test.json", "SUCCESS")
        print_status("Step 16 complete", "SUCCESS")


if __name__ == "__main__":
    Step16HostMassResidual().run()
