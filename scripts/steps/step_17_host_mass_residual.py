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
            log_file_path=self.logs_dir / "step_16_host_mass_residual.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 16: HOST-MASS RESIDUAL TEST (TEP-SPECIFIC BIAS ISOLATION)", "TITLE"
        )

        strat = pd.read_csv(self.results_dir / "stratified_h0.csv")
        trgb = pd.read_csv(self.results_dir / "trgb_hosts_data.csv")

        # --- 1. Cepheid Host-Mass Partial Correlation ---
        print_status("Cepheid: Host-mass partial correlation", "SECTION")

        # Raw correlation
        r_raw, p_raw = stats.pearsonr(strat["sigma_inferred"], strat["h0_derived"])
        print_status(f"Raw: r={r_raw:.3f}, p={p_raw:.4f}", "INFO")

        # Partial correlation controlling for host_logmass
        if "host_logmass" in strat.columns and strat["host_logmass"].notna().sum() > 5:
            valid = strat.dropna(subset=["host_logmass", "sigma_inferred", "h0_derived"])

            # Residuals after regressing out logmass
            slope_m, intercept_m, _, _, _ = stats.linregress(
                valid["host_logmass"], valid["h0_derived"]
            )
            h0_residual = valid["h0_derived"] - (slope_m * valid["host_logmass"] + intercept_m)

            r_part, p_part = stats.pearsonr(valid["sigma_inferred"], h0_residual)
            print_status(
                f"Mass-residual: r={r_part:.3f}, p={p_part:.4f} (N={len(valid)})",
                "INFO",
            )

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
            r_part = p_part = r_both = p_both = np.nan
            print_status("host_logmass missing or insufficient; skipping", "WARNING")

        # --- 2. TRGB Host-Mass Partial Correlation ---
        print_status("TRGB: Host-mass partial correlation", "SECTION")

        # Merge TRGB with stratified to get host_logmass
        trgb["match"] = trgb["galaxy"].str.replace(" ", "").str.upper()
        strat["match"] = strat["normalized_name"].str.replace(" ", "").str.upper()
        merged = pd.merge(
            trgb, strat[["match", "host_logmass"]], on="match", how="inner"
        )

        if len(merged) > 3 and "host_logmass" in merged.columns:
            r_trgb_raw, p_trgb_raw = stats.pearsonr(
                merged["sigma_inferred"], merged["h0_trgb"]
            )
            print_status(f"TRGB raw: r={r_trgb_raw:.3f}, p={p_trgb_raw:.4f}", "INFO")

            valid_trgb = merged.dropna(subset=["host_logmass", "sigma_inferred", "h0_trgb"])
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
                print_status(
                    f"TRGB mass-residual: r={r_trgb_part:.3f}, p={p_trgb_part:.4f} (N={len(valid_trgb)})",
                    "INFO",
                )
            else:
                r_trgb_part = p_trgb_part = np.nan
        else:
            r_trgb_raw = p_trgb_raw = r_trgb_part = p_trgb_part = np.nan
            print_status("TRGB mass data insufficient; skipping", "WARNING")

        # --- 3. Summary ---
        print_status("--- SUMMARY ---", "SECTION")
        headers = ["Channel", "Raw r", "Raw p", "Mass-residual r", "Mass-residual p"]
        rows = [
            ["Cepheid", f"{r_raw:.3f}", f"{p_raw:.4f}",
             f"{r_part:.3f}" if np.isfinite(r_part) else "N/A",
             f"{p_part:.4f}" if np.isfinite(p_part) else "N/A"],
            ["TRGB", f"{r_trgb_raw:.3f}" if np.isfinite(r_trgb_raw) else "N/A",
             f"{p_trgb_raw:.4f}" if np.isfinite(p_trgb_raw) else "N/A",
             f"{r_trgb_part:.3f}" if np.isfinite(r_trgb_part) else "N/A",
             f"{p_trgb_part:.4f}" if np.isfinite(p_trgb_part) else "N/A"],
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
            },
            "trgb": {
                "raw_r": float(r_trgb_raw) if np.isfinite(r_trgb_raw) else None,
                "raw_p": float(p_trgb_raw) if np.isfinite(p_trgb_raw) else None,
                "mass_residual_r": float(r_trgb_part) if np.isfinite(r_trgb_part) else None,
                "mass_residual_p": float(p_trgb_part) if np.isfinite(p_trgb_part) else None,
            },
            "tep_prediction": (
                "If TEP is real: Cepheid mass-residual r should remain significant; "
                "TRGB mass-residual r should collapse toward zero."
            ),
        }

        with open(self.results_dir / "host_mass_residual_test.json", "w") as f:
            json.dump(results, f, indent=2)

        print_status("Saved results to host_mass_residual_test.json", "SUCCESS")
        print_status("Step 16 complete", "SUCCESS")


if __name__ == "__main__":
    Step16HostMassResidual().run()
