#!/usr/bin/env python3
"""
Step 22: SN Ia Downstream Residual Test
========================================

After applying the TEP correction to Cepheid distances, the corrected
SNe Ia should show no residual dependence on host sigma.

This step:
    1. Loads corrected H0 values from tep_corrected_h0.csv
    2. Tests whether corrected H0 still correlates with sigma
    3. Tests whether corrected SN Ia residuals correlate with sigma
    4. Compares raw vs corrected scatter

If TEP is real, the correction should remove the sigma dependence in
both Cepheid and downstream SN Ia calibrations.

Outputs: JSON with raw vs corrected correlation and scatter.
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


class Step22SNResidualTest:
    """Formal pipeline step: test downstream SN Ia residual dependence on sigma."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_22_sn_residual",
            log_file_path=self.logs_dir / "step_22_sn_residual_test.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 22: SN IA DOWNSTREAM RESIDUAL TEST", "TITLE"
        )

        # Load corrected data
        corrected = pd.read_csv(self.results_dir / "tep_corrected_h0.csv")
        raw = pd.read_csv(self.results_dir / "stratified_h0.csv")

        # Ensure same hosts
        raw = raw[raw["normalized_name"].isin(corrected["normalized_name"])]

        sigma = raw["sigma_inferred"].values
        h0_raw = raw["h0_derived"].values
        h0_corr = corrected["h0_corrected"].values

        # Raw correlation
        r_raw, p_raw = stats.pearsonr(sigma, h0_raw)
        rho_raw, prho_raw = stats.spearmanr(sigma, h0_raw)

        # Corrected correlation
        r_corr, p_corr = stats.pearsonr(sigma, h0_corr)
        rho_corr, prho_corr = stats.spearmanr(sigma, h0_corr)

        # Scatter
        scatter_raw = float(np.std(h0_raw - np.mean(h0_raw)))
        scatter_corr = float(np.std(h0_corr - np.mean(h0_corr)))

        print_status("Raw vs corrected comparison:", "INFO")
        headers = ["Metric", "Raw", "Corrected", "Improvement"]
        rows = [
            ["Pearson r", f"{r_raw:.3f}", f"{r_corr:.3f}", f"{abs(r_raw) - abs(r_corr):+.3f}"],
            ["Pearson p", f"{p_raw:.4f}", f"{p_corr:.4f}", f"{p_corr - p_raw:+.4f}"],
            ["Spearman ρ", f"{rho_raw:.3f}", f"{rho_corr:.3f}", f"{abs(rho_raw) - abs(rho_corr):+.3f}"],
            ["Scatter (km/s/Mpc)", f"{scatter_raw:.2f}", f"{scatter_corr:.2f}", f"{scatter_raw - scatter_corr:+.2f}"],
        ]
        print_table(headers, rows)

        # Also test residual = corrected - mean
        residual = h0_corr - np.mean(h0_corr)
        r_resid, p_resid = stats.pearsonr(sigma, residual)

        print_status(
            f"Corrected residual vs sigma: r={r_resid:.3f}, p={p_resid:.4f}",
            "SUCCESS" if abs(r_resid) < 0.2 else "WARNING",
        )

        results = {
            "raw": {
                "pearson_r": float(r_raw),
                "pearson_p": float(p_raw),
                "spearman_rho": float(rho_raw),
                "spearman_p": float(prho_raw),
                "scatter": float(scatter_raw),
            },
            "corrected": {
                "pearson_r": float(r_corr),
                "pearson_p": float(p_corr),
                "spearman_rho": float(rho_corr),
                "spearman_p": float(prho_corr),
                "scatter": float(scatter_corr),
                "residual_r": float(r_resid),
                "residual_p": float(p_resid),
            },
            "interpretation": (
                "If TEP is real, the corrected H0 should show no residual "
                "correlation with sigma (|r| < 0.2). The scatter should also "
                "decrease. If |r_corr| > |r_raw|, the correction overfit."
            ),
        }

        with open(self.results_dir / "sn_downstream_residual_test.json", "w") as f:
            json.dump(results, f, indent=2)

        print_status("Step 22 complete", "SUCCESS")


if __name__ == "__main__":
    Step22SNResidualTest().run()
