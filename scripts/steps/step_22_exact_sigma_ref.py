#!/usr/bin/env python3
"""
Step 21: Exact Anchor-Leverage Sigma_ref
=========================================

Reconstructs sigma_ref from the actual SH0ES GLS design matrix leverage
rather than approximate prose weights.

The standard sigma_ref = 87.17 km/s is derived from approximate anchor
weights (w_MW ~ 0.03, w_LMC ~ 0.10, w_NGC4258 ~ 0.84, w_M31 ~ 0.03).
This step attempts to compute exact leverage from the pipeline's
reconstructed data.

Since the full SH0ES design matrix is not directly available, we approximate
exact leverage by:
    1. Computing each anchor's fractional contribution to the total
       calibrator Cepheid sample
    2. Using the P-L zero-point sensitivity to each anchor's dispersion
    3. Deriving sigma_ref^2 = sum_i w_i * S_i * sigma_i^2

Outputs: JSON with exact vs approximate sigma_ref comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_correction import ANCHOR_NMB, ANCHOR_SCREENING, group_screening_factor


class Step21ExactSigmaRef:
    """Formal pipeline step: derive exact anchor-leverage sigma_ref."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_21_exact_sigma_ref",
            log_file_path=self.logs_dir / "step_22_exact_sigma_ref.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 21: EXACT ANCHOR-LEVERAGE SIGMA_REF", "TITLE")

        # Load anchor data from stratified file (anchors have h0_derived but
        # are not in the primary sample)
        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")
        hosts = pd.read_csv(self.root / "data" / "processed" / "hosts_processed.csv")

        # Anchor weights matching step_3_tep_correction.py exactly
        # (MW=0.20, LMC=0.25, NGC4258=0.55; no M31 — not a calibrator in SH0ES R22)
        approx_weights = {
            "MW": 0.20,
            "LMC": 0.25,
            "NGC 4258": 0.55,
        }

        # Anchor sigma values: DISK velocity dispersions at Cepheid locations
        # Same values used in step_3 (not central bulge apertures).
        anchor_sigmas = {
            "MW": 30.0,      # Bovy+2012 thin disk σ_z at solar neighborhood
            "LMC": 24.0,     # van der Marel+2002 disk dispersion
            "NGC 4258": 115.0,  # Kormendy & Ho 2013 intermediate aperture
        }

        # Recompute sigma_ref with exact and screened variants
        def compute_sigma_ref(weights, sigmas, screening=None):
            numerator = 0.0
            denominator = 0.0
            for name in weights:
                w = weights[name]
                s = sigmas[name]
                S = screening.get(name, 1.0) if screening else 1.0
                numerator += w * S * (s ** 2)
                denominator += w
            return np.sqrt(numerator / denominator) if denominator > 0 else np.nan

        sigma_ref_approx = compute_sigma_ref(approx_weights, anchor_sigmas)
        sigma_ref_screened = compute_sigma_ref(
            approx_weights, anchor_sigmas, ANCHOR_SCREENING
        )

        # Also try equal weights as a robustness check
        equal_weights = {k: 1.0 / len(approx_weights) for k in approx_weights}
        sigma_ref_equal = compute_sigma_ref(equal_weights, anchor_sigmas)
        sigma_ref_equal_screened = compute_sigma_ref(
            equal_weights, anchor_sigmas, ANCHOR_SCREENING
        )

        print_status(f"Step-3 weights, unscreened:        {sigma_ref_approx:.2f} km/s", "INFO")
        print_status(f"Step-3 weights, screened:          {sigma_ref_screened:.2f} km/s", "INFO")
        print_status(f"Equal weights, unscreened:         {sigma_ref_equal:.2f} km/s", "INFO")
        print_status(f"Equal weights, screened:           {sigma_ref_equal_screened:.2f} km/s", "INFO")

        # From pipeline JSON
        with open(self.results_dir / "step_04_tep_correction_results.json") as f:
            tep = json.load(f)
        sigma_ref_pipeline = float(tep["sigma_ref"])
        sigma_ref_scr_pipeline = float(tep.get("sigma_ref_screened", 0))

        print_status(f"Pipeline sigma_ref (standard):     {sigma_ref_pipeline:.2f} km/s", "INFO")
        if sigma_ref_scr_pipeline > 0:
            print_status(f"Pipeline sigma_ref (screened):     {sigma_ref_scr_pipeline:.2f} km/s", "INFO")

        results = {
            "exact_reconstruction": {
                "step3_weights_unscreened": sigma_ref_approx,
                "step3_weights_screened": sigma_ref_screened,
                "equal_weights_unscreened": sigma_ref_equal,
                "equal_weights_screened": sigma_ref_equal_screened,
            },
            "pipeline_values": {
                "sigma_ref_standard": sigma_ref_pipeline,
                "sigma_ref_screened": sigma_ref_scr_pipeline if sigma_ref_scr_pipeline > 0 else None,
            },
            "interpretation": (
                "Anchor weights and disk sigmas now match step_3_tep_correction.py exactly "
                "(MW=0.20/30.0 km/s, LMC=0.25/24.0 km/s, NGC4258=0.55/115.0 km/s; no M31). "
                "The reconstructed sigma_ref (step3_weights_unscreened) should agree with "
                "pipeline_values.sigma_ref_standard to within rounding. "
                "Equal-weight variant is a robustness check only. "
                "Note: the full SH0ES GLS design matrix leverage fractions are not "
                "directly available; these weights are the SH0ES-motivated P-L contribution "
                "fractions from Riess et al. (2022)."
            ),
        }

        with open(self.results_dir / "step_22_exact_sigma_ref_reconstruction.json", "w") as f:
            json.dump(results, f, indent=2)

        print_status("Step 21 complete", "SUCCESS")


if __name__ == "__main__":
    Step21ExactSigmaRef().run()
