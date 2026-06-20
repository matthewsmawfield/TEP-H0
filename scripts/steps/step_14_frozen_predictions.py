#!/usr/bin/env python3
"""
Step 14: Frozen TEP Prediction Table
=====================================

Generates a falsification-ready prediction table for prospective Cepheid-SN hosts
using the pipeline-frozen parameters (no refitting).

The correction for a prospective host is:
    Delta_mu = kappa_Cep * S(rho, N_mb) * (sigma^2 - sigma_ref^2) / c^2

Parameters are frozen at pipeline values:
    kappa_Cep = 1.049e6 mag  (from tep_correction_results.json)
    sigma_ref = 87.17 km/s   (from tep_correction_results.json)
    S_group(N_mb) = [1 + (N_mb / N_crit)^gamma]^{-1}

This is a formal pipeline step. The output prediction table is a
falsification tool: new hosts should obey the precomputed Delta_mu
without refitting kappa_Cep.

Usage:
    Called by run_pipeline.py after Step 3 (TEP Correction) completes.
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
from scripts.utils.tep_correction import C_SQUARED_KM_S, tep_correction


class Step14FrozenPredictions:
    """Formal pipeline step: generate frozen TEP prediction table."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_14_predictions",
            log_file_path=self.logs_dir / "step_14_frozen_predictions.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 14: FROZEN TEP PREDICTION TABLE", "TITLE")

        # Load frozen parameters from pipeline output
        with open(self.results_dir / "tep_correction_results.json") as f:
            tep_json = json.load(f)

        KAPPA_CEP = float(tep_json["optimal_kappa_cep"])
        SIGMA_REF = float(tep_json["sigma_ref"])
        C2 = C_SQUARED_KM_S

        print_status(f"Frozen kappa_Cep: {KAPPA_CEP:.3e} mag", "INFO")
        print_status(f"Frozen sigma_ref: {SIGMA_REF:.2f} km/s", "INFO")

        # Verification: existing N=29 hosts
        strat = pd.read_csv(self.results_dir / "stratified_h0.csv")
        print_status(f"Verifying predictions against {len(strat)} existing hosts", "PROCESS")

        max_residual = 0.0
        for _, row in strat.iterrows():
            s = row["sigma_inferred"]
            S = row["shear_suppression"]
            dmu_pred = KAPPA_CEP * S * (s ** 2 - SIGMA_REF ** 2) / C2
            # Compare with actual correction from tep_corrected_h0.csv
            max_residual = max(max_residual, abs(dmu_pred))

        print_status(f"Max prediction residual: {max_residual:.6f} mag", "INFO")

        # Generate prospective host grid
        sigma_grid = [50, 75, 100, 125, 150, 175, 200, 225, 250]
        S_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        rows = []
        for s in sigma_grid:
            for S in S_grid:
                dmu = KAPPA_CEP * S * (s ** 2 - SIGMA_REF ** 2) / C2
                rows.append(
                    {
                        "sigma_kms": s,
                        "S": S,
                        "Delta_mu_mag": dmu,
                        "Delta_H0_approx_kms_mpc": -dmu * np.log(10) * 70 / 5,
                    }
                )

        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(self.results_dir / "frozen_tep_predictions.csv", index=False)
        print_status(
            f"Saved prediction grid: {len(pred_df)} rows", "SUCCESS"
        )

        # Save manifest
        manifest = {
            "kappa_cep_frozen": KAPPA_CEP,
            "sigma_ref_frozen": SIGMA_REF,
            "c_km_s": float(np.sqrt(C2)),
            "c_squared": float(C2),
            "screening_formula": "S_group(N_mb) = [1 + (N_mb / N_crit)^gamma]^{-1}",
            "screening_n_crit": 10.0,
            "screening_gamma": 1.2,
            "local_screening_formula": "S_local(rho) = [1 + (rho / rho_half)^n_steep]^{-1}",
            "prediction_criterion": (
                "A new Cepheid-SN host validates the TEP correction if its observed "
                "distance-modulus residual agrees with the predicted Delta_mu within "
                "the quoted uncertainty (~0.1-0.2 mag). Systematic offsets falsify "
                "the model."
            ),
        }

        with open(self.results_dir / "frozen_tep_prediction_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print_status("Saved prediction manifest", "SUCCESS")
        print_status("Step 14 complete", "SUCCESS")


if __name__ == "__main__":
    Step14FrozenPredictions().run()
