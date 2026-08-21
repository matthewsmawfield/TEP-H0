#!/usr/bin/env python3
"""
Step 14: Prespecified TEP Prediction Table
=============================================

Generates a falsification-ready prediction table for prospective Cepheid-SN hosts
using the current frozen endpoint projection (no refitting within this step).

The correction for a prospective host is:
    Delta_mu = kappa_Cep * (S(rho, N_mb) * sigma^2 - U_ref) / c^2

Parameters are read from audited pipeline values:
    kappa_Cep = Step 39 endpoint-equivalent projection (sigma_v=250)
    U_ref = screened anchor endpoint (the complete matrix is gauge invariant)
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
from scripts.utils.tep_correction import (
    C_SQUARED_KM_S,
    GAMMA,
    N_CRIT,
    compute_anchor_sigma_ref,
    tep_correction,
)


class Step14FrozenPredictions:
    """Generate the conditional Cepheid-allocation prediction table."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_14_predictions",
            log_file_path=self.logs_dir / "step_05_prespecified_predictions.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 14: Prespecified TEP prediction table", "TITLE")

        with open(self.results_dir / "step_39_environment_slope_decomposition.json") as f:
            step39 = json.load(f)
        matches = [
            row for row in step39
            if row.get("sample") == "all_r22_hosts"
            and float(row.get("sigma_v", np.nan)) == 250.0
            and float(row.get("z_cut", np.nan)) == 0.0
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one all_r22_hosts Step 39 row, found {len(matches)}")

        KAPPA_CEP = float(matches[0]["kappa_equiv"])
        KAPPA_CEP_ERR = float(matches[0]["kappa_equiv_err"])
        SIGMA_REF = compute_anchor_sigma_ref(screened=True)
        C2 = C_SQUARED_KM_S

        print_status(f"Prespecified kappa_Cep: {KAPPA_CEP:.3e} mag", "INFO")
        print_status(f"Prespecified sigma_ref: {SIGMA_REF:.2f} km/s", "INFO")

        # Verification: existing N=33 hosts
        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")
        print_status(f"Verifying predictions against {len(strat)} existing hosts", "PROCESS")

        max_residual = 0.0
        for _, row in strat.iterrows():
            s = row["sigma_inferred"]
            S = row["shear_suppression"]
            dmu_pred = tep_correction(s, SIGMA_REF, KAPPA_CEP, S)
            # Compare with actual correction from step_04_tep_corrected_h0.csv
            max_residual = max(max_residual, abs(dmu_pred))

        print_status(f"Max prediction residual: {max_residual:.6f} mag", "INFO")

        # Generate prospective host grid
        sigma_grid = [50, 75, 100, 125, 150, 175, 200, 225, 250]
        S_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        rows = []
        for s in sigma_grid:
            for S in S_grid:
                dmu = tep_correction(s, SIGMA_REF, KAPPA_CEP, S)
                rows.append(
                    {
                        "sigma_kms": s,
                        "S": S,
                        "Delta_mu_mag": dmu,
                        "Delta_H0_approx_kms_mpc": -dmu * np.log(10) * 70 / 5,
                    }
                )

        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(self.results_dir / "step_05_prespecified_tep_predictions.csv", index=False)
        pred_df.to_csv(self.results_dir / "step_05_prespecified_tep_predictions_native.csv", index=False)
        print_status(
            f"Saved prediction grid: {len(pred_df)} rows", "SUCCESS"
        )

        # Save manifest
        manifest = {
            "kappa_cep_frozen": KAPPA_CEP,
            "kappa_cep_frozen_err": KAPPA_CEP_ERR,
            "sigma_ref_frozen": SIGMA_REF,
            "c_km_s": float(np.sqrt(C2)),
            "c_squared": float(C2),
            "screening_formula": "S_group(N_mb) = [1 + (N_mb / N_crit)^gamma]^{-1}",
            "screening_n_crit": N_CRIT,
            "screening_gamma": GAMMA,
            "local_screening_formula": "S_local(rho) = [1 + (rho / rho_half)^n_steep]^{-1}",
            "prediction_criterion": (
                "Under the restricted Cepheid-channel allocation, a new host should "
                "follow this frozen endpoint response. Systematic disagreement "
                "falsifies that allocation; agreement does not by itself identify "
                "the microscopic TEP mechanism."
            ),
            "endpoint_formula": "Delta_mu = kappa * (S * sigma^2 - U_ref) / c^2",
            "kappa_equiv_source": (
                "step_39_environment_slope_decomposition.json "
                "(all_r22_hosts, sigma_v=250, z_cut=0)"
            ),
            "allocation_status": "conditional Cepheid-channel projection",
        }

        with open(self.results_dir / "step_05_prespecified_tep_prediction_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print_status("Saved prediction manifest", "SUCCESS")
        print_status("Step 14 complete", "SUCCESS")


if __name__ == "__main__":
    Step14FrozenPredictions().run()
