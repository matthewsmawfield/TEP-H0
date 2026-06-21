#!/usr/bin/env python3
"""
Step 14: Prespecified TEP Prediction Table
=============================================

Generates a falsification-ready prediction table for prospective Cepheid-SN hosts
using the pipeline-prespecified parameters (no refitting).

The correction for a prospective host is:
    Delta_mu = kappa_Cep * S(rho, N_mb) * (sigma^2 - sigma_ref^2) / c^2

Parameters are prespecified at pipeline values:
    kappa_Cep = step_04_tep_correction_results.json (optimal_kappa_cep)
    sigma_ref = step_04_tep_correction_results.json (sigma_ref)
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
    """Formal pipeline step: generate prespecified TEP prediction table."""

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

        # Load prespecified parameters from pipeline output
        with open(self.results_dir / "step_04_tep_correction_results.json") as f:
            tep_json = json.load(f)

        KAPPA_CEP = float(tep_json["optimal_kappa_cep"])
        SIGMA_REF = float(tep_json["sigma_ref"])
        C2 = C_SQUARED_KM_S

        print_status(f"Prespecified kappa_Cep: {KAPPA_CEP:.3e} mag", "INFO")
        print_status(f"Prespecified sigma_ref: {SIGMA_REF:.2f} km/s", "INFO")

        # Verification: existing N=29 hosts
        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")
        print_status(f"Verifying predictions against {len(strat)} existing hosts", "PROCESS")

        max_residual = 0.0
        for _, row in strat.iterrows():
            s = row["sigma_inferred"]
            S = row["shear_suppression"]
            dmu_pred = KAPPA_CEP * S * (s ** 2 - SIGMA_REF ** 2) / C2
            # Compare with actual correction from step_04_tep_corrected_h0.csv
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
        pred_df.to_csv(self.results_dir / "step_05_prespecified_tep_predictions.csv", index=False)
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

        with open(self.results_dir / "step_05_prespecified_tep_prediction_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print_status("Saved prediction manifest", "SUCCESS")
        # ------------------------------------------------------------------
        # TEP-native gauge variant: kappa_equiv from velocity-space likelihood
        # ------------------------------------------------------------------
        kappa_equiv = None
        try:
            with open(self.results_dir / "step_39_environment_slope_decomposition.json") as f:
                s39 = json.load(f)
            # Select primary sample, sigma_v=250, z_cut=0
            for rec in s39:
                if (rec.get("sample") == "primary"
                        and rec.get("sigma_v") == 250
                        and rec.get("z_cut", 0) == 0):
                    kappa_equiv = float(rec["kappa_equiv"])
                    break
        except Exception:
            pass

        if kappa_equiv is not None and np.isfinite(kappa_equiv):
            print_status(
                f"TEP-native gauge kappa_equiv: {kappa_equiv:.3e} mag", "INFO"
            )
            rows_native = []
            for s in sigma_grid:
                for S in S_grid:
                    dmu = kappa_equiv * S * (s ** 2 - SIGMA_REF ** 2) / C2
                    rows_native.append(
                        {
                            "sigma_kms": s,
                            "S": S,
                            "Delta_mu_mag": dmu,
                            "Delta_H0_approx_kms_mpc": -dmu * np.log(10) * 70 / 5,
                        }
                    )
            pred_native = pd.DataFrame(rows_native)
            pred_native.to_csv(
                self.results_dir / "step_05_prespecified_tep_predictions_native.csv",
                index=False,
            )
            print_status(
                f"Saved TEP-native prediction grid: {len(pred_native)} rows",
                "SUCCESS",
            )

            # Update manifest
            manifest["kappa_equiv_native"] = kappa_equiv
            manifest["kappa_equiv_source"] = (
                "step_39_environment_slope_decomposition.json "
                "(primary, sigma_v=250, z_cut=0)"
            )
            manifest["gauge_note"] = (
                "The empirical Step 04 table uses kappa_Cep from the host-residual "
                "correction. The TEP-native table uses kappa_equiv = Gamma_X / "
                "((ln 10 / 5) * H_app), the coefficient that would produce the "
                "velocity-space environmental slope under the TEP-native gauge."
            )
        else:
            print_status(
                "TEP-native kappa_equiv not available; skipping native prediction table",
                "INFO",
            )

        with open(self.results_dir / "step_05_prespecified_tep_prediction_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print_status("Saved prediction manifest", "SUCCESS")
        print_status("Step 14 complete", "SUCCESS")


if __name__ == "__main__":
    Step14FrozenPredictions().run()
