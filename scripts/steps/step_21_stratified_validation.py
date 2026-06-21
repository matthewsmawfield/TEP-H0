#!/usr/bin/env python3
"""
Step 20: Physically Stratified Validation
========================================

Train the TEP correction on one physical regime and test on another.
This is stronger than random 70/30 splits because it stresses the
physical axes of the model.

Train/Test regimes:
    1. Low-z (z < 0.005)  -> High-z (z > 0.005)
    2. High-z (z > 0.005) -> Low-z (z < 0.005)
    3. Stellar-σ hosts     -> HI-proxy hosts
    4. HI-proxy hosts      -> Stellar-σ hosts
    5. Isolated (S > 0.9)  -> Group (S < 0.9)
    6. Group (S < 0.9)     -> Isolated (S > 0.9)

For each split:
    - Fit kappa_Cep on train set (minimize slope of H0 vs sigma)
    - Apply to test set (no refitting)
    - Measure residual slope and scatter

If TEP is real, the correction should remove the trend in the held-out
physical regime.

Outputs: JSON with train/test performance for each split.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
from scripts.utils.tep_correction import C_SQUARED_KM_S, tep_correction


class Step20StratifiedValidation:
    """Formal pipeline step: physically stratified train/test validation."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_20_stratified_val",
            log_file_path=self.logs_dir / "step_21_stratified_validation.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 21: Physically stratified validation", "TITLE")

        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")
        prov = pd.read_csv(self.results_dir / "step_07_sigma_provenance_table.csv")

        with open(self.results_dir / "step_04_tep_correction_results.json") as f:
            tep = json.load(f)
        sigma_ref = float(tep["sigma_ref"])
        c2 = C_SQUARED_KM_S

        # Merge provenance for sigma method
        strat = strat.merge(
            prov[["normalized_name", "sigma_method"]], on="normalized_name", how="left"
        )

        sigma = strat["sigma_inferred"].values
        mu = strat["value"].values
        z = strat["z_hd"].values
        S = strat["shear_suppression"].values
        h0 = strat["h0_derived"].values

        n = len(strat)

        # Define splits
        is_lowz = z < 0.005
        is_highz = z >= 0.005
        is_stellar = strat["sigma_method"] == "stellar absorption"
        is_hi = strat["sigma_method"] == "HI linewidth proxy"
        is_isolated = S > 0.9
        is_group = S <= 0.9

        splits = [
            ("lowz_to_highz", is_lowz, is_highz),
            ("highz_to_lowz", is_highz, is_lowz),
            ("stellar_to_hi", is_stellar, is_hi),
            ("hi_to_stellar", is_hi, is_stellar),
            ("isolated_to_group", is_isolated, is_group),
            ("group_to_isolated", is_group, is_isolated),
        ]

        results = []
        for name, train_mask, test_mask in splits:
            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < 5 or n_test < 5:
                print_status(f"{name}: insufficient data (train={n_train}, test={n_test}), skipping", "INFO")
                continue

            # Fit kappa on train
            def fit_kappa(sigma_train, mu_train, z_train, S_train):
                def objective(k):
                    dmu = tep_correction(sigma_train, sigma_ref, k[0], S_train)
                    mu_c = mu_train + dmu
                    mu_fid = 5 * np.log10(299792.458 * z_train) + 25 - 5 * np.log10(70.0)
                    delta_mu = mu_c - mu_fid
                    if len(delta_mu) < 2:
                        return 1e10
                    slope, _ = np.polyfit(sigma_train, delta_mu, 1)
                    return slope ** 2

                res = minimize(objective, x0=[1.0e6], method="Nelder-Mead",
                               options={"xatol": 10.0, "fatol": 1e-6, "maxiter": 500})
                return res.x[0]

            kappa_train = fit_kappa(
                sigma[train_mask], mu[train_mask], z[train_mask], S[train_mask]
            )

            # Apply to test (no refitting)
            dmu_test = tep_correction(sigma[test_mask], sigma_ref, kappa_train, S[test_mask])
            mu_c_test = mu[test_mask] + dmu_test
            mu_fid_test = 5 * np.log10(299792.458 * z[test_mask]) + 25 - 5 * np.log10(70.0)
            delta_mu_test = mu_c_test - mu_fid_test

            # Test residual slope
            if len(delta_mu_test) > 2:
                slope_test, _, _, _, _ = stats.linregress(sigma[test_mask], delta_mu_test)
                r_test, p_test = stats.pearsonr(sigma[test_mask], delta_mu_test)
                scatter_test = float(np.std(delta_mu_test - np.mean(delta_mu_test)))
            else:
                slope_test = r_test = p_test = scatter_test = np.nan

            results.append({
                "split": name,
                "n_train": int(n_train),
                "n_test": int(n_test),
                "kappa_train": float(kappa_train),
                "test_residual_slope": float(slope_test),
                "test_r": float(r_test),
                "test_p": float(p_test),
                "test_scatter": float(scatter_test),
                "passes": bool(abs(r_test) < 0.3) if np.isfinite(r_test) else False,
            })

        df = pd.DataFrame(results)

        print_status("Physically stratified validation results:", "INFO")
        print_table(
            ["Split", "N_train", "N_test", "κ_train", "Test r", "Test p", "Criterion met?"],
            [
                [
                    r["split"],
                    str(r["n_train"]),
                    str(r["n_test"]),
                    f"{r['kappa_train']:.2e}",
                    f"{r['test_r']:.3f}",
                    f"{r['test_p']:.4f}",
                    "YES" if r["passes"] else "NO",
                ]
                for _, r in df.iterrows()
            ],
            title="Stratified Validation (Train -> Test)",
        )

        n_pass = sum(1 for r in results if r["passes"])
        print_status(f"Passed {n_pass}/{len(results)} stratified validation splits", "INFO")

        with open(self.results_dir / "step_21_stratified_validation.json", "w") as f:
            json.dump(results, f, indent=2)

        print_status("Step 21 complete", "SUCCESS")


if __name__ == "__main__":
    Step20StratifiedValidation().run()
