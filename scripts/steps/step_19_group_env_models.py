#!/usr/bin/env python3
"""
Step 18: Group Environment Model Comparison
============================================

Tests four competing models for the H0–sigma relationship:

    1. Baseline:     delta_mu = a + b * sigma^2
    2. Confound:     delta_mu = a + b * sigma^2 + c * N_mb
    3. TEP-local:    delta_mu = a + b * [S_local(rho) * sigma^2]
    4. TEP-full:     delta_mu = a + b * [S_local(rho) * S_group(N_mb) * sigma^2]

If TEP is real, Model 4 should show the strongest signal and lowest scatter.
If group richness is merely a confound, Model 2 should be preferred.

Also tests whether group environment weakens or strengthens the signal:
    - If N_mb is a confound: adding it should weaken significance.
    - If N_mb is a TEP screening mechanism: using S_group(N_mb) should
      strengthen or restore significance.

Outputs: JSON with model comparison (AIC, BIC, scatter, slope significance).
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
from scripts.utils.tep_correction import group_screening_factor


class Step18GroupEnvModels:
    """Formal pipeline step: compare group environment models head-to-head."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_18_group_env",
            log_file_path=self.logs_dir / "step_19_group_env_models.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 18: Group environment model comparison", "TITLE")

        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")
        n = len(strat)

        sigma = strat["sigma_inferred"].values
        h0 = strat["h0_derived"].values
        mu = strat["value"].values
        S_local = strat["shear_suppression"].values
        n_mb = strat["tully_nmb"].fillna(1.0).values
        S_group = np.array([group_screening_factor(x) for x in n_mb])
        S_total = S_local * S_group

        # Response: delta_mu from mean
        mu_mean = np.mean(mu)
        delta_mu = mu - mu_mean

        # Also test with H0 as response
        h0_mean = np.mean(h0)
        delta_h0 = h0 - h0_mean

        models = {
            "baseline_sigma2": {
                "X": np.column_stack([np.ones(n), sigma ** 2]),
                "desc": "Baseline: sigma^2 only",
            },
            "confound_nmb": {
                "X": np.column_stack([np.ones(n), sigma ** 2, n_mb]),
                "desc": "Confound: sigma^2 + N_mb",
            },
            "tep_local": {
                "X": np.column_stack([np.ones(n), S_local * sigma ** 2]),
                "desc": "TEP-local: S_local * sigma^2",
            },
            "tep_full": {
                "X": np.column_stack([np.ones(n), S_total * sigma ** 2]),
                "desc": "TEP-full: S_local * S_group * sigma^2",
            },
            "tep_full_with_nmb": {
                "X": np.column_stack([np.ones(n), S_total * sigma ** 2, n_mb]),
                "desc": "TEP-full + N_mb residual",
            },
        }

        results = []
        for name, model in models.items():
            X = model["X"]
            # Fit H0 ~ X (use delta_h0 for centered response)
            valid = np.all(np.isfinite(X), axis=1) & np.isfinite(delta_h0)
            Xv = X[valid]
            yv = delta_h0[valid]
            nv = len(yv)

            if nv < Xv.shape[1] + 2:
                continue

            # OLS fit
            beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
            pred = Xv @ beta
            resid = yv - pred
            sse = np.sum(resid ** 2)
            sst = np.sum((yv - np.mean(yv)) ** 2)
            r2 = 1 - sse / sst if sst > 0 else 0
            mse = sse / max(nv - Xv.shape[1], 1)
            scatter = float(np.sqrt(mse))

            # t-stat for sigma^2 slope (second coefficient)
            # Standard error from covariance matrix
            cov = mse * np.linalg.inv(Xv.T @ Xv)
            se_beta = np.sqrt(np.diag(cov))
            t_stat = beta[1] / se_beta[1] if se_beta[1] > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), max(nv - Xv.shape[1], 1)))

            # AIC and BIC
            ll = -0.5 * nv * (np.log(2 * np.pi * mse) + 1)
            k = Xv.shape[1]
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(nv)

            results.append({
                "model": name,
                "description": model["desc"],
                "N": nv,
                "k_params": k,
                "r2": float(r2),
                "scatter_kms_mpc": scatter,
                "slope_sigma2": float(beta[1]),
                "slope_se": float(se_beta[1]),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "AIC": float(aic),
                "BIC": float(bic),
                "log_likelihood": float(ll),
            })

        df = pd.DataFrame(results)
        df = df.sort_values("BIC")

        print_status("Model comparison (sorted by BIC, lower is better):", "INFO")
        print_table(
            ["Model", "N", "k", "R²", "Scatter", "t-stat", "p", "AIC", "BIC"],
            [
                [
                    r["description"][:35],
                    str(r["N"]),
                    str(r["k_params"]),
                    f"{r['r2']:.3f}",
                    f"{r['scatter_kms_mpc']:.2f}",
                    f"{r['t_stat']:.2f}",
                    f"{r['p_value']:.4f}",
                    f"{r['AIC']:.1f}",
                    f"{r['BIC']:.1f}",
                ]
                for _, r in df.iterrows()
            ],
            title="Group Environment Model Comparison",
        )

        # Best model
        best = df.iloc[0]
        print_status(
            f"Best model: {best['description']} (BIC={best['BIC']:.1f}, "
            f"scatter={best['scatter_kms_mpc']:.2f}, p={best['p_value']:.4f})",
            "SUCCESS" if best["p_value"] < 0.05 else "WARNING",
        )

        # Save
        df.to_json(
            self.results_dir / "step_19_group_environment_model_comparison.json",
            orient="records",
            indent=2,
        )

        # Also save interpretation
        with open(self.results_dir / "step_19_group_env_model_interpretation.txt", "w") as f:
            f.write("Group environment model comparison\n")
            f.write("=" * 60 + "\n\n")
            f.write("TEP prediction: Model 'tep_full' should be preferred.\n")
            f.write("Confound prediction: Model 'confound_nmb' should be preferred.\n\n")
            for _, r in df.iterrows():
                f.write(f"{r['description']}\n")
                f.write(f"  BIC={r['BIC']:.1f}, scatter={r['scatter_kms_mpc']:.2f}, p={r['p_value']:.4f}\n\n")

        print_status("Step 18 complete", "SUCCESS")


if __name__ == "__main__":
    Step18GroupEnvModels().run()
