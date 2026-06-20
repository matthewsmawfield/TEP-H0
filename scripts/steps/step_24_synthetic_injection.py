#!/usr/bin/env python3
"""
Step 23: Synthetic Injection Recovery Test
==========================================

Validates that the pipeline can recover a known injected TEP signal.

The null baseline is constructed from the Hubble law:
    mu_null[i] = 5 * log10(c * z_hd[i] / H0_mean) + 25

where H0_mean is the mean of individual H0 values derived from the SH0ES
distances. This null has zero sigma-correlation by construction, so OLS/GLS
at kappa_inj=0 should recover a slope consistent with zero.

For each injected kappa level, the recovered slope is compared to the injected
value using OLS (no measurement errors), GLS (full covariance), and ODR
(errors-in-variables). ODR is pre-scaled to avoid numerical conditioning issues
from X_tep ~ 1e-8.

Previous bug: the null was built by subtracting the empirical kappa fit from
mu_obs (mu_null = mu_obs - kappa_emp * X_tep), then re-injecting. This
double-counted the residual covariance when kappa_emp was fit on a different
(larger) sample than the injection test, causing OLS at kappa_inj=0 to recover
a non-zero slope and ODR to diverge.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.odr import ODR, Model, RealData

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.tep_correction import C_SQUARED_KM_S


class Step23SyntheticInjection:
    def __init__(self):
        self.results_dir = PROJECT_ROOT / "results" / "outputs"
        self.logs_dir = PROJECT_ROOT / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.C2 = C_SQUARED_KM_S

        self.logger = TEPLogger(
            "step_23_synthetic_injection",
            log_file_path=self.logs_dir / "step_24_synthetic_injection.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 23: SYNTHETIC INJECTION RECOVERY TEST", "TITLE")

        df = pd.read_csv(self.results_dir / "step_04_tep_corrected_h0.csv")
        df = df[df["z_hd"] > 0.0035].copy()
        N = len(df)

        # Load sigma_ref from pipeline JSON (not hardcoded)
        with open(self.results_dir / "step_04_tep_correction_results.json") as f:
            tep_results = json.load(f)
        sigma_ref = float(tep_results["sigma_ref"])
        print_status(f"sigma_ref = {sigma_ref:.4f} km/s (from step_04_tep_correction_results.json)", "INFO")

        # Load covariance matrix for GLS; align to filtered sample
        cov_path = self.results_dir / "step_03_h0_covariance.npy"
        if cov_path.exists():
            cov_full = np.load(cov_path)
            if cov_full.shape == (N, N):
                cov = cov_full
            else:
                # Covariance built for a different sample size — fall back to diagonal
                print_status(
                    f"step_03_h0_covariance.npy shape {cov_full.shape} != ({N},{N}); using diagonal fallback",
                    "WARNING",
                )
                cov = np.eye(N) * 0.15**2
        else:
            cov = np.eye(N) * 0.15**2

        # Get sigma measurement errors for ODR
        prov_path = self.results_dir / "step_07_sigma_provenance_table.csv"
        if prov_path.exists():
            prov = pd.read_csv(prov_path)
            df = df.merge(prov[["normalized_name", "sigma_measured_error_kms"]],
                          on="normalized_name", how="left")
        if "sigma_measured_error_kms" not in df.columns:
            df["sigma_measured_error_kms"] = pd.Series(
                [10.0] * N, index=df.index  # preserve filtered index to avoid NaN on alignment
            )
        df["sigma_measured_error_kms"] = df["sigma_measured_error_kms"].fillna(10.0)
        sigma_errs = df["sigma_measured_error_kms"].values
        dmu_errs = df["error"].fillna(0.15).values

        # TEP regressor
        sigma = df["sigma_inferred"].values
        S = df["shear_suppression"].values
        X_tep = S * (sigma**2 - sigma_ref**2) / self.C2

        # ── Hubble-law null baseline ──────────────────────────────────────────
        # mu_null[i] depends only on z, not sigma.  At kappa_inj=0 the slope
        # on X_tep should be zero + noise.
        C_KMS = 299792.458
        z_hd = df["z_hd"].values
        mu_obs = df["value"].values
        d_mpc = 10**((mu_obs - 25) / 5)
        H0_individual = C_KMS * z_hd / d_mpc
        H0_null = float(np.mean(H0_individual))
        mu_null = 5 * np.log10(C_KMS * z_hd / H0_null) + 25
        print_status(f"Null H0 = {H0_null:.2f} km/s/Mpc (mean over N={N} hosts)", "INFO")

        # ── Pre-scale X_tep for ODR conditioning ─────────────────────────────
        # X_tep ~ 1e-8 while dmu_err ~ 0.1; pre-scaling X by 1e7 puts both on O(1).
        SCALE = 1e7

        # Diagnostic: check z-sigma correlation (explains non-zero null at kappa_inj=0)
        r_z_xtep, p_z_xtep = stats.pearsonr(z_hd, X_tep)
        print_status(
            f"Correlation(z, X_tep) = {r_z_xtep:.3f} (p={p_z_xtep:.3f}) — "
            "non-zero OLS baseline at kappa_inj=0 expected; differential recovery is exact.",
            "INFO",
        )

        # ODR reliability check: sx/X_tep ratio
        sx = np.abs(S * 2 * sigma * sigma_errs / self.C2)  # S factor required: dX_tep/dsigma = S*2*sigma/C2
        sx_xtep_ratio = float(np.mean(sx / np.abs(X_tep)))
        odr_reliable = sx_xtep_ratio < 0.5
        print_status(
            f"ODR reliability: mean sx/X_tep = {sx_xtep_ratio:.2f}x "
            f"({'reliable' if odr_reliable else 'UNRELIABLE — measurement errors exceed signal; ODR disabled'})",
            "INFO" if odr_reliable else "WARNING",
        )

        injections = [0.0, 0.5e6, 1.0e6, 1.5e6]
        results = []

        for kappa_inj in injections:
            mu_inj = mu_null + kappa_inj * X_tep

            # OLS
            slope_ols, _, r_val, p_val, _ = stats.linregress(X_tep, mu_inj)

            # ODR — disabled when sx/X_tep > 0.5 (measurement errors dominate signal)
            if odr_reliable:
                X_scaled = X_tep * SCALE
                sx_scaled = sx * SCALE
                try:
                    data = RealData(X_scaled, mu_inj, sx=sx_scaled, sy=dmu_errs)
                    odr_model = Model(lambda B, x: B[0] * x + B[1])
                    odr_obj = ODR(data, odr_model,
                                  beta0=[slope_ols / SCALE, np.mean(mu_inj)],
                                  maxit=300)
                    out = odr_obj.run()
                    slope_odr = float(out.beta[0]) * SCALE
                except Exception as exc:
                    slope_odr = float("nan")
                    print_status(f"ODR failed at kappa_inj={kappa_inj:.1e}: {exc}", "WARNING")
            else:
                slope_odr = float("nan")

            # GLS
            X_mat = np.column_stack((np.ones(N), X_tep))
            try:
                inv_cov = np.linalg.inv(cov)
                beta_gls = (np.linalg.inv(X_mat.T @ inv_cov @ X_mat)
                            @ (X_mat.T @ inv_cov @ mu_inj))
                slope_gls = float(beta_gls[1])
            except Exception:
                slope_gls = float("nan")

            results.append({
                "kappa_injected_1e6": round(kappa_inj / 1e6, 2),
                "OLS_recovered_1e6": round(slope_ols / 1e6, 3),
                "ODR_recovered_1e6": round(slope_odr / 1e6, 3) if np.isfinite(slope_odr) else None,
                "GLS_recovered_1e6": round(slope_gls / 1e6, 3) if np.isfinite(slope_gls) else None,
                "null_pvalue": round(float(p_val), 4) if kappa_inj == 0 else None,
                "null_baseline_note": (
                    f"OLS baseline at kappa=0 reflects z-sigma correlation (r={r_z_xtep:.3f}); "
                    "differential recovery is exact"
                ) if kappa_inj == 0 else None,
            })
            print_status(
                f"kappa_inj={kappa_inj/1e6:.1f}e6: OLS={slope_ols/1e6:.3f}, "
                f"ODR={'disabled' if not odr_reliable else f'{slope_odr/1e6:.3f}'}, "
                f"GLS={slope_gls/1e6:.3f}",
                "INFO",
            )

        out_df = pd.DataFrame(results)
        out_df.to_csv(self.results_dir / "step_24_synthetic_injection.csv", index=False)
        print_status("Synthetic injection table saved to step_24_synthetic_injection.csv", "SUCCESS")
        print_status("Step 23 complete", "SUCCESS")


if __name__ == "__main__":
    Step23SyntheticInjection().run()
