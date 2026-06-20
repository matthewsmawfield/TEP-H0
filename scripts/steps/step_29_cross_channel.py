
"""
Step 12: Cross-Channel Consistency Analysis
=============================================

The definitive TEP framework test is cross-channel consistency: different
astrophysical clocks (Cepheid, TRGB, SN Ia, pulsar) couple to the conformal
field with channel-specific strengths.  This step computes channel-specific
kappa values from all available data and tests whether the hierarchy matches
TEP predictions.

TEP Predictions:
- Cepheid (periodic pulsator):  kappa ~ 10^6 mag
- TRGB (non-periodic flash):    kappa << kappa_Cep (weaker or zero)
- SN Ia (explosive):            intermediate (not yet implemented)
- Pulsar (spin-down clock):     kappa_MSP ~ 10^4--10^5 (Paper 10)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
from scripts.utils.tep_correction import tep_correction, C_SQUARED_KM_S
from core.constants import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY


class Step12CrossChannel:
    """Cross-channel consistency: the definitive TEP framework test."""

    def __init__(self):
        self.root_dir = PROJECT_ROOT
        self.results_dir = self.root_dir / "results"
        self.outputs_dir = self.results_dir / "outputs"
        self.logs_dir = self.root_dir / "logs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_12_cross_channel",
            log_file_path=self.logs_dir / "step_12_cross_channel.log",
        )
        set_step_logger(self.logger)

        self.tep_json = self.outputs_dir / "tep_correction_results.json"
        self.trgb_csv = self.outputs_dir / "trgb_hosts_data.csv"
        self.stratified_csv = self.outputs_dir / "stratified_h0.csv"
        self.results_json = self.outputs_dir / "cross_channel_consistency.json"

    def _load_tep_results(self):
        with open(self.tep_json) as f:
            return json.load(f)

    def _slope_min_kappa(self, sigma, mu, z, S, sigma_ref, x0=1.0e6):
        """Optimise kappa to flatten H0 vs sigma slope."""
        if len(sigma) < 5:
            return np.nan, np.nan, np.nan

        def objective(k):
            dmu = tep_correction(sigma, sigma_ref, k[0], S)
            mu_c = mu + dmu
            mu_fid = 5 * np.log10(299792.458 * z) + 25 - 5 * np.log10(70.0)
            delta_mu = mu_c - mu_fid
            slope, _, _, _, _ = stats.linregress(sigma, delta_mu)
            return slope ** 2

        result = minimize(
            objective,
            x0=[x0],
            method="Nelder-Mead",
            options={"xatol": 10.0, "fatol": 1e-6, "maxiter": 500},
        )
        kappa_opt = float(result.x[0])

        dmu = tep_correction(sigma, sigma_ref, kappa_opt, S)
        mu_c = mu + dmu
        d_c = 10 ** ((mu_c - 25) / 5)
        h0 = 299792.458 * z / d_c
        unified_h0 = float(np.mean(h0))
        h0_sem = float(np.std(h0, ddof=1) / np.sqrt(len(h0)))
        return kappa_opt, unified_h0, h0_sem

    def _slope_min_kappa_robust(self, sigma, mu, z, S, sigma_ref, seeds=None):
        """Try multiple seeds and pick the best kappa."""
        if seeds is None:
            seeds = [-2e6, -1e6, -5e5, 0.0, 5e5, 1e6, 2e6, 3e6]
        best_k, best_obj = np.nan, np.inf
        for x0 in seeds:
            k, h0_u, h0_s = self._slope_min_kappa(sigma, mu, z, S, sigma_ref, x0=x0)
            if not np.isnan(k):
                dmu = tep_correction(sigma, sigma_ref, k, S)
                mu_c = mu + dmu
                mu_fid = 5 * np.log10(299792.458 * z) + 25 - 5 * np.log10(70.0)
                delta_mu = mu_c - mu_fid
                obj = stats.linregress(sigma, delta_mu)[0] ** 2
                if obj < best_obj:
                    best_obj = obj
                    best_k = k
        return best_k, best_obj

    def _bootstrap_kappa_err(self, sigma, mu, z, S, sigma_ref, kappa_best, n_boot=200):
        """Bootstrap standard error on slope-minimised kappa."""
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(sigma), size=len(sigma))
            k, _, _ = self._slope_min_kappa(
                sigma[idx], mu[idx], z[idx], S[idx], sigma_ref, x0=kappa_best
            )
            if not np.isnan(k):
                boot.append(k)
        boot = np.array(boot)
        if len(boot) < 10:
            return np.nan
        return float(np.std(boot))

    def _finite_diff_kappa(self, sigma, mu, z, S, sigma_ref):
        """Estimate kappa and error from finite-difference slope cancellation."""
        # Raw delta_mu slope
        mu_fid = 5 * np.log10(299792.458 * z) + 25 - 5 * np.log10(70.0)
        delta_mu_raw = mu - mu_fid
        slope_raw, _, _, _, stderr_raw = stats.linregress(sigma, delta_mu_raw)

        # Evaluate at two bracketing kappa values
        for k_test in [1e6, -1e6]:
            dmu = tep_correction(sigma, sigma_ref, k_test, S)
            mu_c = mu + dmu
            delta_mu_c = mu_c - mu_fid
            slope_c = stats.linregress(sigma, delta_mu_c)[0]
            if k_test == 1e6:
                slope_pos = slope_c
            else:
                slope_neg = slope_c

        ds_dk = (slope_pos - slope_neg) / 2e6
        if abs(ds_dk) < 1e-12:
            return np.nan, np.nan, float(slope_raw), float(stderr_raw)

        kappa_est = -slope_raw / ds_dk
        kappa_err = stderr_raw / abs(ds_dk)
        return float(kappa_est), float(kappa_err), float(slope_raw), float(stderr_raw)

    def run(self):
        print_status("=" * 60, "SECTION")
        print_status("STEP 12: CROSS-CHANNEL CONSISTENCY", "SECTION")
        print_status("The definitive TEP framework test", "INFO")
        print_status("=" * 60, "SECTION")

        tep = self._load_tep_results()
        sigma_ref = tep["sigma_ref"]
        kappa_joint = float(tep["optimal_kappa_cep"])
        kappa_joint_err = float(tep.get("bootstrap_kappa_robust_std", 8.9e5))
        unified_h0 = float(tep["unified_h0"])

        # ============================================================
        # 1. CEPHEID HOST-ONLY (re-optimise on stratified hosts)
        # ============================================================
        df_hosts = pd.read_csv(self.stratified_csv)
        sigma_h = df_hosts["sigma_inferred"].values
        mu_h = df_hosts["value"].values
        z_h = df_hosts["z_hd"].values
        S_h = (
            df_hosts["shear_suppression"].values
            if "shear_suppression" in df_hosts.columns
            else np.ones(len(df_hosts))
        )

        kappa_host, _, _ = self._slope_min_kappa(sigma_h, mu_h, z_h, S_h, sigma_ref)
        kappa_host_err = self._bootstrap_kappa_err(
            sigma_h, mu_h, z_h, S_h, sigma_ref, kappa_host
        )
        print_status(
            f"Cepheid host-only: kappa = {kappa_host:.3e} +/- {kappa_host_err:.3e}",
            "INFO",
        )

        # ============================================================
        # 2. TRGB CHANNEL (finite-difference estimate; slope-min unstable)
        # ============================================================
        df_trgb = pd.read_csv(self.trgb_csv)
        sigma_t = df_trgb["sigma_inferred"].values.astype(float)
        mu_t = df_trgb["mu_trgb"].values.astype(float)
        z_t = df_trgb["z_hd"].values.astype(float)
        S_t = np.ones(len(df_trgb))

        kappa_trgb, kappa_trgb_err, raw_slope, raw_slope_err = self._finite_diff_kappa(
            sigma_t, mu_t, z_t, S_t, sigma_ref
        )
        raw_slope_t = raw_slope / raw_slope_err if raw_slope_err > 0 else np.nan

        print_status(
            f"TRGB:  kappa = {kappa_trgb:.3e} +/- {kappa_trgb_err:.3e} (N={len(df_trgb)})",
            "INFO",
        )
        print_status(f"TRGB raw delta_mu-sigma slope: {raw_slope:.4f} +/- {raw_slope_err:.4f} (t={raw_slope_t:.2f})", "INFO")

        # ============================================================
        # 3. DIFFERENTIAL REGRESSION (matched hosts, redshift-independent)
        # ============================================================
        df_trgb_m = df_trgb.copy()
        df_hosts_m = df_hosts.copy()
        df_trgb_m["match"] = df_trgb_m["galaxy"].str.replace(" ", "").str.upper()
        df_hosts_m["match"] = df_hosts_m["normalized_name"].str.replace(" ", "").str.upper()
        merged = pd.merge(df_trgb_m, df_hosts_m, on="match", suffixes=("_trgb", "_host"))

        R_m = (
            merged["shear_suppression"]
            * (merged["sigma_inferred_trgb"] ** 2 - sigma_ref ** 2)
            / C_SQUARED_KM_S
        )
        delta_mu = merged["mu_trgb"] - merged["value"]

        slope_d, intercept_d, r_d, p_d, stderr_d = stats.linregress(R_m, delta_mu)
        kappa_diff = float(slope_d)
        kappa_diff_err = float(stderr_d)

        print_status(
            f"Differential: kappa_diff = {kappa_diff:.3e} +/- {kappa_diff_err:.3e} (N={len(merged)})",
            "INFO",
        )

        # ============================================================
        # 4. CONSISTENCY TESTS
        # ============================================================
        kappa_gal = KAPPA_GAL
        kappa_gal_err = KAPPA_GAL_UNCERTAINTY

        # Tensions
        tension_diff_vs_cep = (
            (kappa_diff - kappa_host) / np.sqrt(kappa_diff_err ** 2 + kappa_host_err ** 2)
            if kappa_diff_err > 0 and kappa_host_err > 0
            else np.nan
        )
        tension_diff_vs_zero = kappa_diff / kappa_diff_err if kappa_diff_err > 0 else np.nan
        tension_trgb_vs_zero = (
            kappa_trgb / kappa_trgb_err if kappa_trgb_err and kappa_trgb_err > 0 else np.nan
        )

        # Joint chi2 (4 constraints)
        chi2_1 = (kappa_trgb / kappa_trgb_err) ** 2 if kappa_trgb_err and kappa_trgb_err > 0 else 0.0
        chi2_2 = (
            ((kappa_diff - kappa_host) / np.sqrt(kappa_diff_err ** 2 + kappa_host_err ** 2)) ** 2
            if kappa_diff_err > 0 and kappa_host_err > 0
            else 0.0
        )
        chi2_3 = (
            ((kappa_host - kappa_gal) / np.sqrt(kappa_host_err ** 2 + kappa_gal_err ** 2)) ** 2
            if kappa_host_err > 0
            else 0.0
        )
        chi2_4 = (
            ((kappa_joint - kappa_gal) / np.sqrt(kappa_joint_err ** 2 + kappa_gal_err ** 2)) ** 2
            if kappa_joint_err > 0
            else 0.0
        )

        joint_chi2 = chi2_1 + chi2_2 + chi2_3 + chi2_4
        joint_dof = 4
        joint_p = float(1 - stats.chi2.cdf(joint_chi2, joint_dof))

        print_status(f"Joint chi2 = {joint_chi2:.2f} / {joint_dof} (p = {joint_p:.3f})", "INFO")

        # ============================================================
        # 5. BUILD OUTPUT SCHEMA (matches step_9 expectations)
        # ============================================================
        results = {
            "kappa_cep": {
                "kappa_host": kappa_host,
                "kappa_host_err": kappa_host_err,
                "kappa_joint": kappa_joint,
                "kappa_joint_err": kappa_joint_err,
                "unified_h0": unified_h0,
            },
            "kappa_trgb": {
                "kappa_trgb": kappa_trgb,
                "kappa_trgb_err": kappa_trgb_err,
                "n_trgb": len(df_trgb),
                "raw_slope_t": raw_slope_t,
            },
            "kappa_diff": {
                "kappa_diff": kappa_diff,
                "kappa_diff_err": kappa_diff_err,
                "n_diff": len(merged),
            },
            "consistency_tests": {
                "tension_kappa_diff_vs_kappa_cep": float(tension_diff_vs_cep),
                "tension_kappa_diff_vs_zero": float(tension_diff_vs_zero),
                "tension_kappa_trgb_vs_zero": float(tension_trgb_vs_zero),
                "joint_chi2": float(joint_chi2),
                "joint_dof": joint_dof,
                "joint_pvalue": joint_p,
            },
            "theory": {
                "kappa_gal": kappa_gal,
                "kappa_gal_err": kappa_gal_err,
                "kappa_msp": 2.9e4,
                "kappa_msp_err": 4.5e4,
            },
        }

        with open(self.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print_status(f"Results saved to {self.results_json}", "SUCCESS")

        # Summary table
        print_status("\n--- CROSS-CHANNEL SUMMARY ---", "SECTION")
        headers = ["Quantity", "Value", "Error"]
        rows = [
            ["kappa_Cep (host)", f"{kappa_host:.3e}", f"{kappa_host_err:.3e}"],
            ["kappa_Cep (joint)", f"{kappa_joint:.3e}", f"{kappa_joint_err:.3e}"],
            ["kappa_TRGB", f"{kappa_trgb:.3e}", f"{kappa_trgb_err:.3e}"],
            ["kappa_diff", f"{kappa_diff:.3e}", f"{kappa_diff_err:.3e}"],
            ["Joint chi2", f"{joint_chi2:.2f}", f"p={joint_p:.3f}"],
        ]
        print_table(headers, rows)

        print_status("=" * 60, "SECTION")
        print_status("STEP 12 COMPLETE", "SECTION")
        print_status("=" * 60, "SECTION")

        return results


def main():
    step = Step12CrossChannel()
    step.run()


if __name__ == "__main__":
    main()
