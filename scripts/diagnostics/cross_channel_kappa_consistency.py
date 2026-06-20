#!/usr/bin/env python3
"""
Cross-Channel Kappa Consistency Analysis
==========================================

Implements the central TEP validation test that the codebase identifies
but does not execute: cross-channel consistency of the Observable Response
Coefficient across independent distance-indicator channels.

The TEP-H0 docstring (step_3_tep_correction.py, lines 1013-1017) states:
    "Universal-TEP validation requires CROSS-CHANNEL consistency
    (kappa_Cep vs kappa_TRGB vs kappa_SN vs kappa_pulsar). Single-channel
    significance of kappa_Cep from zero is NOT the TEP test."

This script fills that gap by:

1. Fitting kappa_TRGB from TRGB H0 data using the same TEP formalism applied
   to Cepheids. Under TEP, TRGB (non-periodic) should be unaffected:
   kappa_TRGB ~ 0.

2. Fitting kappa_diff from the differential modulus dmu = mu_TRGB - mu_Cep on
   matched hosts. Under TEP: kappa_diff ~ kappa_Cep (Cepheid bias isolated).
   Under a common systematic: kappa_diff ~ 0.

3. Comparing the Cepheid kappa_Cep with the canonical KAPPA_GAL and with
   external pulsar constraints from TEP-COS (Paper 10).

4. Producing a formal chi2 consistency test and tension matrix for all
   available channels.

Outputs:
    results/outputs/cross_channel_kappa_consistency.json
    results/figures/figure_06_cross_channel_kappa.png
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY
from scripts.utils.tep_correction import C_SQUARED_KM_S, tep_correction
from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

try:
    from scripts.utils.plot_style import apply_tep_style
    colors = apply_tep_style()
except ImportError:
    colors = {"blue": "#395d85", "accent": "#b43b4e", "dark": "#301E30", "green": "#4a7c59", "purple": "#6b4c7e"}


class CrossChannelKappaConsistency:
    """Cross-channel Observable Response Coefficient consistency analysis."""

    KAPPA_MSP_EMP = 2.9e4
    KAPPA_MSP_EMP_ERR = 4.5e4
    KAPPA_MSP_BARE_LO = 1.0e6
    KAPPA_MSP_BARE_HI = 1.0e7
    PLANCK_H0 = 67.4

    def __init__(self):
        self.root_dir = PROJECT_ROOT
        self.outputs_dir = self.root_dir / "results" / "outputs"
        self.figures_dir = self.root_dir / "results" / "figures"
        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        self.logs_dir = self.root_dir / "logs"
        for d in (self.outputs_dir, self.figures_dir, self.public_figures_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger("cross_channel_kappa", log_file_path=self.logs_dir / "cross_channel_kappa_consistency.log")
        set_step_logger(self.logger)
        self.tep_json = self.outputs_dir / "tep_correction_results.json"
        self.joint_json = self.outputs_dir / "joint_environmental_screening_model.json"
        self.trgb_csv = self.outputs_dir / "trgb_hosts_data.csv"
        self.stratified_csv = self.outputs_dir / "stratified_h0.csv"
        self.output_json = self.outputs_dir / "cross_channel_kappa_consistency.json"
        self.output_plot = self.figures_dir / "figure_06_cross_channel_kappa.png"

    def _load_kappa_cep(self) -> dict:
        with open(self.tep_json) as f:
            tep = json.load(f)
        with open(self.joint_json) as f:
            joint = json.load(f)
        # Use robust bootstrap std or WLS scaled (both ~0.6e6) instead of raw
        # inflated bootstrap std (~0.89e6). Use scaled joint err (~0.41e6)
        # instead of unscaled formal err (~0.08e6).
        kappa_host_err = float(
            tep.get("bootstrap_kappa_robust_std")
            or tep.get("wls_kappa_err_scaled")
            or tep.get("bootstrap_kappa_std", 8.9e5)
        )
        kappa_joint_err = float(
            joint.get("joint_kappa_err_scaled")
            or joint.get("joint_kappa_err", 8.3e4)
        )
        return {
            "kappa_host": float(tep["optimal_kappa_cep"]),
            "kappa_host_err": kappa_host_err,
            "kappa_joint": float(joint["joint_kappa_cep"]),
            "kappa_joint_err": kappa_joint_err,
            "sigma_ref": float(tep["sigma_ref"]),
            "unified_h0": float(tep["unified_h0"]),
        }

    def _load_trgb_data(self) -> pd.DataFrame | None:
        if not self.trgb_csv.exists():
            print_status("TRGB data not found. Run Step 7 first.", "WARNING")
            return None
        if not self.stratified_csv.exists():
            print_status("Stratified H0 data not found. Run Step 2 first.", "WARNING")
            return None
        trgb = pd.read_csv(self.trgb_csv)
        ceph = pd.read_csv(self.stratified_csv)
        trgb["match"] = trgb["galaxy"].str.replace(" ", "").str.upper()
        ceph["match"] = ceph["normalized_name"].str.replace(" ", "").str.upper()
        merged = pd.merge(
            trgb,
            ceph[["match", "value", "error", "h0_derived", "sigma_inferred",
                  "shear_suppression", "rho_local", "velocity", "z_hd"]],
            on="match",
            how="inner",
            suffixes=("_trgb", "_ceph"),
        )
        merged = merged.rename(columns={
            "value": "mu_ceph",
            "error": "mu_ceph_err",
            "h0_derived": "h0_ceph",
            "sigma_inferred_ceph": "sigma",
            "shear_suppression": "S",
            "velocity": "velocity",
            "z_hd_ceph": "z_hd",
        })
        print_status(f"Matched {len(merged)} hosts for cross-channel analysis", "INFO")
        return merged

    def fit_kappa_trgb(self, df: pd.DataFrame, sigma_ref: float) -> dict:
        """
        Fit kappa_TRGB via weighted linear regression of H0 vs TEP regressor.

        For small corrections:
            H0 ~ H0_base * (1 - (ln 10 / 5) * kappa * x)
        where x = S*(sigma^2 - sigma_ref^2)/c^2.
        So the slope b of H0 vs x is:
            b = -(ln 10 / 5) * H0_base * kappa
        and:
            kappa = -5*b / (ln 10 * H0_base)
        """
        sigma = df["sigma"].values
        S = df["S"].values
        x = S * (sigma ** 2 - sigma_ref ** 2) / C_SQUARED_KM_S
        h0 = df["h0_trgb"].values
        h0_err = df["h0_trgb_err"].values

        # Weighted linear regression: H0 = a + b*x
        weights = 1.0 / h0_err ** 2
        X = np.vstack([np.ones_like(x), x]).T
        W = np.diag(weights)
        cov = np.linalg.inv(X.T @ W @ X)
        params = cov @ (X.T @ W @ h0)
        intercept, slope = params
        intercept_err, slope_err = np.sqrt(np.diag(cov))

        h0_base = float(intercept)
        ln10 = np.log(10)

        # Convert slope to kappa
        kappa = -5.0 * slope / (ln10 * h0_base)
        kappa_err = abs(kappa) * np.sqrt((slope_err / slope) ** 2 + (intercept_err / intercept) ** 2) if slope != 0 else float("nan")

        # Also report the raw slope significance (null test)
        t_slope = abs(slope) / slope_err if slope_err > 0 else float("inf")

        return {
            "kappa_trgb": float(kappa),
            "kappa_trgb_err": float(kappa_err),
            "h0_base_trgb": float(h0_base),
            "h0_base_trgb_err": float(intercept_err),
            "raw_slope": float(slope),
            "raw_slope_err": float(slope_err),
            "raw_slope_t": float(t_slope),
            "n_trgb": int(len(df)),
        }

    def fit_kappa_diff(self, df: pd.DataFrame, sigma_ref: float) -> dict:
        sigma = df["sigma"].values
        S = df["S"].values
        x = S * (sigma ** 2 - sigma_ref ** 2) / C_SQUARED_KM_S
        delta_mu = (df["mu_trgb"] - df["mu_ceph"]).values
        delta_err = np.sqrt(df["mu_trgb_err"]**2 + df["mu_ceph_err"]**2).values

        weights = 1.0 / delta_err ** 2
        X = np.vstack([np.ones_like(x), x]).T
        W = np.diag(weights)
        cov = np.linalg.inv(X.T @ W @ X)
        params = cov @ (X.T @ W @ delta_mu)
        intercept, kappa = params
        intercept_err, kappa_err = np.sqrt(np.diag(cov))

        pred = intercept + kappa * x
        residuals = delta_mu - pred
        chi2 = np.sum((residuals / delta_err) ** 2)
        dof = len(x) - 2
        r, p = stats.pearsonr(x, delta_mu)

        return {
            "kappa_diff": float(kappa),
            "kappa_diff_err": float(kappa_err),
            "intercept": float(intercept),
            "intercept_err": float(intercept_err),
            "chi2": float(chi2),
            "dof": int(dof),
            "chi2_per_dof": float(chi2 / dof) if dof > 0 else float("nan"),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "n_diff": int(len(x)),
        }

    @staticmethod
    def gaussian_tension(val1: float, err1: float, val2: float, err2: float) -> float:
        return abs(val1 - val2) / np.sqrt(err1 ** 2 + err2 ** 2)

    def run_consistency_tests(self, kappa_cep: dict, kappa_trgb: dict, kappa_diff: dict) -> dict:
        results = {}
        for label, kc, ke in [
            ("host", kappa_cep["kappa_host"], kappa_cep["kappa_host_err"]),
            ("joint", kappa_cep["kappa_joint"], kappa_cep["kappa_joint_err"]),
        ]:
            tension = self.gaussian_tension(kc, ke, KAPPA_GAL, KAPPA_GAL_UNCERTAINTY)
            results[f"tension_kappa_{label}_vs_kappa_gal"] = float(tension)
            results[f"consistent_kappa_{label}_vs_kappa_gal"] = bool(tension < 2.0)

        if not np.isnan(kappa_trgb["kappa_trgb_err"]) and kappa_trgb["kappa_trgb_err"] > 0:
            tension_trgb_zero = abs(kappa_trgb["kappa_trgb"]) / kappa_trgb["kappa_trgb_err"]
            results["tension_kappa_trgb_vs_zero"] = float(tension_trgb_zero)
            results["consistent_kappa_trgb_with_zero"] = bool(tension_trgb_zero < 2.0)
        else:
            results["tension_kappa_trgb_vs_zero"] = float("nan")
            results["consistent_kappa_trgb_with_zero"] = False

        kc = kappa_cep["kappa_host"]
        ke = kappa_cep["kappa_host_err"]
        kd = kappa_diff["kappa_diff"]
        kde = kappa_diff["kappa_diff_err"]

        tension_diff_vs_cep = self.gaussian_tension(kd, kde, kc, ke)
        results["tension_kappa_diff_vs_kappa_cep"] = float(tension_diff_vs_cep)
        results["consistent_kappa_diff_with_kappa_cep"] = bool(tension_diff_vs_cep < 2.0)

        tension_diff_zero = abs(kd) / kde if kde > 0 else float("inf")
        results["tension_kappa_diff_vs_zero"] = float(tension_diff_zero)
        results["consistent_kappa_diff_with_zero"] = bool(tension_diff_zero < 2.0)

        chi2 = 0.0
        chi2 += ((kc - KAPPA_GAL) / np.sqrt(ke**2 + KAPPA_GAL_UNCERTAINTY**2)) ** 2
        if not np.isnan(kappa_trgb["kappa_trgb_err"]) and kappa_trgb["kappa_trgb_err"] > 0:
            chi2 += (kappa_trgb["kappa_trgb"] / kappa_trgb["kappa_trgb_err"]) ** 2
        chi2 += ((kd - kc) / np.sqrt(kde**2 + ke**2)) ** 2
        chi2 += ((self.KAPPA_MSP_EMP - kc) / np.sqrt(self.KAPPA_MSP_EMP_ERR**2 + ke**2)) ** 2

        results["joint_chi2"] = float(chi2)
        results["joint_dof"] = 4
        results["joint_chi2_per_dof"] = float(chi2 / 4.0)
        results["joint_pvalue"] = float(stats.chi2.sf(chi2, 4))
        return results

    def plot_results(self, df: pd.DataFrame, kappa_cep: dict, kappa_trgb: dict,
                     kappa_diff: dict, tests: dict, sigma_ref: float):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        sigma = df["sigma"].values
        S = df["S"].values
        x = S * (sigma ** 2 - sigma_ref ** 2) / C_SQUARED_KM_S
        delta_mu = (df["mu_trgb"] - df["mu_ceph"]).values
        delta_err = np.sqrt(df["mu_trgb_err"]**2 + df["mu_ceph_err"]**2).values
        ax.errorbar(
            x * 1e7, delta_mu, yerr=delta_err,
            fmt="o", color=colors["blue"], alpha=0.8, capsize=3,
            markersize=8, label=f"Matched hosts (N={len(df)})",
        )
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = kappa_diff["intercept"] + kappa_diff["kappa_diff"] * x_line
        ax.plot(
            x_line * 1e7, y_line, "--", color=colors["accent"], linewidth=2.5,
            label=f"Fit: kappa_diff = ({kappa_diff['kappa_diff']/1e6:.2f} +/- {kappa_diff['kappa_diff_err']/1e6:.2f}) x 10^6 mag",
        )
        kc = kappa_cep["kappa_host"]
        ke = kappa_cep["kappa_host_err"]
        ax.fill_between(
            x_line * 1e7,
            (kc - ke) * x_line,
            (kc + ke) * x_line,
            color=colors["green"], alpha=0.15,
            label=rf"TEP prediction: kappa_diff = kappa_Cep  ({kc/1e6:.2f} +/- {ke/1e6:.2f}) x 10^6",
        )
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("TEP Regressor S (sigma^2 - sigma_ref^2) / c^2 (x 10^-7)", fontsize=12)
        ax.set_ylabel("Delta mu = mu_TRGB - mu_Cep (mag)", fontsize=12)
        ax.set_title("Differential Modulus: TRGB minus Cepheid", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        channels = []
        vals = []
        errs = []
        colors_list = []
        channels.append("kappa_Cep (host)")
        vals.append(kappa_cep["kappa_host"] / 1e6)
        errs.append(kappa_cep["kappa_host_err"] / 1e6)
        colors_list.append(colors["accent"])
        channels.append("kappa_Cep (joint)")
        vals.append(kappa_cep["kappa_joint"] / 1e6)
        errs.append(kappa_cep["kappa_joint_err"] / 1e6)
        colors_list.append(colors["accent"])
        channels.append("kappa_gal (canonical)")
        vals.append(KAPPA_GAL / 1e6)
        errs.append(KAPPA_GAL_UNCERTAINTY / 1e6)
        colors_list.append(colors["dark"])
        if not np.isnan(kappa_trgb["kappa_trgb_err"]):
            channels.append("kappa_TRGB")
            vals.append(kappa_trgb["kappa_trgb"] / 1e6)
            errs.append(kappa_trgb["kappa_trgb_err"] / 1e6)
            colors_list.append(colors["blue"])
        channels.append("kappa_diff (TRGB-Cep)")
        vals.append(kappa_diff["kappa_diff"] / 1e6)
        errs.append(kappa_diff["kappa_diff_err"] / 1e6)
        colors_list.append(colors["green"])
        channels.append("kappa_MSP^emp (Paper 10)")
        vals.append(self.KAPPA_MSP_EMP / 1e6)
        errs.append(self.KAPPA_MSP_EMP_ERR / 1e6)
        colors_list.append(colors["purple"])
        y_pos = np.arange(len(channels))
        for i, (v, e, c) in enumerate(zip(vals, errs, colors_list)):
            ax.errorbar(v, i, xerr=e, fmt="o", color=c, capsize=5, markersize=10, linewidth=2)
            ax.text(v, i + 0.15, f"{v:.2f} +/- {e:.2f}", ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels, fontsize=11)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("kappa (x 10^6 mag)", fontsize=12)
        ax.set_title("Cross-Channel Observable Response Coefficients", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.text(
            0.98, 0.02,
            f"TEP prediction: kappa_TRGB ~ 0, kappa_diff ~ kappa_Cep\n"
            f"Tension (diff vs Cep): {tests['tension_kappa_diff_vs_kappa_cep']:.2f}s\n"
            f"Joint chi2/dof = {tests['joint_chi2']:.1f}/{tests['joint_dof']} (p={tests['joint_pvalue']:.3f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        plt.tight_layout()
        plt.savefig(self.output_plot, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        shutil.copy(self.output_plot, self.public_figures_dir / "figure_06_cross_channel_kappa.png")
        print_status(f"Saved cross-channel plot to {self.output_plot}", "SUCCESS")

    def run(self):
        print_status("CROSS-CHANNEL KAPPA CONSISTENCY ANALYSIS", "TITLE")
        print_status("Filling the gap between theory and practice: the real TEP test.", "INFO")

        kappa_cep = self._load_kappa_cep()
        df = self._load_trgb_data()

        if df is None or len(df) < 5:
            print_status("Insufficient data for cross-channel analysis.", "WARNING")
            return

        sigma_ref = kappa_cep["sigma_ref"]

        print_status("Fitting kappa_TRGB (TRGB channel)...", "SECTION")
        kappa_trgb = self.fit_kappa_trgb(df, sigma_ref)
        print_status(f"kappa_TRGB = {kappa_trgb['kappa_trgb']:.3e} +/- {kappa_trgb['kappa_trgb_err']:.3e} mag", "INFO")
        print_status(f"  Raw slope H0 vs x: {kappa_trgb['raw_slope']:.3f} +/- {kappa_trgb['raw_slope_err']:.3f} (t = {kappa_trgb['raw_slope_t']:.2f})", "INFO")
        print_status(f"TEP prediction: kappa_TRGB ~ 0 (non-periodic indicator)", "INFO")

        print_status("Fitting kappa_diff (differential TRGB - Cepheid)...", "SECTION")
        kappa_diff = self.fit_kappa_diff(df, sigma_ref)
        print_status(f"kappa_diff = {kappa_diff['kappa_diff']:.3e} +/- {kappa_diff['kappa_diff_err']:.3e} mag", "INFO")
        print_status(f"TEP prediction: kappa_diff ~ kappa_Cep = {kappa_cep['kappa_host']:.3e} mag", "INFO")

        print_status("Running formal consistency tests...", "SECTION")
        tests = self.run_consistency_tests(kappa_cep, kappa_trgb, kappa_diff)

        headers = ["Test", "Tension (sigma)", "Consistent?"]
        rows = [
            ["kappa_Cep (host) vs KAPPA_GAL", f"{tests['tension_kappa_host_vs_kappa_gal']:.2f}", "YES" if tests['consistent_kappa_host_vs_kappa_gal'] else "NO"],
            ["kappa_Cep (joint) vs KAPPA_GAL", f"{tests['tension_kappa_joint_vs_kappa_gal']:.2f}", "YES" if tests['consistent_kappa_joint_vs_kappa_gal'] else "NO"],
            ["kappa_TRGB vs 0 (null)", f"{tests['tension_kappa_trgb_vs_zero']:.2f}", "YES" if tests['consistent_kappa_trgb_with_zero'] else "NO"],
            ["kappa_diff vs kappa_Cep (TEP)", f"{tests['tension_kappa_diff_vs_kappa_cep']:.2f}", "YES" if tests['consistent_kappa_diff_with_kappa_cep'] else "NO"],
            ["kappa_diff vs 0 (common bias)", f"{tests['tension_kappa_diff_vs_zero']:.2f}", "YES" if tests['consistent_kappa_diff_with_zero'] else "NO"],
        ]
        print_table(headers, rows, title="Cross-Channel Consistency Tests")

        print_status("Joint chi2 test of all channels against TEP predictions:", "SECTION")
        print_status(f"  chi2 = {tests['joint_chi2']:.2f} / {tests['joint_dof']} dof", "INFO")
        print_status(f"  chi2/dof = {tests['joint_chi2_per_dof']:.2f}", "INFO")
        print_status(f"  p-value = {tests['joint_pvalue']:.4f}", "INFO")
        if tests['joint_pvalue'] > 0.05:
            print_status("  All channels are mutually consistent with TEP predictions.", "SUCCESS")
        else:
            print_status("  Some channels deviate from TEP predictions at >2sigma.", "WARNING")

        self.plot_results(df, kappa_cep, kappa_trgb, kappa_diff, tests, sigma_ref)

        results = {
            "kappa_cep": kappa_cep,
            "kappa_trgb": kappa_trgb,
            "kappa_diff": kappa_diff,
            "consistency_tests": tests,
            "n_matched_hosts": int(len(df)),
        }
        with open(self.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print_status(f"Saved cross-channel results to {self.output_json}", "SUCCESS")


def main():
    analysis = CrossChannelKappaConsistency()
    analysis.run()


if __name__ == "__main__":
    main()
