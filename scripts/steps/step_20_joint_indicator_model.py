#!/usr/bin/env python3
"""
Step 19: Joint Cepheid + TRGB Indicator Model
=============================================

Fits a joint model separating common host systematics from
indicator-specific clock bias:

    mu_ij = mu_true,i + A_i + B_indicator * X_i + epsilon_ij

where:
    - i = host index
    - j = indicator index (Cepheid or TRGB)
    - A_i = host-specific systematic (common to both indicators)
    - B_Cepheid = Cepheid-specific clock response coefficient
    - B_TRGB = TRGB response (should be << B_Cepheid if TEP is real)
    - X_i = TEP regressor = S_total * (sigma^2 - sigma_ref^2) / c^2

The key test: B_Cepheid > B_TRGB with shared host terms marginalized out.

If B_Cepheid >> B_TRGB, this supports a Cepheid-specific clock-rate bias
superposed on shared host systematics.

If B_Cepheid ≈ B_TRGB, the signal is entirely shared systematic.

Outputs: JSON with B_Cepheid, B_TRGB, their difference, and significance.
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
from scripts.utils.tep_correction import C_SQUARED_KM_S, group_screening_factor


class Step19JointIndicatorModel:
    """Formal pipeline step: joint Cepheid+TRGB indicator model."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_19_joint_indicator",
            log_file_path=self.logs_dir / "step_20_joint_indicator.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 19: Joint Cepheid + TRGB indicator model", "TITLE"
        )

        strat = pd.read_csv(self.results_dir / "step_03_stratified_h0.csv")
        trgb = pd.read_csv(self.results_dir / "step_15_trgb_hosts_data.csv")

        # Load TEP params
        with open(self.results_dir / "step_04_tep_correction_results.json") as f:
            tep = json.load(f)
        sigma_ref = float(tep["sigma_ref"])
        c2 = C_SQUARED_KM_S

        # Merge TRGB with stratified
        trgb["match"] = trgb["galaxy"].str.replace(" ", "").str.upper()
        strat["match"] = strat["normalized_name"].str.replace(" ", "").str.upper()
        merged = pd.merge(
            trgb, strat, on="match", suffixes=("_trgb", "_host"), how="inner"
        )

        n_match = len(merged)
        print_status(f"Matched hosts: N={n_match}", "INFO")

        if n_match < 5:
            print_status("Too few matched hosts; skipping", "INFO")
            return

        # Compute TEP regressor: use SAME regressor as Step 12 cross-channel
        # Step 12 uses: R_m = S_local * (sigma^2 - sigma_ref^2) / c^2
        # NOT S_total = S_local * S_group (which adds noise for isolated hosts)
        # Use the same sigma as Step 3/Step 12 (from step_03_stratified_h0.csv),
        # not the TRGB-file sigma, to ensure regressor consistency.
        sigma = merged["sigma_inferred_host"].values
        S_local = merged["shear_suppression"].values
        R_m = S_local * (sigma ** 2 - sigma_ref ** 2) / c2

        # Robust approach: host-differenced model
        # Under TEP, high-sigma Cepheid distances are underestimated (too small),
        # so mu_Ceph is smaller than it should be, and mu_TRGB - mu_Ceph > 0.
        # The slope of delta_mu on R_m should be positive.
        delta_mu = merged["mu_trgb"].values - merged["value"].values
        delta_err = np.sqrt(merged["error"].values ** 2 + merged["mu_trgb_err"].values ** 2)

        # Differential slope: delta_mu = mu_TRGB - mu_Ceph ~ kappa_diff * R_m
        slope_diff, intercept_diff, _, _, se_diff = stats.linregress(R_m, delta_mu)
        kappa_diff = float(slope_diff)
        kappa_diff_err = float(se_diff)
        t_diff = kappa_diff / kappa_diff_err if kappa_diff_err > 0 else 0
        p_diff = 2 * (1 - stats.t.cdf(abs(t_diff), max(n_match - 2, 1)))

        # Also fit each indicator separately on R_m for comparison
        slope_c, _, _, _, se_c = stats.linregress(R_m, merged["value"].values)
        slope_t, _, _, _, se_t = stats.linregress(R_m, merged["mu_trgb"].values)

        B_ceph = float(slope_c)
        B_trgb = float(slope_t)
        se_ceph = float(se_c)
        se_trgb = float(se_t)

        t_ceph = B_ceph / se_ceph if se_ceph > 0 else 0
        t_trgb = B_trgb / se_trgb if se_trgb > 0 else 0
        p_ceph = 2 * (1 - stats.t.cdf(abs(t_ceph), max(n_match - 2, 1)))
        p_trgb = 2 * (1 - stats.t.cdf(abs(t_trgb), max(n_match - 2, 1)))

        print_status("Joint indicator model results:", "INFO")
        print_status(
            "Using S_local-only regressor (same as Step 12 cross-channel)", "INFO"
        )
        headers = ["Parameter", "Estimate", "SE", "t-stat", "p-value"]
        rows = [
            ["B_Cepheid", f"{B_ceph:.3e}", f"{se_ceph:.3e}", f"{t_ceph:.2f}", f"{p_ceph:.4f}"],
            ["B_TRGB", f"{B_trgb:.3e}", f"{se_trgb:.3e}", f"{t_trgb:.2f}", f"{p_trgb:.4f}"],
            ["kappa_diff = B_TRGB - B_Ceph", f"{kappa_diff:.3e}", f"{kappa_diff_err:.3e}", f"{t_diff:.2f}", f"{p_diff:.4f}"],
        ]
        print_table(headers, rows)

        # Interpretation
        if kappa_diff > 0 and p_diff < 0.05:
            print_status(
                f"Result: kappa_diff = {kappa_diff:.3e} ± {kappa_diff_err:.3e} (t={t_diff:.2f}, p={p_diff:.4f}). "
                "Positive slope means mu_TRGB - mu_Ceph increases with sigma. "
                "Directionally consistent with TEP (Cepheids underestimated at high sigma). "
                f"But magnitude ({kappa_diff:.2e}) is smaller than kappa_Cep ({float(tep['optimal_kappa_cep']):.2e}), "
                "suggesting shared systematic dominates.",
                "INFO",
            )
        elif abs(t_diff) < 2:
            print_status(
                "Result: kappa_diff consistent with zero. "
                "No differential signal detected above shared systematic noise.",
                "WARNING",
            )
        else:
            print_status(
                f"Result: kappa_diff = {kappa_diff:.3e} ± {kappa_diff_err:.3e} (t={t_diff:.2f}, p={p_diff:.4f})",
                "INFO",
            )

        results = {
            "N_hosts": n_match,
            "regressor": "S_local * (sigma^2 - sigma_ref^2) / c^2 (same as Step 12)",
            "B_ceph": B_ceph,
            "B_ceph_se": se_ceph,
            "B_ceph_t": t_ceph,
            "B_ceph_p": p_ceph,
            "B_trgb": B_trgb,
            "B_trgb_se": se_trgb,
            "B_trgb_t": t_trgb,
            "B_trgb_p": p_trgb,
            "kappa_diff": kappa_diff,
            "kappa_diff_err": kappa_diff_err,
            "kappa_diff_t": t_diff,
            "kappa_diff_p": p_diff,
            "interpretation": (
                "kappa_diff > 0 is directionally consistent with TEP "
                "(Cepheids underestimated at high sigma), but magnitude "
                f"({kappa_diff:.2e}) is smaller than kappa_Cep ({float(tep['optimal_kappa_cep']):.2e}), "
                "suggesting shared systematic dominates."
                if kappa_diff > 0 and p_diff < 0.05
                else "kappa_diff consistent with zero: no differential signal detected."
            ),
        }

        with open(self.results_dir / "step_20_joint_indicator_model.json", "w") as f:
            json.dump(results, f, indent=2)

        print_status("Saved results to step_20_joint_indicator_model.json", "SUCCESS")
        print_status("Step 19 complete", "SUCCESS")


if __name__ == "__main__":
    Step19JointIndicatorModel().run()
