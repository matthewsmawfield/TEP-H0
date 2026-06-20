#!/usr/bin/env python3
"""
Step 17: Primary TEP Regressor Audit
=====================================

Compares the same sample, same response variable (H0), same hosts against
different regressors to test whether the TEP-consistent regressor
X_TEP = S_total * (sigma^2 - sigma_ref^2) / c^2 is stronger than raw sigma
or sigma^2.

If TEP is real, the signal should strengthen as the regressor becomes more
physically correct:
    sigma -> sigma^2 -> S_local * sigma^2 -> S_local * S_group * sigma^2

Regressors tested:
    1. sigma (km/s) — empirical proxy
    2. sigma^2 (km^2/s^2) — virial proxy
    3. S_local(rho) * sigma^2 — local-density TEP proxy
    4. S_local(rho) * S_group(N_mb) * sigma^2 — full TEP proxy
    5. stellar mass (log M_*) — confound/null
    6. redshift (z_hd) — flow/null
    7. metallicity proxy (m_b_corr) — astrophysical/null
    8. random shuffled sigma — null control

Outputs: JSON with correlation strengths, significance, and scatter for each.
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
from scripts.utils.tep_correction import C_SQUARED_KM_S


class Step17RegressorAudit:
    """Formal pipeline step: compare TEP regressors head-to-head."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.results_dir = self.root / "results" / "outputs"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.logger = TEPLogger(
            "step_17_regressor_audit",
            log_file_path=self.logs_dir / "step_17_regressor_audit.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status(">>> STEP 17: PRIMARY TEP REGRESSOR AUDIT", "TITLE"
        )

        strat = pd.read_csv(self.results_dir / "stratified_h0.csv")
        n = len(strat)

        # Load TEP parameters
        with open(self.results_dir / "tep_correction_results.json") as f:
            tep = json.load(f)
        sigma_ref = float(tep["sigma_ref"])
        c2 = C_SQUARED_KM_S

        sigma = strat["sigma_inferred"].values
        h0 = strat["h0_derived"].values
        S_local = strat["shear_suppression"].values

        # S_group from tully_nmb
        from scripts.utils.tep_correction import group_screening_factor
        n_mb = strat["tully_nmb"].fillna(1.0).values
        S_group = np.array([group_screening_factor(x) for x in n_mb])
        S_total = S_local * S_group

        # Build regressors
        regressors = {
            "sigma": sigma,
            "sigma_sq": sigma ** 2,
            "S_local_sigma_sq": S_local * sigma ** 2,
            "S_total_sigma_sq": S_total * sigma ** 2,
            "TEP_full": S_total * (sigma ** 2 - sigma_ref ** 2) / c2,
        }

        # Confound / null regressors
        if "host_logmass" in strat.columns:
            regressors["host_logmass"] = strat["host_logmass"].fillna(strat["host_logmass"].median()).values
        regressors["z_hd"] = strat["z_hd"].values
        if "m_b_corr" in strat.columns:
            regressors["m_b_corr"] = strat["m_b_corr"].fillna(strat["m_b_corr"].median()).values
        # Random shuffle null
        rng = np.random.default_rng(42)
        regressors["sigma_shuffled"] = rng.permutation(sigma)

        # Compute correlations
        rows = []
        for name, x in regressors.items():
            valid = np.isfinite(x) & np.isfinite(h0)
            if valid.sum() < 3:
                continue
            r, p = stats.pearsonr(x[valid], h0[valid])
            # Spearman
            rho, prho = stats.spearmanr(x[valid], h0[valid])
            # Scatter around best linear fit
            slope, intercept, _, _, _ = stats.linregress(x[valid], h0[valid])
            pred = slope * x[valid] + intercept
            scatter = float(np.std(h0[valid] - pred))
            rows.append({
                "regressor": name,
                "N": int(valid.sum()),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(prho),
                "scatter_kms_mpc": float(scatter),
                "is_tep_regressor": name in ["S_local_sigma_sq", "S_total_sigma_sq", "TEP_full"],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("pearson_r", key=abs, ascending=False)

        print_status("Regressor comparison (sorted by |Pearson r|):", "INFO")
        print_table(
            ["Regressor", "N", "Pearson r", "p", "Spearman ρ", "p", "Scatter"],
            [
                [
                    r["regressor"],
                    str(r["N"]),
                    f"{r['pearson_r']:.3f}",
                    f"{r['pearson_p']:.4f}",
                    f"{r['spearman_rho']:.3f}",
                    f"{r['spearman_p']:.4f}",
                    f"{r['scatter_kms_mpc']:.2f}",
                ]
                for _, r in df.iterrows()
            ],
            title="TEP Regressor Audit",
        )

        # Key finding: does signal strengthen with more TEP-correct regressor?
        tep_regressors = df[df["is_tep_regressor"]]
        if len(tep_regressors) > 0:
            best_tep = tep_regressors.loc[tep_regressors["pearson_r"].abs().idxmax()]
            print_status(
                f"Best TEP regressor: {best_tep['regressor']} "
                f"(r={best_tep['pearson_r']:.3f}, p={best_tep['pearson_p']:.4f})",
                "SUCCESS" if best_tep["pearson_p"] < 0.05 else "WARNING",
            )

        # Save
        df.to_json(
            self.results_dir / "regressor_audit.json",
            orient="records",
            indent=2,
        )

        # Also save as markdown summary
        summary = {
            "audit_name": "TEP Regressor Audit",
            "sample_N": n,
            "sigma_ref_kms": sigma_ref,
            "best_regressor": df.iloc[0]["regressor"] if len(df) > 0 else None,
            "best_pearson_r": float(df.iloc[0]["pearson_r"]) if len(df) > 0 else None,
            "best_pearson_p": float(df.iloc[0]["pearson_p"]) if len(df) > 0 else None,
            "tep_regressor_rank": int(
                (df["regressor"] == "TEP_full").argmax() + 1
            ) if "TEP_full" in df["regressor"].values else None,
            "all_results": df.to_dict(orient="records"),
        }
        with open(self.results_dir / "regressor_audit_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print_status("Step 17 complete", "SUCCESS")


if __name__ == "__main__":
    Step17RegressorAudit().run()
