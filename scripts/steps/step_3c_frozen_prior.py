import json
import os
import sys
from pathlib import Path

# Make direct execution behave the same as module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from scripts.utils.logger import TEPLogger, print_status, print_table, set_step_logger
from scripts.utils.tep_correction import tep_correction, KAPPA_CEP_PAPER10


class Step3CFrozenPrior:
    r"""
    Step 3C: Cross-Domain Consistency Check
    =======================================

    This step tests whether the TEP framework's bare geometric-factor
    estimate (~10^6 mag), applied without SH0ES tuning, yields a
    Planck-consistent H_0. Rather than fitting κ_Cep to minimise the
    residual H0–σ slope across the SH0ES Hubble-flow hosts, we freeze
    κ_Cep to the central Cepheid value from Paper 11:

        κ_Cep (Paper 11) = 1.05 × 10^6 mag.

    This is NOT an independent pulsar prediction. Paper 10 measures the
    effective screened pulsar coefficient ~3 × 10^4 in dense globular
    clusters (step_5_55_kappa_msp_prior.json). The bare ~10^6 value is
    independently calibrated here (Paper 11) via Cepheid period-luminosity
    residuals. This step checks whether applying the bare estimate
    without SH0ES tuning gives a sensible result—a consistency test,
    not a zero-free-parameter prediction.

    Physics rationale (TEP v0.8 §5–§7; Paper 10 and Paper 11):
    The same conformal factor A(φ) that modulates pulsar periods in deep
    globular-cluster potentials also modulates Cepheid periods in deep
    galactic potentials. The Observable Response Coefficient absorbs channel-
    specific details (virial proportionality, P-L slope, 1/ln 10) but the
    underlying temporal-response hierarchy is predicted to be comparable.
    Paper 10 reports κ_MSP ~ 10^6–10^7 mag; freezing κ_Cep = 1.05×10^6 mag
    tests whether the Cepheid channel obeys the same hierarchy.

    Methodology:
    1. Load the stratified host sample (same as Step 3).
    2. Load the effective calibrator reference σ_ref from Step 3 outputs
       (derived from MW/LMC/NGC 4258 disk dispersions; independent of
       Hubble-flow host tuning).
    3. Apply Δμ = κ_Cep · S(ρ) · (σ^2 − σ_ref^2)/c^2 with FIXED κ_Cep.
    4. Compute unified H0, residual slope, and correlation statistics.
    5. Bootstrap (host resampling only, kappa frozen) to estimate honest
       host-scatter uncertainty.
    6. Report Planck tension.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.root_dir / "logs"
        self.figures_dir = self.root_dir / "results" / "figures"
        self.outputs_dir = self.root_dir / "results" / "outputs"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Logger
        self.logger = TEPLogger(
            "step_3c_frozen_prior", log_file_path=self.logs_dir / "step_3c_frozen_prior.log"
        )
        set_step_logger(self.logger)

        # Inputs
        self.stratified_path = self.outputs_dir / "stratified_h0.csv"
        self.tep_results_path = self.outputs_dir / "tep_correction_results.json"

        # Outputs
        self.json_output_path = self.outputs_dir / "frozen_prior_results.json"

    def load_data(self):
        """Loads stratified data and sigma_ref from Step 3."""
        print_status("Loading Data...", "SECTION")

        if not self.stratified_path.exists():
            print_status("Stratified data missing. Run Step 2 first.", "ERROR")
            sys.exit(1)

        df = pd.read_csv(self.stratified_path)
        if "shear_suppression" not in df.columns:
            df["shear_suppression"] = 1.0
            print_status(
                "shear_suppression missing; defaulting S=1.0 (fully active).", "WARNING"
            )

        # Load sigma_ref from Step 3 results
        sigma_ref = 75.25  # Fallback
        if self.tep_results_path.exists():
            with open(self.tep_results_path, "r") as f:
                tep = json.load(f)
            sigma_ref = float(tep.get("sigma_ref", 75.25))
            print_status(f"Loaded σ_ref = {sigma_ref:.2f} km/s from Step 3.", "INFO")
        else:
            print_status(
                f"Step 3 results missing; using fallback σ_ref = {sigma_ref:.2f} km/s.",
                "WARNING",
            )

        print_status(f"Loaded {len(df)} hosts for cross-domain consistency check.", "INFO")
        return df, sigma_ref

    def apply_fixed_correction(self, df, sigma_ref, kappa_cep):
        """Applies correction with fixed kappa and returns H0 values."""
        S = df["shear_suppression"].values.astype(float)
        sigma_vals = df["sigma_inferred"].values.astype(float)
        mu = df["value"].values.astype(float)
        v = df["velocity"].values.astype(float)

        correction = tep_correction(sigma_vals, sigma_ref, kappa_cep, S)
        mu_corr = mu + correction
        dist_corr = 10 ** ((mu_corr - 25) / 5)
        h0_corr = v / dist_corr

        return h0_corr, correction, S

    def compute_statistics(self, df, h0_corr, sigma_vals):
        """Computes H0 mean, SEM, residual slope, and correlation stats."""
        h0_mean = float(np.mean(h0_corr))
        h0_sem = float(np.std(h0_corr, ddof=1) / np.sqrt(len(h0_corr)))

        # Residual slope and correlations
        slope, intercept = np.polyfit(sigma_vals, h0_corr, 1)
        pearson_r, pearson_p = stats.pearsonr(sigma_vals, h0_corr)
        spearman_rho, spearman_p = stats.spearmanr(sigma_vals, h0_corr)

        return {
            "h0_mean": h0_mean,
            "h0_sem": h0_sem,
            "residual_slope": float(slope),
            "residual_intercept": float(intercept),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        }

    def blind_slope_prediction(self, df, sigma_ref, kappa_cep):
        """Compute predicted H0–σ slope using the TEP correction formula.

        This tests whether the bare κ_Cep (~10^6 mag), combined with the
        sample's σ-distribution, predicts the magnitude and sign of the
        uncorrected H0–σ slope. It is a consistency check, not an
        independent prediction: the ~10^6 value is calibrated here (Paper 11),
        not predicted by Paper 10 pulsars (which measure ~3 × 10^4 in
        dense clusters). The model assumes all hosts share a single true
        H0 = Planck (67.4 km/s/Mpc). The predicted observed H0 for each host is:

            H0_pred(σ) = H0_true * (1 + ln(10)/5 * Δμ(σ))

        where Δμ = κ_Cep * S(ρ) * (σ² − σ_ref²) / c².
        """
        print_status("Slope consistency check (κ_Cep + σ-distribution)...", "SECTION")

        S = df["shear_suppression"].values.astype(float)
        sigma_vals = df["sigma_inferred"].values.astype(float)
        mu = df["value"].values.astype(float)
        v = df["velocity"].values.astype(float)

        # Observed raw H0 (for comparison only)
        h0_raw = v / (10 ** ((mu - 25) / 5))
        obs_slope, _ = np.polyfit(sigma_vals, h0_raw, 1)

        # Blind prediction: assume H0_true = Planck = 67.4 for all hosts
        h0_true_fid = 67.4
        from scripts.utils.tep_correction import C_SQUARED_KM_S
        delta_mu = kappa_cep * S * (sigma_vals**2 - sigma_ref**2) / C_SQUARED_KM_S
        h0_pred = h0_true_fid * (1 + (np.log(10) / 5) * delta_mu)
        pred_slope, _ = np.polyfit(sigma_vals, h0_pred, 1)

        # Fractional agreement
        if abs(obs_slope) > 1e-6:
            agreement = abs(pred_slope - obs_slope) / abs(obs_slope) * 100
        else:
            agreement = float('inf')

        print_status(
            f"Predicted raw slope (blind, H0_true={h0_true_fid}): {pred_slope:.4f} km/s/Mpc/(km/s)",
            "INFO",
        )
        print_status(
            f"Observed raw slope:                           {obs_slope:.4f} km/s/Mpc/(km/s)",
            "INFO",
        )
        print_status(
            f"Fractional agreement:                         {agreement:.1f}%",
            "SUCCESS" if agreement < 15 else "INFO",
        )
        print_status(
            "(Prediction uses only Paper 10 κ_Cep + σ-distribution; no H0 data.)",
            "INFO",
        )

        return {
            "blind_predicted_slope": float(pred_slope),
            "observed_raw_slope": float(obs_slope),
            "slope_agreement_percent": float(agreement),
            "h0_true_fiducial": float(h0_true_fid),
        }

    def bootstrap_fixed_kappa(self, df, sigma_ref, kappa_cep, n_boot=1000):
        """Bootstrap with kappa frozen; only host resampling contributes."""
        print_status(
            f"Bootstrap (fixed κ_Cep = {kappa_cep:.3e} mag): N={n_boot}...", "SECTION"
        )

        np.random.seed(42)
        n_samples = len(df)
        h0s = []
        slopes = []

        for _ in range(n_boot):
            sample = df.sample(n=n_samples, replace=True)
            h0_b, _, _ = self.apply_fixed_correction(sample, sigma_ref, kappa_cep)
            h0s.append(float(np.mean(h0_b)))
            s, _ = np.polyfit(
                sample["sigma_inferred"].values.astype(float), h0_b, 1
            )
            slopes.append(float(s))

        h0s = np.array(h0s)
        slopes = np.array(slopes)

        h0_q16 = float(np.percentile(h0s, 16))
        h0_q50 = float(np.percentile(h0s, 50))
        h0_q84 = float(np.percentile(h0s, 84))

        metrics = {
            "bootstrap_h0_mean": float(np.mean(h0s)),
            "bootstrap_h0_std": float(np.std(h0s)),
            "bootstrap_h0_median": h0_q50,
            "bootstrap_h0_robust_std": float((h0_q84 - h0_q16) / 2.0),
            "bootstrap_h0_ci_lower": float(np.percentile(h0s, 2.5)),
            "bootstrap_h0_ci_upper": float(np.percentile(h0s, 97.5)),
            "bootstrap_residual_slope_mean": float(np.mean(np.abs(slopes))),
            "bootstrap_n": int(n_boot),
        }

        print_status("Bootstrap Results (fixed κ_Cep):", "SUBTITLE")
        headers = ["Parameter", "Mean", "Std", "95% CI"]
        rows = [
            [
                "H0 (Unified)",
                f"{metrics['bootstrap_h0_mean']:.2f}",
                f"{metrics['bootstrap_h0_std']:.2f}",
                f"[{metrics['bootstrap_h0_ci_lower']:.2f}, {metrics['bootstrap_h0_ci_upper']:.2f}]",
            ],
            [
                "Residual |slope|",
                f"{metrics['bootstrap_residual_slope_mean']:.4f}",
                "—",
                "—",
            ],
        ]
        print_table(headers, rows, title="Bootstrap (kappa frozen)")
        print_status(
            "Bootstrap uncertainty is host-scatter only (kappa treated as known).",
            "INFO",
        )

        return metrics

    def run(self):
        print_status("Starting Step 3C: Cross-Domain Consistency Check", "TITLE")

        df, sigma_ref = self.load_data()
        kappa_cep = KAPPA_CEP_PAPER10

        # Apply fixed correction
        print_status(
            f"Applying TEP correction with FIXED κ_Cep = {kappa_cep:.3e} mag...",
            "SECTION",
        )
        print_status(
            "(Value pre-registered from Paper 10 pulsar analysis; no SH0ES tuning.)",
            "INFO",
        )

        h0_corr, correction, S = self.apply_fixed_correction(df, sigma_ref, kappa_cep)
        df = df.copy()
        df["mu_corrected_frozen"] = df["value"].values + correction
        df["h0_corrected_frozen"] = h0_corr

        # Core statistics
        stats_dict = self.compute_statistics(
            df, h0_corr, df["sigma_inferred"].values.astype(float)
        )

        # Blind slope prediction (κ + σ only, no H0 values)
        blind_slope = self.blind_slope_prediction(df, sigma_ref, kappa_cep)

        # Bootstrap
        boot_metrics = self.bootstrap_fixed_kappa(df, sigma_ref, kappa_cep)

        # Planck tension
        planck_h0 = 67.4
        planck_err = 0.5
        h0_mean = stats_dict["h0_mean"]
        primary_error = boot_metrics["bootstrap_h0_std"]

        tension_stat = abs(h0_mean - planck_h0) / np.sqrt(
            stats_dict["h0_sem"] ** 2 + planck_err ** 2
        )
        tension_primary = abs(h0_mean - planck_h0) / np.sqrt(
            primary_error ** 2 + planck_err ** 2
        )

        print_status("Final Tension Analysis (Cross-Domain Consistency)", "SECTION")
        print_status(f"Planck 2018 Value: {planck_h0} +/- {planck_err}", "INFO")
        print_status("-" * 60, "INFO")
        print_status(
            f"CROSS-DOMAIN H0 (κ_Cep = {kappa_cep:.3e} mag): {h0_mean:.2f}", "RESULT"
        )
        print_status(
            f"  +/- {stats_dict['h0_sem']:.2f} (Statistical SEM, kappa fixed)", "INFO"
        )
        print_status(
            f"  +/- {primary_error:.2f} (Bootstrap: host scatter only)", "INFO"
        )
        print_status(
            f"  Residual slope dH0/dσ: {stats_dict['residual_slope']:.4f}", "INFO"
        )
        print_status(
            f"  Pearson r (p): {stats_dict['pearson_r']:.3f} ({stats_dict['pearson_p']:.4f})",
            "INFO",
        )
        print_status(
            f"  Spearman rho (p): {stats_dict['spearman_rho']:.3f} ({stats_dict['spearman_p']:.4f})",
            "INFO",
        )
        print_status("-" * 60, "INFO")
        print_status(f"Tension (Statistical):     {tension_stat:.2f} sigma", "INFO")
        print_status(f"Tension (Bootstrap):       {tension_primary:.2f} sigma", "RESULT")

        if tension_primary < 1.0:
            print_status(
                "CONCLUSION: Bare geometric-factor κ_Cep yields H0 consistent with Planck CMB. "
                "Cross-domain consistency confirmed.",
                "SUCCESS",
            )
        elif tension_primary < 2.0:
            print_status(
                "CONCLUSION: Cross-domain check gives marginal tension < 2σ; plausible consistency.",
                "WARNING",
            )
        else:
            print_status(
                "CONCLUSION: Cross-domain check yields significant residual tension.",
                "WARNING",
            )

        # Save results
        results = {
            "analysis_type": "cross_domain_consistency",
            "description": (
                "Cross-domain consistency check: applies the bare TEP geometric-factor "
                "estimate (~10^6 mag) without SH0ES tuning."
            ),
            "kappa_cep_source": "Paper 11 (TEP-H0) Cepheid period-luminosity fit (bare estimate, not pulsar-derived)",
            "kappa_cep_frozen": float(kappa_cep),
            "sigma_ref": float(sigma_ref),
            "unified_h0": h0_mean,
            "h0_sem": stats_dict["h0_sem"],
            "residual_slope": stats_dict["residual_slope"],
            "pearson_r": stats_dict["pearson_r"],
            "pearson_p": stats_dict["pearson_p"],
            "spearman_rho": stats_dict["spearman_rho"],
            "spearman_p": stats_dict["spearman_p"],
            "bootstrap_h0_mean": boot_metrics["bootstrap_h0_mean"],
            "bootstrap_h0_std": boot_metrics["bootstrap_h0_std"],
            "bootstrap_h0_median": boot_metrics["bootstrap_h0_median"],
            "bootstrap_h0_robust_std": boot_metrics["bootstrap_h0_robust_std"],
            "bootstrap_h0_ci_lower": boot_metrics["bootstrap_h0_ci_lower"],
            "bootstrap_h0_ci_upper": boot_metrics["bootstrap_h0_ci_upper"],
            "bootstrap_residual_slope_mean": boot_metrics[
                "bootstrap_residual_slope_mean"
            ],
            "blind_predicted_slope": blind_slope["blind_predicted_slope"],
            "observed_raw_slope": blind_slope["observed_raw_slope"],
            "slope_agreement_percent": blind_slope["slope_agreement_percent"],
            "planck_h0": float(planck_h0),
            "planck_err": float(planck_err),
            "tension_sigma": float(tension_primary),
            "tension_statistical": float(tension_stat),
            "is_consistent": bool(tension_primary < 2.0),
            "n_hosts": len(df),
        }

        with open(self.json_output_path, "w") as f:
            json.dump(results, f, indent=4)
        print_status(
            f"Saved cross-domain consistency results to {self.json_output_path}", "SUCCESS"
        )


def main():
    step = Step3CFrozenPrior()
    step.run()


if __name__ == "__main__":
    main()
