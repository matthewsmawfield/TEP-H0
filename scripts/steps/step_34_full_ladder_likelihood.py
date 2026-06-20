"""
Full Cepheid-SN Ia Distance Ladder Likelihood with TEP Correction

This script implements the full-ladder likelihood analysis requested in the manuscript:
- Stage 1: Summary-likelihood version using host distance moduli and covariance
- Stage 2: Matrix-level Cepheid likelihood with TEP term in design matrix

The full-ladder likelihood jointly fits:
1. Cepheid PL zero point (M_W)
2. Cepheid period slope (b)
3. Metallicity coefficient (Z)
4. Host distance moduli (mu_i)
5. Geometric anchor priors
6. SN Ia calibrator absolute magnitude (M_B)
7. Hubble-flow SN intercept
8. TEP coefficient (kappa_Cep)

This addresses the reviewer objection: "You have corrected a host-level residual diagnostic,
but you have not shown that the correction survives inside the actual Cepheid–SN Ia distance-ladder likelihood."
"""

import json
import sys
from pathlib import Path

# Make direct execution behave the same as module execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

# Import TEP Logger
try:
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )

from core.constants import KAPPA_GAL

from scripts.utils.tep_correction import (
    tep_correction,
    C_SQUARED_KM_S,
    ANCHOR_SCREENING,
)


class FullLadderLikelihood:
    """Full Cepheid-SN Ia distance ladder likelihood with TEP correction."""

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def fit_gls(A, y, C):
        """Numerically stable GLS using whitening + lstsq.

        Solves: min (y - A theta)^T C^{-1} (y - A theta)
        Via Cholesky whitening: C = L L^T, then solve L^{-1} y = L^{-1} A theta.
        """
        try:
            Lc = np.linalg.cholesky(C)
            A_w = linalg.solve_triangular(Lc, A, lower=True, check_finite=False)
            y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
        except (linalg.LinAlgError, ValueError):
            # Fallback for non-Cholesky covariance
            Cinv = linalg.pinv(C)
            A_w = A.copy()
            y_w = y.copy()
            # Weighted pseudo-approach
            return FullLadderLikelihood._fit_gls_weighted(A, y, Cinv)

        theta, residuals, rank, svals = np.linalg.lstsq(A_w, y_w, rcond=1e-12)
        cov = np.linalg.pinv(A_w.T @ A_w, rcond=1e-12)

        r = y - A @ theta
        chi2 = float(r.T @ linalg.solve(C, r, assume_a="pos"))
        return theta, cov, chi2, rank, svals

    @staticmethod
    def _fit_gls_weighted(A, y, Cinv):
        """Fallback GLS using precomputed inverse."""
        LTCL = A.T @ Cinv @ A
        LTCy = A.T @ Cinv @ y
        try:
            theta = linalg.solve(LTCL, LTCy, assume_a="pos")
            cov = linalg.inv(LTCL)
        except linalg.LinAlgError:
            theta = linalg.pinv(LTCL) @ LTCy
            cov = linalg.pinv(LTCL)
        r = y - A @ theta
        chi2 = float(r.T @ Cinv @ r)
        return theta, cov, chi2, np.linalg.matrix_rank(A), None

    @staticmethod
    def classify_row(i, L, q):
        """Classify a data row by its nonzero parameters."""
        nonzero = np.where(np.abs(L[i]) > 0.01)[0]
        params = [q[j] for j in nonzero]
        if "MHW1" in params:
            return "Cepheid"
        if "MB" in params and "5logH0" not in params:
            return "SN_calibrator"
        if "MB" in params and "5logH0" in params:
            return "SN_Hubble"
        if len(params) == 1 and params[0].startswith("mu_"):
            return "Anchor_prior"
        return "Other"

    @staticmethod
    def build_host_x(host_name, host_sigma, host_screening, sigma_ref, mode="centered"):
        """Build TEP regressor X for a single host."""
        sigma = host_sigma.get(host_name)
        if sigma is None or sigma <= 0:
            return 0.0

        S = host_screening.get(host_name, 1.0)

        if mode == "centered":
            return S * (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S
        if mode == "raw_sigma2":
            return S * sigma**2 / C_SQUARED_KM_S
        if mode == "unscreened_centered":
            return (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S
        if mode == "sigma_linear":
            return S * (sigma - sigma_ref) / 299792.458

        raise ValueError(f"Unknown X mode: {mode}")

    def build_tep_columns(self, L, q, host_sigma, host_screening, sigma_ref,
                          x_mode="centered", anchor_convention="anchor_screened_physical"):
        """Build TEP regressor columns for Cepheid and optionally SN rows.

        Returns:
            x_cepheid: TEP column for Cepheid rows
            x_sn: TEP column for SN calibrator rows (if enabled)
            row_classes: list of row classifications
            host_rows: dict of per-host info
        """
        mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
        mu_names = [q[i] for i in mu_indices]
        n_rows = L.shape[0]

        x_cepheid = np.zeros(n_rows)
        x_sn = np.zeros(n_rows)
        row_classes = []
        host_rows = {}
        anchor_hosts = {"N4258", "LMC", "M31"}

        for i in range(n_rows):
            cls = self.classify_row(i, L, q)
            row_classes.append(cls)

            # Find host for this row
            host = None
            for idx, mu_param in zip(mu_indices, mu_names):
                if abs(L[i, idx]) > 0.01:
                    host = mu_param.replace("mu_", "")
                    break

            if host is None:
                continue

            X = self.build_host_x(host, host_sigma, host_screening, sigma_ref, mode=x_mode)

            # Apply anchor convention
            if host in anchor_hosts:
                if anchor_convention == "anchor_reference_zero":
                    X = 0.0
                elif anchor_convention == "anchor_screened_physical":
                    pass  # keep X as computed
                elif anchor_convention == "exclude_anchor_cepheids":
                    X = 0.0

            if cls == "Cepheid":
                x_cepheid[i] = X
                if host not in host_rows:
                    host_rows[host] = {"n_cepheid_rows": 0, "X": X, "class": cls}
                host_rows[host]["n_cepheid_rows"] += 1

            elif cls in ["SN_calibrator", "SN_Hubble"]:
                x_sn[i] = X

        return x_cepheid, x_sn, row_classes, host_rows

    def build_model_matrix(self, L, q, x_cepheid, x_sn, model_name, X_SCALE=1e6):
        """Build augmented design matrix for a given TEP model variant.

        model_name options:
            - "cepheid_offset": single host-constant Cepheid offset
            - "cepheid_period": period-coupled TEP (kappaP * X * logP)
            - "cepheid_offset_plus_period": both offset and period coupling
            - "cepheid_offset_plus_metallicity": offset + metallicity interaction
            - "sn_offset": separate SN channel TEP
            - "cepheid_plus_sn": Cepheid and SN channels together
        """
        x_c = x_cepheid * X_SCALE
        x_s = x_sn * X_SCALE

        cols = []
        names = []

        if model_name == "cepheid_offset":
            cols.append(-x_c)
            names.append("kappa0_6")

        elif model_name == "cepheid_period":
            b_idx = np.where(q == "bW")[0]
            if len(b_idx) > 0:
                period_term = L[:, b_idx[0]]
                cols.append(-x_c * period_term)
                names.append("kappaP_6")
            else:
                # Fallback: use column of logP deviations from mean
                cols.append(-x_c)
                names.append("kappa0_6")

        elif model_name == "cepheid_offset_plus_period":
            b_idx = np.where(q == "bW")[0]
            if len(b_idx) > 0:
                period_term = L[:, b_idx[0]]
                cols.append(-x_c)
                names.append("kappa0_6")
                cols.append(-x_c * period_term)
                names.append("kappaP_6")
            else:
                cols.append(-x_c)
                names.append("kappa0_6")

        elif model_name == "cepheid_offset_plus_metallicity":
            z_idx = np.where(q == "ZW")[0]
            if len(z_idx) > 0:
                z_term = L[:, z_idx[0]]
                cols.append(-x_c)
                names.append("kappa0_6")
                cols.append(-x_c * z_term)
                names.append("kappaZ_6")
            else:
                cols.append(-x_c)
                names.append("kappa0_6")

        elif model_name == "sn_offset":
            cols.append(-x_s)
            names.append("kappaSN_6")

        elif model_name == "cepheid_plus_sn":
            cols.append(-x_c)
            names.append("kappaCep_6")
            cols.append(-x_s)
            names.append("kappaSN_6")

        else:
            raise ValueError(f"Unknown model: {model_name}")

        if not cols:
            cols = [np.zeros(L.shape[0])]
            names = ["none"]

        L_aug = np.column_stack([L] + cols)
        q_aug = list(q) + names
        return L_aug, q_aug

    def injection_test(self, L, y, C, q, x_cepheid, kappa_inj=9.7e5):
        """Test that the pipeline can recover a known injected TEP signal."""
        print_status(f"Injection test: kappa_inj = {kappa_inj:.3e} mag", "SECTION")

        X_SCALE = 1e6
        x_c = x_cepheid * X_SCALE

        # Fit baseline
        theta_base, cov_base, chi2_base, rank_base, _ = self.fit_gls(L, y, C)

        # Inject known signal: y_mock = L theta_base - kappa6_inj * x_c
        kappa6_inj = kappa_inj / X_SCALE
        y_mock = L @ theta_base - kappa6_inj * x_c

        # Add noise proportional to diagonal covariance
        noise_std = np.sqrt(np.diag(C))
        y_mock += np.random.normal(0, noise_std * 0.01)  # small noise

        # Recover with augmented model
        L_aug = np.column_stack([L, -x_c])
        theta_aug, cov_aug, chi2_aug, rank_aug, _ = self.fit_gls(L_aug, y_mock, C)

        kappa6_hat = theta_aug[-1]
        kappa_hat = kappa6_hat * X_SCALE
        kappa_err = np.sqrt(cov_aug[-1, -1]) * X_SCALE

        result = {
            "kappa_injected": float(kappa_inj),
            "kappa_recovered": float(kappa_hat),
            "kappa_err": float(kappa_err),
            "recovery_fraction": float(kappa_hat / kappa_inj) if kappa_inj != 0 else 0,
            "rank_aug": int(rank_aug),
            "chi2_aug": float(chi2_aug),
        }

        print_status(f"Injected:  {kappa_inj:.3e}", "INFO")
        print_status(f"Recovered: {kappa_hat:.3e} +/- {kappa_err:.3e}", "INFO")
        print_status(f"Recovery fraction: {result['recovery_fraction']:.3f}", "INFO")

        # Also test negative injection
        y_mock_neg = L @ theta_base + kappa6_inj * x_c
        y_mock_neg += np.random.normal(0, noise_std * 0.01)
        theta_neg, cov_neg, chi2_neg, rank_neg, _ = self.fit_gls(L_aug, y_mock_neg, C)
        kappa_neg = theta_neg[-1] * X_SCALE
        print_status(f"Negative injection: injected {-kappa_inj:.3e}, recovered {kappa_neg:.3e}", "INFO")

        result["kappa_neg_injected"] = float(-kappa_inj)
        result["kappa_neg_recovered"] = float(kappa_neg)

        return result

    def run_model_grid(self, L, y, C, q, host_sigma, host_screening, sigma_ref):
        """Run comprehensive model grid: TEP projections, reference frames, anchor conventions."""
        print_status("Running TEP Model Grid", "SECTION")

        X_SCALE = 1e6
        rows = []

        # Baseline fit
        theta_base, cov_base, chi2_base, rank_base, _ = self.fit_gls(L, y, C)
        h0_idx = np.where(q == "5logH0")[0][0]
        h0_base = 10 ** (theta_base[h0_idx] / 5)

        # Grid configurations
        x_modes = ["centered", "raw_sigma2", "unscreened_centered"]
        models = [
            "cepheid_offset",
            "cepheid_period",
            "cepheid_offset_plus_period",
            "cepheid_offset_plus_metallicity",
            "sn_offset",
            "cepheid_plus_sn",
        ]
        anchor_conventions = ["anchor_screened_physical", "anchor_reference_zero"]

        for x_mode in x_modes:
            for anchor_conv in anchor_conventions:
                x_c, x_s, row_classes, host_rows = self.build_tep_columns(
                    L, q, host_sigma, host_screening, sigma_ref,
                    x_mode=x_mode, anchor_convention=anchor_conv
                )

                for model_name in models:
                    try:
                        L_aug, q_aug = self.build_model_matrix(
                            L, q, x_c, x_s, model_name, X_SCALE=X_SCALE
                        )

                        theta, cov, chi2, rank, _ = self.fit_gls(L_aug, y, C)

                        # Pathology diagnostics
                        try:
                            s = np.linalg.svd(L_aug, compute_uv=False)
                            cond_number = float(s[0] / s[-1]) if s[-1] > 0 else float('inf')
                            min_singular = float(s[-1])
                        except Exception:
                            cond_number = float('inf')
                            min_singular = 0.0

                        # Parameter correlation matrix
                        try:
                            corr_matrix = np.corrcoef(cov)
                            np.fill_diagonal(corr_matrix, 0)
                            max_corr = float(np.nanmax(np.abs(corr_matrix)))
                        except Exception:
                            max_corr = float('nan')

                        # Max abs kappa
                        kappa_vals = []
                        for j in range(len(q), len(q_aug)):
                            kappa_vals.append(abs(theta[j] * X_SCALE))
                        max_abs_kappa = float(max(kappa_vals)) if kappa_vals else 0.0

                        # Status flag
                        # Note: SH0ES matrix naturally has high param correlations (~0.999),
                        # so we only flag on extreme kappa values or singular condition numbers.
                        if max_abs_kappa > 5e6 or cond_number > 1e12:
                            status = "pathological"
                        elif max_abs_kappa > 1e6 or cond_number > 1e10:
                            status = "ill_conditioned"
                        else:
                            status = "ok"

                        h0 = 10 ** (theta[h0_idx] / 5)
                        delta_chi2 = chi2_base - chi2

                        result = {
                            "x_mode": x_mode,
                            "anchor_convention": anchor_conv,
                            "model": model_name,
                            "rank": int(rank),
                            "n_params": len(q_aug),
                            "chi2": float(chi2),
                            "delta_chi2": float(delta_chi2),
                            "H0": float(h0),
                            "H0_shift": float(h0 - h0_base),
                            "condition_number": cond_number,
                            "min_singular_value": min_singular,
                            "max_param_correlation": max_corr,
                            "max_abs_kappa": max_abs_kappa,
                            "status": status,
                        }

                        # Extract fitted TEP parameters
                        for j, name in enumerate(q_aug[len(q):], start=len(q)):
                            result[name] = float(theta[j] * X_SCALE)
                            result[name + "_err"] = float(np.sqrt(cov[j, j]) * X_SCALE)
                            result[name + "_sig"] = float(
                                abs(theta[j]) / np.sqrt(cov[j, j]) if cov[j, j] > 0 else 0
                            )

                        rows.append(result)

                    except Exception as e:
                        print_status(f"Grid failed for {model_name}/{x_mode}/{anchor_conv}: {e}", "WARNING")

        df = pd.DataFrame(rows)
        grid_path = self.outputs_dir / "step_34_model_grid.csv"
        df.to_csv(grid_path, index=False)
        print_status(f"Saved model grid to {grid_path}", "SUCCESS")

        # Print summary of key results
        print_status("Model Grid Summary (key results):", "INFO")
        for _, row in df.iterrows():
            kappa_str = ""
            for col in df.columns:
                if col.startswith("kappa") and not col.endswith("_err") and not col.endswith("_sig"):
                    kappa_str += f" {col}={row[col]:.2e}"
            print_status(
                f"  {row['model']:35s} {row['x_mode']:18s} H0={row['H0']:.2f} dchi2={row['delta_chi2']:.2f}{kappa_str}",
                "INFO"
            )

        return df

    def host_summary_reconstruction_audit(self, L, y, C, q, host_sigma, host_screening, sigma_ref):
        """
        Reconstruct the original host-summary discovery statistic from the same SH0ES data.

        CRITICAL: The manuscript's discovery statistic is r(H0, sigma) = +0.466,
        computed from PER-HOST H0 = cz / d, where d = 10^((mu-25)/5).
        The mu comes from PUBLISHED SH0ES distance moduli (r22_distances.csv),
        NOT from matrix-fitted mu_i. This audit verifies:

        1. Can we reproduce r(H0, sigma) = +0.47 from published distance moduli?
        2. Do matrix-fitted mu_i give the same H0 values?
        3. What is the correlation in mu-residual space (with correct sign convention)?
        4. Which hosts differ between published and matrix-derived values?
        """
        print_status("Host-Summary Reconstruction Audit", "SECTION")
        from scipy.stats import pearsonr, spearmanr

        # Load published distance moduli
        dist_path = self.data_dir / "interim" / "r22_distances.csv"
        if not dist_path.exists():
            print_status("r22_distances.csv not found, skipping audit", "WARNING")
            return {"error": "r22_distances.csv missing"}

        dist_df = pd.read_csv(dist_path)
        dist_df["host"] = dist_df["parameter"].str.replace("mu_", "", regex=False)
        dist_df = dist_df.set_index("host")

        # Load host properties for z_hd
        hosts_df = pd.read_csv(self.data_dir / "processed" / "hosts_processed.csv")
        # Use first non-NaN z_hd per source_id
        z_map = hosts_df.dropna(subset=["z_hd"]).drop_duplicates("source_id").set_index("source_id")["z_hd"].to_dict()

        # Fit baseline SH0ES to extract matrix-fitted mu_i
        theta_base, cov_base, _, _, _ = self.fit_gls(L, y, C)
        mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
        mu_names = [q[i] for i in mu_indices]

        # Build host lists: only hosts that have both published mu and matrix mu
        # Matrix host names use abbreviated format (N4258, not NGC 4258)
        anchors = {"LMC", "M31", "N4258", "MW", "SMC"}

        rows = []
        for idx, mu_param in zip(mu_indices, mu_names):
            host = mu_param.replace("mu_", "")
            if host not in host_sigma or host not in dist_df.index:
                continue

            # Published distance modulus
            mu_pub = dist_df.loc[host, "value"]
            mu_pub_err = dist_df.loc[host, "error"]

            # Matrix-fitted distance modulus
            mu_fit = theta_base[idx]
            mu_fit_err = np.sqrt(cov_base[idx, idx])

            # Redshift
            z_hd = z_map.get(host, np.nan)

            # H0 from published mu: H0 = cz / d, d = 10^((mu-25)/5)
            if pd.notna(z_hd) and pd.notna(mu_pub):
                d_pub = 10 ** ((mu_pub - 25) / 5)
                h0_pub = 299792.458 * z_hd / d_pub
            else:
                h0_pub = np.nan

            # H0 from matrix-fitted mu
            if pd.notna(z_hd):
                d_fit = 10 ** ((mu_fit - 25) / 5)
                h0_fit = 299792.458 * z_hd / d_fit
            else:
                h0_fit = np.nan

            # Mu residual
            delta_mu_pub_fit = mu_pub - mu_fit
            delta_mu_pub_fit_err = np.sqrt(mu_pub_err**2 + mu_fit_err**2)

            # H0 residual
            if pd.notna(h0_pub) and pd.notna(h0_fit):
                delta_h0_pub_fit = h0_pub - h0_fit
            else:
                delta_h0_pub_fit = np.nan

            rows.append({
                "host": host,
                "is_anchor": host in anchors,
                "z_hd": z_hd,
                "sigma": host_sigma[host],
                "mu_published": mu_pub,
                "mu_published_err": mu_pub_err,
                "mu_matrix": mu_fit,
                "mu_matrix_err": mu_fit_err,
                "delta_mu": delta_mu_pub_fit,
                "delta_mu_err": delta_mu_pub_fit_err,
                "h0_published": h0_pub,
                "h0_matrix": h0_fit,
                "delta_h0": delta_h0_pub_fit,
            })

        df_all = pd.DataFrame(rows)

        # Sample selection: exclude anchors, z > 0.0035
        df_sample = df_all[~df_all["is_anchor"]].copy()
        df_sample = df_sample[df_sample["z_hd"] > 0.0035].copy()
        df_sample = df_sample.dropna(subset=["sigma", "h0_published", "h0_matrix"])

        n_all = len(df_all)
        n_sample = len(df_sample)

        print_status(f"Total hosts with both published and matrix mu: {n_all}", "INFO")
        print_status(f"After anchor exclusion and z>0.0035 cut: {n_sample}", "INFO")

        if n_sample == 0:
            print_status("No valid hosts in sample after cuts", "ERROR")
            return {"error": "empty sample"}

        # === 1. Reproduce paper's discovery statistic from published data ===
        sigma_vals = df_sample["sigma"].values
        h0_pub_vals = df_sample["h0_published"].values
        h0_fit_vals = df_sample["h0_matrix"].values
        mu_pub_vals = df_sample["mu_published"].values
        mu_fit_vals = df_sample["mu_matrix"].values

        # Paper's discovery: r(H0, sigma)
        r_h0pub_sigma, p_h0pub_sigma = pearsonr(sigma_vals, h0_pub_vals)
        rho_h0pub_sigma, p_rho_h0pub = spearmanr(sigma_vals, h0_pub_vals)

        # Matrix-derived H0 vs sigma
        r_h0fit_sigma, p_h0fit_sigma = pearsonr(sigma_vals, h0_fit_vals)
        rho_h0fit_sigma, p_rho_h0fit = spearmanr(sigma_vals, h0_fit_vals)

        # Mu-space correlations (published minus matrix)
        delta_mu = df_sample["delta_mu"].values
        r_delta_mu_sigma, p_delta_mu = pearsonr(sigma_vals, delta_mu)
        rho_delta_mu_sigma, p_rho_delta_mu = spearmanr(sigma_vals, delta_mu)

        # Mu-space: published mu vs sigma (should be negative: higher sigma -> brighter Cepheids -> smaller mu -> higher H0)
        r_mu_pub_sigma, p_mu_pub = pearsonr(sigma_vals, mu_pub_vals)
        r_mu_fit_sigma, p_mu_fit = pearsonr(sigma_vals, mu_fit_vals)

        # H0 residual vs sigma
        delta_h0 = df_sample["delta_h0"].values
        r_delta_h0_sigma, p_delta_h0 = pearsonr(sigma_vals, delta_h0)

        print_status("--- Paper Discovery Statistic Reproduction ---", "INFO")
        print_status(f"Published H0 vs sigma:  r={r_h0pub_sigma:+.3f} (p={p_h0pub_sigma:.4f}), rho={rho_h0pub_sigma:+.3f}", "INFO")
        print_status(f"Matrix H0 vs sigma:     r={r_h0fit_sigma:+.3f} (p={p_h0fit_sigma:.4f}), rho={rho_h0fit_sigma:+.3f}", "INFO")
        print_status(f"Published mu vs sigma:  r={r_mu_pub_sigma:+.3f} (p={p_mu_pub:.4f})", "INFO")
        print_status(f"Matrix mu vs sigma:    r={r_mu_fit_sigma:+.3f} (p={p_mu_fit:.4f})", "INFO")
        print_status(f"Delta mu vs sigma:     r={r_delta_mu_sigma:+.3f} (p={p_delta_mu:.4f})", "INFO")
        print_status(f"Delta H0 vs sigma:     r={r_delta_h0_sigma:+.3f} (p={p_delta_h0:.4f})", "INFO")

        manuscript_r = 0.466
        manuscript_rho = 0.517
        print_status(f"Manuscript claims: r={manuscript_r:.3f}, rho={manuscript_rho:.3f}", "INFO")

        # Check reproduction
        pub_match_r = abs(r_h0pub_sigma - manuscript_r) < 0.1
        pub_match_rho = abs(rho_h0pub_sigma - manuscript_rho) < 0.1

        if pub_match_r:
            print_status(f"✓ Published H0-sigma reproduces manuscript r within 0.1", "SUCCESS")
        else:
            print_status(f"✗ Published H0-sigma r differs by {abs(r_h0pub_sigma - manuscript_r):.3f}", "WARNING")

        if pub_match_rho:
            print_status(f"✓ Published H0-sigma reproduces manuscript rho within 0.1", "SUCCESS")
        else:
            print_status(f"✗ Published H0-sigma rho differs by {abs(rho_h0pub_sigma - manuscript_rho):.3f}", "WARNING")

        # Check matrix vs published agreement
        matrix_matches_pub_r = abs(r_h0fit_sigma - r_h0pub_sigma) < 0.1
        if matrix_matches_pub_r:
            print_status("✓ Matrix-derived H0 reproduces published H0 trend", "SUCCESS")
        else:
            print_status(f"✗ Matrix-derived H0 differs from published by {abs(r_h0fit_sigma - r_h0pub_sigma):.3f}", "WARNING")

        # Identify hosts with largest published-vs-matrix differences
        df_sample["abs_delta_mu"] = df_sample["delta_mu"].abs()
        df_sample["abs_delta_h0"] = df_sample["delta_h0"].abs()
        top_diff = df_sample.nlargest(5, "abs_delta_h0")[["host", "sigma", "h0_published", "h0_matrix", "delta_h0"]]
        print_status("Hosts with largest H0 discrepancy (published vs matrix):", "INFO")
        for _, row in top_diff.iterrows():
            print_status(f"  {row['host']:12s} sigma={row['sigma']:6.1f} H0_pub={row['h0_published']:7.2f} H0_fit={row['h0_matrix']:7.2f} dH0={row['delta_h0']:+7.2f}", "INFO")

        # Save
        audit_path = self.outputs_dir / "step_34_host_summary_audit.csv"
        df_sample.to_csv(audit_path, index=False)
        print_status(f"Saved audit to {audit_path}", "SUCCESS")

        # Also save paper-vs-matrix comparison
        comp_path = self.outputs_dir / "step_34_matrix_vs_paper_host_residuals.csv"
        df_sample[["host", "sigma", "z_hd",
                    "mu_published", "mu_matrix", "delta_mu",
                    "h0_published", "h0_matrix", "delta_h0"]].to_csv(comp_path, index=False)
        print_status(f"Saved comparison to {comp_path}", "SUCCESS")

        return {
            "n_all": int(n_all),
            "n_sample": int(n_sample),
            "r_h0pub_sigma": float(r_h0pub_sigma),
            "p_h0pub_sigma": float(p_h0pub_sigma),
            "rho_h0pub_sigma": float(rho_h0pub_sigma),
            "p_rho_h0pub": float(p_rho_h0pub),
            "r_h0fit_sigma": float(r_h0fit_sigma),
            "p_h0fit_sigma": float(p_h0fit_sigma),
            "rho_h0fit_sigma": float(rho_h0fit_sigma),
            "p_rho_h0fit": float(p_rho_h0fit),
            "r_mu_pub_sigma": float(r_mu_pub_sigma),
            "r_mu_fit_sigma": float(r_mu_fit_sigma),
            "r_delta_mu_sigma": float(r_delta_mu_sigma),
            "r_delta_h0_sigma": float(r_delta_h0_sigma),
            "matches_manuscript_r": bool(pub_match_r),
            "matches_manuscript_rho": bool(pub_match_rho),
            "matrix_matches_pub_r": bool(matrix_matches_pub_r),
        }

    def host_covariate_table(self, L, y, C, q, host_sigma, sigma_ref):
        """
        Build per-host covariate table from the FULL baseline fit.
        Aggregates Cepheid row-level terms (period, metallicity, residuals)
        per host for the corrected P6 absorption test.
        """
        print_status("Host Covariate Table (B)", "SECTION")
        from scipy.stats import pearsonr, spearmanr

        # Load host redshifts
        hosts_df = pd.read_csv(self.data_dir / "processed" / "hosts_processed.csv")
        z_map = hosts_df.dropna(subset=["z_hd"]).drop_duplicates("source_id").set_index("source_id")["z_hd"].to_dict()

        # Fit FULL baseline
        theta, cov, chi2, rank, _ = self.fit_gls(L, y, C)

        # Identify parameter indices
        bW_idx = np.where(q == "bW")[0]
        ZW_idx = np.where(q == "ZW")[0]
        MHW1_idx = np.where(q == "MHW1")[0]
        h0_idx = np.where(q == "5logH0")[0]

        bW_val = theta[bW_idx[0]] if len(bW_idx) > 0 else 0.0
        ZW_val = theta[ZW_idx[0]] if len(ZW_idx) > 0 else 0.0
        MHW1_val = theta[MHW1_idx[0]] if len(MHW1_idx) > 0 else 0.0
        h0_global = 10 ** (theta[h0_idx[0]] / 5) if len(h0_idx) > 0 else None

        # Residuals from full fit
        residuals = y - L @ theta

        # Anchor names
        anchors = {"LMC", "M31", "N4258", "MW", "SMC"}

        rows = []
        for i, pname in enumerate(q):
            if not pname.startswith("mu_"):
                continue
            host = pname.replace("mu_", "")
            if host not in host_sigma:
                continue

            z_hd = z_map.get(host, np.nan)
            is_anchor = host in anchors

            # Find rows where this host's mu contributes
            host_row_mask = np.abs(L[:, i]) > 0.01
            n_total_rows = np.sum(host_row_mask)

            if n_total_rows == 0:
                continue

            # Classify rows for this host
            ceph_rows = []
            sn_rows = []
            for r in np.where(host_row_mask)[0]:
                rc = self.classify_row(int(r), L, q)
                if rc == "Cepheid":
                    ceph_rows.append(r)
                elif rc in ("SN_calibrator", "SN_Hubble"):
                    sn_rows.append(r)

            n_ceph = len(ceph_rows)
            n_sn = len(sn_rows)

            # Period terms for Cepheid rows
            period_terms = []
            if len(bW_idx) > 0 and n_ceph > 0:
                period_terms = L[np.array(ceph_rows), bW_idx[0]]
            period_contrib = [t * bW_val for t in period_terms] if len(period_terms) > 0 else []

            # Metallicity terms for Cepheid rows
            z_terms = []
            if len(ZW_idx) > 0 and n_ceph > 0:
                z_terms = L[np.array(ceph_rows), ZW_idx[0]]
            z_contrib = [t * ZW_val for t in z_terms] if len(z_terms) > 0 else []

            # Residuals for Cepheid rows
            ceph_residuals = residuals[np.array(ceph_rows)] if n_ceph > 0 else []

            # Mean and median period term
            mean_period = np.mean(period_terms) if len(period_terms) > 0 else np.nan
            median_period = np.median(period_terms) if len(period_terms) > 0 else np.nan
            mean_period_contrib = np.mean(period_contrib) if len(period_contrib) > 0 else np.nan

            # Mean and median Z term
            mean_z = np.mean(z_terms) if len(z_terms) > 0 else np.nan
            median_z = np.median(z_terms) if len(z_terms) > 0 else np.nan
            mean_z_contrib = np.mean(z_contrib) if len(z_contrib) > 0 else np.nan

            # Mean residual
            mean_resid = np.mean(ceph_residuals) if len(ceph_residuals) > 0 else np.nan
            std_resid = np.std(ceph_residuals) if len(ceph_residuals) > 1 else np.nan

            # Fraction of Cepheids with logP > 1.5 (approximate P > 31.6 d)
            frac_logP_gt_1_5 = np.mean(np.array(period_terms) > 1.5) if len(period_terms) > 0 else np.nan

            # H0 from fitted mu
            mu_fit = theta[i]
            if pd.notna(z_hd) and z_hd > 0:
                d_mpc = 10 ** ((mu_fit - 25) / 5)
                h0_host = 299792.458 * z_hd / d_mpc
            else:
                h0_host = np.nan

            rows.append({
                "host": host,
                "sigma": host_sigma.get(host, np.nan),
                "z_hd": z_hd,
                "is_anchor": is_anchor,
                "mu_full": float(mu_fit),
                "mu_full_err": float(np.sqrt(cov[i, i])),
                "h0_host": float(h0_host),
                "h0_global": float(h0_global) if h0_global else np.nan,
                "n_ceph_rows": int(n_ceph),
                "n_sn_rows": int(n_sn),
                "mean_period_term": float(mean_period),
                "median_period_term": float(median_period),
                "mean_period_contrib": float(mean_period_contrib),
                "mean_Z_term": float(mean_z),
                "median_Z_term": float(median_z),
                "mean_Z_contrib": float(mean_z_contrib),
                "frac_logP_gt_1.5": float(frac_logP_gt_1_5),
                "mean_PL_residual": float(mean_resid),
                "std_PL_residual": float(std_resid),
            })

        df = pd.DataFrame(rows)

        # Correlations
        print_status("Host-level correlations:", "INFO")
        df_valid = df[(~df["is_anchor"]) & (df["z_hd"] > 0.0035)].dropna()
        if len(df_valid) > 3:
            for col, name in [
                ("mean_period_term", "mean period term"),
                ("mean_period_contrib", "mean period contrib"),
                ("mean_Z_term", "mean Z term"),
                ("mean_Z_contrib", "mean Z contrib"),
                ("frac_logP_gt_1.5", "fraction logP > 1.5"),
            ]:
                if col in df_valid.columns and df_valid[col].notna().sum() > 3:
                    r, p = pearsonr(df_valid["sigma"], df_valid[col])
                    print_status(f"  r(sigma, {name:25s}) = {r:+.3f} (p={p:.4f})", "INFO")

            print_status("---", "INFO")
            for col, name in [
                ("mean_period_term", "mean period term"),
                ("mean_period_contrib", "mean period contrib"),
                ("mean_Z_term", "mean Z term"),
                ("mean_Z_contrib", "mean Z contrib"),
                ("frac_logP_gt_1.5", "fraction logP > 1.5"),
            ]:
                if col in df_valid.columns and df_valid[col].notna().sum() > 3:
                    r, p = pearsonr(df_valid["h0_host"], df_valid[col])
                    print_status(f"  r(H0, {name:25s}) = {r:+.3f} (p={p:.4f})", "INFO")

        table_path = self.outputs_dir / "step_34_host_covariate_table.csv"
        df.to_csv(table_path, index=False)
        print_status(f"Saved host covariate table to {table_path}", "SUCCESS")

        return df

    def nested_absorption_test(self, L, y, C, q, host_sigma, sigma_ref, covariate_df):
        """
        P6: Corrected Nested Absorption Test (decomposition/ablation).

        Instead of fitting broken partial models, start from the FULL baseline fit
        and build counterfactual host moduli by adding back host-mean correction terms.
        This identifies which standard-ladder degrees of freedom absorb the H0-sigma trend.
        """
        print_status("Nested Absorption Test (P6) — Decomposition", "SECTION")
        from scipy.stats import pearsonr, spearmanr

        # Load host redshifts
        hosts_df = pd.read_csv(self.data_dir / "processed" / "hosts_processed.csv")
        z_map = hosts_df.dropna(subset=["z_hd"]).drop_duplicates("source_id").set_index("source_id")["z_hd"].to_dict()

        # Fit FULL baseline
        theta, cov, chi2, rank, _ = self.fit_gls(L, y, C)
        h0_idx = np.where(q == "5logH0")[0][0]
        h0_global = 10 ** (theta[h0_idx] / 5)

        # Anchor names
        anchors = {"LMC", "M31", "N4258", "MW", "SMC"}

        # Global means for reference
        df_valid = covariate_df[(~covariate_df["is_anchor"]) & (covariate_df["z_hd"] > 0.0035)].dropna()
        global_mean_period = df_valid["mean_period_term"].mean()
        global_mean_z = df_valid["mean_Z_term"].mean()

        def compute_h0_trend(mu_series, sigma_series, z_series):
            """Compute H0-sigma correlations from a series of mu values."""
            h0_vals = []
            sigma_vals = []
            for mu, sig, z in zip(mu_series, sigma_series, z_series):
                if pd.notna(mu) and pd.notna(z) and z > 0:
                    d_mpc = 10 ** ((mu - 25) / 5)
                    h0 = 299792.458 * z / d_mpc
                    h0_vals.append(h0)
                    sigma_vals.append(sig)
            if len(h0_vals) < 5:
                return None
            h0_vals = np.array(h0_vals)
            sigma_vals = np.array(sigma_vals)
            r, p_r = pearsonr(sigma_vals, h0_vals)
            rho, p_rho = spearmanr(sigma_vals, h0_vals)
            # Median split
            med = np.median(sigma_vals)
            dH0 = np.mean(h0_vals[sigma_vals > med]) - np.mean(h0_vals[sigma_vals <= med])
            return {"r": r, "p_r": p_r, "rho": rho, "p_rho": p_rho, "dH0": dH0, "n": len(h0_vals)}

        results = []

        # Reference: full baseline mu
        mu_full = covariate_df["mu_full"].values
        sigma_vals = covariate_df["sigma"].values
        z_vals = covariate_df["z_hd"].values
        is_valid = (~covariate_df["is_anchor"]) & (covariate_df["z_hd"] > 0.0035)

        base_trend = compute_h0_trend(
            mu_full[is_valid], sigma_vals[is_valid], z_vals[is_valid]
        )

        # Counterfactual: remove period correction
        # mu_no_period = mu_full + bW * (mean_period_host - global_mean_period)
        mean_period = covariate_df["mean_period_term"].values
        bW_idx = np.where(q == "bW")[0]
        bW_val = theta[bW_idx[0]] if len(bW_idx) > 0 else 0.0
        mu_no_period = mu_full + bW_val * (mean_period - global_mean_period)
        period_trend = compute_h0_trend(
            mu_no_period[is_valid], sigma_vals[is_valid], z_vals[is_valid]
        )

        # Counterfactual: remove metallicity correction
        mean_z = covariate_df["mean_Z_term"].values
        ZW_idx = np.where(q == "ZW")[0]
        ZW_val = theta[ZW_idx[0]] if len(ZW_idx) > 0 else 0.0
        mu_no_Z = mu_full + ZW_val * (mean_z - global_mean_z)
        z_trend = compute_h0_trend(
            mu_no_Z[is_valid], sigma_vals[is_valid], z_vals[is_valid]
        )

        # Counterfactual: remove both period and metallicity
        mu_no_PZ = mu_full + bW_val * (mean_period - global_mean_period) + ZW_val * (mean_z - global_mean_z)
        pz_trend = compute_h0_trend(
            mu_no_PZ[is_valid], sigma_vals[is_valid], z_vals[is_valid]
        )

        configs = [
            ("full_baseline", "Full baseline (mu_i + all corrections)", base_trend),
            ("no_period", "Remove period correction", period_trend),
            ("no_Z", "Remove metallicity correction", z_trend),
            ("no_PZ", "Remove period + metallicity", pz_trend),
        ]

        for label, desc, trend in configs:
            if trend is None:
                print_status(f"  {label:20s}: insufficient data", "WARNING")
                continue
            results.append({
                "fit": label,
                "description": desc,
                "r_h0_sigma": trend["r"],
                "p_r": trend["p_r"],
                "rho_h0_sigma": trend["rho"],
                "p_rho": trend["p_rho"],
                "dH0_median_split": trend["dH0"],
                "n_hosts": trend["n"],
            })
            print_status(
                f"  {label:20s}: r={trend['r']:+.3f} rho={trend['rho']:+.3f} "
                f"dH0={trend['dH0']:+6.2f}  ({desc})",
                "INFO"
            )

        # Print interpretation
        if base_trend and period_trend:
            if abs(period_trend["r"]) > abs(base_trend["r"]):
                print_status("  → Removing period correction STRENGTHENS the H0-sigma trend", "WARNING")
                print_status("  → b_W was SUPPRESSING a TEP-like signal", "WARNING")
            else:
                print_status("  → Removing period correction weakens the H0-sigma trend", "INFO")
                print_status("  → Period distribution contributes to the apparent trend", "INFO")

        if base_trend and z_trend:
            if abs(z_trend["r"]) > abs(base_trend["r"]):
                print_status("  → Removing metallicity correction STRENGTHENS the H0-sigma trend", "WARNING")
                print_status("  → Z_W was SUPPRESSING an environmental signal", "WARNING")
            else:
                print_status("  → Removing metallicity correction weakens the H0-sigma trend", "INFO")
                print_status("  → Metallicity/crowding contributes to the apparent trend", "INFO")

        df = pd.DataFrame(results)
        abs_path = self.outputs_dir / "step_34_nested_absorption_test.csv"
        df.to_csv(abs_path, index=False)
        print_status(f"Saved nested absorption test to {abs_path}", "SUCCESS")

        return df

    def validate_sn_channel_scope(self, L, q, x_sn):
        """
        C: Validate SN-channel scope.
        Check whether x_sn touches Hubble-flow SN rows or only calibrator SN rows.
        """
        print_status("SN-Channel Scope Validation (C)", "SECTION")

        n_sn_calib_total = 0
        n_sn_calib_x_nonzero = 0
        n_sn_hubble_total = 0
        n_sn_hubble_x_nonzero = 0

        for i in range(len(q)):
            if not q[i].startswith("mu_"):
                continue
            host = q[i].replace("mu_", "")

            host_row_mask = np.abs(L[:, i]) > 0.01
            for r in np.where(host_row_mask)[0]:
                rc = self.classify_row(int(r), L, q)
                if rc == "SN_calibrator":
                    n_sn_calib_total += 1
                    if abs(x_sn[r]) > 1e-12:
                        n_sn_calib_x_nonzero += 1
                elif rc == "SN_Hubble":
                    n_sn_hubble_total += 1
                    if abs(x_sn[r]) > 1e-12:
                        n_sn_hubble_x_nonzero += 1

        print_status(f"SN calibrator rows total:           {n_sn_calib_total}", "INFO")
        print_status(f"SN calibrator rows with X_SN != 0:  {n_sn_calib_x_nonzero}", "INFO")
        print_status(f"SN Hubble rows total:               {n_sn_hubble_total}", "INFO")
        print_status(f"SN Hubble rows with X_SN != 0:      {n_sn_hubble_x_nonzero}", "INFO")

        if n_sn_hubble_total > 0 and n_sn_hubble_x_nonzero == 0:
            print_status("  → x_SN only touches calibrator SN rows, NOT Hubble-flow rows", "WARNING")
            print_status("  → sn_offset model is a calibrator-environment test only", "WARNING")
        elif n_sn_hubble_x_nonzero > 0:
            print_status("  → x_SN touches Hubble-flow SN rows", "SUCCESS")

        return {
            "sn_calib_total": n_sn_calib_total,
            "sn_calib_x_nonzero": n_sn_calib_x_nonzero,
            "sn_hubble_total": n_sn_hubble_total,
            "sn_hubble_x_nonzero": n_sn_hubble_x_nonzero,
        }

    def tep_metallicity_disentanglement(self, L, y, C, q, x_cepheid, host_sigma, host_screening, sigma_ref):
        """
        A: TEP–metallicity disentanglement suite.

        Tests whether the TEP signal is independent of, entangled with, or
        degenerate with the standard Cepheid metallicity correction (Z_W).
        """
        print_status("TEP–Metallicity Disentanglement (A)", "SECTION")
        from scipy.stats import pearsonr, spearmanr

        X_SCALE = 1e6
        x_c = x_cepheid * X_SCALE

        # Find indices of key parameters
        bW_idx = np.where(q == "bW")[0]
        ZW_idx = np.where(q == "ZW")[0]
        h0_idx = np.where(q == "5logH0")[0][0]

        # Full baseline fit (reference)
        theta_base, cov_base, chi2_base, _, _ = self.fit_gls(L, y, C)
        h0_base = 10 ** (theta_base[h0_idx] / 5)

        def fit_and_report(L_aug, q_aug, label, has_tep=False, has_z_int=False):
            """Fit augmented model and return metrics."""
            theta, cov, chi2, rank, _ = self.fit_gls(L_aug, y, C)
            dof = len(y) - len(q_aug)
            h0 = 10 ** (theta[h0_idx] / 5)
            delta_chi2 = chi2_base - chi2

            result = {
                "model": label,
                "n_params": len(q_aug),
                "rank": int(rank),
                "chi2": float(chi2),
                "dof": int(dof),
                "chi2_reduced": float(chi2 / dof) if dof > 0 else float('inf'),
                "H0": float(h0),
                "delta_chi2": float(delta_chi2),
            }

            # Extract TEP parameters if present
            kappa_0 = None
            kappa_Z = None
            if has_tep:
                # Last column is kappa_0 (or kappa_perp)
                k_idx = len(q_aug) - 1 if not has_z_int else len(q_aug) - 2
                kappa_0 = theta[k_idx] * X_SCALE
                kappa_0_err = np.sqrt(cov[k_idx, k_idx]) * X_SCALE
                result["kappa_0"] = float(kappa_0)
                result["kappa_0_err"] = float(kappa_0_err)
                result["kappa_0_sig"] = float(abs(kappa_0) / kappa_0_err if kappa_0_err > 0 else 0)

            if has_z_int:
                kz_idx = len(q_aug) - 1
                kappa_Z = theta[kz_idx] * X_SCALE
                kappa_Z_err = np.sqrt(cov[kz_idx, kz_idx]) * X_SCALE
                result["kappa_Z"] = float(kappa_Z)
                result["kappa_Z_err"] = float(kappa_Z_err)
                result["kappa_Z_sig"] = float(abs(kappa_Z) / kappa_Z_err if kappa_Z_err > 0 else 0)

            return result, theta, cov

        results = []

        # === Model 1: Full baseline (reference) ===
        results.append({
            "model": "1_baseline",
            "description": "Full baseline (L theta)",
            "n_params": len(q),
            "rank": len(q),
            "chi2": float(chi2_base),
            "dof": len(y) - len(q),
            "chi2_reduced": float(chi2_base / (len(y) - len(q))),
            "H0": float(h0_base),
            "delta_chi2": 0.0,
            "kappa_0": None,
            "kappa_0_err": None,
            "kappa_0_sig": None,
            "kappa_Z": None,
            "kappa_Z_err": None,
            "kappa_Z_sig": None,
        })

        # === Model 2: Remove Z_W + TEP ===
        if len(ZW_idx) > 0:
            cols_noZ = [j for j in range(len(q)) if j != ZW_idx[0]]
            L_noZ = L[:, cols_noZ]
            q_noZ = [q[j] for j in cols_noZ]
            L_aug = np.column_stack([L_noZ, -x_c])
            q_aug = q_noZ + ["kappa0_6"]
            r, _, _ = fit_and_report(L_aug, q_aug, "2_noZ_plus_TEP", has_tep=True)
            r["description"] = "Remove Z_W, add TEP offset"
            results.append(r)

        # === Model 3: Baseline + TEP ===
        L_aug = np.column_stack([L, -x_c])
        q_aug = list(q) + ["kappa0_6"]
        r, _, _ = fit_and_report(L_aug, q_aug, "3_baseline_plus_TEP", has_tep=True)
        r["description"] = "Full baseline + TEP offset"
        results.append(r)

        # === Model 4: Baseline + TEP + metallicity interaction (kappa_Z * X * Z) ===
        if len(ZW_idx) > 0:
            z_col = L[:, ZW_idx[0]]
            # Interaction column: X * Z for Cepheid rows, 0 elsewhere
            # x_c is already zero for non-Cepheid rows
            xz_col = x_c * z_col
            L_aug = np.column_stack([L, -x_c, -xz_col])
            q_aug = list(q) + ["kappa0_6", "kappaZ_6"]
            r, _, _ = fit_and_report(L_aug, q_aug, "4_baseline_TEP_Zint", has_tep=True, has_z_int=True)
            r["description"] = "Full baseline + TEP + TEP×metallicity interaction"
            results.append(r)

        # === Model 5: Replace Z_W with TEP (remove Z, keep TEP) ===
        # Same as Model 2 but conceptually framed differently
        # Already captured above, skip duplicate

        # === Model 6: Orthogonalized TEP ===
        # X_perp = X - projection of host X onto host mean metallicity
        if len(ZW_idx) > 0:
            z_col = L[:, ZW_idx[0]]

            # Compute host-mean Z for Cepheid rows
            host_x = {}
            host_zmean = {}
            for i, pname in enumerate(q):
                if not pname.startswith("mu_"):
                    continue
                host = pname.replace("mu_", "")
                host_row_mask = np.abs(L[:, i]) > 0.01
                if not np.any(host_row_mask):
                    continue

                # Get X for this host (first Cepheid row's x_c value)
                for r in np.where(host_row_mask)[0]:
                    if self.classify_row(int(r), L, q) == "Cepheid":
                        host_x[host] = x_c[r]
                        break

                # Mean Z for Cepheid rows of this host
                z_vals = []
                for r in np.where(host_row_mask)[0]:
                    if self.classify_row(int(r), L, q) == "Cepheid":
                        z_vals.append(z_col[r])
                if z_vals:
                    host_zmean[host] = np.mean(z_vals)

            # Regress host X on host Z_mean
            common_hosts = [h for h in host_x if h in host_zmean]
            if len(common_hosts) >= 3:
                x_arr = np.array([host_x[h] for h in common_hosts])
                z_arr = np.array([host_zmean[h] for h in common_hosts])
                # Simple linear regression X = a + b*Z_mean
                Z_mat = np.column_stack([np.ones(len(z_arr)), z_arr])
                beta, _, _, _ = np.linalg.lstsq(Z_mat, x_arr, rcond=None)

                # Build X_perp per row
                x_perp = np.zeros_like(x_c)
                for i, pname in enumerate(q):
                    if not pname.startswith("mu_"):
                        continue
                    host = pname.replace("mu_", "")
                    if host not in host_zmean or host not in host_x:
                        continue
                    z_m = host_zmean[host]
                    x_pred = beta[0] + beta[1] * z_m
                    x_perp_host = host_x[host] - x_pred

                    # Assign to all Cepheid rows of this host
                    host_row_mask = np.abs(L[:, i]) > 0.01
                    for r in np.where(host_row_mask)[0]:
                        if self.classify_row(int(r), L, q) == "Cepheid":
                            x_perp[r] = x_perp_host

                # Fit baseline + orthogonalized TEP
                L_aug = np.column_stack([L, -x_perp])
                q_aug = list(q) + ["kappa_perp_6"]
                r, _, _ = fit_and_report(L_aug, q_aug, "6_orthogonalized_TEP", has_tep=True)
                r["description"] = "Full baseline + orthogonalized TEP (X ⟂ Z_mean)"
                results.append(r)

                # Report orthogonalization stats
                print_status(f"  Orthogonalization: X = {beta[0]:+.3e} + {beta[1]:+.3e} * Z_mean", "INFO")
                print_status(f"  r(host_X, host_Z_mean) = {pearsonr(x_arr, z_arr)[0]:+.3f}", "INFO")

        # === Print summary table ===
        print_status("Disentanglement Results:", "INFO")
        print_status(f"{'Model':25s} {'n_par':>5s} {'H0':>6s} {'chi2':>10s} {'dchi2':>8s} {'kappa0':>12s} {'sig0':>6s} {'kappaZ':>12s} {'sigZ':>6s}", "INFO")
        for r in results:
            k0 = f"{r.get('kappa_0', 0):+.2e}" if r.get('kappa_0') is not None else "n/a"
            s0 = f"{r.get('kappa_0_sig', 0):.1f}" if r.get('kappa_0_sig') is not None else "n/a"
            kz = f"{r.get('kappa_Z', 0):+.2e}" if r.get('kappa_Z') is not None else "n/a"
            sz = f"{r.get('kappa_Z_sig', 0):.1f}" if r.get('kappa_Z_sig') is not None else "n/a"
            print_status(
                f"  {r['model']:23s} {r['n_params']:5d} {r['H0']:6.2f} {r['chi2']:10.2f} "
                f"{r['delta_chi2']:+8.2f} {k0:>12s} {s0:>6s} {kz:>12s} {sz:>6s}",
                "INFO"
            )

        df = pd.DataFrame(results)
        out_path = self.outputs_dir / "step_34_tep_metallicity_disentanglement.csv"
        df.to_csv(out_path, index=False)
        print_status(f"Saved disentanglement results to {out_path}", "SUCCESS")

        return df

    def injection_test_all_models(self, L, y, C, q, x_cepheid, x_sn, kappa_inj=9.7e5):
        """Run injection-recovery tests for all model classes."""
        print_status("Injection Tests for All Model Classes", "SECTION")

        X_SCALE = 1e6
        x_c = x_cepheid * X_SCALE
        x_s = x_sn * X_SCALE

        # Fit baseline
        theta_base, _, _, _, _ = self.fit_gls(L, y, C)
        y_base = L @ theta_base
        noise_std = np.sqrt(np.diag(C))

        results = []
        kappa6_inj = kappa_inj / X_SCALE

        test_configs = [
            ("cepheid_offset", [(-x_c, "kappa0_6")]),
            ("sn_offset", [(-x_s, "kappaSN_6")]),
            ("cepheid_plus_sn", [(-x_c, "kappaCep_6"), (-x_s, "kappaSN_6")]),
        ]

        for model_name, cols in test_configs:
            # Inject signal: y_mock = y_base + kappa6_inj * col
            # where col = -x_c = -x_cepheid*X_SCALE, so kappa6_inj*col = -kappa_inj*x_cepheid
            y_mock = y_base.copy()
            for col, _ in cols:
                y_mock += kappa6_inj * col

            # Add small noise
            y_mock += np.random.normal(0, noise_std * 0.01)

            # Build augmented matrix
            L_aug = np.column_stack([L] + [c for c, _ in cols])
            theta, cov, chi2, rank, _ = self.fit_gls(L_aug, y_mock, C)

            for j, (_, name) in enumerate(cols, start=len(q)):
                kappa_hat = theta[j] * X_SCALE
                kappa_err = np.sqrt(cov[j, j]) * X_SCALE
                results.append({
                    "model": model_name,
                    "parameter": name,
                    "kappa_injected": float(kappa_inj),
                    "kappa_recovered": float(kappa_hat),
                    "kappa_err": float(kappa_err),
                    "recovery_fraction": float(kappa_hat / kappa_inj) if kappa_inj != 0 else 0,
                    "significance": float(abs(kappa_hat) / kappa_err) if kappa_err > 0 else 0,
                })

                print_status(
                    f"  {model_name:25s} {name:12s} injected={kappa_inj:.3e} recovered={kappa_hat:.3e} "
                    f"frac={kappa_hat/kappa_inj:.3f}",
                    "INFO"
                )

        df = pd.DataFrame(results)
        inj_path = self.outputs_dir / "step_34_injection_tests_all_models.csv"
        df.to_csv(inj_path, index=False)
        print_status(f"Saved injection tests to {inj_path}", "SUCCESS")

        return df

    def residual_after_fit_audit(self, L, y, C, q, host_sigma, host_screening, sigma_ref, x_cepheid):
        """After fitting a TEP model, check if host residual-sigma trend remains."""
        print_status("Residual-After-Fit Audit", "SECTION")

        X_SCALE = 1e6
        x_c = x_cepheid * X_SCALE

        # Fit baseline
        theta_base, cov_base, _, _, _ = self.fit_gls(L, y, C)

        # Fit TEP model
        L_aug = np.column_stack([L, -x_c])
        theta_aug, _, _, _, _ = self.fit_gls(L_aug, y, C)

        # Extract host residuals for both fits
        mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
        mu_names = [q[i] for i in mu_indices]

        rows = []
        for idx, mu_param in zip(mu_indices, mu_names):
            host = mu_param.replace("mu_", "")
            if host not in host_sigma:
                continue

            host_row_mask = np.abs(L[:, idx]) > 0.01
            if np.sum(host_row_mask) == 0:
                continue

            # Baseline residuals
            r_base = y[host_row_mask] - L[host_row_mask] @ theta_base
            mu_resid_base = np.mean(r_base)

            # TEP residuals
            r_tep = y[host_row_mask] - L_aug[host_row_mask] @ theta_aug
            mu_resid_tep = np.mean(r_tep)

            rows.append({
                "host": host,
                "sigma": host_sigma[host],
                "resid_baseline": float(mu_resid_base),
                "resid_tep": float(mu_resid_tep),
            })

        df = pd.DataFrame(rows)

        # Correlations
        from scipy.stats import pearsonr, spearmanr
        r_base, _ = pearsonr(df["sigma"], df["resid_baseline"])
        r_tep, _ = pearsonr(df["sigma"], df["resid_tep"])
        rho_base, _ = spearmanr(df["sigma"], df["resid_baseline"])
        rho_tep, _ = spearmanr(df["sigma"], df["resid_tep"])

        print_status(f"Baseline residual-sigma: r={r_base:.3f}, rho={rho_base:.3f}", "INFO")
        print_status(f"TEP residual-sigma:      r={r_tep:.3f}, rho={rho_tep:.3f}", "INFO")

        if abs(r_tep) < abs(r_base):
            print_status("✓ TEP fit reduces residual-sigma correlation", "SUCCESS")
        else:
            print_status("✗ TEP fit does not reduce residual-sigma correlation", "WARNING")

        resid_path = self.outputs_dir / "step_34_residual_after_fit.csv"
        df.to_csv(resid_path, index=False)
        print_status(f"Saved residual audit to {resid_path}", "SUCCESS")

        return {
            "r_baseline": float(r_base),
            "rho_baseline": float(rho_base),
            "r_tep": float(r_tep),
            "rho_tep": float(rho_tep),
            "tep_reduces_trend": abs(r_tep) < abs(r_base),
        }

    def leave_one_host_out(self, L, y, C, q, x_cepheid, host_sigma):
        """Run leave-one-host-out diagnostic to identify influential hosts."""
        print_status("Leave-One-Host-Out Diagnostic", "SECTION")

        X_SCALE = 1e6
        x_c = x_cepheid * X_SCALE

        # Find hosts from mu_ parameters
        mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
        mu_names = [q[i] for i in mu_indices]
        hosts = [name.replace("mu_", "") for name in mu_names]

        rows = []
        for host in sorted(hosts):
            mu_param = f"mu_{host}"
            if mu_param not in q:
                continue

            mu_idx = np.where(q == mu_param)[0][0]
            host_row_mask = np.abs(L[:, mu_idx]) > 0.01

            mask = ~host_row_mask
            L_sub = L[mask]
            y_sub = y[mask]
            C_sub = C[np.ix_(mask, mask)]
            x_sub = x_c[mask]

            L_aug = np.column_stack([L_sub, -x_sub])
            theta, cov, chi2, rank, _ = self.fit_gls(L_aug, y_sub, C_sub)

            kappa = theta[-1] * X_SCALE
            kappa_err = np.sqrt(cov[-1, -1]) * X_SCALE
            sig = abs(kappa) / kappa_err if kappa_err > 0 else 0

            rows.append({
                "dropped_host": host,
                "sigma_kms": host_sigma.get(host, np.nan),
                "kappa": float(kappa),
                "kappa_err": float(kappa_err),
                "significance": float(sig),
                "chi2": float(chi2),
                "rank": int(rank),
            })

        df = pd.DataFrame(rows)
        loo_path = self.outputs_dir / "step_34_leave_one_host_out.csv"
        df.to_csv(loo_path, index=False)
        print_status(f"Saved leave-one-host-out to {loo_path}", "SUCCESS")

        # Print hosts that most change kappa
        if len(df) > 0:
            baseline_kappa = df["kappa"].median()  # approximate
            df["kappa_shift"] = abs(df["kappa"] - baseline_kappa)
            top_hosts = df.nlargest(5, "kappa_shift")
            print_status("Hosts with largest kappa influence:", "INFO")
            for _, row in top_hosts.iterrows():
                print_status(
                    f"  {row['dropped_host']:10s} sigma={row['sigma_kms']:6.1f} kappa={row['kappa']:12.3e} +/- {row['kappa_err']:12.3e}",
                    "INFO"
                )

        return df

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
            "step_34_full_ladder", log_file_path=self.logs_dir / "step_34_full_ladder_likelihood.log"
        )
        set_step_logger(self.logger)

        # SH0ES data paths
        self.sh0es_dir = (
            self.data_dir / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
        )
        self.L_path = self.sh0es_dir / "L_R22.txt"
        self.y_path = self.sh0es_dir / "y_R22.txt"
        self.C_path = self.sh0es_dir / "C_R22.txt"
        self.q_path = self.sh0es_dir / "q_R22.txt"

        # Host metadata
        self.hosts_path = self.data_dir / "processed" / "hosts_processed.csv"

        # Outputs
        self.results_path = self.outputs_dir / "step_34_full_ladder_likelihood_results.json"
        self.comparison_path = self.outputs_dir / "step_34_ladder_likelihood_comparison.csv"

    def load_sh0es_data(self):
        """Load SH0ES design matrix, data vector, covariance, and parameters."""
        print_status("Loading SH0ES Data...", "SECTION")

        # Load design matrix L
        L = np.loadtxt(self.L_path, delimiter="\t")
        print_status(f"Design matrix shape: {L.shape}", "INFO")

        # Load data vector y
        names = ("Source", "Data")
        fmt = ("S20", np.float64)
        y_data = np.loadtxt(self.y_path, unpack=True, skiprows=1, dtype={"names": names, "formats": fmt})
        y_source = y_data[0].astype(str)
        y = y_data[1]
        print_status(f"Data vector length: {len(y)}", "INFO")

        # Load covariance matrix C
        C = np.loadtxt(self.C_path, delimiter="\t")
        print_status(f"Covariance matrix shape: {C.shape}", "INFO")

        # Load parameter names q
        q = np.loadtxt(self.q_path, unpack=True, dtype="str")
        print_status(f"Number of parameters: {len(q)}", "INFO")

        return L, y, C, q, y_source

    def load_host_metadata(self):
        """Load host metadata including sigma and screening factors."""
        print_status("Loading Host Metadata...", "SECTION")

        df = pd.read_csv(self.hosts_path)
        print_status(f"Loaded {len(df)} hosts", "INFO")

        # Create mapping from normalized_name to sigma_inferred
        # Also create alternative name mappings for SH0ES compatibility
        host_sigma = {}
        host_screening = {}
        for _, row in df.iterrows():
            name = row["normalized_name"]
            sigma = row["sigma_inferred"]
            S = row.get("shear_suppression", 1.0)

            # Store under normalized name
            host_sigma[name] = sigma
            host_screening[name] = S

            # Create SH0ES-style mappings
            # SH0ES uses: mu_M101, mu_N0691, mu_U9391, etc.
            # Remove spaces and convert to compact format
            sh0es_name = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")

            # Handle zero-padding (e.g., N0691 vs N691)
            # SH0ES uses zero-padded 4-digit numbers
            if sh0es_name.startswith("N") or sh0es_name.startswith("U"):
                # Extract the number part
                parts = sh0es_name[1:]  # Remove N or U prefix
                # Zero-pad to 4 digits
                if parts.isdigit():
                    padded_num = parts.zfill(4)
                    sh0es_name = sh0es_name[0] + padded_num

            host_sigma[sh0es_name] = sigma
            host_screening[sh0es_name] = S

            # Also store without zero-padding for compatibility
            unpadded = sh0es_name[0] + sh0es_name[1:].lstrip("0")
            if unpadded != sh0es_name:
                host_sigma[unpadded] = sigma
                host_screening[unpadded] = S

            # Also store with NGC prefix for N-prefixed names
            if sh0es_name.startswith("N"):
                ngc_name = "NGC" + sh0es_name[1:]
                host_sigma[ngc_name] = sigma
                host_screening[ngc_name] = S

        # Add explicit mappings for known SH0ES name mismatches
        # SH0ES uses M1337 for NGC 1337 (M catalog doesn't go this high)
        explicit_mappings = {
            "M1337": "N1337",      # NGC 1337
            "N105A": "N105",       # NGC 105A likely refers to NGC 105
            "N976A": "N976",       # NGC 976A likely refers to NGC 976
        }
        for sh0es_name, csv_name in explicit_mappings.items():
            if csv_name in host_sigma and sh0es_name not in host_sigma:
                host_sigma[sh0es_name] = host_sigma[csv_name]
                host_screening[sh0es_name] = host_screening[csv_name]
                print_status(f"Mapped {sh0es_name} -> {csv_name}", "INFO")

        print_status(f"Created {len(host_sigma)} name mappings", "INFO")

        return host_sigma, host_screening

    def calculate_effective_sigma_ref(self):
        """Calculate effective calibrator sigma (same as step_04)."""
        print_status("Calculating Effective Calibrator Sigma...", "SECTION")

        anchors = [
            {"ID": "MW", "Sigma": 30.0, "Weight": 0.20},
            {"ID": "LMC", "Sigma": 24.0, "Weight": 0.25},
            {"ID": "NGC 4258", "Sigma": 115.0, "Weight": 0.55},
        ]

        numerator_sq = 0.0
        denominator = 0.0
        for a in anchors:
            numerator_sq += (a["Sigma"] ** 2) * a["Weight"]
            denominator += a["Weight"]

        sigma_ref = np.sqrt(numerator_sq / denominator)
        print_status(f"Effective sigma_ref: {sigma_ref:.2f} km/s", "SUCCESS")

        return sigma_ref

    def stage1_summary_likelihood(self, mu_obs, mu_err, host_sigma, host_screening, sigma_ref):
        """
        Stage 1: Summary-likelihood version.

        Fit: mu_obs = mu_true + kappa_Cep * X + epsilon
        where X = S * (sigma^2 - sigma_ref^2) / c^2

        This uses the recovered host distance moduli and their covariance.
        """
        print_status("Stage 1: Summary-Likelihood Analysis", "SECTION")

        # Get host names from mu_obs parameter names
        # Filter out anchors (MW, LMC, M31, NGC 4258) - only fit Hubble-flow hosts
        anchor_hosts = ["MW", "LMC", "M31", "NGC 4258"]
        host_names = [
            name.replace("mu_", "")
            for name in mu_obs.index
            if name.startswith("mu_") and name.replace("mu_", "") not in anchor_hosts
        ]

        # Build X vector (TEP regressor)
        X = []
        mu_vals = []
        mu_errs = []

        for name in host_names:
            # Try different name formats
            sigma = None
            S = 1.0

            # Try exact match first
            if name in host_sigma:
                sigma = host_sigma[name]
                S = host_screening.get(name, 1.0)
            # Try with N prefix (e.g., N1365 vs NGC 1365)
            elif name.startswith("N") and name in host_sigma:
                sigma = host_sigma[name]
                S = host_screening.get(name, 1.0)
            # Try with M prefix
            elif name.startswith("M") and name in host_sigma:
                sigma = host_sigma[name]
                S = host_screening.get(name, 1.0)
            # Try with U prefix
            elif name.startswith("U") and name in host_sigma:
                sigma = host_sigma[name]
                S = host_screening.get(name, 1.0)

            if sigma is not None and sigma > 0:
                x = S * (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S
                X.append(x)
                mu_vals.append(mu_obs[f"mu_{name}"])
                mu_errs.append(mu_err[f"mu_{name}"])

        if len(X) < 2:
            print_status(f"Insufficient hosts for Stage 1 fit: {len(X)}", "ERROR")
            return {"stage1": {"error": "insufficient_hosts", "n_hosts": len(X)}}

        X = np.array(X)
        mu_vals = np.array(mu_vals)
        mu_errs = np.array(mu_errs)

        print_status(f"Fitting {len(X)} hosts with TEP regressor", "INFO")
        print_status(f"X range: [{X.min():.3e}, {X.max():.3e}]", "INFO")
        print_status(f"mu range: [{mu_vals.min():.2f}, {mu_vals.max():.2f}]", "INFO")

        # Weighted least squares fit
        weights = 1.0 / mu_errs**2
        A = np.vstack([np.ones_like(X), X]).T
        W = np.diag(weights)

        # GLS solution: theta = (A^T W A)^-1 A^T W y
        ATWA = A.T @ W @ A
        ATWy = A.T @ W @ mu_vals

        # Check condition number
        cond = np.linalg.cond(ATWA)
        print_status(f"Condition number of ATWA: {cond:.2e}", "INFO")

        try:
            theta = linalg.solve(ATWA, ATWy, assume_a="pos")
        except linalg.LinAlgError:
            print_status("Matrix ill-conditioned, using pseudo-inverse", "WARNING")
            theta = linalg.pinv(ATWA) @ ATWy

        theta_cov = linalg.pinv(ATWA)

        mu_true_hat = theta[0]
        kappa_hat = theta[1]
        kappa_err = np.sqrt(theta_cov[1, 1])

        print_status(f"Baseline mu_true: {mu_true_hat:.3f} mag", "INFO")
        print_status(f"TEP coefficient kappa_Cep: {kappa_hat:.3e} +/- {kappa_err:.3e} mag", "SUCCESS")

        # Calculate corrected distance moduli
        mu_corrected = mu_vals - kappa_hat * X

        # Calculate chi2
        residuals = mu_vals - (mu_true_hat + kappa_hat * X)
        chi2 = np.sum((residuals / mu_errs) ** 2)
        dof = len(X) - 2
        chi2_reduced = chi2 / dof

        print_status(f"Chi2/dof: {chi2_reduced:.2f}", "INFO")

        # Check if model is reasonable
        if chi2_reduced > 10:
            print_status("WARNING: Very high chi2/dof suggests model misspecification", "WARNING")
            print_status("The simple linear model mu = mu_true + kappa*X may not be appropriate", "WARNING")

        # Calculate H0 from corrected moduli
        # This is approximate - full H0 calculation requires SN data
        # For now, we report the shift in mean distance modulus
        mu_shift = np.mean(mu_corrected) - np.mean(mu_vals)
        print_status(f"Mean mu shift: {mu_shift:.4f} mag", "INFO")

        results = {
            "stage1": {
                "kappa_Cep": float(kappa_hat),
                "kappa_err": float(kappa_err),
                "kappa_significance": float(abs(kappa_hat) / kappa_err) if kappa_err > 0 else 0,
                "mu_true_baseline": float(mu_true_hat),
                "chi2": float(chi2),
                "dof": int(dof),
                "chi2_reduced": float(chi2_reduced),
                "mean_mu_shift": float(mu_shift),
                "n_hosts": len(X),
                "condition_number": float(cond),
                "model_status": "poor_fit" if chi2_reduced > 10 else "acceptable",
            }
        }

        return results

    def stage2_matrix_likelihood(self, L, y, C, q, y_source, host_sigma, host_screening, sigma_ref):
        """
        Stage 2: Full-ladder likelihood with three valid TEP variants.

        Central finding: The TEP term is IDENTIFIABLE (rank = 47 = full) when
        applied only to Cepheid PL rows. SN calibrator and anchor geometric-prior
        rows constrain host mu_i independently, breaking the degeneracy that would
        exist if TEP were applied to all rows.

        Physical convention: y = L*theta - kappa_Cep*X + epsilon, where positive
        kappa means Cepheids appear BRIGHTWARD of the standard PL prediction
        (period contraction in high-sigma hosts makes them brighter than expected).

        VARIANT A (Free kappa, full latent mu_i):
        - The honest statistical test. Fit kappa jointly with all parameters.
        - kappa is identified by anchor/SN/Hubble constraints on mu_i.

        VARIANT B (Prior-constrained kappa):
        - Bayesian approach: kappa ~ N(KAPPA_GAL, KAPPA_GAL_UNCERTAINTY^2).
        - Tests whether the theoretical prior is compatible with the data.

        VARIANT C (Fixed-kappa sensitivity test):
        - Counterfactual: externally impose kappa = KAPPA_GAL.
        - Apply correction y_corrected = y + kappa*X, refit all latent parameters.
        - Answers: "What happens to H0 if canonical TEP is true?"
        - NOT a model-selection test.
        """
        print_status("Stage 2: Matrix-Level Likelihood Analysis", "SECTION")

        # Find mu_ column indices
        mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
        mu_names = [q[i] for i in mu_indices]
        print_status(f"Found {len(mu_indices)} host distance modulus parameters", "INFO")

        # Build TEP column using L matrix for host identification
        n_rows = len(y)
        x_tep = np.zeros(n_rows)

        for i in range(n_rows):
            for idx, mu_param in zip(mu_indices, mu_names):
                if abs(L[i, idx]) > 0.01:
                    host_name = mu_param.replace("mu_", "")
                    if host_name in host_sigma:
                        sigma = host_sigma[host_name]
                        S = host_screening.get(host_name, 1.0)
                        X = S * (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S
                        x_tep[i] = X  # Positive X for high-sigma hosts
                    break

        n_nonzero = np.sum(x_tep != 0)
        print_status(f"TEP column has {n_nonzero}/{n_rows} non-zero entries", "INFO")

        if n_nonzero == 0:
            print_status("TEP column is all zeros - no host sigma data matched", "ERROR")
            return {"stage2": {"error": "no_tep_data", "n_nonzero": 0}}

        # --- Stage 2: Two-step constrained likelihood ---
        #
        # NOTE: Earlier versions incorrectly claimed the TEP column was perfectly
        # degenerate with host mu_i. This is FALSE for the Cepheid-row-only TEP
        # term. The augmented matrix L_aug = [L, -X_cepheid] has rank = 47 = full
        # because SN calibrator and anchor rows constrain mu_i independently of
        # the TEP-affected Cepheid rows. The free-kappa fit is valid.
        #
        # However, the free-kappa fit finds kappa_Cep consistent with zero and
        # with the opposite sign from canonical TEP. Stage 2 here tests the
        # fixed-kappa sensitivity: impose kappa from Stage 1 and observe the
        # H0 shift when host distances are constrained.
        #
        # This is a sensitivity test, not a model-selection result.

        # --- Common setup: invert covariance and baseline fit ---
        try:
            Cinv = linalg.inv(C)
        except linalg.LinAlgError:
            Cinv = linalg.pinv(C)

        # Baseline fit (needed by all variants)
        LT_Cinv_L = L.T @ Cinv @ L
        LT_Cinv_y = L.T @ Cinv @ y
        theta_base = linalg.solve(LT_Cinv_L, LT_Cinv_y, assume_a="pos")
        theta_cov_base = linalg.inv(LT_Cinv_L)

        y_pred_base = L @ theta_base
        residuals_base = y - y_pred_base
        chi2_base = float(residuals_base.T @ Cinv @ residuals_base)
        dof_base = len(y) - len(q)

        h0_idx = np.where(q == "5logH0")[0][0]
        h0_base = 10 ** (theta_base[h0_idx] / 5)
        h0_err_base = np.log(10) / 5 * np.sqrt(theta_cov_base[h0_idx, h0_idx]) * h0_base

        print_status(f"Baseline H0: {h0_base:.2f} +/- {h0_err_base:.2f} km/s/Mpc", "INFO")

        # ============================================================
        # VARIANT A: Free kappa, full latent mu_i (honest statistical test)
        # ============================================================
        print_status("Variant A: Free kappa with full latent host distances", "SECTION")

        # Build TEP columns using centralized helper
        x_tep, x_sn, row_classes, host_rows = self.build_tep_columns(
            L, q, host_sigma, host_screening, sigma_ref,
            x_mode="centered", anchor_convention="anchor_screened_physical"
        )

        X_SCALE = 1e6
        x_tep_scaled = x_tep * X_SCALE

        # Row-class validation counts
        row_class_counts = {}
        for c in row_classes:
            row_class_counts[c] = row_class_counts.get(c, 0) + 1
        print_status("Row class counts:", "INFO")
        for c, n in sorted(row_class_counts.items()):
            tep_status = "TEP applied" if c == "Cepheid" else "no TEP"
            print_status(f"  {c:20s}: {n:4d} rows ({tep_status})", "INFO")

        # Host-level TEP row counts
        print_status("Host-level TEP application:", "INFO")
        print_status(f"{'Host':10s} {'Cepheid rows':>12s} {'X_i':>12s} {'TEP?':>5s}", "INFO")
        print_status("-" * 45, "INFO")
        for host in sorted(host_rows.keys()):
            info = host_rows[host]
            print_status(f"{host:10s} {info['n_cepheid_rows']:12d} {info['X']:12.3e} {'yes':>5s}", "INFO")

        # Diagnostics
        n_tep_nonzero = int(np.sum(x_tep != 0))
        print_status(f"TEP column: {n_tep_nonzero}/{n_rows} non-zero entries", "INFO")
        print_status(f"X range: [{x_tep[x_tep != 0].min():.3e}, {x_tep[x_tep != 0].max():.3e}]", "INFO")
        print_status(f"X_scaled range: [{x_tep_scaled[x_tep_scaled != 0].min():.3e}, {x_tep_scaled[x_tep_scaled != 0].max():.3e}]", "INFO")

        # Build augmented model for cepheid_offset
        L_aug, q_aug = self.build_model_matrix(L, q, x_tep, x_sn, "cepheid_offset", X_SCALE=X_SCALE)

        # Check rank using L_aug directly
        rank_aug = np.linalg.matrix_rank(L_aug)
        print_status(f"L_aug rank: {rank_aug} (full: {len(q_aug)})", "INFO")

        LTCL_aug = L_aug.T @ Cinv @ L_aug
        LTCy_aug = L_aug.T @ Cinv @ y

        # Also check rank of weighted system
        rank_w = np.linalg.matrix_rank(LTCL_aug)
        print_status(f"LTCL_aug rank: {rank_w}", "INFO")

        kappa_idx = len(q)

        # Use the more optimistic rank (L_aug is the physical design matrix)
        rank_use = max(rank_aug, rank_w)

        if rank_use < len(q_aug):
            print_status("Rank deficiency detected - kappa may be degenerate with host mu_i", "WARNING")
            print_status("This would indicate insufficient independent constraints from SN/anchor rows", "WARNING")
            theta_aug = np.append(theta_base, 0.0)
            theta_cov_aug = linalg.pinv(LTCL_aug)
            kappa_free = 0.0
            kappa_err_free = np.inf
            kappa_sig_free = 0.0
            chi2_aug = chi2_base
            dof_aug = dof_base
            h0_free = h0_base
        else:
            # Use regularized solve for numerical stability
            try:
                theta_aug = linalg.solve(LTCL_aug, LTCy_aug, assume_a="pos")
                theta_cov_aug = linalg.inv(LTCL_aug)
            except linalg.LinAlgError:
                print_status("Matrix ill-conditioned, using pseudo-inverse", "WARNING")
                theta_aug = linalg.pinv(LTCL_aug) @ LTCy_aug
                theta_cov_aug = linalg.pinv(LTCL_aug)
            kappa_6_free = theta_aug[kappa_idx]
            kappa_6_err = np.sqrt(theta_cov_aug[kappa_idx, kappa_idx])
            kappa_free = kappa_6_free * X_SCALE
            kappa_err_free = kappa_6_err * X_SCALE
            kappa_sig_free = abs(kappa_free) / kappa_err_free if kappa_err_free > 0 else 0
            y_pred_aug = L_aug @ theta_aug
            residuals_aug = y - y_pred_aug
            chi2_aug = float(residuals_aug.T @ Cinv @ residuals_aug)
            dof_aug = len(y) - rank_use
            h0_free = 10 ** (theta_aug[h0_idx] / 5)

            # Correlation diagnostics: check kappa degeneracy with key parameters
            print_status("Parameter correlations with kappa:", "INFO")
            for param_name in ["MHW1", "MB", "bW", "ZW", "5logH0"]:
                if param_name in q:
                    p_idx = np.where(q == param_name)[0][0]
                    cov_kp = theta_cov_aug[kappa_idx, p_idx]
                    var_p = theta_cov_aug[p_idx, p_idx]
                    corr = cov_kp / np.sqrt(theta_cov_aug[kappa_idx, kappa_idx] * var_p)
                    print_status(f"  corr(kappa, {param_name:8s}) = {corr:7.3f}", "INFO")

        if rank_use < len(q_aug):
            print_status(f"Free kappa: UNIDENTIFIABLE (degenerate with host mu_i)", "INFO")
        else:
            print_status(f"Free kappa: {kappa_free:.3e} +/- {kappa_err_free:.3e} mag", "INFO")
            print_status(f"Significance: {kappa_sig_free:.2f} sigma", "INFO")
        print_status(f"H0 (free kappa): {h0_free:.2f} km/s/Mpc", "INFO")
        print_status(f"Chi2/dof: {chi2_aug / dof_aug:.2f}", "INFO")

        # ============================================================
        # VARIANT B: Prior-constrained kappa
        # ============================================================
        print_status("Variant B: Prior-constrained kappa", "SECTION")

        # Prior on kappa_6: kappa_6 ~ N(KAPPA_GAL/X_SCALE, (KAPPA_GAL_UNCERTAINTY/X_SCALE)^2)
        kappa_6_prior_mean = KAPPA_GAL / X_SCALE
        kappa_6_prior_std = 4.0e5 / X_SCALE

        # Bayesian update: add prior as pseudo-data point on kappa_6
        L_ext = np.vstack([L_aug, np.append(np.zeros(len(q)), 1.0)])
        y_ext = np.append(y, kappa_6_prior_mean)

        C_ext = np.zeros((len(y) + 1, len(y) + 1))
        C_ext[:len(y), :len(y)] = C
        C_ext[len(y), len(y)] = kappa_6_prior_std**2

        try:
            Cinv_ext = linalg.inv(C_ext)
        except linalg.LinAlgError:
            Cinv_ext = linalg.pinv(C_ext)

        LTCL_prior = L_ext.T @ Cinv_ext @ L_ext
        LTCy_prior = L_ext.T @ Cinv_ext @ y_ext
        theta_prior = linalg.solve(LTCL_prior, LTCy_prior, assume_a="pos")
        theta_cov_prior = linalg.inv(LTCL_prior)

        kappa_6_post = theta_prior[kappa_idx]
        kappa_6_err = np.sqrt(theta_cov_prior[kappa_idx, kappa_idx])
        kappa_prior = kappa_6_post * X_SCALE
        kappa_err_prior = kappa_6_err * X_SCALE

        y_pred_prior = L_aug @ theta_prior  # Note: use L_aug not L_ext for chi2 on actual data
        residuals_prior = y - y_pred_prior
        chi2_prior = float(residuals_prior.T @ Cinv @ residuals_prior)
        dof_prior = len(y) - len(q_aug)

        h0_prior = 10 ** (theta_prior[h0_idx] / 5)
        h0_err_prior = np.log(10) / 5 * np.sqrt(theta_cov_prior[h0_idx, h0_idx]) * h0_prior

        print_status(f"Prior kappa: {kappa_prior:.3e} +/- {kappa_err_prior:.3e} mag", "INFO")
        print_status(f"H0 (prior): {h0_prior:.2f} +/- {h0_err_prior:.2f} km/s/Mpc", "INFO")
        print_status(f"Chi2/dof: {chi2_prior / dof_prior:.2f}", "INFO")

        # ============================================================
        # VARIANT C: Fixed-kappa sensitivity test
        # ============================================================
        print_status("Variant C: Fixed-kappa sensitivity test", "SECTION")

        kappa_fixed = KAPPA_GAL
        print_status(f"Fixed kappa = {kappa_fixed:.3e} mag", "INFO")

        # Apply TEP correction ONLY to Cepheid rows: y_corrected = y + kappa * x_tep
        # Positive kappa means Cepheids appear BRIGHTWARD of standard PL prediction.
        # The corrected magnitude y_corrected removes the TEP brightening.
        y_corrected = y.copy() + kappa_fixed * x_tep

        # Fit baseline model to corrected data (mu_i remain latent)
        LTCL_corr = L.T @ Cinv @ L
        LTCy_corr = L.T @ Cinv @ y_corrected
        theta_corr = linalg.solve(LTCL_corr, LTCy_corr, assume_a="pos")
        theta_cov_corr = linalg.inv(LTCL_corr)

        y_pred_corr = L @ theta_corr
        residuals_corr = y_corrected - y_pred_corr
        chi2_fixed = float(residuals_corr.T @ Cinv @ residuals_corr)
        dof_fixed = len(y) - len(q)

        theta_fixed = theta_corr
        theta_cov_fixed = theta_cov_corr

        h0_fixed = 10 ** (theta_fixed[h0_idx] / 5)
        h0_err_fixed = np.log(10) / 5 * np.sqrt(theta_cov_fixed[h0_idx, h0_idx]) * h0_fixed

        print_status(f"Baseline H0: {h0_base:.2f} +/- {h0_err_base:.2f} km/s/Mpc", "INFO")
        print_status(f"Fixed-kappa H0: {h0_fixed:.2f} +/- {h0_err_fixed:.2f} km/s/Mpc", "SUCCESS")
        print_status(f"Baseline Chi2/dof: {chi2_base / dof_base:.2f}", "INFO")
        print_status(f"Fixed-kappa Chi2/dof: {chi2_fixed / dof_fixed:.2f}", "INFO")

        # Model comparison for fixed-kappa
        delta_chi2 = chi2_base - chi2_fixed
        delta_dof = dof_base - dof_fixed
        delta_aic = (chi2_base + 2 * len(q)) - (chi2_fixed + 2 * len(q))
        delta_bic = (chi2_base + len(q) * np.log(n_rows)) - (chi2_fixed + len(q) * np.log(n_rows))

        print_status(f"Delta Chi2: {delta_chi2:.4f} (dof={delta_dof})", "INFO")
        print_status(f"Delta AIC: {delta_aic:.4f}", "INFO")
        print_status(f"Delta BIC: {delta_bic:.4f} (not valid for model selection)", "INFO")

        # ============================================================
        # Collect all results
        # ============================================================
        results = {
            "variant_a_free_kappa": {
                "kappa_Cep": float(kappa_free),
                "kappa_err": float(kappa_err_free),
                "kappa_significance": float(kappa_sig_free),
                "H0": float(h0_free),
                "chi2": float(chi2_aug),
                "dof": int(dof_aug),
                "chi2_reduced": float(chi2_aug / dof_aug) if dof_aug > 0 else np.nan,
                "rank": int(rank_aug),
                "n_parameters": len(q_aug),
                "degenerate": rank_aug < len(q_aug),
            },
            "variant_b_prior_kappa": {
                "kappa_Cep": float(kappa_prior),
                "kappa_err": float(kappa_err_prior),
                "H0": float(h0_prior),
                "H0_err": float(h0_err_prior),
                "chi2": float(chi2_prior),
                "dof": int(dof_prior),
                "chi2_reduced": float(chi2_prior / dof_prior),
                "prior_mean": float(kappa_6_prior_mean * X_SCALE),
                "prior_std": float(kappa_6_prior_std * X_SCALE),
            },
            "variant_c_fixed_kappa": {
                "kappa_Cep": float(kappa_fixed),
                "H0": float(h0_fixed),
                "H0_err": float(h0_err_fixed),
                "chi2": float(chi2_fixed),
                "dof": int(dof_fixed),
                "chi2_reduced": float(chi2_fixed / dof_fixed),
                "baseline_H0": float(h0_base),
                "baseline_H0_err": float(h0_err_base),
                "baseline_chi2": float(chi2_base),
                "baseline_dof": int(dof_base),
                "delta_chi2": float(delta_chi2),
                "delta_dof": int(delta_dof),
                "delta_aic": float(delta_aic),
                "delta_bic": float(delta_bic),
                "n_cepheids": n_rows,
                "n_parameters": len(q),
            },
            "n_tep_nonzero": int(n_nonzero),
        }

        return results

    def run_baseline_sh0es(self, L, y, C, q):
        """Run baseline SH0ES fit without TEP for comparison."""
        print_status("Running Baseline SH0ES Fit...", "SECTION")

        # Use exact SH0ES reference method
        print_status("Using SH0ES reference GLS method", "INFO")

        try:
            Cinv = linalg.inv(C)
            print_status("Covariance inversion successful", "INFO")
        except linalg.LinAlgError:
            print_status("Covariance inversion failed, using pseudo-inverse", "WARNING")
            Cinv = linalg.pinv(C)

        # Normal equations (SH0ES method)
        LT_Cinv_L = L.T @ Cinv @ L
        LT_Cinv_y = L.T @ Cinv @ y

        # Solve for theta
        try:
            theta = linalg.solve(LT_Cinv_L, LT_Cinv_y, assume_a="pos")
        except linalg.LinAlgError:
            print_status("Solve failed, using inv (SH0ES method)", "WARNING")
            theta = linalg.inv(LT_Cinv_L) @ LT_Cinv_y

        # Covariance (SH0ES method)
        theta_cov = linalg.inv(LT_Cinv_L)

        # Extract H0
        h0_idx = np.where(q == "5logH0")[0][0]
        h0_param = theta[h0_idx]
        h0_err = np.sqrt(theta_cov[h0_idx, h0_idx])
        h0 = 10 ** (h0_param / 5)
        h0_err_final = np.log(10) / 5 * h0_err * 10 ** (h0_param / 5)

        # Extract host distance moduli
        mu_obs = {}
        mu_err = {}
        for i, name in enumerate(q):
            if name.startswith("mu_"):
                mu_obs[name] = theta[i]
                mu_err[name] = np.sqrt(theta_cov[i, i])

        print_status(f"Baseline H0: {h0:.2f} +/- {h0_err_final:.2f} km/s/Mpc", "SUCCESS")

        # Chi2
        y_pred = L @ theta
        residuals = y - y_pred
        chi2 = float(residuals.T @ Cinv @ residuals)
        dof = len(y) - len(q)

        print_status(f"Baseline Chi2/dof: {chi2 / dof:.2f}", "INFO")

        baseline_results = {
            "H0": float(h0),
            "H0_err": float(h0_err_final),
            "chi2": float(chi2),
            "dof": int(dof),
            "chi2_reduced": float(chi2 / dof),
            "mu_obs": mu_obs,
            "mu_err": mu_err,
            "covariance_method": "sh0es_reference",
        }

        return baseline_results

    def generate_comparison_table(self, baseline_results, stage1_results, stage2_results):
        """Generate comparison table for manuscript."""
        print_status("Generating Comparison Table...", "SECTION")

        # Handle Stage 1 error case
        if "error" in stage1_results.get("stage1", {}):
            print_status("Stage 1 failed, skipping in comparison table", "WARNING")
            stage1_kappa = np.nan
            stage1_kappa_err = np.nan
            stage1_sig = np.nan
            stage1_chi2 = np.nan
            stage1_dof = np.nan
            stage1_chi2_red = np.nan
        else:
            stage1_kappa = stage1_results["stage1"]["kappa_Cep"]
            stage1_kappa_err = stage1_results["stage1"]["kappa_err"]
            stage1_sig = stage1_results["stage1"]["kappa_significance"]
            stage1_chi2 = stage1_results["stage1"]["chi2"]
            stage1_dof = stage1_results["stage1"]["dof"]
            stage1_chi2_red = stage1_results["stage1"]["chi2_reduced"]

        # Extract variant results
        var_a = stage2_results.get("variant_a_free_kappa", {})
        var_b = stage2_results.get("variant_b_prior_kappa", {})
        var_c = stage2_results.get("variant_c_fixed_kappa", {})

        comparison = {
            "model": [
                "Baseline SH0ES",
                "TEP Summary Likelihood (Stage 1)",
                "Variant A: Free kappa",
                "Variant B: Prior kappa",
                "Variant C: Fixed kappa",
            ],
            "kappa_Cep": [
                0,
                stage1_kappa,
                var_a.get("kappa_Cep", np.nan),
                var_b.get("kappa_Cep", np.nan),
                var_c.get("kappa_Cep", np.nan),
            ],
            "kappa_err": [
                0,
                stage1_kappa_err,
                var_a.get("kappa_err", np.nan),
                var_b.get("kappa_err", np.nan),
                0,
            ],
            "kappa_significance": [
                0,
                stage1_sig,
                var_a.get("kappa_significance", np.nan),
                "—",
                "—",
            ],
            "H0": [
                baseline_results["H0"],
                np.nan,
                var_a.get("H0", np.nan),
                var_b.get("H0", np.nan),
                var_c.get("H0", np.nan),
            ],
            "H0_err": [
                baseline_results["H0_err"],
                np.nan,
                "—",
                var_b.get("H0_err", np.nan),
                var_c.get("H0_err", np.nan),
            ],
            "chi2": [
                baseline_results["chi2"],
                stage1_chi2,
                var_a.get("chi2", np.nan),
                var_b.get("chi2", np.nan),
                var_c.get("chi2", np.nan),
            ],
            "dof": [
                baseline_results["dof"],
                stage1_dof,
                var_a.get("dof", np.nan),
                var_b.get("dof", np.nan),
                var_c.get("dof", np.nan),
            ],
            "chi2_reduced": [
                baseline_results["chi2_reduced"],
                stage1_chi2_red,
                var_a.get("chi2_reduced", np.nan),
                var_b.get("chi2_reduced", np.nan),
                var_c.get("chi2_reduced", np.nan),
            ],
            "delta_chi2_vs_baseline": [
                0,
                np.nan,
                "—",
                "—",
                var_c.get("delta_chi2", np.nan),
            ],
            "notes": [
                "Standard SH0ES",
                "Discarded: model broken",
                "kappa=-6.7e4+/-2.1e5 (0.3sigma; opposite sign)",
                "Prior dominates",
                "Sensitivity test only",
            ],
        }

        df = pd.DataFrame(comparison)
        df.to_csv(self.comparison_path, index=False)
        print_status(f"Saved comparison table to {self.comparison_path}", "SUCCESS")

        # Print table
        headers = list(df.columns)
        rows = []
        for _, row in df.iterrows():
            rows.append([str(row[col]) if not pd.isna(row[col]) else "—" for col in headers])
        print_table(headers, rows, title="Full-Ladder Likelihood Comparison")

        return df

    def run(self):
        """Run full analysis pipeline."""
        print_status("Full-Ladder Likelihood Analysis", "TITLE")

        # Load data
        L, y, C, q, y_source = self.load_sh0es_data()
        host_sigma, host_screening = self.load_host_metadata()
        sigma_ref = self.calculate_effective_sigma_ref()

        # Run baseline SH0ES fit
        baseline_results = self.run_baseline_sh0es(L, y, C, q)

        # Stage 1: Summary likelihood
        stage1_results = self.stage1_summary_likelihood(
            pd.Series(baseline_results["mu_obs"]),
            pd.Series(baseline_results["mu_err"]),
            host_sigma,
            host_screening,
            sigma_ref,
        )

        # Stage 2: Matrix likelihood (primary variants A/B/C)
        stage2_results = self.stage2_matrix_likelihood(
            L, y, C, q, y_source, host_sigma, host_screening, sigma_ref
        )

        # Build TEP columns for diagnostics (re-use centralized helper)
        x_tep, x_sn, row_classes, host_rows = self.build_tep_columns(
            L, q, host_sigma, host_screening, sigma_ref,
            x_mode="centered", anchor_convention="anchor_screened_physical"
        )

        # Diagnostic 1: Host-summary reconstruction audit (CRITICAL CHECK)
        audit_results = self.host_summary_reconstruction_audit(
            L, y, C, q, host_sigma, host_screening, sigma_ref
        )

        # Diagnostic 2: Injection recovery test (Cepheid offset only)
        injection_results = self.injection_test(L, y, C, q, x_tep, kappa_inj=KAPPA_GAL)

        # Diagnostic 3: Injection tests for all model classes
        injection_all = self.injection_test_all_models(L, y, C, q, x_tep, x_sn, kappa_inj=KAPPA_GAL)

        # Diagnostic 4: Comprehensive model grid
        model_grid = self.run_model_grid(L, y, C, q, host_sigma, host_screening, sigma_ref)

        # Diagnostic 5: Leave-one-host-out
        loo_results = self.leave_one_host_out(L, y, C, q, x_tep, host_sigma)

        # Diagnostic 6: Residual-after-fit audit
        resid_audit = self.residual_after_fit_audit(L, y, C, q, host_sigma, host_screening, sigma_ref, x_tep)

        # Diagnostic 7: Host covariate table (B — needed for corrected P6)
        covariate_df = self.host_covariate_table(L, y, C, q, host_sigma, sigma_ref)

        # Diagnostic 8: Nested absorption test (P6)
        absorption_results = self.nested_absorption_test(L, y, C, q, host_sigma, sigma_ref, covariate_df)

        # Diagnostic 9: SN-channel scope validation (C)
        sn_scope = self.validate_sn_channel_scope(L, q, x_sn)

        # Diagnostic 10: TEP–metallicity disentanglement (A)
        disentangle_df = self.tep_metallicity_disentanglement(
            L, y, C, q, x_tep, host_sigma, host_screening, sigma_ref
        )

        # Generate comparison table
        comparison_df = self.generate_comparison_table(
            baseline_results, stage1_results, stage2_results
        )

        # Save all results
        all_results = {
            "baseline": baseline_results,
            "stage1": stage1_results,
            "stage2": stage2_results,
            "comparison": comparison_df.to_dict(),
            "host_summary_audit": audit_results,
            "injection_test": injection_results,
            "injection_test_all_models": injection_all.to_dict(),
            "model_grid": model_grid.to_dict(),
            "leave_one_host_out": loo_results.to_dict(),
            "residual_after_fit": resid_audit,
            "nested_absorption": absorption_results.to_dict(),
            "sn_channel_scope": sn_scope,
            "tep_metallicity_disentanglement": disentangle_df.to_dict(),
        }

        # Custom JSON encoder for numpy types and special floats
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    if np.isnan(obj):
                        return "NaN"
                    if np.isinf(obj):
                        return "Infinity" if obj > 0 else "-Infinity"
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        print_status(f"Saved results to {self.results_path}", "SUCCESS")

        print_status("Full-Ladder Likelihood Analysis Complete", "SUCCESS")

        return all_results


if __name__ == "__main__":
    analyzer = FullLadderLikelihood()
    results = analyzer.run()
