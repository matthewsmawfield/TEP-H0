
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import sys
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    # Add project root to path if needed
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

class Step4RobustnessChecks:
    r"""
    Step 4: Robustness Checks and Bivariate Analysis
    ================================================
    
    This step subjects the findings to rigorous statistical stress tests to ensure 
    that the observed environmental bias is physical and not an artifact.
    
    Tests Performed:
    1.  **Jackknife Analysis**: We iteratively remove one host galaxy at a time and 
        re-calculate the correlation strength ($r$) and significance ($p$). This 
        ensures that the signal is global and not driven by a single influential outlier.
    2.  **Bivariate Analysis**: We test the "Metallicity Hypothesis". Since Cepheid 
        luminosities depend on metallicity, and mass correlates with metallicity, 
        could the observed $H_0$-$\sigma$ trend be a disguised metallicity effect?
        We calculate **Partial Correlation Coefficients** to isolate the effect of 
        Velocity Dispersion while controlling for Metallicity.
        
    Statistical Tool: Partial Correlation
    $$ r_{xy.z} = \frac{r_{xy} - r_{xz}r_{yz}}{\sqrt{(1-r_{xz}^2)(1-r_{yz}^2)}} $$
    
    If $r(H_0, \sigma | Z)$ remains significant while $r(H_0, Z | \sigma)$ vanishes, 
    it proves the signal is kinematic (TEP), not chemical.
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.logs_dir = self.root_dir / "logs"
        
        self.figures_dir = self.results_dir / "figures"
        self.outputs_dir = self.results_dir / "outputs"
        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.public_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TEPLogger("step_4_robustness", log_file_path=self.logs_dir / "step_4_robustness.log")
        set_step_logger(self.logger)
        
        # Inputs
        self.stratified_path = self.outputs_dir / "stratified_h0.csv"
        self.cepheids_path = self.data_dir / "interim" / "reconstructed_shoes_cepheids.csv"
        self.tep_corrected_path = self.outputs_dir / "tep_corrected_h0.csv"
        self.tep_results_path = self.outputs_dir / "tep_correction_results.json"
        
        # Outputs
        self.stats_path = self.outputs_dir / "bivariate_stats.txt"
        self.covariance_results_path = self.outputs_dir / "covariance_robustness.json"
        self.oos_results_path = self.outputs_dir / "out_of_sample_validation.json"
        self.plot_path = self.figures_dir / "figure_02_bivariate_h0_sigma_metallicity.png"
        self.jackknife_plot_path = self.figures_dir / "supplement_02_jackknife_influence.png"

        self.flow_env_stats_path = self.outputs_dir / "flow_environment_robustness.txt"
        self.zcut_stats_path = self.outputs_dir / "redshift_cut_sensitivity.txt"

        self.h0_cov_path = self.outputs_dir / "h0_covariance.npy"
        self.h0_cov_labels_path = self.outputs_dir / "h0_covariance_labels.json"

    def _load_h0_covariance(self):
        if not self.h0_cov_path.exists() or not self.h0_cov_labels_path.exists():
            return None, None

        cov = np.load(self.h0_cov_path)
        with open(self.h0_cov_labels_path, 'r') as f:
            labels = json.load(f)
        cov = 0.5 * (cov + cov.T)
        return cov, labels

    def _subset_covariance(self, cov, cov_labels, target_labels):
        label_to_idx = {str(lbl): i for i, lbl in enumerate(cov_labels)}
        idx = []
        missing = []
        for lbl in target_labels:
            key = str(lbl)
            if key not in label_to_idx:
                missing.append(key)
                continue
            idx.append(label_to_idx[key])

        if missing:
            raise KeyError(f"Missing {len(missing)} labels in covariance: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        sub = cov[np.ix_(idx, idx)]
        sub = 0.5 * (sub + sub.T)
        
        # Hard assertions against silent ordering bugs
        assert len(target_labels) == sub.shape[0]
        assert sub.shape == (len(target_labels), len(target_labels))
        assert np.allclose(sub, sub.T)
        assert np.min(np.linalg.eigvalsh(sub)) > -1e-10
        
        return sub

    def _regularize_covariance(self, cov):
        cov_reg = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
        if not np.all(np.isfinite(cov_reg)):
            raise ValueError("Covariance matrix contains non-finite values")
        scale = float(np.trace(cov_reg) / cov_reg.shape[0])
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        cov_reg = cov_reg + np.eye(cov_reg.shape[0]) * (1e-12 * scale)
        min_eig = float(np.min(np.linalg.eigvalsh(cov_reg)))
        if min_eig <= 0:
            cov_reg = cov_reg + np.eye(cov_reg.shape[0]) * (-min_eig + 1e-12 * scale)
        return cov_reg

    def _gls_fit(self, X, y, cov):
        try:
            cov_reg = self._regularize_covariance(cov)
            cov_inv = np.linalg.inv(cov_reg)
            
            # Scale columns of X to ensure trace regularization doesn't overwhelmingly penalize small variance columns
            col_norms = np.linalg.norm(X, axis=0)
            col_norms[col_norms < 1e-10] = 1.0
            X_scaled = X / col_norms
            
            XtCi = X_scaled.T @ cov_inv
            fisher_scaled = XtCi @ X_scaled
            
            # Add small regularization for numerical stability using trace of SCALED fisher matrix
            reg = 1e-10 * np.trace(fisher_scaled) / fisher_scaled.shape[0]
            fisher_reg_scaled = fisher_scaled + reg * np.eye(fisher_scaled.shape[0])
            fisher_inv_scaled = np.linalg.inv(fisher_reg_scaled)
            
            beta_scaled = fisher_inv_scaled @ (XtCi @ y)
            
            # Unscale beta and covariance
            beta = beta_scaled / col_norms
            fisher_inv = fisher_inv_scaled / np.outer(col_norms, col_norms)
            
            return beta, fisher_inv
        except np.linalg.LinAlgError:
            # Return NaN values if matrix inversion fails
            beta = np.full(X.shape[1], np.nan)
            fisher_inv = np.full((X.shape[1], X.shape[1]), np.nan)
            return beta, fisher_inv

    def _covariance_aware_tests(self, df):
        cov, cov_labels = self._load_h0_covariance()
        if cov is None or cov_labels is None:
            return None

        target_labels = df['source_id'].astype(str).tolist()
        cov = self._subset_covariance(cov, cov_labels, target_labels)
        cov_reg = self._regularize_covariance(cov)

        sigma = df['sigma_inferred'].values.astype(float)
        y = df['h0_derived'].values.astype(float)
        n = len(y)

        x = sigma - np.mean(sigma)
        X = np.column_stack([np.ones(n), x])

        beta, beta_cov = self._gls_fit(X, y, cov_reg)
        slope = float(beta[1])
        slope_se = float(np.sqrt(beta_cov[1, 1]))
        t_slope = slope / slope_se if slope_se > 0 else np.nan
        p_slope = float(2 * (1 - stats.t.cdf(abs(t_slope), df=n - 2))) if np.isfinite(t_slope) else None

        # Parametric covariance simulation under null (intercept-only)
        X0 = np.ones((n, 1))
        beta0, _ = self._gls_fit(X0, y, cov_reg)
        mu0 = float(beta0[0])

        try:
            L = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(cov_reg)
            w = np.clip(w, 0.0, None)
            L = V @ np.diag(np.sqrt(w))

        n_sims = 20000
        rng = np.random.default_rng(42)
        z = rng.standard_normal((n, n_sims))
        y_sims = mu0 + np.dot(L, z)

        r_obs, _ = stats.pearsonr(sigma, y)
        rho_obs, _ = stats.spearmanr(sigma, y)

        # Compute null distribution correlations
        r_null = np.empty(n_sims)
        rho_null = np.empty(n_sims)
        for i in range(n_sims):
            r_null[i] = stats.pearsonr(sigma, y_sims[:, i])[0]
            rho_null[i] = stats.spearmanr(sigma, y_sims[:, i])[0]

        p_r_cov = float(np.mean(np.abs(r_null) >= abs(r_obs)))
        p_rho_cov = float(np.mean(np.abs(rho_null) >= abs(rho_obs)))

        # Effective N via Kish-like approximation using equicorrelation proxy
        d = np.sqrt(np.diag(cov_reg))
        denom = np.outer(d, d)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(denom > 0, cov_reg / denom, 0.0)
        avg_offdiag = float((np.sum(R) - n) / (n * (n - 1))) if n > 1 else 0.0
        n_eff = float(n / (1 + (n - 1) * max(0.0, avg_offdiag))) if n > 1 else float(n)

        return {
            "n": int(n),
            "n_eff_equicorr": n_eff,
            "avg_offdiag_corr": avg_offdiag,
            "gls_slope_per_kms": slope,
            "gls_slope_se": slope_se,
            "gls_slope_t": float(t_slope) if np.isfinite(t_slope) else None,
            "gls_slope_p": p_slope,
            "pearson_r": float(r_obs),
            "spearman_rho": float(rho_obs),
            "pearson_p_cov": p_r_cov,
            "spearman_p_cov": p_rho_cov,
            "n_sims": int(n_sims),
        }

    def _bayesian_model_comparison(self, df):
        """Bayesian evidence: TEP model vs null using the H0 likelihood.

        We compare two nested models for the uncorrected H0 data:

        Null:     H0_i = H0_0 + ε_i                          (k=1)
        TEP:      H0_i = H0_0 + β · x_i + ε_i               (k=2)
                  where x_i = S_i · (σ_i² − σ_ref²) / c²

        Primary calculation uses diagonal H0 uncertainties propagated from
        the SH0ES distance-modulus errors (consistent with the stratified
        analysis). A supplementary GLS-covariance calculation is reported
        as a cross-check. The χ² is:
            χ² = Σ_i (y_i − ŷ_i)² / σ_i²   (diagonal)
            χ² = (y − Xβ)^T Σ^{-1} (y − Xβ)  (full covariance)

        BIC = χ²_min + k · ln(n)
        ΔBIC = BIC_null − BIC_TEP  (positive → evidence for TEP)
        Bayes factor BF ≈ exp(ΔBIC / 2)

        Jeffreys scale (approximate):
            ΔBIC < 2 : not worth more than a bare mention
            2–6      : positive evidence
            6–10     : strong evidence
            >10      : very strong evidence
        """
        sigma = df["sigma_inferred"].values.astype(float)
        y = df["h0_derived"].values.astype(float)
        mu = df["value"].values.astype(float)
        mu_err = df["error"].values.astype(float)
        S = (
            df["shear_suppression"].values.astype(float)
            if "shear_suppression" in df.columns
            else np.ones(len(df))
        )
        n = len(y)

        sigma_ref = self._load_sigma_ref_val()
        if sigma_ref is None:
            sigma_ref = 87.17
            print_status("σ_ref missing from JSON; using standard fallback 87.17 km/s", "WARNING")

        # TEP regressor: S * (sigma^2 - sigma_ref^2) / c^2
        from scripts.utils.tep_correction import C_SQUARED_KM_S
        x = S * (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S

        # --- Diagonal H0 uncertainties from distance-modulus errors + peculiar velocity ---
        # sigma_H0^2 = (H0 * ln(10)/5 * sigma_mu)^2 + (vpecerr / d)^2
        distance_mpc = pd.to_numeric(df['distance_mpc'], errors='coerce').values
        vpecerr = pd.to_numeric(df.get('vpecerr', pd.Series(np.full(len(df), 250.0))), errors='coerce').fillna(250.0).values
        
        h0_diag_err_mu = y * ((np.log(10) / 5.0) * mu_err)
        h0_diag_err_vpec = vpecerr / distance_mpc
        h0_diag_err = np.sqrt(h0_diag_err_mu**2 + h0_diag_err_vpec**2)
        
        w = 1.0 / h0_diag_err**2
        W = np.diag(w)

        def _wls_fit(X, y, W):
            # Scale matrix for numerical stability with small regressors (1e-7 order)
            col_norms = np.linalg.norm(X, axis=0)
            col_norms[col_norms < 1e-10] = 1.0
            X_scaled = X / col_norms
            
            XWX_scaled = X_scaled.T @ W @ X_scaled
            XWy_scaled = X_scaled.T @ W @ y
            
            reg = 1e-10 * np.trace(XWX_scaled) / XWX_scaled.shape[0]
            XWX_reg_scaled = XWX_scaled + reg * np.eye(XWX_scaled.shape[0])
            
            try:
                beta_scaled = np.linalg.solve(XWX_reg_scaled, XWy_scaled)
                beta = beta_scaled / col_norms
            except np.linalg.LinAlgError:
                beta_scaled = np.linalg.lstsq(XWX_reg_scaled, XWy_scaled, rcond=None)[0]
                beta = beta_scaled / col_norms
            return beta

        # Null model (intercept only)
        X0 = np.ones((n, 1))
        beta0 = _wls_fit(X0, y, W)
        mu0 = float(beta0[0])
        chi2_null_diag = float(np.sum(w * (y - mu0)**2))

        # TEP model (intercept + regressor)
        X = np.column_stack([np.ones(n), x])
        beta = _wls_fit(X, y, W)
        mu_tep = float(beta[0])
        beta_x = float(beta[1])
        chi2_tep_diag = float(np.sum(w * (y - (mu_tep + beta_x * x))**2))

        k_null = 1
        k_tep = 2
        bic_null_diag = chi2_null_diag + k_null * np.log(n)
        bic_tep_diag = chi2_tep_diag + k_tep * np.log(n)
        delta_bic_diag = bic_null_diag - bic_tep_diag
        bf_diag = float(np.exp(delta_bic_diag / 2.0))

        # Implied κ_Cep from H0-space slope
        dH_dmu = -(np.log(10) / 5.0) * mu_tep
        kappa_wls = -beta_x / dH_dmu if abs(dH_dmu) > 1e-6 else float("nan")

        # --- Full-covariance cross-check (GLS) ---
        cov, cov_labels = self._load_h0_covariance()
        cov_results = None
        proj_results = None
        if cov is not None and cov_labels is not None:
            try:
                target_labels = df["source_id"].astype(str).tolist()
                cov_sub = self._subset_covariance(cov, cov_labels, target_labels)
                beta0_gls, _ = self._gls_fit(X0, y, cov_sub)
                beta_gls, _ = self._gls_fit(X, y, cov_sub)
                if not np.isnan(beta0_gls[0]) and not np.isnan(beta_gls[0]):
                    cov_inv = np.linalg.inv(
                        cov_sub + np.eye(n) * (1e-12 * np.trace(cov_sub) / n)
                    )
                    resid0_gls = y - float(beta0_gls[0])
                    chi2_null_gls = float(resid0_gls @ cov_inv @ resid0_gls)
                    resid_tep_gls = y - (beta_gls[0] + beta_gls[1] * x)
                    chi2_tep_gls = float(resid_tep_gls @ cov_inv @ resid_tep_gls)
                    bic_null_gls = chi2_null_gls + k_null * np.log(n)
                    bic_tep_gls = chi2_tep_gls + k_tep * np.log(n)
                    delta_bic_gls = bic_null_gls - bic_tep_gls
                    cov_results = {
                        "null_chi2": float(chi2_null_gls),
                        "tep_chi2": float(chi2_tep_gls),
                        "null_bic": float(bic_null_gls),
                        "tep_bic": float(bic_tep_gls),
                        "delta_bic": float(delta_bic_gls),
                    }
            except Exception:
                pass

            # --- Projected / host-contrast likelihood (primary) ---
            # P = I - (1^T C^{-1} 1)^{-1} * 1 * 1^T * C^{-1}
            # Projects out the shared calibration mode. In projected space:
            #   Null:  E[y_proj] = 0           (k=0)
            #   TEP:   E[y_proj] = β * x_proj  (k=1)
            # n_eff = n - 1. The environmental slope is tested after
            # removing the common calibration zero-point.
            try:
                ones = np.ones(n)
                denom = float(ones @ cov_inv @ ones)
                Pmat = np.eye(n) - np.outer(ones, ones @ cov_inv) / denom

                y_proj = Pmat @ y
                x_proj = Pmat @ x

                # Null in projected space (zero mean)
                chi2_null_proj = float(y_proj @ cov_inv @ y_proj)

                # TEP in projected space (slope only, no intercept)
                xPx = float(x_proj @ cov_inv @ x_proj)
                xPy = float(x_proj @ cov_inv @ y_proj)
                beta_proj = xPy / xPx if abs(xPx) > 1e-12 else 0.0
                chi2_tep_proj = chi2_null_proj - (xPy ** 2) / xPx

                n_eff = n - 1
                k_null_proj = 0
                k_tep_proj = 1
                bic_null_proj = chi2_null_proj + k_null_proj * np.log(n_eff)
                bic_tep_proj = chi2_tep_proj + k_tep_proj * np.log(n_eff)
                delta_bic_proj = bic_null_proj - bic_tep_proj
                bf_proj = float(np.exp(delta_bic_proj / 2.0))

                # Matched-parameter BIC for fair comparison with diagonal/full-covariance
                # In original-space parameter count: null k=1, TEP k=2
                bic_null_proj_matched = chi2_null_proj + k_null * np.log(n)
                bic_tep_proj_matched = chi2_tep_proj + k_tep * np.log(n)
                delta_bic_proj_matched = bic_null_proj_matched - bic_tep_proj_matched

                if delta_bic_proj < 2:
                    strength_proj = "not worth more than a bare mention"
                elif delta_bic_proj < 6:
                    strength_proj = "positive"
                elif delta_bic_proj < 10:
                    strength_proj = "strong"
                else:
                    strength_proj = "very strong"

                proj_results = {
                    "n_eff": int(n_eff),
                    "null_chi2": float(chi2_null_proj),
                    "tep_chi2": float(chi2_tep_proj),
                    "delta_chi2": float(chi2_null_proj - chi2_tep_proj),
                    "delta_bic": float(delta_bic_proj),
                    "bayes_factor": float(bf_proj),
                    "ln_bayes_factor": float(delta_bic_proj / 2.0),
                    "evidence_strength": strength_proj,
                    "beta_proj": float(beta_proj),
                    "null_bic_matched": float(bic_null_proj_matched),
                    "tep_bic_matched": float(bic_tep_proj_matched),
                    "delta_bic_matched": float(delta_bic_proj_matched),
                }
            except Exception as e:
                print_status(f"Projected-likelihood comparison failed: {e}", "WARNING")

        # Jeffreys strength (primary diagonal result)
        if delta_bic_diag < 2:
            strength = "not worth more than a bare mention"
        elif delta_bic_diag < 6:
            strength = "positive"
        elif delta_bic_diag < 10:
            strength = "strong"
        else:
            strength = "very strong"

        # Unified BIC comparison table: diagonal, full-covariance, projected contrast
        print_status("Bayesian Model Comparison", "SECTION")
        headers = ["Likelihood", "Null BIC", "TEP BIC", "ΔBIC", "Interpretation"]
        rows = [
            ["diagonal host scatter", f"{bic_null_diag:.1f}", f"{bic_tep_diag:.1f}", f"{delta_bic_diag:.1f}", "exploratory / host scatter"],
        ]
        if cov_results is not None:
            rows.append(
                [
                    "full covariance GLS slope",
                    f"{cov_results['null_bic']:.1f}",
                    f"{cov_results['tep_bic']:.1f}",
                    f"{cov_results['delta_bic']:.1f}",
                    "free intercept; matches contrast",
                ]
            )
        if proj_results is not None:
            rows.append(
                [
                    "projected host-contrast covariance",
                    f"{proj_results['null_bic_matched']:.1f}",
                    f"{proj_results['tep_bic_matched']:.1f}",
                    f"{proj_results['delta_bic_matched']:.1f}",
                    "primary environmental evidence",
                ]
            )
        print_table(headers, rows, title="Model Comparison")
        if proj_results is not None:
            print_status(
                f"Projected Δχ² = {proj_results['delta_chi2']:.2f}; ΔBIC = {proj_results['delta_bic_matched']:.2f} (headline)",
                "RESULT",
            )
        else:
            print_status(f"Δχ² (null − TEP) = {chi2_null_diag - chi2_tep_diag:.2f}", "INFO")
            print_status(f"ΔBIC = {delta_bic_diag:.2f}  ({strength} evidence for TEP)", "RESULT")
        tep_path = self.outputs_dir / "tep_correction_results.json"
        step3_kappa = None
        try:
            if tep_path.exists():
                with open(tep_path, "r") as f:
                    tep_results = json.load(f)
                step3_kappa = float(tep_results["optimal_kappa_cep"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            step3_kappa = None
        step3_kappa_text = f"{step3_kappa:.3e}" if step3_kappa is not None else "unavailable"
        print_status(
            f"WLS-implied κ_Cep = {kappa_wls:.3e} mag (Step 3 fitted {step3_kappa_text})",
            "INFO",
        )

        result = {
            "n": int(n),
            "sigma_ref": float(sigma_ref),
            "null_chi2": float(chi2_null_diag),
            "null_bic": float(bic_null_diag),
            "tep_chi2": float(chi2_tep_diag),
            "tep_bic": float(bic_tep_diag),
            "delta_chi2": float(chi2_null_diag - chi2_tep_diag),
            "delta_bic": float(delta_bic_diag),
            "bayes_factor": float(bf_diag),
            "ln_bayes_factor": float(delta_bic_diag / 2.0),
            "evidence_strength": strength,
            "kappa_wls_implied": float(kappa_wls),
            "wls_beta_x": float(beta_x),
            "wls_mu_tep": float(mu_tep),
        }
        if cov_results is not None:
            result["gls_crosscheck"] = cov_results
        if proj_results is not None:
            result["projected"] = proj_results
        return result

    def _load_sigma_ref_val(self):
        if not self.tep_results_path.exists():
            return None
        try:
            with open(self.tep_results_path, 'r') as f:
                d = json.load(f)
            return float(d.get('sigma_ref')) if 'sigma_ref' in d else None
        except Exception:
            return None

    def _fit_kappa(self, df, sigma_ref):
        from scripts.utils.tep_correction import tep_correction
        sigma = df['sigma_inferred'].values.astype(float)
        mu = df['value'].values.astype(float)
        v = df['velocity'].values.astype(float)
        S = df['shear_suppression'].values.astype(float) if 'shear_suppression' in df.columns else np.ones(len(df))

        def objective(params):
            kappa_cep = float(params[0])
            corr = tep_correction(sigma, sigma_ref, kappa_cep, S)
            mu_corr = mu + corr
            mu_fid = 5 * np.log10(v) + 25 - 5 * np.log10(70.0)
            delta_mu = mu_corr - mu_fid
            slope, _ = np.polyfit(sigma, delta_mu, 1)
            return float(slope * slope)

        res = minimize(
            objective,
            x0=[1.0e6],
            method='Nelder-Mead',
            options={'xatol': 10.0, 'fatol': 1e-6, 'maxiter': 500},
        )
        return float(res.x[0])

    def _apply_kappa(self, df, kappa_cep, sigma_ref):
        from scripts.utils.tep_correction import tep_correction
        sigma = df['sigma_inferred'].values.astype(float)
        mu = df['value'].values.astype(float)
        v = df['velocity'].values.astype(float)
        S = df['shear_suppression'].values.astype(float) if 'shear_suppression' in df.columns else np.ones(len(df))
        corr = tep_correction(sigma, float(sigma_ref), float(kappa_cep), S)
        mu_corr = mu + corr
        d_corr = 10 ** ((mu_corr - 25.0) / 5.0)
        return v / d_corr

    def perform_out_of_sample_validation(self):
        print_status("Initiating Out-of-Sample Validation (kappa_cep)", "SECTION")

        if not self.stratified_path.exists():
            print_status("Stratified data missing. Run Step 2 first.", "ERROR")
            return

        sigma_ref = self._load_sigma_ref_val()
        if sigma_ref is None:
            print_status("Could not load sigma_ref from Step 3 results. Run Step 3 first.", "ERROR")
            return

        df = pd.read_csv(self.stratified_path)
        required = ['sigma_inferred', 'value', 'velocity', 'h0_derived', 'source_id']
        df = df.dropna(subset=required).reset_index(drop=True)
        n = len(df)
        if n < 10:
            print_status("Insufficient sample size for out-of-sample validation.", "WARNING")
            return

        sigma_all = df['sigma_inferred'].values
        h0_raw = df['h0_derived'].values
        base_slope, _ = np.polyfit(sigma_all, h0_raw, 1)
        base_r, base_p = stats.pearsonr(sigma_all, h0_raw)
        base_rho, base_p_rho = stats.spearmanr(sigma_all, h0_raw)

        rng = np.random.default_rng(42)
        n_repeats = 200
        train_frac = 0.70

        test_slopes = []
        test_r = []
        test_rho = []
        test_h0_mean = []
        kappa_ceps = []

        n_train = int(np.round(train_frac * n))
        for _ in range(n_repeats):
            idx = rng.permutation(n)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]

            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            kappa_hat = self._fit_kappa(train, sigma_ref)
            h0_test = self._apply_kappa(test, kappa_hat, sigma_ref)

            slope_test, _ = np.polyfit(test['sigma_inferred'].values, h0_test, 1)
            r_test = stats.pearsonr(test['sigma_inferred'].values, h0_test)[0]
            rho_test = stats.spearmanr(test['sigma_inferred'].values, h0_test)[0]

            kappa_ceps.append(kappa_hat)
            test_slopes.append(float(slope_test))
            test_r.append(float(r_test))
            test_rho.append(float(rho_test))
            test_h0_mean.append(float(np.mean(h0_test)))

        kappa_ceps = np.array(kappa_ceps)
        test_slopes = np.array(test_slopes)
        test_r = np.array(test_r)
        test_rho = np.array(test_rho)
        test_h0_mean = np.array(test_h0_mean)

        loo_kappa_ceps = []
        loo_pred = np.empty(n)
        for i in range(n):
            train = df.drop(index=i)
            kappa_i = self._fit_kappa(train, sigma_ref)
            loo_kappa_ceps.append(kappa_i)
            # Apply to hold-out
            loo_pred[i] = float(self._apply_kappa(df.iloc[[i]], kappa_i, sigma_ref)[0])

        loo_kappa_ceps = np.array(loo_kappa_ceps)
        loo_slope, _ = np.polyfit(sigma_all, loo_pred, 1)
        loo_r, loo_p = stats.pearsonr(sigma_all, loo_pred)
        loo_rho, loo_p_rho = stats.spearmanr(sigma_all, loo_pred)

        planck_h0 = 67.4
        planck_err = 0.5
        loo_mean = float(np.mean(loo_pred))
        loo_sem = float(np.std(loo_pred, ddof=1) / np.sqrt(n)) if n > 1 else None
        loo_tension = float(abs(loo_mean - planck_h0) / np.sqrt((loo_sem if loo_sem is not None else 0.0) ** 2 + planck_err ** 2)) if loo_sem is not None else None

        headers = ["Validation", "Metric", "Value"]
        rows = [
            ["Baseline", "Slope dH0/dsigma", f"{base_slope:.4f}"],
            ["Baseline", "Pearson r (p)", f"{base_r:.3f} ({base_p:.4f})"],
            ["Baseline", "Spearman rho (p)", f"{base_rho:.3f} ({base_p_rho:.4f})"],
            ["Train/Test", "kappa_cep mean ± std", f"{np.mean(kappa_ceps):.3f} ± {np.std(kappa_ceps):.3f}"],
            ["Train/Test", "test slope median", f"{np.median(test_slopes):.4f}"],
            ["Train/Test", "test |r| median", f"{np.median(np.abs(test_r)):.3f}"],
            ["Train/Test", "test H0 mean (median)", f"{np.median(test_h0_mean):.2f}"],
            ["LOOCV", "kappa_cep mean ± std", f"{np.mean(loo_kappa_ceps):.3f} ± {np.std(loo_kappa_ceps):.3f}"],
            ["LOOCV", "pred slope dH0/dsigma", f"{loo_slope:.4f}"],
            ["LOOCV", "Pearson r (p)", f"{loo_r:.3f} ({loo_p:.4f})"],
            ["LOOCV", "Spearman rho (p)", f"{loo_rho:.3f} ({loo_p_rho:.4f})"],
            ["LOOCV", "H0 mean ± SEM", f"{loo_mean:.2f} ± {loo_sem:.2f}" if loo_sem is not None else f"{loo_mean:.2f}"],
            ["LOOCV", "Planck tension (sigma)", f"{loo_tension:.2f}" if loo_tension is not None else "-"],
        ]
        print_table(headers, rows, title="Out-of-Sample Validation Summary")

        payload = {
            "n": int(n),
            "sigma_ref": float(sigma_ref),
            "planck_h0": float(planck_h0),
            "planck_err": float(planck_err),
            "baseline": {
                "slope": float(base_slope),
                "pearson_r": float(base_r),
                "pearson_p": float(base_p),
                "spearman_rho": float(base_rho),
                "spearman_p": float(base_p_rho),
            },
            "train_test": {
                "n_repeats": int(n_repeats),
                "train_frac": float(train_frac),
                "kappa_cep_mean": float(np.mean(kappa_ceps)),
                "kappa_cep_std": float(np.std(kappa_ceps)),
                "test_slope_median": float(np.median(test_slopes)),
                "test_slope_q16": float(np.percentile(test_slopes, 16)),
                "test_slope_q84": float(np.percentile(test_slopes, 84)),
                "test_abs_pearson_r_median": float(np.median(np.abs(test_r))),
                "test_h0_mean_median": float(np.median(test_h0_mean)),
                "test_h0_mean_q16": float(np.percentile(test_h0_mean, 16)),
                "test_h0_mean_q84": float(np.percentile(test_h0_mean, 84)),
            },
            "loocv": {
                "kappa_cep_mean": float(np.mean(loo_kappa_ceps)),
                "kappa_cep_std": float(np.std(loo_kappa_ceps)),
                "pred_h0_mean": float(loo_mean),
                "pred_h0_sem": float(loo_sem) if loo_sem is not None else None,
                "pred_slope": float(loo_slope),
                "pearson_r": float(loo_r),
                "pearson_p": float(loo_p),
                "spearman_rho": float(loo_rho),
                "spearman_p": float(loo_p_rho),
                "tension_sigma": float(loo_tension) if loo_tension is not None else None,
            },
        }

        with open(self.oos_results_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print_status(f"Saved out-of-sample validation to {self.oos_results_path}", "SUCCESS")

    def calculate_partial_correlation(self, x, y, covar):
        """
        Calculate partial correlation between x and y controlling for covar.
        r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
        
        Returns: (r, p_value)
        """
        df = pd.DataFrame({'x': x, 'y': y, 'z': covar})
        corr = df.corr()
        r_xy = corr.loc['x', 'y']
        r_xz = corr.loc['x', 'z']
        r_yz = corr.loc['y', 'z']
        
        r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        # Calculate p-value using t-distribution
        n = len(x)
        df_val = n - 3  # degrees of freedom for partial correlation
        if abs(r_xy_z) >= 1:
            p_val = 0.0
        else:
            t_stat = r_xy_z * np.sqrt(df_val / (1 - r_xy_z**2))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))
        
        return r_xy_z, p_val

    def _partial_corr_residual_method(self, x, y, covars):
        covars = np.asarray(covars)
        if covars.ndim == 1:
            covars = covars.reshape(-1, 1)

        x = np.asarray(x)
        y = np.asarray(y)

        mask = np.isfinite(x) & np.isfinite(y)
        mask = mask & np.all(np.isfinite(covars), axis=1)

        x = x[mask]
        y = y[mask]
        z = covars[mask]

        n = len(x)
        k = z.shape[1]
        if n <= k + 3:
            return np.nan, np.nan, int(n)

        X = np.column_stack([np.ones(n), z])

        bx, *_ = np.linalg.lstsq(X, x, rcond=None)
        by, *_ = np.linalg.lstsq(X, y, rcond=None)
        rx = x - X @ bx
        ry = y - X @ by

        r = np.corrcoef(rx, ry)[0, 1]

        df_val = n - k - 2
        if not np.isfinite(r) or abs(r) >= 1:
            p_val = 0.0
        else:
            t_stat = r * np.sqrt(df_val / (1 - r**2))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))

        return float(r), float(p_val), int(n)

    def _correlation_suite(self, x, y, n_perm=5000, seed=42):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n = len(x)
        if n < 3:
            return {
                'n': int(n),
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'spearman_rho': np.nan,
                'spearman_p': np.nan,
                'perm_p': np.nan,
            }

        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)

        rng = np.random.default_rng(seed)
        r_perm = np.empty(n_perm)
        for i in range(n_perm):
            yp = rng.permutation(y)
            r_perm[i] = stats.pearsonr(x, yp)[0]
        perm_p = float(np.mean(np.abs(r_perm) >= abs(r_p)))

        return {
            'n': int(n),
            'pearson_r': float(r_p),
            'pearson_p': float(p_p),
            'spearman_rho': float(r_s),
            'spearman_p': float(p_s),
            'perm_p': float(perm_p),
        }

    def _velocity_from_z(self, z_series):
        c = 299792.458
        return c * pd.to_numeric(z_series, errors='coerce')

    def perform_redshift_cut_sensitivity(self):
        print_status("Redshift Cut Sensitivity Scan...", "SECTION")

        # To assess the redshift cut sensitivity, we must start from the full host catalog
        # because the stratified data already has the primary cut applied.
        hosts_path = self.data_dir / "processed" / "hosts_processed.csv"
        dists_path = self.data_dir / "interim" / "r22_distances.csv"
        if not hosts_path.exists() or not dists_path.exists():
            print_status("Raw host/distance data missing.", "ERROR")
            return None
            
        hosts_df = pd.read_csv(hosts_path)
        dists_df = pd.read_csv(dists_path)
        merged = pd.merge(dists_df, hosts_df, on="source_id", how="inner")
        
        # Calculate H0
        merged["distance_mpc"] = 10 ** ((merged["value"] - 25) / 5)
        merged["velocity"] = 299792.458 * merged["z_hd"]
        merged["h0_derived"] = merged["velocity"] / merged["distance_mpc"]
        valid = merged.dropna(subset=["h0_derived", "sigma_inferred", "m_b_corr"]).copy()
        valid["normalized_name"] = valid["normalized_name"].astype(str).str.strip()
        anchors = ["NGC 4258", "LMC", "SMC", "M 31", "MW"]
        df_full = valid[~valid["normalized_name"].isin(anchors)].copy()
        
        if "shear_suppression" not in df_full.columns:
            df_full["shear_suppression"] = 1.0

        cuts = [0.0, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.015]
        rows = []
        
        from scripts.steps.step_04_tep_correction import Step3TEPCorrection
        from scripts.utils.tep_correction import C_SQUARED_KM_S
        step3 = Step3TEPCorrection()
        # Use the same logger
        step3.logger = self.logger
        sigma_ref, _ = step3.calculate_effective_calibrator_sigma()
        
        for zcut in cuts:
            sub = df_full[(pd.to_numeric(df_full['z_hd'], errors='coerce') >= zcut)].copy()
            if len(sub) < 10:
                continue
                
            # Raw Correlation
            suite = self._correlation_suite(sub['sigma_inferred'], sub['h0_derived'])
            
            # TEP Correction
            try:
                kappa = step3.optimize_correction(sub, sigma_ref)
                sub_corr, _, _ = step3.apply_correction(sub, kappa, sigma_ref)
                unified_h0 = sub_corr['h0_corrected'].mean()
                
                # LOOCV
                n = len(sub)
                loocv_preds = []
                for i in range(n):
                    train = sub.drop(sub.index[i])
                    test = sub.iloc[[i]].copy()
                    k_train = step3.optimize_correction(train, sigma_ref)
                    S_test = test["shear_suppression"].values[0]
                    sig_test = test["sigma_inferred"].values[0]
                    mu_corr = test["value"].values[0] + S_test * k_train * (sig_test**2 - sigma_ref**2) / C_SQUARED_KM_S
                    d_corr = 10 ** ((mu_corr - 25) / 5)
                    loocv_preds.append(test["velocity"].values[0] / d_corr)
                loocv_h0 = np.mean(loocv_preds)
                
                # BIC
                x = sub_corr["shear_suppression"].values * (sub_corr["sigma_inferred"].values**2 - sigma_ref**2) / C_SQUARED_KM_S
                h0 = sub_corr["h0_derived"].values
                slope, intercept, r_val, p_val, std_err = stats.linregress(x, h0)
                rss_model = np.sum((h0 - (intercept + slope * x))**2)
                rss_null = np.sum((h0 - np.mean(h0))**2)
                bic_null = n * np.log(rss_null / n) + 1 * np.log(n)
                bic_model = n * np.log(rss_model / n) + 2 * np.log(n)
                delta_bic = bic_null - bic_model  # positive = evidence for TEP (consistent with _bayesian_model_comparison convention)
            except Exception as e:
                print_status(f"Error computing TEP stats for zcut {zcut}: {e}", "WARNING")
                kappa = np.nan
                unified_h0 = np.nan
                loocv_h0 = np.nan
                delta_bic = np.nan
                
            suite['zcut'] = float(zcut)
            suite['kappa_1e6'] = kappa / 1e6
            suite['h0_corr'] = unified_h0
            suite['loocv_h0'] = loocv_h0
            suite['delta_bic'] = delta_bic
            rows.append(suite)

        out = pd.DataFrame(rows)
        out.to_csv(self.zcut_stats_path, index=False)
        print_status(f"Saved redshift cut sensitivity results to {self.zcut_stats_path}", "SUCCESS")

        print_table(
            ["z_cut", "N", "Pearson r", "Spearman ρ", "Corr H0", "LOOCV H0", "κ_Cep (10^6)", "ΔBIC"],
            [[
                f"{r['zcut']:.4f}",
                str(int(r['n'])),
                f"{r['pearson_r']:.3f}",
                f"{r['spearman_rho']:.3f}",
                f"{r['h0_corr']:.2f}",
                f"{r['loocv_h0']:.2f}",
                f"{r['kappa_1e6']:.4f}",
                f"{r['delta_bic']:.2f}",
            ] for _, r in out.iterrows()],
            title="Redshift Cut Sensitivity (TEP Correction Stability)"
        )

        return out

    def perform_flow_environment_robustness(self, n_perm=5000, n_mc=5000):
        print_status("Flow/Environment Confound Robustness...", "SECTION")

        if not self.stratified_path.exists():
            print_status("Stratified data missing. Run Step 2 first.", "ERROR")
            return

        df = pd.read_csv(self.stratified_path)
        required = ['sigma_inferred', 'h0_derived', 'z_hd', 'distance_mpc']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print_status(f"Missing columns for flow/env robustness: {missing}", "ERROR")
            return

        df['z_hd'] = pd.to_numeric(df['z_hd'], errors='coerce')

        env_nmb = pd.to_numeric(df.get('tully_nmb', np.nan), errors='coerce').fillna(1.0)
        env_logpK = pd.to_numeric(df.get('tully_logpK', np.nan), errors='coerce')

        base_suite = self._correlation_suite(df['sigma_inferred'], df['h0_derived'], n_perm=n_perm)

        r_h0_sigma_z, p_h0_sigma_z, n_z = self._partial_corr_residual_method(
            df['h0_derived'],
            df['sigma_inferred'],
            df[['z_hd']].values,
        )

        r_h0_sigma_z_env, p_h0_sigma_z_env, n_z_env = self._partial_corr_residual_method(
            df['h0_derived'],
            df['sigma_inferred'],
            np.column_stack([df['z_hd'].values, env_nmb.values]),
        )

        df_env = df.copy()
        df_env['tully_logpK'] = env_logpK
        df_env = df_env.dropna(subset=['tully_logpK']).copy()
        r_h0_sigma_z_logpK, p_h0_sigma_z_logpK, n_z_logpK = self._partial_corr_residual_method(
            df_env['h0_derived'],
            df_env['sigma_inferred'],
            np.column_stack([df_env['z_hd'].values, df_env['tully_logpK'].values]),
        )

        alt_rows = []
        for zcol in ['z_hd', 'z_cmb', 'z_hel']:
            if zcol not in df.columns:
                continue
            vel = self._velocity_from_z(df[zcol])
            h0_alt = vel / pd.to_numeric(df['distance_mpc'], errors='coerce')
            suite = self._correlation_suite(df['sigma_inferred'], h0_alt, n_perm=n_perm)
            suite['z_definition'] = zcol
            alt_rows.append(suite)
        alt_df = pd.DataFrame(alt_rows) if alt_rows else pd.DataFrame()

        with open(self.flow_env_stats_path, 'w') as f:
            f.write("Flow / Environment Robustness\n")
            f.write("==============================\n\n")
            f.write(f"Baseline Pearson r(H0, Sigma): {base_suite['pearson_r']:.6f} (perm p={base_suite['perm_p']:.6f}) N={base_suite['n']}\n")
            f.write(f"Partial r(H0, Sigma | zHD): {r_h0_sigma_z:.6f} (p={p_h0_sigma_z:.6f}) N={n_z}\n")
            f.write(f"Partial r(H0, Sigma | zHD, Tully Nmb): {r_h0_sigma_z_env:.6f} (p={p_h0_sigma_z_env:.6f}) N={n_z_env}\n")
            f.write(f"Partial r(H0, Sigma | zHD, Tully logpK): {r_h0_sigma_z_logpK:.6f} (p={p_h0_sigma_z_logpK:.6f}) N={n_z_logpK}\n\n")

            if not alt_df.empty:
                f.write("Alternative velocity definitions (H0 = cz/d):\n")
                for _, r in alt_df.iterrows():
                    f.write(
                        f"  {r['z_definition']}: Pearson r={r['pearson_r']:.6f} (perm p={r['perm_p']:.6f}) N={int(r['n'])}\n"
                    )
                f.write("\n")

        print_status(f"Saved flow/environment robustness results to {self.flow_env_stats_path}", "SUCCESS")

        headers = ["Test", "Statistic", "p-value", "N"]
        rows = [
            ["Baseline r(H0,Sigma)", f"{base_suite['pearson_r']:.3f}", f"{base_suite['perm_p']:.4f}", str(base_suite['n'])],
            ["Partial r(H0,Sigma|z)", f"{r_h0_sigma_z:.3f}", f"{p_h0_sigma_z:.4f}", str(n_z)],
            ["Partial r(H0,Sigma|z,Nmb)", f"{r_h0_sigma_z_env:.3f}", f"{p_h0_sigma_z_env:.4f}", str(n_z_env)],
            ["Partial r(H0,Sigma|z,logpK)", f"{r_h0_sigma_z_logpK:.3f}", f"{p_h0_sigma_z_logpK:.4f}", str(n_z_logpK)],
        ]
        print_table(headers, rows, title="Flow + Environment Controls")

    def perform_jackknife_analysis(self):
        """Performs Jackknife robustness analysis."""
        print_status("Initiating Jackknife Analysis...", "SECTION")
        
        # We perform jackknife on the CORRELATION coefficient (Sigma vs H0)
        # using the pre-corrected data to show the signal is robust.
        if not self.stratified_path.exists():
            print_status("Stratified data missing. Run Step 2 first.", "ERROR")
            return
            
        df = pd.read_csv(self.stratified_path)

        # Check required columns
        required = ['sigma_inferred', 'h0_derived', 'normalized_name']
        if not all(col in df.columns for col in required):
            print_status(f"Missing columns for Jackknife. Have {df.columns.tolist()}", "ERROR")
            return

        # Ensure no NaNs
        df = df.dropna(subset=required)
        n = len(df)
        print_status(f"Loaded {n} hosts for Jackknife stability test.", "INFO")

        # 1. Baseline Correlations (Full Sample)
        # Pearson (parametric)
        r_base, p_base = stats.pearsonr(df['sigma_inferred'], df['h0_derived'])
        
        # Spearman (non-parametric, rank-based) - more robust to outliers
        rho_spearman, p_spearman = stats.spearmanr(df['sigma_inferred'], df['h0_derived'])
        
        # Bootstrap p-value for Pearson correlation (non-parametric significance)
        n_bootstrap = 10000
        np.random.seed(42)
        bootstrap_r = []
        sigma_vals = df['sigma_inferred'].values
        h0_vals = df['h0_derived'].values
        for _ in range(n_bootstrap):
            # Permute one variable to break correlation (null hypothesis)
            perm_idx = np.random.permutation(n)
            r_perm, _ = stats.pearsonr(sigma_vals[perm_idx], h0_vals)
            bootstrap_r.append(r_perm)
        bootstrap_r = np.array(bootstrap_r)
        # Two-tailed p-value: fraction of permuted r >= observed |r|
        p_bootstrap = np.mean(np.abs(bootstrap_r) >= abs(r_base))
        
        # Display correlation results
        headers = ["Test", "Statistic", "p-value", "Interpretation"]
        rows = [
            ["Pearson r", f"{r_base:.4f}", f"{p_base:.4f}", "Parametric"],
            ["Spearman ρ", f"{rho_spearman:.4f}", f"{p_spearman:.4f}", "Non-parametric (rank)"],
            ["Bootstrap p", "-", f"{p_bootstrap:.4f}", f"Permutation (N={n_bootstrap})"]
        ]

        cov_results = None
        try:
            cov_results = self._covariance_aware_tests(df)
            if cov_results is not None:
                rows.append(["GLS slope", f"{cov_results['gls_slope_t']:.3f}", f"{cov_results['gls_slope_p']:.4f}", "Covariance-aware Wald test"])
                rows.append(["Pearson p (cov)", f"{cov_results['pearson_r']:.4f}", f"{cov_results['pearson_p_cov']:.4f}", "Parametric MVN null"])
                rows.append(["Spearman p (cov)", f"{cov_results['spearman_rho']:.4f}", f"{cov_results['spearman_p_cov']:.4f}", "Parametric MVN null"])
        except (KeyError, ValueError, np.linalg.LinAlgError) as e:
            print_status(f"Covariance-aware tests failed: {e}", "WARNING")

        # Bayesian model comparison (TEP vs null)
        bayes_results = None
        try:
            bayes_results = self._bayesian_model_comparison(df)
            if bayes_results is not None and cov_results is not None:
                cov_results["bayesian_comparison"] = bayes_results
                rows.append([
                    f"Bayes factor",
                    f"{bayes_results['bayes_factor']:.1f}",
                    f"ΔBIC={bayes_results['delta_bic']:.1f}",
                    f"{bayes_results['evidence_strength']} evidence for TEP",
                ])
        except Exception as e:
            print_status(f"Bayesian comparison failed: {e}", "WARNING")

        print_table(headers, rows, title="Correlation Tests (H0 vs Sigma)")

        if cov_results is not None:
            with open(self.covariance_results_path, 'w') as f:
                json.dump(cov_results, f, indent=2)
            print_status(f"Saved covariance-aware results to {self.covariance_results_path}", "SUCCESS")
        
        if p_bootstrap < 0.01:
            print_status(f"Bootstrap p = {p_bootstrap:.4f} < 0.01: Correlation is HIGHLY SIGNIFICANT.", "SUCCESS")
        elif p_bootstrap < 0.05:
            print_status(f"Bootstrap p = {p_bootstrap:.4f} < 0.05: Correlation is SIGNIFICANT.", "SUCCESS")
        else:
            print_status(f"Bootstrap p = {p_bootstrap:.4f}: Correlation is NOT significant.", "WARNING")

        # 2. Jackknife Loop
        r_values = []
        influential_points = []
        
        for i in range(n):
            # Leave one out
            subset = df.drop(index=i)
            r, _ = stats.pearsonr(subset['sigma_inferred'], subset['h0_derived'])
            r_values.append(r)
            
            # Check influence (positive delta means removing host INCREASED r, negative means DECREASED r)
            # If removing a host kills the correlation (r -> 0), it's a driver.
            delta_r = r - r_base
            name = df.iloc[i]['normalized_name']
            influential_points.append({'Host': name, 'r_jack': r, 'delta_r': delta_r})

        # Convert to DF for table
        jack_df = pd.DataFrame(influential_points)
        jack_df = jack_df.sort_values('r_jack')
        
        # Determine Stability
        r_min = min(r_values)
        r_max = max(r_values)
        is_stable = (r_min > 0.3) and all(r > 0 for r in r_values)  # all jackknife r positive and above floor
        
        print_status(f"Jackknife Range: r in [{r_min:.4f}, {r_max:.4f}]", "RESULT")
        
        if r_min > 0.4:
            print_status("CONCLUSION: Correlation is ROBUST. No single host drives the trend.", "SUCCESS")
        else:
            print_status("CONCLUSION: Correlation is somewhat fragile.", "WARNING")
            
        # Display Most Influential Points (Top 3 reducers of r)
        # If removing them drops r significantly, they are supporting the trend strongly.
        print_table(["Host", "r (without)", "Delta r"], 
                   [[row['Host'], f"{row['r_jack']:.4f}", f"{row['delta_r']:+.4f}"] for _, row in jack_df.head(3).iterrows()],
                   title="Most Influential Hosts (Supports Trend)")

        # Plotting Jackknife Influence
        print_status("Generating Jackknife influence plot...", "PROCESS")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'light_blue': '#4b6785', 'green': '#4a2650'}
            
        plt.figure(figsize=(14, 9))

        # Sort by delta_r for cleaner plot
        jack_df = jack_df.sort_values('delta_r')
        
        bar_colors = [colors['accent'] if d >= 0 else colors['blue'] for d in jack_df['delta_r']]
        plt.bar(jack_df['Host'], jack_df['delta_r'], color=bar_colors, alpha=0.8)
        plt.axhline(0, color=colors['dark'], linewidth=1.5)
        plt.xticks(rotation=90, fontsize=10)
        plt.ylabel(r"$\Delta r = r_{\rm leave-one-out} - r_{\rm full}$")
        plt.title("Jackknife Influence: No single host drives the correlation")
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        # plt.savefig(self.jackknife_plot_path, dpi=300)
        # print_status(f"Saved Jackknife plot to {self.jackknife_plot_path}", "SUCCESS")
        plt.close()

        # Copy to public
        # public_jack = self.public_figures_dir / "supplement_02_jackknife_influence.png"
        # shutil.copy(self.jackknife_plot_path, public_jack)
        # print_status(f"Copied Jackknife plot to {public_jack}", "SUCCESS")

    def perform_bivariate_analysis(self):
        """Performs bivariate analysis (H0 vs Sigma + Metallicity)."""
        print_status("Initiating Bivariate Analysis...", "SECTION")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'light_blue': '#4b6785', 'green': '#4a2650'}
            
        if not self.stratified_path.exists():
            print_status("Stratified data missing. Run Step 2 first.", "ERROR")
            return
            
        df = pd.read_csv(self.stratified_path)

        # We need metallicity. In this dataset, we use Mass as a proxy if Z is missing, 
        # or load external Z if available. For now, we assume 'host_logmass' is the proxy 
        # since Mass-Metallicity relation is tight.
        # Alternatively, we can look for specific [O/H] columns if added later.
        if 'host_logmass' not in df.columns:
            print_status("Host mass (metallicity proxy) missing.", "ERROR")
            return
            
        # Rename for clarity
        df['metallicity_proxy'] = df['host_logmass']
        
        valid = df.dropna(subset=['h0_derived', 'sigma_inferred', 'metallicity_proxy'])
        n = len(valid)
        print_status(f"Loaded {n} hosts for Bivariate Analysis.", "INFO")
        
        # Variables
        y = valid['h0_derived']
        x1 = valid['sigma_inferred'] # Primary interest
        x2 = valid['metallicity_proxy'] # Confound
        
        # 1. Raw Correlations
        r_y_x1, p_y_x1 = stats.pearsonr(y, x1)
        r_y_x2, p_y_x2 = stats.pearsonr(y, x2)
        r_x1_x2, p_x1_x2 = stats.pearsonr(x1, x2)
        
        # 2. Partial Correlations
        # r_y_x1.x2 = (r_y_x1 - r_y_x2 * r_x1_x2) / sqrt((1-r_y_x2^2)(1-r_x1_x2^2))
        def partial_corr(r_xy, r_xz, r_yz):
            return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            
        pr_y_x1_x2 = partial_corr(r_y_x1, r_y_x2, r_x1_x2)
        pr_y_x2_x1 = partial_corr(r_y_x2, r_y_x1, r_x1_x2)
        
        # Significance (t-statistic)
        # df = n - 2 - k (k=1 control) = n - 3
        dof = n - 3
        t_y_x1_x2 = pr_y_x1_x2 * np.sqrt(dof / (1 - pr_y_x1_x2**2))
        p_y_x1_x2 = 2 * stats.t.sf(np.abs(t_y_x1_x2), dof)
        
        t_y_x2_x1 = pr_y_x2_x1 * np.sqrt(dof / (1 - pr_y_x2_x1**2))
        p_y_x2_x1 = 2 * stats.t.sf(np.abs(t_y_x2_x1), dof)
        
        # Reporting
        headers = ["Relation", "Correlation Type", "Coefficient", "p-value"]
        rows = [
            ["H0 vs Sigma", "Pearson (Raw)", f"{r_y_x1:.3f}", f"{p_y_x1:.4f}"],
            ["H0 vs Metallicity", "Pearson (Raw)", f"{r_y_x2:.3f}", f"{p_y_x2:.4f}"],
            ["Sigma vs Metallicity", "Pearson (Raw)", f"{r_x1_x2:.3f}", f"{p_x1_x2:.4f}"],
            ["H0 vs Sigma | Z", "Partial", f"{pr_y_x1_x2:.3f}", f"{p_y_x1_x2:.4f}"],
            ["H0 vs Z | Sigma", "Partial", f"{pr_y_x2_x1:.3f}", f"{p_y_x2_x1:.4f}"]
        ]
        print_table(headers, rows, title="Bivariate Analysis Results")
        
        # Save Stats
        with open(self.stats_path, 'w') as f:
            f.write("Bivariate Analysis Stats\n")
            f.write(f"Pearson r(H0, Sigma): {r_y_x1:.4f} (p={p_y_x1:.4f})\n")
            f.write(f"Pearson r(H0, Metal): {r_y_x2:.4f} (p={p_y_x2:.4f})\n")
            f.write(f"Pearson r(Sigma, Metal): {r_x1_x2:.4f} (p={p_x1_x2:.4f})\n")
            f.write(f"Partial r(H0, Sigma | Metal): {pr_y_x1_x2:.4f} (p={p_y_x1_x2:.4f})\n")
            f.write(f"Partial r(H0, Metal | Sigma): {pr_y_x2_x1:.4f} (p={p_y_x2_x1:.4f})\n")
        print_status(f"Saved stats to {self.stats_path}", "SUCCESS")
        
        # Plotting (Partial Regression Plots / Added Variable Plots)
        # To visualize partial correlation, we regress Y on Z, and X on Z, then plot residuals
        
        # 1. H0 vs Sigma | Z
        slope_y_z, intercept_y_z, _, _, _ = stats.linregress(x2, y)
        resid_y_z = y - (slope_y_z * x2 + intercept_y_z)
        
        slope_x_z, intercept_x_z, _, _, _ = stats.linregress(x2, x1)
        resid_x_z = x1 - (slope_x_z * x2 + intercept_x_z)
        
        # 2. H0 vs Z | Sigma
        slope_y_x, intercept_y_x, _, _, _ = stats.linregress(x1, y)
        resid_y_x = y - (slope_y_x * x1 + intercept_y_x)
        
        slope_z_x, intercept_z_x, _, _, _ = stats.linregress(x1, x2)
        resid_z_x = x2 - (slope_z_x * x1 + intercept_z_x)
        
        plt.figure(figsize=(14, 9))

        # Identify high-dispersion outlier for visual emphasis
        outlier_mask = (
            valid['normalized_name'].values == 'NGC 4639'
            if 'normalized_name' in valid.columns
            else np.zeros(len(valid), dtype=bool)
        )

        # Plot 1
        plt.subplot(1, 2, 1)
        plt.scatter(
            resid_x_z[~outlier_mask], resid_y_z[~outlier_mask],
            alpha=0.7, color=colors['blue'], s=60, edgecolor='white',
            label='Other hosts',
        )
        if np.any(outlier_mask):
            plt.scatter(
                resid_x_z[outlier_mask], resid_y_z[outlier_mask],
                color='#FF8C00', s=120, edgecolor='white', linewidth=1.5,
                zorder=5, label='NGC 4639',
            )

        # Fit line
        m, b = np.polyfit(resid_x_z, resid_y_z, 1)
        xp = np.linspace(resid_x_z.min(), resid_x_z.max(), 100)
        plt.plot(xp, m*xp + b, color=colors['dark'], linestyle='--', linewidth=2, label=f'Partial $r={pr_y_x1_x2:.3f}, p={p_y_x1_x2:.3f}$')

        plt.xlabel(r'Residual $\sigma$ (controlling for metallicity $Z$)')
        plt.ylabel(r'Residual $H_0$ (controlling for metallicity $Z$)')
        plt.title(r'$H_0$ vs $\sigma$ (Partial Residuals)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 2
        plt.subplot(1, 2, 2)
        plt.scatter(resid_z_x, resid_y_x, alpha=0.7, color=colors['accent'], s=60, edgecolor='white')
        
        # Fit line
        m2, b2 = np.polyfit(resid_z_x, resid_y_x, 1)
        xp2 = np.linspace(resid_z_x.min(), resid_z_x.max(), 100)
        plt.plot(xp2, m2*xp2 + b2, color=colors['dark'], linestyle='--', linewidth=2, label=f'Partial $r={pr_y_x2_x1:.3f}, p={p_y_x2_x1:.3f}$')
        
        plt.xlabel(r'Residual $Z$ (controlling for $\sigma$)')
        plt.ylabel(r'Residual $H_0$ (controlling for $\sigma$)')
        plt.title(r'$H_0$ vs Metallicity (Partial Residuals)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Use same y-axis limits for both panels for fair visual comparison
        y_min = min(resid_y_z.min(), resid_y_x.min())
        y_max = max(resid_y_z.max(), resid_y_x.max())
        plt.subplot(1, 2, 1)
        plt.ylim(y_min - 1, y_max + 1)
        plt.subplot(1, 2, 2)
        plt.ylim(y_min - 1, y_max + 1)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved bivariate plot to {self.plot_path}", "SUCCESS")
        plt.close()
        
        # Copy to public
        public_biv = self.public_figures_dir / "figure_02_bivariate_h0_sigma_metallicity.png"
        shutil.copy(self.plot_path, public_biv)
        print_status(f"Copied bivariate plot to {public_biv}", "SUCCESS")

    def generate_plot(self, h0, sigma, metal, part_corr_sigma, part_corr_metal):
        """Generates the bivariate analysis plot."""
        print_status("Generating partial regression plots...", "PROCESS")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'light_blue': '#4b6785', 'green': '#4a2650'}
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        
        # Residuals of H0 given Metal vs Residuals of Sigma given Metal
        slope_h_m, intercept_h_m, _, _, _ = stats.linregress(metal, h0)
        slope_s_m, intercept_s_m, _, _, _ = stats.linregress(metal, sigma)
        
        resid_h0_given_metal = h0 - (slope_h_m * metal + intercept_h_m)
        resid_sigma_given_metal = sigma - (slope_s_m * metal + intercept_s_m)
        
        # Residuals of H0 given Sigma vs Residuals of Metal given Sigma
        slope_h_s, intercept_h_s, _, _, _ = stats.linregress(sigma, h0)
        slope_m_s, intercept_m_s, _, _, _ = stats.linregress(sigma, metal)
        
        resid_h0_given_sigma = h0 - (slope_h_s * sigma + intercept_h_s)
        resid_metal_given_sigma = metal - (slope_m_s * sigma + intercept_m_s)
        
        # Left Panel: H0 residuals vs Sigma residuals
        ax1 = axes[0]
        ax1.scatter(resid_sigma_given_metal, resid_h0_given_metal, 
                   color=colors['blue'], alpha=0.7, s=80, edgecolor=colors['dark'], linewidth=0.5)
        
        m1, c1 = np.polyfit(resid_sigma_given_metal, resid_h0_given_metal, 1)
        x_range1 = np.array([min(resid_sigma_given_metal), max(resid_sigma_given_metal)])
        ax1.plot(x_range1, m1*x_range1 + c1, color=colors['blue'], linewidth=3, linestyle='--')
        
        ax1.set_title(f'Effect of Velocity Dispersion\n(Controlling for Metallicity)')
        ax1.set_xlabel(r'Residual $\sigma$ [km/s]')
        ax1.set_ylabel(r'Residual $H_0$ [km/s/Mpc]')
        ax1.text(0.05, 0.90, f'Partial $r = {part_corr_sigma:.3f}$', transform=ax1.transAxes, 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors['dark']))
        
        # Right Panel: H0 residuals vs Metallicity residuals
        ax2 = axes[1]
        ax2.scatter(resid_metal_given_sigma, resid_h0_given_sigma, 
                   color=colors['accent'], alpha=0.7, s=80, edgecolor=colors['dark'], linewidth=0.5)
        
        m2, c2 = np.polyfit(resid_metal_given_sigma, resid_h0_given_sigma, 1)
        x_range2 = np.array([min(resid_metal_given_sigma), max(resid_metal_given_sigma)])
        ax2.plot(x_range2, m2*x_range2 + c2, color=colors['accent'], linewidth=3, linestyle='--')
        
        ax2.set_title(f'Effect of Metallicity\n(Controlling for Velocity Dispersion)')
        ax2.set_xlabel(r'Residual Metallicity [dex]')
        ax2.set_ylabel(r'Residual $H_0$ [km/s/Mpc]')
        ax2.text(0.05, 0.90, f'Partial $r = {part_corr_metal:.3f}$', transform=ax2.transAxes, 
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors['dark']))
        
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
        print_status(f"Saved bivariate plot to {self.plot_path}", "SUCCESS")
        
        # Copy to public
        shutil.copy(self.plot_path, self.public_figures_dir / "figure_02_bivariate_h0_sigma_metallicity.png")

    def _provenance_eiv_model(self, df):
        """Provenance-aware errors-in-variables regression.

        H0,i = β0 + β1 * X_TEP,i + γ_method[i] + ε_i

        where method is: stellar absorption, HI linewidth, rotation proxy.
        σ measurement uncertainty (especially for HI/rotation proxies) is
        propagated into X_TEP error and included in the total variance.
        """
        print_status("Provenance-Aware Errors-in-Variables Model", "SECTION")

        prov_path = self.outputs_dir / "sigma_provenance_table.csv"
        if not prov_path.exists():
            print_status("Sigma provenance table missing; generating it via Step 4b.", "WARNING")
            try:
                from scripts.steps.step_07_aperture_sensitivity import Step4bApertureSensitivity

                Step4bApertureSensitivity().run()
                set_step_logger(self.logger)
            except Exception as exc:
                print_status(f"Could not generate sigma provenance table: {exc}", "WARNING")
                return None
            if not prov_path.exists():
                print_status("Sigma provenance table still missing; skipping EIV model.", "WARNING")
                return None

        prov = pd.read_csv(prov_path)
        df = df.copy()
        df["normalized_name"] = df["normalized_name"].astype(str).str.strip()
        prov["normalized_name"] = prov["normalized_name"].astype(str).str.strip()

        merged = pd.merge(
            df,
            prov[
                [
                    "normalized_name",
                    "sigma_method",
                    "sigma_measured_error_total_kms",
                ]
            ],
            on="normalized_name",
            how="left",
        )

        def classify_method(m):
            s = str(m).lower().strip()
            if "stellar absorption" in s:
                return "stellar_absorption"
            if "hi proxy" in s or "hi linewidth" in s or "calibrated_vmax" in s:
                return "HI_linewidth"
            if "vrot" in s or "proxy" in s:
                return "rotation_proxy"
            return "other"

        merged["method_class"] = merged["sigma_method"].apply(classify_method)

        sigma = merged["sigma_inferred"].values.astype(float)
        y = merged["h0_derived"].values.astype(float)
        mu = merged["value"].values.astype(float)
        mu_err = merged["error"].values.astype(float)
        S = (
            merged["shear_suppression"].values.astype(float)
            if "shear_suppression" in merged.columns
            else np.ones(len(merged))
        )

        sigma_ref = self._load_sigma_ref_val()
        if sigma_ref is None:
            sigma_ref = 87.17
            print_status("σ_ref missing from JSON; using standard fallback 87.17 km/s", "WARNING")

        from scripts.utils.tep_correction import C_SQUARED_KM_S
        x = S * (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S

        is_hi = (merged["method_class"] == "HI_linewidth").astype(float).values
        is_rot = (merged["method_class"] == "rotation_proxy").astype(float).values
        X = np.column_stack([np.ones(len(y)), x, is_hi, is_rot])

        # H0 measurement error from distance modulus
        h0_err = y * (np.log(10) / 5.0) * mu_err

        # σ measurement error from provenance
        sigma_err = pd.to_numeric(
            merged["sigma_measured_error_total_kms"], errors="coerce"
        ).fillna(0.0).values

        # Propagate σ error to X_TEP error: dX/dσ = S * 2σ / c^2
        dx_dsigma = S * 2.0 * sigma / C_SQUARED_KM_S
        x_err = np.abs(dx_dsigma * sigma_err)

        # First fit with H0 errors only (for initial guess)
        w_base = 1.0 / (h0_err**2 + 1e-10)
        W_base = np.diag(w_base)
        
        col_norms = np.linalg.norm(X, axis=0)
        col_norms[col_norms < 1e-10] = 1.0
        X_scaled = X / col_norms
        
        XWX_base_scaled = X_scaled.T @ W_base @ X_scaled
        XWy_base_scaled = X_scaled.T @ W_base @ y
        
        reg = 1e-10 * np.trace(XWX_base_scaled) / XWX_base_scaled.shape[0]
        XWX_base_reg_scaled = XWX_base_scaled + reg * np.eye(XWX_base_scaled.shape[0])
        
        try:
            beta_base_scaled = np.linalg.solve(XWX_base_reg_scaled, XWy_base_scaled)
            beta_base = beta_base_scaled / col_norms
        except np.linalg.LinAlgError:
            beta_base_scaled = np.linalg.lstsq(XWX_base_reg_scaled, XWy_base_scaled, rcond=None)[0]
            beta_base = beta_base_scaled / col_norms

        # Total variance using rigorous Orthogonal Distance Regression (ODR)
        from scipy import odr

        def f_model(B, x_data):
            # B[0] = intercept, B[1] = slope, B[2] = gamma_hi, B[3] = gamma_rot
            # x_data[0] = x_tep, x_data[1] = is_hi, x_data[2] = is_rot
            return B[0] + B[1] * x_data[0] + B[2] * x_data[1] + B[3] * x_data[2]

        linear_model = odr.Model(f_model)
        
        # x input data
        x_data = np.vstack([x, is_hi, is_rot])
        
        # x error data (treat is_hi and is_rot as exact by setting tiny errors)
        sx_data = np.vstack([x_err + 1e-10, np.full_like(x_err, 1e-10), np.full_like(x_err, 1e-10)])
        
        mydata = odr.RealData(x_data, y, sx=sx_data, sy=h0_err + 1e-10)
        myodr = odr.ODR(mydata, linear_model, beta0=beta_base)
        myoutput = myodr.run()
        
        beta = myoutput.beta
        beta_se = myoutput.sd_beta

        t_beta1 = beta[1] / beta_se[1] if beta_se[1] > 0 else np.nan
        df_resid = len(y) - X.shape[1]
        p_beta1 = (
            2 * (1 - stats.t.cdf(abs(t_beta1), df=df_resid))
            if np.isfinite(t_beta1)
            else np.nan
        )

        gamma_hi = beta[2]
        gamma_rot = beta[3]
        gamma_hi_se = beta_se[2]
        gamma_rot_se = beta_se[3]

        headers = ["Parameter", "Estimate", "SE", "t", "p"]
        rows = [
            ["β0 (intercept)", f"{beta[0]:.2f}", f"{beta_se[0]:.2f}", "-", "-"],
            [
                "β1 (X_TEP slope)",
                f"{beta[1]:.3e}",
                f"{beta_se[1]:.3e}",
                f"{t_beta1:.2f}" if np.isfinite(t_beta1) else "-",
                f"{p_beta1:.4f}" if np.isfinite(p_beta1) else "-",
            ],
            ["γ_HI (HI offset)", f"{gamma_hi:.2f}", f"{gamma_hi_se:.2f}", "-", "-"],
            [
                "γ_rot (rot offset)",
                f"{gamma_rot:.2f}",
                f"{gamma_rot_se:.2f}",
                "-",
                "-",
            ],
        ]
        print_table(headers, rows)

        # Proxy dilution summary
        n_stellar = int((merged["method_class"] == "stellar_absorption").sum())
        n_hi = int((merged["method_class"] == "HI_linewidth").sum())
        n_rot = int((merged["method_class"] == "rotation_proxy").sum())
        mean_x_err_stellar = float(
            np.mean(x_err[merged["method_class"] == "stellar_absorption"])
        ) if n_stellar > 0 else None
        mean_x_err_hi = float(
            np.mean(x_err[merged["method_class"] == "HI_linewidth"])
        ) if n_hi > 0 else None
        mean_x_err_rot = float(
            np.mean(x_err[merged["method_class"] == "rotation_proxy"])
        ) if n_rot > 0 else None

        print_status(
            f"Method counts: stellar={n_stellar}, HI={n_hi}, rot={n_rot}",
            "INFO",
        )
        hi_str = f"HI={mean_x_err_hi:.2e}" if mean_x_err_hi is not None else "HI=N/A"
        rot_str = f"rot={mean_x_err_rot:.2e}" if mean_x_err_rot is not None else "rot=N/A"
        print_status(
            f"Mean X_TEP uncertainty from σ error: {hi_str}, {rot_str}",
            "INFO",
        )

        result = {
            "beta": [float(v) for v in beta],
            "beta_se": [float(v) for v in beta_se],
            "beta1_t": float(t_beta1) if np.isfinite(t_beta1) else None,
            "beta1_p": float(p_beta1) if np.isfinite(p_beta1) else None,
            "gamma_HI": float(gamma_hi),
            "gamma_HI_se": float(gamma_hi_se),
            "gamma_rotation": float(gamma_rot),
            "gamma_rotation_se": float(gamma_rot_se),
            "n_stellar": n_stellar,
            "n_hi": n_hi,
            "n_rot": n_rot,
            "mean_x_err_stellar": mean_x_err_stellar,
            "mean_x_err_hi": mean_x_err_hi,
            "mean_x_err_rot": mean_x_err_rot,
        }

        # Save to the same covariance_robustness JSON for synthesis access
        cov_path = self.covariance_results_path
        if cov_path.exists():
            with open(cov_path, "r") as f:
                cov_data = json.load(f)
            cov_data["provenance_eiv"] = result
            with open(cov_path, "w") as f:
                json.dump(cov_data, f, indent=2)

        return result

    def run(self):
        print_status("Starting Step 4: Robustness Checks", "TITLE")
        self.perform_jackknife_analysis()
        self.perform_out_of_sample_validation()
        self.perform_bivariate_analysis()
        self.perform_redshift_cut_sensitivity()
        self.perform_flow_environment_robustness()
        if self.stratified_path.exists():
            df = pd.read_csv(self.stratified_path)
            self._provenance_eiv_model(df)
        print_status("Step 4 Complete.", "SUCCESS")

if __name__ == "__main__":
    Step4RobustnessChecks().run()
