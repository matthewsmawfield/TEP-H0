
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import sys
import json
import shutil

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
        self.plot_path = self.figures_dir / "bivariate_h0_sigma_metallicity.png"
        self.jackknife_plot_path = self.figures_dir / "jackknife_influence.png"

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
        return sub

    def _gls_fit(self, X, y, cov):
        cov_reg = cov + np.eye(cov.shape[0]) * (1e-12 * np.trace(cov) / cov.shape[0])
        cov_inv = np.linalg.inv(cov_reg)
        XtCi = X.T @ cov_inv
        fisher = XtCi @ X
        fisher_inv = np.linalg.inv(fisher)
        beta = fisher_inv @ (XtCi @ y)
        return beta, fisher_inv

    def _covariance_aware_tests(self, df):
        cov, cov_labels = self._load_h0_covariance()
        if cov is None or cov_labels is None:
            return None

        target_labels = df['source_id'].astype(str).tolist()
        cov = self._subset_covariance(cov, cov_labels, target_labels)

        sigma = df['sigma_inferred'].values.astype(float)
        y = df['h0_derived'].values.astype(float)
        n = len(y)

        x = sigma - np.mean(sigma)
        X = np.column_stack([np.ones(n), x])

        beta, beta_cov = self._gls_fit(X, y, cov)
        slope = float(beta[1])
        slope_se = float(np.sqrt(beta_cov[1, 1]))
        t_slope = slope / slope_se if slope_se > 0 else np.nan
        p_slope = float(2 * (1 - stats.t.cdf(abs(t_slope), df=n - 2))) if np.isfinite(t_slope) else None

        # Parametric covariance simulation under null (intercept-only)
        X0 = np.ones((n, 1))
        beta0, _ = self._gls_fit(X0, y, cov)
        mu0 = float(beta0[0])

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 0.0, None)
            L = V @ np.diag(np.sqrt(w))

        n_sims = 20000
        rng = np.random.default_rng(42)
        z = rng.standard_normal((n, n_sims))
        y_sims = mu0 + (L @ z)

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
        d = np.sqrt(np.diag(cov))
        denom = np.outer(d, d)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(denom > 0, cov / denom, 0.0)
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

    def _load_sigma_ref(self):
        if not self.tep_results_path.exists():
            return None
        try:
            with open(self.tep_results_path, 'r') as f:
                d = json.load(f)
            return float(d.get('sigma_ref')) if 'sigma_ref' in d else None
        except Exception:
            return None

    def _fit_alpha(self, df, sigma_ref):
        sigma = df['sigma_inferred'].values.astype(float)
        mu = df['value'].values.astype(float)
        v = df['velocity'].values.astype(float)

        def objective(params):
            alpha = float(params[0])
            corr = alpha * np.log10(sigma / sigma_ref)
            mu_corr = mu + corr
            d_corr = 10 ** ((mu_corr - 25.0) / 5.0)
            h0_corr = v / d_corr
            slope, _ = np.polyfit(sigma, h0_corr, 1)
            return float(slope * slope)

        res = minimize(objective, x0=[0.7], method='Nelder-Mead', tol=1e-2)
        return float(res.x[0])

    def _apply_alpha(self, df, alpha, sigma_ref):
        sigma = df['sigma_inferred'].values.astype(float)
        mu = df['value'].values.astype(float)
        v = df['velocity'].values.astype(float)
        corr = float(alpha) * np.log10(sigma / float(sigma_ref))
        mu_corr = mu + corr
        d_corr = 10 ** ((mu_corr - 25.0) / 5.0)
        return v / d_corr

    def perform_out_of_sample_validation(self):
        print_status("Initiating Out-of-Sample Validation (Alpha)", "SECTION")

        if not self.stratified_path.exists():
            print_status("Stratified data missing. Run Step 2 first.", "ERROR")
            return

        sigma_ref = self._load_sigma_ref()
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
        alphas = []

        n_train = int(np.round(train_frac * n))
        for _ in range(n_repeats):
            idx = rng.permutation(n)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]

            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            alpha_hat = self._fit_alpha(train, sigma_ref)
            h0_test = self._apply_alpha(test, alpha_hat, sigma_ref)

            slope_test, _ = np.polyfit(test['sigma_inferred'].values, h0_test, 1)
            r_test = stats.pearsonr(test['sigma_inferred'].values, h0_test)[0]
            rho_test = stats.spearmanr(test['sigma_inferred'].values, h0_test)[0]

            alphas.append(alpha_hat)
            test_slopes.append(float(slope_test))
            test_r.append(float(r_test))
            test_rho.append(float(rho_test))
            test_h0_mean.append(float(np.mean(h0_test)))

        alphas = np.array(alphas)
        test_slopes = np.array(test_slopes)
        test_r = np.array(test_r)
        test_rho = np.array(test_rho)
        test_h0_mean = np.array(test_h0_mean)

        loo_alphas = []
        loo_pred = np.empty(n)
        for i in range(n):
            train = df.drop(index=i)
            alpha_i = self._fit_alpha(train, sigma_ref)
            loo_alphas.append(alpha_i)
            loo_pred[i] = float(self._apply_alpha(df.iloc[[i]], alpha_i, sigma_ref)[0])

        loo_alphas = np.array(loo_alphas)
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
            ["Train/Test", "alpha mean ± std", f"{np.mean(alphas):.3f} ± {np.std(alphas):.3f}"],
            ["Train/Test", "test slope median", f"{np.median(test_slopes):.4f}"],
            ["Train/Test", "test |r| median", f"{np.median(np.abs(test_r)):.3f}"],
            ["Train/Test", "test H0 mean (median)", f"{np.median(test_h0_mean):.2f}"],
            ["LOOCV", "alpha mean ± std", f"{np.mean(loo_alphas):.3f} ± {np.std(loo_alphas):.3f}"],
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
                "alpha_mean": float(np.mean(alphas)),
                "alpha_std": float(np.std(alphas)),
                "test_slope_median": float(np.median(test_slopes)),
                "test_slope_q16": float(np.percentile(test_slopes, 16)),
                "test_slope_q84": float(np.percentile(test_slopes, 84)),
                "test_abs_pearson_r_median": float(np.median(np.abs(test_r))),
                "test_h0_mean_median": float(np.median(test_h0_mean)),
                "test_h0_mean_q16": float(np.percentile(test_h0_mean, 16)),
                "test_h0_mean_q84": float(np.percentile(test_h0_mean, 84)),
            },
            "loocv": {
                "alpha_mean": float(np.mean(loo_alphas)),
                "alpha_std": float(np.std(loo_alphas)),
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

        if not self.stratified_path.exists():
            print_status("Stratified data missing. Run Step 2 first.", "ERROR")
            return None

        df = pd.read_csv(self.stratified_path)
        if 'z_hd' not in df.columns:
            print_status("No z_hd column found in stratified data.", "ERROR")
            return None

        cuts = [0.0035, 0.005, 0.01, 0.02]
        rows = []
        for zcut in cuts:
            sub = df[(pd.to_numeric(df['z_hd'], errors='coerce') >= zcut)].copy()
            suite = self._correlation_suite(sub['sigma_inferred'], sub['h0_derived'])
            suite['zcut'] = float(zcut)
            rows.append(suite)

        out = pd.DataFrame(rows)
        out.to_csv(self.zcut_stats_path, index=False)
        print_status(f"Saved redshift cut sensitivity results to {self.zcut_stats_path}", "SUCCESS")

        print_table(
            ["z_cut", "N", "Pearson r", "Spearman ρ", "Perm p"],
            [[
                f"{r['zcut']:.4f}",
                str(int(r['n'])),
                f"{r['pearson_r']:.3f}",
                f"{r['spearman_rho']:.3f}",
                f"{r['perm_p']:.4f}",
            ] for _, r in out.iterrows()],
            title="Redshift Cut Sensitivity (H0 vs Sigma)"
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

        if 'vpecerr' in df.columns:
            vpecerr = pd.to_numeric(df['vpecerr'], errors='coerce').fillna(250.0).values
        else:
            vpecerr = np.full(len(df), 250.0)

        vel_hd = self._velocity_from_z(df['z_hd']).values
        d = pd.to_numeric(df['distance_mpc'], errors='coerce').values
        sigma = pd.to_numeric(df['sigma_inferred'], errors='coerce').values

        mask = np.isfinite(vel_hd) & np.isfinite(d) & np.isfinite(sigma) & np.isfinite(vpecerr)
        vel_hd = vel_hd[mask]
        d = d[mask]
        sigma = sigma[mask]
        vpecerr = vpecerr[mask]

        mc = None
        if len(vel_hd) >= 3:
            rng = np.random.default_rng(42)
            r_draws = np.empty(n_mc)
            for i in range(n_mc):
                v_draw = vel_hd + rng.normal(0.0, vpecerr)
                h0_draw = v_draw / d
                r_draws[i] = stats.pearsonr(sigma, h0_draw)[0]

            mc = {
                'n_mc': int(n_mc),
                'n_hosts': int(len(vel_hd)),
                'r_mean': float(np.mean(r_draws)),
                'r_std': float(np.std(r_draws)),
                'r_p2p5': float(np.percentile(r_draws, 2.5)),
                'r_p50': float(np.percentile(r_draws, 50)),
                'r_p97p5': float(np.percentile(r_draws, 97.5)),
                'p_r_le_0': float(np.mean(r_draws <= 0.0)),
            }

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

            if mc is not None:
                f.write("Monte Carlo with residual peculiar-velocity uncertainty:\n")
                f.write(f"  N_draw={mc['n_mc']} N_hosts={mc['n_hosts']}\n")
                f.write(f"  r_mean={mc['r_mean']:.6f} r_std={mc['r_std']:.6f}\n")
                f.write(f"  r_95CI=[{mc['r_p2p5']:.6f}, {mc['r_p97p5']:.6f}]\n")
                f.write(f"  P(r<=0)={mc['p_r_le_0']:.6f}\n")

        print_status(f"Saved flow/environment robustness results to {self.flow_env_stats_path}", "SUCCESS")

        headers = ["Test", "Statistic", "p-value", "N"]
        rows = [
            ["Baseline r(H0,Sigma)", f"{base_suite['pearson_r']:.3f}", f"{base_suite['perm_p']:.4f}", str(base_suite['n'])],
            ["Partial r(H0,Sigma|z)", f"{r_h0_sigma_z:.3f}", f"{p_h0_sigma_z:.4f}", str(n_z)],
            ["Partial r(H0,Sigma|z,Nmb)", f"{r_h0_sigma_z_env:.3f}", f"{p_h0_sigma_z_env:.4f}", str(n_z_env)],
            ["Partial r(H0,Sigma|z,logpK)", f"{r_h0_sigma_z_logpK:.3f}", f"{p_h0_sigma_z_logpK:.4f}", str(n_z_logpK)],
        ]
        print_table(headers, rows, title="Flow + Environment Controls")

        if mc is not None:
            print_table(
                ["MC Metric", "Value"],
                [
                    ["N_hosts", str(mc['n_hosts'])],
                    ["r_mean", f"{mc['r_mean']:.3f}"],
                    ["r_std", f"{mc['r_std']:.3f}"],
                    ["r_95CI", f"[{mc['r_p2p5']:.3f}, {mc['r_p97p5']:.3f}]"],
                    ["P(r<=0)", f"{mc['p_r_le_0']:.4f}"],
                ],
                title="v_pec Monte Carlo Robustness"
            )

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
        except Exception as e:
            print_status(f"Covariance-aware tests failed: {e}", "WARNING")

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
        is_stable = (r_min > 0.3) and (min([p for p in r_values if p > 0]) > 0) # Basic check
        
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
            colors = {'blue': '#395d85', 'dark': '#301E30'}
            
        plt.figure(figsize=(14, 9))

        # Sort by delta_r for cleaner plot
        jack_df = jack_df.sort_values('delta_r')
        
        plt.bar(jack_df['Host'], jack_df['delta_r'], color=colors['blue'], alpha=0.8)
        plt.axhline(0, color=colors['dark'], linewidth=1.5)
        plt.xticks(rotation=90, fontsize=10)
        plt.ylabel(r"$\Delta r$ (Change in Correlation when removed)")
        plt.title("Jackknife Influence Analysis")
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(self.jackknife_plot_path, dpi=300)
        print_status(f"Saved Jackknife plot to {self.jackknife_plot_path}", "SUCCESS")
        plt.close()
        
        # Copy to public
        public_jack = self.public_figures_dir / "jackknife_influence.png"
        shutil.copy(self.jackknife_plot_path, public_jack)
        print_status(f"Copied Jackknife plot to {public_jack}", "SUCCESS")

    def perform_bivariate_analysis(self):
        """Performs bivariate analysis (H0 vs Sigma + Metallicity)."""
        print_status("Initiating Bivariate Analysis...", "SECTION")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30'}
            
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
        
        # Plot 1
        plt.subplot(1, 2, 1)
        plt.scatter(resid_x_z, resid_y_z, alpha=0.7, color=colors['blue'], s=60, edgecolor='white')
        
        # Fit line
        m, b = np.polyfit(resid_x_z, resid_y_z, 1)
        xp = np.linspace(resid_x_z.min(), resid_x_z.max(), 100)
        plt.plot(xp, m*xp + b, color=colors['dark'], linestyle='--', linewidth=2, label=f'r={pr_y_x1_x2:.3f}')
        
        plt.xlabel(r'Residual $\sigma$ (controlling for $Z$)')
        plt.ylabel(r'Residual $H_0$ (controlling for $Z$)')
        plt.title('H0 vs Sigma (Partial)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 2
        plt.subplot(1, 2, 2)
        plt.scatter(resid_z_x, resid_y_x, alpha=0.7, color=colors['accent'], s=60, edgecolor='white')
        
        # Fit line
        m2, b2 = np.polyfit(resid_z_x, resid_y_x, 1)
        xp2 = np.linspace(resid_z_x.min(), resid_z_x.max(), 100)
        plt.plot(xp2, m2*xp2 + b2, color=colors['dark'], linestyle='--', linewidth=2, label=f'r={pr_y_x2_x1:.3f}')
        
        plt.xlabel(r'Residual $Z$ (controlling for $\sigma$)')
        plt.ylabel(r'Residual $H_0$ (controlling for $\sigma$)')
        plt.title('H0 vs Metallicity (Partial)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved bivariate plot to {self.plot_path}", "SUCCESS")
        plt.close()
        
        # Copy to public
        public_biv = self.public_figures_dir / "bivariate_h0_sigma_metallicity.png"
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
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30'}
            
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
        shutil.copy(self.plot_path, self.public_figures_dir / "bivariate_h0_sigma_metallicity.png")

    def run(self):
        print_status("Starting Step 4: Robustness Checks", "TITLE")
        self.perform_jackknife_analysis()
        self.perform_out_of_sample_validation()
        self.perform_bivariate_analysis()
        self.perform_redshift_cut_sensitivity()
        self.perform_flow_environment_robustness()
        print_status("Step 4 Complete.", "SUCCESS")

if __name__ == "__main__":
    Step4RobustnessChecks().run()
