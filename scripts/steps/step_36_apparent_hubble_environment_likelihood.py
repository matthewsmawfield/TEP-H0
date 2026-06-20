#!/usr/bin/env python3
"""
step_36_apparent_hubble_environment_likelihood.py

Apparent Hubble Constant — Environment Test

Tests whether the apparent local Hubble slope H_i = cz_i / d_i correlates with
environment (TEP regressor X_i), after controlling for metallicity, period, and
redshift.

Models (weighted least squares on N hosts):
  0: H_i = H_app + ε_i
  1: H_i = H_app + β_X X_i + ε_i
  2: H_i = H_app + β_X X_i + β_Z <Z>_i + ε_i
  3: H_i = H_app + β_X X_i + β_Z <Z>_i + β_P <P>_i + ε_i
  4: H_i = H_app + β_X X_i + β_Z <Z>_i + β_P <P>_i + β_z z_i + ε_i

Error model:
  σ_H² = (0.4605 · H_i · σ_μ)² + (σ_v / d_i)² + σ_int²

Sweep σ_v = 150, 250, 500 km/s.

Validation:
  - LOOCV (leave-one-host-out cross-validation)
  - Permutation p-value for β_X
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SH0ES_DIR = DATA_DIR / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458


def print_status(msg, level="INFO"):
    prefix = {
        "SECTION": "=" * 60,
        "INFO": "[INFO]",
        "SUCCESS": "[SUCCESS]",
        "WARNING": "[WARNING]",
        "ERROR": "[ERROR]",
    }.get(level, "[INFO]")
    if level == "SECTION":
        print(f"\n{prefix}\n{msg}\n{prefix}")
    else:
        print(f"{prefix} {msg}")


def build_host_x(sigma, sigma_ref, S=1.0):
    """Compute TEP regressor X for a host."""
    if sigma is None or sigma <= 0 or sigma_ref <= 0:
        return 0.0
    return S * (sigma**2 - sigma_ref**2) / (C_KM_S ** 2)


def center_scale(v):
    """Center to zero mean. Optionally scale to unit variance."""
    return v - np.mean(v)


def compute_vif(X):
    """Compute VIF for each regressor (columns 1 onward, skipping intercept)."""
    vifs = {}
    for j in range(1, X.shape[1]):
        # Regress column j on all other columns (excluding intercept)
        other_cols = [k for k in range(1, X.shape[1]) if k != j]
        if len(other_cols) == 0:
            vifs[f"col_{j}"] = 1.0
            continue
        X_other = X[:, other_cols]
        y_col = X[:, j]
        # Simple OLS
        beta_ols = np.linalg.lstsq(X_other, y_col, rcond=None)[0]
        y_pred = X_other @ beta_ols
        ss_res = np.sum((y_col - y_pred)**2)
        ss_tot = np.sum((y_col - np.mean(y_col))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2 + 1e-12)
        vifs[f"col_{j}"] = float(vif)
    return vifs


def _wls_stable(X, y, w):
    """Stable WLS using element-wise weights, avoiding dense diag(w)."""
    # XWX = sum_i w_i * x_i x_i^T  (outer product)
    # XWy = sum_i w_i * x_i * y_i
    n, p = X.shape
    XWX = np.zeros((p, p))
    XWy = np.zeros(p)
    for i in range(n):
        xi = X[i]
        XWX += w[i] * np.outer(xi, xi)
        XWy += w[i] * xi * y[i]
    try:
        beta = np.linalg.solve(XWX, XWy)
        cov = np.linalg.inv(XWX)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XWX, XWy, rcond=None)[0]
        cov = np.linalg.pinv(XWX, rcond=1e-12)
    return beta, cov


def fit_mle_with_scatter(X, y, sigma_base, sigma_int_guess=5.0):
    """
    Fit model H = X @ beta with intrinsic scatter sigma_int.

    sigma_base: (n,) baseline error per host (distance + peculiar velocity)

    Maximizes log-likelihood:
        ln L = -0.5 * sum[(y_i - y_pred_i)^2 / (sigma_base_i^2 + sigma_int^2)
               + ln(sigma_base_i^2 + sigma_int^2)]

    Returns beta, cov_beta, sigma_int, sigma_int_err, logL, chi2_equiv
    """
    from scipy.optimize import minimize_scalar

    n, p = X.shape
    sigma_base = np.asarray(sigma_base)
    # Floor to prevent zero division
    sigma_base = np.maximum(sigma_base, 0.01)

    def neg_logL(sigma_int):
        if sigma_int < 0:
            return 1e10
        var = sigma_base**2 + sigma_int**2
        w = 1.0 / var
        beta, _ = _wls_stable(X, y, w)
        resid = y - X @ beta
        ll = -0.5 * np.sum(resid**2 / var + np.log(var))
        return -ll

    # Optimize sigma_int
    res = minimize_scalar(neg_logL, bounds=(0.0, 50.0), method="bounded")
    sigma_int = float(res.x)

    # Final WLS fit
    var = sigma_base**2 + sigma_int**2
    w = 1.0 / var
    beta, cov = _wls_stable(X, y, w)

    resid = y - X @ beta
    chi2_equiv = float(np.sum(resid**2 / var))
    logL = -neg_logL(sigma_int)

    # Sigma_int uncertainty via profile likelihood curvature
    eps = 0.1
    ll0 = -neg_logL(sigma_int)
    ll_p = -neg_logL(sigma_int + eps)
    ll_m = -neg_logL(max(sigma_int - eps, 0.0))
    d2 = (ll_p - 2*ll0 + ll_m) / eps**2
    sigma_int_err = float(1.0 / np.sqrt(max(-d2, 1e-6))) if d2 < 0 else np.nan

    return beta, cov, sigma_int, sigma_int_err, logL, chi2_equiv


def loocv_mle(X, y, sigma_base, sigma_int):
    """Leave-one-out cross-validation with fixed sigma_int."""
    n = len(y)
    errors = np.zeros(n)
    sigma_base = np.maximum(sigma_base, 0.01)
    var = sigma_base**2 + sigma_int**2
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w = 1.0 / var[mask]
        beta_i, _ = _wls_stable(X[mask], y[mask], w)
        pred = X[i] @ beta_i
        errors[i] = (y[i] - pred) ** 2
    return float(np.mean(errors)), errors


def permutation_test_mle(X, y, sigma_base, sigma_int, n_perm=500, seed=42):
    """Permutation test for beta_X (column 1)."""
    if X.shape[1] < 2:
        return np.nan, 0.0

    rng = np.random.default_rng(seed)
    n = len(y)

    beta_orig, _, _, _, _, _ = fit_mle_with_scatter(X, y, sigma_base, sigma_int)
    beta_x_orig = beta_orig[1]

    perm_stats = np.zeros(n_perm)
    for p in range(n_perm):
        X_perm = X.copy()
        idx = np.arange(n)
        rng.shuffle(idx)
        X_perm[:, 1] = X[idx, 1]
        beta_p, _, _, _, _, _ = fit_mle_with_scatter(X_perm, y, sigma_base, sigma_int)
        perm_stats[p] = beta_p[1]

    p_value = float(np.mean(np.abs(perm_stats) >= np.abs(beta_x_orig)))
    return p_value, beta_x_orig


def run_model(model_name, X, y, sigma_base, n_perm=500):
    """Fit a single model with intrinsic scatter and return results dict."""
    n, p = X.shape

    # Check for all-zero columns (skip if found after column 0)
    for j in range(1, p):
        if np.allclose(X[:, j], 0.0):
            return {
                "model": model_name,
                "n_hosts": n,
                "n_params": p + 1,
                "dof": n - p - 1,
                "chi2": np.nan,
                "chi2_reduced": np.nan,
                "logL": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "H_app": np.nan,
                "H_app_err": np.nan,
                "sigma_int": np.nan,
                "sigma_int_err": np.nan,
                "beta_X": np.nan,
                "beta_X_err": np.nan,
                "beta_X_sig": np.nan,
                "beta_X_p_perm": np.nan,
                "loocv_mse": np.nan,
                "max_vif": np.nan,
                "status": "skipped_zero_column",
            }

    # Fit with MLE scatter
    beta, cov, sigma_int, sigma_int_err, logL, chi2 = fit_mle_with_scatter(
        X, y, sigma_base
    )
    se = np.sqrt(np.diag(cov))

    # LOOCV
    loocv_mse, loocv_errors = loocv_mle(X, y, sigma_base, sigma_int)

    # Permutation test
    p_perm, beta_x = permutation_test_mle(X, y, sigma_base, sigma_int, n_perm=n_perm)

    # VIFs
    vifs = compute_vif(X)
    max_vif = max(vifs.values()) if vifs else 1.0

    # AIC, BIC using logL
    n_params = p + 1
    aic = -2 * logL + 2 * n_params
    bic = -2 * logL + n_params * np.log(n)

    result = {
        "model": model_name,
        "n_hosts": n,
        "n_params": n_params,
        "dof": n - p - 1,
        "chi2": chi2,
        "chi2_reduced": chi2 / (n - p - 1) if (n - p - 1) > 0 else np.inf,
        "logL": logL,
        "AIC": aic,
        "BIC": bic,
        "H_app": float(beta[0]),
        "H_app_err": float(se[0]),
        "sigma_int": sigma_int,
        "sigma_int_err": sigma_int_err,
        "beta_X": float(beta[1]) if len(beta) > 1 else np.nan,
        "beta_X_err": float(se[1]) if len(se) > 1 else np.nan,
        "beta_X_sig": float(abs(beta[1]) / se[1]) if len(beta) > 1 and se[1] > 0 else np.nan,
        "beta_X_p_perm": p_perm,
        "loocv_mse": loocv_mse,
        "max_vif": max_vif,
        "status": "ok",
    }

    # Add other coefficients
    coef_names = ["beta_Z", "beta_P", "beta_z"]
    for i, name in enumerate(coef_names):
        idx = i + 2
        if len(beta) > idx:
            result[name] = float(beta[idx])
            result[f"{name}_err"] = float(se[idx])
            result[f"{name}_sig"] = float(abs(beta[idx]) / se[idx]) if se[idx] > 0 else np.nan

    return result


def load_sh0es_data():
    """Load SH0ES design matrix, data vector, covariance, and parameter names."""
    L = np.loadtxt(SH0ES_DIR / "L_R22.txt", delimiter="\t")
    names = ("Source", "Data")
    fmt = ("S20", np.float64)
    y_data = np.loadtxt(
        SH0ES_DIR / "y_R22.txt", unpack=True, skiprows=1,
        dtype={"names": names, "formats": fmt},
    )
    y = y_data[1]
    C = np.loadtxt(SH0ES_DIR / "C_R22.txt", delimiter="\t")
    q = np.loadtxt(SH0ES_DIR / "q_R22.txt", unpack=True, dtype="str")
    return L, y, C, q


def load_host_metadata():
    """Load host sigma, redshift, and shear suppression from processed metadata."""
    df = pd.read_csv(HOSTS_PATH)
    host_sigma = {}
    host_z = {}
    host_S = {}
    for _, row in df.iterrows():
        name = row["normalized_name"]
        sigma = row["sigma_inferred"]
        z_hd = row["z_hd"]
        S = row.get("shear_suppression", 1.0)
        if pd.isna(S):
            S = 1.0
        host_sigma[name] = sigma
        host_S[name] = float(S)
        if pd.notna(z_hd) and z_hd > 0:
            host_z[name] = z_hd
        # SH0ES-style compact names
        compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
        if compact.startswith(("N", "U")):
            parts = compact[1:]
            if parts.isdigit():
                padded = compact[0] + parts.zfill(4)
                host_sigma[padded] = sigma
                host_S[padded] = float(S)
                if pd.notna(z_hd) and z_hd > 0:
                    host_z[padded] = z_hd
                unpadded = compact[0] + parts.lstrip("0")
                if unpadded != padded:
                    host_sigma[unpadded] = sigma
                    host_S[unpadded] = float(S)
                    if pd.notna(z_hd) and z_hd > 0:
                        host_z[unpadded] = z_hd
        if compact.startswith("N"):
            ngc_name = "NGC" + compact[1:]
            host_sigma[ngc_name] = sigma
            host_S[ngc_name] = float(S)
            if pd.notna(z_hd) and z_hd > 0:
                host_z[ngc_name] = z_hd
    explicit = {"M1337": "N1337", "N105A": "N105", "N976A": "N976"}
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_sigma and sh0es_name not in host_sigma:
            host_sigma[sh0es_name] = host_sigma[csv_name]
            host_S[sh0es_name] = host_S[csv_name]
            if csv_name in host_z:
                host_z[sh0es_name] = host_z[csv_name]
    return host_sigma, host_z, host_S


def compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref):
    """Compute per-host quantities from SH0ES matrix and metadata."""
    from scipy import linalg

    # Fit baseline GLS to get mu_i
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except (linalg.LinAlgError, ValueError):
        Cinv = linalg.pinv(C)
        A_w = L.copy()
        y_w = y.copy()
    theta, residuals, rank, svals = np.linalg.lstsq(A_w, y_w, rcond=1e-12)

    # Extract mu_i and errors
    mu_params = [(q[i].replace("mu_", ""), i) for i in range(len(q)) if q[i].startswith("mu_")]

    # Build host data
    hosts = []
    mus = []
    mu_errs = []
    sigmas = []
    zs = []
    is_anchors = []
    host_period_terms = []
    host_z_terms = []

    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}

    for host_name, mu_idx in mu_params:
        if host_name not in host_sigma:
            continue
        mu_fit = theta[mu_idx]

        # Check if this host has Cepheid rows
        has_ceph = False
        period_terms = []
        z_terms = []
        for r in range(L.shape[0]):
            if abs(L[r, mu_idx]) > 0.01:
                nonzero = np.where(np.abs(L[r]) > 0.01)[0]
                params = [q[j] for j in nonzero]
                if "MHW1" in params:
                    has_ceph = True
                    # Extract period and metallicity coefficients from design matrix
                    bW_idx = np.where(q == "bW")[0]
                    ZW_idx = np.where(q == "ZW")[0]
                    if len(bW_idx) > 0:
                        period_terms.append(L[r, bW_idx[0]])
                    if len(ZW_idx) > 0:
                        z_terms.append(L[r, ZW_idx[0]])

        if not has_ceph:
            continue

        # Get covariance for mu
        try:
            cov = np.linalg.pinv(A_w.T @ A_w, rcond=1e-12)
            mu_err = np.sqrt(cov[mu_idx, mu_idx])
        except:
            mu_err = 0.05

        hosts.append(host_name)
        mus.append(mu_fit)
        mu_errs.append(mu_err)
        sigmas.append(host_sigma[host_name])
        zs.append(host_z.get(host_name, np.nan))
        is_anchors.append(host_name in anchor_hosts)
        host_period_terms.append(period_terms)
        host_z_terms.append(z_terms)

    # Build dataframe
    df = pd.DataFrame({
        "host": hosts,
        "mu": mus,
        "mu_err": mu_errs,
        "sigma": sigmas,
        "z_hd": zs,
        "is_anchor": is_anchors,
        "mean_period_term": [np.mean(pts) if pts else 0.0 for pts in host_period_terms],
        "mean_Z_term": [np.mean(zts) if zts else 0.0 for zts in host_z_terms],
    })

    return df


def _build_covariates_for_subset(df_subset, host_S, sigma_ref):
    """Build screened X, Z, P, z covariates for a subset, recentered within subset."""
    mu = df_subset["mu"].values
    mu_err = df_subset["mu_err"].fillna(0.05).values
    z = df_subset["z_hd"].values
    sigma_host = df_subset["sigma"].values
    d = 10 ** ((mu - 25.0) / 5.0)
    H_apparent = (C_KM_S * z) / d

    # Screened TEP regressor
    X_tep = np.array([
        build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
        for s, h in zip(sigma_host, df_subset["host"].values)
    ])

    # Load Z and P controls from covariate table if available
    Z_mean = np.zeros(len(df_subset))
    P_mean = np.zeros(len(df_subset))
    cov_table_path = OUT_DIR / "step_34_host_covariate_table.csv"
    if cov_table_path.exists():
        df_cov = pd.read_csv(cov_table_path)
        z_map = {row["host"]: row.get("mean_Z_term", 0.0) for _, row in df_cov.iterrows()}
        p_map = {row["host"]: row.get("mean_period_term", 0.0) for _, row in df_cov.iterrows()}
        for i, host in enumerate(df_subset["host"].values):
            Z_mean[i] = z_map.get(host, 0.0)
            P_mean[i] = p_map.get(host, 0.0)

    # Center within THIS subset
    X_tep_c = center_scale(X_tep)
    Z_c = center_scale(Z_mean)
    P_c = center_scale(P_mean)
    z_c = center_scale(z)

    return H_apparent, mu_err, z, d, X_tep_c, Z_c, P_c, z_c


def run():
    print_status("Step 36: Apparent Hubble Environment Likelihood", "SECTION")

    # Load data
    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_S = load_host_metadata()
    print_status(f"Design matrix: {L.shape}, loaded {len(host_sigma)} sigma mappings", "INFO")

    # Compute sigma_ref
    sigma_ref = np.sqrt(
        (30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55)
    )

    # Compute host-level covariates from matrix fit
    df_hosts = compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref)
    print_status(f"Computed {len(df_hosts)} calibrator hosts with Cepheid rows", "INFO")

    # Primary sample: non-anchor, valid redshift, z >= 0.0035 (N=29)
    PRIMARY_Z_CUT = 0.0035
    df_primary = df_hosts[
        (~df_hosts["is_anchor"])
        & df_hosts["z_hd"].notna()
        & (df_hosts["z_hd"] >= PRIMARY_Z_CUT)
    ].copy()

    # Sensitivity: all non-anchor with valid redshift (N=35)
    df_sensitivity = df_hosts[
        (~df_hosts["is_anchor"])
        & df_hosts["z_hd"].notna()
    ].copy()

    if len(df_primary) < 5:
        print_status(f"Only {len(df_primary)} valid primary hosts — aborting", "ERROR")
        return []

    print_status(f"Primary sample (z >= {PRIMARY_Z_CUT}): {len(df_primary)} hosts", "INFO")
    print_status(f"Sensitivity sample (all non-anchor): {len(df_sensitivity)} hosts", "INFO")

    # Run models on both primary and sensitivity samples
    sigma_v_values = [150, 250, 500]
    all_results = []

    for sample_name, df_sample in [("primary", df_primary), ("sensitivity", df_sensitivity)]:
        H_apparent, mu_err, z, d, X_tep_c, Z_c, P_c, z_c = _build_covariates_for_subset(
            df_sample, host_S, sigma_ref
        )
        print_status(
            f"Sample '{sample_name}': H_app range {H_apparent.min():.2f} to {H_apparent.max():.2f}, "
            f"mean {H_apparent.mean():.2f}",
            "INFO",
        )

        for sigma_v in sigma_v_values:
            sigma_H_mu = 0.4605 * H_apparent * mu_err
            sigma_H_v = sigma_v / d
            sigma_base = np.sqrt(sigma_H_mu**2 + sigma_H_v**2)

            models = [
                ("0_intercept_only", np.column_stack([np.ones(len(H_apparent))])),
                ("1_X", np.column_stack([np.ones(len(H_apparent)), X_tep_c])),
                ("2_X_Z", np.column_stack([np.ones(len(H_apparent)), X_tep_c, Z_c])),
                ("3_X_Z_P", np.column_stack([np.ones(len(H_apparent)), X_tep_c, Z_c, P_c])),
                ("4_X_Z_P_z", np.column_stack([np.ones(len(H_apparent)), X_tep_c, Z_c, P_c, z_c])),
            ]

            for name, X in models:
                r = run_model(name, X, H_apparent, sigma_base)
                r["sigma_v"] = sigma_v
                r["sample"] = sample_name
                all_results.append(r)

    # Print summary
    print_status("Summary Table", "SECTION")
    print_status(
        f"{'sample':>10s} {'sigma_v':>7s} {'Model':>12s} {'N':>3s} {'H_app':>7s} {'s_int':>5s} "
        f"{'beta_X':>10s} {'sig_X':>6s} {'p_perm':>7s} {'VIF':>5s} {'LOOCV':>8s}",
        "INFO",
    )
    for r in all_results:
        h = f"{r['H_app']:.2f}"
        si = f"{r['sigma_int']:.1f}"
        bx = f"{r['beta_X']:+.3e}" if not np.isnan(r.get('beta_X', np.nan)) else "n/a"
        sx = f"{r['beta_X_sig']:.1f}" if not np.isnan(r.get('beta_X_sig', np.nan)) else "n/a"
        pp = f"{r['beta_X_p_perm']:.3f}" if not np.isnan(r.get('beta_X_p_perm', np.nan)) else "n/a"
        vf = f"{r['max_vif']:.1f}"
        print_status(
            f"  {r['sample']:10s} {r['sigma_v']:5d} {r['model']:12s} {r['n_hosts']:3d} {h:>7s} {si:>5s} {bx:>10s} "
            f"{sx:>6s} {pp:>7s} {vf:>5s} {r['loocv_mse']:8.2f}",
            "INFO",
        )

    # Save
    df_out = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "step_36_apparent_hubble_environment.csv"
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    import json
    out_json = OUT_DIR / "step_36_apparent_hubble_environment.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    return all_results


if __name__ == "__main__":
    run()
