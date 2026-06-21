#!/usr/bin/env python3
"""
step_37_velocity_robustness.py

Velocity Robustness Suite for Apparent Hubble Environment Test

Stress-tests whether Step 36's positive beta_X survives under:
  1. Peculiar-velocity sensitivity sweep (sigma_v = 150..1000)
  2. Redshift cut sweep (remove local peculiar-velocity dominated hosts)
  3. Leave-one-host-out jackknife
  4. Student-t robust regression (downweight outliers)
  5. Redshift-bin-preserving permutation test
  6. Bootstrap confidence intervals

Output: robustness table showing beta_X significance versus assumptions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

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


def load_sh0es_data():
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
    from scipy import linalg
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except (linalg.LinAlgError, ValueError):
        A_w = L.copy()
        y_w = y.copy()
    theta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=1e-12)

    mu_params = [(q[i].replace("mu_", ""), i) for i in range(len(q)) if q[i].startswith("mu_")]
    hosts, mus, mu_errs, sigmas, zs, is_anchors = [], [], [], [], [], []
    host_period_terms, host_z_terms = [], []
    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}

    for host_name, mu_idx in mu_params:
        if host_name not in host_sigma:
            continue
        mu_fit = theta[mu_idx]
        has_ceph = False
        period_terms, z_terms = [], []
        for r in range(L.shape[0]):
            if abs(L[r, mu_idx]) > 0.01:
                nonzero = np.where(np.abs(L[r]) > 0.01)[0]
                params = [q[j] for j in nonzero]
                if "MHW1" in params:
                    has_ceph = True
                    bW_idx = np.where(q == "bW")[0]
                    ZW_idx = np.where(q == "ZW")[0]
                    if len(bW_idx) > 0:
                        period_terms.append(L[r, bW_idx[0]])
                    if len(ZW_idx) > 0:
                        z_terms.append(L[r, ZW_idx[0]])
        if not has_ceph:
            continue
        try:
            A_w_pinv = np.linalg.pinv(A_w, rcond=1e-12)
            mu_err = float(np.linalg.norm(A_w_pinv[mu_idx, :]))
        except Exception:
            mu_err = 0.05
        hosts.append(host_name)
        mus.append(mu_fit)
        mu_errs.append(mu_err)
        sigmas.append(host_sigma[host_name])
        zs.append(host_z.get(host_name, np.nan))
        is_anchors.append(host_name in anchor_hosts)
        host_period_terms.append(period_terms)
        host_z_terms.append(z_terms)

    df = pd.DataFrame({
        "host": hosts, "mu": mus, "mu_err": mu_errs,
        "sigma": sigmas, "z_hd": zs, "is_anchor": is_anchors,
        "mean_period_term": [np.mean(pts) if pts else 0.0 for pts in host_period_terms],
        "mean_Z_term": [np.mean(zts) if zts else 0.0 for zts in host_z_terms],
    })
    return df


def build_host_x(sigma, sigma_ref, S=1.0):
    if sigma is None or sigma <= 0 or sigma_ref <= 0:
        return 0.0
    return S * (sigma**2 - sigma_ref**2) / (C_KM_S ** 2)


def center_scale(v):
    return v - np.mean(v)


def _wls_stable(X, y, w):
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


def fit_mle_with_scatter(X, y, sigma_base):
    from scipy.optimize import minimize_scalar
    n, p = X.shape
    sigma_base = np.asarray(sigma_base)
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

    res = minimize_scalar(neg_logL, bounds=(0.0, 50.0), method="bounded")
    sigma_int = float(res.x)
    var = sigma_base**2 + sigma_int**2
    w = 1.0 / var
    beta, cov = _wls_stable(X, y, w)
    resid = y - X @ beta
    chi2 = float(np.sum(resid**2 / var))
    logL = -neg_logL(sigma_int)
    se = np.sqrt(np.diag(cov))
    return beta, cov, sigma_int, chi2, logL, se


def fit_student_t(X, y, sigma_base, nu=4.0):
    """Robust regression with Student-t errors (nu=4 => heavy tails).
    
    NOTE: sigma_int floor is raised to 0.5 to avoid unrealistically small
    errors when the heavy-tailed likelihood drives scatter to zero.
    """
    n, p = X.shape
    sigma_base = np.maximum(np.asarray(sigma_base), 0.01)

    def neg_logL(params):
        beta = params[:p]
        sigma_int = max(params[p], 0.5)
        var = sigma_base**2 + sigma_int**2
        resid = y - X @ beta
        ll = np.sum(
            -0.5 * (nu + 1) * np.log(1 + resid**2 / (nu * var))
            - 0.5 * np.log(var)
        )
        return -ll

    beta_init, _, _, _, _, _ = fit_mle_with_scatter(X, y, sigma_base)
    x0 = np.concatenate([beta_init, [3.0]])
    bounds = [(None, None)] * p + [(0.5, 50.0)]
    res = optimize.minimize(neg_logL, x0, method="L-BFGS-B", bounds=bounds)
    beta = res.x[:p]
    sigma_int = res.x[p]
    # Approximate covariance from Hessian
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            hess = optimize.approx_fprime(res.x, lambda x: optimize.approx_fprime(x, neg_logL, 1e-5), 1e-5)
            cov = np.linalg.pinv(hess, rcond=1e-12)
            se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(p, np.nan)
    return beta, sigma_int, se


def run_standard(X, y, sigma_base, model_name, sigma_v):
    beta, cov, sigma_int, chi2, logL, se = fit_mle_with_scatter(X, y, sigma_base)
    n_params = X.shape[1] + 1
    n, p = X.shape
    aic = -2 * logL + 2 * n_params
    bic = -2 * logL + n_params * np.log(n)
    return {
        "test": "standard",
        "model": model_name,
        "sigma_v": sigma_v,
        "z_cut": 0.0,
        "N_hosts": n,
        "H_app": float(beta[0]),
        "H_app_err": float(se[0]),
        "sigma_int": sigma_int,
        "beta_X": float(beta[1]) if len(beta) > 1 else np.nan,
        "beta_X_err": float(se[1]) if len(se) > 1 else np.nan,
        "beta_X_sig": float(abs(beta[1]) / se[1]) if len(beta) > 1 and se[1] > 0 else np.nan,
        "chi2": chi2,
        "AIC": aic,
        "BIC": bic,
    }


def run_loho(X, y, sigma_base, sigma_v, host_names):
    """Leave-one-host-out: remove each host and refit model 1."""
    n = len(y)
    beta_Xs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta, _, _, _, _, se = fit_mle_with_scatter(X[mask], y[mask], sigma_base[mask])
        beta_Xs.append(float(beta[1]) if len(beta) > 1 else np.nan)

    mean_bx = float(np.nanmean(beta_Xs))
    std_bx = float(np.nanstd(beta_Xs))
    min_bx = float(np.nanmin(beta_Xs))
    max_bx = float(np.nanmax(beta_Xs))
    n_positive = int(np.sum(np.array(beta_Xs) > 0))

    # Identify the most influential host
    influence = np.array(beta_Xs) - mean_bx
    max_influence_idx = int(np.nanargmax(np.abs(influence)))

    return {
        "test": "loho",
        "model": "1_X",
        "sigma_v": sigma_v,
        "z_cut": 0.0,
        "N_hosts": n,
        "beta_X_mean": mean_bx,
        "beta_X_std": std_bx,
        "beta_X_min": min_bx,
        "beta_X_max": max_bx,
        "n_positive": n_positive,
        "most_influential_host": host_names[max_influence_idx] if max_influence_idx < len(host_names) else "n/a",
    }


def permutation_binned(X, y, sigma_base, sigma_v, z, n_perm=500, n_bins=4, seed=42):
    """Permute X within redshift bins to preserve redshift-environment coupling."""
    if X.shape[1] < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    n = len(y)

    beta_orig, _, _, _, _, _ = fit_mle_with_scatter(X, y, sigma_base)
    beta_x_orig = beta_orig[1]

    # Create redshift bins
    z_sorted = np.argsort(z)
    bin_edges = np.array_split(z_sorted, n_bins)

    perm_stats = np.zeros(n_perm)
    for p in range(n_perm):
        X_perm = X.copy()
        for bin_idx in bin_edges:
            idx = bin_idx.copy()
            rng.shuffle(idx)
            X_perm[bin_idx, 1] = X[idx, 1]
        beta_p, _, _, _, _, _ = fit_mle_with_scatter(X_perm, y, sigma_base)
        perm_stats[p] = beta_p[1]

    p_value = float(np.mean(np.abs(perm_stats) >= np.abs(beta_x_orig)))
    return p_value


def bootstrap_hosts(X, y, sigma_base, n_boot=1000, seed=42):
    """Bootstrap by resampling hosts with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    beta_X_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        beta, _, _, _, _, se = fit_mle_with_scatter(X[idx], y[idx], sigma_base[idx])
        beta_X_boot.append(float(beta[1]) if len(beta) > 1 else np.nan)

    beta_X_boot = np.array(beta_X_boot)
    # Bias-corrected percentile CI
    ci_low = float(np.percentile(beta_X_boot, 2.5))
    ci_high = float(np.percentile(beta_X_boot, 97.5))
    ci_16 = float(np.percentile(beta_X_boot, 16))
    ci_84 = float(np.percentile(beta_X_boot, 84))
    mean = float(np.mean(beta_X_boot))
    std = float(np.std(beta_X_boot))
    frac_positive = float(np.mean(beta_X_boot > 0))

    return {
        "beta_X_boot_mean": mean,
        "beta_X_boot_std": std,
        "beta_X_ci_low": ci_low,
        "beta_X_ci_high": ci_high,
        "beta_X_ci_16": ci_16,
        "beta_X_ci_84": ci_84,
        "frac_positive": frac_positive,
    }


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
    cov_path = OUT_DIR / "step_34_host_covariate_table.csv"
    if cov_path.exists():
        df_cov = pd.read_csv(cov_path)
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
    print_status("Step 37: Velocity Robustness Suite", "SECTION")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_S = load_host_metadata()
    sigma_ref = np.sqrt(
        (30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55)
    )

    df_hosts = compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref)
    print_status(f"Computed {len(df_hosts)} calibrator hosts", "INFO")

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

    print_status(f"Primary sample (z >= {PRIMARY_Z_CUT}): {len(df_primary)} hosts", "INFO")
    print_status(f"Sensitivity sample (all non-anchor): {len(df_sensitivity)} hosts", "INFO")

    # ========================================================================
    # TEST 1: Standard model sigma_v sweep on PRIMARY sample
    # ========================================================================
    print_status("TEST 1: Standard model sigma_v sweep (PRIMARY)", "SECTION")
    sigma_v_values = [150, 250, 300, 500, 750, 1000]
    results = []

    H_pri, mu_err_pri, z_pri, d_pri, X_pri, Z_pri, P_pri, zc_pri = _build_covariates_for_subset(
        df_primary, host_S, sigma_ref
    )

    for sigma_v in sigma_v_values:
        sigma_H_mu = 0.4605 * H_pri * mu_err_pri
        sigma_H_v = sigma_v / d_pri
        sigma_base = np.sqrt(sigma_H_mu**2 + sigma_H_v**2)

        # Model 0: intercept
        X0 = np.column_stack([np.ones(len(H_pri))])
        r0 = run_standard(X0, H_pri, sigma_base, "0_intercept_only", sigma_v)
        r0["sample"] = "primary"
        results.append(r0)

        # Model 1: +X
        X1 = np.column_stack([np.ones(len(H_pri)), X_pri])
        r1 = run_standard(X1, H_pri, sigma_base, "1_X", sigma_v)
        r1["sample"] = "primary"
        results.append(r1)

        # Model 2: +X+Z
        X2 = np.column_stack([np.ones(len(H_pri)), X_pri, Z_pri])
        r2 = run_standard(X2, H_pri, sigma_base, "2_X_Z", sigma_v)
        r2["sample"] = "primary"
        results.append(r2)

    # ========================================================================
    # TEST 1b: Standard model sigma_v=250 on SENSITIVITY sample
    # ========================================================================
    print_status("TEST 1b: Standard model sigma_v=250 (SENSITIVITY)", "SECTION")
    H_sens, mu_err_sens, z_sens, d_sens, X_sens, Z_sens, P_sens, zc_sens = _build_covariates_for_subset(
        df_sensitivity, host_S, sigma_ref
    )
    sigma_v_sens = 250
    sigma_H_mu = 0.4605 * H_sens * mu_err_sens
    sigma_H_v = sigma_v_sens / d_sens
    sigma_base_sens = np.sqrt(sigma_H_mu**2 + sigma_H_v**2)

    X1_sens = np.column_stack([np.ones(len(H_sens)), X_sens])
    r_sens = run_standard(X1_sens, H_sens, sigma_base_sens, "1_X", sigma_v_sens)
    r_sens["sample"] = "sensitivity"
    results.append(r_sens)

    # ========================================================================
    # TEST 2: Redshift cut sweep on PRIMARY sample (sigma_v=250)
    # ========================================================================
    print_status("TEST 2: Redshift cut sweep (sigma_v=250, primary)", "SECTION")
    z_cuts = [0.0035, 0.005, 0.0075, 0.01]
    sigma_v_cut = 250
    sigma_H_mu = 0.4605 * H_pri * mu_err_pri
    sigma_H_v = sigma_v_cut / d_pri
    sigma_base_full = np.sqrt(sigma_H_mu**2 + sigma_H_v**2)

    for z_cut in z_cuts:
        mask = z_pri >= z_cut
        n_cut = mask.sum()
        print_status(f"  z >= {z_cut}: N={n_cut}", "INFO")
        if n_cut < 10:
            continue

        H_cut = H_pri[mask]
        sb_cut = sigma_base_full[mask]
        # Recenter X within the subset
        X_subset = center_scale(X_pri[mask])
        X_cut = np.column_stack([np.ones(n_cut), X_subset])

        beta, cov, si, chi2, logL, se = fit_mle_with_scatter(X_cut, H_cut, sb_cut)
        n_params = X_cut.shape[1] + 1
        results.append({
            "test": "z_cut",
            "model": "1_X",
            "sample": "primary",
            "sigma_v": sigma_v_cut,
            "z_cut": z_cut,
            "N_hosts": n_cut,
            "H_app": float(beta[0]),
            "H_app_err": float(se[0]),
            "sigma_int": si,
            "beta_X": float(beta[1]),
            "beta_X_err": float(se[1]),
            "beta_X_sig": float(abs(beta[1]) / se[1]) if se[1] > 0 else np.nan,
            "chi2": chi2,
            "AIC": -2 * logL + 2 * n_params,
            "BIC": -2 * logL + n_params * np.log(n_cut),
        })

    # ========================================================================
    # TEST 3: Leave-one-host-out on PRIMARY sample
    # ========================================================================
    print_status("TEST 3: Leave-one-host-out (primary)", "SECTION")
    sigma_v_loho = 250
    sigma_H_mu = 0.4605 * H_pri * mu_err_pri
    sigma_H_v = sigma_v_loho / d_pri
    sigma_base_loho = np.sqrt(sigma_H_mu**2 + sigma_H_v**2)
    X1_pri = np.column_stack([np.ones(len(H_pri)), X_pri])

    loho_result = run_loho(X1_pri, H_pri, sigma_base_loho, sigma_v_loho, df_primary["host"].values)
    results.append({**loho_result, "sample": "primary", "chi2": np.nan, "AIC": np.nan, "BIC": np.nan})
    print_status(
        f"  LOHO: beta_X = {loho_result['beta_X_mean']:.3e} "
        f"+/- {loho_result['beta_X_std']:.3e}, "
        f"range [{loho_result['beta_X_min']:.3e}, {loho_result['beta_X_max']:.3e}], "
        f"positive in {loho_result['n_positive']}/{loho_result['N_hosts']} hosts",
        "INFO",
    )
    print_status(f"  Most influential host: {loho_result['most_influential_host']}", "INFO")

    # ========================================================================
    # TEST 4: Student-t robust regression on PRIMARY sample
    # ========================================================================
    print_status("TEST 4: Student-t robust regression (nu=4, primary)", "SECTION")
    beta_t, si_t, se_t = fit_student_t(X1_pri, H_pri, sigma_base_loho, nu=4.0)
    # Do not report broken analytic sig; report sign/stability only
    results.append({
        "test": "student_t",
        "model": "1_X",
        "sample": "primary",
        "sigma_v": sigma_v_loho,
        "z_cut": 0.0,
        "N_hosts": len(H_pri),
        "H_app": float(beta_t[0]),
        "sigma_int": si_t,
        "beta_X": float(beta_t[1]),
        "beta_X_err": np.nan,
        "beta_X_sig": np.nan,
        "status": "sign_only_hessian_unreliable",
        "chi2": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
    })
    print_status(
        f"  Student-t: beta_X = {beta_t[1]:.3e} (sign only), sigma_int = {si_t:.2f}",
        "INFO",
    )

    # ========================================================================
    # TEST 5: Binned permutation on PRIMARY sample
    # ========================================================================
    print_status("TEST 5: Redshift-bin-preserving permutation (primary)", "SECTION")
    p_binned = permutation_binned(X1_pri, H_pri, sigma_base_loho, sigma_v_loho, z_pri, n_perm=500, n_bins=4)
    results.append({
        "test": "perm_binned",
        "model": "1_X",
        "sample": "primary",
        "sigma_v": sigma_v_loho,
        "z_cut": 0.0,
        "N_hosts": len(H_pri),
        "beta_X_p_binned": p_binned,
    })
    print_status(f"  Binned permutation p-value: {p_binned:.3f}", "INFO")

    # ========================================================================
    # TEST 6: Bootstrap on PRIMARY sample
    # ========================================================================
    print_status("TEST 6: Bootstrap (N=1000, primary)", "SECTION")
    boot = bootstrap_hosts(X1_pri, H_pri, sigma_base_loho, n_boot=1000)
    results.append({
        "test": "bootstrap",
        "model": "1_X",
        "sample": "primary",
        "sigma_v": sigma_v_loho,
        "z_cut": 0.0,
        "N_hosts": len(H_pri),
        **boot,
    })
    print_status(
        f"  Bootstrap: beta_X = {boot['beta_X_boot_mean']:.3e} +/- {boot['beta_X_boot_std']:.3e}",
        "INFO",
    )
    print_status(
        f"  95% CI: [{boot['beta_X_ci_low']:.3e}, {boot['beta_X_ci_high']:.3e}], "
        f"frac_positive = {boot['frac_positive']:.3f}",
        "INFO",
    )

    # ========================================================================
    # Summary table
    # ========================================================================
    print_status("Summary Table", "SECTION")
    print_status(
        f"{'Test':>12s} {'sample':>10s} {'sigma_v':>7s} {'z_cut':>6s} {'N':>3s} "
        f"{'H_app':>7s} {'s_int':>5s} {'beta_X':>10s} {'sig_X':>6s}",
        "INFO",
    )
    for r in results:
        if "beta_X" not in r or np.isnan(r.get("beta_X", np.nan)):
            continue
        h = f"{r.get('H_app', np.nan):.2f}" if not np.isnan(r.get('H_app', np.nan)) else "n/a"
        si = f"{r.get('sigma_int', np.nan):.1f}" if not np.isnan(r.get('sigma_int', np.nan)) else "n/a"
        bx = f"{r['beta_X']:+.3e}"
        sx = f"{r.get('beta_X_sig', np.nan):.1f}" if not np.isnan(r.get('beta_X_sig', np.nan)) else "n/a"
        print_status(
            f"  {r.get('test', 'n/a'):10s} {r.get('sample', 'n/a'):10s} {r.get('sigma_v', 0):5d} "
            f"{r.get('z_cut', 0):6.4f} {r.get('N_hosts', 0):3d} {h:>7s} {si:>5s} {bx:>10s} {sx:>6s}",
            "INFO",
        )

    # ========================================================================
    # Old-vs-new comparison output
    # ========================================================================
    print_status("Old-vs-New Comparison", "SECTION")
    comparison = []
    # Extract key results for comparison
    for r in results:
        if r.get("model") == "1_X" and r.get("test") in ("standard", "z_cut"):
            comparison.append({
                "sample": r.get("sample", "n/a"),
                "screening": "screened",
                "sigma_v": r.get("sigma_v", np.nan),
                "N_hosts": r.get("N_hosts", np.nan),
                "z_cut": r.get("z_cut", 0.0),
                "beta_X": r.get("beta_X", np.nan),
                "beta_X_err": r.get("beta_X_err", np.nan),
                "beta_X_sig": r.get("beta_X_sig", np.nan),
                "redshift_binned_permutation_p": r.get("beta_X_p_binned", np.nan),
                "LOOCV": r.get("loocv_mse", np.nan),
            })

    if comparison:
        df_comp = pd.DataFrame(comparison)
        comp_csv = OUT_DIR / "step_37_old_vs_patched_comparison.csv"
        df_comp.to_csv(comp_csv, index=False)
        print_status(f"Saved comparison to {comp_csv}", "SUCCESS")
        print_status(df_comp.to_string(index=False, na_rep="—"), "INFO")

    # Save full results
    df_out = pd.DataFrame(results)
    out_csv = OUT_DIR / "step_37_velocity_robustness.csv"
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    out_json = OUT_DIR / "step_37_velocity_robustness.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    # ========================================================================
    # CSV/JSON consistency assertions
    # ========================================================================
    df_json = pd.read_json(out_json)
    df_csv = pd.read_csv(out_csv)

    assert len(df_json) == len(df_csv), \
        f"JSON/CSV row mismatch: {len(df_json)} vs {len(df_csv)}"
    assert set(df_json.columns).issubset(set(df_csv.columns)), \
        f"CSV missing columns: {set(df_json.columns) - set(df_csv.columns)}"

    primary_rows = df_csv[
        (df_csv["test"] == "standard")
        & (df_csv["sample"] == "primary")
        & (df_csv["model"] == "1_X")
    ]
    assert set(primary_rows["N_hosts"]) == {29}, \
        f"Primary standard model 1_X has N_hosts != 29: {set(primary_rows['N_hosts'])}"
    assert np.all(primary_rows["beta_X"] > 0), \
        f"Primary standard model 1_X has non-positive beta_X: {primary_rows['beta_X'].values}"
    print_status("CSV/JSON consistency assertions passed", "SUCCESS")

    return results


if __name__ == "__main__":
    run()
