#!/usr/bin/env python3
"""
step_38_hierarchical_timefield_ladder.py

Hierarchical Timefield Ladder

Jointly models Cepheid-distance bias (κ_Cep) and apparent-redshift/environment
bias (β_X) in velocity space.

Model grid:
  H0      : κ=0, β=0               — null
  Hβ      : κ=0, β=free            — apparent environment only
  K0      : κ=free, β=0            — Cepheid bias only
  Kfixβ   : κ=fixed canonical, β=free  — canonical + residual
  Kpriorβ : κ=Gaussian prior, β=free   — theory-guided mixed
  Kβ      : κ=free, β=free         — empirical mixed (likely degenerate)

Physics:
  d_true = d_obs × 10^(κ_Cep·X_i / 5)
  cz_model,i = d_true,i × (H_app + β_X·X_i)
  σ_cz² = σ_v² + (ln(10)/5 · cz_model,i · σ_μ,i)² + σ_int,v²

Runs σ_v = {150, 250, 500} km/s on primary (N=29) and redshift cuts.
Includes mandatory debug checks: injection recovery, contours, correlation,
LOHO sign stability, bootstrap sign stability, primary vs z-cut sensitivity.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

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
LN10_OVER_5 = np.log(10) / 5.0
KAPPA_CANONICAL = 970000.0  # mag, from TEP canonical value
KAPPA_PRIOR_MEAN = 960000.0  # mag
KAPPA_PRIOR_SIGMA = 400000.0  # mag


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
    """Center to zero mean."""
    return v - np.mean(v)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except (linalg.LinAlgError, ValueError):
        A_w = L.copy()
        y_w = y.copy()
    theta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=1e-12)

    mu_params = [(q[i].replace("mu_", ""), i) for i in range(len(q)) if q[i].startswith("mu_")]
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
        has_ceph = False
        period_terms = []
        z_terms = []
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
    d_obs = 10 ** ((mu - 25.0) / 5.0)
    cz_obs = C_KM_S * z

    X_tep = np.array([
        build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
        for s, h in zip(sigma_host, df_subset["host"].values)
    ])

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

    X_tep_c = center_scale(X_tep)
    Z_c = center_scale(Z_mean)
    P_c = center_scale(P_mean)
    z_c = center_scale(z)

    return cz_obs, mu_err, z, d_obs, X_tep_c, Z_c, P_c, z_c


# ---------------------------------------------------------------------------
# Model fitting in velocity space
# ---------------------------------------------------------------------------
# Internal scaling factors to make optimizer parameters O(1)
BETA_SCALE = 1e7
KAPPA_SCALE = 1e5


def _velocity_model(cz_obs, d_obs, X, H_app, kappa, beta):
    """Compute predicted cz and variance for the hierarchical model."""
    # d_true = d_obs * 10^(kappa * X / 5)
    d_true = d_obs * np.power(10.0, kappa * X / 5.0)
    # cz_model = d_true * (H_app + beta * X)
    cz_model = d_true * (H_app + beta * X)
    return cz_model


def _neg_logL_velocity(params, cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, sigma_int_guess=5.0):
    """
    Negative log-likelihood for velocity-space hierarchical model.

    params are internally scaled O(1) values:
      beta_param  = beta_actual  / BETA_SCALE  (beta_actual  ~ 1e7  → param ~ 1)
      kappa_param = kappa_actual / KAPPA_SCALE (kappa_actual ~ 1e5  → param ~ 1)

    params layout depends on model_type:
      H0      : [H_app, sigma_int_v]
      Hβ      : [H_app, beta_param, sigma_int_v]
      K0      : [H_app, kappa_param, sigma_int_v]
      Kfixβ   : [H_app, beta_param, sigma_int_v]  (kappa fixed)
      Kpriorβ : [H_app, kappa_param, beta_param, sigma_int_v]
      Kβ      : [H_app, kappa_param, beta_param, sigma_int_v]
    """
    sigma_int_v = max(params[-1], 0.01)

    if model_type == "H0":
        H_app = params[0]
        kappa = 0.0
        beta = 0.0
    elif model_type == "Hβ":
        H_app = params[0]
        kappa = 0.0
        beta = params[1] * BETA_SCALE
    elif model_type == "K0":
        H_app = params[0]
        kappa = params[1] * KAPPA_SCALE
        beta = 0.0
    elif model_type == "Kfixβ":
        H_app = params[0]
        kappa = KAPPA_CANONICAL
        beta = params[1] * BETA_SCALE
    elif model_type in ("Kpriorβ", "Kβ"):
        H_app = params[0]
        kappa = params[1] * KAPPA_SCALE
        beta = params[2] * BETA_SCALE
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    cz_model = _velocity_model(cz_obs, d_obs, X, H_app, kappa, beta)
    resid = cz_obs - cz_model

    # Variance: peculiar velocity + distance uncertainty + intrinsic scatter
    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
    var = np.maximum(var, 0.01)

    ll = -0.5 * np.sum(resid**2 / var + np.log(var))

    # Gaussian prior on actual kappa for Kpriorβ
    if model_type == "Kpriorβ":
        ll += -0.5 * ((kappa - KAPPA_PRIOR_MEAN) / KAPPA_PRIOR_SIGMA) ** 2

    return -ll


def fit_model(cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, sigma_int_guess=5.0):
    """Fit a model variant and return parameter estimates and uncertainties."""
    n = len(cz_obs)

    # Initial guess (scaled parameters are O(1))
    H_app_init = np.median(cz_obs / d_obs)
    if model_type == "H0":
        x0 = np.array([H_app_init, sigma_int_guess])
        bounds = [(30.0, 90.0), (0.01, 50.0)]
    elif model_type == "Hβ":
        x0 = np.array([H_app_init, 2.35, sigma_int_guess])
        bounds = [(30.0, 90.0), (-100.0, 100.0), (0.01, 50.0)]
    elif model_type == "K0":
        x0 = np.array([H_app_init, 0.0, sigma_int_guess])
        bounds = [(30.0, 90.0), (-50.0, 50.0), (0.01, 50.0)]
    elif model_type == "Kfixβ":
        x0 = np.array([H_app_init, 2.35, sigma_int_guess])
        bounds = [(30.0, 90.0), (-100.0, 100.0), (0.01, 50.0)]
    elif model_type in ("Kpriorβ", "Kβ"):
        x0 = np.array([H_app_init, KAPPA_PRIOR_MEAN / KAPPA_SCALE, 2.35, sigma_int_guess])
        bounds = [(30.0, 90.0), (-50.0, 50.0), (-100.0, 100.0), (0.01, 50.0)]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    res = optimize.minimize(
        _neg_logL_velocity,
        x0,
        args=(cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, sigma_int_guess),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        # Fallback: try with different initializations
        for H_init in [65.0, 70.0, 75.0, 80.0]:
            x0_try = x0.copy()
            x0_try[0] = H_init
            res_try = optimize.minimize(
                _neg_logL_velocity,
                x0_try,
                args=(cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, sigma_int_guess),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res_try.fun < res.fun:
                res = res_try

    # Parameter extraction (convert scaled params back to physical values)
    if model_type == "H0":
        H_app, sigma_int_v = res.x[0], res.x[1]
        kappa, beta = 0.0, 0.0
    elif model_type == "Hβ":
        H_app, beta, sigma_int_v = res.x[0], res.x[1] * BETA_SCALE, res.x[2]
        kappa = 0.0
    elif model_type == "K0":
        H_app, kappa, sigma_int_v = res.x[0], res.x[1] * KAPPA_SCALE, res.x[2]
        beta = 0.0
    elif model_type == "Kfixβ":
        H_app, beta, sigma_int_v = res.x[0], res.x[1] * BETA_SCALE, res.x[2]
        kappa = KAPPA_CANONICAL
    else:  # Kpriorβ, Kβ
        H_app, kappa, beta, sigma_int_v = res.x[0], res.x[1] * KAPPA_SCALE, res.x[2] * BETA_SCALE, res.x[3]

    # Compute Hessian for uncertainties (in scaled parameter space)
    try:
        hess = optimize.approx_fprime(
            res.x,
            lambda x: optimize.approx_fprime(
                x,
                lambda p: _neg_logL_velocity(p, cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, sigma_int_guess),
                1e-5,
            ),
            1e-5,
        )
        cov = np.linalg.pinv(hess, rcond=1e-12)
        se_scaled = np.sqrt(np.maximum(np.diag(cov), 0))
    except:
        se_scaled = np.full(len(res.x), np.nan)

    # Convert uncertainties back to physical units
    se = se_scaled.copy()
    if model_type == "Hβ":
        se[1] *= BETA_SCALE
    elif model_type == "K0":
        se[1] *= KAPPA_SCALE
    elif model_type == "Kfixβ":
        se[1] *= BETA_SCALE
    elif model_type in ("Kpriorβ", "Kβ"):
        se[1] *= KAPPA_SCALE
        se[2] *= BETA_SCALE

    # Model prediction and chi2
    cz_model = _velocity_model(cz_obs, d_obs, X, H_app, kappa, beta)
    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
    resid = cz_obs - cz_model
    chi2 = float(np.sum(resid**2 / var))
    logL = -res.fun

    # Degrees of freedom
    if model_type == "H0":
        n_params = 2
    elif model_type in ("Hβ", "K0", "Kfixβ"):
        n_params = 3
    else:
        n_params = 4
    dof = n - n_params

    return {
        "model": model_type,
        "n_hosts": n,
        "n_params": n_params,
        "dof": dof,
        "H_app": float(H_app),
        "H_app_err": float(se[0]),
        "kappa_Cep": float(kappa),
        "kappa_Cep_err": float(se[1]) if model_type in ("K0", "Kpriorβ", "Kβ") else np.nan,
        "beta_X": float(beta),
        "beta_X_err": float(se[-2]) if model_type in ("Hβ", "Kfixβ", "Kpriorβ", "Kβ") else np.nan,
        "sigma_int_v": float(sigma_int_v),
        "sigma_int_v_err": float(se[-1]),
        "chi2": chi2,
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
        "logL": float(logL),
        "AIC": -2 * logL + 2 * n_params,
        "BIC": -2 * logL + n_params * np.log(n),
        "status": "converged" if res.success else "fallback",
        "n_iter": int(res.nit) if hasattr(res, "nit") else -1,
    }


# ---------------------------------------------------------------------------
# Mandatory debug checks
# ---------------------------------------------------------------------------
def injection_recovery(cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, true_kappa, true_beta, seed=42):
    """Inject known (kappa, beta), fit, and recover."""
    rng = np.random.default_rng(seed)
    n = len(cz_obs)

    # Generate synthetic cz with injected signal
    cz_model_true = _velocity_model(cz_obs, d_obs, X, 73.0, true_kappa, true_beta)
    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model_true) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + 2.0**2
    noise = rng.normal(0, np.sqrt(var))
    cz_synth = cz_model_true + noise

    result = fit_model(cz_synth, d_obs, X, sigma_mu, sigma_v, model_type)
    recovered_kappa = result.get("kappa_Cep", 0.0)
    recovered_beta = result.get("beta_X", 0.0)

    return {
        "injected_kappa": true_kappa,
        "injected_beta": true_beta,
        "recovered_kappa": recovered_kappa,
        "recovered_beta": recovered_beta,
        "kappa_bias": recovered_kappa - true_kappa if true_kappa != 0 else np.nan,
        "beta_bias": recovered_beta - true_beta if true_beta != 0 else np.nan,
    }


def likelihood_contour(cz_obs, d_obs, X, sigma_mu, sigma_v, n_grid=40):
    """Compute likelihood contour on a coarse kappa-beta grid around the MLE."""
    result = fit_model(cz_obs, d_obs, X, sigma_mu, sigma_v, "Kβ")
    kappa_mle = result["kappa_Cep"]
    beta_mle = result["beta_X"]

    # Grid in physical units, ±3 sigma around MLE
    dk = max(abs(result.get("kappa_Cep_err", 5e5)), 5e5)
    db = max(abs(result.get("beta_X_err", 5e7)), 5e7)

    kappa_grid = np.linspace(kappa_mle - 3 * dk, kappa_mle + 3 * dk, n_grid)
    beta_grid = np.linspace(beta_mle - 3 * db, beta_mle + 3 * db, n_grid)
    logL_grid = np.full((n_grid, n_grid), -np.inf)

    for i, k in enumerate(kappa_grid):
        for j, b in enumerate(beta_grid):
            # Profile out H_app and sigma_int_v
            def _profile(params):
                H_app, sigma_int_v = params[0], max(params[1], 0.01)
                cz_model = _velocity_model(cz_obs, d_obs, X, H_app, k, b)
                resid = cz_obs - cz_model
                sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
                var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
                var = np.maximum(var, 0.01)
                return 0.5 * np.sum(resid**2 / var + np.log(var))

            res_prof = optimize.minimize(
                _profile, [70.0, 3.0], method="L-BFGS-B",
                bounds=[(30.0, 90.0), (0.01, 50.0)]
            )
            logL_grid[j, i] = -res_prof.fun

    return {
        "kappa_mle": kappa_mle,
        "beta_mle": beta_mle,
        "kappa_grid": kappa_grid.tolist(),
        "beta_grid": beta_grid.tolist(),
        "logL_grid": logL_grid.tolist(),
    }


def parameter_correlation(cz_obs, d_obs, X, sigma_mu, sigma_v):
    """Estimate corr(kappa, beta) from finite-difference Hessian at MLE (scaled params)."""
    result = fit_model(cz_obs, d_obs, X, sigma_mu, sigma_v, "Kβ")
    # MLE in scaled parameter space
    x_mle = np.array([
        result["H_app"],
        result["kappa_Cep"] / KAPPA_SCALE,
        result["beta_X"] / BETA_SCALE,
        result["sigma_int_v"],
    ])

    try:
        hess = optimize.approx_fprime(
            x_mle,
            lambda x: optimize.approx_fprime(
                x,
                lambda p: _neg_logL_velocity(p, cz_obs, d_obs, X, sigma_mu, sigma_v, "Kβ"),
                1e-5,
            ),
            1e-5,
        )
        cov = np.linalg.pinv(hess, rcond=1e-12)
        corr = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if cov[i, i] > 0 and cov[j, j] > 0:
                    corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
        kappa_beta_corr = float(corr[1, 2])
    except:
        kappa_beta_corr = np.nan

    return {
        "corr_kappa_beta": kappa_beta_corr,
        "kappa_Cep": result["kappa_Cep"],
        "beta_X": result["beta_X"],
    }


def loho_sign_stability(cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, host_names):
    """Leave-one-host-out sign stability for beta_X."""
    n = len(cz_obs)
    beta_Xs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        res = fit_model(cz_obs[mask], d_obs[mask], X[mask], sigma_mu[mask], sigma_v, model_type)
        beta_Xs.append(res.get("beta_X", np.nan))

    beta_arr = np.array(beta_Xs)
    n_positive = int(np.sum(beta_arr > 0))
    most_influential = host_names[np.nanargmax(np.abs(beta_arr - np.nanmean(beta_arr)))]

    return {
        "beta_X_mean": float(np.nanmean(beta_arr)),
        "beta_X_std": float(np.nanstd(beta_arr)),
        "beta_X_min": float(np.nanmin(beta_arr)),
        "beta_X_max": float(np.nanmax(beta_arr)),
        "n_positive": n_positive,
        "N_hosts": n,
        "most_influential_host": most_influential,
    }


def bootstrap_sign_stability(cz_obs, d_obs, X, sigma_mu, sigma_v, model_type, n_boot=1000, seed=42):
    """Bootstrap sign stability for beta_X."""
    rng = np.random.default_rng(seed)
    n = len(cz_obs)
    beta_Xs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        res = fit_model(cz_obs[idx], d_obs[idx], X[idx], sigma_mu[idx], sigma_v, model_type)
        beta_Xs.append(res.get("beta_X", np.nan))

    beta_arr = np.array(beta_Xs)
    ci_low, ci_high = np.percentile(beta_arr, [2.5, 97.5])
    return {
        "beta_X_boot_mean": float(np.nanmean(beta_arr)),
        "beta_X_boot_std": float(np.nanstd(beta_arr)),
        "beta_X_ci_low": float(ci_low),
        "beta_X_ci_high": float(ci_high),
        "frac_positive": float(np.mean(beta_arr > 0)),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run():
    print_status("Step 38: Hierarchical Timefield Ladder", "SECTION")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_S = load_host_metadata()
    sigma_ref = np.sqrt(
        (30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55)
    )

    df_hosts = compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref)
    print_status(f"Computed {len(df_hosts)} calibrator hosts", "INFO")

    PRIMARY_Z_CUT = 0.0035
    df_primary = df_hosts[
        (~df_hosts["is_anchor"])
        & df_hosts["z_hd"].notna()
        & (df_hosts["z_hd"] >= PRIMARY_Z_CUT)
    ].copy()

    print_status(f"Primary sample (z >= {PRIMARY_Z_CUT}): {len(df_primary)} hosts", "INFO")

    # Build covariates for primary
    cz_pri, mu_err_pri, z_pri, d_pri, X_pri, Z_pri, P_pri, zc_pri = _build_covariates_for_subset(
        df_primary, host_S, sigma_ref
    )

    # Model list
    models = ["H0", "Hβ", "K0", "Kfixβ", "Kpriorβ", "Kβ"]
    sigma_v_values = [150, 250, 500]
    results = []
    debug_results = []

    for sigma_v in sigma_v_values:
        print_status(f"sigma_v = {sigma_v} km/s", "SECTION")

        for model_type in models:
            print_status(f"  Fitting {model_type}...", "INFO")
            res = fit_model(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v, model_type)
            res["sigma_v"] = sigma_v
            res["sample"] = "primary"
            res["z_cut"] = 0.0
            results.append(res)

            # Key diagnostics
            if model_type in ("Hβ", "K0", "Kfixβ", "Kβ"):
                bx = res.get("beta_X", np.nan)
                bx_err = res.get("beta_X_err", np.nan)
                sig = abs(bx) / bx_err if bx_err and bx_err > 0 else np.nan
                kx = res.get("kappa_Cep", np.nan)
                kx_err = res.get("kappa_Cep_err", np.nan)
                k_sig = abs(kx) / kx_err if kx_err and kx_err > 0 else np.nan
                print_status(
                    f"    {model_type}: H0={res['H_app']:.2f}, "
                    f"beta_X={bx:+.3e} ({sig:.1f}σ), "
                    f"kappa={kx:.3e} ({k_sig:.1f}σ), "
                    f"sigma_int_v={res['sigma_int_v']:.2f}, "
                    f"chi2/dof={res['chi2']:.1f}/{res['dof']}",
                    "INFO",
                )

    # ========================================================================
    # Redshift cut sensitivity
    # ========================================================================
    print_status("Redshift cut sensitivity", "SECTION")
    z_cuts = [0.005, 0.0075]
    for z_cut in z_cuts:
        mask = z_pri >= z_cut
        n_cut = mask.sum()
        print_status(f"  z >= {z_cut}: N={n_cut}", "INFO")
        if n_cut < 10:
            continue

        cz_cut = cz_pri[mask]
        d_cut = d_pri[mask]
        X_cut = center_scale(X_pri[mask])
        mu_err_cut = mu_err_pri[mask]

        for sigma_v in [250]:
            for model_type in ["H0", "Hβ", "K0", "Kβ"]:
                res = fit_model(cz_cut, d_cut, X_cut, mu_err_cut, sigma_v, model_type)
                res["sigma_v"] = sigma_v
                res["sample"] = "primary"
                res["z_cut"] = z_cut
                results.append(res)

    # ========================================================================
    # Debug checks (on primary, sigma_v=250, Kβ model)
    # ========================================================================
    print_status("Debug checks (primary, sigma_v=250, Kβ)", "SECTION")
    sigma_v_dbg = 250

    # 1. Injection recovery: kappa only
    print_status("  Injection recovery: kappa=2e5, beta=0", "INFO")
    inj_k = injection_recovery(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, "K0", true_kappa=2e5, true_beta=0.0)
    debug_results.append({"check": "injection_kappa_only", **inj_k})
    print_status(f"    recovered kappa={inj_k['recovered_kappa']:.3e}", "INFO")

    # 2. Injection recovery: beta only
    print_status("  Injection recovery: kappa=0, beta=2e7", "INFO")
    inj_b = injection_recovery(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, "Hβ", true_kappa=0.0, true_beta=2e7)
    debug_results.append({"check": "injection_beta_only", **inj_b})
    print_status(f"    recovered beta={inj_b['recovered_beta']:.3e}", "INFO")

    # 3. Injection recovery: both
    print_status("  Injection recovery: kappa=2e5, beta=2e7", "INFO")
    inj_kb = injection_recovery(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, "Kβ", true_kappa=2e5, true_beta=2e7)
    debug_results.append({"check": "injection_kappa_beta", **inj_kb})
    print_status(
        f"    recovered kappa={inj_kb['recovered_kappa']:.3e}, beta={inj_kb['recovered_beta']:.3e}",
        "INFO",
    )

    # 4. Likelihood contour
    print_status("  Likelihood contour in kappa-beta plane", "INFO")
    contour = likelihood_contour(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, n_grid=30)
    debug_results.append({"check": "likelihood_contour", "contour": contour})

    # 5. Parameter correlation
    print_status("  Parameter correlation corr(kappa, beta)", "INFO")
    corr = parameter_correlation(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg)
    debug_results.append({"check": "parameter_correlation", **corr})
    print_status(f"    corr(kappa, beta) = {corr['corr_kappa_beta']:.3f}", "INFO")

    # 6. LOHO sign stability
    print_status("  LOHO sign stability", "INFO")
    loho = loho_sign_stability(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, "Kβ", df_primary["host"].values)
    debug_results.append({"check": "loho", **loho})
    print_status(
        f"    beta_X = {loho['beta_X_mean']:.3e} +/- {loho['beta_X_std']:.3e}, "
        f"positive in {loho['n_positive']}/{loho['N_hosts']}",
        "INFO",
    )

    # 7. Bootstrap sign stability
    print_status("  Bootstrap sign stability (N=500)", "INFO")
    boot = bootstrap_sign_stability(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, "Kβ", n_boot=500)
    debug_results.append({"check": "bootstrap", **boot})
    print_status(
        f"    beta_X = {boot['beta_X_boot_mean']:.3e} +/- {boot['beta_X_boot_std']:.3e}, "
        f"95% CI [{boot['beta_X_ci_low']:.3e}, {boot['beta_X_ci_high']:.3e}], "
        f"frac_positive={boot['frac_positive']:.3f}",
        "INFO",
    )

    # ========================================================================
    # Summary: key diagnostic
    # ========================================================================
    print_status("Summary: what do the data prefer?", "SECTION")
    for r in results:
        if r["sample"] != "primary" or r["z_cut"] != 0.0 or r["sigma_v"] != 250:
            continue
        k = r.get("kappa_Cep", np.nan)
        k_err = r.get("kappa_Cep_err", np.nan)
        k_sig = abs(k) / k_err if k_err and k_err > 0 else np.nan
        b = r.get("beta_X", np.nan)
        b_err = r.get("beta_X_err", np.nan)
        b_sig = abs(b) / b_err if b_err and b_err > 0 else np.nan
        print_status(
            f"  {r['model']:7s}: H0={r['H_app']:.2f}, "
            f"kappa={k:+.3e} ({k_sig:.1f}σ), "
            f"beta={b:+.3e} ({b_sig:.1f}σ), "
            f"chi2/dof={r['chi2']:.1f}/{r['dof']}",
            "INFO",
        )

    # ========================================================================
    # Save
    # ========================================================================
    df_out = pd.DataFrame(results)
    out_csv = OUT_DIR / "step_38_hierarchical_timefield_ladder.csv"
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    out_json = OUT_DIR / "step_38_hierarchical_timefield_ladder.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    debug_json = OUT_DIR / "step_38_debug_checks.json"
    with open(debug_json, "w") as f:
        json.dump(debug_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved debug checks to {debug_json}", "SUCCESS")

    return results, debug_results


if __name__ == "__main__":
    run()
