#!/usr/bin/env python3
"""
step_40_flow_sky_controls.py

Flow / Sky-Position Controls for Apparent Hubble Environment

Tests whether Gamma_X > 0 survives explicit controls for redshift trend,
sky dipole/quadrupole, and group offsets.

Core model: cz_i = d_i (H_app + Gamma_X X_i) + controls + v_i

Also runs stricter permutation tests:
  - redshift-bin preserving
  - sky-bin preserving
  - redshift x sky cross-bin preserving
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SH0ES_DIR = DATA_DIR / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10) / 5.0
GAMMA_SCALE = 1e7


def print_status(msg, level="INFO"):
    prefix = {"SECTION": "=" * 60, "INFO": "[INFO]", "SUCCESS": "[SUCCESS]",
              "WARNING": "[WARNING]", "ERROR": "[ERROR]"}.get(level, "[INFO]")
    if level == "SECTION":
        print(f"\n{prefix}\n{msg}\n{prefix}")
    else:
        print(f"{prefix} {msg}")


def build_host_x(sigma, sigma_ref, S=1.0):
    if sigma is None or sigma <= 0 or sigma_ref <= 0:
        return 0.0
    return S * (sigma**2 - sigma_ref**2) / (C_KM_S ** 2)


def center_scale(v):
    return v - np.mean(v)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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
    host_sigma, host_z, host_S = {}, {}, {}
    host_ra, host_dec, host_group = {}, {}, {}
    for _, row in df.iterrows():
        name = row["normalized_name"]
        sigma = row["sigma_inferred"]
        z_hd = row["z_hd"]
        S = row.get("shear_suppression", 1.0)
        if pd.isna(S):
            S = 1.0
        ra = row.get("ra", np.nan)
        dec = row.get("dec", np.nan)
        group = row.get("pgc", np.nan)

        for key in [name]:
            host_sigma[key] = sigma
            host_S[key] = float(S)
            host_ra[key] = float(ra) if pd.notna(ra) else np.nan
            host_dec[key] = float(dec) if pd.notna(dec) else np.nan
            host_group[key] = int(group) if pd.notna(group) else np.nan
            if pd.notna(z_hd) and z_hd > 0:
                host_z[key] = z_hd

        compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
        if compact.startswith(("N", "U")):
            parts = compact[1:]
            if parts.isdigit():
                for alias in [compact[0] + parts.zfill(4),
                              compact[0] + parts.lstrip("0")]:
                    host_sigma[alias] = sigma
                    host_S[alias] = float(S)
                    host_ra[alias] = float(ra) if pd.notna(ra) else np.nan
                    host_dec[alias] = float(dec) if pd.notna(dec) else np.nan
                    host_group[alias] = int(group) if pd.notna(group) else np.nan
                    if pd.notna(z_hd) and z_hd > 0:
                        host_z[alias] = z_hd
        if compact.startswith("N"):
            ngc_name = "NGC" + compact[1:]
            host_sigma[ngc_name] = sigma
            host_S[ngc_name] = float(S)
            host_ra[ngc_name] = float(ra) if pd.notna(ra) else np.nan
            host_dec[ngc_name] = float(dec) if pd.notna(dec) else np.nan
            host_group[ngc_name] = int(group) if pd.notna(group) else np.nan
            if pd.notna(z_hd) and z_hd > 0:
                host_z[ngc_name] = z_hd

    explicit = {"M1337": "N1337", "N105A": "N105", "N976A": "N976"}
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_sigma and sh0es_name not in host_sigma:
            host_sigma[sh0es_name] = host_sigma[csv_name]
            host_S[sh0es_name] = host_S[csv_name]
            host_ra[sh0es_name] = host_ra.get(csv_name, np.nan)
            host_dec[sh0es_name] = host_dec.get(csv_name, np.nan)
            host_group[sh0es_name] = host_group.get(csv_name, np.nan)
            if csv_name in host_z:
                host_z[sh0es_name] = host_z[csv_name]

    return host_sigma, host_z, host_S, host_ra, host_dec, host_group


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
    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}

    for host_name, mu_idx in mu_params:
        if host_name not in host_sigma:
            continue
        mu_fit = theta[mu_idx]
        has_ceph = False
        for r in range(L.shape[0]):
            if abs(L[r, mu_idx]) > 0.01:
                nonzero = np.where(np.abs(L[r]) > 0.01)[0]
                params = [q[j] for j in nonzero]
                if "MHW1" in params:
                    has_ceph = True
                    break
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

    return pd.DataFrame({
        "host": hosts, "mu": mus, "mu_err": mu_errs,
        "sigma": sigmas, "z_hd": zs, "is_anchor": is_anchors,
    })


def _build_covariates(df_subset, host_S, host_ra, host_dec, host_group, sigma_ref):
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
    X_tep_c = center_scale(X_tep)

    ra = np.array([host_ra.get(h, np.nan) for h in df_subset["host"].values])
    dec = np.array([host_dec.get(h, np.nan) for h in df_subset["host"].values])
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    n_x = np.cos(dec_rad) * np.cos(ra_rad)
    n_y = np.cos(dec_rad) * np.sin(ra_rad)
    n_z = np.sin(dec_rad)

    groups = np.array([host_group.get(h, np.nan) for h in df_subset["host"].values])
    has_group = ~np.isnan(groups)

    return cz_obs, mu_err, z, d_obs, X_tep_c, n_x, n_y, n_z, groups, has_group


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def _neg_logL(params, cz_obs, d_obs, X, sigma_mu, sigma_v, model_type,
              n_x=None, n_y=None, n_z=None, z=None, groups=None, has_group=None):
    sigma_int_v = max(params[-1], 0.01)
    idx = 0
    H_app = params[idx]; idx += 1
    gamma_param = params[idx]; idx += 1
    Gamma_X = 0.0 if model_type == "M0" else gamma_param * GAMMA_SCALE

    cz_model = d_obs * (H_app + Gamma_X * X)

    if model_type in ("M2", "M6"):
        alpha_z = params[idx]; idx += 1
        cz_model += d_obs * alpha_z * z
    if model_type in ("M3", "M4", "M6"):
        Dx, Dy, Dz = params[idx], params[idx+1], params[idx+2]
        idx += 3
        cz_model += Dx * n_x + Dy * n_y + Dz * n_z
    if model_type in ("M4", "M6"):
        Qxx, Qyy, Qxy = params[idx], params[idx+1], params[idx+2]
        idx += 3
        cz_model += Qxx * (n_x**2 - 1/3) + Qyy * (n_y**2 - 1/3) + Qxy * n_x * n_y
    if model_type in ("M5", "M6") and has_group is not None and np.any(has_group):
        unique_groups = np.unique(groups[has_group])
        goff = {}
        for g in unique_groups:
            goff[g] = params[idx]; idx += 1
        for i in range(len(cz_obs)):
            if has_group[i] and groups[i] in goff:
                cz_model[i] += goff[groups[i]]

    resid = cz_obs - cz_model
    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
    var = np.maximum(var, 0.01)
    return 0.5 * np.sum(resid**2 / var + np.log(var))


def fit_model(cz_obs, d_obs, X, sigma_mu, sigma_v, model_type,
              n_x=None, n_y=None, n_z=None, z=None, groups=None, has_group=None):
    n = len(cz_obs)
    H_app_init = np.median(cz_obs / d_obs)
    x0 = [H_app_init, 2.0, 3.0]
    bounds = [(30.0, 90.0), (-100.0, 100.0), (0.01, 50.0)]

    if model_type in ("M2", "M6"):
        x0.append(0.0); bounds.append((-1e4, 1e4))
    if model_type in ("M3", "M4", "M6"):
        x0.extend([0.0, 0.0, 0.0])
        bounds.extend([(-5000.0, 5000.0)] * 3)
    if model_type in ("M4", "M6"):
        x0.extend([0.0, 0.0, 0.0])
        bounds.extend([(-5000.0, 5000.0)] * 3)
    if model_type in ("M5", "M6") and has_group is not None and np.any(has_group):
        unique_groups = np.unique(groups[has_group])
        x0.extend([0.0] * len(unique_groups))
        bounds.extend([(-5000.0, 5000.0)] * len(unique_groups))

    x0 = np.array(x0)

    def obj(p):
        return _neg_logL(p, cz_obs, d_obs, X, sigma_mu, sigma_v, model_type,
                         n_x, n_y, n_z, z, groups, has_group)

    res = optimize.minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    for H_init in [65.0, 70.0, 75.0, 80.0]:
        for g_init in [1.0, 2.0, 2.3, 2.5, 3.0, 0.0, -1.0]:
            x0_try = x0.copy()
            x0_try[0] = H_init
            x0_try[1] = g_init
            res_try = optimize.minimize(obj, x0_try, method="L-BFGS-B", bounds=bounds)
            if res_try.fun < res.fun:
                res = res_try

    idx = 0
    H_app = res.x[idx]; idx += 1
    gamma_param = res.x[idx]; idx += 1
    Gamma_X = 0.0 if model_type == "M0" else gamma_param * GAMMA_SCALE
    sigma_int_v = res.x[idx]; idx += 1

    alpha_z = np.nan
    Dx = Dy = Dz = np.nan
    Qxx = Qyy = Qxy = np.nan
    group_offsets = {}

    if model_type in ("M2", "M6"):
        alpha_z = res.x[idx]; idx += 1
    if model_type in ("M3", "M4", "M6"):
        Dx, Dy, Dz = res.x[idx], res.x[idx+1], res.x[idx+2]
        idx += 3
    if model_type in ("M4", "M6"):
        Qxx, Qyy, Qxy = res.x[idx], res.x[idx+1], res.x[idx+2]
        idx += 3
    if model_type in ("M5", "M6") and has_group is not None and np.any(has_group):
        unique_groups = np.unique(groups[has_group])
        for g in unique_groups:
            group_offsets[int(g)] = float(res.x[idx]); idx += 1

    try:
        hess = optimize.approx_fprime(
            res.x, lambda x: optimize.approx_fprime(x, obj, 1e-5), 1e-5,
        )
        cov = np.linalg.pinv(hess, rcond=1e-12)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se = np.full(len(res.x), np.nan)

    gamma_se = se[1] * GAMMA_SCALE if model_type != "M0" else np.nan
    gamma_sig = abs(Gamma_X) / gamma_se if gamma_se and gamma_se > 0 else np.nan

    cz_model = d_obs * (H_app + Gamma_X * X)
    if model_type in ("M2", "M6"):
        cz_model += d_obs * alpha_z * z
    if model_type in ("M3", "M4", "M6"):
        cz_model += Dx * n_x + Dy * n_y + Dz * n_z
    if model_type in ("M4", "M6"):
        cz_model += Qxx * (n_x**2 - 1/3) + Qyy * (n_y**2 - 1/3) + Qxy * n_x * n_y
    if model_type in ("M5", "M6") and has_group is not None and np.any(has_group):
        for i in range(len(cz_obs)):
            if has_group[i] and groups[i] in group_offsets:
                cz_model[i] += group_offsets[groups[i]]

    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
    resid = cz_obs - cz_model
    chi2 = float(np.sum(resid**2 / var))
    logL = -res.fun
    dof = n - len(res.x)

    return {
        "model": model_type, "n_hosts": n, "n_params": len(res.x), "dof": dof,
        "H_app": float(H_app), "H_app_err": float(se[0]),
        "Gamma_X": float(Gamma_X), "Gamma_X_err": float(gamma_se), "Gamma_X_sig": float(gamma_sig),
        "alpha_z": float(alpha_z), "dipole_x": float(Dx), "dipole_y": float(Dy), "dipole_z": float(Dz),
        "quadrupole_xx": float(Qxx), "quadrupole_yy": float(Qyy), "quadrupole_xy": float(Qxy),
        "sigma_int_v": float(sigma_int_v), "chi2": chi2,
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
        "logL": float(logL), "AIC": -2 * logL + 2 * len(res.x),
        "BIC": -2 * logL + len(res.x) * np.log(n),
        "status": "converged" if res.success else "fallback",
        "group_offsets": group_offsets,
    }


# ---------------------------------------------------------------------------
# Fast WLS for permutation tests (avoids slow optimizer)
# ---------------------------------------------------------------------------
def _wls_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v):
    """Fast weighted least squares for Gamma_X only."""
    n = len(cz_obs)
    H_app_init = np.median(cz_obs / d_obs)
    # Use fixed H_app, fit only Gamma_X
    y = cz_obs - d_obs * H_app_init
    x = d_obs * X
    # Weights: 1 / sigma_v^2 (distance uncertainty subdominant for permutation)
    w = 1.0 / (sigma_v**2 + (LN10_OVER_5 * np.abs(cz_obs) * sigma_mu)**2)
    w = np.maximum(w, 1e-10)
    Xmat = np.column_stack([np.ones(n), x])
    W = np.diag(w)
    beta = np.linalg.lstsq(Xmat.T @ W @ Xmat, Xmat.T @ W @ y, rcond=None)[0]
    return beta[1]  # Gamma_X


def _permute_within_bins(X, bins, seed=42):
    rng = np.random.default_rng(seed)
    X_perm = X.copy()
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() > 1:
            X_perm[mask] = rng.permutation(X_perm[mask])
    return X_perm


def permutation_suite(cz_obs, d_obs, X, sigma_mu, sigma_v, z, n_x, n_y, n_z, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(cz_obs)
    gamma_true = abs(_wls_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v))
    results = {}

    # Standard
    gp = []
    for _ in range(n_perm):
        Xp = rng.permutation(X)
        gp.append(abs(_wls_gamma(cz_obs, d_obs, Xp, sigma_mu, sigma_v)))
    results["standard"] = float(np.mean(np.array(gp) >= gamma_true))

    # Redshift-bin
    z_bins = np.digitize(z, bins=np.percentile(z, [25, 50, 75]))
    gp = []
    for _ in range(n_perm):
        Xp = _permute_within_bins(X, z_bins, seed=rng.integers(0, 1e9))
        gp.append(abs(_wls_gamma(cz_obs, d_obs, Xp, sigma_mu, sigma_v)))
    results["redshift_binned"] = float(np.mean(np.array(gp) >= gamma_true))

    # Sky-bin (RA octants)
    ra = np.rad2deg(np.arctan2(n_y, n_x))
    ra_bins = np.digitize(ra, bins=np.linspace(ra.min(), ra.max(), 5))
    gp = []
    for _ in range(n_perm):
        Xp = _permute_within_bins(X, ra_bins, seed=rng.integers(0, 1e9))
        gp.append(abs(_wls_gamma(cz_obs, d_obs, Xp, sigma_mu, sigma_v)))
    results["sky_binned"] = float(np.mean(np.array(gp) >= gamma_true))

    # Redshift x sky cross-bin
    cross_bins = z_bins * 10 + ra_bins
    gp = []
    for _ in range(n_perm):
        Xp = _permute_within_bins(X, cross_bins, seed=rng.integers(0, 1e9))
        gp.append(abs(_wls_gamma(cz_obs, d_obs, Xp, sigma_mu, sigma_v)))
    results["redshift_sky_binned"] = float(np.mean(np.array(gp) >= gamma_true))

    results["gamma_true"] = float(gamma_true)
    return results


# ---------------------------------------------------------------------------
# LOHO and bootstrap
# ---------------------------------------------------------------------------
def loho_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, host_names):
    n = len(cz_obs)
    gammas = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r = fit_model(cz_obs[mask], d_obs[mask], X[mask], sigma_mu[mask], sigma_v, "M1")
        gammas.append(r["Gamma_X"])
    gammas_arr = np.array(gammas)
    most_influential = host_names[np.nanargmax(np.abs(gammas_arr - np.nanmean(gammas_arr)))]
    return {
        "Gamma_X_mean": float(np.nanmean(gammas_arr)),
        "Gamma_X_std": float(np.nanstd(gammas_arr)),
        "Gamma_X_min": float(np.nanmin(gammas_arr)),
        "Gamma_X_max": float(np.nanmax(gammas_arr)),
        "n_positive": int(np.sum(gammas_arr > 0)),
        "N_hosts": n,
        "most_influential_host": most_influential,
    }


def bootstrap_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(cz_obs)
    gammas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = fit_model(cz_obs[idx], d_obs[idx], X[idx], sigma_mu[idx], sigma_v, "M1")
        gammas.append(r["Gamma_X"])
    gammas_arr = np.array(gammas)
    ci_low, ci_high = np.percentile(gammas_arr, [2.5, 97.5])
    return {
        "Gamma_X_boot_mean": float(np.mean(gammas_arr)),
        "Gamma_X_boot_std": float(np.std(gammas_arr)),
        "Gamma_X_ci_low": float(ci_low),
        "Gamma_X_ci_high": float(ci_high),
        "frac_positive": float(np.mean(gammas_arr > 0)),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run():
    print_status("Step 40: Flow / Sky Controls", "SECTION")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_S, host_ra, host_dec, host_group = load_host_metadata()
    sigma_ref = np.sqrt((30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55))

    df_hosts = compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref)
    print_status(f"Computed {len(df_hosts)} calibrator hosts", "INFO")

    PRIMARY_Z_CUT = 0.0035
    df_primary = df_hosts[
        (~df_hosts["is_anchor"]) & df_hosts["z_hd"].notna() & (df_hosts["z_hd"] >= PRIMARY_Z_CUT)
    ].copy()
    print_status(f"Primary sample (z >= {PRIMARY_Z_CUT}): {len(df_primary)} hosts", "INFO")

    cz_pri, mu_err_pri, z_pri, d_pri, X_pri, n_x, n_y, n_z, groups, has_group = \
        _build_covariates(df_primary, host_S, host_ra, host_dec, host_group, sigma_ref)

    n_ra = np.sum(~np.isnan(n_x))
    n_group = np.sum(has_group)
    # Count unique groups with multiple members
    group_counts = {}
    for g in groups[has_group]:
        group_counts[g] = group_counts.get(g, 0) + 1
    n_meaningful_groups = sum(1 for c in group_counts.values() if c >= 2)
    print_status(f"Sky coordinates: {n_ra}/{len(df_primary)}", "INFO")
    print_status(f"Group IDs: {n_group}/{len(df_primary)}, meaningful groups (>=2 hosts): {n_meaningful_groups}", "INFO")

    sigma_v_values = [150, 250, 500]
    model_types = ["M0", "M1", "M2", "M3", "M4"]
    # Only add M5/M6 if there are groups with multiple members
    if n_meaningful_groups >= 2:
        model_types.append("M5")
    else:
        print_status("Skipping M5/M6 group models: no meaningful groups found", "INFO")
    model_types.append("M6")

    results = []
    for sigma_v in sigma_v_values:
        print_status(f"sigma_v = {sigma_v} km/s", "SECTION")
        for model_type in model_types:
            print_status(f"  Fitting {model_type}...", "INFO")
            res = fit_model(
                cz_pri, d_pri, X_pri, mu_err_pri, sigma_v, model_type,
                n_x=n_x, n_y=n_y, n_z=n_z, z=z_pri,
                groups=groups, has_group=has_group if n_meaningful_groups >= 2 else None,
            )
            res["sigma_v"] = sigma_v
            res["sample"] = "primary"
            results.append(res)

            gx = res["Gamma_X"]
            gx_err = res["Gamma_X_err"]
            sig = abs(gx) / gx_err if gx_err and gx_err > 0 else np.nan
            sig_str = f"{sig:.1f}σ" if np.isfinite(sig) and sig < 1000 else "n/a"
            dipole_mag = np.sqrt(res["dipole_x"]**2 + res["dipole_y"]**2 + res["dipole_z"]**2)
            dipole_str = f"{dipole_mag:.1f}" if np.isfinite(dipole_mag) else "n/a"
            print_status(
                f"    {model_type}: H0={res['H_app']:.2f}, "
                f"Gamma_X={gx:+.3e} ({sig_str}), "
                f"chi2/dof={res['chi2']:.1f}/{res['dof']}, "
                f"dipole={dipole_str}",
                "INFO",
            )

    # Permutation suite
    print_status("Permutation suite (sigma_v=250)", "SECTION")
    perm = permutation_suite(cz_pri, d_pri, X_pri, mu_err_pri, 250, z_pri, n_x, n_y, n_z, n_perm=2000)
    for k, v in perm.items():
        if k == "gamma_true":
            print_status(f"  gamma_true = {v:.3e}", "INFO")
        else:
            print_status(f"  {k}: p = {v:.4f}", "INFO")

    # LOHO and bootstrap
    print_status("LOHO and bootstrap (M1, sigma_v=250)", "SECTION")
    loho = loho_gamma(cz_pri, d_pri, X_pri, mu_err_pri, 250, df_primary["host"].values)
    print_status(
        f"  LOHO: Gamma_X = {loho['Gamma_X_mean']:.3e} +/- {loho['Gamma_X_std']:.3e}, "
        f"positive in {loho['n_positive']}/{loho['N_hosts']}",
        "INFO",
    )
    boot = bootstrap_gamma(cz_pri, d_pri, X_pri, mu_err_pri, 250, n_boot=1000)
    print_status(
        f"  Bootstrap: Gamma_X = {boot['Gamma_X_boot_mean']:.3e} +/- {boot['Gamma_X_boot_std']:.3e}, "
        f"95% CI [{boot['Gamma_X_ci_low']:.3e}, {boot['Gamma_X_ci_high']:.3e}], "
        f"frac_positive={boot['frac_positive']:.3f}",
        "INFO",
    )

    # Model comparison summary
    print_status("Model comparison (primary, sigma_v=250)", "SECTION")
    print(f"{'Model':>6s} {'N_param':>7s} {'Gamma_X':>14s} {'sig':>5s} {'chi2/dof':>10s} {'AIC':>10s} {'BIC':>10s}")
    print("-" * 80)
    for r in results:
        if r["sigma_v"] != 250:
            continue
        gx = r["Gamma_X"]
        gx_err = r["Gamma_X_err"]
        sig = abs(gx) / gx_err if gx_err and gx_err > 0 else np.nan
        sig_out = f"{sig:5.1f}" if np.isfinite(sig) and sig < 1000 else "  n/a"
        print(f"{r['model']:>6s} {r['n_params']:>7d} {gx:>+14.3e} {sig_out:>5s} {r['chi2']:.1f}/{r['dof']} {r['AIC']:>10.1f} {r['BIC']:>10.1f}")

    # Save
    df_out = pd.DataFrame(results)
    out_csv = OUT_DIR / "step_40_flow_sky_controls.csv"
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    out_json = OUT_DIR / "step_40_flow_sky_controls.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    test_results = {"permutation": perm, "loho": loho, "bootstrap": boot}
    test_json = OUT_DIR / "step_40_statistical_tests.json"
    with open(test_json, "w") as f:
        json.dump(test_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved test results to {test_json}", "SUCCESS")

    return results, test_results


if __name__ == "__main__":
    run()
