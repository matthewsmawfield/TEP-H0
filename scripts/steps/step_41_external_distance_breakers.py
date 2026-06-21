#!/usr/bin/env python3
"""
step_41_external_distance_breakers.py

External Distance Breakers: Separate κ_Cep from β_X

Uses non-Cepheid distance channels (TRGB) to break the degeneracy between
Cepheid-distance bias (κ_Cep) and apparent-redshift bias (β_X).

Core equations:
    μ_Cepheid,i = μ_true,i - κ_Cep * X_i + ε_Cep,i
    μ_ext,i     = μ_true,i + Δ_m + ε_ext,i
    cz_i        = d_true,i * (H_app + β_X * X_i) + v_i

For hosts with both Cepheid and external distances:
    Δμ_i = μ_Cep,i - μ_ext,i = -κ_Cep * X_i - Δ_m + ε_i

Model grid:
    E0:       κ=0, β=0   (null)
    Eβ:       κ=0, β free
    Eκ:       κ free, β=0
    Eκβ:      κ free, β free
    Eκpriorβ: κ ~ N(κ_canonical, σ_κ²), β free
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SH0ES_DIR = DATA_DIR / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
TRGB_PATH = BASE_DIR / "results" / "outputs" / "step_15_trgb_hosts_data.csv"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10) / 5.0
GAMMA_SCALE = 1e7
KAPPA_SCALE = 1e5
KAPPA_CANONICAL = 9.7e5
KAPPA_PRIOR_SIGMA = 4.0e5


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
    host_sigma, host_S = {}, {}
    host_z = {}
    for _, row in df.iterrows():
        name = row["normalized_name"]
        sigma = row["sigma_inferred"]
        z_hd = row["z_hd"]
        S = row.get("shear_suppression", 1.0)
        if pd.isna(S):
            S = 1.0
        for key in [name]:
            host_sigma[key] = sigma
            host_S[key] = float(S)
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
                    if alias not in host_z and pd.notna(z_hd) and z_hd > 0:
                        host_z[alias] = z_hd
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


def load_external_distances():
    """Load external distance data. Currently TRGB only."""
    records = []
    if TRGB_PATH.exists():
        df = pd.read_csv(TRGB_PATH)
        for _, row in df.iterrows():
            match_name = row.get("match_name", "")
            galaxy = row.get("galaxy", "")
            name = str(match_name) if pd.notna(match_name) and str(match_name) else str(galaxy).replace(" ", "").replace("NGC", "N")
            records.append({
                "host": name,
                "mu_ext": float(row["mu_trgb"]),
                "mu_ext_err": float(row["mu_trgb_err"]),
                "method": "TRGB",
                "calibration_family": "Freedman2024",
                "is_independent_of_cepheids": True,
                "reference": row.get("reference", ""),
            })
    return pd.DataFrame(records)


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
        "host": hosts, "mu_cep": mus, "mu_cep_err": mu_errs,
        "sigma": sigmas, "z_hd": zs, "is_anchor": is_anchors,
    })


# ---------------------------------------------------------------------------
# Part 1: Differential κ test (hosts with both Cepheid and external)
# ---------------------------------------------------------------------------
def fit_differential_kappa(df_merged, sigma_ref, host_S):
    """Fit Δμ = μ_Cep - μ_ext = a - κ*X + ε. Returns κ_Cep estimate."""
    mu_cep = df_merged["mu_cep"].values
    mu_cep_err = df_merged["mu_cep_err"].values
    mu_ext = df_merged["mu_ext"].values
    mu_ext_err = df_merged["mu_ext_err"].values
    sigmas = df_merged["sigma"].values
    hosts = df_merged["host"].values

    X = np.array([build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
                  for s, h in zip(sigmas, hosts)])
    X_c = center_scale(X)

    # Internal scaling: X_scale = X * KAPPA_SCALE for numerical stability
    X_scale = X_c * KAPPA_SCALE

    dmu = mu_cep - mu_ext
    dmu_err = np.sqrt(mu_cep_err**2 + mu_ext_err**2)

    w = 1.0 / dmu_err**2
    n = len(dmu)
    Xmat = np.column_stack([np.ones(n), X_scale])
    W = np.diag(w)
    beta = np.linalg.lstsq(Xmat.T @ W @ Xmat, Xmat.T @ W @ dmu, rcond=None)[0]

    a = beta[0]
    kappa_param = -beta[1]  # kappa = kappa_param * KAPPA_SCALE
    kappa_fit = kappa_param * KAPPA_SCALE

    cov = np.linalg.pinv(Xmat.T @ W @ Xmat, rcond=1e-12)
    a_err = np.sqrt(cov[0, 0])
    kappa_param_err = np.sqrt(cov[1, 1])
    kappa_err = kappa_param_err * KAPPA_SCALE
    kappa_sig = abs(kappa_fit) / kappa_err if kappa_err > 0 else np.nan

    resid = dmu - (a - kappa_param * X_scale)
    chi2 = np.sum(w * resid**2)
    dof = n - 2

    return {
        "N_hosts": n,
        "a_offset": float(a),
        "a_offset_err": float(a_err),
        "kappa_Cep": float(kappa_fit),
        "kappa_Cep_err": float(kappa_err),
        "kappa_Cep_sig": float(kappa_sig),
        "chi2": float(chi2),
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
        "dof": dof,
    }


# ---------------------------------------------------------------------------
# Part 2: Velocity β test using external-informed μ_true
# ---------------------------------------------------------------------------
def fit_velocity_beta(df_primary, df_merged, kappa_est, sigma_ref, host_S, sigma_v):
    """
    Fit cz = d_true * (H_app + β*X) + v
    where d_true uses external μ (when available) or Cepheid-corrected μ.
    """
    # Build μ_true dictionary
    mu_true_dict = {}
    mu_true_err_dict = {}

    a_offset = df_merged["mu_cep"].mean() - df_merged["mu_ext"].mean() if len(df_merged) > 0 else 0.0

    for _, row in df_merged.iterrows():
        host = row["host"]
        mu_true_dict[host] = row["mu_ext"] + a_offset
        mu_true_err_dict[host] = row["mu_ext_err"]

    for _, row in df_primary.iterrows():
        host = row["host"]
        if host not in mu_true_dict:
            X = build_host_x(row["sigma"], sigma_ref, S=host_S.get(host, 1.0))
            mu_true_dict[host] = row["mu_cep"] + kappa_est * X
            mu_true_err_dict[host] = row["mu_cep_err"]

    hosts = df_primary["host"].values
    mu = np.array([mu_true_dict[h] for h in hosts])
    mu_err = np.array([mu_true_err_dict.get(h, 0.05) for h in hosts])
    z = df_primary["z_hd"].values
    sigmas = df_primary["sigma"].values

    d_obs = 10 ** ((mu - 25.0) / 5.0)
    cz_obs = C_KM_S * z

    X = np.array([build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
                  for s, h in zip(sigmas, hosts)])
    X_c = center_scale(X)

    # Internal scaling: X_scale = X * GAMMA_SCALE for numerical stability
    X_scale = X_c * GAMMA_SCALE

    y = cz_obs / d_obs
    x = X_scale
    w = d_obs**2 / (sigma_v**2 + (LN10_OVER_5 * cz_obs * mu_err)**2)
    w = np.maximum(w, 1e-10)

    n = len(y)
    Xmat = np.column_stack([np.ones(n), x])
    W = np.diag(w)
    beta = np.linalg.lstsq(Xmat.T @ W @ Xmat, Xmat.T @ W @ y, rcond=None)[0]

    H_app = beta[0]
    beta_param = beta[1]  # beta_X = beta_param * GAMMA_SCALE
    beta_X = beta_param * GAMMA_SCALE

    cov = np.linalg.pinv(Xmat.T @ W @ Xmat, rcond=1e-12)
    H_err = np.sqrt(cov[0, 0])
    beta_param_err = np.sqrt(cov[1, 1])
    beta_err = beta_param_err * GAMMA_SCALE
    beta_sig = abs(beta_X) / beta_err if beta_err > 0 else np.nan

    resid = y - (H_app + beta_param * x)
    chi2 = np.sum(w * resid**2)
    dof = n - 2

    return {
        "N_hosts": n,
        "H_app": float(H_app),
        "H_app_err": float(H_err),
        "beta_X": float(beta_X),
        "beta_X_err": float(beta_err),
        "beta_X_sig": float(beta_sig),
        "chi2": float(chi2),
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
        "dof": dof,
        "kappa_used": float(kappa_est),
    }


# ---------------------------------------------------------------------------
# Part 3: Joint model for hosts with both distances
# ---------------------------------------------------------------------------
def _neg_logL_joint(params, mu_cep, mu_cep_err, mu_ext, mu_ext_err,
                    cz_obs, X, sigma_v, n_both):
    idx = 0
    mu_true = params[idx:idx + n_both]
    idx += n_both
    H_app = params[idx]; idx += 1
    kappa_param = params[idx]; idx += 1
    beta_param = params[idx]; idx += 1
    sigma_int_v = max(params[idx], 0.01); idx += 1
    delta_m = params[idx]; idx += 1

    kappa_Cep = kappa_param * KAPPA_SCALE
    beta_X = beta_param * GAMMA_SCALE

    logL = 0.0
    for i in range(n_both):
        resid = mu_cep[i] - (mu_true[i] - kappa_Cep * X[i])
        logL += 0.5 * (resid**2 / mu_cep_err[i]**2 + np.log(mu_cep_err[i]**2))
        resid = mu_ext[i] - (mu_true[i] + delta_m)
        logL += 0.5 * (resid**2 / mu_ext_err[i]**2 + np.log(mu_ext_err[i]**2))
        d_true = 10 ** ((mu_true[i] - 25.0) / 5.0)
        cz_model = d_true * (H_app + beta_X * X[i])
        resid = cz_obs[i] - cz_model
        var = sigma_v**2 + (LN10_OVER_5 * np.abs(cz_model) * 0.05)**2 + sigma_int_v**2
        logL += 0.5 * (resid**2 / var + np.log(var))

    return logL


def fit_joint_model(df_merged, sigma_ref, host_S, sigma_v, model_type):
    mu_cep = df_merged["mu_cep"].values
    mu_cep_err = df_merged["mu_cep_err"].values
    mu_ext = df_merged["mu_ext"].values
    mu_ext_err = df_merged["mu_ext_err"].values
    sigmas = df_merged["sigma"].values
    hosts = df_merged["host"].values
    z = df_merged["z_hd"].values

    n = len(mu_cep)
    X = np.array([build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
                  for s, h in zip(sigmas, hosts)])

    mu_true_init = mu_cep.copy()
    H_app_init = np.median(C_KM_S * z / (10 ** ((mu_cep - 25.0) / 5.0)))

    x0 = list(mu_true_init)
    x0.extend([H_app_init, 0.0, 0.0, 3.0, 0.0])

    bounds = []
    for _ in range(n):
        bounds.append((20.0, 40.0))
    bounds.extend([
        (30.0, 90.0), (-100.0, 100.0), (-100.0, 100.0), (0.01, 50.0), (-5.0, 5.0),
    ])

    if model_type == "E0":
        bounds[n + 1] = (0.0, 0.0)  # kappa_param = 0
        bounds[n + 2] = (0.0, 0.0)  # beta_param = 0
    elif model_type == "Eβ":
        bounds[n + 1] = (0.0, 0.0)  # kappa_param = 0
    elif model_type == "Eκ":
        bounds[n + 2] = (0.0, 0.0)  # beta_param = 0

    def obj(p):
        return _neg_logL_joint(p, mu_cep, mu_cep_err, mu_ext, mu_ext_err,
                               C_KM_S * z, X, sigma_v, n)

    res = optimize.minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

    # Multi-start
    for H_init in [65.0, 70.0, 75.0, 80.0]:
        for k_init in [0.0, 5.0, 10.0, -5.0]:
            for b_init in [0.0, 1.0, 2.0, 2.3, 3.0]:
                x0_try = list(mu_true_init)
                x0_try.extend([H_init, k_init, b_init, 3.0, 0.0])
                res_try = optimize.minimize(obj, x0_try, method="L-BFGS-B", bounds=bounds)
                if res_try.fun < res.fun:
                    res = res_try

    idx = 0
    mu_true = res.x[idx:idx + n]
    idx += n
    H_app = res.x[idx]; idx += 1
    kappa_param = res.x[idx]; idx += 1
    beta_param = res.x[idx]; idx += 1
    sigma_int_v = res.x[idx]; idx += 1
    delta_m = res.x[idx]; idx += 1

    kappa_Cep = kappa_param * KAPPA_SCALE
    beta_X = beta_param * GAMMA_SCALE

    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            hess = optimize.approx_fprime(
                res.x, lambda x: optimize.approx_fprime(x, obj, 1e-5), 1e-5,
            )
            cov = np.linalg.pinv(hess, rcond=1e-12)
            se = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se = np.full(len(res.x), np.nan)

    kappa_se = se[n + 1] * KAPPA_SCALE
    beta_se = se[n + 2] * GAMMA_SCALE
    kappa_sig = abs(kappa_Cep) / kappa_se if kappa_se > 0 else np.nan
    beta_sig = abs(beta_X) / beta_se if beta_se > 0 else np.nan

    logL = -res.fun
    n_params = len(res.x)
    dof = n - n_params if n > n_params else 0

    return {
        "model": model_type,
        "N_hosts": n,
        "n_params": n_params,
        "dof": dof,
        "H_app": float(H_app),
        "H_app_err": float(se[n]),
        "kappa_Cep": float(kappa_Cep),
        "kappa_Cep_err": float(kappa_se),
        "kappa_Cep_sig": float(kappa_sig),
        "beta_X": float(beta_X),
        "beta_X_err": float(beta_se),
        "beta_X_sig": float(beta_sig),
        "delta_m": float(delta_m),
        "sigma_int_v": float(sigma_int_v),
        "logL": float(logL),
        "AIC": -2 * logL + 2 * n_params,
        "BIC": -2 * logL + n_params * np.log(n),
        "status": "converged" if res.success else "fallback",
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run():
    print_status("Step 41: External Distance Breakers", "SECTION")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_S = load_host_metadata()
    sigma_ref = np.sqrt((30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55))

    df_cep = compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref)
    print_status(f"Cepheid hosts: {len(df_cep)}", "INFO")

    df_ext = load_external_distances()
    print_status(f"External distances: {len(df_ext)} hosts (TRGB)", "INFO")

    # Merge
    merged = []
    for _, cep_row in df_cep.iterrows():
        host = cep_row["host"]
        ext_matches = df_ext[df_ext["host"] == host]
        if len(ext_matches) == 0:
            alt = host.replace("N", "NGC ").replace(" ", "")
            ext_matches = df_ext[df_ext["host"] == alt]
        if len(ext_matches) > 0:
            ext = ext_matches.iloc[0]
            merged.append({
                "host": host,
                "mu_cep": cep_row["mu_cep"],
                "mu_cep_err": cep_row["mu_cep_err"],
                "mu_ext": ext["mu_ext"],
                "mu_ext_err": ext["mu_ext_err"],
                "method": ext["method"],
                "sigma": cep_row["sigma"],
                "z_hd": cep_row["z_hd"],
                "is_anchor": cep_row["is_anchor"],
            })
    df_merged = pd.DataFrame(merged)
    print_status(f"Merged Cepheid+TRGB overlap: {len(df_merged)} hosts", "INFO")

    if len(df_merged) == 0:
        print_status("No overlap between Cepheid and external distances. Step 41 cannot run.", "ERROR")
        return {}

    # Primary sample (non-anchor, z >= 0.0035)
    PRIMARY_Z_CUT = 0.0035
    df_primary = df_cep[
        (~df_cep["is_anchor"]) & df_cep["z_hd"].notna() & (df_cep["z_hd"] >= PRIMARY_Z_CUT)
    ].copy()
    df_merged_primary = df_merged[
        (~df_merged["is_anchor"]) & df_merged["z_hd"].notna() & (df_merged["z_hd"] >= PRIMARY_Z_CUT)
    ].copy()
    print_status(f"Primary Cepheid sample: {len(df_primary)} hosts", "INFO")
    print_status(f"Primary merged sample: {len(df_merged_primary)} hosts", "INFO")

    # ========================================================================
    # Part 1: Differential κ test
    # ========================================================================
    print_status("Part 1: Differential κ test (μ_Cep - μ_ext vs X)", "SECTION")
    diff_result = fit_differential_kappa(df_merged_primary, sigma_ref, host_S)
    print_status(
        f"  N={diff_result['N_hosts']}, κ_Cep = {diff_result['kappa_Cep']:+.3e} "
        f"({diff_result['kappa_Cep_sig']:.1f}σ), "
        f"Δμ_offset = {diff_result['a_offset']:.3f} +/- {diff_result['a_offset_err']:.3f} mag, "
        f"χ²/dof = {diff_result['chi2']:.1f}/{diff_result['dof']}",
        "INFO",
    )

    # ========================================================================
    # Part 2: Velocity β with external-informed μ_true
    # ========================================================================
    print_status("Part 2: Velocity β with external-informed μ_true", "SECTION")
    sigma_v_values = [150, 250, 500]
    velocity_results = []
    for sigma_v in sigma_v_values:
        # Test with κ=0, κ=canonical, and κ=diff_result
        for kappa_label, kappa_val in [("κ=0", 0.0), ("κ=canonical", KAPPA_CANONICAL), ("κ=fit", diff_result["kappa_Cep"])]:
            res = fit_velocity_beta(df_primary, df_merged_primary, kappa_val, sigma_ref, host_S, sigma_v)
            res["sigma_v"] = sigma_v
            res["kappa_label"] = kappa_label
            velocity_results.append(res)
            print_status(
                f"  σ_v={sigma_v}, {kappa_label}: H_app={res['H_app']:.2f}, "
                f"β_X={res['beta_X']:+.3e} ({res['beta_X_sig']:.1f}σ)",
                "INFO",
            )

    # ========================================================================
    # Part 3: Joint model for merged hosts
    # ========================================================================
    print_status("Part 3: Joint model (latent μ_true)", "SECTION")
    joint_results = []
    for sigma_v in sigma_v_values:
        for model_type in ["E0", "Eβ", "Eκ", "Eκβ", "Eκpriorβ"]:
            if model_type == "Eκpriorβ":
                # Weak Gaussian prior on κ around canonical
                # Implemented as penalty in objective
                def obj_prior(p):
                    base = _neg_logL_joint(p, df_merged_primary["mu_cep"].values,
                                           df_merged_primary["mu_cep_err"].values,
                                           df_merged_primary["mu_ext"].values,
                                           df_merged_primary["mu_ext_err"].values,
                                           C_KM_S * df_merged_primary["z_hd"].values,
                                           np.array([build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
                                                     for s, h in zip(df_merged_primary["sigma"].values,
                                                                     df_merged_primary["host"].values)]),
                                           sigma_v, len(df_merged_primary))
                    kappa_param = p[len(df_merged_primary) + 1]
                    kappa_Cep = kappa_param * KAPPA_SCALE
                    penalty = 0.5 * ((kappa_Cep - KAPPA_CANONICAL) / KAPPA_PRIOR_SIGMA)**2
                    return base + penalty

                x0 = list(df_merged_primary["mu_cep"].values)
                x0.extend([70.0, KAPPA_CANONICAL / KAPPA_SCALE, 2.3, 3.0, 0.0])
                bounds = []
                for _ in range(len(df_merged_primary)):
                    bounds.append((20.0, 40.0))
                bounds.extend([
                    (30.0, 90.0), (-100.0, 100.0), (-100.0, 100.0), (0.01, 50.0), (-5.0, 5.0),
                ])
                res = optimize.minimize(obj_prior, x0, method="L-BFGS-B", bounds=bounds)
                for H_init in [65.0, 70.0, 75.0, 80.0]:
                    for k_init in [0.0, 5.0, 10.0, -5.0]:
                        for b_init in [0.0, 1.0, 2.0, 2.3, 3.0]:
                            x0_try = list(df_merged_primary["mu_cep"].values)
                            x0_try.extend([H_init, k_init, b_init, 3.0, 0.0])
                            res_try = optimize.minimize(obj_prior, x0_try, method="L-BFGS-B", bounds=bounds)
                            if res_try.fun < res.fun:
                                res = res_try

                n = len(df_merged_primary)
                idx = n
                H_app = res.x[idx]; idx += 1
                kappa_param = res.x[idx]; idx += 1
                beta_param = res.x[idx]; idx += 1
                sigma_int_v = res.x[idx]; idx += 1
                delta_m = res.x[idx]; idx += 1
                kappa_Cep = kappa_param * KAPPA_SCALE
                beta_X = beta_param * GAMMA_SCALE

                try:
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        hess = optimize.approx_fprime(
                            res.x, lambda x: optimize.approx_fprime(x, obj_prior, 1e-5), 1e-5,
                        )
                        cov = np.linalg.pinv(hess, rcond=1e-12)
                        se = np.sqrt(np.maximum(np.diag(cov), 0))
                except Exception:
                    se = np.full(len(res.x), np.nan)

                kappa_se = se[n + 1] * KAPPA_SCALE
                beta_se = se[n + 2] * GAMMA_SCALE
                kappa_sig = abs(kappa_Cep) / kappa_se if kappa_se > 0 else np.nan
                beta_sig = abs(beta_X) / beta_se if beta_se > 0 else np.nan
                logL = -res.fun + 0.5 * ((kappa_Cep - KAPPA_CANONICAL) / KAPPA_PRIOR_SIGMA)**2
                n_params = len(res.x)

                result = {
                    "model": model_type,
                    "sigma_v": sigma_v,
                    "N_hosts": n,
                    "n_params": n_params,
                    "H_app": float(H_app),
                    "kappa_Cep": float(kappa_Cep),
                    "kappa_Cep_err": float(kappa_se),
                    "kappa_Cep_sig": float(kappa_sig),
                    "beta_X": float(beta_X),
                    "beta_X_err": float(beta_se),
                    "beta_X_sig": float(beta_sig),
                    "delta_m": float(delta_m),
                    "sigma_int_v": float(sigma_int_v),
                    "logL": float(logL),
                    "AIC": -2 * logL + 2 * n_params,
                    "BIC": -2 * logL + n_params * np.log(n),
                    "status": "converged" if res.success else "fallback",
                }
            else:
                result = fit_joint_model(df_merged_primary, sigma_ref, host_S, sigma_v, model_type)
                result["sigma_v"] = sigma_v

            joint_results.append(result)
            print_status(
                f"  {model_type} (σ_v={sigma_v}): "
                f"κ={result['kappa_Cep']:+.3e} ({result.get('kappa_Cep_sig', np.nan):.1f}σ), "
                f"β={result['beta_X']:+.3e} ({result.get('beta_X_sig', np.nan):.1f}σ), "
                f"AIC={result['AIC']:.1f}",
                "INFO",
            )

    # ========================================================================
    # Model comparison
    # ========================================================================
    print_status("Joint model comparison (σ_v=250)", "SECTION")
    print(f"{'Model':>8s} {'N_param':>7s} {'κ_Cep':>14s} {'β_X':>14s} {'AIC':>10s} {'BIC':>10s}")
    print("-" * 80)
    for r in joint_results:
        if r["sigma_v"] != 250:
            continue
        ksig = r.get("kappa_Cep_sig", np.nan)
        bsig = r.get("beta_X_sig", np.nan)
        print(f"{r['model']:>8s} {r['n_params']:>7d} {r['kappa_Cep']:>+14.3e} {r['beta_X']:>+14.3e} {r['AIC']:>10.1f} {r['BIC']:>10.1f}")

    # ========================================================================
    # Save
    # ========================================================================
    out = {
        "differential_kappa": diff_result,
        "velocity_beta": velocity_results,
        "joint_model": joint_results,
        "N_merged_hosts": len(df_merged_primary),
        "N_primary_hosts": len(df_primary),
    }

    out_json = OUT_DIR / "step_41_external_distance_breakers.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    out_csv = OUT_DIR / "step_41_velocity_beta.csv"
    pd.DataFrame(velocity_results).to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    joint_csv = OUT_DIR / "step_41_joint_model.csv"
    pd.DataFrame(joint_results).to_csv(joint_csv, index=False)
    print_status(f"Saved CSV to {joint_csv}", "SUCCESS")

    return out


if __name__ == "__main__":
    run()
