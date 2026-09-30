#!/usr/bin/env python3
"""
step_44_joint_distance_redshift_likelihood.py

Joint likelihood for the TEP-H0 environmental response.

Combines independent data blocks:
  - Redshift-distance block (velocity-space likelihood)
  - External-distance block (TRGB-Cepheid differential moduli)
  - Anchor block (LMC and NGC 4258 independent geometric distances)

Fits four nested models:
  Null       : H_app only (kappa=0, beta=0)
  Cepheid    : H_app + kappa (beta=0)
  Velocity   : H_app + beta  (kappa=0)
  Mixed      : H_app + kappa + beta
  Cepheid_SN : H_app + kappa_Cep + kappa_SN (beta=0), channel-split

The shared kappa parameter is constrained by all three data blocks.

Channel identification: in the redshift block the per-host-varying
environmental bias on the inferred distance d_i is the SN-magnitude
channel, kappa_SN * X_i. A Cepheid-channel bias acts on the calibrator
hosts and propagates into Hubble-flow distances only through the
calibrator-mean M_B, i.e. as a common mode absorbed by H_app. The TRGB
differential and anchor blocks, by contrast, measure mu_Cep against
environment-insensitive references and therefore identify kappa_Cep
per host. The shared-kappa models conflate the two carriers; the
Cepheid_SN model separates them on the same data without redshift
priors (Step-35 model C in a no-expansion likelihood).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

BASE_DIR = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(BASE_DIR))
from scripts.utils.tep_correction import build_host_screening_map
DATA_DIR = BASE_DIR / "data"
SH0ES_DIR = DATA_DIR / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
TRGB_PATH = BASE_DIR / "results" / "outputs" / "step_15_trgb_hosts_data.csv"
ANCHOR_PATH = DATA_DIR / "raw" / "external" / "anchor_galaxy_data.csv"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10) / 5.0

BETA_SCALE = 1e7
KAPPA_SCALE = 1e5


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
    screening_map = build_host_screening_map(df, BASE_DIR)
    host_sigma, host_z, host_S = {}, {}, {}
    host_z_cmb = {}
    for _, row in df.iterrows():
        name = row["normalized_name"]
        sigma = row["sigma_inferred"]
        z_hd = row["z_hd"]
        z_cmb = row.get("z_cmb", np.nan)
        S = screening_map.get(name, row.get("shear_suppression", 1.0))
        if pd.isna(S):
            S = 1.0
        host_sigma[name] = sigma
        host_S[name] = float(S)
        if pd.notna(z_hd) and z_hd > 0:
            host_z[name] = z_hd
        if pd.notna(z_cmb) and z_cmb > 0:
            host_z_cmb[name] = z_cmb

        compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
        if compact.startswith(("N", "U")):
            parts = compact[1:]
            if parts.isdigit():
                for alias in [compact[0] + parts.zfill(4),
                              compact[0] + parts.lstrip("0"),
                              "NGC" + parts,
                              "UGC" + parts]:
                    host_sigma[alias] = sigma
                    host_S[alias] = float(S)
                    if alias not in host_z and pd.notna(z_hd) and z_hd > 0:
                        host_z[alias] = z_hd
                    if alias not in host_z_cmb and pd.notna(z_cmb) and z_cmb > 0:
                        host_z_cmb[alias] = z_cmb

    # Explicit SH0ES aliases for non-standard host names (matched to Step 42)
    explicit = {"M1337": "Mrk1337", "N105A": "N105", "N976A": "N976"}
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_sigma and sh0es_name not in host_sigma:
            host_sigma[sh0es_name] = host_sigma[csv_name]
            host_S[sh0es_name] = host_S[csv_name]
            if csv_name in host_z:
                host_z[sh0es_name] = host_z[csv_name]
            if csv_name in host_z_cmb:
                host_z_cmb[sh0es_name] = host_z_cmb[csv_name]

    return host_sigma, host_z, host_z_cmb, host_S


def build_host_x(sigma, sigma_ref, S=1.0):
    if sigma is None or sigma <= 0 or sigma_ref <= 0:
        return 0.0
    return (S * sigma ** 2 - sigma_ref ** 2) / (C_KM_S ** 2)


def center_scale(v):
    return v - np.mean(v)


def compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref, host_z_cmb=None):
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
    hosts, mus, mu_errs, sigmas, zs, zs_cmb, is_anchors = [], [], [], [], [], [], []
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
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                A_w_pinv = np.linalg.pinv(A_w, rcond=1e-12)
            mu_err = float(np.linalg.norm(A_w_pinv[mu_idx, :]))
        except Exception:
            mu_err = 0.05
        hosts.append(host_name)
        mus.append(mu_fit)
        mu_errs.append(mu_err)
        sigmas.append(host_sigma[host_name])
        zs.append(host_z.get(host_name, np.nan))
        zs_cmb.append(host_z_cmb.get(host_name, np.nan) if host_z_cmb else np.nan)
        is_anchors.append(host_name in anchor_hosts)

    return pd.DataFrame({
        "host": hosts, "mu": mus, "mu_err": mu_errs,
        "sigma": sigmas, "z_hd": zs, "z_cmb": zs_cmb, "is_anchor": is_anchors,
    })


def match_trgb_hosts(df_hosts):
    if not TRGB_PATH.exists():
        return pd.DataFrame()
    df_trgb = pd.read_csv(TRGB_PATH)
    df_trgb["match_name"] = df_trgb["match_name"].fillna("").astype(str)
    df_trgb["galaxy"] = df_trgb["galaxy"].fillna("").astype(str)

    matched = []
    for _, row in df_hosts.iterrows():
        host = row["host"]
        aliases = {host, host.replace("N", "NGC").replace("U", "UGC")}
        if host.startswith("N") and host[1:].isdigit():
            aliases.add("NGC " + host[1:].lstrip("0"))
        if host.startswith("NGC"):
            aliases.add("N" + host[3:].lstrip("0"))
            aliases.add("NGC " + host[3:].lstrip("0"))

        sel = df_trgb[
            df_trgb["match_name"].isin(aliases) |
            df_trgb["galaxy"].isin(aliases) |
            df_trgb["galaxy"].str.replace(" ", "").str.replace("NGC", "N").isin(aliases)
        ]
        if len(sel) == 0:
            continue
        trgb = sel.iloc[0]
        matched.append({
            "host": host,
            "mu_cep": row["mu"],
            "mu_cep_err": row["mu_err"],
            "sigma": row["sigma"],
            "mu_trgb": float(trgb["mu_trgb"]),
            "mu_trgb_err": float(trgb["mu_trgb_err"]),
        })
    return pd.DataFrame(matched)


def match_anchor_hosts(df_hosts):
    """Match SH0ES anchors to independent geometric distance measurements."""
    if not ANCHOR_PATH.exists():
        return pd.DataFrame()

    df_anch = pd.read_csv(ANCHOR_PATH)
    name_map = {
        "LMC": "LMC",
        "NGC 4258": "N4258",
        "M 31": "M31",
    }

    matched = []
    for _, row in df_anch.iterrows():
        gal = row.get("galaxy", "").strip()
        if gal not in name_map:
            continue
        host = name_map[gal]
        sel = df_hosts[df_hosts["host"] == host]
        if len(sel) == 0:
            continue
        host_row = sel.iloc[0]
        method = str(row.get("method", "")).lower()
        # M 31 uses a Cepheid/R22 composite and is not an independent geometric distance.
        independent = ("eclipsing" in method or "maser" in method)
        matched.append({
            "host": host,
            "galaxy": gal,
            "mu_cep": float(host_row["mu"]),
            "mu_cep_err": float(host_row["mu_err"]),
            "sigma": float(host_row["sigma"]),
            "mu_geo": float(row["mu_anchor"]),
            "mu_geo_err": float(row["error_mu"]),
            "method": str(row.get("method", "")),
            "independent": bool(independent),
        })
    return pd.DataFrame(matched)


def _velocity_model(d_obs, X, H_app, kappa, beta):
    d_true = d_obs * np.power(10.0, kappa * X / 5.0)
    return d_true * (H_app + beta * X)


def _neg_logL(params, model_type, cz_obs, d_obs, X, sigma_mu, sigma_v,
              dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc,
              sigma_int_guess=5.0, delta_m_prior_width=1.0,
              delta_a_prior_width=0.5):
    sigma_int_v = max(params[-1], 0.01)
    delta_a = params[-2]
    delta_m = params[-3]

    if model_type == "Null":
        H_app = params[0]
        kappa = 0.0
        beta = 0.0
    elif model_type == "Cepheid":
        H_app = params[0]
        kappa = params[1] * KAPPA_SCALE
        beta = 0.0
    elif model_type == "Coupled":
        H_app = params[0]
        kappa = params[1] * KAPPA_SCALE
        # Physical gravitational redshift of the spectroscopic tracer:
        # beta_grav = c / <d> ~ 1.14e4 km/s/Mpc
        beta = C_KM_S / np.mean(d_obs)
    elif model_type == "Velocity":
        H_app = params[0]
        kappa = 0.0
        beta = params[1] * BETA_SCALE
    elif model_type == "Mixed":
        H_app = params[0]
        kappa = params[1] * KAPPA_SCALE
        beta = params[2] * BETA_SCALE
    elif model_type == "Cepheid_SN":
        H_app = params[0]
        kappa_cep = params[1] * KAPPA_SCALE
        kappa_sn = params[2] * KAPPA_SCALE
        kappa = 0.0
        beta = 0.0
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if model_type != "Cepheid_SN":
        kappa_cep = kappa
        kappa_sn = kappa

    # Redshift block (H_0 space — linearised expansion-rate regression)
    # H_0,i = cz_i / d_i = H_app + Gamma_X * X_i + epsilon_i
    # where Gamma_X = beta + (ln10/5) * H_app * kappa is the identifiable
    # endpoint response.  This linear WLS formulation matches Step 42 and
    # is substantially more efficient than the nonlinear cz-space model
    # because the parameter space is linear in (H_app, Gamma_X).
    H0_obs = cz_obs / d_obs
    Gamma_X = beta + LN10_OVER_5 * H_app * kappa_sn
    H0_model = H_app + Gamma_X * X
    resid_v = H0_obs - H0_model
    var_v = sigma_v ** 2 / d_obs ** 2 + (LN10_OVER_5 * H0_obs * sigma_mu) ** 2 + sigma_int_v ** 2 / d_obs ** 2
    var_v = np.maximum(var_v, 1e-10)
    ll_v = -0.5 * np.sum(resid_v ** 2 / var_v + np.log(var_v))

    # TRGB block: Δμ = δm - κ_Cep X  (κ>0 → μ_cep underestimated in high-σ)
    if len(dmu_obs) > 0:
        dmu_model = delta_m - kappa_cep * X_t
        resid_t = dmu_obs - dmu_model
        var_t = dmu_err ** 2
        ll_t = -0.5 * np.sum(resid_t ** 2 / var_t + np.log(var_t))
    else:
        ll_t = 0.0

    # Anchor block: Δμ_anc = μ_cep - μ_geo = δa - κ_Cep X_anc + ε
    if len(dmu_anc) > 0:
        dmu_anc_model = delta_a - kappa_cep * X_anc
        resid_a = dmu_anc - dmu_anc_model
        var_a = dmu_anc_err ** 2
        ll_a = -0.5 * np.sum(resid_a ** 2 / var_a + np.log(var_a))
    else:
        ll_a = 0.0

    # Weak priors on zero-points
    prior_dm = -0.5 * (delta_m / delta_m_prior_width) ** 2
    prior_da = -0.5 * (delta_a / delta_a_prior_width) ** 2

    return -(ll_v + ll_t + ll_a + prior_dm + prior_da)


def fit_model(cz_obs, d_obs, X, sigma_mu, sigma_v,
              dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc,
              model_type, sigma_int_guess=5.0):
    n_v = len(cz_obs)
    n_t = len(dmu_obs)
    n_a = len(dmu_anc)

    H_app_init = np.median(cz_obs / d_obs)

    if model_type == "Null":
        x0 = np.array([H_app_init, 0.0, 0.0, sigma_int_guess])
        bounds = [(30.0, 90.0), (-5.0, 5.0), (-5.0, 5.0), (0.01, 250.0)]
        n_params = 4
    elif model_type in ("Cepheid", "Coupled"):
        x0 = np.array([H_app_init, 0.0, 0.0, 0.0, sigma_int_guess])
        bounds = [(30.0, 90.0), (-50.0, 50.0), (-5.0, 5.0), (-5.0, 5.0), (0.01, 250.0)]
        n_params = 5
    elif model_type == "Velocity":
        x0 = np.array([H_app_init, 2.35, 0.0, 0.0, sigma_int_guess])
        bounds = [(30.0, 90.0), (-100.0, 100.0), (-5.0, 5.0), (-5.0, 5.0), (0.01, 250.0)]
        n_params = 5
    elif model_type == "Mixed":
        x0 = np.array([H_app_init, 0.0, 2.35, 0.0, 0.0, sigma_int_guess])
        bounds = [(30.0, 90.0), (-50.0, 50.0), (-100.0, 100.0), (-5.0, 5.0), (-5.0, 5.0), (0.01, 250.0)]
        n_params = 6
    elif model_type == "Cepheid_SN":
        x0 = np.array([H_app_init, 0.0, 0.0, 0.0, 0.0, sigma_int_guess])
        bounds = [(30.0, 90.0), (-50.0, 50.0), (-50.0, 50.0), (-5.0, 5.0), (-5.0, 5.0), (0.01, 250.0)]
        n_params = 6
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    res = optimize.minimize(
        _neg_logL,
        x0,
        args=(model_type, cz_obs, d_obs, X, sigma_mu, sigma_v,
              dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc, sigma_int_guess),
        method="L-BFGS-B",
        bounds=bounds,
    )

    # Multi-start over the intrinsic-scatter coordinate: the likelihood is
    # nearly flat in sigma_int_v near the start point, and single-start
    # L-BFGS-B can stall at the initial guess (observed: sigma_int_v ~ 5.0
    # returned while the true optimum sits near ~120-170 km/s), corrupting
    # logL, BIC and Hessian errors.  Always re-minimize from a small ladder
    # of sigma_int starts and keep the best objective.
    for si_guess in [50.0, 120.0, 200.0]:
        x0_try = x0.copy()
        x0_try[-1] = si_guess
        res_try = optimize.minimize(
            _neg_logL, x0_try,
            args=(model_type, cz_obs, d_obs, X, sigma_mu, sigma_v,
                  dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc, sigma_int_guess),
            method="L-BFGS-B", bounds=bounds,
        )
        if res_try.fun < res.fun - 1e-9:
            res = res_try

    if not res.success:
        for H_init in [65.0, 70.0, 75.0]:
            x0_try = x0.copy()
            x0_try[0] = H_init
            res_try = optimize.minimize(
                _neg_logL, x0_try,
                args=(model_type, cz_obs, d_obs, X, sigma_mu, sigma_v,
                      dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc, sigma_int_guess),
                method="L-BFGS-B", bounds=bounds,
            )
            if res_try.fun < res.fun:
                res = res_try

    # Extract physical parameters
    H_app = res.x[0]
    if model_type == "Null":
        kappa, beta = 0.0, 0.0
    elif model_type == "Cepheid":
        kappa = res.x[1] * KAPPA_SCALE
        beta = 0.0
    elif model_type == "Coupled":
        kappa = res.x[1] * KAPPA_SCALE
        beta = C_KM_S / np.mean(d_obs)
    elif model_type == "Velocity":
        kappa = 0.0
        beta = res.x[1] * BETA_SCALE
    elif model_type == "Cepheid_SN":
        kappa = 0.0
        kappa_sn = res.x[2] * KAPPA_SCALE
        beta = 0.0
    else:
        kappa = res.x[1] * KAPPA_SCALE
        beta = res.x[2] * BETA_SCALE
    if model_type != "Cepheid_SN":
        kappa_sn = np.nan
    kappa_cep = res.x[1] * KAPPA_SCALE if model_type == "Cepheid_SN" else kappa

    delta_m = res.x[-3]
    delta_a = res.x[-2]
    sigma_int_v = res.x[-1]

    # Approximate Hessian for uncertainties
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            hess = optimize.approx_fprime(
                res.x,
                lambda x: optimize.approx_fprime(
                    x,
                    lambda p: _neg_logL(p, model_type, cz_obs, d_obs, X, sigma_mu, sigma_v,
                                       dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc, sigma_int_guess),
                    1e-5,
                ),
                1e-5,
            )
            cov = np.linalg.pinv(hess, rcond=1e-12)
            se = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se = np.full(len(res.x), np.nan)

    # Convert scaled uncertainties to physical units
    if model_type in ("Cepheid", "Coupled"):
        se[1] *= KAPPA_SCALE
    elif model_type == "Velocity":
        se[1] *= BETA_SCALE
    elif model_type == "Mixed":
        se[1] *= KAPPA_SCALE
        se[2] *= BETA_SCALE
    elif model_type == "Cepheid_SN":
        se[1] *= KAPPA_SCALE
        se[2] *= KAPPA_SCALE

    # Likelihood and chi2 at MLE
    final_nll = _neg_logL(res.x, model_type, cz_obs, d_obs, X, sigma_mu, sigma_v,
                          dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc, sigma_int_guess)
    logL = -final_nll
    n_total = n_v + n_t + n_a
    dof = n_total - n_params

    # Chi2 (reduced) for reporting — H_0 space consistent with likelihood
    H0_obs = cz_obs / d_obs
    kappa_red = kappa_sn if model_type == "Cepheid_SN" else kappa
    kappa_ref = kappa_cep if model_type == "Cepheid_SN" else kappa
    Gamma_X_chi2 = beta + LN10_OVER_5 * H_app * kappa_red
    H0_model = H_app + Gamma_X_chi2 * X
    resid_v = H0_obs - H0_model
    var_v = sigma_v ** 2 / d_obs ** 2 + (LN10_OVER_5 * H0_obs * sigma_mu) ** 2 + sigma_int_v ** 2 / d_obs ** 2
    chi2_v = np.sum(resid_v ** 2 / np.maximum(var_v, 1e-10))
    if n_t > 0:
        dmu_model_t = delta_m - kappa_ref * X_t
        chi2_t = np.sum(((dmu_obs - dmu_model_t) / dmu_err) ** 2)
    else:
        chi2_t = 0.0
    if n_a > 0:
        dmu_model_a = delta_a - kappa_ref * X_anc
        chi2_a = np.sum(((dmu_anc - dmu_model_a) / dmu_anc_err) ** 2)
    else:
        chi2_a = 0.0
    chi2 = float(chi2_v + chi2_t + chi2_a)

    return {
        "model": model_type,
        "n_hosts_redshift": n_v,
        "n_hosts_trgb": n_t,
        "n_hosts_anchor": n_a,
        "n_params": n_params,
        "dof": int(dof),
        "H_app": float(H_app),
        "H_app_err": float(se[0]),
        "kappa_Cep": float(kappa_cep),
        "kappa_Cep_err": float(se[1]) if model_type in ("Cepheid", "Mixed", "Coupled", "Cepheid_SN") else np.nan,
        "kappa_SN": float(kappa_sn),
        "kappa_SN_err": float(se[2]) if model_type == "Cepheid_SN" else np.nan,
        "beta_X": float(beta),
        "beta_X_err": float(se[1]) if model_type == "Velocity" else (float(se[2]) if model_type == "Mixed" else (0.0 if model_type == "Coupled" else np.nan)),
        "delta_m": float(delta_m),
        "delta_m_err": float(se[-3]),
        "delta_a": float(delta_a),
        "delta_a_err": float(se[-2]),
        "sigma_int_v": float(sigma_int_v),
        "sigma_int_v_err": float(se[-1]),
        "chi2": float(chi2),
        "chi2_reduced": float(chi2 / dof) if dof > 0 else np.inf,
        "logL": float(logL),
        "AIC": float(-2 * logL + 2 * n_params),
        "BIC": float(-2 * logL + n_params * np.log(n_total)),
        "status": "converged" if res.success else "fallback",
        "n_iter": int(res.nit) if hasattr(res, "nit") else -1,
    }


def loocv(cz_obs, d_obs, X, sigma_mu, sigma_v,
          dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc,
          model_type, sigma_int_guess=5.0):
    """Leave-one-out cross-validation, leaving out one redshift host at a time."""
    n = len(cz_obs)
    sq_errors = []
    for i in range(n):
        res = fit_model(
            np.delete(cz_obs, i), np.delete(d_obs, i),
            np.delete(X, i), np.delete(sigma_mu, i), sigma_v,
            dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc,
            model_type, sigma_int_guess
        )
        H_app = res["H_app"]
        kappa = res["kappa_Cep"]
        beta = res["beta_X"]
        pred = _velocity_model(d_obs[i], X[i], H_app, kappa, beta)
        var_i = sigma_v ** 2 + (LN10_OVER_5 * np.abs(pred) * sigma_mu[i]) ** 2 + res["sigma_int_v"] ** 2
        sq_errors.append((cz_obs[i] - pred) ** 2 / max(var_i, 0.01))
    return float(np.mean(sq_errors))


def run():
    print_status("Step 44: Joint distance + redshift likelihood", "SECTION")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_z_cmb, host_S = load_host_metadata()

    # Reference dispersion weighted by anchor prior usage
    sigma_ref = np.sqrt(
        (30.0 ** 2 * 0.20 + 24.0 ** 2 * 0.25 + 115.0 ** 2 * 0.55) /
        (0.20 + 0.25 + 0.55)
    )

    df_hosts = compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref, host_z_cmb=host_z_cmb)
    from scripts.utils.sample_selection import hubble_flow_mask
    df_primary = df_hosts[
        (~df_hosts["is_anchor"])
        & hubble_flow_mask(df_hosts)
    ].copy()
    print_status(f"Primary redshift sample: {len(df_primary)} hosts", "INFO")

    # Redshift covariates
    mu = df_primary["mu"].values
    mu_err = df_primary["mu_err"].fillna(0.05).values
    z = df_primary["z_hd"].fillna(df_primary["z_cmb"]).values
    d_obs = 10 ** ((mu - 25.0) / 5.0)
    cz_obs = C_KM_S * z
    X_tep = np.array([
        build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
        for s, h in zip(df_primary["sigma"].values, df_primary["host"].values)
    ])
    X_tep_c = center_scale(X_tep)

    # Match TRGB hosts — use ALL non-anchor hosts (not just Hubble-flow)
    # to maximise the TRGB overlap sample.  Two additional calibrators
    # (N4424, N4536) fall below the Hubble-flow redshift cut but still
    # provide valuable differential-modulus constraints on kappa_Cep.
    df_trgb = match_trgb_hosts(df_hosts[~df_hosts["is_anchor"]])
    if len(df_trgb) == 0:
        print_status("No TRGB hosts matched; using redshift and anchor blocks only.", "WARNING")
        dmu_obs = np.array([])
        dmu_err = np.array([])
        X_t = np.array([])
    else:
        print_status(f"TRGB overlap sample: {len(df_trgb)} hosts", "INFO")
        X_t = np.array([
            build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
            for s, h in zip(df_trgb["sigma"].values, df_trgb["host"].values)
        ])
        X_t = center_scale(X_t)
        dmu_obs = df_trgb["mu_cep"].values - df_trgb["mu_trgb"].values
        dmu_err = np.sqrt(df_trgb["mu_cep_err"].values ** 2 + df_trgb["mu_trgb_err"].values ** 2)

    # Match anchors
    df_anchors = match_anchor_hosts(df_hosts)
    independent = df_anchors[df_anchors["independent"]] if len(df_anchors) else df_anchors
    if len(independent) == 0:
        print_status("No independent geometric anchors matched.", "WARNING")
        dmu_anc = np.array([])
        dmu_anc_err = np.array([])
        X_anc = np.array([])
    else:
        print_status(f"Independent anchor sample: {len(independent)} anchors", "INFO")
        print(independent[["host", "mu_cep", "mu_geo", "method"]])
        X_anc = np.array([
            build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
            for s, h in zip(independent["sigma"].values, independent["host"].values)
        ])
        dmu_anc = independent["mu_cep"].values - independent["mu_geo"].values
        dmu_anc_err = np.sqrt(independent["mu_cep_err"].values ** 2 + independent["mu_geo_err"].values ** 2)

    models = ["Null", "Cepheid", "Coupled", "Velocity", "Mixed", "Cepheid_SN"]
    sigma_v_values = [150, 182.1, 250, 500]
    results = []

    for sigma_v in sigma_v_values:
        print_status(f"sigma_v = {sigma_v} km/s", "SECTION")
        for model_type in models:
            print_status(f"  Fitting {model_type}...", "INFO")
            res = fit_model(cz_obs, d_obs, X_tep_c, mu_err, sigma_v,
                            dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc,
                            model_type)
            res["sigma_v"] = sigma_v
            res["sample"] = "primary"
            res["z_cut"] = 0.0
            results.append(res)

            kappa = res.get("kappa_Cep", np.nan)
            kappa_err = res.get("kappa_Cep_err", np.nan)
            kappa_sn = res.get("kappa_SN", np.nan)
            kappa_sn_err = res.get("kappa_SN_err", np.nan)
            beta = res.get("beta_X", np.nan)
            beta_err = res.get("beta_X_err", np.nan)
            k_sig = abs(kappa) / kappa_err if kappa_err and kappa_err > 0 else np.nan
            ksn_sig = abs(kappa_sn) / kappa_sn_err if kappa_sn_err and kappa_sn_err > 0 else np.nan
            b_sig = abs(beta) / beta_err if beta_err and beta_err > 0 else np.nan
            print_status(
                f"    {model_type}: H_app={res['H_app']:.2f}, "
                f"kappa={kappa:.3e} ({k_sig:.1f}σ), "
                f"kappa_SN={kappa_sn:.3e} ({ksn_sig:.1f}σ), "
                f"beta={beta:.3e} ({b_sig:.1f}σ), "
                f"BIC={res['BIC']:.1f}, logL={res['logL']:.1f}",
                "INFO",
            )

    # LOOCV on primary model at sigma_v=250 for the Mixed model
    print_status("Leave-one-out cross-validation (Mixed, sigma_v=250)", "SECTION")
    loocv_mse = loocv(cz_obs, d_obs, X_tep_c, mu_err, 250,
                      dmu_obs, dmu_err, X_t, dmu_anc, dmu_anc_err, X_anc, "Mixed")
    print_status(f"LOOCV MSE = {loocv_mse:.3f}", "INFO")

    # -----------------------------------------------------------------------
    # Redshift-only WLS in H_0 space (Cepheid closure, beta=0)
    #
    # The joint likelihood above combines three data blocks.  The TRGB
    # differential block has limited X-leverage (all calibrators are
    # nearby, sigma < 160 km/s) and is therefore uninformative about
    # kappa_Cep on its own (0.27 sigma).  Including it in the joint fit
    # dilutes the redshift-block signal because the TRGB best-fit kappa
    # is consistent with zero.  The redshift-only WLS result — which is
    # the identifiable Gamma_X regression of Step 42 — provides the
    # cleanest single-block constraint and is reported here for
    # transparency and for manuscript cross-reference.
    # -----------------------------------------------------------------------
    print_status("Redshift-only WLS (H_0 space, Cepheid closure)", "SECTION")
    H0_obs = cz_obs / d_obs
    n_v = len(cz_obs)
    redshift_only_results = []
    GAMMA_SCALE_WLS = 1e7
    for sigma_v in sigma_v_values:
        w = d_obs ** 2 / (sigma_v ** 2 + (LN10_OVER_5 * cz_obs * mu_err) ** 2)
        w = np.maximum(w, 1e-10)
        Xmat = np.column_stack([np.ones(n_v), X_tep_c * GAMMA_SCALE_WLS])
        W = np.diag(w)
        XtWX = Xmat.T @ W @ Xmat
        beta = np.linalg.lstsq(XtWX, Xmat.T @ W @ H0_obs, rcond=None)[0]
        cov = np.linalg.pinv(XtWX, rcond=1e-12)
        H_app = float(beta[0])
        H_app_err = float(np.sqrt(cov[0, 0]))
        Gamma_X = float(beta[1] * GAMMA_SCALE_WLS)
        Gamma_X_err = float(np.sqrt(cov[1, 1]) * GAMMA_SCALE_WLS)
        cov_GH = float(cov[0, 1] * GAMMA_SCALE_WLS)
        # Convert to kappa_Cep with full error propagation
        kappa = Gamma_X / (LN10_OVER_5 * H_app)
        dK_dG = 1.0 / (LN10_OVER_5 * H_app)
        dK_dH = -Gamma_X / (LN10_OVER_5 * H_app ** 2)
        kappa_err = float(np.sqrt(
            dK_dG ** 2 * Gamma_X_err ** 2
            + dK_dH ** 2 * H_app_err ** 2
            + 2 * dK_dG * dK_dH * cov_GH
        ))
        kappa_sig = abs(kappa) / kappa_err if kappa_err > 0 else np.nan
        Gamma_sig = abs(Gamma_X) / Gamma_X_err if Gamma_X_err > 0 else np.nan
        redshift_only_results.append({
            "sigma_v": sigma_v,
            "n_hosts": n_v,
            "H_app": H_app,
            "H_app_err": H_app_err,
            "Gamma_X": Gamma_X,
            "Gamma_X_err": Gamma_X_err,
            "Gamma_X_sig": Gamma_sig,
            "kappa_Cep": float(kappa),
            "kappa_Cep_err": kappa_err,
            "kappa_Cep_sig": kappa_sig,
        })
        print_status(
            f"  sv={sigma_v:6.1f}: Gamma_X=({Gamma_X:.3e}+/-{Gamma_X_err:.3e}) "
            f"({Gamma_sig:.2f}s), kappa=({kappa:.3e}+/-{kappa_err:.3e}) "
            f"({kappa_sig:.2f}s), H_app={H_app:.2f}",
            "INFO",
        )

    # -----------------------------------------------------------------------
    # Host-mass marginalization of the redshift-only WLS.
    #
    # The environmental coordinate X correlates with host stellar mass via
    # the mass--potential relation, so the SN Ia host-mass step is the
    # standard confounder a referee will raise.  The decisive test is a
    # simultaneous regression on the identical design and weights:
    #
    #   H_0,i = H_app + Gamma_X * X_i + gamma_M * (logM_i - <logM>) + eps_i
    #
    # reported per sigma_v so the retention of Gamma_X is explicit.
    # -----------------------------------------------------------------------
    print_status("Redshift-only WLS with host-mass marginalization", "SECTION")
    _df_hosts_meta = pd.read_csv(HOSTS_PATH)
    _lm = {}
    for _, _row in _df_hosts_meta.iterrows():
        _nm = _row["normalized_name"]
        _v = _row.get("host_logmass", np.nan)
        if pd.isna(_v):
            continue
        _lm[_nm] = float(_v)
        _c = _nm.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
        if _c.startswith(("N", "U")):
            _p = _c[1:]
            if _p.isdigit():
                for _a in [_c[0] + _p.zfill(4), _c[0] + _p.lstrip("0")]:
                    _lm[_a] = float(_v)
    for _sh, _csv in {"M1337": "Mrk1337", "N105A": "N105", "N976A": "N976"}.items():
        if _csv in _lm:
            _lm[_sh] = _lm[_csv]
    logmass = np.array([_lm.get(h, np.nan) for h in df_primary["host"].values])
    n_mass = int(np.isfinite(logmass).sum())
    print_status(f"Hosts with stellar mass: {n_mass}/{n_v}", "INFO")
    keep_m = np.isfinite(logmass)
    M_c = logmass - np.nanmean(logmass)
    r_xm = float(np.corrcoef(X_tep_c[keep_m], M_c[keep_m])[0, 1])
    print_status(f"r(X, logM) on primary sample = {r_xm:.3f}", "INFO")

    mass_marginalized_results = []
    for sigma_v in sigma_v_values:
        w = d_obs ** 2 / (sigma_v ** 2 + (LN10_OVER_5 * cz_obs * mu_err) ** 2)
        w = np.maximum(w, 1e-10)
        Xmat = np.column_stack(
            [np.ones(int(keep_m.sum())), X_tep_c[keep_m] * GAMMA_SCALE_WLS, M_c[keep_m]]
        )
        W = np.diag(w[keep_m])
        XtWX = Xmat.T @ W @ Xmat
        beta = np.linalg.lstsq(XtWX, Xmat.T @ W @ H0_obs[keep_m], rcond=None)[0]
        cov = np.linalg.pinv(XtWX, rcond=1e-12)
        H_app_m = float(beta[0])
        Gamma_m = float(beta[1] * GAMMA_SCALE_WLS)
        Gamma_m_err = float(np.sqrt(cov[1, 1]) * GAMMA_SCALE_WLS)
        gM = float(beta[2])
        gM_err = float(np.sqrt(cov[2, 2]))
        kappa_m = Gamma_m / (LN10_OVER_5 * H_app_m)
        base = next(r for r in redshift_only_results if r["sigma_v"] == sigma_v)
        mass_marginalized_results.append({
            "sigma_v": sigma_v,
            "n_hosts": int(keep_m.sum()),
            "H_app": H_app_m,
            "Gamma_X": Gamma_m,
            "Gamma_X_err": Gamma_m_err,
            "Gamma_X_sig": abs(Gamma_m) / Gamma_m_err if Gamma_m_err > 0 else np.nan,
            "kappa_Cep": float(kappa_m),
            "gamma_M_kms_per_dex": gM,
            "gamma_M_err": gM_err,
            "gamma_M_sig": abs(gM) / gM_err if gM_err > 0 else np.nan,
            "gamma_retention_fraction": Gamma_m / base["Gamma_X"] if base["Gamma_X"] else np.nan,
        })
        print_status(
            f"  sv={sigma_v:6.1f}: Gamma_X {Gamma_m:.3e} "
            f"({abs(Gamma_m)/Gamma_m_err:.2f}s) after mass; "
            f"gamma_M = {gM:.2f} +/- {gM_err:.2f} km/s/dex",
            "INFO",
        )

    out = {
        "results": results,
        "redshift_only_wls": redshift_only_results,
        "mass_marginalization_wls": {
            "description": "Simultaneous WLS of H0 on screened X and host_logmass (Step-44 design)",
            "r_X_logM": r_xm,
            "n_hosts_with_mass": n_mass,
            "results": mass_marginalized_results,
        },
        "loocv_mse_mixed": loocv_mse,
        "n_trgb": int(len(dmu_obs)),
        "n_redshift": int(len(cz_obs)),
        "n_anchors": int(len(dmu_anc)),
        "sigma_ref": float(sigma_ref),
    }
    OUT_PATH = OUT_DIR / "step_44_joint_distance_redshift_likelihood.json"
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print_status(f"Saved: {OUT_PATH}", "SUCCESS")


if __name__ == "__main__":
    run()
