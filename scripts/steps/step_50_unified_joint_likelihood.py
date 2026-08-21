#!/usr/bin/env python3
"""
step_50_unified_joint_likelihood.py

Diagnostic Summary-Block Likelihood for the TEP-H0 Environmental Response

WARNING: this step is retained for provenance and is not used for inference.
It reuses baseline-fitted host moduli and M_B summaries, approximates the
calibrator transfer, and fails reference-gauge invariance.  The raw SH0ES
matrix likelihood is implemented in Steps 34/45; the generative endpoint
likelihood is implemented in Steps 39/42/44.

This step jointly fits:
    (H_0, kappa_Cep, Gamma_redshift-env, sigma_v, sigma_int)

Model per host i (non-anchor, with Cepheid + SN calibrator data):

    mu_i = mu^Cep_obs,i + kappa_Cep * X_i         (TEP-corrected modulus)
    m_{B,i} = M_B + mu_i + epsilon_SN              (calibrator SN)
    cz_i = D(mu_i) * [H_0 + Gamma_env * X_i] + v_i (redshift-distance)

Where TRGB data exist:

    mu^TRGB_obs,i = mu_i - kappa_TRGB * X_i + epsilon_TRGB

Hubble-flow SNe:

    m_{B,j} = M_B + 5*log10(cz_j / H_0) + 25 + epsilon_HF

Key corrections from reviewer feedback (v2):
  1. NO Step 44 Gaussian prior on kappa in the historical "raw_data" mode.
     The redshift-distance block alone constrains kappa when sigma_v is
     fixed from the null model (the sigma_v–kappa degeneracy is broken).
  2. sigma_v and sigma_int are FIXED from the null-model fit.  The null
     model determines the velocity dispersion from the scatter of cz
     around the Hubble flow (independent of kappa, which measures the
     TREND with X, not the scatter).  Fixing sigma_v allows the RD block
     to detect the kappa*X slope.
  3. kappa_TRGB is FIXED to zero.  TRGB is a stellar-evolution clock,
     physically distinct from Cepheid pulsation.  TEP predicts a
     negligible TRGB response, so the TRGB block becomes a direct
     constraint on kappa_Cep:  mu_cep + kappa*X - mu_trgb = delta_m.
  4. A "propagation" cross-check mode retains the Step 44 prior but
     removes the RD block, confirming that the two approaches agree.

Outputs
-------
  results/outputs/step_50_unified_joint_likelihood.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from scripts.utils.tep_correction import (
    ANCHOR_SIGMA,
    ANCHOR_WEIGHTS,
    build_host_screening_map,
    compute_anchor_sigma_ref,
)

DATA_DIR = BASE_DIR / "data"
SH0ES_DIR = DATA_DIR / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
TRGB_PATH = BASE_DIR / "results" / "outputs" / "step_15_trgb_hosts_data.csv"
ANCHOR_PATH = DATA_DIR / "raw" / "external" / "anchor_galaxy_data.csv"
PANTHEON_PATH = DATA_DIR / "raw" / "Pantheon+SH0ES.dat"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
C_SQUARED_KM_S = C_KM_S ** 2
LN10 = np.log(10.0)
LN10_OVER_5 = LN10 / 5.0

H0_PLANCK = 67.4
H0_PLANCK_ERR = 0.5
H0_SH0ES = 73.04
H0_SH0ES_ERR = 1.04

KAPPA_SCALE = 1e5
BETA_SCALE = 1e7


def print_status(msg, level="INFO"):
    prefix = {
        "SECTION": "=" * 70,
        "INFO": "[INFO]",
        "SUCCESS": "[SUCCESS]",
        "WARNING": "[WARNING]",
        "ERROR": "[ERROR]",
    }.get(level, "[INFO]")
    if level == "SECTION":
        print(f"\n{prefix}\n{msg}\n{prefix}")
    else:
        print(f"{prefix} {msg}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sh0es_data():
    L = np.loadtxt(SH0ES_DIR / "L_R22.txt", delimiter="\t")
    names = ("Source", "Data")
    fmt = ("S20", np.float64)
    y_data = np.loadtxt(SH0ES_DIR / "y_R22.txt", unpack=True, skiprows=1,
                        dtype={"names": names, "formats": fmt})
    y = y_data[1]
    C = np.loadtxt(SH0ES_DIR / "C_R22.txt", delimiter="\t")
    q = np.loadtxt(SH0ES_DIR / "q_R22.txt", unpack=True, dtype="str")
    return L, y, C, q


def load_host_metadata():
    df = pd.read_csv(HOSTS_PATH)
    screening_map = build_host_screening_map(df, BASE_DIR)

    host_sigma, host_S, host_z = {}, {}, {}
    for _, row in df.iterrows():
        name = row["normalized_name"]
        sigma = row["sigma_inferred"]
        z_hd = row["z_hd"]
        S = screening_map.get(name, 1.0)
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
                for alias in [compact[0] + parts.zfill(4), compact[0] + parts.lstrip("0")]:
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
    return host_sigma, host_z, host_S


# ---------------------------------------------------------------------------
# Fisher-matrix anchor weights
# ---------------------------------------------------------------------------

def compute_fisher_anchor_weights(L, y, C, q):
    """Compute anchor weights from the SH0ES Fisher matrix using
    leave-one-anchor-out information loss."""
    anchor_mu_params = {"mu_N4258": "NGC 4258", "mu_LMC": "LMC"}

    anchor_rows = {}
    for mu_param in anchor_mu_params:
        idx = np.where(q == mu_param)[0]
        if len(idx) == 0:
            continue
        mu_idx = idx[0]
        rows = []
        for i in range(L.shape[0]):
            nonzero = np.where(np.abs(L[i]) > 0.01)[0]
            if len(nonzero) == 1 and nonzero[0] == mu_idx:
                rows.append(i)
        anchor_rows[mu_param] = rows

    def get_h0_var(L_use, y_use, C_use):
        try:
            Lc = np.linalg.cholesky(C_use)
            A_w = linalg.solve_triangular(Lc, L_use, lower=True, check_finite=False)
            y_w = linalg.solve_triangular(Lc, y_use, lower=True, check_finite=False)
        except:
            A_w = L_use.copy()
            y_w = y_use.copy()
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            F = A_w.T @ A_w
            Cov = np.linalg.pinv(F, rcond=1e-10)
        h0_idx = np.where(q == "5logH0")[0]
        if len(h0_idx) == 0:
            return 1e10
        var = Cov[h0_idx[0], h0_idx[0]]
        return var if np.isfinite(var) and var > 0 and var < 1e6 else 1e10

    var_full = get_h0_var(L, y, C)
    if var_full >= 1e9:
        print_status("Fisher matrix unstable; using standard weights.", "WARNING")
        return {"MW": 0.20, "LMC": 0.25, "NGC 4258": 0.55}

    info = {}
    for mu_param, anchor_name in anchor_mu_params.items():
        rows = anchor_rows.get(mu_param, [])
        if len(rows) == 0:
            info[anchor_name] = 0.0
            continue
        mask = np.ones(L.shape[0], dtype=bool)
        mask[rows] = False
        var_loo = get_h0_var(L[mask], y[mask], C[np.ix_(mask, mask)])
        if var_loo < 1e9 and var_loo > var_full:
            info[anchor_name] = 1.0 / var_full - 1.0 / var_loo
        else:
            info[anchor_name] = 0.0

    weights = {"MW": 0.20}
    total_info = sum(info.values())
    if total_info > 0:
        for anchor_name, inf in info.items():
            weights[anchor_name] = inf / total_info * 0.80
    else:
        weights["LMC"] = 0.25
        weights["NGC 4258"] = 0.55

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def compute_sigma_ref(weights, screened=False):
    return compute_anchor_sigma_ref(screened=screened, weights=weights)


# ---------------------------------------------------------------------------
# TEP environmental covariate
# ---------------------------------------------------------------------------

def build_host_x(sigma, sigma_ref, S=1.0):
    if sigma is None or sigma <= 0 or sigma_ref <= 0:
        return 0.0
    return (S * sigma ** 2 - sigma_ref ** 2) / C_SQUARED_KM_S


# ---------------------------------------------------------------------------
# Extract Cepheid moduli and SN data from SH0ES baseline fit
# ---------------------------------------------------------------------------

def extract_baseline_data(L, y, C, q, host_sigma, host_z, host_S):
    """Extract per-host Cepheid moduli, SN calibrator magnitudes, and
    Hubble-flow SN data from the baseline SH0ES fit."""
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except:
        A_w = L.copy()
        y_w = y.copy()

    theta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=1e-12)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        cov = np.linalg.pinv(A_w.T @ A_w, rcond=1e-12)

    mu_params = [(q[i].replace("mu_", ""), i) for i in range(len(q)) if q[i].startswith("mu_")]
    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}

    hosts = []
    for host_name, mu_idx in mu_params:
        if host_name in anchor_hosts:
            continue
        if host_name not in host_sigma:
            continue
        z = host_z.get(host_name)
        if z is None or z <= 0:
            continue
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
        mu_fit = theta[mu_idx]
        mu_err = np.sqrt(cov[mu_idx, mu_idx]) if np.isfinite(cov[mu_idx, mu_idx]) else 0.05
        hosts.append({
            "host": host_name,
            "mu_cep": float(mu_fit),
            "mu_cep_err": float(mu_err),
            "sigma": host_sigma[host_name],
            "z": float(z),
            "S": host_S.get(host_name, 1.0),
        })

    mb_idx = np.where(q == "MB")[0]
    h0_idx = np.where(q == "5logH0")[0]
    M_B_base = float(theta[mb_idx[0]]) if len(mb_idx) > 0 else -19.25
    five_log_h0 = float(theta[h0_idx[0]]) if len(h0_idx) > 0 else np.log10(73.04) * 5
    H0_base = 10 ** (five_log_h0 / 5.0)

    mb_i = mb_idx[0] if len(mb_idx) > 0 else None
    h0_i = h0_idx[0] if len(h0_idx) > 0 else None
    calibrators = []
    all_mu_params = [(q[i].replace("mu_", ""), i) for i in range(len(q)) if q[i].startswith("mu_")]
    for i in range(L.shape[0]):
        nonzero = np.where(np.abs(L[i]) > 0.01)[0]
        params = [q[j] for j in nonzero]
        is_sn = mb_i is not None and abs(L[i, mb_i]) > 0.01
        is_hf = h0_i is not None and abs(L[i, h0_i]) > 0.01
        is_ceph = "MHW1" in params
        if is_sn and not is_hf and not is_ceph:
            host = None
            mu_idx_host = None
            for hn, mi in all_mu_params:
                if abs(L[i, mi]) > 0.01:
                    host = hn
                    mu_idx_host = mi
                    break
            if host is None:
                continue
            sigma_host = host_sigma.get(host)
            if sigma_host is None or sigma_host <= 0:
                continue
            mu_fit_host = float(theta[mu_idx_host])
            sigma_sn = float(np.sqrt(C[i, i])) if C[i, i] > 0 else 0.15
            mu_err_host = float(np.sqrt(cov[mu_idx_host, mu_idx_host])) if np.isfinite(cov[mu_idx_host, mu_idx_host]) else 0.05
            calibrators.append({
                "host": host,
                "m_b_obs": float(y[i]),
                "m_b_err": sigma_sn,
                "mu_cep": mu_fit_host,
                "mu_cep_err": mu_err_host,
                "sigma": sigma_host,
                "S": host_S.get(host, 1.0),
            })

    hf_y = []
    hf_sigma = []
    for i in range(L.shape[0]):
        nonzero = np.where(np.abs(L[i]) > 0.01)[0]
        params = [q[j] for j in nonzero]
        is_sn = mb_i is not None and abs(L[i, mb_i]) > 0.01
        is_hf = h0_i is not None and abs(L[i, h0_i]) > 0.01
        is_ceph = "MHW1" in params
        has_mu = any(p.startswith("mu_") for p in params)
        if is_sn and is_hf and not is_ceph and not has_mu:
            hf_y.append(float(y[i]))
            hf_sigma.append(float(np.sqrt(C[i, i])) if C[i, i] > 0 else 0.15)

    hubble_flow = {
        "y": np.array(hf_y),
        "sigma": np.array(hf_sigma),
    }

    return hosts, hubble_flow, M_B_base, H0_base, calibrators


# ---------------------------------------------------------------------------
# Match TRGB hosts
# ---------------------------------------------------------------------------

def match_trgb_hosts(hosts, sigma_ref):
    if not TRGB_PATH.exists():
        return []
    df_trgb = pd.read_csv(TRGB_PATH)
    df_trgb["match_name"] = df_trgb["match_name"].fillna("").astype(str)
    df_trgb["galaxy"] = df_trgb["galaxy"].fillna("").astype(str)

    matched = []
    for h in hosts:
        host = h["host"]
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
            "mu_cep": h["mu_cep"],
            "mu_cep_err": h["mu_cep_err"],
            "mu_trgb": float(trgb["mu_trgb"]),
            "mu_trgb_err": float(trgb["mu_trgb_err"]),
            "sigma": h["sigma"],
            "S": h["S"],
        })
    return matched


# ---------------------------------------------------------------------------
# Unified likelihood (v2 — corrected)
# ---------------------------------------------------------------------------

def total_neg_logL(params, hosts, hubble_flow, matched_trgb, sigma_ref,
                   model_type="Mixed", calibrators=None,
                   fix_sigma_v=None, fix_sigma_int=None,
                   kappa_trgb_fixed=0.0,
                   use_rd_block=True, use_kappa_prior=False,
                   kappa_prior=None, M_B_baseline=-19.253):
    """Total negative log-likelihood for the unified model (v2).

    M_B is DETERMINED by kappa through the mean TEP correction:
        M_B = M_B_baseline - kappa * <X_cal>

    Key v2 changes:
      - sigma_v / sigma_int can be FIXED from the null-model fit,
        breaking the sigma_v–kappa degeneracy that made the RD block
        uninformative in v1.
      - kappa_TRGB is FIXED to zero (TRGB = stellar evolution clock,
        negligible TEP response).  The TRGB block becomes a direct
        constraint on kappa_Cep.
      - The Step 44 prior is OFF by default (use_kappa_prior=False),
        eliminating double-counting of the redshift-distance data.
      - The RD block can be disabled (use_rd_block=False) for the
        propagation cross-check mode.

    Parameter layout depends on which parameters are fixed:

    If fix_sigma_v is None (free):
        Null:    [sigma_v, sigma_int, delta_m, H0]
        Cepheid: [kappa, sigma_v, sigma_int, delta_m, H0]
        Mixed:   [kappa, Gamma_env, sigma_v, sigma_int, delta_m, H0]

    If fix_sigma_v is not None (fixed):
        Null:    [delta_m, H0]
        Cepheid: [kappa, delta_m, H0]
        Mixed:   [kappa, Gamma_env, delta_m, H0]
    """
    has_trgb = len(matched_trgb) > 0
    sv = fix_sigma_v
    si = fix_sigma_int

    idx = 0
    if model_type == "Null":
        kappa = 0.0
        Gamma_env = 0.0
        if sv is None:
            sigma_v = params[idx]; idx += 1
            sigma_int = params[idx]; idx += 1
        else:
            sigma_v = sv
            sigma_int = si if si is not None else 5.0
        if has_trgb:
            delta_m = params[idx]; idx += 1
            H0 = params[idx]; idx += 1
        else:
            delta_m = 0.0
            H0 = params[idx]; idx += 1
        kappa_trgb = 0.0
    elif model_type == "Cepheid":
        kappa = params[idx] * KAPPA_SCALE; idx += 1
        Gamma_env = 0.0
        if sv is None:
            sigma_v = params[idx]; idx += 1
            sigma_int = params[idx]; idx += 1
        else:
            sigma_v = sv
            sigma_int = si if si is not None else 5.0
        if has_trgb:
            delta_m = params[idx]; idx += 1
            H0 = params[idx]; idx += 1
        else:
            delta_m = 0.0
            H0 = params[idx]; idx += 1
        kappa_trgb = kappa_trgb_fixed
    elif model_type == "Mixed":
        kappa = params[idx] * KAPPA_SCALE; idx += 1
        Gamma_env = params[idx] * BETA_SCALE; idx += 1
        if sv is None:
            sigma_v = params[idx]; idx += 1
            sigma_int = params[idx]; idx += 1
        else:
            sigma_v = sv
            sigma_int = si if si is not None else 5.0
        if has_trgb:
            delta_m = params[idx]; idx += 1
            H0 = params[idx]; idx += 1
        else:
            delta_m = 0.0
            H0 = params[idx]; idx += 1
        kappa_trgb = kappa_trgb_fixed
    else:
        raise ValueError(f"Unknown model: {model_type}")

    if H0 is None or H0 <= 0:
        return 1e10

    X_hosts = np.array([
        build_host_x(h["sigma"], sigma_ref, S=h["S"])
        for h in hosts
    ])

    # --- M_B propagation from calibrator-host TEP corrections ---
    # The host moduli used here come from the baseline SH0ES solution.  A true
    # calibrator-SN residual likelihood would need Cepheid-only host moduli to
    # avoid double-counting the same SN rows that helped determine the baseline
    # mu_i.  Here calibrators only determine the mean zero-point propagation.
    if calibrators is not None and len(calibrators) > 0:
        # Host-weighted mean (not SN-weighted) to avoid bias from hosts with multiple SNe
        seen_hosts = {}
        for c in calibrators:
            if c["host"] not in seen_hosts:
                seen_hosts[c["host"]] = build_host_x(c["sigma"], sigma_ref, S=c["S"])
        mean_X_cal = np.mean(list(seen_hosts.values()))
    else:
        mean_X_cal = np.mean(X_hosts)
    M_B = M_B_baseline - kappa * mean_X_cal

    # --- Block 1: Hubble-flow SNe ---
    y_hf = hubble_flow["y"]
    sigma_hf = hubble_flow["sigma"]
    resid_hf = y_hf - M_B + 5 * np.log10(H0)
    chi2_hf = np.sum(resid_hf ** 2 / sigma_hf ** 2)
    n_hf = len(y_hf)
    logL_hf_norm = -0.5 * np.sum(np.log(2 * np.pi * sigma_hf ** 2))

    # --- Block 2: Redshift-distance (optional) ---
    # Per-host variance: sigma_v^2 (peculiar velocity, global) +
    # (cz_i * ln(10)/5 * mu_cep_err_i)^2 (Cepheid modulus, per-host)
    if use_rd_block:
        mu_corrected = np.array([h["mu_cep"] + kappa * build_host_x(h["sigma"], sigma_ref, S=h["S"])
                                  for h in hosts])
        d_true = 10 ** ((mu_corrected - 25.0) / 5.0)
        cz_obs = np.array([C_KM_S * h["z"] for h in hosts])
        cz_model = d_true * (H0 + Gamma_env * X_hosts)
        resid_rd = cz_obs - cz_model
        # Per-host Cepheid modulus velocity uncertainty
        sigma_cz_cep = cz_obs * LN10_OVER_5 * np.array([h["mu_cep_err"] for h in hosts])
        var_rd = sigma_v ** 2 + sigma_cz_cep ** 2
        chi2_rd = np.sum(resid_rd ** 2 / var_rd)
        n_rd = len(hosts)
        logL_rd_norm = -0.5 * np.sum(np.log(2 * np.pi * var_rd))
    else:
        chi2_rd = 0.0
        n_rd = 0
        logL_rd_norm = 0.0

    # --- Block 3: TRGB differential (kappa_TRGB fixed to 0) ---
    # The differential uncertainty is sqrt(mu_cep_err^2 + mu_trgb_err^2)
    # because mu_cep and mu_trgb are independent measurements.
    if len(matched_trgb) > 0:
        X_trgb = np.array([
            build_host_x(m["sigma"], sigma_ref, S=m["S"])
            for m in matched_trgb
        ])
        dmu_obs = np.array([m["mu_cep"] + kappa * build_host_x(m["sigma"], sigma_ref, S=m["S"])
                            - m["mu_trgb"] for m in matched_trgb])
        dmu_model = delta_m + kappa_trgb * X_trgb
        dmu_err = np.array([np.sqrt(m["mu_cep_err"]**2 + m["mu_trgb_err"]**2)
                            for m in matched_trgb])
        resid_trgb = dmu_obs - dmu_model
        chi2_trgb = np.sum(resid_trgb ** 2 / dmu_err ** 2)
        n_trgb = len(matched_trgb)
        logL_trgb_norm = -0.5 * np.sum(np.log(2 * np.pi * dmu_err ** 2))
    else:
        chi2_trgb = 0.0
        n_trgb = 0
        logL_trgb_norm = 0.0

    # --- Total likelihood ---
    logL = (-0.5 * (chi2_hf + chi2_rd + chi2_trgb)
            + logL_hf_norm + logL_rd_norm + logL_trgb_norm)

    # Weak Gaussian prior on delta_m (regularization, sigma=1.0 mag)
    # This prevents delta_m from running to unphysical values when the
    # TRGB block has few hosts. The prior is negligible for the
    # fitted values (delta_m ~ 0.02 mag, penalty ~ 0.0002).
    if n_trgb > 0:
        logL += -0.5 * (delta_m / 1.0) ** 2

    # Gaussian prior on kappa from Step 44 (propagation mode only)
    if use_kappa_prior and kappa_prior is not None and model_type != "Null":
        kappa_prior_val, kappa_prior_err = kappa_prior
        logL += -0.5 * ((kappa - kappa_prior_val) / kappa_prior_err) ** 2

    nll = -logL
    if not np.isfinite(nll):
        return 1e10
    return nll


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_unified(hosts, hubble_flow, matched_trgb, sigma_ref, model_type="Mixed",
                calibrators=None,
                fix_sigma_v=None, fix_sigma_int=None,
                kappa_trgb_fixed=0.0,
                use_rd_block=True, use_kappa_prior=False,
                kappa_prior=None, M_B_baseline=-19.253):
    args = (hosts, hubble_flow, matched_trgb, sigma_ref, model_type, calibrators,
            fix_sigma_v, fix_sigma_int, kappa_trgb_fixed,
            use_rd_block, use_kappa_prior, kappa_prior, M_B_baseline)

    has_trgb = len(matched_trgb) > 0
    sv_fixed = fix_sigma_v is not None

    if model_type == "Null":
        if sv_fixed:
            if has_trgb:
                x0 = np.array([0.0, 72.0])
                bounds = [(-5.0, 5.0), (60.0, 80.0)]
            else:
                x0 = np.array([72.0])
                bounds = [(60.0, 80.0)]
        else:
            if has_trgb:
                x0 = np.array([250.0, 5.0, 0.0, 73.0])
                bounds = [(10.0, 500.0), (0.01, 50.0), (-5.0, 5.0), (60.0, 80.0)]
            else:
                x0 = np.array([250.0, 5.0, 73.0])
                bounds = [(10.0, 500.0), (0.01, 50.0), (60.0, 80.0)]
    elif model_type == "Cepheid":
        if sv_fixed:
            if has_trgb:
                x0 = np.array([8.0, 0.0, 70.0])
                bounds = [(-50.0, 50.0), (-5.0, 5.0), (60.0, 80.0)]
            else:
                x0 = np.array([8.0, 70.0])
                bounds = [(-50.0, 50.0), (60.0, 80.0)]
        else:
            if has_trgb:
                x0 = np.array([5.0, 250.0, 5.0, 0.0, 71.0])
                bounds = [(-50.0, 50.0), (10.0, 500.0), (0.01, 50.0), (-5.0, 5.0), (60.0, 80.0)]
            else:
                x0 = np.array([5.0, 250.0, 5.0, 71.0])
                bounds = [(-50.0, 50.0), (10.0, 500.0), (0.01, 50.0), (60.0, 80.0)]
    elif model_type == "Mixed":
        if sv_fixed:
            if has_trgb:
                x0 = np.array([8.0, 2.35, 0.0, 70.0])
                bounds = [(-50.0, 50.0), (-100.0, 100.0), (-5.0, 5.0), (60.0, 80.0)]
            else:
                x0 = np.array([8.0, 2.35, 70.0])
                bounds = [(-50.0, 50.0), (-100.0, 100.0), (60.0, 80.0)]
        else:
            if has_trgb:
                x0 = np.array([5.0, 2.35, 250.0, 5.0, 0.0, 70.0])
                bounds = [(-50.0, 50.0), (-100.0, 100.0), (10.0, 500.0),
                          (0.01, 50.0), (-5.0, 5.0), (60.0, 80.0)]
            else:
                x0 = np.array([5.0, 2.35, 250.0, 5.0, 70.0])
                bounds = [(-50.0, 50.0), (-100.0, 100.0), (10.0, 500.0),
                          (0.01, 50.0), (60.0, 80.0)]
    else:
        raise ValueError(f"Unknown model: {model_type}")

    best_res = None
    init_points = [x0]
    if model_type in ("Cepheid", "Mixed"):
        for kappa_init in [0.0, 5.0, 10.0, 15.0, 20.0]:
            x0_try = x0.copy()
            x0_try[0] = kappa_init
            x0_try[-1] = 69.0
            init_points.append(x0_try)
            x0_try2 = x0_try.copy()
            x0_try2[-1] = 73.0
            init_points.append(x0_try2)

    for x0_try in init_points:
        res = optimize.minimize(
            total_neg_logL, x0_try, args=args,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if best_res is None or (res.fun < best_res.fun and np.isfinite(res.fun)):
            best_res = res

    res = best_res
    p = res.x

    # Extract parameters
    idx = 0
    if model_type == "Null":
        kappa = 0.0
        Gamma_env = 0.0
        if not sv_fixed:
            sigma_v = p[idx]; idx += 1
            sigma_int = p[idx]; idx += 1
        else:
            sigma_v = fix_sigma_v
            sigma_int = fix_sigma_int if fix_sigma_int is not None else 5.0
        if has_trgb:
            delta_m = p[idx]; idx += 1
            H0 = p[idx]; idx += 1
        else:
            delta_m = 0.0
            H0 = p[idx]; idx += 1
        kappa_trgb = 0.0
    elif model_type == "Cepheid":
        kappa = p[idx] * KAPPA_SCALE; idx += 1
        Gamma_env = 0.0
        if not sv_fixed:
            sigma_v = p[idx]; idx += 1
            sigma_int = p[idx]; idx += 1
        else:
            sigma_v = fix_sigma_v
            sigma_int = fix_sigma_int if fix_sigma_int is not None else 5.0
        if has_trgb:
            delta_m = p[idx]; idx += 1
            H0 = p[idx]; idx += 1
        else:
            delta_m = 0.0
            H0 = p[idx]; idx += 1
        kappa_trgb = kappa_trgb_fixed
    elif model_type == "Mixed":
        kappa = p[idx] * KAPPA_SCALE; idx += 1
        Gamma_env = p[idx] * BETA_SCALE; idx += 1
        if not sv_fixed:
            sigma_v = p[idx]; idx += 1
            sigma_int = p[idx]; idx += 1
        else:
            sigma_v = fix_sigma_v
            sigma_int = fix_sigma_int if fix_sigma_int is not None else 5.0
        if has_trgb:
            delta_m = p[idx]; idx += 1
            H0 = p[idx]; idx += 1
        else:
            delta_m = 0.0
            H0 = p[idx]; idx += 1
        kappa_trgb = kappa_trgb_fixed

    # Compute M_B propagation from calibrator-host TEP corrections
    X_hosts_fit = np.array([build_host_x(h["sigma"], sigma_ref, S=h["S"]) for h in hosts])
    if calibrators is not None and len(calibrators) > 0:
        seen_hosts = {}
        for c in calibrators:
            if c["host"] not in seen_hosts:
                seen_hosts[c["host"]] = build_host_x(c["sigma"], sigma_ref, S=c["S"])
        mean_X_cal = np.mean(list(seen_hosts.values()))
    else:
        mean_X_cal = np.mean(X_hosts_fit)
    mean_correction = kappa * mean_X_cal
    M_B = M_B_baseline - mean_correction

    # Compute chi2 blocks
    y_hf = hubble_flow["y"]
    sigma_hf = hubble_flow["sigma"]
    resid_hf = y_hf - M_B + 5 * np.log10(H0)
    chi2_hf = np.sum(resid_hf ** 2 / sigma_hf ** 2)
    n_hf = len(y_hf)

    if use_rd_block:
        mu_corr = np.array([h["mu_cep"] + kappa * build_host_x(h["sigma"], sigma_ref, S=h["S"]) for h in hosts])
        d_true = 10 ** ((mu_corr - 25.0) / 5.0)
        cz_obs = np.array([C_KM_S * h["z"] for h in hosts])
        cz_model = d_true * (H0 + Gamma_env * X_hosts_fit)
        sigma_cz_cep = cz_obs * LN10_OVER_5 * np.array([h["mu_cep_err"] for h in hosts])
        var_rd = sigma_v ** 2 + sigma_cz_cep ** 2
        chi2_rd = np.sum((cz_obs - cz_model) ** 2 / var_rd)
        n_rd = len(hosts)
    else:
        chi2_rd = 0.0
        n_rd = 0

    if has_trgb:
        X_trgb = np.array([build_host_x(m["sigma"], sigma_ref, S=m["S"]) for m in matched_trgb])
        dmu_obs = np.array([m["mu_cep"] + kappa * build_host_x(m["sigma"], sigma_ref, S=m["S"])
                            - m["mu_trgb"] for m in matched_trgb])
        dmu_model = delta_m + kappa_trgb * X_trgb
        dmu_err = np.array([np.sqrt(m["mu_cep_err"]**2 + m["mu_trgb_err"]**2)
                            for m in matched_trgb])
        chi2_trgb = np.sum((dmu_obs - dmu_model) ** 2 / dmu_err ** 2)
        n_trgb = len(matched_trgb)
    else:
        chi2_trgb = 0.0
        n_trgb = 0

    chi2_prior = 0.0
    if use_kappa_prior and kappa_prior is not None and model_type != "Null":
        kp_val, kp_err = kappa_prior
        chi2_prior = ((kappa - kp_val) / kp_err) ** 2

    total_chi2 = chi2_hf + chi2_rd + chi2_trgb + chi2_prior
    n_total = n_hf + n_rd + n_trgb
    n_params = len(p)
    dof = n_total - n_params
    logL = -res.fun

    # Hessian-based uncertainties
    try:
        n_p = len(res.x)
        hess = np.zeros((n_p, n_p))
        # Per-parameter step size: eps_i = max(1e-6, 1e-4 * |x_i|)
        # This avoids the problem where a fixed eps=1e-3 is too large
        # for small parameters (e.g. delta_m ~ 0.02) and too small
        # for large parameters (e.g. H0 ~ 72).
        eps_vec = np.maximum(1e-6, 1e-4 * np.abs(res.x))
        f0 = total_neg_logL(res.x, *args)
        for i in range(n_p):
            ei = eps_vec[i]
            for j in range(n_p):
                ej = eps_vec[j]
                if i == j:
                    xp = res.x.copy(); xp[i] += ei
                    xm = res.x.copy(); xm[i] -= ei
                    hess[i, i] = (total_neg_logL(xp, *args) - 2*f0 + total_neg_logL(xm, *args)) / ei**2
                elif j > i:
                    xp = res.x.copy(); xp[i] += ei; xp[j] += ej
                    xm = res.x.copy(); xm[i] -= ei; xm[j] -= ej
                    xpi = res.x.copy(); xpi[i] += ei; xpi[j] -= ej
                    xmi = res.x.copy(); xmi[i] -= ei; xmi[j] += ej
                    hess[i, j] = (total_neg_logL(xp, *args) - total_neg_logL(xpi, *args)
                                  - total_neg_logL(xmi, *args) + total_neg_logL(xm, *args)) / (4*ei*ej)
                    hess[j, i] = hess[i, j]
        cov = np.linalg.pinv(hess, rcond=1e-10)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except:
        se = np.full(len(p), np.nan)

    H0_err_stat = se[-1]
    if model_type == "Null":
        kappa_err = np.nan
    elif model_type == "Cepheid":
        kappa_err = se[0] * KAPPA_SCALE
    elif model_type == "Mixed":
        kappa_err = se[0] * KAPPA_SCALE

    # SH0ES systematic uncertainty
    H0_SYS = 1.00
    H0_err = np.sqrt(H0_err_stat ** 2 + H0_SYS ** 2) if np.isfinite(H0_err_stat) else H0_SYS

    tension_planck = abs(H0 - H0_PLANCK) / np.sqrt(H0_err ** 2 + H0_PLANCK_ERR ** 2) if H0_err and H0_err > 0 else np.nan
    tension_sh0es = abs(H0 - H0_SH0ES) / np.sqrt(H0_err ** 2 + H0_SH0ES_ERR ** 2) if H0_err and H0_err > 0 else np.nan

    kappa_sig = abs(kappa) / kappa_err if kappa_err and kappa_err > 0 else np.nan

    aic = -2 * logL + 2 * n_params
    bic = -2 * logL + n_params * np.log(max(n_total, 1))

    return {
        "model": model_type,
        "H0": float(H0),
        "H0_err": float(H0_err) if np.isfinite(H0_err) else np.nan,
        "kappa_Cep": float(kappa),
        "kappa_Cep_err": float(kappa_err) if np.isfinite(kappa_err) else np.nan,
        "kappa_Cep_sigma": float(kappa_sig) if np.isfinite(kappa_sig) else np.nan,
        "Gamma_env": float(Gamma_env),
        "sigma_v": float(sigma_v),
        "sigma_int": float(sigma_int),
        "kappa_TRGB": float(kappa_trgb),
        "delta_m": float(delta_m),
        "M_B": float(M_B),
        "mean_correction": float(mean_correction),
        "chi2_hf": float(chi2_hf),
        "n_hf": int(n_hf),
        "chi2_rd": float(chi2_rd),
        "n_rd": int(n_rd),
        "chi2_trgb": float(chi2_trgb),
        "n_trgb": int(n_trgb),
        "chi2_prior": float(chi2_prior),
        "chi2_total": float(total_chi2),
        "dof": int(dof),
        "logL": float(logL),
        "AIC": float(aic),
        "BIC": float(bic),
        "tension_planck": float(tension_planck) if np.isfinite(tension_planck) else np.nan,
        "tension_sh0es": float(tension_sh0es) if np.isfinite(tension_sh0es) else np.nan,
        "n_params": int(n_params),
        "n_total": int(n_total),
        "sigma_v_fixed": sv_fixed,
        "use_rd_block": use_rd_block,
        "use_kappa_prior": use_kappa_prior,
        "kappa_trgb_fixed": float(kappa_trgb_fixed),
        "kappa_Cep_sigma_hess": float(kappa_sig) if np.isfinite(kappa_sig) else np.nan,
        "status": "converged" if res.success else "fallback",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print_status("Step 50: Diagnostic Summary-Block Likelihood", "SECTION")
    print_status(
        "Diagnostic only: reuses fitted summaries and fails reference-gauge invariance; "
        "do not use for manuscript inference.",
        "WARNING",
    )
    print_status(
        "Joint fit of (H_0, kappa_Cep, Gamma_env, sigma_v, sigma_int)\n"
        "with Cepheid moduli + SN calibrators + Hubble-flow SNe +\n"
        "redshift-distance + TRGB in a single likelihood.\n\n"
        "v2 corrections:\n"
        "  1. sigma_v FIXED from null-model fit (breaks sigma_v-kappa degeneracy)\n"
        "  2. NO Step 44 prior in primary mode (eliminates double-counting)\n"
        "  3. kappa_TRGB FIXED to 0 (TRGB = stellar evolution clock)\n"
        "  4. Propagation cross-check mode (Step 44 prior, no RD block)",
        "INFO",
    )

    print_status("Loading data...", "SECTION")
    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_S = load_host_metadata()
    print_status(f"SH0ES design matrix: {L.shape}", "INFO")

    print_status("Computing anchor reference dispersion...", "SECTION")
    fisher_weights = compute_fisher_anchor_weights(L, y, C, q)
    weights = dict(ANCHOR_WEIGHTS)
    print_status(f"  Canonical SH0ES-motivated anchor weights: {weights}", "INFO")
    print_status(f"  Fisher diagnostic weights (not used for primary sigma_ref): {fisher_weights}", "INFO")

    sigma_ref_std = compute_sigma_ref(weights, screened=False)
    sigma_ref_scr = compute_sigma_ref(weights, screened=True)
    print_status(f"  Standard  sigma_ref = {sigma_ref_std:.2f} km/s", "INFO")
    print_status(f"  Screened  sigma_ref = {sigma_ref_scr:.2f} km/s", "INFO")

    print_status("Extracting baseline data...", "SECTION")
    hosts, hubble_flow, M_B_base, H0_base, calibrators = extract_baseline_data(
        L, y, C, q, host_sigma, host_z, host_S
    )
    print_status(f"  Hosts with Cepheids + redshifts: {len(hosts)}", "INFO")
    print_status(f"  Calibrator SNe: {len(calibrators)}", "INFO")
    print_status(f"  Hubble-flow SNe: {len(hubble_flow['y'])}", "INFO")
    print_status(f"  Baseline M_B = {M_B_base:.3f}, H0 = {H0_base:.2f}", "INFO")

    # Load kappa prior from Step 44 (for propagation mode only)
    step44_path = OUT_DIR / "step_44_joint_distance_redshift_likelihood.json"
    kappa_prior = None
    if step44_path.exists():
        d44 = json.load(open(step44_path))
        for r in d44.get("results", []):
            if r.get("sigma_v") == 250 and r.get("model") == "Velocity":
                GX = r.get("beta_X", 0)
                GX_err = r.get("beta_X_err", 0)
                H_app = r.get("H_app", 70)
                kappa_equiv = GX * 5.0 / (H_app * np.log(10))
                kappa_equiv_err = GX_err * 5.0 / (H_app * np.log(10))
                kappa_prior = (kappa_equiv, kappa_equiv_err)
                break
    if kappa_prior is None:
        kappa_prior = (7.5e5, 4.2e5)
    print_status(f"  kappa prior from Step 44 (for propagation mode): {kappa_prior[0]:.3e} ± {kappa_prior[1]:.3e}", "INFO")

    print_status("Matching TRGB hosts...", "SECTION")
    matched_trgb = match_trgb_hosts(hosts, sigma_ref_scr)
    print_status(f"  TRGB overlap: {len(matched_trgb)} hosts", "INFO")

    results = []

    # ===================================================================
    # MODE 1: Historical raw_data label; diagnostic summary-block fit only
    # ===================================================================
    for sigma_ref, sr_label in [(sigma_ref_std, "standard"), (sigma_ref_scr, "screened")]:
        print_status(f"\n{'='*70}", "INFO")
        print_status(f"MODE 1: SUMMARY-BLOCK DIAGNOSTIC — sigma_ref={sigma_ref:.2f} ({sr_label})", "INFO")
        print_status(f"  No Step 44 prior, sigma_v fixed from null, kappa_TRGB=0", "INFO")
        print_status(f"{'='*70}", "INFO")

        # Step 1a: Fit null model with FREE sigma_v to determine velocity dispersion
        print_status(f"  Fitting Null (free sigma_v)...", "INFO")
        null_free = fit_unified(hosts, hubble_flow, matched_trgb, sigma_ref, "Null",
                                calibrators=calibrators,
                                fix_sigma_v=None, fix_sigma_int=None,
                                kappa_trgb_fixed=0.0,
                                use_rd_block=True, use_kappa_prior=False,
                                kappa_prior=None, M_B_baseline=M_B_base)
        null_free["sigma_ref"] = float(sigma_ref)
        null_free["sigma_ref_label"] = sr_label
        null_free["anchor_weights"] = {k: float(v) for k, v in weights.items()}
        null_free["mode"] = "raw_data"
        null_free["kappa_prior"] = None
        results.append(null_free)

        sv_fix = null_free["sigma_v"]
        si_fix = null_free["sigma_int"]
        print_status(
            f"    Null (free): H0={null_free['H0']:.2f}±{null_free['H0_err']:.2f}, "
            f"σ_v={sv_fix:.1f}, σ_int={si_fix:.1f}, "
            f"χ²={null_free['chi2_total']:.1f}/{null_free['dof']}",
            "SUCCESS",
        )

        # Step 1a-fair: Fit null with FIXED sigma_v for fair BIC comparison
        null_fixed = fit_unified(hosts, hubble_flow, matched_trgb, sigma_ref, "Null",
                                 calibrators=calibrators,
                                 fix_sigma_v=sv_fix, fix_sigma_int=si_fix,
                                 kappa_trgb_fixed=0.0,
                                 use_rd_block=True, use_kappa_prior=False,
                                 kappa_prior=None, M_B_baseline=M_B_base)
        null_fixed["sigma_ref"] = float(sigma_ref)
        null_fixed["sigma_ref_label"] = sr_label
        null_fixed["anchor_weights"] = {k: float(v) for k, v in weights.items()}
        null_fixed["mode"] = "raw_data_null_fixed"
        null_fixed["kappa_prior"] = None
        results.append(null_fixed)
        print_status(
            f"    Null (fixed σ_v): H0={null_fixed['H0']:.2f}±{null_fixed['H0_err']:.2f}, "
            f"χ²={null_fixed['chi2_total']:.1f}/{null_fixed['dof']}, "
            f"BIC={null_fixed['BIC']:.1f} (n_params={null_fixed['n_params']})",
            "SUCCESS",
        )

        # Step 1b: Fit TEP models with sigma_v FIXED from null
        for model_type in ["Cepheid", "Mixed"]:
            print_status(f"  Fitting {model_type} (fixed sigma_v={sv_fix:.1f})...", "INFO")
            res = fit_unified(hosts, hubble_flow, matched_trgb, sigma_ref, model_type,
                              calibrators=calibrators,
                              fix_sigma_v=sv_fix, fix_sigma_int=si_fix,
                              kappa_trgb_fixed=0.0,
                              use_rd_block=True, use_kappa_prior=False,
                              kappa_prior=None, M_B_baseline=M_B_base)
            res["sigma_ref"] = float(sigma_ref)
            res["sigma_ref_label"] = sr_label
            res["anchor_weights"] = {k: float(v) for k, v in weights.items()}
            res["mode"] = "raw_data"
            res["kappa_prior"] = None
            results.append(res)

            print_status(
                f"    {model_type}: H0={res['H0']:.2f}±{res['H0_err']:.2f}, "
                f"κ={res['kappa_Cep']:.3e}±{res['kappa_Cep_err']:.3e} ({res['kappa_Cep_sigma']:.1f}σ), "
                f"Γ={res['Gamma_env']:.3e}, "
                f"χ²={res['chi2_total']:.1f}/{res['dof']}, "
                f"BIC={res['BIC']:.1f}, "
                f"Planck={res['tension_planck']:.2f}σ",
                "SUCCESS",
            )

    # ===================================================================
    # MODE 2: Propagation cross-check — Step 44 prior, no RD block
    # ===================================================================
    for sigma_ref, sr_label in [(sigma_ref_std, "standard"), (sigma_ref_scr, "screened")]:
        print_status(f"\n{'='*70}", "INFO")
        print_status(f"MODE 2: PROPAGATION (cross-check) — sigma_ref={sigma_ref:.2f} ({sr_label})", "INFO")
        print_status(f"  Step 44 prior, NO RD block, kappa_TRGB=0", "INFO")
        print_status(f"{'='*70}", "INFO")

        # Fit null (no RD block, no prior — just HF + TRGB)
        null_prop = fit_unified(hosts, hubble_flow, matched_trgb, sigma_ref, "Null",
                                calibrators=calibrators,
                                fix_sigma_v=sv_fix, fix_sigma_int=si_fix,
                                kappa_trgb_fixed=0.0,
                                use_rd_block=False, use_kappa_prior=False,
                                kappa_prior=None, M_B_baseline=M_B_base)
        null_prop["sigma_ref"] = float(sigma_ref)
        null_prop["sigma_ref_label"] = sr_label
        null_prop["anchor_weights"] = {k: float(v) for k, v in weights.items()}
        null_prop["mode"] = "propagation"
        null_prop["kappa_prior"] = None
        results.append(null_prop)

        print_status(
            f"    Null: H0={null_prop['H0']:.2f}±{null_prop['H0_err']:.2f}, "
            f"χ²={null_prop['chi2_total']:.1f}/{null_prop['dof']}",
            "SUCCESS",
        )

        for model_type in ["Cepheid", "Mixed"]:
            print_status(f"  Fitting {model_type} (propagation)...", "INFO")
            res = fit_unified(hosts, hubble_flow, matched_trgb, sigma_ref, model_type,
                              calibrators=calibrators,
                              fix_sigma_v=sv_fix, fix_sigma_int=si_fix,
                              kappa_trgb_fixed=0.0,
                              use_rd_block=False, use_kappa_prior=True,
                              kappa_prior=kappa_prior, M_B_baseline=M_B_base)
            res["sigma_ref"] = float(sigma_ref)
            res["sigma_ref_label"] = sr_label
            res["anchor_weights"] = {k: float(v) for k, v in weights.items()}
            res["mode"] = "propagation"
            res["kappa_prior"] = {"value": kappa_prior[0], "err": kappa_prior[1]}
            results.append(res)

            print_status(
                f"    {model_type}: H0={res['H0']:.2f}±{res['H0_err']:.2f}, "
                f"κ={res['kappa_Cep']:.3e}±{res['kappa_Cep_err']:.3e} ({res['kappa_Cep_sigma']:.1f}σ), "
                f"Γ={res['Gamma_env']:.3e}, "
                f"χ²={res['chi2_total']:.1f}/{res['dof']}, "
                f"BIC={res['BIC']:.1f}, "
                f"Planck={res['tension_planck']:.2f}σ",
                "SUCCESS",
            )

    # ===================================================================
    # Model comparison (fair BIC: all models with fixed sigma_v)
    # ===================================================================
    for sr_label in ["standard", "screened"]:
        print_status(f"\nModel comparison ({sr_label}, raw-data, fair BIC)", "SECTION")
        raw = [r for r in results if r["mode"] in ("raw_data", "raw_data_null_fixed") and r["sigma_ref_label"] == sr_label]
        null_fixed = next(r for r in raw if r["mode"] == "raw_data_null_fixed")
        null_free_bic = next(r["BIC"] for r in raw if r["mode"] == "raw_data" and r["model"] == "Null")
        print_status(f"  Null (free σ_v):  BIC={null_free_bic:.1f} (n_params=4, for reference)", "INFO")
        print_status(f"  Null (fixed σ_v): BIC={null_fixed['BIC']:.1f} (n_params=2, fair comparison)", "INFO")
        for res in raw:
            if res["mode"] == "raw_data_null_fixed":
                continue
            delta_bic_fair = res["BIC"] - null_fixed["BIC"]
            delta_bic_free = res["BIC"] - null_free_bic
            # LRT significance: the LRT statistic is Lambda = 2*delta_logL = delta_chi2
            # (for Gaussian likelihoods), which follows chi-squared(n_extra).
            # For n_extra=1: sigma = sqrt(Lambda) = sqrt(delta_chi2).
            delta_chi2 = null_fixed["chi2_total"] - res["chi2_total"]
            n_extra = res["n_params"] - null_fixed["n_params"]
            lrt_sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0
            res["kappa_Cep_sigma_lrt"] = float(lrt_sigma)
            print_status(
                f"  {res['model']:10s}: H0={res['H0']:.2f}±{res['H0_err']:.2f}, "
                f"κ={res['kappa_Cep']:.3e} (Hess:{res['kappa_Cep_sigma']:.1f}σ, LRT:{lrt_sigma:.1f}σ), "
                f"BIC={res['BIC']:.1f}, ΔBIC(fair)={delta_bic_fair:+.1f}, "
                f"Planck={res['tension_planck']:.2f}σ",
                "INFO",
            )

    print_status("\nModel comparison (screened, propagation mode)", "SECTION")
    prop_screened = [r for r in results if r["mode"] == "propagation" and r["sigma_ref_label"] == "screened"]
    null_bic_prop = next(r["BIC"] for r in prop_screened if r["model"] == "Null")
    for res in prop_screened:
        delta_bic = res["BIC"] - null_bic_prop
        print_status(
            f"  {res['model']:10s}: H0={res['H0']:.2f}±{res['H0_err']:.2f}, "
            f"κ={res['kappa_Cep']:.3e} ({res['kappa_Cep_sigma']:.1f}σ), "
            f"BIC={res['BIC']:.1f}, ΔBIC={delta_bic:+.1f}, "
            f"Planck={res['tension_planck']:.2f}σ",
            "INFO",
        )

    # Chi2 decomposition for raw-data mode
    print_status("\nChi2 decomposition (standard, raw-data)", "SECTION")
    raw_std = [r for r in results if r["mode"] in ("raw_data", "raw_data_null_fixed") and r["sigma_ref_label"] == "standard"]
    for res in raw_std:
        print_status(
            f"  {res['model']:10s}: χ²_hf={res['chi2_hf']:.2f}/{res['n_hf']}, "
            f"χ²_rd={res['chi2_rd']:.2f}/{res['n_rd']}, "
            f"χ²_trgb={res['chi2_trgb']:.2f}/{res['n_trgb']}, "
            f"χ²_prior={res['chi2_prior']:.2f}, "
            f"total={res['chi2_total']:.2f}/{res['dof']}",
            "INFO",
        )

    out = {
        "description": "Diagnostic summary-block approximation: CalibratorSN+HubbleFlow+RedshiftDistance+TRGB",
        "validity": "diagnostic_only",
        "exclusion_reason": (
            "Reuses baseline-fitted host moduli and M_B summaries, approximates the "
            "calibrator transfer, and fails reference-gauge invariance."
        ),
        "v2_corrections": [
            "sigma_v fixed from null-model fit (breaks sigma_v-kappa degeneracy)",
            "No Step 44 prior in raw-data mode (eliminates double-counting)",
            "kappa_TRGB fixed to 0 (TRGB = stellar evolution clock, negligible TEP response)",
            "Propagation cross-check mode (Step 44 prior, no RD block)",
        ],
        "anchor_weights": {k: float(v) for k, v in weights.items()},
        "anchor_weights_fisher_diagnostic": {k: float(v) for k, v in fisher_weights.items()},
        "sigma_ref_standard": float(sigma_ref_std),
        "sigma_ref_screened": float(sigma_ref_scr),
        "n_hosts": len(hosts),
        "n_calibrators": len(calibrators),
        "n_hubble_flow": len(hubble_flow["y"]),
        "n_trgb_matched": len(matched_trgb),
        "M_B_baseline": float(M_B_base),
        "H0_baseline": float(H0_base),
        "results": results,
    }
    out_path = OUT_DIR / "step_50_unified_joint_likelihood.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print_status(f"\nSaved: {out_path}", "SUCCESS")


if __name__ == "__main__":
    run()
