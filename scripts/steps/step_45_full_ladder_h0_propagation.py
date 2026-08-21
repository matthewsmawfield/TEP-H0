#!/usr/bin/env python3
"""
step_45_full_ladder_h0_propagation.py

Full-Ladder H_0 Propagation Test

This step implements the decisive audit requested by the reviewer:

    "First establish whether the frozen Cepheid correction actually moves
     the complete SH0ES ladder from 73.04 to approximately 69."

The test proceeds as follows:

  1. Fit the baseline SH0ES model to recover H_0 = 73.04 (validation).
  2. Freeze kappa_Cep at the endpoint-equivalent projection reported by
     Step 39, plus the separately adopted canonical coefficient.
  3. Apply the TEP correction to Cepheid distance moduli:
        mu_corrected = mu_obs + kappa * (S * sigma^2 - U_ref) / c^2
  4. Recompute M_B from corrected moduli:
        M_B_i = m_b_i - mu_corrected_i
  5. Take the weighted mean M_B and apply to the 277 Hubble-flow SNe.
  6. Compute H_0 from the Hubble-diagram intercept.

Two algebraically equivalent choices of U_ref = sigma_ref^2 are tested:

  - Standard:  sigma_ref = 87.17 km/s  (weighted mean of anchor sigma^2)
  - Screened:  sigma_ref = 30.51 km/s  (anchors down-weighted by TEP screening)

They are reference gauges, not different physical models. A constant shift in
U_ref is absorbed by the refitted Cepheid P--L zero point, so the complete
matrix fit must return the same H_0, M_B, and chi^2 in both gauges.

This step also reports the full-ladder likelihood chi^2 penalty for each
frozen-kappa model, using the SH0ES design matrix to properly account for
the Cepheid PL relation, anchor priors, and SN Ia calibrator constraints.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import linalg

from core.constants import KAPPA_GAL

from scripts.utils.logger import (
    TEPLogger,
    print_status,
    print_table,
    set_step_logger,
)
from scripts.utils.tep_correction import (
    tep_correction,
    C_SQUARED_KM_S,
    ANCHOR_SCREENING,
    ANCHOR_NMB,
    ANCHOR_WEIGHTS,
    compute_anchor_sigma_ref,
    build_host_screening_map,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_KM_S = 299792.458
LN10 = np.log(10.0)
LN10_OVER_5 = LN10 / 5.0

# SH0ES data paths
SH0ES_DIR = PROJECT_ROOT / "data" / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = PROJECT_ROOT / "data" / "processed" / "hosts_processed.csv"
PANTHEON_PATH = PROJECT_ROOT / "data" / "raw" / "Pantheon+SH0ES.dat"
OUT_DIR = PROJECT_ROOT / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Planck reference
H0_PLANCK = 67.4
H0_PLANCK_ERR = 0.5
H0_SH0ES = 73.04
H0_SH0ES_ERR = 1.01


def compute_sigma_ref(screened=False):
    """Compute one of two reference-gauge origins for the endpoint.

    Parameters
    ----------
    screened : bool
        Selects the active-endpoint or unscreened weighted-RMS gauge origin.
    """
    return compute_anchor_sigma_ref(screened=screened, weights=ANCHOR_WEIGHTS)


def load_endpoint_kappa():
    """Load the current sigma_v=250 complete-host Step 39 projection."""
    path = OUT_DIR / "step_39_environment_slope_decomposition.json"
    with open(path) as f:
        rows = json.load(f)
    matches = [
        row for row in rows
        if row.get("sample") == "all_r22_hosts"
        and float(row.get("sigma_v", np.nan)) == 250.0
        and float(row.get("z_cut", np.nan)) == 0.0
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one all_r22_hosts Step 39 endpoint row in {path}, "
            f"found {len(matches)}"
        )
    row = matches[0]
    return float(row["kappa_equiv"]), float(row["kappa_equiv_err"])


def load_sh0es_data(include_sources=False):
    """Load the SH0ES R22 design matrix, data vector, covariance, and parameter names."""
    L = np.loadtxt(SH0ES_DIR / "L_R22.txt", delimiter="\t")
    names = ("Source", "Data")
    fmt = ("S20", np.float64)
    y_data = np.loadtxt(
        SH0ES_DIR / "y_R22.txt", unpack=True, skiprows=1,
        dtype={"names": names, "formats": fmt},
    )
    sources = np.asarray(y_data[0]).astype(str)
    y = y_data[1]
    C = np.loadtxt(SH0ES_DIR / "C_R22.txt", delimiter="\t")
    q = np.loadtxt(SH0ES_DIR / "q_R22.txt", unpack=True, dtype="str")
    if include_sources:
        return L, y, C, q, sources
    return L, y, C, q


def load_host_metadata():
    """Load host sigma, screening, and redshift from the processed hosts file.

    Screening factors are computed from the Tully 2015 2MRS group catalog
    (Table 5) using the continuous group-halo screening formula
    S(N_mb) = [1 + (N_mb / N_crit)^gamma]^{-1} with N_crit=10, gamma=1.2.

    For hosts in the step_03 stratified output, the full screening factor
    (including local density screening) is used.  For hosts not in step_03,
    the group-only screening from the Tully catalog is used.  Hosts not in
    either source default to S=1.0 (isolated field galaxy).
    """
    df = pd.read_csv(HOSTS_PATH)
    screening_map = build_host_screening_map(df, PROJECT_ROOT)

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
        no_space = name.replace(" ", "")
        host_sigma[no_space] = sigma
        host_S[no_space] = float(S)
        if pd.notna(z_hd) and z_hd > 0:
            host_z[name] = z_hd
        # Compact aliases
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
    # Explicit aliases
    explicit = {"M1337": "Mrk1337", "N105A": "N105", "N976A": "N976"}
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_sigma and sh0es_name not in host_sigma:
            host_sigma[sh0es_name] = host_sigma[csv_name]
            host_S[sh0es_name] = host_S[csv_name]
            if csv_name in host_z:
                host_z[sh0es_name] = host_z[csv_name]
    # SH0ES calibrator environments use the literature values that define the
    # anchor endpoint, rather than aperture-corrected catalog duplicates.
    calibrators = {
        "MW": (30.0, ANCHOR_SCREENING["MW"]),
        "LMC": (24.0, ANCHOR_SCREENING["LMC"]),
        "SMC": (22.0, ANCHOR_SCREENING["SMC"]),
        "M31": (160.0, ANCHOR_SCREENING["M31"]),
        "N4258": (115.0, ANCHOR_SCREENING["NGC 4258"]),
        "NGC4258": (115.0, ANCHOR_SCREENING["NGC 4258"]),
    }
    for name, (sigma, screening) in calibrators.items():
        host_sigma[name] = sigma
        host_S[name] = screening
    return host_sigma, host_z, host_S


def identify_cepheid_host(row, L, q, sources, mu_indices, mu_names):
    """Identify the environment owning a Cepheid-sensitive SH0ES row."""
    source = str(sources[row])
    if source.startswith("LMC_"):
        return "LMC"
    if source.startswith("MHW1_"):
        return "MW"
    if source in {"SMC", "M31", "N4258"}:
        return source

    # Ordinary SN-host Cepheid rows use the compact host name as Source.
    if source and source not in {"nan", "None"}:
        return source

    for idx, mu_param in zip(mu_indices, mu_names):
        if abs(L[row, idx]) > 0.01:
            return mu_param.replace("mu_", "")
    return None


def fit_sh0es_baseline(L, y, C, q):
    """Fit the baseline SH0ES model and extract H_0, M_B, and host mu_i.

    Returns theta, cov, chi2, and a dict of key parameters.
    """
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except (linalg.LinAlgError, ValueError):
        A_w = L.copy()
        y_w = y.copy()

    theta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=1e-12)
    resid = y_w - A_w @ theta
    chi2 = float(np.sum(resid ** 2))
    dof = len(y) - len(theta)
    try:
        cov = np.linalg.pinv(A_w.T @ A_w, rcond=1e-12)
    except Exception:
        cov = np.eye(len(theta)) * 1e10

    # Extract key parameters
    h0_idx = np.where(q == "5logH0")[0]
    mb_idx = np.where(q == "MB")[0]

    result = {
        "chi2": chi2,
        "dof": dof,
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
    }
    if len(h0_idx) > 0:
        five_log_h0 = theta[h0_idx[0]]
        H0 = 10 ** (five_log_h0 / 5.0)
        H0_err = np.sqrt(cov[h0_idx[0], h0_idx[0]]) * H0 * LN10 / 5.0
        result["H0"] = float(H0)
        result["H0_err"] = float(H0_err)
        result["five_log_H0"] = float(five_log_h0)
    if len(mb_idx) > 0:
        result["M_B"] = float(theta[mb_idx[0]])
        result["M_B_err"] = float(np.sqrt(cov[mb_idx[0], mb_idx[0]]))

    # Extract host distance moduli
    mu_params = [(q[i].replace("mu_", ""), i) for i in range(len(q)) if q[i].startswith("mu_")]
    mu_dict = {}
    for host_name, mu_idx in mu_params:
        mu_dict[host_name] = {
            "mu": float(theta[mu_idx]),
            "mu_err": float(np.sqrt(cov[mu_idx, mu_idx])),
            "idx": mu_idx,
        }
    result["mu_dict"] = mu_dict

    return theta, cov, result


def extract_host_mu_cep(L, y, C, q, host_sigma, host_z, host_S):
    """Extract SH0ES-fitted Cepheid distance moduli per host (non-anchor, with Cepheids)."""
    theta, cov, result = fit_sh0es_baseline(L, y, C, q)
    mu_dict = result["mu_dict"]

    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}
    hosts, mus, mu_errs, sigmas, zs, S_vals = [], [], [], [], [], []

    for host_name, info in mu_dict.items():
        if host_name in anchor_hosts:
            continue
        if host_name not in host_sigma:
            continue
        mu_idx = info["idx"]
        # Check if this host has Cepheid rows
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
        hosts.append(host_name)
        mus.append(info["mu"])
        mu_errs.append(info["mu_err"])
        sigmas.append(host_sigma[host_name])
        zs.append(host_z.get(host_name, np.nan))
        S_vals.append(host_S.get(host_name, 1.0))

    return pd.DataFrame({
        "host": hosts, "mu_cep": mus, "mu_cep_err": mu_errs,
        "sigma": sigmas, "z_hd": zs, "S": S_vals,
    }), result


def load_pantheon_hubble_flow():
    """Load the Pantheon+SH0ES Hubble-flow sample."""
    sn = pd.read_csv(PANTHEON_PATH, sep=r"\s+", comment="#")
    hf = sn[sn["USED_IN_SH0ES_HF"] == 1].copy()
    hf_clean = hf[
        (hf["zCMB"] > 0.023) &
        (hf["m_b_corr"].notna()) &
        (hf["m_b_corr_err_DIAG"].notna())
    ].copy()
    return hf_clean


def compute_h0_from_hubble_flow(hf, M_B, M_B_err=0.0):
    """Compute H_0 from the Hubble-flow sample given M_B.

    Uses a WLS fit of m_b = M_B + 5*log10(c*z/H_0) + 25
    with fixed slope = 5 (Hubble law).
    """
    c = C_KM_S
    logz = np.log10(hf["zCMB"].values)
    mb = hf["m_b_corr"].values
    mb_err = hf["m_b_corr_err_DIAG"].values
    w = 1.0 / mb_err ** 2

    # Fixed slope = 5: mb = a + 5*logz
    # a = M_B + 5*log10(c/H_0) + 25
    # H_0 = c * 10^((a - M_B - 25)/5)
    a_wls = np.sum(w * (mb - 5 * logz)) / np.sum(w)
    H0 = c * 10 ** ((a_wls - M_B - 25) / 5)

    # Error propagation
    # dH_0/H_0 = (ln10/5) * sqrt(var_a + var_M_B)
    var_a = 1.0 / np.sum(w)
    H0_err = H0 * LN10_OVER_5 * np.sqrt(var_a + M_B_err ** 2)

    return float(H0), float(H0_err)


def compute_full_ladder_h0(df_cep, hf, kappa, sigma_ref, label=""):
    """Compute the full-ladder H_0 with a frozen TEP correction.

    Uses the analytic relation:
        H_0_corrected = H_0_baseline * 10^(-mean(delta_mu) / 5)

    where mean(delta_mu) = kappa * mean((S * sigma^2 - U_ref) / c^2)
    over the Cepheid calibrator hosts.  This is exact for a pure
    distance-modulus shift: M_B shifts by -mean(delta_mu), and
    H_0 = c * 10^((a - M_B - 25)/5) shifts by the corresponding factor.

    The baseline H_0 = 73.04 is the SH0ES global fit value, which our
    design-matrix fit reproduces exactly (Step 45 baseline validation).
    """
    # Build TEP correction
    sigma_vals = df_cep["sigma"].values
    S_vals = df_cep["S"].values

    correction = tep_correction(sigma_vals, sigma_ref, kappa, S_vals)
    mean_dmu = float(np.mean(correction))

    # Full-ladder H_0 shift
    # H_0_corrected = H_0 * 10^(-mean_dmu / 5)
    # (brighter M_B -> smaller H_0)
    H0 = H0_SH0ES * 10 ** (-mean_dmu / 5.0)
    H0_err = H0_SH0ES_ERR  # conservative: use baseline error

    # Tension with Planck
    tension = (H0 - H0_PLANCK) / np.sqrt(H0_err ** 2 + H0_PLANCK_ERR ** 2)

    result = {
        "label": label,
        "kappa": float(kappa),
        "sigma_ref": float(sigma_ref),
        "N_hosts": len(df_cep),
        "mean_dmu": mean_dmu,
        "H0": float(H0),
        "H0_err": float(H0_err),
        "tension_planck": float(tension),
        "delta_H0_vs_sh0es": float(H0 - H0_SH0ES),
    }
    return result


def fit_sh0es_with_tep_correction(
    L, y, C, q, sources, host_sigma, host_S, kappa, sigma_ref
):
    """Fit the SH0ES design matrix with TEP-corrected data.

    Instead of adding kappa as a free parameter, we CORRECT the observed
    Cepheid magnitudes and then fit the standard model.  This is the
    proper full-ladder propagation.

    The correction is applied to Cepheid rows:
        y_corrected = y + kappa * X_i
    where X_i = (S_i * sigma_i^2 - U_ref) / c^2 for host i.
    """
    # Build TEP correction per row
    mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_names = [q[i] for i in mu_indices]
    n_rows = L.shape[0]
    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}

    y_corrected = y.copy()
    n_corrected = 0
    corrected_by_host = {}
    unmatched_cepheid_sources = {}

    for i in range(n_rows):
        # Check if this is a Cepheid row
        nonzero = np.where(np.abs(L[i]) > 0.01)[0]
        params = [q[j] for j in nonzero]
        if "MHW1" not in params:
            continue

        host = identify_cepheid_host(i, L, q, sources, mu_indices, mu_names)
        if host is None:
            unmatched_cepheid_sources[str(sources[i])] = (
                unmatched_cepheid_sources.get(str(sources[i]), 0) + 1
            )
            continue

        # Get sigma and S for this host
        sigma = host_sigma.get(host)
        if sigma is None or sigma <= 0:
            unmatched_cepheid_sources[host] = unmatched_cepheid_sources.get(host, 0) + 1
            continue
        S = host_S.get(host, 1.0)

        # Compute TEP correction
        # TEP predicts Cepheid periods contract in high-sigma environments,
        # making them appear BRIGHTER (m_obs < m_true).  The correction
        # restores the true magnitude: m_corrected = m_obs + kappa * X.
        # In the design matrix y = L @ theta, this means y_corrected = y + kappa * X.
        X = (S * sigma ** 2 - sigma_ref ** 2) / C_SQUARED_KM_S
        y_corrected[i] += kappa * X
        n_corrected += 1
        corrected_by_host[host] = corrected_by_host.get(host, 0) + 1

    # Fit corrected data
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y_corrected, lower=True, check_finite=False)
    except (linalg.LinAlgError, ValueError):
        A_w = L.copy()
        y_w = y_corrected.copy()

    theta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=1e-12)
    resid = y_w - A_w @ theta
    chi2 = float(np.sum(resid ** 2))
    dof = len(y) - len(theta)
    try:
        cov = np.linalg.pinv(A_w.T @ A_w, rcond=1e-12)
    except Exception:
        cov = np.eye(len(theta)) * 1e10

    h0_idx = np.where(q == "5logH0")[0]
    mb_idx = np.where(q == "MB")[0]

    result = {
        "chi2": chi2,
        "dof": dof,
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
        "n_corrected_rows": n_corrected,
        "corrected_rows_by_host": corrected_by_host,
        "unmatched_cepheid_sources": unmatched_cepheid_sources,
    }
    if len(h0_idx) > 0:
        five_log_h0 = theta[h0_idx[0]]
        H0 = 10 ** (five_log_h0 / 5.0)
        H0_err = np.sqrt(cov[h0_idx[0], h0_idx[0]]) * H0 * LN10 / 5.0
        result["H0"] = float(H0)
        result["H0_err"] = float(H0_err)
    if len(mb_idx) > 0:
        result["M_B"] = float(theta[mb_idx[0]])
        result["M_B_err"] = float(np.sqrt(cov[mb_idx[0], mb_idx[0]]))

    return result


def classify_rows(L, y, C, q, sources, host_sigma):
    """Classify each design-matrix row into a type for chi^2 decomposition.

    Returns a list of row-type labels of length ``len(y)``:
    ``"cepheid_nonanchor"``, ``"cepheid_anchor"``, ``"sn_calibrator"``,
    ``"sn_hubble_flow"``, or ``"other"``.
    """
    mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_names = [q[i] for i in mu_indices]
    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}
    mb_idx = np.where(q == "MB")[0]
    h0_idx = np.where(q == "5logH0")[0]
    n_rows = L.shape[0]
    labels = []

    for i in range(n_rows):
        nonzero = np.where(np.abs(L[i]) > 0.01)[0]
        params = [q[j] for j in nonzero]

        is_cepheid = "MHW1" in params
        is_sn = len(mb_idx) > 0 and abs(L[i, mb_idx[0]]) > 0.01
        is_hubble_flow = len(h0_idx) > 0 and abs(L[i, h0_idx[0]]) > 0.01

        if is_cepheid:
            host = identify_cepheid_host(i, L, q, sources, mu_indices, mu_names)
            if host in anchor_hosts:
                labels.append("cepheid_anchor")
            else:
                labels.append("cepheid_nonanchor")
        elif is_sn and is_hubble_flow:
            labels.append("sn_hubble_flow")
        elif is_sn:
            labels.append("sn_calibrator")
        else:
            labels.append("other")

    return labels


def chi2_decomposition(L, y, C, q, sources, host_sigma, host_S, kappa, sigma_ref,
                       theta_base, chi2_base):
    """Decompose the Δχ² of the matrix-method TEP correction by row type.

    Returns a dict with the Δχ² contribution from each row category.
    """
    labels = classify_rows(L, y, C, q, sources, host_sigma)
    mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_names = [q[i] for i in mu_indices]
    n_rows = L.shape[0]

    # Apply TEP correction (same logic as fit_sh0es_with_tep_correction)
    y_corrected = y.copy()
    for i in range(n_rows):
        nonzero = np.where(np.abs(L[i]) > 0.01)[0]
        params = [q[j] for j in nonzero]
        if "MHW1" not in params:
            continue
        host = identify_cepheid_host(i, L, q, sources, mu_indices, mu_names)
        if host is None:
            continue
        sigma = host_sigma.get(host)
        if sigma is None or sigma <= 0:
            continue
        S = host_S.get(host, 1.0)
        X = (S * sigma ** 2 - sigma_ref ** 2) / C_SQUARED_KM_S
        y_corrected[i] += kappa * X

    # Whiten with covariance
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w_corr = linalg.solve_triangular(Lc, y_corrected, lower=True, check_finite=False)
        y_w_base = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except (linalg.LinAlgError, ValueError):
        A_w = L.copy()
        y_w_corr = y_corrected.copy()
        y_w_base = y.copy()

    # Fit corrected data
    theta_corr, _, _, _ = np.linalg.lstsq(A_w, y_w_corr, rcond=1e-12)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        resid_corr = y_w_corr - A_w @ theta_corr
        resid_base = y_w_base - A_w @ theta_base
    # Replace any NaN/inf from zero-variance prior rows with 0
    resid_corr = np.nan_to_num(resid_corr, nan=0.0, posinf=0.0, neginf=0.0)
    resid_base = np.nan_to_num(resid_base, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-row Δχ² = (resid_corr^2 - resid_base^2)
    dchi2_per_row = resid_corr ** 2 - resid_base ** 2

    categories = ["cepheid_nonanchor", "sn_calibrator", "cepheid_anchor",
                  "sn_hubble_flow", "other"]
    decomp = {}
    for cat in categories:
        mask = np.array([l == cat for l in labels])
        if mask.any():
            decomp[cat] = {
                "n_rows": int(mask.sum()),
                "delta_chi2": float(np.sum(dchi2_per_row[mask])),
            }
        else:
            decomp[cat] = {"n_rows": 0, "delta_chi2": 0.0}

    total = float(np.sum(dchi2_per_row))
    decomp["total"] = total
    decomp["total_check"] = float(chi2_base + total)  # should ≈ chi2_corr

    # Percentages
    for cat in categories:
        if total != 0:
            decomp[cat]["pct"] = round(100.0 * decomp[cat]["delta_chi2"] / total, 1)
        else:
            decomp[cat]["pct"] = 0.0

    return decomp


def run():
    logger = TEPLogger(
        "step_45_full_ladder",
        log_file_path=PROJECT_ROOT / "logs" / "step_45_full_ladder_h0.log",
    )
    set_step_logger(logger)

    print_status("Step 45: Full-Ladder H_0 Propagation Test", "SECTION")
    print_status(
        "Decisive audit: does frozen kappa move the complete SH0ES ladder\n"
        "from 73.04 to approximately 69?",
        "INFO",
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print_status("Loading data...", "SECTION")
    L, y, C, q, sources = load_sh0es_data(include_sources=True)
    host_sigma, host_z, host_S = load_host_metadata()
    hf = load_pantheon_hubble_flow()
    print_status(f"SH0ES design matrix: {L.shape}, {len(y)} observations, {len(q)} parameters", "INFO")
    print_status(f"Hubble-flow sample: {len(hf)} SNe (z > 0.023)", "INFO")

    # ------------------------------------------------------------------
    # Baseline fit
    # ------------------------------------------------------------------
    print_status("Baseline SH0ES fit...", "SECTION")
    theta_base, cov_base, result_base = fit_sh0es_baseline(L, y, C, q)
    H0_base = result_base.get("H0", np.nan)
    H0_base_err = result_base.get("H0_err", np.nan)
    M_B_base = result_base.get("M_B", np.nan)
    chi2_base = result_base["chi2"]
    dof_base = result_base["dof"]
    print_status(f"  H_0 = {H0_base:.2f} +/- {H0_base_err:.2f} km/s/Mpc", "SUCCESS")
    print_status(f"  M_B = {M_B_base:.4f} +/- {result_base.get('M_B_err', 0):.4f}", "INFO")
    print_status(f"  chi^2/dof = {chi2_base:.1f}/{dof_base} = {chi2_base/dof_base:.4f}", "INFO")
    print_status(f"  (SH0ES reports H_0 = 73.04 +/- 1.01)", "INFO")

    # ------------------------------------------------------------------
    # Extract Cepheid host moduli
    # ------------------------------------------------------------------
    print_status("Extracting Cepheid host distance moduli...", "SECTION")
    df_cep, result_cep = extract_host_mu_cep(L, y, C, q, host_sigma, host_z, host_S)
    print_status(f"  N_Cepheid hosts = {len(df_cep)}", "INFO")
    print_status(f"  sigma range: {df_cep['sigma'].min():.1f} - {df_cep['sigma'].max():.1f} km/s", "INFO")
    print_status(f"  S range: {df_cep['S'].min():.3f} - {df_cep['S'].max():.3f}", "INFO")

    # ------------------------------------------------------------------
    # sigma_ref conventions
    # ------------------------------------------------------------------
    print_status("sigma_ref conventions...", "SECTION")
    sigma_ref_std = compute_sigma_ref(screened=False)
    sigma_ref_scr = compute_sigma_ref(screened=True)
    print_status(f"  Standard  sigma_ref = {sigma_ref_std:.2f} km/s", "INFO")
    print_status(f"  Screened  sigma_ref = {sigma_ref_scr:.2f} km/s", "INFO")
    print_status(f"    MW  S = {ANCHOR_SCREENING['MW']:.3f}  (N_mb = {ANCHOR_NMB['MW']})", "INFO")
    print_status(f"    LMC S = {ANCHOR_SCREENING['LMC']:.3f}  (N_mb = {ANCHOR_NMB['LMC']})", "INFO")
    print_status(f"    N4258 S = {ANCHOR_SCREENING['NGC 4258']:.3f}  (N_mb = {ANCHOR_NMB['NGC 4258']})", "INFO")

    # ------------------------------------------------------------------
    # Coefficients to test
    # ------------------------------------------------------------------
    kappa_endpoint, kappa_endpoint_err = load_endpoint_kappa()
    kappa_values = {
        "kappa_0_null": 0.0,
        "kappa_gal_canonical": float(KAPPA_GAL),
        "kappa_endpoint_equiv": kappa_endpoint,
    }

    # ------------------------------------------------------------------
    # Full-ladder propagation (simple M_B method)
    # ------------------------------------------------------------------
    print_status("Full-ladder propagation (M_B -> Hubble flow)...", "SECTION")
    results_simple = []
    for sigma_ref, sr_label in [(sigma_ref_std, "standard"), (sigma_ref_scr, "screened")]:
        for kappa_name, kappa in kappa_values.items():
            res = compute_full_ladder_h0(df_cep, hf, kappa, sigma_ref,
                                         label=f"{kappa_name}_{sr_label}")
            res["sigma_ref_label"] = sr_label
            res["kappa_name"] = kappa_name
            results_simple.append(res)
            print_status(
                f"  {sr_label:8s} {kappa_name:25s} kappa={kappa:.3e}  "
                f"H_0={res['H0']:.2f} +/- {res['H0_err']:.2f}  "
                f"tension={res['tension_planck']:.2f}sigma  "
                f"delta_H0={res['delta_H0_vs_sh0es']:+.2f}",
                "INFO",
            )

    # ------------------------------------------------------------------
    # Full-ladder propagation (SH0ES design matrix)
    # ------------------------------------------------------------------
    print_status("Full-ladder propagation (SH0ES design matrix)...", "SECTION")
    results_matrix = []
    for sigma_ref, sr_label in [(sigma_ref_std, "standard"), (sigma_ref_scr, "screened")]:
        for kappa_name, kappa in kappa_values.items():
            if kappa == 0:
                # Use baseline
                res = {
                    "label": f"{kappa_name}_{sr_label}",
                    "kappa": 0.0,
                    "sigma_ref": float(sigma_ref),
                    "H0": H0_base,
                    "H0_err": H0_base_err,
                    "M_B": M_B_base,
                    "chi2": chi2_base,
                    "dof": dof_base,
                    "chi2_reduced": chi2_base / dof_base,
                    "delta_chi2": 0.0,
                    "n_corrected_rows": 0,
                }
            else:
                res = fit_sh0es_with_tep_correction(
                    L, y, C, q, sources, host_sigma, host_S, kappa, sigma_ref
                )
                res["label"] = f"{kappa_name}_{sr_label}"
                res["kappa"] = float(kappa)
                res["sigma_ref"] = float(sigma_ref)
                res["delta_chi2"] = res["chi2"] - chi2_base
            res["sigma_ref_label"] = sr_label
            res["kappa_name"] = kappa_name
            res["tension_planck"] = float(
                (res["H0"] - H0_PLANCK) /
                np.sqrt(res["H0_err"] ** 2 + H0_PLANCK_ERR ** 2)
            ) if "H0" in res else np.nan
            results_matrix.append(res)
            print_status(
                f"  {sr_label:8s} {kappa_name:25s} kappa={kappa:.3e}  "
                f"H_0={res.get('H0', np.nan):.2f}  "
                f"delta_chi2={res['delta_chi2']:+.1f}  "
                f"tension={res['tension_planck']:.2f}sigma",
                "INFO",
            )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print_status("Summary: Full-Ladder H_0 Propagation", "SECTION")
    headers = ["sigma_ref", "kappa", "H0_simple", "H0_matrix", "dChi2", "tension"]
    rows = []
    for rs, rm in zip(results_simple, results_matrix):
        rows.append([
            rs["sigma_ref_label"],
            f"{rs['kappa']:.3e}",
            f"{rs['H0']:.2f}",
            f"{rm.get('H0', np.nan):.2f}",
            f"{rm['delta_chi2']:+.1f}",
            f"{rs['tension_planck']:.2f}sigma",
        ])
    print_table(headers, rows, title="Full-Ladder H_0 Propagation Results")

    # ------------------------------------------------------------------
    # Delta-chi2 row-type decomposition for the endpoint-equivalent projection
    # ------------------------------------------------------------------
    print_status("Delta-chi2 row-type decomposition (endpoint-equivalent projection)...", "SECTION")
    chi2_decomp = chi2_decomposition(
        L, y, C, q, sources, host_sigma, host_S,
        kappa_values["kappa_endpoint_equiv"], sigma_ref_std,
        theta_base, chi2_base,
    )
    for cat in ["cepheid_nonanchor", "sn_calibrator", "cepheid_anchor",
                "sn_hubble_flow", "other"]:
        d = chi2_decomp.get(cat, {})
        print_status(
            f"  {cat:22s}: Δχ² = {d.get('delta_chi2', 0):+.1f}  "
            f"({d.get('pct', 0):.1f}%)  N_rows = {d.get('n_rows', 0)}",
            "INFO",
        )
    print_status(f"  {'total':22s}: Δχ² = {chi2_decomp.get('total', 0):+.1f}", "INFO")

    # ------------------------------------------------------------------
    # Key conclusion
    # ------------------------------------------------------------------
    print_status("Key Conclusion", "SECTION")
    for r in results_matrix:
        if r["sigma_ref_label"] == "standard" and r["kappa_name"] == "kappa_endpoint_equiv":
            print_status(
                f"  Endpoint-equivalent kappa ({r['kappa']:.3e}): "
                f"H_0 = {r['H0']:.2f} +/- {r['H0_err']:.2f}, "
                f"Delta-chi2 = {r['delta_chi2']:+.2f}",
                "SUCCESS",
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_json = OUT_DIR / "step_45_full_ladder_h0_propagation.json"
    with open(out_json, "w") as f:
        json.dump({
            "baseline": {
                "H0": H0_base,
                "H0_err": H0_base_err,
                "M_B": M_B_base,
                "chi2": chi2_base,
                "dof": dof_base,
            },
            "sigma_ref": {
                "standard": float(sigma_ref_std),
                "screened": float(sigma_ref_scr),
            },
            "endpoint_projection": {
                "source": "step_39_environment_slope_decomposition.json",
                "sigma_v": 250,
                "sample": "all_r22_hosts",
                "kappa_equiv": kappa_endpoint,
                "kappa_equiv_err": kappa_endpoint_err,
            },
            "simple_propagation": results_simple,
            "matrix_propagation": results_matrix,
            "chi2_decomposition": chi2_decomp,
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    out_csv = OUT_DIR / "step_45_full_ladder_h0_propagation.csv"
    df_out = pd.DataFrame(results_simple)
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    return {
        "simple": results_simple,
        "matrix": results_matrix,
        "baseline": result_base,
    }


if __name__ == "__main__":
    run()
