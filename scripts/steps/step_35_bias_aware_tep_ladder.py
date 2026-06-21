#!/usr/bin/env python3
"""
step_35_bias_aware_tep_ladder.py

Bias-Aware TEP Ladder: Redshift-Distance Prior Sensitivity

The standard SH0ES full-ladder fit treats host distance moduli (mu_i) as
free latent parameters. In this model, any host-level environmental signal
(e.g., TEP) is absorbed into mu_i, making TEP undetectable.

This step has TWO branches:

  PRIMARY: Standard SH0ES + TEP terms (no redshift priors).
           This is the proper distance ladder. mu_i are free.

  SENSITIVITY: Add redshift-distance priors for non-anchor calibrator hosts
               with KNOWN redshift, sweeping peculiar-velocity uncertainty.
               This is a prior-driven sensitivity test, NOT a truth model.
               Nearby calibrator hosts have large peculiar-velocity errors,
               so these priors are weak and model-dependent.

Models (fitted in both branches):
  A: Standard (kappa_Cep = 0, kappa_SN = 0)
  B: Cepheid-bias TEP (kappa_Cep free, kappa_SN = 0)
  C: Dynamic-time local ladder (kappa_Cep free, kappa_SN free)

WARNING: Redshift priors on nearby calibrators assume a Hubble law,
which is inconsistent with a no-expansion TEP theory. This branch
is a STANDARD-EXPANSION sensitivity test only.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_KM_S = 299792.458  # speed of light in km/s
C_SQUARED_KM_S = C_KM_S ** 2
SIGMA_VPEC_KM_S = 200.0  # typical peculiar velocity in km/s


def print_status(msg, level="INFO"):
    """Print with status prefix."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sh0es_data():
    """Load SH0ES design matrix, data vector, covariance, and parameter names."""
    L = np.loadtxt(SH0ES_DIR / "L_R22.txt", delimiter="\t")

    names = ("Source", "Data")
    fmt = ("S20", np.float64)
    y_data = np.loadtxt(
        SH0ES_DIR / "y_R22.txt",
        unpack=True,
        skiprows=1,
        dtype={"names": names, "formats": fmt},
    )
    y_source = y_data[0].astype(str)
    y = y_data[1]

    C = np.loadtxt(SH0ES_DIR / "C_R22.txt", delimiter="\t")
    q = np.loadtxt(SH0ES_DIR / "q_R22.txt", unpack=True, dtype="str")

    print_status(f"Design matrix shape: {L.shape}", "INFO")
    print_status(f"Data vector length: {len(y)}", "INFO")
    print_status(f"Covariance shape: {C.shape}", "INFO")
    print_status(f"Number of parameters: {len(q)}", "INFO")

    return L, y, C, q, y_source


def load_host_metadata():
    """Load host sigma and redshift from processed metadata."""
    df = pd.read_csv(HOSTS_PATH)

    host_sigma = {}
    host_z = {}
    host_screening = {}

    for _, row in df.iterrows():
        name = row["normalized_name"]
        sigma = row["sigma_inferred"]
        z_hd = row["z_hd"]
        S = row.get("shear_suppression", 1.0)

        host_sigma[name] = sigma
        host_screening[name] = S
        if pd.notna(z_hd) and z_hd > 0:
            host_z[name] = z_hd

        # SH0ES-style compact names
        compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
        if compact.startswith(("N", "U")):
            parts = compact[1:]
            if parts.isdigit():
                padded = compact[0] + parts.zfill(4)
                host_sigma[padded] = sigma
                host_screening[padded] = S
                if pd.notna(z_hd) and z_hd > 0:
                    host_z[padded] = z_hd

                unpadded = compact[0] + parts.lstrip("0")
                if unpadded != padded:
                    host_sigma[unpadded] = sigma
                    host_screening[unpadded] = S
                    if pd.notna(z_hd) and z_hd > 0:
                        host_z[unpadded] = z_hd

        # NGC prefix variants
        if compact.startswith("N"):
            ngc_name = "NGC" + compact[1:]
            host_sigma[ngc_name] = sigma
            host_screening[ngc_name] = S
            if pd.notna(z_hd) and z_hd > 0:
                host_z[ngc_name] = z_hd

    # Explicit mappings
    explicit = {"M1337": "N1337", "N105A": "N105", "N976A": "N976"}
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_sigma and sh0es_name not in host_sigma:
            host_sigma[sh0es_name] = host_sigma[csv_name]
            host_screening[sh0es_name] = host_screening[csv_name]
            if csv_name in host_z:
                host_z[sh0es_name] = host_z[csv_name]

    print_status(f"Loaded {len(host_sigma)} sigma mappings, {len(host_z)} redshift mappings", "INFO")
    return host_sigma, host_z, host_screening


def classify_row(i, L, q):
    """Classify a data row by its nonzero parameters."""
    nonzero = np.where(np.abs(L[i]) > 0.01)[0]
    params = [q[j] for j in nonzero]
    if "MHW1" in params:
        return "Cepheid"
    if "MB" in params and "5logH0" not in params:
        return "SN_calibrator"
    if "MB" in params and "5logH0" in params:
        return "SN_Hubble"
    if len(params) == 1 and params[0].startswith("mu_"):
        return "Anchor_prior"
    return "Other"


def build_host_x(host_name, host_sigma, host_screening, sigma_ref, mode="centered"):
    """Build TEP regressor X for a single host."""
    sigma = host_sigma.get(host_name)
    if sigma is None or sigma <= 0:
        return 0.0
    S = host_screening.get(host_name, 1.0)
    if mode == "centered":
        return S * (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S
    if mode == "raw_sigma2":
        return S * sigma**2 / C_SQUARED_KM_S
    if mode == "unscreened_centered":
        return (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S
    if mode == "sigma_linear":
        return S * (sigma - sigma_ref) / C_KM_S
    raise ValueError(f"Unknown X mode: {mode}")


# ---------------------------------------------------------------------------
# GLS fitting
# ---------------------------------------------------------------------------
def fit_gls(A, y, C):
    """Fit weighted least squares via Cholesky whitening.

    Solves: min (y - A theta)^T C^{-1} (y - A theta)
    Via Cholesky: C = Lc Lc^T, then solve Lc^{-1} y = Lc^{-1} A theta.
    """
    from scipy import linalg

    C = np.atleast_1d(C)
    use_cholesky = False
    if C.ndim == 1:
        # Whitening: divide by sqrt(variance)
        sqrt_C = np.sqrt(C)
        A_w = A / sqrt_C[:, None]
        y_w = y / sqrt_C
    else:
        try:
            Lc = np.linalg.cholesky(C)
            A_w = linalg.solve_triangular(Lc, A, lower=True, check_finite=False)
            y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
            use_cholesky = True
        except (linalg.LinAlgError, ValueError):
            A_w = A.copy()
            y_w = y.copy()

    theta, residuals, rank, svals = np.linalg.lstsq(A_w, y_w, rcond=1e-12)

    # Chi2 = ||y_w - A_w theta||^2
    if residuals is not None and residuals.size > 0 and np.isfinite(residuals[0]):
        chi2 = float(residuals[0])
    else:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            r_w = y_w - A_w @ theta
            chi2 = float(r_w.T @ r_w)

    # Covariance from SVD of whitened design matrix (avoids A_w.T @ A_w overflow)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _, s, Vt = np.linalg.svd(A_w, full_matrices=False)
    if s.size == 0:
        cov = np.full((A.shape[1], A.shape[1]), np.nan)
    else:
        tol = 1e-12 * s[0]
        s_inv2 = np.zeros_like(s)
        mask = s > tol
        if np.any(mask):
            s_inv2[mask] = 1.0 / np.maximum(s[mask] ** 2, tol ** 2)
        cov = (Vt.T * s_inv2) @ Vt
        cov = np.where(np.isfinite(cov), cov, np.nan)

    return theta, cov, chi2, rank


# ---------------------------------------------------------------------------
# Redshift-distance prior construction
# ---------------------------------------------------------------------------
def build_redshift_priors(L, q, y, C, y_source, host_z, sigma_vpec=SIGMA_VPEC_KM_S):
    """
    For each non-anchor calibrator host with known redshift, add a prior row:
        mu_host + 5logH0 = 5*log10(c*z) + 25 + noise

    Hard validation rules (per user request):
      - Skip anchor hosts (they have geometric priors)
      - Skip z > 0.05  (these are Hubble-flow hosts, not calibrators)
      - Skip z < 0.0035 (too local; peculiar velocities completely dominate)
      - Skip if sigma_mu_from_vpec > 0.5 mag (prior would be too weak/noisy)

    Returns augmented L, y, C, q, y_source.
    """
    h0_idx = np.where(q == "5logH0")[0]
    if len(h0_idx) == 0:
        print_status("No 5logH0 parameter found — cannot add redshift priors", "WARNING")
        return L, y, C, q, y_source

    h0_idx = h0_idx[0]

    # Find all host parameters
    mu_params = [(i, q[i].replace("mu_", "")) for i in range(len(q)) if q[i].startswith("mu_")]

    prior_rows_L = []
    prior_rows_y = []
    prior_rows_C = []
    prior_rows_source = []

    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}
    n_added = 0
    n_skipped_anchor = 0
    n_skipped_zrange = 0
    n_skipped_noisy = 0

    for mu_idx, host_name in mu_params:
        if host_name in anchor_hosts:
            n_skipped_anchor += 1
            continue
        if host_name not in host_z:
            continue

        z = host_z[host_name]

        # Hard validation
        if z <= 0 or z > 0.5:
            continue
        if z > 0.05:
            n_skipped_zrange += 1
            continue  # Hubble-flow host, not a local calibrator
        if z < 0.0035:
            n_skipped_zrange += 1
            continue  # too local; peculiar velocities dominate

        # Peculiar velocity uncertainty in magnitude
        dm = (5.0 / np.log(10)) * (sigma_vpec / (C_KM_S * z))
        if dm > 0.5:
            n_skipped_noisy += 1
            print_status(
                f"Skipping {host_name}: z={z:.5f}, sigma_mu(vpec)={dm:.3f} mag > 0.5 mag",
                "INFO",
            )
            continue

        # Compute apparent distance modulus from redshift
        mu_app = 5.0 * np.log10(C_KM_S * z) + 25.0
        var = dm ** 2

        # Build prior row: mu_host + 5logH0 = 5log10(c*z) + 25
        row = np.zeros(len(q))
        row[mu_idx] = 1.0
        row[h0_idx] = 1.0

        prior_rows_L.append(row)
        prior_rows_y.append(mu_app)
        prior_rows_C.append(var)
        prior_rows_source.append(f"z_prior_{host_name}")
        n_added += 1

    print_status(
        f"Redshift prior stats: added={n_added}, skipped_anchor={n_skipped_anchor}, "
        f"skipped_zrange={n_skipped_zrange}, skipped_noisy={n_skipped_noisy}",
        "INFO",
    )

    if n_added == 0:
        print_status("No redshift priors could be added", "INFO")
        return L, y, C, q, y_source

    # Stack design matrix and data vector
    L_aug = np.vstack([L, np.array(prior_rows_L)])
    y_aug = np.hstack([y, np.array(prior_rows_y)])
    y_source_aug = np.hstack([y_source, np.array(prior_rows_source)])

    # Augment covariance as block-diagonal: [C, 0; 0, C_prior]
    n_new = n_added
    C_prior = np.diag(np.array(prior_rows_C))
    C_aug = np.zeros((L.shape[0] + n_new, L.shape[0] + n_new))
    C_aug[:L.shape[0], :L.shape[0]] = C
    C_aug[L.shape[0]:, L.shape[0]:] = C_prior

    print_status(f"Added {n_added} redshift-distance prior rows", "SUCCESS")
    print_status(f"Augmented matrix shape: {L_aug.shape}", "INFO")

    return L_aug, y_aug, C_aug, q, y_source_aug


# ---------------------------------------------------------------------------
# TEP column construction
# ---------------------------------------------------------------------------
def build_tep_columns(L, q, host_sigma, host_screening, sigma_ref, x_mode="centered"):
    """Build TEP regressor columns for Cepheid and SN rows."""
    mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_names = [q[i] for i in mu_indices]
    n_rows = L.shape[0]

    x_cepheid = np.zeros(n_rows)
    x_sn = np.zeros(n_rows)
    anchor_hosts = {"N4258", "LMC", "M31", "MW", "SMC"}

    for i in range(n_rows):
        host = None
        for idx, mu_param in zip(mu_indices, mu_names):
            if abs(L[i, idx]) > 0.01:
                host = mu_param.replace("mu_", "")
                break
        if host is None:
            continue

        X = build_host_x(host, host_sigma, host_screening, sigma_ref, mode=x_mode)

        # Anchor convention: physical screening
        if host in anchor_hosts:
            pass  # keep X as computed

        rc = classify_row(i, L, q)
        if rc == "Cepheid":
            x_cepheid[i] = X
        elif rc in ("SN_calibrator", "SN_Hubble"):
            x_sn[i] = X

    return x_cepheid, x_sn


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_bias_aware_model(L, y, C, q, x_cepheid, x_sn, model_name, X_SCALE=1e6):
    """
    Fit one of the bias-aware TEP models.

    model_name:
        "A_standard"          : kappa_Cep = 0, kappa_SN = 0
        "B_cepheid_tep"       : kappa_Cep free, kappa_SN = 0
        "C_full_tep"          : kappa_Cep free, kappa_SN free
    """
    x_c = x_cepheid * X_SCALE
    x_s = x_sn * X_SCALE

    if model_name == "A_standard":
        L_fit = L.copy()
        q_fit = list(q)

    elif model_name == "B_cepheid_tep":
        L_fit = np.column_stack([L, -x_c])
        q_fit = list(q) + ["kappaCep_6"]

    elif model_name == "C_full_tep":
        L_fit = np.column_stack([L, -x_c, -x_s])
        q_fit = list(q) + ["kappaCep_6", "kappaSN_6"]

    else:
        raise ValueError(f"Unknown model: {model_name}")

    theta, cov, chi2, rank = fit_gls(L_fit, y, C)
    dof = len(y) - len(q_fit)

    # Extract key parameters
    h0_idx = np.where(np.array(q_fit) == "5logH0")[0]
    h0 = 10 ** (theta[h0_idx[0]] / 5) if len(h0_idx) > 0 else np.nan

    result = {
        "model": model_name,
        "n_params": len(q_fit),
        "rank": int(rank),
        "chi2": float(chi2),
        "dof": int(dof),
        "chi2_reduced": float(chi2 / dof) if dof > 0 else float('inf'),
        "H0": float(h0),
    }

    # Extract TEP parameters
    if model_name in ("B_cepheid_tep", "C_full_tep"):
        k_idx = q_fit.index("kappaCep_6")
        kappa_cep = theta[k_idx] * X_SCALE
        kappa_cep_err = np.sqrt(cov[k_idx, k_idx]) * X_SCALE
        result["kappa_Cep"] = float(kappa_cep)
        result["kappa_Cep_err"] = float(kappa_cep_err)
        result["kappa_Cep_sig"] = float(abs(kappa_cep) / kappa_cep_err if kappa_cep_err > 0 else 0)

    if model_name == "C_full_tep":
        k_idx = q_fit.index("kappaSN_6")
        kappa_sn = theta[k_idx] * X_SCALE
        kappa_sn_err = np.sqrt(cov[k_idx, k_idx]) * X_SCALE
        result["kappa_SN"] = float(kappa_sn)
        result["kappa_SN_err"] = float(kappa_sn_err)
        result["kappa_SN_sig"] = float(abs(kappa_sn) / kappa_sn_err if kappa_sn_err > 0 else 0)

    return result, theta, cov, q_fit, L_fit


# ---------------------------------------------------------------------------
# Residual trend analysis
# ---------------------------------------------------------------------------
def analyze_residual_trend(theta, L, y, C, q, host_sigma, sigma_ref):
    """Compute H0-sigma residual trend after fit."""
    from scipy.stats import pearsonr, spearmanr

    # Predicted y
    y_pred = L @ theta
    residuals = y - y_pred

    # Get host-level mu and H0
    mu_idx_map = {q[i].replace("mu_", ""): i for i in range(len(q)) if q[i].startswith("mu_")}

    hosts = []
    mu_fits = []
    sigmas = []

    for host, idx in mu_idx_map.items():
        if host not in host_sigma or host_sigma[host] <= 0:
            continue
        mu_fit = theta[idx]
        # Check if this host has Cepheid rows (only calibrator hosts matter)
        host_row_mask = np.abs(L[:, idx]) > 0.01
        has_ceph = any(classify_row(int(r), L, q) == "Cepheid" for r in np.where(host_row_mask)[0])
        if not has_ceph:
            continue

        z_val = 0.0
        # Try to get redshift for H0 computation
        host_z_map = {}
        # We don't have host_z here; compute H0 from mu if we had z
        # For now, just compute mu
        hosts.append(host)
        mu_fits.append(mu_fit)
        sigmas.append(host_sigma[host])

    if len(hosts) < 3:
        return {"n_hosts": len(hosts), "r": np.nan, "p_r": np.nan, "rho": np.nan, "p_rho": np.nan}

    # For bias-aware model, the residual trend should be analyzed differently
    # since mu_i are now constrained. Instead, look at Cepheid residuals vs sigma
    ceph_residuals = []
    ceph_sigmas = []
    for i in range(L.shape[0]):
        if classify_row(i, L, q) == "Cepheid":
            for host, idx in mu_idx_map.items():
                if abs(L[i, idx]) > 0.01 and host in host_sigma:
                    ceph_residuals.append(residuals[i])
                    ceph_sigmas.append(host_sigma[host])
                    break

    if len(ceph_residuals) < 10:
        return {"n_hosts": len(hosts), "r": np.nan, "p_r": np.nan, "rho": np.nan, "p_rho": np.nan}

    r, p_r = pearsonr(ceph_sigmas, ceph_residuals)
    rho, p_rho = spearmanr(ceph_sigmas, ceph_residuals)

    return {
        "n_hosts": len(hosts),
        "r": float(r),
        "p_r": float(p_r),
        "rho": float(rho),
        "p_rho": float(p_rho),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _fit_models(L, y, C, q, x_cepheid, x_sn, host_sigma, sigma_ref, branch_label):
    """Fit models A/B/C on a given data branch and return results list."""
    models = ["A_standard", "B_cepheid_tep", "C_full_tep"]
    results = []

    for model_name in models:
        print_status(f"  [{branch_label}] Fitting {model_name}...", "INFO")
        res, theta, cov, q_fit, L_fit = fit_bias_aware_model(
            L, y, C, q, x_cepheid, x_sn, model_name
        )
        res["branch"] = branch_label
        res["description"] = {
            "A_standard": "Standard (no TEP)",
            "B_cepheid_tep": "Cepheid-bias TEP",
            "C_full_tep": "Full TEP (Cepheid + SN)",
        }[model_name]

        trend = analyze_residual_trend(theta, L_fit, y, C, q_fit, host_sigma, sigma_ref)
        res.update(trend)
        results.append(res)
    return results


def run():
    print_status("Step 35: Bias-Aware TEP Ladder", "SECTION")

    # Load data
    L, y, C, q, y_source = load_sh0es_data()
    host_sigma, host_z, host_screening = load_host_metadata()

    # Calculate sigma_ref
    sigma_ref = np.sqrt(
        (30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55)
    )
    print_status(f"Effective sigma_ref: {sigma_ref:.2f} km/s", "INFO")

    # Build TEP columns
    x_cepheid, x_sn = build_tep_columns(
        L, q, host_sigma, host_screening, sigma_ref, x_mode="centered"
    )

    # Count rows with nonzero TEP
    n_ceph_x = np.sum(np.abs(x_cepheid) > 0)
    n_sn_x = np.sum(np.abs(x_sn) > 0)
    print_status(f"Cepheid rows with X != 0: {n_ceph_x}", "INFO")
    print_status(f"SN rows with X != 0: {n_sn_x}", "INFO")

    # ========================================================================
    # BRANCH 1: PRIMARY — Standard SH0ES + TEP (NO redshift priors)
    # ========================================================================
    print_status("BRANCH 1: Primary (no redshift priors)", "SECTION")
    primary_results = _fit_models(
        L, y, C, q, x_cepheid, x_sn, host_sigma, sigma_ref, "primary"
    )

    # ========================================================================
    # BRANCH 2: SENSITIVITY — Redshift priors with sigma_v sweep
    # ========================================================================
    sigma_v_values = [150, 250, 500, 1000]
    sensitivity_results = []

    for sigma_v in sigma_v_values:
        print_status(
            f"BRANCH 2: Redshift prior sensitivity (sigma_v = {sigma_v} km/s)",
            "SECTION",
        )
        L_p, y_p, C_p, q_p, y_src_p = build_redshift_priors(
            L, q, y, C, y_source, host_z, sigma_vpec=sigma_v
        )

        n_prior = len(y_p) - len(y)
        if n_prior == 0:
            print_status("No priors added — skipping this sigma_v", "INFO")
            continue

        x_c_aug = np.hstack([x_cepheid, np.zeros(n_prior)])
        x_s_aug = np.hstack([x_sn, np.zeros(n_prior)])

        branch_label = f"sigma_v_{sigma_v}"
        branch_res = _fit_models(
            L_p, y_p, C_p, q_p, x_c_aug, x_s_aug, host_sigma, sigma_ref, branch_label
        )
        sensitivity_results.extend(branch_res)

    # ========================================================================
    # Combine and save
    # ========================================================================
    all_results = primary_results + sensitivity_results

    # Compute AIC, BIC within each branch relative to Model A
    for res in all_results:
        k = res["n_params"]
        n_data = res["dof"] + k
        res["AIC"] = float(res["chi2"] + 2 * k)
        res["BIC"] = float(res["chi2"] + k * np.log(n_data))

    # Print primary summary
    print_status("Primary Model Comparison (no redshift priors):", "INFO")
    print_status(
        f"{'Model':20s} {'n_par':>5s} {'H0':>6s} {'chi2':>10s} {'AIC':>10s} {'BIC':>10s} "
        f"{'kappaCep':>12s} {'sigCep':>6s} {'kappaSN':>12s} {'sigSN':>6s}",
        "INFO",
    )
    for res in primary_results:
        kc = f"{res.get('kappa_Cep', 0):+.2e}" if res.get('kappa_Cep') is not None else "n/a"
        sc = f"{res.get('kappa_Cep_sig', 0):.1f}" if res.get('kappa_Cep_sig') is not None else "n/a"
        ks = f"{res.get('kappa_SN', 0):+.2e}" if res.get('kappa_SN') is not None else "n/a"
        ss = f"{res.get('kappa_SN_sig', 0):.1f}" if res.get('kappa_SN_sig') is not None else "n/a"
        print_status(
            f"  {res['model']:18s} {res['n_params']:5d} {res['H0']:6.2f} {res['chi2']:10.2f} "
            f"{res['AIC']:10.2f} {res['BIC']:10.2f} {kc:>12s} {sc:>6s} {ks:>12s} {ss:>6s}",
            "INFO",
        )

    # Print sensitivity summary
    print_status("Sensitivity Branch (redshift priors, sigma_v sweep):", "INFO")
    print_status(
        f"{'Branch':15s} {'Model':20s} {'n_par':>5s} {'H0':>6s} {'chi2':>10s} {'AIC':>10s} {'BIC':>10s} "
        f"{'kappaCep':>12s} {'sigCep':>6s}",
        "INFO",
    )
    for res in sensitivity_results:
        if res["model"] != "A_standard":
            continue  # only show baseline for sensitivity summary
        kc = f"{res.get('kappa_Cep', 0):+.2e}" if res.get('kappa_Cep') is not None else "n/a"
        sc = f"{res.get('kappa_Cep_sig', 0):.1f}" if res.get('kappa_Cep_sig') is not None else "n/a"
        print_status(
            f"  {res['branch']:13s} {res['model']:18s} {res['n_params']:5d} {res['H0']:6.2f} "
            f"{res['chi2']:10.2f} {res['AIC']:10.2f} {res['BIC']:10.2f} {kc:>12s} {sc:>6s}",
            "INFO",
        )

    # Save results
    df = pd.DataFrame(all_results)
    out_path = OUT_DIR / "step_35_bias_aware_tep_ladder.csv"
    df.to_csv(out_path, index=False)
    print_status(f"Saved results to {out_path}", "SUCCESS")

    # Save JSON
    import json
    json_path = OUT_DIR / "step_35_bias_aware_tep_ladder.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {json_path}", "SUCCESS")

    return all_results


if __name__ == "__main__":
    run()
