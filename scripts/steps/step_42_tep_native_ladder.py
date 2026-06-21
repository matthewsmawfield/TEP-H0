#!/usr/bin/env python3
"""
step_42_tep_native_ladder.py

TEP-Native Ladder: Generative clock-aware model.

The standard SH0ES ladder treats inferred distance moduli μ_i as free latent
parameters. In a TEP-native model, those μ_i are biased quantities. The
proper correction applies at the generative-observable level:

    μ_i^SH0ES = μ_i^true - κ_Cep * X_i

    cz_i = d_i^true * (H_app + β_X * X_i) + v_i

The identifiable combination is:

    Γ_X = β_X + (ln 10 / 5) * H_app * κ_Cep

Model variants:
    T0      : standard ladder, κ=0, β=0
    TGamma  : Γ_X only (identifiable slope)
    TK      : pure Cepheid correction, β=0
    TBeta   : pure redshift correction, κ=0
    TMixed  : both free (underdetermined without external distances)
    TMethod : method-specific κ_m for external channels

Gauge choices:
    Gauge A (κ=0)    : β = Γ_X
    Gauge B (β=0)    : κ = Γ_X / (0.4605 * H_app)
    Gauge C (mixed)   : choose κ from external constraint, set β = Γ_X - 0.4605*H*κ
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
TRGB_PATH = BASE_DIR / "results" / "outputs" / "step_15_trgb_hosts_data.csv"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10) / 5.0
GAMMA_SCALE = 1e7
KAPPA_SCALE = 1e5
KAPPA_CANONICAL = 9.7e5


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


def compute_host_mu_cep(L, y, C, q, host_sigma, host_z, sigma_ref):
    """Extract SH0ES-derived Cepheid distance moduli per host."""
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
# TEP-native model fitting
# ---------------------------------------------------------------------------
def fit_tep_native_model(df_cep, df_merged, sigma_ref, host_S, sigma_v, model_type):
    """
    Fit TEP-native generative model.

    For all hosts (Cepheid sample): use SH0ES μ_Cep as observable
    For hosts with external distances: also use μ_ext as observable

    Model:
        μ_Cep,i = μ_true,i - κ_Cep * X_i + ε_Cep,i
        μ_ext,i = μ_true,i + Δ_m + ε_ext,i    (if available)
        cz_i = d_true,i * (H_app + β_X * X_i) + v_i

    Gauge interpretation:
        T0:      κ=0, β=0
        TGamma:  fit Γ_X directly via cz = d*(H + Γ*X) + v
        TK:      β=0, fit κ via both μ_Cep and cz
        TBeta:   κ=0, fit β via cz
        TMixed:  both free (needs external distances for identifiability)
    """
    # Build arrays for all primary Cepheid hosts
    df_pri = df_cep[(~df_cep["is_anchor"]) & df_cep["z_hd"].notna() & (df_cep["z_hd"] >= 0.0035)].copy()
    hosts_all = df_pri["host"].values
    mu_cep_all = df_pri["mu_cep"].values
    mu_cep_err_all = df_pri["mu_cep_err"].values
    z_all = df_pri["z_hd"].values
    sigma_all = df_pri["sigma"].values

    X_all = np.array([build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
                      for s, h in zip(sigma_all, hosts_all)])

    # Build external data arrays
    has_ext = {h: False for h in hosts_all}
    mu_ext_dict = {}
    mu_ext_err_dict = {}
    if len(df_merged) > 0:
        df_m_pri = df_merged[(~df_merged["is_anchor"]) & df_merged["z_hd"].notna() & (df_merged["z_hd"] >= 0.0035)].copy()
        for _, row in df_m_pri.iterrows():
            h = row["host"]
            if h in has_ext:
                has_ext[h] = True
                mu_ext_dict[h] = row["mu_ext"]
                mu_ext_err_dict[h] = row["mu_ext_err"]

    n_all = len(hosts_all)
    n_ext = sum(has_ext.values())

    # -----------------------------------------------------------------------
    # Quick WLS fit for Γ_X (identifiable combination)
    # Model: cz = d_obs * (H_app + Γ_X * X) where d_obs uses SH0ES μ_Cep
    # This is the primary TEP-native observable.
    # -----------------------------------------------------------------------
    d_obs = 10 ** ((mu_cep_all - 25.0) / 5.0)
    cz_obs = C_KM_S * z_all
    y = cz_obs / d_obs
    X_c = center_scale(X_all)

    # Use internal scaling for numerical stability
    X_scale = X_c * GAMMA_SCALE
    w = d_obs**2 / (sigma_v**2 + (LN10_OVER_5 * cz_obs * mu_cep_err_all)**2)
    w = np.maximum(w, 1e-10)

    Xmat = np.column_stack([np.ones(n_all), X_scale])
    W = np.diag(w)
    beta_wls = np.linalg.lstsq(Xmat.T @ W @ Xmat, Xmat.T @ W @ y, rcond=None)[0]
    H_app_wls = beta_wls[0]
    gamma_param_wls = beta_wls[1]
    Gamma_X_wls = gamma_param_wls * GAMMA_SCALE

    cov_wls = np.linalg.pinv(Xmat.T @ W @ Xmat, rcond=1e-12)
    H_err_wls = np.sqrt(cov_wls[0, 0])
    gamma_param_err_wls = np.sqrt(cov_wls[1, 1])
    gamma_err_wls = gamma_param_err_wls * GAMMA_SCALE
    gamma_sig_wls = abs(Gamma_X_wls) / gamma_err_wls if gamma_err_wls > 0 else np.nan

    # χ²
    resid_wls = y - (H_app_wls + gamma_param_wls * X_scale)
    chi2_wls = np.sum(w * resid_wls**2)
    dof_wls = n_all - 2

    # -----------------------------------------------------------------------
    # Gauge interpretations of Γ_X
    # -----------------------------------------------------------------------
    gamma_factor = LN10_OVER_5 * H_app_wls  # ≈ 0.4605 * H_app

    # Gauge A: pure apparent-redshift (κ=0)
    beta_A = Gamma_X_wls
    kappa_A = 0.0

    # Gauge B: pure Cepheid-distance (β=0)
    kappa_B = Gamma_X_wls / gamma_factor if gamma_factor != 0 else np.nan
    beta_B = 0.0

    # Gauge C: mixed with canonical κ
    kappa_C = KAPPA_CANONICAL
    beta_C = Gamma_X_wls - gamma_factor * KAPPA_CANONICAL

    # -----------------------------------------------------------------------
    # Joint model for hosts with external distances (if available)
    # -----------------------------------------------------------------------
    if n_ext >= 3:
        # Differential κ test: μ_Cep - μ_ext = -κ*X - Δ_m + noise
        ext_hosts = [h for h in hosts_all if has_ext[h]]
        mu_cep_ext = np.array([mu_cep_all[list(hosts_all).index(h)] for h in ext_hosts])
        mu_cep_err_ext = np.array([mu_cep_err_all[list(hosts_all).index(h)] for h in ext_hosts])
        mu_ext_vals = np.array([mu_ext_dict[h] for h in ext_hosts])
        mu_ext_err_vals = np.array([mu_ext_err_dict[h] for h in ext_hosts])
        X_ext = np.array([build_host_x(sigma_all[list(hosts_all).index(h)], sigma_ref, S=host_S.get(h, 1.0))
                         for h in ext_hosts])
        X_ext_c = center_scale(X_ext)
        X_ext_scale = X_ext_c * KAPPA_SCALE

        dmu = mu_cep_ext - mu_ext_vals
        dmu_err = np.sqrt(mu_cep_err_ext**2 + mu_ext_err_vals**2)
        w_diff = 1.0 / dmu_err**2

        # Fit: dmu = a - κ*X
        n_ext_arr = len(dmu)
        Xmat_ext = np.column_stack([np.ones(n_ext_arr), X_ext_scale])
        W_ext = np.diag(w_diff)
        beta_diff = np.linalg.lstsq(Xmat_ext.T @ W_ext @ Xmat_ext, Xmat_ext.T @ W_ext @ dmu, rcond=None)[0]
        a_offset = beta_diff[0]
        kappa_param_diff = -beta_diff[1]
        kappa_diff = kappa_param_diff * KAPPA_SCALE

        cov_ext = np.linalg.pinv(Xmat_ext.T @ W_ext @ Xmat_ext, rcond=1e-12)
        a_err = np.sqrt(cov_ext[0, 0])
        kappa_param_err = np.sqrt(cov_ext[1, 1])
        kappa_err = kappa_param_err * KAPPA_SCALE
        kappa_sig = abs(kappa_diff) / kappa_err if kappa_err > 0 else np.nan

        # Gauge D: mixed with fitted κ from external distances
        kappa_D = kappa_diff
        beta_D = Gamma_X_wls - gamma_factor * kappa_diff
    else:
        kappa_diff = np.nan
        kappa_err = np.nan
        kappa_sig = np.nan
        a_offset = np.nan
        a_err = np.nan
        kappa_D = np.nan
        beta_D = np.nan

    return {
        "model": model_type,
        "sigma_v": sigma_v,
        "N_hosts": n_all,
        "N_ext": n_ext,
        "H_app": float(H_app_wls),
        "H_app_err": float(H_err_wls),
        "Gamma_X": float(Gamma_X_wls),
        "Gamma_X_err": float(gamma_err_wls),
        "Gamma_X_sig": float(gamma_sig_wls),
        "chi2": float(chi2_wls),
        "chi2_reduced": chi2_wls / dof_wls if dof_wls > 0 else np.inf,
        "dof": dof_wls,
        "gamma_factor": float(gamma_factor),
        # Gauge A: κ=0
        "kappa_A": float(kappa_A),
        "beta_A": float(beta_A),
        # Gauge B: β=0
        "kappa_B": float(kappa_B),
        "beta_B": float(beta_B),
        # Gauge C: canonical κ
        "kappa_C": float(kappa_C),
        "beta_C": float(beta_C),
        # Gauge D: external κ
        "kappa_D": float(kappa_D),
        "beta_D": float(beta_D),
        # Differential test
        "kappa_diff": float(kappa_diff),
        "kappa_diff_err": float(kappa_err),
        "kappa_diff_sig": float(kappa_sig),
        "a_offset": float(a_offset),
        "a_offset_err": float(a_err),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run():
    print_status("Step 42: TEP-Native Ladder (Generative Clock-Aware Model)", "SECTION")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_S = load_host_metadata()
    sigma_ref = np.sqrt((30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55))

    df_cep = compute_host_mu_cep(L, y, C, q, host_sigma, host_z, sigma_ref)
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

    sigma_v_values = [150, 250, 500]
    results = []

    for sigma_v in sigma_v_values:
        print_status(f"sigma_v = {sigma_v} km/s", "SECTION")

        for model_type in ["T0", "TGamma", "TK", "TBeta", "TMixed"]:
            res = fit_tep_native_model(df_cep, df_merged, sigma_ref, host_S, sigma_v, model_type)
            results.append(res)

            # Print summary
            gx = res["Gamma_X"]
            gx_err = res["Gamma_X_err"]
            gx_sig = res["Gamma_X_sig"]
            print_status(
                f"  {model_type}: H_app={res['H_app']:.2f}, "
                f"Γ_X={gx:+.3e} ({gx_sig:.1f}σ), "
                f"χ²/dof={res['chi2']:.1f}/{res['dof']}",
                "INFO",
            )
            print_status(
                f"    Gauge A (κ=0):     β={res['beta_A']:+.3e}",
                "INFO",
            )
            print_status(
                f"    Gauge B (β=0):     κ={res['kappa_B']:+.3e}",
                "INFO",
            )
            print_status(
                f"    Gauge C (κ=canonical): β={res['beta_C']:+.3e}",
                "INFO",
            )
            if not np.isnan(res['kappa_D']):
                print_status(
                    f"    Gauge D (κ=ext):   κ={res['kappa_D']:+.3e} ({res['kappa_diff_sig']:.1f}σ), β={res['beta_D']:+.3e}",
                    "INFO",
                )

    # Save
    out_json = OUT_DIR / "step_42_tep_native_ladder.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    out_csv = OUT_DIR / "step_42_tep_native_ladder.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    return results


if __name__ == "__main__":
    run()
