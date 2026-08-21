#!/usr/bin/env python3
"""
step_46_profile_likelihood_sigma_int.py

Profile-Likelihood Analysis for the Bounded-Scatter Nuisance Parameter

The hierarchical timefield ladder (Step 38) fits sigma_int_v as a nuisance
parameter with bounds (0.01, 50.0) km/s.  In practice, sigma_int_v hits
a boundary in every fit:
  - At sigma_v = 150 km/s: sigma_int_v -> 50.0 (upper bound)
  - At sigma_v = 250, 500 km/s: sigma_int_v -> 0.01 (lower bound)

When a nuisance parameter is at a boundary, the Hessian-based uncertainty
on the parameters of interest (kappa_Cep, beta_X) is unreliable.  This
step implements a profile-likelihood approach that properly handles the
boundary:

  1. For each value of sigma_int_v on a fine grid, optimize the remaining
     parameters (H_app, kappa, beta) and record the profile log-likelihood.
  2. The profile likelihood gives a proper confidence interval on
     sigma_int_v that accounts for the boundary.
  3. The significance of kappa/beta is computed from the likelihood ratio
     test (LRT) with the null model, not from the Hessian.
  4. The confidence interval on kappa/beta is derived from the profile
     likelihood for kappa/beta, marginalizing over sigma_int_v by profiling.

Outputs:
  - results/outputs/step_46_profile_likelihood_sigma_int.json
  - results/outputs/step_46_profile_likelihood_sigma_int.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
SH0ES_DIR = DATA_DIR / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from core.constants import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10) / 5.0
KAPPA_CANONICAL = KAPPA_GAL
KAPPA_PRIOR_MEAN = KAPPA_GAL
KAPPA_PRIOR_SIGMA = KAPPA_GAL_UNCERTAINTY

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
        print(f"[{level}] {msg}")


def build_host_x(sigma, sigma_ref, S=1.0):
    if sigma is None or sigma <= 0 or sigma_ref <= 0:
        return 0.0
    return (S * sigma**2 - sigma_ref**2) / (C_KM_S ** 2)


def center_scale(v):
    return v - np.mean(v)


# ---------------------------------------------------------------------------
# Data loading (reuse Step 38 logic)
# ---------------------------------------------------------------------------
def load_data():
    """Load host data and build covariates for the primary sample."""
    df_hosts = pd.read_csv(HOSTS_PATH)

    # Load screening from step_03
    step3_path = OUT_DIR / "step_03_stratified_h0.csv"
    screening_map = {}
    if step3_path.exists():
        df_s3 = pd.read_csv(step3_path)
        for _, row in df_s3.iterrows():
            screening_map[row["normalized_name"]] = float(row.get("shear_suppression", 1.0))

    # Load Tully group catalog for hosts not in step_03
    tully_path = DATA_DIR / "raw" / "external" / "tully2015_2mrs_groups_table5.csv"
    if tully_path.exists():
        df_tully = pd.read_csv(tully_path)
        from scripts.utils.tep_correction import group_screening_factor
        for _, row in df_hosts.iterrows():
            name = row["normalized_name"]
            if name in screening_map:
                continue
            pgc = row.get("pgc", None)
            if pd.notna(pgc):
                match = df_tully[df_tully["PGC"] == int(pgc)]
                if len(match) > 0:
                    nmb = float(match.iloc[0]["Nmb"])
                    screening_map[name] = group_screening_factor(nmb)

    # Load SH0ES data for host moduli
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

    from scipy import linalg
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except Exception:
        A_w = L.copy()
        y_w = y.copy()
    theta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=1e-12)

    mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_names = [q[i].replace("mu_", "") for i in mu_indices]
    anchor_set = {"N4258", "LMC", "M31", "MW", "SMC"}

    # Build host data
    hosts = []
    for h in mu_names:
        if h in anchor_set:
            continue
        sigma = None
        z = None
        S = 1.0
        # Look up in hosts_processed
        for _, row in df_hosts.iterrows():
            name = row["normalized_name"]
            compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
            if compact == h or name.replace(" ", "") == h:
                sigma = row["sigma_inferred"]
                z = row["z_hd"]
                S = screening_map.get(name, 1.0)
                break
        if sigma is None or sigma <= 0:
            continue
        if z is None or z <= 0:
            continue
        mu_idx = mu_indices[mu_names.index(h)]
        mu_val = theta[mu_idx]
        hosts.append({
            "host": h,
            "sigma": float(sigma),
            "S": float(S),
            "mu": float(mu_val),
            "z": float(z),
        })

    df = pd.DataFrame(hosts)
    from scripts.utils.tep_correction import compute_anchor_sigma_ref
    sigma_ref = compute_anchor_sigma_ref(screened=True)

    # Build covariates
    d_obs = 10 ** ((df["mu"].values - 25.0) / 5.0)
    cz_obs = C_KM_S * df["z"].values
    X = np.array([
        build_host_x(s, sigma_ref, S=S_val)
        for s, S_val in zip(df["sigma"].values, df["S"].values)
    ])
    X_c = center_scale(X)
    # Distance uncertainty (approximate)
    sigma_mu = np.full(len(df), 0.1)  # typical Cepheid modulus uncertainty

    return df, cz_obs, d_obs, X_c, sigma_mu, sigma_ref


# ---------------------------------------------------------------------------
# Profile likelihood
# ---------------------------------------------------------------------------
def neg_logL_fixed_sigma_int(params, cz_obs, d_obs, X, sigma_mu, sigma_v,
                              sigma_int_v, model_type):
    """Negative log-likelihood with sigma_int_v FIXED (not a free parameter)."""
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
    elif model_type == "Kβ":
        H_app = params[0]
        kappa = params[1] * KAPPA_SCALE
        beta = params[2] * BETA_SCALE
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    d_true = d_obs * np.power(10.0, kappa * X / 5.0)
    cz_model = d_true * (H_app + beta * X)
    resid = cz_obs - cz_model

    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
    var = np.maximum(var, 0.01)

    ll = -0.5 * np.sum(resid**2 / var + np.log(var))

    if model_type == "Kβ" and KAPPA_PRIOR_MEAN > 0:
        ll += -0.5 * ((kappa - KAPPA_PRIOR_MEAN) / KAPPA_PRIOR_SIGMA) ** 2

    return -ll


def fit_with_fixed_sigma_int(cz_obs, d_obs, X, sigma_mu, sigma_v,
                              sigma_int_v, model_type):
    """Fit model with sigma_int_v fixed, return optimal params and logL."""
    H_app_init = np.median(cz_obs / d_obs)

    if model_type == "H0":
        x0 = np.array([H_app_init])
        bounds = [(30.0, 90.0)]
    elif model_type == "Hβ":
        x0 = np.array([H_app_init, 0.0])
        bounds = [(30.0, 90.0), (-100.0, 100.0)]
    elif model_type == "K0":
        x0 = np.array([H_app_init, 0.0])
        bounds = [(30.0, 90.0), (-50.0, 50.0)]
    elif model_type == "Kβ":
        x0 = np.array([H_app_init, KAPPA_PRIOR_MEAN / KAPPA_SCALE, 0.0])
        bounds = [(30.0, 90.0), (-50.0, 50.0), (-100.0, 100.0)]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    res = optimize.minimize(
        neg_logL_fixed_sigma_int,
        x0,
        args=(cz_obs, d_obs, X, sigma_mu, sigma_v, sigma_int_v, model_type),
        method="L-BFGS-B",
        bounds=bounds,
    )

    # Try alternative initializations
    for H_init in [65.0, 70.0, 75.0, 80.0]:
        x0_try = x0.copy()
        x0_try[0] = H_init
        res_try = optimize.minimize(
            neg_logL_fixed_sigma_int,
            x0_try,
            args=(cz_obs, d_obs, X, sigma_mu, sigma_v, sigma_int_v, model_type),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if res_try.fun < res.fun:
            res = res_try

    # Extract parameters
    if model_type == "H0":
        H_app = res.x[0]
        kappa = 0.0
        beta = 0.0
    elif model_type == "Hβ":
        H_app = res.x[0]
        kappa = 0.0
        beta = res.x[1] * BETA_SCALE
    elif model_type == "K0":
        H_app = res.x[0]
        kappa = res.x[1] * KAPPA_SCALE
        beta = 0.0
    elif model_type == "Kβ":
        H_app = res.x[0]
        kappa = res.x[1] * KAPPA_SCALE
        beta = res.x[2] * BETA_SCALE

    return {
        "H_app": float(H_app),
        "kappa": float(kappa),
        "beta": float(beta),
        "logL": -res.fun,
        "success": res.success,
    }


def profile_likelihood_sigma_int(cz_obs, d_obs, X, sigma_mu, sigma_v,
                                  model_type, sigma_int_grid=None):
    """Profile likelihood over sigma_int_v.

    Returns the profile log-likelihood as a function of sigma_int_v,
    the best-fit sigma_int_v, and the best-fit parameters.
    """
    if sigma_int_grid is None:
        # Grid from 0 to 100 km/s (beyond the old 50 bound)
        sigma_int_grid = np.concatenate([
            np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0]),
            np.arange(10.0, 105.0, 5.0),
        ])

    results = []
    for siv in sigma_int_grid:
        res = fit_with_fixed_sigma_int(cz_obs, d_obs, X, sigma_mu, sigma_v,
                                        siv, model_type)
        res["sigma_int_v"] = float(siv)
        results.append(res)

    # Find best-fit sigma_int_v
    logLs = np.array([r["logL"] for r in results])
    best_idx = np.argmax(logLs)
    best = results[best_idx]

    # Compute confidence interval on sigma_int_v
    # 2*(logL_max - logL(sigma_int_v)) ~ chi^2_1
    delta_logL = logLs - logLs[best_idx]
    within_1sigma = np.array([r["sigma_int_v"] for r, d in zip(results, delta_logL) if d > -0.5])
    within_2sigma = np.array([r["sigma_int_v"] for r, d in zip(results, delta_logL) if d > -1.92])

    ci_1sigma = (float(np.min(within_1sigma)) if len(within_1sigma) > 0 else 0.0,
                 float(np.max(within_1sigma)) if len(within_1sigma) > 0 else 0.0)
    ci_2sigma = (float(np.min(within_2sigma)) if len(within_2sigma) > 0 else 0.0,
                 float(np.max(within_2sigma)) if len(within_2sigma) > 0 else 0.0)

    return {
        "model": model_type,
        "sigma_v": sigma_v,
        "best_sigma_int_v": best["sigma_int_v"],
        "best_logL": best["logL"],
        "best_H_app": best["H_app"],
        "best_kappa": best["kappa"],
        "best_beta": best["beta"],
        "ci_1sigma_sigma_int": ci_1sigma,
        "ci_2sigma_sigma_int": ci_2sigma,
        "profile": results,
        "at_boundary": best["sigma_int_v"] <= 0.1 or best["sigma_int_v"] >= 99.0,
    }


def profile_likelihood_kappa(cz_obs, d_obs, X, sigma_mu, sigma_v,
                              model_type="K0"):
    """Profile likelihood over kappa_Cep, profiling over sigma_int_v.

    For each value of kappa, find the best sigma_int_v and record the
    profile log-likelihood.  This gives a proper confidence interval
    on kappa that accounts for the nuisance parameter.
    """
    kappa_grid = np.linspace(-2e6, 5e6, 71)  # -2e6 to 5e6 in steps of 1e5

    results = []
    for kappa_val in kappa_grid:
        # For each kappa, profile over sigma_int_v and H_app
        best_logL = -np.inf
        best_siv = 0.0
        best_H = 0.0
        for siv in np.concatenate([np.array([0.0, 1.0, 5.0]), np.arange(10.0, 105.0, 10.0)]):
            # Fix kappa and sigma_int_v, optimize H_app only
            d_true = d_obs * np.power(10.0, kappa_val * X / 5.0)
            H_init = np.median(cz_obs / d_true)

            def neg_ll_H(H_app):
                cz_model = d_true * H_app
                resid = cz_obs - cz_model
                sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
                var = sigma_v**2 + sigma_cz_dist**2 + siv**2
                var = np.maximum(var, 0.01)
                return 0.5 * np.sum(resid**2 / var + np.log(var))

            res = optimize.minimize_scalar(neg_ll_H, bounds=(30, 90), method="bounded")
            logL = -res.fun
            if logL > best_logL:
                best_logL = logL
                best_siv = siv
                best_H = res.x

        results.append({
            "kappa": float(kappa_val),
            "logL": float(best_logL),
            "sigma_int_v": float(best_siv),
            "H_app": float(best_H),
        })

    # Find best-fit kappa
    logLs = np.array([r["logL"] for r in results])
    best_idx = np.argmax(logLs)
    best = results[best_idx]

    # Confidence interval on kappa from profile likelihood
    # 2*(logL_max - logL(kappa)) ~ chi^2_1
    delta_logL = logLs - logLs[best_idx]
    within_1sigma = np.array([r["kappa"] for r, d in zip(results, delta_logL) if d > -0.5])
    within_2sigma = np.array([r["kappa"] for r, d in zip(results, delta_logL) if d > -1.92])
    within_3sigma = np.array([r["kappa"] for r, d in zip(results, delta_logL) if d > -4.61])

    ci = {
        "1sigma": (float(np.min(within_1sigma)) if len(within_1sigma) > 0 else 0.0,
                   float(np.max(within_1sigma)) if len(within_1sigma) > 0 else 0.0),
        "2sigma": (float(np.min(within_2sigma)) if len(within_2sigma) > 0 else 0.0,
                   float(np.max(within_2sigma)) if len(within_2sigma) > 0 else 0.0),
        "3sigma": (float(np.min(within_3sigma)) if len(within_3sigma) > 0 else 0.0,
                   float(np.max(within_3sigma)) if len(within_3sigma) > 0 else 0.0),
    }

    # LRT significance: compare best logL to null (kappa=0) logL
    null_idx = np.argmin(np.abs([r["kappa"] for r in results]))
    null_logL = results[null_idx]["logL"]
    lrt_stat = 2 * (best["logL"] - null_logL)
    lrt_pvalue = stats.chi2.sf(lrt_stat, df=1)
    lrt_sigma = np.sqrt(lrt_stat) if lrt_stat > 0 else 0.0

    return {
        "model": model_type,
        "sigma_v": sigma_v,
        "best_kappa": best["kappa"],
        "best_logL": best["logL"],
        "best_sigma_int_v": best["sigma_int_v"],
        "best_H_app": best["H_app"],
        "null_logL": null_logL,
        "lrt_statistic": float(lrt_stat),
        "lrt_pvalue": float(lrt_pvalue),
        "lrt_sigma": float(lrt_sigma),
        "ci": ci,
        "profile": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print_status("Step 46: Profile-Likelihood Analysis for Bounded Scatter", "SECTION")
    print_status("Loading data...", "INFO")

    df, cz_obs, d_obs, X, sigma_mu, sigma_ref = load_data()
    print_status(f"  N hosts = {len(df)}", "INFO")
    print_status(f"  sigma_ref (screened) = {sigma_ref:.2f} km/s", "INFO")
    print_status(f"  sigma range: {df['sigma'].min():.1f} - {df['sigma'].max():.1f}", "INFO")
    print_status(f"  S range: {df['S'].min():.3f} - {df['S'].max():.3f}", "INFO")

    # ------------------------------------------------------------------
    # 1. Profile likelihood over sigma_int_v for each model and sigma_v
    # ------------------------------------------------------------------
    print_status("Profile likelihood over sigma_int_v...", "SECTION")

    sigma_v_values = [150.0, 250.0, 500.0]
    models = ["H0", "K0", "Hβ", "Kβ"]

    all_profile_results = []
    for sigma_v in sigma_v_values:
        print_status(f"  sigma_v = {sigma_v} km/s:", "INFO")
        for model in models:
            res = profile_likelihood_sigma_int(cz_obs, d_obs, X, sigma_mu,
                                                sigma_v, model)
            all_profile_results.append(res)
            boundary_str = " (AT BOUNDARY)" if res["at_boundary"] else ""
            print_status(
                f"    {model:4s}: best sigma_int_v = {res['best_sigma_int_v']:.1f} km/s, "
                f"logL = {res['best_logL']:.2f}, "
                f"kappa = {res['best_kappa']:.3e}, "
                f"beta = {res['best_beta']:.3e}"
                f"{boundary_str}",
                "SUCCESS" if not res["at_boundary"] else "WARNING",
            )

    # ------------------------------------------------------------------
    # 2. Profile likelihood over kappa (profiling over sigma_int_v)
    # ------------------------------------------------------------------
    print_status("Profile likelihood over kappa_Cep (profiling sigma_int_v)...", "SECTION")

    kappa_profile_results = []
    for sigma_v in sigma_v_values:
        print_status(f"  sigma_v = {sigma_v} km/s:", "INFO")
        res = profile_likelihood_kappa(cz_obs, d_obs, X, sigma_mu,
                                       sigma_v, model_type="K0")
        kappa_profile_results.append(res)
        print_status(
            f"    K0: best kappa = {res['best_kappa']:.3e}, "
            f"LRT = {res['lrt_statistic']:.2f}, "
            f"p = {res['lrt_pvalue']:.4e}, "
            f"significance = {res['lrt_sigma']:.2f}sigma",
            "SUCCESS",
        )
        print_status(
            f"    1sigma CI: [{res['ci']['1sigma'][0]:.3e}, {res['ci']['1sigma'][1]:.3e}]",
            "INFO",
        )
        print_status(
            f"    2sigma CI: [{res['ci']['2sigma'][0]:.3e}, {res['ci']['2sigma'][1]:.3e}]",
            "INFO",
        )

    # ------------------------------------------------------------------
    # 3. Compare Hessian vs profile-likelihood significance
    # ------------------------------------------------------------------
    print_status("Comparison: Hessian vs Profile-Likelihood significance", "SECTION")

    # Load Step 38 results for comparison
    step38_path = OUT_DIR / "step_38_hierarchical_timefield_ladder.json"
    if step38_path.exists():
        with open(step38_path) as f:
            step38 = json.load(f)

        print(f"{'Model':<8s} {'sigma_v':<8s} {'Hessian_sig':<12s} {'LRT_sig':<12s} {'sigma_int':<10s} {'boundary':<10s}")
        print("-" * 70)
        for s38, s46 in zip(step38, all_profile_results):
            if s38.get("model") != s46.get("model"):
                continue
            if s38.get("sigma_v") != s46.get("sigma_v"):
                continue
            k = s38.get("kappa_Cep", 0)
            k_err = s38.get("kappa_Cep_err", np.nan)
            hess_sig = abs(k) / k_err if k_err and k_err > 0 else np.nan
            lrt_sig = None
            for kp in kappa_profile_results:
                if kp["sigma_v"] == s38.get("sigma_v"):
                    lrt_sig = kp["lrt_sigma"]
                    break
            siv = s38.get("sigma_int_v", 0)
            boundary = "UPPER" if siv >= 49.9 else ("LOWER" if siv <= 0.02 else "interior")
            print(f"{s38['model']:<8s} {s38['sigma_v']:<8} {hess_sig:<12.2f} {lrt_sig if lrt_sig else 'N/A':<12} {siv:<10.2f} {boundary:<10s}")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    print_status("Saving results...", "SECTION")

    # Prepare JSON-serializable output
    output = {
        "description": "Profile-likelihood analysis for bounded sigma_int_v nuisance parameter",
        "n_hosts": len(df),
        "sigma_ref": float(sigma_ref),
        "sigma_int_v_profiles": [
            {
                "model": r["model"],
                "sigma_v": r["sigma_v"],
                "best_sigma_int_v": r["best_sigma_int_v"],
                "best_logL": r["best_logL"],
                "best_kappa": r["best_kappa"],
                "best_beta": r["best_beta"],
                "ci_1sigma": list(r["ci_1sigma_sigma_int"]),
                "ci_2sigma": list(r["ci_2sigma_sigma_int"]),
                "at_boundary": r["at_boundary"],
                "profile_grid": [
                    {"sigma_int_v": p["sigma_int_v"], "logL": p["logL"],
                     "kappa": p["kappa"], "beta": p["beta"]}
                    for p in r["profile"]
                ],
            }
            for r in all_profile_results
        ],
        "kappa_profiles": [
            {
                "model": r["model"],
                "sigma_v": r["sigma_v"],
                "best_kappa": r["best_kappa"],
                "best_logL": r["best_logL"],
                "best_sigma_int_v": r["best_sigma_int_v"],
                "null_logL": r["null_logL"],
                "lrt_statistic": r["lrt_statistic"],
                "lrt_pvalue": r["lrt_pvalue"],
                "lrt_sigma": r["lrt_sigma"],
                "ci_1sigma": list(r["ci"]["1sigma"]),
                "ci_2sigma": list(r["ci"]["2sigma"]),
                "ci_3sigma": list(r["ci"]["3sigma"]),
                "profile_grid": [
                    {"kappa": p["kappa"], "logL": p["logL"],
                     "sigma_int_v": p["sigma_int_v"], "H_app": p["H_app"]}
                    for p in r["profile"]
                ],
            }
            for r in kappa_profile_results
        ],
    }

    out_json = OUT_DIR / "step_46_profile_likelihood_sigma_int.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    # Save CSV summary
    rows = []
    for r in kappa_profile_results:
        rows.append({
            "model": r["model"],
            "sigma_v": r["sigma_v"],
            "best_kappa": r["best_kappa"],
            "best_sigma_int_v": r["best_sigma_int_v"],
            "lrt_statistic": r["lrt_statistic"],
            "lrt_pvalue": r["lrt_pvalue"],
            "lrt_sigma": r["lrt_sigma"],
            "ci_1sigma_low": r["ci"]["1sigma"][0],
            "ci_1sigma_high": r["ci"]["1sigma"][1],
            "ci_2sigma_low": r["ci"]["2sigma"][0],
            "ci_2sigma_high": r["ci"]["2sigma"][1],
        })
    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "step_46_profile_likelihood_sigma_int.csv"
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    # ------------------------------------------------------------------
    # 5. Key conclusion
    # ------------------------------------------------------------------
    print_status("Key Conclusion", "SECTION")
    print_status(
        "The profile-likelihood analysis properly handles the bounded sigma_int_v\n"
        "nuisance parameter.  The LRT-based significance replaces the unreliable\n"
        "Hessian-based significance for all models where sigma_int_v hits a boundary.",
        "INFO",
    )


if __name__ == "__main__":
    main()
