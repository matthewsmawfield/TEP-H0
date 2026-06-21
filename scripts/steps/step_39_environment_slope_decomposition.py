#!/usr/bin/env python3
"""
step_39_environment_slope_decomposition.py

Environment Slope Decomposition

Fits the identifiable combined environmental coefficient directly:

    cz_i = d_i (H_app + Gamma_X * X_i) + v_i

For small X, the Step 38 hierarchical model expands to:

    cz ≈ d_obs * H_app + d_obs * [beta_X + (ln10/5) * H_app * kappa_Cep] * X

So the identifiable combination is:

    Gamma_X = beta_X + (ln10/5) * H_app * kappa_Cep

This script reports:
  - Gamma_X and its uncertainty
  - kappa_equiv = Gamma_X / ((ln10/5) * H_app)  — the kappa that would produce this slope
  - Delta_Gamma_canonical = Gamma_X - (ln10/5) * H_app * KAPPA_CANONICAL

Includes permutation test, bootstrap, redshift-cut sensitivity, and LOHO.
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
KAPPA_CANONICAL = 970000.0
KAPPA_PRIOR_MEAN = 960000.0
KAPPA_PRIOR_SIGMA = 400000.0


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
    if sigma is None or sigma <= 0 or sigma_ref <= 0:
        return 0.0
    return S * (sigma**2 - sigma_ref**2) / (C_KM_S ** 2)


def center_scale(v):
    return v - np.mean(v)


# ---------------------------------------------------------------------------
# Data loading (identical to step_38)
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
    hosts = []
    mus = []
    mu_errs = []
    sigmas = []
    zs = []
    is_anchors = []

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

    df = pd.DataFrame({
        "host": hosts,
        "mu": mus,
        "mu_err": mu_errs,
        "sigma": sigmas,
        "z_hd": zs,
        "is_anchor": is_anchors,
    })
    return df


def _build_covariates_for_subset(df_subset, host_S, sigma_ref):
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

    return cz_obs, mu_err, z, d_obs, X_tep_c


# ---------------------------------------------------------------------------
# Gamma_X fitting
# ---------------------------------------------------------------------------
GAMMA_SCALE = 1e7


def fit_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, sigma_int_guess=5.0):
    """Fit cz = d_obs * (H_app + Gamma_X * X) + noise.

    Internally scales Gamma_X by GAMMA_SCALE so optimizer parameters are O(1).
    """
    n = len(cz_obs)

    def neg_logL(params):
        H_app, gamma_param, sigma_int_v = params[0], params[1], max(params[2], 0.01)
        Gamma_X = gamma_param * GAMMA_SCALE
        cz_model = d_obs * (H_app + Gamma_X * X)
        resid = cz_obs - cz_model
        sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
        var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
        var = np.maximum(var, 0.01)
        return 0.5 * np.sum(resid**2 / var + np.log(var))

    H_app_init = np.median(cz_obs / d_obs)
    x0 = np.array([H_app_init, 2.0, sigma_int_guess])
    bounds = [(30.0, 90.0), (-100.0, 100.0), (0.01, 50.0)]

    res = optimize.minimize(neg_logL, x0, method="L-BFGS-B", bounds=bounds)
    # Always scan multiple initial guesses; L-BFGS-B can get stuck on flat likelihood ridges
    for H_init in [65.0, 70.0, 75.0, 80.0]:
        for g_init in [1.0, 2.0, 2.3, 2.5, 3.0, 0.0, -1.0]:
            x0_try = np.array([H_init, g_init, sigma_int_guess])
            res_try = optimize.minimize(neg_logL, x0_try, method="L-BFGS-B", bounds=bounds)
            if res_try.fun < res.fun:
                res = res_try

    H_app, gamma_param, sigma_int_v = res.x[0], res.x[1], res.x[2]
    Gamma_X = gamma_param * GAMMA_SCALE

    # Hessian for uncertainties (in scaled parameter space)
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            hess = optimize.approx_fprime(
                res.x,
                lambda x: optimize.approx_fprime(x, neg_logL, 1e-5),
                1e-5,
            )
            cov = np.linalg.pinv(hess, rcond=1e-12)
            se_scaled = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se_scaled = np.full(3, np.nan)

    # Convert uncertainties back to physical units
    se = se_scaled.copy()
    se[1] *= GAMMA_SCALE

    cz_model = d_obs * (H_app + Gamma_X * X)
    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
    resid = cz_obs - cz_model
    chi2 = float(np.sum(resid**2 / var))
    logL = -res.fun if hasattr(res, 'fun') else np.nan
    dof = n - 3

    # Derived quantities
    kappa_equiv = Gamma_X / (LN10_OVER_5 * H_app) if H_app > 0 else np.nan
    kappa_equiv_err = np.nan
    try:
        dkappa_dG = 1.0 / (LN10_OVER_5 * H_app)
        dkappa_dH = -Gamma_X / (LN10_OVER_5 * H_app**2)
        kappa_equiv_err = np.sqrt(
            (dkappa_dG * se[1])**2 + (dkappa_dH * se[0])**2
            + 2 * dkappa_dG * dkappa_dH * cov[0, 1]
        )
    except Exception:
        pass

    gamma_canonical = LN10_OVER_5 * H_app * KAPPA_CANONICAL
    delta_gamma_canonical = Gamma_X - gamma_canonical

    return {
        "n_hosts": n,
        "n_params": 3,
        "dof": dof,
        "H_app": float(H_app),
        "H_app_err": float(se[0]),
        "Gamma_X": float(Gamma_X),
        "Gamma_X_err": float(se[1]),
        "Gamma_X_sig": float(abs(Gamma_X) / se[1]) if se[1] > 0 else np.nan,
        "sigma_int_v": float(sigma_int_v),
        "sigma_int_v_err": float(se[2]),
        "kappa_equiv": float(kappa_equiv),
        "kappa_equiv_err": float(kappa_equiv_err),
        "kappa_equiv_sig": float(abs(kappa_equiv) / kappa_equiv_err) if kappa_equiv_err and kappa_equiv_err > 0 else np.nan,
        "Delta_Gamma_canonical": float(delta_gamma_canonical),
        "chi2": chi2,
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
        "logL": float(logL) if np.isfinite(logL) else np.nan,
        "AIC": -2 * logL + 6 if np.isfinite(logL) else np.nan,
        "BIC": -2 * logL + 3 * np.log(n) if np.isfinite(logL) else np.nan,
        "status": "converged" if res.success else "fallback",
    }


# ---------------------------------------------------------------------------
# Permutation and bootstrap tests
# ---------------------------------------------------------------------------
def permutation_test(cz_obs, d_obs, X, sigma_mu, sigma_v, n_perm=5000, seed=42):
    """Standard permutation test for Gamma_X significance."""
    rng = np.random.default_rng(seed)
    n = len(cz_obs)
    res_true = fit_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v)
    gamma_true = abs(res_true["Gamma_X"])

    gamma_perm = []
    for _ in range(n_perm):
        X_perm = rng.permutation(X)
        res_perm = fit_gamma(cz_obs, d_obs, X_perm, sigma_mu, sigma_v)
        gamma_perm.append(abs(res_perm["Gamma_X"]))

    gamma_perm = np.array(gamma_perm)
    p_value = float(np.mean(gamma_perm >= gamma_true))
    return {
        "permutation_p": p_value,
        "gamma_true": float(gamma_true),
        "gamma_perm_mean": float(np.mean(gamma_perm)),
        "gamma_perm_std": float(np.std(gamma_perm)),
    }


def bootstrap_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, n_boot=1000, seed=42):
    """Bootstrap for Gamma_X confidence interval and sign stability."""
    rng = np.random.default_rng(seed)
    n = len(cz_obs)
    gammas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        res = fit_gamma(cz_obs[idx], d_obs[idx], X[idx], sigma_mu[idx], sigma_v)
        gammas.append(res["Gamma_X"])

    gammas = np.array(gammas)
    ci_low, ci_high = np.percentile(gammas, [2.5, 97.5])
    return {
        "Gamma_X_boot_mean": float(np.mean(gammas)),
        "Gamma_X_boot_std": float(np.std(gammas)),
        "Gamma_X_ci_low": float(ci_low),
        "Gamma_X_ci_high": float(ci_high),
        "frac_positive": float(np.mean(gammas > 0)),
    }


def loho_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, host_names):
    """Leave-one-host-out for Gamma_X."""
    n = len(cz_obs)
    gammas = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        res = fit_gamma(cz_obs[mask], d_obs[mask], X[mask], sigma_mu[mask], sigma_v)
        gammas.append(res["Gamma_X"])

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


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run():
    print_status("Step 39: Environment Slope Decomposition", "SECTION")

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

    cz_pri, mu_err_pri, z_pri, d_pri, X_pri = _build_covariates_for_subset(
        df_primary, host_S, sigma_ref
    )

    sigma_v_values = [150, 250, 500]
    results = []

    for sigma_v in sigma_v_values:
        print_status(f"sigma_v = {sigma_v} km/s", "SECTION")

        res = fit_gamma(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v)
        res["sigma_v"] = sigma_v
        res["sample"] = "primary"
        res["z_cut"] = 0.0
        results.append(res)

        print_status(
            f"  Gamma_X = {res['Gamma_X']:.3e} +/- {res['Gamma_X_err']:.3e} "
            f"({res['Gamma_X_sig']:.1f}σ)",
            "INFO",
        )
        print_status(
            f"  H_app = {res['H_app']:.2f}, "
            f"kappa_equiv = {res['kappa_equiv']:.3e} +/- {res['kappa_equiv_err']:.3e} "
            f"({res['kappa_equiv_sig']:.1f}σ)",
            "INFO",
        )
        print_status(
            f"  Delta_Gamma_canonical = {res['Delta_Gamma_canonical']:.3e}",
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
            res = fit_gamma(cz_cut, d_cut, X_cut, mu_err_cut, sigma_v)
            res["sigma_v"] = sigma_v
            res["sample"] = "primary"
            res["z_cut"] = z_cut
            results.append(res)

    # ========================================================================
    # Statistical tests (primary, sigma_v=250)
    # ========================================================================
    print_status("Statistical tests (primary, sigma_v=250)", "SECTION")
    sigma_v_dbg = 250

    # Permutation
    print_status("  Permutation test (N=5000)...", "INFO")
    perm = permutation_test(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, n_perm=5000)
    print_status(
        f"    permutation p = {perm['permutation_p']:.4f}, "
        f"gamma_true = {perm['gamma_true']:.3e}",
        "INFO",
    )

    # Bootstrap
    print_status("  Bootstrap (N=1000)...", "INFO")
    boot = bootstrap_gamma(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, n_boot=1000)
    print_status(
        f"    Gamma_X = {boot['Gamma_X_boot_mean']:.3e} +/- {boot['Gamma_X_boot_std']:.3e}, "
        f"95% CI [{boot['Gamma_X_ci_low']:.3e}, {boot['Gamma_X_ci_high']:.3e}], "
        f"frac_positive={boot['frac_positive']:.3f}",
        "INFO",
    )

    # LOHO
    print_status("  LOHO sign stability...", "INFO")
    loho = loho_gamma(cz_pri, d_pri, X_pri, mu_err_pri, sigma_v_dbg, df_primary["host"].values)
    print_status(
        f"    Gamma_X = {loho['Gamma_X_mean']:.3e} +/- {loho['Gamma_X_std']:.3e}, "
        f"positive in {loho['n_positive']}/{loho['N_hosts']}",
        "INFO",
    )

    # ========================================================================
    # Summary table
    # ========================================================================
    print_status("Summary: identifiable environmental slope", "SECTION")
    print(f"{'sample':>10s} {'sig_v':>5s} {'z_cut':>6s} {'N':>3s} "
          f"{'Gamma_X':>14s} {'sig':>5s} {'kappa_equiv':>12s} {'ke_sig':>6s} "
          f"{'dG_canon':>14s}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['sample']:>10s} {r['sigma_v']:>5.0f} {r['z_cut']:>6.4f} {r['n_hosts']:>3d} "
            f"{r['Gamma_X']:>+14.3e} {r['Gamma_X_sig']:>5.1f} "
            f"{r['kappa_equiv']:>+12.3e} {r['kappa_equiv_sig']:>6.1f} "
            f"{r['Delta_Gamma_canonical']:>+14.3e}"
        )

    # ========================================================================
    # Save
    # ========================================================================
    df_out = pd.DataFrame(results)
    out_csv = OUT_DIR / "step_39_environment_slope_decomposition.csv"
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")

    out_json = OUT_DIR / "step_39_environment_slope_decomposition.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved JSON to {out_json}", "SUCCESS")

    # Save tests
    test_results = {
        "permutation": perm,
        "bootstrap": boot,
        "loho": loho,
    }
    test_json = OUT_DIR / "step_39_statistical_tests.json"
    with open(test_json, "w") as f:
        json.dump(test_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved test results to {test_json}", "SUCCESS")

    return results, test_results


if __name__ == "__main__":
    run()
