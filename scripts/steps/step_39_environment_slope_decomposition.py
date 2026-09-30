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
from core.constants import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY
from scripts.utils.tep_correction import build_host_screening_map
DATA_DIR = BASE_DIR / "data"
SH0ES_DIR = DATA_DIR / "raw" / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10) / 5.0
KAPPA_CANONICAL = KAPPA_GAL  # from core.constants
KAPPA_PRIOR_MEAN = KAPPA_GAL  # from core.constants
KAPPA_PRIOR_SIGMA = KAPPA_GAL_UNCERTAINTY  # from core.constants


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
    return (S * sigma**2 - sigma_ref**2) / (C_KM_S ** 2)


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
    screening_map = build_host_screening_map(df, BASE_DIR)
    host_sigma = {}
    host_z = {}
    host_z_cmb = {}
    host_S = {}
    for _, row in df.iterrows():
        name = row["normalized_name"]
        source_id = str(row.get("source_id", "")).strip()
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
        if source_id:
            host_sigma[source_id] = sigma
            host_S[source_id] = float(S)
            if pd.notna(z_hd) and z_hd > 0:
                host_z[source_id] = z_hd
            if pd.notna(z_cmb) and z_cmb > 0:
                host_z_cmb[source_id] = z_cmb
        compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
        if compact.startswith(("N", "U")):
            parts = compact[1:]
            if parts.isdigit():
                padded = compact[0] + parts.zfill(4)
                host_sigma[padded] = sigma
                host_S[padded] = float(S)
                if pd.notna(z_hd) and z_hd > 0:
                    host_z[padded] = z_hd
                if pd.notna(z_cmb) and z_cmb > 0:
                    host_z_cmb[padded] = z_cmb
                unpadded = compact[0] + parts.lstrip("0")
                if unpadded != padded:
                    host_sigma[unpadded] = sigma
                    host_S[unpadded] = float(S)
                    if pd.notna(z_hd) and z_hd > 0:
                        host_z[unpadded] = z_hd
                    if pd.notna(z_cmb) and z_cmb > 0:
                        host_z_cmb[unpadded] = z_cmb
        if compact.startswith("N"):
            ngc_name = "NGC" + compact[1:]
            host_sigma[ngc_name] = sigma
            host_S[ngc_name] = float(S)
            if pd.notna(z_hd) and z_hd > 0:
                host_z[ngc_name] = z_hd
            if pd.notna(z_cmb) and z_cmb > 0:
                host_z_cmb[ngc_name] = z_cmb
    explicit = {"M101": "M 101", "M1337": "Mrk 1337", "N105A": "N105", "N976A": "N976"}
    host_mass = {}
    for _, row in df.iterrows():
        name = row["normalized_name"]
        source_id = str(row.get("source_id", "")).strip()
        m = row.get("host_logmass", np.nan)
        if pd.notna(m):
            host_mass[name] = float(m)
            if source_id:
                host_mass[source_id] = float(m)
            compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
            if compact.startswith(("N", "U")):
                parts = compact[1:]
                if parts.isdigit():
                    padded = compact[0] + parts.zfill(4)
                    host_mass[padded] = float(m)
                    unpadded = compact[0] + parts.lstrip("0")
                    if unpadded != padded:
                        host_mass[unpadded] = float(m)
            if compact.startswith("N"):
                ngc_name = "NGC" + compact[1:]
                host_mass[ngc_name] = float(m)
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_mass and sh0es_name not in host_mass:
            host_mass[sh0es_name] = host_mass[csv_name]
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_sigma and sh0es_name not in host_sigma:
            host_sigma[sh0es_name] = host_sigma[csv_name]
            host_S[sh0es_name] = host_S[csv_name]
            if csv_name in host_z:
                host_z[sh0es_name] = host_z[csv_name]
            if csv_name in host_z_cmb:
                host_z_cmb[sh0es_name] = host_z_cmb[csv_name]
    return host_sigma, host_z, host_z_cmb, host_S, host_mass


def compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref, host_z_cmb=None, host_mass=None):
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
    zs_cmb = []
    masses = []
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
        zs_cmb.append(host_z_cmb.get(host_name, np.nan) if host_z_cmb else np.nan)
        if host_mass is not None:
            masses.append(host_mass.get(host_name, np.nan))
        else:
            masses.append(np.nan)
        is_anchors.append(host_name in anchor_hosts)

    df = pd.DataFrame({
        "host": hosts,
        "mu": mus,
        "mu_err": mu_errs,
        "sigma": sigmas,
        "z_hd": zs,
        "z_cmb": zs_cmb,
        "host_logmass": masses,
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


def velocity_likelihood_hessian(params, cz_obs, d_obs, x_scaled, sigma_mu, sigma_v):
    """Exact curvature of the Gaussian likelihood, including model-dependent variance.

    Differencing log-likelihoods twice at 1e-5 loses precision and can create
    negative variances. Work in the same normalized coordinates as the fit.
    """
    H, gamma, scatter = params
    design = np.column_stack([d_obs, d_obs * x_scaled])
    model = design @ np.asarray([H, gamma])
    residual = cz_obs - model
    a2 = (LN10_OVER_5 * sigma_mu)**2
    variance = sigma_v**2 + a2 * model**2 + scatter**2
    vm = 2 * a2 * model
    vs = 2 * scatter
    first_v = 1 / variance - residual**2 / variance**2
    second_v = -1 / variance**2 + 2 * residual**2 / variance**3
    fmm = (1 / variance + 2 * residual * vm / variance**2
           + a2 * first_v + .5 * vm**2 * second_v)
    fms = residual * vs / variance**2 + .5 * vm * vs * second_v
    fss = first_v + .5 * vs**2 * second_v
    hessian = np.empty((3, 3))
    hessian[:2, :2] = design.T @ (fmm[:, None] * design)
    hessian[:2, 2] = design.T @ fms
    hessian[2, :2] = hessian[:2, 2]
    hessian[2, 2] = np.sum(fss)
    return hessian


def fit_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, sigma_int_guess=5.0,
              compute_profile=False, multi_start=False,
              compute_uncertainty=True):
    """Fit cz = d_obs * (H_app + Gamma_X * X) + noise.

    Normalize the actual regressor for numerical conditioning. The canonical
    potential contrast is O(1e-7), whereas callers also supply O(1) response
    functions. A fixed 1e7 multiplier makes those fits ill-conditioned.
    Returned coefficients and their covariance remain in the caller's units;
    the existing physical Gamma_X bounds are preserved.
    """
    n = len(cz_obs)
    X = np.asarray(X, dtype=float)
    x_norm = float(np.max(np.abs(X)))
    gamma_scale = 1.0 / x_norm if x_norm > 0 else GAMMA_SCALE

    def neg_logL(params):
        H_app, gamma_param, sigma_int_v = params[0], params[1], max(params[2], 0.01)
        Gamma_X = gamma_param * gamma_scale
        cz_model = d_obs * (H_app + Gamma_X * X)
        resid = cz_obs - cz_model
        sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
        var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
        var = np.maximum(var, 0.01)
        return 0.5 * np.sum(resid**2 / var + np.log(var))

    ratio = cz_obs / d_obs
    linear_design = np.column_stack([np.ones(n), X * gamma_scale])
    H_app_init, gamma_init = np.linalg.lstsq(linear_design, ratio, rcond=None)[0]
    x0 = np.array([H_app_init, gamma_init, sigma_int_guess])
    gamma_bound = 100.0 * GAMMA_SCALE / gamma_scale
    # sigma_int_v enters the variance only through sigma_v**2 + sigma_int_v**2,
    # so it must be allowed to reach the likelihood-implied total scatter
    # (measured ~210 km/s on this sample); a 50 km/s cap made every
    # sigma_v < ~210 variant under-dispersed and inflated its significance.
    bounds = [(30.0, 90.0), (-gamma_bound, gamma_bound), (0.01, 250.0)]

    res = optimize.minimize(neg_logL, x0, method="L-BFGS-B", bounds=bounds)
    if multi_start:
        for H_init in [65.0, 70.0, 75.0, 80.0]:
            for g_init in [1.0, 2.0, 2.5, 3.0, 0.0, -1.0]:
                x0_try = np.array([H_init, g_init, sigma_int_guess])
                res_try = optimize.minimize(neg_logL, x0_try, method="L-BFGS-B", bounds=bounds)
                if res_try.fun < res.fun:
                    res = res_try

    # Multi-start over sigma_int_v: the likelihood is nearly flat in this
    # coordinate (exactly degenerate with sigma_v in the variance), and
    # single-start L-BFGS-B can stall at the initial guess while reporting
    # success.  Keep the best objective across a small scatter ladder.
    for si_init in [50.0, 120.0, 200.0]:
        x0_try = np.array([res.x[0], res.x[1], si_init])
        res_try = optimize.minimize(neg_logL, x0_try, method="L-BFGS-B", bounds=bounds)
        if res_try.fun < res.fun - 1e-9:
            res = res_try

    H_app, gamma_param, sigma_int_v = res.x[0], res.x[1], res.x[2]
    Gamma_X = gamma_param * gamma_scale

    # Hessian for uncertainties (in scaled parameter space)
    cov_physical = np.full((3, 3), np.nan)
    uncertainty_method = "not_computed"
    try:
        if not compute_uncertainty:
            raise RuntimeError("uncertainty calculation disabled")
        hess = velocity_likelihood_hessian(
            res.x, cz_obs, d_obs, X * gamma_scale, sigma_mu, sigma_v)
        # At a constrained optimum, invert curvature only for free parameters.
        # A bound parameter has no symmetric Wald error; leave its SE missing.
        free = np.array([not (np.isclose(value, low, rtol=0, atol=1e-5)
                              or np.isclose(value, high, rtol=0, atol=1e-5))
                         for value, (low, high) in zip(res.x, bounds)])
        active_hess = hess[np.ix_(free, free)]
        np.linalg.cholesky(active_hess)  # never turn negative variance into zero
        cov_scaled = np.full((3, 3), np.nan)
        cov_scaled[np.ix_(free, free)] = np.linalg.inv(active_hess)
        units = np.array([1., gamma_scale, 1.])
        cov_physical = cov_scaled * units[:, None] * units[None, :]
        se = np.sqrt(np.diag(cov_physical))
        uncertainty_method = ("analytic_hessian" if np.all(free)
                              else "analytic_hessian_conditional_on_active_bounds")
    except Exception:
        se = np.full(3, np.nan)
        if compute_uncertainty:
            uncertainty_method = "unavailable_nonpositive_or_singular_curvature"

    cz_model = d_obs * (H_app + Gamma_X * X)
    sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
    var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
    resid = cz_obs - cz_model
    chi2 = float(np.sum(resid**2 / var))
    logL = -res.fun if hasattr(res, 'fun') else np.nan
    dof = n - 3

    # One-parameter nested likelihood-ratio check against Gamma_X = 0.
    def neg_logL_null(params):
        return neg_logL([params[0], 0.0, params[1]])

    if compute_uncertainty or compute_profile:
        null_res = optimize.minimize(
            neg_logL_null,
            [H_app, sigma_int_v],
            method="L-BFGS-B",
            bounds=[bounds[0], bounds[2]],
        )
        delta_2logL = max(0.0, 2.0 * (null_res.fun - res.fun))
        lrt_sig = np.sign(Gamma_X) * np.sqrt(delta_2logL)
    else:
        delta_2logL = np.nan
        lrt_sig = np.nan

    profile_low = np.nan
    profile_high = np.nan
    if compute_profile:
        target = res.fun + 0.5

        def profile_nll(gamma_param_fixed):
            def objective_hs(hs):
                return neg_logL([hs[0], gamma_param_fixed, hs[1]])

            prof = optimize.minimize(
                objective_hs,
                [H_app, sigma_int_v],
                method="L-BFGS-B",
                bounds=[bounds[0], bounds[2]],
            )
            return prof.fun

        def find_profile_limit(direction):
            center = gamma_param
            step = max(0.1, 0.1 * abs(center))
            inner = center
            outer = center + direction * step
            while bounds[1][0] < outer < bounds[1][1] and profile_nll(outer) < target:
                inner = outer
                step *= 1.8
                outer = center + direction * step
            outer = min(max(outer, bounds[1][0]), bounds[1][1])
            if profile_nll(outer) < target:
                return np.nan
            left, right = sorted([inner, outer])
            return optimize.brentq(
                lambda g: profile_nll(g) - target,
                left,
                right,
            )

        try:
            profile_low = find_profile_limit(-1) * gamma_scale
            profile_high = find_profile_limit(+1) * gamma_scale
            profile_err = 0.5 * (profile_high - profile_low)
            if np.isfinite(profile_err) and profile_err > 0:
                se[1] = profile_err
                uncertainty_method += "; Gamma_X_profile_likelihood"
        except (ValueError, RuntimeError):
            pass

    # Derived Cepheid-only equivalent after selecting the profile uncertainty.
    kappa_equiv = Gamma_X / (LN10_OVER_5 * H_app) if H_app > 0 else np.nan
    kappa_equiv_err = np.nan
    try:
        dkappa_dG = 1.0 / (LN10_OVER_5 * H_app)
        dkappa_dH = -Gamma_X / (LN10_OVER_5 * H_app**2)
        kappa_equiv_err = np.sqrt(
            (dkappa_dG * se[1])**2 + (dkappa_dH * se[0])**2
            + 2 * dkappa_dG * dkappa_dH * cov_physical[0, 1]
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
        "Gamma_X_lrt_sig": float(lrt_sig),
        "delta_2logL_vs_null": float(delta_2logL),
        "Gamma_X_profile_low": float(profile_low),
        "Gamma_X_profile_high": float(profile_high),
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
        "optimizer_gamma_scale": float(gamma_scale),
        "optimizer_message": str(res.message),
        "uncertainty_method": uncertainty_method,
    }


# ---------------------------------------------------------------------------
# Permutation and bootstrap tests
# ---------------------------------------------------------------------------
def permutation_test(cz_obs, d_obs, X, sigma_mu, sigma_v, n_perm=5000, seed=42):
    """Standard permutation test for Gamma_X significance."""
    rng = np.random.default_rng(seed)
    n = len(cz_obs)
    res_true = fit_gamma(
        cz_obs, d_obs, X, sigma_mu, sigma_v,
        multi_start=True, compute_uncertainty=False,
    )
    gamma_true = abs(res_true["Gamma_X"])

    gamma_perm = []
    for _ in range(n_perm):
        X_perm = rng.permutation(X)
        res_perm = fit_gamma(
            cz_obs, d_obs, X_perm, sigma_mu, sigma_v,
            compute_uncertainty=False,
        )
        gamma_perm.append(abs(res_perm["Gamma_X"]))

    gamma_perm = np.array(gamma_perm)
    exceedances = int(np.sum(gamma_perm >= gamma_true))
    p_value = float((exceedances + 1) / (n_perm + 1))
    return {
        "permutation_p": p_value,
        "permutation_exceedances": exceedances,
        "n_permutations": int(n_perm),
        "gamma_true": float(gamma_true),
        "gamma_perm_mean": float(np.mean(gamma_perm)),
        "gamma_perm_std": float(np.std(gamma_perm)),
    }


def bootstrap_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, n_boot=1000, seed=42,
                    strata=None):
    """Bootstrap Gamma_X, optionally conditional on fixed observed strata.

    A binary threshold can lose its entire rare group in a pairs bootstrap.
    Then intercept and slope are not separately identifiable: arbitrary
    optimizer slopes must not enter the interval. Report such draws explicitly.
    With strata, preserve each group's observed count; the resulting interval
    is conditional on those counts and the selected response/threshold.
    """
    rng = np.random.default_rng(seed)
    n = len(cz_obs)
    gammas = []
    unidentified = 0
    failed = 0
    groups = None
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != (n,):
            raise ValueError("strata must have one label per host")
        groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    for _ in range(n_boot):
        idx = (rng.integers(0, n, size=n) if groups is None else
               np.concatenate([rng.choice(group, len(group), replace=True)
                               for group in groups]))
        if np.ptp(X[idx]) == 0:
            unidentified += 1
            continue
        res = fit_gamma(
            cz_obs[idx], d_obs[idx], X[idx], sigma_mu[idx], sigma_v,
            compute_uncertainty=False,
        )
        if res["status"] != "converged":
            failed += 1
        gammas.append(res["Gamma_X"])

    gammas = np.array(gammas)
    ci_low, ci_high = np.percentile(gammas, [2.5, 97.5]) if len(gammas) else (np.nan, np.nan)
    return {
        "Gamma_X_boot_mean": float(np.mean(gammas)) if len(gammas) else np.nan,
        "Gamma_X_boot_std": float(np.std(gammas)) if len(gammas) else np.nan,
        "Gamma_X_ci_low": float(ci_low),
        "Gamma_X_ci_high": float(ci_high),
        "frac_positive": float(np.mean(gammas > 0)) if len(gammas) else np.nan,
        "n_attempted": int(n_boot),
        "n_identifiable": int(len(gammas)),
        "n_unidentifiable": int(unidentified),
        "n_optimizer_fallback": int(failed),
        "resampling": "stratified_pairs" if groups is not None else "pairs",
        "group_sizes": [int(len(g)) for g in groups] if groups is not None else None,
        "interval_scope": ("Conditional on observed stratum counts and the selected model/shape" if groups is not None else
                           "Identifiable pairs-bootstrap draws only; inspect n_unidentifiable before interpreting coverage"),
    }


def loho_gamma(cz_obs, d_obs, X, sigma_mu, sigma_v, host_names):
    """Leave-one-host-out for Gamma_X."""
    n = len(cz_obs)
    gammas = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        res = fit_gamma(
            cz_obs[mask], d_obs[mask], X[mask], sigma_mu[mask], sigma_v,
            compute_uncertainty=False,
        )
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


def fit_multivariate_gamma(cz_obs, d_obs, regressor_dict, sigma_mu, sigma_v, sigma_int_guess=5.0):
    """Fit cz = d_obs * (H_app + sum_k c_k * regressor_k) + noise.

    regressor_dict: dict of {name: (array, scale_factor)}
    """
    reg_names = list(regressor_dict.keys())
    reg_arrays = [regressor_dict[k][0] for k in reg_names]
    reg_scales = [regressor_dict[k][1] for k in reg_names]
    k_dim = len(reg_names)

    def neg_logL(params):
        H_app = params[0]
        c_params = params[1:1+k_dim]
        sigma_int_v = max(params[1+k_dim], 0.01)
        cz_model = d_obs * H_app
        for arr, scale, c in zip(reg_arrays, reg_scales, c_params):
            cz_model = cz_model + d_obs * (c * scale * arr)
        resid = cz_obs - cz_model
        sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
        var = sigma_v**2 + sigma_cz_dist**2 + sigma_int_v**2
        var = np.maximum(var, 0.01)
        return 0.5 * np.sum(resid**2 / var + np.log(var))

    if k_dim > 0:
        design = [np.ones(len(cz_obs))] + [arr * scale for arr, scale in zip(reg_arrays, reg_scales)]
        A = np.column_stack(design)
        coeffs = np.linalg.lstsq(A, cz_obs / d_obs, rcond=None)[0]
    else:
        coeffs = [np.mean(cz_obs / d_obs)]
    x0 = np.concatenate([coeffs, [sigma_int_guess]])
    # See fit_gamma: sigma_int_v must reach the likelihood-implied total
    # scatter (~210 km/s); a 50 km/s cap under-disperses low-sigma_v fits.
    bounds = [(30.0, 90.0)] + [(-100.0, 100.0)] * k_dim + [(0.01, 250.0)]

    res = optimize.minimize(neg_logL, x0, method="L-BFGS-B", bounds=bounds)
    for si_init in [50.0, 120.0, 200.0]:
        x0_try = res.x.copy()
        x0_try[-1] = si_init
        res_try = optimize.minimize(neg_logL, x0_try, method="L-BFGS-B", bounds=bounds)
        if res_try.fun < res.fun - 1e-9:
            res = res_try

    cov_physical = np.full((len(x0), len(x0)), np.nan)
    try:
        hess = optimize.approx_fprime(res.x, lambda x: optimize.approx_fprime(x, neg_logL, 1e-5), 1e-5)
        hess = 0.5 * (hess + hess.T)
        # When the intrinsic scatter sits on a bound, the variance direction is
        # singular; report covariance conditional on the boundary value.
        active = np.ones(len(x0), dtype=bool)
        lo, hi = bounds[-1]
        if res.x[-1] <= lo + 1e-3 or res.x[-1] >= hi - 1e-3:
            active[-1] = False
        cov_scaled = np.zeros((len(x0), len(x0)))
        cov_scaled[np.ix_(active, active)] = np.linalg.pinv(
            hess[np.ix_(active, active)], rcond=1e-12
        )
        jac_diag = [1.0] + reg_scales + [1.0]
        jac = np.diag(jac_diag)
        cov_physical = jac @ cov_scaled @ jac
        se = np.sqrt(np.maximum(np.diag(cov_physical), 0))
    except Exception:
        se = np.full(len(x0), np.nan)

    out = {
        "H_app": float(res.x[0]),
        "H_app_err": float(se[0]),
        "logL": float(-res.fun),
        "AIC": float(2 * len(x0) + 2 * res.fun),
        "BIC": float(len(x0) * np.log(len(cz_obs)) + 2 * res.fun),
        "sigma_int_v": float(res.x[-1]),
    }
    for i, name in enumerate(reg_names):
        val = float(res.x[1+i] * reg_scales[i])
        err = float(se[1+i])
        sig = float(val / err) if (err > 0 and np.isfinite(err)) else np.nan
        out[name] = val
        out[f"{name}_err"] = err
        out[f"{name}_sig"] = sig
        out[f"{name}_covH"] = float(cov_physical[0, 1 + i])

    return out, res, neg_logL


def run_host_mass_analysis(cz_pri, d_pri, X_pri, mu_err_pri, df_primary, sigma_ref, host_S=None):
    """Evaluate host-mass decorrelation and joint likelihood marginalization against SN Ia mass step."""
    print_status("Host-Mass Decoupling & Mass-Step Marginalization", "SECTION")

    M = df_primary["host_logmass"].values
    valid_m = np.isfinite(M)
    if valid_m.sum() < 10:
        print_status("Insufficient host mass data for marginalization analysis", "WARNING")
        return {}

    M_c = M - np.mean(M)
    step_10 = (M >= 10.0).astype(float)
    step_10_c = step_10 - np.mean(step_10)

    # 1. Correlations
    r_XM, p_XM = stats.pearsonr(X_pri, M)
    rho_XM, prho_XM = stats.spearmanr(X_pri, M)
    r_Xs, p_Xs = stats.pearsonr(X_pri, step_10)
    rho_Xs, prho_Xs = stats.spearmanr(X_pri, step_10)

    sig_raw = df_primary["sigma"].values
    r_sig_M, p_sig_M = stats.pearsonr(sig_raw, M)
    r_sig2_M, p_sig2_M = stats.pearsonr(sig_raw**2, M)

    print_status(f"Decorrelation check: r(sigma, M*) = {r_sig_M:.3f} (p = {p_sig_M:.2e}), r(sigma^2, M*) = {r_sig2_M:.3f} (p = {p_sig2_M:.2e})", "INFO")
    print_status(f"TEP screened potential: r(X_TEP, M*) = {r_XM:.3f} (p = {p_XM:.4f}), rho(X_TEP, M*) = {rho_XM:.3f} (p = {prho_XM:.4f})", "INFO")
    print_status(f"TEP vs SN Ia Mass Step: r(X_TEP, Step_10) = {r_Xs:.3f} (p = {p_Xs:.4f}), rho(X_TEP, Step_10) = {rho_Xs:.3f} (p = {prho_Xs:.4f})", "INFO")

    # 2. Orthogonalized coordinates
    slope_xm, int_xm, _, _, _ = stats.linregress(M, X_pri)
    X_resid_M = X_pri - (slope_xm * M + int_xm)

    slope_xs, int_xs, _, _, _ = stats.linregress(step_10, X_pri)
    X_resid_step = X_pri - (slope_xs * step_10 + int_xs)

    # 3. Model fittings across velocity dispersions
    sigma_v_values = [150.0, 182.1, 250.0]
    marginalization_results = []

    U_ref = sigma_ref**2
    X_cosmic = -U_ref / (C_KM_S**2)
    # The intercept sits at <X> on the *screened* coordinate that the fit
    # actually uses, X_i = (S_i sigma_i^2 - sigma_ref^2)/c^2. Evaluating the
    # mean with S=1 mixes coordinate axes and overstates the extrapolation
    # distance to the cosmic point.
    host_names = df_primary["host"].values
    if host_S is None:
        host_S = {}
    mean_X_raw = np.mean([
        build_host_x(s, sigma_ref, S=host_S.get(h, 1.0))
        for s, h in zip(sig_raw, host_names)
    ])
    dX_to_cosmic = X_cosmic - mean_X_raw

    for sv in sigma_v_values:
        out_null, res_null, _ = fit_multivariate_gamma(cz_pri, d_pri, {}, mu_err_pri, sv)
        out_tep, res_tep, _ = fit_multivariate_gamma(cz_pri, d_pri, {"Gamma_X": (X_pri, 1e7)}, mu_err_pri, sv)
        out_step, res_step, _ = fit_multivariate_gamma(cz_pri, d_pri, {"gamma_step": (step_10_c, 1.0)}, mu_err_pri, sv)
        out_m, res_m, _ = fit_multivariate_gamma(cz_pri, d_pri, {"gamma_M": (M_c, 1.0)}, mu_err_pri, sv)
        out_j_step, res_j_step, _ = fit_multivariate_gamma(cz_pri, d_pri, {"Gamma_X": (X_pri, 1e7), "gamma_step": (step_10_c, 1.0)}, mu_err_pri, sv)
        out_j_m, res_j_m, _ = fit_multivariate_gamma(cz_pri, d_pri, {"Gamma_X": (X_pri, 1e7), "gamma_M": (M_c, 1.0)}, mu_err_pri, sv)
        out_res_step, res_res_step, _ = fit_multivariate_gamma(cz_pri, d_pri, {"Gamma_X_resid_step": (X_resid_step, 1e7)}, mu_err_pri, sv)
        out_res_m, res_res_m, _ = fit_multivariate_gamma(cz_pri, d_pri, {"Gamma_X_resid_M": (X_resid_M, 1e7)}, mu_err_pri, sv)

        # Likelihood ratio statistics
        lrt_tep_standalone = np.sqrt(max(0.0, 2.0 * (out_tep["logL"] - out_null["logL"])))
        lrt_tep_given_step = np.sqrt(max(0.0, 2.0 * (out_j_step["logL"] - out_step["logL"])))
        lrt_step_given_tep = np.sqrt(max(0.0, 2.0 * (out_j_step["logL"] - out_tep["logL"])))
        lrt_tep_given_m = np.sqrt(max(0.0, 2.0 * (out_j_m["logL"] - out_m["logL"])))
        lrt_m_given_tep = np.sqrt(max(0.0, 2.0 * (out_j_m["logL"] - out_tep["logL"])))
        lrt_res_step = np.sqrt(max(0.0, 2.0 * (out_res_step["logL"] - out_null["logL"])))
        lrt_res_m = np.sqrt(max(0.0, 2.0 * (out_res_m["logL"] - out_null["logL"])))

        # Cosmic H0 projections (with full (H_app, Gamma_X) covariance propagation)
        def hcos_err(out):
            var = (out["H_app_err"] ** 2
                   + dX_to_cosmic ** 2 * out["Gamma_X_err"] ** 2
                   + 2 * dX_to_cosmic * out.get("Gamma_X_covH", 0.0))
            return float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan

        H0_cosmic_tep = out_tep["H_app"] + out_tep["Gamma_X"] * dX_to_cosmic
        H0_cosmic_j_step = out_j_step["H_app"] + out_j_step["Gamma_X"] * dX_to_cosmic
        H0_cosmic_j_m = out_j_m["H_app"] + out_j_m["Gamma_X"] * dX_to_cosmic

        row_entry = {
            "sigma_v": sv,
            "Gamma_X_standalone": out_tep["Gamma_X"],
            "Gamma_X_standalone_err": out_tep["Gamma_X_err"],
            "Gamma_X_standalone_lrt": lrt_tep_standalone,
            "H_app_standalone": out_tep["H_app"],
            "H0_cosmic_standalone": H0_cosmic_tep,
            "H0_cosmic_standalone_err": hcos_err(out_tep),
            "Gamma_X_joint_step": out_j_step["Gamma_X"],
            "Gamma_X_joint_step_err": out_j_step["Gamma_X_err"],
            "Gamma_X_joint_step_lrt": lrt_tep_given_step,
            "gamma_step_joint": out_j_step["gamma_step"],
            "gamma_step_joint_err": out_j_step["gamma_step_err"],
            "gamma_step_joint_lrt": lrt_step_given_tep,
            "H_app_joint_step": out_j_step["H_app"],
            "H0_cosmic_joint_step": H0_cosmic_j_step,
            "H0_cosmic_joint_step_err": hcos_err(out_j_step),
            "retention_vs_step": out_j_step["Gamma_X"] / out_tep["Gamma_X"],
            "Gamma_X_joint_m": out_j_m["Gamma_X"],
            "Gamma_X_joint_m_err": out_j_m["Gamma_X_err"],
            "Gamma_X_joint_m_lrt": lrt_tep_given_m,
            "gamma_m_joint": out_j_m["gamma_M"],
            "gamma_m_joint_err": out_j_m["gamma_M_err"],
            "gamma_m_joint_lrt": lrt_m_given_tep,
            "H_app_joint_m": out_j_m["H_app"],
            "H0_cosmic_joint_m": H0_cosmic_j_m,
            "H0_cosmic_joint_m_err": hcos_err(out_j_m),
            "retention_vs_m": out_j_m["Gamma_X"] / out_tep["Gamma_X"],
            "Gamma_X_resid_step": out_res_step["Gamma_X_resid_step"],
            "Gamma_X_resid_step_err": out_res_step["Gamma_X_resid_step_err"],
            "Gamma_X_resid_step_lrt": lrt_res_step,
            "Gamma_X_resid_m": out_res_m["Gamma_X_resid_M"],
            "Gamma_X_resid_m_err": out_res_m["Gamma_X_resid_M_err"],
            "Gamma_X_resid_m_lrt": lrt_res_m,
            "aic_null": out_null["AIC"],
            "aic_tep": out_tep["AIC"],
            "aic_step": out_step["AIC"],
            "aic_m": out_m["AIC"],
            "aic_joint_step": out_j_step["AIC"],
            "aic_joint_m": out_j_m["AIC"],
        }
        marginalization_results.append(row_entry)

        print_status(f"  sigma_v = {sv:5.1f} km/s: Standalone Gamma_X = {out_tep['Gamma_X']:+.3e} ({lrt_tep_standalone:.2f}σ)", "INFO")
        print_status(f"    Joint w/ Mass Step: Gamma_X = {out_j_step['Gamma_X']:+.3e} ({lrt_tep_given_step:.2f}σ, retention {out_j_step['Gamma_X']/out_tep['Gamma_X']*100:.1f}%), Step = {out_j_step['gamma_step']:+.2f} ({lrt_step_given_tep:.2f}σ)", "INFO")
        print_status(f"    Joint w/ log M*:    Gamma_X = {out_j_m['Gamma_X']:+.3e} ({lrt_tep_given_m:.2f}σ, retention {out_j_m['Gamma_X']/out_tep['Gamma_X']*100:.1f}%), Mass = {out_j_m['gamma_M']:+.2f} ({lrt_m_given_tep:.2f}σ)", "INFO")
        print_status(f"    Mass-Step Resid:    Gamma_X = {out_res_step['Gamma_X_resid_step']:+.3e} ({lrt_res_step:.2f}σ)", "INFO")
        print_status(f"    Continuous Resid:   Gamma_X = {out_res_m['Gamma_X_resid_M']:+.3e} ({lrt_res_m:.2f}σ)", "INFO")

    out_csv = OUT_DIR / "step_39_host_mass_marginalization.csv"
    pd.DataFrame(marginalization_results).to_csv(out_csv, index=False)
    print_status(f"Saved mass marginalization CSV to {out_csv}", "SUCCESS")

    out_json = OUT_DIR / "step_39_host_mass_marginalization.json"
    meta_summary = {
        "decorrelation_metrics": {
            "r_X_logM": float(r_XM),
            "p_X_logM": float(p_XM),
            "rho_X_logM": float(rho_XM),
            "p_rho_X_logM": float(prho_XM),
            "r_X_step10": float(r_Xs),
            "p_X_step10": float(p_Xs),
            "rho_X_step10": float(rho_Xs),
            "p_rho_X_step10": float(prho_Xs),
            "r_sigma_logM": float(r_sig_M),
            "p_sigma_logM": float(p_sig_M),
            "r_sigma2_logM": float(r_sig2_M),
            "p_sigma2_logM": float(p_sig2_M),
        },
        "marginalization_models": marginalization_results,
    }
    with open(out_json, "w") as f:
        json.dump(meta_summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved mass marginalization JSON to {out_json}", "SUCCESS")

    return meta_summary


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run():
    print_status("Step 39: Environment Slope Decomposition", "SECTION")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_z_cmb, host_S, host_mass = load_host_metadata()
    sigma_ref = np.sqrt(
        (30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55) / (0.20 + 0.25 + 0.55)
    )

    df_hosts = compute_host_covariates(L, y, C, q, host_sigma, host_z, sigma_ref, host_z_cmb=host_z_cmb, host_mass=host_mass)
    print_status(f"Computed {len(df_hosts)} calibrator hosts", "INFO")

    df_primary = df_hosts[
        (~df_hosts["is_anchor"])
        & df_hosts["z_hd"].notna()
        & (df_hosts["z_hd"] > 0)
    ].copy()

    print_status(
        f"Primary sample (complete R22 positive-redshift set): {len(df_primary)} hosts",
        "INFO",
    )

    cz_pri, mu_err_pri, z_pri, d_pri, X_pri = _build_covariates_for_subset(
        df_primary, host_S, sigma_ref
    )

    sigma_v_values = [150, 182.1, 250, 500]
    results = []

    for sigma_v in sigma_v_values:
        print_status(f"sigma_v = {sigma_v} km/s", "SECTION")

        res = fit_gamma(
            cz_pri, d_pri, X_pri, mu_err_pri, sigma_v,
            compute_profile=True, multi_start=True,
        )
        res["sigma_v"] = sigma_v
        res["sample"] = "all_r22_hosts"
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
    z_cuts = [0.0035, 0.005, 0.0075]
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
            res["sample"] = "z_hd_cut"
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
    # Host-mass decoupling and marginalization analysis
    # ========================================================================
    mass_results = run_host_mass_analysis(cz_pri, d_pri, X_pri, mu_err_pri, df_primary, sigma_ref, host_S)

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
        "mass_marginalization": mass_results,
    }
    test_json = OUT_DIR / "step_39_statistical_tests.json"
    with open(test_json, "w") as f:
        json.dump(test_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print_status(f"Saved test results to {test_json}", "SUCCESS")

    return results, test_results


if __name__ == "__main__":
    run()
