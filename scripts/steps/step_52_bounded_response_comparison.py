#!/usr/bin/env python3
"""
step_52_bounded_response_comparison.py

Bounded-Response Model Comparison and Within-Host Disentanglement

Motivation
----------
The canonical TEP-H0 environmental regressor is the potential contrast

    X_i = (S_i sigma_i^2 - sigma_ref^2) / c^2  ~ 1.8e-7

which is the linearised (small-X) endpoint of the clock-amplitude response.
A fitted slope against this coordinate returns coefficients of order 1e6
simply because the coordinate is of order 1e-7; the coefficient is a
coordinate-projection factor, not a fundamental coupling.  TEP-0 (Paper 0)
predicts a bounded environmental response: the shear-sector transfer
S_Sigma(E) saturates in the screened and unscreened limits, so a response
that is linear in X out to arbitrarily deep potentials is not the
theory-consistent functional form.

This step therefore performs a controlled model comparison of response
functionals g_i(sigma, S) in the velocity-space likelihood

    cz_i = d_i (H_app + a * g_i) + v_i ,      Var_i = sigma_v^2
           + ((ln10/5) cz_i sigma_mu_i)^2 + sigma_int^2 ,

identical to step_39, against:

  linear_X        canonical screened potential contrast (reference model)
  unscreened_X    (sigma^2 - sigma_ref^2)/c^2
  log_sigma       log10(S sigma^2 / sigma_ref^2)   (v0.2 bounded form)
  sigma_grad      (S sigma - sigma_ref)/c          (velocity-gradient proxy)
  screening_S     S_i                              (shear-transmission proxy;
                                                    independent of sigma)
  step_sigma      Theta(sigma - sigma_c)           [sigma_c profiled]
  tanh_sigma      tanh((sigma - sigma_t)/w)        [sigma_t, w profiled;
                                                    phenomenological proxy
                                                    for S_Sigma saturation]

Models are compared on equal footing by AIC/BIC (scanned shape parameters
carry an explicit penalty) and, decisively, by leave-one-host-out
predictive mean-squared error in which the profiled shape parameters are
re-selected on the training subset inside every fold.  A permutation test
and bootstrap are reported for the best-performing model with its shape
frozen, and the leave-one-host-out sign stability is retained.

Part B performs the corresponding disentanglement at ladder level: the
within-host period-coupled term kappa_P (X * (logP - 1) on Cepheid rows)
is fitted jointly with the metallicity-coupled term kappa_Z
(X * [O/H] on Cepheid rows) — a combination absent from the step_34
model grid — so that the ~3 sigma environment-period structure can be
separated from a conventional period-metallicity systematic.  Both anchor
conventions are reported.

Part C contains observable-level injection-recovery validation (the
period-coupled term is injected into the data vector at design-matrix
level, correcting the latent-modulus mismatch identified in step_43) and
a demonstration that a host-uniform modulus shift — which the free mu_i
parameters absorb exactly — cannot generate a spurious kappa_P.

Consistency with TEP-0: the two-metric structure, the universal coupling
beta_A = -1, and the screening conventions are unchanged.  The bounded
functionals tested here are phenomenological proxies for the shear-sector
response S_Sigma(E); their status relative to a derived functional form
is recorded explicitly in the outputs.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from core.constants import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY
from scripts.utils.sample_selection import hubble_flow_mask, Z_CUT
from scripts.utils.tep_correction import build_host_screening_map
from scripts.steps.step_39_environment_slope_decomposition import (
    C_KM_S,
    LN10_OVER_5,
    print_status,
    load_sh0es_data,
    load_host_metadata,
    compute_host_covariates,
    fit_gamma,
    fit_multivariate_gamma,
    permutation_test,
    bootstrap_gamma,
    loho_gamma,
    center_scale,
)

DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KAPPA_CANONICAL = KAPPA_GAL

# Profiled-shape scan grids
STEP_GRID = np.arange(45.0, 145.0, 5.0)
TANH_CENTER_GRID = np.arange(40.0, 150.0, 10.0)
TANH_WIDTH_GRID = np.array([10.0, 20.0, 40.0, 80.0])


# ---------------------------------------------------------------------------
# Regressor construction
# ---------------------------------------------------------------------------
def build_regressors(sigma, S, sigma_ref):
    """Return dict of fixed-shape regressors g_i = f(sigma_i, S_i).

    All are evaluated per host; centering is applied later per dataset so
    that the amplitude is orthogonal to H_app by construction.
    """
    sigma = np.asarray(sigma, dtype=float)
    S = np.asarray(S, dtype=float)
    U = S * sigma ** 2                      # screened endpoint density
    U_ref = sigma_ref ** 2

    return {
        "linear_X": (U - U_ref) / (C_KM_S ** 2),
        "unscreened_X": (sigma ** 2 - U_ref) / (C_KM_S ** 2),
        "log_sigma": np.log10(U / U_ref),
        "sigma_grad": (S * sigma - sigma_ref) / C_KM_S,
        "screening_S": S.copy(),
    }


def build_step(sigma, sigma_c):
    return (np.asarray(sigma) > sigma_c).astype(float)


def build_tanh(sigma, sigma_t, w):
    return np.tanh((np.asarray(sigma) - sigma_t) / w)


# ---------------------------------------------------------------------------
# Generic velocity-space fit and CV
# ---------------------------------------------------------------------------
def fit_model(cz, d, g, sigma_mu, sigma_v, compute_profile=False):
    """Thin wrapper around step_39's fit_gamma with an arbitrary regressor.

    fit_gamma solves cz = d (H_app + Gamma * g); with g of order unity the
    returned Gamma is the amplitude in km/s/Mpc.  The internal GAMMA_SCALE
    bookkeeping does not affect the physical result.
    """
    return fit_gamma(cz, d, g, sigma_mu, sigma_v,
                     compute_profile=compute_profile,
                     compute_uncertainty=True)


def fit_amplitude_only(cz, d, g, sigma_mu, sigma_v):
    """Fast amplitude fit used inside CV/scans (no Hessian, no profile)."""
    return fit_gamma(cz, d, g, sigma_mu, sigma_v,
                     compute_uncertainty=False, compute_profile=False)


def scan_step(cz, d, sigma, sigma_mu, sigma_v, mask=None):
    """Profile the step coordinate over sigma_c; return best fit + table."""
    best = None
    table = []
    for sc in STEP_GRID:
        g = center_scale(build_step(sigma, sc))
        if np.all(g == g[0]):  # degenerate constant column
            continue
        r = fit_amplitude_only(cz, d, g, sigma_mu, sigma_v)
        table.append({"sigma_c": float(sc), "neg2logL": -2.0 * r["logL"],
                      "amplitude": r["Gamma_X"]})
        if best is None or r["logL"] > best["logL"]:
            best = dict(r, sigma_c=float(sc))
    return best, table


def scan_tanh(cz, d, sigma, sigma_mu, sigma_v):
    """Profile tanh over (sigma_t, w); return best fit + table."""
    best = None
    table = []
    for st in TANH_CENTER_GRID:
        for w in TANH_WIDTH_GRID:
            g = center_scale(build_tanh(sigma, st, w))
            if np.all(g == g[0]):
                continue
            r = fit_amplitude_only(cz, d, g, sigma_mu, sigma_v)
            table.append({"sigma_t": float(st), "w": float(w),
                          "neg2logL": -2.0 * r["logL"],
                          "amplitude": r["Gamma_X"]})
            if best is None or r["logL"] > best["logL"]:
                best = dict(r, sigma_t=float(st), w=float(w))
    return best, table


def loo_cv(cz, d, sigma, S, sigma_ref, sigma_mu, sigma_v, host):
    """Leave-one-host-out predictive MSE for every model.

    For profiled-shape models the shape parameter is re-selected on the
    training subset within each fold, so the CV error includes the
    shape-selection variance.
    """
    n = len(cz)
    regs = build_regressors(sigma, S, sigma_ref)
    models = list(regs.keys()) + ["step_sigma", "tanh_sigma"]
    sse = {m: [] for m in models}

    for i in range(n):
        tr = np.arange(n) != i
        # fixed-shape models
        for name, g_all in regs.items():
            g_tr = g_all[tr] - np.mean(g_all[tr])
            g_te = g_all[i] - np.mean(g_all[tr])
            r = fit_amplitude_only(cz[tr], d[tr], g_tr, sigma_mu[tr], sigma_v)
            pred = d[i] * (r["H_app"] + r["Gamma_X"] * g_te)
            sse[name].append((cz[i] - pred) ** 2)
        # step (re-scan sigma_c on the training fold)
        b, _ = scan_step(cz[tr], d[tr], sigma[tr], sigma_mu[tr], sigma_v)
        g_te = float(sigma[i] > b["sigma_c"]) - np.mean(
            build_step(sigma[tr], b["sigma_c"]))
        pred = d[i] * (b["H_app"] + b["Gamma_X"] * g_te)
        sse["step_sigma"].append((cz[i] - pred) ** 2)
        # tanh (re-scan sigma_t, w on the training fold)
        b, _ = scan_tanh(cz[tr], d[tr], sigma[tr], sigma_mu[tr], sigma_v)
        g_te = np.tanh((sigma[i] - b["sigma_t"]) / b["w"]) - np.mean(
            build_tanh(sigma[tr], b["sigma_t"], b["w"]))
        pred = d[i] * (b["H_app"] + b["Gamma_X"] * g_te)
        sse["tanh_sigma"].append((cz[i] - pred) ** 2)

    return {m: float(np.mean(v)) for m, v in sse.items()}


def delta_mu_table(model_g_of_sigma, amplitude, H_app, sigmas_eval):
    """Predicted distance-modulus bias per unit coordinate.

    cz = d (H_app + a g)  <=>  Delta_mu = (5/ln10) a g / H_app.
    Reports the predicted response at fixed sigma for the unscreened
    case (S = 1) so that extrapolation behaviour is directly comparable.
    """
    return {
        float(s): float((5.0 / np.log(10)) * amplitude
                        * model_g_of_sigma(s) / H_app)
        for s in sigmas_eval
    }


# ---------------------------------------------------------------------------
# Part B helpers (ladder-level kappa_P x kappa_Z)
# ---------------------------------------------------------------------------
def ladder_disentanglement():
    """Joint period-coupled x metallicity-coupled fit on the full ladder.

    Reuses step_34 machinery (FullLadderLikelihood) so the design matrix,
    covariance, host assignment, and anchor conventions are identical to
    the published analysis.
    """
    from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood

    fl = FullLadderLikelihood()
    L, y, C, q, y_source = fl.load_sh0es_data()
    host_sigma, host_screening = fl.load_host_metadata()
    sigma_ref = fl.calculate_effective_sigma_ref()

    b_idx = np.where(q == "bW")[0][0]
    z_idx = np.where(q == "ZW")[0][0]
    period_term = L[:, b_idx]
    z_term = L[:, z_idx]
    X_SCALE = 1e6

    results = []
    inj = {}
    n_rows = len(y)

    for anchor_conv in ["anchor_screened_physical", "anchor_reference_zero"]:
        x_c, x_s, _, _ = fl.build_tep_columns(
            L, q, host_sigma, host_screening, sigma_ref,
            x_mode="centered", anchor_convention=anchor_conv,
            y_source=y_source,
        )
        xc = x_c * X_SCALE
        col0 = -xc                       # host-constant Cepheid offset
        colP = -xc * period_term         # period-coupled
        colZ = -xc * z_term              # metallicity-coupled

        theta_base, cov_base, chi2_base, _, _ = fl.fit_gls(L, y, C)
        h0_idx = np.where(q == "5logH0")[0][0]

        def fit(tag, cols, names):
            L_aug = np.column_stack([L] + cols) if cols else L
            q_aug = list(q) + names
            theta, cov, chi2, rank, _ = fl.fit_gls(L_aug, y, C)
            k = len(q_aug)
            row = {
                "anchor_convention": anchor_conv,
                "model": tag,
                "n_params": k,
                "chi2": float(chi2),
                "delta_chi2": float(chi2_base - chi2),
                "AIC": float(chi2 + 2 * k),
                "BIC": float(chi2 + k * np.log(n_rows)),
                "H0": float(10 ** (theta[h0_idx] / 5)),
                "cond_number": float(
                    np.linalg.cond(L_aug) if np.isfinite(L_aug).all() else np.inf),
                "rank": int(rank),
            }
            for j, nm in enumerate(names):
                jdx = len(q) + j
                row[nm] = float(theta[jdx] * X_SCALE)
                err = float(np.sqrt(cov[jdx, jdx]) * X_SCALE) if cov[jdx, jdx] > 0 else np.nan
                row[nm + "_err"] = err
                row[nm + "_sig"] = float(abs(theta[jdx]) / np.sqrt(cov[jdx, jdx])) \
                    if cov[jdx, jdx] > 0 else np.nan
            # kappaP-kappaZ parameter correlation when both present
            if "kappaP_6" in names and "kappaZ_6" in names:
                iP = len(q) + names.index("kappaP_6")
                iZ = len(q) + names.index("kappaZ_6")
                denom = np.sqrt(cov[iP, iP] * cov[iZ, iZ])
                row["corr_kappaP_kappaZ"] = float(cov[iP, iZ] / denom) \
                    if denom > 0 else np.nan
            return row, theta, cov

        combos = [
            ("baseline", [], []),
            ("kappa0_only", [col0], ["kappa0_6"]),
            ("kappaP_only", [colP], ["kappaP_6"]),
            ("kappaZ_only", [colZ], ["kappaZ_6"]),
            ("kappa0_plus_kappaP", [col0, colP], ["kappa0_6", "kappaP_6"]),
            ("kappaP_plus_kappaZ", [colP, colZ], ["kappaP_6", "kappaZ_6"]),
            ("kappa0_kappaP_kappaZ", [col0, colP, colZ],
             ["kappa0_6", "kappaP_6", "kappaZ_6"]),
        ]
        for tag, cols, names in combos:
            row, _, _ = fit(tag, cols, names)
            results.append(row)

        # --- Observable-level injection-recovery for kappa_P (only on the
        # primary anchor convention to keep the table compact) -------------
        if anchor_conv == "anchor_screened_physical":
            rng = np.random.default_rng(7)
            noise = np.sqrt(np.diag(C))
            for k_true in (KAPPA_CANONICAL, -KAPPA_CANONICAL, 3.0e5):
                y_mock = L @ theta_base + (k_true / X_SCALE) * colP
                y_mock = y_mock + rng.normal(0, 0.01 * noise)
                L_aug = np.column_stack([L, colP])
                th, cv, _, _, _ = fl.fit_gls(L_aug, y_mock, C)
                rec = th[-1] * X_SCALE
                err = np.sqrt(cv[-1, -1]) * X_SCALE
                inj[f"kappaP_inj_{k_true:+.2e}"] = {
                    "injected": float(k_true),
                    "recovered": float(rec),
                    "err": float(err),
                    "recovery_fraction": float(rec / k_true) if k_true else np.nan,
                }

            # Degeneracy check: a host-uniform modulus shift must NOT
            # produce a kappa_P.  Inject delta_mu_i = k_host * X_i on every
            # row of each host (Cepheid and SN-calibrator rows alike), then
            # fit the period-coupled model.
            mu_idx = [i for i, p in enumerate(q) if p.startswith("mu_")]
            x_host = {}
            for i in range(n_rows):
                host = None
                for idx in mu_idx:
                    if abs(L[i, idx]) > 0.01:
                        host = q[idx].replace("mu_", "")
                        break
                if host is not None and host not in x_host:
                    x_host[host] = fl.build_host_x(
                        host, host_sigma, host_screening, sigma_ref,
                        mode="centered")
            uniform_col = np.zeros(n_rows)
            for i in range(n_rows):
                for idx in mu_idx:
                    if abs(L[i, idx]) > 0.01:
                        host = q[idx].replace("mu_", "")
                        uniform_col[i] = x_host.get(host, 0.0)
                        break
            k_host = KAPPA_CANONICAL
            y_mock = L @ theta_base - (k_host / X_SCALE) * (uniform_col * X_SCALE)
            y_mock = y_mock + rng.normal(0, 0.01 * noise)
            L_aug = np.column_stack([L, colP])
            th, cv, _, _, _ = fl.fit_gls(L_aug, y_mock, C)
            inj["uniform_modulus_injection"] = {
                "injected_uniform_kappa": float(k_host),
                "kappaP_recovered": float(th[-1] * X_SCALE),
                "kappaP_err": float(np.sqrt(cv[-1, -1]) * X_SCALE),
                "note": "A host-uniform modulus shift should yield kappa_P ~ 0 "
                        "(free mu_i absorb it); nonzero recovery would "
                        "indicate cross-channel leakage.",
            }

    return pd.DataFrame(results), inj


# ---------------------------------------------------------------------------
# Part C: median-split stratification (v0.2 discovery statistic)
# ---------------------------------------------------------------------------
def stratification_check(df, cz, d, sigma_mu, sigma_v):
    """Median split on sigma; weighted-mean H0 per half.

    Weights use the full per-host cz variance under the null model so the
    comparison is consistent with the likelihood used elsewhere.
    """
    sig = df["sigma"].values
    h0 = cz / d
    var = (sigma_v / d) ** 2 + (LN10_OVER_5 * h0 * sigma_mu) ** 2
    med = np.median(sig)
    out = {"sigma_median": float(med)}
    for tag, m in (("low", sig <= med), ("high", sig > med)):
        w = 1.0 / var[m]
        out[f"H0_{tag}"] = float(np.sum(h0[m] * w) / np.sum(w))
        out[f"H0_{tag}_err"] = float(1.0 / np.sqrt(np.sum(w)))
        out[f"n_{tag}"] = int(m.sum())
        out[f"sigma_range_{tag}"] = [float(sig[m].min()), float(sig[m].max())]
    out["delta_H0"] = out["H0_high"] - out["H0_low"]
    out["delta_H0_err"] = float(
        np.sqrt(out["H0_low_err"] ** 2 + out["H0_high_err"] ** 2))
    out["delta_H0_sig"] = abs(out["delta_H0"]) / out["delta_H0_err"] \
        if out["delta_H0_err"] > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(reuse_ladder_from=None, output_dir=None):
    print_status("Step 52: Bounded-Response Model Comparison", "SECTION")
    out_dir = Path(output_dir) if output_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    reused_ladder = None
    if reuse_ladder_from is not None:
        cache_path = Path(reuse_ladder_from).resolve()
        cache_bytes = cache_path.read_bytes()
        reused_ladder = json.loads(cache_bytes)
        for key in ("kappaP_kappaZ_disentanglement", "injection_tests"):
            if key not in reused_ladder:
                raise ValueError(f"Ladder cache lacks {key}: {cache_path}")
        if not reused_ladder["kappaP_kappaZ_disentanglement"]:
            raise ValueError("Ladder cache contains no fits")

    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_z_cmb, host_S, host_mass = load_host_metadata()
    sigma_ref = np.sqrt(
        (30.0 ** 2 * 0.20 + 24.0 ** 2 * 0.25 + 115.0 ** 2 * 0.55)
        / (0.20 + 0.25 + 0.55)
    )
    df_hosts = compute_host_covariates(
        L, y, C, q, host_sigma, host_z, sigma_ref,
        host_z_cmb=host_z_cmb, host_mass=host_mass)
    df_all = df_hosts[
        (~df_hosts["is_anchor"]) & df_hosts["z_hd"].notna()
        & (df_hosts["z_hd"] > 0)].copy()
    print_status(f"Calibrator host sample: {len(df_all)}", "INFO")

    sig_all = df_all["sigma"].values
    S_all = np.array([host_S.get(h, 1.0) for h in df_all["host"]])
    d_all = 10 ** ((df_all["mu"].values - 25.0) / 5.0)
    muerr_all = df_all["mu_err"].fillna(0.05).values
    cz_hd = C_KM_S * df_all["z_hd"].values
    cz_cmb = C_KM_S * df_all["z_cmb"].fillna(df_all["z_hd"]).values

    # Dataset configurations: (label, cz vector, selection mask, sigma_v)
    hf_mask = hubble_flow_mask(df_all, z_cut=Z_CUT).values
    hf_mask_005 = hubble_flow_mask(df_all, z_cut=0.005).values
    configs = []
    for sv in (150.0, 250.0, 500.0):
        configs.append((f"all_z_hd_sv{int(sv)}", cz_hd,
                        np.ones(len(df_all), bool), sv))
    for sv in (150.0, 250.0):
        configs.append((f"hf_z035_sv{int(sv)}", cz_hd, hf_mask, sv))
    configs.append(("hf_z005_sv250", cz_hd, hf_mask_005, 250.0))
    configs.append(("all_z_cmb_sv250", cz_cmb, np.ones(len(df_all), bool), 250.0))
    configs.append(("hf_z035_cmb_sv250", cz_cmb, hf_mask, 250.0))

    rows = []
    cv_results = {}
    stat_tests = {}
    strat = {}

    for label, cz, mask, sv in configs:
        cz_d = cz[mask]
        d_d = d_all[mask]
        s_d = sig_all[mask]
        S_d = S_all[mask]
        m_d = muerr_all[mask]
        print_status(f"{label}: N={mask.sum()}, sigma_v={sv:.0f}", "SECTION")

        regs = build_regressors(s_d, S_d, sigma_ref)

        # Fixed-shape models
        fitted = {}
        for name, g in regs.items():
            g_c = g - np.mean(g)
            r = fit_model(cz_d, d_d, g_c, m_d, sv)
            fitted[name] = (g_c, r)
            rows.append({
                "dataset": label, "model": name, "shape_params": 0,
                "amplitude": r["Gamma_X"], "amplitude_err": r["Gamma_X_err"],
                "amplitude_sig": r["Gamma_X_sig"],
                "lrt_sig": r["Gamma_X_lrt_sig"],
                "delta_2logL_vs_null": r["delta_2logL_vs_null"],
                "H_app": r["H_app"], "logL": r["logL"],
                "AIC": r["AIC"], "BIC": r["BIC"],
                "sigma_int_v": r["sigma_int_v"],
                "chi2": r["chi2"], "n_hosts": r["n_hosts"],
                "optimizer_status": r["status"],
                "optimizer_message": r["optimizer_message"],
                "uncertainty_method": r["uncertainty_method"],
            })

        # Scanned models (in-sample profile)
        b_step, step_tab = scan_step(cz_d, d_d, s_d, m_d, sv)
        b_tanh, tanh_tab = scan_tanh(cz_d, d_d, s_d, m_d, sv)
        for name, best, k_shape in (("step_sigma", b_step, 1),
                                    ("tanh_sigma", b_tanh, 2)):
            rows.append({
                "dataset": label, "model": name, "shape_params": k_shape,
                "shape_best": {k: v for k, v in best.items()
                               if k in ("sigma_c", "sigma_t", "w")},
                "amplitude": best["Gamma_X"], "amplitude_err": best["Gamma_X_err"],
                "amplitude_sig": best["Gamma_X_sig"],
                "lrt_sig": best["Gamma_X_lrt_sig"],
                "delta_2logL_vs_null": best["delta_2logL_vs_null"],
                "H_app": best["H_app"], "logL": best["logL"],
                "AIC": -2 * best["logL"] + 2 * (3 + k_shape),
                "BIC": -2 * best["logL"] + (3 + k_shape) * np.log(mask.sum()),
                "sigma_int_v": best["sigma_int_v"],
                "chi2": best["chi2"], "n_hosts": best["n_hosts"],
                "optimizer_status": best["status"],
                "optimizer_message": best["optimizer_message"],
                "uncertainty_method": best["uncertainty_method"],
            })
            fitted[name] = (None, best)

        # LOO-CV (all configs)
        print_status(f"  LOO-CV ({label})...", "INFO")
        cv = loo_cv(cz_d, d_d, s_d, S_d, sigma_ref, m_d, sv,
                    df_all["host"].values[mask])
        cv_results[label] = cv
        for m, v in cv.items():
            for r in rows:
                if r["dataset"] == label and r["model"] == m:
                    r["cv_mse"] = v
        order = sorted(cv.items(), key=lambda kv: kv[1])
        print_status("  CV MSE ranking: " + ", ".join(
            f"{m}={v:.0f}" for m, v in order), "INFO")

        # Permutation/bootstrap/LOHO on best model (frozen shape) for the
        # two headline configs
        if label in ("all_z_hd_sv250", "hf_z035_sv250"):
            best_name = order[0][0]
            print_status(f"  Best model by CV: {best_name}", "INFO")
            if best_name == "step_sigma":
                g_best = center_scale(build_step(s_d, b_step["sigma_c"]))
            elif best_name == "tanh_sigma":
                g_best = center_scale(
                    build_tanh(s_d, b_tanh["sigma_t"], b_tanh["w"]))
            else:
                g_best = fitted[best_name][0]
            perm = permutation_test(cz_d, d_d, g_best, m_d, sv, n_perm=2000)
            boot = bootstrap_gamma(cz_d, d_d, g_best, m_d, sv, n_boot=1000,
                                   strata=g_best if best_name == "step_sigma" else None)
            pairs_diagnostic = (bootstrap_gamma(cz_d, d_d, g_best, m_d, sv, n_boot=1000)
                                if best_name == "step_sigma" else None)
            loho = loho_gamma(cz_d, d_d, g_best, m_d, sv,
                              df_all["host"].values[mask])
            stat_tests[label] = {
                "best_model": best_name,
                "test_scope": "Conditional on the selected model and frozen shape; not a global model-search p-value",
                "frozen_shape": {k: v for k, v in
                                 (b_step if best_name == "step_sigma"
                                  else b_tanh if best_name == "tanh_sigma"
                                  else {}).items()
                                 if k in ("sigma_c", "sigma_t", "w")},
                "permutation": perm,
                "bootstrap": boot,
                "unstratified_bootstrap_diagnostic": pairs_diagnostic,
                "loho": loho,
            }
            print_status(
                f"  perm p = {perm['permutation_p']:.4f}; "
                f"bootstrap CI [{boot['Gamma_X_ci_low']:.3e}, "
                f"{boot['Gamma_X_ci_high']:.3e}]; "
                f"LOHO positive {loho['n_positive']}/{loho['N_hosts']}",
                "INFO")

        # Extrapolation table for the canonical bounded candidates
        a_lin = fitted["linear_X"][1]["Gamma_X"]
        H_lin = fitted["linear_X"][1]["H_app"]
        a_log = fitted["log_sigma"][1]["Gamma_X"]
        H_log = fitted["log_sigma"][1]["H_app"]
        sig_eval = [50.0, 90.0, 150.0, 250.0, 400.0]
        strat[label] = {
            "delta_mu_linear_X": delta_mu_table(
                lambda s: (s ** 2 - sigma_ref ** 2) / C_KM_S ** 2
                - np.mean(regs["linear_X"]),
                a_lin, H_lin, sig_eval),
            "delta_mu_log_sigma": delta_mu_table(
                lambda s: np.log10(s ** 2 / sigma_ref ** 2)
                - np.mean(regs["log_sigma"]),
                a_log, H_log, sig_eval),
        }

    df_cmp = pd.DataFrame(rows)
    cmp_csv = out_dir / "step_52_bounded_response_comparison.csv"
    df_cmp.to_csv(cmp_csv, index=False)
    print_status(f"Saved comparison table to {cmp_csv}", "SUCCESS")

    # ------------------------------------------------------------------
    # Stratification cross-check (median split)
    # ------------------------------------------------------------------
    print_status("Median-split stratification (v0.2 statistic)", "SECTION")
    for label, cz, mask, sv in (("all_z_hd_sv250", cz_hd,
                                 np.ones(len(df_all), bool), 250.0),
                                ("hf_z035_sv250", cz_hd, hf_mask, 250.0)):
        strat[label + "__median_split"] = stratification_check(
            df_all[mask], cz[mask], d_all[mask], muerr_all[mask], sv)
        s = strat[label + "__median_split"]
        print_status(
            f"  {label}: low-sigma H0={s['H0_low']:.2f}+/-{s['H0_low_err']:.2f} "
            f"(n={s['n_low']}), high-sigma H0={s['H0_high']:.2f}"
            f"+/-{s['H0_high_err']:.2f} (n={s['n_high']}), "
            f"Delta={s['delta_H0']:+.2f} ({s['delta_H0_sig']:.1f}σ)", "INFO")

    # ------------------------------------------------------------------
    # Part B: kappa_P x kappa_Z disentanglement
    # ------------------------------------------------------------------
    print_status("Ladder-level kappa_P x kappa_Z disentanglement", "SECTION")
    if reused_ladder is None:
        df_ladder, injections = ladder_disentanglement()
        ladder_provenance = {"mode": "recomputed"}
    else:
        df_ladder = pd.DataFrame(reused_ladder["kappaP_kappaZ_disentanglement"])
        injections = reused_ladder["injection_tests"]
        ladder_provenance = {
            "mode": "reused_unchanged",
            "source": str(cache_path),
            "sha256": hashlib.sha256(cache_bytes).hexdigest(),
            "scope": "GLS ladder and its injections do not call fit_gamma; reuse is appropriate when only the host optimizer changed and ladder inputs are unchanged",
        }
        print_status("Reused independent GLS ladder results; no full-ladder rerun.", "INFO")
    lad_csv = out_dir / "step_52_kappaP_kappaZ_disentanglement.csv"
    df_ladder.to_csv(lad_csv, index=False)
    print_status(f"Saved disentanglement table to {lad_csv}", "SUCCESS")
    for _, r in df_ladder.iterrows():
        if r["model"] in ("kappaP_plus_kappaZ", "kappa0_kappaP_kappaZ"):
            print_status(
                f"  [{r['anchor_convention']}] {r['model']}: "
                f"kappaP={r.get('kappaP_6', np.nan):.2e} "
                f"({r.get('kappaP_6_sig', np.nan):.1f}σ), "
                f"kappaZ={r.get('kappaZ_6', np.nan):.2e} "
                f"({r.get('kappaZ_6_sig', np.nan):.1f}σ), "
                f"corr={r.get('corr_kappaP_kappaZ', np.nan):.2f}, "
                f"dchi2={r['delta_chi2']:.1f}", "INFO")

    # ------------------------------------------------------------------
    # Save master JSON
    # ------------------------------------------------------------------
    out = {
        "description": __doc__.strip().split("\n")[0],
        "model_comparison": rows,
        "loo_cv": cv_results,
        "statistical_tests": stat_tests,
        "extrapolation": strat,
        "kappaP_kappaZ_disentanglement": df_ladder.to_dict("records"),
        "injection_tests": injections,
        "ladder_provenance": ladder_provenance,
        "host_optimizer_source_sha256": hashlib.sha256(
            (BASE_DIR / "scripts/steps/step_39_environment_slope_decomposition.py").read_bytes()).hexdigest(),
        "interpretation": {
            "coordinate_note": (
                "large coefficients on the linear_X channel are a "
                "coordinate-projection artefact (X ~ 1e-7); the physically "
                "meaningful quantity is the predicted Delta_mu response."),
            "bounded_form_status": (
                "log_sigma, step_sigma and tanh_sigma are phenomenological "
                "proxies for a saturating S_Sigma(E) response; none is "
                "yet derived from the fixed scalar action."),
            "screening_S_status": (
                "screening_S uses the group-environment shear-transmission "
                "factor S_i, independent of sigma, as the environmental "
                "coordinate."),
        },
    }
    out_json = out_dir / "step_52_bounded_response_comparison.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating)
                  else str(x))
    print_status(f"Saved master JSON to {out_json}", "SUCCESS")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-ladder-from", type=Path,
                        help="Reuse unchanged GLS ladder/injection fields from a prior step-52 JSON; rerun host comparisons only")
    parser.add_argument("--output-dir", type=Path, help="Write a separate reviewable result set")
    args = parser.parse_args()
    run(reuse_ladder_from=args.reuse_ladder_from, output_dir=args.output_dir)
