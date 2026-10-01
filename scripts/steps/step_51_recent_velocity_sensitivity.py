#!/usr/bin/env python3
"""Recent Manticore-Local posterior-summary sensitivity test.

This is deliberately separate from the primary Pantheon+/Carrick endpoint.
The Stiskalek et al. table contains joint posterior summaries from a model that
already fits distances, selection, and the velocity field. Regressing those
summaries is useful as a reproducible compatibility diagnostic, but it is not
an independent likelihood or an additional detection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from scripts.steps.step_39_environment_slope_decomposition import (
    C_KM_S,
    GAMMA_SCALE,
    LN10_OVER_5,
    build_host_x,
    compute_host_covariates,
    load_host_metadata,
    load_sh0es_data,
)

INPUT = BASE_DIR / "data" / "raw" / "external" / "stiskalek2026_manticore_hosts.csv"
OUT_JSON = BASE_DIR / "results" / "outputs" / "step_51_recent_velocity_sensitivity.json"
OUT_CSV = BASE_DIR / "results" / "outputs" / "step_51_recent_velocity_sensitivity.csv"
MANTICORE_RESIDUAL_KMS = 172.5
CARRICK_RESIDUAL_KMS = 216.6


def fit_heteroscedastic(df: pd.DataFrame, residual_kms: float) -> dict:
    """Fit cz=d(H+Gamma X) with published per-host velocity errors."""
    d = df["distance_mpc"].to_numpy(float)
    cz = df["cz_cos_kms"].to_numpy(float)
    x = df["X"].to_numpy(float)
    mu_err = df["mu_error_mag"].to_numpy(float)
    vpec_err = df["vpec_error_kms"].to_numpy(float)

    def nll(params):
        h_app, gamma_scaled = params
        model = d * (h_app + gamma_scaled * GAMMA_SCALE * x)
        variance = (
            residual_kms**2
            + vpec_err**2
            + (LN10_OVER_5 * np.abs(model) * mu_err) ** 2
        )
        return 0.5 * np.sum((cz - model) ** 2 / variance + np.log(variance))

    result = optimize.minimize(
        nll,
        [71.0, 0.0],
        method="L-BFGS-B",
        bounds=[(30.0, 90.0), (-100.0, 100.0)],
    )
    hessian = optimize.approx_fprime(
        result.x,
        lambda p: optimize.approx_fprime(p, nll, 1e-5),
        1e-5,
    )
    covariance = np.linalg.pinv(hessian, rcond=1e-12)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    null = optimize.minimize(
        lambda p: nll([p[0], 0.0]),
        [result.x[0]],
        method="L-BFGS-B",
        bounds=[(30.0, 90.0)],
    )
    delta_2logl = max(0.0, 2.0 * (null.fun - result.fun))
    gamma = result.x[1] * GAMMA_SCALE
    gamma_err = errors[1] * GAMMA_SCALE
    return {
        "n_hosts": int(len(df)),
        "H_app": float(result.x[0]),
        "H_app_err": float(errors[0]),
        "Gamma_X": float(gamma),
        "Gamma_X_err": float(gamma_err),
        "Gamma_X_sig": float(abs(gamma) / gamma_err),
        "delta_2logL_vs_null": float(delta_2logl),
        "Gamma_X_lrt_sig": float(np.sign(gamma) * np.sqrt(delta_2logl)),
        "residual_velocity_kms": float(residual_kms),
        "status": "converged" if result.success else "optimizer_warning",
    }


def add_environment(table, host_sigma, host_s, sigma_ref, screened=True):
    out = table.copy()
    out["sigma"] = out["host"].map(host_sigma)
    out["S"] = out["host"].map(host_s)
    if out[["sigma", "S"]].isna().any().any():
        missing = out.loc[out["sigma"].isna() | out["S"].isna(), "host"].tolist()
        raise ValueError(f"Missing host environment metadata: {missing}")
    s_values = out["S"].to_numpy(float) if screened else np.ones(len(out))
    x_raw = np.array(
        [build_host_x(sigma, sigma_ref, S) for sigma, S in zip(out["sigma"], s_values)]
    )
    out["X"] = x_raw - np.mean(x_raw)
    return out


def run():
    table = pd.read_csv(INPUT, comment="#")
    if len(table) != 35 or table["host"].duplicated().any():
        raise ValueError("The pinned Stiskalek v3 table must contain 35 unique hosts")

    host_sigma, host_z, host_z_cmb, host_s, _host_mass = load_host_metadata()
    sigma_ref = np.sqrt(30.0**2 * 0.20 + 24.0**2 * 0.25 + 115.0**2 * 0.55)

    table["distance_mpc"] = 10 ** ((table["mu_mag"] - 25.0) / 5.0)
    table["cz_cos_kms"] = table["cz_cmb_kms"] - table["vpec_manticore_kms"]

    screened = add_environment(table, host_sigma, host_s, sigma_ref, screened=True)
    unscreened = add_environment(table, host_sigma, host_s, sigma_ref, screened=False)
    result_screened = fit_heteroscedastic(screened, MANTICORE_RESIDUAL_KMS)
    result_unscreened = fit_heteroscedastic(unscreened, MANTICORE_RESIDUAL_KMS)

    # Same 35 hosts with the primary SH0ES GLS distances and Pantheon+/Carrick
    # redshifts. This overlap check separates sample composition from velocity
    # field and selection-model changes.
    L, y, C, q = load_sh0es_data()
    r22 = compute_host_covariates(
        L, y, C, q, host_sigma, host_z, sigma_ref, host_z_cmb=host_z_cmb
    )
    overlap = r22.merge(table[["host"]], on="host", how="inner", validate="one_to_one")
    overlap = overlap.rename(columns={"mu": "mu_mag", "mu_err": "mu_error_mag"})
    overlap["distance_mpc"] = 10 ** ((overlap["mu_mag"] - 25.0) / 5.0)
    overlap["cz_cos_kms"] = C_KM_S * overlap["z_hd"]
    overlap["vpec_error_kms"] = 0.0
    overlap = add_environment(overlap, host_sigma, host_s, sigma_ref, screened=True)
    result_carrick_overlap = fit_heteroscedastic(overlap, CARRICK_RESIDUAL_KMS)

    records = []
    for name, result in [
        ("manticore_screened", result_screened),
        ("manticore_unscreened", result_unscreened),
        ("carrick_r22_same_35", result_carrick_overlap),
    ]:
        records.append({"model": name, **result})
    pd.DataFrame(records).to_csv(OUT_CSV, index=False)

    payload = {
        "status": "PASS",
        "source": {
            "citation": "Stiskalek et al. 2025, MNRAS, staf2260",
            "doi": "10.1093/mnras/staf2260",
            "arxiv_version": "2509.09665v3",
            "version_date": "2026-06-10",
            "table": "Table B1",
            "data_kind": "joint posterior summaries",
        },
        "interpretation": (
            "Reproducible posterior-summary sensitivity only; it is not an "
            "independent reanalysis of the unreleased Manticore field samples."
        ),
        "models": {row["model"]: {k: v for k, v in row.items() if k != "model"} for row in records},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    for row in records:
        print(
            f"{row['model']}: N={row['n_hosts']}, "
            f"Gamma_X={row['Gamma_X']:+.3e} +/- {row['Gamma_X_err']:.3e} "
            f"({row['Gamma_X_sig']:.2f} sigma; LRT={row['Gamma_X_lrt_sig']:+.2f})"
        )
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_CSV}")
    return payload


if __name__ == "__main__":
    run()
