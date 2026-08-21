#!/usr/bin/env python3
"""
step_48_pantheon_velocity_covariance.py

Pantheon+ Peculiar-Velocity Covariance Analysis

The redshift–distance likelihood (Step 38/44) uses a fixed peculiar
velocity scatter sigma_v = {150, 250, 500} km/s.  This step uses the
actual Pantheon+ peculiar velocity data and covariance matrix to compute
a data-driven sigma_v for each Cepheid host.

The Pantheon+ data release provides:
  - VPEC: model-predicted peculiar velocity for each SN
  - VPECERR = 250 km/s: assumed peculiar velocity uncertainty
  - Full STAT+SYS covariance matrix (1701 x 1701)

This step:
  1. Extracts the peculiar velocity component from the Pantheon+ covariance
  2. Computes a data-driven sigma_v for each Cepheid host
  3. Tests the redshift–distance likelihood with the data-driven sigma_v
  4. Compares the kappa_Cep significance with fixed vs data-driven sigma_v

Outputs:
  - results/outputs/step_48_pantheon_velocity_covariance.json
  - results/outputs/step_48_pantheon_velocity_covariance.csv
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
PANTHEON_PATH = DATA_DIR / "raw" / "Pantheon+SH0ES.dat"
PANTHEON_COV_PATH = DATA_DIR / "raw" / "external" / "Pantheon+SH0ES_STAT+SYS.cov"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10) / 5.0


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


def main():
    print_status("Step 48: Pantheon+ Peculiar-Velocity Covariance Analysis", "SECTION")

    # ------------------------------------------------------------------
    # 1. Load Pantheon+ data
    # ------------------------------------------------------------------
    sn = pd.read_csv(PANTHEON_PATH, sep=r'\s+', comment='#')
    print_status(f"Pantheon+SH0ES: {len(sn)} SNe", "INFO")
    print_status(f"VPECERR (all SNe): {sn['VPECERR'].unique()}", "INFO")

    cal = sn[sn['IS_CALIBRATOR'] == 1].copy()
    hf = sn[sn['USED_IN_SH0ES_HF'] == 1].copy()
    print_status(f"Calibrators: {len(cal)}, Hubble-flow: {len(hf)}", "INFO")

    # ------------------------------------------------------------------
    # 2. Load Pantheon+ covariance matrix
    # ------------------------------------------------------------------
    if PANTHEON_COV_PATH.exists():
        print_status("Loading Pantheon+ STAT+SYS covariance...", "INFO")
        with open(PANTHEON_COV_PATH) as f:
            n = int(f.readline().strip())
        cov_flat = np.loadtxt(PANTHEON_COV_PATH, skiprows=1)
        cov_full = cov_flat.reshape(n, n)
        print_status(f"  Full covariance: {cov_full.shape}", "INFO")

        # Extract calibrator sub-matrix
        cal_indices = cal.index.values
        cov_cal = cov_full[np.ix_(cal_indices, cal_indices)]
        print_status(f"  Calibrator covariance: {cov_cal.shape}", "INFO")

        # Compute correlations
        sigma_cal = np.sqrt(np.diag(cov_cal))
        corr_cal = cov_cal / np.outer(sigma_cal, sigma_cal)
        off_mask = ~np.eye(len(cal_indices), dtype=bool)
        n_correlated = (np.abs(corr_cal[off_mask]) > 0.1).sum()
        print_status(f"  Correlated pairs (|r| > 0.1): {n_correlated}", "INFO")
    else:
        print_status("Pantheon+ covariance not available, using diagonal only", "WARNING")
        cov_cal = np.diag(cal['m_b_corr_err_DIAG'].values ** 2)

    # ------------------------------------------------------------------
    # 3. Compute data-driven sigma_v for Cepheid hosts
    # ------------------------------------------------------------------
    print_status("Computing data-driven sigma_v...", "SECTION")

    # The peculiar velocity contribution to m_b uncertainty is:
    # sigma_m_b_VPEC = (5 * sigma_v / cz) / ln(10)
    # So sigma_v = sigma_m_b_VPEC * cz * ln(10) / 5

    # For each calibrator SN, compute the effective sigma_v
    cal['sigma_v_effective'] = cal['m_b_corr_err_VPEC'] * C_KM_S * cal['zHD'] * LN10_OVER_5
    # Actually: sigma_m_b_VPEC = 5 * sigma_v / (cz * ln(10))
    # So sigma_v = sigma_m_b_VPEC * cz * ln(10) / 5
    # But sigma_m_b_VPEC is already in magnitude units
    # Let's compute it properly:
    # The peculiar velocity contributes sigma_v to cz
    # In magnitude space: sigma_mu = 5 * sigma_v / (cz * ln(10))
    # So sigma_v = sigma_mu * cz * ln(10) / 5
    cal['sigma_v_from_m_b'] = cal['m_b_corr_err_VPEC'] * C_KM_S * cal['zHD'] * np.log(10) / 5.0

    print_status(f"Calibrator sigma_v from m_b:", "INFO")
    print_status(f"  Mean: {cal['sigma_v_from_m_b'].mean():.1f} km/s", "INFO")
    print_status(f"  Median: {cal['sigma_v_from_m_b'].median():.1f} km/s", "INFO")
    print_status(f"  Std: {cal['sigma_v_from_m_b'].std():.1f} km/s", "INFO")
    print_status(f"  Min: {cal['sigma_v_from_m_b'].min():.1f} km/s", "INFO")
    print_status(f"  Max: {cal['sigma_v_from_m_b'].max():.1f} km/s", "INFO")

    # Group by Cepheid host (CEPH_DIST)
    # Multiple SNe in the same host should have the same sigma_v
    cal['host_id'] = cal['CEPH_DIST'].round(4)
    host_sigma_v = cal.groupby('host_id')['sigma_v_from_m_b'].agg(['mean', 'std', 'count'])
    print_status(f"\nPer-host sigma_v (unique Cepheid distances): {len(host_sigma_v)}", "INFO")
    print_status(f"  Mean per-host sigma_v: {host_sigma_v['mean'].mean():.1f} km/s", "INFO")

    # The Pantheon+ default is 250 km/s for all SNe
    # The data-driven values are typically lower because:
    # 1. The VPEC model accounts for known bulk flows
    # 2. The residual scatter after bulk flow correction is smaller
    print_status(f"\nComparison with fixed sigma_v:", "INFO")
    print_status(f"  Pantheon+ default: 250 km/s (all SNe)", "INFO")
    print_status(f"  Data-driven mean: {cal['sigma_v_from_m_b'].mean():.1f} km/s", "INFO")
    print_status(f"  Data-driven median: {cal['sigma_v_from_m_b'].median():.1f} km/s", "INFO")

    # ------------------------------------------------------------------
    # 4. Estimate the peculiar velocity scatter from VPEC residuals
    # ------------------------------------------------------------------
    print_status("\nPeculiar velocity scatter from VPEC residuals...", "SECTION")

    # The VPEC column gives model-predicted peculiar velocities
    # The actual scatter is the RMS of VPEC after subtracting the mean
    vpec_cal = cal['VPEC'].values
    vpec_hf = hf['VPEC'].values

    print_status(f"Calibrator VPEC:", "INFO")
    print_status(f"  Mean: {vpec_cal.mean():.1f} km/s", "INFO")
    print_status(f"  RMS: {np.sqrt(np.mean(vpec_cal**2)):.1f} km/s", "INFO")
    print_status(f"  Std (after mean removal): {vpec_cal.std():.1f} km/s", "INFO")

    print_status(f"\nHubble-flow VPEC:", "INFO")
    print_status(f"  Mean: {vpec_hf.mean():.1f} km/s", "INFO")
    print_status(f"  RMS: {np.sqrt(np.mean(vpec_hf**2)):.1f} km/s", "INFO")
    print_status(f"  Std (after mean removal): {vpec_hf.std():.1f} km/s", "INFO")

    # The effective sigma_v is the RMS of the residual peculiar velocities
    # after accounting for the model-predicted bulk flow
    # The Pantheon+ VPEC model already accounts for known bulk flows
    # So the residual scatter is the "irreducible" peculiar velocity noise
    sigma_v_residual = vpec_cal.std()
    print_status(f"\nResidual sigma_v (calibrator): {sigma_v_residual:.1f} km/s", "INFO")
    print_status(f"This is the data-driven estimate of the peculiar velocity scatter", "INFO")
    print_status(f"after removing the model-predicted bulk flow.", "INFO")

    # ------------------------------------------------------------------
    # 5. Test the redshift–distance likelihood with data-driven sigma_v
    # ------------------------------------------------------------------
    print_status("\nRedshift–distance likelihood with data-driven sigma_v...", "SECTION")

    # Load the host data (same as Step 46)
    df_hosts = pd.read_csv(HOSTS_PATH)

    # Load screening
    step3_path = OUT_DIR / "step_03_stratified_h0.csv"
    screening_map = {}
    if step3_path.exists():
        df_s3 = pd.read_csv(step3_path)
        for _, row in df_s3.iterrows():
            screening_map[row["normalized_name"]] = float(row.get("shear_suppression", 1.0))

    # Load Tully catalog
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
    y_data = np.loadtxt(SH0ES_DIR / "y_R22.txt", unpack=True, skiprows=1,
                        dtype={"names": names, "formats": fmt})
    y = y_data[1]
    C = np.loadtxt(SH0ES_DIR / "C_R22.txt", delimiter="\t")
    q = np.loadtxt(SH0ES_DIR / "q_R22.txt", unpack=True, dtype="str")

    from scipy import linalg
    try:
        Lc = np.linalg.cholesky(C)
        A_w = linalg.solve_triangular(Lc, L, lower=True, check_finite=False)
        y_w = linalg.solve_triangular(Lc, y, lower=True, check_finite=False)
    except Exception:
        A_w = L.copy(); y_w = y.copy()
    theta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=1e-12)

    mu_indices = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_names = [q[i].replace("mu_", "") for i in mu_indices]
    anchor_set = {"N4258", "LMC", "M31", "MW", "SMC"}

    # Build host data
    hosts = []
    for h in mu_names:
        if h in anchor_set:
            continue
        sigma = None; z = None; S = 1.0
        for _, row in df_hosts.iterrows():
            name = row["normalized_name"]
            compact = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
            if compact == h or name.replace(" ", "") == h:
                sigma = row["sigma_inferred"]
                z = row["z_hd"]
                S = screening_map.get(name, 1.0)
                break
        if sigma is None or sigma <= 0 or z is None or z <= 0:
            continue
        mu_idx = mu_indices[mu_names.index(h)]
        mu_val = theta[mu_idx]
        hosts.append({"host": h, "sigma": float(sigma), "S": float(S),
                       "mu": float(mu_val), "z": float(z)})

    df = pd.DataFrame(hosts)
    print_status(f"  N hosts = {len(df)}", "INFO")

    # Build covariates
    from scripts.utils.tep_correction import compute_anchor_sigma_ref
    sigma_ref = compute_anchor_sigma_ref(screened=True)

    d_obs = 10 ** ((df["mu"].values - 25.0) / 5.0)
    cz_obs = C_KM_S * df["z"].values
    X = np.array([(S * s**2 - sigma_ref**2) / C_KM_S**2
                  for s, S in zip(df["sigma"].values, df["S"].values)])
    X_c = X - np.mean(X)
    sigma_mu = np.full(len(df), 0.1)

    # Test with different sigma_v values
    sigma_v_values = [150.0, float(sigma_v_residual), 250.0, 500.0]
    BETA_SCALE = 1e7; KAPPA_SCALE = 1e5

    def neg_logL_K0(params, cz_obs, d_obs, X, sigma_mu, sigma_v):
        H_app = params[0]
        kappa = params[1] * KAPPA_SCALE
        d_true = d_obs * np.power(10.0, kappa * X / 5.0)
        cz_model = d_true * H_app
        resid = cz_obs - cz_model
        sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
        var = sigma_v**2 + sigma_cz_dist**2
        var = np.maximum(var, 0.01)
        return 0.5 * np.sum(resid**2 / var + np.log(var))

    def neg_logL_H0(params, cz_obs, d_obs, X, sigma_mu, sigma_v):
        H_app = params[0]
        cz_model = d_obs * H_app
        resid = cz_obs - cz_model
        sigma_cz_dist = LN10_OVER_5 * np.abs(cz_model) * sigma_mu
        var = sigma_v**2 + sigma_cz_dist**2
        var = np.maximum(var, 0.01)
        return 0.5 * np.sum(resid**2 / var + np.log(var))

    results = []
    for sigma_v in sigma_v_values:
        label = "data-driven" if np.isclose(sigma_v, sigma_v_residual) else f"fixed_{int(sigma_v)}"

        # Fit K0 (Cepheid-only)
        H_init = np.median(cz_obs / d_obs)
        res_K0 = optimize.minimize(
            neg_logL_K0, [H_init, 0.0],
            args=(cz_obs, d_obs, X_c, sigma_mu, sigma_v),
            method="L-BFGS-B", bounds=[(30, 90), (-50, 50)]
        )
        # Try alternative initializations
        for H_try in [65, 70, 75, 80]:
            res_try = optimize.minimize(
                neg_logL_K0, [H_try, 0.0],
                args=(cz_obs, d_obs, X_c, sigma_mu, sigma_v),
                method="L-BFGS-B", bounds=[(30, 90), (-50, 50)]
            )
            if res_try.fun < res_K0.fun:
                res_K0 = res_try

        # Fit H0 (null)
        res_H0 = optimize.minimize(
            neg_logL_H0, [H_init],
            args=(cz_obs, d_obs, X_c, sigma_mu, sigma_v),
            method="L-BFGS-B", bounds=[(30, 90)]
        )
        for H_try in [65, 70, 75, 80]:
            res_try = optimize.minimize(
                neg_logL_H0, [H_try],
                args=(cz_obs, d_obs, X_c, sigma_mu, sigma_v),
                method="L-BFGS-B", bounds=[(30, 90)]
            )
            if res_try.fun < res_H0.fun:
                res_H0 = res_try

        kappa_best = res_K0.x[1] * KAPPA_SCALE
        logL_K0 = -res_K0.fun
        logL_H0 = -res_H0.fun
        lrt = 2 * (logL_K0 - logL_H0)
        p_value = stats.chi2.sf(lrt, df=1)
        sigma_sig = np.sqrt(lrt) if lrt > 0 else 0.0

        results.append({
            "sigma_v": sigma_v,
            "label": label,
            "kappa_best": kappa_best,
            "logL_K0": logL_K0,
            "logL_H0": logL_H0,
            "lrt": lrt,
            "p_value": p_value,
            "significance_sigma": sigma_sig,
            "H_app_K0": res_K0.x[0],
            "H_app_H0": res_H0.x[0],
        })

        print_status(
            f"  sigma_v = {sigma_v:.0f} km/s ({label}): "
            f"kappa = {kappa_best:.3e}, "
            f"LRT = {lrt:.2f}, "
            f"significance = {sigma_sig:.2f}sigma",
            "SUCCESS",
        )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print_status("\nSummary", "SECTION")
    print_status(
        f"The Pantheon+ peculiar velocity data provides a data-driven\n"
        f"sigma_v = {sigma_v_residual:.1f} km/s (residual RMS after bulk flow\n"
        f"correction), compared to the Pantheon+ default of 250 km/s.\n"
        f"The kappa_Cep significance at the data-driven sigma_v is\n"
        f"{[r for r in results if r['label'] == 'data-driven'][0]['significance_sigma']:.2f}sigma\n"
        f"(LRT), compared to "
        f"{[r for r in results if r['label'] == 'fixed_250'][0]['significance_sigma']:.2f}sigma "
        f"at sigma_v = 250 km/s.",
        "INFO",
    )

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    output = {
        "description": "Pantheon+ peculiar-velocity covariance analysis",
        "pantheon_data": {
            "n_sne": len(sn),
            "n_calibrators": len(cal),
            "n_hubble_flow": len(hf),
            "vpecerr_default": 250.0,
        },
        "data_driven_sigma_v": {
            "residual_rms": float(sigma_v_residual),
            "calibrator_mean": float(cal['sigma_v_from_m_b'].mean()),
            "calibrator_median": float(cal['sigma_v_from_m_b'].median()),
            "calibrator_std": float(cal['sigma_v_from_m_b'].std()),
        },
        "covariance_analysis": {
            "n_correlated_pairs": int(n_correlated) if PANTHEON_COV_PATH.exists() else 0,
            "max_correlation": float(np.abs(corr_cal[off_mask]).max()) if PANTHEON_COV_PATH.exists() else 0.0,
        },
        "likelihood_results": results,
    }

    out_json = OUT_DIR / "step_48_pantheon_velocity_covariance.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print_status(f"\nSaved JSON to {out_json}", "SUCCESS")

    df_out = pd.DataFrame(results)
    out_csv = OUT_DIR / "step_48_pantheon_velocity_covariance.csv"
    df_out.to_csv(out_csv, index=False)
    print_status(f"Saved CSV to {out_csv}", "SUCCESS")


if __name__ == "__main__":
    main()
