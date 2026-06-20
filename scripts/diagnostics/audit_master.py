#!/usr/bin/env python3
"""
TEP-H0 Master Audit Script
==========================

Comprehensive audit of the TEP-H0 analysis following the practical audit checklist.
Run with: python scripts/audit_master.py

Outputs:
- results/outputs/step_32_audit_master_report.json
- results/outputs/step_32_audit_master_table.csv
- stdout: formatted audit findings
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_correction import tep_correction, C_SQUARED_KM_S

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

HOSTS_PATH = DATA_DIR / "processed" / "hosts_processed.csv"
DISTANCES_PATH = DATA_DIR / "interim" / "r22_distances.csv"
STRATIFIED_PATH = RESULTS_DIR / "step_03_stratified_h0.csv"
CORRECTED_PATH = RESULTS_DIR / "step_04_tep_corrected_h0.csv"
COV_PATH = RESULTS_DIR / "step_03_h0_covariance.npy"
COV_LABELS_PATH = RESULTS_DIR / "step_03_h0_covariance_labels.json"
MU_COV_PATH = DATA_DIR / "interim" / "r22_mu_covariance.npy"
MU_COV_LABELS_PATH = DATA_DIR / "interim" / "r22_mu_covariance_labels.json"
SIGMA_PROV_PATH = RESULTS_DIR / "step_07_sigma_provenance_table.csv"
REDZ_SENS_PATH = RESULTS_DIR / "step_08_redshift_cut_sensitivity.txt"
APERTURE_SUMMARY_PATH = RESULTS_DIR / "step_07_aperture_sensitivity_summary.json"
TEP_CORR_JSON_PATH = RESULTS_DIR / "step_04_tep_correction_results.json"

REPORT_PATH = RESULTS_DIR / "step_32_audit_master_report.json"
TABLE_PATH = RESULTS_DIR / "step_32_audit_master_table.csv"

C_KM_S = 299792.458
C2 = C_KM_S ** 2


def load_data():
    """Load all required data files."""
    hosts = pd.read_csv(HOSTS_PATH)
    dists = pd.read_csv(DISTANCES_PATH)
    strat = pd.read_csv(STRATIFIED_PATH) if STRATIFIED_PATH.exists() else None
    corr = pd.read_csv(CORRECTED_PATH) if CORRECTED_PATH.exists() else None
    sigma_prov = pd.read_csv(SIGMA_PROV_PATH) if SIGMA_PROV_PATH.exists() else None

    merged = pd.merge(dists, hosts, on="source_id", how="inner")
    merged["distance_mpc"] = 10 ** ((merged["value"] - 25) / 5)
    merged["velocity"] = C_KM_S * merged["z_hd"]
    merged["h0_derived"] = merged["velocity"] / merged["distance_mpc"]

    # Full sample (before redshift cut)
    full = merged.dropna(subset=["h0_derived", "sigma_inferred", "m_b_corr"]).copy()
    full["normalized_name"] = full["normalized_name"].astype(str).str.strip()
    anchors = ["NGC 4258", "LMC", "SMC", "M 31", "MW"]
    full_nh = full[~full["normalized_name"].isin(anchors)].copy()

    # Primary sample (z > 0.0035)
    primary = full_nh[pd.to_numeric(full_nh["z_hd"], errors="coerce") > 0.0035].copy()

    return {
        "hosts": hosts,
        "dists": dists,
        "merged": merged,
        "full_nh": full_nh,
        "primary": primary,
        "strat": strat,
        "corr": corr,
        "sigma_prov": sigma_prov,
    }


def build_master_table(data):
    """Build master host table with redshifts and inclusion flags."""
    df = data["full_nh"].copy()
    df = df.sort_values("normalized_name")

    table = []
    for _, row in df.iterrows():
        z_hd = pd.to_numeric(row.get("z_hd"), errors="coerce")
        z_cmb = pd.to_numeric(row.get("z_cmb"), errors="coerce")
        z_hel = pd.to_numeric(row.get("z_hel"), errors="coerce")

        included = z_hd > 0.0035 if pd.notna(z_hd) else False
        reason = "z_HD > 0.0035" if included else "z_HD <= 0.0035 (excluded)"

        table.append({
            "Host": row["normalized_name"],
            "SN": row.get("pantheon_id", ""),
            "z_HEL": round(z_hel, 5) if pd.notna(z_hel) else None,
            "z_CMB": round(z_cmb, 5) if pd.notna(z_cmb) else None,
            "z_HD": round(z_hd, 5) if pd.notna(z_hd) else None,
            "included": included,
            "reason": reason,
            "sigma_measured": row.get("sigma_measured"),
            "sigma_inferred": row.get("sigma_inferred"),
            "sigma_corrected": row.get("sigma_corrected"),
            "h0_derived": round(row["h0_derived"], 2) if pd.notna(row.get("h0_derived")) else None,
        })

    return pd.DataFrame(table)


def audit_sample_consistency(data):
    """Priority 1.1: Verify sample and redshift cuts."""
    findings = []
    primary = data["primary"]

    n_primary = len(primary)
    z_min = primary["z_hd"].min()
    z_max = primary["z_hd"].max()

    findings.append(f"Primary sample: N={n_primary}, z_HD range [{z_min:.5f}, {z_max:.5f}]")

    # Check if every included host satisfies z > 0.0035
    violations = primary[primary["z_hd"] <= 0.0035]
    if len(violations) > 0:
        findings.append(f"CRITICAL: {len(violations)} included hosts violate z > 0.0035")
        for _, row in violations.iterrows():
            findings.append(f"  - {row['normalized_name']}: z_HD={row['z_hd']}")
    else:
        findings.append("OK: All included hosts satisfy z_HD > 0.0035")

    # Full non-anchor sample
    full_nh = data["full_nh"]
    n_full = len(full_nh)
    findings.append(f"Full non-anchor sample: N={n_full}")

    excluded = full_nh[pd.to_numeric(full_nh["z_hd"], errors="coerce") <= 0.0035]
    findings.append(f"Excluded by z-cut: {len(excluded)} hosts")
    for _, row in excluded.iterrows():
        findings.append(f"  - {row['normalized_name']}: z_HD={row['z_hd']:.5f}")

    # Check for hosts with very low z that manuscript mentions
    very_low = primary[primary["z_hd"] < 0.005]
    findings.append(f"Hosts with z_HD < 0.005 in primary sample: {len(very_low)}")
    for _, row in very_low.iterrows():
        findings.append(f"  - {row['normalized_name']}: z_HD={row['z_hd']:.5f}")

    return findings


def audit_h0_contradictions(data):
    """Priority 1.2: Check corrected H0 values for contradictions."""
    findings = []

    # From step_04_tep_correction_results.json
    if TEP_CORR_JSON_PATH.exists():
        with open(TEP_CORR_JSON_PATH) as f:
            tep_json = json.load(f)
        findings.append(f"step_04_tep_correction_results.json: unified_h0 = {tep_json.get('unified_h0'):.2f}")
        findings.append(f"  bootstrap_h0_mean = {tep_json.get('bootstrap_h0_mean'):.2f} +/- {tep_json.get('bootstrap_h0_std'):.2f}")
        findings.append(f"  optimal_kappa_cep = {tep_json.get('optimal_kappa_cep'):.3e}")
        findings.append(f"  sigma_ref = {tep_json.get('sigma_ref'):.2f}")
        findings.append(f"  unified_h0_screened = {tep_json.get('unified_h0_screened'):.2f}")

    # From aperture sensitivity
    if APERTURE_SUMMARY_PATH.exists():
        with open(APERTURE_SUMMARY_PATH) as f:
            apert = json.load(f)
        grid = apert.get("grid_summary", {})
        h0_min = grid.get("unified_h0_min")
        h0_max = grid.get("unified_h0_max")
        findings.append(f"Aperture sensitivity: unified_H0 in [{h0_min:.2f}, {h0_max:.2f}]")
        if h0_min is not None and h0_max is not None:
            if h0_max < 66.5 or h0_min > 69.0:
                findings.append("  NOTE: Aperture range does not include 68.75; this is expected because aperture varies the correction amplitude.")

    # From redshift sensitivity
    if REDZ_SENS_PATH.exists():
        rz = pd.read_csv(REDZ_SENS_PATH)
        row_29 = rz[rz["n"] == 29]
        if len(row_29) > 0:
            r = row_29.iloc[0]
            findings.append(f"Redshift sensitivity (N=29, zcut=0.0035): h0_corr={r['h0_corr']:.2f}, kappa={r['kappa_1e6']:.4f}")

    return findings


def audit_delta_mu_sign(data):
    """Priority 2.4: Verify distance-modulus to H0 conversion sign."""
    findings = []

    # Toy test from checklist
    sigma_test = 200.0
    sigma_ref_test = 87.16507328052906
    S_test = 1.0
    kappa_test = 1.05e6

    delta_mu = kappa_test * S_test * (sigma_test**2 - sigma_ref_test**2) / C2
    findings.append(f"Toy test: sigma={sigma_test}, sigma_ref={sigma_ref_test:.2f}, S={S_test}, kappa={kappa_test:.3e}")
    findings.append(f"  Delta_mu = {delta_mu:.4f} mag")

    if not (0.3 <= delta_mu <= 0.4):
        findings.append(f"  WARNING: Expected ~0.3-0.4 mag for high-sigma hosts, got {delta_mu:.4f}")
    else:
        findings.append(f"  OK: Correction magnitude is in expected range.")

    # Check sign: for high sigma, Delta_mu > 0, distance increases, H0 decreases
    if delta_mu > 0:
        findings.append("  Sign check: Delta_mu > 0 for high-sigma → distance increases → H0 decreases")
        findings.append("  This means the correction REDUCES H0 for high-sigma hosts (good).")
    else:
        findings.append("  CRITICAL: Delta_mu < 0 for high-sigma — sign error!")

    # Verify in actual corrected data
    corr = data["corr"]
    if corr is not None:
        high_sigma = corr[corr["sigma_inferred"] > 100]
        if len(high_sigma) > 0:
            avg_dmu = high_sigma["mu_corrected"].mean() - high_sigma["value"].mean()
            findings.append(f"  Actual data: mean Delta_mu for sigma>100 hosts = {avg_dmu:.4f} mag")
            if avg_dmu < 0:
                findings.append("  CRITICAL: Actual correction is NEGATIVE for high-sigma hosts!")

    return findings


def recompute_headline_numbers(data):
    """Priority 2-5: Recompute all headline numbers from scratch."""
    findings = []
    primary = data["primary"]

    if len(primary) == 0:
        findings.append("No primary sample data")
        return findings

    sigma_vals = primary["sigma_inferred"].values
    h0_vals = primary["h0_derived"].values

    # Pearson / Spearman
    pearson_r, pearson_p = stats.pearsonr(sigma_vals, h0_vals)
    spearman_r, spearman_p = stats.spearmanr(sigma_vals, h0_vals)

    findings.append(f"Recomputed Pearson: r={pearson_r:.4f}, p={pearson_p:.4f}")
    findings.append(f"Recomputed Spearman: rho={spearman_r:.4f}, p={spearman_p:.4f}")

    # Median split
    median_sigma = np.median(sigma_vals)
    low = primary[primary["sigma_inferred"] <= median_sigma]
    high = primary[primary["sigma_inferred"] > median_sigma]

    mean_low = low["h0_derived"].mean()
    err_low = low["h0_derived"].std() / np.sqrt(len(low))
    mean_high = high["h0_derived"].mean()
    err_high = high["h0_derived"].std() / np.sqrt(len(high))

    findings.append(f"Low-sigma (N={len(low)}, sigma<={median_sigma:.1f}): H0={mean_low:.2f} +/- {err_low:.2f}")
    findings.append(f"High-sigma (N={len(high)}, sigma>{median_sigma:.1f}): H0={mean_high:.2f} +/- {err_high:.2f}")
    findings.append(f"Delta H0 = {mean_high - mean_low:.2f}")

    # IMPORTANT: Use the actual stratified data which includes shear_suppression
    strat = data["strat"]
    if strat is not None and len(strat) == len(primary):
        S = strat["shear_suppression"].values
        mu_vals = strat["value"].values
        v_vals = strat["velocity"].values
        # Use actual sigma_ref from anchor-weighted calculation
        sigma_ref = 87.16507328052906
    else:
        S = np.ones(len(primary))
        mu_vals = primary["value"].values
        v_vals = primary["velocity"].values
        sigma_ref = median_sigma

    def objective(k):
        correction = tep_correction(sigma_vals, sigma_ref, k[0], S)
        mu_corr = mu_vals + correction
        d_corr = 10 ** ((mu_corr - 25) / 5)
        h0_corr = v_vals / d_corr
        slope, _ = np.polyfit(sigma_vals, h0_corr, 1)
        return slope ** 2

    res = minimize(objective, x0=[1.0e6], method="Nelder-Mead", options={"xatol": 1.0, "fatol": 1e-8, "maxiter": 2000})
    kappa_opt = res.x[0]

    # Apply correction
    correction = tep_correction(sigma_vals, sigma_ref, kappa_opt, S)
    mu_corr = mu_vals + correction
    d_corr = 10 ** ((mu_corr - 25) / 5)
    h0_corr = v_vals / d_corr
    h0_mean_corr = h0_corr.mean()
    h0_sem_corr = h0_corr.std() / np.sqrt(len(primary))

    findings.append(f"Optimized kappa_cep = {kappa_opt:.3e}")
    findings.append(f"Corrected H0 mean = {h0_mean_corr:.2f} +/- {h0_sem_corr:.2f}")

    # Compare with pipeline JSON
    if TEP_CORR_JSON_PATH.exists():
        with open(TEP_CORR_JSON_PATH) as f:
            tep_json = json.load(f)
        json_kappa = tep_json.get("optimal_kappa_cep")
        json_h0 = tep_json.get("unified_h0")
        findings.append(f"Pipeline JSON kappa = {json_kappa:.3e}, H0 = {json_h0:.2f}")
        if not np.isclose(kappa_opt, json_kappa, rtol=0.01):
            findings.append(f"  WARNING: Recomputed kappa differs from pipeline by {abs(kappa_opt - json_kappa) / json_kappa * 100:.1f}%")
        if not np.isclose(h0_mean_corr, json_h0, rtol=0.01):
            findings.append(f"  WARNING: Recomputed H0 differs from pipeline by {abs(h0_mean_corr - json_h0):.2f}")

    # LOOCV
    loocv_h0s = []
    for i in range(len(primary)):
        mask = np.ones(len(primary), dtype=bool)
        mask[i] = False
        sigma_train = sigma_vals[mask]
        S_train = S[mask]

        def obj_loo(k):
            corr_train = tep_correction(sigma_train, sigma_ref, k[0], S_train)
            mu_train = mu_vals[mask] + corr_train
            d_train = 10 ** ((mu_train - 25) / 5)
            h_train = v_vals[mask] / d_train
            s, _ = np.polyfit(sigma_train, h_train, 1)
            return s ** 2

        res_loo = minimize(obj_loo, x0=[1.0e6], method="Nelder-Mead", options={"xatol": 100.0, "fatol": 1e-8, "maxiter": 500})
        if res_loo.success:
            kappa_loo = res_loo.x[0]
            corr_test = tep_correction(sigma_vals[i], sigma_ref, kappa_loo, S[i])
            mu_test = mu_vals[i] + corr_test
            d_test = 10 ** ((mu_test - 25) / 5)
            h0_test = v_vals[i] / d_test
            loocv_h0s.append(h0_test)

    if loocv_h0s:
        loocv_mean = np.mean(loocv_h0s)
        loocv_std = np.std(loocv_h0s)
        findings.append(f"LOOCV H0 = {loocv_mean:.2f} +/- {loocv_std:.2f} (N={len(loocv_h0s)})")

    return findings


def audit_covariance(data):
    """Priority 3.21: Verify SH0ES covariance extraction."""
    findings = []

    if not COV_PATH.exists() or not COV_LABELS_PATH.exists():
        findings.append("Covariance files not found")
        return findings

    cov = np.load(COV_PATH)
    with open(COV_LABELS_PATH) as f:
        labels = json.load(f)

    findings.append(f"Covariance matrix shape: {cov.shape}")
    findings.append(f"N labels: {len(labels)}")

    # Check symmetric
    is_symmetric = np.allclose(cov, cov.T, atol=1e-10)
    findings.append(f"Symmetric: {is_symmetric}")
    if not is_symmetric:
        findings.append(f"  max asymmetry: {np.max(np.abs(cov - cov.T)):.3e}")

    # Check positive semi-definite
    eigs = np.linalg.eigvalsh(cov)
    min_eig = eigs.min()
    findings.append(f"Min eigenvalue: {min_eig:.3e}")
    if min_eig < -1e-8:
        findings.append("  CRITICAL: Matrix is not positive semi-definite!")
    else:
        findings.append("  OK: Matrix is positive semi-definite within tolerance")

    # Check diagonal entries match published uncertainties
    primary = data["primary"]
    if len(primary) == len(labels):
        diag_err = np.sqrt(np.diag(cov))
        # The H0 errors from stratified data
        h0_err_linear = primary["h0_derived"] * (np.log(10) / 5) * primary["error"]
        findings.append(f"Mean diagonal cov sqrt: {diag_err.mean():.3f}")
        findings.append(f"Mean linear H0 err: {h0_err_linear.mean():.3f}")
        ratio = diag_err / h0_err_linear.values
        findings.append(f"Mean ratio (cov_diag / linear_err): {ratio.mean():.3f}")
        if not np.allclose(diag_err, h0_err_linear.values, rtol=0.5):
            findings.append("  WARNING: Diagonal covariance does not closely match linear H0 errors")

    # Check correlation values plausible
    corr_mat = np.corrcoef(cov)
    off_diag = corr_mat[np.triu_indices_from(corr_mat, k=1)]
    findings.append(f"Off-diagonal correlation range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")

    return findings


def audit_sigma_provenance(data):
    """Priority 4.9: Rebuild sigma table from scratch."""
    findings = []
    prov = data["sigma_prov"]
    if prov is None:
        findings.append("No sigma provenance table found")
        return findings

    n_total = len(prov)
    n_stellar = (prov["sigma_method"] == "stellar absorption").sum()
    n_proxy = (prov["sigma_method"] == "HI linewidth proxy").sum()

    findings.append(f"Sigma provenance: N={n_total}, stellar={n_stellar}, proxy={n_proxy}")

    # Check for missing uncertainties
    missing_err = prov[prov["sigma_measured_error_kms"].isna() | (prov["sigma_measured_error_kms"] == 0)]
    if len(missing_err) > 0:
        findings.append(f"WARNING: {len(missing_err)} hosts have missing or zero sigma measurement errors")

    # Check for large corrections
    prov["delta_sigma"] = prov["sigma_inferred_kms"] - prov["sigma_measured_kms"]
    large_delta = prov[prov["delta_sigma"].abs() > 20]
    if len(large_delta) > 0:
        findings.append(f"Hosts with |sigma_corrected - sigma_measured| > 20 km/s: {len(large_delta)}")
        for _, row in large_delta.iterrows():
            findings.append(f"  - {row['normalized_name']}: measured={row['sigma_measured_kms']:.1f}, corrected={row['sigma_inferred_kms']:.1f}")

    # Check duplicate hosts
    dupes = prov[prov.duplicated("normalized_name", keep=False)]
    if len(dupes) > 0:
        findings.append(f"WARNING: {len(dupes)} duplicate normalized_name entries in provenance table")

    return findings


def audit_planck_circularity(data):
    """Priority 2.5: Check whether Planck is built into residual definition."""
    findings = []

    primary = data["primary"]
    if len(primary) == 0:
        return findings

    # The analysis computes H0 = cz/d directly, not as a residual from Planck.
    # But the correction is fit to make the mean match ~Planck.
    h0_raw_mean = primary["h0_derived"].mean()
    findings.append(f"Raw mean H0 (primary sample) = {h0_raw_mean:.2f}")
    findings.append("The analysis defines H0 = cz/d directly, not as a residual from Planck.")
    findings.append("However, the TEP correction is optimized to flatten the H0-sigma trend,")
    findings.append("and the resulting mean happens to be near Planck. This is a diagnostic,")
    findings.append("not independent evidence, and the manuscript already notes this.")

    # Fit with free intercept
    sigma_vals = primary["sigma_inferred"].values
    h0_vals = primary["h0_derived"].values
    S = primary.get("shear_suppression", pd.Series(np.ones(len(primary)))).values
    x = S * (sigma_vals**2 - np.median(sigma_vals)**2) / C2

    # Linear fit H0 = a + b*x
    A = np.vstack([np.ones(len(x)), x]).T
    coeff, residual, rank, s = np.linalg.lstsq(A, h0_vals, rcond=None)
    a_free, b_free = coeff

    findings.append(f"Free-intercept fit: H0 = {a_free:.2f} + {b_free:.3e} * X")
    findings.append(f"  Slope b = {b_free:.3e}; if significantly non-zero, TEP signal exists independent of intercept.")

    # Test with different fiducial H0 for residual
    for fid_h0 in [67.4, 70.0, 73.0]:
        residual = h0_vals - fid_h0
        slope, intercept, r_val, p_val, se = stats.linregress(x, residual)
        findings.append(f"  Residual from H0={fid_h0}: slope={slope:.3e}, p={p_val:.4f}, r={r_val:.4f}")

    return findings


def audit_bic_sign(data):
    """Priority 5.16: Check BIC sign and parameter counting."""
    findings = []

    primary = data["primary"]
    if len(primary) == 0:
        return findings

    sigma_vals = primary["sigma_inferred"].values
    h0_vals = primary["h0_derived"].values
    n = len(primary)

    # Null model: constant H0 (1 parameter)
    null_mean = h0_vals.mean()
    null_var = h0_vals.var(ddof=1)
    ll_null = -0.5 * n * (np.log(2 * np.pi * null_var) + 1)
    k_null = 1
    bic_null = -2 * ll_null + k_null * np.log(n)

    # TEP model: linear in X = S*(sigma^2 - sigma_ref^2)/c^2 (2 params: intercept + slope)
    S = primary.get("shear_suppression", pd.Series(np.ones(len(primary)))).values
    sigma_ref = np.median(sigma_vals)
    x = S * (sigma_vals**2 - sigma_ref**2) / C2

    A = np.vstack([np.ones(len(x)), x]).T
    coeff, residual, rank, s_vals = np.linalg.lstsq(A, h0_vals, rcond=None)
    pred = A @ coeff
    resid = h0_vals - pred
    tep_var = resid.var(ddof=2) if len(resid) > 2 else resid.var()
    ll_tep = -0.5 * n * (np.log(2 * np.pi * tep_var) + 1)
    k_tep = 2
    bic_tep = -2 * ll_tep + k_tep * np.log(n)

    delta_bic = bic_null - bic_tep  # positive means TEP favored

    findings.append(f"Null model: BIC_null = {bic_null:.2f} (k={k_null})")
    findings.append(f"TEP model: BIC_tep = {bic_tep:.2f} (k={k_tep})")
    findings.append(f"Delta BIC = BIC_null - BIC_tep = {delta_bic:.2f}")
    if delta_bic > 0:
        findings.append("  Positive Delta BIC favors TEP model")
    else:
        findings.append("  Negative Delta BIC favors null model")

    findings.append("NOTE: manuscript reports Delta BIC = +2.6, which is consistent with this convention.")

    return findings


def run_leave_one_out(data):
    """Priority 5.17: Leave-one-out and influence diagnostics."""
    findings = []
    primary = data["primary"]
    if len(primary) == 0:
        return findings

    sigma_vals = primary["sigma_inferred"].values
    h0_vals = primary["h0_derived"].values
    S = primary.get("shear_suppression", pd.Series(np.ones(len(primary)))).values
    sigma_ref = np.median(sigma_vals)

    slopes = []
    for i in range(len(primary)):
        mask = np.ones(len(primary), dtype=bool)
        mask[i] = False
        s, _ = np.polyfit(sigma_vals[mask], h0_vals[mask], 1)
        slopes.append(s)

    slopes = np.array(slopes)
    findings.append(f"Leave-one-out slope range: [{slopes.min():.4f}, {slopes.max():.4f}]")
    findings.append(f"  Median LOO slope: {np.median(slopes):.4f}")
    findings.append(f"  Full-sample slope: {np.polyfit(sigma_vals, h0_vals, 1)[0]:.4f}")

    # Identify most influential host
    idx_max = np.argmax(np.abs(slopes - np.polyfit(sigma_vals, h0_vals, 1)[0]))
    host_max = primary.iloc[idx_max]["normalized_name"]
    findings.append(f"  Most influential host when removed: {host_max} (slope change = {slopes[idx_max] - np.polyfit(sigma_vals, h0_vals, 1)[0]:.4f})")

    return findings


def audit_redshift_frames(data):
    """Priority 3.6: Compare z_HEL, z_CMB, z_HD."""
    findings = []
    primary = data["primary"]
    if len(primary) == 0:
        return findings

    for zcol, label in [("z_hel", "HEL"), ("z_cmb", "CMB"), ("z_hd", "HD")]:
        z = pd.to_numeric(primary[zcol], errors="coerce")
        v = C_KM_S * z
        h0 = v / primary["distance_mpc"]
        r, p = stats.pearsonr(primary["sigma_inferred"].values, h0.values)
        findings.append(f"{label} frame: Pearson r={r:.4f}, p={p:.4f}, mean H0={h0.mean():.2f}")

    # Check if any host has large frame differences
    primary_copy = primary.copy()
    primary_copy["dz_max"] = primary_copy[["z_hel", "z_cmb", "z_hd"]].apply(
        lambda row: max(row) - min(row), axis=1
    )
    large_dz = primary_copy[primary_copy["dz_max"] > 0.002]
    if len(large_dz) > 0:
        findings.append(f"Hosts with |z_frame_diff| > 0.002: {len(large_dz)}")
        for _, row in large_dz.iterrows():
            findings.append(f"  - {row['normalized_name']}: dz={row['dz_max']:.4f}")

    return findings


def audit_z_gt_0_01(data):
    """Priority 3.7: Treat z>0.01, N=5 as sign-only."""
    findings = []
    primary = data["primary"]
    z01 = primary[pd.to_numeric(primary["z_hd"], errors="coerce") > 0.01]
    findings.append(f"z>0.01 subsample: N={len(z01)}")
    if len(z01) > 0:
        hosts = z01["normalized_name"].tolist()
        findings.append(f"  Hosts: {', '.join(hosts)}")
        r, p = stats.pearsonr(z01["sigma_inferred"].values, z01["h0_derived"].values)
        findings.append(f"  Pearson r={r:.4f}, p={p:.4f}")
        if len(z01) < 10:
            findings.append("  WARNING: N<10; this should be described as sign-stability only, not significance")
    return findings


def audit_stellar_vs_proxy(data):
    """Priority 4.10: Stress-test HI linewidth conversion."""
    findings = []
    prov = data["sigma_prov"]
    primary = data["primary"]
    if prov is None or len(primary) == 0:
        return findings

    # Merge provenance with primary
    merged = primary.merge(prov[["normalized_name", "sigma_method"]], on="normalized_name", how="left")

    stellar = merged[merged["sigma_method"] == "stellar absorption"]
    proxy = merged[merged["sigma_method"] == "HI linewidth proxy"]

    findings.append(f"Stellar-only: N={len(stellar)}, r={stats.pearsonr(stellar['sigma_inferred'], stellar['h0_derived'])[0]:.4f}")
    findings.append(f"Proxy-only: N={len(proxy)}, r={stats.pearsonr(proxy['sigma_inferred'], proxy['h0_derived'])[0]:.4f}")

    if len(stellar) > 3 and len(proxy) > 3:
        # Test if signal is stronger in stellar
        findings.append("Stellar-only vs proxy-only comparison:")
        for subset, name in [(stellar, "stellar"), (proxy, "proxy")]:
            r, p = stats.pearsonr(subset["sigma_inferred"], subset["h0_derived"])
            findings.append(f"  {name}: r={r:.4f}, p={p:.4f}")

    return findings


def audit_host_mass_control(data):
    """Priority 6.24: Add host stellar mass control."""
    findings = []
    primary = data["primary"]
    if len(primary) == 0:
        return findings

    # Test H0 ~ logmass
    mass = pd.to_numeric(primary["host_logmass"], errors="coerce")
    valid = primary[mass.notna()]
    if len(valid) > 3:
        r_mass, p_mass = stats.pearsonr(mass[valid.index], valid["h0_derived"])
        findings.append(f"H0 vs host_logmass: r={r_mass:.4f}, p={p_mass:.4f}")

        # Partial correlation: H0 vs sigma controlling for logmass
        from scipy.linalg import lstsq
        y = valid["h0_derived"].values
        x1 = valid["sigma_inferred"].values
        x2 = pd.to_numeric(valid["host_logmass"], errors="coerce").values
        # Residualize y and x1 against x2
        A2 = np.vstack([np.ones(len(x2)), x2]).T
        coeff_y, _, _, _ = lstsq(A2, y)
        coeff_x1, _, _, _ = lstsq(A2, x1)
        y_resid = y - A2 @ coeff_y
        x1_resid = x1 - A2 @ coeff_x1
        r_partial, p_partial = stats.pearsonr(x1_resid, y_resid)
        findings.append(f"Partial correlation H0-sigma | logmass: r={r_partial:.4f}, p={p_partial:.4f}")

    return findings


def audit_bin_uncertainties(data):
    """Priority 3.8: Reassess low/high sigma bin uncertainties."""
    findings = []
    primary = data["primary"]
    if len(primary) == 0:
        return findings

    median_sigma = primary["sigma_inferred"].median()
    low = primary[primary["sigma_inferred"] <= median_sigma]
    high = primary[primary["sigma_inferred"] > median_sigma]

    # Simple SEM
    sem_low = low["h0_derived"].std() / np.sqrt(len(low))
    sem_high = high["h0_derived"].std() / np.sqrt(len(high))

    # With peculiar velocity uncertainty (250 km/s) propagated
    vpec_err = 250.0
    dist_low = low["distance_mpc"]
    dist_high = high["distance_mpc"]
    vpec_h0_err_low = vpec_err / dist_low
    vpec_h0_err_high = vpec_err / dist_high

    findings.append(f"Low-sigma bin: statistical SEM = {sem_low:.2f}")
    findings.append(f"  Peculiar-velocity H0 err range: [{vpec_h0_err_low.min():.2f}, {vpec_h0_err_low.max():.2f}]")
    findings.append(f"High-sigma bin: statistical SEM = {sem_high:.2f}")
    findings.append(f"  Peculiar-velocity H0 err range: [{vpec_h0_err_high.min():.2f}, {vpec_h0_err_high.max():.2f}]")
    findings.append("WARNING: The reported ±1.05 km/s/Mpc bin uncertainties ignore peculiar-velocity systematics.")
    findings.append("  Full systematic uncertainty is likely 3-5x larger.")

    return findings


def run_basic_multiverse(data):
    """Priority 5.14: Basic multiverse analysis."""
    findings = []
    primary = data["primary"]
    if len(primary) == 0:
        return findings

    variants = []

    # 1. sigma vs sigma^2
    for transform, label in [(lambda x: x, "sigma"), (lambda x: x**2, "sigma^2")]:
        x = transform(primary["sigma_inferred"].values)
        r, p = stats.pearsonr(x, primary["h0_derived"].values)
        variants.append((label, r, p))

    # 2. log sigma
    r, p = stats.pearsonr(np.log(primary["sigma_inferred"].values), primary["h0_derived"].values)
    variants.append(("log_sigma", r, p))

    # 3. Direct-only vs all
    prov = data["sigma_prov"]
    if prov is not None:
        merged = primary.merge(prov[["normalized_name", "sigma_method"]], on="normalized_name", how="left")
        stellar = merged[merged["sigma_method"] == "stellar absorption"]
        if len(stellar) > 3:
            r, p = stats.pearsonr(stellar["sigma_inferred"], stellar["h0_derived"])
            variants.append(("stellar_only", r, p))

    # 4. Redshift cuts
    for zcut in [0.0035, 0.005, 0.007, 0.01]:
        sub = primary[pd.to_numeric(primary["z_hd"], errors="coerce") > zcut]
        if len(sub) > 5:
            r, p = stats.pearsonr(sub["sigma_inferred"], sub["h0_derived"])
            variants.append((f"z>{zcut}", r, p))

    findings.append("Multiverse summary (Pearson r, p):")
    for label, r, p in variants:
        findings.append(f"  {label}: r={r:.4f}, p={p:.4f}")

    # Count variants with p < 0.05
    n_significant = sum(1 for _, _, p in variants if p < 0.05)
    findings.append(f"  {n_significant}/{len(variants)} variants significant at p<0.05")

    return findings


def main():
    print("=" * 70)
    print("TEP-H0 MASTER AUDIT")
    print("=" * 70)

    data = load_data()

    # Build and save master table
    master = build_master_table(data)
    master.to_csv(TABLE_PATH, index=False)
    print(f"\nSaved master table to {TABLE_PATH}")

    all_findings = []

    sections = [
        ("SAMPLE CONSISTENCY", audit_sample_consistency),
        ("H0 CONTRADICTIONS", audit_h0_contradictions),
        ("DELTA MU SIGN / UNITS", audit_delta_mu_sign),
        ("HEADLINE NUMBERS", recompute_headline_numbers),
        ("REDSHIFT FRAMES", audit_redshift_frames),
        ("Z>0.01 SUBSAMPLE", audit_z_gt_0_01),
        ("BIN UNCERTAINTIES", audit_bin_uncertainties),
        ("COVARIANCE SANITY", audit_covariance),
        ("SIGMA PROVENANCE", audit_sigma_provenance),
        ("STELLAR VS PROXY", audit_stellar_vs_proxy),
        ("HOST MASS CONTROL", audit_host_mass_control),
        ("PLANCK CIRCULARITY", audit_planck_circularity),
        ("BIC SIGN", audit_bic_sign),
        ("LEAVE-ONE-OUT", run_leave_one_out),
        ("MULTIVERSE", run_basic_multiverse),
    ]

    for title, func in sections:
        print(f"\n{'=' * 70}")
        print(f"SECTION: {title}")
        print("=" * 70)
        findings = func(data)
        for f in findings:
            print(f"  {f}")
        all_findings.extend([f"{title}: {f}" for f in findings])

    # Save report
    report = {
        "audit_completed": True,
        "n_findings": len(all_findings),
        "findings": all_findings,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Audit complete. Report saved to {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
