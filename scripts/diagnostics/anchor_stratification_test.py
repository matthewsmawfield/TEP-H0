#!/usr/bin/env python3
"""
Anchor Stratification Test: Check for Internal P-L Tension Between Calibrators

This script tests whether the three SH0ES geometric anchors (MW, LMC, NGC 4258)
show systematic P-L zero-point differences that correlate with their velocity
dispersions - a prediction of TEP if the anchors themselves are affected.

If TEP is real:
- LMC (σ = 24 km/s): Should yield BRIGHTEST zero-point (shallowest potential)
- MW (σ = 30 km/s): Intermediate
- NGC 4258 (σ = 115 km/s): Should yield FAINTEST zero-point (deepest potential)

The test fits independent P-L relations to each anchor and compares zero-points.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.constants import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY
from scripts.utils.plot_style import apply_tep_style
from scripts.utils.logger import print_status, print_table

# Constants
ANCHOR_SIGMA = {
    'N4258': 115.0,   # NGC 4258: maser host, deep potential
    'LMC': 24.0,      # LMC: low-mass satellite
    'MW': 30.0,       # Milky Way thin disk
    'M31': 160.0,     # M31: massive spiral, deepest potential
}

# Known distance moduli for anchors (geometric distances)
ANCHOR_MU = {
    'N4258': 29.397,  # Maser distance (Humphreys+2013, Reid+2019)
    'LMC': 18.477,    # DEBs (Pietrzynski+2019)
    'MW': 0.0,        # Parallax (by definition)
    'M31': 24.407,    # Cepheid/TRGB consensus (Riess+2012, de Grijs+2014)
}


def load_cepheid_data():
    """Load reconstructed SH0ES Cepheid data."""
    data_path = Path(__file__).parent.parent.parent / "data" / "interim" / "reconstructed_shoes_cepheids.csv"
    
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return None
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded {len(df)} Cepheid measurements", "INFO")
    
    # L_col_bW contains (log P - 1), so log P = L_col_bW + 1
    df['log_P'] = df['L_col_bW'] + 1.0
    df['W'] = df['Data']  # Wesenheit magnitude
    
    return df


def extract_anchor_samples(df):
    """Extract Cepheid samples for each geometric anchor."""
    anchors = {}
    
    # NGC 4258
    mask_n4258 = df['Source'] == 'N4258'
    anchors['N4258'] = df[mask_n4258].copy()
    
    # LMC (both ground and HST)
    mask_lmc = df['Source'].str.startswith('LMC')
    anchors['LMC'] = df[mask_lmc].copy()
    
    # M31 - adds crucial leverage with highest σ
    mask_m31 = df['Source'] == 'M31'
    anchors['M31'] = df[mask_m31].copy()
    
    # MW would require parallax data which is handled differently in SH0ES
    # For now, we'll work with N4258, LMC, and M31
    
    print_status("Anchor Sample Sizes:", "SECTION")
    headers = ["Anchor", "N_Cepheids", "σ (km/s)", "μ_geo (mag)"]
    rows = []
    for name, sample in anchors.items():
        if name in ANCHOR_SIGMA:
            rows.append([name, len(sample), f"{ANCHOR_SIGMA[name]:.0f}", f"{ANCHOR_MU[name]:.3f}"])
    print_table(headers, rows)
    
    return anchors


def fit_pl_relation(log_p, W, W_err=None):
    """
    Fit P-L relation: W = M_W + b_W * (log P - 1)
    
    Returns:
        M_W: Zero-point (absolute magnitude at log P = 1)
        b_W: Slope
        M_W_err: Uncertainty on zero-point
        b_W_err: Uncertainty on slope
    """
    # Design matrix: [1, (log P - 1)]
    X = np.column_stack([np.ones_like(log_p), log_p - 1.0])
    if W_err is None:
        # Unweighted OLS
        beta, residuals, rank, s = np.linalg.lstsq(X, W, rcond=None)

        # Estimate uncertainties from residuals
        n = len(W)
        p = 2
        if n > p:
            sigma2 = np.sum((W - X @ beta)**2) / (n - p)
            XtX = X.T @ X
            reg = 1e-10 * np.trace(XtX) / XtX.shape[0]
            XtX_reg = XtX + reg * np.eye(XtX.shape[0])
            try:
                cov = sigma2 * np.linalg.inv(XtX_reg)
                se = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                se = np.full(X.shape[1], np.nan)
        else:
            se = [np.nan, np.nan]
    else:
        # Weighted least squares
        W_inv = np.diag(1.0 / W_err**2)
        XtWX = X.T @ W_inv @ X
        XtWy = X.T @ W_inv @ W
        reg = 1e-10 * np.trace(XtWX) / XtWX.shape[0]
        XtWX_reg = XtWX + reg * np.eye(XtWX.shape[0])
        try:
            beta = np.linalg.solve(XtWX_reg, XtWy)
            cov = np.linalg.inv(XtWX_reg)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            beta = np.array([np.nan, np.nan])
            se = np.full(X.shape[1], np.nan)
    
    M_W, b_W = beta
    M_W_err, b_W_err = se
    
    return M_W, b_W, M_W_err, b_W_err


def compute_absolute_zeropoint(anchor_name, M_W_apparent):
    """
    Convert apparent zero-point to absolute using known distance modulus.
    
    M_W_absolute = M_W_apparent - μ_geo
    """
    mu_geo = ANCHOR_MU[anchor_name]
    return M_W_apparent - mu_geo


def run_anchor_stratification_test():
    """Main test: fit P-L independently to each anchor and compare zero-points."""
    print_status("=" * 70, "INFO")
    print_status("ANCHOR STRATIFICATION TEST", "SECTION")
    print_status("Testing for internal P-L tension between geometric calibrators", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = load_cepheid_data()
    if df is None:
        return None
    
    # Extract anchor samples
    anchors = extract_anchor_samples(df)
    
    # Fit P-L relation to each anchor independently
    print_status("Independent P-L Fits:", "SECTION")
    
    results = {}
    for name, sample in anchors.items():
        if len(sample) < 10:
            print_status(f"Skipping {name}: insufficient data (N={len(sample)})", "WARNING")
            continue
        
        # Fit P-L
        M_W, b_W, M_W_err, b_W_err = fit_pl_relation(
            sample['log_P'].values,
            sample['W'].values
        )
        
        # Convert to absolute magnitude
        M_W_abs = compute_absolute_zeropoint(name, M_W)
        
        results[name] = {
            'N': len(sample),
            'sigma': ANCHOR_SIGMA[name],
            'mu_geo': ANCHOR_MU[name],
            'M_W_apparent': M_W,
            'M_W_absolute': M_W_abs,
            'M_W_err': M_W_err,
            'b_W': b_W,
            'b_W_err': b_W_err,
            'log_P_mean': sample['log_P'].mean(),
            'log_P_std': sample['log_P'].std(),
        }
    
    # Display results table
    headers = ["Anchor", "N", "σ (km/s)", "M_W (abs)", "± err", "b_W (slope)", "± err"]
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            r['N'],
            f"{r['sigma']:.0f}",
            f"{r['M_W_absolute']:.3f}",
            f"{r['M_W_err']:.3f}",
            f"{r['b_W']:.3f}",
            f"{r['b_W_err']:.3f}"
        ])
    print_table(headers, rows, title="Independent P-L Fit Results")
    
    # Test for correlation between M_W and σ
    if len(results) >= 2:
        sigmas = np.array([r['sigma'] for r in results.values()])
        M_Ws = np.array([r['M_W_absolute'] for r in results.values()])
        M_W_errs = np.array([r['M_W_err'] for r in results.values()])
        names = list(results.keys())
        
        print_status("Zero-Point vs Velocity Dispersion:", "SECTION")
        
        # ===== MULTI-ANCHOR REGRESSION (sigma^2/c^2 form) =====
        # Physics-derived form consistent with TEP correction:
        # M_W = a + κ_anchor * (σ^2 - σ_ref^2)/c^2
        # This directly tests if anchor zero-points correlate with potential depth
        
        sigma_ref = 87.17  # Same reference as main analysis
        c_km_s = 299792.458
        # Physics-derived regressor: (sigma^2 - sigma_ref^2)/c^2 (matches step_3 correction form)
        sigma_regressor = (sigmas**2 - sigma_ref**2) / c_km_s**2
        
        # Weighted linear regression
        weights = 1.0 / M_W_errs**2
        
        # Design matrix
        X = np.column_stack([np.ones_like(sigma_regressor), sigma_regressor])
        W_mat = np.diag(weights)
        
        # Weighted least squares
        XtWX = X.T @ W_mat @ X
        XtWy = X.T @ W_mat @ M_Ws
        reg = 1e-10 * np.trace(XtWX) / XtWX.shape[0]
        XtWX_reg = XtWX + reg * np.eye(XtWX.shape[0])
        try:
            beta = np.linalg.solve(XtWX_reg, XtWy)
            cov = np.linalg.inv(XtWX_reg)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            beta = np.array([np.nan, np.nan])
            se = np.array([np.nan, np.nan])
        
        intercept, kappa_anchor = beta
        intercept_err, kappa_anchor_err = se
        
        # Compute chi-squared and p-value
        residuals = M_Ws - (intercept + kappa_anchor * sigma_regressor)
        chi2 = np.sum((residuals / M_W_errs)**2)
        dof = len(M_Ws) - 2
        
        # Correlation coefficient (using sigma_regressor for physics consistency)
        r_pearson, p_pearson = stats.pearsonr(sigma_regressor, M_Ws)
        
        print_status(f"Multi-Anchor Regression (N={len(names)} systems):", "SECTION")
        print_status(f"  κ_anchor = {kappa_anchor:.3e} ± {kappa_anchor_err:.3e} mag", "INFO")
        if kappa_anchor_err > 0:
            print_status(f"  Significance: {abs(kappa_anchor)/kappa_anchor_err:.1f}σ", "INFO")
        print_status(f"  Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})", "INFO")
        print_status(f"  χ²/dof = {chi2:.2f}/{dof}", "INFO")
        
        # Compare with host κ_Cep from pipeline output
        kappa_host = np.nan
        kappa_host_err = np.nan
        try:
            project_root = Path(__file__).parent.parent.parent
            tep_path = project_root / "results" / "outputs" / "step_04_tep_correction_results.json"
            if tep_path.exists():
                with open(tep_path, "r") as f:
                    tep = json.load(f)
                if isinstance(tep, dict) and 'optimal_kappa_cep' in tep:
                    kappa_host = float(tep['optimal_kappa_cep'])
                if isinstance(tep, dict):
                    kappa_host_err = float(
                        tep.get('bootstrap_kappa_robust_std')
                        or tep.get('wls_kappa_err_scaled')
                        or tep.get('bootstrap_kappa_std', 8.9e5)
                    )
        except Exception:
            pass
        
        if np.isfinite(kappa_host) and np.isfinite(kappa_host_err) and np.isfinite(kappa_anchor_err):
            tension = abs(kappa_anchor - kappa_host) / np.sqrt(kappa_anchor_err**2 + kappa_host_err**2)
            print_status(f"  κ_Cep (hosts) = {kappa_host:.3e} ± {kappa_host_err:.3e} mag", "INFO")
            print_status(f"  Tension with host κ_Cep: {tension:.1f}σ", "INFO")
        else:
            print_status(f"  Note: Host κ_Cep comparison unavailable", "INFO")
        
        # Pairwise comparisons
        print_status("Pairwise Comparisons:", "SECTION")
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:
                    delta_MW = results[name1]['M_W_absolute'] - results[name2]['M_W_absolute']
                    delta_MW_err = np.sqrt(results[name1]['M_W_err']**2 + results[name2]['M_W_err']**2)
                    sig = abs(delta_MW) / delta_MW_err
                    print_status(f"  Δ M_W ({name1} - {name2}) = {delta_MW:+.3f} ± {delta_MW_err:.3f} mag ({sig:.1f}σ)", "INFO")
        
        results['regression'] = {
            'kappa_anchor': float(kappa_anchor),
            'kappa_anchor_err': float(kappa_anchor_err),
            'intercept': float(intercept),
            'intercept_err': float(intercept_err),
            'r_pearson': float(r_pearson),
            'p_pearson': float(p_pearson),
            'chi2': float(chi2),
            'dof': int(dof),
            'n_anchors': len(names),
            'kappa_host': float(kappa_host) if np.isfinite(kappa_host) else None,
            'kappa_host_err': float(kappa_host_err) if np.isfinite(kappa_host_err) else None,
            'tension_with_host': float(tension) if np.isfinite(tension) else None,
        }
        
        # Pairwise comparison: N4258 vs LMC (largest sigma contrast)
        if 'N4258' in results and 'LMC' in results:
            delta_MW = results['N4258']['M_W_absolute'] - results['LMC']['M_W_absolute']
            delta_MW_err = np.sqrt(results['N4258']['M_W_err']**2 + results['LMC']['M_W_err']**2)
            # Sigma^2/c^2 difference for this pair
            delta_sigma_sq = (ANCHOR_SIGMA['N4258']**2 - ANCHOR_SIGMA['LMC']**2) / c_km_s**2
            # Expected delta from host kappa_Cep if anchors were affected
            if np.isfinite(kappa_host):
                expected_delta = kappa_host * delta_sigma_sq
            else:
                expected_delta = None
            
            results['test'] = {
                'delta_MW': float(delta_MW),
                'delta_MW_err': float(delta_MW_err),
                'delta_sigma_sq_over_c2': float(delta_sigma_sq),
                'significance': float(abs(delta_MW) / delta_MW_err),
                'tep_prediction_host_kappa': float(expected_delta) if expected_delta is not None else None,
            }
    
    # Create figure
    create_anchor_comparison_figure(results)
    
    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "outputs" / "step_27_anchor_stratification_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    output_data = {}
    for k, v in results.items():
        if isinstance(v, dict):
            output_data[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                            for kk, vv in v.items()}
        else:
            output_data[k] = v
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print_status(f"Results saved to: {output_path}", "SUCCESS")
    
    return results


def create_anchor_comparison_figure(results):
    """Create visualization of anchor P-L comparison."""
    apply_tep_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Zero-point vs σ
    ax1 = axes[0]
    
    sigmas = []
    M_Ws = []
    M_W_errs = []
    names = []
    
    for name, r in results.items():
        # Skip non-anchor entries
        if name in ['test', 'regression']:
            continue
        if 'sigma' not in r:
            continue
        sigmas.append(r['sigma'])
        M_Ws.append(r['M_W_absolute'])
        M_W_errs.append(r['M_W_err'])
        names.append(name)
    
    sigmas = np.array(sigmas)
    M_Ws = np.array(M_Ws)
    M_W_errs = np.array(M_W_errs)
    
    ax1.errorbar(sigmas, M_Ws, yerr=M_W_errs, fmt='o', markersize=12, 
                 capsize=5, capthick=2, color='#2E86AB', ecolor='#2E86AB',
                 label='Anchor Zero-Points')
    
    for i, name in enumerate(names):
        ax1.annotate(name, (sigmas[i], M_Ws[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    ax1.set_xlabel(r'Velocity Dispersion $\sigma$ (km/s)', fontsize=14)
    ax1.set_ylabel(r'P-L Zero-Point $M_W$ (mag)', fontsize=14)
    ax1.set_title('Anchor Zero-Points vs Environment', fontsize=14, fontweight='bold')
    
    # Add TEP prediction line if we have the test results
    if 'test' in results and len(sigmas) >= 2:
        sigma_range = np.linspace(min(sigmas)*0.8, max(sigmas)*1.2, 100)
        # Use LMC as reference
        lmc_idx = names.index('LMC') if 'LMC' in names else 0
        # Load fitted kappa from step_04_tep_correction_results.json
        import json
        json_path = Path(__file__).resolve().parents[2] / "results" / "outputs" / "step_04_tep_correction_results.json"
        if json_path.exists():
            with open(json_path) as f:
                tep_results = json.load(f)
            kappa_cep_ref = float(tep_results.get("optimal_kappa_cep", KAPPA_GAL))
        else:
            kappa_cep_ref = KAPPA_GAL
        sigma_ref = sigmas[lmc_idx]
        M_W_ref = M_Ws[lmc_idx]
        
        M_W_pred = M_W_ref + kappa_cep_ref * ((sigma_range**2 - sigma_ref**2) / 299792.458**2)
        kappa_mantissa = kappa_cep_ref / 1e5
        ax1.plot(sigma_range, M_W_pred, '--', color='#C73E1D', alpha=0.7,
                label=rf'TEP prediction ($\kappa_{{Cep}} = {kappa_mantissa:.2f}\times 10^5$)')
        ax1.legend(fontsize=11)
    
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Slope comparison
    ax2 = axes[1]
    
    x_pos = np.arange(len(names))
    slopes = [results[n]['b_W'] for n in names]
    slope_errs = [results[n]['b_W_err'] for n in names]
    
    bars = ax2.bar(x_pos, slopes, yerr=slope_errs, capsize=5, 
                   color=['#2E86AB', '#A23B72'], alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    # Add SH0ES global slope reference
    shoes_bw = -3.299  # From Riess+2022
    ax2.axhline(shoes_bw, color='#C73E1D', linestyle='--', linewidth=2,
               label=f'SH0ES Global: $b_W = {shoes_bw}$')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'P-L Slope $b_W$', fontsize=14)
    ax2.set_title('Independent P-L Slopes', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Path(__file__).parent.parent.parent / "results" / "figures" / "step_27_anchor_stratification_test.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print_status(f"Figure saved to: {fig_path}", "SUCCESS")
    
    plt.close()


if __name__ == "__main__":
    results = run_anchor_stratification_test()
    
    if results and 'test' in results:
        print_status("=" * 70, "INFO")
        print_status("INTERPRETATION", "SECTION")
        print_status("=" * 70, "INFO")
        
        t = results['test']
        if t['significance'] > 2.0:
            print_status(f"SIGNIFICANT internal tension detected ({t['significance']:.1f}σ)", "WARNING")
            print_status("This suggests the anchors may themselves be affected by TEP.", "INFO")
            print_status("The current σ_ref calculation partially accounts for this.", "INFO")
        else:
            print_status(f"No significant internal tension ({t['significance']:.1f}σ)", "SUCCESS")
            print_status("Anchors are consistent with a common P-L relation.", "INFO")
            print_status("The referee concern is addressed: anchor bias is subdominant.", "INFO")
