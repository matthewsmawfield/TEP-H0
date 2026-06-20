#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import sys
import json

# Add root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.tep_correction import tep_correction

def run_redshift_sensitivity():
    print("--- Running Boundary-Redshift Sensitivity Analysis ---")
    base_dir = Path(__file__).resolve().parents[2]
    df = pd.read_csv(base_dir / "results" / "outputs" / "step_04_tep_corrected_h0.csv")
    
    with open(base_dir / "results" / "outputs" / "step_04_tep_correction_results.json", "r") as f:
        tep = json.load(f)
    
    sigma_ref = tep['sigma_ref_screened']
    kappa_full = tep['optimal_kappa_cep']
    
    # Redshift cuts from 0.0 to 0.010 in steps of 0.001
    # Plus standard cuts 0.0035 and 0.007
    z_cuts = sorted(list(np.arange(0.000, 0.0101, 0.001)) + [0.0035, 0.007])
    
    results = []
    
    for z_cut in z_cuts:
        df_sub = df[df['z_hd'] >= z_cut]
        N_hosts = len(df_sub)
        
        if N_hosts < 5:
            # Too few hosts to fit reasonably
            results.append({
                'z_min': z_cut,
                'N_hosts': N_hosts,
                'kappa_cep': np.nan,
                'pct_change': np.nan
            })
            continue
            
        sigma_vals = df_sub["sigma_inferred"].values
        S = df_sub["shear_suppression"].values
        mu_obs = df_sub["value"].values
        v_obs = df_sub["velocity"].values
        
        def objective(params):
            kappa_cep = params[0]
            correction = tep_correction(sigma_vals, sigma_ref, kappa_cep, S)
            mu_corr = mu_obs + correction
            mu_fid = 5 * np.log10(v_obs) + 25 - 5 * np.log10(70.0)
            delta_mu = mu_corr - mu_fid
            slope, _ = np.polyfit(sigma_vals, delta_mu, 1)
            return slope**2
            
        res = minimize(objective, x0=[kappa_full], method="Nelder-Mead", options={"xatol": 1.0, "fatol": 1e-8, "maxiter": 2000})
        kappa_sub = res.x[0]
        
        delta_kappa = kappa_sub - kappa_full
        pct_change = (delta_kappa / kappa_full) * 100
        
        results.append({
            'z_min': z_cut,
            'N_hosts': N_hosts,
            'kappa_cep': kappa_sub,
            'pct_change': pct_change
        })
        
    df_res = pd.DataFrame(results)
    
    print(f"\nFull sample kappa_Cep: {kappa_full:.2e} mag")
    print(f"{'z_min':<10} {'N_hosts':<10} {'Kappa_Cep':<12} {'% Change':<10}")
    print("-" * 50)
    for _, r in df_res.iterrows():
        k = f"{r['kappa_cep']:.2e}" if pd.notnull(r['kappa_cep']) else "N/A"
        pct = f"{r['pct_change']:>8.2f}%" if pd.notnull(r['pct_change']) else "N/A"
        print(f"{r['z_min']:<10.4f} {r['N_hosts']:<10} {k:<12} {pct:<10}")
              
    output_path = base_dir / "results" / "outputs" / "step_08_redshift_cut_sensitivity.csv"
    df_res.to_csv(output_path, index=False)
    print(f"\nSaved boundary-redshift table to {output_path}")

if __name__ == '__main__':
    run_redshift_sensitivity()
