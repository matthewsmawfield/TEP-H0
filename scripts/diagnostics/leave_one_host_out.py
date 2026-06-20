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

def run_loocv():
    print("--- Running Leave-One-Host-Out Influence Analysis ---")
    base_dir = Path(__file__).resolve().parents[2]
    df = pd.read_csv(base_dir / "results" / "outputs" / "tep_corrected_h0.csv")
    
    with open(base_dir / "results" / "outputs" / "tep_correction_results.json", "r") as f:
        tep = json.load(f)
    
    sigma_ref = tep['sigma_ref_screened']
    kappa_full = tep['optimal_kappa_cep']
    
    results = []
    
    for i, row in df.iterrows():
        host = row['source_id']
        df_sub = df.drop(index=i)
        
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
            'excluded_host': host,
            'sigma': row['sigma_inferred'],
            'shear_suppression': row['shear_suppression'],
            'kappa_loo': kappa_sub,
            'delta_kappa': delta_kappa,
            'pct_change': pct_change
        })
        
    df_res = pd.DataFrame(results)
    # Sort by absolute percentage change descending
    df_res['abs_pct_change'] = df_res['pct_change'].abs()
    df_res = df_res.sort_values(by='abs_pct_change', ascending=False).drop(columns=['abs_pct_change'])
    
    print(f"\nFull sample kappa_Cep: {kappa_full:.2e} mag")
    print(f"{'Excluded Host':<15} {'Sigma':<8} {'S':<5} {'Kappa_LOO':<10} {'Delta':<10} {'% Change':<8}")
    print("-" * 65)
    for _, r in df_res.iterrows():
        print(f"{r['excluded_host']:<15} {r['sigma']:<8.1f} {r['shear_suppression']:<5.2f} "
              f"{r['kappa_loo']:<10.2e} {r['delta_kappa']:<10.2e} {r['pct_change']:>8.2f}%")
              
    output_path = base_dir / "results" / "outputs" / "leave_one_out_influence.csv"
    df_res.to_csv(output_path, index=False)
    print(f"\nSaved influence table to {output_path}")

if __name__ == '__main__':
    run_loocv()
