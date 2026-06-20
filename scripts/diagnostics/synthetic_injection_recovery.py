#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Fix path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def linear_func(p, x):
    return p[0] * x + p[1]

def run_synthetic_injection():
    print("--- Running Synthetic Injection Test (ODR Slope Anomaly) ---")
    np.random.seed(42)
    N = 29
    c2 = 299792.458**2

    # Draw true sigma from uniform 50-200 km/s, true H0 = 68.0 km/s/Mpc
    sigma_true = np.random.uniform(50, 200, N)
    sigma_ref = 87.17
    H0_true = 68.0

    # Assume distance d ~ 10-40 Mpc. The exact distance matters less, let's use z = 0.0035 to 0.015
    z_true = np.random.uniform(0.0035, 0.015, N)
    d_true = (299792.458 * z_true) / H0_true
    mu_true = 5 * np.log10(d_true) + 25

    # Errors: sigma_err ~ 15 km/s, mu_err ~ 0.15 mag
    sigma_err = np.random.uniform(5, 25, N)
    mu_err = np.random.uniform(0.10, 0.20, N)

    kappa_injections = [0.0, 0.5e6, 1.0e6, 1.5e6]
    
    results = []
    
    for kappa in kappa_injections:
        # Inject TEP bias into distance modulus. 
        # In TEP, high potential depth -> clock contracts -> inferred distance is smaller.
        # Thus mu_obs = mu_true - delta_mu_bias
        delta_mu_bias = kappa * (sigma_true**2 - sigma_ref**2) / c2
        
        mu_obs_correct = mu_true - delta_mu_bias + np.random.normal(0, mu_err, N)
        sigma_obs = sigma_true + np.random.normal(0, sigma_err, N)
        
        v_obs = 299792.458 * z_true + np.random.normal(0, 200, N)
        d_obs_correct = 10**((mu_obs_correct - 25) / 5)
        H0_obs_correct = v_obs / d_obs_correct
        
        # H0 error roughly propagated from mu_err and v_err
        H0_err_c = H0_obs_correct * np.sqrt((mu_err * np.log(10) / 5)**2 + (200 / v_obs)**2)
        
        # 1. OLS
        slope_ols_c, _ = np.polyfit(sigma_obs, H0_obs_correct, 1)
        
        # 2. ODR
        linear_model = Model(linear_func)
        data_c = RealData(sigma_obs, H0_obs_correct, sx=sigma_err, sy=H0_err_c)
        odr_c = ODR(data_c, linear_model, beta0=[slope_ols_c, np.mean(H0_obs_correct)])
        out_c = odr_c.run()
        slope_odr_c = out_c.beta[0]
        
        expected_slope = H0_true * (np.log(10)/5) * kappa * 2 * np.mean(sigma_true) / c2

        results.append({
            "kappa": kappa,
            "expected_slope": expected_slope,
            "slope_ols": slope_ols_c,
            "slope_odr": slope_odr_c,
            "odr_ols_ratio": slope_odr_c / slope_ols_c if slope_ols_c != 0 else np.nan
        })

        print(f"Injection: kappa = {kappa:.1e}")
        print(f"  Expected H0 slope : {expected_slope:.4f}")
        print(f"  Recovered OLS     : {slope_ols_c:.4f}")
        print(f"  Recovered ODR     : {slope_odr_c:.4f} (Ratio ODR/OLS: {slope_odr_c/slope_ols_c:.2f})")
        print("-" * 40)
        
    df_res = pd.DataFrame(results)
    output_path = Path(__file__).resolve().parents[2] / "results" / "outputs" / "synthetic_odr_recovery.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df_res.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    run_synthetic_injection()
