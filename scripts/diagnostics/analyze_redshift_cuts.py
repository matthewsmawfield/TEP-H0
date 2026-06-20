import pandas as pd
import numpy as np
from pathlib import Path
import sys
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.steps.step_04_tep_correction import Step3TEPCorrection
from scripts.steps.step_08_robustness_checks import Step4RobustnessChecks
from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_correction import C_SQUARED_KM_S

def run_cut(name, df_subset):
    print(f"\n--- Running cut: {name} (N={len(df_subset)}) ---")
    
    if len(df_subset) < 10:
        print("Sample too small.")
        return None
        
    temp_path = PROJECT_ROOT / "results" / "outputs" / f"temp_stratified_h0_{name.replace(' ', '_')}.csv"
    df_subset.to_csv(temp_path, index=False)
    
    step3 = Step3TEPCorrection()
    step3.logger = TEPLogger(f"temp_step3_{name.replace(' ', '_')}")
    set_step_logger(step3.logger)
    step3.input_path = temp_path
    
    df_loaded = step3.load_data()
    sigma_ref, sigma_ref_screened = step3.calculate_effective_calibrator_sigma()
    
    kappa = step3.optimize_correction(df_loaded, sigma_ref_screened)
    df_corr, _, _ = step3.apply_correction(df_loaded, kappa, sigma_ref_screened)
    unified_h0 = df_corr['h0_corrected'].mean()
    
    # Manually run LOOCV
    n = len(df_loaded)
    loocv_preds = []
    
    for i in range(n):
        train = df_loaded.drop(index=i)
        test = df_loaded.iloc[[i]].copy()
        kappa_train = step3.optimize_correction(train, sigma_ref_screened)
        
        S = test["shear_suppression"].values[0]
        sigma = test["sigma_inferred"].values[0]
        mu_raw = test["value"].values[0]
        vel = test["velocity"].values[0]
        
        mu_corr = mu_raw + S * kappa_train * (sigma**2 - sigma_ref_screened**2) / C_SQUARED_KM_S
        d_corr = 10 ** ((mu_corr - 25) / 5)
        h0_corr = vel / d_corr
        
        loocv_preds.append(h0_corr)
    
    loocv_h0 = np.mean(loocv_preds)
    
    # Manually run BIC
    from scipy import stats
    df_temp = df_corr.copy()
    S = df_temp["shear_suppression"].values
    x = S * (df_temp["sigma_inferred"].values**2 - sigma_ref_screened**2) / C_SQUARED_KM_S
    h0 = df_temp["h0_derived"].values
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, h0)
    rss_model = np.sum((h0 - (intercept + slope * x))**2)
    rss_null = np.sum((h0 - np.mean(h0))**2)
    
    bic_null = n * np.log(rss_null / n) + 1 * np.log(n)
    bic_model = n * np.log(rss_model / n) + 2 * np.log(n)
    delta_bic = bic_model - bic_null
    
    print(f"Corrected H0: {unified_h0:.2f}")
    print(f"kappa_cep:    {kappa/1e6:.4f} * 10^6")
    print(f"LOOCV H0:     {loocv_h0:.2f}")
    print(f"Delta BIC:    {delta_bic:.2f} (negative means TEP favoured)")
    
    return {
        "Name": name,
        "N": len(df_subset),
        "Corrected_H0": unified_h0,
        "Kappa_1e6": kappa/1e6,
        "LOOCV_H0": loocv_h0,
        "Delta_BIC": delta_bic
    }

def main():
    stratified_path = PROJECT_ROOT / "results" / "outputs" / "stratified_h0.csv"
    if not stratified_path.exists():
        print("Run step 2 first.")
        return
        
    df = pd.read_csv(stratified_path)
    
    results = []
    
    results.append(run_cut("Full Sample (N=36)", df))
    
    df_29 = df[df['z_hd'] > 0.0035].copy().reset_index(drop=True)
    results.append(run_cut("Hubble-flow (z>0.0035, N=29)", df_29))
    
    df_23 = df[df['z_hd'] > 0.005].copy().reset_index(drop=True)
    results.append(run_cut("Strict flow (z>0.005, N=23)", df_23))
    
    df_28 = df[(df['z_hd'] > 0.0035) & (~df['normalized_name'].str.contains('4639'))].copy().reset_index(drop=True)
    results.append(run_cut("z>0.0035 without NGC 4639 (N=28)", df_28))
    
    print("\n\n=== SUMMARY ===")
    print(f"{'Sample':<35} | {'N':<4} | {'H0_corr':<8} | {'Kappa':<8} | {'LOOCV':<8} | {'dBIC':<8}")
    print("-" * 80)
    for r in results:
        if r is not None:
            print(f"{r['Name']:<35} | {r['N']:<4} | {r['Corrected_H0']:<8.2f} | {r['Kappa_1e6']:<8.4f} | {r['LOOCV_H0']:<8.2f} | {r['Delta_BIC']:<8.2f}")

if __name__ == '__main__':
    main()
