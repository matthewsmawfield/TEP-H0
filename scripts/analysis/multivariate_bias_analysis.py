
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Apply TEP Style
try:
    from scripts.utils.plot_style import apply_tep_style
    colors = apply_tep_style()
except ImportError:
    colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'light_blue': '#4b6785', 'green': '#4a2650'}

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
FIGURES_DIR = ROOT_DIR / "site/public/figures"

def load_and_merge_data():
    """Load stratified H0 data and merge with auxiliary astrophysical params."""
    
    # 1. Main H0 vs Sigma Data
    h0_data = pd.read_csv(OUTPUTS_DIR / "stratified_h0.csv")
    
    # 2. Cepheid Period Data (Aggregated per host)
    cepheids = pd.read_csv(DATA_DIR / "interim/reconstructed_shoes_cepheids.csv")
    cepheids['logP'] = cepheids['L_col_bW'] + 1.0
    host_periods = cepheids.groupby('Source')['logP'].mean().reset_index()
    host_periods.rename(columns={'Source': 'source_id', 'logP': 'mean_logP'}, inplace=True)
    
    # 3. SN Color Data (from Pantheon+)
    # We need to re-match or extract if not present in stratified_h0.csv
    # stratified_h0.csv has 'pantheon_id', but not 'c' (color) or 'x1' (stretch)
    pantheon_path = DATA_DIR / "raw/Pantheon+SH0ES.dat"
    if pantheon_path.exists():
        pan_df = pd.read_csv(pantheon_path, sep=r'\s+')
        pan_subset = pan_df[['CID', 'c', 'cERR', 'x1', 'x1ERR']].copy()
        pan_subset.rename(columns={'CID': 'pantheon_id'}, inplace=True)
    else:
        print("Pantheon data not found!")
        return None

    # Merge
    # H0 data has 'source_id' and 'pantheon_id'
    merged = pd.merge(h0_data, host_periods, on='source_id', how='left')
    merged = pd.merge(merged, pan_subset, on='pantheon_id', how='left')
    
    # Filter for regression (drop NaNs)
    # We need H0, Sigma, MeanLogP, Color
    analysis_df = merged.dropna(subset=['h0_derived', 'sigma_inferred', 'mean_logP', 'c']).copy()
    
    print(f"Data loaded. N={len(analysis_df)} hosts available for multivariate analysis.")
    return analysis_df

import json

def run_regression(df):
    """Run OLS regressions to test robustness of Sigma dependence."""
    
    print("\n--- Multivariate Regression Analysis ---")
    
    # Standardize variables for comparable coefficients
    df_std = df.copy()
    cols = ['h0_derived', 'sigma_inferred', 'mean_logP', 'c', 'x1', 'host_logmass']
    for col in cols:
        df_std[col] = (df[col] - df[col].mean()) / df[col].std()
    
    models = {}
    summaries = []
    structured_results = {}
    
    # Model 1: H0 ~ Sigma (Baseline)
    X1 = sm.add_constant(df_std[['sigma_inferred']])
    y = df_std['h0_derived']
    model1 = sm.OLS(y, X1).fit()
    models['Baseline'] = model1
    summaries.append("Model 1: H0 ~ Sigma\n" + str(model1.summary()))
    
    # Model 2: H0 ~ Sigma + MeanLogP (Age Control)
    X2 = sm.add_constant(df_std[['sigma_inferred', 'mean_logP']])
    model2 = sm.OLS(y, X2).fit()
    models['AgeControl'] = model2
    summaries.append("Model 2: H0 ~ Sigma + MeanLogP (Age Proxy)\n" + str(model2.summary()))
    
    # Model 3: H0 ~ Sigma + Color + Stretch (Dust/SN Physics Control)
    X3 = sm.add_constant(df_std[['sigma_inferred', 'c', 'x1']])
    model3 = sm.OLS(y, X3).fit()
    models['DustControl'] = model3
    summaries.append("Model 3: H0 ~ Sigma + Color + Stretch\n" + str(model3.summary()))
    
    # Model 4: Full Multivariate
    X4 = sm.add_constant(df_std[['sigma_inferred', 'mean_logP', 'c', 'host_logmass']])
    model4 = sm.OLS(y, X4).fit()
    models['Full'] = model4
    summaries.append("Model 4: H0 ~ Sigma + Age + Dust + Mass\n" + str(model4.summary()))
    
    # Print summaries
    for s in summaries:
        print(s)
        print("\n" + "="*80 + "\n")

    # Save structured results
    for name, model in models.items():
        structured_results[name] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'params': model.params.to_dict(),
            'bse': model.bse.to_dict(),
            'pvalues': model.pvalues.to_dict(),
            'nobs': model.nobs
        }
    
    # Write to files
    summary_path = OUTPUTS_DIR / "multivariate_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("\n\n".join(summaries))
    print(f"Saved full regression summaries to {summary_path}")
        
    json_path = OUTPUTS_DIR / "multivariate_analysis_results.json"
    with open(json_path, 'w') as f:
        json.dump(structured_results, f, indent=4)
    print(f"Saved structured results to {json_path}")
    
    return models

def plot_coefficients(models):
    """Plot standardized coefficients to compare effect sizes."""
    
    data = []
    
    for name, model in models.items():
        params = model.params
        bse = model.bse
        pvals = model.pvalues
        
        for term in params.index:
            if term == 'const': continue
            data.append({
                'Model': name,
                'Term': term,
                'Coef': params[term],
                'Error': bse[term],
                'Pval': pvals[term]
            })
            
    res_df = pd.DataFrame(data)
    
    # Map terms to readable names
    term_map = {
        'sigma_inferred': 'Potential (σ)',
        'mean_logP': 'Period (Age)',
        'c': 'Color (Dust)',
        'x1': 'Stretch',
        'host_logmass': 'Stellar Mass'
    }
    res_df['TermLabel'] = res_df['Term'].map(term_map)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Offset slightly for clarity
    offsets = {'Baseline': 0, 'AgeControl': -0.1, 'DustControl': 0.1, 'Full': 0.2}
    
    for name, group in res_df.groupby('Model'):
        y_pos = np.arange(len(group)) + offsets[name]
        # We need to align by Term actually...
        pass
    
    # Better plot: Group by Term
    terms = res_df['Term'].unique()
    
    plt.figure(figsize=(14, 9))
    
    model_order = ['Baseline', 'AgeControl', 'DustControl', 'Full']
    term_order = ['sigma_inferred', 'mean_logP', 'c', 'host_logmass', 'x1']
    
    # Create position map
    y_base = np.arange(len(term_order)) * -1.5 # Spaced out
    y_map = {t: y for t, y in zip(term_order, y_base)}
    
    offset_step = 0.2
    
    # Map colors to model names using our palette
    model_colors = {
        'Baseline': colors['dark'],
        'AgeControl': colors['blue'],
        'DustControl': colors['green'],
        'Full': colors['accent']
    }
    
    for i, model_name in enumerate(model_order):
        subset = res_df[res_df['Model'] == model_name]
        if subset.empty: continue
        
        ys = [y_map[t] + (i - 1.5) * offset_step for t in subset['Term']]
        
        plt.errorbar(subset['Coef'], ys, xerr=subset['Error'], 
                     fmt='o', label=model_name, capsize=4, 
                     color=model_colors.get(model_name, 'gray'),
                     markersize=8, linewidth=2)
        
    plt.yticks(list(y_map.values()), [term_map.get(t, t) for t in term_order])
    plt.axvline(0, color=colors['dark'], linestyle='--', alpha=0.5)
    plt.xlabel('Standardized Coefficient (Impact on H0)')
    plt.title('Robustness of Potential Dependence vs Astrophysical Confounders')
    plt.legend()
    plt.grid(True, axis='x', alpha=0.3, linestyle=':')
    
    out_path = FIGURES_DIR / "multivariate_robustness.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved coefficient plot to {out_path}")
    plt.close()

def main():
    df = load_and_merge_data()
    if df is not None:
        models = run_regression(df)
        plot_coefficients(models)

if __name__ == "__main__":
    main()
