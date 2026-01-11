
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import json

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

class Step6MultivariateAnalysis:
    r"""
    Step 6: Multivariate Analysis of Astrophysical Systematics
    ==========================================================
    
    This step performs a rigorous multivariate regression analysis to determine if the 
    observed H0-Sigma correlation is driven by mundane astrophysical confounders.
    
    We test the following hypothesis:
    H0 ~ Sigma + Age (Period) + Dust (Color) + Mass
    
    If Sigma remains significant while other factors are not, the TEP hypothesis (gravitational potential)
    is supported over astrophysical systematics.
    
    Inputs:
        - results/outputs/stratified_h0.csv (H0 & Sigma)
        - data/interim/reconstructed_shoes_cepheids.csv (Periods)
        - data/raw/Pantheon+SH0ES.dat (SN Colors)
        
    Outputs:
        - results/outputs/multivariate_analysis_results.json
        - results/outputs/multivariate_analysis_summary.txt
        - site/public/figures/multivariate_robustness.png
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.outputs_dir = self.results_dir / "outputs"
        self.figures_dir = self.root_dir / "site" / "public" / "figures"
        self.logs_dir = self.root_dir / "logs"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TEPLogger("step_6_multivariate", log_file_path=self.logs_dir / "step_6_multivariate.log")
        set_step_logger(self.logger)
        
        # Inputs
        self.h0_path = self.outputs_dir / "stratified_h0.csv"
        self.cepheid_path = self.data_dir / "interim" / "reconstructed_shoes_cepheids.csv"
        self.pantheon_path = self.data_dir / "raw" / "Pantheon+SH0ES.dat"
        
        # Outputs
        self.summary_path = self.outputs_dir / "multivariate_analysis_summary.txt"
        self.json_path = self.outputs_dir / "multivariate_analysis_results.json"
        self.plot_path = self.figures_dir / "multivariate_robustness.png"

    def load_and_merge_data(self):
        """Load stratified H0 data and merge with auxiliary astrophysical params."""
        print_status("Loading and Merging Data...", "SECTION")
        
        if not self.h0_path.exists():
            print_status("Stratified H0 data not found. Run Step 2 first.", "ERROR")
            return None
            
        # 1. Main H0 vs Sigma Data
        h0_data = pd.read_csv(self.h0_path)
        
        # 2. Cepheid Period Data (Aggregated per host)
        if self.cepheid_path.exists():
            cepheids = pd.read_csv(self.cepheid_path)
            # Recover LogP
            # L_col_bW corresponds to (logP - 1)
            cepheids['logP'] = cepheids['L_col_bW'] + 1.0
            host_periods = cepheids.groupby('Source')['logP'].mean().reset_index()
            host_periods.rename(columns={'Source': 'source_id', 'logP': 'mean_logP'}, inplace=True)
        else:
            print_status("Cepheid data not found.", "ERROR")
            return None
        
        # 3. SN Color Data (from Pantheon+)
        if self.pantheon_path.exists():
            pan_df = pd.read_csv(self.pantheon_path, sep=r'\s+')
            pan_subset = pan_df[['CID', 'c', 'cERR', 'x1', 'x1ERR']].copy()
            pan_subset.rename(columns={'CID': 'pantheon_id'}, inplace=True)
        else:
            print_status("Pantheon data not found.", "ERROR")
            return None
    
        # Merge
        # H0 data has 'source_id' and 'pantheon_id'
        merged = pd.merge(h0_data, host_periods, on='source_id', how='left')
        merged = pd.merge(merged, pan_subset, on='pantheon_id', how='left')
        
        # Filter for regression (drop NaNs)
        analysis_df = merged.dropna(subset=['h0_derived', 'sigma_inferred', 'mean_logP', 'c']).copy()
        
        print_status(f"Merged Data: N={len(analysis_df)} hosts available for analysis.", "INFO")
        return analysis_df

    def run_regression(self, df):
        """Run OLS regressions to test robustness of Sigma dependence."""
        print_status("Running Multivariate Regression...", "SECTION")
        
        # Standardize variables for comparable coefficients
        df_std = df.copy()
        cols = ['h0_derived', 'sigma_inferred', 'mean_logP', 'c', 'x1', 'host_logmass']
        for col in cols:
            if col in df.columns:
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
        
        # Write files
        with open(self.summary_path, 'w') as f:
            f.write("\n\n".join(summaries))
        print_status(f"Saved regression summaries to {self.summary_path}", "SUCCESS")
            
        with open(self.json_path, 'w') as f:
            json.dump(structured_results, f, indent=4)
        print_status(f"Saved structured results to {self.json_path}", "SUCCESS")
        
        # Check significance of Sigma in full model
        pval_sigma = model4.pvalues['sigma_inferred']
        if pval_sigma < 0.05:
            print_status(f"Sigma remains significant in full model (p={pval_sigma:.4f}).", "SUCCESS")
        else:
            print_status(f"Sigma loses significance in full model (p={pval_sigma:.4f}).", "WARNING")
            
        return models

    def plot_coefficients(self, models):
        """Plot standardized coefficients."""
        print_status("Generating Robustness Plot...", "PROCESS")
        
        # Apply Style if available
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'green': '#4a2650'}
            
        data = []
        for name, model in models.items():
            params = model.params
            bse = model.bse
            for term in params.index:
                if term == 'const': continue
                data.append({
                    'Model': name,
                    'Term': term,
                    'Coef': params[term],
                    'Error': bse[term]
                })
                
        res_df = pd.DataFrame(data)
        
        term_map = {
            'sigma_inferred': 'Potential (σ)',
            'mean_logP': 'Period (Age)',
            'c': 'Color (Dust)',
            'x1': 'Stretch',
            'host_logmass': 'Stellar Mass'
        }
        res_df['TermLabel'] = res_df['Term'].map(term_map)
        
        plt.figure(figsize=(14, 9))
        
        model_order = ['Baseline', 'AgeControl', 'DustControl', 'Full']
        term_order = ['sigma_inferred', 'mean_logP', 'c', 'host_logmass', 'x1']
        
        y_base = np.arange(len(term_order)) * -1.5 
        y_map = {t: y for t, y in zip(term_order, y_base)}
        
        offset_step = 0.2
        
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
        
        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved plot to {self.plot_path}", "SUCCESS")
        plt.close()

    def run(self):
        print_status("Starting Step 6: Multivariate Analysis", "TITLE")
        
        df = self.load_and_merge_data()
        if df is not None:
            models = self.run_regression(df)
            self.plot_coefficients(models)
            
        print_status("Step 6 Complete.", "SUCCESS")

def main():
    step = Step6MultivariateAnalysis()
    step.run()

if __name__ == "__main__":
    main()
