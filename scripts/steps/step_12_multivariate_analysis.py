
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError as e:
    raise ImportError("statsmodels is required for multivariate analysis. Install with: pip install statsmodels") from e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    observed H_0–σ correlation is driven by mundane astrophysical confounders.
    
    We test the following hypothesis:
    H_0 ~ σ + Age (Period) + Dust (Color) + Mass
    
    If σ remains significant while other factors are not, the TEP hypothesis (gravitational potential)
    is supported over astrophysical systematics.
    
    Inputs:
        - results/outputs/step_03_stratified_h0.csv (H_0 & σ)
        - data/interim/reconstructed_shoes_cepheids.csv (Periods)
        - data/raw/Pantheon+SH0ES.dat (SN Colors)
        
    Outputs:
        - results/outputs/step_12_multivariate_analysis_results.json
        - results/outputs/step_12_multivariate_analysis_summary.txt
        - results/figures/step_12_figure_12_multivariate_robustness.png
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.outputs_dir = self.results_dir / "outputs"
        self.figures_dir = self.results_dir / "figures"
        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        self.logs_dir = self.root_dir / "logs"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.public_figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TEPLogger("step_6_multivariate", log_file_path=self.logs_dir / "step_12_multivariate_analysis.log")
        set_step_logger(self.logger)
        
        # Inputs
        self.h0_path = self.outputs_dir / "step_03_stratified_h0.csv"
        self.cepheid_path = self.data_dir / "interim" / "reconstructed_shoes_cepheids.csv"
        self.pantheon_path = self.data_dir / "raw" / "Pantheon+SH0ES.dat"
        
        # Outputs
        self.summary_path = self.outputs_dir / "step_12_multivariate_analysis_summary.txt"
        self.json_path = self.outputs_dir / "step_12_multivariate_analysis_results.json"
        self.plot_path = self.figures_dir / "step_12_figure_12_multivariate_robustness.png"

    def load_and_merge_data(self):
        """Load stratified H0 data and merge with auxiliary astrophysical params."""
        print_status("Loading and Merging Data...", "SECTION")
        
        if not self.h0_path.exists():
            print_status("Stratified H0 data not found. Run Step 2 first.", "ERROR")
            return None
            
        # 1. Main H_0 vs σ data
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
            pan_subset = pan_subset.groupby('CID').mean().reset_index()
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
        cols = ['h0_derived', 'sigma_inferred', 'mean_logP', 'c', 'x1', 'host_logmass', 'z_hd', 'tully_nmb']
        for col in cols:
            if col in df.columns:
                df_std[col] = (df[col] - df[col].mean()) / df[col].std()
        if 'tully_nmb' in df_std.columns:
            df_std['tully_nmb'] = df_std['tully_nmb'].fillna(0.0)
        
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

        # Model 5: Include flow and large-scale environment controls when present.
        flow_cols = ['sigma_inferred', 'mean_logP', 'c', 'host_logmass']
        if 'z_hd' in df_std.columns:
            flow_cols.append('z_hd')
        if 'tully_nmb' in df_std.columns:
            flow_cols.append('tully_nmb')
        X5 = sm.add_constant(df_std[flow_cols])
        model5 = sm.OLS(y, X5).fit()
        models['FlowEnvironment'] = model5
        summaries.append("Model 5: H0 ~ Sigma + Age + Dust + Mass + zHD + Group Richness\n" + str(model5.summary()))
        
        # Save structured results
        for name, model in models.items():
            robust = model.get_robustcov_results(cov_type='HC3')
            robust_params = dict(zip(model.params.index, robust.params))
            robust_bse = dict(zip(model.params.index, robust.bse))
            robust_pvalues = dict(zip(model.params.index, robust.pvalues))
            structured_results[name] = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'params': model.params.to_dict(),
                'bse': model.bse.to_dict(),
                'pvalues': model.pvalues.to_dict(),
                'hc3_params': robust_params,
                'hc3_bse': robust_bse,
                'hc3_pvalues': robust_pvalues,
                'nobs': model.nobs
            }
        
        # Write files
        with open(self.summary_path, 'w') as f:
            f.write("\n\n".join(summaries))
        print_status(f"Saved regression summaries to {self.summary_path}", "SUCCESS")
            
        with open(self.json_path, 'w') as f:
            json.dump(structured_results, f, indent=4)
        print_status(f"Saved structured results to {self.json_path}", "SUCCESS")
        
        # The primary confounder test controls ordinary astrophysical nuisance
        # terms. The flow/environment model is a deliberately saturated stress
        # test because group richness is also a TEP screening mediator.
        full_robust = model4.get_robustcov_results(cov_type='HC3')
        full_terms = list(model4.params.index)
        full_sigma_idx = full_terms.index('sigma_inferred')
        pval_sigma_full = float(full_robust.pvalues[full_sigma_idx])

        flow_robust = model5.get_robustcov_results(cov_type='HC3')
        flow_terms = list(model5.params.index)
        flow_sigma_idx = flow_terms.index('sigma_inferred')
        pval_sigma_flow = float(flow_robust.pvalues[flow_sigma_idx])

        structured_results['_interpretation'] = {
            'primary_model': 'Full',
            'primary_reason': (
                'Controls age, SN color, and stellar mass. This is the primary '
                'astrophysical-confound model.'
            ),
            'primary_sigma_hc3_p': pval_sigma_full,
            'stress_model': 'FlowEnvironment',
            'stress_reason': (
                'Adds redshift and group richness. This is a conservative stress '
                'test because group richness can mediate TEP screening rather than '
                'act as a pure nuisance covariate.'
            ),
            'stress_sigma_hc3_p': pval_sigma_flow,
        }
        with open(self.json_path, 'w') as f:
            json.dump(structured_results, f, indent=4)

        if pval_sigma_full < 0.05:
            print_status(
                f"Sigma remains significant in primary astrophysical HC3 model (p={pval_sigma_full:.4f}).",
                "SUCCESS",
            )
        else:
            print_status(
                f"Sigma is not significant in primary astrophysical HC3 model (p={pval_sigma_full:.4f}).",
                "WARNING",
            )
        print_status(
            "Flow/environment model is a saturated mediator stress test; "
            f"sigma HC3 p={pval_sigma_flow:.4f}.",
            "INFO",
        )
            
        return models

    def plot_coefficients(self, models):
        """Plot standardized coefficients."""
        print_status("Generating Robustness Plot...", "PROCESS")
        
        # Apply Style if available
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'light_blue': '#4b6785', 'green': '#4a2650'}
            
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
            'sigma_inferred': 'Velocity Dispersion (σ) [Potential Proxy]',
            'mean_logP': 'Period (Age)',
            'c': 'Color (Dust)',
            'x1': 'Stretch',
            'host_logmass': 'Stellar Mass',
            'z_hd': 'Redshift',
            'tully_nmb': 'Group Richness'
        }
        res_df['TermLabel'] = res_df['Term'].map(term_map)
        
        plt.figure(figsize=(14, 9))
        
        model_order = ['Baseline', 'AgeControl', 'DustControl', 'Full', 'FlowEnvironment']
        term_order = ['sigma_inferred', 'mean_logP', 'c', 'host_logmass', 'x1', 'z_hd', 'tully_nmb']
        
        y_base = np.arange(len(term_order)) * -1.5 
        y_map = {t: y for t, y in zip(term_order, y_base)}
        
        offset_step = 0.2
        
        model_colors = {
            'Baseline': colors['dark'],
            'AgeControl': colors['blue'],
            'DustControl': colors['green'],
            'Full': colors['accent'],
            'FlowEnvironment': colors.get('light_blue', colors['blue'])
        }
        
        model_labels = {
            'Baseline': 'Baseline (σ only)',
            'AgeControl': '+ Period/Age control',
            'DustControl': '+ Color/Dust control',
            'Full': '+ All astrophysical controls',
            'FlowEnvironment': '+ Flow environment controls'
        }
        
        for i, model_name in enumerate(model_order):
            subset = res_df[res_df['Model'] == model_name]
            if subset.empty: continue
            
            ys = [y_map[t] + (i - 1.5) * offset_step for t in subset['Term']]
            # Mute non-sigma covariates: smaller paler markers
            ms = [10 if t == 'sigma_inferred' else 6 for t in subset['Term']]
            lw = [2.5 if t == 'sigma_inferred' else 1.5 for t in subset['Term']]
            
            for j, (_, row) in enumerate(subset.iterrows()):
                c = colors['accent'] if row['Term'] == 'sigma_inferred' else '#999999'
                alpha = 1.0 if row['Term'] == 'sigma_inferred' else 0.45
                plt.errorbar(row['Coef'], ys[j], xerr=row['Error'],
                             fmt='o', capsize=3 if row['Term'] != 'sigma_inferred' else 4,
                             color=c, alpha=alpha,
                             markersize=ms[j], linewidth=lw[j])
            # Dummy legend entry
            plt.errorbar([], [], fmt='o', color=model_colors.get(model_name, 'gray'),
                         markersize=8, linewidth=2, label=model_labels.get(model_name, model_name))
            
        plt.yticks(list(y_map.values()), [term_map.get(t, t) for t in term_order])
        # Solid, prominent zero-effect line; faint grid behind it
        plt.axvline(0, color='black', linestyle='-', linewidth=2.0, zorder=100)
        plt.xlabel(r'Standardized Coefficient (Impact on $H_0$)')
        plt.title('Multivariate Robustness of the Host-Potential Signal')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, axis='x', alpha=0.15, linestyle=':')
        
        # Highlight sigma row with background shading
        sigma_y = y_map['sigma_inferred']
        plt.axhspan(sigma_y - 0.6, sigma_y + 0.6, color=colors['accent'], alpha=0.1, zorder=0)
        
        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved plot to {self.plot_path}", "SUCCESS")
        plt.close()

        # Copy to public figures for site build
        public_plot_path = self.public_figures_dir / "step_12_figure_12_multivariate_robustness.png"
        shutil.copy(self.plot_path, public_plot_path)
        print_status(f"Copied plot to {public_plot_path}", "SUCCESS")

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
