"""
Step 6: Enhanced Robustness Analysis
=====================================

Addresses key weaknesses identified in referee review:
1. Stellar-absorption-only subsample analysis (excluding HI proxy hosts)
2. Local galaxy density as additional environment control
3. High-σ subsample (Kormendy & Ho 2013 gold standard sources only)

This step strengthens the paper by demonstrating that the H0-σ correlation:
- Persists in the homogeneous stellar-absorption subsample
- Is not driven by HI linewidth proxy uncertainties
- Remains after controlling for local galaxy density
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import sys
import json

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table


class Step6EnhancedRobustness:
    """Enhanced robustness checks addressing referee concerns."""
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.outputs_dir = self.results_dir / "outputs"
        self.logs_dir = self.root_dir / "logs"
        
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TEPLogger("step_6_enhanced", log_file_path=self.logs_dir / "step_6_enhanced_robustness.log")
        set_step_logger(self.logger)
        
        # Inputs
        self.stratified_path = self.outputs_dir / "stratified_h0.csv"
        self.sigma_provenance_path = self.outputs_dir / "sigma_provenance_table.csv"
        
        # Outputs
        self.enhanced_results_path = self.outputs_dir / "enhanced_robustness_results.json"
        self.subsample_stats_path = self.outputs_dir / "subsample_sensitivity.txt"
    
    def run(self):
        """Execute enhanced robustness analysis."""
        print_status("=" * 60, "SECTION")
        print_status("STEP 6: ENHANCED ROBUSTNESS ANALYSIS", "SECTION")
        print_status("Addressing referee concerns on σ heterogeneity and environment", "INFO")
        print_status("=" * 60, "SECTION")
        
        # Load data
        df = pd.read_csv(self.stratified_path)
        sigma_prov = pd.read_csv(self.sigma_provenance_path)
        
        print_status(f"Loaded {len(df)} hosts from stratified analysis", "INFO")
        print_status(f"Loaded {len(sigma_prov)} entries from σ provenance table", "INFO")
        
        results = {}
        
        # 1. Stellar-absorption-only subsample
        results['stellar_absorption'] = self._analyze_stellar_absorption_subsample(df, sigma_prov)
        
        # 2. High-quality σ subsample (Kormendy & Ho + SDSS only)
        results['gold_standard'] = self._analyze_gold_standard_subsample(df, sigma_prov)
        
        # 3. Local density control
        results['density_control'] = self._analyze_density_control(df)
        
        # 4. TEP correction on subsamples
        results['subsample_tep'] = self._tep_correction_subsamples(df, sigma_prov)
        
        # Save results
        with open(self.enhanced_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Write summary
        self._write_summary(results)
        
        print_status("=" * 60, "SECTION")
        print_status("ENHANCED ROBUSTNESS ANALYSIS COMPLETE", "SECTION")
        print_status("=" * 60, "SECTION")
        
        return results
    
    def _analyze_stellar_absorption_subsample(self, df, sigma_prov):
        """Analyze only hosts with stellar absorption σ measurements."""
        print_status("\n--- STELLAR ABSORPTION SUBSAMPLE ---", "SECTION")
        
        prov = sigma_prov.copy()
        if 'normalized_name' in prov.columns:
            prov['normalized_name'] = prov['normalized_name'].astype(str).str.strip()
        for c in ['sigma_method', 'sigma_source']:
            if c in prov.columns:
                prov[c] = prov[c].astype(str).str.strip()

        def _method_class(v: str) -> str:
            s = str(v).lower().strip()
            if 'stellar absorption' in s:
                return 'stellar absorption'
            if 'hi' in s or 'w50' in s or 'vrot' in s or 'linewidth' in s or 'proxy' in s:
                return 'HI linewidth proxy'
            return 'unknown'

        def _source_class(v: str) -> str:
            s = str(v).lower().strip()
            if 'kormendy' in s:
                return 'Kormendy&Ho2013'
            if 'sdss' in s:
                return 'SDSS DR7'
            if 'ho+2009' in s or 'ho+09' in s or 'j/apjs/183/1' in s:
                return 'Ho+2009'
            if 'hyperleda' in s and 'hi' in s:
                return 'HyperLEDA HI'
            if 'hyperleda' in s:
                return 'HyperLEDA'
            if 'ho 2007' in s or 'j/apj/668/94' in s:
                return 'Ho2007'
            if 'bass' in s:
                return 'BASS DR2'
            if '6dfgs' in s:
                return '6dFGSv'
            if 'apj 929:84' in s or 'j/apj/929/84' in s:
                return 'ApJ 929:84'
            if 'mnras 482:1427' in s or 'j/mnras/482/1427' in s:
                return 'MNRAS 482:1427'
            return str(v).strip()

        if 'sigma_method' in prov.columns:
            prov['sigma_method_class'] = prov['sigma_method'].apply(_method_class)
        else:
            prov['sigma_method_class'] = 'unknown'

        if 'sigma_source' in prov.columns:
            prov['sigma_source_class'] = prov['sigma_source'].apply(_source_class)
        else:
            prov['sigma_source_class'] = ''

        # Identify stellar absorption hosts
        stellar_hosts = prov[prov['sigma_method_class'] == 'stellar absorption']['normalized_name'].tolist()
        hi_hosts = prov[prov['sigma_method_class'] == 'HI linewidth proxy']['normalized_name'].tolist()
        
        print_status(f"Stellar absorption hosts: {len(stellar_hosts)}", "INFO")
        print_status(f"HI linewidth proxy hosts: {len(hi_hosts)}", "INFO")
        
        # Filter main dataframe
        df_stellar = df[df['normalized_name'].isin(stellar_hosts)].copy()
        df_hi = df[df['normalized_name'].isin(hi_hosts)].copy()
        
        # Full sample statistics
        full_r, full_p = stats.pearsonr(df['sigma_corrected'], df['h0_derived'])
        full_rho, full_rho_p = stats.spearmanr(df['sigma_corrected'], df['h0_derived'])
        
        # Stellar-only statistics
        if len(df_stellar) >= 5:
            stellar_r, stellar_p = stats.pearsonr(df_stellar['sigma_corrected'], df_stellar['h0_derived'])
            stellar_rho, stellar_rho_p = stats.spearmanr(df_stellar['sigma_corrected'], df_stellar['h0_derived'])
        else:
            stellar_r, stellar_p, stellar_rho, stellar_rho_p = np.nan, np.nan, np.nan, np.nan
        
        # Stratification in stellar-only
        if len(df_stellar) >= 5:
            median_sigma = df_stellar['sigma_corrected'].median()
            low_sigma = df_stellar[df_stellar['sigma_corrected'] <= median_sigma]
            high_sigma = df_stellar[df_stellar['sigma_corrected'] > median_sigma]
            
            h0_low = low_sigma['h0_derived'].mean()
            h0_high = high_sigma['h0_derived'].mean()
            delta_h0 = h0_high - h0_low
        else:
            h0_low, h0_high, delta_h0, median_sigma = np.nan, np.nan, np.nan, np.nan
        
        results = {
            'n_stellar': len(df_stellar),
            'n_hi': len(df_hi),
            'n_total': len(df),
            'full_sample': {
                'pearson_r': float(full_r),
                'pearson_p': float(full_p),
                'spearman_rho': float(full_rho),
                'spearman_p': float(full_rho_p)
            },
            'stellar_only': {
                'pearson_r': float(stellar_r) if not np.isnan(stellar_r) else None,
                'pearson_p': float(stellar_p) if not np.isnan(stellar_p) else None,
                'spearman_rho': float(stellar_rho) if not np.isnan(stellar_rho) else None,
                'spearman_p': float(stellar_rho_p) if not np.isnan(stellar_rho_p) else None,
                'median_sigma': float(median_sigma) if not np.isnan(median_sigma) else None,
                'h0_low_sigma': float(h0_low) if not np.isnan(h0_low) else None,
                'h0_high_sigma': float(h0_high) if not np.isnan(h0_high) else None,
                'delta_h0': float(delta_h0) if not np.isnan(delta_h0) else None
            },
            'stellar_hosts': stellar_hosts,
            'hi_hosts': hi_hosts
        }
        
        # Print table
        print_status("\nSubsample Comparison:", "INFO")
        headers = ["Subsample", "N", "Pearson r", "p-value", "Spearman ρ", "p-value"]
        rows = [
            ["Full Sample", len(df), f"{full_r:.3f}", f"{full_p:.4f}", f"{full_rho:.3f}", f"{full_rho_p:.4f}"],
            ["Stellar Absorption Only", len(df_stellar), 
             f"{stellar_r:.3f}" if not np.isnan(stellar_r) else "N/A",
             f"{stellar_p:.4f}" if not np.isnan(stellar_p) else "N/A",
             f"{stellar_rho:.3f}" if not np.isnan(stellar_rho) else "N/A",
             f"{stellar_rho_p:.4f}" if not np.isnan(stellar_rho_p) else "N/A"]
        ]
        print_table(headers, rows)
        
        if not np.isnan(delta_h0):
            print_status(f"\nStellar-only stratification: ΔH0 = {delta_h0:.2f} km/s/Mpc (high - low σ)", "RESULT")
        
        return results
    
    def _analyze_gold_standard_subsample(self, df, sigma_prov):
        """Analyze only hosts with gold-standard σ (Kormendy & Ho, SDSS)."""
        print_status("\n--- GOLD STANDARD σ SUBSAMPLE ---", "SECTION")
        
        # Identify gold standard sources
        gold_sources = ['Kormendy&Ho2013', 'SDSS DR7', 'Ho+2009']
        prov = sigma_prov.copy()
        if 'normalized_name' in prov.columns:
            prov['normalized_name'] = prov['normalized_name'].astype(str).str.strip()
        for c in ['sigma_source', 'sigma_method']:
            if c in prov.columns:
                prov[c] = prov[c].astype(str).str.strip()

        if 'sigma_source_class' not in prov.columns:
            def _source_class(v: str) -> str:
                s = str(v).lower().strip()
                if 'kormendy' in s:
                    return 'Kormendy&Ho2013'
                if 'sdss' in s:
                    return 'SDSS DR7'
                if 'ho+2009' in s or 'ho+09' in s or 'j/apjs/183/1' in s:
                    return 'Ho+2009'
                return str(v).strip()

            prov['sigma_source_class'] = prov['sigma_source'].apply(_source_class) if 'sigma_source' in prov.columns else ''

        gold_prov = prov[prov['sigma_source_class'].isin(gold_sources)]
        gold_hosts = gold_prov['normalized_name'].tolist()
        
        print_status(f"Gold standard hosts (K&H13 + SDSS + Ho09): {len(gold_hosts)}", "INFO")
        for src in gold_sources:
            n = len(sigma_prov[sigma_prov['sigma_source'] == src])
            print_status(f"  - {src}: {n} hosts", "INFO")
        
        df_gold = df[df['normalized_name'].isin(gold_hosts)].copy()
        
        if len(df_gold) >= 5:
            gold_r, gold_p = stats.pearsonr(df_gold['sigma_corrected'], df_gold['h0_derived'])
            gold_rho, gold_rho_p = stats.spearmanr(df_gold['sigma_corrected'], df_gold['h0_derived'])
        else:
            gold_r, gold_p, gold_rho, gold_rho_p = np.nan, np.nan, np.nan, np.nan
        
        results = {
            'n_gold': len(df_gold),
            'gold_sources': gold_sources,
            'gold_hosts': gold_hosts,
            'pearson_r': float(gold_r) if not np.isnan(gold_r) else None,
            'pearson_p': float(gold_p) if not np.isnan(gold_p) else None,
            'spearman_rho': float(gold_rho) if not np.isnan(gold_rho) else None,
            'spearman_p': float(gold_rho_p) if not np.isnan(gold_rho_p) else None
        }
        
        if not np.isnan(gold_r):
            print_status(f"Gold standard subsample: r = {gold_r:.3f}, p = {gold_p:.4f}", "RESULT")
        
        return results
    
    def _analyze_density_control(self, df):
        """Control for local galaxy density using rho_local."""
        print_status("\n--- LOCAL DENSITY CONTROL ---", "SECTION")
        
        # Check if rho_local is available
        if 'rho_local' not in df.columns:
            print_status("rho_local column not found - skipping density control", "WARNING")
            return {'available': False}
        
        # Filter to hosts with valid rho_local
        df_valid = df[df['rho_local'].notna() & (df['rho_local'] > 0)].copy()
        print_status(f"Hosts with valid local density: {len(df_valid)}/{len(df)}", "INFO")
        
        if len(df_valid) < 10:
            print_status("Insufficient hosts with valid density for partial correlation", "WARNING")
            return {'available': False, 'n_valid': len(df_valid)}
        
        # Log transform density for better distribution
        df_valid['log_rho'] = np.log10(df_valid['rho_local'])
        
        # Compute partial correlation: r(H0, σ | log_rho)
        h0 = df_valid['h0_derived'].values
        sigma = df_valid['sigma_corrected'].values
        log_rho = df_valid['log_rho'].values
        
        # Residual method for partial correlation
        def partial_corr(x, y, z):
            """Compute partial correlation r(x,y|z) using residual method."""
            # Regress x on z
            slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
            resid_x = x - (slope_xz * z + intercept_xz)
            
            # Regress y on z
            slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
            resid_y = y - (slope_yz * z + intercept_yz)
            
            # Correlation of residuals
            r, p = stats.pearsonr(resid_x, resid_y)
            return r, p
        
        # Baseline correlation
        base_r, base_p = stats.pearsonr(sigma, h0)
        
        # Partial correlation controlling for density
        partial_r, partial_p = partial_corr(h0, sigma, log_rho)
        
        # Also check correlation of H0 with density
        rho_h0_r, rho_h0_p = stats.pearsonr(log_rho, h0)
        rho_sigma_r, rho_sigma_p = stats.pearsonr(log_rho, sigma)
        
        results = {
            'available': True,
            'n_valid': len(df_valid),
            'baseline_r': float(base_r),
            'baseline_p': float(base_p),
            'partial_r_h0_sigma_given_rho': float(partial_r),
            'partial_p': float(partial_p),
            'r_h0_rho': float(rho_h0_r),
            'p_h0_rho': float(rho_h0_p),
            'r_sigma_rho': float(rho_sigma_r),
            'p_sigma_rho': float(rho_sigma_p)
        }
        
        print_status("\nDensity Control Results:", "INFO")
        headers = ["Metric", "Value"]
        rows = [
            ["Baseline r(H0, σ)", f"{base_r:.3f} (p = {base_p:.4f})"],
            ["Partial r(H0, σ | log ρ)", f"{partial_r:.3f} (p = {partial_p:.4f})"],
            ["r(H0, log ρ)", f"{rho_h0_r:.3f} (p = {rho_h0_p:.4f})"],
            ["r(σ, log ρ)", f"{rho_sigma_r:.3f} (p = {rho_sigma_p:.4f})"]
        ]
        print_table(headers, rows)
        
        print_status(f"\nSignal after density control: r = {partial_r:.3f} (p = {partial_p:.4f})", "RESULT")
        
        return results
    
    def _tep_correction_subsamples(self, df, sigma_prov):
        """Apply TEP correction to subsamples and verify consistency."""
        print_status("\n--- TEP CORRECTION ON SUBSAMPLES ---", "SECTION")
        
        sigma_ref = 75.25  # Fixed reference from anchors
        
        def optimize_alpha(subset_df):
            """Optimize alpha for a subset."""
            if len(subset_df) < 5:
                return np.nan, np.nan, np.nan
            
            def objective(alpha):
                mu_corr = subset_df['value'].values + alpha * np.log10(subset_df['sigma_corrected'].values / sigma_ref)
                d_corr = 10 ** ((mu_corr - 25) / 5)
                h0_corr = 299792.458 * subset_df['z_hd'].values / d_corr
                slope, _, _, _, _ = stats.linregress(subset_df['sigma_corrected'].values, h0_corr)
                return slope ** 2
            
            result = minimize(objective, x0=0.8, method='Nelder-Mead')
            alpha_opt = result.x[0]
            
            # Compute unified H0
            mu_corr = subset_df['value'].values + alpha_opt * np.log10(subset_df['sigma_corrected'].values / sigma_ref)
            d_corr = 10 ** ((mu_corr - 25) / 5)
            h0_corr = 299792.458 * subset_df['z_hd'].values / d_corr
            unified_h0 = np.mean(h0_corr)
            h0_std = np.std(h0_corr) / np.sqrt(len(h0_corr))
            
            return alpha_opt, unified_h0, h0_std
        
        # Full sample
        alpha_full, h0_full, err_full = optimize_alpha(df)
        
        # Stellar absorption only
        stellar_hosts = sigma_prov[sigma_prov['sigma_method'] == 'stellar absorption']['normalized_name'].tolist()
        df_stellar = df[df['normalized_name'].isin(stellar_hosts)].copy()
        alpha_stellar, h0_stellar, err_stellar = optimize_alpha(df_stellar)
        
        # Gold standard only
        gold_sources = ['Kormendy&Ho2013', 'SDSS DR7', 'Ho+2009']
        gold_prov = sigma_prov[sigma_prov['sigma_source'].isin(gold_sources)]
        gold_hosts = gold_prov['normalized_name'].tolist()
        df_gold = df[df['normalized_name'].isin(gold_hosts)].copy()
        alpha_gold, h0_gold, err_gold = optimize_alpha(df_gold)
        
        results = {
            'full_sample': {
                'n': len(df),
                'alpha': float(alpha_full) if not np.isnan(alpha_full) else None,
                'unified_h0': float(h0_full) if not np.isnan(h0_full) else None,
                'h0_error': float(err_full) if not np.isnan(err_full) else None
            },
            'stellar_only': {
                'n': len(df_stellar),
                'alpha': float(alpha_stellar) if not np.isnan(alpha_stellar) else None,
                'unified_h0': float(h0_stellar) if not np.isnan(h0_stellar) else None,
                'h0_error': float(err_stellar) if not np.isnan(err_stellar) else None
            },
            'gold_standard': {
                'n': len(df_gold),
                'alpha': float(alpha_gold) if not np.isnan(alpha_gold) else None,
                'unified_h0': float(h0_gold) if not np.isnan(h0_gold) else None,
                'h0_error': float(err_gold) if not np.isnan(err_gold) else None
            }
        }
        
        print_status("\nTEP Correction Comparison:", "INFO")
        headers = ["Subsample", "N", "α_opt", "Unified H0 (km/s/Mpc)"]
        rows = [
            ["Full Sample", len(df), 
             f"{alpha_full:.3f}" if not np.isnan(alpha_full) else "N/A",
             f"{h0_full:.2f} ± {err_full:.2f}" if not np.isnan(h0_full) else "N/A"],
            ["Stellar Only", len(df_stellar),
             f"{alpha_stellar:.3f}" if not np.isnan(alpha_stellar) else "N/A",
             f"{h0_stellar:.2f} ± {err_stellar:.2f}" if not np.isnan(h0_stellar) else "N/A"],
            ["Gold Standard", len(df_gold),
             f"{alpha_gold:.3f}" if not np.isnan(alpha_gold) else "N/A",
             f"{h0_gold:.2f} ± {err_gold:.2f}" if not np.isnan(h0_gold) else "N/A"]
        ]
        print_table(headers, rows)
        
        return results
    
    def _write_summary(self, results):
        """Write summary to text file."""
        with open(self.subsample_stats_path, 'w') as f:
            f.write("ENHANCED ROBUSTNESS ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Stellar absorption
            sa = results['stellar_absorption']
            f.write("1. STELLAR ABSORPTION SUBSAMPLE\n")
            f.write(f"   N_stellar = {sa['n_stellar']}, N_HI = {sa['n_hi']}\n")
            if sa['stellar_only']['pearson_r'] is not None:
                f.write(f"   Pearson r = {sa['stellar_only']['pearson_r']:.3f} (p = {sa['stellar_only']['pearson_p']:.4f})\n")
                f.write(f"   Spearman ρ = {sa['stellar_only']['spearman_rho']:.3f} (p = {sa['stellar_only']['spearman_p']:.4f})\n")
                f.write(f"   ΔH0 (high - low σ) = {sa['stellar_only']['delta_h0']:.2f} km/s/Mpc\n")
            f.write("\n")
            
            # Gold standard
            gs = results['gold_standard']
            f.write("2. GOLD STANDARD σ SUBSAMPLE\n")
            f.write(f"   N = {gs['n_gold']} (sources: {', '.join(gs['gold_sources'])})\n")
            if gs['pearson_r'] is not None:
                f.write(f"   Pearson r = {gs['pearson_r']:.3f} (p = {gs['pearson_p']:.4f})\n")
            f.write("\n")
            
            # Density control
            dc = results['density_control']
            f.write("3. LOCAL DENSITY CONTROL\n")
            if dc.get('available'):
                f.write(f"   N with valid density = {dc['n_valid']}\n")
                f.write(f"   Baseline r(H0, σ) = {dc['baseline_r']:.3f} (p = {dc['baseline_p']:.4f})\n")
                f.write(f"   Partial r(H0, σ | log ρ) = {dc['partial_r_h0_sigma_given_rho']:.3f} (p = {dc['partial_p']:.4f})\n")
            else:
                f.write("   Insufficient data for density control\n")
            f.write("\n")
            
            # TEP on subsamples
            tep = results['subsample_tep']
            f.write("4. TEP CORRECTION ON SUBSAMPLES\n")
            for name, data in tep.items():
                if data['unified_h0'] is not None:
                    f.write(f"   {name}: α = {data['alpha']:.3f}, H0 = {data['unified_h0']:.2f} ± {data['h0_error']:.2f}\n")
            
        print_status(f"\nSummary written to: {self.subsample_stats_path}", "INFO")


def main():
    step = Step6EnhancedRobustness()
    return step.run()


if __name__ == "__main__":
    main()
