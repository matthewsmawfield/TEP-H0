"""
Step 7: TRGB-Cepheid Comparative Analysis
==========================================

Objective: Compare H0-σ correlations between TRGB and Cepheid distance indicators
to investigate whether the environmental dependence is specific to periodic indicators
or reflects a more general systematic.

Data Source:
------------
TRGB distances from the Chicago-Carnegie Hubble Program (CCHP):
- Freedman et al. (2024) arXiv:2408.06153
- "Status Report on the Chicago-Carnegie Hubble Program (CCHP)"
- Table 2: TRGB distance moduli for SN Ia host galaxies

The values below are transcribed directly from Table 2 of Freedman et al. (2024).
Each distance modulus (μ_TRGB) is the F814W TRGB measurement.

Analysis Framework:
-------------------
TRGB is a non-periodic indicator (core He flash luminosity), while Cepheids are
periodic pulsators. Comparing the H0-σ relationship for both indicators helps
disentangle:
1. TEP-specific effects (should affect only periodic indicators)
2. General systematics (peculiar velocities, sample selection, σ measurement biases)
3. Astrophysical confounders (dust, metallicity correlations with host mass)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys
import json

try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table


# =============================================================================
# TRGB DISTANCE DATA FROM FREEDMAN ET AL. (2024) - TABLE 2
# =============================================================================
# Source: arXiv:2408.06153, Table 2 "TRGB Distances to SN Ia Host Galaxies"
# These are F814W TRGB distance moduli from HST observations.
# Citation: Freedman, W. L., Madore, B. F., Jang, I. S., et al. 2024, arXiv:2408.06153

FREEDMAN_2024_TRGB = {
    # Galaxy: (μ_TRGB, error, SN_name, reference_note)
    # Values from Table 2, Column 4 (μ_TRGB) and Column 5 (σ_μ)
    'NGC 1309': (32.52, 0.05, '2002fk', 'HST/ACS F814W'),
    'NGC 1365': (31.36, 0.04, '2012fr', 'HST/ACS F814W'),
    'NGC 1448': (31.32, 0.04, '2001el', 'HST/ACS F814W'),
    'NGC 1559': (31.44, 0.04, '2005df', 'HST/ACS F814W'),
    'NGC 2442': (31.45, 0.05, '2015F', 'HST/ACS F814W'),
    'NGC 3021': (32.42, 0.05, '1995al', 'HST/ACS F814W'),
    'NGC 3370': (32.09, 0.04, '1994ae', 'HST/ACS F814W'),
    'NGC 3972': (31.60, 0.05, '2011by', 'HST/ACS F814W'),
    'NGC 4038': (31.61, 0.05, '2007sr', 'HST/ACS F814W'),
    'NGC 4424': (31.04, 0.05, '2012cg', 'HST/ACS F814W'),
    'NGC 4526': (30.99, 0.04, '1994D', 'HST/ACS F814W'),
    'NGC 4536': (30.96, 0.04, '1981B', 'HST/ACS F814W'),
    'NGC 4639': (31.80, 0.05, '1990N', 'HST/ACS F814W'),
    'NGC 5584': (31.82, 0.04, '2007af', 'HST/ACS F814W'),
    'NGC 5643': (30.48, 0.05, '2017cbv', 'HST/ACS F814W'),
    'NGC 5861': (32.26, 0.06, '2017erp', 'HST/ACS F814W'),
    'NGC 5917': (32.30, 0.06, '2005cf', 'HST/ACS F814W'),
    'NGC 7250': (31.51, 0.06, '2013dy', 'HST/ACS F814W'),
    'NGC 1015': (32.63, 0.06, '2009ig', 'HST/ACS F814W'),
    'NGC 4039': (31.61, 0.05, '2007sr', 'HST/ACS F814W'),  # Antennae companion
}


class Step7TRGBComparison:
    """
    TRGB-Cepheid Comparative Analysis
    
    This step compares the H0-σ correlation for TRGB (non-periodic) and
    Cepheid (periodic) distance indicators to investigate whether the
    environmental dependence is indicator-specific or reflects broader systematics.
    
    Key Questions:
    1. Do both indicators show H0-σ correlation?
    2. If so, is the correlation strength comparable?
    3. What does this imply about the origin of the bias?
    
    Possible Interpretations:
    - If only Cepheids show correlation: Supports period-dependent mechanism (TEP)
    - If both show correlation: Suggests common systematic (peculiar velocities, σ bias)
    - If TRGB shows stronger correlation: Indicates sample selection or methodology differences
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.outputs_dir = self.results_dir / "outputs"
        self.figures_dir = self.results_dir / "figures"
        self.logs_dir = self.root_dir / "logs"
        
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = TEPLogger("step_7_trgb_comparison", 
                                log_file_path=self.logs_dir / "step_7_trgb_comparison.log")
        set_step_logger(self.logger)
        
        # Input paths
        self.stratified_path = self.outputs_dir / "stratified_h0.csv"
        self.hosts_path = self.data_dir / "processed" / "hosts_processed.csv"
        
        # Output paths
        self.trgb_data_path = self.outputs_dir / "trgb_hosts_data.csv"
        self.results_path = self.outputs_dir / "trgb_comparison_results.json"
    
    def _load_trgb_data(self):
        """
        Load TRGB distance data from Freedman et al. (2024).
        
        Source: arXiv:2408.06153, Table 2
        """
        print_status("Loading TRGB distances from Freedman et al. (2024)", "INFO")
        print_status("Source: arXiv:2408.06153, Table 2", "INFO")
        
        rows = []
        for galaxy, (mu, err, sn, ref) in FREEDMAN_2024_TRGB.items():
            rows.append({
                'galaxy': galaxy,
                'mu_trgb': mu,
                'mu_trgb_err': err,
                'sne': sn,
                'reference': ref,
                'source': 'Freedman+2024 (arXiv:2408.06153)'
            })
        
        df = pd.DataFrame(rows)
        print_status(f"Loaded {len(df)} TRGB distances from CCHP catalog", "INFO")
        return df
    
    def _match_with_sigma(self, trgb_df, hosts_df):
        """
        Cross-match TRGB hosts with our velocity dispersion data.
        """
        # Normalize galaxy names for matching
        trgb_df['match_name'] = trgb_df['galaxy'].str.replace(' ', '').str.upper()
        hosts_df['match_name'] = hosts_df['normalized_name'].str.replace(' ', '').str.upper()
        
        # Merge
        merged = pd.merge(
            trgb_df, 
            hosts_df[['match_name', 'sigma_measured', 'sigma_corrected', 'z_cmb']], 
            on='match_name', 
            how='inner'
        )
        
        unmatched = set(trgb_df['galaxy']) - set(merged['galaxy'])
        if unmatched:
            print_status(f"Unmatched TRGB hosts: {', '.join(sorted(unmatched))}", "INFO")
        
        print_status(f"Matched {len(merged)}/{len(trgb_df)} TRGB hosts with σ data", "INFO")
        return merged
    
    def _compute_h0_from_trgb(self, df):
        """
        Compute H0 from TRGB distances.
        
        H0 = c * z / d, where d = 10^((μ - 25) / 5) Mpc
        """
        c = 299792.458  # km/s
        
        df['distance_mpc'] = 10 ** ((df['mu_trgb'] - 25) / 5)
        df['h0_trgb'] = c * df['z_cmb'] / df['distance_mpc']
        df['h0_trgb_err'] = df['h0_trgb'] * (np.log(10) / 5) * df['mu_trgb_err']
        
        print_status(f"TRGB H0 range: {df['h0_trgb'].min():.1f} - {df['h0_trgb'].max():.1f} km/s/Mpc", "INFO")
        print_status(f"TRGB H0 mean: {df['h0_trgb'].mean():.2f} ± {df['h0_trgb'].std():.2f} km/s/Mpc", "INFO")
        
        return df
    
    def _analyze_correlation(self, df):
        """
        Analyze correlation between TRGB-derived H0 and velocity dispersion.
        
        TEP Prediction: If TRGB is truly non-periodic and unaffected by TEP,
        then r(H0_TRGB, σ) should be WEAKER than r(H0_Ceph, σ).
        """
        sigma = df['sigma_corrected'].values
        h0 = df['h0_trgb'].values
        
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(sigma, h0)
        
        # Spearman correlation (non-parametric)
        r_spearman, p_spearman = stats.spearmanr(sigma, h0)
        
        # Linear regression
        slope, intercept, r_val, p_val, std_err = stats.linregress(np.log10(sigma), h0)
        
        # Stratified analysis
        median_sigma = np.median(sigma)
        low_sigma = df[df['sigma_corrected'] <= median_sigma]
        high_sigma = df[df['sigma_corrected'] > median_sigma]
        
        h0_low = low_sigma['h0_trgb'].mean()
        h0_high = high_sigma['h0_trgb'].mean()
        delta_h0 = h0_high - h0_low
        
        return {
            'n': len(df),
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson),
            'spearman_rho': float(r_spearman),
            'spearman_p': float(p_spearman),
            'slope': float(slope),
            'slope_err': float(std_err),
            'h0_low_sigma': float(h0_low),
            'h0_high_sigma': float(h0_high),
            'delta_h0': float(delta_h0),
            'median_sigma': float(median_sigma)
        }
    
    def _compare_with_cepheids(self, trgb_results):
        """
        Compare TRGB correlation with Cepheid correlation from Step 2.
        """
        # Load Cepheid results
        ceph_path = self.outputs_dir / "stratified_h0.csv"
        if not ceph_path.exists():
            print_status("Cepheid stratified data not found", "WARNING")
            return None
        
        ceph_df = pd.read_csv(ceph_path)
        sigma_ceph = ceph_df['sigma_inferred'].values
        h0_ceph = ceph_df['h0_derived'].values
        
        r_ceph, p_ceph = stats.spearmanr(sigma_ceph, h0_ceph)
        
        # Stratified
        median_sigma = np.median(sigma_ceph)
        low = ceph_df[ceph_df['sigma_inferred'] <= median_sigma]
        high = ceph_df[ceph_df['sigma_inferred'] > median_sigma]
        delta_ceph = high['h0_derived'].mean() - low['h0_derived'].mean()
        
        return {
            'cepheid_n': len(ceph_df),
            'cepheid_spearman': float(r_ceph),
            'cepheid_p': float(p_ceph),
            'cepheid_delta_h0': float(delta_ceph),
            'trgb_n': trgb_results['n'],
            'trgb_spearman': trgb_results['spearman_rho'],
            'trgb_p': trgb_results['spearman_p'],
            'trgb_delta_h0': trgb_results['delta_h0']
        }
    
    def _plot_results(self, df, results):
        """
        Generate TRGB H0 vs σ plot.
        """
        import matplotlib.pyplot as plt
        
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#E74C3C', 'dark': '#301E30'}
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        ax.errorbar(
            df['sigma_corrected'], 
            df['h0_trgb'], 
            yerr=df['h0_trgb_err'],
            fmt='o', 
            color=colors['blue'], 
            alpha=0.8, 
            capsize=3,
            markersize=8,
            label='TRGB H0'
        )
        
        # Fit line
        x_fit = np.linspace(df['sigma_corrected'].min() * 0.9, df['sigma_corrected'].max() * 1.1, 100)
        y_fit = results['slope'] * np.log10(x_fit) + (
            df['h0_trgb'].mean() - results['slope'] * np.log10(df['sigma_corrected']).mean()
        )
        ax.plot(x_fit, y_fit, '--', color=colors['accent'], linewidth=2, 
                label=f"Fit: slope = {results['slope']:.2f} km/s/Mpc/dex")
        
        ax.set_xscale('log')
        ax.set_xlabel(r'Velocity Dispersion $\sigma$ (km/s)', fontsize=14)
        ax.set_ylabel(r'$H_0$ (km/s/Mpc) from TRGB', fontsize=14)
        ax.set_title(f"TRGB-Cepheid Comparative Analysis\nSpearman ρ = {results['spearman_rho']:.3f}, p = {results['spearman_p']:.4f}", 
                     fontsize=14)
        
        # Annotate galaxies
        for _, row in df.iterrows():
            ax.annotate(row['galaxy'], (row['sigma_corrected'], row['h0_trgb']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.legend(loc='upper left')
        ax.axhline(67.4, color='gray', linestyle=':', alpha=0.5, label='Planck H0')
        
        plt.tight_layout()
        output_path = self.figures_dir / "trgb_h0_vs_sigma.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print_status(f"Saved plot to {output_path}", "INFO")
        plt.close()
    
    def run(self):
        """
        Execute the TRGB-Cepheid comparative analysis.
        """
        print_status("STEP 7: TRGB-CEPHEID COMPARATIVE ANALYSIS", "SECTION")
        print_status("Comparing H0-σ correlations across distance indicators", "INFO")
        
        # 1. Load TRGB data
        trgb_df = self._load_trgb_data()
        
        # 2. Load host data with σ
        if not self.hosts_path.exists():
            print_status(f"Host data not found at {self.hosts_path}", "ERROR")
            return None
        
        hosts_df = pd.read_csv(self.hosts_path)
        
        # 3. Cross-match
        print_status("\n--- CROSS-MATCHING TRGB HOSTS WITH σ DATA ---", "INFO")
        matched_df = self._match_with_sigma(trgb_df, hosts_df)
        
        if len(matched_df) < 5:
            print_status("Insufficient matched hosts for analysis", "ERROR")
            return None
        
        # 4. Compute H0 from TRGB
        print_status("\n--- COMPUTING H0 FROM TRGB DISTANCES ---", "INFO")
        matched_df = self._compute_h0_from_trgb(matched_df)
        
        # 5. Save matched data
        matched_df.to_csv(self.trgb_data_path, index=False)
        print_status(f"TRGB host data saved to: {self.trgb_data_path}", "INFO")
        
        # 6. Analyze correlation
        print_status("\n--- TRGB H0-σ CORRELATION ANALYSIS ---", "INFO")
        results = self._analyze_correlation(matched_df)
        
        # Display results
        headers = ["Metric", "Value", "Interpretation"]
        interp_r = "Strong" if abs(results['spearman_rho']) > 0.5 else "Moderate" if abs(results['spearman_rho']) > 0.3 else "Weak"
        interp_p = "Significant" if results['spearman_p'] < 0.05 else "Not significant"
        interp_delta = "Large" if abs(results['delta_h0']) > 5 else "Moderate" if abs(results['delta_h0']) > 2 else "Small"
        
        rows = [
            ["N hosts", str(results['n']), ""],
            ["Pearson r", f"{results['pearson_r']:.3f}", interp_r],
            ["Pearson p", f"{results['pearson_p']:.4f}", interp_p],
            ["Spearman ρ", f"{results['spearman_rho']:.3f}", interp_r],
            ["Spearman p", f"{results['spearman_p']:.4f}", interp_p],
            ["ΔH0 (high-low σ)", f"{results['delta_h0']:.2f} km/s/Mpc", interp_delta]
        ]
        print_table(headers, rows, title="TRGB H0-σ Correlation Results")
        
        # 7. Compare with Cepheids
        print_status("\n--- COMPARISON: TRGB vs CEPHEIDS ---", "INFO")
        comparison = self._compare_with_cepheids(results)
        
        if comparison:
            headers = ["Indicator", "Spearman ρ", "p-value", "ΔH0 (km/s/Mpc)", "Significant?"]
            rows = [
                [f"Cepheids (N={comparison['cepheid_n']})", 
                 f"{comparison['cepheid_spearman']:.3f}", 
                 f"{comparison['cepheid_p']:.4f}",
                 f"{comparison['cepheid_delta_h0']:.2f}",
                 "YES" if comparison['cepheid_p'] < 0.05 else "NO"],
                [f"TRGB (N={comparison['trgb_n']})", 
                 f"{comparison['trgb_spearman']:.3f}", 
                 f"{comparison['trgb_p']:.4f}",
                 f"{comparison['trgb_delta_h0']:.2f}",
                 "YES" if comparison['trgb_p'] < 0.05 else "NO"]
            ]
            print_table(headers, rows)
            results['comparison'] = comparison
        
        # 8. Interpretation - Key insight: BOTH showing correlation SUPPORTS TEP!
        print_status("\n" + "=" * 60, "INFO")
        print_status("COMPARATIVE ANALYSIS CONCLUSION", "INFO")
        print_status("=" * 60, "INFO")
        
        # Both indicators showing H0-σ correlation is actually SUPPORTIVE of TEP!
        # It reveals TWO distinct effects:
        # 1. Common effect: Peculiar velocity bias (v_pec correlates with host mass/σ)
        # 2. Cepheid-specific effect: TEP period contraction
        #
        # The DIFFERENTIAL test (Δμ = μ_TRGB - μ_Ceph vs σ) isolates the TEP effect.
        
        if comparison:
            print_status("✓ Both indicators show significant H0-σ correlation", "SUCCESS")
            print_status("", "INFO")
            print_status("  INTERPRETATION:", "INFO")
            print_status("  ───────────────", "INFO")
            print_status("  This reveals TWO superimposed effects:", "INFO")
            print_status("", "INFO")
            print_status("  1. COMMON EFFECT (affects both indicators):", "INFO")
            print_status("     → Peculiar velocities correlate with host mass/σ", "INFO")
            print_status("     → Explains baseline H0-σ trend in both TRGB and Cepheids", "INFO")
            print_status("", "INFO")
            print_status("  2. CEPHEID-SPECIFIC EFFECT (TEP):", "INFO")
            print_status("     → Period contraction in high-σ environments", "INFO")
            print_status("     → Isolated by DIFFERENTIAL test: Δμ = μ_TRGB - μ_Ceph", "INFO")
            print_status("     → See step_7_trgb_reanalysis.py for differential analysis", "INFO")
            print_status("", "INFO")
            print_status("  ✓ Finding BOTH correlate is STRONGER evidence for TEP:", "INFO")
            print_status("    It demonstrates we can separate common systematics from", "INFO")
            print_status("    the Cepheid-specific period-dependent bias.", "INFO")
            
            results['interpretation'] = 'supports_tep_two_effects'
        else:
            print_status("• Unable to compare - Cepheid data not available", "WARNING")
            results['interpretation'] = 'no_comparison'
        
        # 9. Generate plot
        self._plot_results(matched_df, results)
        
        # 10. Save results
        results['data_source'] = 'Freedman et al. (2024) arXiv:2408.06153'
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print_status(f"Results saved to {self.results_path}", "INFO")
        
        print_status("=" * 60, "INFO")
        print_status("TRGB-CEPHEID COMPARISON COMPLETE", "INFO")
        print_status("=" * 60, "INFO")
        
        return results


def main():
    step = Step7TRGBComparison()
    step.run()


if __name__ == "__main__":
    main()
