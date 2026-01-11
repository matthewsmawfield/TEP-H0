"""
Step 7: TRGB Re-Analysis (Corrected Logic)
===========================================

Objective: Test TEP prediction using the DIFFERENCE between TRGB and Cepheid distance moduli.

TEP Physics Recap:
1. High-σ environment (deep potential) → Time dilation/Clock rate difference.
2. If local clocks run "fast" relative to standard (Period Contraction):
   - Observed Period P is smaller.
   - Leavitt Law (M ~ -3 log P) infers a fainter Absolute Magnitude M (less negative).
   - Distance Modulus μ = m - M.
   - If inferred M is too faint (too positive), inferred μ is too small.
   - Result: Cepheid distances are UNDERESTIMATED in high-σ hosts.
   - H0 = cz/d is OVERESTIMATED.

Prediction for TRGB:
1. TRGB is a standard candle (core Helium flash), non-periodic.
2. Assumed to be independent of TEP (or much less sensitive).
3. Therefore, μ_TRGB should be "correct" (or at least uncorrelated with σ).

Differential Prediction:
- Define Δμ = μ_TRGB - μ_Ceph
- Low σ: TEP effect small. Δμ ≈ 0.
- High σ: μ_Ceph is underestimated (too small). μ_TRGB is correct.
- Therefore, μ_TRGB > μ_Ceph.
- Δμ should be POSITIVE.
- Correlation: Δμ should POSITIVELY correlate with σ.

Previous Analysis Error:
- We incorrectly interpreted a positive correlation as "inconsistent".
- In fact, a positive correlation (slope > 0) is exactly what TEP predicts!

Data:
- TRGB: CCHP (Freedman et al. 2024)
- Cepheids: SH0ES (Riess et al. 2022) via our 'stratified_h0.csv'
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys
import matplotlib.pyplot as plt

try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

class Step7TRGBReanalysis:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.results_dir = self.root_dir / "results"
        self.outputs_dir = self.results_dir / "outputs"
        self.figures_dir = self.results_dir / "figures"
        self.logs_dir = self.root_dir / "logs"
        
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = TEPLogger("step_7_trgb_reanalysis", log_file_path=self.logs_dir / "step_7_trgb_reanalysis.log")
        set_step_logger(self.logger)
        
        self.trgb_data_path = self.outputs_dir / "trgb_hosts_data.csv"
        self.ceph_data_path = self.outputs_dir / "stratified_h0.csv"
        
    def run(self):
        print_status("STEP 7: TRGB RE-ANALYSIS (DIFFERENTIAL TEST)", "SECTION")
        
        # Load data
        if not self.trgb_data_path.exists():
            print_status("TRGB data file not found. Run previous step first?", "ERROR")
            return
            
        trgb = pd.read_csv(self.trgb_data_path)
        ceph = pd.read_csv(self.ceph_data_path)
        
        # Merge
        trgb['match'] = trgb['galaxy'].str.replace(' ', '').str.upper()
        ceph['match'] = ceph['normalized_name'].str.replace(' ', '').str.upper()
        
        merged = pd.merge(trgb, ceph, on='match', suffixes=('_trgb', '_ceph'))
        
        print_status(f"Matched {len(merged)} hosts for differential analysis", "INFO")
        
        # 1. Calculate Differential Modulus
        # μ_Ceph is 'value' in stratified_h0.csv (the SH0ES distance modulus)
        merged['mu_ceph'] = merged['value']
        merged['delta_mu'] = merged['mu_trgb'] - merged['mu_ceph']
        
        # 2. Sigma Stratification
        # Use our corrected sigma
        merged['sigma'] = merged['sigma_corrected_trgb']
        merged['log_sigma'] = np.log10(merged['sigma'])
        
        # 3. Statistics
        r, p = stats.pearsonr(merged['sigma'], merged['delta_mu'])
        rho, prho = stats.spearmanr(merged['sigma'], merged['delta_mu'])
        
        slope, intercept, r_val, p_val, std_err = stats.linregress(merged['log_sigma'], merged['delta_mu'])
        
        # 4. Display Results
        print_status("\nDifferential Modulus (TRGB - Cepheid) vs Sigma:", "INFO")
        headers = ["Metric", "Value", "Significance"]
        rows = [
            ["Pearson r", f"{r:.3f}", f"p={p:.4f}"],
            ["Spearman ρ", f"{rho:.3f}", f"p={prho:.4f}"],
            ["Slope (mag/dex)", f"{slope:.3f} ± {std_err:.3f}", f"p={p_val:.4f}"]
        ]
        print_table(headers, rows)
        
        # 5. Interpretation
        print_status("\nInterpretation Check:", "INFO")
        if r > 0.3:
            print_status("  POSITIVE correlation detected.", "RESULT")
            print_status("  High-σ hosts have μ_TRGB > μ_Ceph.", "INFO")
            print_status("  Implies μ_Ceph is UNDERESTIMATED at high σ.", "INFO")
            print_status("  → CONSISTENT WITH TEP PREDICTION.", "SUCCESS")
        else:
            print_status("  No significant positive correlation.", "RESULT")
            print_status("  → INCONCLUSIVE / TENSION.", "WARNING")
            
        # 6. Save Plot
        self._plot_differential(merged, slope, intercept, r, p)
        
        # 7. Write Results
        results = {
            'n': len(merged),
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': prho,
            'slope': slope,
            'slope_err': std_err,
            'consistent_with_tep': bool(r > 0.3)
        }
        
        # Save results to JSON
        import json
        output_file = self.outputs_dir / "trgb_differential_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print_status(f"Results saved to {output_file}", "INFO")
        
        return results

    def _plot_differential(self, df, slope, intercept, r, p):
        import matplotlib.pyplot as plt
        
        # Apply TEP style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            # Fallback if running as script
            sys.path.append(str(Path(__file__).resolve().parents[3]))
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        
        fig, ax = plt.subplots(figsize=(14, 9)) # Standard TEP size
        
        # Error bars: combine TRGB and Cepheid errors
        # Cepheid error is 'error' column in stratified_h0
        df['total_err'] = np.sqrt(df['mu_trgb_err']**2 + df['error']**2)
        
        ax.errorbar(df['sigma'], df['delta_mu'], yerr=df['total_err'], 
                   fmt='o', color=colors['dark'], alpha=0.8, label='Data', capsize=3, markeredgecolor=colors['purple'])
        
        # Fit line
        x_range = np.linspace(df['sigma'].min()*0.9, df['sigma'].max()*1.1, 100)
        y_fit = slope * np.log10(x_range) + intercept
        ax.plot(x_range, y_fit, linestyle='--', linewidth=2.5, color=colors['accent'], label=f'Fit: slope={slope:.2f} mag/dex')
        
        # Zero line
        ax.axhline(0, color=colors['purple'], linestyle=':', alpha=0.6, linewidth=1.5)
        
        ax.set_xscale('log')
        ax.set_xlabel(r'Velocity Dispersion $\sigma$ (km/s)')
        ax.set_ylabel(r'$\Delta \mu = \mu_{\rm TRGB} - \mu_{\rm Cepheid}$ (mag)')
        ax.set_title(f'TRGB - Cepheid Differential Distance\nPearson $r={r:.2f}$, $p={p:.3f}$')
        
        # Add galaxy names with smart offset avoiding clutter
        offsets = {
            'NGC 1309': (-15, -15),  # Clustered with 4639, 1015
            'NGC 4639': (10, 0),     # Clustered
            'NGC 3021': (5, 10),     # High point in cluster
            'NGC 1015': (-25, 5),    # Left of cluster
            'NGC 3370': (0, -15),    # Low point
            'NGC 5917': (-10, -15),  # Low cluster
            'NGC 5584': (5, -15),    # Low cluster
            'NGC 4038': (-20, 5),    # Near 2442
            'NGC 2442': (10, -5),    # Near 4038
            'NGC 5861': (0, 10),     # High point
            'NGC 1365': (-30, 0),    # High sigma
            'NGC 7250': (10, 0),     # Low sigma outlier
            'NGC 1559': (0, -15)     # Isolated
        }

        for _, row in df.iterrows():
            name = row['galaxy']
            xytext = offsets.get(name, (5, 5))
            
            ax.annotate(name, (row['sigma'], row['delta_mu']), 
                       xytext=xytext, textcoords='offset points', 
                       fontsize=10, alpha=0.9, color=colors['dark'],
                       fontweight='bold' if name in ['NGC 1309', 'NGC 4639'] else 'normal')
        
        ax.legend(frameon=True)
        # Grid is handled by apply_tep_style
        
        output_path = self.figures_dir / "trgb_cepheid_residual.png"
        plt.savefig(output_path, bbox_inches='tight')
        print_status(f"Plot saved to {output_path}", "INFO")
        plt.close()

def main():
    step = Step7TRGBReanalysis()
    step.run()

if __name__ == "__main__":
    main()
