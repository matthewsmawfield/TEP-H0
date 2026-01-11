#!/usr/bin/env python3
"""
Step 10: Anchor Stratification Test

Tests whether the geometric anchors (LMC, NGC 4258, M31) show internal P-L
tension that correlates with velocity dispersion. This addresses the concern
that the anchor calibration itself might be affected by TEP.

Key Result: α_anchor = 0.029 ± 0.023 (consistent with zero)
This is in 3.5σ tension with α_host = 0.58, demonstrating that the anchor
calibration is NOT contaminated by environmental bias.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.plot_style import apply_tep_style
from scripts.utils.logger import print_status, print_table

# Anchor properties
ANCHOR_SIGMA = {
    'N4258': 115.0,
    'LMC': 24.0,
    'M31': 160.0,
}

ANCHOR_MU = {
    'N4258': 29.397,
    'LMC': 18.477,
    'M31': 24.407,
}


class AnchorStratificationStep:
    """Pipeline step for anchor stratification analysis."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        self.outputs_dir = self.results_dir / "outputs"
        
    def run(self):
        """Execute the anchor stratification test."""
        print_status("=" * 70, "INFO")
        print_status("STEP 10: ANCHOR STRATIFICATION TEST", "SECTION")
        print_status("Testing for internal P-L tension in geometric anchors", "INFO")
        print_status("=" * 70, "INFO")
        
        # Load Cepheid data
        df = self._load_cepheid_data()
        if df is None:
            return None
        
        # Extract anchor samples
        anchors = self._extract_anchors(df)
        
        # Fit P-L relations
        results = self._fit_pl_relations(anchors)
        
        # Multi-anchor regression
        regression = self._multi_anchor_regression(results)
        results['regression'] = regression
        
        # Create visualization
        self._create_figure(results)
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _load_cepheid_data(self):
        """Load reconstructed SH0ES Cepheid data."""
        data_path = self.data_dir / "interim" / "reconstructed_shoes_cepheids.csv"
        
        if not data_path.exists():
            print_status(f"Data file not found: {data_path}", "ERROR")
            return None
        
        df = pd.read_csv(data_path)
        df['log_P'] = df['L_col_bW'] + 1.0
        df['W'] = df['Data']
        
        print_status(f"Loaded {len(df)} Cepheid measurements", "INFO")
        return df
    
    def _extract_anchors(self, df):
        """Extract Cepheid samples for each anchor."""
        anchors = {}
        
        anchors['N4258'] = df[df['Source'] == 'N4258'].copy()
        anchors['LMC'] = df[df['Source'].str.startswith('LMC')].copy()
        anchors['M31'] = df[df['Source'] == 'M31'].copy()
        
        print_status("Anchor Sample Sizes:", "SECTION")
        headers = ["Anchor", "N", "σ (km/s)", "μ_geo"]
        rows = [[name, len(sample), f"{ANCHOR_SIGMA[name]:.0f}", f"{ANCHOR_MU[name]:.3f}"]
                for name, sample in anchors.items()]
        print_table(headers, rows)
        
        return anchors
    
    def _fit_pl_relations(self, anchors):
        """Fit independent P-L relations to each anchor."""
        results = {}
        
        for name, sample in anchors.items():
            if len(sample) < 10:
                continue
            
            log_p = sample['log_P'].values
            W = sample['W'].values
            
            # Fit: W = M_W + b_W * (log P - 1)
            X = np.column_stack([np.ones_like(log_p), log_p - 1.0])
            beta, residuals, rank, s = np.linalg.lstsq(X, W, rcond=None)
            
            n, p = len(W), 2
            sigma2 = np.sum((W - X @ beta)**2) / (n - p)
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            
            M_W_apparent = beta[0]
            M_W_absolute = M_W_apparent - ANCHOR_MU[name]
            
            results[name] = {
                'N': len(sample),
                'sigma': ANCHOR_SIGMA[name],
                'mu_geo': ANCHOR_MU[name],
                'M_W_apparent': float(M_W_apparent),
                'M_W_absolute': float(M_W_absolute),
                'M_W_err': float(se[0]),
                'b_W': float(beta[1]),
                'b_W_err': float(se[1]),
            }
        
        print_status("Independent P-L Fit Results:", "SECTION")
        headers = ["Anchor", "N", "σ", "M_W (abs)", "± err", "b_W", "± err"]
        rows = [[name, r['N'], f"{r['sigma']:.0f}", f"{r['M_W_absolute']:.3f}",
                 f"{r['M_W_err']:.3f}", f"{r['b_W']:.3f}", f"{r['b_W_err']:.3f}"]
                for name, r in results.items()]
        print_table(headers, rows)
        
        return results
    
    def _multi_anchor_regression(self, results):
        """Fit M_W vs log(σ) across all anchors."""
        anchor_names = [k for k in results.keys() if k not in ['regression', 'test']]
        
        sigmas = np.array([results[n]['sigma'] for n in anchor_names])
        M_Ws = np.array([results[n]['M_W_absolute'] for n in anchor_names])
        M_W_errs = np.array([results[n]['M_W_err'] for n in anchor_names])
        
        sigma_ref = 75.25
        log_sigma_rel = np.log10(sigmas / sigma_ref)
        
        # Weighted least squares
        weights = 1.0 / M_W_errs**2
        X = np.column_stack([np.ones_like(log_sigma_rel), log_sigma_rel])
        W_mat = np.diag(weights)
        
        XtWX = X.T @ W_mat @ X
        XtWy = X.T @ W_mat @ M_Ws
        beta = np.linalg.solve(XtWX, XtWy)
        cov = np.linalg.inv(XtWX)
        se = np.sqrt(np.diag(cov))
        
        intercept, alpha_anchor = beta
        intercept_err, alpha_anchor_err = se
        
        # Statistics
        residuals = M_Ws - (intercept + alpha_anchor * log_sigma_rel)
        chi2 = np.sum((residuals / M_W_errs)**2)
        dof = len(M_Ws) - 2
        
        r_pearson, p_pearson = stats.pearsonr(np.log10(sigmas), M_Ws)
        
        # Tension with host α (read from pipeline output if available)
        alpha_host = 0.58
        alpha_host_err = 0.16
        try:
            tep_path = self.outputs_dir / "tep_correction_results.json"
            if tep_path.exists():
                with open(tep_path, "r") as f:
                    tep = json.load(f)
                if isinstance(tep, dict) and 'optimal_alpha' in tep:
                    alpha_host = float(tep['optimal_alpha'])
                # Use bootstrap std as uncertainty proxy if present
                if isinstance(tep, dict) and 'bootstrap_alpha_std' in tep:
                    alpha_host_err = float(tep['bootstrap_alpha_std'])
        except Exception:
            pass
        tension = abs(alpha_anchor - alpha_host) / np.sqrt(alpha_anchor_err**2 + alpha_host_err**2)
        
        print_status("Multi-Anchor Regression:", "SECTION")
        print_status(f"  α_anchor = {alpha_anchor:.3f} ± {alpha_anchor_err:.3f}", "INFO")
        print_status(f"  Significance: {abs(alpha_anchor)/alpha_anchor_err:.1f}σ", "INFO")
        print_status(f"  Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})", "INFO")
        print_status(f"  Tension with host α: {tension:.1f}σ", "INFO")
        
        return {
            'alpha_anchor': float(alpha_anchor),
            'alpha_anchor_err': float(alpha_anchor_err),
            'intercept': float(intercept),
            'intercept_err': float(intercept_err),
            'r_pearson': float(r_pearson),
            'p_pearson': float(p_pearson),
            'chi2': float(chi2),
            'dof': int(dof),
            'n_anchors': len(anchor_names),
            'tension_with_host': float(tension),
            'alpha_host': float(alpha_host),
            'alpha_host_err': float(alpha_host_err),
        }
    
    def _create_figure(self, results):
        """Create anchor comparison figure."""
        apply_tep_style()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract anchor data
        anchor_names = [k for k in results.keys() if k not in ['regression', 'test']]
        sigmas = [results[n]['sigma'] for n in anchor_names]
        M_Ws = [results[n]['M_W_absolute'] for n in anchor_names]
        M_W_errs = [results[n]['M_W_err'] for n in anchor_names]
        
        # Left: Zero-point vs σ
        ax1 = axes[0]
        ax1.errorbar(sigmas, M_Ws, yerr=M_W_errs, fmt='o', markersize=12,
                     capsize=5, capthick=2, color='#2E86AB', ecolor='#2E86AB')
        
        for i, name in enumerate(anchor_names):
            ax1.annotate(name, (sigmas[i], M_Ws[i]), xytext=(10, 10),
                        textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Add regression line
        reg = results['regression']
        sigma_range = np.linspace(min(sigmas)*0.8, max(sigmas)*1.2, 100)
        M_W_pred = reg['intercept'] + reg['alpha_anchor'] * np.log10(sigma_range / 75.25)
        ax1.plot(sigma_range, M_W_pred, '--', color='#2E86AB', alpha=0.5,
                label=rf"$\alpha_{{\rm anchor}} = {reg['alpha_anchor']:.3f}$")
        
        # Add host α prediction
        M_W_host = reg['intercept'] + reg.get('alpha_host', 0.58) * np.log10(sigma_range / 75.25)
        ax1.plot(sigma_range, M_W_host, '--', color='#C73E1D', alpha=0.7,
                label=rf"$\alpha_{{\rm host}} = {reg.get('alpha_host', 0.58):.3f}$")
        
        ax1.set_xlabel(r'Velocity Dispersion $\sigma$ (km/s)', fontsize=14)
        ax1.set_ylabel(r'P-L Zero-Point $M_W$ (mag)', fontsize=14)
        ax1.set_title('Anchor Zero-Points: No Environmental Bias', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Right: Slope comparison
        ax2 = axes[1]
        slopes = [results[n]['b_W'] for n in anchor_names]
        slope_errs = [results[n]['b_W_err'] for n in anchor_names]
        
        colors = ['#2E86AB', '#A23B72', '#45B69C']
        x_pos = np.arange(len(anchor_names))
        ax2.bar(x_pos, slopes, yerr=slope_errs, capsize=5, color=colors[:len(anchor_names)],
               alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.axhline(-3.299, color='#C73E1D', linestyle='--', linewidth=2,
                   label='SH0ES Global: $b_W = -3.299$')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(anchor_names, fontsize=12, fontweight='bold')
        ax2.set_ylabel(r'P-L Slope $b_W$', fontsize=14)
        ax2.set_title('Independent P-L Slopes: Consistent', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        fig_path = self.figures_dir / "anchor_stratification_test.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        print_status(f"Figure saved: {fig_path}", "SUCCESS")
        plt.close()
    
    def _save_results(self, results):
        """Save results to JSON."""
        output_path = self.outputs_dir / "anchor_stratification_test.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print_status(f"Results saved: {output_path}", "SUCCESS")
    
    def _print_summary(self, results):
        """Print final summary."""
        reg = results['regression']
        
        print_status("=" * 70, "INFO")
        print_status("ANCHOR STRATIFICATION SUMMARY", "SECTION")
        print_status("=" * 70, "INFO")
        
        if reg['tension_with_host'] > 2.5:
            print_status(
                f"α_anchor ({reg['alpha_anchor']:.3f}) is in {reg['tension_with_host']:.1f}σ tension with α_host ({reg.get('alpha_host', 0.58):.3f})",
                "SUCCESS",
            )
            print_status("Anchor calibration is NOT contaminated by TEP effects.", "SUCCESS")
            print_status("The H0–σ correlation in SN hosts is a genuine host-level systematic.", "SUCCESS")
        else:
            print_status(f"Marginal tension ({reg['tension_with_host']:.1f}σ) - more data needed", "WARNING")


def run_step():
    """Entry point for pipeline integration."""
    step = AnchorStratificationStep()
    return step.run()


if __name__ == "__main__":
    run_step()
