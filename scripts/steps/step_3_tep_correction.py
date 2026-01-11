
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
import sys
import os
import json
import shutil

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    # Add project root to path if needed
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

class Step3TEPCorrection:
    r"""
    Step 3: TEP Correction and Unification
    ======================================
    
    This step applies the Temporal Equivalence Principle (TEP) correction to the 
    Cepheid distance moduli to resolve the Hubble Tension.
    
    The Physics:
    TEP predicts that clocks in deeper gravitational potentials run faster relative 
    to a universal cosmic time. Since Cepheids are standard clocks (Leavitt Law), 
    their periods are contracted in high-density environments (high $\sigma$).
    
    $$ P_{\rm obs} = P_{\rm true} \cdot (1 - \Phi/c^2)^\alpha \approx P_{\rm true} \cdot (1 - \epsilon) $$
    
    This leads to an underestimated luminosity and distance modulus. The correction 
    restores the distance modulus to the value it would have in the calibrator environment:
    
    $$ \mu_{\rm corr} = \mu_{\rm obs} + \alpha \cdot \log_{10}\left(\frac{\sigma}{\sigma_{\rm ref}}\right) $$
    
    Where:
    - $\alpha$: The coupling constant (related to the scalar field screening).
    - $\sigma_{\rm ref}$: The effective velocity dispersion of the calibrator sample (MW, LMC, N4258).
    
    Objective:
    Find the optimal $\alpha$ that minimizes the correlation between corrected $H_0$ and $\sigma$, 
    thereby removing the environmental bias.
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.root_dir / "logs"
        self.figures_dir = self.root_dir / "results" / "figures"
        self.outputs_dir = self.root_dir / "results" / "outputs"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TEPLogger("step_3_correction", log_file_path=self.logs_dir / "step_3_correction.log")
        set_step_logger(self.logger)
        
        # Inputs
        self.input_path = self.outputs_dir / "stratified_h0.csv"
        
        # Outputs
        self.corrected_output_path = self.outputs_dir / "tep_corrected_h0.csv"
        self.json_output_path = self.outputs_dir / "tep_correction_results.json"
        self.plot_path = self.figures_dir / "tep_correction_comparison.png"
        
        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        self.public_figures_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Loads the stratified dataset."""
        print_status("Loading Data...", "SECTION")
        
        if not self.input_path.exists():
            print_status("Input file missing. Please run Step 2 first.", "ERROR")
            sys.exit(1)
            
        df = pd.read_csv(self.input_path)
        print_status(f"Loaded {len(df)} hosts for correction.", "INFO")
        return df

    def calculate_effective_calibrator_sigma(self):
        """
        Calculates the effective velocity dispersion of the anchor sample.
        
        WEIGHTING RATIONALE (SH0ES-motivated):
        The weights reflect each anchor's contribution to the P-L zero-point calibration,
        NOT simply the distance precision. From Riess et al. (2022), the contributions are:
        
        - MW (~20%): Many Cepheids with individual Gaia parallaxes, but higher scatter
        - LMC (~25%): ~70 Cepheids with excellent HST photometry, precise DEB distance
        - N4258 (~55%): ~139 Cepheids, gold-standard maser distance, and critically:
          it is the ONLY anchor that is a GALAXY (like the SN hosts being corrected)
        
        NGC 4258 is weighted most heavily because it provides the most relevant 
        calibration environment for extragalactic distance measurements.
        
        CRITICAL: ANCHOR SIGMA VALUES
        These are DISK velocity dispersions at Cepheid locations, NOT central bulge values.
        Cepheids reside in galactic disks at R ~ 4-8 kpc, where the local potential is
        shallower than the nuclear region. Using central bulge dispersions (e.g., Ho+2009
        gives NGC 4258 = 148 km/s) would overestimate the effective potential.
        
        Sources:
        - MW: Bovy+2012 thin disk σ_z at solar neighborhood = 30 km/s
        - LMC: van der Marel+2002 disk dispersion = 24 km/s  
        - N4258: Kormendy & Ho 2013 intermediate-aperture value = 115 km/s
          (NOT the central bulge value of 148 km/s from Ho+2009)
        
        The resulting σ_ref ≈ 75 km/s is validated by the empirical result that
        low-σ SN hosts (σ < 90 km/s) yield H0 = 67.8 km/s/Mpc, matching Planck.
        """
        print_status("Calculating Effective Calibrator Sigma...", "SECTION")
        
        # Anchor velocity dispersions: DISK values at Cepheid locations
        # These are intentionally different from the central bulge values in the
        # regenerated sigma catalog, which uses Ho+2009 nuclear apertures.
        anchors = [
            {
                "ID": "MW",
                "Sigma": 30.0,
                "Desc": "Milky Way Disk (Bovy+2012)",
                "Weight": 0.20  # ~270 Cepheids, individual parallaxes, higher scatter
            },
            {
                "ID": "LMC",
                "Sigma": 24.0,
                "Desc": "LMC Disk (vdMarel+2002)",
                "Weight": 0.25  # ~70 Cepheids, excellent photometry
            },
            {
                "ID": "N4258",
                "Sigma": 115.0,
                "Desc": "NGC 4258 Disk (K&H2013)",
                "Weight": 0.55  # ~139 Cepheids, gold anchor, galaxy environment
            }
        ]
        
        print_status("Using SH0ES-motivated weights (based on P-L contribution).", "INFO")
        
        # Display Anchor Table
        headers = ["Anchor", "Sigma (km/s)", "Weight", "Description"]
        rows = []
        numerator = 0
        denominator = 0
        
        for a in anchors:
            rows.append([a["ID"], f"{a['Sigma']:.1f}", f"{a['Weight']:.2f}", a["Desc"]])
            numerator += a["Sigma"] * a["Weight"]
            denominator += a["Weight"]
            
        print_table(headers, rows, title="Geometric Anchor Sample")
        
        sigma_ref = numerator / denominator
        print_status(f"Effective Reference Sigma (σ_ref): {sigma_ref:.2f} km/s", "SUCCESS")
        
        return sigma_ref

    def optimize_correction(self, df, sigma_ref):
        """Finds the optimal correction parameter alpha."""
        print_status("Optimizing TEP Coupling (Alpha)...", "SECTION")
        
        # Objective function: minimize H0 vs σ correlation
        # We want the corrected H0 to be independent of environment (slope ~ 0)
        def objective(params):
            alpha = params[0]
            
            # Correction model: mu_corr = mu_obs + alpha * log10(sigma / sigma_ref)
            correction = alpha * np.log10(df['sigma_inferred'] / sigma_ref)
            mu_corr = df['value'] + correction
            
            d_corr = 10**((mu_corr - 25)/5)
            h0_corr = df['velocity'] / d_corr
            
            # Minimize squared slope of H0 vs Sigma
            # We use polyfit slope instead of correlation to handle scale
            slope, _ = np.polyfit(df['sigma_inferred'], h0_corr, 1)
            
            return slope**2
            
        # Optimize
        initial_guess = [0.5]
        res = minimize(objective, x0=initial_guess, method='Nelder-Mead', tol=1e-4)
        best_alpha = res.x[0]
        
        print_status(f"Optimization converged: {res.success}", "INFO")
        print_status(f"Optimal Coupling Constant (α): {best_alpha:.4f}", "SUCCESS")
        
        return best_alpha

    def apply_correction(self, df, alpha, sigma_ref):
        """Applies the correction and calculates stats."""
        print_status("Applying Conformal Correction...", "SECTION")
        
        correction = alpha * np.log10(df['sigma_inferred'] / sigma_ref)
        df['mu_corrected'] = df['value'] + correction
        df['dist_corrected'] = 10**((df['mu_corrected'] - 25)/5)
        df['h0_corrected'] = df['velocity'] / df['dist_corrected']
        
        # Sample Correction Table
        headers = ["Host", "Sigma", "H0 (Raw)", "Corr (mag)", "H0 (TEP)"]
        rows = []
        sample = df.sample(5, random_state=42).sort_values('sigma_inferred')
        for _, row in sample.iterrows():
            rows.append([
                row['normalized_name'],
                f"{row['sigma_inferred']:.1f}",
                f"{row['h0_derived']:.2f}",
                f"{alpha * np.log10(row['sigma_inferred'] / sigma_ref):+.3f}",
                f"{row['h0_corrected']:.2f}"
            ])
        print_table(headers, rows, title="Sample Corrections")
        
        h0_mean = df['h0_corrected'].mean()
        # Standard error of the mean (simple)
        h0_sem = df['h0_corrected'].std() / np.sqrt(len(df))
        
        print_status("-" * 60, "INFO")
        print_status(f"UNIFIED H0 (Statistical): {h0_mean:.2f} +/- {h0_sem:.2f} km/s/Mpc", "SUCCESS")
        print_status("Note: This error (SEM) assumes alpha is fixed/known perfectly.", "INFO")
        print_status("-" * 60, "INFO")
        
        return df, h0_mean, h0_sem

    def bootstrap_analysis(self, df, sigma_ref, n_boot=1000):
        """Performs bootstrap resampling to estimate robustness."""
        print_status(f"Bootstrap Uncertainty Analysis (N={n_boot})...", "SECTION")
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        alphas = []
        h0s = []
        
        n_samples = len(df)
        
        # Suppress optimization warnings for speed
        import warnings
        warnings.filterwarnings('ignore')
        
        for _ in range(n_boot):
            # Resample with replacement
            sample = df.sample(n=n_samples, replace=True)
            
            # Re-optimize alpha inline
            # We capture the variation in alpha due to sample selection
            def obj(a):
                corr = a * np.log10(sample['sigma_inferred'] / sigma_ref)
                mc = sample['value'] + corr
                dc = 10**((mc - 25)/5)
                hc = sample['velocity'] / dc
                slope, _ = np.polyfit(sample['sigma_inferred'], hc, 1)
                return slope**2
            
            res = minimize(obj, x0=[0.7], method='Nelder-Mead', tol=1e-2)
            alpha_boot = res.x[0]
            
            # Calc H0 with the re-optimized alpha
            corr = alpha_boot * np.log10(sample['sigma_inferred'] / sigma_ref)
            mc = sample['value'] + corr
            dc = 10**((mc - 25)/5)
            hc = sample['velocity'] / dc
            h0_boot = hc.mean()
            
            alphas.append(alpha_boot)
            h0s.append(h0_boot)
            
        alphas = np.array(alphas)
        h0s = np.array(h0s)
        warnings.resetwarnings()
        
        metrics = {
            "bootstrap_alpha_mean": float(np.mean(alphas)),
            "bootstrap_alpha_std": float(np.std(alphas)),
            "bootstrap_h0_mean": float(np.mean(h0s)),
            "bootstrap_h0_std": float(np.std(h0s))
        }
        
        # Bootstrap Results Table
        headers = ["Parameter", "Mean", "Std Dev (Robust Error)", "95% CI"]
        rows = [
            [
                "Alpha (α)", 
                f"{metrics['bootstrap_alpha_mean']:.3f}", 
                f"{metrics['bootstrap_alpha_std']:.3f}",
                f"[{np.percentile(alphas, 2.5):.3f}, {np.percentile(alphas, 97.5):.3f}]"
            ],
            [
                "H0 (Unified)", 
                f"{metrics['bootstrap_h0_mean']:.2f}", 
                f"{metrics['bootstrap_h0_std']:.2f}",
                f"[{np.percentile(h0s, 2.5):.2f}, {np.percentile(h0s, 97.5):.2f}]"
            ]
        ]
        print_table(headers, rows, title="Bootstrap Results (Includes Optimization Uncertainty)")
        
        return metrics

    def sensitivity_analysis(self, df):
        """Analyzes sensitivity of H0 to sigma_ref."""
        print_status("Sensitivity Analysis (Sigma Ref Scan)...", "PROCESS")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'green': '#4a2650', 'dark': '#301E30', 'light_blue': '#4b6785'}

        sigma_refs = np.linspace(30, 130, 20)
        h0_results = []
        
        planck_h0 = 67.4
        
        for sr in sigma_refs:
            # Re-optimize alpha for this sr
            alpha = self.optimize_correction(df, sr)
            
            # Apply
            correction = alpha * np.log10(df['sigma_inferred'] / sr)
            mu_corr = df['value'] + correction
            dist_corr = 10**((mu_corr - 25)/5)
            h0_corr = df['velocity'] / dist_corr
            
            h0_mean = h0_corr.mean()
            h0_results.append(h0_mean)
            
        # Plot
        plt.figure(figsize=(14, 9))
        plt.plot(sigma_refs, h0_results, marker='o', color=colors['blue'], label='Unified H0', linewidth=2.5)
        plt.axhline(planck_h0, color=colors['accent'], linestyle='--', label='Planck CMB', linewidth=2.5)
        plt.fill_between(sigma_refs, planck_h0 - 0.5, planck_h0 + 0.5, color=colors['accent'], alpha=0.15)
        
        plt.xlabel(r'Reference $\sigma_{ref}$ (km/s)')
        plt.ylabel(r'Unified $H_0$ (km/s/Mpc)')
        plt.title('Sensitivity of H0 to Calibrator Reference')
        plt.legend()
        plt.tight_layout()
        
        path = self.figures_dir / "sensitivity_h0_vs_sigmaref.png"
        plt.savefig(path, dpi=300)
        print_status(f"Saved sensitivity plot to {path}", "SUCCESS")
        plt.close()
        
        # Copy to public
        public_path = self.public_figures_dir / "sensitivity_h0_vs_sigmaref.png"
        shutil.copy(path, public_path)
        print_status(f"Copied sensitivity plot to {public_path}", "SUCCESS")
        
        return list(zip(sigma_refs, h0_results))

    def plot_comparison(self, df, h0_mean):
        """Generates comparison plots."""
        print_status("Generating Comparison Plots...", "PROCESS")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'green': '#4a2650', 'dark': '#301E30', 'light_blue': '#4b6785'}
        
        # Use a wide figure for side-by-side
        plt.figure(figsize=(14, 9))
        
        # Original
        plt.subplot(1, 2, 1)
        plt.scatter(df['sigma_inferred'], df['h0_derived'], alpha=0.8, s=80, color=colors.get('light_blue', '#4b6785'), label='Original', edgecolor='white', linewidth=0.5)
        
        if len(df) > 1:
            z = np.polyfit(df['sigma_inferred'], df['h0_derived'], 1)
            p = np.poly1d(z)
            x = np.linspace(df['sigma_inferred'].min(), df['sigma_inferred'].max(), 100)
            plt.plot(x, p(x), color=colors['dark'], linestyle='--', linewidth=3, label='Trend')
            
        plt.title(f"Original Data\nMean H0: {df['h0_derived'].mean():.2f}")
        plt.xlabel(r'Velocity Dispersion $\sigma$ (km/s)')
        plt.ylabel(r'$H_0$ (km/s/Mpc)')
        plt.ylim(55, 85)
        plt.legend()
        
        # Corrected
        plt.subplot(1, 2, 2)
        plt.scatter(df['sigma_inferred'], df['h0_corrected'], alpha=0.8, s=80, color=colors['blue'], label='TEP Corrected', edgecolor='white', linewidth=0.5)
        
        if len(df) > 1:
            z2 = np.polyfit(df['sigma_inferred'], df['h0_corrected'], 1)
            p2 = np.poly1d(z2)
            plt.plot(x, p2(x), color=colors['blue'], linestyle='--', linewidth=3, label='Trend')
            
        plt.axhline(67.4, color=colors['accent'], linestyle=':', linewidth=2.5, label='Planck CMB')
        plt.title(f"TEP Corrected\nMean H0: {h0_mean:.2f}")
        plt.xlabel(r'Velocity Dispersion $\sigma$ (km/s)')
        plt.ylim(55, 85)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved comparison plot to {self.plot_path}", "SUCCESS")
        plt.close()
        
        # Copy to public
        public_path = self.public_figures_dir / "tep_correction_comparison.png"
        shutil.copy(self.plot_path, public_path)
        print_status(f"Copied comparison plot to {public_path}", "SUCCESS")

    def run(self):
        print_status("Starting Step 3: TEP Correction", "TITLE")
        
        df = self.load_data()
        
        # 1. Dynamic Sigma Ref
        sigma_ref = self.calculate_effective_calibrator_sigma()
        
        # 2. Optimize
        alpha = self.optimize_correction(df, sigma_ref)
        
        # 3. Apply
        final_df, h0_mean, h0_sem = self.apply_correction(df, alpha, sigma_ref)
        
        # 4. Generate Comparison Plot
        self.plot_comparison(final_df, h0_mean)
        
        # 5. Bootstrap
        boot_metrics = self.bootstrap_analysis(final_df, sigma_ref)
        
        # 6. Sensitivity
        self.sensitivity_analysis(final_df)
        
        # Combine Error Budget
        # We report Bootstrap STD as the robust error (conservative).
        robust_error = boot_metrics['bootstrap_h0_std']
        
        # Planck Comparison
        planck_h0 = 67.4
        planck_err = 0.5
        
        # Tension Calculation
        # We calculate tension using both errors for transparency
        tension_stat = abs(h0_mean - planck_h0) / np.sqrt(h0_sem**2 + planck_err**2)
        tension_robust = abs(h0_mean - planck_h0) / np.sqrt(robust_error**2 + planck_err**2)
        
        print_status("Final Tension Analysis", "SECTION")
        print_status(f"Planck 2018 Value: {planck_h0} +/- {planck_err}", "INFO")
        print_status("-" * 60, "INFO")
        print_status(f"TEP Unified Value: {h0_mean:.2f}", "INFO")
        print_status(f"  +/- {h0_sem:.2f} (Statistical SEM)", "INFO")
        print_status(f"  +/- {robust_error:.2f} (Robust Bootstrap)", "INFO")
        print_status("-" * 60, "INFO")
        print_status(f"Tension (Statistical): {tension_stat:.2f} sigma", "RESULT")
        print_status(f"Tension (Robust):      {tension_robust:.2f} sigma", "RESULT")
        
        if tension_robust < 1.0:
            print_status("CONCLUSION: Result is consistent with Planck CMB (Robust).", "SUCCESS")
        else:
            print_status("CONCLUSION: Tension remains.", "WARNING")
        
        # Compile and Save Results
        results = {
            "optimal_alpha": float(alpha),
            "sigma_ref": float(sigma_ref),
            "unified_h0": float(h0_mean),
            "h0_sem": float(h0_sem),
            "bootstrap_h0_mean": float(boot_metrics['bootstrap_h0_mean']),
            "bootstrap_h0_std": float(boot_metrics['bootstrap_h0_std']),
            "bootstrap_alpha_mean": float(boot_metrics['bootstrap_alpha_mean']),
            "bootstrap_alpha_std": float(boot_metrics['bootstrap_alpha_std']),
            "planck_h0": float(planck_h0),
            "tension_sigma": float(tension_robust),
            "is_consistent": bool(tension_robust < 1.0),
            "n_hosts": len(final_df)
        }
        
        with open(self.json_output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print_status(f"Saved results JSON to {self.json_output_path}", "SUCCESS")
        
        final_df.to_csv(self.corrected_output_path, index=False)
        print_status(f"Saved corrected data to {self.corrected_output_path}", "SUCCESS")

def main():
    step = Step3TEPCorrection()
    step.run()

if __name__ == "__main__":
    main()
