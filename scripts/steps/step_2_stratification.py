
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json
import shutil

# Astronomy imports (environment catalog)
try:
    from astroquery.vizier import Vizier
except Exception:
    Vizier = None

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    # Add project root to path if needed
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

class Step2Stratification:
    r"""
    Step 2: Host Stratification and H0 Analysis
    ===========================================
    
    This step performs the core phenomenological test of the TEP hypothesis:
    Does the inferred Hubble Constant ($H_0$) depend on the gravitational potential 
    depth ($\sigma$) of the host galaxy?
    
    Methodology:
    1.  **Calculate Individual H0**: For each SN Ia host, we calculate $H_0 = v / d$, 
        where $v = cz$ (Pantheon+ CMB frame redshift) and $d$ is the distance derived 
        from the SH0ES Cepheid distance modulus ($\mu$).
    2.  **Stratification**: We split the sample into two bins based on the median 
        velocity dispersion ($\sigma_{\rm med}$).
        - **Low Density ($\sigma < \sigma_{\rm med}$)**: Shallow potentials, similar to calibrators.
        - **High Density ($\sigma > \sigma_{\rm med}$)**: Deep potentials, predicted to show TEP bias.
    3.  **Bias Quantification**: We calculate the mean $H_0$ in each bin and the 
        Pearson correlation coefficient $r$ between $\sigma$ and $H_0$.
    
    A significant difference between the bins ($\Delta H_0 > 0$) or a strong positive correlation 
    confirms the presence of an environmental bias consistent with TEP Period Contraction.
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
        self.logger = TEPLogger("step_2_stratification", log_file_path=self.logs_dir / "step_2_stratification.log")
        set_step_logger(self.logger)
        
        # Inputs
        self.hosts_path = self.data_dir / "processed" / "hosts_processed.csv"
        self.distances_path = self.data_dir / "interim" / "r22_distances.csv"

        self.mu_cov_path = self.data_dir / "interim" / "r22_mu_covariance.npy"
        self.mu_cov_labels_path = self.data_dir / "interim" / "r22_mu_covariance_labels.json"

        self.env_catalog_cache_path = self.data_dir / "raw" / "external" / "tully2015_2mrs_groups_table5.csv"
        
        # Outputs
        self.stratified_output_path = self.outputs_dir / "stratified_h0.csv"
        self.json_output_path = self.outputs_dir / "stratification_results.json"
        self.plot_path = self.figures_dir / "h0_vs_sigma.png"

        self.h0_cov_path = self.outputs_dir / "h0_covariance.npy"
        self.h0_cov_labels_path = self.outputs_dir / "h0_covariance_labels.json"

    def load_and_merge(self):
        """Loads data and merges host properties with distances."""
        print_status("Loading and Merging Datasets...", "SECTION")
        
        if not self.hosts_path.exists() or not self.distances_path.exists():
            print_status("Input files missing. Please run Step 1 first.", "ERROR")
            sys.exit(1)
            
        hosts_df = pd.read_csv(self.hosts_path)
        dists_df = pd.read_csv(self.distances_path)
        
        # Merge on source_id
        merged = pd.merge(dists_df, hosts_df, on='source_id', how='inner')
        print_status(f"Merged {len(merged)} galaxies (Distances + Properties).", "INFO")
        return merged

    def calculate_h0(self, df):
        """Calculates H0 for each host."""
        print_status("Calculating Individual H0 Values...", "SECTION")
        
        c = 299792.458 # km/s
        
        # v = c * z_hd
        df['velocity'] = c * df['z_hd']
        
        # d = 10^((mu - 25)/5)
        # 'value' column in r22_distances is mu
        df['distance_mpc'] = 10 ** ((df['value'] - 25) / 5)
        
        # H0 = v / d
        df['h0_derived'] = df['velocity'] / df['distance_mpc']
        
        # Filter valid entries (require Sigma, H0, and minimum redshift)
        valid = df.dropna(subset=['h0_derived', 'sigma_inferred']).copy()
        
        # Ensure normalized_name is stripped and string
        valid['normalized_name'] = valid['normalized_name'].astype(str).str.strip()
        
        # Exclude Anchors/Calibrators from H0 sample (they are not SN hosts in this context)
        # N4258, LMC, SMC, M31, MW
        anchors = ['NGC 4258', 'LMC', 'SMC', 'M 31', 'MW']
        
        valid = valid[~valid['normalized_name'].isin(anchors)].copy()
        
        n_before = len(valid)
        
        # MINIMUM REDSHIFT CUT: z > 0.0035
        # Rationale: At low redshift, peculiar velocities (v_pec ~ 300 km/s) dominate
        # the recession velocity. For z = 0.0035, v_rec = cz ≈ 1050 km/s, so 
        # v_pec/v_rec ≈ 29%. This is a reasonable compromise between sample size
        # and peculiar velocity contamination.
        # We use z > 0.0035 (rather than z > 0.01 as in some studies) to maximize
        # sample size while keeping peculiar velocity errors below ~30%.
        # Reference: Scolnic et al. (2022), Pantheon+ methodology
        MIN_REDSHIFT = 0.0035
        valid = valid[valid['z_hd'] >= MIN_REDSHIFT].copy()
        n_excluded = n_before - len(valid)
        
        if n_excluded > 0:
            print_status(f"Excluded {n_excluded} low-z hosts (z < {MIN_REDSHIFT}) to minimize peculiar velocity errors.", "WARNING")
        
        print_status(f"Final Sample Size: {len(valid)} SN Ia Hosts", "SUCCESS")
        
        # Display Sample
        headers = ["Host", "z_HD", "mu (mag)", "D (Mpc)", "H0 (km/s/Mpc)"]
        rows = []
        for _, row in valid.head(5).iterrows():
            rows.append([
                row['normalized_name'],
                f"{row['z_hd']:.4f}",
                f"{row['value']:.3f}",
                f"{row['distance_mpc']:.1f}",
                f"{row['h0_derived']:.2f}"
            ])
        print_table(headers, rows, title="Sample H0 Calculations")
        
        return valid

    def _load_tully_2015_table5(self):
        if Vizier is None:
            return None

        self.env_catalog_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.env_catalog_cache_path.exists():
            try:
                return pd.read_csv(self.env_catalog_cache_path)
            except Exception:
                return None

        v = Vizier(
            columns=[
                'PGC',
                'Nest',
                'Nmb',
                'logpK',
                'Mvir',
                'Mlum',
            ],
            row_limit=-1,
        )

        tables = v.get_catalogs('J/AJ/149/171/table5')
        if not tables:
            return None

        t = tables[0].to_pandas()
        t.to_csv(self.env_catalog_cache_path, index=False)
        return t

    def annotate_large_scale_environment(self, df):
        print_status("Annotating Large-Scale Environment (Tully 2015 groups)...", "SECTION")

        if 'pgc' not in df.columns:
            df['tully_nest'] = np.nan
            df['tully_nmb'] = np.nan
            df['tully_logpK'] = np.nan
            df['tully_mvir_tmsun'] = np.nan
            df['tully_mlum_tmsun'] = np.nan
            df['tully_is_group'] = False
            df['tully_is_cluster'] = False
            print_status("No PGC identifiers available in merged data; skipping group crossmatch.", "WARNING")
            return df

        t = self._load_tully_2015_table5()
        if t is None or len(t) == 0:
            df['tully_nest'] = np.nan
            df['tully_nmb'] = np.nan
            df['tully_logpK'] = np.nan
            df['tully_mvir_tmsun'] = np.nan
            df['tully_mlum_tmsun'] = np.nan
            df['tully_is_group'] = False
            df['tully_is_cluster'] = False
            print_status("Could not load Tully 2015 group catalog; skipping group crossmatch.", "WARNING")
            return df

        cols = ['PGC', 'Nest', 'Nmb', 'logpK', 'Mvir', 'Mlum']
        missing = [c for c in cols if c not in t.columns]
        if missing:
            df['tully_nest'] = np.nan
            df['tully_nmb'] = np.nan
            df['tully_logpK'] = np.nan
            df['tully_mvir_tmsun'] = np.nan
            df['tully_mlum_tmsun'] = np.nan
            df['tully_is_group'] = False
            df['tully_is_cluster'] = False
            print_status(f"Tully 2015 table missing columns: {missing}. Skipping.", "WARNING")
            return df

        t = t[cols].copy()
        t = t.dropna(subset=['PGC']).copy()
        try:
            t['PGC'] = t['PGC'].astype(int)
        except Exception:
            t['PGC'] = pd.to_numeric(t['PGC'], errors='coerce').astype('Int64')

        df = df.copy()
        df['pgc_int'] = pd.to_numeric(df['pgc'], errors='coerce').astype('Int64')

        merged = df.merge(t, left_on='pgc_int', right_on='PGC', how='left')

        merged['tully_nest'] = merged['Nest']
        merged['tully_nmb'] = merged['Nmb']
        merged['tully_logpK'] = merged['logpK']
        merged['tully_mvir_tmsun'] = merged['Mvir']
        merged['tully_mlum_tmsun'] = merged['Mlum']

        merged['tully_is_group'] = merged['tully_nmb'].fillna(1) >= 2
        merged['tully_is_cluster'] = merged['tully_nmb'].fillna(1) >= 5

        merged = merged.drop(columns=['PGC', 'Nest', 'Nmb', 'logpK', 'Mvir', 'Mlum'], errors='ignore')

        n_tagged = int(merged['tully_nest'].notna().sum())
        print_status(f"Matched {n_tagged}/{len(merged)} hosts to Tully 2015 group catalog.", "INFO")
        return merged

    def _load_mu_covariance(self):
        if not self.mu_cov_path.exists() or not self.mu_cov_labels_path.exists():
            return None, None

        mu_cov = np.load(self.mu_cov_path)
        with open(self.mu_cov_labels_path, 'r') as f:
            mu_labels = json.load(f)

        return mu_cov, mu_labels

    def _subset_covariance(self, cov, cov_labels, target_labels):
        label_to_idx = {str(lbl): i for i, lbl in enumerate(cov_labels)}
        idx = []
        missing = []
        for lbl in target_labels:
            key = str(lbl)
            if key not in label_to_idx:
                missing.append(key)
                continue
            idx.append(label_to_idx[key])

        if missing:
            raise KeyError(f"Missing {len(missing)} labels in covariance: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        sub = cov[np.ix_(idx, idx)]
        sub = 0.5 * (sub + sub.T)
        return sub

    def _mu_to_h0_covariance(self, mu_values, h0_values, mu_cov):
        dH_dmu = -(np.log(10.0) / 5.0) * h0_values
        J = np.diag(dH_dmu)
        h0_cov = J @ mu_cov @ J
        h0_cov = 0.5 * (h0_cov + h0_cov.T)
        return h0_cov

    def calculate_densities(self, df):
        r"""
        Estimates the local stellar mass density for SH0ES hosts to verify
        the "unscreened" regime assumption.
        
        Physics:
        SH0ES Cepheids typically reside in the disks of spiral galaxies at 
        radii r ~ 0.3 - 0.8 R25.
        We model the disk as an exponential profile:
        rho(r) ~ (M / 4 pi Rd^2 zd) * exp(-r/Rd)
        
        Assumptions:
        - Rd ~ R25 / 3.2
        - zd ~ 0.1 Rd (Thin disk scale height)
        - Typical Cepheid Radius r_cep ~ 0.55 R25 ~ 1.8 Rd
        """
        print_status("Estimating Host Environmental Densities...", "SECTION")
        
        # Filter hosts with Mass and Size info
        valid = df.dropna(subset=['host_logmass', 'r25_arcsec', 'distance_mpc']).copy()
        
        if len(valid) == 0:
            print_status("Insufficient data for density estimation (missing RC3 radii or Mass).", "WARNING")
            # If no density can be calculated, return safe defaults
            return np.nan, np.nan, np.nan, df
            
        # Calculate Physical Radius R25 in kpc
        # theta = R / D -> R = D * theta
        # r25_arcsec to radians: r25 / 206265
        valid['r25_kpc'] = valid['distance_mpc'] * 1000 * (valid['r25_arcsec'] / 206265.0)
        
        # Calculate Scale Length Rd (kpc)
        valid['rd_kpc'] = valid['r25_kpc'] / 3.2
        
        # Scale Height zd (kpc)
        valid['zd_kpc'] = 0.1 * valid['rd_kpc']
        
        # Mass in Solar Masses
        valid['mass_sol'] = 10**valid['host_logmass']
        
        # Evaluate Density at typical Cepheid radius (1.8 Rd)
        # rho = (M / (4 pi Rd^2 zd)) * exp(-1.8)
        # Note: M_disk is roughly M_total for these spirals
        
        def get_rho(row):
            if row['rd_kpc'] <= 0 or row['zd_kpc'] <= 0: return np.nan
            prefactor = row['mass_sol'] / (4 * np.pi * (row['rd_kpc']**2) * row['zd_kpc'])
            density = prefactor * np.exp(-1.8)
            # Convert kpc^-3 to pc^-3 (1 kpc^3 = 1e9 pc^3)
            return density / 1e9
            
        valid['rho_local'] = valid.apply(get_rho, axis=1)
        
        # Merge rho_local back to main df for export
        df['rho_local'] = np.nan
        # We need to map by index to update correctly
        df.update(valid['rho_local'])
        
        # Stats
        rho_mean = valid['rho_local'].mean()
        rho_min = valid['rho_local'].min()
        rho_max = valid['rho_local'].max()
        
        print_status(f"Calculated densities for {len(valid)} hosts.", "INFO")
        if not np.isnan(rho_mean):
            print_status(f"Mean Host Density: {rho_mean:.4f} M_sun/pc^3", "RESULT")
            print_status(f"Density Range: [{rho_min:.4f}, {rho_max:.4f}]", "INFO")
            
            # Effective galactic transition density from Paper 7 (SPARC normalization): rho_trans ~ 0.5 M_sun/pc^3
            rho_trans = 0.5

            # Identify hosts above the effective transition density
            transition_regime = valid[valid['rho_local'] > rho_trans]
            if len(transition_regime) > 0:
                print_status(
                    f"Found {len(transition_regime)} hosts above the effective transition density (rho > {rho_trans}):",
                    "WARNING",
                )
                for _, row in transition_regime.iterrows():
                    print_status(f"  - {row['normalized_name']}: {row['rho_local']:.3f} M_sun/pc^3", "WARNING")
            
            if rho_mean < rho_trans:
                print_status(
                    "CONCLUSION: Sample mean is below the effective galactic transition density (rho_trans).",
                    "SUCCESS",
                )
            else:
                print_status(
                    "CONCLUSION: Sample mean is near/above the effective galactic transition density (rho_trans).",
                    "WARNING",
                )
        
        return rho_mean, rho_min, rho_max, df

    def stratify_and_analyze(self, df):
        """Stratifies by Sigma and analyzes H0 bias."""
        print_status("Stratification Analysis (Low vs High Density)", "SECTION")

        df = df.reset_index(drop=True)
        
        median_sigma = df['sigma_inferred'].median()
        
        low_sigma = df[df['sigma_inferred'] <= median_sigma]
        high_sigma = df[df['sigma_inferred'] > median_sigma]
        
        mean_low = low_sigma['h0_derived'].mean()
        err_low = low_sigma['h0_derived'].std() / np.sqrt(len(low_sigma))
        
        mean_high = high_sigma['h0_derived'].mean()
        err_high = high_sigma['h0_derived'].std() / np.sqrt(len(high_sigma))
        
        diff = mean_high - mean_low
        
        # Calculate Correlation
        corr = df['sigma_inferred'].corr(df['h0_derived'])

        # Covariance-aware uncertainties (if available)
        cov_low_err = None
        cov_high_err = None
        cov_all_mean_err = None
        cov_diff_err = None
        cov_available = False
        try:
            mu_cov, mu_labels = self._load_mu_covariance()
            if mu_cov is not None and mu_labels is not None:
                sample_labels = df['source_id'].astype(str).tolist()
                mu_cov_sub = self._subset_covariance(mu_cov, mu_labels, sample_labels)
                h0_vals = df['h0_derived'].values
                mu_vals = df['value'].values
                h0_cov = self._mu_to_h0_covariance(mu_vals, h0_vals, mu_cov_sub)

                np.save(self.h0_cov_path, h0_cov)
                with open(self.h0_cov_labels_path, 'w') as f:
                    json.dump(sample_labels, f, indent=2)

                ones = np.ones(len(df))
                cov_all_mean_err = float(np.sqrt(ones @ h0_cov @ ones) / len(df))

                pos_low = low_sigma.index.to_numpy(dtype=int)
                pos_high = high_sigma.index.to_numpy(dtype=int)

                cov_low = h0_cov[np.ix_(pos_low, pos_low)]
                cov_high = h0_cov[np.ix_(pos_high, pos_high)]
                ones_low = np.ones(len(pos_low))
                ones_high = np.ones(len(pos_high))
                cov_low_err = float(np.sqrt(ones_low @ cov_low @ ones_low) / len(pos_low))
                cov_high_err = float(np.sqrt(ones_high @ cov_high @ ones_high) / len(pos_high))

                w = np.zeros(len(df))
                w[pos_high] = 1.0 / len(pos_high)
                w[pos_low] = -1.0 / len(pos_low)
                cov_diff_err = float(np.sqrt(w @ h0_cov @ w))
                cov_available = True
        except Exception as e:
            print_status(f"Could not compute covariance-aware uncertainties: {e}", "WARNING")
        
        # Results Table
        headers = ["Bin", "Sigma Range", "N", "Mean H0", "Std Err"]
        rows = [
            [
                "Low Density", 
                f"<= {median_sigma:.1f}", 
                str(len(low_sigma)), 
                f"{mean_low:.2f}", 
                f"{cov_low_err:.2f}" if cov_available and cov_low_err is not None else f"{err_low:.2f}"
            ],
            [
                "High Density", 
                f"> {median_sigma:.1f}", 
                str(len(high_sigma)), 
                f"{mean_high:.2f}", 
                f"{cov_high_err:.2f}" if cov_available and cov_high_err is not None else f"{err_high:.2f}"
            ],
            [
                "Difference",
                "-",
                "-",
                f"+{diff:.2f}",
                "-"
            ]
        ]
        print_table(headers, rows, title="Stratified H0 Results")
        
        print_status(f"Median Velocity Dispersion: {median_sigma:.2f} km/s", "INFO")
        print_status(f"Correlation (Sigma vs H0): r = {corr:.3f}", "TEST")
        
        if diff > 3.0:
            print_status(f"Significant Environmental Bias Detected (+{diff:.2f} km/s/Mpc)", "WARNING")
        
        metrics = {
            "median_sigma": float(median_sigma),
            "low_density": {
                "n": int(len(low_sigma)),
                "mean_h0": float(mean_low),
                "std_err": float(err_low),
                "cov_err": float(cov_low_err) if cov_low_err is not None else None
            },
            "high_density": {
                "n": int(len(high_sigma)),
                "mean_h0": float(mean_high),
                "std_err": float(err_high),
                "cov_err": float(cov_high_err) if cov_high_err is not None else None
            },
            "difference": float(diff),
            "correlation_r": float(corr),
            "h0_mean_cov_err": float(cov_all_mean_err) if cov_all_mean_err is not None else None,
            "difference_cov_err": float(cov_diff_err) if cov_diff_err is not None else None,
            "h0_covariance_saved": bool(cov_available)
        }
        
        return df, metrics

    def plot_results(self, df, corr):
        """Generates analysis plots."""
        print_status("Generating Diagnostic Plots...", "PROCESS")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            # Fallback if style file missing
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30'}
        
        plt.figure(figsize=(14, 9))
        
        # Data
        plt.scatter(df['sigma_inferred'], df['h0_derived'], 
                   alpha=0.8, 
                   color=colors['blue'], 
                   s=100,
                   edgecolor='white',
                   label='SN Ia Hosts',
                   zorder=3)
        
        # Regression line
        if len(df) > 1:
            z = np.polyfit(df['sigma_inferred'], df['h0_derived'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['sigma_inferred'].min(), df['sigma_inferred'].max(), 100)
            plt.plot(x_range, p(x_range), 
                    color=colors['accent'], 
                    linestyle='--', 
                    linewidth=2.5,
                    alpha=0.9, 
                    label=f'Trend (r={corr:.2f})',
                    zorder=4)

        plt.xlabel(r'Velocity Dispersion $\sigma$ (km/s)')
        plt.ylabel(r'Derived $H_0$ (km/s/Mpc)')
        plt.title('Hubble Constant vs Host Potential Depth')
        plt.legend(loc='upper left', frameon=True)
        # Grid handled by style
        plt.tight_layout()
        
        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved plot to {self.plot_path}", "SUCCESS")
        plt.close()
        
        # Copy to public figures
        public_path = self.root_dir / "site" / "public" / "figures" / "h0_vs_sigma.png"
        shutil.copy(self.plot_path, public_path)
        print_status(f"Copied plot to {public_path}", "SUCCESS")

    def run(self):
        print_status("Starting Step 2: Stratification", "TITLE")
        
        merged = self.load_and_merge()
        analyzed = self.calculate_h0(merged)
        
        # Add density calculation
        rho_mean, rho_min, rho_max, analyzed = self.calculate_densities(analyzed)

        analyzed = self.annotate_large_scale_environment(analyzed)
        
        final_df, metrics = self.stratify_and_analyze(analyzed)
        
        # Add density metrics
        metrics['shoes_density_mean'] = float(rho_mean) if not np.isnan(rho_mean) else None
        metrics['shoes_density_min'] = float(rho_min) if not np.isnan(rho_min) else None
        metrics['shoes_density_max'] = float(rho_max) if not np.isnan(rho_max) else None
        
        final_df.to_csv(self.stratified_output_path, index=False)
        print_status(f"Saved stratified data to {self.stratified_output_path}", "SUCCESS")
        
        # Save JSON
        with open(self.json_output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print_status(f"Saved analysis metrics to {self.json_output_path}", "SUCCESS")
        
        self.plot_results(final_df, metrics['correlation_r'])
        
        print_status("Step 2 Complete.", "SUCCESS")

def main():
    step = Step2Stratification()
    step.run()

if __name__ == "__main__":
    main()
