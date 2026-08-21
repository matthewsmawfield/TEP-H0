
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import TEP Logger & Utilities
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
    from scripts.utils.aperture_correction import jorgensen_aperture_correction
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
    from scripts.utils.aperture_correction import jorgensen_aperture_correction

class Step1bApertureCorrection:
    r"""
    Step 1b: Aperture Correction
    ============================
    
    The authoritative catalog now uses inclination-corrected global rotation
    speeds transformed to ``u_phi = V_rot / sqrt(2)``.  Global rotation speeds
    have no spectroscopic aperture radius, so the Jorgensen correction is not
    applicable.  This step preserves the historical output columns while
    explicitly recording a unit correction factor.
    
    The Physics:
    Velocity dispersion ($\sigma$) measurements depend on the fraction of the galaxy 
    covered by the spectroscopic slit or fiber. Measurements taken with smaller 
    apertures tend to probe the central, hotter regions, yielding higher $\sigma$.
    
    We apply the standard power-law correction from Jorgensen et al. (1995):
    
    $$ \log \sigma_{\rm norm} = \log \sigma_{\rm obs} + \beta \log \left( \frac{r_{\rm ap}}{r_{\rm norm}} \right) $$
    
    Where:
    - $\beta = 0.04$ (Empirical slope for early-type/bulge-dominated galaxies)
    - $r_{\rm ap}$: The radius of the observation aperture (assumed 1.5" for typical literature slits)
    - $r_{\rm norm} = R_{\rm eff} / 8$: The standard physical radius for normalization.
    
    This ensures that we are comparing "apples to apples" when stratifying hosts by potential depth.
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.processed_dir = self.root_dir / "data" / "processed"
        self.logs_dir = self.root_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.hosts_path = self.processed_dir / "hosts_processed.csv"
        self.metadata_path = self.processed_dir / "hosts_metadata_enriched.csv"
        
        # Initialize Logger
        self.logger = TEPLogger("step_1b_aperture", log_file_path=self.logs_dir / "step_02_aperture_correction.log")
        set_step_logger(self.logger)

    def apply_aperture_corrections(self):
        print_status("Initializing Aperture Correction Protocol", "SECTION")
        
        if not self.hosts_path.exists() or not self.metadata_path.exists():
            missing = [p for p in [self.hosts_path, self.metadata_path] if not p.exists()]
            raise FileNotFoundError(
                f"Aperture correction inputs missing: {missing}. "
                "Run Step 1 and fetch_metadata.py first."
            )

        # Load Data
        hosts_df = pd.read_csv(self.hosts_path)
        meta_df = pd.read_csv(self.metadata_path)
        # LMC has separate ground/HST rows in the ladder.  Metadata are
        # galaxy-level, so deduplicate before merging to avoid a Cartesian
        # expansion of those observations.
        meta_df = meta_df.drop_duplicates(subset=["normalized_name"], keep="first")
        
        print_status(f"Loaded {len(hosts_df)} hosts from pipeline processing.", "INFO")
        
        # Merge metadata if not already present
        # We use 'normalized_name' as key
        if 'r_eff_arcsec' not in hosts_df.columns:
            print_status("Merging enriched metadata (R_eff, R25)...", "PROCESS")
            # Select cols to merge
            cols_to_merge = ['normalized_name', 'r25_arcsec', 'r_eff_arcsec']
            hosts_df = hosts_df.merge(meta_df[cols_to_merge], on='normalized_name', how='left')
        
        # Correction Parameters
        BETA = 0.04
        ASSUMED_APERTURE_RADIUS = 1.5 # arcsec (Typical 3" slit/fiber)
        
        print_status(
            "HyperLEDA V_rot is a global kinematic measure; aperture correction disabled.",
            "INFO",
        )
        hosts_df['sigma_corrected'] = hosts_df['sigma_measured']
        
        # Calculate deltas for reporting
        hosts_df['sigma_delta'] = hosts_df['sigma_corrected'] - hosts_df['sigma_measured']
        
        # Update inferred sigma to use the corrected value
        hosts_df['sigma_inferred'] = hosts_df['sigma_corrected']
        
        # Verification Table
        headers = ["Host", "R_eff (\")", "Raw Sigma", "Corr Sigma", "Delta"]
        rows = []
        
        # Show top 5 corrected hosts
        corrected_sample = hosts_df.dropna(subset=['sigma_delta', 'r_eff_arcsec'])
        if len(corrected_sample) > 0:
            for _, row in corrected_sample.head(5).iterrows():
                rows.append([
                    row['normalized_name'],
                    f"{row['r_eff_arcsec']:.1f}",
                    f"{row['sigma_measured']:.1f}",
                    f"{row['sigma_corrected']:.1f}",
                    f"{row['sigma_delta']:+.2f}"
                ])
        print_table(headers, rows, title="Aperture Correction Samples")
        
        # Log changes
        n_corrected = 0
        mean_change = hosts_df['sigma_delta'].mean()
        
        print_status(f"Applied corrections to {n_corrected}/{len(hosts_df)} hosts.", "SUCCESS")
        print_status(f"Mean Kinematic Scale Change: {mean_change:+.2f} km/s", "INFO")
        
        # Save
        hosts_df.to_csv(self.hosts_path, index=False)
        print_status(f"Updated {self.hosts_path} with homogenized velocity dispersions.", "SUCCESS")
        
    def run(self):
        print_status("Starting Step 1b: Aperture Correction", "TITLE")
        self.apply_aperture_corrections()
        print_status("Step 1b Complete.", "SUCCESS")

if __name__ == "__main__":
    Step1bApertureCorrection().run()
