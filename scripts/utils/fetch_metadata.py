
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import sys

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    # Add project root to path if needed
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

# Paths
root_dir = Path(__file__).resolve().parents[2]
data_dir = root_dir / "data"
processed_dir = data_dir / "processed"
input_path = processed_dir / "hosts_processed.csv"
output_path = processed_dir / "hosts_metadata_enriched.csv"

def fetch_galaxy_metadata():
    """
    Fetches supplementary metadata for host galaxies from the Third Reference Catalog 
    of Bright Galaxies (RC3) via Vizier.
    
    Target Data:
    - D25: Isophotal diameter at surface brightness 25 mag/arcsec^2.
    - Used to estimate Effective Radius (R_eff) for aperture corrections.
    """
    print_status("Fetching Galaxy Metadata (RC3)", "SECTION")
    
    if not input_path.exists():
        print_status("Input file missing (hosts_processed.csv). Run Step 1 first.", "ERROR")
        return

    df = pd.read_csv(input_path)
    print_status(f"Loaded {len(df)} hosts for metadata enrichment.", "INFO")
    
    # Vizier Catalog: RC3 (VII/155) for D25
    # D25 in RC3 is typically log10(0.1 arcmin). 
    # LogD25 = log10(D25_0.1arcmin)
    # D25_arcmin = 10**LogD25 * 0.1
    # R25_arcsec = D25_arcmin * 60 / 2
    
    print_status("Querying Vizier (Catalog VII/155)...", "PROCESS")
    Vizier.ROW_LIMIT = 1
    
    df['log_d25'] = np.nan
    df['r25_arcsec'] = np.nan
    
    found_count = 0
    
    for i, row in df.iterrows():
        name = row['normalized_name']
        
        try:
            # Query by name
            cats = Vizier.query_object(name, catalog='VII/155')
            if cats and len(cats) > 0:
                cat = cats[0]
                if 'D25' in cat.columns:
                    val = cat['D25'][0]
                    if isinstance(val, (float, np.float32, np.float64)) and not np.isnan(val):
                        df.at[i, 'log_d25'] = val
                        # Convert to Radius in Arcsec
                        # val is log10(diameter in 0.1 arcmin)
                        # Diameter in 0.1 arcmin = 10^val
                        # Diameter in arcmin = 10^val * 0.1
                        # Radius in arcmin = 10^val * 0.05
                        # Radius in arcsec = 10^val * 0.05 * 60 = 10^val * 3
                        r25 = (10**val) * 3.0
                        df.at[i, 'r25_arcsec'] = r25
                        found_count += 1
                        # Debug log for first few
                        if i < 3:
                            print_status(f"  {name}: logD25={val:.2f} -> R25={r25:.1f}''", "DEBUG")
        except Exception as e:
            # print_status(f"Error querying {name}: {e}", "DEBUG")
            pass
            
    # Estimate Effective Radius (Re)
    # For disk galaxies, Re approx 0.5 * R25 is a common rough scaling 
    # (or R25 approx 3.2 Rd, Re approx 1.68 Rd -> Re/R25 ~ 0.5)
    # We will use R_eff = 0.5 * R25 as a working proxy for aperture correction normalization.
    df['r_eff_arcsec'] = df['r25_arcsec'] * 0.5
    
    # Results Table
    print_status(f"Metadata Retrieval Complete. Found RC3 data for {found_count}/{len(df)} hosts.", "SUCCESS")
    
    headers = ["Host", "log(D25)", "R25 ('')", "R_eff ('')"]
    rows = []
    # Show top 5 found
    found_df = df.dropna(subset=['r25_arcsec']).head(5)
    for _, row in found_df.iterrows():
        rows.append([
            row['normalized_name'],
            f"{row['log_d25']:.2f}",
            f"{row['r25_arcsec']:.1f}",
            f"{row['r_eff_arcsec']:.1f}"
        ])
    print_table(headers, rows, title="Sample Galaxy Metadata (RC3)")
    
    # Save
    df.to_csv(output_path, index=False)
    print_status(f"Saved enriched metadata to {output_path}", "SUCCESS")

if __name__ == "__main__":
    # Create a local logger if running directly
    logger = TEPLogger("fetch_metadata")
    set_step_logger(logger)
    fetch_galaxy_metadata()
