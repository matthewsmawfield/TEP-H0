
import numpy as np
import pandas as pd
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
rc3_path = data_dir / "raw" / "external" / "rc3_d25_hosts.csv"


def read_commented_csv(path):
    """Read a committed catalog snapshot while ignoring provenance comments."""
    return pd.read_csv(path, comment="#")

def fetch_galaxy_metadata():
    """
    Merges supplementary metadata from an exact-PGC subset of the Third
    Reference Catalogue of Bright Galaxies (RC3).
    
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
    
    if not rc3_path.exists():
        raise FileNotFoundError(f"Pinned RC3 subset missing: {rc3_path}")

    print_status("Merging pinned RC3 VII/155 values by exact PGC identifier...", "PROCESS")
    rc3 = read_commented_csv(rc3_path)
    if rc3["pgc"].duplicated().any():
        raise ValueError("Pinned RC3 subset contains duplicate PGC identifiers")

    df["pgc_int"] = pd.to_numeric(df["pgc"], errors="coerce").astype("Int64")
    rc3["pgc"] = pd.to_numeric(rc3["pgc"], errors="raise").astype("Int64")
    df = df.merge(
        rc3[["pgc", "log_d25", "e_log_d25", "d25_uncertain"]],
        left_on="pgc_int",
        right_on="pgc",
        how="left",
        validate="many_to_one",
        suffixes=("", "_rc3"),
    )
    df["log_d25"] = pd.to_numeric(df["log_d25"], errors="coerce")
    df["r25_arcsec"] = np.power(10.0, df["log_d25"]) * 3.0
    found_count = int(df["log_d25"].notna().sum())
            
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
    
    if found_count != df["pgc_int"].notna().sum():
        missing = df.loc[df["pgc_int"].notna() & df["log_d25"].isna(), "normalized_name"].tolist()
        raise ValueError(f"Pinned RC3 subset is incomplete for PGC-tagged rows: {missing}")

    df = df.drop(columns=["pgc_int", "pgc_rc3"], errors="ignore")
    df.to_csv(output_path, index=False)
    print_status(f"Saved enriched metadata to {output_path}", "SUCCESS")

if __name__ == "__main__":
    # Create a local logger if running directly
    logger = TEPLogger("fetch_metadata")
    set_step_logger(logger)
    fetch_galaxy_metadata()
