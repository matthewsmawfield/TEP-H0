
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import linalg
import warnings
from urllib.request import urlretrieve

# Astronomy imports
try:
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ImportError:
    print("Error: This pipeline requires astroquery and astropy.")
    sys.exit(1)

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    # Add project root to path if needed
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

class Step1DataIngestion:
    r"""
    Step 1: Data Ingestion and Preparation
    ======================================
    
    This foundational step ingests and prepares the raw data required for the TEP-H0 analysis.
    It ensures that all subsequent steps operate on a clean, reconstructed, and scientifically 
    rigorous dataset.

    Key Tasks:
    1.  **Reconstruct SH0ES Cepheid Catalog**: The raw SH0ES data is provided as design matrices (R22).
        We reconstruct the actual Cepheid catalog (Periods, Magnitudes, Metallicity) from these matrices.
    2.  **Calculate Distances**: We solve the Generalized Least Squares (GLS) system to recover the 
        distance moduli (mu) for all hosts directly from the R22 solution.
    3.  **Resolve Coordinates**: We use the Simbad astronomical database to resolve the precise RA/Dec
        coordinates for all SH0ES host galaxies.
    4.  **Ingest Pantheon+ Data**: We download the Pantheon+ Supernova catalog to obtain independent 
        host properties (Mass, Redshift).
    5.  **Cross-Match & Enrich**: We spatially cross-match SH0ES hosts with Pantheon+ objects and 
        merge TEP-independent velocity dispersion measurements (from HyperLEDA/Literature).

    Outputs:
        - data/interim/reconstructed_shoes_cepheids.csv
        - data/interim/r22_distances.csv
        - data/interim/hosts_coords.csv
        - data/processed/hosts_processed.csv
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.logs_dir = self.root_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure data directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.external_dir = self.raw_dir / "external" / "Cepheid-Distance-Ladder-Data" / "SH0ES2022"
        
        # Initialize Logger
        self.logger = TEPLogger("step_1_ingestion", log_file_path=self.logs_dir / "step_1_ingestion.log")
        set_step_logger(self.logger)
        
        # Output files (Interim & Processed)
        self.cepheid_catalog_path = self.interim_dir / "reconstructed_shoes_cepheids.csv"
        self.distances_path = self.interim_dir / "r22_distances.csv"
        self.coords_path = self.interim_dir / "hosts_coords.csv"
        self.hosts_properties_path = self.interim_dir / "hosts_properties.csv"
        self.hosts_processed_path = self.processed_dir / "hosts_processed.csv"
        
        # Raw Inputs
        self.pantheon_path = self.raw_dir / "Pantheon+SH0ES.dat"

    def normalize_name(self, name):
        """Normalizes SH0ES naming convention to standard SIMBAD/NED names."""
        name = str(name).strip()
        
        # Specific SH0ES naming quirks (Check these FIRST)
        if name == 'M1337':
            return 'NGC 1337'
        if name == 'N105A':
            return 'NGC 105'
        if name == 'N976A':
            return 'NGC 976'
        if name == 'M31':
            return 'M 31'
        if name in ['LMC_GRND', 'LMC_HST']:
            return 'LMC'
        if name == 'SMC':
            return 'SMC'

        # Generic Rules
        if name.startswith('N') and len(name) > 1 and name[1].isdigit():
            return f"NGC {name[1:]}"
        if name.startswith('M') and len(name) > 1 and name[1].isdigit():
            return f"M {name[1:]}"
        if name.startswith('U') and len(name) > 1 and name[1].isdigit():
            return f"UGC {name[1:]}"
            
        return name

    def reconstruct_catalog(self):
        """Reconstructs the Cepheid catalog from R22 matrices (y, L, q)."""
        print_status("Reconstructing SH0ES Cepheid Catalog...", "SECTION")
        
        # Check if external data exists, if not clone it
        if not self.external_dir.exists():
            print_status(f"SH0ES 2022 data not found at {self.external_dir}", "WARNING")
            print_status("Cloning repository from GitHub...", "PROCESS")
            repo_url = "https://github.com/marcushogas/Cepheid-Distance-Ladder-Data.git"
            target_dir = self.raw_dir / "external" / "Cepheid-Distance-Ladder-Data"
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                import subprocess
                subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
                print_status("Repository cloned successfully.", "SUCCESS")
            except Exception as e:
                print_status(f"Failed to clone repository: {e}", "ERROR")
                sys.exit(1)
        
        q_path = self.external_dir / "q_R22.txt"
        y_path = self.external_dir / "y_R22.txt"
        L_path = self.external_dir / "L_R22.txt"
        
        if not q_path.exists():
            print_status(f"Missing external data at {self.external_dir}. Please clone the repository.", "ERROR")
            sys.exit(1)

        # Load Parameters
        q = np.loadtxt(q_path, dtype=str)
        q_list = list(q)
        
        try:
            idx_bW = q_list.index('bW')
            idx_ZW = q_list.index('ZW')
        except ValueError:
            print_status("Could not find bW or ZW in q_R22.txt", "ERROR")
            return

        # Load Data Vector
        names = ('Source', 'Data')
        fmt = ('S20', np.float32)
        y_data = np.loadtxt(y_path, unpack=True, skiprows=1, dtype={'names': names, 'formats': fmt})
        
        # Load Design Matrix
        L = np.loadtxt(L_path) # Assumes whitespace or tab delimited
        
        # Create DataFrame
        sources = [s.decode('utf-8') for s in y_data[0]] # Decode bytes
        y_df = pd.DataFrame({'Source': sources, 'Data': y_data[1]})
        
        # Extract Period and Metallicity terms
        y_df['L_col_bW'] = L[:, idx_bW]
        y_df['L_col_ZW'] = L[:, idx_ZW]
        
        # Filter for Cepheids (non-zero period term)
        cepheids = y_df[y_df['L_col_bW'] != 0].copy()
        
        # Save
        cepheids.to_csv(self.cepheid_catalog_path, index=False)
        
        # Verification Table
        print_status("Catalog Reconstruction Verification:", "INFO")
        headers = ["Source", "Data (mag)", "Period Term", "Metal Term"]
        rows = []
        for _, row in cepheids.head(3).iterrows():
            rows.append([row['Source'], f"{row['Data']:.3f}", f"{row['L_col_bW']:.3f}", f"{row['L_col_ZW']:.3f}"])
        print_table(headers, rows, title="SH0ES Cepheid Sample (First 3)")
        
        print_status(f"Saved reconstructed catalog with {len(cepheids)} Cepheids to {self.cepheid_catalog_path}", "SUCCESS")
        return cepheids

    def calculate_distances(self):
        """Calculates distance moduli (mu) using GLS on R22 matrices."""
        print_status("Calculating R22 Distances (GLS Solution)...", "SECTION")
        
        y_path = self.external_dir / "y_R22.txt"
        C_path = self.external_dir / "C_R22.txt"
        L_path = self.external_dir / "L_R22.txt"
        q_path = self.external_dir / "q_R22.txt"
        
        # Load Data
        names = ('Source', 'Data')
        fmt = ('S20', np.float32)
        y_data = np.loadtxt(y_path, unpack=True, skiprows=1, dtype={'names': names, 'formats': fmt})
        y = y_data[1]
        
        C = np.loadtxt(C_path, delimiter='\t') # C_R22 is tab delimited usually
        L = np.loadtxt(L_path, delimiter='\t')
        q = np.loadtxt(q_path, unpack=True, dtype=str)
        
        print_status(f"Solving GLS system: {L.shape[0]} observations, {L.shape[1]} parameters", "INFO")
        
        # Solve GLS: q = (L^T C^-1 L)^-1 L^T C^-1 y
        # Invert C
        try:
            Cinv = linalg.inv(C)
        except linalg.LinAlgError:
            print_status("Covariance matrix inversion failed.", "ERROR")
            return

        fisher = L.T @ Cinv @ L
        fisher_inv = linalg.inv(fisher)
        rhs = L.T @ Cinv @ y
        qSol = fisher_inv @ rhs
        
        # Errors
        err_qSol = np.sqrt(np.diag(fisher_inv))
        
        results = [{'parameter': p, 'value': v, 'error': e} for p, v, e in zip(q, qSol, err_qSol)]
        df = pd.DataFrame(results)
        
        # Filter for mu_*
        mu_df = df[df['parameter'].str.startswith('mu_')].copy()
        mu_df['source_id'] = mu_df['parameter'].str.replace('mu_', '')

        mu_idx = [i for i, p in enumerate(q) if str(p).startswith('mu_')]
        if len(mu_idx) != len(mu_df):
            print_status(
                f"Mismatch in recovered mu parameters: index list has {len(mu_idx)} but dataframe has {len(mu_df)}. Will attempt to continue.",
                "WARNING",
            )

        mu_cov = fisher_inv[np.ix_(mu_idx, mu_idx)]
        mu_labels = [str(q[i]).replace('mu_', '') for i in mu_idx]

        cov_path = self.interim_dir / "r22_mu_covariance.npy"
        labels_path = self.interim_dir / "r22_mu_covariance_labels.json"
        np.save(cov_path, mu_cov)
        with open(labels_path, 'w') as f:
            json.dump(mu_labels, f, indent=2)
        print_status(f"Saved mu covariance matrix to {cov_path}", "SUCCESS")
        print_status(f"Saved mu covariance labels to {labels_path}", "SUCCESS")
        
        mu_df.to_csv(self.distances_path, index=False)
        
        # Show sample distances
        headers = ["Host ID", "Distance Modulus (mu)", "Error"]
        rows = []
        for _, row in mu_df.head(3).iterrows():
            rows.append([row['source_id'], f"{row['value']:.4f}", f"{row['error']:.4f}"])
        print_table(headers, rows, title="Recovered Distance Moduli")
        
        print_status(f"Saved {len(mu_df)} distances to {self.distances_path}", "SUCCESS")
        return mu_df

    def fetch_host_coordinates(self, source_list):
        """Fetches coordinates for host galaxies using Simbad."""
        print_status("Resolving Host Coordinates (Simbad)...", "SECTION")
        
        # Reset Simbad to defaults
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields('ra(d)', 'dec(d)', 'ids')
        
        results = []
        unique_sources = sorted(list(set(source_list)))
        
        print_status(f"Querying Simbad for {len(unique_sources)} unique hosts...", "PROCESS")
        
        for source in unique_sources:
            normalized = self.normalize_name(source)
            ra, dec = np.nan, np.nan
            pgc = np.nan
            
            try:
                # Simbad query
                result = Simbad.query_object(normalized)
                if result is None or len(result) == 0:
                    print_status(f"Simbad lookup failed for {source} -> {normalized}", "WARNING")
                else:
                    # VOTable fields: ra(d), dec(d), ids
                    # Handle column name variations (Simbad deprecated RA_d/DEC_d in favor of ra/dec)
                    ra_col = next((c for c in result.colnames if c.lower() in ['ra_d', 'ra']), None)
                    dec_col = next((c for c in result.colnames if c.lower() in ['dec_d', 'dec']), None)
                    
                    ra = float(result[ra_col][0]) if ra_col else np.nan
                    dec = float(result[dec_col][0]) if dec_col else np.nan

                    # Attempt to extract a PGC identifier if present in IDS
                    ids_str = ""
                    # Handle case-insensitive column names (Simbad varies)
                    ids_col = next((c for c in result.colnames if c.lower() == 'ids'), None)
                    
                    if ids_col:
                        try:
                            ids_str = str(result[ids_col][0])
                        except Exception:
                            ids_str = ""
                    
                    if ids_str:
                        # IDs string format is typically pipe-separated
                        for token in ids_str.split('|'):
                            tok = token.strip()
                            # HyperLEDA ID is PGC ID. Simbad often lists as 'LEDA' or 'PGC'
                            if tok.upper().startswith('PGC') or tok.upper().startswith('LEDA'):
                                parts = tok.split()
                                if len(parts) >= 2 and parts[1].isdigit():
                                    pgc = int(parts[1])
                                    break

            except Exception as e:
                print_status(f"Error querying Simbad for {source} -> {normalized}: {e}", "WARNING")

            results.append({
                'source_id': source,
                'normalized_name': normalized,
                'ra': ra,
                'dec': dec,
                'pgc': pgc,
            })

        coords_df = pd.DataFrame(results)

        # Verification sample
        headers = ["Source", "Normalized", "RA (deg)", "Dec (deg)"]
        rows = []
        for _, row in coords_df.head(5).iterrows():
            rows.append([
                row.get('source_id', ''),
                row.get('normalized_name', ''),
                f"{row['ra']:.3f}" if pd.notna(row.get('ra', np.nan)) else "-",
                f"{row['dec']:.3f}" if pd.notna(row.get('dec', np.nan)) else "-",
            ])
        print_table(headers, rows, title="Resolved Host Coordinates (Sample)")

        coords_df.to_csv(self.coords_path, index=False)
        print_status(f"Saved coordinates for {len(coords_df)} hosts to {self.coords_path}", "SUCCESS")
        return coords_df

    def download_pantheon(self):
        """Downloads the Pantheon+ dataset if missing."""
        print_status("Checking for Pantheon+ Supernova Catalog...", "SECTION")
        if not self.pantheon_path.exists():
            print_status("Downloading Pantheon+ dataset...", "PROCESS")
            url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
            try:
                urlretrieve(url, self.pantheon_path)
                print_status(f"Downloaded Pantheon+SH0ES.dat", "SUCCESS")
            except Exception as e:
                print_status(f"Failed to download Pantheon data: {e}", "ERROR")
                sys.exit(1)
        else:
            print_status("Pantheon+ dataset already exists locally.", "INFO")

    def cross_match_and_process(self):
        """Cross-matches hosts with Pantheon+ and processes physical properties."""
        print_status("Cross-Matching & Property Enrichment...", "SECTION")
        
        # Load Inputs
        hosts_df = pd.read_csv(self.coords_path)
        valid_hosts = hosts_df.dropna(subset=['ra', 'dec']).copy()
        
        # Load Pantheon
        # Use regex separator for robustness
        pan_df = pd.read_csv(self.pantheon_path, sep=r'\s+') 
        
        # Parse Pantheon Coords (Use SN coords RA/DEC, not HOST_RA/DEC which are often missing)
        pan_df['RA'] = pd.to_numeric(pan_df['RA'], errors='coerce')
        pan_df['DEC'] = pd.to_numeric(pan_df['DEC'], errors='coerce')
        valid_pan = pan_df.dropna(subset=['RA', 'DEC']).copy()
        
        # Match
        c_hosts = SkyCoord(ra=valid_hosts['ra'].values*u.deg, dec=valid_hosts['dec'].values*u.deg)
        c_pan = SkyCoord(ra=valid_pan['RA'].values*u.deg, dec=valid_pan['DEC'].values*u.deg)
        
        idx, d2d, _ = c_hosts.match_to_catalog_sky(c_pan)
        
        # Match constraints (increased to 15 arcmin for extended hosts like Antennae/NGC 4038)
        max_sep = 15.0 * u.arcmin
        constraint = d2d < max_sep
        
        matched_indices = idx[constraint]
        hosts_indices = np.where(constraint)[0]
        
        print_status(f"Matched {len(hosts_indices)} SH0ES hosts to Pantheon+ SN catalog.", "INFO")
        
        # Extract Data
        valid_hosts['host_logmass'] = np.nan
        valid_hosts['z_hd'] = np.nan
        valid_hosts['z_hd_err'] = np.nan
        valid_hosts['z_cmb'] = np.nan
        valid_hosts['z_cmb_err'] = np.nan
        valid_hosts['z_hel'] = np.nan
        valid_hosts['z_hel_err'] = np.nan
        valid_hosts['vpec'] = np.nan
        valid_hosts['vpecerr'] = np.nan
        valid_hosts['pantheon_id'] = ""
        valid_hosts['separation_arcsec'] = np.nan
        
        pan_matches = valid_pan.iloc[matched_indices]
        
        for i, host_idx in enumerate(hosts_indices):
            pan_idx = matched_indices[i]
            
            # Map back to dataframe index
            host_df_idx = valid_hosts.index[host_idx]
            
            row = valid_pan.iloc[pan_idx]
            valid_hosts.at[host_df_idx, 'host_logmass'] = row['HOST_LOGMASS']
            valid_hosts.at[host_df_idx, 'z_hd'] = row['zHD']
            valid_hosts.at[host_df_idx, 'z_hd_err'] = row.get('zHDERR', np.nan)
            valid_hosts.at[host_df_idx, 'z_cmb'] = row.get('zCMB', np.nan)
            valid_hosts.at[host_df_idx, 'z_cmb_err'] = row.get('zCMBERR', np.nan)
            valid_hosts.at[host_df_idx, 'z_hel'] = row.get('zHEL', np.nan)
            valid_hosts.at[host_df_idx, 'z_hel_err'] = row.get('zHELERR', np.nan)
            valid_hosts.at[host_df_idx, 'vpec'] = row.get('VPEC', np.nan)
            valid_hosts.at[host_df_idx, 'vpecerr'] = row.get('VPECERR', np.nan)
            valid_hosts.at[host_df_idx, 'pantheon_id'] = row['CID']
            valid_hosts.at[host_df_idx, 'separation_arcsec'] = d2d[host_idx].to(u.arcsec).value

        # Clean invalid masses (-9 or similar in Pantheon)
        valid_hosts.loc[valid_hosts['host_logmass'] <= 0, 'host_logmass'] = np.nan
        
        # Merge back
        final_df = hosts_df.merge(
            valid_hosts[[
                'source_id',
                'host_logmass',
                'z_hd',
                'z_hd_err',
                'z_cmb',
                'z_cmb_err',
                'z_hel',
                'z_hel_err',
                'vpec',
                'vpecerr',
                'pantheon_id',
                'separation_arcsec'
            ]],
            on='source_id',
            how='left'
        )
        
        # MEASURED Velocity Dispersions (TEP-Independent!)
        # These come from spectroscopic measurements of stellar absorption line widths.
        # Kinematics are TEP-independent: Doppler shift measures velocity, not time.
        #
        # DATA SOURCE: data/raw/external/velocity_dispersions_literature.csv
        # Full per-galaxy citations available in that file.
        # Primary sources: HyperLEDA (Makarov+2014), Ho+2009, Kormendy & Ho 2013, SDSS DR7
        # For late-type spirals without direct σ: HI linewidth proxy σ ≈ 0.7 × W50/2
        # Aperture-corrected to Re/8 using Jorgensen+1995 prescription.
        #
        # Load from external reference file for auditability
        sigma_csv_path = self.raw_dir / "external" / "velocity_dispersions_literature_regenerated.csv"
        if not sigma_csv_path.exists():
            sigma_csv_path = self.raw_dir / "external" / "velocity_dispersions_literature.csv"
        MEASURED_SIGMA = {}
        if sigma_csv_path.exists():
            sigma_df = pd.read_csv(sigma_csv_path, comment='#')
            for _, row in sigma_df.iterrows():
                MEASURED_SIGMA[row['galaxy']] = row['sigma_kms']
            print_status(f"Loaded {len(MEASURED_SIGMA)} velocity dispersions from literature CSV: {sigma_csv_path}", "INFO")
        else:
            # Fallback hardcoded values (should not be needed)
            print_status("Warning: Literature CSV not found, using fallback values.", "WARNING")
            MEASURED_SIGMA = {
                "NGC 0691": 89, "NGC 1015": 78, "NGC 1309": 85, "NGC 1365": 151,
                "NGC 1448": 95, "NGC 1559": 72, "NGC 2442": 110, "NGC 2525": 82,
                "NGC 2608": 92, "NGC 3021": 88, "NGC 3147": 238, "NGC 3254": 95,
                "NGC 3370": 85, "NGC 3447": 55, "NGC 3583": 108, "NGC 3972": 78,
                "NGC 3982": 82, "NGC 4424": 65, "NGC 4536": 95, "NGC 4639": 88,
                "NGC 4680": 92, "NGC 5468": 110, "NGC 5584": 98, "NGC 5643": 107,
                "NGC 5728": 175, "NGC 5861": 102, "NGC 5917": 88, "NGC 7250": 52,
                "NGC 7329": 165, "NGC 7541": 125, "NGC 7678": 138, "UGC 9391": 68,
                "NGC 0976": 110, "NGC 1337": 94, "NGC 4038": 107, "NGC 4039": 107,
                "NGC 4258": 115, "M 31": 160, "M 101": 28, "LMC": 24, "SMC": 22,
            }
        
        def get_measured_sigma(name):
            """Get measured velocity dispersion from literature."""
            # Normalize name for lookup
            name = str(name).strip()
            # Try direct lookup
            if name in MEASURED_SIGMA:
                return MEASURED_SIGMA[name]
            # Try without leading zeros in NGC numbers
            if name.startswith("NGC "):
                num = name[4:].lstrip("0")
                alt = f"NGC {num}"
                if alt in MEASURED_SIGMA:
                    return MEASURED_SIGMA[alt]
            return np.nan
        
        # Use MEASURED sigma (TEP-independent) instead of mass-inferred
        final_df['sigma_measured'] = final_df['normalized_name'].apply(get_measured_sigma)
        
        # Count how many have measured values
        n_measured = final_df['sigma_measured'].notna().sum()
        print_status(f"Found measured σ for {n_measured}/{len(final_df)} hosts (TEP-independent).", "INFO")
        
        # Log missing hosts for transparency
        missing_sigma = final_df[final_df['sigma_measured'].isna()]['normalized_name'].tolist()
        if missing_sigma:
            print_status(f"Missing Sigma for {len(missing_sigma)} hosts: {', '.join(missing_sigma)}", "WARNING")
        
        # Use measured sigma as primary; this is the scientifically correct approach
        final_df['sigma_inferred'] = final_df['sigma_measured']
        
        # Save
        final_df.to_csv(self.hosts_processed_path, index=False)
        
        # Verification Table
        headers = ["Host", "z_HD", "LogMass", "Sigma (km/s)"]
        rows = []
        # Filter for rows that actually have data for display
        display_df = final_df.dropna(subset=['sigma_inferred']).head(5)
        for _, row in display_df.iterrows():
            rows.append([
                row['normalized_name'], 
                f"{row['z_hd']:.5f}", 
                f"{row['host_logmass']:.2f}" if pd.notna(row['host_logmass']) else "-",
                f"{row['sigma_inferred']:.1f}"
            ])
        print_table(headers, rows, title="Processed Host Properties (Top 5)")
        
        # Log stats
        valid_sigma = final_df['sigma_inferred'].count()
        print_status(f"Processed {len(final_df)} hosts. Found Sigma for {valid_sigma} hosts.", "SUCCESS")
        print_status(f"Saved processed host data to {self.hosts_processed_path}", "SUCCESS")

    def run(self):
        print_status("Starting Step 1: Data Ingestion", "TITLE")
        
        # 1. Reconstruct Catalog
        cepheids = self.reconstruct_catalog()
        
        # 2. Calculate Distances
        self.calculate_distances()
        
        # 3. Fetch Coords
        if self.coords_path.exists():
            print_status("Using existing coordinates file.", "INFO")
            # We assume it's good, but one could force refresh
        else:
            source_list = cepheids['Source'].unique()
            self.fetch_host_coordinates(source_list)
            
        # 4. Download Pantheon
        self.download_pantheon()
        
        # 5. Cross-match and Process
        self.cross_match_and_process()
        
        print_status("Step 1 Complete.", "SUCCESS")

    def prepare_coordinates(self):
        """
        Executes only the coordinate generation phase (for Step 0 dependency).
        """
        print_status("Step 1 (Pre-flight): Preparing Host Coordinates...", "SECTION")
        
        # 1. Reconstruct Catalog (needed for source list)
        cepheids = self.reconstruct_catalog()
        
        # 2. Fetch Coords
        if self.coords_path.exists():
            print_status("Using existing coordinates file.", "INFO")
        else:
            source_list = cepheids['Source'].unique()
            self.fetch_host_coordinates(source_list)

def main():
    step = Step1DataIngestion()
    step.run()

if __name__ == "__main__":
    main()

