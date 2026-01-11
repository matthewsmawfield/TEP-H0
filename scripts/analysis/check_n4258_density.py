
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.logger import print_status

def check_n4258_density():
    print_status("Checking NGC 4258 (M106) Density Environment", "SECTION")
    
    # 1. Load Cepheids
    data_path = Path("data/interim/reconstructed_shoes_cepheids.csv")
    if not data_path.exists():
        print_status("Data file not found", "ERROR")
        return

    df = pd.read_csv(data_path)
    n4258 = df[df['Source'] == 'N4258'].copy()
    print_status(f"Loaded {len(n4258)} Cepheids for NGC 4258", "INFO")
    
    # 2. NGC 4258 Structural Parameters
    # Distance: 7.58 Mpc (Reid et al. 2019) -> mu = 29.397
    dist_mpc = 7.58
    # Inclination: 72 deg (HyperLEDA/Literature)
    inc_deg = 72.0
    pa_deg = 150.0 # Approx PA
    
    # Mass and Scale Length
    # M_star approx 1.7e10 to 1e11 depending on source. 
    # SH0ES hosts file had logM ~ 9.3? That seems low. 
    # Kormendy & Ho (2013) give sigma=115.
    # Leda gives vmax ~ 208 km/s. 
    # Let's use a standard spiral model consistent with v_rot ~ 210 km/s.
    # V^2 = GM/R. 
    # Let's use the density formula from Step 2: rho(r) ~ (M / 4pi Rd^2 zd) * exp(-r/Rd)
    # We need Rd. R25 is roughly 18 arcmin?
    # RC3 D25 = 18.6 arcmin. R25 = 9.3 arcmin.
    r25_arcmin = 9.3
    r25_kpc = dist_mpc * 1000 * np.radians(r25_arcmin / 60.0)
    print_status(f"R25 (kpc): {r25_kpc:.2f}", "INFO")
    
    # Scale length Rd approx R25 / 3.2
    rd_kpc = r25_kpc / 3.2
    zd_kpc = 0.1 * rd_kpc
    print_status(f"Scale Length Rd (kpc): {rd_kpc:.2f}", "INFO")
    
    # Estimate Mass from Vmax if Mass is uncertain
    # Vmax = 208 km/s (HyperLeda)
    # M_total(<R) ~ V^2 R / G
    # Let's use the mass implied by the Step 2 calculation for consistency if possible,
    # but the hosts file value (9.28) is suspicious. 
    # Let's check what a typical mass for N4258 is.
    # Castro et al 2016: Stellar mass logM = 10.75. 
    # This is much higher than 9.28. 9.28 might be the bulge mass? Or just wrong data.
    # We will compute density for both Mass scenarios.
    
    mass_high = 10**10.75
    mass_low = 10**9.28 
    
    # 3. Calculate Deprojected Radii for Cepheids
    # We don't have RA/DEC in the reconstructed file for N4258 (it just has Period, W, Source).
    # Wait, the reconstructed file might not have coordinates.
    # Let's check the columns.
    
    if 'RA' not in n4258.columns:
        print_status("No RA/Dec in reconstructed file. Using statistical distribution assumption.", "WARNING")
        # Most Cepheids are found at 0.3 - 0.8 R25.
        # Let's assume r = 0.5 * R25 as a representative location.
        r_rep_kpc = 0.5 * r25_kpc
        print_status(f"Assuming representative radius r = 0.5 R25 = {r_rep_kpc:.2f} kpc", "INFO")
    else:
        # If we had coords we'd deproject. 
        pass

    def calc_density(mass, r_kpc):
        prefactor = mass / (4 * np.pi * (rd_kpc**2) * zd_kpc)
        rho_kpc = prefactor * np.exp(-r_kpc / rd_kpc)
        rho_pc = rho_kpc / 1e9
        return rho_pc

    rho_high = calc_density(mass_high, r_rep_kpc)
    rho_low = calc_density(mass_low, r_rep_kpc)
    
    print_status("-" * 40, "INFO")
    print_status(f"Density at r={r_rep_kpc:.1f} kpc (0.5 R25):", "RESULT")
    print_status(f"  Scenario A (LogM=10.75, Standard): {rho_high:.4f} M_sun/pc^3", "RESULT")
    print_status(f"  Scenario B (LogM=9.28,  File val): {rho_low:.4f} M_sun/pc^3", "RESULT")
    
    # Transition threshold
    rho_trans = 0.5
    print_status("-" * 40, "INFO")
    print_status(f"Screening Threshold: {rho_trans} M_sun/pc^3", "INFO")
    
    if rho_high > rho_trans:
        print_status("CONCLUSION (Std Mass): NGC 4258 is SCREENED.", "SUCCESS")
    else:
        print_status("CONCLUSION (Std Mass): NGC 4258 is UNSCREENED.", "WARNING")

if __name__ == "__main__":
    check_n4258_density()
