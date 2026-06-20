#!/usr/bin/env python3
"""
Crossmatch SH0ES hosts with SDSS MaNGA DR17 to find homogeneous IFS velocity dispersions.
Optimized for M4 Pro with parallel processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Use all available cores on M4 Pro
NUM_WORKERS = os.cpu_count() or 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_shoes_hosts():
    """Load SH0ES host coordinates."""
    hosts_path = PROJECT_ROOT / "data" / "interim" / "hosts_coords.csv"
    df = pd.read_csv(hosts_path)
    # Exclude anchors
    anchors = ['LMC', 'SMC', 'M 31', 'M 101', 'NGC 4258']
    df = df[~df['normalized_name'].isin(anchors)]
    df = df[~df['source_id'].str.contains('LMC|SMC|M31|M101|N4258', case=False, na=False)]
    return df

def query_manga_vizier(shoes_df, radius_arcmin=5.0):
    """
    Query MaNGA catalog via VizieR (more reliable than SDSS SkyServer).
    Uses astroquery for parallel cone searches.
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("Installing astroquery...")
        import subprocess
        subprocess.run(['pip', 'install', 'astroquery'], check=True)
        from astroquery.vizier import Vizier
    
    # MaNGA DR17 catalog on VizieR: VII/292 (MaNGA)
    # Also check SDSS DR17 spectroscopic catalog for velocity dispersions
    vizier = Vizier(columns=['*'], row_limit=-1)
    
    print(f"Querying VizieR for {len(shoes_df)} targets...")
    
    results = []
    for _, row in shoes_df.iterrows():
        name = row['normalized_name']
        coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg)
        
        try:
            # Query SDSS DR17 spectroscopic catalog for velocity dispersion
            # Catalog: V/154 (SDSS DR17 specObj)
            result = vizier.query_region(
                coord, 
                radius=radius_arcmin*u.arcmin,
                catalog=['V/154/sdss17']  # SDSS DR17 spectroscopic
            )
            
            if result and len(result) > 0 and len(result[0]) > 0:
                tbl = result[0]
                # Look for velocity dispersion column
                if 'velDisp' in tbl.colnames:
                    best = tbl[0]
                    sigma = float(best['velDisp']) if best['velDisp'] else np.nan
                    sigma_err = float(best['e_velDisp']) if 'e_velDisp' in tbl.colnames and best['e_velDisp'] else np.nan
                    
                    if np.isfinite(sigma) and sigma > 0:
                        results.append({
                            'shoes_name': name,
                            'shoes_ra': row['ra'],
                            'shoes_dec': row['dec'],
                            'sdss_ra': float(best['RA_ICRS']),
                            'sdss_dec': float(best['DE_ICRS']),
                            'stellar_sigma': sigma,
                            'stellar_sigma_err': sigma_err,
                            'source': 'SDSS_DR17_spec',
                            'status': 'match'
                        })
                        print(f"  ✓ {name}: σ = {sigma:.1f} ± {sigma_err:.1f} km/s (SDSS spec)")
                        continue
            
            print(f"  ✗ {name}: No SDSS spectroscopic match")
            results.append({'shoes_name': name, 'status': 'no_match'})
            
        except Exception as e:
            print(f"  ⚠ {name}: Query error - {e}")
            results.append({'shoes_name': name, 'status': 'error', 'error': str(e)})
    
    matches = [r for r in results if r.get('status') == 'match']
    return pd.DataFrame(matches) if matches else pd.DataFrame()


def query_hyperleda_sigma(shoes_df):
    """
    Query HyperLeda for velocity dispersions - most complete source for nearby galaxies.
    Uses the ledacat.cgi endpoint and parses HTML response.
    """
    import urllib.request
    import re
    import time
    
    print(f"\nQuerying HyperLeda for {len(shoes_df)} targets (sequential to avoid rate limits)...")
    
    results = []
    for idx, row in shoes_df.iterrows():
        name = row['normalized_name']
        # Clean name for HyperLeda query (remove spaces)
        query_name = name.replace(' ', '')
        
        try:
            # Use ledacat.cgi which returns full galaxy data
            url = f"http://leda.univ-lyon1.fr/ledacat.cgi?o={query_name}"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 TEP-H0/1.0'})
            with urllib.request.urlopen(req, timeout=20) as response:
                html = response.read().decode('latin-1')
            
            # Parse vdis from HTML: look for "vdis</a></td><td> XXX.X ± Y.Y"
            vdis_match = re.search(r'vdis\.html[^>]*>[^<]*</a></td><td>\s*([\d.]+)\s*(?:&#177;|±)\s*([\d.]+)', html)
            
            if vdis_match:
                sigma = float(vdis_match.group(1))
                sigma_err = float(vdis_match.group(2))
                results.append({
                    'shoes_name': name,
                    'stellar_sigma': sigma,
                    'stellar_sigma_err': sigma_err,
                    'source': 'HyperLeda',
                    'status': 'match'
                })
                print(f"  ✓ {name}: σ = {sigma:.1f} ± {sigma_err:.1f} km/s")
            else:
                # Check if galaxy exists but has no vdis
                if 'objname' in html.lower() or query_name.lower() in html.lower():
                    print(f"  ✗ {name}: Galaxy found but no σ measurement")
                else:
                    print(f"  ✗ {name}: Not found in HyperLeda")
                results.append({'shoes_name': name, 'status': 'no_match'})
            
            # Small delay to be nice to the server
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  ⚠ {name}: {e}")
            results.append({'shoes_name': name, 'status': 'error'})
    
    matches = [r for r in results if r.get('status') == 'match']
    return pd.DataFrame(matches) if matches else pd.DataFrame()

def crossmatch(shoes_df, manga_df, radius_arcsec=30.0):
    """
    Vectorized crossmatch using astropy's match_coordinates_sky.
    Much faster than loop-based matching.
    """
    if manga_df.empty:
        return pd.DataFrame()
    
    print(f"Running vectorized sky crossmatch (radius = {radius_arcsec}\")...")
    
    shoes_coords = SkyCoord(
        ra=shoes_df['ra'].values * u.deg,
        dec=shoes_df['dec'].values * u.deg
    )
    
    manga_coords = SkyCoord(
        ra=manga_df['ra'].values * u.deg,
        dec=manga_df['dec'].values * u.deg
    )
    
    # Vectorized matching - O(N log N) instead of O(N*M)
    idx, sep2d, _ = match_coordinates_sky(shoes_coords, manga_coords)
    
    matches = []
    for i, (name, match_idx, sep) in enumerate(zip(shoes_df['normalized_name'], idx, sep2d)):
        sep_arcsec = sep.arcsec
        
        if sep_arcsec < radius_arcsec:
            match = {
                'shoes_name': name,
                'shoes_ra': shoes_df.iloc[i]['ra'],
                'shoes_dec': shoes_df.iloc[i]['dec'],
                'manga_plateifu': manga_df.iloc[match_idx]['plateifu'],
                'manga_ra': manga_df.iloc[match_idx]['ra'],
                'manga_dec': manga_df.iloc[match_idx]['dec'],
                'separation_arcsec': sep_arcsec,
                'stellar_sigma_1re': manga_df.iloc[match_idx]['stellar_sigma_1re'],
                'stellar_sigma_1re_err': manga_df.iloc[match_idx]['stellar_sigma_1re_err'],
                'nsa_z': manga_df.iloc[match_idx]['nsa_z'],
            }
            matches.append(match)
            print(f"  ✓ {name}: σ = {match['stellar_sigma_1re']:.1f} ± {match['stellar_sigma_1re_err']:.1f} km/s (sep = {sep_arcsec:.1f}\")")
        else:
            print(f"  ✗ {name}: No MaNGA match within {radius_arcsec}\" (nearest = {sep_arcsec:.1f}\")")
    
    return pd.DataFrame(matches)

def main():
    print("=" * 60)
    print("SH0ES Velocity Dispersion Search")
    print(f"Using {NUM_WORKERS} parallel workers")
    print("=" * 60)
    
    # Load SH0ES hosts
    shoes_df = load_shoes_hosts()
    print(f"\nLoaded {len(shoes_df)} SH0ES host galaxies (excluding anchors).\n")
    
    # Try HyperLeda first (best coverage for nearby galaxies)
    matches = query_hyperleda_sigma(shoes_df)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULT: {len(matches)} / {len(shoes_df)} SH0ES hosts have stellar σ in HyperLeda.")
    print("=" * 60)
    
    if not matches.empty:
        # Clean up status column before saving
        if 'status' in matches.columns:
            matches = matches.drop(columns=['status'])
        
        out_path = PROJECT_ROOT / "results" / "outputs" / "hyperleda_sigma_crossmatch.csv"
        matches.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")
        print("\nMatched hosts:")
        print(matches[['shoes_name', 'stellar_sigma', 'stellar_sigma_err']].to_string(index=False))
    else:
        print("\nNo matches found in HyperLeda.")

if __name__ == "__main__":
    main()
