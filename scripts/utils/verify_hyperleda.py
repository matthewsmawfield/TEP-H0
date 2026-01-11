"""
HyperLEDA Velocity Dispersion Verification Script
==================================================

This script verifies the velocity dispersion values in our literature CSV 
against the HyperLEDA database (http://leda.univ-lyon1.fr/).

HyperLEDA API:
- Base URL: http://leda.univ-lyon1.fr/ledacat.cgi
- Query format: objname=<galaxy>&out=<parameters>
- Key parameter: vdis (velocity dispersion in km/s)

References:
- Makarov, D., et al. 2014, A&A, 570, A13 (HyperLEDA description)
- http://leda.univ-lyon1.fr/leda/param/vdis.html (vdis documentation)

Usage:
    python scripts/utils/verify_hyperleda.py
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import sys
import time
import json
from io import StringIO

# VizieR (stable mirror for HyperLEDA PGC/name resolution)
try:
    from astroquery.vizier import Vizier
except ImportError:
    Vizier = None

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from scripts.utils.logger import print_status, print_table
except ImportError:
    def print_status(msg, level="INFO"):
        print(f"[{level}] {msg}")
    def print_table(headers, rows, title=None):
        if title:
            print(f"\n{title}")
        print(" | ".join(headers))
        for row in rows:
            print(" | ".join(str(x) for x in row))


class HyperLEDAVerifier:
    """
    Verify velocity dispersion measurements against HyperLEDA database.
    
    HyperLEDA (http://leda.univ-lyon1.fr/) is the authoritative database
    for extragalactic kinematics, maintained by Lyon Observatory.
    """
    
    # HyperLEDA query endpoint (use OHP mirror; Lyon endpoint often wraps output in HTML)
    LEDA_URL = "http://atlas.obs-hp.fr/hyperleda/ledacat.cgi"
    
    # VizieR catalog for HyperLEDA object resolution (PGC numbers)
    VIZIER_PGC_CATALOG = "VII/237/pgc"

    # VizieR catalog for HyperLEDA HI kinematics (log(2Vm))
    VIZIER_HI_CATALOG = "VII/238/hidat"
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.lit_csv_path = self.data_dir / "raw" / "external" / "velocity_dispersions_literature.csv"
        self.output_path = self.data_dir / "raw" / "external" / "velocity_dispersions_verified.csv"
        self.report_path = self.root_dir / "results" / "outputs" / "hyperleda_verification_report.json"

    def _resolve_pgc_vizier(self, galaxy_name):
        """Resolve a galaxy name to a PGC identifier using VizieR (VII/237/pgc)."""
        if Vizier is None:
            return None

        try:
            v = Vizier(columns=['PGC'])
            v.ROW_LIMIT = 1
            cats = v.query_object(str(galaxy_name).strip(), catalog=self.VIZIER_PGC_CATALOG)
            if not cats or len(cats) == 0 or len(cats[0]) == 0:
                return None
            pgc = cats[0]['PGC'][0]
            return int(pgc) if pgc is not None else None
        except Exception:
            return None

    def _query_hi_log2vm_vizier(self, pgc):
        if Vizier is None or pgc is None:
            return None

        try:
            v = Vizier(columns=['PGC', 'log(2Vm)'])
            v.ROW_LIMIT = 1
            cats = v.query_constraints(catalog=self.VIZIER_HI_CATALOG, PGC=int(pgc))
            if not cats or len(cats) == 0 or len(cats[0]) == 0:
                return None
            val = cats[0]['log(2Vm)'][0]
            if val is None:
                return None
            # Astropy tables may return masked values
            try:
                if hasattr(val, 'mask') and bool(val.mask):
                    return None
            except Exception:
                pass
            return float(val)
        except Exception:
            return None

    def _query_leda_single_field(self, obj, field):
        """Query HyperLEDA for a single field (robust to missing columns)."""
        params = {
            'o': str(obj).strip(),
            'd': str(field).strip(),
            'a': 't',
        }

        try:
            response = requests.get(self.LEDA_URL, params=params, timeout=10)
            response.raise_for_status()

            if "Internal database error" in response.text:
                return None

            # Parse first non-comment, non-HTML line
            for line in response.text.split('\n'):
                stripped = line.strip()
                if not stripped or stripped.startswith('#') or stripped.startswith('<'):
                    continue

                # In single-field mode, response often looks like: "64.4" or "NGC7541 64.4"
                tokens = stripped.split()
                if len(tokens) == 0:
                    return None
                candidate = tokens[-1]
                try:
                    return float(candidate)
                except ValueError:
                    return None

            return None

        except requests.RequestException:
            return None

    
    def _calculate_sigma_from_w50(self, w50):
        """
        Estimate velocity dispersion from HI 21cm line width.
        
        The standard approximation for late-type galaxies:
        σ ≈ 0.7 × (W50 / 2) / sin(i)
        
        For a typical inclination average, this simplifies to:
        σ ≈ 0.35 × W50 (approximate)
        
        Reference: Kormendy & Ho (2013), Equation 5
        Note: This is an indirect estimate, not a direct measurement.
        """
        if w50 is None or w50 <= 0:
            return None
        return 0.35 * w50

    def _sigma_from_vrot_proxy(self, vrot):
        """Proxy used in prior audits: sigma ≈ Vrot / 1.7 (Kormendy & Ho 2013)."""
        if vrot is None or not np.isfinite(vrot) or vrot <= 0:
            return None
        return float(vrot) / 1.7

    def _sigma_from_log2vm(self, log2vm):
        if log2vm is None or not np.isfinite(log2vm):
            return None
        # HyperLEDA HI table provides log10(2*Vmax) in km/s, so W50 ~ 2*Vmax ~ 10**log(2Vm)
        w50_est = 10 ** float(log2vm)
        return 0.35 * w50_est
    
    def verify_all(self):
        """
        Verify all galaxies in the literature CSV against HyperLEDA.
        """
        print_status("=" * 60, "INFO")
        print_status("HYPERLEDA VELOCITY DISPERSION VERIFICATION", "INFO")
        print_status("=" * 60, "INFO")
        
        # Load our literature values
        if not self.lit_csv_path.exists():
            print_status(f"Literature CSV not found: {self.lit_csv_path}", "ERROR")
            return None
        
        lit_df = pd.read_csv(self.lit_csv_path, comment='#')
        print_status(f"Loaded {len(lit_df)} galaxies from literature CSV", "INFO")
        
        # Query HyperLEDA for each galaxy
        results = []
        matched = 0
        discrepant = 0
        
        print_status("\nQuerying HyperLEDA database (VizieR PGC resolution + ledacat.cgi)...", "INFO")
        
        for _, row in lit_df.iterrows():
            galaxy = row['galaxy']
            lit_sigma = row['sigma_kms']
            lit_method = row.get('method', 'unknown')
            lit_source = row.get('source', 'unknown')

            # Resolve to PGC if possible (stable object key)
            pgc = self._resolve_pgc_vizier(galaxy)
            leda_obj = f"PGC {pgc}" if pgc is not None else galaxy

            hi_log2vm = self._query_hi_log2vm_vizier(pgc)

            # Query fields individually (robust to missing/shifted values)
            leda_vdis = self._query_leda_single_field(leda_obj, 'vdis')
            leda_e_vdis = self._query_leda_single_field(leda_obj, 'e_vdis')
            leda_vrot = self._query_leda_single_field(leda_obj, 'vrot')
            leda_w50 = self._query_leda_single_field(leda_obj, 'w50')

            time.sleep(0.5)  # Rate limiting

            if leda_vdis is None and leda_vrot is None and leda_w50 is None:
                print_status(f"  {galaxy}: NO DATA FOUND in HyperLEDA (via {leda_obj})", "WARNING")
                results.append({
                    'galaxy': galaxy,
                    'lit_sigma': lit_sigma,
                    'lit_method': lit_method,
                    'lit_source': lit_source,
                    'leda_obj': leda_obj,
                    'leda_vdis': None,
                    'leda_e_vdis': None,
                    'leda_vrot': None,
                    'leda_w50': None,
                    'sigma_from_vrot': None,
                    'sigma_from_w50': None,
                    'status': 'NOT_FOUND',
                    'discrepancy_pct': None
                })
                continue

            sigma_from_w50 = self._calculate_sigma_from_w50(leda_w50)
            sigma_from_vrot = self._sigma_from_vrot_proxy(leda_vrot)
            sigma_from_hi = self._sigma_from_log2vm(hi_log2vm)

            # Compare apples-to-apples when possible:
            # - stellar absorption measurements should match vdis
            # - HI linewidth proxies should match a proxy (prefer vrot/1.7; fallback to 0.35*w50)
            method_str = str(lit_method).lower()
            if 'hi linewidth' in method_str or 'proxy' in method_str:
                comparison_sigma = (
                    sigma_from_hi
                    if sigma_from_hi is not None
                    else (sigma_from_vrot if sigma_from_vrot is not None else sigma_from_w50)
                )
            else:
                # For direct stellar absorption, do NOT automatically treat vrot/HI as a substitute,
                # since that masks classification mistakes.
                comparison_sigma = leda_vdis
            
            if comparison_sigma is not None and np.isfinite(comparison_sigma) and comparison_sigma > 0:
                discrepancy = abs(lit_sigma - comparison_sigma) / comparison_sigma * 100
                matched += 1
                
                if discrepancy > 20:
                    status = 'DISCREPANT'
                    discrepant += 1
                    print_status(f"  {galaxy}: σ_lit={lit_sigma:.0f}, σ_LEDA={comparison_sigma:.0f} "
                               f"({discrepancy:.1f}% off) ⚠", "WARNING")
                else:
                    status = 'VERIFIED'
                    print_status(f"  {galaxy}: σ_lit={lit_sigma:.0f}, σ_LEDA={comparison_sigma:.0f} "
                               f"({discrepancy:.1f}% off) ✓", "INFO")
            else:
                status = 'NO_SIGMA_DATA'
                discrepancy = None
                print_status(f"  {galaxy}: No σ data in HyperLEDA", "INFO")
            
            results.append({
                'galaxy': galaxy,
                'lit_sigma': lit_sigma,
                'lit_method': lit_method,
                'lit_source': lit_source,
                'leda_obj': leda_obj,
                'pgc': pgc,
                'hi_log2vm': hi_log2vm,
                'leda_vdis': leda_vdis,
                'leda_e_vdis': leda_e_vdis,
                'leda_vrot': leda_vrot,
                'leda_w50': leda_w50,
                'sigma_from_vrot': sigma_from_vrot,
                'sigma_from_w50': sigma_from_w50,
                'sigma_from_hi': sigma_from_hi,
                'status': status,
                'discrepancy_pct': discrepancy
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        print_status("\n" + "=" * 60, "INFO")
        print_status("VERIFICATION SUMMARY", "INFO")
        print_status("=" * 60, "INFO")
        
        n_total = len(results_df)
        n_verified = len(results_df[results_df['status'] == 'VERIFIED'])
        n_discrepant = len(results_df[results_df['status'] == 'DISCREPANT'])
        n_not_found = len(results_df[results_df['status'] == 'NOT_FOUND'])
        n_no_data = len(results_df[results_df['status'] == 'NO_SIGMA_DATA'])
        
        headers = ["Status", "Count", "Percentage"]
        rows = [
            ["VERIFIED (< 20% diff)", str(n_verified), f"{100*n_verified/n_total:.1f}%"],
            ["DISCREPANT (> 20% diff)", str(n_discrepant), f"{100*n_discrepant/n_total:.1f}%"],
            ["NOT FOUND in HyperLEDA", str(n_not_found), f"{100*n_not_found/n_total:.1f}%"],
            ["NO σ DATA in HyperLEDA", str(n_no_data), f"{100*n_no_data/n_total:.1f}%"],
        ]
        print_table(headers, rows, title="Verification Results")
        
        # Save verified CSV
        results_df.to_csv(self.output_path, index=False)
        print_status(f"\nVerified data saved to: {self.output_path}", "INFO")
        
        # Save report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'source': 'HyperLEDA via OHP mirror (ledacat.cgi) with VizieR PGC resolution (VII/237/pgc)',
            'reference': 'Makarov et al. 2014, A&A, 570, A13',
            'n_total': n_total,
            'n_verified': n_verified,
            'n_discrepant': n_discrepant,
            'n_not_found': n_not_found,
            'n_no_data': n_no_data,
            'verification_rate': n_verified / n_total if n_total > 0 else 0,
            'discrepancies': results_df[results_df['status'] == 'DISCREPANT'][
                ['galaxy', 'lit_sigma', 'leda_vdis', 'discrepancy_pct']
            ].to_dict('records')
        }
        
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print_status(f"Verification report saved to: {self.report_path}", "INFO")
        
        # Final assessment
        if n_discrepant == 0 and n_verified > 0:
            print_status("\n✓ ALL VERIFIED VALUES MATCH HYPERLEDA WITHIN 20%", "SUCCESS")
        elif n_discrepant > 0:
            print_status(f"\n⚠ {n_discrepant} DISCREPANT VALUES REQUIRE REVIEW", "WARNING")
        
        return results_df
    
    def add_leda_column(self):
        """
        Add HyperLEDA object IDs to the literature CSV for citation purposes.
        """
        # This would add PGC numbers for each galaxy
        # PGC (Principal Galaxy Catalog) numbers are unique HyperLEDA identifiers
        pass


def main():
    verifier = HyperLEDAVerifier()
    results = verifier.verify_all()
    
    if results is not None:
        print_status("\n" + "=" * 60, "INFO")
        print_status("HYPERLEDA VERIFICATION COMPLETE", "INFO")
        print_status("=" * 60, "INFO")


if __name__ == "__main__":
    main()
