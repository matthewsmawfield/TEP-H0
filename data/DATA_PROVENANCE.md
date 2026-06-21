# TEP-H0 Data Provenance

This document provides complete provenance information for all external data used in the TEP-H0 analysis pipeline.

## 1. SH0ES Cepheid Distance Ladder Data

**Source:** Riess et al. (2022) - SH0ES Team  
**Reference:** Riess, A. G., et al. 2022, ApJ, 934, L7  
**arXiv:** 2112.04510  
**Data Repository:** https://github.com/marcushogas/Cepheid-Distance-Ladder-Data

**Files Used:**
- `q_R22.txt` - Quality flags
- `y_R22.txt` - Observables vector
- `L_R22.txt` - Design matrix
- `C_R22.txt` - Covariance matrix

**Ingestion:** Automated download via `scripts/steps/step_1_data_ingestion.py`

---

## 2. Pantheon+ SN Ia Distances

**Source:** Scolnic et al. (2022) - Pantheon+ Collaboration  
**Reference:** Scolnic, D., et al. 2022, ApJ, 938, 113  
**arXiv:** 2112.03863  
**Data Repository:** https://github.com/PantheonPlusSH0ES/DataRelease

**File Used:** `Pantheon+SH0ES.dat`

**Ingestion:** Automated download via `scripts/steps/step_1_data_ingestion.py`

---

## 3. Velocity Dispersions (σ) — FULLY TRACEABLE MASTER DATA

**Status:** Every value is traceable to an original literature source with ADS bibcode.

**Primary File:** `data/raw/external/velocity_dispersions_literature.csv` (master, committed to git)

This is the ONLY file containing velocity dispersion data. Every entry includes:
- ADS bibcode linking to the peer-reviewed source publication
- Source URL to the ADS abstract page
- Measurement method (stellar absorption, HI linewidth proxy, etc.)
- Provenance notes documenting any corrections
- Date accessed for audit trail
- Traceability confidence rating (HIGH / MEDIUM)

There are no other sigma input files. The pipeline reads exclusively from this file.
**Audit Report:** `results/outputs/pipeline_audit_report.json`

### Sources with ADS Bibcodes

| Source | Bibcode | Reference | Method | N |
|--------|---------|-----------|--------|---|
| HyperLEDA | 2014A&A...570A..13M | Makarov et al. 2014, A&A, 570, A13 | Central stellar σ | 15 |
| Ho+2009 | 2009ApJS..183....1H | Ho, L. C., et al. 2009, ApJS, 183, 1 | Long-slit spectroscopy | 6 |
| Kormendy&Ho2013 | 2013ARA&A..51..511K | Kormendy, J. & Ho, L. C. 2013, ARA&A, 51, 511 | Compilation | 4 |
| BASS DR2 | 2022ApJS..261....6K | Koss et al. 2022, ApJS, 261, 6 | X-ray AGN host σ | 1 |
| SDSS DR7 | 2009ApJS..182..543A | Abazajian et al. 2009, ApJS, 182, 543 | Fiber spectroscopy | 1 |
| 6dFGSv | 2014MNRAS.443.1231C | Campbell et al. 2014, MNRAS, 443, 1231 | Fundamental Plane | 19 |
| Saulder+2019 | 2019MNRAS.482.1427S | Saulder et al. 2019, MNRAS, 482, 1427 | FP Vdisp30Hd | 1 |
| Ho 2007 | 2007ApJ...668...94H | Ho, L. C. 2007, ApJ, 668, 94 | Kinematics catalog | 1 |

### Methodology
- **Direct measurements:** Central stellar velocity dispersion from absorption line fitting (Ho+2009, BASS DR2, SDSS DR7, Kormendy&Ho2013)
- **HI linewidth proxy:** σ from W50 via Campbell+2014 6dFGSv catalog, calibrated via HyperLEDA
- **Aperture correction:** Jorgensen et al. (1995) power-law normalized to R_eff/8

### Traceability Guarantee
Each entry includes:
- `source_bibcode`: ADS bibcode for direct literature verification
- `source_url`: ADS abstract URL
- `notes`: Detailed provenance including LEDA reference codes and discrepancy flags
- `traceability_confidence`: HIGH (single clear source) / MEDIUM (multiple sources agree)

### Reproducibility
The catalog is committed to git as canonical data. To verify against original sources:
```bash
python scripts/utils/verify_hyperleda.py
```

External databases update over time; the committed values represent the most accurate measurements as of 2026-06-18.

---

## 4. Galaxy Metadata (RC3 Sizes) — CANONICAL DATA

**File:** `data/processed/hosts_metadata_enriched.csv`

**"Enriched" meaning:** This file contains the base galaxy metadata *plus* structural parameters from the **Third Reference Catalog of Bright Galaxies (RC3, de Vaucouleurs et al., VizieR VII/155)**:
- `log_d25` — Logarithm of isophotal diameter at 25 mag/arcsec² (in 0.1 arcmin)
- `r25_arcsec` — Isophotal radius derived from D25 (arcsec)
- `r_eff_arcsec` — Effective (half-light) radius, estimated as 0.5 × R25 (arcsec)

These are **published catalog values**, not manual tweaks. They are required for the Jorgensen et al. (1995) aperture correction applied in Step 1b.

**Reproducibility:** The RC3 metadata is committed to git as canonical data. The `fetch_galaxy_metadata.py` utility can re-query Vizier VII/155, but the pipeline preserves the committed file by default. Re-querying requires:

```bash
rm data/processed/hosts_metadata_enriched.csv
python scripts/utils/fetch_metadata.py
```

---

## 5. Hosts Processed Data — CANONICAL DATA

**File:** `data/processed/hosts_processed.csv`

This file contains the merged SH0ES host data (coordinates, redshifts, Pantheon+ properties) combined with the measured velocity dispersions and aperture-corrected sigma values. It is the primary input to Step 2 (Stratification).

**Reproducibility:** This file is committed to git. The pipeline preserves it by default to prevent external database drift from changing the analysis inputs. Step 1 will only overwrite it if the file does not already exist.

---

## 6. TRGB Distances

**Source:** Chicago-Carnegie Hubble Program (CCHP)  
**Reference:** Freedman, W. L., et al. 2024, arXiv:2408.06153  
**Title:** "Status Report on the Chicago-Carnegie Hubble Program (CCHP)"

**Data:** TRGB distance moduli from HST/ACS F814W photometry  
**Table:** Table 2 of Freedman et al. (2024)

**Ingestion:** Values transcribed in `scripts/steps/step_7_trgb_comparison.py`

---

## 7. M31 Cepheid Catalog

**Source:** Kodric et al. (2018)  
**Reference:** Kodric, M., et al. 2018, AJ, 156, 130  
**VizieR Catalog:** J/AJ/156/130

**Ingestion:** Automated VizieR query via `astroquery.vizier` in `scripts/steps/step_5_m31_analysis.py`

---

## 8. LMC Cepheid Catalog

**Source:** OGLE-IV Survey  
**Reference:** Soszyński, I., et al. 2015, Acta Astronomica, 65, 297  
**VizieR Catalog:** J/AcA/65/297

**Ingestion:** Automated VizieR query via `astroquery.vizier` in `scripts/steps/step_7_lmc_replication.py`

---

## 9. Tully Group Catalog (Large-Scale Environment)

**Source:** Tully (2015)  
**Reference:** Tully, R. B. 2015, AJ, 149, 171  
**VizieR Catalog:** J/AJ/149/171/table5

**Ingestion:** Automated VizieR query in `scripts/steps/step_2_stratification.py`

---

## Verification Commands

```bash
# Verify velocity dispersions against HyperLEDA
python scripts/utils/verify_hyperleda.py

# Run full pipeline (uses committed canonical data)
python scripts/run_pipeline.py

# Force full data rebuild from live databases
python scripts/run_pipeline.py --rebuild-sigma

# Check TRGB data source
python -c "from scripts.steps.step_7_trgb_comparison import FREEDMAN_2024_TRGB; print(FREEDMAN_2024_TRGB)"
```

---

## Data Quality Flags

| Dataset | Automated Download | VizieR Query | Manual Transcription | Verified | Canonical / Pinned |
|---------|-------------------|--------------|---------------------|----------|-------------------|
| SH0ES | ✓ | - | - | ✓ | ✓ (committed R22 matrices) |
| Pantheon+ | ✓ | - | - | ✓ | ✓ (committed .dat file) |
| Velocity Dispersions | - | - | ✓ | ✓ | ✓ (committed literature CSV) |
| RC3 Metadata (D25, R_eff) | - | ✓ | - | ✓ | ✓ (committed enriched CSV) |
| TRGB Distances | - | - | ✓ | ✓ | ✓ (hardcoded in script) |
| M31 Cepheids | - | ✓ | - | ✓ | - (live query) |
| LMC Cepheids | - | ✓ | - | ✓ | - (live query) |

---

*Last updated: 2026-06-21*
*Pipeline version: v0.7*
