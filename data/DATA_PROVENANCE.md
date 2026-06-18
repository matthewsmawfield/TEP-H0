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

## 3. Velocity Dispersions (σ) — CANONICAL DATA

**Status:** These values are pinned/committed. The pipeline uses the committed file by default.

**Primary File:** `data/raw/external/velocity_dispersions_literature_regenerated.csv` (canonical, committed to git)

**Fallback File:** `data/raw/external/velocity_dispersions_literature.csv` (legacy)

### Sources:

| Source | Reference | Method | N galaxies |
|--------|-----------|--------|------------|
| HyperLEDA | Makarov et al. 2014, A&A, 570, A13 | Central stellar σ | ~15 |
| Ho et al. 2009 | Ho, L. C., et al. 2009, ApJS, 183, 1 | Long-slit spectroscopy | ~5 |
| Kormendy & Ho 2013 | Kormendy, J. & Ho, L. C. 2013, ARA&A, 51, 511 | Compilation | ~3 |
| SDSS DR7 | Abazajian et al. 2009, ApJS, 182, 543 | Fiber spectroscopy | ~5 |

### Methodology:
- **Direct measurements:** Central stellar velocity dispersion from absorption line fitting
- **HI linewidth proxy:** σ ≈ 0.7 × W50/2 (for galaxies without direct σ measurements)
- **Aperture correction:** Jorgensen et al. (1995) power-law normalized to R_eff/8

### Reproducibility Note
The velocity dispersion catalog is **committed to git as canonical data** to ensure exact reproducibility. External databases (HyperLEDA, VizieR) are updated over time; re-querying them may return different values. To force a fresh download from live databases:

```bash
python scripts/run_pipeline.py --rebuild-sigma
```

To verify the committed values against HyperLEDA:
```bash
python scripts/utils/verify_hyperleda.py
```

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
| Velocity Dispersions | - | Partial | ✓ | ✓ | ✓ (committed regenerated CSV) |
| RC3 Metadata (D25, R_eff) | - | ✓ | - | ✓ | ✓ (committed enriched CSV) |
| TRGB Distances | - | - | ✓ | ✓ | ✓ (hardcoded in script) |
| M31 Cepheids | - | ✓ | - | ✓ | - (live query) |
| LMC Cepheids | - | ✓ | - | ✓ | - (live query) |

---

*Last updated: 2026-06-18*
*Pipeline version: v0.7*
