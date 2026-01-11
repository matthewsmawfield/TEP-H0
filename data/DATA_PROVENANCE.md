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

## 3. Velocity Dispersions (σ)

**File:** `data/raw/external/velocity_dispersions_literature.csv`

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

### Verification:
Run `python scripts/utils/verify_hyperleda.py` to cross-check values against HyperLEDA database.

---

## 4. TRGB Distances

**Source:** Chicago-Carnegie Hubble Program (CCHP)  
**Reference:** Freedman, W. L., et al. 2024, arXiv:2408.06153  
**Title:** "Status Report on the Chicago-Carnegie Hubble Program (CCHP)"

**Data:** TRGB distance moduli from HST/ACS F814W photometry  
**Table:** Table 2 of Freedman et al. (2024)

**Ingestion:** Values transcribed in `scripts/steps/step_7_trgb_comparison.py`

---

## 5. M31 Cepheid Catalog

**Source:** Kodric et al. (2018)  
**Reference:** Kodric, M., et al. 2018, AJ, 156, 130  
**VizieR Catalog:** J/AJ/156/130

**Ingestion:** Automated VizieR query via `astroquery.vizier` in `scripts/steps/step_5_m31_analysis.py`

---

## 6. LMC Cepheid Catalog

**Source:** OGLE-IV Survey  
**Reference:** Soszyński, I., et al. 2015, Acta Astronomica, 65, 297  
**VizieR Catalog:** J/AcA/65/297

**Ingestion:** Automated VizieR query via `astroquery.vizier` in `scripts/steps/step_7_lmc_replication.py`

---

## 7. Tully Group Catalog (Large-Scale Environment)

**Source:** Tully (2015)  
**Reference:** Tully, R. B. 2015, AJ, 149, 171  
**VizieR Catalog:** J/AJ/149/171/table5

**Ingestion:** Automated VizieR query in `scripts/steps/step_2_stratification.py`

---

## Verification Commands

```bash
# Verify velocity dispersions against HyperLEDA
python scripts/utils/verify_hyperleda.py

# Run full pipeline with data regeneration
python scripts/run_pipeline.py --regenerate

# Check TRGB data source
python -c "from scripts.steps.step_7_trgb_comparison import FREEDMAN_2024_TRGB; print(FREEDMAN_2024_TRGB)"
```

---

## Data Quality Flags

| Dataset | Automated Download | VizieR Query | Manual Transcription | Verified |
|---------|-------------------|--------------|---------------------|----------|
| SH0ES | ✓ | - | - | ✓ |
| Pantheon+ | ✓ | - | - | ✓ |
| Velocity Dispersions | - | Partial | ✓ | Pending |
| TRGB Distances | - | - | ✓ | ✓ |
| M31 Cepheids | - | ✓ | - | ✓ |
| LMC Cepheids | - | ✓ | - | ✓ |

---

*Last updated: 2026-01-11*
