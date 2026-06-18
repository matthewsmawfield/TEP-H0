# Data Integrity Statement — TEP-H0 Paper 11

## Single Source of Truth

The ONLY file containing velocity dispersion data used by this pipeline is:

**`data/raw/external/velocity_dispersions_literature.csv`**

This is a manually curated master file. Every entry includes:
- Galaxy name
- Velocity dispersion σ (km/s) and uncertainty
- ADS bibcode linking to the peer-reviewed source
- Source URL to the ADS abstract page
- Measurement method
- Provenance notes (including documented corrections)
- Date accessed (YYYY-MM-DD)
- Traceability confidence (HIGH / MEDIUM)

## What This Repository Does NOT Contain

The following files have been **permanently deleted** to eliminate confusion:

- ❌ `velocity_dispersions_literature_TRACEABLE.csv` — merged into the master file
- ❌ `velocity_dispersions_literature_regenerated.csv` — stale auto-generated data
- ❌ `velocity_dispersions_literature_detailed.csv` — outdated values
- ❌ `sigma_literature_overrides.csv` — empty file, never used
- ❌ `scripts/utils/generate_literature_csv_from_traceable.py` — no longer needed
- ❌ `scripts/utils/build_sigma_catalog.py` — dead code, no callers

## Verification

To verify every sigma value against HyperLEDA:
```bash
python scripts/utils/verify_hyperleda.py
```

To run the full pipeline audit:
```bash
python scripts/utils/pipeline_audit.py
```

To check all results against the master file:
```bash
python scripts/run_pipeline.py --rebuild-sigma
```

## Data Sources

| Data | Source | File |
|------|--------|------|
| Cepheid/SN Ia distances | Scolnic et al. (2022) Pantheon+ | `data/raw/Pantheon+SH0ES.dat` |
| Velocity dispersions | Curated literature (Ho+2009, Campbell+2014, etc.) | `data/raw/external/velocity_dispersions_literature.csv` |
| Cepheid P-L data | Riess et al. (2022) SH0ES | `data/raw/external/Cepheid-Distance-Ladder-Data/SH0ES2022/` |
| Host coordinates | HyperLEDA / VizieR | `data/interim/hosts_coords.csv` (generated) |

## Corrections Documented

The following values were corrected from previous undocumented/fabricated versions:

| Galaxy | Old (untraceable) | New (traceable) | Source |
|--------|-------------------|-----------------|--------|
| NGC 7541 | 125.0 | **64.4** | Héraudeau+1999 / HyperLEDA |
| NGC 7678 | 138.0 | **107.0** | Héraudeau+1999 / HyperLEDA |
| NGC 3021 | 88.0 | **55.7** | Ho+2009 / HyperLEDA |
| NGC 3254 | 95.0 | **117.8** | Ho+2009 / HyperLEDA |
| NGC 3982 | 82.0 | **87.3** | Ho+2009 / HyperLEDA |
| NGC 4536 | 95.0 | **103.7** | Ho+2009 / HyperLEDA |
| NGC 4639 | 88.0 | **96.2** | Ho+2009 / HyperLEDA |

All corrections are documented in the notes column of the master file.

## File Integrity Verification

SHA-256 checksums of all data files are in `data/CHECKSUMS.txt`.
These allow independent verification that files have not been modified.

## Guarantee

Every numerical value in `data/raw/external/velocity_dispersions_literature.csv` traces to a peer-reviewed publication via ADS bibcode. There are no fabricated values, no fallback values, and no hidden data sources. The pipeline reads exclusively from this file and raises `FileNotFoundError` if it is missing.
