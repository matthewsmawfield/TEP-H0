# Velocity Dispersion Data — Single Source of Truth

## Overview

This directory contains the **authoritative velocity dispersion compilation** for TEP-H0 Paper 11. There is exactly **one** master file. All pipeline scripts read exclusively from this file. There are no auto-generated inputs, no fallbacks, and no overrides.

## File Structure

### `velocity_dispersions_literature.csv` — THE MASTER FILE

**This is the only file containing velocity dispersion data.**

- 43 galaxies with measured central velocity dispersions
- Every value has full traceability:
  - `galaxy` — Galaxy identifier
  - `sigma_kms` — Central velocity dispersion in km/s
  - `error_kms` — Uncertainty in km/s
  - `source_bibcode` — ADS bibcode of peer-reviewed source
  - `source_url` — Direct link to ADS abstract
  - `method` — Measurement method (stellar absorption / HI linewidth proxy / stellar kinematics / disk kinematics)
  - `notes` — Provenance notes, including documented corrections
  - `traceability_confidence` — HIGH (direct stellar absorption) or MEDIUM (HI linewidth proxy or large uncertainty)
  - `date_accessed` — Date data was retrieved/verified (YYYY-MM-DD)

## Removed Files (No Longer Present)

The following files were removed to eliminate confusion and enforce a single source of truth:

- `velocity_dispersions_literature_TRACEABLE.csv` — Merged into the master file above
- `velocity_dispersions_literature_regenerated.csv` — Stale automated query results; could silently produce different scientific results
- `velocity_dispersions_literature_detailed.csv` — Old file with outdated values
- `sigma_literature_overrides.csv` — Empty file, never used
- `scripts/utils/generate_literature_csv_from_traceable.py` — No longer needed; master file is direct input

## How to Update a Value

1. Edit `velocity_dispersions_literature.csv` directly with the new value, bibcode, URL, and notes
2. Document the change in the notes column (e.g., "Corrected from X to Y per [source]")
3. Update `date_accessed` to the current date
4. Re-run the full pipeline: `python scripts/run_pipeline.py --rebuild-sigma`
5. Verify with `python scripts/utils/pipeline_audit.py`

## Verification

Run `python scripts/utils/pipeline_audit.py` after any change. The audit checks:
- All pipeline results are consistent with the master file
- All narrative files (abstract, results, discussion, conclusion, README, site) contain the current headline numbers
- No forbidden values are present
- All required output files exist

## Contact

For questions about velocity dispersion provenance, check the notes column in the master file or consult the audit report (`results/outputs/pipeline_audit_report.json`).
