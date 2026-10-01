# TEP-H0 Data File Manifest
# 
# This file provides a complete inventory of ALL data files in this repository,
# their roles, and whether they are inputs, outputs, or auxiliary files.
# 
# The SINGLE SOURCE OF TRUTH for velocity dispersion data is:
#   data/raw/external/velocity_dispersions_literature.csv
#
# NO OTHER FILE contains velocity dispersion input data.
# All other sigma-related files are pipeline outputs.
#

## INPUT DATA (Published / Curated)

data/raw/Pantheon+SH0ES.dat
  Source: Scolnic et al. (2022) ApJ 938 113
  Role: Primary input — Cepheid and SN Ia distances
  Status: Published data, downloaded from official repository

data/raw/external/velocity_dispersions_literature.csv
  Source: Curated from peer-reviewed literature (Ho+2009, Campbell+2014 6dFGSv,
          Heraudeau+1999, Kormendy & Ho 2013, Koss+2022, Saulder+2019, Riess+2022,
          van der Marel+2002, Harris & Zaritsky 2006, Alves & Nelson 2000)
  Role: SINGLE SOURCE OF TRUTH for the host inner-potential velocity scale sigma_*
  Status: Manually curated master file with ADS bibcodes
  Columns: galaxy, pgc, vrot_kms, vrot_error_kms, sigma_kms, error_kms,
           source_bibcode, source_url, method, notes, traceability_confidence,
           date_accessed
  Coordinate semantics:
    sigma_kms is the LOCAL INNER-POTENTIAL scale sigma_* used by the primary TEP
    coordinate X_i = (S_i sigma_{*,i}^2 - U_ref)/c^2: direct stellar-absorption
    dispersions where measured (method = "stellar absorption" / "stellar
    kinematics"), documented H I linewidth proxies otherwise.
    vrot_kms is the pinned HyperLEDA maximum rotation velocity retained ONLY as
    the audit/robustness coordinate u_phi = V_rot/sqrt(2); it is not consumed as
    the primary coordinate (inclination deprojection is unstable for
    low-inclination hosts and it probes the outer halo, not the Cepheid
    environment).
  Provenance note: versions v0.9/v0.10 had erroneously populated sigma_kms with
    the homogeneous u_phi rotation proxy, attenuating the environmental signal
  Bibcode note: bibcode 2014MNRAS.443.1231C (Campbell et al. 2014, 6dFGSv data
    release) legitimately carries two method labels because that single data
    release publishes both products used here: Fundamental Plane stellar
    absorption dispersions (used directly, e.g. NGC 1015 at 106.5 km/s) and
    W50 21-cm linewidths (converted sigma ~= W50/2.83 for the "HI linewidth
    proxy" rows). The method column, not the bibcode, distinguishes the
    measurement channel.
    (Gamma_X 2.35e7 -> 1.16e7). The sigma_* construction was restored with full
    per-row provenance; the rotation column is kept for audit only.
  CRITICAL: This is the ONLY file the pipeline reads for sigma values.

data/raw/external/anchor_galaxy_data.csv
  Source: Published SH0ES anchor galaxy Cepheid data
  Role: Anchor galaxy Cepheid photometry (LMC, NGC 4258, MW, M31)
  Status: Published data

data/raw/external/trgb_distances_freedman2024.csv
  Source: Freedman et al. (2024) CCHP
  Role: TRGB distance moduli for cross-probe comparison
  Status: Published data from Chicago-Carnegie Hubble Program

data/raw/external/tully2015_2mrs_groups_table5.csv
  Source: Tully (2015) 2MRS group catalog
  Role: Group membership counts (N_mb) for environment controls
  Status: Published catalog

data/raw/external/hyperleda_inclinations.csv
  Source: HyperLEDA (pinned query)
  Role: Galaxy inclinations and rotation-curve parameters for the u_phi
        audit coordinate and the Step 64 inclination-proxy audit
  Status: Published catalog query, per-host rows

data/raw/external/rc3_d25_hosts.csv
  Source: RC3 (de Vaucouleurs et al. 1991)
  Role: Isophotal diameters D25 and host cross-identifications
  Status: Published catalog

data/raw/external/stiskalek2026_manticore_hosts.csv
  Source: Stiskalek et al. (2026), Manticore local-Universe reconstruction
  Role: Local density-field estimates for host environments
  Status: Published data

data/raw/external/Pantheon+SH0ES_STAT+SYS.cov
  Source: Scolnic et al. (2022) Pantheon+ release
  Role: Full statistical+systematic covariance matrix for Pantheon+ SH0ES
  Status: Published data from official repository

data/raw/external/Cepheid-Distance-Ladder-Data/
  Source: Riess et al. (2022) SH0ES Team
  Role: Git submodule — Cepheid P-L data
  Status: Published data from official repository
  Contents:
    - C_R22.txt: Cepheid light curve parameters
    - L_R22.txt: Cepheid luminosities
    - q_R22.txt: Quality flags
    - y_R22.txt: Cepheid properties
    - README.md, Read_and_Fit_R22.ipynb: Documentation

data/interim/hosts_coords.csv
  Source: Generated from coordinate queries
  Role: Intermediate — host galaxy coordinates
  Status: Pipeline-generated from literature names

data/interim/hosts_properties.csv
  Source: Generated from metadata compilation
  Role: Intermediate — host galaxy properties
  Status: Pipeline-generated

data/interim/r22_distances.csv
  Source: Extracted from data/raw/Pantheon+SH0ES.dat
  Role: Intermediate — R22 distance moduli with error bars
  Status: Pipeline-generated (step_1_data_ingestion)

data/interim/r22_mu_covariance.npy
  Source: Computed from Pantheon+SH0ES covariance
  Role: Intermediate — Full GLS distance-modulus covariance matrix (29×29)
  Status: Pipeline-generated (step_1_data_ingestion)

data/interim/r22_mu_covariance_labels.json
  Source: Generated alongside covariance matrix
  Role: Intermediate — Host labels for covariance matrix ordering
  Status: Pipeline-generated (step_1_data_ingestion)

data/interim/kodric_hst_cepheids.csv
  Source: Kodric et al. (2018), ApJ 864 59
  Role: Intermediate — HST J/H band M31 Cepheid photometry
  Status: Pipeline-processed from published catalog

data/interim/reconstructed_shoes_cepheids.csv
  Source: Reconstructed from SH0ES distance ladder
  Role: Intermediate — Per-Cepheid distance moduli reconstruction
  Status: Pipeline-generated (step_1_data_ingestion)

data/processed/hosts_metadata_enriched.csv
  Source: Pipeline output
  Role: Processed — Host metadata with cross-matched properties
  Status: Generated by step_1_data_ingestion

## OUTPUT DATA (Generated by Pipeline)

data/processed/hosts_processed.csv
  Source: Pipeline output
  Role: Final processed host data
  Status: Generated by scripts/steps/step_1_data_ingestion.py
  Note: Reads sigma from data/raw/external/velocity_dispersions_literature.csv ONLY

results/outputs/stratified_h0.csv
  Source: Pipeline output
  Role: Stratified H0 results per host
  Status: Generated by TEP analysis pipeline

results/outputs/tep_correction_results.json
  Source: Pipeline output
  Role: Primary TEP statistical results
  Status: Generated by TEP analysis

results/outputs/TEP_FINAL_ROBUSTNESS_REPORT.md
  Source: Pipeline output
  Role: Human-readable robustness report
  Status: Generated by step 9

results/outputs/pipeline_audit_report.json
  Source: Pipeline output
  Role: Audit check results
  Status: Generated by scripts/utils/pipeline_audit.py

results/outputs/velocity_dispersions_verified.csv
  Source: Pipeline output (verification)
  Role: HyperLEDA cross-check of master file values
  Status: Generated by scripts/utils/verify_hyperleda.py
  IMPORTANT: This is a DIAGNOSTIC OUTPUT, not an input. It cross-checks
  the master file against HyperLEDA but does not override anything.

results/outputs/sigma_regeneration_report.json
  Source: Pipeline output (metadata)
  Role: Reports which sigma source was used
  Status: Generated by step_0_sigma_catalog.py
  IMPORTANT: Despite the name, this is just a metadata report. The pipeline
  does NOT regenerate sigma values; it reads directly from the master file.

results/outputs/sigma_provenance_table.csv
  Source: Pipeline output
  Role: Per-host sigma provenance summary
  Status: Generated by step 4b

## REMOVED FILES (Previously Existed, Now Deleted)

data/raw/external/velocity_dispersions_literature_TRACEABLE.csv
  Status: DELETED — merged into velocity_dispersions_literature.csv

data/raw/external/velocity_dispersions_literature_regenerated.csv
  Status: DELETED — stale auto-generated file with inconsistent values

data/raw/external/velocity_dispersions_literature_detailed.csv
  Status: DELETED — outdated values superseded by master

data/raw/external/sigma_literature_overrides.csv
  Status: DELETED — empty file, never used

scripts/utils/generate_literature_csv_from_traceable.py
  Status: DELETED — no longer needed, master is direct input

scripts/utils/build_sigma_catalog.py
  Status: DELETED — dead code, no callers in pipeline

## VERIFICATION

To verify data integrity, run:
  python scripts/utils/pipeline_audit.py

To verify sigma values against HyperLEDA:
  python scripts/utils/verify_hyperleda.py

## Checksums

File integrity checksums (SHA-256) are in `data/CHECKSUMS.txt`.
These allow independent verification that data files have not been modified.

To verify:
```bash
# Extract just the hash and filename
cut -d' ' -f1,2 data/CHECKSUMS.txt > /tmp/checksums_simple.txt
sha256sum -c /tmp/checksums_simple.txt
```
