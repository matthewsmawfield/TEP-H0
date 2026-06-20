# TEP-H0 Analysis Pipeline

Entry point for reproducing all results and figures in Paper 11: *The Cepheid Bias: Resolving the Hubble Tension*.

## Quick Start

```bash
cd /Users/matthewsmawfield/www/Temporal\ Equivalence\ Principle/TEP-H0
python3 scripts/run_pipeline.py
```

This executes the full pipeline and populates `results/figures/` and `results/outputs/` with fresh data.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 00 | `step_00_sigma_catalog.py` | Build and validate the velocity-dispersion compilation from literature and catalog sources. |
| 01 | `step_01_data_ingestion.py` | Downloads SH0ES/Pantheon+ data, reconstructs catalogs, matches hosts. |
| 02 | `step_02_aperture_correction.py` | Fetches RC3 metadata and applies aperture normalization to velocity dispersions. |
| 03 | `step_03_stratification.py` | Calculates H₀, stratifies by σ, and detects environmental bias. |
| 04 | `step_04_tep_correction.py` | Optimizes κ_Cep, applies the TEP correction, and unifies H₀. |
| 05 | `step_05_frozen_predictions.py` | Generates frozen falsification-ready prediction table for prospective hosts. |
| 06 | `step_06_shear_suppression_viz.py` | Generates shear-suppression visualization. |
| 07 | `step_07_aperture_sensitivity.py` | Tests stability against aperture size and correction parameters. |
| 08 | `step_08_robustness_checks.py` | Jackknife, Bootstrap, and Peculiar Velocity Monte Carlo tests. |
| 09 | `step_09_hierarchical_sigma.py` | Hierarchical measurement-error model for σ (method-specific bias & scatter; ODR slope). |
| 10 | `step_10_m31_analysis.py` | Analyzes Inner vs Outer Cepheids in M31 (ground-based). |
| 11 | `step_11_m31_radial_suppression.py` | Continuous and step-function radial suppression models. |
| 12 | `step_12_multivariate_analysis.py` | Controls for Age, Dust, and Stellar Mass confounds. |
| 13 | `step_13_enhanced_robustness.py` | Enhanced robustness tests. |
| 14 | `step_14_lmc_replication.py` | Replicates differential analysis in LMC (null control). |
| 15 | `step_15_trgb_comparison.py` | TRGB differential-insensitivity analysis. |
| 16 | `step_16_trgb_reanalysis.py` | TRGB differential-insensitivity reanalysis. |
| 17 | `step_17_host_mass_residual.py` | Host-mass residual test: isolates TEP-specific signal from shared systematics (Cepheid vs TRGB). |
| 18 | `step_18_regressor_audit.py` | Primary TEP regressor audit: compares σ, σ², S_local·σ², S_total·σ², confounds, and null controls. |
| 19 | `step_19_group_env_models.py` | Group environment model comparison: tests whether N_mb is a confound or a TEP screening mechanism. |
| 20 | `step_20_joint_indicator_model.py` | Joint Cepheid+TRGB indicator model: separates common host systematics from indicator-specific clock bias. |
| 21 | `step_21_stratified_validation.py` | Physically stratified validation: train on one physical regime, test on another. |
| 22 | `step_22_exact_sigma_ref.py` | Exact anchor-leverage σ_ref reconstruction from multiple weighting schemes. |
| 23 | `step_23_sn_residual_test.py` | SN Ia downstream residual test: does TEP correction remove σ dependence in corrected H0? |
| 24 | `step_24_synthetic_injection.py` | Synthetic signal injection and recovery test. |
| 25 | `step_25_leave_one_out.py` | Leave-One-Out influence analysis. |
| 26 | `step_26_m31_phat_analysis.py` | High-resolution HST analysis of M31 Cepheids (PHAT). |
| 27 | `step_27_anchor_stratification.py` | Tests for TEP effects in geometric anchors (MW, LMC, NGC 4258). |
| 28 | `step_28_local_gravity_closure.py` | Converts fitted Cepheid response into explicit local source-charge prediction. |
| 29 | `step_29_cross_channel.py` | Cross-channel consistency test (Cepheid + TRGB + pulsar). |
| 30 | `step_30_cosmology_inference.py` | Cosmological inference models and predictions. |
| 31 | `step_31_final_synthesis.py` | Generates final manuscript figures and summary tables. |
| 32 | `step_32_comprehensive_audit.py` | Comprehensive audit: sample consistency, headline recomputation, covariance, provenance, ODR, multiple-testing. |
| 32b| `scripts/utils/pipeline_audit.py` | Lightweight pipeline self-check (legacy). |
| 33 | `step_33_stellar_validation.py` | MESA/RSP/GYRE stellar-structure validation (optional, post-pipeline). |

## Options

```bash
python3 scripts/run_pipeline.py --skip-sigma-step      # Skip sigma catalog rebuild
python3 scripts/run_pipeline.py --rebuild-sigma        # Force rebuild sigma catalog
python3 scripts/run_pipeline.py --use-lit-overrides    # Use literature overrides
python3 scripts/run_pipeline.py --skip-audit           # Skip post-run audit
python3 scripts/run_pipeline.py --run-stellar-validation  # Run Step 13
```

## Folder Structure

```
scripts/
  run_pipeline.py          # Master orchestrator — single entry point for all results
  steps/                   # Formal pipeline steps (numbered, ordered, reproducible)
    step_0_sigma_catalog.py
    step_1_data_ingestion.py
    ...
    step_15_hierarchical_sigma.py
  utils/                   # Shared utilities (tep_correction, logger, plot_style, etc.)
  diagnostics/             # Post-hoc diagnostic scripts (NOT part of reproducible pipeline)
    analyze_redshift_cuts.py
    check_astrophysical_bias.py
    check_n4258_density.py
    ...
```

**Rule:** Every number quoted in the manuscript must trace to an output produced by
`scripts/run_pipeline.py`. The `diagnostics/` folder contains standalone exploratory
scripts; they are not guaranteed to be reproducible or maintained.

## Audit

After running the pipeline, validate internal consistency:

```bash
python3 scripts/utils/pipeline_audit.py
```

## Outputs

- `results/figures/` — PNG figures referenced in the manuscript.
- `results/outputs/` — JSON/CSV tables (TEP correction, robustness, multivariate, etc.).
- `logs/` — Step-by-step execution logs.
