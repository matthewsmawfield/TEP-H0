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
| 0 | `step_0_sigma_catalog.py` | Build and validate the velocity-dispersion compilation from literature and catalog sources. |
| 1 | `step_1_data_ingestion.py` | Downloads SH0ES/Pantheon+ data, reconstructs catalogs, matches hosts. |
| 1b | `step_1b_aperture_correction.py` | Fetches RC3 metadata and applies aperture normalization to velocity dispersions. |
| 2 | `step_2_stratification.py` | Calculates H₀, stratifies by σ, and detects environmental bias. |
| 2b | `step_2b_shear_suppression_viz.py` | Generates shear-suppression visualization. |
| 3 | `step_3_tep_correction.py` | Optimizes κ_Cep, applies the TEP correction, and unifies H₀. |
| 4 | `step_4_robustness_checks.py` | Jackknife, Bootstrap, and Peculiar Velocity Monte Carlo tests. |
| 4b | `step_4b_aperture_sensitivity.py` | Tests stability against aperture size and correction parameters. |
| 5 | `step_5_m31_analysis.py` | Analyzes Inner vs Outer Cepheids in M31 (ground-based). |
| 5b | `step_5b_m31_radial_suppression.py` | Continuous and step-function radial suppression models. |
| 6 | `step_6_multivariate_analysis.py` | Controls for Age, Dust, and Stellar Mass confounds. |
| 6e | `step_6_enhanced_robustness.py` | Enhanced robustness tests. |
| 7 | `step_7_lmc_replication.py` | Replicates differential analysis in LMC (null control). |
| 7t | `step_7_trgb_comparison.py` / `step_7_trgb_reanalysis.py` | TRGB differential-insensitivity analysis. |
| 8 | `step_8_m31_phat_analysis.py` | High-resolution HST analysis of M31 Cepheids (PHAT). |
| 9 | `step_9_final_synthesis.py` | Generates final manuscript figures and summary tables. |
| 10 | `step_10_anchor_stratification.py` | Tests for TEP effects in geometric anchors (MW, LMC, NGC 4258). |
| 13 | `step_13_stellar_validation.py` | MESA/RSP/GYRE stellar-structure validation (optional, post-pipeline). |

## Options

```bash
python3 scripts/run_pipeline.py --skip-sigma-step      # Skip sigma catalog rebuild
python3 scripts/run_pipeline.py --rebuild-sigma        # Force rebuild sigma catalog
python3 scripts/run_pipeline.py --use-lit-overrides    # Use literature overrides
python3 scripts/run_pipeline.py --skip-audit           # Skip post-run audit
python3 scripts/run_pipeline.py --run-stellar-validation  # Run Step 13
```

## Audit

After running the pipeline, validate internal consistency:

```bash
python3 scripts/utils/pipeline_audit.py
```

## Outputs

- `results/figures/` — PNG figures referenced in the manuscript.
- `results/outputs/` — JSON/CSV tables (TEP correction, robustness, multivariate, etc.).
- `logs/` — Step-by-step execution logs.
