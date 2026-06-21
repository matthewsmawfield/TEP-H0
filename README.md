# The Cepheid Bias: Resolving the Hubble Tension

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18209702.svg)](https://doi.org/10.5281/zenodo.18209702)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-H0: Cepheid Bias](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.7 (Kingston upon Hull)  
**Date:** First published: 11 January 2026 · Last updated: 21 June 2026  
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702)  
**Website:** [https://mlsmawfield.com/tep/h0/](https://mlsmawfield.com/tep/h0/)  
**Paper Series:** TEP Series: Paper 11 (Cosmological Observations)

## Abstract

The Hubble Tension—the persistent 5σ discrepancy between local distance-ladder measurements (H₀ ≈ 73 km/s/Mpc) and early-universe CMB inference (H₀ = 67.4 ± 0.5 km/s/Mpc)—represents a significant challenge in precision cosmology. This paper tests whether a component of the Hubble tension can be represented as an environment-dependent Cepheid clock bias, as predicted by the Temporal Equivalence Principle (TEP).

This study tests the hypothesis that the discrepancy arises from a violation of the isochrony axiom—the assumption that proper time accumulation is independent of the local gravitational environment. Under scalar-tensor theories that break the Strong Equivalence Principle (such as TEP), Cepheid variable stars act as environment-dependent "standard clocks." In deep gravitational potentials (high velocity dispersion σ) and active-shear environments, enhanced scalar field activity is predicted to induce period contraction relative to calibration environments. When interpreted through a universal Period-Luminosity relation, this clock-rate anomaly would mimic diminished luminosity, leading to underestimated distances and an inflated local Hubble constant.

The standard SH0ES full-ladder likelihood yields a baseline H₀ = 73.04 ± 1.01 km/s/Mpc. A standard-ladder projection test inserts a TEP environmental column into the SH0ES design matrix while keeping the host-level distance moduli μᵢ as free latent parameters; the environmental signal is absorbed by the inferred μᵢ, yielding κ_Cep = -0.067 ± 0.210×10⁶ mag (consistent with zero). This null result is expected: the standard SH0ES model is not TEP-native, and any host-constant environmental bias is algebraically equivalent to shifting a host's inferred modulus.

The proper test must therefore be applied at the redshift-distance level, where distances are tied to an independent velocity scale. A velocity-space likelihood analysis models czᵢ = dᵢ^true (H_app + Γ_X Xᵢ) + vᵢ and identifies a structurally robust combined environmental slope Γ_X = +2.35×10⁷ ± 1.00×10⁷ (2.3σ at σ_v = 250 km/s; 3.4σ at σ_v = 150 km/s). The signal survives explicit controls for redshift trend, sky dipole (~100 km/s), quadrupole, and group-offset models; binned permutation tests confirm it is not driven by redshift or sky selection. Leave-one-host-out cross-validation gives 29/29 positive signs; bootstrap resampling gives 99.9% positive fraction.

The velocity-space likelihood identifies a combined environmental slope. In a general phenomenological model this slope can contain both Cepheid clock bias and residual velocity-sector/environmental terms. The TEP-native gauge sets the non-Cepheid velocity-sector component β_X to zero, corresponding to the hypothesis tested here: the observed environmental slope is dominantly a Cepheid clock-transport bias. In that gauge—treating the environmental slope as a pure Cepheid clock-rate bias (β_X = 0)—the equivalent response coefficient is κ_Cep ≈ 7.34×10⁵ mag, consistent with the canonical TEP parameter κ_gal = 9.7×10⁵ mag (~10⁶). The velocity-space fit yields H_app = 69.47 ± 1.49 km/s/Mpc; in the TEP-native gauge this corresponds to a Cepheid clock-bias correction that brings the local distance scale into agreement with the CMB inference. As a historical residual-space cross-check, the empirical one-parameter correction pipeline yields H₀^TEP = 68.84 km/s/Mpc (bootstrap mean 68.92 ± 1.44), reducing the Hubble tension from ≈5σ to ≈1σ relative to Planck. This value is obtained in the pure-Cepheid TEP-native gauge (β_X=0); the gauge-independent empirical detection is Γ_X. A residual-based empirical cross-check gives κ_Cep = (1.27 ± 0.46) × 10⁶ mag, consistent with the generative inference. The inferred coefficient places this probe in the same response-coefficient regime as the millisecond-pulsar spin-down excess (Paper 10). External TRGB distances (N=13 overlap) give κ_Cep = +3.2×10⁵ (0.8σ)—underpowered to independently break the κ–β degeneracy, yet directionally consistent with and supportive of the TEP-native gauge.

A differential M31 analysis yields an "Inner Fainter" signal consistent with TEP shear suppression, providing auxiliary support for the continuous screening mechanism.

## Key Findings

Analysis of 29 SH0ES Cepheid hosts reveals a correlation between derived H₀ and host galaxy velocity dispersion (ρ = 0.517, p = 0.0041; Pearson r = 0.466, p = 0.0109). A velocity-space likelihood analysis identifies a robust combined environmental slope Γ_X ≈ +2.35×10⁷ (2.3σ). In the TEP-native pure-Cepheid gauge (β_X = 0), this corresponds to an equivalent Cepheid response coefficient κ_equiv ≈ 7.34×10⁵ mag, consistent with the canonical TEP parameter κ_gal = 9.7×10⁵ mag. The empirical one-parameter residual-space correction gives κ_Cep^emp ≈ (1.27 ± 0.46) × 10⁶ mag and H₀^TEP = 68.84 km/s/Mpc, reducing the Planck tension from ≈5σ to ≈1σ. A standard-ladder projection test demonstrates that environmental signal is absorbed by free latent host moduli in the non-native SH0ES gauge, validating the need for the generative-observable correction.

---

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Consistency Test | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Synthesis | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 6** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Cross-Scale Consistency of ρ_T | [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) |
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate | [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) |
| **Paper 8** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging | [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | **TEP-H0** (This repo) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | The Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | [TEP-WB](https://github.com/matthewsmawfield/TEP-WB) | The Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) |
| **Paper 15** | [TEP-EFA](https://github.com/matthewsmawfield/TEP-EFA) | Temporal Equivalence Principle: Temporal Shear in the Earth Flyby Anomaly | [10.5281/zenodo.19454863](https://doi.org/10.5281/zenodo.19454863) |
| **Paper 16** | [TEP-J0437](https://github.com/matthewsmawfield/TEP-J0437) | Synchronization Holonomy in Pulsar Scintillation | [10.5281/zenodo.19454620](https://doi.org/10.5281/zenodo.19454620) |
| **Paper 17** | [TEP-LLR](https://github.com/matthewsmawfield/TEP-LLR) | Lunar Laser Ranging and the Nordtvedt Effect | [10.5281/zenodo.19446029](https://doi.org/10.5281/zenodo.19446029) |

## Directory Structure

```
TEP-H0/
├── data/                          # Raw and processed data
│   ├── raw/                       # Original SH0ES/Gaia datasets
│   ├── processed/                 # Stratified and enriched host data
│   └── interim/                   # Intermediate calculation files
├── scripts/
│   ├── steps/                     # Analysis pipeline steps (1-10)
│   ├── utils/                     # Shared utility functions
│   └── analysis/                  # Exploratory analysis scripts
├── results/
│   ├── outputs/                   # Analysis results (JSON, CSV, Tables)
│   └── figures/                   # Generated plots (PNG)
├── site/
│   ├── components/                # Manuscript HTML sections
│   └── public/                    # Static assets
├── 11-TEP-H0-v0.7-KingstonUponHull.md  # Full manuscript (Markdown)
└── requirements.txt               # Python dependencies
```

## Installation

```bash
# Clone repository
git clone https://github.com/matthewsmawfield/TEP-H0.git
cd TEP-H0

# Install dependencies
pip install -r requirements.txt
```

## Essential Data Files

- `results/outputs/step_04_tep_corrected_h0.csv` - Final TEP-corrected Hubble Constant values for all hosts (empirical residual-space correction).
- `data/processed/hosts_processed.csv` - Stratified host galaxy data with velocity dispersions.
- `results/outputs/step_04_tep_correction_results.json` - Optimized TEP response coefficient (κ_Cep^emp), σ_ref, and statistics.
- `results/outputs/step_39_environment_slope_decomposition.json` - Velocity-space environmental slope Γ_X and TEP-native equivalent coefficient κ_equiv.
- `results/outputs/step_34_full_ladder_likelihood_results.json` - Standard-ladder projection test showing absorption of environmental signal by latent host moduli.
- `results/outputs/step_43_toy_recovery_experiment.json` - Toy recovery experiment validating gauge absorption and native TEP detection.
- `results/outputs/step_07_aperture_sensitivity_grid.csv` - Sensitivity analysis data for aperture corrections.

## Reproduction Steps

The analysis pipeline is fully automated and reproducible. The master script `scripts/run_pipeline.py` executes the following steps in sequence:

| Manuscript Section | Analysis Step | Script | Description |
|-------------------|---------------|--------|-------------|
| **2.1** | Data Ingestion | `step_1_data_ingestion.py` | Downloads SH0ES/Pantheon+ data, reconstructs catalogs, matches hosts. |
| **2.2** | Aperture Correction | `step_1b_aperture_correction.py` | Fetches RC3 metadata and applies aperture normalization to velocity dispersions. |
| **3.1** | Stratification | `step_2_stratification.py` | Calculates H₀, stratifies by σ, and detects environmental bias. |
| **3.3** | TEP Correction | `step_3_tep_correction.py` | Optimizes κ_Cep, applies the TEP correction, and unifies H₀. |
| **3.6** | Robustness | `step_4_robustness_checks.py` | Performs Jackknife, Bootstrap, and Peculiar Velocity Monte Carlo tests. |
| **4.1** | Sensitivity | `step_4b_aperture_sensitivity.py` | Tests stability against aperture size and correction parameters. |
| **3.8** | M31 Differential | `step_5_m31_analysis.py` | Analyzes Inner vs Outer Cepheids in M31 (Ground-based). |
| **4.2** | Multivariate | `step_6_multivariate_analysis.py` | Controls for Age, Dust, and Stellar Mass confounds. |
| **3.8** | LMC Control | `step_7_lmc_replication.py` | Replicates differential analysis in LMC (Null Control). |
| **4.8** | M31 PHAT | `step_8_m31_phat_analysis.py` | High-resolution HST analysis of M31 Cepheids. |
| **Fig 1-9** | Final Synthesis | `step_9_final_synthesis.py` | Generates final manuscript figures and summary tables. |
| **3.5** | Anchor Test | `step_10_anchor_stratification.py` | Tests for TEP effects in geometric anchors (MW, LMC, NGC 4258). |

### Running the Pipeline

To reproduce all results and figures:

```bash
python3 scripts/run_pipeline.py
```

This will populate `results/figures/` and `results/outputs/` with fresh data.

## Audit & Reproducibility

To support careful external scrutiny, this repository includes machine-checkable audit artifacts alongside the paper outputs:

- **Pipeline audit (sanity checks):**
  - Run: `python3 scripts/utils/pipeline_audit.py`
  - Purpose: validates internal consistency between key CSV/JSON outputs and critical manuscript values.
- **Primary audit reports:**
  - `AUDIT_REPORT_GENERATED.md`
  - `DEEP_AUDIT_REPORT.md`
  - `DEEP_AUDIT_LOGIC_REPORT.md`
  - `results/outputs/TEP_FINAL_ROBUSTNESS_REPORT.md`

### Data provenance (non-exhaustive)

- **Hubble-flow SN / redshifts:** `data/raw/Pantheon+SH0ES.dat` (Pantheon+SH0ES release).
- **SH0ES host distance moduli:** reconstructed from SH0ES R22 inputs (see `data/raw/external/Cepheid-Distance-Ladder-Data/SH0ES2022/`).
- **Velocity dispersions:** manually curated from peer-reviewed literature with ADS bibcodes (see `data/raw/external/velocity_dispersions_literature.csv`). This is the single master file — the only source of sigma data used by the pipeline.
- **M31 HST Cepheids:** VizieR catalog `J/ApJ/864/59` (Kodric et al. 2018), summarized in `results/outputs/m31_phat_robustness_summary.json`.

### Interpretation scope

The analysis demonstrates a statistically significant host-level H₀–σ correlation in the SH0ES host set and shows that a TEP-motivated conformal correction yields a Planck-consistent unified value within uncertainties. The **anchor consistency test** (LMC, NGC 4258, M31) yields κ_anchor ≈ 0 in the σ²/c² convention and is interpreted through group-halo screening, producing a direct environmental prediction for future field-versus-group distance-ladder tests.

## Citation

```bibtex
@article{smawfield2026cepheidbias,
  title={The Cepheid Bias: Resolving the Hubble Tension},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18209702},
  note={Preprint v0.7 (Kingston upon Hull)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
