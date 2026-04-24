# The Cepheid Bias: Resolving the Hubble Tension

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18209702.svg)](https://doi.org/10.5281/zenodo.18209702)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-H0: Cepheid Bias](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.5 (Kingston upon Hull)  
**Date:** 24 April 2026  
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702)  
**Website:** [https://mlsmawfield.com/tep/h0/](https://mlsmawfield.com/tep/h0/)  
**Paper Series:** TEP Series: Paper 11 (Cosmological Observations)

## Abstract

The Hubble Tension—the persistent 5σ discrepancy between local distance-ladder measurements (H₀ ≈ 73 km/s/Mpc) and early-universe CMB inference (H₀ = 67.4 ± 0.5 km/s/Mpc)—represents a significant challenge in precision cosmology. This study proposes that the tension arises from a systematic, environment-dependent bias in Cepheid-based distances, as predicted by the Temporal Equivalence Principle (TEP).

This study tests the hypothesis that the discrepancy arises from a violation of the isochrony axiom—the assumption that proper time accumulation is independent of the local gravitational environment. Under scalar-tensor theories that break the Strong Equivalence Principle (such as TEP), Cepheid variable stars act as environment-dependent "standard clocks." In deep gravitational potentials (high velocity dispersion σ) and active-shear environments, enhanced scalar field activity is predicted to induce period contraction relative to calibration environments. When interpreted through a universal Period-Luminosity relation, this clock-rate anomaly would mimic diminished luminosity, leading to underestimated distances and an inflated local Hubble constant.

Analysis of the SH0ES Cepheid sample (N=29), stratified by host galaxy velocity dispersion (a TEP-independent kinematic observable), reveals a statistically significant correlation between host potential depth and derived H₀ (Spearman ρ = 0.434, p = 0.019; Pearson r = 0.428, p = 0.021). A median-split stratification at σ_med ≈ 90 km/s yields H₀ = 67.82 ± 1.62 km/s/Mpc (low-σ; N=15) versus 72.45 ± 2.32 km/s/Mpc (high-σ; N=14), implying ΔH₀ = 4.63 km/s/Mpc. Because published σ values are heterogeneous (direct stellar absorption and calibrated HI/rotation proxies), measurement methodology is treated as a first-class provenance variable and covariance-aware significance tests are reported using the full SH0ES GLS distance-modulus covariance.

Application of the TEP conformal correction Δμ = α_eff·S(ρ)·(σ² − σ_ref²)/c²—derived from the TEP period-contraction P_obs = P_true(1 − |Φ|/c²)^α_int combined with the virial relation |Φ| ∝ σ²—with optimized effective coupling α_eff = (9.6 ± 4.0) × 10⁵ mag and effective calibrator reference σ_ref = 75.25 km/s yields a unified local Hubble constant of H₀ = 68.37 ± 1.54 km/s/Mpc, corresponding to a Planck tension of 0.60σ. The inferred α_eff ~ 10⁶ places this probe in the same coupling regime as the millisecond-pulsar spin-down excess (Paper 10), eliminating the cross-probe mismatch of earlier phenomenological log₁₀ σ scalings.

Out-of-sample validation (train/test splits and LOOCV) shows that the optimized coupling is stable and removes the residual environmental trend in held-out hosts. A differential analysis within M31 yields an "Inner Fainter" signal in HST photometry. Within the TEP v0.7 framework, this sign is consistent with continuous shear suppression: the high-density M31 bulge experiences progressive attenuation of Temporal Shear (suppression factor S ≈ 0.05 at R < 1 kpc), while the lower-density SN Ia host disks remain in the active-shear regime (⟨S⟩ = 0.946). On this interpretation, the M31 signal marks the empirical mapping of a continuous density-dependent suppression profile across a single galaxy.

The anchor–host mismatch (geometric anchors show near-zero coupling, α_anchor ≈ (7.2 ± 7.9) × 10⁴ mag, in marginal 2.2σ tension with the host-level α ≈ 9.6 × 10⁵ mag) finds a natural resolution in group halo screening: all three anchors (LMC, NGC 4258, M31) are members of galaxy groups, embedding them in deep ambient potentials that trigger chameleon-type screening regardless of internal disk densities. The SN Ia hosts, selected for smooth Hubble flow, are biased toward isolated field galaxies that lack this external screening. This framework generates a falsifiable prediction: the TEP distance-ladder bias should be unique to isolated field galaxies and suppressed in group/cluster environments.

## Key Findings

Analysis of 29 SH0ES Cepheid hosts reveals a significant correlation between derived H₀ and host galaxy velocity dispersion (ρ = 0.434, p = 0.019). High-σ hosts yield H₀ = 72.45 km/s/Mpc while low-σ hosts yield 67.82 km/s/Mpc—a 4.63 km/s/Mpc environmental bias. Applying the physics-derived TEP conformal correction (α ≈ 9.6 × 10⁵ mag with σ² /c² scaling) eliminates this trend, yielding a unified H₀ = 68.37 ± 1.54 km/s/Mpc, reducing Planck tension from 5σ to 0.60σ. A differential analysis within M31 (HST PHAT) detects an "Inner Fainter" signal (+0.68 mag, 3.6σ), explained by density-dependent screening: the high-density bulge is screened while the outer disk is not. The anchor–host mismatch is resolved by group halo screening—all anchors reside in galaxy groups, while SN Ia hosts are biased toward isolated field environments.

---

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Validation of Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Validation of the Temporal Equivalence Principle | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 6** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Unifying Atomic, Galactic, and Compact Object Scales | [10.5281/zenodo.18064366](https://doi.org/10.5281/zenodo.18064366) |
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: A Runaway Black Hole as a Gravitational Soliton | [10.5281/zenodo.18059251](https://doi.org/10.5281/zenodo.18059251) |
| **Paper 8** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Global Time Echoes: Optical Validation of the Temporal Equivalence Principle via Satellite Laser Ranging | [10.5281/zenodo.18064582](https://doi.org/10.5281/zenodo.18064582) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109761](https://doi.org/10.5281/zenodo.18109761) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | **TEP-H0** (This repo) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | The Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | [TEP-WB](https://github.com/matthewsmawfield/TEP-WB) | The Temporal Equivalence Principle: Density-Dependent Screening in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102062](https://doi.org/10.5281/zenodo.19102062) |

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
├── 11manuscript-tep-h0.md         # Full manuscript (Markdown)
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

- `results/outputs/tep_corrected_h0.csv` - Final TEP-corrected Hubble Constant values for all hosts.
- `data/processed/hosts_processed.csv` - Stratified host galaxy data with velocity dispersions.
- `results/outputs/tep_correction_results.json` - Optimized TEP parameters (α, σ_ref) and statistics.
- `results/outputs/aperture_sensitivity_grid.csv` - Sensitivity analysis data for aperture corrections.

## Reproduction Steps

The analysis pipeline is fully automated and reproducible. The master script `scripts/run_pipeline.py` executes the following steps in sequence:

| Manuscript Section | Analysis Step | Script | Description |
|-------------------|---------------|--------|-------------|
| **2.1** | Data Ingestion | `step_1_data_ingestion.py` | Downloads SH0ES/Pantheon+ data, reconstructs catalogs, matches hosts. |
| **2.2** | Aperture Correction | `step_1b_aperture_correction.py` | Fetches RC3 metadata and applies aperture normalization to velocity dispersions. |
| **3.1** | Stratification | `step_2_stratification.py` | Calculates H₀, stratifies by σ, and detects environmental bias. |
| **3.3** | TEP Correction | `step_3_tep_correction.py` | Optimizes TEP coupling α, applies correction, and unifies H₀. |
| **3.6** | Robustness | `step_4_robustness_checks.py` | Performs Jackknife, Bootstrap, and Peculiar Velocity Monte Carlo tests. |
| **4.1** | Sensitivity | `step_4b_aperture_sensitivity.py` | Tests stability against aperture size and correction parameters. |
| **3.8** | M31 Differential | `step_5_m31_analysis.py` | Analyzes Inner vs Outer Cepheids in M31 (Ground-based). |
| **4.2** | Multivariate | `step_6_multivariate_analysis.py` | Controls for Age, Dust, and Stellar Mass confounds. |
| **3.8** | LMC Control | `step_7_lmc_replication.py` | Replicates differential analysis in LMC (Null Control). |
| **4.8** | M31 PHAT | `step_8_m31_phat_analysis.py` | High-resolution HST analysis of M31 Cepheids. |
| **Fig 1-9** | Final Synthesis | `step_9_final_synthesis.py` | Generates final manuscript figures and summary tables. |
| **3.5** | Anchor Test | `step_10_anchor_stratification.py` | Tests for TEP effects in geometric anchors (MW, LMC, N4258). |

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
- **Velocity dispersions:** compiled/regenerated from public catalogs and literature (HyperLEDA, Ho+2009, SDSS; see `data/raw/external/velocity_dispersions_literature*.csv`).
- **M31 HST Cepheids:** VizieR catalog `J/ApJ/864/59` (Kodric et al. 2018), summarized in `results/outputs/m31_phat_robustness_summary.json`.

### Interpretation scope

The analysis demonstrates a statistically significant host-level H₀–σ correlation in the SH0ES host set and shows that a TEP-motivated conformal correction yields a Planck-consistent unified value within uncertainties. The **anchor consistency test** (LMC, NGC 4258, M31) yields α_anchor ≈ 0 and is in tension with the host-level coupling; this **anchors-vs-hosts mismatch is treated as an open question** and is explicitly discussed as such.

## Citation

```bibtex
@article{smawfield2026cepheidbias,
  title={The Cepheid Bias: Resolving the Hubble Tension},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18209702},
  note={Preprint v0.5 (Kingston upon Hull)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthewsmawfield@gmail.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
