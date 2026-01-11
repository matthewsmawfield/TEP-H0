# The Cepheid Bias: Resolving the Hubble Tension

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18209703.svg)](https://doi.org/10.5281/zenodo.18209703)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-H0: Cepheid Bias](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.2 (Kingston upon Hull)  
**Date:** 11 January 2026  
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.18209703](https://doi.org/10.5281/zenodo.18209703)  
**Website:** [https://mlsmawfield.com/tep/h0/](https://mlsmawfield.com/tep/h0/)  
**Paper Series:** TEP Series: Paper 12 (Cosmological Observations)

## Abstract

    The Hubble Tension—the persistent 5σ discrepancy between local distance-ladder measurements ($H_0 \approx 73$ km/s/Mpc) and early-universe CMB inference ($H_0 = 67.4 \pm 0.5$ km/s/Mpc)—represents a significant challenge in precision cosmology. This study proposes that the tension arises from a systematic, environment-dependent bias in Cepheid-based distances, as predicted by the Temporal Equivalence Principle (TEP).

    TEP posits that proper time accumulation depends on the local gravitational potential. Cepheid variable stars, acting as "standard clocks" via their period-luminosity relation, experience differential time flow governed by their host galaxy environment. Consistent with the anomalous spin-down rates observed in globular cluster pulsars (Paper 11), Cepheids in deep gravitational potentials (high velocity dispersion $\sigma$) experience period contraction relative to calibration environments. When interpreted through a universal P-L relation, this period deficit masquerades as diminished luminosity, causing systematic underestimation of distances to SN Ia host galaxies and consequent overestimation of $H_0$.

    Analysis of the SH0ES Cepheid sample ($N=29$), stratified by host galaxy velocity dispersion (a TEP-independent kinematic observable), reveals a statistically significant correlation between host potential depth and derived $H_0$ (Spearman $\rho = 0.434$, $p = 0.019$; Pearson $r = 0.428$, $p = 0.021$). A median-split stratification at $\sigma_{\rm med} \approx 90$ km/s yields $H_0 = 67.82 \pm 1.62$ km/s/Mpc (low-$\sigma$; $N=15$) versus $72.45 \pm 2.32$ km/s/Mpc (high-$\sigma$; $N=14$), implying $\Delta H_0 = 4.63$ km/s/Mpc. Because published $\sigma$ values are heterogeneous (direct stellar absorption and calibrated HI/rotation proxies), we treat measurement methodology as a first-class provenance variable and report covariance-aware significance tests using the full SH0ES GLS distance-modulus covariance.

    Application of the TEP conformal correction with an optimized coupling $\alpha = 0.58 \pm 0.16$ and effective calibrator reference $\sigma_{\rm ref} = 75.25$ km/s yields a unified local Hubble constant of $H_0 = 68.66 \pm 1.51$ km/s/Mpc, corresponding to a Planck tension of $0.79\sigma$. Out-of-sample validation (train/test splits and LOOCV) shows that the optimized coupling is stable and removes the residual environmental trend in held-out hosts. A differential analysis within M31 yields an “Inner Fainter” signal in HST photometry. This result, initially counter-intuitive, is resolved by the theory's density-dependent screening mechanism: the high-density M31 bulge ($\rho > \rho_{\rm trans}$) is screened (restoring standard physics), while the lower-density SN Ia host disks remain unscreened. The M31 signal thus marks the empirical crossing of the screening threshold.

## Key Results

- **Potential-Dependent H0:** Significant correlation (Spearman $\rho=0.434$, $p=0.019$) between derived $H_0$ and host velocity dispersion $\sigma$.
- **Bias Correction:** Application of TEP conformal correction eliminates environmental dependence.
- **Unified Value:** Corrected local **$H_0 = 68.66 \pm 1.51$ km/s/Mpc**, in agreement with Planck 2018 ($0.8\sigma$ tension).
- **Robustness:** Signal persists after controlling for metallicity, age (period), and dust (color).
- **Mechanism Check:** Differential analysis of M31 Cepheids (Inner vs Outer) shows "Inner Fainter" signal consistent with TEP screening.

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Temporal Evolution | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Validation | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 6** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Validation of TEP | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 7** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density | [10.5281/zenodo.18064366](https://doi.org/10.5281/zenodo.18064366) |
| **Paper 8** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake | [10.5281/zenodo.18059251](https://doi.org/10.5281/zenodo.18059251) |
| **Paper 9** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Satellite Laser Ranging Validation | [10.5281/zenodo.18064582](https://doi.org/10.5281/zenodo.18064582) |
| **Paper 10** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109761](https://doi.org/10.5281/zenodo.18109761) |
| **Paper 11** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 12** | **TEP-H0** (This repo) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209703](https://doi.org/10.5281/zenodo.18209703) |

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
├── 12manuscript-tep-h0.md         # Full manuscript (Markdown)
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
- `results/outputs/tep_correction_results.json` - Optimized TEP parameters ($\alpha$, $\sigma_{\rm ref}$) and statistics.
- `results/outputs/aperture_sensitivity_grid.csv` - Sensitivity analysis data for aperture corrections.

## Reproduction Steps

The analysis pipeline is fully automated and reproducible. The master script `scripts/run_pipeline.py` executes the following steps in sequence:

| Manuscript Section | Analysis Step | Script | Description |
|-------------------|---------------|--------|-------------|
| **2.1** | Data Ingestion | `step_1_data_ingestion.py` | Downloads SH0ES/Pantheon+ data, reconstructs catalogs, matches hosts. |
| **2.2** | Aperture Correction | `step_1b_aperture_correction.py` | Fetches RC3 metadata and applies aperture normalization to velocity dispersions. |
| **3.1** | Stratification | `step_2_stratification.py` | Calculates $H_0$, stratifies by $\sigma$, and detects environmental bias. |
| **3.3** | TEP Correction | `step_3_tep_correction.py` | Optimizes TEP coupling $\alpha$, applies correction, and unifies $H_0$. |
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

The analysis demonstrates a statistically significant host-level $H_0$–$\sigma$ correlation in the SH0ES host set and shows that a TEP-motivated conformal correction yields a Planck-consistent unified value within uncertainties. The **anchor consistency test** (LMC, NGC 4258, M31) yields $\alpha_{\rm anchor}\approx 0$ and is in tension with the host-level coupling; this **anchors-vs-hosts mismatch is treated as an open question** and is explicitly discussed as such.

## Citation

```bibtex
@article{smawfield2026h0,
  title={The Cepheid Bias: Resolving the Hubble Tension},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18209703},
  note={Preprint v0.2 (Kingston upon Hull)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthewsmawfield@gmail.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
