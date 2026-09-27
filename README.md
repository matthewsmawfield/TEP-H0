# The Cepheid Bias: Resolving the Hubble Tension

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18209702.svg)](https://doi.org/10.5281/zenodo.18209702)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-H0: Cepheid Bias](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.10 (Kingston upon Hull)  
**First published:** 11 January 2026 · **Last updated:** 16 September 2026
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702)  
**Website:** [https://mlsmawfield.com/tep/h0/](https://mlsmawfield.com/tep/h0/)  
**Paper Series:** TEP Series: Paper 11 (Cosmological Observations)

## Abstract

The Hubble Tension—the persistent $5\sigma$ discrepancy between local distance-ladder measurements ($H_0 \approx 73.0\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and early-universe CMB inference ($H_0 = 67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$)—represents a significant challenge in precision cosmology. This paper tests whether a component of the Hubble tension can be represented as an environment-dependent Cepheid clock bias, as predicted by the Temporal Equivalence Principle (TEP).

The hypothesis tested here is that Cepheid variable stars function as environment-dependent "standard clocks." Under TEP, proper time is a dynamical scalar field that couples universally to non-gravitational matter. Because stellar pulsation rates operate in local proper time, calibrating classical Cepheids in deep-potential SN Ia host galaxies against diffuse, low-mass anchor galaxies systematically misestimates host distance moduli. When interpreted through a universal Period--Luminosity relation, this clock-rate anomaly mimics diminished luminosity, leading to underestimated distances and an inflated local Hubble constant.

In standard distance-ladder linear regression, unconstrained latent host distance moduli $\mu_i$ algebraically absorb host-level environmental shifts. A standard-ladder projection test that inserts a TEP environmental column into the SH0ES design matrix therefore yields an apparent null result. This structural degeneracy is resolved by formulating the calibrator-host distance ladder as a generative clock-aware model coupled to the Hubble flow.

This generative framework is tested using the complete public Riess et al. (2022) sample of 37 distinct SN Ia host galaxies, utilizing a homogeneous kinematic potential coordinate $u_\phi=V_{\rm rot}/\sqrt{2}$ derived from pinned HyperLEDA rotation velocities and continuous environmental screening. Across all 37 hosts, the generative endpoint likelihood yields $\Gamma_X=(1.156\pm0.958)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical $250\ {\rm km/s}$ velocity variance, strengthening to $1.64\sigma$ under data-driven Pantheon+ velocity scatter ($182.1\ {\rm km/s}$) and $1.97\sigma$ at $150\ {\rm km/s}$, with 100% leave-one-host-out sign stability (37 of 37 refits positive). In the cosmologically constrained host-level expansion-rate likelihood across the 33 Hubble-flow hosts (Step 42), the combined endpoint response is recovered at $\Gamma_X=(1.165\pm0.979)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$; $2.07\sigma$ at $150\ {\rm km/s}$; and $1.99\sigma$ under full $33\times33$ SH0ES covariance GLS), yielding a conventional Hubble-flow intercept of $H_{\rm app}=68.36\pm0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ to $68.62\pm1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$. When evaluated in a joint multi-block framework with independent TRGB distances and geometric anchors (Step 44), the multi-block likelihood consistently favours an environmental response under either single-parameter allocation ($\kappa_{\rm Cep} = (0.400 \pm 0.270)\times 10^6\ {\rm mag}$ or $\beta_X = (1.631 \pm 1.250)\times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$), resolving the local Hubble constant to $H_{\rm app} = 68.76\pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$. Under the restricted TEP Cepheid-channel closure ($\beta_X=0$), the full response is allocated to the Cepheid channel ($\kappa_{\rm Cep}^{\rm equiv} = (0.369 \pm 0.310)\times 10^6\ {\rm mag}$); this allocation is physically specified but remains conditional on host-specific aperture validation.

Propagating the TEP potential correction through the distance ladder yields a unified local Hubble constant of $H_0=66.65\pm1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, consistent with Planck CMB observations at $0.45\sigma$ ($0.31\sigma$ bootstrap); the TEP-CMB inference ($H_0 = 66.70 \pm 0.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$) agrees with the TEP-corrected local value at $0.03\sigma$. Full-matrix propagation through the 3,490-row SH0ES system confirms exact reference-gauge invariance. Finally, single-galaxy differential tests provide internal signatures with the sign predicted by TEP free from host-to-host peculiar velocity systematics: M31 inner versus outer Cepheids exhibit an empirical Period--Luminosity offset of $+0.356 \pm 0.136\ {\rm mag}$ ($2.6\sigma$), increasing to $+0.630 \pm 0.195\ {\rm mag}$ ($3.24\sigma$) under spatial PHAT matching; however, M31 fails matched controls (colour-matched $-0.017 \pm 0.128$ mag, $0.1\sigma$; eW-matched $-0.103 \pm 0.129$ mag, $0.8\sigma$), so the M31 signal is treated as a debug signal, not evidence. The LMC gradient ($+0.0284 \pm 0.0086$ mag, $3.3\sigma$) is robust to period and colour matching and remains the strongest single-galaxy result.

## Key Findings

- **M31 internal Cepheids (Pan-STARRS):** $\Delta W = +0.356 \pm 0.136$ mag ($2.6\sigma$).
- **M31 HST PHAT footprint:** $\Delta W = +0.630 \pm 0.195$ mag ($3.24\sigma$).
- **LMC OGLE-IV Cepheids (radial stratification):** $\Delta W = +0.0284 \pm 0.0086$ mag ($3.3\sigma$).
- **37-host endpoint likelihood (canonical 250 km/s):** $\Gamma_X = (1.156 \pm 0.958) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$; $100\%$ positive leave-one-host-out stability (37 of 37 refits).
- **Flow-velocity sensitivity:** $1.97\sigma$ at 150 km/s; $1.64\sigma$ at data-driven Pantheon+ residual scatter $182.1$ km/s; $0.54\sigma$ at 500 km/s.
- **Cosmologically constrained expansion likelihood (Step 42, 33 Hubble-flow hosts):** $\Gamma_X = (1.165 \pm 0.979) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$ at $\sigma_v=250$; $2.07\sigma$ at $\sigma_v=150$; $1.99\sigma$ under full $33\times33$ SH0ES covariance GLS), $H_{\rm app} = 68.36 \pm 0.98$ to $68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$.
- **Joint multi-block likelihood (Step 44):** Consistently favours environmental response; $\kappa_{\rm Cep} = (0.400 \pm 0.270) \times 10^6\ {\rm mag}$ or $\beta_X = (1.631 \pm 1.250) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$; $H_{\rm app} = 68.76 \pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$.
- **Unified host-level reconstruction (Step 04):** $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, $0.45\sigma$ from Planck ($0.31\sigma$ bootstrap); $0.03\sigma$ from TEP-CMB ($H_0 = 66.70 \pm 0.58$).
- **Full-matrix SH0ES propagation:** exact reference-gauge invariance through the 3,490-row system.

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
| **Paper 15** | [TEP-EFA](https://github.com/matthewsmawfield/TEP-EFA) | Temporal Equivalence Principle: Temporal Shear in the Earth Flyby Anomaly | [10.5281/zenodo.19454862](https://doi.org/10.5281/zenodo.19454862) |
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
│   ├── steps/                     # Analysis pipeline steps
│   ├── utils/                     # Shared utility functions
│   └── analysis/                  # Exploratory analysis scripts
├── results/
│   ├── outputs/                   # Analysis results (JSON, CSV, Tables)
│   └── figures/                   # Generated plots (PNG)
├── site/
│   ├── components/                # Manuscript HTML sections
│   └── public/                    # Static assets
├── 11-TEP-H0-v0.10-KingstonUponHull.md  # Full manuscript (Markdown)
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

- `results/outputs/step_04_tep_corrected_h0.csv` - Per-host residual-flattening diagnostic.
- `data/processed/hosts_processed.csv` - Stratified host galaxy data with velocity dispersions.
- `results/outputs/step_04_tep_correction_results.json` - Optimized TEP response coefficient ($\kappa_{\rm Cep}^{\rm emp}$), $\sigma_{\rm ref}$, and Planck-tension metrics.
- `results/outputs/step_39_environment_slope_decomposition.json` - Redshift-distance environmental slope $\Gamma_X$ and Cepheid-dominant equivalent coefficient $\kappa_{\rm equiv}$.
- `results/outputs/step_42_tep_native_ladder.json` - TEP-native generative distance-ladder detection.
- `results/outputs/step_34_full_ladder_likelihood_results.json` - Standard-ladder projection test demonstrating absorption of environmental signal by latent host moduli.
- `results/outputs/step_43_toy_recovery_experiment.json` - Toy recovery experiment validating gauge absorption and native TEP detection.
- `results/outputs/step_07_aperture_sensitivity_grid.csv` - Sensitivity analysis data for aperture corrections.

## Reproduction Steps

The authoritative pipeline rebuilds the input tables, descriptive diagnostics,
primary endpoint likelihood, flow/sky controls, SH0ES matrix fit, conditional
full-ladder projection, and regression gates.

```bash
python3 scripts/run_pipeline.py
```

The run ends with `scripts/utils/pipeline_audit.py`, after the current
inferential artifacts have been written. Historical Steps 28--31 and 49, plus
the gauge-failing Step 50 summary, are excluded from the authoritative path.
See `scripts/README.md` for the step-level status dictionary.


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

The analysis identifies a coherent multi-scale TEP signal: internal galaxy
Period--Luminosity offsets in M31 and the LMC, a homogeneous 37-host
environmental endpoint, and cosmologically constrained expansion
likelihoods ($H_{\rm app} = 68.36$–$68.76\ {\rm km\,s^{-1}\,Mpc^{-1}}$). The unified TEP correction reconciles the local distance scale
with Planck CMB cosmology without invoking early-universe modifications.
Local-gravity closure, pulsar coefficient matching, and the gauge-failing Step
50 summary are not inferential results of this repository.

## Citation

```bibtex
@article{smawfield2026cepheidbias,
  title={The Cepheid Bias: Resolving the Hubble Tension},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18209702},
  note={Preprint v0.10 (Kingston upon Hull)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
