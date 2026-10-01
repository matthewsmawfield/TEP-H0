# The Cepheid Bias: Resolving the Hubble Tension

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18209702.svg)](https://doi.org/10.5281/zenodo.18209702)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-H0: Cepheid Bias](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.11 (Kingston upon Hull)  
**First published:** 11 January 2026 · **Last updated:** 1 October 2026
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702)  
**Website:** [https://mlsmawfield.com/tep/h0/](https://mlsmawfield.com/tep/h0/)  
**Paper Series:** TEP Series: Paper 11 (Cosmological Observations)

## Abstract






The Hubble Tension—the persistent 5σ discrepancy between local distance-ladder measurements (H_0 ≈ 73.0 km s⁻¹ Mpc⁻¹) and early-universe CMB inference (H_0 = 67.4 ± 0.5 km s⁻¹ Mpc⁻¹)—represents a significant challenge in precision cosmology. This paper demonstrates that the discrepancy is resolved by an environment-dependent Cepheid clock bias, as predicted by the Temporal Equivalence Principle (TEP). The resolving mechanism is environmental clock calibration: Cepheid variable stars function as environment-dependent "standard clocks." Under TEP, proper time is a dynamical scalar field that couples universally to non-gravitational matter. Because stellar pulsation rates operate in local proper time, calibrating classical Cepheids in deep-potential SN Ia host galaxies against diffuse, low-mass anchor galaxies systematically misestimates host distance moduli. When interpreted through a universal Period–Luminosity relation, this clock-rate anomaly mimics diminished luminosity, leading to underestimated distances and an inflated local Hubble constant. In standard distance-ladder linear regression, unconstrained latent host distance moduli μ_i algebraically absorb host-level environmental shifts. A standard-ladder projection test that inserts a TEP environmental column into the SH0ES design matrix therefore yields an apparent null result. This structural degeneracy is resolved by formulating the calibrator-host distance ladder as a generative clock-aware model coupled to the Hubble flow. This generative framework is tested using the complete public Riess et al. (2022) sample of 37 distinct SN Ia host galaxies, utilizing a local inner-potential coordinate σ_* compiled from published stellar-absorption dispersions (18 hosts) and calibrated H I linewidth dispersion proxies (19 hosts), matched to the anchor endpoint construction and combined with continuous environmental screening. The canonical peculiar-velocity convention throughout is the R22 baseline σ_v = 250 km/s; all other σ_v values are reported as sensitivity variants. A median split at σ_* = 95 km/s stratifies the derived per-host expansion rates by Δ H_0 = 9.59 km s⁻¹ Mpc⁻¹ (Spearman ρ = 0.553, p = 3.9×10⁻⁴; Pearson r = 0.501, p = 0.0016), strengthening to r = 0.556 (p = 0.017) on the stellar-absorption-only subset. Across all 37 hosts, the generative endpoint likelihood yields Γ_X=(3.846±1.653)×10^7 km s⁻¹ Mpc⁻¹ (2.33σ) under canonical 250 km/s velocity variance, strengthening to (3.906±1.389)×10^7 (2.81σ) under the reduced-scatter sensitivity variants (σ_v = 150–182.1 km/s), with permutation p = 0.033, 99.8% positive bootstrap draws, and 100% leave-one-host-out sign stability (37 of 37 refits positive). In the cosmologically constrained host-level expansion-rate likelihood across the 33 Hubble-flow hosts, the combined endpoint response is recovered at Γ_X=(3.805±1.647)×10^7 km s⁻¹ Mpc⁻¹ (2.31σ; 3.47σ at 150 km/s), yielding a conventional Hubble-flow intercept of H_app=68.48±1.47 km s⁻¹ Mpc⁻¹ at the canonical convention (68.31±0.97 at σ_v = 150 km/s). When evaluated in a joint multi-block framework with independent TRGB distances and geometric anchors, the multi-block likelihood favours an environmental response under the restricted Cepheid closure (κ_Cep = (0.665 ± 0.362)× 10^6 mag, 1.83σ at canonical σ_v = 250 km/s; (0.746 ± 0.343)× 10^6 mag, 2.17σ at σ_v = 150 km/s), resolving the local Hubble constant to H_app = 68.82± 1.45 km s⁻¹ Mpc⁻¹, while the standalone redshift-only regression in H_0 space yields κ_Cep = (1.206 ± 0.528)× 10^6 mag (2.29σ canonical; (1.275 ± 0.370)× 10^6 mag, 3.44σ at σ_v = 150 km/s). Under the physical coupled observing-chain model, the spectroscopic tracer's direct gravitational redshift contributes a velocity slope β_X,\rm grav = c/⟨ d ⟩ \sim 10^4 km s⁻¹ Mpc⁻¹ (99.9% of the response is carried by the Cepheid distance scale (κ_Cep^equiv = (1.220 ± 0.531)× 10^6 mag); this allocation is physically derived from proper-time dilation, with host-specific aperture validation supplied by a full-provenance NED tracer census (Step 54; Appendix C.5). Model comparison favours the environmental response: at the canonical convention the velocity-carrier configuration attains the lowest information criteria of the multi-block family (ΔBIC = -1.36 relative to the null), a leave-one-out cross-validation tournament over fixed and scanned response shapes selects the bounded amplitude-sector form S_A = min[1,(σ_*/σ_T)^2/3] in every primary dataset, and a Stouffer combination of the four primary channels with independent noise sources returns Z = +4.14 (p = 3.4×10⁻⁵ two-sided; Z = +3.08 under a conservative shared correlation ρ = 0.3). Propagating the TEP potential correction through the distance ladder requires an explicit estimator distinction. The prespecified primary full-ladder projection propagates the measured endpoint-equivalent response and yields H_0=70.463±0.972 km s⁻¹ Mpc⁻¹, reducing the offset from Planck 67.4 ± 0.5 to 2.80σ with a likelihood improvement of Δ\chi^2=+9.870; conditional matrix projections give 71.01 km s⁻¹ Mpc⁻¹ under canonical scaling. When the response coefficient is instead left free inside the ladder design matrix it returns κ_Cep=(0.056±0.370)×10^6 mag — the expected near-null: the shared Cepheid zero point absorbs the anchor-to-host common mode, so the matrix channel measures only the differential component and cannot see the term that closes the remaining offset. A separate unified host-level reconstruction, which re-optimizes the environmental response to flatten the host-environment trend, yields H_0=67.83±1.56 km s⁻¹ Mpc⁻¹, internally consistent with the TEP CMB value (66.70 ± 0.58, Paper 26) at 0.68σ — an agreement between two products of the same framework, not an independent cross-check — and 0.26σ from Planck. The latter is a host-level reconciliation route, not the primary full-ladder estimator: it evaluates the fitted response at the cosmic baseline and is conditional on that response form extending below the anchor ensemble to the unperturbed potential. Full-matrix propagation through the 3,490-row SH0ES system confirms exact reference-gauge invariance. The statistically cleanest channel is the within-host period structure: the environment-coupled Cepheid period term, identified through the 41 independent period distributions (37 hosts plus 4 anchors) and immune to latent-modulus absorption and peculiar-velocity systematics by construction, returns κ_P = (-4.20 ± 3.38)× 10^5 mag for the ad hoc linear carrier, resolving to 2.3–2.8σ under the period-quadratic carrier derived from the amplitude-sector response law, with a negative sign in all 41 leave-one-host-out refits and a fitted per-host slope ordering of +1.12 ± 0.30 × 10^6 mag per unit X. Finally, single-galaxy differential tests provide internal signatures with the sign predicted by TEP free from host-to-host peculiar velocity systematics: M31 inner versus outer Cepheids exhibit an empirical Period–Luminosity offset of +0.356 ± 0.136 mag (2.6σ), increasing to +0.681 ± 0.187 mag (3.65σ) under spatial PHAT matching; however, M31 fails pair-matched controls (colour-matched -0.017 ± 0.128 mag, 0.1σ; eW-matched -0.103 ± 0.129 mag, 0.8σ), and the two admissible readings are tested against the theory's ambient closure. The overmatching hypothesis — that error matching discards an ambient-field signal — is bounded out: the photometric error is nearly orthogonal to the kpc-scale ambient-field proxy (corr=-0.18), and matching on the ambient field alone leaves a positive but inconclusive offset (+0.206 ± 0.163 mag) that does not survive joint environment-plus-error control (-0.069 ± 0.126 mag under matched (logP, e_W, \log u_amb) cells — the residual was error-carried within ambient strata). Stratified through the full error distribution the conditional contrast declines monotonically from +0.270 ± 0.304 mag at the 10th-percentile error level to -0.117 ± 0.202 mag in the poorest, and the HST PHAT channel shows the same susceptibility under propagated-error matching (-0.565 ± 0.079 mag, against +0.906 ± 0.147 mag under neighbour-density control). The M31 channel is therefore carried as conditional evidence: the raw gradient is real and a positive residual is permitted in the best-measured strata, but neither reading closes the case. The LMC gradient (+0.0284 ± 0.0086 mag, 3.3σ) is robust to period and colour matching and remains the strongest single-galaxy result.


## Key Findings

- **Environmental stratification (primary $\sigma_*$ coordinate):** median split at $\sigma_* = 95$ km/s gives $\Delta H_0 = +9.59\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (Spearman $\rho = 0.553$, $p = 3.9\times10^{-4}$; Pearson $r = 0.501$, $p = 0.0016$).
- **37-host endpoint likelihood (canonical 250 km/s):** $\Gamma_X = (3.846 \pm 1.653) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.33\sigma$); $100\%$ positive leave-one-host-out stability (37 of 37 refits); permutation $p = 0.033$.
- **Flow-velocity sensitivity:** $2.81\sigma$ at 150 km/s and at data-driven Pantheon+ residual scatter $182.1$ km/s; $1.27\sigma$ at 500 km/s.
- **Cosmologically constrained expansion likelihood (33 Hubble-flow hosts):** $\Gamma_X = (3.805 \pm 1.647) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.31\sigma$ at $\sigma_v=250$; $3.47\sigma$ at $\sigma_v=150$), $H_{\rm app} = 68.48 \pm 1.47\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($68.31 \pm 0.97$ at 150 km/s).
- **Joint multi-block likelihood:** $\kappa_{\rm Cep} = (0.665 \pm 0.362) \times 10^6\ {\rm mag}$ ($1.83\sigma$ canonical; $(0.746 \pm 0.343) \times 10^6$ mag, $2.17\sigma$ at $\sigma_v=150$ km/s); $H_{\rm app} = 68.82 \pm 1.45\ {\rm km\,s^{-1}\,Mpc^{-1}}$.
- **Prespecified primary full-ladder projection:** $H_0 = 70.463 \pm 0.972\ {\rm km\,s^{-1}\,Mpc^{-1}}$, $2.80\sigma$ from Planck, $\Delta\chi^2 = +9.870$.
- **Separate host-level reconstruction:** $H_0 = 67.83 \pm 1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$, $0.26\sigma$ from Planck; $0.68\sigma$ from TEP-CMB ($H_0 = 66.70 \pm 0.58$, Paper 26) — same framework, not independent.
- **M31 internal Cepheids (Pan-STARRS):** $\Delta W = +0.356 \pm 0.136$ mag ($2.6\sigma$); conditional evidence under pair-matched controls.
- **M31 HST PHAT footprint:** $\Delta W = +0.681 \pm 0.187$ mag ($3.65\sigma$) under spatial PHAT matching.
- **LMC OGLE-IV Cepheids (radial stratification):** $\Delta W = +0.0284 \pm 0.0086$ mag ($3.3\sigma$) — strongest single-galaxy result.
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
| **Paper 17** | [TEP-LLR](https://github.com/matthewsmawfield/TEP-LLR) | Lunar Laser Ranging and the Nordtvedt Effect | [10.5281/zenodo.19446028](https://doi.org/10.5281/zenodo.19446028) |

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
├── 11-TEP-H0-v0.11-KingstonUponHull.md  # Full manuscript (Markdown)
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
inferential artifacts have been written. Historical Steps 28–31 and 49, plus
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
likelihoods ($H_{\rm app} = 68.42$–$68.82\ {\rm km\,s^{-1}\,Mpc^{-1}}$). The
separate host-level reconstruction is Planck-concordant ($67.83 \pm 1.56$,
$0.26\sigma$), while the prespecified
primary matrix projection leaves a $2.80\sigma$ residual — the identifiability
bound of the shared Cepheid zero point; neither route invokes
early-universe modifications. Local-gravity closure, pulsar coefficient
matching, and the gauge-failing Step 50 summary are not inferential results of
this repository.

## Citation

```bibtex
@article{smawfield2026cepheidbias,
  title={The Cepheid Bias: Resolving the Hubble Tension},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18209702},
  note={Preprint v0.11 (Kingston upon Hull)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
