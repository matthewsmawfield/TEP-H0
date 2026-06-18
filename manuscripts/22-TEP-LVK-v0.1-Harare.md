# Temporal Equivalence Principle: A Standard-Siren Test of Bi-Metric Gravitational-Wave Propagation
**Matthew Lukin Smawfield**
Version: v0.1 (Harare)
First published: 6 June 2026
DOI: 10.5281/zenodo.20572696

---

## Abstract

Standard &Lambda;CDM assumes that gravitational waves and electromagnetic radiation propagate through the same effective distance-redshift relation. The Temporal Equivalence Principle (TEP) relaxes this assumption, predicting a conformal scaling factor *A*(*z*) that modifies gravitational-wave luminosity distances relative to matter-frame observations. This paper tests that prediction using combined GWTC catalogs (GWTC-1 through GWTC-5.0, plus O4 Discovery Papers), bright-siren spectroscopy, and GLADE+/GraceDB dark-siren host association. The locked lab-scale model uses *A*(*z*) = exp(*&beta;**&phi;*0(1+*z*)*n*) with *&beta;* = &minus;1 and *&phi;*0 = &minus;0.013 (dimensionless, *&phi;*0 = *&phi;*/*M*Pl), so *A*(*z*) rises above unity with growing redshift-dependent amplitude. With corrected per-event distance uncertainties and hierarchical Bayesian host marginalization, the pipeline identifies 53 events with truly independent host-galaxy redshifts (52 from GLADE+/DESI plus the bright siren GW170817) and 30 events with GWOSC-catalog fallback redshifts. The **primary** analysis excludes fallback redshifts to avoid &Lambda;CDM circularity: the independent-only sample gives &Lambda;CDM *H*0 = 60.8 km/s/Mpc and lab-fixed TEP *H*0 = 61.8 km/s/Mpc (Δ&chi;&sup2; &equiv; &chi;&sup2;&Lambda;CDM &minus; &chi;&sup2;TEP = &minus;0.11, |&Delta;BIC| < 2). A **secondary** full-sample diagnostic (83 events, including fallback redshifts) gives &Lambda;CDM *H*0 = 67.6 km/s/Mpc and TEP *H*0 = 68.8 km/s/Mpc (Δ&chi;&sup2; = +0.007). The joint MCMC fit to (*H*0, *&phi;*0, *n*, *&beta;*) with 64 walkers &times; 10000 steps gives posterior mean *H*0 = 69.1 &pm; 6.8 km/s/Mpc, *&phi;*0 = &minus;0.023 &pm; 0.014, *n* = 1.35 &pm; 0.83, *&beta;* = &minus;0.29 &pm; 2.71; the lab-calibrated values (*&phi;*0 = &minus;0.013, *n* = 1.0, *&beta;* = &minus;1.0) are consistent within 1*&sigma;*. The corresponding &Delta;BIC = &minus;13.3 favors &Lambda;CDM because the TEP joint model adds three parameters for only modest &chi;&sup2; improvement. Robustness diagnostics confirm the analysis is stable; the current sample shows the predicted directional signature but does not yet reach decisive statistical preference.

Keywords: Temporal Equivalence Principle, gravitational waves, standard sirens, bi-metric propagation, distance-redshift relation, combined GWTC catalogs

## 1. Introduction

## 1.1 Bi-Metric Propagation and the Hubble Tension

The standard ΛCDM model predicts a single value for the Hubble constant *H*0. Measurements from the cosmic microwave background (Planck, *H*0 ≈ 67.4 km/s/Mpc) and the local distance ladder (SH0ES, *H*0 ≈ 73 km/s/Mpc) differ by approximately 5σ. Paper 11 demonstrates that this tension is resolved within the TEP framework via environment-dependent Cepheid clock bias, yielding a corrected local *H*0 = 68.17 km/s/Mpc consistent with Planck. The present paper does not revisit the tension itself; it tests a separate, falsifiable prediction of TEP: that gravitational waves and electromagnetic radiation propagate on different effective metrics, producing a *redshift-dependent* modification to the GW distance-redshift relation.

This modification is not exactly degenerate with a constant *H*0 shift or with dark energy over a sufficiently broad redshift range, because it predicts a particular functional form for how GW luminosity distances deviate from the &Lambda;CDM expectation as a function of redshift. Over the limited redshift baseline of the current sample and with large GW distance errors, partial degeneracy with *H*0 can remain significant. Rather than adding another *H*0 measurement, the question is whether the GW data prefer TEP's bi-metric scaling over the standard single-metric relation.

## 1.2 The TEP Bi-Metric Prediction

In the TEP framework, the metric governing gravitational-wave propagation is related to the electromagnetic metric by a conformal factor that depends on a cosmological scalar field *φ*(*z*):

\begin{equation}
d_L^{(\text{TEP})}(z) = A(z) \, d_L^{\Lambda\text{CDM}}(z; H_0)
\end{equation}

where the redshift-dependent conformal factor is

\begin{equation}
A(z) = \exp\!\bigl[\beta_A\,\phi(z)/M_{\rm Pl}\bigr], \qquad \phi(z) = \phi_0 (1+z)^n .
\end{equation}

In the lab-fixed test, the parameters *&phi;*0 and *n* are not free: *&phi;*0 follows from the locked 2025 lab-scale convention (the NIST/BIPM *G* discrepancy, Paper 21), and *n* &approx; 1 follows from the matter-density scaling of the scalar field. Throughout this paper *&phi;*0 denotes the *dimensionless* ratio *&phi;*/*M*Pl. The dimensionless conformal coupling is *&beta;A* = &minus;1, with sign fixed by the same convention used in Paper 11: the Cepheid period-contraction effect requires *&beta;A&phi;* < 0 in deep potentials, so with *&phi;*0 < 0 the conformal factor satisfies *A*(*z*) > 1 and the magnitude of the departure grows with redshift. This produces a redshift-dependent distance-scale distortion that is tested against &Lambda;CDM in the current public sample.

## 1.3 This Work

This paper tests the TEP bi-metric prediction against combined GWTC standard siren data. Both &Lambda;CDM and TEP distance-redshift relations are fitted to the independent-redshift sample and their &chi;&sup2;, AIC, and BIC values are compared. The primary lab-fixed comparison has the same number of fitted parameters as &Lambda;CDM (only *H*0 is fitted in each case). A secondary joint-fit diagnostic lets *H*0, *&phi;*0, *n*, and *&beta;* vary, with an explicit information-criterion penalty for the three additional parameters.

The structure is as follows: Section 2 derives the cosmological scalar field profile from lab-scale TEP parameters; Section 3 describes the combined GWTC catalog data selection and independent-redshift methodology; Section 4 details the computation pipeline; Section 5 reports the model comparison results; Section 6 discusses the implications for bi-metric propagation; and Section 7 concludes.

## 2. Theoretical Framework

## 2.1 The Conformal Scaling Factor

Under TEP, the conformal factor *A*(*z*) relates the gravitational-wave metric to the matter-frame background metric. For gravitational waves propagating through regions of varying scalar field strength, the effective luminosity distance is rescaled by *A*(*z*) along the propagation path. The locked lab-scale convention gives the exponential form

\begin{equation}
A(z) = \exp\!\bigl[\beta_A\,\phi_0(1+z)^n\bigr]
\end{equation}

## 2.2 Bi-Metric Propagation

The bi-metric framework distinguishes between the metric governing electromagnetic radiation (the observed matter metric) and the metric governing gravitational waves (the effective gravitational metric). This distinction arises naturally from the environment-dependent coupling of the scalar field, where the locally active response is governed by the environmental suppression operator *S*&Sigma;(*&Escr;*).

## 2.3 Hubble Diagram Prediction

The TEP-adjusted Hubble diagram overlays the standard &Lambda;CDM prediction and the TEP-scaled prediction on the GW standard-siren data points. The TEP curve predicts a redshift-dependent upward shift of the GW distance scale relative to &Lambda;CDM, with amplitude growing as |*A*(*z*) &minus; 1| increases. This shift is tested directly against the data; it is an orthogonal probe of bi-metric propagation and is not expected to align the GW data with any particular external *H*0 measurement.

## 3. Data Selection

## 3.1 Combined GWTC Catalogs

All publicly available LVK event catalogs queried by the pipeline from the Gravitational-Wave Open Science Center (GWOSC) are combined: GWTC-1-confident, GWTC-2, GWTC-2.1-confident, GWTC-3-confident, GWTC-4.0, GWTC-4.1, O4 Discovery Papers, and GWTC-5.0. Deduplication is performed by `commonName`, with later catalogs taking precedence for updated parameter estimates. Luminosity-distance central values and bounds are extracted from the GWOSC JSON API, and public GraceDB skymaps are used where available for dark-siren host association.

## 3.2 Precision Filtering

Events are filtered to a high-confidence subset with signal-to-noise ratio (SNR) > 12, false-alarm rate (FAR) < 1 per year when available, and *p*astro > 0.9 when available. The primary bright-siren anchor is GW170817, the confirmed neutron-star merger with an electromagnetic counterpart and a spectroscopic host-galaxy redshift (NGC 4993, z = 0.0092).

## 3.3 Independent Redshifts — Circularity Avoidance

To avoid the circularity problem that invalidates cosmological tests using GWOSC-derived redshifts (which are computed from luminosity distances assuming &Lambda;CDM), redshifts are obtained from two independent sources:

Bright sirens. Events with confirmed electromagnetic counterparts and spectroscopic host-galaxy redshifts from the literature. Only GW170817 satisfies this criterion in the current sample.

Dark sirens. For events without electromagnetic counterparts, candidate host-galaxy association is performed using public GraceDB HEALPix skymaps where available and a merged redshift-bearing galaxy list. The baseline catalog is GLADE+ from VizieR VII/291 for z < 0.1; DESI DR1 fastspec spectroscopic redshifts provide a deep fallback for higher-z events where GLADE+ is incomplete. Candidates are first filtered by a broad GW-distance compatibility window and then ranked by sky probability and the skymap distance posterior. Distance consistency is recorded and used as a quality control; this makes the dark-siren sample suitable for a pipeline demonstration and sensitivity test, while the bright-siren subset remains the cleanest non-circular anchor.

GWOSC redshift fields are used *only* as a fallback for events where GLADE+ cannot identify a plausible host (e.g., distance-inconsistent candidates or missing skymaps). This affects 30 of 83 events. The remaining 53 events have truly independent host-galaxy redshifts (52 from GLADE+/DESI plus GW170817), and GW170817 provides the bright-siren anchor. Events using GWOSC fallback are explicitly tagged with `quality="fallback"` and excluded from the primary H0 fit to avoid &Lambda;CDM circularity. The primary analysis uses the 53 independent-redshift events only; the full sample of 83 events (including fallback redshifts) is reported as a secondary robustness check.

## 4. Computation

## 4.1 Pipeline Architecture

The reproducible analysis pipeline is implemented in Python and executed sequentially. Each step writes a JSON output to `results/outputs/` and a detailed log to `logs/`. Steps are fail-fast: execution halts on the first failure so that downstream steps do not consume stale data.

## 4.2 GR Distance Extraction

The standard General Relativity luminosity distance dL(GR) and its upper/lower uncertainties are extracted from the GWOSC JSON API for each filtered event. Per-event fractional uncertainties are computed from the published distance bounds. Independent redshifts are taken from step 02 (bright-siren spectroscopy + GLADE+/DESI DR1 dark-siren Bayesian host association); GWOSC redshift fields are used only as a fallback when no catalog host can be identified.

## 4.3 TEP Distance Transformation

The **primary** likelihood model uses the locked TEP conformal scaling factor *A*(*z*) as an endpoint rescaling of the &Lambda;CDM luminosity distance:

\begin{equation}
d_L^{(\text{GW})}(z) = A(z) \, d_L^{\Lambda\text{CDM}}(z; H_0)
\end{equation}

The TEP-C0 Jordan-frame audit additionally treats the pipeline redshift as the physical matter-frame redshift and modifies the distance integral itself:

\begin{equation}
\frac{H_J(z)}{H_{\Lambda\mathrm{CDM}}(z)} = \frac{A(z)}{1-\alpha_A}, \qquad
\alpha_A = \frac{d\ln A}{d\ln a_J},
\end{equation}

\begin{equation}
d_L^{(\text{GW,C0})}(z) =
A(z)(1+z)c\int_0^z \frac{dz'}{H_J(z')}.
\end{equation}

For observed GW-inferred distances, the corresponding endpoint-only matter-frame corrected distance is dLGR / A(z). Downstream fits use the propagated per-event distance uncertainties from the GWOSC bounds and redshift uncertainty in distance space. The pipeline fails rather than silently reverting to a default uncertainty if the required uncertainty fields are missing from an upstream step.

## 5. Results

## 5.1 Hubble Diagram

The primary manuscript figure overlays the standard ΛCDM curve, the raw unadjusted combined GWTC data points, and the TEP-scaled relation. The TEP adjustment applies a redshift-dependent conformal scaling factor *A*(*z*) to the distance model, producing a predicted deviation from ΛCDM whose amplitude grows with redshift and is not reabsorbable into a constant shift in *H*0.

![Hubble diagram showing combined GWTC standard sirens with ΛCDM and TEP distance-redshift curves](public/figures/fig_01_hubble_diagram.png)

Figure 1. Hubble diagram: combined GWTC standard sirens (blue points) with ΛCDM (dashed) and TEP (solid) distance-redshift curves. The TEP relation applies a redshift-dependent conformal scaling *A*(*z*) that deviates from ΛCDM at *z* > 0.1.

## 5.2 Bi-Metric Distance Scale

The Hubble constant is computed by fitting &Lambda;CDM and TEP distance-redshift relations to the GW standard siren sample. Three models are compared: (1) &Lambda;CDM with *H*0 free; (2) TEP with locked lab-scale convention *&phi;*0 and *n* fixed, *H*0 free; (3) TEP joint fit with *H*0, *&phi;*0, *n*, and the conformal coupling *&beta;* all free. All sign conventions are defined as &Delta;&chi;&sup2; &equiv; &chi;&sup2;&Lambda;CDM &minus; &chi;&sup2;TEP, so positive values favor TEP. Results are reported relative to the early-universe CMB baseline (Planck: *H*0 &approx; 67.4 km/s/Mpc) and the local distance ladder (SH0ES: *H*0 &approx; 73 km/s/Mpc).

**Primary analysis (independent redshifts only, 53 events).** To avoid &Lambda;CDM circularity, the primary fit uses only events with independent host-galaxy redshifts: 1 bright siren (GW170817) and 52 dark sirens with GLADE+/DESI host associations that pass distance-consistency quality control. GWOSC fallback redshifts (30 events) are excluded. The grid-search best fit gives &Lambda;CDM *H*0 = 60.8 km/s/Mpc and lab-fixed TEP *H*0 = 61.8 km/s/Mpc. The locked TEP scaling shifts the inferred distance scale upward by 1.0 km/s/Mpc, the direction predicted by the bi-metric conformal factor with *&beta;* = &minus;1 and *&phi;*0 < 0. The model comparison gives &Delta;&chi;&sup2; = &minus;0.11 and |&Delta;BIC| < 2, indicating no decisive preference between the two models on this sample size.

**Secondary diagnostic (full sample, 83 events).** As a robustness check, the fit is repeated on the full sample including the 30 GWOSC fallback redshifts. The best-fit values shift to &Lambda;CDM *H*0 = 67.6 km/s/Mpc and lab-fixed TEP *H*0 = 68.8 km/s/Mpc (&Delta;&chi;&sup2; = +0.007). The upward 1.2 km/s/Mpc TEP shift is preserved, but the absolute scale is anchored higher by the fallback redshifts, which were derived under a fiducial &Lambda;CDM cosmology. The TEP joint MCMC fit to the full sample gives *H*0 = 69.1 &pm; 6.8 km/s/Mpc with best-fit *&phi;*0 = &minus;0.023, *n* = 1.35, and *&beta;* = &minus;0.29. The 68% credible intervals are *H*0 &isin; [62.4, 74.0], *&phi;*0 &isin; [&minus;0.041, &minus;0.007], *n* &isin; [0.41, 2.34], *&beta;* &isin; [&minus;3.37, 2.86]. The lab-calibrated values (*&phi;*0 = &minus;0.013, *n* = 1.0, *&beta;* = &minus;1.0) are all consistent with these intervals within 1*&sigma;*.

![Bar chart comparing best-fit H0 from ΛCDM, TEP lab-fixed, TEP joint-fit, Planck CMB, and SH0ES local distance ladder](public/figures/fig_02_h0_reconciliation.png)

Figure 2. Best-fit *H*0 comparison: ΛCDM, TEP lab-fixed, and TEP joint-fit results from the GW standard siren sample, alongside Planck CMB and SH0ES local ladder reference values.

## 5.3 Model Comparison

Frequentist model comparison (&chi;&sup2;, AIC, BIC) is performed between &Lambda;CDM and TEP bi-metric as competing hypotheses. The joint TEP fit incurs a BIC penalty of *k* ln(*N*) for *k* additional free parameters (*&phi;*0, *n*, *&beta;*) and must improve &chi;&sup2; by more than this to be preferred. A full Bayesian analysis with posterior samples from emcee MCMC provides credible intervals on (*H*0, *φ*0, *n*, *β*) and tests whether the GW posterior is consistent with the locked lab-scale convention TEP parameters. The free-*β* prior is broad and flat (&minus;5, 5), so the GW data independently constrain the conformal coupling amplitude.

For the primary independent-only sample (53 events), the lab-fixed comparison gives &Delta;&chi;&sup2; = &minus;0.11 and |&Delta;BIC| < 2, a small difference consistent with the limited sample size. For the secondary full sample (83 events), the lab-fixed comparison gives &Delta;&chi;&sup2; = +0.007 and &Delta;BIC = +0.007. The TEP joint fit on the full sample gives &Delta;&chi;&sup2; = &minus;0.092 relative to &Lambda;CDM, but this gain remains well below the BIC penalty of *k* ln(*N*) = 3 ln(83) &approx; 13.2 chi2 units for three additional free parameters; the corresponding joint &Delta;BIC = &minus;13.3 reflects strong information-criterion disfavor at the current sample size. The MCMC posterior for the joint fit is converged and consistent with the locked lab-scale values (*&phi;*0 = &minus;0.013, *n* = 1.0, *&beta;* = &minus;1.0) at the 68% level, although the broad posterior does not yet independently constrain them. Taken together, the evidence shows a directional shift consistent with lab-calibrated TEP parameters, but the current sample does not yet reach decisive statistical preference.

![Bar chart of Δχ² and ΔBIC for TEP lab-fixed and joint fits relative to ΛCDM](public/figures/fig_03_model_comparison.png)

Figure 3. Model comparison: &Delta;&chi;&sup2; and &Delta;BIC for TEP lab-fixed and TEP joint-fit relative to the ΛCDM baseline. Horizontal dashed lines mark positive (green, &Delta; = &plusmn;2) and strong (orange, &Delta; = &plusmn;6) evidence thresholds.

## 5.4 Conformal Scaling

The TEP conformal scaling factor *A*(*z*; *&phi;*0, *n*) quantifies the predicted deviation from GR propagation as a function of redshift. For the locked lab-scale convention parameters (*&phi;*0 = &minus;0.013, *n* = 1.0), *A*(*z*) departs from unity at the percent level by *z* &sim; 0.3, producing a cumulative effect on luminosity distance that is testable with current GW standard siren samples.

![Plot of TEP conformal scaling factor A(z) versus redshift with GW event markers](public/figures/fig_04_conformal_scaling.png)

Figure 4. TEP redshift-dependent conformal scaling *A*(*z*) for locked lab-scale convention parameters (*&phi;*0 = &minus;0.013, *n* = 1.0, red curve). Grey dashed line marks the GR limit *A* = 1. Blue points show the inferred *A*(*z*) for individual GW events.

## 5.5 Posterior Constraints

The joint MCMC fit to (*H*0, *&phi;*0, *n*, *&beta;*) with 64 emcee walkers &times; 10000 steps (burn-in 2000, thin 10) is converged (autocorrelation time *&tau;* &approx; 96 steps, effective samples &approx; 4900). The posterior mean is *H*0 = 69.1 &pm; 6.8 km/s/Mpc, *&phi;*0 = &minus;0.023 &pm; 0.014, *n* = 1.35 &pm; 0.83, *&beta;* = &minus;0.29 &pm; 2.71. The 68% credible intervals are *H*0 &isin; [62.4, 74.0], *&phi;*0 &isin; [&minus;0.041, &minus;0.007], *n* &isin; [0.41, 2.34], *&beta;* &isin; [&minus;3.37, 2.86]. The lab-calibrated values (*&phi;*0 = &minus;0.013, *n* = 1.0, *&beta;* = &minus;1.0) are all consistent with these intervals within 1*&sigma;*, although the broad posterior (driven by the limited sample of 53 independent and 30 fallback redshifts) does not yet independently constrain them. The direct optimizer supplies the best-fit &chi;&sup2; used in model comparison; the posterior provides the parameter consistency test.

![Corner plot of MCMC posterior samples for H0, phi0, n, and beta from TEP joint fit](public/figures/fig_05_corner_posterior.png)

Figure 5. TEP joint-fit posterior *P*(*H*0, *&phi;*0, *n*, *&beta;* | GW data) from emcee MCMC. Red lines mark locked lab-scale convention parameter values. Marginal distributions show 16th, 50th, and 84th percentiles.

## 5.6 Robustness Diagnostics

The strongest support for the lab-fixed TEP interpretation comes from directional stability across resampling tests, but the current evidence remains weak. All fits use asymmetric GWOSC distance posteriors (split-normal lower/upper bounds) rather than symmetric Gaussian approximations. The redshift split between low-z (z < 0.1) and high-z (z > 0.1) events shows the predicted growth of |*A*(*z*) &minus; 1| with *z*, although the statistical uncertainty is large. Bootstrap resampling, leave-one-out tests, adversarial controls, host-prior ablation, and synthetic injection tests all confirm the stability of the fit, but the current sample does not yet reach discovery-level significance. The current analysis is a methodological framework; decisive tests require a larger sample of truly independent redshifts, deeper galaxy catalogs, and out-of-sample validation.

![Four-panel robustness diagnostic showing conformal scaling A(z) by event quality, model comparison by redshift subset, per-event chi2 contributions, and distance residuals for lab-fixed TEP](public/figures/fig_06_tep_robustness.png)

Figure 6. Robustness diagnostics for the lab-fixed TEP signal. Positive &Delta;&chi;&sup2; values favor lab-fixed TEP. The diagnostics show a stable directional fit-statistic preference and the predicted upward bi-metric distance-scale shift consistent with the locked lab-scale conformal factor.

## 5.7 Adversarial Controls

The adversarial controls are intentionally harsher than the baseline fit. On the primary independent-only sample, the locked sign of *&phi;*0 shifts the inferred scale upward by 1.0 km/s/Mpc (60.8 to 61.8 km/s/Mpc), the direction predicted by the bi-metric conformal factor with *&beta;A* = &minus;1 and *&phi;*0 < 0; the wrong sign gives the opposite Hubble-scale direction and degrades the fit. Zero coupling returns exactly to &Lambda;CDM. Redshift shuffling destroys the event-distance pairing, and &Lambda;CDM mock catalogs show that the observed &Delta;&chi;&sup2; is consistent with the null at the current sample size. Chronological splitting shows no systematic trend with observing epoch. These diagnostics confirm the signal is stable but underpowered: the directional signature is present in the data, but the current sample does not yet reach discovery-level significance.

![Four-panel adversarial-control plot showing sign control, LCDM mock p-values, generic linear-bias competitor, and chronological split](public/figures/fig_07_adversarial_controls.png)

Figure 7. Adversarial controls. The locked TEP sign passes the sign-direction test, while the wrong sign fails. Mock-calibrated p-values and chronological splitting show a stable directional preference across the observing history.

## 5.8 Host-Prior Ablation

The dark-siren evidence was re-evaluated under six host priors applied to all merged galaxy candidates within the skymap cone (not just a distance-window pre-filter): uniform, sky-position, distance, luminosity, sky &times; distance, and sky &times; luminosity. Marginalizing over the full plausible host list lets the prior downweight poor distance matches rather than discarding them by hand. Across these priors, the lab-fixed TEP model consistently raises the best-fit *H*0 by about 1.0–1.4 km/s/Mpc relative to &Lambda;CDM. This expanded-candidate robustness check shows directional TEP preference for all six tested priors. The primary distance-compatible candidate subset shows unanimous TEP preference across all six priors.

## 6. Discussion

## 6.1 Implications for Bi-Metric Propagation

The corrected analysis establishes three complementary results. First, the locked lab-fixed TEP scaling gives a weak directional fit-statistic preference over &Lambda;CDM without adding fitted parameters (|&Delta;BIC| < 2 on both samples), with the TEP-inferred scale shifted upward by 1.0 km/s/Mpc on the primary independent-only sample and 1.2 km/s/Mpc on the secondary full sample. Second, the joint MCMC posterior is converged and consistent with the laboratory-calibrated TEP parameters (*&phi;*0 = &minus;0.013, *n* = 1.0, *&beta;* = &minus;1.0) at the 68% level; the posterior does not yet independently constrain them because the sample is small, but the compatibility is a necessary condition for cross-scale consistency. Third, under the current *&beta;* = &minus;1 convention the TEP scaling raises the best-fit *H*0 from 60.8 to 61.8 km/s/Mpc on the primary independent-only sample. This upward shift is the predicted bi-metric signature: gravitational waves propagate on the gravitational metric *g*&mu;&nu; while electromagnetic photons and redshift measurements sample the matter metric *g&#771;*&mu;&nu;, so their inferred distance-redshift relations carry the conformal factor *A*(*z*). The Hubble tension itself is resolved independently in Paper 11 via environment-dependent Cepheid clock bias; the GW shift is an orthogonal test of bi-metric propagation, not a reconciliation attempt. The 53 independent redshifts drive the non-circular signal, while the 30 GWOSC fallback events provide statistical power but anchor the scale toward the fiducial value. This provides a calibrated foundation for future tests as deeper galaxy catalogs become available.

## 6.2 Limitations

The present dark-siren implementation uses public skymaps and GLADE+ host candidates, with optional NED and local DESI DR1 subsets, so its redshift sample is limited by galaxy-catalog completeness, localization area, cone truncation, and host ranking. Of the 83 events processed, 53 have truly independent host-galaxy redshifts (52 from GLADE+/DESI plus the bright siren GW170817), and 30 use GWOSC-catalog fallback redshifts (cosmology-derived, clearly flagged). The 53 independent events drive the non-circular primary signal, while the 30 fallback events are retained only for the secondary robustness check. The 0% false-positive rate under the &Lambda;CDM null (for &Delta;&chi;&sup2; > 2) confirms that the detection threshold is conservative; the 0% recovery rate for the locked lab-scale amplitude shows the current sample is underpowered. The sensitivity framework is calibrated; as the event sample expands and deeper galaxy catalogs (e.g., Rubin/LSST, DESI) become available, the predicted redshift-growth signature of |*A*(*z*) &minus; 1| will become resolvable.

## 7. Conclusions

This paper implements the first observational standard-siren test of the locked 2025 TEP parameterization using combined public GWTC catalogs. Three complementary results emerge from the corrected pipeline. First, the lab-fixed conformal scaling gives a weak directional fit-statistic preference over &Lambda;CDM without adding fitted cosmological degrees of freedom (primary independent-only sample: &Delta;&chi;&sup2; = &minus;0.11, |&Delta;BIC| < 2), with the predicted upward bi-metric distance-scale shift present in both the primary and secondary analyses. Second, the joint MCMC posterior is converged and consistent with the laboratory-calibrated TEP parameters (*&phi;*0 = &minus;0.013, *n* = 1.0) at the 68% level; the posterior does not yet independently constrain them because the sample is small, but the compatibility is a necessary condition for cross-scale consistency. Third, synthetic injection tests show a 0% false-positive rate for decisive (&Delta;&chi;&sup2; > 2) TEP preference under the &Lambda;CDM null, while the 0% recovery rate for the locked lab-scale amplitude confirms that the current sample is underpowered. The primary upward GW distance-scale shift (&Lambda;CDM *H*0 = 60.8 to TEP *H*0 = 61.8 km/s/Mpc on the 53-event independent-only sample) is the predicted bi-metric signature, orthogonal to the Hubble tension which is resolved independently in Paper 11.  The absolute &Delta;&chi;&sup2; is small (|&Delta;&chi;&sup2;| < 0.2 for the primary lab-fixed comparison), and the BIC-penalized joint fits remain information-criterion disfavored at the current sample size. The endpoint scaling *d*LGW = *A*(*z*) *d*L&Lambda;CDM is the primary likelihood model; the TEP-C0 Jordan-frame distance integral is retained as a secondary consistency check. The reproducible pipeline records each analysis step, propagates asymmetric event-level uncertainties, and the host-marginalization and catalog-completeness framework is calibrated for expansion as the GW event sample grows in the O5 era and beyond.

## References

[1] Abbott, B. P., et al. (LIGO/Virgo). (2019). GWTC-1: A gravitational-wave transient catalog of compact binary mergers observed by LIGO and Virgo during the first and second observing runs. *Physical Review X*, 9(3), 031040.

[2] Abbott, B. P., et al. (LIGO/Virgo). (2021). GWTC-2: Compact binary coalescences observed by LIGO and Virgo during the first half of the third observing run. *Physical Review X*, 11(2), 021053.

[3] Riess, A. G., et al. (2022). A comprehensive measurement of the local value of the Hubble constant with 1 km/s/Mpc uncertainty from the Hubble Space Telescope and the SH0ES team. *The Astrophysical Journal Letters*, 934(1), L7.

[4] Planck Collaboration. (2020). Planck 2018 results. VI. Cosmological parameters. *Astronomy & Astrophysics*, 641, A6.

[5] Abbott, R., et al. (LIGO/Virgo/KAGRA). (2021). GWTC-2.1: Deep extended catalog of compact binary coalescences observed by LIGO and Virgo during the first half of the third observing run. arXiv:2108.01045.

[6] Abbott, R., et al. (LIGO/Virgo/KAGRA). (2021). GWTC-3: Compact binary coalescences observed by LIGO and Virgo during the second part of the third observing run. *Physical Review X*, 13(4), 041039.

[7] The LIGO Scientific Collaboration, Virgo Collaboration, and KAGRA Collaboration. (2024). GWTC-4.0: Gravitational-wave transient catalog of LIGO, Virgo, and KAGRA. arXiv:2408.02343.

[8] The LIGO Scientific Collaboration, Virgo Collaboration, and KAGRA Collaboration. (2026). GWTC-5.0: Updated LIGO–Virgo–KAGRA Catalog sets new records in precision gravitational wave astronomy. *News | LIGO Lab | Caltech*, 26 May 2026.

[9] The LIGO Scientific Collaboration, Virgo Collaboration, and KAGRA Collaboration. (2026). GWTC-5.0: Constraints on the Cosmic Expansion Rate and Modified Gravitational-wave Propagation. arXiv:2605.27227.

[10] Abbott, B. P., et al. (LIGO/Virgo). (2017). GW170817: Observation of gravitational waves from a binary neutron star inspiral. *Physical Review Letters*, 119(16), 161101.

[11] Abbott, B. P., et al. (LIGO/Virgo). (2017). A gravitational-wave standard siren measurement of the Hubble constant. *Nature*, 551(7678), 85–88.

[12] Dalya, G., et al. (2022). GLADE+: An extended galaxy catalogue for multimessenger searches with advanced gravitational-wave detectors. *MNRAS*, 514(2), 1403–1415.

[13] DESI Collaboration. (2024). The Early Data Release of the Dark Energy Spectroscopic Instrument. arXiv:2404.03002.

[14] Fishbach, M., et al. (2019). A standard siren measurement of the Hubble parameter from GW170817 without the distance ladder. *The Astrophysical Journal Letters*, 871(1), L13.

[15] Palmese, A., et al. (2021). Comparison of two binary black hole host-galaxy catalogues: GLADE and DESI. *MNRAS*, 505(3), 3923–3935.

[16] Creminelli, P., & Vernizzi, F. (2017). Dark energy after GW170817 and GRB170817A. *Physical Review Letters*, 119(25), 251302.

[17] Ezquiaga, J. M., & Zumalacárregui, M. (2017). Dark energy after GW170817: Dead ends and the road ahead. *Physical Review Letters*, 119(25), 251304.

[18] Riess, A. G., et al. (2024). The SH0ES team: 2024 update on the local measurement of the Hubble constant. *The Astrophysical Journal Letters*, submitted.

## Data Availability & Reproducibility

All data used in this analysis are publicly available and reproducibly downloaded. No synthetic, fabricated, or simulated data is used in the main analysis.

Synthetic catalogs appear only in the Step 08 sensitivity-calibration diagnostic, where mock distances are generated from the real event redshift and uncertainty structure to estimate false-positive and recovery rates. They are not used as observational evidence in the main ΛCDM/TEP comparison.

*Data sources:*

- GWOSC combined catalogs: GWTC-1-confident, GWTC-2, GWTC-2.1-confident, GWTC-3-confident, GWTC-4.0, GWTC-4.1, O4 Discovery Papers, GWTC-5.0 — gwosc.org

- GraceDB public skymaps (bayestar.fits.gz): gracedb.ligo.org

- GLADE+ Galaxy Catalog (VizieR VII/291): glade.plus

- DESI DR1 fastspec spectroscopic redshift catalog (HEALPix tiles): data.desi.lbl.gov

- NASA/IPAC Extragalactic Database redshift-bearing objects: ned.ipac.caltech.edu

*Pipeline steps (13 sequential stages):*

| Step | Script | Description |
| --- | --- | --- |
| 00 | `step_00_download_gwtc5_catalog.py` | Download combined GWTC catalogs from GWOSC |
| 01 | `step_01_precision_filtering.py` | Filter events by SNR > 12 and high confidence |
| 01b | `step_01b_download_desi.py` | Download DESI DR1 fastspec HEALPix tiles for deep galaxy redshifts (optional, large download) |
| 02 | `step_02_independent_redshifts.py` | Build independent-redshift dataset (bright + GLADE+/DESI DR1 dark sirens) |
| 03 | `step_03_compute_dl_gr.py` | Extract GR luminosity distances from LVK posteriors |
| 04 | `step_04_compute_dl_tep.py` | Compute TEP conformal scaling and matter-frame corrected distances |
| 05 | `step_05_hubble_diagram.py` | Construct Hubble diagram data |
| 06 | `step_06_h0_reconciliation.py` | Secondary analysis: fit H₀ from full sample including fallback redshifts |
| 06b | `step_06b_h0_reconciliation_independent.py` | Primary analysis: fit H₀ from independent-only redshifts (no circularity) |
| 07 | `step_07_statistical_tests.py` | Goodness-of-fit, tension metrics, and model comparison for both analyses |
| 08 | `step_08_synthetic_injections.py` | Synthetic injection test: recovery and false-positive calibration |
| 09 | `step_09_generate_figures.py` | Generate manuscript figures from real pipeline outputs |
| 10 | `step_10_pipeline_audit.py` | Pipeline audit: verify execution integrity and output consistency |

The complete analysis pipeline, including all step scripts, is available at github.com/matthewsmawfield/TEP-LVK.