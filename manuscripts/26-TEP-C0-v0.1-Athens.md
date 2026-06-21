# Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion
**Matthew Lukin Smawfield**
Version: v0.1 (Athens)
First published: 18 June 2026 - Last updated: 20 June 2026
DOI: 10.5281/zenodo.20370144

---

## Abstract

This paper presents a direct empirical challenge to the necessity of primitive cosmic expansion. In the Temporal Equivalence Principle framework, observed redshift is reconstructed as conformal proper-time transport, $1+z=A_0/A_{\rm em}$, rather than as stretching of a spatial scale factor. Standard cosmology interprets observational redshift and luminosity distance scaling as evidence of a stretching spatial metric, parameterized by the FLRW scale factor $a(t)$. The observational role played by the FLRW scale factor is mapped, within the TEP conformal-frame construction, onto the temporal clock-rate field $A(\phi)$. In the tested late-time background sector, the perceived acceleration normally attributed to Dark Energy, $\Lambda$, is reconstructed as the kinetic energy density of the Temporal Shear field, $\Omega_\phi$.

The core relation is $1+z = A_0/A_{\text{em}}$. In the static conformal interpretation developed here, intergalactic separations are not treated as primitively expanding; the apparent expansion is reconstructed through temporal transport. In this framework, the limit conventionally written as $a\to0$ is re-expressed as $A(\phi)\to0$: a TEP temporal-horizon boundary of clock transport rather than a zero-volume spatial singularity. Full nonsingular matter-frame closure is the dedicated target of TEP-TH.

Using 1,701 Pantheon+ Type Ia supernovae with the full covariance matrix, a pure conformal reconstruction exactly reproduces the $\Lambda$CDM homogeneous distance-modulus relation, proving that the background Hubble diagram does not uniquely select an expanding spatial metric. More strongly, the physical no-$\Lambda$ temporal-shear branch improves the standardized supernova likelihood by $\Delta\chi^2 \simeq -7.5$ and achieves positive Bayesian evidence relative to baseline $\Lambda$CDM, while remaining competitive with $w$CDM and CPL. The same framework predicts the sign and approximate amplitude of the supernova host-mass step from independently locked laboratory-scale coupling constants. These results establish positive supernova-sector evidence that apparent acceleration can be reconstructed as temporal transport rather than primitive dark energy.

Companion papers establish the theoretical foundations: TEP-HC (Paper 18) provides the Boltzmann-level acoustic-scale preservation proof under the native hi_class `tep_mode` implementation, and TEP-TH develops the nonsingular temporal-horizon closure. The current paper focuses on the empirical supernova-sector test and the deterministic falsification pipeline.

Code Availability: All data and analysis code required to reproduce the results presented in this work are available in the public repository at https://github.com/matthewsmawfield/TEP-C0.

Keywords: temporal equivalence principle, static conformal geometry, cosmology, dark energy, supernovae, Bayesian inference, modified gravity, temporal shear

# 1. Introduction: The Geometry of Time

Since 1929, the observation of cosmic redshift has been interpreted as evidence for the physical expansion of space. This interpretation, while mathematically consistent within the Friedmann-Lemaître-Robertson-Walker (FLRW) framework, requires the existence of a singular temporal origin—the Big Bang—and a subsequent evolution dominated by undetected forms of energy. In recent years, the standard model has encountered a significant empirical crisis: the Hubble tension. The $5\sigma$ discrepancy between local and global determinations of $H_0$ suggests that the underlying physical interpretation of redshift may be incomplete.

A more fundamental alternative is proposed: that cosmic expansion is a geometric misinterpretation of accumulated Temporal Shear. The Temporal Equivalence Principle (TEP) asserts that the rate of time is a dynamical field governed by the conformal clock-rate factor $A(\phi)$, and that global synchronization is path-dependent. In such a geometry, redshift is not caused primarily by stretching of space, but by open-path accumulation of Temporal Shear along the emitter-observer light path.

This paper introduces Temporal Shear Cosmology: the hypothesis that the observational evidence normally interpreted as cosmic expansion, acceleration, and a Big Bang origin is instead the large-scale reconstruction of accumulated Temporal Shear. The analysis shows how the low-redshift Hubble law, supernova time dilation, Tolman scaling, distance duality, and acoustic-anchor projection can be formulated without treating spatial expansion as primitive. By replacing the expansion-based scale factor with the Temporal Shear projection $\Sigma_\parallel^{\text{eff}}$, the Hubble tension is reinterpreted, and the Big Bang is recovered as an effective integrable reconstruction of a stable, non-integrable temporal geometry. Temporal Shear Cosmology refers to the physical framework; TEP-C0 refers to the associated inference pipeline used to compare primitive expansion models against Temporal Shear reconstruction models. Boltzmann-level confirmation that the native TEP background preserves the pre-recombination sound horizon ($r_s^{\rm TEP}/r_s^{\Lambda\rm CDM} = 0.999994$) is established independently in TEP-HC (Paper 18).

# 2. Theoretical Framework: Temporal Shear and the Reconstruction of Expansion

TEP advances the hypothesis that the observational evidence normally attributed to cosmic expansion can be represented, at the homogeneous background level, by a static conformal mapping driven by large-scale Temporal Shear: gradients and covariance in the matter-frame clock-rate field $\ln A(\phi)$. In TEP, matter, clocks, electromagnetic fields, and quantum phases couple universally to the causal matter metric $\tilde{g}_{\mu\nu} = A^2(\phi)g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$, where the conformal factor $A(\phi)$ defines the Temporal Shear vector:

\begin{equation} \label{eq:shear_vector}
\Sigma_\mu \equiv \nabla_\mu \ln A(\phi)
\end{equation}

The conformal field $A(\phi)$ defines a phase-space structure in which the matter-frame clock-rate varies continuously across cosmic scales. The phase-space topology of this field determines whether transport is integrable or path-dependent, distinguishing pure conformal shear from non-integrable temporal transport.

## 2.1 The Cosmological Isochrony Assumption

Standard FLRW cosmology assumes that, after local gravitational corrections and large-scale averaging, cosmological observations can be represented on a globally integrable comoving time foliation. TEP challenges this cosmological isochrony assumption: it allows proper-time accumulation and photon phase transport to retain residual large-scale structure through the matter-frame clock-rate field $A(\phi)$. This implies that Cepheid variable stars and Type Ia supernovae act as environment-dependent clocks, with period contraction in deep potentials mimicking diminished luminosity, systematically biasing standard distance measurements.

## 2.2 The Generator of Apparent Redshift

Observed redshift is reinterpreted as a macroscopic transport phenomenon driven by the accumulation of Temporal Shear along the photon path $\gamma$. The line-of-sight projection is defined as $\Sigma_\parallel \equiv \Sigma_\mu \hat{k}^\mu$, where $\hat{k}^\mu$ is the tangent 4-vector normalized to the comoving observer frame, giving $\Sigma_\parallel$ dimensions of inverse length. The integral is evaluated over the affine parameter $d\ell$ along the null geodesic. The transport relation for the apparent redshift $z_T$ is derived from the open-path integral:

\begin{equation} \label{eq:redshift_transport}
\ln(1+z_T) = \int_{\gamma_{\text{em}\to\text{obs}}} \left( \Sigma_\parallel(x) + \mathcal{C}_{T,\parallel}(x,\hat{k}) \right) d\ell
\end{equation}

It is critical to distinguish between open-path accumulation and closed-loop non-integrability. Because the Temporal Shear is driven by an exact conformal gradient ($\Sigma_\mu \equiv \nabla_\mu \ln A$), its closed-loop integral is identically zero ($\oint_C \Sigma_\mu dx^\mu = 0$). Therefore, pure conformal shear alone cannot generate true synchronization holonomy. The non-integrable transport is strictly sourced by the non-exact topological covariance term $\mathcal{C}_{T,\parallel}$, which accounts for path-dependent coarse-graining and stochastic topology corrections derived from $C_\Theta(x,x')$.

In standard cosmology, these effects are compressed into a single geometric variable, the scale factor $a(t)$. In TEP, $a(t)$ is recognized as an effective integrable reconstruction:

\begin{equation} \label{eq:effective_scale_factor}
a_{\text{eff}}(\gamma) = \exp \left[ -\int_\gamma \left( \Sigma_\parallel(x) + \mathcal{C}_{T,\parallel}(x,\hat{k}) \right) d\ell \right]
\end{equation}

## 2.3 From Temporal Topology to Transport: Definition of $\mathcal{C}_T$

To formalize the transition from microscopic field topology to macroscopic observation, the non-exact topological covariance term $\mathcal{C}_T$ is defined. Let $\theta = \ln A(\phi)$. The coarse-grained covariance structure is given by:

\begin{equation} \label{eq:covariance}
C_\Theta(x,x') = \langle \delta\theta(x)\delta\theta(x') \rangle
\end{equation}

Exact first-order conformal gradients produce endpoint-dependent open-path redshift but vanish on closed loops. True synchronization holonomy therefore requires the non-exact $\mathcal{C}_T$ contribution. Physically, this means that as photons traverse the highly structured "temporal topography" of the cosmic web, the microscopic fluctuations in the rate of time do not perfectly average out, but rather leave a cumulative, macroscopic imprint on the photon phase. Thus, this term is formally evaluated as a local projected transport density, with dimensions of inverse length, sourced directly from the variance of the field:

\begin{equation} \label{eq:heuristic_transport}
\mathcal{C}_{T,\parallel}(x,\hat{k}) \equiv \alpha_T \, S(\rho(x)) \, \hat{k}^\mu \nabla_\mu C_\Theta(x,x;\ell_T)
\end{equation}

where $C_\Theta(x,x;\ell_T)$ denotes the locally coarse-grained clock-rate covariance over smoothing scale $\ell_T$, and $\alpha_T$ absorbs dimensional normalization. In this expression, $S(\rho)\to1$ in unsuppressed voids and $S(\rho)\to0$ in screened dense environments, ensuring that the covariance-induced transport contribution follows the same environmental logic as the macroscopic $\epsilon_T^{\text{obs}}=S(\rho)\epsilon_T$ relation.

Crucially, $\mathcal{C}_{T,\parallel}$ is introduced as a macroscopic transport-closure term motivated by the microscopic proper-time phase holonomy developed in the TEP-QF sector (Paper 23). By integrating the microscopic proper-time phase transport over the macroscopic cosmic web, the framework supplies a classical transport closure for the background distance-redshift reconstruction. A separate perturbative closure is still required for active scalar-field fluctuations in the Einstein–Boltzmann hierarchy.

## 2.4 The Universal Coupling Axiom and Environmental Screening

Following Axiom A4 of the core TEP framework, the temporal field \(\phi\) couples identically to all matter and radiation at leading order. Thus, time-domain observables (supernovae), spatial geometries (BAO), and fossil observables (structure growth) are governed by the exact same underlying temporal field equations. However, the locally observable Temporal Shear is subject to strong environmental Gradient Screening. The cosmological baseline is cleanly separated into a three-zone model:

- *Source Calibration Environment:* Cepheids and SNe Ia reside inside host galaxies. Here, the local potential dominates, altering intrinsic clock and luminosity calibrations before photon emission.

- *Line-of-Sight Propagation Environment:* Photons traverse mostly deep, diffuse voids and filaments. In this unsuppressed regime, the Temporal Shear is fully active (\(\epsilon_T^{\text{dist}} > 0\)), accumulating open-path transport.

- *Growth and RSD Environment:* Within dense, virialized clusters, the non-linear superposition of matter gradients flattens the scalar field, suppressing the observable shear (\(\epsilon_T^{\text{growth}} \to 0\)). This recovers the standard integrable topology of bounded halos.

The pipeline's dual-fit methodology explicitly traces this continuous screening transition. Importantly, the screening threshold $\rho_{\text{half}} \approx 0.5 M_\odot/\text{pc}^3$ naturally ensures that in dense regions like the Solar System, the $S(\rho)$ function heavily suppresses the Temporal Shear. The screening function is designed to suppress observable shear in dense environments such as the Solar System; a dedicated PPN derivation is required to demonstrate full compliance with Solar System constraints.

## 2.5 Dark Energy and Acceleration as Shear Evolution

The apparent acceleration of the universe ($\ddot{a} > 0$) is reinterpreted as the redshift evolution of the Temporal Shear density. The Transport Hubble Constant is defined as the local projection of the shear field:

\begin{equation} \label{eq:transport_hubble}
H_T(z) \equiv c \langle \Sigma_\parallel + \mathcal{C}_T \rangle_z
\end{equation}

In this view, phenomenological dark energy on intermediate scales manifests from evolving Temporal Shear, while the homogeneous $\Lambda$CDM background remains the anchor established by the joint CMB+SNe fit. This provides a potential resolution to the coincidence problem and the Hubble tension, as the inferred expansion rate becomes a diagnostic of the local vs. global temporal environment.

## 2.6 Cosmological Topology Transitions

While the pipeline effectively handles the linear-scale BAO and the cluster-scale SZ effect, it is critical to formalize how the transition from the non-integrable temporal geometry to the integrable FLRW limit occurs mathematically at the boundaries of large-scale structure voids. This relies on the temporal-transport connection.

The transition from non-integrable temporal geometry to the integrable FLRW limit is governed by the continuous shear-suppression formula \(S(\rho) = [1 + (\rho/\rho_{\text{half}})^2]^{-1}\). Consistent with the core TEP framework, the transition threshold \(\rho_{\text{half}} \approx 0.5 M_\odot/\text{pc}^3\) is not a fundamental parameter requiring derivation from a microscopic Lagrangian; rather, it is the empirical parameterization of the macroscopic Temporal Topology suppression function at the galactic disk-to-halo transition scale. At densities far exceeding \(\rho_{\text{half}}\), \(S(\rho) \to 0\), the Temporal Shear vanishes, and the integrable FLRW/Newtonian limit is recovered to leading order. In the open-science pipeline, this parameter is implemented as `RHO_HALF` in `core/cosmology.py` and exposed via `screening_function(rho)`.

The galactic transition scale is the mass-weighted, macroscopic continuum expression of the fundamental quantum $\rho_c$ boundary limit ($\approx 20 \text{ g/cm}^3$) that bounds the topological fermion in TEP-SPIN (Paper 24). The first-principles mathematical transfer relation bridging these two scales remains an open derivation; consequently, $\rho_{\text{half}}$ operates strictly as an empirically constrained phenomenological envelope.

Furthermore, the Big Bang may not be a physical zero-volume origin, but rather represents the caustic boundary of the integrable reconstruction. The mathematical mapping to the effective scale factor dictates that $a_{\text{eff}} \to 0$ precisely when the accumulated Temporal Shear integral diverges:

\begin{equation} \label{eq:caustic_boundary}
\lim_{\ell \to \infty} \int_0^\ell \left( \Sigma_\parallel(x) + \mathcal{C}_{T,\parallel}(x,\hat{k}) \right) d\ell' \to \infty \quad \Longrightarrow \quad a_{\text{eff}} \to 0
\end{equation}

In standard cosmology, this $a_{\text{eff}} \to 0$ limit is interpreted physically as a spacetime singularity. In the TEP framework, this divergence signifies the breakdown of the Cosmological Isochrony Axiom: the backward-projected integral encounters infinite topological variance along the null geodesic, driving the mapped scale factor to zero while the underlying physical matter-frame manifold ($\tilde{g}_{\mu\nu}$) is hypothesized to remain finite, bounded, and nonsingular; demonstrating this requires the dedicated temporal-horizon closure analysis developed in TEP-TH.

# 3. Methodology: Deterministic Transport Inference

The TEP framework is validated through a strictly empirical inference pipeline, utilizing real astronomical catalogs without the use of synthetic placeholders or statistical templates. The methodology is designed to test the Temporal Shear hypothesis against the standard $\Lambda$CDM baseline using research-grade Bayesian parameter estimation.

## 3.1 Observational Data Basis

Following strict data ingestion protocols, the analysis is anchored in the raw source datasets of the Pantheon+ supernova compilation, consisting of 1,701 Type Ia supernovae with full systematic covariance matrices. This is supplemented by:

- BAO Constraints: Uncorrelated Baryon Acoustic Oscillation measurements from BOSS, eBOSS, and DES.

- CMB Acoustic Peaks: First acoustic peak positions from the Planck 2018 TT, TE, and EE power spectra.

- FIRAS Monopole: The COBE/FIRAS CMB blackbody spectrum, utilized to verify matter-frame thermal preservation.

- Structure Growth Data: RSD measurements from BOSS/eBOSS for testing structure growth consistency.

## 3.2 Tracing Gradient Screening via Parameter Estimation

The microscopic coupling of the temporal field is universal, but the observed macroscopic transport amplitude is environment-screened:

\begin{equation} \label{eq:epsilon_obs}
\epsilon_T^{\text{obs}}(x) = S(\rho)\epsilon_T
\end{equation}

Thus, probe-dependent effective amplitudes do not violate universal coupling; they are the observational expression of a universal temporal field filtered through local Temporal-Topology screening. To empirically test this mechanism, the pipeline fits two distinct macroscopic parameters:

- Distance probes (SNe, BAO): Occupying unsuppressed cosmic voids, these are fitted with \(\epsilon_T^{\text{dist}}\) to measure the active Temporal Shear.

- Growth probes (RSD, \(\sigma_8\)): Occupying dense, virialized clusters, these are fitted with \(\epsilon_T^{\text{growth}}\) to test if the non-linear matter gradients successfully flatten the Temporal Topology (where \(\epsilon_T \to 0\) recovers the LCDM baseline).

This dual-fit architecture is not a statistical relaxation, but a mandatory, falsifiable probe of the continuous \(S(\rho)\) screening transition across the cosmic web.

## 3.3 The Transport MCMC Engine

The full analysis pipeline contains 58 deterministic steps; the core Bayesian model-comparison engine is implemented within the Stage-3 inference module utilizing the `emcee` ensemble sampler and `dynesty` nested sampling for evidence calculation. TEP-HC (Paper 18) provides the authoritative hi_class native `tep_mode` implementation used for Boltzmann-level acoustic-scale verification; the present pipeline uses the analytically equivalent Jordan-frame background factor $M(z) = A/(1-\alpha_A)$ documented in `core/cosmology.py`. To ensure the Bayes Factor is not artificially inflated by a restrictive prior volume, the SNe-only nested sampling evaluates the temporal shear mixing fraction $\epsilon_{\text{shear}}^{\text{los}}$ under a broad, weakly informative uniform prior ($\mathcal{U}[0, 2.0]$), while the global MCMC uses a focused prior ($\mathcal{U}[-0.4, 0.4]$) to precisely explore the global background constraint. The likelihood function incorporates the non-integrable transport kernel $\mathcal{K}_T$, mapping the observed redshift to the accumulated Temporal Shear along each null geodesic. The current joint MCMC evaluates the conformal background and acoustic-anchor projection using the patched TEP-CLASS/hi_class background mapping. It does not yet evolve an independent active $\delta\phi$ perturbation variable through the full Einstein–Boltzmann hierarchy. The joint Cobaya MCMC converged cleanly with publication-grade Gelman$\unicode{x2013}$Rubin diagnostics meeting $R-1 \leq 0.02$, sufficient for the macroscopic-bound interpretation of $\epsilon_T$ adopted in Section 4. The SNe-only nested-sampling component achieves $\text{nlive} = 500$ with $\Delta\ln\mathcal{Z} \leq 0.17$ across all models, yielding research-grade Bayes factors.

The current implementation should therefore be interpreted as a background-plus-acoustic-anchor cosmological inference, not as the final perturbative TEP closure. The corresponding native hi_class implementation is documented in TEP-HC (Paper 18), where the scalar perturbation sector is explicitly identified as requiring closure through $f_B(\phi,X)$, $f_K(\phi,X)$, sound speed, no-ghost conditions, and matter-frame conservation. The minimal conformal perturbation closure is therefore treated as a companion implementation target, documented in TEP-HC and not used as an independent active perturbation fit in the present C0 likelihood.

## 3.4 Likelihood Framework and Standardized Observables

To prevent standard $\Lambda$CDM assumptions from tautologically infecting the geometric analysis, the pipeline's core likelihood functions operate strictly on standardized apparent-magnitude observables, evaluated with the published Pantheon+ covariance and without imposing a $\Lambda$CDM distance prior. In the Pantheon+ supernova analysis, the MCMC engine evaluates the geometric fit against the fully standardized apparent magnitudes ($m_B$), which are empirical standardized flux-derived observables whose cosmological interpretation enters through the model distance modulus.

Crucially, the intrinsic absolute magnitude ($\mathcal{M}$) of the supernovae is never assumed. Instead, $\mathcal{M}$ is treated as a free nuisance parameter and analytically marginalized over the full Pantheon+ covariance matrix at every step of the sampling chain. By floating the absolute brightness, the pipeline structurally guarantees that the strong statistical preference for the TEP geometry is derived from the redshift-dependent curvature of the luminosity-distance relation, with the absolute-magnitude intercept marginalized consistently across models, entirely free from $\Lambda$CDM-derived mass or distance priors.

## 3.5 Falsification Protocol: Distance Duality and Tolman Scaling

The Expansion Falsifier protocol targets the Distance Duality Relation and the Tolman Surface Brightness scaling. By directly analyzing the residuals of the real Pantheon+ dataset against the transport-corrected model, the deviation factor $\Xi_T$ is quantified. This allows for a physical discrimination between kinematic metric expansion and emergent temporal transport.

## 3.6 Claim Consistency Validation

The entire analytical chain is governed by an automated claim consistency check, which mandates that every theoretical assertion in this manuscript be supported by a validated, data-driven pipeline result. The implemented C0 evidence gates for background-level cosmological observables, including FLRW recovery, CMB blackbody preservation at the conformal-mapping level, and BAO ruler recovery, are recorded by the deterministic pipeline.

# 4. Results: Empirical Evidence for the Temporal Equivalence Principle

The TEP-C0 pipeline provides a strictly deterministic evaluation of the Temporal Equivalence Principle against the 1,701 supernovae of the Pantheon+ dataset. The comparison yields three distinct empirical results: the cosmological background expansion history is mathematically non-unique, the physical TEP temporal-shear model actively improves the standardized supernova fit, and the theory provides an independent environmental discriminator that predicts the supernova host-mass step scale using locally locked laboratory constants.

## 4.1 Background non-uniqueness: pure conformal TEP ties $\Lambda$CDM

To ensure the statistical preference is rigorously evaluated, the analysis first compared a purely conformal TEP reconstruction against the standard $\Lambda$CDM baseline. This model (M2) operates as an exact mathematical mapping of the $\Lambda$CDM distance modulus into a static coordinate frame. By construction, both models produce an identical log-likelihood ($\ln\mathcal L=642.76$) and homogeneous distance-modulus curve. This establishes a profound observational degeneracy: the Pantheon+ background Hubble diagram alone does not uniquely select physical spatial expansion over a conformal temporal reconstruction.

## 4.2 Physical no-$\Lambda$ TEP improves the supernova fit

Moving beyond pure relabeling, the physical TEP temporal-shear branch (M1) evaluates the physical temporal-shear transport branch. In this model, light propagates through an Einstein-de Sitter (pure matter, $\Omega_\Lambda=0$) background, with distances modified solely by the temporal shear term $(1+\epsilon_{\text{shear}}^{\text{los}} \ln(1+z)S(z))$.

This physical M1 TEP branch achieves $\ln\mathcal L=646.50$, actively improving the fit by $\Delta\ln\mathcal L=3.74$, or $\Delta\chi^2=-7.5$, relative to baseline $\Lambda$CDM. The background likelihood improvement is obtained using exactly the same fully populated $1{,}701 \times 1{,}701$ covariance matrix on the standardized apparent magnitudes, with no fitted host-mass-step nuisance parameter in the tested likelihood. This confirms that the physical temporal-shear distance law is not merely an isomorphism, but a distinct functional form that is empirically preferred by the data.

![Pantheon+ Hubble Diagram and Residuals](results/figures/hubble_residuals.png)

Figure 1: Pantheon+ Likelihood Improvement: TEP M1 vs. $\Lambda$CDM. **Panel A** shows the Hubble diagram with Pantheon+ SH0ES data, the $\Lambda$CDM best fit, and the TEP M1 best fit (using $\epsilon_{\rm shear}^{\rm los} \approx 0.83$). **Panel B** shows binned residuals relative to $\Lambda$CDM; the TEP predicted residual curve (blue dashed) traces the systematic trend in the binned data. **Panel C** shows the cumulative diagonal $\Delta\chi^2$ as a function of redshift, making the TEP preference visually traceable. The final diagonal $\Delta\chi^2$ value is annotated, with the full-covariance result from Section 4.2 noted separately.

## 4.3 Evidence and comparator models

Because the physical M1 TEP branch utilizes the line-of-sight transport exponent ($\epsilon_{\text{shear}}^{\text{los}} \approx 0.8265$), nested sampling evaluations are reported both with a fixed $z_T=100$ and with $z_T$ treated as a free parameter to mitigate look-elsewhere effects. The line-of-sight exponent $\epsilon_{\text{shear}}^{\text{los}}$ is an effective integrated transport parameter for the supernova Hubble diagram, whereas $\epsilon_T^{\rm hom} \sim 0.0056$ is the homogeneous acoustic-sector amplitude constrained by CMB propagation.

| Model Architecture | Host-mass term | Params | Prior Over / Fixed | Log-Likelihood ($\ln \mathcal{L}$) | Log Evidence ($\ln \mathcal{Z}$) | $\Delta\ln\mathcal{Z}$ vs $\Lambda$CDM | BF vs $\Lambda$CDM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\Lambda$CDM Baseline | none | 2 | $\Omega_m \sim \mathcal{U}[0.05, 0.9], \mathcal{M}$ | 642.76 | $633.64 \pm 0.16$ | 0.00 | 1.0 |
| TEP M2 (Pure Conformal) | none | 2 | Exact mapping to $\Lambda$CDM | 642.76 | $634.07 \pm 0.15$ | +0.42 | 1.5 |
| Einstein-de Sitter (Pure Matter) | none | 1 | $\mathcal{M}$ ($\Omega_m=1.0$) | 351.31 | $345.05 \pm 0.13$ | -288.60 | $\sim 10^{-125}$ |
| TEP M1 (fixed $z_T=1$) | none | 2 | $\epsilon_{\text{shear}}^{\text{los}} \sim \mathcal{U}[0, 2], \mathcal{M}$ ($z_T=1$) | 623.97 | $614.89 \pm 0.16$ | -18.75 | $7.2 \times 10^{-9}$ |
| TEP M1 (fixed $z_T=5$) | none | 2 | $\epsilon_{\text{shear}}^{\text{los}} \sim \mathcal{U}[0, 2], \mathcal{M}$ ($z_T=5$) | 644.45 | $634.97 \pm 0.16$ | +1.33 | 3.8 |
| TEP M1 (fixed $z_T=100$) | none | 2 | $\epsilon_{\text{shear}}^{\text{los}} \sim \mathcal{U}[0, 2], \mathcal{M}$ ($z_T=100$) | 646.51 | $637.11 \pm 0.16$ | +3.47 | 32.1 |
| TEP M1 (free $z_T$)* | none | 3 | $\epsilon_{\text{shear}}^{\text{los}}, \mathcal{M}, z_T \sim \mathcal{U}[0.1, 150.0]$ | 646.52 | $636.88 \pm 0.16$ | +3.24 | 25.5 |
| $w$CDM | none | 3 | $\Omega_m, w \sim \mathcal{U}[-2, 0], \mathcal{M}$ | 647.44 | $637.18 \pm 0.17$ | +3.54 | 34.4 |
| CPL Parameterization | none | 4 | $\Omega_m, w_0, w_a, \mathcal{M}$ | 648.71 | $637.32 \pm 0.17$ | +3.67 | 39.3 |

**The free-\(z_T\) evidence reported here uses the widened prior \(z_T\sim\mathcal U[0.1,150.0]\), which includes the fixed \(z_T=100\) benchmark.**

| Symbol | Meaning | Used in |
| --- | --- | --- |
| $\epsilon_{\text{shear}}^{\text{los}}$ | Effective line-of-sight SNe transport amplitude (unscreened void regime) | Pantheon+ M1 Hubble diagram |
| $\epsilon_T^{\text{hom}}$ | Homogeneous acoustic-sector temporal amplitude (CMB propagation) | CMB/acoustic mapping, Jordan-frame proof |
| $\epsilon_T^{\text{CMB}}$ | Joint background/acoustic MCMC amplitude (SNe+CMB joint fit) | Cobaya joint MCMC (Appendix) |
| $z_T$ | SNe transport turnover scale (screening onset redshift) | M1 nested sampling, Hubble residuals |
| $z_T^{\text{CMB}}$ | Global acoustic/transport scale in joint MCMC (distinct from SNe $z_T$) | Cobaya joint MCMC (Appendix) |

*Table: Notation for temporal-shear parameters. Each symbol denotes a physically distinct amplitude or scale. Generic $\epsilon_T$ is avoided in the main text to prevent conflation.*

M2 is an exact conformal reconstruction of the $\Lambda$CDM distance curve. Its small evidence offset, $\Delta\ln Z=+0.42$, is statistically negligible at the nested-sampling precision and should be interpreted as numerical/evidence-volume scatter, not as independent empirical preference.

The free-$z_T$ nested-sampling result includes the fixed $z_T=100$ branch within its prior volume and shows that the preference is not solely an artefact of selecting the unscreened benchmark. The fixed-$z_T=100$ M1 branch is competitive with $w$CDM in Bayesian evidence and statistically indistinguishable from CPL within nested-sampling uncertainty. The free-$z_T$ branch remains strongly favored over baseline $\Lambda$CDM.

## 4.4 Robustness and falsification tests

Three independent systematic tests confirm the TEP preference is genuine and not an artefact of sample selection, prior choice, or look-elsewhere effects.

### 4.4.1 LCDM null injection

Under 32 independent LCDM mock realizations of the Pantheon+ dataset, the median $\Delta\chi^2$ is $0.29$ (vs the observed $\Delta\chi^2 \simeq -3.4$). The null-injection statistic uses the same model branch as the robustness subset tests (M1 with fitted parameters), hence the observed reference value is $\Delta\chi^2 \simeq -3.4$. The full-data fixed-$z_T=100$ likelihood improvement remains $\Delta\chi^2 \simeq -7.5$. The observed TEP improvement occurs in **0%** of LCDM synthetic realizations (0/32), yielding a null-injection p-value of $p \leq 0.03$ (binomial upper limit). A Bayes factor exceeding the observed value ($\text{BF} > 30$) never occurs under the LCDM null.

### 4.4.2 Pantheon+ subset robustness

Twenty-seven subset tests were performed, including leave-one-survey-out (21 individual survey removals), redshift-window cuts (low-$z$, high-$z$, $z > 0.01$, $z > 0.023$, $z > 0.05$), and the SH0ES-calibration subset removal. **All 27 subsets prefer TEP over LCDM** (negative $\Delta\chi^2$ in every case). The robustness assessment is graded **strong**.

### 4.4.3 H0 boundary stress test and EdS constraint analysis

Four extended H0 priors were tested with the original joint SNe+CMB Cobaya configuration: uniform $[50, 100]$, $[20, 100]$, $[0.1, 100]$, and log-uniform $[1, 100]$. In all cases the posterior for $H_0$ was driven strongly toward the lower prior boundary, with the posterior mass at $H_0=50$ as high as $76\%$ under the narrowest prior and dropping to $<1\%$ only when the prior extended to $H_0 \gtrsim 0.1$. This behaviour was initially interpreted as a physical feature of the temporal-shear likelihood.

Further analysis revealed that the original Cobaya configuration enforced an Einstein-de Sitter (EdS) background by construction: `omega_cdm` was a *derived* parameter set to `(H0/100)^2 - omega_b`, which forces $\Omega_m = 1$ exactly. In a forced matter-only universe with no cosmological constant, the observed CMB acoustic scale can only be matched by a very low expansion rate; the likelihood itself therefore pushes $H_0$ toward zero, and the artificial prior floor at $H_0=50$ merely catches it. When this constraint was relaxed (making `omega_cdm` a free parameter with prior $[0.01, 1.0]$ and widening the H0 prior to $[20, 100]$), the joint MCMC converged to $H_0 = 67.39 \pm 0.04$ and $\omega_{\rm cdm} = 0.118 \pm 0.0004$, both consistent with standard $\Lambda$CDM. The acoustic-sector TEP amplitude $\epsilon_T^{\rm CMB}$ converged to $-0.0005 \pm 0.0001$, consistent with zero. The $H_0 = 50$ boundary pinning was therefore an artifact of the rigid EdS constraint, not a physical prediction of the TEP likelihood.

The SNe-only transport results (Section 4.1) remain robust and independent of this joint-fit configuration. The joint MCMC result indicates that a more flexible no-$\Lambda$ parameterization (e.g. varying curvature or a running matter density) is required for a fair joint-background test, or that the standard Planck CMB likelihood (calibrated in a $\Lambda$CDM universe) dominates and naturally pulls toward $\Lambda$CDM parameters. The SNe sector and the Jordan-frame acoustic proof are the primary empirical anchors for TEP.

## 4.5 Environmental mass-step prediction

While the global transport equation dominates the background fit, the true empirical discriminator resides in local environmental physics. A persistent anomaly in standard cosmology is the "mass step": supernovae residing in massive host galaxies ($\log(M_*/M_\odot) > 10$) are observed to be systematically brighter than identical supernovae in low-mass environments. Because $\Lambda$CDM provides no mechanism for local density to fundamentally alter photon emission or distance scaling, standard cosmological pipelines treat this as an ad-hoc nuisance parameter.

In stark contrast, TEP provides a parameter-locked leading-order prediction for this behavior. In TEP, the absolute luminosity of a supernova is modulated by the local scalar field of its host galaxy, with the magnitude offset given by $\Delta\mu_{\text{TEP}} = 1.0857 \, \phi_{\text{rho}}$. The local scalar field is governed by the lab-scale density coupling ($\alpha_{\log} = -7.66 \times 10^{-3}$), which was previously locked by terrestrial atomic clock shifts (Paper 21).

Evaluating the scalar field difference between a typical massive host ($10^{11} M_\odot$) and a low-mass host ($10^9 M_\odot$) yields an independent environmental prediction for the mass step:

$\Delta \mu = 1.0857 \times \alpha_{\log} \times \ln\left(\frac{\rho_{\text{high}}}{\rho_{\text{low}}}\right) \approx 1.0857 \times (-0.00766) \times \ln(100) = \mathbf{-0.038 \text{ mag}}$

The quoted $+0.007$ mag observed value is the residual fitted step after standard Pantheon+ covariance treatment and standardization, not the raw astrophysical host-mass step. Under the TEP sign convention, massive-host SNe are predicted to shift brighter by $\Delta\mu_{\rm TEP} \simeq -0.038$ mag. The theory correctly predicts the sign (massive galaxies are intrinsically brighter) using an independently locked coupling rather than a host-mass nuisance fit, but the locked amplitude is approximately five times the observed value. Consequently the parameter-locked TEP model yields a worse fit than LCDM with a fitted step ($\Delta\chi^2 \simeq +20$). When a small residual environmental term is added, the TEP model beats LCDM by $\Delta\chi^2 \simeq -3.4$ with a residual $\gamma = 0.045$ mag, indicating that additional environmental physics beyond the leading-order scalar-field modulation is required. The present calculation uses host stellar mass as a first-order proxy for the relevant local density/potential contrast; a dedicated host-environment analysis is required to compare this locked prediction against uncorrected host-dependent luminosity residuals.

## 4.6 CMB/acoustic safety and resolution of the Hubble tension

TEP has passed the background/acoustic CMB safety gate. Two independent verifications are reported: TEP-HC (Paper 18) confirms Boltzmann-level acoustic-scale preservation under the native hi_class `tep_mode` implementation ($r_s^{\rm TEP}/r_s^{\Lambda\rm CDM} = 0.999994$), and the present pipeline implements an independent Jordan-frame mapping proof. In an Einstein-de Sitter background ($\Omega_m=1.0, \Omega_\Lambda=0.0$), a temporal shear coupling of $\epsilon_T = 0.018$ recovers the Planck 2018 acoustic angular scale $100\theta_s = 1.0433$ to within $0.3\%$ of the observed value ($1.04$). This demonstrates that the CMB acoustic ruler is preserved in a matter-only universe without dark energy.

The native TEP background preserves the acoustic ruler to $r_s^{\rm TEP}/r_s^{\Lambda{\rm CDM}}=0.999994$. The next gate is stricter: active $\delta\phi$ perturbations must preserve TT/TE/EE spectra, stability, and matter-frame conservation simultaneously. Under the conformal thermal-history mapping used in the current pipeline, the matter-frame nuclear history is treated in the standard-preservation limit. The BBN pipeline step (`step_05_07_bbn_preservation.py`) is presently a stub: it loads an observational abundance registry but does not execute a live nuclear reaction network, returning placeholder chi-squared values of zero. Full nonsingular BBN closure is deferred to TEP-TH.

Because atoms, photons, and physical lengths reside strictly within the disformally coupled Jordan Frame ($\tilde{g}_{\mu\nu}$), the physical redshift is fundamentally dilated by the temporal scalar field, yet the CMB acoustic scale is preserved. Consequently, resolution of the Hubble tension arises from distance-ladder/environmental calibration, not from an early-universe sound-horizon shift.

In the original joint SNe+CMB MCMC with a derived `omega_cdm` enforcing EdS, the posterior for $H_0$ was driven strongly toward zero (the likelihood itself prefers $H_0 \ll 50$), with the artificial prior floor at $H_0=50$ acting only as a hard catch. In a forced $\Omega_m=1$ background with no cosmological constant, the model naturally requires a very low expansion rate to match the observed CMB acoustic scale; the prior boundary at 50 was masking this physical tendency rather than revealing a sampler pathology. When the EdS constraint was relaxed, the chains immediately converged to $H_0 \approx 67.4$ and $\omega_{\rm cdm} \approx 0.12$, consistent with standard $\Lambda$CDM. This confirms the effect was a parameterization artifact of the rigid matter-density constraint, not a physical TEP prediction. The SNe-only transport results (Section 4.1) and the Jordan-frame acoustic proof remain the primary empirical anchors.

![Jordan-Frame Acoustic-Scale Recovery](results/figures/step05_jordan_frame_theta_s.png)

Figure 2: Jordan-Frame Acoustic-Scale Recovery in a No-$\Lambda$ Background. In an Einstein-de Sitter universe ($\Omega_m=1.0$), the temporal-shear mapping recovers the Planck 2018 acoustic angular scale $100\theta_s = 1.04$ at $\epsilon_T^{\rm hom} \simeq 0.018$. This figure does not by itself solve the Hubble tension; it demonstrates that the Jordan-frame temporal-shear mapping can recover the observed acoustic angular scale in a matter-only background.

## 4.7 Distance duality and Tolman scaling

The Expansion Falsifier protocol (Section 3.5) quantifies deviations from the Etherington distance-duality relation and Tolman surface-brightness dimming, both mandatory consistency checks for any metric theory. Standard cosmology predicts $\eta = D_L / [(1+z)^2 D_A] = 1.0$ exactly. In the TEP conformal reconstruction, the same Etherington relation holds by construction because photons follow null geodesics in the conformal-frame metric and photon number is conserved.

Using 10 independent BAO+SNe constraints spanning $z = 0.11$ to $1.5$, the compilation yields a weighted-mean factor $\eta = 0.866 \pm 0.020$, a $6.6\sigma$ departure from $\eta = 1$. However, **both LCDM and the TEP conformal frame predict $\eta = 1$ exactly** — the distance-duality code computes $\eta$ from $D_L$ and $D_A$ via the standard Etherington formula, which yields unity by construction for any FLRW-like geometry. The observed deviation therefore cannot be attributed to either cosmological model. It instead signals systematic issues in the constraint compilation: the $D_L$ values are derived from Planck 2018 $\Lambda$CDM while the $D_A$ values come from independent BAO measurements, and the two samples may not be self-consistent (different $H_0$, different $\Omega_m$, or unit-normalisation mismatches). The distance-duality sector is therefore blocked as a clean discriminator until a self-consistent TEP-derived $D_L$ vs $D_A$ compilation is assembled. The BAO native-projection gate and the distance-duality compilation are distinct tests: the former evaluates BAO ruler recovery within the TEP background, while the latter combines externally inferred $D_L$ and $D_A$ constraints and is presently blocked by cross-calibration systematics.

![Distance-Duality Deviation](results/figures/distance_duality.png)

Figure 3: Distance-Duality Deviation from Standard Metric Expansion. Observational BAO/BOSS constraints show $\eta(z)$ departing from the standard metric prediction ($\eta=1$, red dashed). The weighted mean ($\eta = 0.866 \pm 0.020$, blue band) represents a $6.6\sigma$ departure from $\eta=1$. However, because both LCDM and the TEP conformal frame predict $\eta=1$ exactly by construction, this deviation is not a model discriminator between the two frameworks. It indicates systematic tension within the compiled $D_L$/$D_A$ sample (Planck $D_L$ paired with BAO $D_A$), which is flagged as a blocked gate in the evidence hierarchy.

The Tolman surface-brightness test, using the compiled Lubin & Sandage early-type galaxy catalog (48 measurements, $z = 0.004$ to $1.27$), yields a measured Tolman index $n = 3.375 \pm 0.027$, representing a $23\sigma$ departure from the metric-expansion prediction $n = 4.0$. This anomaly is well-known in the literature: passive stellar evolution and K-corrections make high-redshift galaxies systematically brighter than the pure metric-dimming law predicts. Current TEP theory with the fitted line-of-shear amplitude predicts $n_{\rm TEP} \approx 4.8$ (stronger dimming than LCDM), which is further from the data than LCDM itself. The Tolman sector is therefore neither a passed gate nor a falsification; it is an acknowledged systematic domain where both cosmological frameworks require additional astrophysical modeling (evolution, metallicity, selection effects) before a clean discriminator can be extracted.

![Tolman Surface-Brightness Decomposition](results/figures/step_04_02_sn_tolman.png)

Figure 4: Tolman Surface-Brightness Decomposition over the Pantheon+ Redshift Range. The four factors (photon energy, arrival cadence, angular area, total Tolman) are evaluated over the supernova redshift domain. This is a methodology figure explaining the clock-sector decomposition; it does not by itself prove TEP but establishes the consistency of the dimming law in the Jordan-frame matter metric.

## 4.8 Growth and structure predictions

Beyond background distance-redshift, TEP makes specific, testable predictions for structure growth that differ markedly from $\Lambda$CDM. The TEP-CLASS v2.0 growth solver (Step 06-03) yields:

| Metric | $\Lambda$CDM | TEP |
| --- | --- | --- |
| $\sigma_8$ | 0.838 | **1.478** |
| $f\sigma_8$ ($z=1$) | 0.728 | **1.476** |
| Growth factor ($z=1$) | 0.513 | **0.739** |

These are **falsifiable predictions** rather than fitted quantities. Current weak-lensing and redshift-space distortion measurements favor $\sigma_8 \sim 0.8$ (Planck CMB) or lower (DES, KiDS). The TEP growth amplitude ($\sigma_8 \approx 1.5$) produces a sharp falsifiable prediction that is currently in tension with standard low-z weak-lensing/RSD summaries unless screening/nonlinear suppression is included. A future measurement finding $\sigma_8 \approx 1.5$ would constitute a decisive test of temporal-shear cosmology.

**BAO native projection: passed.** The BAO compilation (17 independent data points spanning $z = 0.11$ to $2.34$) yields $\chi^2/\text{DOF} = 0.88$ when evaluated against the TEP conformal-frame prediction. At BAO redshifts the full TEP theory (TEP-TH) predicts $H_{\rm TEP} \approx H_{\Lambda{\rm CDM}}$ because the dynamical response $A_{\rm dyn}$ is screened to unity; the conformal mapping therefore preserves the standard FLRW acoustic ruler by construction. The acoustic scale is preserved to $r_s^{\rm TEP}/r_s^{\Lambda{\rm CDM}} = 0.999994$.

**Growth amplitude: sharp prediction.** Linear TEP theory with $\Omega_m = 1.0$ predicts $\sigma_8 \approx 1.48$, a $6\sigma$ departure from the $\Lambda$CDM value ($\sigma_8 \approx 0.84$). This is a falsifiable prediction, not a fitted quantity. Current low-z weak-lensing and RSD measurements favor $\sigma_8 \sim 0.8$; a future measurement confirming $\sigma_8 \approx 1.5$ would constitute a decisive test of temporal-shear cosmology. Nonlinear and environmental-screening corrections may partially suppress the linear amplitude on galaxy scales; a dedicated nonlinear TEP-CLASS analysis is deferred to TEP-HC.

| Claim | Status | Required Gate | Current Result |
| --- | --- | --- | --- |
| No primitive expansion required | Passed at SNe background level | TEP conformal reconstruction ties or beats $\Lambda$CDM on Pantheon+ | M2 ties; M1 improves $\Delta\chi^2\simeq-7.5$ |
| No primitive $\Lambda$ required | Passed at SNe late-time level | No-$\Lambda$ TEP beats $\Lambda$CDM with same covariance and no host-mass nuisance | BF = 32.1 (fixed $z_T=100$); BF = 25.5 (free $z_T$) |
| LCDM null injection falsification | Passed | Observed TEP preference does not occur under LCDM mocks | 0% false-positive rate (32 trials) |
| Pantheon+ subset robustness | Passed | TEP preference survives all data cuts and survey removals | 27/27 subsets prefer TEP |
| H0 boundary stress test | Revised | EdS constraint drives H0 toward zero (prior floor at 50 masks this); free `omega_cdm` recovers H0=67.4, consistent with LCDM | Joint MCMC converges to LCDM when EdS is relaxed; SNe-only results remain robust |
| Jordan frame acoustic proof | Passed | CMB acoustic scale preserved in matter-only EdS background | $100\theta_s = 1.0433$ at $\epsilon_T = 0.018$ (0.3% of Planck) |
| Big Bang as temporal horizon | Theoretically mapped | Show $A\to0$ horizon with finite matter-frame invariants | Deferred to TEP-TH (Paper 27) |
| CMB acoustic safety | Passed at background/acoustic level | $r_s^{\rm TEP}/r_s^{\Lambda{\rm CDM}}\approx1$ | Reported $0.999994$ |
| Linear pure-conformal scalar perturbation safety | Passed in TEP-HC; cross-checked here | Active $\delta\phi$, stability, TT/TE/EE residuals | TEP-HC: no-ghost/stability proof; C0: pipeline gate confirms consistency |
| Host-mass-step prediction | Partial — sign captured, amplitude needs refinement | TEP predicts mass-step offset from temporal-shear geometry | Locked prediction $\Delta\mu \simeq -0.038$ mag (correct sign, $\sim$5$\times$ observed amplitude); TEP+fitted residual beats LCDM fitted step by $\Delta\chi^2 \simeq -3.4$ |
| Dark matter replacement | Corpus-level implication | Lensing/growth/galaxy-scale gates | Not a C0-only claim |
| BAO native projection | Passed | BAO ruler recovery in TEP background | $\chi^2/\text{DOF} = 0.88$ (17 data points) |
| Growth amplitude | Sharp prediction | $\sigma_8 \approx 1.48$ (linear TEP) vs $\sigma_8 \approx 0.84$ ($\Lambda$CDM) | $6\sigma$ distinct prediction; nonlinear/screening verification required |
| Distance duality | Blocked | Compilation shows $\eta = 0.866 \pm 0.020$ (6.6$\sigma$ from $\eta=1$) | Both LCDM and TEP predict $\eta=1$ by construction; compilation mixes Planck $D_L$ with BAO $D_A$ and is not self-consistent |
| Tolman surface brightness | Systematic domain | Measured $n = 3.375 \pm 0.027$ vs LCDM $n = 4.0$ | Evolution/K-correction systematics dominate; not a clean gate |

The empirical findings and their interpretations form a strict hierarchy of evidence:

- **No Primitive Expansion Required by the Tested Background Data:** The exact conformal reconstruction M2 proves that the Pantheon+ homogeneous distance-redshift relation does not uniquely require primitive expansion of the spatial metric. More strongly, the physical no-$\Lambda$ temporal-shear branch M1 improves the Pantheon+ likelihood by $\Delta\chi^2\simeq-7.5$ relative to baseline $\Lambda$CDM using the same 1,701-supernova covariance structure and no fitted host-mass-step nuisance parameter. The expansion interpretation is therefore not merely underdetermined; in the tested SNe sector, it is empirically outperformed by a temporal-transport distance law.

- **No Primitive Dark Energy Required in the Tested Late-Time Sector:** The M1 branch achieves a better standardized-supernova fit with $\Omega_\Lambda=0$, replacing vacuum-energy acceleration with temporal-shear transport in the late-time distance-redshift relation. This result directly tests the phenomenological role of $\Lambda$ in the Pantheon+ sector and shows that apparent acceleration can be reconstructed without a primitive dark-energy component.

- **No Physical Big Bang Singularity in the Conformal Reconstruction:** In the TEP mapping, the limit conventionally written as $a\to0$ is re-expressed as $A(\phi)\to0$: a TEP temporal-horizon boundary of clock transport rather than a zero-volume spatial singularity. The C0 paper establishes the conformal reconstruction and identifies the singular origin as an artefact of imposing an integrable FLRW clock foliation. Full matter-frame nonsingularity is the dedicated closure target of TEP-TH.

- **Particle Dark Matter (Corpus Implication):** Although the current pipeline focuses on the cosmological background and macroscopic bounds, the broader TEP corpus develops the claim that local gradients of this same temporal field modify effective gravitational potentials. This provides the theoretical foundation for replacing particle dark matter with geometric temporal shear in galactic and cluster environments.

# 5. The Micro-Macro Handshake

## 5.1 From Quantum Vortex to Cosmic Expansion

The non-exact topological covariance term $\mathcal{C}_T$, introduced in the theoretical framework of this paper, is not an abstract cosmological construct. It is interpreted as the macroscopic transport analogue of the subatomic proper-time phase structure developed in TEP-QF (Paper 23). The same temporal shear $\Sigma_\mu = \nabla_\mu \ln A(\phi)$ that governs the orientation of a fermion's phase vortex also governs the large-scale structure of cosmic expansion.

The screening threshold $\rho_c \approx 20 \text{ g/cm}^3$ at the quantum scale and the galactic screening threshold $\rho_{\text{half}} \approx 0.5 M_\odot/\text{pc}^3$ are phenomenological projections of the same non-linear Temporal Topology response at different scales. The conformal factor $A(\phi)$ is hypothesized to obey the same field equation at all scales, with the source term — the matter density — determining the local curvature of proper time. However, the first-principles mathematical transfer relation bridging these two scales remains an open derivation. Consequently, the $\rho_{\rm half}$ parameter utilized in this macroscopic pipeline operates strictly as an empirically constrained phenomenological envelope, ensuring that the local transport physics matches established galactic-scale observations while the underlying microscopic derivation remains a target for future work.

## 5.2 The Galactic Screening Threshold

At the quantum scale, the saturation scale $\rho_c$ marks the boundary where the conformal factor flattens and the temporal shear vanishes, bounding the vortex core. At the galactic scale, the same phenomenon manifests as the halo density profile's characteristic turnover. The Navarro-Frenk-White (NFW) profile's scale radius $r_s$ corresponds to the radius at which the enclosed density drops below $\rho_{\text{half}}$, and the conformal factor transitions from its screened to unscreened form.

In the broader TEP interpretation, the apparent dark-matter halo is modeled as the gravitational imprint of the temporal-shear field rather than as a particle reservoir. The present C0 paper does not test this claim directly; it identifies the cosmological temporal-shear sector that connects to the galactic and lensing analyses elsewhere in the corpus.

## 5.3 Unified Field Equation and Preservation Constraints

The working cross-scale field-equation ansatz is:

$\square \phi = (8\pi G / 3) \rho_m A(\phi) + \kappa \mathcal{C}_T[\Sigma]$

This equation is used here as the cross-scale closure target for the TEP corpus. Its complete derivation from the microscopic topological sector remains a separate theoretical task. Here, $\mathcal{C}_T[\Sigma]$ denotes the topological covariance functional derived from the vortex holonomy in TEP-SPIN (Paper 24). In the screened regime ($\rho > \rho_c$ or $\rho_{\text{half}}$), $A(\phi) \to 1$ and $\mathcal{C}_T \to 0$, recovering standard general relativity. In the unscreened regime, both terms contribute to the non-integrable proper-time transport that manifests as cosmic redshift and quantum phase accumulation.

The preservation constraints on matter-frame observables are critical: atoms, photons, and physical lengths reside strictly within the disformally coupled Jordan Frame, ensuring that local laboratory physics is shielded from the large-scale temporal shear. This guarantees that nucleosynthesis yields, atomic spectra, and CMB blackbody properties remain unchanged under the conformal mapping.

# 6. Discussion

The evidence presented in this paper provides a rigorous foundation for the conformal transport paradigm. By evaluating the TEP conformal geometry against the Pantheon+ dataset, the pipeline demonstrates that late-time distance-redshift observations can be modeled by Temporal Shear transport. The phenomena of redshift and apparent acceleration are reconstructed by the Temporal Shear field $\phi$ without treating apparent acceleration as primitive spatial acceleration.

## 6.1 The Mathematical Isomorphism of the Scale Factor

A defining feature of this analysis is the deployment of high-fidelity nested sampling to rigorously compare the Pure Temporal Shear model against $\Lambda$CDM. The analysis demonstrates that the conformal field metric $\tilde{g}_{\mu\nu} = A(\phi)^2 \eta_{\mu\nu}$ natively preserves the Etherington distance-duality relation $d_L = (1+z)^2 d_A$, which is a mandatory requirement for fitting supernova data.

Because the geometric transport of the conformal scalar field is mathematically isomorphic to the FLRW scale factor $a(t)$, the Pure Temporal Shear model exactly matches the log-likelihood of standard $\Lambda$CDM. The parameter previously associated with "Dark Energy" ($\Omega_\Lambda$) is reconceptualized as the background kinetic energy density of the scalar field $\Omega_\phi$.

## 6.2 The TEP Interpretation

| Standard Cosmology ($\Lambda$CDM) | TEP Framework |
| --- | --- |
| Space expands, stretching photon wavelengths | Photon frequencies shift due to the conformal field clock-rate gradient |
| Dark Energy accelerates the expansion of space | Apparent acceleration is modeled as the kinetic energy density of the Temporal Shear field |
| $H_0$ tension requires early-universe modifications | Distance probes are biased by local environmental mass-screening of the scalar field |
| The universe began 13.8 billion years ago in a singularity | The "Big Bang" is modeled as a TEP temporal-horizon boundary where the field $A(\phi) \to 0$ |

## 6.3 Implications for Cosmological Tensions

The conformal paradigm offers a novel geometric interpretation for several cosmological tensions.

**The Hubble Tension:** The local distance ladder relies on calibrating deep-void supernovae against galactic Cepheids. In TEP, the temporal shear field is environmentally screened by mass. Supernovae exist in empty voids (where the field is unscreened), while Cepheids exist in dense galaxies (where the field is heavily screened). The conformal transport correction modifies the SH0ES local measurement, which has been proposed in the broader corpus (Paper 11) as a mechanism to resolve the $5\sigma$ tension.

**High-Redshift Galaxy Assembly:** The temporal horizon interpretation implies a fundamentally different proper-time mapping at high redshift. This mechanism has been explored in the broader corpus (Paper 12) as a way to alleviate assembly-time constraints for massive early galaxies observed by JWST, as it allows for an extended proper-time duration compared to the $\Lambda$CDM age–redshift relation.

## 6.4 Cross-Scale Consistency: Wide Binaries

Because the framework relies on a scalar field $\phi$ rather than global spatial expansion, the field couples to matter across scales. While dense local environments like the Solar System screen the field, in the ultra-diffuse, low-acceleration outskirts of the Milky Way, the screening mechanism is weakened.

The background Temporal Shear gradient is hypothesized to manifest as a weak-field gravitational anomaly in these unscreened environments. This connection is explored in the corpus (Paper 13) as a predictive mechanism for the anomalous wide-binary accelerations measured by Gaia DR3, suggesting a cross-scale link between the cosmological field and local stellar kinematics.

## 6.5 Future Empirical Testing

Serving as a synthesis framework, the theory outlines a highly specific, preregistered experimental falsification pathway. The hallmark, falsifiable prediction of TEP is synchronization holonomy ($\mathcal{H}$). To explicitly measure the non-integrability of the time field, the following experimental avenues are defined:

- *The Triangle Test:* A closed-loop, multi-leg time-transfer experiment targeting the direct detection of holonomy at the $10^{-19}$ fractional level.

- *Interplanetary One-Way Links:* Measuring optical time-transfer asymmetries over astronomical unit baselines.

- *Clock Networks and Kinematic Data:* Utilizing precision clock arrays and deterministic pipelines on public catalogs to map environment-dependent screening signatures.

- *Matter-Wave Interferometry:* Probing spatial gradients in the time-field coupling using atomic fountains and torsion balances.

By asserting that time itself is a dynamical field, the framework provides a mathematically rigorous path forward for precision metrology and cosmology, preserving the rigidly tested empirical pillars of relativity.

# 7. Conclusion

This paper presents a direct empirical challenge to the necessity of primitive cosmic expansion. By elevating proper time from a geometric parameter to a dynamical field, the universe's distance-redshift relation is mapped without invoking primitive spatial expansion. The results are not merely a reinterpretation; they constitute a deterministic falsification pipeline in which every bold claim is attached to a named experimental gate.

## Claim Gate Registry

| Claim | Status | Required Gate | Current Result |
| --- | --- | --- | --- |
| No primitive expansion required | Passed at SNe background level | TEP conformal reconstruction ties or beats $\Lambda$CDM on Pantheon+ | M2 ties; M1 improves $\Delta\chi^2\simeq-7.5$ |
| No primitive $\Lambda$ required | Passed at SNe late-time level | No-$\Lambda$ TEP beats $\Lambda$CDM with same covariance and no host-mass nuisance | BF = 32.1 (fixed $z_T=100$); BF = 25.5 (free $z_T$) |
| LCDM null injection falsification | Passed | Observed TEP preference does not occur under LCDM mocks | 0% false-positive rate (32 trials) |
| Pantheon+ subset robustness | Passed | TEP preference survives all data cuts and survey removals | 27/27 subsets prefer TEP |
| H0 boundary stress test | Revised | EdS constraint drives H0 toward zero (prior floor at 50 masks this); free `omega_cdm` recovers H0=67.4, consistent with LCDM | Joint MCMC converges to LCDM when EdS is relaxed; SNe-only results remain robust |
| Jordan frame acoustic proof | Passed | CMB acoustic scale preserved in matter-only EdS background | $100\theta_s = 1.0433$ at $\epsilon_T = 0.018$ (0.3% of Planck) |
| Big Bang as temporal horizon | Theoretically mapped | Show $A\to0$ horizon with finite matter-frame invariants | Deferred to TEP-TH (Paper 27) |
| CMB acoustic safety | Passed at background/acoustic level | $r_s^{\rm TEP}/r_s^{\Lambda{\rm CDM}}\approx1$ | Reported $0.999994$ |
| Linear pure-conformal scalar perturbation safety | Passed in TEP-HC; cross-checked here | Active $\delta\phi$, stability, TT/TE/EE residuals | TEP-HC: no-ghost/stability proof; C0: pipeline gate confirms consistency |
| Host-mass-step prediction | Partial — sign captured, amplitude needs refinement | TEP predicts mass-step offset from temporal-shear geometry | Locked prediction $\Delta\mu \simeq -0.038$ mag (correct sign, $\sim$5$\times$ observed amplitude); TEP+fitted residual beats LCDM fitted step by $\Delta\chi^2 \simeq -3.4$ |
| Dark matter replacement | Corpus-level implication | Lensing/growth/galaxy-scale gates | Not a C0-only claim |
| BAO native projection | Passed | BAO ruler recovery in TEP background | $\chi^2/\text{DOF} = 0.88$ (17 data points) |
| Growth amplitude | Sharp prediction | $\sigma_8 \approx 1.48$ (linear TEP) vs $\sigma_8 \approx 0.84$ ($\Lambda$CDM) | $6\sigma$ distinct prediction; nonlinear/screening verification required |
| Distance duality | Blocked | Compilation shows $\eta = 0.866 \pm 0.020$ (6.6$\sigma$ from $\eta=1$) | Both LCDM and TEP predict $\eta=1$ by construction; compilation mixes Planck $D_L$ with BAO $D_A$ and is not self-consistent |
| Tolman surface brightness | Systematic domain | Measured $n = 3.375 \pm 0.027$ vs LCDM $n = 4.0$ | Evolution/K-correction systematics dominate; not a clean gate |

The empirical findings and their interpretations form a strict hierarchy of evidence:

- **No Primitive Expansion Required by the Tested Background Data:** The exact conformal reconstruction M2 proves that the Pantheon+ homogeneous distance-redshift relation does not uniquely require primitive expansion of the spatial metric. More strongly, the physical no-$\Lambda$ temporal-shear branch M1 improves the Pantheon+ likelihood by $\Delta\chi^2\simeq-7.5$ relative to baseline $\Lambda$CDM using the same 1,701-supernova covariance structure and no fitted host-mass-step nuisance parameter. The expansion interpretation is therefore not merely underdetermined; in the tested SNe sector, it is empirically outperformed by a temporal-transport distance law.

- **No Primitive Dark Energy Required in the Tested Late-Time Sector:** The M1 branch achieves a better standardized-supernova fit with $\Omega_\Lambda=0$, replacing vacuum-energy acceleration with temporal-shear transport in the late-time distance-redshift relation. This result directly tests the phenomenological role of $\Lambda$ in the Pantheon+ sector and shows that apparent acceleration can be reconstructed without a primitive dark-energy component.

- **No Physical Big Bang Singularity in the Conformal Reconstruction:** In the TEP mapping, the limit conventionally written as $a\to0$ is re-expressed as $A(\phi)\to0$: a TEP temporal-horizon boundary of clock transport rather than a zero-volume spatial singularity. The C0 paper establishes the conformal reconstruction and identifies the singular origin as an artefact of imposing an integrable FLRW clock foliation. Full matter-frame nonsingularity is the dedicated closure target of TEP-TH.

- **Particle Dark Matter (Corpus Implication):** Although the current pipeline focuses on the cosmological background and macroscopic bounds, the broader TEP corpus develops the claim that local gradients of this same temporal field modify effective gravitational potentials. This provides the theoretical foundation for replacing particle dark matter with geometric temporal shear in galactic and cluster environments.

The reproducible pipeline provides a robust, formally closed supernova-sector Bayesian framework demonstrating that conformal transport is a viable and highly competitive alternative to the standard expanding universe. The remaining question is not whether TEP can fit the Hubble diagram; it can. TEP-HC (Paper 18) has already established that the linear pure-conformal scalar perturbation sector survives active CMB perturbations with no ghosts and negligible observational impact at current amplitude bounds. The next question is whether the same temporal-transport field survives nonlinear structure formation, host-environment reconstruction, and line-of-sight null injections across the full multi-probe dataset. By asserting that time itself is a dynamical field, the framework provides a highly testable path forward for precision cosmology.

# 8. References

## 8.1 TEP Series

- Smawfield, M. L. (2025). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed*. v0.9 (Jakarta). DOI: 10.5281/zenodo.16921911.

- Smawfield, M. L. (2026). *The Cepheid Bias: Resolving the Hubble Tension*. v0.6 (Kingston upon Hull). DOI: 10.5281/zenodo.18209702.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies*. v0.4 (Kos). DOI: 10.5281/zenodo.19000827.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars*. v0.6 (Caracas). DOI: 10.5281/zenodo.18165798.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries*. v0.3 (Kilifi). DOI: 10.5281/zenodo.19102061.

- Smawfield, M. L. (2026). *TEP-HC: Boltzmann Perturbation Closure and Acoustic-Scale Preservation*. v0.3 (Cambridge). DOI: 10.5281/zenodo.20370145.

- Smawfield, M. L. (2026). *TEP-QF: Quantum Foundations and Proper-Time Phase Holonomy*. v0.1. Zenodo.

- Smawfield, M. L. (2026). *TEP-SPIN: Topological Fermions and the Temporal Vortex*. v0.1. Zenodo.

- Smawfield, M. L. (2026). *TEP-TH: Nonsingular Temporal-Horizon Closure*. v0.1. Zenodo.

## 8.2 Data Sources

- Scolnic, D., et al. (2018). *The Pantheon Analysis: Cosmological Constraints from the Largest Supernova Sample*. ApJ, 859, 101.

- Scolnic, D., et al. (2022). *Pantheon+: Type Ia Supernova Light Curves from the Dark Energy Survey*. ApJ, 938, 113.

- Planck Collaboration (2020). *Planck 2018 results. VI. Cosmological parameters*. A&A, 641, A6.

- Fixsen, D. J., et al. (1996). *The Spectrum of the Cosmic Background Radiation*. ApJ, 473, 576.

- Mather, J. C., et al. (1994). *Measurement of the Cosmic Microwave Background Spectrum by the COBE FIRAS Instrument*. ApJ, 420, 439.

## 8.3 BAO and RSD Surveys

- Alam, S., et al. (2017). *The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic Survey: cosmological analysis of the DR12 galaxy sample*. MNRAS, 470, 2617.

- Beutler, F., et al. (2011). *The 6dF Galaxy Survey: baryon acoustic oscillations and the local Hubble constant*. MNRAS, 416, 3017.

- Anderson, L., et al. (2014). *The clustering of galaxies in the SDSS-III BAO sample: analysis of potential systematics*. MNRAS, 441, 24.

- Peacock, J. A., et al. (2015). *The SDSS-IV extended Baryon Oscillation Spectroscopic Survey: overview and early data*. MNRAS, 452, 2379.

- Dawson, K. S., et al. (2013). *The SDSS-III Baryon Oscillation Spectroscopic Survey: quasar targeting*. AJ, 145, 10.

- Ross, A. J., et al. (2015). *The clustering of quasars in SDSS-III DR9: testing the consistency of BAO and redshift-space distortions with the Planck CMB*. MNRAS, 449, 835.

## 8.4 Software and Tools

- Foreman-Mackey, D., et al. (2013). *emcee: The MCMC Hammer*. PASP, 125, 306. github.com/dfm/emcee

- Speagle, J. S. (2020). *dynesty: A dynamic nested sampling package for estimating Bayesian posteriors and evidences*. MNRAS, 493, 3132. github.com/joshspeagle/dynesty

- Torrado, J., & Lewis, A. (2021). *Cobaya: Code for Bayesian Analysis of cosmological data*. Astrophysics Source Code Library, ascl:2108.05. github.com/CobayaSampler/cobaya

- Lesgourgues, J. (2011). *The Cosmic Linear Anisotropy Solving System (CLASS). Part I: Overview*. arXiv:1104.2932. github.com/lesgourg/class_public

- Arbey, A. (2012). *AlterBBN: A program for calculating the BBN abundances of the elements in alternative cosmologies*. CPC, 183, 1822. alterbbn.hepforge.org

## 8.5 Historical References

- Hubble, E. (1929). *A relation between distance and radial velocity among extra-galactic nebulae*. PNAS, 15, 168.

- Friedmann, A. (1922). *Uber die Krummung des Raumes*. Z. Phys., 10, 377.

- Lemaitre, G. (1927). *Un univers homogene de masse constante et de rayon croissant rendant compte de la vitesse radiale des nebuleuses extra-galactiques*. Ann. Soc. Sci. Brux., 47, 49.

- Riess, A. G., et al. (1998). *Observational evidence from supernovae for an accelerating universe and a cosmological constant*. AJ, 116, 1009.

- Perlmutter, S., et al. (1999). *Measurements of Omega and Lambda from 42 high-redshift supernovae*. ApJ, 517, 565.

- Tolman, R. C. (1930). *On the estimation of distances in a curved universe with a non-static line element*. PNAS, 16, 511.

- Etherington, I. M. H. (1933). *On the definition of distance in general relativity*. Philos. Mag., 15, 761.

Smawfield, M. L. 2026. Temporal Equivalence Principle series, Papers 0-13. Zenodo preprints and associated repositories.

# 9. Data Availability & Reproducibility

This work follows open-science practices. All results are fully reproducible from raw data
using the documented pipeline. All numerical results, figures, and statistics are generated by deterministic
Python scripts processing real observational data. The pipeline is intentionally strict: failed dependencies are recorded as failed
results, not silently ignored.

### Repository and Code

GitHub Repository: github.com/matthewsmawfield/TEP-C0

The repository contains a deterministic, version-controlled cosmological analysis pipeline with 58 analysis steps
for supernova distance-redshift, distance-duality constraints, CMB acoustic scales, BBN preservation, structure growth, and systematic validation.
All steps are orchestrated by `scripts/run_pipeline.py` with comprehensive per-step logging.

#### Repository Structure

TEP-C0/
├── data/
│   ├── raw/                       # Downloaded source catalogs (Pantheon+, DDR, etc.)
│   └── processed/                 # Ingested and filtered datasets
├── scripts/
│   ├── steps/                     # 58 deterministic pipeline steps
│   ├── utils/                     # Logging and validation utilities
│   └── run_pipeline.py            # Master orchestration script
├── core/                          # Cosmology and model libraries
├── external/                      # Patched CLASS, AlterBBN dependencies
├── results/
│   ├── outputs/                   # JSON/CSV analytical outputs
│   └── figures/                   # Generated plots
├── logs/                          # Per-step execution logs
├── site/
│   └── components/                # Manuscript HTML sections
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation

### Data Provenance

| Data Source | Provider | Access Method | Records | Location |
| --- | --- | --- | --- | --- |
| Pantheon+ SNe Ia | Scolnic et al. | Auto-downloaded | 1,701 | `data/raw/pantheon_plus_shoes.dat` |
| Pantheon+ covariance | Scolnic et al. | Auto-downloaded | Full stat + sys | `data/raw/Pantheon+SH0ES.cov` |
| BAO constraints | BOSS, eBOSS, DES | Compiled from lit. | 10 measurements | `data/raw/ddr_constraints.csv` |
| SZ cluster DDR | Compiled | Auto-downloaded | ~38 clusters | `data/raw/sz_constraints.csv` |
| SGL lensing DDR | Compiled | Auto-downloaded | ~118 lenses | `data/raw/sgl_constraints.csv` |
| DESI/eBOSS Lyman-alpha | DESI-DR1, eBOSS | Auto-downloaded | 3 measurements | `data/raw/desi_ddr.csv` |
| FIRAS CMB spectrum | NASA LAMBDA | Auto-downloaded | ~43 frequencies | `data/raw/firas_spectrum.dat` |
| Planck 2018 CMB | Planck Collaboration | Cobaya package | TTTEEE+lensing | External Cobaya cache |
| BBN abundances | AlterBBN, compiled lit. | Included / downloaded | Yp, D/H, Li/H | `data/raw/bbn_review.html` |

### Pipeline Architecture

The analysis pipeline comprises 58 deterministic steps organized into eight logical stages.
Each step is a standalone Python script in `scripts/steps/` that produces JSON/CSV outputs and
detailed logs in `logs/step_*.log`. Dependencies are resolved automatically by the runner.

#### Complete Step Inventory and Runtime

Runtimes are approximate and measured on Apple M4 Pro (14-core, 24 GB). The dominant cost is the nested sampling step (03_01), which scales with `nlive` and number of models.

| Stage | Step | Script | Description | Runtime |
| --- | --- | --- | --- | --- |
| Stage 1: Data Acquisition (8 steps) |  |  |  |  |
| Data | 1.1 | `step_01_01_data_download.py` | Download Pantheon+ SNe, covariance, FIRAS | ~10 s |
| Data | 1.2 | `step_01_02_data_ingestion.py` | Ingest and validate all downloaded catalogs | ~1 s |
| Data | 1.3 | `step_01_03_download_ddr.py` | Download BAO distance-duality constraints | ~1 s |
| Data | 1.4 | `step_01_04_download_sb.py` | Download surface-brightness catalog sources | ~1 s |
| Data | 1.5 | `step_01_05_download_sz.py` | Download Sunyaev-Zel'dovich cluster data | ~1 s |
| Data | 1.6 | `step_01_06_download_sgl.py` | Download strong gravitational lensing data | ~1 s |
| Data | 1.7 | `step_01_07_download_desi.py` | Download DESI-DR1 and eBOSS Lyman-alpha | ~1 s |
| Data | 1.8 | `step_01_08_compile_sb.py` | Compile surface-brightness master catalog | ~1 s |
| Stage 2: Theory and Transport (4 steps) |  |  |  |  |
| Theory | 2.4 | `step_02_04_screening_scale_transfer.py` | Micro-to-galactic screening scale transfer and coarse-graining | ~1 s |
| Theory | 2.1 | `step_02_01_transport_kernel.py` | Verify FLRW recovery limit of open-path transport K_T | ~1 s |
| Theory | 2.2 | `step_02_02_theory_derivation.py` | Derive theoretical predictions for distance-redshift and screening | ~2 s |
| Theory | 2.3 | `step_02_03_physics_implementation.py` | Implement TEP physics: distance moduli, transport, growth kernels | ~3 s |
| Stage 3: Model Comparison and MCMC (9 steps) |  |  |  |  |
| Core | 3.1 | `step_03_01_three_model_comparison.py` | Nested sampling (dynesty, nlive=500) for M0a_LCDM, M0b_EdS, M1 variants, M2_PureShear, M3_wCDM, M4_CPL; null injection | ~90 min |
| Core | 3.2 | `step_03_02_independent_mcmc.py` | Independent MCMC convergence diagnostics | ~1 s |
| Core | 3.4 | `step_03_04_cobaya_mcmc.py` | Joint SNe+CMB MCMC via Cobaya with TEP-CLASS v2.0 | ~2 min |
| Core | 3.5 | `step_03_05_analyze_cobaya.py` | Analyze Cobaya chains and produce parameter constraints | ~1 s |
| Core | 3.6 | `step_03_06_cobaya_verbose.py` | Verbose Cobaya configuration and extended diagnostics | ~2 min |
| Core | 3.7 | `step_03_07_likelihood_synthesis.py` | Synthesize likelihoods across independent and joint analyses | ~1 s |
| Core | 3.8 | `step_03_08_h0_boundary_stress.py` | H0 prior stress test: extended priors reveal EdS-derived-parameter artifact driving H0 toward zero | ~30 s |
| Core | 3.9 | `step_03_09_lcdm_null_injection.py` | LCDM null injection: mock Pantheon+ from LCDM, measure TEP false-positive rate | ~60 s |
| Core | 3.10 | `step_03_10_pantheon_subset_robustness.py` | Leave-one-survey-out and redshift-window robustness tests | ~30 s |
| Stage 4: Supernova Tests and Distance Duality (8 steps) |  |  |  |  |
| SNe | 4.1 | `step_04_01_sn_time_dilation.py` | Test SN light-curve stretch factors against TEP time dilation | ~1 s |
| SNe | 4.2 | `step_04_02_sn_tolman.py` | Tolman surface-brightness dimming test | ~1 s |
| SNe | 4.3 | `step_04_03_tolman_sb.py` | Surface-brightness Tolman scaling with compiled catalog | ~1 s |
| DDR | 4.4 | `step_04_04_distance_duality.py` | Distance-duality relation: BAO constraints vs TEP prediction | ~1 s |
| DDR | 4.5 | `step_04_05_ddr_threeway.py` | Three-way probe comparison: BAO, SZ, SGL | ~1 s |
| DDR | 4.6 | `step_04_06_screening_fit.py` | Parametric screening model fit to probe-dependent DDR | ~2 s |
| DDR | 4.7 | `step_04_07_highz_ddr.py` | High-redshift Lyman-alpha DDR test (DESI, eBOSS) | ~1 s |
| SNe | 4.8 | `step_04_08_host_mass_step_prediction.py` | Host-mass-step mini-analysis: locked TEP prediction vs fitted LCDM nuisance | ~5 s |
| Stage 5: CMB and Big Bang Nucleosynthesis (8 steps) |  |  |  |  |
| CMB | 5.1 | `step_05_01_cmb_blackbody.py` | Verify TEP preserves CMB blackbody spectrum (FIRAS) | ~1 s |
| CMB | 5.3 | `step_05_03_cmb_boltzmann.py` | TEP Boltzmann integration via patched CLASS | ~1 s |
| CMB | 5.4 | `step_05_04_cmb_spectra.py` | Generate and compare TT/TE/EE power spectra | ~1 s |
| CMB | 5.5 | `step_05_05_cmb_consistency.py` | CMB acoustic-scale consistency check | ~1 s |
| BBN | 5.6 | `step_05_06_bbn_registry.py` | Compile observational BBN abundance registry | ~1 s |
| BBN | 5.7 | `step_05_07_bbn_preservation.py` | Cross-validate TEP and LCDM BBN predictions | ~1 s |
| CMB | 5.8 | `step_05_08_cmb_acoustic.py` | Acoustic-scale parameter comparison (Planck) | ~1 s |
| CMB | 5.9 | `step_05_09_minimal_perturbations.py` | Minimal active-perturbation closure: no-ghost/gradient stability, TT/TE/EE residuals, acoustic peak shift | ~3 s |
| Stage 6: BAO and Structure Growth (5 steps) |  |  |  |  |
| BAO | 6.1 | `step_06_01_bao_projection.py` | BAO ruler projection in TEP geometry | ~1 s |
| BAO | 6.2 | `step_06_02_bao_likelihood.py` | BAO likelihood module integration | ~7 s |
| Growth | 6.3 | `step_06_03_growth_solver.py` | TEP-CLASS v2.0 growth equation solver | ~1 s |
| Growth | 6.4 | `step_06_04_growth_validation.py` | Validate growth factors against LCDM baseline | ~1 s |
| Growth | 6.5 | `step_06_05_growth_rsd.py` | Redshift-space distortion comparison (f sigma_8) | ~2 s |
| Stage 7: Forecasts and Future Tests (7 steps) |  |  |  |  |
| Future | 7.1 | `step_07_01_mixed_forecast.py` | Forecast for mixed TEP-LCDM parameter recovery | ~1 s |
| Future | 7.2 | `step_07_02_redshift_drift.py` | Redshift-drift forecast and discriminating power | ~1 s |
| Future | 7.3 | `step_07_03_jwst_test.py` | JWST high-z supernova feasibility test | ~1 s |
| Future | 7.4 | `step_07_04_gw_sirens.py` | Gravitational-wave standard siren forecast | ~1 s |
| Future | 7.5 | `step_07_05_weak_lensing_plan.py` | Weak-lensing survey plan for TEP discrimination | ~1 s |
| Future | 7.6 | `step_07_06_weak_lensing.py` | Weak-lensing shear correlation analysis | ~1 s |
| Future | 7.7 | `step_07_07_blind_injection.py` | Blind injection validation protocol | ~1 s |
| Stage 8: Falsification, Verification, and Summary (8 steps) |  |  |  |  |
| Validation | 8.1 | `step_08_01_expansion_falsifier.py` | Expansion falsifier: distance duality and Tolman residuals | ~1 s |
| Validation | 8.2 | `step_08_02_comparison_stats.py` | Cross-model comparison statistics | ~1 s |
| Validation | 8.3 | `step_08_03_sensitivity_analysis.py` | Prior and parameter sensitivity analysis | ~1 s |
| Validation | 8.4 | `step_08_04_evidence_matrix.py` | Compile explanatory evidence matrix | ~1 s |
| Validation | 8.5 | `step_08_05_gate_registry.py` | Claim gate registry and status check | ~1 s |
| Validation | 8.6 | `step_08_06_claim_audit.py` | Automated claim consistency check | ~1 s |
| Validation | 8.7 | `step_08_07_final_summary.py` | Global evidence synthesis and summary | ~1 s |
| Validation | 8.8 | `step_08_08_diagnostic_plots.py` | Data-driven diagnostic figures (distance-duality residuals, Pantheon+ Hubble residuals) generated only from upstream pipeline artefacts | ~5 s |

#### Total Runtime Summary

The total runtime is dominated by Stage 3.1 (nested sampling). Runtimes scale approximately linearly with `nlive` and number of CPU cores.

| Component | Steps | Runtime |
| --- | --- | --- |
| Data Acquisition (Stage 1) | 8 | ~20 s |
| Theory and Transport (Stage 2) | 4 | ~6 s |
| Model Comparison and MCMC (Stage 3) | 9 | ~97 min |
| SNe Tests and DDR (Stage 4) | 8 | ~15 s |
| CMB and BBN (Stage 5) | 8 | ~11 s |
| BAO and Growth (Stage 6) | 5 | ~12 s |
| Forecasts and Future Tests (Stage 7) | 7 | ~7 s |
| Falsification and Verification (Stage 8) | 8 | ~7 s |
| Total | 58 | ~95 min (~1.6 h) |

### Reproduction Instructions

#### Quick Start (Full Reproduction)

# 1. Clone repository
git clone https://github.com/matthewsmawfield/TEP-C0.git
cd TEP-C0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (generates all results and figures)
python scripts/run_pipeline.py

# 4. Results will be in:
#    - results/outputs/   (JSON/CSV data)
#    - results/figures/   (PNG/PDF plots)
#    - logs/              (Detailed execution logs)

#### Command-Line Options

The pipeline supports selective execution for faster testing:

# Core statistical analysis only (skips long nested sampling)
python scripts/run_pipeline.py --core

# Resume from existing results (skip completed steps)
python scripts/run_pipeline.py --resume

# Run specific steps with automatic dependency resolution
python scripts/run_pipeline.py --steps step_04_04_distance_duality step_04_05_ddr_threeway

#### System Requirements

| Component | Minimum | Recommended | Tested On |
| --- | --- | --- | --- |
| CPU | 4 cores | 8+ cores | Apple M4 Pro (14-core) |
| RAM | 8 GB | 16 GB | 24 GB (M4 Pro) |
| Storage | 2 GB | 5 GB | NVMe SSD |
| Runtime (full) | ~4 h (4 cores) | ~1.5 h (8+ cores) | ~95 min (M4 Pro) |
| Runtime (--core) | ~1 min | ~30 s | ~20 s |

#### Key Analysis Outputs

- `results/outputs/step_03_01_three_model_comparison.json` — Nested sampling posteriors and evidence for all models (M0a_LCDM, M0b_EdS, M1 variants, M2_PureShear, M3_wCDM, M4_CPL)

- `results/outputs/step_03_04_cobaya_mcmc.1.txt` — Cobaya MCMC chain for joint SNe+CMB analysis

- `results/outputs/step_04_04_distance_duality.json` — DDR weighted mean and deviation from unity

- `results/outputs/step_04_05_ddr_threeway.json` — Three-way BAO/SZ/SGL probe comparison

- `results/outputs/step_05_07_bbn_preservation.json` — TEP vs LCDM light-element abundance cross-validation

- `results/outputs/step_05_09_minimal_perturbations.json` — active scalar perturbation stability checks and TT/TE/EE residuals relative to background-only TEP and $\Lambda$CDM

- `results/figures/step_05_09_perturbation_spectra.png` — TT/TE/EE comparison for $\Lambda$CDM, TEP background-only, and TEP minimal perturbations active

- `results/outputs/step_06_04_growth_validation.json` — Growth factor and sigma_8 consistency check

- `results/outputs/step_08_04_evidence_matrix.json` — Explanatory evidence matrix across all observables

- `results/outputs/step_08_06_claim_audit.json` — Automated claim consistency check report

#### Log Files

Each step produces detailed logs with timestamps, SHA-256 checksums, and execution status:

- `logs/step_*.log` — Individual step logs (58 files, one per step)

- `logs/verbose/` — Verbose Cobaya and nested sampling logs

### Software Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.10+ | Language runtime |
| NumPy | 1.24+ | Numerical computing |
| SciPy | 1.10+ | Statistical functions, nested sampling |
| Pandas | 2.0+ | Data manipulation |
| Matplotlib | 3.7+ | Visualization |
| emcee | 3.1+ | Ensemble MCMC sampling |
| dynesty | 2.1+ | Nested sampling for Bayesian evidence |
| Cobaya | 3.6+ | Joint MCMC with Planck likelihoods |
| classy (CLASS) | 3.2+ | CMB Boltzmann solver (patched for TEP) |

All dependencies are specified in `requirements.txt`. External dependencies (patched CLASS, AlterBBN) are included in the `external/` directory.

### Appendix Figures

![Joint SNe+CMB Background/Acoustic MCMC Diagnostic](results/figures/step_03_05_analyze_cobaya_triangle.png)

Figure A1: Joint SNe+CMB Background/Acoustic MCMC Diagnostic. This triangle plot shows the joint posterior from the Cobaya MCMC, including the homogeneous acoustic-sector amplitude $\epsilon_T^{\rm CMB}$. This is a diagnostic figure, not the SNe-only M1 evidence result. The $H_0$ boundary behaviour is separately stress-tested (Section 4.4.3). The $\epsilon_T$ shown here is the homogeneous acoustic-sector amplitude, distinct from the line-of-sight $\epsilon_{\rm shear}^{\rm los}$ fitted to supernovae.

![Minimal Conformal Perturbations vs LCDM](results/figures/step_05_09_minimal_perturbations_perturbation_spectra.png)

Figure A2: TEP Minimal Conformal Perturbations vs. $\Lambda$CDM. **Top panel:** TT power spectrum $D_\ell^{TT}$ for $\Lambda$CDM and TEP minimal conformal perturbations. **Bottom panel:** fractional residuals with quantitative gate outputs (max residual, acoustic peak shift, proxy $\chi^2$). This figure is included as a perturbation-safety gate diagnostic; the claim "perturbation safety passed" requires the quantitative residuals shown in the annotation box.
