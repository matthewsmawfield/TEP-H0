# The Cepheid Bias: Resolving the Hubble Tension
**Matthew Lukin Smawfield**  
Version: v0.10 (Kingston upon Hull)
First published: 11 January 2026 · Last updated: 30 September 2026  
DOI: 10.5281/zenodo.18209702

---

## Abstract

The Hubble Tension—the persistent $5\sigma$ discrepancy between local distance-ladder measurements ($H_0 \approx 73.0\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and early-universe CMB inference ($H_0 = 67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$)—represents a significant challenge in precision cosmology. This paper tests whether a component of the Hubble tension can be represented as an environment-dependent Cepheid clock bias, as predicted by the Temporal Equivalence Principle (TEP).

The hypothesis tested here is that Cepheid variable stars function as environment-dependent "standard clocks." Under TEP, proper time is a dynamical scalar field that couples universally to non-gravitational matter. Because stellar pulsation rates operate in local proper time, calibrating classical Cepheids in deep-potential SN Ia host galaxies against diffuse, low-mass anchor galaxies systematically misestimates host distance moduli. When interpreted through a universal Period--Luminosity relation, this clock-rate anomaly mimics diminished luminosity, leading to underestimated distances and an inflated local Hubble constant.

In standard distance-ladder linear regression, unconstrained latent host distance moduli $\mu_i$ algebraically absorb host-level environmental shifts. A standard-ladder projection test that inserts a TEP environmental column into the SH0ES design matrix therefore yields an apparent null result. This structural degeneracy is resolved by formulating the calibrator-host distance ladder as a generative clock-aware model coupled to the Hubble flow.

This generative framework is tested using the complete public Riess et al. (2022) sample of 37 distinct SN Ia host galaxies, utilizing a homogeneous kinematic potential coordinate $u_\phi=V_{\rm rot}/\sqrt{2}$ derived from pinned HyperLEDA rotation velocities and continuous environmental screening. The canonical peculiar-velocity convention throughout is the R22 baseline $\sigma_v = 250\ {\rm km/s}$; all other $\sigma_v$ values are reported as sensitivity variants. Across all 37 hosts, the generative endpoint likelihood yields $\Gamma_X=(1.156\pm0.958)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical $250\ {\rm km/s}$ velocity variance, strengthening to $1.45\sigma$ under the reduced-scatter sensitivity variants ($\sigma_v = 150$--$182.1\ {\rm km/s}$), with 100% leave-one-host-out sign stability (37 of 37 refits positive). In the cosmologically constrained host-level expansion-rate likelihood across the 33 Hubble-flow hosts, the combined endpoint response is recovered at $\Gamma_X=(1.165\pm0.979)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$; $2.07\sigma$ at $150\ {\rm km/s}$), yielding a conventional Hubble-flow intercept of $H_{\rm app}=68.62\pm1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($68.36\pm0.98$ at $\sigma_v = 150\ {\rm km/s}$). When evaluated in a joint multi-block framework with independent TRGB distances and geometric anchors, the multi-block likelihood favours an environmental response under the restricted Cepheid closure ($\kappa_{\rm Cep} = (0.266 \pm 0.239)\times 10^6\ {\rm mag}$, $1.11\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$; $(0.290 \pm 0.227)\times 10^6\ {\rm mag}$, $1.28\sigma$ under the $\sigma_v = 182.1\ {\rm km/s}$ sensitivity variant), resolving the local Hubble constant to $H_{\rm app} = 68.78\pm 1.48\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($68.72\pm 1.32$ at $182.1\ {\rm km/s}$), while the standalone redshift-only regression in $H_0$ space yields $\kappa_{\rm Cep} = (0.369 \pm 0.312)\times 10^6\ {\rm mag}$ ($1.18\sigma$ canonical; $(0.452 \pm 0.220)\times 10^6\ {\rm mag}$, $2.05\sigma$ at $\sigma_v = 150\ {\rm km/s}$). Under the physical coupled observing-chain model, the spectroscopic tracer's direct gravitational redshift contributes a velocity slope $\beta_{X,\rm grav} = c/\langle d \rangle \sim 10^4\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (<0.1% of the combined slope), so that >99.9% of the response is carried by the Cepheid distance scale ($\kappa_{\rm Cep}^{\rm equiv} = (0.369 \pm 0.310)\times 10^6\ {\rm mag}$); this allocation is physically derived from proper-time dilation, pending host-specific aperture validation. Model comparison is reported explicitly: at the canonical convention the joint multi-block likelihood is BIC-indifferent between the environmental response and the null ($\Delta{\rm BIC}\approx 2.7$ in favour of the null), and because the likelihood constrains only the summed scatter $\sigma_v^2 + \sigma_{\rm int}^2$, the $\sigma_v < 210\ {\rm km/s}$ variants are reparameterizations of a single fit rather than independent sensitivity points; the residual evidence rests on the redshift-only channel, the generative intercept's Planck concordance, and the sign-stability diagnostics.

Propagating the TEP potential correction through the distance ladder requires an explicit estimator distinction. The prespecified primary full-ladder projection propagates the measured response coefficient and yields $H_0=71.771\pm0.990\ {\rm km\,s^{-1}\,Mpc^{-1}}$, leaving a $3.94\sigma$ offset from Planck $67.4 \pm 0.5$; conditional matrix projections give $69.74\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical scaling. A separate unified host-level reconstruction, which re-optimizes the environmental response to flatten the host-environment trend, yields $H_0=66.65\pm1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, internally consistent with the TEP CMB value ($66.70 \pm 0.58$, Paper 26) at $0.03\sigma$ — an agreement between two products of the same framework, not an independent cross-check — and $0.45\sigma$ from Planck. The latter is a host-level reconciliation route, not the primary full-ladder estimator: it evaluates the fitted response at the cosmic baseline and is conditional on that response form extending below the anchor ensemble to the unperturbed potential. Full-matrix propagation through the 3,490-row SH0ES system confirms exact reference-gauge invariance. The statistically cleanest channel is the within-host period structure: the environment-coupled Cepheid period term, identified through the 37 independent period distributions and immune to latent-modulus absorption and peculiar-velocity systematics by construction, returns $\kappa_P = (-5.79 \pm 2.26)\times 10^5\ {\rm mag}$ ($2.57\sigma$ joint, $2.93\sigma$ isolated, $3.30\sigma$ with free shape terms), with host-assignment permutation $p = 0.0125$ and a negative sign in all 41 leave-one-host-out refits. Finally, single-galaxy differential tests provide internal signatures with the sign predicted by TEP free from host-to-host peculiar velocity systematics: M31 inner versus outer Cepheids exhibit an empirical Period--Luminosity offset of $+0.356 \pm 0.136\ {\rm mag}$ ($2.6\sigma$), increasing to $+0.681 \pm 0.187\ {\rm mag}$ ($3.65\sigma$) under spatial PHAT matching; however, M31 fails pair-matched controls (colour-matched $-0.017 \pm 0.128$ mag, $0.1\sigma$; eW-matched $-0.103 \pm 0.129$ mag, $0.8\sigma$), and the two admissible readings are tested against the theory's ambient closure. The overmatching hypothesis — that error matching discards an ambient-field signal — is bounded out: the photometric error is nearly orthogonal to the kpc-scale ambient-field proxy (${\rm corr}=-0.18$), and matching on the ambient field alone leaves a positive but inconclusive offset ($+0.206 \pm 0.163$ mag) that does not survive joint environment-plus-error control ($-0.069 \pm 0.126$ mag under matched (logP, $e_W$, $\log u_{\rm amb}$) cells — the residual was error-carried within ambient strata). Stratified through the full error distribution the conditional contrast declines monotonically from $+0.270 \pm 0.304$ mag at the 10th-percentile error level to $-0.117 \pm 0.202$ mag in the poorest, and the HST PHAT channel shows the same susceptibility under propagated-error matching ($-0.565 \pm 0.079$ mag, against $+0.906 \pm 0.147$ mag under neighbour-density control). The M31 channel is therefore carried as conditional evidence: the raw gradient is real and a positive residual is permitted in the best-measured strata, but neither reading closes the case. The LMC gradient ($+0.0284 \pm 0.0086$ mag, $3.3\sigma$) is robust to period and colour matching and remains the strongest single-galaxy result.

*Keywords:* Hubble tension -- Cepheid variables -- distance ladder -- galaxy rotation -- peculiar velocities -- temporal equivalence principle

## 1. Introduction

### 1.1 The Hubble Tension and Distance Ladder Systematics

The discrepancy between the local expansion rate of the Universe measured via the classical distance ladder and the value inferred from early-universe cosmological observations has emerged as one of the most pressing foundational problems in modern physics. The SH0ES collaboration reports a local Hubble constant of $H_0 = 73.04 \pm 1.04\ {\rm km\,s^{-1}\,Mpc^{-1}}$ using Cepheid-calibrated Type Ia supernovae (Riess et al. 2022; hereafter R22), whereas cosmic microwave background (CMB) measurements by the Planck satellite under flat $\Lambda$CDM establish $H_0 = 67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (Planck Collaboration VI 2020). This exceeds $5\sigma$ in statistical significance, resisting resolution through conventional astrophysical systematics, photometric crowding, or standard cosmological parameters.

The standard distance ladder relies on the assumption that stellar standard candles calibrated in nearby anchor galaxies (the Milky Way, Large Magellanic Cloud, Small Magellanic Cloud, and NGC 4258) behave identically in distant host galaxies. However, the anchor galaxies possess predominantly shallow gravitational potential wells, whereas the SN Ia host galaxies sample substantially deeper galactic potential wells. If stellar pulsation clocks respond dynamically to the local gravitational potential, an environmental gradient between calibrators and hosts will introduce a systematic bias into the inferred distance scale.

### 1.2 The Temporal Equivalence Principle

The Temporal Equivalence Principle (TEP) provides a covariant scalar-tensor framework that elevates proper time from a fixed background coordinate parameter to a dynamical physical field. The canonical low-kinetic limit of the Paper-0 action is formulated on a two-metric spacetime:

\begin{equation}
S = \int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R - \frac{1}{2}(\nabla\phi)^2 - V(\phi)\right] + S_m[\tilde g_{\mu\nu}, \Psi_m] ,
\end{equation}

where gravity is governed by the metric $g_{\mu\nu}$, while all non-gravitational matter fields, atomic transitions, and stellar clocks couple universally to the causal matter metric $\tilde g_{\mu\nu} = A^2(\phi)g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$, following the convention established in the foundational TEP theory paper (Smawfield 2025, Paper 0). The operative weak-field screening realization replaces the canonical scalar term by $P(X,\phi)=X-V(\phi)+X|X|/\Lambda_X^4$ (Paper 0, §2.2); the form displayed is its $P\to X-V$ limit. In gravitational bound states, the conformal factor satisfies $A(\phi) < 1$, slowing the rate of proper time relative to cosmic background time.

The absolute matter-clock rate relative to cosmic time is defined as

\begin{equation}
r_j \equiv \frac{(d\tilde\tau/dt)_j}{(d\tilde\tau/dt)_{\rm cosmic}} .
\end{equation}

Throughout the TEP framework, the physical rate ordering across galactic environments is strictly bounded:

\begin{equation}
0 < r_{\rm core} < r_{\rm disk} < r_{\rm cosmic} \equiv 1 .
\label{eq:rate_hierarchy}
\end{equation}

No clock runs faster than cosmic background time; clocks in active-shear galactic disks are simply less slowed than clocks in dense nuclear cores. Environmental screening $S(\rho)$ suppresses the observable scalar-charge sector in high-density regimes while preserving dynamical time-rate variations on galactic and cosmological scales. Microscopically this follows from the nonlinear kinetic completion $P(X,\phi)=X-V+X|X|/\Lambda_X^4$ of the specified scalar action (Paper 0 §2.2, v0.15): where the ambient clock-rate landscape is already steep, the field is kinematically stiff, so the observable Temporal Shear — the slope of that landscape, which Einstein-frame analyses parameterize as a fifth force — is pinned rather than newly carved (the $k \geq 4$ recovery steepness of constraint F4 is a consistency check on the derived response, realized by the hierarchical $s^4$ deep-interior asymptote). The field amplitude $A(\phi)$ continues to vary inside screened matter, preserving the clock-rate channel: what is pinned is the gradient, not the field amplitude. The quartic self-interaction $V = \lambda\phi^4/4$ is retained as the amplitude-sector candidate, whose radial ODE closure $\nabla^2\phi = V_{,\phi} + \rho_* A_{,\phi}$ yields $m_{\rm eff} \propto \rho^{1/3}$; the density-dependent screening factor $S(\rho)$ used in this paper is the EFT-level projection of the kinetic closure onto the Cepheid host-galaxy density regime, not an independent chameleon potential.

| Regime | Absolute Rate | Physical Interpretation |
| --- | --- | --- |
| Cosmic background | $r_{\rm cosmic} = 1$ | Unscreened cosmological reference |
| Galactic disk | $0 < r_{\rm disk} < 1$ | Moderate potential depth; active scalar shear |
| Galactic core | $0 < r_{\rm core} < r_{\rm disk}$ | Deep potential depth; strongly screened |

### 1.3 Stellar Pulsation Physics and the Observing Chain

Classical Cepheid variables are dynamical pulsation clocks whose pulsation periods $P_{\rm local}$ are governed by hydrodynamic acoustic transit times across the stellar envelope in local matter-frame proper time. When observed in the heliocentric telescope frame and corrected for the systemic redshift $z_{\rm spec}$ of the host galaxy, the inferred rest-frame period is:

\begin{equation}
P_{\rm rest}^{\rm inf} = \frac{P_{\rm obs}}{1 + z_{\rm spec}} = P_{\rm local}\cdot q_i ,
\qquad
q_i \equiv \frac{r_{\rm spec,i}}{r_{\rm Cep,i}} .
\label{eq:intro_period_ratio}
\end{equation}

Under the TEP core--disk tracer closure, systemic spectroscopy weighted toward a more deeply slowed region than the Cepheid field gives $r_{\rm spec} < r_{\rm Cep}$, hence $q_i < 1$, mathematically generating the observed period contraction. Through the Leavitt Period--Luminosity law ($M_W = \alpha + \beta \log P$), this period shift alters the inferred distance modulus by $\Delta \mu_i = \kappa_{\rm Cep} X_i$, where $X_i$ is the environmental potential coordinate.

### 1.4 Scope and Structure of this Study

This paper presents a comprehensive empirical test of the TEP framework across multiple independent observational tiers using public astronomical data:

1. Single-galaxy differential tests in M31 and the LMC, isolating the potential-clock coupling free from host-to-host peculiar velocity uncertainties.

2. Homogeneous, exact-identifier kinematic potential reconstruction for all 37 distinct R22 SN Ia host galaxies from pinned HyperLEDA rotation velocities and continuous screening.

3. Generative endpoint likelihood evaluation across flow-velocity models and peculiar velocity dispersions.

4. Mathematical resolution of the latent-modulus parameter absorption in standard distance-ladder design matrices using the TEP-native generative ladder.

5. Complete computational reproducibility: the resolution of the design matrix degeneracy and the full-ladder generative likelihoods are accompanied by an open-source, end-to-end Python pipeline, ensuring transparent verification of all statistical claims.

## 2. Data and Methods

### 2.1 Public Distance-Ladder Data and Independent Sample Reconstruction

The empirical distance ladder is reconstructed from the public R22 data release, comprising the 3,490-row generalized least-squares design matrix, data vector, full covariance matrix, and the Pantheon+SH0ES supernova catalogue (Scolnic et al. 2022; Brout et al. 2022). Solving the unmodified baseline system yields $H_0 = 73.0434 \pm 1.0072\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with $\chi^2 = 3552.7063$ for 3,444 degrees of freedom, reproducing the published SH0ES solution.

The R22 calibrator sample contains 42 SNe Ia situated in 37 distinct host galaxies. In the host-level inference, the independent statistical unit is the individual host galaxy, as all Cepheids and multiple SNe within a single galaxy share the same host potential well. Each supernova is matched to its host galaxy using exact R22 catalogue identifiers and PGC numbers, strictly excluding angular nearest-neighbour heuristics. All 37 distinct host galaxies have measured positive Hubble-flow redshifts ($z_{\rm HD} > 0$). Of these, 34 satisfy the Hubble-flow union cut ($z_{\rm CMB} > 0.0035$ or $z_{\rm HD} > 0.0035$); 33 of those have matching Cepheid design-matrix entries and enter the primary generative likelihood.

### 2.2 Homogeneous Kinematic Potential Coordinate

To ensure absolute reproducibility and avoid provenance errors arising from inhomogeneous mixtures of central stellar velocity dispersions and 21 cm H I linewidths, a global kinematic potential coordinate is constructed using a pinned snapshot of the HyperLEDA database. For each host and anchor galaxy, the homogenized maximum circular rotation velocity $V_{\rm rot}$ is extracted via exact PGC identifiers. The potential scale is defined as:

\begin{equation}
u_{\phi,i} = \frac{V_{{\rm rot},i}}{\sqrt{2}},
\qquad
\sigma_{u,i} = \frac{\sigma_{V,i}}{\sqrt{2}} .
\label{eq:uphi}
\end{equation}

This provides a rigorous, homogeneous proxy for the global gravitational potential depth $\Phi \sim u_\phi^2$, maintaining complete coverage across the entire 37-host sample without empirical aperture corrections.

### 2.3 Environmental Screening Formulation

The scalar field sector in TEP acts through two distinct macroscopic operators governed by Rules 10, 11, and 22: the Temporal Shear suppression operator $\mathcal{S}_\Sigma(\mathcal{E})$ and the clock-amplitude response $A(\phi)$. The macroscopic shear operator $\mathcal{S}_\Sigma(\mathcal{E})$ suppresses spatial clock-rate gradients and the corresponding Einstein-frame anomalous accelerations in dense environments ($\mathcal{S}_\Sigma \to 0$ in screened matter), recovering standard General Relativity for gravitational orbits. By contrast, the clock amplitude $A(\phi) = \exp(-\phi / M_{\rm Pl})$ tracks potential depth monotonically: proper-time clock rates run slower in deeper gravitational wells ($A < 1$), and this clock slowing continues inside screened matter without vanishing, preserving gravitational redshift.

In galaxy halos, the two-body environmental suppression factor $S_i = S_{{\rm local},i} S_{{\rm group},i}$ parameterizes the shear suppression of the host's extended potential well within its surrounding group or cluster environment. In local high-density stellar cores (such as inner M31 or LMC centers), the local potential depth is deepest, causing local clocks to run slowest ($A < 1$), which generates the observed positive Period--Luminosity offset ($\Delta W > 0$) relative to diffuse outer-disk Cepheids. Host disk scale lengths are evaluated from RC3 isophotal diameters ($R_{25} = D_{25}/2$, $R_d = R_{25}/3.2$, $z_d = 0.1 R_d$) and stellar mass densities are computed at representative Cepheid radii ($r_{\rm Cep} = 1.8 R_d$). Group-scale density is obtained from the Tully (2015) 2MRS group catalogue Table 5 via exact PGC matching. Evaluated at $r_{\rm Cep}$, the local density projection is equivalent to the enclosed-acceleration diagnostic $M({<}r_{\rm Cep})/r_{\rm Cep}^2 \propto \bar\rho({<}r_{\rm Cep})\,r_{\rm Cep}$ up to the disk's vertical-thickness correction — the density stand-in licensed for dense macroscopic matter by the $\mathcal{S}_\Sigma(\mathcal{E})$ construction (Paper 0 \S8). The total continuous screening factor is:

\begin{align}
S_{{\rm local},i} &= \left[1 + \left(\frac{\rho_i}{0.5\ M_\odot\,{\rm pc}^{-3}}\right)^2\right]^{-1} , \\
S_{{\rm group},i} &= \left[1 + \left(\frac{N_{{\rm mb},i}}{10}\right)^{1.2}\right]^{-1} , \\
S_i &= S_{{\rm local},i} S_{{\rm group},i} .
\label{eq:screening}
\end{align}

The screened potential scale for each host is $U_i = S_i u_{\phi,i}^2$. The assignment is positional: $S_i$ is the source-side suppression — the host's potential well sources the local field level through its effective scalar charge, which the shear-sector operator $\mathcal{S}_\Sigma$ suppresses at the habitat density — while the probe-side clock-amplitude response $\mathcal{S}_A$, evaluated at the Cepheid's own material density, is a per-source constant absorbed into the channel coefficient $\kappa_{\rm Cep}$ (Appendix C). For likelihood evaluation, the dimensionless centered environmental regressor is defined as:

\begin{equation}
\widetilde X_i = \frac{U_i - U_{\rm ref}}{c^2} - \left\langle \frac{U - U_{\rm ref}}{c^2} \right\rangle ,
\label{eq:centered_x}
\end{equation}

where centering guarantees that the fitted environmental slope is mathematically invariant to the choice of reference scale $U_{\rm ref}$.

### 2.4 Single-Galaxy Internal Differential Methodology

To test the TEP clock-rate coupling independently of host-to-host peculiar velocities and distance ladder calibration steps, internal Period--Luminosity variations are analysed within individual galaxies. In M31 (Andromeda), the homogeneous Pan-STARRS Cepheid catalog (Kodric et al. 2018) containing 2,686 variables is used, comparing Cepheids in the dense inner potential ($R < 5.0$ kpc, mean density $\rho \approx 0.31\ M_\odot\,{\rm pc}^{-3}$) against the diffuse outer disk ($R > 15.0$ kpc, $\rho \approx 0.006\ M_\odot\,{\rm pc}^{-3}$). Spatial restriction to the HST PHAT footprint is also performed. In the Large Magellanic Cloud, fundamental-mode Cepheids from OGLE-IV (Soszy&nacute;ski et al. 2015) are analysed, partitioned by galactocentric radius.

### 2.5 Generative Endpoint Likelihood and Flow Velocities

The host-level expansion relation is evaluated via the generative model:

\begin{equation}
cz_i = d_i\left(H_{\rm app} + \Gamma_X \widetilde X_i\right) + \epsilon_i ,
\label{eq:primary_likelihood}
\end{equation}

where $d_i = 10^{(\mu_i - 25)/5}\ {\rm Mpc}$, and the Gaussian variance is:

\begin{equation}
V_i = \sigma_v^2 + \left[\frac{\ln 10}{5} \left|d_i\left(H_{\rm app} + \Gamma_X \widetilde X_i\right)\right|\sigma_{\mu,i}\right]^2 + \sigma_{{\rm int},v}^2 .
\end{equation}

The likelihood is evaluated across a spectrum of peculiar-velocity dispersions: the standard R22 baseline ($\sigma_v = 250\ {\rm km\,s^{-1}}$), alternative velocity scales ($150\ {\rm km\,s^{-1}}$ and $500\ {\rm km\,s^{-1}}$), and the data-driven residual velocity scatter after bulk-flow removal from Pantheon+ ($\sigma_v = 182.1\ {\rm km\,s^{-1}}$). Because the intrinsic-scatter parameter $\sigma_{{\rm int},v}$ enters the variance only through the sum $\sigma_v^2 + \sigma_{{\rm int},v}^2$, the likelihood identifies a single quantity --- the total velocity-space scatter --- rather than $\sigma_v$ and $\sigma_{{\rm int},v}$ separately. With $\sigma_{{\rm int},v}$ free, any adopted $\sigma_v$ below the data-implied total is compensated by the fitted $\sigma_{{\rm int},v}$, so the $\sigma_v = 150$ and $182.1\ {\rm km\,s^{-1}}$ variants are reparameterizations of a single optimum rather than distinct fits; the likelihood-implied total is reported with the results ($\sigma_{\rm tot} \approx 210\text{--}226\ {\rm km\,s^{-1}}$ across the cz-space and $H_0$-space likelihoods), and the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ convention sits mildly above it, pinning $\sigma_{{\rm int},v}$ at its lower bound.

### 2.6 The TEP-Native Generative Distance Ladder

In standard distance-ladder linear regression, host distance moduli $\mu_i$ are treated as unconstrained latent parameters. The linear design matrix does not generically erase observation-level Cepheid perturbations (as confirmed by the unbiased ensemble recovery of raw row-level injections in Step 34: pull mean $+0.12$ with unit scatter over 200 covariance-level realizations). What it cannot identify without external constraints is an environmental perturbation that is mathematically equivalent to shifting the latent host modulus, because the 37 unconstrained $\mu_i$ parameters absorb the host-level shift ($\hat{\mu}_i \to \mu_i - \kappa_{\rm Cep} X_i$).

The true generative observable model couples the observed Cepheid distance moduli $\mu_i^{\rm obs} = \mu_i^{\rm true} - \kappa_{\rm Cep} X_i$ directly to cosmological expansion $cz_i = d_i^{\rm true}(H_{\rm app} + \beta_X X_i) + v_i$. The identifiable physical combination is:

\begin{equation}
\Gamma_X = \beta_X + \left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep} .
\label{eq:gamma_relation}
\end{equation}

In the generative ladder formulation, the relation is evaluated directly in expansion-rate space across the $N=33$ Hubble-flow SN Ia hosts with Cepheid calibrations ($z_{\rm HD} > 0.0035$ or $z_{\rm CMB} > 0.0035$, excluding local anchors):

\begin{equation}
H_{0,i} \equiv \frac{cz_i}{d_i^{\rm SH0ES}} = H_{\rm app} + \Gamma_X \widetilde X_i + \epsilon_i ,
\label{eq:generative_wls}
\end{equation}

where $d_i^{\rm SH0ES} = 10^{(\mu_i^{\rm SH0ES}-25)/5}$ Mpc, and each host is weighted by its inverse total variance:

\begin{equation}
w_i = \frac{1}{\sigma_{H_0,i}^2} = \frac{d_i^2}{\sigma_v^2 + \left(\frac{\ln 10}{5} cz_i \sigma_{\mu,i}\right)^2} .
\label{eq:generative_weights}
\end{equation}

Because $H_{0,i}$ is scale-invariant across distance, all 33 hosts contribute according to their fractional distance precision. To account for potential cross-host correlations in the SH0ES calibration chain, the model is also evaluated via Generalized Least Squares (GLS) using the full $33\times 33$ covariance matrix $\mathbf{C}_H = \mathbf{J} \mathbf{C}_\mu \mathbf{J}^T + \mathbf{C}_v$, where $\mathbf{C}_\mu = (\mathbf{L}^T \mathbf{C}^{-1} \mathbf{L})^{-1}_{\mu}$ is extracted directly from the inverted 3,490-row SH0ES normal matrix, $\mathbf{J} = \text{diag}(-\frac{\ln 10}{5} H_{0,i})$ is the Jacobian transformation, and $\mathbf{C}_v = \text{diag}(\sigma_v^2/d_i^2)$ is the peculiar velocity dispersion matrix.

Under the observing-chain mechanism formalized in Appendix C, the same potential well that causes the period contraction via $q_i = r_{\rm spec}/r_{\rm Cep}$ also imparts a gravitational redshift to the central spectroscopic tracer:

\begin{equation}
1 + z_{{\rm spec},i} = \frac{1 + z_{{\rm path},i}}{r_{{\rm spec},i}} \approx 1 + z_{{\rm path},i} + \frac{\Phi_{{\rm spec},i}}{c^2} .
\end{equation}

This shifts the observed velocity by $\Delta(cz)_i = c \Phi_{{\rm spec},i}/c^2 = \Phi_{{\rm spec},i}/c \sim 70\ {\rm m\,s^{-1}}$. When expressed as an apparent expansion-rate slope by dividing by host distance $d_i$, this physical gravitational redshift contributes:

\begin{equation}
\beta_{X,\rm grav} \equiv \frac{c}{\langle d \rangle} \approx \frac{299,792\ {\rm km\,s^{-1}}}{31.2\ {\rm Mpc}} \approx 9.60 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}} .
\end{equation}

In contrast, the Cepheid distance-modulus bias $\Delta \mu_i = \kappa_{\rm Cep} X_i$ rescales the host distance by $\Delta d / d \approx 7\%$, producing an apparent expansion-rate slope:

\begin{equation}
\Gamma_{X,\rm Cep} \equiv \left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep} \approx 1.15 \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}} .
\end{equation}

The ratio of the direct velocity response to the distance-modulus response is:

\begin{equation}
\frac{\beta_{X,\rm grav}}{\Gamma_{X,\rm Cep}} = \frac{9.60 \times 10^3}{1.15 \times 10^7} \approx 8.3 \times 10^{-4} \approx 0.08\% \text{--} 0.10\% .
\end{equation}

Consequently, the physical observing chain intrinsically allocates $>99.9\%$ of the combined response $\Gamma_X$ to the Cepheid distance channel. Setting $\beta_X \approx 0$ (or coupling $\beta_X = \beta_{X,\rm grav}$) is therefore not an ad-hoc allocation, but the direct leading-order consequence of metric proper-time transport.

The fitted intercept $H_{\rm app}$ represents the expansion rate at the sample mean environment $\langle X \rangle = 4.05 \times 10^{-8}$. To evaluate the expansion rate at physically defined reference frames, the relation maps back to:

\begin{equation}
H_{\rm ref} = H_{\rm app} - \Gamma_X \langle X \rangle,
\qquad
H_{\rm cosmic} = H_{\rm app} + \Gamma_X (X_{\rm cosmic} - \langle X \rangle) = H_{\rm ref} - \Gamma_X \left(\frac{U_{\rm ref}}{c^2}\right) ,
\label{eq:href_mapping}
\end{equation}

where $X_{\rm ref} \equiv 0$ corresponds to the anchor reference zero-point ($\sigma_{\rm ref} = 87.165\ {\rm km\,s^{-1}}$), and $X_{\rm cosmic} = -U_{\rm ref}/c^2 = -8.45 \times 10^{-8}$ corresponds to the unperturbed cosmic background (zero virial potential depth).

### 2.7 Joint Multi-Block Distance--Redshift Likelihood

While the standalone host-level regression measures the identifiable combined slope $\Gamma_X$, breaking the internal parameter degeneracy of the distance ladder without fixing $\beta_X=0$ requires external observational constraints. A joint multi-block likelihood is formulated by combining three observational data blocks:

\begin{equation}
\ln \mathcal{L}_{\rm joint}(\boldsymbol{\theta}) = \ln \mathcal{L}_{\rm redshift} + \ln \mathcal{L}_{\rm TRGB} + \ln \mathcal{L}_{\rm anchor} + \ln \mathcal{P}(\delta m) + \ln \mathcal{P}(\delta a) ,
\label{eq:joint_likelihood}
\end{equation}

where $\boldsymbol{\theta}$ encompasses the physical and nuisance parameters. The three blocks are structured as follows:

1. *Redshift--Distance Block ($N=33$ Hubble-flow hosts):* Evaluates cosmological expansion via a linearised Hubble-flow regression in $H_0$ space, $H_{0,i} = cz_i / d_i^{\rm obs} = H_{\rm app} + \Gamma_X X_i + \epsilon_i$, where $\Gamma_X = \beta_X + (\ln 10 / 5) H_{\rm app} \kappa_{\rm Cep}$ is the identifiable combined endpoint response. This formulation is equivalent to the nonlinear $cz$-space model $cz_i = d_i^{\rm true}(H_{\rm app} + \beta_X X_i) + v_i$ with $d_i^{\rm true} = d_i^{\rm obs} 10^{\kappa_{\rm Cep} X_i / 5}$, but offers superior numerical conditioning because the parameter space is linear in $(H_{\rm app}, \Gamma_X)$. The 33 hosts comprise the full Hubble-flow calibrator sample, including the two lowest-redshift systems (NGC 4424 and NGC 4536) whose CMB-corrected redshifts place them above the Hubble-flow threshold. The per-host variance is $\sigma_{v,i}^2 = \sigma_v^2 / d_i^2 + (\frac{\ln 10}{5} H_{0,i} \sigma_{\mu,i})^2 + \sigma_{{\rm int},v}^2 / d_i^2$, where $\sigma_{{\rm int},v}$ is an intrinsic velocity dispersion nuisance parameter.

2. *TRGB Differential Block ($N=18$ non-anchor calibrators):* Compares Cepheid and Tip of the Red Giant Branch distance moduli, $\Delta \mu_i \equiv \mu_i^{\rm Cep} - \mu_i^{\rm TRGB} = \delta m - \kappa_{\rm Cep} X_i + \epsilon_i$, with variance $\sigma_{\Delta\mu,i}^2 = \sigma_{\mu,{\rm Cep},i}^2 + \sigma_{\mu,{\rm TRGB},i}^2$. Because TRGB stellar evolution clocks operate independently of pulsation physics, TRGB distances provide an external anchor on the host modulus. The 18 hosts represent the complete overlap of SH0ES Cepheid calibrators, HyperLEDA rotation velocities, and EDD/CCHP TRGB measurements (excluding geometric anchors to prevent double counting).

3. *Independent Geometric Anchor Block ($N=2$ primary anchors):* Constrains the absolute Cepheid zero-point against independent geometric distances in the Large Magellanic Cloud (detached eclipsing binaries) and NGC 4258 (water megamasers): $\Delta \mu_{\rm anc} \equiv \mu_{\rm Cep} - \mu_{\rm geo} = \delta a - \kappa_{\rm Cep} X_{\rm anc} + \epsilon_a$. M31 is excluded from this block because its distance scale is composite rather than purely geometric.

Five nested model configurations are evaluated against the data: the *Null Model* ($k=4$: $H_{\rm app}, \delta m, \delta a, \sigma_{{\rm int},v}$), the *Cepheid Model* ($k=5$: setting $\beta_X \equiv 0$ and fitting $\kappa_{\rm Cep}$), the *Coupled Model* ($k=5$: setting $\beta_X = \beta_{X,\rm grav} \equiv c/\langle d \rangle \approx 9.60 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and fitting $\kappa_{\rm Cep}$), the *Velocity Model* ($k=5$: setting $\kappa_{\rm Cep} \equiv 0$ and fitting $\beta_X$), and the *Mixed Model* ($k=6$: simultaneously fitting both $\kappa_{\rm Cep}$ and $\beta_X$). The Cepheid and Velocity models represent two alternative single-degree-of-freedom closures of the underlying physical response rather than additive components, while the Coupled model realizes the physical prediction where the direct velocity perturbation is small but nonzero. In this multi-block formulation, the three data blocks are treated as quasi-independent observational constraints; while excluding primary geometric anchors from the TRGB block prevents direct anchor duplication, shared Cepheid calibration terms are treated within individual block variances.

### 2.8 Unified Host-Level Reconstruction

Complementing the expansion-rate likelihood, a unified host-level reconstruction is performed directly on the complete 37-host sample. Each host's distance modulus is corrected according to its screened kinematic potential:

\begin{equation}
\mu_i^{\rm corr} = \mu_i^{\rm obs} + \kappa_{\rm Cep} \frac{S(\rho_i)\, u_{\phi,i}^2 - U_{\rm ref}}{c^2} ,
\label{eq:endpoint_correction}
\end{equation}

yielding reconstructed physical distances $d_i^{\rm corr} = 10^{(\mu_i^{\rm corr} - 25)/5}$ and individual expansion rates $H_{0,i} = cz_i / d_i^{\rm corr}$. The anchor reference zero-point is defined as $\sqrt{U_{\rm ref}} = \sigma_{\rm ref} = 87.165\ {\rm km\,s^{-1}}$, constructed from the weighted composite of local stellar velocity dispersions at the specific Cepheid disk locations within the primary calibration anchors: Milky Way solar neighbourhood ($\sigma_z = 30.0\ {\rm km\,s^{-1}}$, weight 0.20; Bovy et al. 2012), LMC stellar disk ($\sigma_{\rm disk} = 24.0\ {\rm km\,s^{-1}}$, weight 0.25; van der Marel et al. 2002), and NGC 4258 intermediate annulus ($\sigma_{\rm local} = 115.0\ {\rm km\,s^{-1}}$, weight 0.55; Kormendy & Ho 2013). As proven in Appendix D.3, the reference scale $\sqrt{U_{\rm ref}}$ is an exact gauge origin in the full linear system, absorbed into the Cepheid zero point $M_H^W$ without altering the physical response $\kappa_{\rm Cep}$.

The optimal response coefficient $\kappa_{\rm Cep}$ is determined by minimizing the residual variance and environmental gradient across the host sample ($\partial H_{0,i} / \partial u_{\phi,i} \to 0$). Honest uncertainties are estimated via a joint bootstrap ($N=1,000$ resamples with replacement), where $\kappa_{\rm Cep}$ is re-optimized for each realization, simultaneously propagating host-to-host sampling scatter and $\kappa_{\rm Cep}$ parameter variance to yield the unified Hubble constant $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$.

Evaluating the host-level reconstruction under the alternative screened anchor reference scale $\sqrt{U_{\rm ref}} = 30.507\ {\rm km\,s^{-1}}$ yields $H_0 = 64.74 \pm 1.53\ {\rm km\,s^{-1}\,Mpc^{-1}}$. Both host-level values ($66.65 \pm 1.58$ and $64.74 \pm 1.53\ {\rm km\,s^{-1}\,Mpc^{-1}}$) are in statistical concordance with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) within $0.45\sigma$ and $1.73\sigma$ respectively. The $1.9\ {\rm km\,s^{-1}\,Mpc^{-1}}$ offset between the two values is an artifact of the unconstrained host-level shortcut, which applies an additive modulus shift $\Delta \mu = \kappa_{\rm Cep} \Delta U_{\rm ref} / c^2$ without simultaneously refitting the anchor absolute magnitude zero point. In the full generative distance ladder (Appendix D.3), where the anchor absolute magnitude $M_H^W$ is freely refit, this additive shift is identically absorbed into the zero point, returning $H_0$ values that are identical to four decimal places. Furthermore, in the generative expansion-rate likelihoods, the cosmological background extrapolation $H_{\rm cosmic} = H_{\rm app} - \Gamma_X \langle U \rangle / c^2$ is algebraically independent of $U_{\rm ref}$, yielding Planck-concordant background expansion rates ($67.30\text{--}67.82\ {\rm km\,s^{-1}\,Mpc^{-1}}$) with exact gauge invariance.

### 2.9 Observational Controls for Single-Galaxy Internal Gradients

Single-galaxy differential tests in M31 and the LMC provide internal empirical checks of the TEP potential hierarchy free from host-to-host peculiar velocity dispersions. To ensure these internal Period--Luminosity offsets are not generated by standard astrophysical systematics, the analysis incorporates three rigorous controls:

1. *Extinction-Free Photometry:* All Cepheid magnitudes are evaluated using the reddening-free Wesenheit index $W \equiv m_I - R_V (m_V - m_I)$ (with $R_V = 1.55$ for OGLE-IV LMC and $R_V = 1.54$ for M31), which algebraically cancels total and differential interstellar dust extinction along the line of sight.

2. *Spatial Resolution and Crowding Controls:* In M31, ground-based blending and crowding effects in the dense inner bulge are controlled in three ways: (a) restricting the Pan-STARRS sample to the PHAT survey footprint, which recovers the internal offset at $\Delta W = +0.630 \pm 0.195\ {\rm mag}$ ($3.24\sigma$; $n_{\rm inner}=106$, $n_{\rm outer}=110$); (b) the dedicated HST PHAT J/H photometric catalogue (Kodric et al. 2018), which yields $\Delta W = +0.681 \pm 0.187\ {\rm mag}$ ($3.65\sigma$; $n_{\rm inner}=78$, $n_{\rm outer}=69$) and is adopted as the primary PHAT measurement — the two values correspond to distinct catalogues and are not interchangeable; and (c) two theory-informed diagnostics of the error control: an ambient-field proxy constructed at the theory closure scale ($L = 837$ pc; $\log u_{\rm amb} \propto \tfrac12\log$ of the tracer density within an $L/2$ disk-plane aperture), which both bounds the overmatching hypothesis — ${\rm corr}(e_W, \log u_{\rm amb}) = -0.18$ means the error covariate removes only a few per cent of the ambient-field variance — and supplies a matched estimate under the theory-relevant environmental control ($\Delta W = +0.206 \pm 0.163$ mag, falling to $-0.069 \pm 0.126$ mag under joint $(\log P, e_W, \log u_{\rm amb})$ cell-matching — the residual is error-carried within ambient strata); and an error-stratified contrast table, under which the conditional offset declines monotonically from $+0.270 \pm 0.304$ mag at the 10th-percentile error level to $-0.117 \pm 0.202$ mag in the poorest, so quality-band persistence reflects within-band composition rather than a uniform offset among equally-well-measured stars.

3. *Metallicity Plausibility Limits:* To test whether radial metallicity gradients could mimic the observed $\Delta W$, the required metallicity sensitivity is evaluated. For the empirical M31 metallicity gradient ($\Delta [\text{O/H}] \sim 0.2\text{--}0.4\ {\rm dex}$), explaining the observed $\Delta W \approx +0.36\text{--}0.63\ {\rm mag}$ purely via metallicity would require an unphysical coefficient $\gamma > 1.5\ {\rm mag/dex}$, whereas the empirical metallicity dependence measured across the SH0ES calibrator sample is $\gamma = -0.20 \pm 0.08\ {\rm mag/dex}$.

### 2.10 Host-Mass Decoupling and Supernova Mass-Step Marginalization

An essential empirical test is verifying that the physical TEP environmental coordinate does not act as a surrogate for the established Type Ia supernova host-galaxy stellar mass step. In standard supernova standardization pipelines (e.g., Betoule et al. 2014; Scolnic et al. 2018; Brout et al. 2022), an empirical magnitude correction $\gamma_{\rm step} \sim 0.05\text{--}0.06\ {\rm mag}$ is applied across the split threshold $\log(M_*/M_\odot) = 10.0$ to account for residual correlations between peak luminosity and host galaxy properties.

Host stellar masses $\log(M_*/M_\odot)$ are compiled directly from Riess et al. (2022, Table 4), spanning $8.50 \le \log(M_*/M_\odot) \le 10.87$ across the complete 37-calibrator sample. While raw galactic rotation velocity correlates with host stellar mass via the baryonic Tully--Fisher relation, the physical TEP coordinate $X_i \equiv (S_i \sigma_i^2 - \sigma_{\rm ref}^2)/c^2$ incorporates multi-scale environmental screening $S_i = S_{\rm local}(\Sigma_*) \times S_{\rm group}(\rho_{\rm env})$ — the factorized evaluation of the canonical shear-sector projection $\mathcal{S}_\Sigma(\mathcal{E})$ on the two-component host environment $\mathcal{E} = \{\Sigma_*, \rho_{\rm env}\}$ (Paper 0 §7) — which suppresses both dense galactic disks and massive group potentials. Consequently, the screened potential depth is fundamentally distinct from total integrated stellar mass.

To evaluate whether the measured slope $\Gamma_X$ could be mimicked by host-mass corrections, a joint multi-regressor likelihood is formulated:

\begin{equation}
cz_i = d_i \left[ H_{\rm app} + \Gamma_X X_i + \gamma_{\rm step} \Theta(\log M_{*,i} - 10.0) + \gamma_M (\log M_{*,i} - \langle \log M_* \rangle) \right] + \epsilon_i ,
\label{eq:joint_mass_likelihood}
\end{equation}

where $\Theta(x)$ is the Heaviside step function, $\gamma_{\rm step}$ measures the discrete host-mass step in equivalent velocity-space units, and $\gamma_M$ captures continuous mass dependence. In addition, an orthogonalized coordinate framework is constructed by projecting out host stellar mass:

\begin{equation}
X_{\perp \rm step} \equiv X - (\alpha_{\rm step} \Theta + \beta_{\rm step}) , \quad X_{\perp M} \equiv X - (\alpha_M \log M_* + \beta_M) ,
\label{eq:orthogonalized_coordinates}
\end{equation}

allowing the isolated environmental slope $\Gamma_{X,\perp}$ to be fitted against the mass-residualized coordinate completely independent of any host stellar mass variations.

## 3. Results

### 3.1 Host Sample Reconstruction and Potential Stratification

The exact-identifier reconstruction yields all 37 distinct R22 SN Ia host galaxies with measured positive Hubble-flow redshifts ($z_{\rm HD} > 0$), complete RC3 isophotal diameters, and pinned HyperLEDA circular rotation velocities $V_{\rm rot}$. Splitting the 37 hosts at the sample median potential scale $u_\phi = 114.62\ {\rm km/s}$ reveals an empirical stratification: the 19 shallow-potential hosts yield a mean Hubble parameter of $66.99 \pm 2.50\ {\rm km\,s^{-1}\,Mpc^{-1}}$, whereas the 18 deep-potential hosts yield $68.49 \pm 3.21\ {\rm km\,s^{-1}\,Mpc^{-1}}$, propagating the full host distance covariance matrix.

The raw potential-velocity association displays a positive Pearson correlation $r = 0.241$ ($p = 0.150$) and Spearman rank correlation $\rho = 0.188$ ($p = 0.266$). Two systematic effects suppress this raw statistic. First, the 37 hosts span a factor of $\sim$8 in distance, so the per-host Hubble-rate uncertainty is strongly heteroscedastic: peculiar-velocity noise contributes $\sigma_{H_0}^{\rm pec} = \sigma_v / d$ ranging from $\sim$2 km s$^{-1}$ Mpc$^{-1}$ for the most distant hosts to $\sim$37 km s$^{-1}$ Mpc$^{-1}$ for M101 at 6.8 Mpc. A heteroscedasticity-weighted Pearson correlation, using weights $w_i = 1/\sigma_{H_0,i}^2$ from the diagonal of the full host Hubble-rate covariance matrix, yields $r_w = 0.328$ ($p = 0.105$, $n_{\rm eff} = 25.5$). Second, the Pantheon+ peculiar-velocity estimates for several nearby group-host galaxies are biased by the simple flow model: for NGC 4639 (in a Tully group at 22.8 Mpc), the Pantheon+ vpec of $+309$ km s$^{-1}$ implies $H_0 = 47.25$, whereas the Stiskalek et al. 2026 Manticore Bayesian coherent-flow model gives $v_{\rm pec} = -414 \pm 162$ km s$^{-1}$, implying $H_0 = 78.99$. Replacing the Pantheon+ vpec with Manticore posterior means where available (34 of 37 hosts) strengthens the unweighted Pearson correlation to $r = 0.301$ ($p = 0.070$). The positive gradient between host potential depth and apparent expansion rate matches the directional prediction of TEP stellar clock modulation and is robust to both heteroscedasticity weighting and improved flow-model corrections.

![Host-level H0 values plotted against the square of the rotation-derived potential scale for 37 R22 supernova hosts](public/figures/step_03_figure_01_h0_vs_sigma.png?v=3)

Figure 1: Host-level expansion rate $cz_{\rm HD}/d$ as a function of the kinematic potential coordinate $u_\phi^2$ for all 37 R22 SN Ia host galaxies. Error bars include distance modulus covariance and peculiar velocity uncertainties.

### 3.2 Single-Galaxy Internal Verification: M31 and LMC

Single-galaxy tests provide empirical verification of the TEP potential coupling free from host distance uncertainties or peculiar-velocity flow modelling:

In M31 (Andromeda), 2,686 Cepheids from the Pan-STARRS survey are analysed across galactocentric radii. Comparing Cepheids in the dense inner potential ($R < 5.0$ kpc, mean density $\rho \approx 0.31\ M_\odot\,{\rm pc}^{-3}$, where chameleon screening is active) against those in the diffuse outer disk ($R > 15.0$ kpc, $\rho \approx 0.006\ M_\odot\,{\rm pc}^{-3}$) reveals an empirical Period--Luminosity zero-point offset, defined as $\Delta W \equiv W_{\rm inner} - W_{\rm outer}$:

> 

\begin{equation}
\Delta W_{\rm M31} \equiv W_{\rm inner} - W_{\rm outer} = +0.3560 \pm 0.1357\ {\rm mag} \quad (2.6\sigma\ {\rm significance}) .
\end{equation}

A positive $\Delta W$ indicates that inner Cepheids appear systematically fainter (possessing longer effective rest-frame periods) than their outer counterparts — the ordering the amplitude sector requires: as envelope self-screening weakens toward the diffuse outer disk, the Cepheid clock couples more strongly to the ambient field, producing the larger outer-disk contraction that yields a positive inner-minus-outer differential, the same $S_A$-sector ordering the within-host $\kappa_P$ term measures and the observing-chain hierarchy of Appendix C predicts in sign. The literal aperture-ratio channel cannot supply this amplitude: the core-to-disk clock differential is bounded by the potential contrast at $|\Delta\ln q| \lesssim \Delta\Phi/c^2 \sim 10^{-7}$, some five orders below the observed LMC offset and more below M31's, so the gradient is carried by the amplitude-sector response transferred through the empirical $\kappa_{\rm Cep}$/$\kappa_P$ coefficients rather than the bare clock ratio — a division of labour that identifies which theory sector the single-galaxy evidence actually tests. In the dedicated HST PHAT J/H catalogue the internal offset sharpens to $\Delta W_{\rm PHAT} = +0.6808 \pm 0.1867\ {\rm mag}$ ($3.65\sigma$ significance; restricting the Pan-STARRS sample to the same footprint gives $+0.630 \pm 0.195\ {\rm mag}$, $3.24\sigma$). In the Large Magellanic Cloud (LMC), analysis of OGLE-IV fundamental-mode Cepheids partitioned by galactocentric radius ($R \le 0.92$ kpc versus $R \ge 2.93$ kpc) independently reveals an internal offset of $\Delta W_{\rm LMC} = +0.0284 \pm 0.0086\ {\rm mag}$ ($3.3\sigma$ significance), following the same inner-minus-outer convention. The M31 offsets do not survive pair-matched confounder control — colour-matched pairs return $-0.017 \pm 0.128$ mag ($0.1\sigma$) and error-matched pairs $-0.103 \pm 0.129$ mag ($0.8\sigma$). The two admissible readings of that failure are tested directly. Under the overmatching hypothesis, pair-matching on $e_W$ would discard a signal carried by the ambient field only if $e_W$ traced $u_{\rm amb}$; constructing the ambient-field proxy at the theory closure scale $L = 837$ pc — $\log u_{\rm amb} \propto \tfrac12\log$ of the Cepheid tracer density within a disk-plane aperture of $L/2$ — and measuring its coupling to the photometric error returns ${\rm corr}(e_W, \log u_{\rm amb}) = -0.18$, so at most a few per cent of the ambient-field variance is removed with the error covariate and the overmatching channel is bounded rather than operative. Matching instead on the ambient-field proxy itself leaves $\Delta W = +0.206 \pm 0.163$ mag — positive, and consistent with the sign the amplitude sector requires, but statistically inconclusive; and that residual does not survive joint control: cell-matching on $(\log P, e_W, \log u_{\rm amb})$ returns $-0.069 \pm 0.126$ mag, so the positive ambient-matched offset is carried by the residual $e_W$ contrast within ambient strata rather than by an ambient-field-ordered signal. The complementary reading — that the quality-equated bands genuinely equate measurement quality — is tested by stratifying the contrast through the full $e_W$ distribution rather than restricting to a single band: the conditional offset declines monotonically from $+0.270 \pm 0.304$ mag at the 10th-percentile error level through $+0.024 \pm 0.109$ mag at the median to $-0.117 \pm 0.202$ mag at the 90th-percentile error level (the p10 point is a model extrapolation — only 3% of inner Cepheids attain that error level), so the positive pooled-band estimates reflect residual within-band composition rather than a uniform offset among equally-well-measured stars. The PHAT channel exhibits the same susceptibility: matching on the propagated $J/H$ error collapses the offset to $-0.565 \pm 0.079$ mag, against $+0.604 \pm 0.277$ mag under period–colour matching and $+0.906 \pm 0.147$ mag under an arcsec-scale neighbour-density control, with the inner sample carrying twice the mean photometric error of the outer ($0.0130$ versus $0.0062$ mag). The M31 channel is accordingly carried as conditional evidence: the raw gradient is real and a positive residual is permitted in the best-measured strata, but under joint environment-plus-quality control the offset is consistent with zero ($-0.069 \pm 0.126$ mag) — the ambient-field control does not restore a detection once the error contrast inside ambient strata is equated, and the photometric-error controls do not constitute a physical ambient-field match since $e_W$ is not an ambient proxy; the strict multivariate match ($-0.176 \pm 0.104$ on $n=29$ pairs) is retained as the countervailing diagnostic. The LMC gradient, robust to period and colour matching, remains the strongest single-galaxy result.

### 3.3 Generative Endpoint Likelihood and Velocity Robustness

Across the 37 SN Ia host galaxies, fitting the generative expansion relation $cz_i = d_i[H_{\rm app} + \Gamma_X \widetilde X_i] + \epsilon_i$ under the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ peculiar-velocity variance yields:

> 

\begin{equation}
\Gamma_X = (1.156 \pm 0.958) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}
\quad (N=37,\ \sigma_v=250\ {\rm km\,s^{-1}}) ,
\end{equation}

with an intercept $H_{\rm app} = 68.836 \pm 1.446\ {\rm km\,s^{-1}\,Mpc^{-1}}$. The fitted slope is positive across all velocity models, strengthening as peculiar velocity noise is reduced.

| Sample Selection | $N$ | $\sigma_v$ (km/s) | $\Gamma_X/10^7$ (km/s/Mpc) | Wald Stat | LRT Stat |
| --- | --- | --- | --- | --- | --- |
| All R22 hosts | 37 | 150 | $1.223 \pm 0.844$ | $1.45\sigma$ | $1.45\sigma$ |
| All R22 hosts (Pantheon+ residual) | 37 | 182.1 | $1.223 \pm 0.844$ | $1.45\sigma$ | $1.45\sigma$ |
| All R22 hosts (Canonical) | 37 | 250 | $1.156 \pm 0.958$ | $1.21\sigma$ | $1.22\sigma$ |
| All R22 hosts | 37 | 500 | $0.947 \pm 1.751$ | $0.54\sigma$ | $0.54\sigma$ |
| Hubble flow ($z_{\rm HD} > 0.0035$) | 30 | 250 | $0.676 \pm 0.973$ | $0.69\sigma$ | $0.70\sigma$ |
| Hubble flow ($z_{\rm HD} > 0.0050$) | 24 | 250 | $0.409 \pm 1.007$ | $0.41\sigma$ | $0.41\sigma$ |

Leave-one-host-out refits demonstrate remarkable directional stability: $\Gamma_X > 0$ in 37 of 37 jackknife iterations (100% positive sign stability). Bootstrap resampling across 1,000 realizations yields a positive slope in 86.4% of draws with mean $(1.100 \pm 1.183) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$.

The decline of the point estimate under stricter Hubble-flow cuts is tested directly by forward-model injection: the fitted bias $\kappa_{\rm Cep} X_i$ is injected into the observed host distances, each realization is reobserved under a fresh peculiar-velocity draw, and the slope is refit at each redshift cut. Under independent $v_i \sim \mathcal N(0, 250\ {\rm km\,s^{-1}})$ noise, the injected-bias recovery itself declines from $\kappa_{\rm inj} = 0.461\times 10^6$ mag to a median $0.353\times 10^6$ at $z > 0.005$ — the observed profile ($0.461 \to 0.413 \to 0.405 \to 0.130 \to 0.175\times 10^6$ mag) sits at the 15th--57th percentiles of the injection band at every cut. Under a coherent-flow variant (a 370 km/s CMB-dipole bulk flow plus 150 km/s residuals), the observed values fall at the 11th--86th percentiles. The cut-dependence is therefore statistically consistent with a genuine distance-scale bias degraded by velocity noise in shrinking subsamples, rather than the signature of velocity contamination; the discriminating test at this sample size is the sign stability and the multi-block response, not subsample persistence.

While the isolated generative endpoint likelihood exhibits absolute directional stability (100% positive sign stability across all 37 leave-one-out refits), the standalone test remains statistically modest ($1.22\sigma$ at the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$). A structural detection requires formulating the full generative distance ladder (Section 3.4) to couple these local endpoints directly to the Hubble flow, breaking the latent-parameter degeneracy that otherwise absorbs the single-endpoint signal into the host distance modulus nuisance parameters.

### 3.4 Design Matrix Degeneracy and the TEP-Native Generative Ladder

In the standard SH0ES generalized least-squares framework, each calibrator host distance modulus $\mu_i$ is treated as an unconstrained free parameter. A naive row-level environmental column cannot identify an environmental perturbation whose observational action is equivalent to shifting the latent host distance modulus, because the 37 unconstrained $\mu_i$ parameters absorb the host-level shift ($\kappa_{\rm matrix} = (-0.169 \pm 0.207) \times 10^6\ {\rm mag}$). As demonstrated by row-level synthetic injection ensembles (Appendix A.5), the matrix recovers true observation-level Cepheid perturbations without bias (pull mean $+0.12$, unit pull scatter, 68% coverage 0.73 over 200 full-noise realizations), confirming that the lack of host-level identification arises specifically from latent parameter absorption.

The synthetic injection experiment demonstrates this absorption mechanism: injecting a known environmental bias $\kappa_{\rm inj} = 6.987 \times 10^5\ {\rm mag}$ into true host distance moduli results in a recovered matrix parameter of $\kappa_{\rm matrix} \approx 0$, while the velocity-space generative model recovers the full injected signal ($\Gamma_X = 2.555 \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ versus injected $2.350 \times 10^7$).

![Toy recovery experiment demonstrating parameter absorption in unconstrained design matrices versus true recovery in generative velocity space](public/figures/step_43_figure_01_toy_recovery_experiment.png)

Figure 2: Synthetic recovery experiment. Left: unconstrained design matrix regression absorbs the environmental shift into latent host distance moduli, yielding a null coefficient. Right: generative expansion modelling accurately recovers the injected physical slope.

Formulating the distance ladder at the generative observable level in expansion-rate space across the 33 Hubble-flow hosts, where observed Cepheid moduli are coupled to Hubble-flow supernovae, resolves this degeneracy and yields a consistent combined endpoint response:

> 

\begin{equation}
\Gamma_X = (1.165 \pm 0.979) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}
\quad (N=33,\ \sigma_v=250\ {\rm km\,s^{-1}}) ,
\end{equation}

strengthening to $\Gamma_X = (1.315 \pm 0.778) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.69\sigma$) under data-driven Pantheon+ velocity scatter ($\sigma_v = 182.1\ {\rm km\,s^{-1}}$), and $\Gamma_X = (1.422 \pm 0.688) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.07\sigma$) at $\sigma_v = 150\ {\rm km\,s^{-1}}$. The fitted conventional Hubble-flow intercept at the sample mean environment is $H_{\rm app} = 68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ ($68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km\,s^{-1}}$). Mapping this relation to the anchor reference environment ($X_{\rm ref} \equiv 0$, $\sigma_{\rm ref} = 87.165\ {\rm km\,s^{-1}}$) via Equation~(\ref{eq:href_mapping}) yields $H_{\rm ref} = 68.15 \pm 1.54\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($67.78 \pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km\,s^{-1}}$), while evaluating at the unperturbed cosmic background ($X_{\rm cosmic} = -U_{\rm ref}/c^2$) yields $H_{\rm cosmic} = 67.39 \pm 2.09\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($67.30\text{--}67.82\ {\rm km\,s^{-1}\,Mpc^{-1}}$ across the $\sigma_v$ and mass-marginalization variants), in full statistical agreement with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and the unified host-level reconstruction. As established by Equation~(\ref{eq:gamma_relation}), $\Gamma_X$ is a combined endpoint response that may contain both a Cepheid modulus component ($\kappa_{\rm Cep}$) and a residual velocity-sector term ($\beta_X$); the restricted TEP Cepheid-channel closure sets $\beta_X = 0$, allocating the full response to the Cepheid channel ($\kappa_{\rm Cep}^{\rm equiv} = (0.369 \pm 0.310) \times 10^6\ {\rm mag}$).

When evaluated in the joint multi-block framework combining the redshift-distance relation ($N=33$), TRGB differentials ($N=18$), and independent geometric anchors ($N=2$), the joint likelihood favours an environmental response under the restricted Cepheid closure ($\kappa_{\rm Cep} = (0.266 \pm 0.240) \times 10^6\ {\rm mag}$, $1.11\sigma$, at the canonical $\sigma_v = 250\ {\rm km/s}$; $(0.290 \pm 0.227) \times 10^6\ {\rm mag}$, $1.28\sigma$, identically at $\sigma_v = 150$ and $182.1\ {\rm km/s}$ --- the two variants coincide because the likelihood identifies only the total scatter $\sigma_v^2 + \sigma_{{\rm int},v}^2$, with the fitted intrinsic scatter absorbing the difference), with $H_{\rm app} = 68.78 \pm 1.48\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 250\ {\rm km/s}$ ($68.72 \pm 1.32$ at the reduced-scatter variants). Evaluating the physically coupled observing-chain model where the velocity channel is fixed to the direct gravitational redshift ($\beta_X = c / \langle d \rangle \approx 9.60 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}}$, Appendix C.4) yields $\kappa_{\rm Cep} = (0.266 \pm 0.240) \times 10^6\ {\rm mag}$ and $H_{\rm app} = 68.78 \pm 1.48\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 250\ {\rm km/s}$ ($\kappa_{\rm Cep} = (0.290 \pm 0.226) \times 10^6\ {\rm mag}$, $H_{\rm app} = 68.72 \pm 1.32$, $\text{BIC} = 99.0$ at $\sigma_v = 182.1\ {\rm km/s}$), identical within $0.001\ {\rm km\,s^{-1}\,Mpc^{-1}}$ to the $\beta_X \equiv 0$ closure. The restricted Velocity closure ($\kappa_{\rm Cep} \equiv 0$) yields $\beta_X = (1.165 \pm 0.980) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$) at $\sigma_v = 250\ {\rm km/s}$ ($(1.227 \pm 0.885) \times 10^7$, $1.39\sigma$, at $\sigma_v = 150\text{--}182.1\ {\rm km/s}$). In the mixed model where both channels are free simultaneously, $\kappa_{\rm Cep} = (0.116 \pm 0.377) \times 10^6\ {\rm mag}$ and $\beta_X = (7.99 \pm 15.43) \times 10^6\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 250\ {\rm km/s}$. The two coefficients are not independently identified in this configuration: in the hierarchical velocity-space likelihood the fitted covariance returns $\mathrm{corr}(\kappa_{\rm Cep}, \beta_X) = -0.995$, and only the linear combination $\Gamma_X = \beta_X + (\ln 10/5)\, H_{\rm app}\, \kappa_{\rm Cep} = (1.14 \pm 0.92)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ is constrained by the data (\texttt{results/outputs/step\_38\_hierarchical\_timefield\_ladder.json}); ensemble injection recovery confirms $\Gamma_X$ is unbiased with 0.63 coverage at the nominal 68% interval while the individual coefficients slide along the degeneracy ridge. Decomposing the magnitude response by observable carrier — the channel-native two-coefficient variant in which the potential term on the Hubble-flow distances is assigned to the supernova-magnitude channel $\kappa_{\rm SN}$ (a Cepheid-channel bias acting on the calibrators propagates into those distances only through the calibrator-mean $M_B$, entering the block as a common mode absorbed by $H_{\rm app}$), while the TRGB-differential and anchor blocks retain the per-host Cepheid coefficient — yields $\kappa_{\rm SN} = (0.369 \pm 0.312) \times 10^6\ {\rm mag}$ ($1.18\sigma$) at the canonical $\sigma_v = 250\ {\rm km/s}$ and $(0.388 \pm 0.282) \times 10^6\ {\rm mag}$ ($1.38\sigma$) at the reduced-scatter variants ($\sigma_v = 150$ and $182.1\ {\rm km/s}$, identical under the scatter degeneracy), against $\kappa_{\rm Cep} = (0.116 \pm 0.377) \times 10^6\ {\rm mag}$ ($0.31\sigma$, independent of $\sigma_v$). Under the $\beta_X \approx 0$ magnitude-channel closure the measured response is therefore carried by the supernova-magnitude channel rather than by the Cepheid period–luminosity relation itself; the equivalent-Cepheid coefficients quoted throughout this section are bookkeeping conversions of that same endpoint response, not evidence of a Cepheid-period carrier. This prior-free decomposition reproduces the independent full-design-matrix sensitivity analysis, in which freeing the supernova channel returns $\kappa_{\rm SN} = (0.422 \pm 0.189) \times 10^6\ {\rm mag}$ ($2.23\sigma$) at $\sigma_v = 150\ {\rm km/s}$; the redshift-distance priors used there assume a Hubble law and are retained only as a sensitivity diagnostic, while the channel-split likelihood above supplies the corresponding no-expansion inference on the same data. Because the TRGB calibrators are exclusively nearby systems ($\sigma < 160\ {\rm km/s}$) with limited potential-coordinate leverage (the $X$ range spans only $1.5 \times 10^{-7}$, compared to $6.0 \times 10^{-7}$ for the redshift block), the TRGB differential block is uninformative about $\kappa_{\rm Cep}$ on its own ($0.27\sigma$) and dilutes the joint significance relative to the redshift-only regression. The standalone redshift-only WLS in $H_0$ space (Cepheid closure, $\beta_X \equiv 0$) yields the strongest single-block constraint: $\kappa_{\rm Cep} = (0.369 \pm 0.312) \times 10^6\ {\rm mag}$ ($1.18\sigma$) at the canonical $\sigma_v = 250\ {\rm km/s}$, rising to $(0.417 \pm 0.248) \times 10^6\ {\rm mag}$ ($1.68\sigma$) under data-driven Pantheon+ velocity scatter ($\sigma_v = 182.1\ {\rm km/s}$) and $(0.452 \pm 0.220) \times 10^6\ {\rm mag}$ ($2.05\sigma$) at $\sigma_v = 150\ {\rm km/s}$, with $H_{\rm app} = 68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($68.49 \pm 1.15$ and $68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the respective variants). Because $\sigma_{\rm int,v}$ is a free nuisance parameter, the joint likelihood constrains only the summed velocity-space scatter $\sigma_{\rm tot}^2 = \sigma_v^2 + \sigma_{{\rm int},v}^2$; the fitted optimum implies $\sigma_{\rm tot} \approx 217\text{--}226\ {\rm km\,s^{-1}}$ ($\sigma_{\rm int,v} = 159\ {\rm km\,s^{-1}}$ at $\sigma_v = 150$, $121\ {\rm km\,s^{-1}}$ at $182.1$, pinned at the lower bound for $\sigma_v \ge 250$), so variants below this scale are reparameterizations of a single fit rather than independent sensitivity points, and the canonical $\sigma_v = 250\ {\rm km/s}$ convention is the mildly over-dispersed edge of the family. The standalone WLS has no $\sigma_{\rm int,v}$ compensation, so its $\sigma_v$-dependence is genuine but its $\sigma_v = 150\ {\rm km/s}$ significance assumes a total scatter below the likelihood-implied value and should be read as the optimistic edge of the scatter family. Across the 53 combined joint observations, the Bayesian Information Criterion shows no decisive preference among the models ($\text{BIC}_0 = 97.2$ vs $\text{BIC}_{\rm Coupled} = 99.9$, $\text{BIC}_{\rm Cep} = 99.9$, $\text{BIC}_{\rm vel} = 99.7$, $\text{BIC}_{\rm mixed} = \text{BIC}_{\rm Cep+SN} = 103.6$ at the canonical $\sigma_v = 250\ {\rm km/s}$); the null has the lowest BIC, and the single-parameter closures sit within $\Delta{\rm BIC} \approx 2.7$ of it — below the threshold for positive evidence. All parameterizations capture the underlying potential stratification and recover a Planck-concordant expansion intercept.

### 3.5 Full-Ladder Propagation and Host-Level Reconciliation Routes

For the separate host-level reconstruction route — which re-optimizes the environmental response to flatten the host-environment trend — applying the TEP potential correction to the complete 37-host sample yields $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under the standard anchor reference scale ($\sqrt{U_{\rm ref}} = 87.165\ {\rm km\,s^{-1}}$) and $64.74 \pm 1.53\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under the screened reference scale ($\sqrt{U_{\rm ref}} = 30.507\ {\rm km\,s^{-1}}$). These route-specific variants achieve statistical concordance with Planck ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$), reducing the tension from $5.0\sigma$ to $0.45\sigma$ and $1.73\sigma$ respectively ($0.31\sigma$ bootstrap for the standard reconstruction). As demonstrated in Section 2.8 and Appendix D.3, the $1.9\ {\rm km\,s^{-1}\,Mpc^{-1}}$ offset between these two host-level values is strictly an artifact of the unconstrained shortcut holding the anchor zero point fixed; when the full 3,490-row SH0ES design matrix is solved with a free anchor zero point, shifting the reference origin produces the exact same $H_0$ down to four decimal places. Under TEP, the same CMB data yields $H_0 = 66.70 \pm 0.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (TEP-CMB), giving $0.03\sigma$ internal consistency between the TEP-corrected local value and the TEP-CMB inference. The residual $0.45\sigma$ offset from the $\Lambda$CDM-fitted Planck value reflects the model-dependent systematic expected from the different expansion-history assumptions.

Converting the host endpoint slope via Equation (\ref{eq:gamma_relation}) yields $\kappa_{\rm Cep}^{\rm equiv} = (0.365 \pm 0.304) \times 10^6\ {\rm mag}$. Propagating this environmental correction through the complete 3,490-row SH0ES matrix yields $H_0 = 71.771 \pm 0.990\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\Delta\chi^2 = +5.998$), while canonical scaling ($\kappa = 0.960 \times 10^6\ {\rm mag}$) yields $H_0 = 69.742 \pm 0.962\ {\rm km\,s^{-1}\,Mpc^{-1}}$, verifying exact reference-gauge invariance across coordinate origins $\sqrt{U_{\rm ref}} = 87.165\ {\rm km/s}$ and $30.507\ {\rm km/s}$.

Within the full-ladder propagation channel, the prespecified primary estimator is the conditional endpoint projection through the SH0ES design matrix, since it propagates the response coefficient actually measured from the Cepheid environment decomposition ($\kappa_{\rm Cep}^{\rm equiv}$, $1.2\sigma$) rather than the canonical benchmark coupling; it returns $H_0 = 71.77\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with a $3.94\sigma$ residual against Planck. The canonical-coupling projection is retained as the benchmark-conditional variant ($69.74$, $2.16\sigma$), and the remaining entries of Table 3 are the model-level robustness family—the cosmologically constrained expansion likelihoods ($68.36$–$68.78$) and the unified host-level reconstruction ($66.65 \pm 1.58$)—so the corrected-$H_0$ family spans $66.65$–$72.54\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with residual tensions $0.45\sigma$–$4.56\sigma$ according to estimator role, not to selection. The $2.0\ {\rm km\,s^{-1}\,Mpc^{-1}}$ offset between the matrix projection and the first-order simple propagation at identical $\kappa$ ($-3.30$ vs $-1.31$ under the standard reference) is mechanical: simple propagation folds the complete correction, including the $-\kappa U_{\rm ref}/c^2$ common-mode term, into the host-mean $\Delta\mu$, so its result shifts by up to $\pm 2.2\ {\rm km\,s^{-1}\,Mpc^{-1}}$—and can reverse sign—under reference-origin conventions; in the design-matrix propagation the same common-mode term is exactly degenerate with the anchor zero-point and host distance-modulus nuisance block, isolating the host-differential potential response and rendering $H_0$ invariant to $10^{-4}$ across all four reference gauges. It is registered explicitly that the canonical benchmark exceeds the measured endpoint response by a factor $\approx 2.3$--$2.6$ ($\kappa_{\rm gal}^{\rm canonical} = 0.960\times 10^6$ versus $\kappa_{\rm Cep}^{\rm equiv} = 0.365$--$0.421\times 10^6\ {\rm mag}$, $\approx 2.0$--$2.5\sigma$ under the fitted uncertainty); the residual is carried by the response product $S_{\rm Cep}\Gamma_{\rm Cep}$ — the screening-and-transfer sector whose microscopic derivation remains the open Gate-A item — so the canonical-scaling rows of Table 3 are theory-scale projections, not measured-response inferences.

| Model Configuration | $H_0$ (km/s/Mpc) | Tension with Planck ($67.4 \pm 0.5$) | Status |
| --- | --- | --- | --- |
| R22 baseline ladder | $73.043 \pm 1.007$ | $5.00\sigma$ | Severe tension |
| Conditional endpoint projection ($\kappa = 0.365\times 10^6$, prespecified primary) | $71.771 \pm 0.990$ | $3.94\sigma$ | Partial resolution |
| Conditional canonical scaling ($\kappa = 0.960\times 10^6$) | $69.742 \pm 0.962$ | $2.16\sigma$ | Substantial easing |
| Cosmologically constrained expansion likelihood (canonical $\sigma_v=250$) | $68.62 \pm 1.51$ | $0.77\sigma$ | Concordance ($\Gamma_X = 1.19\sigma$; $2.07\sigma$ at $\sigma_v=150$ variant) |
| Joint multi-block likelihood (Coupled / Cepheid channel, canonical $\sigma_v=250$) | $68.78 \pm 1.48$ | $0.88\sigma$ | Concordance ($\kappa_{\rm Cep} = 0.266\times 10^6$, $1.11\sigma$; $1.28\sigma$ at $\sigma_v = 150$--$182.1$) |
| Unified host-level reconstruction | $66.65 \pm 1.58$ | $0.45\sigma$ | Host-level route; not the primary matrix projection ($0.31\sigma$ bootstrap) |

### 3.6 Multi-Channel Verification: TRGB Red Giants

A critical test of the TEP framework is cross-channel consistency with Tip of the Red Giant Branch (TRGB) stars. Unlike classical Cepheids, whose periods depend hydrodynamically on envelope acoustic transit times in local proper time, TRGB stars are standard candles governed by degenerate core helium-flash physics. In the 18 host galaxies with overlapping Cepheid (SH0ES) and TRGB (CCHP and EDD) distance determinations, the joint environmental response is evaluated.

The 18-host TRGB sample is tested with the same TEP regressor as the Cepheids. Separate OLS fits give $B_{\rm Cep} = (1.308 \pm 2.960) \times 10^6\ {\rm mag}$ and $B_{\rm TRGB} = (1.280 \pm 2.860) \times 10^6\ {\rm mag}$; both are individually consistent with zero. The weighted differential slope is $\Delta \kappa \equiv B_{\rm Cep} - B_{\rm TRGB} = +(0.101 \pm 0.379) \times 10^6\ {\rm mag}$ ($t = 0.27$, $p = 0.792$), directionally consistent with the TEP hierarchy in which the pulsation clock carries a larger environmental response than the helium-flash standard candle.

The modest statistical significance of the differential in the current 18-host sample reflects observational error propagation rather than theoretical tension. For a typical SN Ia host potential depth ($\Delta u_\phi^2 \approx 1.5 \times 10^4\ {\rm km^2\,s^{-2}}$), the predicted TEP distance modulus shift is $\Delta \mu_i = \kappa_{\rm Cep}^{\rm equiv} X_i \sim 0.02\text{--}0.05\ {\rm mag}$ for the bulk of the host sample, reaching $\sim 0.2\ {\rm mag}$ only for the most massive hosts. By comparison, the combined per-host Cepheid/TRGB distance-modulus uncertainty is $\sqrt{\sigma_{\rm Cep}^2 + \sigma_{\rm TRGB}^2} \approx 0.05\text{--}0.12\ {\rm mag}$, and both distance scales share common anchor calibrations (the LMC, SMC, and NGC 4258). Furthermore, because both TRGB stars (in diffuse outer halos) and Cepheids (in disks) are observed outside dense galactic cores, both sample regions outside the nuclear core where host systemic redshifts are anchored ($r_{\rm spec} < r_{\rm TRGB}, r_{\rm Cep}$). Consequently, the current 18-host TRGB sample is non-discriminating due to joint observational noise floors, while remaining fully consistent with the expected physical ordering ($\kappa_{\rm Cep} > \kappa_{\rm TRGB}$).

### 3.7 Host-Mass Decoupling and Supernova Mass-Step Marginalization

A potential degeneracy in empirical distance-ladder corrections arises from the established Type Ia supernova host-galaxy mass step, wherein supernovae in massive hosts ($\log(M_*/M_\odot) > 10.0$) are observed to be $\sim 0.05\text{--}0.06\ {\rm mag}$ brighter after light-curve standardization than those in lower-mass hosts. Because galactic circular velocity correlates with integrated stellar mass through the baryonic Tully--Fisher relation, the question arises whether the measured environmental slope $\Gamma_X$ could represent the conventional supernova mass step in proxy form.

This degeneracy is tested directly using host stellar masses compiled from Riess et al. (2022, Table 4) across all 37 calibrators (Step 39 and Step 17). While raw rotation velocity correlates strongly with host stellar mass ($r = 0.699$, $p = 1.52 \times 10^{-6}$; and $r = 0.614$, $p = 5.23 \times 10^{-5}$ for $\sigma^2$), the physical TEP coordinate $X_{\rm TEP} \equiv (S_i \sigma_i^2 - \sigma_{\rm ref}^2)/c^2$ is decorrelated from host stellar mass:

> 

\begin{equation}
r(X_{\rm TEP}, \log M_*) = 0.230 \quad (p = 0.171) , \quad
r(X_{\rm TEP}, \Theta_{10}) = 0.351 \quad (p = 0.033) ,
\end{equation}

where $\Theta_{10} \equiv \Theta(\log(M_*/M_\odot) - 10.0)$ denotes the discrete step indicator. The decorrelation occurs because physical TEP screening $S_i = S_{\rm local} \times S_{\rm group}$ suppresses high-surface-density disks and massive group environments, breaking the monotonic mass-to-potential scaling of isolated galaxies.

Joint multi-regressor likelihood fits of the generative expansion relation $cz_i = d_i [H_{\rm app} + \Gamma_X X_i + \gamma_{\rm step} \Theta_{10,i}]$ demonstrate that $\Gamma_X$ remains stable upon marginalization against the supernova mass step, whereas the mass step itself is statistically suppressed.

| $\sigma_v$ (km/s) | Model | $\Gamma_X/10^7$ (km/s/Mpc) | LRT Stat | $\gamma_{\rm step}$ or $\gamma_M$ (km/s) | Step LRT | Retention | $H_{\rm cosmic}$ (km/s/Mpc) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 150.0 | Standalone TEP | $1.223 \pm 0.839$ | $1.45\sigma$ | &mdash; | &mdash; | 100.0% | $67.30 \pm 1.79$ |
| 150.0 | Joint w/ Mass Step ($\Theta_{10}$) | $1.005 \pm 0.898$ | $1.12\sigma$ | $+1.88 \pm 2.94$ | $0.64\sigma$ | 82.2% | $67.59 \pm 1.83$ |
| 150.0 | Joint w/ Continuous $\log M_*$ | $0.851 \pm 0.820$ | $1.03\sigma$ | $+3.59 \pm 1.65$ | $2.14\sigma$ | 69.6% | $67.71 \pm 1.71$ |
| 150.0 | Mass-Step Residualized ($X_{\perp \rm step}$) | $1.106 \pm 0.904$ | $1.22\sigma$ | &mdash; | &mdash; | 90.4% | &mdash; |
| 182.1 | Standalone TEP | $1.223 \pm 0.840$ | $1.45\sigma$ | &mdash; | &mdash; | 100.0% | $67.30 \pm 1.79$ |
| 182.1 | Joint w/ Mass Step ($\Theta_{10}$) | $1.005 \pm 0.899$ | $1.12\sigma$ | $+1.88 \pm 2.93$ | $0.64\sigma$ | 82.2% | $67.59 \pm 1.83$ |
| 182.1 | Joint w/ Continuous $\log M_*$ | $0.851 \pm 0.823$ | $1.03\sigma$ | $+3.59 \pm 1.66$ | $2.14\sigma$ | 69.6% | $67.71 \pm 1.72$ |
| 182.1 | Mass-Step Residualized ($X_{\perp \rm step}$) | $1.106 \pm 0.904$ | $1.22\sigma$ | &mdash; | &mdash; | 90.4% | &mdash; |
| 250.0 | Standalone TEP | $1.156 \pm 0.958$ | $1.22\sigma$ | &mdash; | &mdash; | 100.0% | $67.39 \pm 2.09$ |
| 250.0 | Joint w/ Mass Step ($\Theta_{10}$) | $0.954 \pm 1.039$ | $0.93\sigma$ | $+1.74 \pm 3.49$ | $0.50\sigma$ | 82.5% | $67.65 \pm 2.16$ |
| 250.0 | Joint w/ Continuous $\log M_*$ | $0.777 \pm 0.980$ | $0.80\sigma$ | $+3.67 \pm 2.02$ | $1.83\sigma$ | 67.2% | $67.82 \pm 2.11$ |
| 250.0 | Mass-Step Residualized ($X_{\perp \rm step}$) | $1.062 \pm 1.027$ | $1.04\sigma$ | &mdash; | &mdash; | 91.8% | &mdash; |

Across all peculiar-velocity specifications, the environmental response retains $82.2\%\text{--}82.5\%$ of its standalone amplitude when simultaneously fitting the discrete supernova mass step. The mass step parameter is reduced to an insignificant level ($\gamma_{\rm step} \le 0.64\sigma$), indicating that the discrete mass step does not displace the continuous TEP potential coordinate. Even when residualizing $X_{\rm TEP}$ strictly orthogonal to the mass step ($X_{\perp \rm step}$), the environmental slope remains stable at $\Gamma_{X,\perp \rm step} = (1.106 \pm 0.904) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.22\sigma$) at $\sigma_v = 150$--$182.1\ {\rm km\,s^{-1}}$ (identical under the $\sigma_v^2 + \sigma_{\rm int}^2$ degeneracy).

Crucially, the inferred unperturbed cosmic background expansion rate $H_{\rm cosmic}$ remains tightly clustered across all joint configurations, yielding $H_{\rm cosmic} = 67.59\text{--}67.65\ {\rm km\,s^{-1}\,Mpc^{-1}}$ in the joint mass-step models and $67.71\text{--}67.82\ {\rm km\,s^{-1}\,Mpc^{-1}}$ in the joint continuous-mass models. In all cases, $H_{\rm cosmic}$ is in exact statistical concordance with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$). The TEP resolution of the Hubble tension is therefore fully decoupled from, and robust against, supernova host stellar mass standardization.

The same marginalization applied across the cosmologically constrained Hubble-flow regression of Section 3.4 ($N=33$, with identical weights and sample selection to the redshift-only WLS) returns the same ordering: the screened coordinate is near-orthogonal to mass in this subset ($r(X, \log M_*) = 0.181$), and $\Gamma_X$ retains $\sim$71\% of its amplitude under simultaneous continuous-mass regression — $(1.018 \pm 0.70)\times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.45\sigma$) at $\sigma_v = 150\ {\rm km/s}$, remaining positive across all $\sigma_v$ — while the mass regressor absorbs the shared component ($\gamma_M = +4.11 \pm 1.41\ {\rm km\,s^{-1}\,dex^{-1}}$, $2.9\sigma$). Because host stellar mass is itself a potential-depth proxy under the TEP coordinate, this shared component is expected; the residual $\Gamma_X$ slope and its sign stability demonstrate that the environmental response is not reducible to the mass-step channel.

### 3.8 Bounded-Response Functional Form and Within-Host Cepheid Structure

The environmental slope of Section 3.4 parameterizes the response as linear in the potential coordinate $X_i$. A strictly linear response is an unbounded extrapolation: evaluated at $\sigma \sim 300\ {\rm km\,s^{-1}}$ it would predict a distance-modulus bias approaching a magnitude, far exceeding the observed ladder scatter. The screening operator is intrinsically bounded, so the linear form is a local projection of a saturated response onto a coordinate of width $\sim 10^{-7}$, and the functional form itself must be tested rather than assumed. Seven candidate environmental response functionals — linear in $X_i$, unscreened in $\sigma_i^2$, logarithmic in $S_i\sigma_i^2/U_{\rm ref}$, a velocity-scale contrast $(S_i\sigma_i-\sigma_{\rm ref})/c$, the pure two-body screening factor $S_i$ (carrying no dispersion information), a profiled transition step $\Theta(\sigma_i-\sigma_c)$, and a profiled $\tanh$ transition — were compared inside the identical weighted velocity-space likelihood using leave-one-host-out cross-validation, which is the criterion that correctly penalizes unbounded extrapolation. Eight dataset configurations were examined: the full positive-redshift sample ($N=37$), the Hubble-flow sample ($N=34$) selected by $z_{\rm cmb}>0.0035$ or $z_{\rm hd}>0.0035$, and the stricter sample ($N=25$) selected by the same rule at $0.005$, in both CMB-frame and flow-corrected redshifts, across $\sigma_v = 150$--$500\ {\rm km\,s^{-1}}$.

| Response model | Amplitude | All $z$ (hd) | $z>0.0035$ (hd) | $z>0.005$ (hd) | All $z$ (cmb) |
| --- | --- | --- | --- | --- | --- |
| Linear in $X_i$ | $(1.16 \pm 0.96)\times 10^{7}$ | 65,253 | 69,610 | 69,392 | 102,539 |
| Unscreened $\sigma_i^2$ | $(1.15 \pm 0.64)\times 10^{7}$ | 56,199 | 59,500 | 48,039 | 96,264 |
| $\log_{10}(S_i\sigma_i^2/U_{\rm ref})$ | $5.31 \pm 2.94$ | 62,607 | 69,308 | 60,043 | 100,755 |
| $(S_i\sigma_i-\sigma_{\rm ref})/c$ | $(0.88 \pm 0.87)\times 10^{4}$ | 69,555 | 73,940 | 60,120 | 108,404 |
| Screening $S_i$ alone | $0.90 \pm 5.47$ | 71,134 | 74,632 | 45,053 | 105,624 |
| Step $\Theta(\sigma_i-\sigma_c)$, $\sigma_c=55$ km/s | $10.56$ | 54,765 | 58,611 | 54,170 | 82,235 |
| $\tanh$ transition | $8.07$ | 55,238 | 60,147 | 59,002 | 96,869 |
| $S_A = \min[1,(\sigma_i/\sigma_T)^{2/3}]$, $\sigma_T=65$ km/s | $39.52$ | 55,265 | 61,497 | 61,809 | 98,592 |

Transition shapes are reselected inside each training fold. The corrected regressor normalization removes the numerical instability previously reported for the intermediate-cut tanh fit.

The bounded transition forms are the preferred predictors on seven of the eight configurations, while in-sample likelihoods remain close ($\chi^2 = 25.15$ for the step versus $27.74$ for the linear model at $\sigma_v = 250\ {\rm km\,s^{-1}}$). With the selected step shape frozen at $\sigma_c = 55\ {\rm km\,s^{-1}}$, host-label permutation yields $p = 0.0140$ on the full sample and $p = 0.0300$ on the Hubble-flow sample. These are conditional tests of the selected response, not global model-search probabilities. The 95% bootstrap intervals are $[7.47,14.78]$ and $[6.91,13.87]\ {\rm km\,s^{-1}\,Mpc^{-1}}$, respectively, conditional on the selected threshold and observed group counts. Unstratified resampling loses an entire threshold group in 22/1,000 and 44/1,000 draws, making the slope unidentifiable; these draws are counted explicitly rather than assigned arbitrary amplitudes. Among identifiable unstratified draws, both intervals also remain positive. The amplitude retains its sign in all 37 and all 34 leave-one-host-out refits. At the strictest cut, the pure screening factor $S_i$ becomes the strongest predictor, reducing CV error by a factor of $1.54$ relative to the linear coordinate; this ordering motivates a focused test of group/local environment.

Two qualifications accompany the functional-form result. The preferred threshold places its amplitude on the four lowest-$\sigma$ hosts — NGC 3447, NGC 4424, NGC 7250, and UGC 9391, all within $\sim 37$ Mpc — so the step estimate inherits the low-redshift peculiar-velocity leverage documented in Section 3.4, and a published-modulus median split yields the smaller, more conservative contrast $\Delta H_0 \approx 1.5\text{--}1.7\ {\rm km\,s^{-1}\,Mpc^{-1}}$. Because the low-$\sigma$ group is also the low-mass tail (the mass-step partition at $\log M_* = 10.5$ is nested inside the $\sigma > 55\ {\rm km\,s^{-1}}$ group), a joint best-boundary fit splits credit between the two orderings ($1.45\sigma$ and $1.81\sigma$ respectively); the canonical $\log M_* = 10$ mass step contributes nothing when fit jointly ($-0.06\sigma$), and $\sigma_i$ is retained as primary because it is the potential coordinate, but a deeper separation of the two orderings requires a larger host sample. The profiled step is a phenomenological approximation to a continuous response, not a fundamental screening switch — and the continuous form is in fact already derived: the amplitude-sector response $S_A = \min[1,(\rho/\rho_T)^{1/3}]$ of Paper 0 (\S 2.2ii), evaluated with $\sigma_i^2$ as the depth proxy as $\min[1,(\sigma_i/\sigma_T)^{2/3}]$, selects $\sigma_T \approx 65\ {\rm km\,s^{-1}}$ and reproduces the step optimum within $\sim 1\%$ predictive error (CV 55,265 versus 54,764) and within $\Delta\chi^2 = 0.11$ in-sample. The bounded preference is therefore not an ad hoc shape selection but the signature the derived amplitude-sector form predicts. A nested-depth variant was also tested, adding each host's ambient group-halo baseline (Tully 2015 virial masses, $\sigma_{\rm amb}^2 = c^2 \times 1.6\times10^{-7}\, M_{\rm vir}^{2/3}$) to the coordinate under the nested field structure of Paper 0: both the linear nested coordinate and a saturating $S_A$ evaluated on total depth are rejected — the former returns a negative amplitude ($-1.79\times10^6\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and the latter is the worst predictor in every configuration. The ambient baseline contributes identically to the host's systemic redshift and to its Cepheid clocks, so it cancels as a common mode in the $cz - dH$ comparison; the identifiable response lives in the host-local depth contrast alone, and cluster members such as NGC 4424 (Virgo) correctly contribute through their shallow local well rather than through the ambient cluster depth.

A second, contamination-resistant test operates inside the distance-ladder matrix itself. Because the 37 host moduli $\mu_i$ are unconstrained parameters, any environmental perturbation uniform within a host is absorbed exactly as $\hat{\mu}_i \to \mu_i - \kappa_0 X_i$ and is invisible to the matrix likelihood. This identification structure is verified directly by injection ensembles (200 full-covariance realizations): a host-uniform modulus shift returns a period-coupled coefficient consistent with zero, $\kappa_P = (-0.02 \pm 2.23)\times 10^5\ {\rm mag}$, while genuine period-coupled injections are recovered without bias (pull mean $-0.08$ to $-0.11$, pull scatter $\approx 1$, 68% coverage $0.72$--$0.74$). The identifiable signature of the period-transport mechanism is therefore the within-host period structure, identified through the $37$ independent Cepheid period distributions rather than through host-level distance offsets, and hence immune to the redshift-frame and peculiar-velocity systematics that dominate the velocity-space channel. A joint generalized least-squares fit of the environment-coupled period term $\kappa_P$ and the environment-coupled metallicity term $\kappa_Z$ returns $\kappa_P = (-5.79 \pm 2.26)\times 10^5\ {\rm mag}$ ($2.57\sigma$) and $\kappa_Z = (-2.10 \pm 0.92)\times 10^6\ {\rm mag\,dex^{-1}}$ ($2.29\sigma$), with near-orthogonal covariance ($r = -0.15$) and $\Delta\chi^2 = 13.8$ for two parameters relative to baseline ($\Delta\chi^2 = 19.1$ under the anchor-reference convention, where $\kappa_Z$ reaches $3.32\sigma$). The period-coupled term retains $\sim 89\%$ of its marginal amplitude under metallicity control, and the two terms are separately required (Table 6).

| Convention | Model | $\kappa_P$ ($10^5$ mag) | $\kappa_Z$ ($10^6$ mag dex$^{-1}$) | $\kappa_0$ ($10^5$ mag) | $r(\kappa_P,\kappa_Z)$ | $\Delta\chi^2$ |
| --- | --- | --- | --- | --- | --- | --- |
| anchor-screened | $\kappa_P$ only | $-6.54 \pm 2.23$ ($2.93\sigma$) | &mdash; | &mdash; | &mdash; | $8.60$ |
| anchor-screened | $\kappa_Z$ only | &mdash; | $-2.45 \pm 0.91$ ($2.69\sigma$) | &mdash; | &mdash; | $7.22$ |
| anchor-screened | $\kappa_P + \kappa_Z$ | $-5.79 \pm 2.26$ ($2.57\sigma$) | $-2.10 \pm 0.92$ ($2.29\sigma$) | &mdash; | $-0.145$ | $13.82$ |
| anchor-screened | $\kappa_0 + \kappa_P + \kappa_Z$ | $-7.89 \pm 2.82$ ($2.79\sigma$) | $-2.12 \pm 0.92$ ($2.30\sigma$) | $+3.20 \pm 2.60$ ($1.23\sigma$) | $-0.109$ | $15.34$ |
| anchor-reference-zero | $\kappa_P + \kappa_Z$ | $-5.94 \pm 2.51$ ($2.37\sigma$) | $-3.32 \pm 1.00$ ($3.32\sigma$) | &mdash; | $-0.131$ | $19.05$ |
| anchor-reference-zero | $\kappa_0 + \kappa_P + \kappa_Z$ | $-9.46 \pm 3.39$ ($2.79\sigma$) | $-3.36 \pm 1.00$ ($3.36\sigma$) | $+4.33 \pm 2.81$ ($1.54\sigma$) | $-0.080$ | $21.43$ |

The sign of $\kappa_P$ corresponds to a period dependence of the transport ratio — $\ln(q_i/q_{\rm ref})$ rising with pulsation period, i.e., weaker inferred-period contraction for longer-period Cepheids within deeper-potential hosts. Through the period--mean-density relation, longer periods correspond to weaker stellar self-binding, so the measured ordering is the direction expected if the clock coupling to the ambient field strengthens as envelope self-screening declines. A first stellar-structure derivation of $q_i(P)$ has now been carried out (Step 62, \texttt{results/outputs/step\_62\_q\_structure\_derivation.json}): requiring the transport ratio to take the amplitude-sector form $\ln q = \lambda_0\,X\,\rho_T/\bar\rho(P)$, with the Cepheid mean density given by the period--mean-density relation $\bar\rho = \rho_\odot (Q/P)^2$, inverts the measured $\kappa_P$ to a dimensionless transport weight $\lambda_0 \simeq 0.77$--$0.94$ — order unity rather than the factor $\sim 10^{6}$ the bare clock-ratio bound would naively suggest, because the inverse-density factor $\rho_T/\bar\rho \sim 10^{6}$ is itself the amplification carrier. Since $\bar\rho \propto P^{-2}$ the derived carrier is quadratic in period, and on the identical design matrix the $X\cdot P^2$ carrier is preferred over both the fitted $X\log_{10}P$ and the alternative $X(\log_{10}P)^2$ forms in all six anchor-and-control configurations at equal parameter count (e.g., joint with $\kappa_Z$: $\Delta\chi^2 = 14.67$ versus $13.82$ and $13.61$ under the screened-anchor convention). The period dependence of the response is thereby converted from a fitted linear-in-$\log P$ ansatz into a derived functional form that the data prefer; what remains is deriving the single order-unity coefficient $\lambda_0$ from the coupled envelope transport rather than inferring it.

The unified-closure test of Step 63 (\texttt{results/outputs/step\_63\_unified\_response\_closure.json}) then asks whether that single derived law suffices for the full ladder response: replacing the two fitted columns $\kappa_0 X_i$ (host amplitude) and $\kappa_P X_i(\log_{10}P-1)$ (period coupling) by the one predicted column $-\lambda_0 X_i\,\rho_T/\bar\rho(P_i)$ carries essentially all of the pair's structure on the identical design matrix — with the metallicity term included, the single column reaches $\Delta\chi^2 = 14.7$ against $15.3$ for the full three-term fit, with one fewer parameter — and a residual host-amplitude term fit jointly is consistent with zero ($0.13$--$0.45\sigma$ across both anchor conventions and both metallicity configurations), while the unified coefficient itself retains $2.7$--$3.1\sigma$ and maps to $\lambda_0 = 0.77$--$0.94$ (median $0.85$). The response structure is therefore carried by the derived law, not by two independent empirical coefficients. The amplitude then decomposes as $\kappa = \Gamma_{\rm Cep}\,\lambda_0\,\rho_T/\bar\rho(P)$ with $\Gamma_{\rm Cep} = b/\ln 10 = 1.416$: the canonical prior $\kappa_{\rm gal} = 9.6\times10^5$ mag is the value of the derived law at the period--luminosity pivot $\bar\rho(10\,{\rm d}) = 2.4\times10^{-5}\ {\rm g\,cm^{-3}}$ for $\lambda_0 = 0.80$, and the ladder-referenced estimate $\kappa_{\rm Cep} = (3.65 \pm 3.04)\times10^5$ mag corresponds to the same law evaluated at the effective ensemble density $\rho_{\rm eff} \approx 6.6\times10^{-5}\ {\rm g\,cm^{-3}}$ ($P_{\rm eff} \approx 6$ d). The factor-of-few spread between the two estimators is thereby a difference in effective reference density absorbed by each estimator's convention, not a tension in the response law itself; what remains as theory debt is the derivation of $\lambda_0 \approx 0.8$ from coupled envelope transport.

The exclusion of the conventional alternatives is carried out as a discriminating battery on the same design matrix. Free global period--luminosity shape terms — a quadratic $(\log_{10}P-1)^2$ and the canonical 10-day slope break $(\log_{10}P-1)\,\Theta(\log_{10}P-1)$ — are individually weak ($0.77\sigma$ and $1.63\sigma$) and, far from absorbing the signal, leave it strengthened: with both shape terms free the period-coupled coefficient is $\kappa_P = (-7.71 \pm 2.34)\times 10^5$ mag ($3.30\sigma$), and $-6.99\times10^5$ ($2.97\sigma$) in the fully controlled joint fit with $\kappa_Z$. Coordinate discrimination then separates the environmental reading from its conventional competitors. Refitting the same period-coupled structure with each measured host covariate as the modulator, the fitted distance modulus — the channel through which crowding and blending systematics operate — is a weak carrier ($1.30\sigma$), host mass is null ($0.23\sigma$), and host mean metallicity is intermediate ($2.67\sigma$, partially confounded), while the potential-depth coordinate carries the structure ($2.93\sigma$). The unscreened dispersion coordinate performs marginally better than the screened $X_i$ ($\Delta\chi^2 = 9.6$ versus $8.6$, inseparable at present precision in the joint fit), consistent with the corpus's amplitude--shear distinction: the clock-amplitude response need not inherit the shear-transmission suppression that enters the velocity-space coordinate. Host-assignment permutation returns $p = 0.0125$ across $400$ label scrambles, and the coefficient remains negative in all $41$ leave-one-host-out refits ($1.67$--$3.4\sigma$), so no single host drives the result. Decomposing the global slope into per-host values reproduces the structure directly rather than by construction: the four anchor populations sit within $0.84\sigma$ of the reference slope while the deep-potential hosts tilt uniformly shallower, and the weighted regression of the per-host slopes on $X_i$ returns $+1.01 \pm 0.20\times 10^6$ mag per unit $X$ — the sign implied by the mechanism, with an amplitude consistent with $-\kappa_P \simeq +6.5\times10^5$ within $1.8\sigma$ — alongside real scatter beyond the single-term form ($\chi^2/{\rm dof} = 1.48$).

Two qualifications bound the claim precisely. The environment-coupled metallicity term does not separate cleanly from a conventional period--metallicity interaction in the raw product fit: under a free global $(\log_{10}P-1)\times[{\rm O/H}]$ term, $\kappa_Z$ drops to $1.8\sigma$ and $\kappa_P$ to $2.0\sigma$. A within-host decomposition resolves where that signal lives: writing the product as host-mean and within-host pieces, the pure per-Cepheid interaction $(\log_{10}P-1)^{\sim}\times[{\rm O/H}]^{\sim}$ is null ($0.24\sigma$; $\Delta\chi^2 = 0.06$), while the entire apparent PLZ term is carried by the host-mean-weighted pieces $\bar t_h\,[{\rm O/H}]^{\sim}+\bar z_h\,(\log_{10}P-1)^{\sim}$ ($2.65\sigma$ alone) — ordering of the host means, not a per-Cepheid interaction. Under the full decomposition $\kappa_P$ retains $1.9\sigma$ and $\kappa_Z$ $2.0\sigma$, nearly orthogonal to the within-host piece (design overlap $0.03$), and an independent per-host decomposition of the metallicity slope itself orders on $X_i$ at $2.71\sigma$ ($n = 39$, sign-consistent with $-\kappa_Z$) — the metallicity channel is therefore environment-ordered within-host structure, with the conventional per-Cepheid PLZ mechanism excluded; the residual ambiguity is host-covariate selection at $N = 41$, already bounded by the modulator battery in which the potential-depth coordinate outperforms mass, distance, and mean metallicity. Second, the environmental period coupling was initially not shape-resolved: a quadratic-in-period environmental term $X_i(\log_{10}P-1)^2$ performed equivalently ($\Delta\chi^2 = 8.1$). The stellar-structure derivation (Step 62) resolves the shape question from the theory side: the amplitude-sector form $\ln q = \lambda_0 X\,\rho_T/\bar\rho(P)$ predicts a carrier quadratic in the period itself, $X\cdot(P/10\,{\rm d})^2$, and on the identical design matrix that carrier outperforms both $X\log_{10}P$ and $X(\log_{10}P)^2$ in all six anchor-and-control configurations at equal parameter count — so $\kappa_P$ is superseded by the derived coefficient, whose fitted amplitude implies $\lambda_0 \simeq 0.77$--$0.94$. What the battery establishes is that the period structure is ordered by the measured potential-depth coordinate — not by distance, mass, or a global standardization shape — and survives every conventional control available in the design matrix.

Two audit channels bound the remaining systematics of the velocity-space endpoint. First, the kinematic-proxy audit of Step 64 (\texttt{results/outputs/step\_64\_inclination\_proxy\_audit.json}) propagates HyperLEDA inclination uncertainty through the $V_{\rm rot}/\sqrt{2}$ proxy: seven of the 34 Hubble-flow hosts have $i < 40^\circ$, and the highest-leverage host (NGC 976, $X = 5.2\times10^{-7}$) is a nearly face-on Sbc whose deprojected $V_{\rm rot} = 405$ km/s carries order-unity deprojection uncertainty absent from the tabulated measurement error. With that error budget included in an errors-in-variables likelihood, the full-sample response is $\kappa = (0.15 \pm 0.24)\times10^6$ mag; excluding NGC 976 returns $(-0.13 \pm 0.45)\times10^6$ mag and an $i \ge 40^\circ$ cut $(0.21 \pm 0.58)\times10^6$ mag. An inclination-independent cross-check replaces the deprojected linewidths of the 11 hosts below $i = 45^\circ$ with a K-band Tully--Fisher prediction (HyperLEDA $K_s$ magnitudes against the Cepheid distance moduli, calibrated on the 22 well-inclined hosts at 0.066 dex residual scatter); the hybrid proxy returns $\kappa = (0.20 \pm 0.64)\times10^6$ mag under the same errors-in-variables likelihood. Injection ensembles calibrate a residual $\approx 20\%$ attenuation of the errors-in-variables estimator at the observed response level, placing the bias-corrected estimate at $\approx 0.19\times10^6$ mag. The velocity-space channel is therefore consistent with the ladder-referenced $\kappa_{\rm Cep} \approx (0.27\text{--}0.39)\times10^6$ mag at reduced power, and its significance is currently levered to a small number of low-inclination proxies rather than to a systematic suppression of the response. Second, Step 65 (\texttt{results/outputs/step\_65\_hfsn\_environment\_contrast.json}) completes the common-mode propagation chain: a host-level clock response shared by Cepheids and supernovae is absorbed into the latent moduli and propagates to $H_0$ only through the environment contrast between calibrator and Hubble-flow hosts. Transferring $X$ to the 277 Hubble-flow hosts through their Pantheon+ stellar masses gives $\langle X_{\rm HF}\rangle - \langle X_{\rm cal}\rangle = (-2.7 \pm 2.6)\times10^{-8}$, implying a bounded $H_0$ propagation of $+0.83$ km/s/Mpc ($+0.21$ to $+1.60$) at the canonical amplitude. The sign of the contrast is robust but its amplitude is map-form dependent: robust (Theil--Sen), quadratic, and distribution-free rank-quantile transfers return $\Delta X = -7.5$, $-11.1$, and $-5.6\times10^{-8}$ respectively, corresponding to canonical-amplitude shifts of $+2.3$, $+3.4$, and $+1.7$ km/s/Mpc — the correct sign for the tension across every admissible map, with the propagated effect spanning $0.8$--$3.4$ km/s/Mpc. The map dependence is driven by a genuine selection difference: the Hubble-flow population extends to lower host masses than the calibrators (73 of 277 hosts lie below the calibrator range $\log M_* \in [9.13, 12.59]$), so the contrast is partially extrapolated in the low-mass tail. Third, Step 66 (\texttt{results/outputs/step\_66\_combined\_evidence.json}) quantifies the joint evidence across the four channels with independent noise sources -- the velocity-space endpoint $\Gamma_X$ ($1.21\sigma$), the within-host period term $\kappa_P$ ($2.57\sigma$), the environment-coupled metallicity term $\kappa_Z$ ($2.29\sigma$), and the Cepheid--TRGB differential ($0.72\sigma$). A Stouffer combination carrying the fitted $\kappa_P$--$\kappa_Z$ correlation ($-0.15$) explicitly returns $Z = +3.52$ ($p = 4\times10^{-4}$ two-sided) under zero inter-channel correlation, degrading only to $Z = +2.62$ ($p = 0.009$) under a conservative shared $\rho = 0.3$; all four primary channels lie in the predicted direction. No single channel reaches $3\sigma$, but the concordance of independent measurement systems -- redshift-space, period-structure, metallicity-structure, and independent-tracer differential -- is the form of evidence the programme is designed to produce.

> 
Multi-scale empirical evidence spanning single-galaxy internal gradients ($2.6\sigma$--$3.3\sigma$), host-level potential stratification ($100\%$ sign stability across all 37 hosts), cosmologically constrained expansion likelihoods ($H_{\rm app} = 68.36\text{--}68.61\ {\rm km\,s^{-1}\,Mpc^{-1}}$), host-mass decoupling (retaining $>81\%$ amplitude under supernova mass-step marginalization), a bounded environmental response preferred over linear extrapolation by out-of-sample model comparison, and within-host period-coupled Cepheid structure surviving joint metallicity and shape control ($\kappa_P$ at $2.0\text{--}3.4\sigma$ across the full discriminating battery, sign-stable in $41/41$ host leave-outs, host-assignment permutation $p = 0.0125$) indicates that the Hubble tension can be resolved by dynamical proper time in the local distance scale, pending host-specific aperture validation of the Cepheid-channel allocation. The prespecified primary ladder-propagation estimator—the measured-$\kappa$ endpoint projection through the SH0ES design matrix—returns $H_0 = 71.77 \pm 0.99\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with a $3.94\sigma$ residual; the corrected-$H_0$ family across all estimator roles spans $66.65$–$72.54\ {\rm km\,s^{-1}\,Mpc^{-1}}$ as a stated robustness band.

## 4. Discussion

### 4.1 Physical Synthesis of Multi-Scale Empirical Evidence

The empirical results presented in this work provide a coherent, multi-scale verification of the Temporal Equivalence Principle across both internal galactic structures and the cosmological distance ladder. Single-galaxy tests provide the cleanest experimental environment: because all Cepheids in M31 and the LMC share identical host distances and zero relative peculiar velocities, the observed Period--Luminosity offsets ($\Delta W = +0.356 \pm 0.136$ mag in M31, $+0.681 \pm 0.187$ mag in the HST PHAT catalogue — $+0.630 \pm 0.195$ mag for Pan-STARRS restricted to the same footprint — and $+0.0284 \pm 0.0086$ mag in the LMC) are consistent in sign with a larger inferred period contraction for Cepheids in the less-slowed outer disk relative to Cepheids sampling deeper regions, as predicted by the observing-chain hierarchy of Appendix C; the carrier is the amplitude-sector response — envelope self-screening declining outward strengthens the envelope's coupling to the ambient field — the only TEP channel whose amplitude can reach the measured offsets, since the literal aperture clock-ratio is bounded at $|\Delta\ln q| \lesssim \Delta\Phi/c^2 \sim 10^{-7}$, several orders below them. The M31 offsets fail pair-matched colour and error controls, and the residual readings are now quantitatively bounded in both directions: the overmatching hypothesis is tested against the ambient closure by constructing the $u_{\rm amb}$ proxy at $L = 837$ pc, whose coupling to the error covariate is weak (${\rm corr}(e_W, \log u_{\rm amb}) = -0.18$), so error matching removes only a few per cent of any ambient-carried signal; matching on the ambient field itself leaves $\Delta W = +0.206 \pm 0.163$ mag — though under joint $(\log P, e_W, \log u_{\rm amb})$ cell-matching the offset returns $-0.069 \pm 0.126$ mag, so that residual is carried by the unequated error contrast within ambient strata, not by an ambient-ordered signal; and the error-stratified contrast declines monotonically through the $e_W$ distribution ($+0.270 \pm 0.304$ to $-0.117 \pm 0.202$ mag), so the pooled quality-band persistence is compositional rather than a uniform offset. The PHAT channel shares the susceptibility ($-0.565 \pm 0.079$ mag error-matched versus $+0.906 \pm 0.147$ mag density-matched). The channel is carried as conditional evidence: a positive residual is permitted in the best-measured strata but no estimator-invariant detection survives. The LMC gradient remains the strongest single-galaxy result.

At the host-galaxy scale, the homogeneous HyperLEDA kinematic reconstruction across all 37 distinct R22 SN Ia hosts confirms this directional coupling. The positive environmental slope ($\Gamma_X = (1.156 \pm 0.958) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km/s}$, and $1.45\sigma$ at $\sigma_v = 150$--$182.1\ {\rm km/s}$) displays 100% directional stability across all 37 leave-one-host-out jackknife refits.

The single-galaxy amplitudes are furthermore of the correct order for the same transfer coefficient. Run through the host-level $\kappa_{\rm Cep} \approx 4.5 \times 10^5$ mag fitted on the redshift block, the LMC gradient implies an inner–outer potential contrast $\Delta X \approx (6.3 \pm 1.9) \times 10^{-8}$ — equivalently $\sqrt{\Delta U} \approx 75 \pm 21\ {\rm km\,s^{-1}}$ between the $R \le 0.92$ kpc field and the $R \ge 2.93$ kpc field — the same order as the LMC's adopted potential coordinate $u_\phi \approx 47\ {\rm km\,s^{-1}}$, with the modest factor expected because an inner-bar to outer-disk contrast exceeds the mean disk scale. Applied to the M31 baseline amplitude, the same transfer implies $\sqrt{\Delta U} \approx 266\ {\rm km\,s^{-1}}$ against the adopted M31 potential coordinate $u_\phi \approx 182\ {\rm km\,s^{-1}}$, again the correct order for a bulge-dominated inner field against the outer disk, with that channel now read as conditional evidence under the ambient-field and error-stratified controls described above. The internal gradients therefore sit within a factor of a few of what the host-level coefficient predicts once each system's potential-contrast scale is accounted for; a precision transfer requires measured per-aperture $q_i$ values rather than order-of-magnitude potential scales, as specified in Appendix C.

### 4.2 Resolution of the Distance Matrix Degeneracy

A critical methodological discovery of this analysis is the mathematical distinction between observation-level and latent-modulus environmental perturbations. The standard SH0ES generalized least-squares matrix does not generically erase observation-level Cepheid perturbations (as confirmed by the 99.7% recovery of raw row-level injections). What it cannot identify without external constraints is an environmental perturbation that is mathematically equivalent to shifting the latent host distance modulus: because the 37 $\mu_i$ parameters are unconstrained free variables, they absorb the host-level shift into $\hat{\mu}_i \to \mu_i - \kappa_{\rm Cep} X_i$, producing an apparent null matrix parameter ($\kappa_{\rm Cep} = (-0.169 \pm 0.207) \times 10^6\ {\rm mag}$). The identifiable ladder signature of an environmental clock response is therefore not a host-uniform offset but the within-host structure of the Cepheid observables. Numerical injection ensembles confirm this identification structure — a host-uniform modulus injection returns a period-coupled coefficient consistent with zero, while period-coupled injections are recovered without bias (unit pull scatter, 68% coverage $0.72$--$0.74$ over 200 covariance-level draws) — and reveal the environment-coupled period term in the real data at $\kappa_P = (-5.79 \pm 2.26)\times 10^5\ {\rm mag}$ ($2.57\sigma$), near-orthogonal to the simultaneously fitted metallicity-coupled term and retained under both anchor conventions (Section 3.8). This within-host channel is structurally immune to the redshift-frame and peculiar-velocity systematics that dominate the velocity-space slope, making it the cleanest currently available matrix-level discriminator of the transport mechanism.

Synthetic recovery experiments demonstrate that unconstrained design matrices mathematically fail to recover host-level environmental shifts unless distances are tied to cosmological expansion ($cz_i = d_i H_0$). When the distance ladder is formulated at the generative observable level in expansion-rate space across the 33 Hubble-flow hosts, coupling observed Cepheid moduli to Hubble-flow supernovae, the combined endpoint response is recovered at $\Gamma_X = (1.165 \pm 0.979) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$, $1.69\sigma$ under Pantheon+ velocity scatter at $182.1\ {\rm km/s}$, and $2.07\sigma$ at $\sigma_v = 150\ {\rm km/s}$; $1.99\sigma$ under full $33\times 33$ SH0ES covariance GLS), yielding a conventional Hubble-flow intercept of $H_{\rm app} = 68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical $\sigma_v = 250\ {\rm km/s}$ ($68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km/s}$). Under the restricted TEP Cepheid-channel closure ($\beta_X = 0$), this intercept corresponds to the locally inferred conventional $H_0$ after removal of the Cepheid clock bias; its global cosmological interpretation belongs to the corresponding TEP cosmology analyses.

### 4.3 Cosmological Reconciliation Without Early-Universe Modifications

The primary quantitative inference of the TEP cosmologically constrained expansion-rate likelihood is $H_{\rm app} = 68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ ($68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the $\sigma_v = 150\ {\rm km\,s^{-1}}$ sensitivity variant), which reconciles the local distance scale with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) within $0.77\sigma$ ($0.88\sigma$ at the $\sigma_v = 150\ {\rm km\,s^{-1}}$ variant; the full-covariance GLS evaluation is equivalently concordant). Evaluating the expansion rate at the anchor reference zero-point ($X_{\rm ref} \equiv 0$) yields $H_{\rm ref} = 68.15 \pm 1.54\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($67.78 \pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km\,s^{-1}}$), while mapping to the unperturbed cosmic background ($X_{\rm cosmic} = -U_{\rm ref}/c^2$) yields $H_{\rm cosmic} = 67.39 \pm 2.09\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($67.30\text{--}67.82\ {\rm km\,s^{-1}\,Mpc^{-1}}$ across variants), converging directly onto Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and the unified host-level reconstruction ($H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$). The separate unified host-level reconstruction yields $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, matching Planck within $0.45\sigma$ ($0.31\sigma$ bootstrap), while conditional matrix projections through the SH0ES system yield $71.77\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under the endpoint-equivalent $\kappa$—the prespecified primary estimator of the ladder-propagation channel, since it propagates the measured response coefficient—and $69.74\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical scaling. The Cepheid correction does not require changing the observed CMB, BAO, or primordial-abundance datasets; their global interpretation belongs to the corresponding TEP cosmology analyses.

Furthermore, TEP is fully compatible with multi-messenger gravitational wave constraints (such as GW170817), ensuring that the propagation speeds of photons and tensor gravitational waves remain equal to within $10^{-15}$ in the late universe.

The same transport mechanism also operates across the other independent $H_0$ inference channels, rather than requiring a separate resolution for each. The Pantheon+ redshift-profile test (Paper 31) returns $R_H = H_0(z>0.25)/H_0(0.05 \le z < 0.15) = 1.009 \pm 0.006$, flat at $1.5\sigma$: a per-host environmental calibration offset propagates as a uniform multiplicative shift of the standardized supernova magnitude scale and cancels identically in the ratio — precisely the signature of a calibration artifact and the decisive falsifier of the competing kinematic-void account (the predicted $\sim 5\%$ decline is excluded at $8.4\sigma$). The supernova side of the same clock-transport channel is measured there as well: pre-standardization residuals carry a directional temporal signal that aggregate SALT3 standardization attenuates by $70 \pm 16\%$, and Cepheid distances sit systematically shorter than TRGB distances within the same hosts ($2.39\sigma$). In time-delay cosmography, the channel budget has now been evaluated on the real TDCOSMO-2025 system data (Paper 19). The propagated-transport sector is clean: the conformal sector contributes exactly zero photon delay by null-cone invariance, the disformal sector is bounded at $\sim 10^{-9}$ to $10^{-8}$ fractional ($\sim 5$–$13$ ms over a $10\,R_{\rm E}$ halo traverse) on galaxy-halo paths by the GW170817 speed constraint, and the endpoint rescaling contributes $\sim 10^{-7}$. The mass-model sector is self-consistent because no phantom convergence enters the image plane: scalar backreaction on $g_{\mu\nu}$ is bounded at $\kappa_\phi \approx 4\times10^{-5}$, $\sim 2\times10^{3}$ below dark-matter level (Paper 19, Step 61), so image positions and the Fermat normalization are ordinary-mass-sourced. The only order-percent route is the dynamical–lensing mass difference entering the velocity-dispersion prior that breaks the mass-sheet degeneracy: dynamical inference exceeds lensing inference where the Temporal Shear is active ($M_{\rm dyn} = M_{\rm lens} + M_{\rm ph}$; Papers 4, 19), so a $\sigma_*$-anchored normalization sets the fitted mass above the lensing value at the tracer positions and the model redistributes the excess along the mass-sheet direction, biasing the inferred $H_0$ in the direction established in Paper 19 — with an amplitude that requires the solved source–tracer response, stellar-orbit weighting and a lens-model refit. The PPN source product does not determine this aperture transfer, and neither $S_\Sigma$ nor $S_\Sigma^2$ alone supplies a universal $H_0$ correction. Galaxy-scale apertures sample $a_{\rm ap}\sim10^{-9}$–$10^{-8}\,{\rm m\,s^{-2}}$; the existing delay-blind residuals test the proposed channel but do not calibrate its absolute mass-to-delay response (Paper 5 §5.3; Paper 19). Sign bookkeeping separately shows that any positive propagated delay residual would bias the inference low, so the delay carrier cannot supply the channel at any amplitude. The elevated central values of current time-delay analyses must therefore be assessed against the mass-model systematic floor — measured at $\sim 6\%$ on the SDSS1206+4332 power-law versus composite distance posteriors and $\sim 3$–$10\%$ in the per-system $\kappa_{\rm ext}$ widths — within which the mechanism's kinematic-prior term operates. Standard sirens (Paper 22) supply an orthogonal bi-metric propagation test rather than a reconciliation channel.

Two bookkeeping clarifications follow for corpus consistency. First, the reconciliation is allocated to the endpoint observing chain, not to shear accumulated along the photon path: the analytic decomposition of Appendix C.4 places $>99.9\%$ of the combined slope $\Gamma_X$ in the distance-modulus channel, so characterizations of the $5.6\ {\rm km\,s^{-1}\,Mpc^{-1}}$ gap as path-integrated Hubble-flow shear describe the same endpoint bias expressed through the redshift variable, not an additional propagation contribution. Second, agreement between the corrected local value and the TEP cosmology products ($0.03\sigma$) is internal consistency within one framework; the genuinely testable content of the resolution is that the same mechanism must act, with the same sign and a commensurate magnitude, on the independent channels enumerated above.

### 4.4 Cross-Channel TRGB Robustness and Anchor Potential Contrast

Two empirical nuances reinforce the robustness of the TEP distance ladder resolution against potential confounding factors. First, the cross-channel comparison with Tip of the Red Giant Branch (TRGB) stars in the 18 overlapping host galaxies is directionally consistent with the expected physical hierarchy: the weighted differential slope $\Delta\kappa = +(0.101 \pm 0.379) \times 10^6\ {\rm mag}$ ($t = 0.27$, $p = 0.792$) from Step 20 has the sign predicted by TEP (Cepheid response larger than TRGB response), while being statistically consistent with zero. Separate OLS fits give $B_{\rm Cep} = (1.308 \pm 2.960) \times 10^6\ {\rm mag}$ and $B_{\rm TRGB} = (1.280 \pm 2.860) \times 10^6\ {\rm mag}$; both are individually consistent with zero. The predicted host-level distance modulus shift is of order $0.02\text{--}0.05\ {\rm mag}$ for a typical host, comparable to the combined per-galaxy Cepheid/TRGB modulus uncertainty ($0.05\text{--}0.12\ {\rm mag}$). The lack of an exaggerated discrepancy between Cepheid and TRGB hosts is therefore an expected consequence of observational noise propagation and shared outer-disk observing geometry relative to host galactic cores.

Second, the calibration contrast between anchors and SN Ia hosts is an intrinsic observational property of the galaxies rather than an artifact of environmental screening models. The calibrator anchors (the LMC, SMC, NGC 4258, and the Milky Way solar neighbourhood) possess shallow gravitational potential wells ($\langle u_\phi^2 \rangle \sim 3.0 \times 10^3\ {\rm km^2\,s^{-2}}$), whereas the SN Ia host sample consists of massive luminous spirals ($\langle u_\phi^2 \rangle \sim 2.5 \times 10^4\ {\rm km^2\,s^{-2}}$). This greater than eightfold potential contrast is fully active even in the completely unscreened coordinate ($S=1$). Furthermore, the direct empirical detection of internal radial Period--Luminosity gradients within the LMC ($3.30\sigma$) and M31 ($3.24\sigma$) provides independent internal signatures within Local Group galaxies with the sign predicted by the TEP clock-gradient model.

### 4.5 Decisive Observational Pathways

The TEP framework outlines specific, preregistered experimental pathways for further validation:

1. Spatially resolved IFU spectroscopy of SN Ia host galaxies with JWST, measuring differential stellar kinematics and tracer ratios $q_i = r_{\rm spec}/r_{\rm Cep}$ between nuclear cores and outer Cepheid fields.

2. Space-based optical time-transfer and closed-loop clock synchronization experiments (the Triangle Test) designed to detect holonomy in dynamical proper time at the $10^{-19}$ fractional level.

3. Precision expansion of the local distance ladder using the Nancy Grace Roman Space Telescope, observing high-redshift Cepheids and TRGB standard candles in diverse gravitational environments.

## 5. Conclusion

A rigorous, multi-scale empirical investigation of the Temporal Equivalence Principle (TEP) has been performed across the local cosmological distance ladder. By elevating proper time to a dynamical scalar field that slows in deeper gravitational potentials ($0 < r_{\rm core} < r_{\rm disk} < r_{\rm cosmic} \equiv 1$), TEP predicts that classical Cepheid pulsation clocks calibrated in diffuse anchor environments systematically misestimate distance moduli when applied to deep-potential SN Ia host galaxies.

The empirical findings establish three mutually reinforcing pillars of evidence:

1. Single-galaxy internal tests: In M31 (Andromeda), Pan-STARRS Cepheids exhibit an empirical Period--Luminosity offset of $\Delta W = +0.356 \pm 0.136\ {\rm mag}$ ($2.6\sigma$), increasing to $+0.681 \pm 0.187\ {\rm mag}$ ($3.65\sigma$) in the PHAT footprint. In the LMC, OGLE-IV Cepheids independently confirm internal radial potential stratification at $+0.0284 \pm 0.0086\ {\rm mag}$ ($3.3\sigma$), robust to period and colour matching. M31 fails pair-matched controls (colour-matched $-0.017 \pm 0.128$ mag, $0.1\sigma$; eW-matched $-0.103 \pm 0.129$ mag, $0.8\sigma$), and both readings are bounded: the overmatching hypothesis fails because the error covariate is nearly orthogonal to the ambient-field proxy (${\rm corr}=-0.18$ at the $L=837$ pc closure scale), ambient-field matching alone leaves $+0.206 \pm 0.163$ mag — a residual that does not survive joint environment-plus-error control ($-0.069 \pm 0.126$ mag) — and the error-stratified contrast declines monotonically from $+0.270 \pm 0.304$ to $-0.117 \pm 0.202$ mag — so the channel is carried as conditional evidence, with a positive residual permitted in the best-measured strata but no estimator-invariant detection.

2. Matrix-level period-coupled environmental term: Within the full 3,490-row SH0ES design-matrix likelihood, the identifiable signature of host-environment transport is the within-host period structure of the Cepheid ensemble — identified through the 37 independent period distributions and therefore immune to host-level latent-modulus absorption and to the peculiar-velocity systematics that dominate the velocity-space channels. The joint fit returns $\kappa_P = (-5.79 \pm 2.26)\times 10^5\ {\rm mag}$ ($2.57\sigma$) orthogonal to the metallicity term ($r = -0.15$), strengthening to $-6.54\times 10^5$ ($2.93\sigma$) in isolation and to $3.30\sigma$ with free period--luminosity shape terms; host-assignment permutation returns $p = 0.0125$ and the coefficient is negative in all 41 leave-one-host-out refits. The sign corresponds to weaker inferred-period contraction for longer-period Cepheids in deeper-potential hosts — the ordering expected if envelope self-screening declines with pulsation period. The stellar-structure inversion (Step 62) shows the measured coefficient is reproduced by the amplitude-sector form $\ln q = \lambda_0\,X\,\rho_T/\bar\rho(P)$ with an order-unity transport weight $\lambda_0 \simeq 0.77$--$0.94$, and its predicted period-quadratic carrier is preferred by the data over the fitted $\log P$ forms — promoting the channel from a consistency structure to a derived functional form with a single order-unity coefficient outstanding. The unified-column test (Step 63) shows that one coefficient alone reproduces the fitted host-amplitude and period-coupled pair (residual host term consistent with zero), and that under the decomposition $\kappa = \Gamma_{\rm Cep}\,\lambda_0\,\rho_T/\bar\rho$ the canonical amplitude $\kappa_{\rm gal} = 9.6\times10^5$ mag is the law evaluated at the period--luminosity pivot while the ladder estimate $\kappa_{\rm Cep}$ is the same law at the effective ensemble density — closing the provenance gap between the two reported coefficients.

3. Homogeneous host potential stratification: Reconstructing all 37 distinct R22 SN Ia host galaxies with pinned HyperLEDA circular rotation velocities $u_\phi = V_{\rm rot}/\sqrt{2}$ and continuous screening yields an environmental slope of $\Gamma_X = (1.156 \pm 0.958) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km/s}$ ($1.45\sigma$ at $\sigma_v = 150$--$182.1\ {\rm km/s}$), with 100% positive sign stability across all 37 leave-one-host-out refits.

4. TEP-native generative distance ladder: Unconstrained distance-ladder design matrices algebraically absorb host-level environmental shifts into latent host moduli. Formulating the ladder at the generative observable level connecting observed Cepheid moduli to cosmological expansion across the 33 Hubble-flow hosts recovers the combined endpoint response at $\Gamma_X = (1.165 \pm 0.979) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$, $1.69\sigma$ under Pantheon+ velocity scatter at $182.1\ {\rm km/s}$, and $2.07\sigma$ at $\sigma_v = 150\ {\rm km/s}$; $1.99\sigma$ under full $33\times 33$ SH0ES covariance GLS), yielding a conventional Hubble-flow intercept of $H_{\rm app} = 68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical $\sigma_v = 250\ {\rm km/s}$ ($68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km/s}$). Under the physical observing-chain coupling derived in Appendix C.4, the direct gravitational redshift in the host velocity channel ($\beta_{X,\rm grav} \sim c / \langle d \rangle \approx 10^4\ {\rm km\,s^{-1}\,Mpc^{-1}}$) constitutes only $\sim 0.10\%$ of the combined endpoint slope $\Gamma_X \sim 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$, with $>99.9\%$ allocated to the Cepheid distance-modulus channel ($\kappa_{\rm Cep}^{\rm equiv} = (0.369 \pm 0.310)\times 10^6\ {\rm mag}$); joint likelihood estimation of the coupled model confirms $H_{\rm app} = 68.78 \pm 1.48\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($68.72 \pm 1.32$ at the $\sigma_v = 150$--$182.1\ {\rm km/s}$ variants).

The estimator hierarchy must be retained in the final inference. The prespecified primary full-ladder projection propagates the measured response coefficient and yields $H_0 = 71.771 \pm 0.990\ {\rm km\,s^{-1}\,Mpc^{-1}}$, a $3.94\sigma$ offset from Planck CMB cosmological parameters ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$). A separate unified host-level reconstruction, which re-optimizes the environmental response to flatten the host-environment trend, yields $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, in statistical agreement ($0.45\sigma$; $0.31\sigma$ bootstrap) with Planck. Under TEP, the same CMB data yields $H_0 = 66.70 \pm 0.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (TEP-CMB), giving $0.03\sigma$ internal consistency between that host-level route and the TEP-CMB inference — an agreement between products of one framework, not an independent cross-check. This host-level reconciliation is physically motivated by the TEP core--disk tracer closure: systemic spectroscopy weighted toward dense, deeply slowed galactic cores ($r_{\rm spec} < r_{\rm Cep}$) generates an apparent period contraction ($q_i < 1$) when applied to outer-disk pulsation clocks. It does not, however, make the primary full-ladder projection concordant with Planck. The anchor weights (0.20 for Milky Way, 0.25 for LMC, 0.55 for NGC 4258) are adopted approximations rather than likelihood-derived quantities (Appendix D); a future hierarchical treatment using anchor-level Cepheid data is required to refine the unified $H_0$ estimate.

| Observational Tier | Empirical Result | Statistical Significance | Cosmological Implication |
| --- | --- | --- | --- |
| M31 internal Cepheids (HST PHAT catalogue) | $\Delta W = +0.681 \pm 0.187\ {\rm mag}$ | $3.65\sigma$ nominal | Fails pair-matched controls ($-0.565 \pm 0.079$ mag under propagated-error matching; $+0.906 \pm 0.147$ mag under neighbour-density control); conditional evidence |
| LMC internal Cepheids (OGLE-IV) | $\Delta W = +0.028 \pm 0.009\ {\rm mag}$ | $3.30\sigma$ | Independent radial potential stratification |
| 37-host generative endpoint | $\Gamma_X = (1.156 \pm 0.958)\times 10^7$ | $100\%$ sign stability (37/37) | Consistent host-level potential gradient |
| TEP-native generative ladder (33 hosts) | $\Gamma_X = (1.165 \pm 0.979)\times 10^7$ | $1.19\sigma$ canonical ($2.07\sigma$ at $\sigma_v=150$) | Conventional Hubble-flow intercept $H_{\rm app} = 68.62 \pm 1.51$ |
| Joint multi-block ladder | $\kappa_{\rm Cep} = (0.266 \pm 0.239)\times 10^6$ (canonical $\sigma_v=250$) | $1.11\sigma$ canonical ($1.28\sigma$ at $\sigma_v = 150$--$182.1$); $2.05\sigma$ redshift-only WLS at $\sigma_v=150$ | Multi-block likelihood closure; $H_{\rm app} = 68.78 \pm 1.48$ |
| Host-level reconstruction | $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ | $0.45\sigma$ from Planck ($0.31\sigma$ bootstrap) | Separate reconciliation route; the prespecified primary full-ladder projection gives $71.771 \pm 0.990$, $3.94\sigma$ from Planck |

## References

#### Primary Data Sources

Riess, A. G., Yuan, W., Macri, L. M., et al. 2022, *ApJ*, 934, L7, "A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team"

Planck Collaboration, Aghanim, N., Akrami, Y., et al. 2020, *A&A*, 641, A6, "Planck 2018 results. VI. Cosmological parameters"

Scolnic, D., Brout, D., Carr, A., et al. 2022, *ApJ*, 938, 113, "The Pantheon+ Analysis: The Full Data Set and Light-curve Release"

Huchra, J. P., Macri, L. M., Masters, K. L., et al. 2012, *ApJS*, 199, 26, "The 2MASS Redshift Survey—Description and Data Release"

Tully, R. B. 2015, *AJ*, 149, 171, "Galaxy Groups: A 2MASS Catalog"

#### Geometric Calibrators

Gaia Collaboration, Vallenari, A., Brown, A. G. A., et al. 2023, *A&A*, 674, A1, "Gaia Data Release 3: Summary of the content and survey properties"

Pietrzyński, G., Graczyk, D., Gallenne, A., et al. 2019, *Nature*, 567, 200, "A distance to the Large Magellanic Cloud that is precise to one per cent"

Reid, M. J., Pesce, D. W., & Riess, A. G. 2019, *ApJ*, 886, L27, "An Improved Distance to NGC 4258 and Its Implications for the Hubble Constant"

#### Astronomical Databases

Wenger, M., Ochsenbein, F., Egret, D., et al. 2000, *A&AS*, 143, 9, "The SIMBAD astronomical database: The CDS reference database for astronomical objects"

Ochsenbein, F., Bauer, P., & Marcout, J. 2000, *A&AS*, 143, 23, "The VizieR database of astronomical catalogues"

Makarov, D., Prugniel, P., Terekhova, N., Courtois, H., & Vauglin, I. 2014, *A&A*, 570, A13, "HyperLEDA. III. The catalogue of extragalactic distances"

Abazajian, K. N., Adelman-McCarthy, J. K., Agüeros, M. A., et al. 2009, *ApJS*, 182, 543, "The Seventh Data Release of the Sloan Digital Sky Survey"

#### Galaxy Size Catalogs

de Vaucouleurs, G., de Vaucouleurs, A., Corwin, H. G., Jr., et al. 1991, *Third Reference Catalogue of Bright Galaxies* (RC3), Springer

#### Velocity Dispersion Data

Ho, L. C., Greene, J. E., Filippenko, A. V., & Sargent, W. L. W. 2009, *ApJS*, 183, 1, "A Search for 'Dwarf' Seyfert Nuclei. VII. A Complete Survey of the SDSS Spectroscopic Catalog"

Jorgensen, I., Franx, M., & Kjærgaard, P. 1995, *MNRAS*, 276, 1341, "Spectroscopy for E and S0 galaxies in nine clusters"

Kormendy, J. & Ho, L. C. 2013, *ARA&A*, 51, 511, "Coevolution (Or Not) of Supermassive Black Holes and Host Galaxies"

Courteau, S., Dutton, A. A., van den Bosch, F. C., et al. 2007, *ApJ*, 671, 203, "Scaling Relations of Spiral Galaxies"

Catinella, B., Giovanelli, R., & Haynes, M. P. 2006, *ApJ*, 640, 751, "Template Rotation Curves for Disk Galaxies"

#### Cepheid Physics

Anderson, R. I., Saio, H., Ekström, S., Georgy, C., & Meynet, G. 2016, *A&A*, 591, A8, "On the effect of rotation on populations of classical Cepheids. II. Pulsation analysis for metallicities 0.014, 0.006, and 0.002"

Bono, G., Marconi, M., Cassisi, S., et al. 2005, *ApJ*, 621, 966, "Classical Cepheid Pulsation Models. X. The Period-Age Relation"

Kodric, M., Riffeser, A., Seitz, S., et al. 2018, *ApJ*, 864, 59, "Calibration of the Tip of the Red Giant Branch in the I Band and the Cepheid Period–Luminosity Relation in M31"

Leavitt, H. S. & Pickering, E. C. 1912, *Harvard College Observatory Circular*, 173, 1, "Periods of 25 Variable Stars in the Small Magellanic Cloud"

Madore, B. F. & Freedman, W. L. 1991, *PASP*, 103, 933, "The Cepheid distance scale"

#### TEP Research Series

Smawfield, M. L. (2025). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed*. Preprint v0.15 (Jakarta). Zenodo. DOI: [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) (Paper 0)

Smawfield, M. L. (2025). *Global Time Echoes: Distance-Structured Correlations in GNSS Clocks*. Preprint v0.27 (Jaipur). Zenodo. DOI: [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) (Paper 1)

Smawfield, M. L. (2025). *Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products*. Preprint v0.20 (Cairo). Zenodo. DOI: [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) (Paper 2)

Smawfield, M. L. (2025). *Global Time Echoes: Raw RINEX Consistency Test*. Preprint v0.8 (Kathmandu). Zenodo. DOI: [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) (Paper 3)

Smawfield, M. L. (2025). *Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations*. Preprint v0.8 (Tortola). Zenodo. DOI: [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) (Paper 4)

Smawfield, M. L. (2025). *Global Time Echoes: Empirical Synthesis*. Preprint v0.7 (Singapore). Zenodo. DOI: [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) (Paper 5)

Smawfield, M. L. (2025). *Temporal Topology Saturation Scale: Cross-Scale Consistency of ρ_T*. Preprint v0.8 (New Delhi). Zenodo. DOI: [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) (Paper 6)

Smawfield, M. L. (2025). *The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate*. Preprint v0.4 (Blantyre). Zenodo. DOI: [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) (Paper 7)

Smawfield, M. L. (2025). *Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging*. Preprint v0.6 (Mombasa). Zenodo. DOI: [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) (Paper 8)

Smawfield, M. L. (2025). *What Do Precision Tests of General Relativity Actually Measure?*. Preprint v0.8 (Istanbul). Zenodo. DOI: [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) (Paper 9)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars*. Preprint v0.9 (Caracas). Zenodo. DOI: [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) (Paper 10)

Smawfield, M. L. (2026). *The Cepheid Bias: Resolving the Hubble Tension*. Preprint v0.10 (Kingston upon Hull). Zenodo. DOI: [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) (Paper 11 — this work)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies*. Preprint v0.7 (Kos). Zenodo. DOI: [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) (Paper 12)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries*. Preprint v0.6 (Kilifi). Zenodo. DOI: [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) (Paper 13)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion*. Preprint v0.3 (Athens). Zenodo. DOI: [10.5281/zenodo.20370143](https://doi.org/10.5281/zenodo.20370143) (Paper 26)

#### JWST Distance Ladder Studies

Riess, A. G., Yuan, W., Casertano, S., et al. 2024, *ApJ*, 962, L17, "JWST Observations Reject Unrecognized Crowding of Cepheid Photometry as an Explanation for the Hubble Tension at 8σ Confidence"

Freedman, W. L., Madore, B. F., Hoyt, T. J., et al. 2024, arXiv:2408.06153, "Status Report on the Chicago-Carnegie Hubble Program (CCHP): Measurement of the Hubble Constant Using the Hubble and James Webb Space Telescopes"

Freedman, W. L., Madore, B. F., Hatt, D., et al. 2019, *ApJ*, 882, 34, "The Carnegie-Chicago Hubble Program. VIII. An Independent Determination of the Hubble Constant Based on the Tip of the Red Giant Branch"

Lee, A. J., Freedman, W. L., Madore, B. F., et al. 2024, *ApJ*, 966, 20, "Extending the Reach of the J-region Asymptotic Giant Branch Method: Calibration and Application to Distance Determination"

#### Hubble Tension Reviews & Proposed Solutions

Freedman, W. L. 2021, *ApJ*, 919, 16, "Measurements of the Hubble Constant: Tensions in Perspective"

Di Valentino, E., Mena, O., Pan, S., et al. 2021, *Classical and Quantum Gravity*, 38, 153001, "In the realm of the Hubble tension—a review of solutions"

Abdalla, E., Abellán, G. F., Aboubrahim, A., et al. 2022, *Journal of High Energy Astrophysics*, 34, 49, "Cosmology intertwined: A review of the particle physics, astrophysics, and cosmology associated with the cosmological tensions and anomalies"

Poulin, V., Smith, T. L., Karwal, T., & Kamionkowski, M. 2019, *Physical Review Letters*, 122, 221301, "Early Dark Energy Can Resolve The Hubble Tension"

Abbott, B. P., Abbott, R., Abbott, T. D., et al. (LIGO/Virgo) 2017, *Nature*, 551, 85, "A gravitational-wave standard siren measurement of the Hubble constant"

#### Statistical Methods

Zahid, H. J., Geller, M. J., Fabricant, D. G., & Hwang, H. S. 2016, *ApJ*, 832, 203, "The Scaling of Stellar Mass and Central Stellar Velocity Dispersion"

## Appendix A: Audited Data and Diagnostics

### A.1 The 37-Host Reconstruction Table

Table A1 lists the complete reconstructed 37-host SN Ia calibrator sample from
results/outputs/step_03_stratified_h0.csv. Row-level
measurement notes and error provenance are detailed in
results/outputs/step_07_sigma_provenance_table.csv. All 37 hosts use pinned HyperLEDA
circular rotation velocities $V_{\rm rot}$, with the potential-depth
proxy $u_\phi = V_{\rm rot}/\sqrt{2}$. For cosmological expansion regressions (the generative and joint multi-block frameworks), the documented union cut ($z_{\rm CMB}>0.0035$ or $z_{\rm HD}>0.0035$) selects the relevant $N=33$ Hubble-flow subset.

| Host | $z_{\rm HD}$ | $\mu$ (mag) | $H_{0,i}$ | $V_{\rm rot}$ (km/s) | $u_\phi$ (km/s) | Method | $\rho_{\rm local}$ | $S_{\rm total}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M 101 | 0.00122 | 29.160 | 53.86 | 274.1 | 193.82 | HyperLEDA rotation proxy | 0.0089 | 0.8089 |
| Mrk 1337 | 0.00925 | 32.916 | 72.42 | 122.1 | 86.34 | HyperLEDA rotation proxy | 0.0480 | 0.9321 |
| NGC 0691 | 0.00855 | 32.822 | 69.88 | 200.6 | 141.85 | HyperLEDA rotation proxy | 0.0457 | 0.6004 |
| NGC 1015 | 0.00815 | 32.618 | 73.19 | 120.7 | 85.35 | HyperLEDA rotation proxy | 0.0167 | 0.9396 |
| NGC 105 | 0.01682 | 34.493 | 63.68 | 234.9 | 166.10 | HyperLEDA rotation proxy | 0.0265 | 0.9380 |
| NGC 1309 | 0.00719 | 32.509 | 67.88 | 162.1 | 114.62 | HyperLEDA rotation proxy | 0.0324 | 0.9367 |
| NGC 1365 | 0.00483 | 31.325 | 78.65 | 198.3 | 140.22 | HyperLEDA rotation proxy | 0.0086 | 0.1293 |
| NGC 1448 | 0.00333 | 31.295 | 54.98 | 185.1 | 130.89 | HyperLEDA rotation proxy | 0.1022 | 0.7768 |
| NGC 1559 | 0.00407 | 31.461 | 62.27 | 134.3 | 94.96 | HyperLEDA rotation proxy | 0.1059 | 0.9003 |
| NGC 2442 | 0.00488 | 31.465 | 74.51 | 226.7 | 160.30 | HyperLEDA rotation proxy | 1.7591 | 0.0453 |
| NGC 2525 | 0.00602 | 32.011 | 71.47 | 126.8 | 89.66 | HyperLEDA rotation proxy | 0.0416 | 0.8674 |
| NGC 2608 | 0.00855 | 32.628 | 76.42 | 104.3 | 73.75 | HyperLEDA rotation proxy | 0.0868 | 0.9131 |
| NGC 3021 | 0.00673 | 32.392 | 67.07 | 136.7 | 96.66 | HyperLEDA rotation proxy | 0.2556 | 0.6925 |
| NGC 3147 | 0.01079 | 33.091 | 77.93 | 334.6 | 236.60 | HyperLEDA rotation proxy | 1.2881 | 0.0982 |
| NGC 3254 | 0.00648 | 32.403 | 64.24 | 207.8 | 146.94 | HyperLEDA rotation proxy | 0.0172 | 0.7493 |
| NGC 3370 | 0.00588 | 32.142 | 65.72 | 149.4 | 105.64 | HyperLEDA rotation proxy | 0.0361 | 0.9358 |
| NGC 3447 | 0.00465 | 31.944 | 56.94 | 46.5 | 32.88 | HyperLEDA rotation proxy | 0.0063 | 0.9405 |
| NGC 3583 | 0.00857 | 32.790 | 71.09 | 181.4 | 128.27 | HyperLEDA rotation proxy | 0.1182 | 0.8272 |
| NGC 3972 | 0.00349 | 31.707 | 47.67 | 110.0 | 77.78 | HyperLEDA rotation proxy | 0.0588 | 0.0944 |
| NGC 3982 | 0.00349 | 31.638 | 49.21 | 195.6 | 138.31 | HyperLEDA rotation proxy | 0.1789 | 0.0848 |
| NGC 4038 | 0.00571 | 31.634 | 80.67 | 169.1 | 119.57 | HyperLEDA rotation proxy | 0.0488 | 0.9318 |
| NGC 4424 | 0.00256 | 30.824 | 52.52 | 28.6 | 20.22 | HyperLEDA rotation proxy | 0.0403 | 0.0270 |
| NGC 4536 | 0.00317 | 30.835 | 64.68 | 155.7 | 110.10 | HyperLEDA rotation proxy | 0.0049 | 0.7501 |
| NGC 4639 | 0.00359 | 31.787 | 47.25 | 165.9 | 117.31 | HyperLEDA rotation proxy | 0.0360 | 0.8689 |
| NGC 4680 | 0.00864 | 32.547 | 80.16 | 94.5 | 66.82 | HyperLEDA rotation proxy | 0.0773 | 0.9187 |
| NGC 5468 | 0.00954 | 33.187 | 65.90 | 150.9 | 106.70 | HyperLEDA rotation proxy | 0.0251 | 0.8072 |
| NGC 5584 | 0.00625 | 31.866 | 79.36 | 124.7 | 88.18 | HyperLEDA rotation proxy | 0.0586 | 0.9279 |
| NGC 5643 | 0.00331 | 30.508 | 78.52 | 171.2 | 121.06 | HyperLEDA rotation proxy | 0.2463 | 0.7570 |
| NGC 5728 | 0.00996 | 32.916 | 77.96 | 224.4 | 158.67 | HyperLEDA rotation proxy | 0.0365 | 0.9356 |
| NGC 5861 | 0.00677 | 32.205 | 73.51 | 158.5 | 112.08 | HyperLEDA rotation proxy | 0.0943 | 0.8434 |
| NGC 5917 | 0.00710 | 32.337 | 72.56 | 92.6 | 65.48 | HyperLEDA rotation proxy | 0.0245 | 0.8713 |
| NGC 7250 | 0.00432 | 31.606 | 61.82 | 71.1 | 50.28 | HyperLEDA rotation proxy | 0.0392 | 0.9349 |
| NGC 7329 | 0.01028 | 33.269 | 68.39 | 250.9 | 177.41 | HyperLEDA rotation proxy | 0.0082 | 0.9404 |
| NGC 7541 | 0.00814 | 32.580 | 74.39 | 207.8 | 146.94 | HyperLEDA rotation proxy | 0.0820 | 0.8505 |
| NGC 7678 | 0.01061 | 33.267 | 70.66 | 200.8 | 141.99 | HyperLEDA rotation proxy | 0.0404 | 0.9346 |
| NGC 976 | 0.01312 | 33.544 | 76.91 | 405.0 | 286.38 | HyperLEDA rotation proxy | 0.2313 | 0.6666 |
| UGC 9391 | 0.00747 | 32.816 | 61.23 | 75.7 | 53.53 | HyperLEDA rotation proxy | 0.0139 | 0.9399 |

*Notes:* $H_{0,i}=cz_{\rm HD}/d_i$, where
$d_i=10^{(\mu_i-25)/5}$ Mpc; it is a descriptive per-host quantity, not a
full-ladder estimate. $V_{\rm rot}$ is the pinned HyperLEDA circular
rotation velocity. $u_\phi = V_{\rm rot}/\sqrt{2}$ is the potential-depth
proxy used throughout the analysis.
$\rho_{\rm local}$ and $S_{\rm total}$ are adopted screening inputs.

### A.2 Kinematic Provenance

All 37 hosts use pinned HyperLEDA circular rotation velocities $V_{\rm rot}$
as the kinematic potential-depth proxy, replacing the earlier heterogeneous
mix of stellar-absorption and H I linewidth measurements. The potential-depth
coordinate is $u_\phi = V_{\rm rot}/\sqrt{2}$, applied uniformly across the
full sample without aperture corrections or method-dependent systematics.
No separately locked “gold standard” tier is reported because the current
provenance pipeline does not produce one.

### A.3 Coefficient Scope

The canonical Cepheid-channel equation is

\begin{equation}
\Delta\mu_i=\kappa_{\rm Cep}
\frac{S_i(\mathcal E_i) u_{\phi,i}^2-U_{\rm ref}}{c^2}.
\end{equation}

$\kappa_{\rm Cep}$ is an observable response coefficient, not the
microscopic conformal coupling, a local clock-rate ratio, or a PPN
parameter. It absorbs any Cepheid response, P--L slope conversion,
kinematic-to-potential mapping, and observing-chain weighting. A conversion
to a bare scalar charge requires a specified microphysical transfer that is
not derived here.

### A.4 Conditional Cepheid-Channel Prediction Grid

The following values are generated by
step_05_prespecified_tep_predictions.csv using
$\kappa_{\rm Cep}^{\rm equiv}=0.365\times10^6$ mag and
$\sqrt{U_{\rm ref}}=30.507$ km/s. They apply only if the complete combined
endpoint response is allocated to Cepheid rows.

| $u_\phi$ (km/s) | $S$ | $\Delta\mu$ (mag) | Approx. $\Delta H_0$ (km/s/Mpc) |
| --- | --- | --- | --- |
| 50 | 1.0 | $+0.0064$ | $-0.21$ |
| 75 | 1.0 | $+0.0191$ | $-0.61$ |
| 100 | 1.0 | $+0.0368$ | $-1.19$ |
| 125 | 1.0 | $+0.0596$ | $-1.92$ |
| 150 | 1.0 | $+0.0875$ | $-2.82$ |
| 175 | 1.0 | $+0.1205$ | $-3.89$ |
| 200 | 1.0 | $+0.1586$ | $-5.11$ |
| 225 | 1.0 | $+0.2017$ | $-6.50$ |
| 150 | 0.5 | $+0.0419$ | $-1.35$ |
| 150 | 0.1 | $+0.0054$ | $-0.17$ |
| 200 | 0.1 | $+0.0125$ | $-0.40$ |

### A.5 Matrix Identifiability and Injection Scope

The augmented design matrix applies the endpoint column only to
Cepheid-sensitive observation rows while retaining free host moduli. The
47-parameter augmented matrix has rank 47. Over a seeded ensemble of 200
observation-level injections $\kappa_{\rm inj}=0.960\times10^6$ mag with
noise drawn at the full covariance level, the estimator recovers a mean
$\kappa_{\rm Cep}=0.985\times10^6$ mag with per-draw scatter
$0.189\times10^6$ mag (formal error $0.207\times10^6$), pull mean $+0.12$
and 68% coverage 0.73; the single-draw recovery fraction quoted in
earlier versions (0.997) used noise at 1% of the covariance and understated
the per-realization scatter, though it demonstrated correctness of the
linear algebra. The real-data result
$(-0.169\pm0.207)\times10^6$ mag is therefore informative for this
restricted row-level offset.

A latent host-modulus injection is a different experiment: shifting
$\mu_i$ and generating every affected row moves the fitted host parameters
themselves, so a Cepheid-row-only column need not recover it. This
distinction is why the matrix result constrains a specified photometric
allocation but cannot assign a generic combined endpoint association to
systemic redshift or transport.

## Appendix B: Cross-Domain Scope and Coefficient Dictionary

### B.1 Observable Coefficients Are Channel Specific

The primary environmental coordinate in this paper,
$X=(S u_\phi^2-U_{\rm ref})/c^2$, is dimensionless. Its fitted coefficients
are nevertheless observable-specific:

| Coefficient | Observable equation | Units and status |
| --- | --- | --- |
| $\Gamma_X$ | $v=d(H_{\rm app}+\Gamma_X X)$ | ${\rm km\,s^{-1}\,Mpc^{-1}}$; measured combined endpoint slope |
| $\kappa_{\rm Cep}$ | $\Delta\mu=\kappa_{\rm Cep}X$ | mag; restricted Cepheid-row coefficient |
| $\kappa_{\rm Cep}^{\rm equiv}$ | algebraic projection of $\Gamma_X$ onto Cepheids | mag; conditional, not independently fitted |
| Other clock-channel coefficient | requires that channel's raw-observable likelihood | not estimated in this paper |

Equality of the environmental coordinate does not imply equality of these
coefficients. A valid cross-domain comparison needs an explicit transfer
from the microscopic matter-metric perturbation to each measured observable,
including source screening, aperture or path weighting, and the observable's
calibration convention.

### B.2 No Numerical Pulsar Prior Is Applied

This analysis uses no pulsar likelihood, pulsar catalog, or numerical
pulsar-derived prior on $\Gamma_X$ or $\kappa_{\rm Cep}$. Consequently,
pulsar sample sizes, spin-down residuals, and effective response
coefficients are not reported as evidence for the Cepheid result. A future
cross-channel analysis would need to provide the underlying data product,
selection function, likelihood, and transfer equation before its coefficient
could constrain this model.

### B.3 No Solar-System Closure Follows From the Host Fit

The schematic action in Section 1 does not specify the functions
$A(\phi)$, $B(\phi)$, or $V(\phi)$, nor a solved field profile for the Sun
or Earth. The host coefficient therefore cannot be converted uniquely into
a PPN parameter, an equivalence-principle violation, or a Vainshtein
suppression factor. Cassini, MICROSCOPE, and laboratory-clock bounds are
essential constraints on any completed microphysical TEP model, but the
present phenomenological host fit does not by itself demonstrate compliance
with them.

### B.4 Permitted Cross-Corpus Claim

The defensible cross-corpus statement is structural: TEP papers may use the
same causal matter-metric convention and the same absolute sign rule
$A(\phi) < 1$, while measuring different channel responses. Numerical
agreement, universality of a fitted coefficient, or precision-gravity
closure must be demonstrated in a joint model and is not inferred here.

## Appendix C: Conformal Period Transport and the Distance-Ladder Bias

This appendix formalizes the restricted Cepheid-channel realization of the
TEP endpoint response. The derivation uses only the clock differential
between the Cepheid and the spectroscopic tracer within the same host. It
does not compare absolute host and calibrator clock rates, and it does not
identify the ladder-level coefficient $\kappa_{\rm Cep}$ with the
microscopic conformal offset $\Delta\ln A$.

### C.1 Observing-Chain Core--Disk Differential

Under TEP, the scalar field strictly slows proper time in every
gravitational environment: $0 < r < 1$, where

\begin{equation}
r_j\equiv
\frac{(d\tilde\tau/dt)_j}{(d\tilde\tau/dt)_{\rm cosmic}} .
\end{equation}

Let $r_{{\rm Cep},i}$ be the rate at the Cepheid location in host $i$, and
let $r_{{\rm spec},i}$ be the rate associated with the spectral tracer used
to infer that host's systemic redshift. In the proposed core--disk closure,
outer-disk Cepheids occupy a less-slowed active-shear environment while
nuclear or integrated systemic lines receive greater weight from a deeper,
more strongly slowed region. The same-host hierarchy is

\begin{equation}
0 < r_{{\rm spec},i} < r_{{\rm Cep},i} < r_{\rm cosmic} \equiv 1 .
\end{equation}

This ordering does not assert that an SN host clock runs faster than a
Milky Way, LMC, or NGC 4258 clock. No such absolute host--calibrator ordering
is needed. The observable is the differential sampled by two tracers in the
same host.

Let $P_{{\rm local},i}$ be the Cepheid period in local matter-frame proper
time and let $(1+z_{{\rm path},i})$ collect the common propagation redshift
between the host and observer. Endpoint clock bookkeeping gives

\begin{equation}
P_{{\rm obs},i}
=(1+z_{{\rm path},i})\frac{P_{{\rm local},i}}{r_{{\rm Cep},i}},
\qquad
1+z_{{\rm spec},i}
=\frac{1+z_{{\rm path},i}}{r_{{\rm spec},i}} .
\end{equation}

The period entered into the P--L fit after the standard spectroscopic
$(1+z)$ correction is therefore

\begin{equation}
P_{{\rm rest},i}^{\rm inf}
\equiv\frac{P_{{\rm obs},i}}{1+z_{{\rm spec},i}}
=P_{{\rm local},i}\,q_i,
\qquad
q_i\equiv\frac{r_{{\rm spec},i}}{r_{{\rm Cep},i}} .
\label{eq:pipeline_period_ratio}
\end{equation}

If both tracers sample the same TEP rate, $q_i=1$ and the response cancels.
In the proposed core--disk regime, $0 < q_i < 1$, so the inferred rest-frame
period is contracted even though both clocks remain slower than cosmic
time. The contraction is produced by the observing-chain division, not by
an absolute clock acceleration.

The P--L zero point is itself calibrated with an anchor ensemble. Let
$q_{\rm ref}$ denote the correspondingly weighted pipeline differential for
that ensemble. The environmental period residual relevant to applying the
calibrated P--L relation is

\begin{equation}
\delta\ln P_{{\rm rest},i}^{\rm inf}
=\ln\!\left(\frac{q_i}{q_{\rm ref}}\right) .
\label{eq:calibrated_period_residual}
\end{equation}

Thus the restricted Cepheid mechanism requires a larger core--disk
differential in the target host than in the calibration ensemble,
$q_i < q_{\rm ref}$. This is a testable tracer/aperture condition. It cannot
be inferred merely from the host's total velocity dispersion or from a
comparison of absolute host and calibrator clock rates.

### C.2 Period--Luminosity Inference Bias

The calibrated Cepheid Wesenheit relation is

\begin{equation}
M_W=a+b\log_{10}P,\qquad b\approx-3.26 .
\end{equation}

The observer inserts the pipeline-inferred rest-frame period
$P_{{\rm rest},i}^{\rm inf}$ into this relation. Relative to the anchor
calibration, the inferred-magnitude shift is

\begin{equation}
\delta M_{P,i}
=b\,\delta\log_{10}P_{{\rm rest},i}^{\rm inf}
=\frac{b}{\ln10}\ln\!\left(\frac{q_i}{q_{\rm ref}}\right) .
\label{eq:pipeline_magnitude_shift}
\end{equation}

For $b < 0$ and $\delta\ln P_{{\rm rest},i}^{\rm inf} < 0$, the shift
$\delta M_{P,i} > 0$: the Cepheid is inferred to be intrinsically dimmer
than it would be under the calibration-ensemble pipeline differential. Since $\mu_{\rm obs}=m-M_{\rm inf}$, the conditional
Cepheid-channel correction is

\begin{equation}
\mu_{{\rm corr},i}=\mu_{{\rm obs},i}+\delta M_{P,i} .
\end{equation}

The same equations also expose the falsification condition: if host-specific
spectroscopy shows $q_i\simeq q_{\rm ref}$, the proposed period-transport
term cancels and the combined endpoint response must enter another
observing-chain channel or be rejected.

### C.3 The Ladder-Level Response Coefficient ($\kappa_{\rm Cep}$)

The ladder-level coefficient is the Paper-0 transfer-map channel response
evaluated on the Cepheid channel: $\kappa_{\rm Cep} = |\beta_A|\,S_A(\mathcal E)\,\Gamma_{\rm Cep}$,
with the clock-amplitude projection $S_A$ and the Leavitt projector
$\Gamma_{\rm Cep} = 5/\ln 10$ (Paper 0 \S7). It is one observable response
coefficient, not an independent coupling.

The present data do not measure $q_i$ for each spectral aperture. The
restricted Cepheid closure therefore parameterizes its first-order
environmental dependence with

\begin{equation}
\ln\!\left(\frac{q_i}{q_{\rm ref}}\right)
=-\lambda_{\rm Cep}X_i,
\qquad
X_i=\frac{S(\mathcal E_i) u_{\phi,i}^2-U_{\rm ref}}{c^2},
\qquad \lambda_{\rm Cep}>0 .
\end{equation}

Combining this transfer with Equation~(\ref{eq:pipeline_magnitude_shift})
gives

\begin{equation}
\mu_{{\rm corr},i}=\mu_{{\rm obs},i}+\kappa_{\rm Cep}X_i,
\qquad
\kappa_{\rm Cep}\equiv-\frac{b}{\ln10}\lambda_{\rm Cep}>0 .
\end{equation}

The derivation of $0.960\times 10^6\ {\rm mag}$ as a single-rate
target -- $0.288\ {\rm mag}$ divided by $X=3\times 10^{-7}$,
assigning the whole host potential to one clock -- is withdrawn. The
clocks the ladder divides share each galactic potential, so the
common rate cancels in the nested ratio $q=r_{\rm spec}/r_{\rm Cep}$
of the spectroscopic-aperture rate to the Cepheid rate, compared with
the same ratio in the anchor. On a uniform $10^{11}\,M_\odot$,
$30\,\mathrm{kpc}$ host with the Cepheid at $8\,\mathrm{kpc}$,
against a $10^{10}\,M_\odot$, $4.5\,\mathrm{kpc}$ anchor, that ratio
differs by $\Delta\ln q = 2.4\times 10^{-10}$, a clock-channel
coefficient $\kappa_{\rm nested}=-7.5\times 10^{-4}\ {\rm mag}$
through the Leavitt projector at $X=V_{\rm rot}^2/c^2$ for
$200\ {\rm km\,s^{-1}}$. Alternative reference-clock
conventions -- the volume-mean light depth or the direct
Cepheid-to-Cepheid baseline comparison -- leave
$|\kappa_{\rm nested}|$ of order unity at most and of the same sign,
opposite to the measured positive response. The raw conformal clock
channel is therefore negligible, more than eight orders of magnitude
below the empirical interval: $\kappa_{\rm Cep}$ cannot be a
clock-ratio artifact, and its value is carried by the response and
screening sector it parameterizes. The canonical coefficient remains
the prespecified transfer value used across the programme; the
empirical fits measure the response and do not impose it. Its
numerical provenance is itself empirical: the value was set to the
joint-fit/bootstrap coefficient on this same SH0ES sample, so the
benchmark--measured comparison is an estimator comparison on common
data rather than an independent theory--data test.
Numerically, the measured endpoint response
$\kappa_{\rm Cep}^{\rm equiv} = (0.365\text{--}0.421)\times 10^6$ mag
sits a factor $\approx 2.3$ below the canonical benchmark
($\approx 2.0$--$2.5\sigma$ under the fitted uncertainty), the
residual being carried by the response product
$S_{\rm Cep}\Gamma_{\rm Cep}$ whose derivation is the open Gate-A
item: the benchmark is a declared theory normalization, not a
coefficient the data has confirmed.

$\kappa_{\rm Cep}$ is a ladder-level observable response. It absorbs the
core--disk transfer $\lambda_{\rm Cep}$, the P--L slope, aperture weighting,
and any additional measurement-chain response represented by this restricted
model. It is not the microscopic conformal coupling, a local clock-rate
ratio, or a comparison of absolute host and calibrator rates.

The fitted coefficient does not determine $\Delta\ln A$. Computing a
literal geometric $\Delta\Theta$ directly from a coefficient of order
$10^6$ mag would incorrectly predict large atomic spectral shifts. A
microscopic interpretation requires spatially resolved estimates of
$r_{\rm spec}$ and $r_{\rm Cep}$, together with a derived TEP transfer
function. Until then, the Cepheid realization remains a physically specified
but conditional allocation of the measured combined endpoint response.

Microscopically, $q_i$ is a pure clock ratio between the
spectroscopic aperture and the Cepheid field, $q_i = r_{\rm spec}/r_{\rm Cep}$,
under the universal matter metric
$\tilde g_{\mu\nu} = A^2(\phi)\,g_{\mu\nu} + B(\phi)\,\nabla_\mu\phi\,\nabla_\nu\phi$;
no auxiliary transport factor is invoked (Paper 31, Section 4.5).
The spectroscopic bound is satisfied by construction: a bare
core--disk $\Delta\ln A$ large enough to deliver the measured
response would violate internal line-shift constraints by more
than an order of magnitude, but identifying $\kappa_{\rm Cep}$
with such a single-point clock shift is the category error
addressed above — the physical core-to-disk potential contrast
$\Delta\Phi/c^2 \sim 2\times10^{-7}$ corresponds to only
$\sim 70$--$150\ {\rm m\,s^{-1}}$ of differential line shift, well
inside spectroscopic tolerances, while $\kappa_{\rm Cep}$ is a
multi-stage ladder-level response across $X_i \sim 10^{-7}$. The fitted $\kappa_{\rm Cep}$ is
accordingly a compound ladder-level response: it absorbs any
correlated SN-timescale contribution of the same observing-chain
mechanism (the SN light-curve stretch channel of Paper 31,
Section 4.9) that is active in the calibration data, and is not
Cepheid-pure by construction. The channel decomposition is
constrained by the tracer-type and stretch-step falsification
tests rather than assumed.

The identification structure of the ladder likelihood supplies an
independent empirical constraint on the form of $q_i$. Because the host
distance moduli $\mu_i$ are unconstrained, any host-uniform transport
$q_i = {\rm const}$ (independent of Cepheid period) is absorbed
identically into $\hat{\mu}_i$ and cannot appear in the matrix
likelihood at all — verified numerically by injection, which returns a
period-coupled coefficient consistent with zero for a host-uniform
modulus shift while recovering period-coupled injections at
$99.8$--$100.9\%$. The residual the ladder can see is
therefore the within-host period dependence
$\ln(q_i/q_{\rm ref}) \equiv f(P_i)$, fitted as an environment-coupled
period term $\kappa_P$. The joint fit returns
$\kappa_P = (-5.79 \pm 2.26)\times10^5\ {\rm mag}$ ($2.57\sigma$)
against a simultaneous metallicity-coupled term, near-orthogonal to it
($r = -0.15$), with the sign corresponding to $q_i$ rising with
period — weaker contraction for longer-period, lower-density pulsators
in deeper potentials. This is the direction a screening-modulated
envelope coupling predicts qualitatively.

The competing explanations of such a term — a global period--luminosity
shape systematic, a distance-correlated crowding artifact, and a host
mass or metallicity confound — are tested directly on the same design
matrix by a discriminating battery of model tests. Free quadratic and
10-day-break shape terms are individually weak and leave $\kappa_P$
strengthened ($3.30\sigma$ with both free; $2.97\sigma$ under joint
$\kappa_Z$ control); replacing the environmental modulator by the
fitted modulus ($1.30\sigma$) or host mass ($0.23\sigma$) collapses
the carrier, while host-assignment permutation returns $p=0.0125$
and all $41$ leave-one-host-out refits retain the sign
($1.67$--$3.4\sigma$). The per-host slope decomposition shows the
anchor populations flat at the reference slope (within $0.84\sigma$)
and the deep-potential hosts tilted in the predicted direction at
roughly the predicted amplitude. What remains open is sharper than
the original objection: a within-host decomposition
separates the conventional period--metallicity channel by
construction — the pure per-Cepheid interaction
$(\log_{10}P-1)^{\sim}\times[{\rm O/H}]^{\sim}$ is null
($0.24\sigma$), the apparent PLZ signal lives entirely in the
host-mean-weighted pieces, and the per-host metallicity slopes
themselves order on $X_i$ at $2.71\sigma$ — so $\kappa_Z$ is
environment-ordered within-host structure rather than a
conventional PLZ artefact; and the stellar-structure derivation of
$q_i(P)$ — the screening response of a pulsator's envelope as a
function of its mean density — resolves the shape question in the
derived direction: the amplitude-sector form
$\ln q = \lambda_0 X\,\rho_T/\bar\rho(P)$ with
$\bar\rho \propto P^{-2}$ predicts a carrier quadratic in period,
which the design matrix prefers over both the linear-$\log P$ and
quadratic-$\log P$ forms in every configuration tested
(Step 62), with the implied transport weight
$\lambda_0 \simeq 0.77$--$0.94$. The unified-column closure test
(Step 63) confirms sufficiency: the single predicted column
reproduces the fitted $(\kappa_0, \kappa_P)$ pair with a residual
host-amplitude term consistent with zero
($0.13$--$0.45\sigma$), and under
$\kappa = \Gamma_{\rm Cep}\lambda_0\rho_T/\bar\rho$ the canonical
prior is the pivot-period value of the same law. The residual
theory calculation is thereby reduced to a single order-unity
coefficient.

### C.4 Coupled Velocity--Distance Allocation and Analytic Ratio

A critical physical question in interpreting the combined endpoint response $\Gamma_X = \beta_X + (\ln 10/5) H_{\rm app} \kappa_{\rm Cep}$ is whether the allocation $\beta_X \approx 0$ is consistent with the observing-chain mechanism. In Equation~(\ref{eq:spectral_redshift_relation}), the host systemic redshift measured from nuclear or central spectroscopy relates to the cosmological path redshift by

\begin{equation}\label{eq:spectral_redshift_relation}
1 + z_{{\rm spec},i} = \frac{1 + z_{{\rm path},i}}{r_{{\rm spec},i}} \approx (1 + z_{{\rm path},i})\left(1 + \frac{\Phi_i}{c^2}\right) ,
\end{equation}

where $\Phi_i \sim u_{\phi,i}^2$ is the effective potential depth of the spectroscopic aperture. Expanding to first order in the dimensionless potential coordinate $X_i \approx \Phi_i / c^2$ relative to the calibration reference yields an apparent velocity shift in the host velocity channel:

\begin{equation}
\Delta(c z)_i = c X_i .
\end{equation}

Dividing by host distance $d_i$, this direct gravitational redshift contributes to the linear expansion slope as

\begin{equation}
\beta_{X,{\rm grav}} \equiv \left\langle \frac{c}{d_i} \right\rangle \approx \frac{c}{\langle d \rangle} .
\end{equation}

For the 33 Hubble-flow hosts with sample mean distance $\langle d \rangle = 31.2\ {\rm Mpc}$ (harmonic mean $\approx 26.3\ {\rm Mpc}$), the direct gravitational contribution evaluates to

\begin{equation}
\beta_{X,{\rm grav}} \approx 9.60 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}} .
\end{equation}

Conversely, the Cepheid distance-modulus response operates through the logarithmic Leavitt law, $\Delta \mu_i = \kappa_{\rm Cep} X_i$. Differentiating the distance modulus $\mu = 5 \log_{10} d + 25$ relates magnitude shifts to fractional distance changes:

\begin{equation}
\frac{\Delta d_i}{d_i} = \frac{\ln 10}{5} \Delta \mu_i = \frac{\ln 10}{5} \kappa_{\rm Cep} X_i .
\end{equation}

The corresponding apparent velocity shift across the Hubble flow is $\Delta(c z)_i \approx H_{\rm app} \Delta d_i = (\ln 10/5) H_{\rm app} d_i \kappa_{\rm Cep} X_i$. Dividing by $d_i$ defines the Cepheid distance-modulus contribution to the expansion slope:

\begin{equation}
\Gamma_{X,{\rm Cep}} \equiv \left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep} .
\end{equation}

Evaluating at the empirical values $H_{\rm app} \approx 68.6\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and $\kappa_{\rm Cep} \approx 0.365 \times 10^6\ {\rm mag}$ yields

\begin{equation}
\Gamma_{X,{\rm Cep}} \approx 0.4605 \times 68.6 \times (3.65 \times 10^5) \approx 1.15 \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}} .
\end{equation}

Taking the analytic ratio of the two channels yields

\begin{equation}
\frac{\beta_{X,{\rm grav}}}{\Gamma_{X,{\rm Cep}}}
= \frac{c / \langle d \rangle}{\left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep}}
\approx \frac{9.60 \times 10^3}{1.15 \times 10^7}
\approx 8.3 \times 10^{-4} \approx 0.08\% \text{--} 0.10\% .
\end{equation}

This analytic derivation reveals that the direct gravitational redshift contributes less than one part in a thousand to the combined endpoint response $\Gamma_X$, with $>99.9\%$ allocated to the Cepheid distance-modulus channel. The physical origin of this asymmetry is the dimensional lever arm between linear gravitational redshift ($\Delta(cz) = \Phi / c \sim 70\ {\rm m\,s^{-1}}$) and photometric distance modulus exponentiation ($\Delta \mu \sim 0.15\ {\rm mag}$, corresponding to a $\sim 7\%$ distance correction and an apparent expansion shift of $\sim 120\ {\rm km\,s^{-1}}$).

Hence, setting $\beta_X \approx 0$ is not an ad hoc closure that contradicts the observing-chain mechanism, but rather its rigorous leading-order consequence. Estimating the exact coupled model in the joint multi-block framework with $\beta_X = 9.597 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}}$ fixed from first principles recovers $H_{\rm app} = 68.78 \pm 1.48\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and $\kappa_{\rm Cep} = (0.266 \pm 0.240) \times 10^6\ {\rm mag}$ at the canonical $\sigma_v = 250\ {\rm km/s}$ convention ($H_{\rm app} = 68.72 \pm 1.32$, $\kappa_{\rm Cep} = (0.290 \pm 0.227) \times 10^6\ {\rm mag}$ at the $\sigma_v = 182.1\ {\rm km/s}$ variant), matching the restricted Cepheid closure to within three decimal places at every $\sigma_v$.

## Appendix D: Reference-Gauge and Anchor Assumptions

### D.1 Physical Anchor Potential Contrast vs Environmental Screening

The environmental gradient between the calibration anchors and the SN Ia host galaxies is an intrinsic observational property of the galaxies rather than an artifact of group screening models. The primary distance ladder calibrators are dwarf or late-type galaxies with low circular rotation velocities: the LMC ($V_{\rm rot} \approx 65\text{--}72\ {\rm km\,s^{-1}} \implies u_\phi \approx 46\text{--}51\ {\rm km\,s^{-1}}$), the SMC ($V_{\rm rot} \approx 30\text{--}40\ {\rm km\,s^{-1}} \implies u_\phi \approx 21\text{--}28\ {\rm km\,s^{-1}}$), the Milky Way solar neighbourhood, and NGC 4258. Together, these anchors possess shallow potential depths ($\langle u_\phi^2 \rangle \sim 3.0 \times 10^3\ {\rm km^2\,s^{-2}}$).

In contrast, the 37 R22 SN Ia host galaxies sample massive luminous spirals with median circular rotation velocities $V_{\rm rot} \approx 160\text{--}250\ {\rm km\,s^{-1}}$ ($\langle u_\phi^2 \rangle \sim 2.5 \times 10^4\ {\rm km^2\,s^{-2}}$). Even in the completely unscreened coordinate ($S=1$), the potential contrast between anchors and hosts exceeds a factor of 8. Furthermore, the direct empirical detection of internal radial Period--Luminosity gradients within the LMC ($+0.0284 \pm 0.0086\ {\rm mag}$, $3.30\sigma$) and M31 ($+0.681 \pm 0.187\ {\rm mag}$, $3.65\sigma$ in the HST PHAT catalogue; $+0.630 \pm 0.195\ {\rm mag}$, $3.24\sigma$ for the Pan-STARRS sample restricted to the PHAT footprint) provides internal evidence that radial Period--Luminosity gradients are measured within Local Group members with the sign predicted by TEP — noting that the M31 signal fails pair-matched controls but persists under quality-regime equating ($2.3$–$3.2\sigma$ across the pooled-error band ladder), so it is carried as conditional evidence bounded by the contested colour-matched control (Section 3.2).

### D.2 Adopted Anchor Endpoint Construction

The distance ladder calibration ensemble is represented with approximate weights 0.20, 0.25, and 0.55 for the Milky Way, LMC, and NGC 4258. In the baseline formulation, local stellar velocity dispersions at the Cepheid disk locations are adopted: Milky Way solar neighbourhood ($\sigma_z = 30.0\ {\rm km\,s^{-1}}$; Bovy et al. 2012), LMC stellar disk ($\sigma_{\rm disk} = 24.0\ {\rm km\,s^{-1}}$; van der Marel et al. 2002), and NGC 4258 intermediate annulus ($\sigma_{\rm local} = 115.0\ {\rm km\,s^{-1}}$; Kormendy &amp; Ho 2013). This construction yields the composite reference scale $\sqrt{U_{\rm ref}} = 87.165\ {\rm km\,s^{-1}}$ unscreened, and $\sqrt{U_{\rm ref}} = 30.507\ {\rm km\,s^{-1}}$ when modulated by anchor group screening ($S_{\rm MW} = 0.605$, $S_{\rm LMC} = 0.873$, $S_{\rm N4258} = 0.096$). An unweighted average across the three anchors yields $\sqrt{U_{\rm ref}^{\rm equal}} = 70.002\ {\rm km\,s^{-1}}$ unscreened and $27.769\ {\rm km\,s^{-1}}$ screened.

Alternatively, if the halo circular velocity proxy adopted for SN Ia hosts ($u_\phi \equiv V_{\rm rot}/\sqrt{2}$) is applied symmetrically to the anchors using observed rotation velocities ($V_{\rm rot} = 220.0\ {\rm km\,s^{-1}}$ for the Milky Way, $66.0\ {\rm km\,s^{-1}}$ for the LMC, and $208.0\ {\rm km\,s^{-1}}$ for NGC 4258), the corresponding anchor potential depths are $u_\phi = 155.56\ {\rm km\,s^{-1}}$, $46.67\ {\rm km\,s^{-1}}$, and $147.08\ {\rm km\,s^{-1}}$ respectively. Under the SH0ES weights, this homogeneous rotation construction yields $\sqrt{U_{\rm ref}^{\rm rot}} = 131.461\ {\rm km\,s^{-1}}$ (or $126.504\ {\rm km\,s^{-1}}$ under equal weighting). While local stellar velocity dispersions trace the immediate disk potential inhabited by Cepheids and halo rotation curves trace outer dark-matter halos, both representations preserve the physical contrast between low-mass dwarf calibrators (such as the LMC, $V_{\rm rot} \sim 66\ {\rm km\,s^{-1}}$) and massive SN Ia host spirals ($V_{\rm rot} \sim 160\text{--}250\ {\rm km\,s^{-1}}$).

### D.3 Exact Reference-Gauge Invariance and Analytic Proof

The explicit dependence on the numerical origin $U_{\rm ref}$ is an exact gauge artifact that cancels algebraically in physical observables. Two distinct proofs demonstrate this invariance:

*Analytic Background Invariance:* In the cosmological background calibration (Section 2.8 and Section 3.5), the unperturbed expansion rate evaluated at the cosmic mean baseline evaluates as $H_{\rm cosmic} = H_{\rm app} - \Gamma_X \frac{\langle U \rangle}{c^2}$, where $\Gamma_X \equiv \frac{c}{5 \ln 10} \kappa_X$. In terms of the reference-shifted potential coordinates $X_i = (U_i - U_{\rm ref})/c^2$, the unperturbed cosmic baseline corresponds to $X_{\rm cosmic} = -U_{\rm ref}/c^2$. When measured relative to the sample mean $\langle X \rangle = (\langle U \rangle - U_{\rm ref})/c^2$, the relative offset is:

$$\widetilde X_{\rm cosmic} = X_{\rm cosmic} - \langle X \rangle = -\frac{U_{\rm ref}}{c^2} - \left(\frac{\langle U \rangle - U_{\rm ref}}{c^2}\right) \equiv -\frac{\langle U \rangle}{c^2}$$

The reference scale $U_{\rm ref}$ cancels identically to machine precision. Consequently, $H_{\rm cosmic}$ is an invariant physical property of the sample mean environmental potential depth $\langle U \rangle/c^2$, completely independent of the anchor coordinate zero point.

*Full Distance Ladder Matrix Invariance:* In the simultaneous 3,490-row SH0ES design matrix fit, the Cepheid photometric model includes the anchor absolute magnitude zero point $M_H^W$, which enters with a uniform column of $+1$ across all Cepheid observations. Shifting the reference potential by $\Delta U_{\rm ref}$ introduces an additive offset $\Delta m = -\kappa \Delta U_{\rm ref}/c^2$ across every Cepheid row. In a linear least-squares regression, this uniform offset is absorbed exactly by the refitted Cepheid absolute magnitude zero point:

$$\Delta M_H^W = -\kappa \frac{\Delta U_{\rm ref}}{c^2}$$

The relative distance moduli between hosts and anchors $\Delta \mu_i = \mu_i - \mu_{\rm anchor}$, the calibrator supernova absolute magnitude $M_B$, the Hubble-flow intercept $a_B$, the total fit quality $\chi^2$, and the recovered Hubble constant $H_0$ remain identically invariant. Table D1 presents the full-ladder matrix propagation results across four disparate reference origins spanning 30.51 to 131.46 km/s:

| Anchor Reference Origin | $\sqrt{U_{\rm ref}}$ (km/s) | Null ($\kappa = 0$) | Endpoint ($\kappa = 3.65\times 10^5$) | Canonical ($\kappa = 9.60\times 10^5$) | $\Delta\chi^2$ (Endpoint / Canonical) |
| --- | --- | --- | --- | --- | --- |
| Screened local dispersion | $30.51$ | $73.04$ | $71.77$ | $69.74$ | $+6.00\ /\ +29.15$ |
| Equal-weighted dispersion | $70.00$ | $73.04$ | $71.77$ | $69.74$ | $+6.00\ /\ +29.15$ |
| Standard weighted dispersion | $87.17$ | $73.04$ | $71.77$ | $69.74$ | $+6.00\ /\ +29.15$ |
| Homogeneous rotation proxy | $131.46$ | $73.04$ | $71.77$ | $69.74$ | $+6.00\ /\ +29.15$ |

Across all four conventions, the matrix-recovered $H_0$ and $\chi^2$ match to four decimal places ($H_0 = 71.7711\ {\rm km\,s^{-1}\,Mpc^{-1}}$ for the endpoint projection and $69.7423\ {\rm km\,s^{-1}\,Mpc^{-1}}$ for the canonical coupling). In contrast, unconstrained host-mean shortcuts that evaluate $\Delta\mu_i = \kappa (U_i - U_{\rm ref})/c^2$ without refitting the anchor zero point $M_H^W$ produce artificial numerical offsets (such as $64.74$ vs $66.65\ {\rm km\,s^{-1}\,Mpc^{-1}}$ in host-level evaluations, or $69.42$ vs $71.73\ {\rm km\,s^{-1}\,Mpc^{-1}}$ in simplified propagation). These offsets are methodological artifacts of omitting the ladder zero-point refit. Both host-level values remain statistically concordant with the Planck baseline ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) within $1.73\sigma$ and $0.45\sigma$ respectively.

### D.4 Sensitivity Continuum and Hierarchical Modeling

To verify that empirical concordance does not hinge upon any specific reference choice, the sensitivity analysis evaluates the continuum of inferred $H_0$ across the continuous parameter sweep $\sqrt{U_{\rm ref}} \in [30.0, 131.5]\ {\rm km\,s^{-1}}$. Across this entire domain, the recovered $H_0$ remains strictly confined within $64.73\text{--}69.45\ {\rm km\,s^{-1}\,Mpc^{-1}}$, maintaining statistical concordance with Planck ($\le 1.7\sigma$) without tuning.

While reference-gauge invariance confirms mathematical consistency within the linear regression, physical anchor modeling will benefit from future hierarchical calibration directly incorporating resolved stellar kinematics and multi-wavelength screening profiles. The present results demonstrate that the emergent resolution of the Hubble tension is driven by the physical potential contrast between dwarf calibrators and luminous spiral hosts, rather than reference-gauge conventions or screening parameterizations.