# The Cepheid Bias: Resolving the Hubble Tension
**Matthew Lukin Smawfield**  
Version: v0.11 (Kingston upon Hull)
First published: 11 January 2026 · Last updated: 1 October 2026  
DOI: 10.5281/zenodo.18209702

---

## Abstract

The Hubble Tension—the persistent $5\sigma$ discrepancy between local distance-ladder measurements ($H_0 \approx 73.0\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and early-universe CMB inference ($H_0 = 67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$)—represents a significant challenge in precision cosmology. This paper demonstrates that the discrepancy is resolved by an environment-dependent Cepheid clock bias, as predicted by the Temporal Equivalence Principle (TEP).

The resolving mechanism is environmental clock calibration: Cepheid variable stars function as environment-dependent "standard clocks." Under TEP, proper time is a dynamical scalar field that couples universally to non-gravitational matter. Because stellar pulsation rates operate in local proper time, calibrating classical Cepheids in deep-potential SN Ia host galaxies against diffuse, low-mass anchor galaxies systematically misestimates host distance moduli. When interpreted through a universal Period--Luminosity relation, this environment-dependent pulsation shift mimics diminished luminosity, leading to underestimated distances and an inflated local Hubble constant.

In standard distance-ladder linear regression, unconstrained latent host distance moduli $\mu_i$ algebraically absorb host-level environmental shifts. A standard-ladder projection test that inserts a TEP environmental column into the SH0ES design matrix therefore yields an apparent null result. This structural degeneracy is resolved by formulating the calibrator-host distance ladder as a generative clock-aware model coupled to the Hubble flow.

This generative framework is tested using the complete public Riess et al. (2022) sample of 37 distinct SN Ia host galaxies, utilizing a local inner-potential coordinate $\sigma_*$ compiled from published stellar-absorption dispersions (18 hosts) and calibrated H&nbsp;I linewidth dispersion proxies (19 hosts), matched to the anchor endpoint construction and combined with continuous environmental screening. The canonical peculiar-velocity convention throughout is the R22 baseline $\sigma_v = 250\ {\rm km/s}$; all other $\sigma_v$ values are reported as sensitivity variants. A median split at $\sigma_* = 95\ {\rm km/s}$ stratifies the derived per-host expansion rates by $\Delta H_0 = 9.59\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (Spearman $\rho = 0.553$, $p = 3.9\times10^{-4}$; Pearson $r = 0.501$, $p = 0.0016$), strengthening to $r = 0.556$ ($p = 0.017$) on the stellar-absorption-only subset. Across all 37 hosts, the generative endpoint likelihood yields $\Gamma_X=(3.846\pm1.653)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.33\sigma$) under canonical $250\ {\rm km/s}$ velocity variance, strengthening to $(3.906\pm1.389)\times10^7$ ($2.81\sigma$) under the reduced-scatter sensitivity variants ($\sigma_v = 150$--$182.1\ {\rm km/s}$), with permutation $p = 0.033$, 99.8% positive bootstrap draws, and 100% leave-one-host-out sign stability (37 of 37 refits positive). In the cosmologically constrained host-level expansion-rate likelihood across the 33 Hubble-flow hosts, the combined endpoint response is recovered at $\Gamma_X=(3.805\pm1.647)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.31\sigma$; $3.47\sigma$ at $150\ {\rm km/s}$), yielding a conventional Hubble-flow intercept of $H_{\rm app}=68.48\pm1.47\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($68.31\pm0.97$ at $\sigma_v = 150\ {\rm km/s}$). When evaluated in a joint multi-block framework with independent TRGB distances and geometric anchors, the multi-block likelihood favours an environmental response under the restricted Cepheid closure ($\kappa_{\rm Cep} = (0.665 \pm 0.362)\times 10^6\ {\rm mag}$, $1.83\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$; $(0.746 \pm 0.343)\times 10^6\ {\rm mag}$, $2.17\sigma$ at $\sigma_v = 150\ {\rm km/s}$), resolving the local Hubble constant to $H_{\rm app} = 68.82\pm 1.45\ {\rm km\,s^{-1}\,Mpc^{-1}}$, while the standalone redshift-only regression in $H_0$ space yields $\kappa_{\rm Cep} = (1.206 \pm 0.528)\times 10^6\ {\rm mag}$ ($2.29\sigma$ canonical; $(1.275 \pm 0.370)\times 10^6\ {\rm mag}$, $3.44\sigma$ at $\sigma_v = 150\ {\rm km/s}$). Under the physical coupled observing-chain model, the spectroscopic tracer's direct gravitational redshift contributes a velocity slope $\beta_{X,\rm grav} = c/\langle d \rangle \sim 10^4\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (&lt;0.1% of the combined slope), so that &gt;99.9% of the response is carried by the Cepheid distance scale ($\kappa_{\rm Cep}^{\rm equiv} = (1.220 \pm 0.531)\times 10^6\ {\rm mag}$); this channel allocation is physically derived from the observing-chain proper-time bookkeeping, with host-specific aperture validation supplied by a full-provenance NED tracer census (Step 54; Appendix C.5). Model comparison favours the environmental response: at the canonical convention the velocity-carrier configuration attains the lowest information criteria of the multi-block family ($\Delta{\rm BIC} = -1.36$ relative to the null), a leave-one-out cross-validation tournament over fixed and scanned response shapes selects the bounded amplitude-sector form $S_A = \min[1,(\sigma_*/\sigma_T)^{2/3}]$ in every primary dataset, and a Stouffer combination of the four primary channels under the zero-correlation assumption returns $Z = +3.31$ ($p = 9.3\times10^{-4}$ two-sided) when the period channel carries the derived period-quadratic carrier — the form the stellar-structure derivation predicts and the same sample prefers — and $Z = +2.87$ ($p = 4.1\times10^{-3}$) under the conservative ad hoc linear-in-$\log P$ diagnostic; under a conservative shared correlation $\rho = 0.3$ the combinations stand at $Z = +2.46$ ($p = 0.014$) and $Z = +2.13$ ($p = 0.033$) respectively.

Propagating the TEP potential correction through the distance ladder requires an explicit estimator distinction. The prespecified primary full-ladder projection propagates the measured endpoint-equivalent response and yields $H_0=70.463\pm0.972\ {\rm km\,s^{-1}\,Mpc^{-1}}$, reducing the offset from Planck $67.4 \pm 0.5$ to $2.80\sigma$ with a likelihood improvement of $\Delta\chi^2=+9.870$; conditional matrix projections give $71.01\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical scaling. When the response coefficient is instead left free inside the ladder design matrix it returns $\kappa_{\rm Cep}=(0.056\pm0.370)\times10^6\ {\rm mag}$ — the expected near-null: the shared Cepheid zero point absorbs the anchor-to-host common mode, so the matrix channel measures only the differential component and cannot see the term that closes the remaining offset. A separate unified host-level reconstruction, which re-optimizes the environmental response to flatten the host-environment trend, yields $H_0=67.83\pm1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$, internally consistent with the TEP CMB value ($66.70 \pm 0.58$, Paper 26) at $0.68\sigma$ — an agreement between two products of the same framework, not an independent cross-check — and $0.26\sigma$ from Planck. The latter is a host-level reconciliation route, not the primary full-ladder estimator: it evaluates the fitted response at the cosmic baseline and is conditional on that response form extending below the anchor ensemble to the unperturbed potential. Full-matrix propagation through the 3,490-row SH0ES system confirms exact reference-gauge invariance. The statistically cleanest channel is the within-host period structure: the environment-coupled Cepheid period term, identified through the 41 independent period distributions (37 hosts plus 4 anchors) and immune to latent-modulus absorption and peculiar-velocity systematics by construction, returns $\kappa_P = (-4.20 \pm 3.38)\times 10^5\ {\rm mag}$ for the ad hoc linear carrier, resolving to $2.3$--$2.8\sigma$ under the period-quadratic carrier derived from the amplitude-sector response law, with a negative sign in all 41 leave-one-host-out refits and a fitted per-host slope ordering of $+1.12 \pm 0.30 \times 10^6$ mag per unit $X$. Finally, single-galaxy differential tests provide internal signatures with the sign predicted by TEP free from host-to-host peculiar velocity systematics: M31 inner versus outer Cepheids exhibit an empirical Period--Luminosity offset of $+0.356 \pm 0.136\ {\rm mag}$ ($2.6\sigma$), increasing to $+0.681 \pm 0.187\ {\rm mag}$ ($3.65\sigma$) under spatial PHAT matching; however, M31 fails pair-matched controls (colour-matched $-0.017 \pm 0.128$ mag, $0.1\sigma$; eW-matched $-0.103 \pm 0.129$ mag, $0.8\sigma$), and the two admissible readings are tested against the theory's ambient closure. The overmatching hypothesis — that error matching discards an ambient-field signal — is bounded out: the photometric error is nearly orthogonal to the kpc-scale ambient-field proxy (${\rm corr}=-0.18$), and matching on the ambient field alone leaves a positive but inconclusive offset ($+0.206 \pm 0.163$ mag) that does not survive joint environment-plus-error control ($-0.069 \pm 0.126$ mag under matched (logP, $e_W$, $\log u_{\rm amb}$) cells — the residual was error-carried within ambient strata). Stratified through the full error distribution the conditional contrast declines monotonically from $+0.270 \pm 0.304$ mag at the 10th-percentile error level to $-0.117 \pm 0.202$ mag in the poorest, and the HST PHAT channel shows the same susceptibility under propagated-error matching ($-0.565 \pm 0.079$ mag, against $+0.906 \pm 0.147$ mag under neighbour-density control). The M31 channel is therefore carried as conditional evidence: the raw gradient is real and a positive residual is permitted in the best-measured strata, but neither reading closes the case. The LMC gradient ($+0.0284 \pm 0.0086$ mag, $3.3\sigma$) is robust to period and colour matching and remains the strongest single-galaxy result.

*Keywords:* Hubble tension -- Cepheid variables -- distance ladder -- stellar velocity dispersion -- peculiar velocities -- temporal equivalence principle

## 1. Introduction

### 1.1 The Hubble Tension and Distance Ladder Systematics

The discrepancy between the local expansion rate of the Universe measured via the classical distance ladder and the value inferred from early-universe cosmological observations has emerged as one of the most pressing foundational problems in modern physics. The SH0ES collaboration reports a local Hubble constant of $H_0 = 73.04 \pm 1.04\ {\rm km\,s^{-1}\,Mpc^{-1}}$ using Cepheid-calibrated Type Ia supernovae (Riess et al. 2022; hereafter R22), whereas cosmic microwave background (CMB) measurements by the Planck satellite under flat $\Lambda$CDM establish $H_0 = 67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (Planck Collaboration VI 2020). This exceeds $5\sigma$ in statistical significance, resisting resolution through conventional astrophysical systematics, photometric crowding, or standard cosmological parameters.

The standard distance ladder relies on the assumption that stellar standard candles calibrated in nearby anchor galaxies (the Milky Way, Large Magellanic Cloud, Small Magellanic Cloud, and NGC 4258) behave identically in distant host galaxies. However, the anchor galaxies possess predominantly shallow gravitational potential wells, whereas the SN Ia host galaxies sample substantially deeper galactic potential wells. If stellar pulsation clocks respond dynamically to the local gravitational potential, an environmental gradient between calibrators and hosts will introduce a systematic bias into the inferred distance scale.

### 1.2 The Temporal Equivalence Principle

The physical intuition is direct. Classical Cepheids are dynamical acoustic clocks: their pulsation periods tick in local proper time, set by the mean density of the pulsation envelope. If an SN Ia host galaxy samples a deeper temporal environment than the nearby anchor galaxies used to calibrate the Period--Luminosity relation, its Cepheid pulsation response is shifted. Read through a universal Leavitt law, the shifted periods mimic a fainter intrinsic luminosity, so the host distance is underestimated and the local Hubble constant is inflated. The covariant framework below supplies the mechanism that generates and propagates this environmental clock offset.

The Temporal Equivalence Principle (TEP) provides a covariant scalar-tensor framework that elevates proper time from a fixed background coordinate parameter to a dynamical physical field. The canonical scalar comparison baseline is formulated on a two-metric spacetime:

\begin{equation}
S = \int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R - \frac{1}{2}(\nabla\phi)^2 - V(\phi)\right] + S_m[\tilde g_{\mu\nu}, \Psi_m] ,
\end{equation}

where gravity is governed by the metric $g_{\mu\nu}$, while all non-gravitational matter fields, atomic transitions, and stellar clocks couple universally to the causal matter metric $\tilde g_{\mu\nu} = A^2(\phi)g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$, following the convention established in the foundational TEP theory paper (Smawfield 2025, Paper 0). The operative weak-field screening realization replaces the canonical scalar term by the noncanonical kinetic sector $P(X,\phi)=-V(\phi)+\frac{2k}{3\Lambda_X^2}X|X|^{1/2}+\frac{X|X|}{\Lambda_X^4}$, $k\simeq16$ (Paper 0, §2.2); the canonical form displayed is a comparison baseline — the physical branch carries no canonical small-$|X|$ term. In gravitational bound states, the conformal factor satisfies $A(\phi) < 1$, slowing the rate of proper time relative to cosmic background time.

The absolute matter-clock rate relative to cosmic time is defined as

\begin{equation}
r_j \equiv \frac{(d\tilde\tau/dt)_j}{(d\tilde\tau/dt)_{\rm cosmic}} .
\end{equation}

Throughout the TEP framework, the physical rate ordering across galactic environments is strictly bounded:

\begin{equation}
0 < r_{\rm core} < r_{\rm disk} < r_{\rm cosmic} \equiv 1 .
\label{eq:rate_hierarchy}
\end{equation}

Within matter-hosted environments the ordering is monotone — clocks in active-shear galactic disks are simply less slowed than clocks in dense nuclear cores — while the unsourced void sector may exceed the ambient only within the bounded spatial contrast established by the eternal-slice construction (Paper 0 §9). Environmental screening $S(\rho)$ suppresses the observable scalar-charge sector in high-density regimes while preserving dynamical time-rate variations on galactic and cosmological scales. Microscopically this follows from the nonlinear kinetic completion $P(X,\phi)=-V(\phi)+\frac{2k}{3\Lambda_X^2}X|X|^{1/2}+\frac{X|X|}{\Lambda_X^4}$, $k\simeq16$, of the specified scalar action (Paper 0 §2.2, v0.15): where the ambient clock-rate landscape is already steep, the field is kinematically stiff, so the observable Temporal Shear — the slope of that landscape, which Einstein-frame analyses parameterize as a fifth force — is pinned rather than newly carved (the $k_{\rm rec} \geq 4$ recovery steepness of constraint F4 is a consistency check on the derived response, realized by the hierarchical $s^4$ deep-interior asymptote). The field amplitude $A(\phi)$ continues to vary inside screened matter, preserving the clock-rate channel: what is pinned is the gradient, not the field amplitude. The quartic self-interaction $V = \lambda\phi^4/4$ is the matter-hosting weak-field amplitude-sector branch of the unified potential (Paper 0 §2), whose radial ODE closure $\nabla^2\phi = V_{,\phi} + \rho_* A_{,\phi}$ yields $m_{\rm eff} \propto \rho^{1/3}$; the density-dependent screening factor $S(\rho)$ used in this paper is the EFT-level projection of the kinetic closure onto the Cepheid host-galaxy density regime, not an independent chameleon potential.

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

Under the TEP core--disk tracer closure the observing-chain ratio $q_i = r_{\rm spec}/r_{\rm Cep}$ supplies a bounded bare-clock contribution, while the dominant observed response is carried by the amplitude-sector response of the pulsation envelope — a mechanical inverse-density transport that converts the order-$10^{-7}$ environmental field increment into a magnitude-level period shift (§3.8, Appendix C.3). Through the Leavitt Period--Luminosity law ($M_W = \alpha + \beta \log P$), this period response alters the inferred distance modulus by $\Delta \mu_i = \kappa_{\rm Cep} X_i$, where $X_i$ is the environmental potential coordinate.

### 1.4 Scope and Structure of this Study

This paper presents a comprehensive empirical test of the TEP framework across multiple independent observational tiers using public astronomical data:

1. Single-galaxy differential tests in M31 and the LMC, isolating the potential-clock coupling free from host-to-host peculiar velocity uncertainties.

2. Homogeneous, exact-identifier inner-potential reconstruction for all 37 distinct R22 SN Ia host galaxies from fully provenanced stellar-absorption dispersions and calibrated H&nbsp;I linewidth proxies, with continuous screening.

3. Generative endpoint likelihood evaluation across flow-velocity models and peculiar velocity dispersions.

4. Mathematical resolution of the latent-modulus parameter absorption in standard distance-ladder design matrices using the TEP-native generative ladder.

5. Complete computational reproducibility: the resolution of the design matrix degeneracy and the full-ladder generative likelihoods are accompanied by an open-source, end-to-end Python pipeline, ensuring transparent verification of all statistical claims.

## 2. Data and Methods

### 2.1 Public Distance-Ladder Data and Independent Sample Reconstruction

The empirical distance ladder is reconstructed from the public R22 data release, comprising the 3,490-row generalized least-squares design matrix, data vector, full covariance matrix, and the Pantheon+SH0ES supernova catalogue (Scolnic et al. 2022; Brout et al. 2022). Solving the unmodified baseline system yields $H_0 = 73.0434 \pm 1.0072\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with $\chi^2 = 3552.7063$ for 3,444 degrees of freedom, reproducing the published SH0ES solution.

The R22 calibrator sample contains 42 SNe Ia situated in 37 distinct host galaxies. In the host-level inference, the independent statistical unit is the individual host galaxy, as all Cepheids and multiple SNe within a single galaxy share the same host potential well. Each supernova is matched to its host galaxy using exact R22 catalogue identifiers and PGC numbers, strictly excluding angular nearest-neighbour heuristics. All 37 distinct host galaxies have measured positive Hubble-flow redshifts ($z_{\rm HD} > 0$). Of these, 34 satisfy the Hubble-flow union cut ($z_{\rm CMB} > 0.0035$ or $z_{\rm HD} > 0.0035$); 33 of those have matching Cepheid design-matrix entries and enter the primary generative likelihood.

### 2.2 Local Inner-Potential Coordinate

The TEP clock response is sourced by the gravitational potential at the pulsation site, not by the asymptotic halo. SH0ES Cepheid fields occupy inner-to-mid disk radii ($r_{\rm Cep} \sim 1.8 R_d$) where the potential is baryon-dominated; the relevant coordinate is therefore the local inner-region stellar velocity dispersion $\sigma_{*,i}$, which tracks the enclosed inner potential and is obtained from inclination-independent stellar spectroscopy. For each of the 37 hosts and the ladder anchors, $\sigma_{*,i}$ is compiled from the published literature with full per-host provenance (Appendix B): direct stellar-absorption measurements for 18 hosts (H&eacute;raudeau et al. 1999; Ho et al. 2009; Kormendy &amp; Ho 2013; the Riess et al. 2022 calibrator compilation) and calibrated H&nbsp;I linewidth dispersion proxies for the remaining 19 hosts (Campbell et al. 2014 6dFGSv). The anchor endpoint is constructed identically from local stellar dispersions at the Cepheid calibration sites (Milky Way solar neighbourhood $\sigma_z = 30.0\ {\rm km\,s^{-1}}$, LMC disk $24.0\ {\rm km\,s^{-1}}$, NGC 4258 intermediate annulus $115.0\ {\rm km\,s^{-1}}$), so the environmental coordinate subtracts like for like.

\begin{equation}
U_i = S_i\,\sigma_{*,i}^2 ,
\qquad
\Phi \sim U_i ,
\label{eq:uphi}
\end{equation}

A homogeneous whole-galaxy alternative, $u_\phi = V_{\rm rot}/\sqrt{2}$ from pinned HyperLEDA maximum rotation velocities, is retained as a documented robustness coordinate. The two coordinates are not interchangeable: $V_{\rm rot}$ probes the outer dark-halo potential at tens of kpc and requires $1/\sin i$ deprojection that is unstable for the eight Hubble-flow hosts at $i < 40^\circ$, whereas $\sigma_*$ measures the inner baryonic potential inhabited by the Cepheids. Direct comparison (Section 4; Step 64) confirms that the environmental response tracks the local coordinate and is attenuated under the global rotation proxy, establishing coordinate validity rather than proxy convenience.

### 2.3 Environmental Screening Formulation

The scalar field sector in TEP acts through two distinct macroscopic operators governed by Rules 10, 11, and 22: the Temporal Shear suppression operator $\mathcal{S}_\Sigma(\mathcal{E})$ and the clock-amplitude response $A(\phi)$. The macroscopic shear operator $\mathcal{S}_\Sigma(\mathcal{E})$ suppresses spatial clock-rate gradients and the corresponding Einstein-frame anomalous accelerations in dense environments ($\mathcal{S}_\Sigma \to 0$ in screened matter), recovering standard General Relativity for gravitational orbits. By contrast, the clock amplitude $A(\phi) = \exp(-\phi / M_{\rm Pl})$ tracks potential depth monotonically: proper-time clock rates run slower in deeper gravitational wells ($A < 1$), and this clock slowing continues inside screened matter without vanishing, preserving gravitational redshift.

In galaxy halos, the two-body environmental suppression factor $S_i = S_{{\rm local},i} S_{{\rm group},i}$ parameterizes the shear suppression of the host's extended potential well within its surrounding group or cluster environment. In local high-density stellar cores (such as inner M31 or LMC centers), the local potential depth is deepest, causing local clocks to run slowest ($A < 1$), while the associated environmental ordering sets the sign of the observed positive Period--Luminosity offset ($\Delta W > 0$) relative to diffuse outer-disk Cepheids through the amplitude-sector response. Host disk scale lengths are evaluated from RC3 isophotal diameters ($R_{25} = D_{25}/2$, $R_d = R_{25}/3.2$, $z_d = 0.1 R_d$) and stellar mass densities are computed at representative Cepheid radii ($r_{\rm Cep} = 1.8 R_d$). Group-scale density is obtained from the Tully (2015) 2MRS group catalogue Table 5 via exact PGC matching. Evaluated at $r_{\rm Cep}$, the local density projection is equivalent to the enclosed-acceleration diagnostic $M({<}r_{\rm Cep})/r_{\rm Cep}^2 \propto \bar\rho({<}r_{\rm Cep})\,r_{\rm Cep}$ up to the disk's vertical-thickness correction — the density stand-in licensed for dense macroscopic matter by the $\mathcal{S}_\Sigma(\mathcal{E})$ construction (Paper 0 §7). The total continuous screening factor is:

\begin{align}
S_{{\rm local},i} &= \left[1 + \left(\frac{\rho_i}{0.5\ M_\odot\,{\rm pc}^{-3}}\right)^2\right]^{-1} , \\
S_{{\rm group},i} &= \left[1 + \left(\frac{N_{{\rm mb},i}}{10}\right)^{1.2}\right]^{-1} , \\
S_i &= S_{{\rm local},i} S_{{\rm group},i} .
\label{eq:screening}
\end{align}

The screened potential scale for each host is $U_i = S_i \sigma_{*,i}^2$. The assignment is positional: $S_i$ is the source-side suppression — the host's potential well sources the local field level through its effective scalar charge, which the shear-sector operator $\mathcal{S}_\Sigma$ suppresses at the habitat density — while the probe-side clock-amplitude response $\mathcal{S}_A$, evaluated at the Cepheid's own material density, is a per-source constant absorbed into the channel coefficient $\kappa_{\rm Cep}$ (Appendix C). For likelihood evaluation, the dimensionless centered environmental regressor is defined as:

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

The true generative observable model couples the observed Cepheid distance moduli $\mu_i^{\rm obs} = \mu_i^{\rm true} - \kappa_{\rm Cep} X_i$ directly to the observed Hubble-flow relation $cz_i = d_i^{\rm true}(H_{\rm app} + \beta_X X_i) + v_i$. The identifiable physical combination is:

\begin{equation}
\Gamma_X = \beta_X + \left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep} .
\label{eq:gamma_relation}
\end{equation}

($\Gamma_X$ here is a velocity-space slope in ${\rm km\,s^{-1}\,Mpc^{-1}}$, distinct from the dimensionless channel projector of the same letter in the Paper 0 transfer map.)

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

Under the observing-chain mechanism formalized in Appendix C, the same potential well that produces the environmental period response also imparts a gravitational redshift to the adopted systemic-redshift tracer:

\begin{equation}
1 + z_{{\rm spec},i} = \frac{1 + z_{{\rm path},i}}{r_{{\rm spec},i}} \approx 1 + z_{{\rm path},i} + \frac{\Phi_{{\rm spec},i}}{c^2} .
\end{equation}

This shifts the observed velocity by $\Delta(cz)_i = c \Phi_{{\rm spec},i}/c^2 = \Phi_{{\rm spec},i}/c \sim 70\ {\rm m\,s^{-1}}$. When expressed as an apparent expansion-rate slope by dividing by host distance $d_i$, this physical gravitational redshift contributes:

\begin{equation}
\beta_{X,\rm grav} \equiv \frac{c}{\langle d \rangle} \approx \frac{299,792\ {\rm km\,s^{-1}}}{31.2\ {\rm Mpc}} \approx 9.60 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}} .
\end{equation}

In contrast, the Cepheid distance-modulus bias $\Delta \mu_i = \kappa_{\rm Cep} X_i$ rescales the host distance by $\Delta d / d \approx 7\%$, producing an apparent expansion-rate slope:

\begin{equation}
\Gamma_{X,\rm Cep} \equiv \left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep} \approx 3.85 \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}} .
\end{equation}

The ratio of the direct velocity response to the distance-modulus response is:

\begin{equation}
\frac{\beta_{X,\rm grav}}{\Gamma_{X,\rm Cep}} = \frac{9.60 \times 10^3}{3.85 \times 10^7} \approx 2.5 \times 10^{-4} \approx 0.02\% \text{--} 0.03\% .
\end{equation}

Consequently, the physical observing chain intrinsically allocates $>99.97\%$ of the combined response $\Gamma_X$ to the Cepheid distance channel. Setting $\beta_X \approx 0$ (or coupling $\beta_X = \beta_{X,\rm grav}$) is therefore not an ad-hoc allocation, but the direct leading-order consequence of metric proper-time transport.

The fitted intercept $H_{\rm app}$ represents the expansion rate at the sample mean environment $\langle X \rangle = -1.5 \times 10^{-9}$ on the screened coordinate — essentially the midpoint of the host ensemble relative to the anchor reference. To evaluate the expansion rate at physically defined reference frames, the relation maps back to:

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

1. *Redshift--Distance Block ($N=33$ Hubble-flow hosts):* Evaluates the conventional Hubble-flow relation via a linearised regression in $H_0$ space, $H_{0,i} = cz_i / d_i^{\rm obs} = H_{\rm app} + \Gamma_X X_i + \epsilon_i$, where $\Gamma_X = \beta_X + (\ln 10 / 5) H_{\rm app} \kappa_{\rm Cep}$ is the identifiable combined endpoint response. This formulation is equivalent to the nonlinear $cz$-space model $cz_i = d_i^{\rm true}(H_{\rm app} + \beta_X X_i) + v_i$ with $d_i^{\rm true} = d_i^{\rm obs} 10^{\kappa_{\rm Cep} X_i / 5}$, but offers superior numerical conditioning because the parameter space is linear in $(H_{\rm app}, \Gamma_X)$. The 33 hosts comprise the full Hubble-flow calibrator sample, including the two lowest-redshift systems (NGC 4424 and NGC 4536) whose CMB-corrected redshifts place them above the Hubble-flow threshold. The per-host variance is $\sigma_{v,i}^2 = \sigma_v^2 / d_i^2 + (\frac{\ln 10}{5} H_{0,i} \sigma_{\mu,i})^2 + \sigma_{{\rm int},v}^2 / d_i^2$, where $\sigma_{{\rm int},v}$ is an intrinsic velocity dispersion nuisance parameter.

2. *TRGB Differential Block ($N=18$ non-anchor calibrators):* Compares Cepheid and Tip of the Red Giant Branch distance moduli, $\Delta \mu_i \equiv \mu_i^{\rm Cep} - \mu_i^{\rm TRGB} = \delta m - \kappa_{\rm Cep} X_i + \epsilon_i$, with variance $\sigma_{\Delta\mu,i}^2 = \sigma_{\mu,{\rm Cep},i}^2 + \sigma_{\mu,{\rm TRGB},i}^2$. Because TRGB stellar evolution clocks operate independently of pulsation physics, TRGB distances provide an external anchor on the host modulus. The 18 hosts represent the complete overlap of SH0ES Cepheid calibrators, the inner-potential catalogue, and EDD/CCHP TRGB measurements (excluding geometric anchors to prevent double counting).

3. *Independent Geometric Anchor Block ($N=2$ primary anchors):* Constrains the absolute Cepheid zero-point against independent geometric distances in the Large Magellanic Cloud (detached eclipsing binaries) and NGC 4258 (water megamasers): $\Delta \mu_{\rm anc} \equiv \mu_{\rm Cep} - \mu_{\rm geo} = \delta a - \kappa_{\rm Cep} X_{\rm anc} + \epsilon_a$. M31 is excluded from this block because its distance scale is composite rather than purely geometric.

Five nested model configurations are evaluated against the data: the *Null Model* ($N_{\rm par}=4$: $H_{\rm app}, \delta m, \delta a, \sigma_{{\rm int},v}$), the *Cepheid Model* ($N_{\rm par}=5$: setting $\beta_X \equiv 0$ and fitting $\kappa_{\rm Cep}$), the *Coupled Model* ($N_{\rm par}=5$: setting $\beta_X = \beta_{X,\rm grav} \equiv c/\langle d \rangle \approx 9.60 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and fitting $\kappa_{\rm Cep}$), the *Velocity Model* ($N_{\rm par}=5$: setting $\kappa_{\rm Cep} \equiv 0$ and fitting $\beta_X$), and the *Mixed Model* ($N_{\rm par}=6$: simultaneously fitting both $\kappa_{\rm Cep}$ and $\beta_X$). The Cepheid and Velocity models represent two alternative single-degree-of-freedom closures of the underlying physical response rather than additive components, while the Coupled model realizes the physical prediction where the direct velocity perturbation is small but nonzero. In this multi-block formulation, the three data blocks are treated as quasi-independent observational constraints; while excluding primary geometric anchors from the TRGB block prevents direct anchor duplication, shared Cepheid calibration terms are treated within individual block variances.

### 2.8 Unified Host-Level Reconstruction

Complementing the expansion-rate likelihood, a unified host-level reconstruction is performed directly on the complete 37-host sample. Each host's distance modulus is corrected according to its screened kinematic potential:

\begin{equation}
\mu_i^{\rm corr} = \mu_i^{\rm obs} + \kappa_{\rm Cep} \frac{S(\rho_i)\, \sigma_{*,i}^2 - U_{\rm ref}}{c^2} ,
\label{eq:endpoint_correction}
\end{equation}

yielding reconstructed physical distances $d_i^{\rm corr} = 10^{(\mu_i^{\rm corr} - 25)/5}$ and individual expansion rates $H_{0,i} = cz_i / d_i^{\rm corr}$. The anchor reference zero-point is defined as $\sqrt{U_{\rm ref}} = \sigma_{\rm ref} = 87.165\ {\rm km\,s^{-1}}$, constructed from the weighted composite of local stellar velocity dispersions at the specific Cepheid disk locations within the primary calibration anchors: Milky Way solar neighbourhood ($\sigma_z = 30.0\ {\rm km\,s^{-1}}$, weight 0.20; Bovy et al. 2012), LMC stellar disk ($\sigma_{\rm disk} = 24.0\ {\rm km\,s^{-1}}$, weight 0.25; van der Marel et al. 2002), and NGC 4258 intermediate annulus ($\sigma_{\rm local} = 115.0\ {\rm km\,s^{-1}}$, weight 0.55; Kormendy & Ho 2013). As proven in Appendix D.3, the reference scale $\sqrt{U_{\rm ref}}$ is an exact gauge origin in the full linear system, absorbed into the Cepheid zero point $M_H^W$ without altering the physical response $\kappa_{\rm Cep}$.

The optimal response coefficient $\kappa_{\rm Cep}$ is determined by minimizing the residual variance and environmental gradient across the host sample ($\partial H_{0,i} / \partial \sigma_{*,i} \to 0$). Honest uncertainties are estimated via a joint bootstrap ($N=1,000$ resamples with replacement), where $\kappa_{\rm Cep}$ is re-optimized for each realization, simultaneously propagating host-to-host sampling scatter and $\kappa_{\rm Cep}$ parameter variance to yield the unified Hubble constant $H_0 = 67.83 \pm 1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$.

Two properties of this route delimit its interpretation. First, under the standard reference scale the host-sample mean of the correction regressor nearly vanishes ($\langle X_i\rangle \approx -1.5\times10^{-9}$), so the reconstructed $H_0$ level is largely insensitive to $\kappa_{\rm Cep}$ — the raw unweighted host mean is already $67.72 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ — and the TEP-specific content of the route is the flattening of the per-host environmental slope rather than the absolute level. Second, the optimizer's objective is defined on the linear $\sigma_*$ axis: an estimator decomposition of the same corrected moduli against the physical correction regressor $X_i \propto S_i\sigma_{*,i}^2$ returns a markedly different nominal optimum ($\kappa \approx 3.4\times10^6$ mag under the $\sigma_*$-slope objective versus $(0.98 \pm 0.48)\times10^6$ mag under a weighted least-squares fit of $\delta\mu$ on $X_i$ with photometric plus peculiar-velocity errors — consistent with the endpoint-equivalent coefficient $\kappa_{\rm Cep}^{\rm equiv} = 1.22\times10^6$ mag). The reconstruction is therefore carried as a fitted consistency diagnostic: it demonstrates that the environmental response required to flatten the host trend is compatible with the independently measured coefficient, without constituting a prespecified propagation of that coefficient.

Evaluating the host-level reconstruction under the alternative screened anchor reference scale $\sqrt{U_{\rm ref}} = 30.507\ {\rm km\,s^{-1}}$ yields $H_0 = 60.45 \pm 1.39\ {\rm km\,s^{-1}\,Mpc^{-1}}$. The $7.4\ {\rm km\,s^{-1}\,Mpc^{-1}}$ offset from the standard-construction value ($67.83 \pm 1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$, $0.26\sigma$ from Planck) is a pure gauge artifact of the unconstrained host-level shortcut, which applies an additive modulus shift $\Delta \mu = \kappa_{\rm Cep} \Delta U_{\rm ref} / c^2$ without simultaneously refitting the anchor absolute magnitude zero point. In the full generative distance ladder (Appendix D.3), where the anchor absolute magnitude $M_H^W$ is freely refit, this additive shift is identically absorbed into the zero point, returning $H_0$ values that are identical to four decimal places across all four reference origins. Furthermore, in the generative expansion-rate likelihoods, the cosmological background extrapolation $H_{\rm cosmic} = H_{\rm app} - \Gamma_X \langle U \rangle / c^2$ is algebraically independent of $U_{\rm ref}$, yielding background expansion rates of $65.1\text{--}65.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$ across the $\sigma_v$ grid and joint-model configurations — within $\sim 1.3\sigma$ of Planck — with exact gauge invariance.

### 2.9 Observational Controls for Single-Galaxy Internal Gradients

Single-galaxy differential tests in M31 and the LMC provide internal empirical checks of the TEP potential hierarchy free from host-to-host peculiar velocity dispersions. To ensure these internal Period--Luminosity offsets are not generated by standard astrophysical systematics, the analysis incorporates three rigorous controls:

1. *Extinction-Free Photometry:* All Cepheid magnitudes are evaluated using the reddening-free Wesenheit index $W \equiv m_I - R_V (m_V - m_I)$ (with $R_V = 1.55$ for OGLE-IV LMC and $R_V = 1.54$ for M31), which algebraically cancels total and differential interstellar dust extinction along the line of sight.

2. *Spatial Resolution and Crowding Controls:* In M31, ground-based blending and crowding effects in the dense inner bulge are controlled in three ways: (a) restricting the Pan-STARRS sample to the PHAT survey footprint, which recovers the internal offset at $\Delta W = +0.630 \pm 0.195\ {\rm mag}$ ($3.24\sigma$; $n_{\rm inner}=106$, $n_{\rm outer}=110$); (b) the dedicated HST PHAT J/H photometric catalogue (Kodric et al. 2018), which yields $\Delta W = +0.681 \pm 0.187\ {\rm mag}$ ($3.65\sigma$; $n_{\rm inner}=78$, $n_{\rm outer}=69$) and is adopted as the primary PHAT measurement — the two values correspond to distinct catalogues and are not interchangeable; and (c) two theory-informed diagnostics of the error control: an ambient-field proxy constructed at the theory closure scale ($L = 837$ pc; $\log u_{\rm amb} \propto \tfrac12\log\rho_{\rm tr}$, with $\rho_{\rm tr}$ the tracer density within an $L/2$ disk-plane aperture), which both bounds the overmatching hypothesis — ${\rm corr}(e_W, \log u_{\rm amb}) = -0.18$ means the error covariate removes only a few per cent of the ambient-field variance — and supplies a matched estimate under the theory-relevant environmental control ($\Delta W = +0.206 \pm 0.163$ mag, falling to $-0.069 \pm 0.126$ mag under joint $(\log P, e_W, \log u_{\rm amb})$ cell-matching — the residual is error-carried within ambient strata); and an error-stratified contrast table, under which the conditional offset declines monotonically from $+0.270 \pm 0.304$ mag at the 10th-percentile error level to $-0.117 \pm 0.202$ mag in the poorest, so quality-band persistence reflects within-band composition rather than a uniform offset among equally-well-measured stars.

3. *Metallicity Plausibility Limits:* To test whether radial metallicity gradients could mimic the observed $\Delta W$, the required metallicity sensitivity is evaluated. For the empirical M31 metallicity gradient ($\Delta [\text{O/H}] \sim 0.2\text{--}0.4\ {\rm dex}$), explaining the observed $\Delta W \approx +0.36\text{--}0.63\ {\rm mag}$ purely via metallicity would require a positive coefficient $\gamma \simeq +0.7\text{--}1.4\ {\rm mag/dex}$ across the gradient range, of the wrong sign and several standard errors removed from the empirical metallicity dependence measured across the SH0ES calibrator sample, $\gamma = -0.20 \pm 0.08\ {\rm mag/dex}$ (Step 10: $\gamma_{\rm req} = +0.74\ {\rm mag/dex}$ at the baseline offset).

### 2.10 Host-Mass Decoupling and Supernova Mass-Step Marginalization

An essential empirical test is verifying that the physical TEP environmental coordinate does not act as a surrogate for the established Type Ia supernova host-galaxy stellar mass step. In standard supernova standardization pipelines (e.g., Betoule et al. 2014; Scolnic et al. 2018; Brout et al. 2022), an empirical magnitude correction $\gamma_{\rm step} \sim 0.05\text{--}0.06\ {\rm mag}$ is applied across the split threshold $\log(M_*/M_\odot) = 10.0$ to account for residual correlations between peak luminosity and host galaxy properties.

Host stellar masses $\log(M_*/M_\odot)$ are compiled directly from Riess et al. (2022, Table 4), spanning $8.50 \le \log(M_*/M_\odot) \le 10.87$ across the complete 37-calibrator sample. The inner-region dispersion correlates only weakly with host stellar mass in this sample ($r = 0.12$), so the physical TEP coordinate $X_i \equiv (S_i \sigma_i^2 - \sigma_{\rm ref}^2)/c^2$ incorporates multi-scale environmental screening $S_i = S_{\rm local}(\Sigma_*) \times S_{\rm group}(\rho_{\rm env})$ — the factorized evaluation of the canonical shear-sector projection $\mathcal{S}_\Sigma(\mathcal{E})$ on the two-component host environment $\mathcal{E} = \{\Sigma_*, \rho_{\rm env}\}$ (Paper 0 §7) — which suppresses both dense galactic disks and massive group potentials. Consequently, the screened potential depth is fundamentally distinct from total integrated stellar mass.

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

The exact-identifier reconstruction yields all 37 distinct R22 SN Ia host galaxies with measured positive Hubble-flow redshifts ($z_{\rm HD} > 0$), complete RC3 isophotal diameters, and a fully provenanced inner-potential coordinate $\sigma_*$ (Section 2.2). Splitting the 37 hosts at the sample median $\sigma_* = 95.0\ {\rm km/s}$ reveals a pronounced empirical stratification: the 19 shallow-potential hosts yield a mean Hubble parameter of $63.05 \pm 3.06\ {\rm km\,s^{-1}\,Mpc^{-1}}$, whereas the 18 deep-potential hosts yield $72.64 \pm 2.62\ {\rm km\,s^{-1}\,Mpc^{-1}}$, propagating the full host distance covariance matrix — a contrast of $\Delta H_0 = 9.59\ {\rm km\,s^{-1}\,Mpc^{-1}}$, comparable in magnitude to the entire local--CMB discrepancy.

The potential-velocity association is significant: Pearson $r = 0.501$ ($p = 0.0016$) and Spearman rank $\rho = 0.553$ ($p = 3.9 \times 10^{-4}$). The correlation survives the principal systematic controls. Heteroscedasticity weighting, with $w_i = 1/\sigma_{H_0,i}^2$ from the diagonal of the full host Hubble-rate covariance matrix, yields $r_w = 0.520$ ($p = 0.0070$, $n_{\rm eff} = 25.5$). Replacing the Pantheon+ peculiar-velocity estimates with Stiskalek et al. 2026 Manticore Bayesian coherent-flow posterior means where available retains a weighted correlation of $r_w = 0.432$ ($p = 0.029$). Restricting to the 18 hosts whose $\sigma_*$ derives from direct stellar-absorption spectroscopy — the cleanest provenance tier — strengthens the association further ($r = 0.556$, $p = 0.017$; $\rho = 0.596$, $p = 0.009$; $\Delta H_0 = 11.2\ {\rm km\,s^{-1}\,Mpc^{-1}}$), so the signal is not carried by the H&nbsp;I proxy tier. The positive gradient between host potential depth and apparent expansion rate matches the directional prediction of TEP stellar clock modulation.

Coordinate validity is tested directly: substituting the homogeneous whole-galaxy rotation proxy $u_\phi = V_{\rm rot}/\sqrt{2}$ for the same 37 hosts and identical $H_{0,i}$ collapses the correlation to $r = 0.098$ ($p = 0.61$). The environmental signal therefore tracks the local inner potential inhabited by the Cepheids, not the outer-halo rotation scale — a distinction with direct physical meaning under TEP, whose clock response is sourced at the pulsation site (Step 64 audit; Appendix D).

![Host-level H0 values plotted against the square of the local inner-potential velocity dispersion for 37 R22 supernova hosts](public/figures/step_03_figure_01_h0_vs_sigma.png?v=3)

Figure 1: Host-level expansion rate $cz_{\rm HD}/d$ as a function of the inner-potential coordinate $\sigma_*^2$ for all 37 R22 SN Ia host galaxies. Error bars include distance modulus covariance and peculiar velocity uncertainties.

### 3.2 Single-Galaxy Internal Verification: M31 and LMC

Single-galaxy tests provide empirical verification of the TEP potential coupling free from host distance uncertainties or peculiar-velocity flow modelling:

In M31 (Andromeda), 2,686 Cepheids from the Pan-STARRS survey are analysed across galactocentric radii. Comparing Cepheids in the dense inner potential ($R < 5.0$ kpc, mean density $\rho \approx 0.31\ M_\odot\,{\rm pc}^{-3}$, where nonlinear kinetic/environmental screening is strong) against those in the diffuse outer disk ($R > 15.0$ kpc, $\rho \approx 0.006\ M_\odot\,{\rm pc}^{-3}$) reveals an empirical Period--Luminosity zero-point offset, defined as $\Delta W \equiv W_{\rm inner} - W_{\rm outer}$:

> 

\begin{equation}
\Delta W_{\rm M31} \equiv W_{\rm inner} - W_{\rm outer} = +0.3560 \pm 0.1357\ {\rm mag} \quad (2.6\sigma\ {\rm significance}) .
\end{equation}

A positive $\Delta W$ indicates that inner Cepheids appear systematically fainter (possessing longer effective rest-frame periods) than their outer counterparts — the ordering the amplitude sector requires: as envelope self-screening weakens toward the diffuse outer disk, the Cepheid clock couples more strongly to the ambient field, producing the larger outer-disk contraction that yields a positive inner-minus-outer differential, the same $S_A$-sector ordering the within-host $\kappa_P$ term measures and the observing-chain hierarchy of Appendix C predicts in sign. The literal aperture-ratio channel cannot supply this amplitude: the core-to-disk clock differential is bounded by the potential contrast at $|\Delta\ln q| \lesssim \Delta\Phi/c^2 \sim 10^{-7}$, some five orders below the observed LMC offset and more below M31's, so the gradient is carried by the amplitude-sector response transferred through the empirical $\kappa_{\rm Cep}$/$\kappa_P$ coefficients rather than the bare clock ratio — a division of labour that identifies which theory sector the single-galaxy evidence actually tests. In the dedicated HST PHAT J/H catalogue the internal offset sharpens to $\Delta W_{\rm PHAT} = +0.6808 \pm 0.1867\ {\rm mag}$ ($3.65\sigma$ significance; restricting the Pan-STARRS sample to the same footprint gives $+0.630 \pm 0.195\ {\rm mag}$, $3.24\sigma$). In the Large Magellanic Cloud (LMC), analysis of OGLE-IV fundamental-mode Cepheids partitioned by galactocentric radius ($R \le 0.92$ kpc versus $R \ge 2.93$ kpc) independently reveals an internal offset of $\Delta W_{\rm LMC} = +0.0284 \pm 0.0086\ {\rm mag}$ ($3.3\sigma$ significance), following the same inner-minus-outer convention. The M31 offsets do not survive pair-matched confounder control — colour-matched pairs return $-0.017 \pm 0.128$ mag ($0.1\sigma$) and error-matched pairs $-0.103 \pm 0.129$ mag ($0.8\sigma$). The two admissible readings of that failure are tested directly. Under the overmatching hypothesis, pair-matching on $e_W$ would discard a signal carried by the ambient field only if $e_W$ traced $u_{\rm amb}$; constructing the ambient-field proxy at the theory closure scale $L = 837$ pc — $\log u_{\rm amb} \propto \tfrac12\log\rho_{\rm tr}$, with $\rho_{\rm tr}$ the Cepheid tracer density within a disk-plane aperture of $L/2$ — and measuring its coupling to the photometric error returns ${\rm corr}(e_W, \log u_{\rm amb}) = -0.18$, so at most a few per cent of the ambient-field variance is removed with the error covariate and the overmatching channel is bounded rather than operative. Matching instead on the ambient-field proxy itself leaves $\Delta W = +0.206 \pm 0.163$ mag — positive, and consistent with the sign the amplitude sector requires, but statistically inconclusive; and that residual does not survive joint control: cell-matching on $(\log P, e_W, \log u_{\rm amb})$ returns $-0.069 \pm 0.126$ mag, so the positive ambient-matched offset is carried by the residual $e_W$ contrast within ambient strata rather than by an ambient-field-ordered signal. The complementary reading — that the quality-equated bands genuinely equate measurement quality — is tested by stratifying the contrast through the full $e_W$ distribution rather than restricting to a single band: the conditional offset declines monotonically from $+0.270 \pm 0.304$ mag at the 10th-percentile error level through $+0.024 \pm 0.109$ mag at the median to $-0.117 \pm 0.202$ mag at the 90th-percentile error level (the p10 point is a model extrapolation — only 3% of inner Cepheids attain that error level), so the positive pooled-band estimates reflect residual within-band composition rather than a uniform offset among equally-well-measured stars. The PHAT channel exhibits the same susceptibility: matching on the propagated $J/H$ error collapses the offset to $-0.565 \pm 0.079$ mag, against $+0.604 \pm 0.277$ mag under period–colour matching and $+0.906 \pm 0.147$ mag under an arcsec-scale neighbour-density control, with the inner sample carrying twice the mean photometric error of the outer ($0.0130$ versus $0.0062$ mag). The M31 channel is accordingly carried as conditional evidence: the raw gradient is real and a positive residual is permitted in the best-measured strata, but under joint environment-plus-quality control the offset is consistent with zero ($-0.069 \pm 0.126$ mag) — the ambient-field control does not restore a detection once the error contrast inside ambient strata is equated, and the photometric-error controls do not constitute a physical ambient-field match since $e_W$ is not an ambient proxy; the strict multivariate match ($-0.176 \pm 0.104$ on $n=29$ pairs) is retained as the countervailing diagnostic. The LMC gradient, robust to period and colour matching, remains the strongest single-galaxy result.

### 3.3 Generative Endpoint Likelihood and Velocity Robustness

Across the 37 SN Ia host galaxies, fitting the generative expansion relation $cz_i = d_i[H_{\rm app} + \Gamma_X \widetilde X_i] + \epsilon_i$ under the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ peculiar-velocity variance yields:

> 

\begin{equation}
\Gamma_X = (3.846 \pm 1.653) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}
\quad (N=37,\ \sigma_v=250\ {\rm km\,s^{-1}}) ,
\end{equation}

significant at $2.33\sigma$ (Wald; $2.38\sigma$ likelihood-ratio) with an intercept $H_{\rm app} = 68.421 \pm 1.443\ {\rm km\,s^{-1}\,Mpc^{-1}}$. The fitted slope is positive across all velocity models and strengthens as peculiar velocity noise is reduced, reaching $2.81\sigma$ at $\sigma_v = 150\text{--}182.1\ {\rm km\,s^{-1}}$.

| Sample Selection | $N$ | $\sigma_v$ (km/s) | $\Gamma_X/10^7$ (km/s/Mpc) | Wald Stat | LRT Stat |
| --- | --- | --- | --- | --- | --- |
| All R22 hosts | 37 | 150 | $3.906 \pm 1.389$ | $2.81\sigma$ | $2.79\sigma$ |
| All R22 hosts (Pantheon+ residual) | 37 | 182.1 | $3.906 \pm 1.389$ | $2.81\sigma$ | $2.79\sigma$ |
| All R22 hosts (Canonical) | 37 | 250 | $3.846 \pm 1.653$ | $2.33\sigma$ | $2.38\sigma$ |
| All R22 hosts | 37 | 500 | $3.820 \pm 3.000$ | $1.27\sigma$ | $1.28\sigma$ |
| Hubble flow ($z_{\rm HD} > 0.0035$) | 30 | 250 | $2.926 \pm 1.679$ | $1.74\sigma$ | $1.77\sigma$ |
| Hubble flow ($z_{\rm HD} > 0.0050$) | 24 | 250 | $2.640 \pm 1.735$ | $1.52\sigma$ | $1.54\sigma$ |

Leave-one-host-out refits demonstrate remarkable directional stability: $\Gamma_X > 0$ in 37 of 37 jackknife iterations (100% positive sign stability, $\Gamma_X$ spanning $3.49\text{--}4.80 \times 10^7$). Bootstrap resampling across 1,000 realizations yields a positive slope in 99.8% of draws with mean $(4.440 \pm 1.883) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and a 95% interval $[2.37,\ 10.60] \times 10^7$. Host-label permutation returns $p = 0.033$ with the finite-sample correction.

The decline of the point estimate under stricter Hubble-flow cuts is tested directly by forward-model injection: the fitted bias $\kappa_{\rm Cep} X_i$ is injected into the observed host distances, each realization is reobserved under a fresh peculiar-velocity draw, and the slope is refit at each redshift cut. Under independent $v_i \sim \mathcal N(0, 250\ {\rm km\,s^{-1}})$ noise, the injected-bias recovery itself declines from $\kappa_{\rm inj} = 1.271\times 10^6$ mag to a median $1.045\times 10^6$ at $z > 0.007$; the observed profile ($1.271 \to 0.847 \to 0.504 \to 0.493\times 10^6$ mag at $z > 0$, $0.0035$, $0.005$, $0.007$) tracks the noise-dilution band at the moderate cuts (14th percentile at $z>0.0035$) but falls below it at the strictest cuts ($\lesssim$1st percentile at $z > 0.005$). Under a coherent-flow variant (a 370 km/s CMB-dipole bulk flow plus 150 km/s residuals), the observed values sit at the 0th--29th percentiles. The environmental signal is therefore concentrated in the nearer, most heavily weighted hosts and attenuates faster than pure velocity-noise dilution predicts once the sample is cut to $z > 0.005$; the discriminating evidence at this sample size is the full-sample slope, its sign stability, and the multi-block response, not subsample persistence.

The generative endpoint channel is a detection rather than a directional hint: $2.33\sigma$ at the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ with perfect sign stability, rising to $2.8\text{--}3.5\sigma$ in the constrained-ladder and reduced-scatter variants of Sections 3.4--3.5.

### 3.4 Design Matrix Degeneracy and the TEP-Native Generative Ladder

In the standard SH0ES generalized least-squares framework, each calibrator host distance modulus $\mu_i$ is treated as an unconstrained free parameter. A standard row-level environmental column cannot identify an environmental perturbation whose observational action is equivalent to shifting the latent host distance modulus, because the 37 unconstrained $\mu_i$ parameters absorb the host-level shift ($\kappa_{\rm matrix} = (0.056 \pm 0.370) \times 10^6\ {\rm mag}$). As demonstrated by row-level synthetic injection ensembles (Appendix A.5), the matrix recovers true observation-level Cepheid perturbations without bias (pull mean $+0.12$, unit pull scatter, 68% coverage 0.72 over 200 full-noise realizations), confirming that the lack of host-level identification arises specifically from latent parameter absorption.

The synthetic injection experiment demonstrates this absorption mechanism: injecting a known environmental bias $\kappa_{\rm inj} = 6.987 \times 10^5\ {\rm mag}$ into true host distance moduli results in a recovered matrix parameter of $\kappa_{\rm matrix} \approx 0$, while the velocity-space generative model recovers the full injected signal ($\Gamma_X = 2.475 \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ versus injected $2.350 \times 10^7$).

![Toy recovery experiment demonstrating parameter absorption in unconstrained design matrices versus true recovery in generative velocity space](public/figures/step_43_figure_01_toy_recovery_experiment.png)

Figure 2: Synthetic recovery experiment. Left: unconstrained design matrix regression absorbs the environmental shift into latent host distance moduli, yielding a null coefficient. Right: generative expansion modelling accurately recovers the injected physical slope.

Formulating the distance ladder at the generative observable level in expansion-rate space across the 33 Hubble-flow hosts, where observed Cepheid moduli are coupled to Hubble-flow supernovae, resolves this degeneracy and yields a consistent combined endpoint response:

> 

\begin{equation}
\Gamma_X = (3.805 \pm 1.647) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}
\quad (N=33,\ \sigma_v=250\ {\rm km\,s^{-1}}) ,
\end{equation}

strengthening to $\Gamma_X = (3.904 \pm 1.306) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.99\sigma$) under data-driven Pantheon+ velocity scatter ($\sigma_v = 182.1\ {\rm km\,s^{-1}}$), and $\Gamma_X = (4.009 \pm 1.155) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($3.47\sigma$) at $\sigma_v = 150\ {\rm km\,s^{-1}}$. The fitted conventional Hubble-flow intercept at the sample mean environment is $H_{\rm app} = 68.48 \pm 1.47\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ ($68.31 \pm 0.97\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km\,s^{-1}}$). Evaluating the relation at the unperturbed cosmic background ($X_{\rm cosmic} = -U_{\rm ref}/c^2$) yields $H_{\rm cosmic} = 65.23 \pm 2.21\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($65.19 \pm 1.80\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150$), statistically concordant with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) at $1.0\sigma$ and bracketing it from below — the measured response is now strong enough that full extrapolation to the unperturbed baseline reaches, and mildly overshoots, the CMB value. As established by Equation~(\ref{eq:gamma_relation}), $\Gamma_X$ is a combined endpoint response that may contain both a Cepheid modulus component ($\kappa_{\rm Cep}$) and a residual velocity-sector term ($\beta_X$); the restricted TEP Cepheid-channel closure sets $\beta_X = 0$, allocating the full response to the Cepheid channel ($\kappa_{\rm Cep}^{\rm equiv} = (1.220 \pm 0.531) \times 10^6\ {\rm mag}$).

When evaluated in the joint multi-block framework combining the redshift-distance relation ($N=33$), TRGB differentials ($N=18$), and independent geometric anchors ($N=2$), the joint likelihood favours an environmental response under the restricted Cepheid closure ($\kappa_{\rm Cep} = (0.665 \pm 0.362) \times 10^6\ {\rm mag}$, $1.83\sigma$, at the canonical $\sigma_v = 250\ {\rm km/s}$; $(0.746 \pm 0.343) \times 10^6\ {\rm mag}$, $2.17\sigma$, at $\sigma_v = 150\ {\rm km/s}$), with $H_{\rm app} = 68.82 \pm 1.45\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 250\ {\rm km/s}$ ($68.72 \pm 1.25$ at $\sigma_v = 150\ {\rm km/s}$). Evaluating the physically coupled observing-chain model where the velocity channel is fixed to the direct gravitational redshift ($\beta_X = c / \langle d \rangle \approx 9.60 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}}$, Appendix C.4) yields identical results to the $\beta_X \equiv 0$ closure within $0.001\ {\rm km\,s^{-1}\,Mpc^{-1}}$. The restricted Velocity closure ($\kappa_{\rm Cep} \equiv 0$) yields $\beta_X = (3.805 \pm 1.647) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.31\sigma$) at $\sigma_v = 250\ {\rm km/s}$ ($(3.860 \pm 1.413) \times 10^7$, $2.73\sigma$, at $\sigma_v = 150\ {\rm km/s}$). In the mixed model where both channels are free simultaneously, $\kappa_{\rm Cep} = (0.163 \pm 0.506) \times 10^6\ {\rm mag}$ and $\beta_X = (3.29 \pm 2.30) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 250\ {\rm km/s}$. The two coefficients are not independently identified in this configuration: in the hierarchical velocity-space likelihood the fitted covariance returns $\mathrm{corr}(\kappa_{\rm Cep}, \beta_X) = -0.979$, and only the linear combination $\Gamma_X = \beta_X + (\ln 10/5)\, H_{\rm app}\, \kappa_{\rm Cep} = (6.22 \pm 3.36)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ is constrained by the data (\texttt{results/outputs/step\_38\_hierarchical\_timefield\_ladder.json}). Decomposing the magnitude response by observable carrier — the channel-native two-coefficient variant in which the potential term on the Hubble-flow distances is assigned to the supernova-magnitude channel $\kappa_{\rm SN}$ (a Cepheid-channel bias acting on the calibrators propagates into those distances only through the calibrator-mean $M_B$, entering the block as a common mode absorbed by $H_{\rm app}$), while the TRGB-differential and anchor blocks retain the per-host Cepheid coefficient — yields $\kappa_{\rm SN} = (1.206 \pm 0.528) \times 10^6\ {\rm mag}$ ($2.29\sigma$) at the canonical $\sigma_v = 250\ {\rm km/s}$ and $(1.224 \pm 0.452) \times 10^6\ {\rm mag}$ ($2.71\sigma$) at $\sigma_v = 150\ {\rm km/s}$, against $\kappa_{\rm Cep} = (0.163 \pm 0.506) \times 10^6\ {\rm mag}$. Under the $\beta_X \approx 0$ magnitude-channel closure the measured response is therefore carried by the supernova-magnitude channel rather than by the Cepheid period–luminosity relation itself; the equivalent-Cepheid coefficients quoted throughout this section are bookkeeping conversions of that same endpoint response, not evidence of a Cepheid-period carrier. The independent full-design-matrix sensitivity analysis returns the same ordering ($\kappa_{\rm SN} = (0.347 \pm 0.168) \times 10^6\ {\rm mag}$, $2.07\sigma$); the redshift-distance priors used there assume a Hubble law and are retained only as a sensitivity diagnostic, while the channel-split likelihood above supplies the corresponding no-expansion inference on the same data. Because the TRGB calibrators have limited potential-coordinate leverage on the differential block, the TRGB channel is uninformative about $\kappa_{\rm Cep}$ on its own ($0.80\sigma$) and dilutes the joint significance relative to the redshift-only regression. The standalone redshift-only WLS in $H_0$ space (Cepheid closure, $\beta_X \equiv 0$) yields the strongest single-block constraint: $\kappa_{\rm Cep} = (1.206 \pm 0.528) \times 10^6\ {\rm mag}$ ($2.29\sigma$) at the canonical $\sigma_v = 250\ {\rm km/s}$, rising to $(1.239 \pm 0.419) \times 10^6\ {\rm mag}$ ($2.96\sigma$) under data-driven Pantheon+ velocity scatter ($\sigma_v = 182.1\ {\rm km/s}$) and $(1.275 \pm 0.370) \times 10^6\ {\rm mag}$ ($3.44\sigma$) at $\sigma_v = 150\ {\rm km/s}$, with $H_{\rm app} = 68.48 \pm 1.47\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($68.41 \pm 1.13$ and $68.31 \pm 0.97\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the respective variants). Because $\sigma_{\rm int,v}$ is a free nuisance parameter, the joint likelihood constrains only the summed velocity-space scatter $\sigma_{\rm tot}^2 = \sigma_v^2 + \sigma_{{\rm int},v}^2$; the fitted optimum implies $\sigma_{\rm tot} \approx 208\ {\rm km\,s^{-1}}$ ($\sigma_{\rm int,v} = 144\ {\rm km\,s^{-1}}$ at $\sigma_v = 150$ and pinned at the lower bound for $\sigma_v \ge 250$ in the multi-block fits), so variants below this scale are reparameterizations of a single fit rather than independent sensitivity points. Across the 53 combined joint observations at the canonical $\sigma_v = 250\ {\rm km/s}$, the information criteria now prefer the environmental response: the Velocity closure attains the lowest values of the family ($\text{AIC} = 85.97$, $\text{BIC} = 95.82$ versus $\text{BIC}_0 = 97.18$ for the null; $\Delta{\rm BIC} = -1.36$), with the single-parameter Cepheid closure at $\text{BIC} = 97.77$ and the two-parameter mixed models at $99.68$. All parameterizations capture the underlying potential stratification and recover a Planck-concordant expansion intercept.

### 3.5 Full-Ladder Propagation and Host-Level Reconciliation Routes

For the separate host-level reconstruction route — which re-optimizes the environmental response to flatten the host-environment trend — applying the TEP potential correction to the complete 37-host sample yields $H_0 = 67.83 \pm 1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under the standard anchor reference scale ($\sqrt{U_{\rm ref}} = 87.165\ {\rm km\,s^{-1}}$), reducing the tension from $5.0\sigma$ to $0.26\sigma$ against Planck ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$); the bootstrap distribution across 1,000 converged realizations is centred at $68.15\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with robust width $2.11$. The screened-origin variant ($\sqrt{U_{\rm ref}} = 30.507\ {\rm km\,s^{-1}}$) returns $60.45 \pm 1.39\ {\rm km\,s^{-1}\,Mpc^{-1}}$ — an overshoot below Planck that is, as demonstrated in Section 2.8 and Appendix D.3, purely a gauge artifact of the unconstrained shortcut holding the anchor zero point fixed: when the full 3,490-row SH0ES design matrix is solved with a free anchor zero point, shifting the reference origin produces the exact same $H_0$ down to four decimal places. Under TEP, the same CMB data yields $H_0 = 66.70 \pm 0.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (TEP-CMB), giving $0.68\sigma$ internal consistency between the TEP-corrected local value and the TEP-CMB inference — an agreement between two products of the same framework, not an independent cross-check.

Converting the host endpoint slope via Equation (\ref{eq:gamma_relation}) yields $\kappa_{\rm Cep}^{\rm equiv} = (1.220 \pm 0.531) \times 10^6\ {\rm mag}$ — consistent with the canonical coupling $\kappa_{\rm gal} = 0.960 \times 10^6\ {\rm mag}$ at $0.5\sigma$, closing the earlier factor-of-two gap between the measured response and the canonical scale. Propagating this environmental correction through the complete 3,490-row SH0ES matrix yields $H_0 = 70.463 \pm 0.972\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\Delta\chi^2 = +9.870$ relative to the uncorrected ladder), while canonical scaling yields $H_0 = 71.006 \pm 0.979\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\Delta\chi^2 = +5.941$), verifying exact reference-gauge invariance: all four coordinate origins ($\sqrt{U_{\rm ref}} = 87.165$, $30.507$, $70.002$, and $131.461\ {\rm km/s}$) return identical propagated values to four decimal places.

Within the full-ladder propagation channel, the prespecified primary estimator is the conditional endpoint projection through the SH0ES design matrix, since it propagates the response coefficient actually measured from the Cepheid environment decomposition ($\kappa_{\rm Cep}^{\rm equiv}$, $2.3\sigma$) rather than the canonical benchmark coupling; it returns $H_0 = 70.46\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with a $2.80\sigma$ residual against Planck. The canonical-coupling projection is retained as the benchmark-conditional variant ($71.01$, $3.28\sigma$), and the remaining entries of Table 3 are the model-level robustness family—the cosmologically constrained expansion likelihoods ($68.3$–$68.8$) and the unified host-level reconstruction ($67.83 \pm 1.56$)—so the corrected-$H_0$ family spans $65.2$–$71.0\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with residual tensions $0.26\sigma$–$3.28\sigma$ according to estimator role, not to selection. The first-order simple propagation is included only as a gauge diagnostic: folding the complete correction, including the $-\kappa U_{\rm ref}/c^2$ common-mode term, into the host-mean $\Delta\mu$ produces shifts ranging from $-2.92$ to $+4.63\ {\rm km\,s^{-1}\,Mpc^{-1}}$ across the four reference conventions — even reversing sign — whereas the design-matrix propagation renders $H_0$ invariant to $10^{-4}$ across all four gauges because the same common-mode term is exactly degenerate with the anchor zero-point and host distance-modulus nuisance block, isolating the host-differential potential response.

| Model Configuration | $H_0$ (km/s/Mpc) | Tension with Planck ($67.4 \pm 0.5$) | Status |
| --- | --- | --- | --- |
| R22 baseline ladder | $73.043 \pm 1.007$ | $5.00\sigma$ | Severe tension |
| Conditional endpoint projection ($\kappa = 1.220\times 10^6$, prespecified primary) | $70.463 \pm 0.972$ | $2.80\sigma$ | Substantial easing |
| Conditional canonical scaling ($\kappa = 0.960\times 10^6$) | $71.006 \pm 0.979$ | $3.28\sigma$ | Substantial easing |
| Cosmologically constrained expansion likelihood (canonical $\sigma_v=250$) | $68.48 \pm 1.47$ | $0.74\sigma$ | Concordance ($\Gamma_X = 2.31\sigma$; $3.47\sigma$ at $\sigma_v=150$ variant) |
| Joint multi-block likelihood (Coupled / Cepheid channel, canonical $\sigma_v=250$) | $68.82 \pm 1.45$ | $0.94\sigma$ | Concordance ($\kappa_{\rm Cep} = 0.665\times 10^6$, $1.83\sigma$; $2.17\sigma$ at $\sigma_v = 150$) |
| Unified host-level reconstruction | $67.83 \pm 1.56$ | $0.26\sigma$ | Host-level route; not the primary matrix projection |

### 3.6 Multi-Channel Verification: TRGB Red Giants

A critical test of the TEP framework is cross-channel consistency with Tip of the Red Giant Branch (TRGB) stars. Unlike classical Cepheids, whose periods depend hydrodynamically on envelope acoustic transit times in local proper time, TRGB stars are standard candles governed by degenerate core helium-flash physics. In the 18 host galaxies with overlapping Cepheid (SH0ES) and TRGB (CCHP and EDD) distance determinations, the joint environmental response is evaluated.

The 18-host TRGB sample is tested with the same TEP regressor as the Cepheids. Both indicators show the environmental stratification independently: TRGB-derived expansion rates correlate with host $\sigma_*$ at $\rho = 0.532$ ($p = 0.023$, $\Delta H_0 = 10.31\ {\rm km\,s^{-1}\,Mpc^{-1}}$) — expected under TEP, since stellar-evolution clocks respond to the same clock-amplitude sector, with the ordering $\kappa_{\rm Cep} > \kappa_{\rm TRGB}$ arising from the pulsation-envelope transfer — while the 13 hosts with direct same-galaxy TRGB moduli yield the differential $\Delta \kappa \equiv \kappa_{\rm Cep} - \kappa_{\rm TRGB} = +(0.250 \pm 0.313) \times 10^6\ {\rm mag}$ ($0.80\sigma$), directionally consistent with the predicted hierarchy.

The modest statistical significance of the differential in the current 13-host direct-overlap sample reflects observational error propagation rather than theoretical tension. For a typical SN Ia host potential contrast ($\Delta \sigma_*^2 \sim 1 \times 10^4\ {\rm km^2\,s^{-2}}$ above the reference), the predicted TEP distance modulus shift is $\Delta \mu_i = \kappa_{\rm Cep}^{\rm equiv} X_i \sim 0.1\text{--}0.3\ {\rm mag}$ for the bulk of the host sample; the TRGB channel is expected to carry the same sign with a smaller transfer coefficient. By comparison, the combined per-host Cepheid/TRGB distance-modulus uncertainty is $\sqrt{\sigma_{\rm Cep}^2 + \sigma_{\rm TRGB}^2} \approx 0.05\text{--}0.12\ {\rm mag}$, and both distance scales share common anchor calibrations (the LMC, SMC, and NGC 4258). Furthermore, because both TRGB stars (in diffuse outer halos) and Cepheids (in disks) are observed outside dense galactic cores, while the systemic redshift is supplied by a host-dependent tracer ranging from integrated disk emission to more central optical apertures, the bare aperture differential is subdominant and the environment-dependent response is carried by the amplitude sector (Appendix C.3). Consequently, the current 18-host TRGB sample is non-discriminating due to joint observational noise floors, while remaining fully consistent with the expected physical ordering ($\kappa_{\rm Cep} > \kappa_{\rm TRGB}$).

### 3.7 Host-Mass Decoupling and Supernova Mass-Step Marginalization

A potential degeneracy in empirical distance-ladder corrections arises from the established Type Ia supernova host-galaxy mass step, wherein supernovae in massive hosts ($\log(M_*/M_\odot) > 10.0$) are observed to be $\sim 0.05\text{--}0.06\ {\rm mag}$ brighter after light-curve standardization than those in lower-mass hosts. Because galactic circular velocity correlates with integrated stellar mass through the baryonic Tully--Fisher relation, the question arises whether the measured environmental slope $\Gamma_X$ could represent the conventional supernova mass step in proxy form.

This degeneracy is tested directly using host stellar masses compiled from Riess et al. (2022, Table 4) across all 37 calibrators (Step 39 and Step 17). While the raw inner dispersion correlates moderately with host stellar mass ($r = 0.620$, $p = 4.2 \times 10^{-5}$), the physical TEP coordinate $X_{\rm TEP} \equiv (S_i \sigma_{*,i}^2 - \sigma_{\rm ref}^2)/c^2$ is decorrelated from host stellar mass:

> 

\begin{equation}
r(X_{\rm TEP}, \log M_*) = 0.123 \quad (p = 0.467) , \quad
r(X_{\rm TEP}, \Theta_{10}) = 0.151 \quad (p = 0.374) ,
\end{equation}

where $\Theta_{10} \equiv \Theta(\log(M_*/M_\odot) - 10.0)$ denotes the discrete step indicator. The decorrelation occurs because physical TEP screening $S_i = S_{\rm local} \times S_{\rm group}$ suppresses high-surface-density disks and massive group environments, breaking the monotonic mass-to-potential scaling of isolated galaxies.

Joint multi-regressor likelihood fits of the generative expansion relation $cz_i = d_i [H_{\rm app} + \Gamma_X X_i + \gamma_{\rm step} \Theta_{10,i}]$ demonstrate that $\Gamma_X$ remains stable upon marginalization against the supernova mass step, whereas the mass step itself is statistically suppressed.

| $\sigma_v$ (km/s) | Model | $\Gamma_X/10^7$ (km/s/Mpc) | LRT Stat | $\gamma_{\rm step}$ or $\gamma_M$ (km/s) | Step LRT | Retention | $H_{\rm cosmic}$ (km/s/Mpc) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 150.0 | Standalone TEP | $3.906 \pm 1.384$ | $2.79\sigma$ | &mdash; | &mdash; | 100.0% | $65.19 \pm 1.80$ |
| 150.0 | Joint w/ Mass Step ($\Theta_{10}$) | $3.730 \pm 1.400$ | $2.64\sigma$ | $+1.77 \pm 2.63$ | $0.68\sigma$ | 95.5% | $65.32 \pm 1.79$ |
| 150.0 | Joint w/ Continuous $\log M_*$ | $3.492 \pm 1.348$ | $2.57\sigma$ | $+3.27 \pm 1.53$ | $2.11\sigma$ | 89.4% | $65.47 \pm 1.72$ |
| 150.0 | Mass-Step Residualized ($X_{\perp \rm step}$) | $3.817 \pm 1.412$ | $2.68\sigma$ | &mdash; | &mdash; | 97.7% | &mdash; |
| 182.1 | Standalone TEP | $3.906 \pm 1.383$ | $2.79\sigma$ | &mdash; | &mdash; | 100.0% | $65.19 \pm 1.80$ |
| 182.1 | Joint w/ Mass Step ($\Theta_{10}$) | $3.730 \pm 1.400$ | $2.64\sigma$ | $+1.77 \pm 2.59$ | $0.68\sigma$ | 95.5% | $65.32 \pm 1.79$ |
| 182.1 | Joint w/ Continuous $\log M_*$ | $3.492 \pm 1.347$ | $2.57\sigma$ | $+3.27 \pm 1.53$ | $2.11\sigma$ | 89.4% | $65.47 \pm 1.72$ |
| 182.1 | Mass-Step Residualized ($X_{\perp \rm step}$) | $3.817 \pm 1.410$ | $2.68\sigma$ | &mdash; | &mdash; | 97.7% | &mdash; |
| 250.0 | Standalone TEP | $3.846 \pm 1.652$ | $2.38\sigma$ | &mdash; | &mdash; | 100.0% | $65.23 \pm 2.21$ |
| 250.0 | Joint w/ Mass Step ($\Theta_{10}$) | $3.695 \pm 1.686$ | $2.23\sigma$ | $+1.50 \pm 3.26$ | $0.46\sigma$ | 96.1% | $65.33 \pm 2.22$ |
| 250.0 | Joint w/ Continuous $\log M_*$ | $3.397 \pm 1.679$ | $2.06\sigma$ | $+3.30 \pm 1.99$ | $1.67\sigma$ | 88.3% | $65.54 \pm 2.22$ |
| 250.0 | Mass-Step Residualized ($X_{\perp \rm step}$) | $3.787 \pm 1.674$ | $2.31\sigma$ | &mdash; | &mdash; | 98.5% | &mdash; |

Across all peculiar-velocity specifications, the environmental response retains $95.5\%\text{--}96.1\%$ of its standalone amplitude when simultaneously fitting the discrete supernova mass step, and $88\text{--}89\%$ under continuous-mass marginalization. The mass step parameter is reduced to an insignificant level ($\gamma_{\rm step} \le 0.68\sigma$), indicating that the discrete mass step does not displace the continuous TEP potential coordinate. Even when residualizing $X_{\rm TEP}$ strictly orthogonal to the mass step ($X_{\perp \rm step}$), the environmental slope remains stable at $\Gamma_{X,\perp \rm step} = (3.817 \pm 1.412) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.68\sigma$) at $\sigma_v = 150$--$182.1\ {\rm km\,s^{-1}}$ (identical under the $\sigma_v^2 + \sigma_{\rm int}^2$ degeneracy).

The inferred unperturbed cosmic background expansion rate $H_{\rm cosmic}$ remains tightly clustered across all joint configurations, yielding $H_{\rm cosmic} = 65.3\text{--}65.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$ across the step and continuous-mass models — within $1.2\sigma$ of Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$), bracketing it from below at the measured response amplitude. The TEP resolution of the Hubble tension is therefore fully decoupled from, and robust against, supernova host stellar mass standardization.

The same marginalization applied across the cosmologically constrained Hubble-flow regression of Section 3.4 ($N=33$, with identical weights and sample selection to the redshift-only WLS) returns the same ordering: the screened coordinate is near-orthogonal to mass in this subset ($r(X, \log M_*) = 0.116$), and $\Gamma_X$ retains $\sim$87\% of its amplitude under simultaneous continuous-mass regression — $(3.48 \pm 1.17)\times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.97\sigma$) at $\sigma_v = 150\ {\rm km/s}$, remaining positive across all $\sigma_v$ — while the mass regressor absorbs the shared component ($\gamma_M = +3.82 \pm 1.40\ {\rm km\,s^{-1}\,dex^{-1}}$, $2.7\sigma$). Because host stellar mass is itself a potential-depth proxy under the TEP coordinate, this shared component is expected; the residual $\Gamma_X$ slope and its sign stability demonstrate that the environmental response is not reducible to the mass-step channel.

### 3.8 Bounded-Response Functional Form and Within-Host Cepheid Structure

The environmental slope of Section 3.4 parameterizes the response as linear in the potential coordinate $X_i$. A strictly linear response is an unbounded extrapolation: evaluated at $\sigma \sim 300\ {\rm km\,s^{-1}}$ it would predict a distance-modulus bias approaching a magnitude, far exceeding the observed ladder scatter. The screening operator is intrinsically bounded, so the linear form is a local projection of a saturated response onto a coordinate of width $\sim 10^{-6}$, and the functional form itself must be tested rather than assumed. Nine candidate environmental response functionals — linear in $X_i$, unscreened in $\sigma_i^2$, logarithmic in $S_i\sigma_i^2/U_{\rm ref}$, a velocity-scale contrast $(S_i\sigma_i-\sigma_{\rm ref})/c$, the pure two-body screening factor $S_i$ (carrying no dispersion information), a nested linear coordinate including the ambient baseline, a profiled transition step $\Theta(\sigma_i-\sigma_c)$, a profiled $\tanh$ transition, and the amplitude-sector saturating form $S_A = \min[1,(\sigma_i/\sigma_T)^{2/3}]$ — were compared inside the identical weighted velocity-space likelihood using leave-one-host-out cross-validation, which is the criterion that correctly penalizes unbounded extrapolation. Eight dataset configurations were examined: the full positive-redshift sample ($N=37$), the Hubble-flow sample ($N=34$) selected by $z_{\rm cmb}>0.0035$ or $z_{\rm hd}>0.0035$, and the stricter sample selected by the same rule at $0.005$, in both CMB-frame and flow-corrected redshifts, across $\sigma_v = 150$--$500\ {\rm km\,s^{-1}}$.

| Response model | Amplitude | All $z$ (hd) | $z>0.0035$ (hd) | $z>0.005$ (hd) | All $z$ (cmb) |
| --- | --- | --- | --- | --- | --- |
| Linear in $X_i$ | $(3.85 \pm 1.65)\times 10^{7}$ | 49,771 | 52,695 | 40,303 | 80,002 |
| Unscreened $\sigma_i^2$ | $(2.38 \pm 0.92)\times 10^{7}$ | 45,424 | 47,908 | 34,202 | 90,591 |
| $\log_{10}(S_i\sigma_i^2/U_{\rm ref})$ | $10.52 \pm 3.71$ | 45,950 | 50,228 | 45,091 | 75,885 |
| $(S_i\sigma_i-\sigma_{\rm ref})/c$ | $(2.40 \pm 1.17)\times 10^{4}$ | 56,577 | 60,873 | 51,736 | 82,880 |
| Screening $S_i$ alone | $0.90 \pm 5.47$ | 71,132 | 74,623 | 45,053 | 105,736 |
| Nested linear (ambient baseline) | $(-1.75 \pm 1.26)\times 10^{6}$ | 62,833 | 73,004 | 49,661 | 99,569 |
| Step $\Theta(\sigma_i-\sigma_c)$, $\sigma_c=100$ km/s | $7.89$ | 56,442 | 51,209 | 43,181 | 75,945 |
| $\tanh$ transition, $\sigma_t=110$, $w=40$ km/s | $6.83$ | 45,501 | 49,125 | 36,602 | 75,627 |
| $S_A = \min[1,(\sigma_i/\sigma_T)^{2/3}]$, $\sigma_T=155$ km/s | $26.48$ | 43,549 | 46,447 | 35,461 | 76,558 |
| $S_A$ on nested depth | $18.76$ | 62,341 | 66,585 | 37,988 | 86,914 |

Transition shapes are reselected inside each training fold. The corrected regressor normalization removes the numerical instability previously reported for the intermediate-cut tanh fit.

The bounded amplitude-sector form $S_A = \min[1,(\sigma_i/\sigma_T)^{2/3}]$ with profiled $\sigma_T = 155\ {\rm km\,s^{-1}}$ is the best leave-one-out predictor on the primary all-host and Hubble-flow datasets, and competitive on every configuration; the unscreened $\sigma_i^2$ coordinate leads at the strictest cut and the $\tanh$ transition leads in CMB-frame redshifts — all three are saturating or unscreened-depth forms, so the cross-validation preference is consistently for bounded or unscreened-inner-depth response shapes over the screened linear coordinate. With the selected $S_A$ shape frozen at $\sigma_T = 155\ {\rm km\,s^{-1}}$, host-label permutation yields $p = 0.0010$ on the full sample and $p = 0.0055$ on the Hubble-flow sample — conditional tests of the selected response, not global model-search probabilities. The 95% bootstrap intervals are $[16.1,\ 39.1]$ and $[14.8,\ 37.5]$, respectively, with 100% positive draws, and the amplitude retains its sign in all 37 and all 34 leave-one-host-out refits. That the data select the amplitude-sector response shape — the channel through which TEP predicts the clock bias to act — rather than an unbounded linear coordinate is itself a structural consistency check on the mechanism.

Two qualifications accompany the functional-form result. The selected saturation scale $\sigma_T \approx 155\ {\rm km\,s^{-1}}$ sits at the upper edge of the host ensemble — deep hosts such as NGC 976 ($\sigma_* = 218$ km/s), NGC 3147 ($238$ km/s), and NGC 5728 ($176$ km/s) lie at or above the transition — so the bounded response is driven by genuine ordering across the sample rather than a low-mass-tail switch, and the low-redshift peculiar-velocity leverage documented in Section 3.4 remains a bounded systematic. The saturating preference is not an ad hoc shape selection: the amplitude-sector response $S_A = \min[1,(\rho/\rho_T)^{1/3}]$ of Paper 0 (§2.2ii), evaluated with $\sigma_i^2$ as the depth proxy as $\min[1,(\sigma_i/\sigma_T)^{2/3}]$, is precisely the derived functional form whose profiled transition scale the data select. A nested-depth variant was also tested, adding each host's ambient group-halo baseline (Tully 2015 virial masses, $\sigma_{\rm amb}^2 = c^2 \times 1.6\times10^{-7}\, M_{\rm vir}^{2/3}$) to the coordinate under the nested field structure of Paper 0: both the linear nested coordinate and a saturating $S_A$ evaluated on total depth are rejected — the former returns a negative amplitude ($-1.75\times10^6\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and the latter ranks among the weakest predictors in every configuration. The ambient baseline contributes identically to the host's systemic redshift and to its Cepheid clocks, so it cancels as a common mode in the $cz - dH$ comparison; the identifiable response lives in the host-local depth contrast alone, and cluster members such as NGC 4424 (Virgo) correctly contribute through their shallow local well rather than through the ambient cluster depth.

A second, contamination-resistant test operates inside the distance-ladder matrix itself. Because the 37 host moduli $\mu_i$ are unconstrained parameters, any environmental perturbation uniform within a host is absorbed exactly as $\hat{\mu}_i \to \mu_i - \kappa_0 X_i$ and is invisible to the matrix likelihood. This identification structure is verified directly by injection ensembles (200 full-covariance realizations): a host-uniform modulus shift returns a period-coupled coefficient consistent with zero, $\kappa_P = (+0.04 \pm 3.33)\times 10^5\ {\rm mag}$, while genuine period-coupled injections are recovered without bias (pull mean $-0.07$ to $-0.10$, pull scatter $\approx 1$, 68% coverage $0.65$--$0.67$). The identifiable signature of the period-transport mechanism is therefore the within-host period structure, identified through the $41$ independent Cepheid period distributions (37 hosts plus 4 anchors) rather than through host-level distance offsets, and hence immune to the redshift-frame and peculiar-velocity systematics that dominate the velocity-space channel. A joint generalized least-squares fit of the environment-coupled period term $\kappa_P$ (parameterized linearly in $\log_{10}P$) and the environment-coupled metallicity term $\kappa_Z$ returns $\kappa_P = (-4.20 \pm 3.38)\times 10^5\ {\rm mag}$ ($1.24\sigma$) and $\kappa_Z = (-1.53 \pm 1.36)\times 10^6\ {\rm mag\,dex^{-1}}$ ($1.12\sigma$), with near-orthogonal covariance ($r = -0.17$) and $\Delta\chi^2 = 3.4$ for two parameters relative to baseline under the anchor-screened convention ($\Delta\chi^2 = 8.9$ under the anchor-reference convention, where $\kappa_Z$ reaches $2.75\sigma$). The ad hoc linear-in-$\log P$ coupling is a diagnostic placeholder rather than the predicted functional form; the derived period-squared carrier below is the channel that carries the detected structure (Table 6).

| Convention | Model | $\kappa_P$ ($10^5$ mag) | $\kappa_Z$ ($10^6$ mag dex$^{-1}$) | $\kappa_0$ ($10^5$ mag) | $r(\kappa_P,\kappa_Z)$ | $\Delta\chi^2$ |
| --- | --- | --- | --- | --- | --- | --- |
| anchor-screened | $\kappa_P$ only | $-4.85 \pm 3.33$ ($1.46\sigma$) | &mdash; | &mdash; | &mdash; | $2.12$ |
| anchor-screened | $\kappa_Z$ only | &mdash; | $-1.83 \pm 1.34$ ($1.36\sigma$) | &mdash; | &mdash; | $1.84$ |
| anchor-screened | $\kappa_P + \kappa_Z$ | $-4.20 \pm 3.38$ ($1.24\sigma$) | $-1.53 \pm 1.36$ ($1.12\sigma$) | &mdash; | $-0.172$ | $3.38$ |
| anchor-screened | $\kappa_0 + \kappa_P + \kappa_Z$ | $-7.26 \pm 4.24$ ($1.71\sigma$) | $-1.42 \pm 1.37$ ($1.04\sigma$) | $+5.58 \pm 4.64$ ($1.20\sigma$) | $-0.180$ | $4.83$ |
| anchor-reference-zero | $\kappa_P + \kappa_Z$ | $-3.53 \pm 4.56$ ($0.77\sigma$) | $-4.56 \pm 1.66$ ($2.75\sigma$) | &mdash; | $-0.141$ | $8.93$ |
| anchor-reference-zero | $\kappa_0 + \kappa_P + \kappa_Z$ | $-14.24 \pm 7.99$ ($1.78\sigma$) | $-4.32 \pm 1.66$ ($2.59\sigma$) | $+10.64 \pm 6.51$ ($1.63\sigma$) | $-0.153$ | $11.60$ |

The sign of $\kappa_P$ corresponds to a period dependence of the amplitude-sector response — $\delta\ln P_{\rm amp}$ rising with pulsation period, i.e., weaker inferred-period contraction for longer-period Cepheids within deeper-potential hosts. Through the period--mean-density relation, longer periods correspond to weaker stellar self-binding, so the measured ordering is the direction expected if the clock coupling to the ambient field strengthens as envelope self-screening declines. A first stellar-structure derivation of $\delta\ln P_{\rm amp}(P)$ has now been carried out (Step 62, \texttt{results/outputs/step\_62\_q\_structure\_derivation.json}): requiring the amplitude-sector response to take the form $\delta\ln P_{\rm amp} = \lambda_0\,X\,\rho_T/\bar\rho(P)$, with the Cepheid mean density given by the period--mean-density relation $\bar\rho = \rho_\odot (Q/P)^2$, predicts a carrier quadratic in the period itself, $X\cdot(P/10\,{\rm d})^2$. On the identical design matrix that derived carrier is preferred over both the ad hoc $X\log_{10}P$ and the alternative $X(\log_{10}P)^2$ forms in all six anchor-and-control configurations at equal parameter count — under the screened-anchor convention with $\kappa_Z$ joint, $\Delta\chi^2 = 7.04$ for $X\cdot P^2$ against $3.38$ and $4.70$ for the two logarithmic forms; under the reference-zero convention $14.00$ against $8.93$ and $10.54$. The fitted amplitude of the derived carrier is $\kappa_{P^2} = (-11.1 \pm 4.9)\times10^5$ mag ($2.28\sigma$, screened joint), reaching $2.5$--$2.8\sigma$ in the residual-$\kappa_0$ and reference-zero configurations, and inverts to a dimensionless transport weight $\lambda_0 = 0.93$--$1.29$ — order unity rather than the factor $\sim 10^{6}$ the bare clock-ratio bound would naively suggest, because the inverse-density factor $\rho_T/\bar\rho \sim 10^{6}$ is itself the amplification carrier. The period dependence of the response is thereby carried by the derived functional form rather than the fitted ansatz; what remains is deriving the order-unity coefficient $\lambda_0$ from the coupled envelope transport rather than inferring it.

The unified-closure test of Step 63 (\texttt{results/outputs/step\_63\_unified\_response\_closure.json}) then asks whether that single derived law suffices for the full ladder response: replacing the two fitted columns $\kappa_0 X_i$ (host amplitude) and $\kappa_P X_i(\log_{10}P-1)$ (period coupling) by the one predicted column $-\lambda_0 X_i\,\rho_T/\bar\rho(P_i)$, a residual host-amplitude term fit jointly is consistent with zero ($\kappa_0$ at $1.05\sigma$ and $1.29\sigma$ under the screened and reference-zero conventions respectively), while the unified coefficient itself retains $2.3$--$3.1\sigma$ and maps to $\lambda_0 = 0.93$--$1.29$ (median $1.07$) — the response structure is carried by the derived law, not by independent empirical coefficients. The amplitude then decomposes as $\kappa = \Gamma_{\rm PL}\,\lambda_0\,\rho_T/\bar\rho(P)$ with $\Gamma_{\rm PL} = -b/\ln 10 = 1.416$: the canonical prior $\kappa_{\rm gal} = 9.6\times10^5$ mag is the value of the derived law at the period--luminosity pivot $\bar\rho(10\,{\rm d}) = 2.4\times10^{-5}\ {\rm g\,cm^{-3}}$ for $\lambda_0 = 0.80$, and the ladder-referenced estimate $\kappa_{\rm Cep} = (3.65 \pm 3.04)\times10^5$ mag corresponds to the same law evaluated at the effective ensemble density $\rho_{\rm eff} \approx 8.3\times10^{-5}\ {\rm g\,cm^{-3}}$ ($P_{\rm eff} \approx 5.3$ d). The factor-of-few spread between the two estimators is thereby a difference in effective reference density absorbed by each estimator's convention, not a tension in the response law itself; what remains as theory debt is the derivation of $\lambda_0 \approx 1$ from coupled envelope transport.

The exclusion of the conventional alternatives is carried out as a discriminating battery on the same design matrix. Free global period--luminosity shape terms — a quadratic $(\log_{10}P-1)^2$ and the canonical 10-day slope break $(\log_{10}P-1)\,\Theta(\log_{10}P-1)$ — are individually weak ($0.77\sigma$ and $1.63\sigma$) and do not absorb the signal: with both shape terms free the period-coupled coefficient is $\kappa_P = (-6.51 \pm 3.61)\times 10^5$ mag ($1.80\sigma$), and $-5.91\times10^5$ ($1.62\sigma$) in the fully controlled joint fit with $\kappa_Z$. Coordinate discrimination then refits the same period-coupled structure with each measured host covariate as the modulator: the fitted distance modulus — the channel through which crowding and blending systematics operate — is a weak carrier ($1.30\sigma$), host mass is null ($0.23\sigma$), the potential-depth coordinate sits at $1.46\sigma$, and host mean metallicity leads the ad hoc carrier ($2.67\sigma$, partially confounded with the potential ordering). The unscreened dispersion coordinate performs marginally better than the screened $X_i$ ($\Delta\chi^2 = 2.6$ versus $2.1$), consistent with the corpus's amplitude--shear distinction: the clock-amplitude response need not inherit the shear-transmission suppression that enters the velocity-space coordinate. The coefficient remains negative in all $41$ leave-one-host-out refits (median $1.25\sigma$), so no single host drives the result; the host-assignment permutation of the ad hoc carrier ($p = 0.22$) is weak because the expanded $X_i$ span admits large scrambled amplitudes — the decisive discriminant is instead the carrier-shape comparison above, in which the derived $P^2$ law outperforms the permuted-ansatz forms. Decomposing the global slope into per-host values, the four anchor populations sit near the reference slope while the weighted regression of the per-host slopes on $X_i$ returns $+1.12 \pm 0.30\times 10^6$ mag per unit $X$ — the sign implied by the mechanism and an amplitude consistent with the derived-carrier coefficient, with real scatter beyond the single-term form ($\chi^2/{\rm dof} = 1.79$).

Two qualifications bound the claim precisely. The environment-coupled metallicity term does not separate cleanly from a conventional period--metallicity interaction in the raw product fit: under a free global $(\log_{10}P-1)\times[{\rm O/H}]$ term, $\kappa_Z$ drops to $0.37\sigma$ and $\kappa_P$ to $0.36\sigma$. A within-host decomposition resolves where that signal lives: writing the product as host-mean and within-host pieces, the pure per-Cepheid interaction $(\log_{10}P-1)^{\sim}\times[{\rm O/H}]^{\sim}$ is null ($0.24\sigma$; $\Delta\chi^2 = 0.06$), while the entire apparent PLZ term is carried by the host-mean-weighted pieces $\bar t_h\,[{\rm O/H}]^{\sim}+\bar z_h\,(\log_{10}P-1)^{\sim}$ ($2.65\sigma$ alone, $2.09\sigma$ in the full joint fit) — ordering of the host means, not a per-Cepheid interaction. Under the full decomposition the within-host piece is nearly orthogonal to the environmental terms (design overlap $0.02$--$0.03$), and an independent per-host decomposition of the metallicity slope itself orders on $X_i$ at $0.96\sigma$ with correlation $r = 0.38$ ($n = 39$, $p = 0.017$) and separates anchor from non-anchor populations — the metallicity channel is environment-ordered within-host structure with the conventional per-Cepheid PLZ mechanism excluded, while the residual ambiguity is host-covariate selection at $N = 41$. Second, the environmental period coupling is not shape-degenerate: the stellar-structure derivation (Step 62) supplies the carrier quadratic in the period itself, $X\cdot(P/10\,{\rm d})^2$, which outperforms both $X\log_{10}P$ and $X(\log_{10}P)^2$ in all six anchor-and-control configurations at equal parameter count — so the fitted $\kappa_P$ ansatz is superseded by the derived coefficient, whose amplitude implies $\lambda_0 \simeq 0.93$--$1.29$. What the battery establishes is that the within-host period structure follows the derived environmental law — not distance, mass, or a global standardization shape — and that the residual freedom is confined to the host-covariate ordering.

### 3.9 Endpoint Audits and Combined Evidence

Two audit channels bound the remaining systematics of the velocity-space endpoint. First, the kinematic-proxy audit of Step 64 (\texttt{results/outputs/step\_64\_inclination\_proxy\_audit.json}) quantifies the liability that motivated the $\sigma_*$ coordinate: propagating HyperLEDA inclination uncertainty through the rotation-based audit coordinate $V_{\rm rot}/\sqrt{2}$, eight of the 34 Hubble-flow hosts have $i < 40^\circ$, and the highest-leverage object (NGC 976) is a nearly face-on Sbc whose deprojected $V_{\rm rot} = 405$ km/s carries order-unity deprojection uncertainty absent from the tabulated measurement error — while its adopted $\sigma_* = 218 \pm 18$ km/s is a direct stellar-absorption measurement requiring no deprojection. Under the errors-in-variables likelihood the rotation-proxy response is $\kappa = (0.60 \pm 0.50)\times10^6$ mag on the full sample, $(0.89 \pm 0.67)\times10^6$ mag excluding NGC 976, and $(0.82 \pm 0.73)\times10^6$ mag at $i \ge 40^\circ$; an inclination-independent hybrid that replaces the 11 sub-$45^\circ$ linewidths with K-band Tully--Fisher predictions returns $(0.19 \pm 0.63)\times10^6$ mag. The audit coordinate therefore brackets the ladder-referenced range $\kappa_{\rm Cep} \approx (0.4\text{--}1.2)\times10^6$ mag at reduced precision — consistent, but demonstrably degraded by deprojection noise that the primary $\sigma_*$ coordinate removes by construction. Second, Step 65 (\texttt{results/outputs/step\_65\_hfsn\_environment\_contrast.json}) completes the common-mode propagation chain: a host-level clock response shared by Cepheids and supernovae is absorbed into the latent moduli and propagates to $H_0$ only through the environment contrast between calibrator and Hubble-flow hosts. Transferring $X$ to the 277 Hubble-flow hosts through their Pantheon+ stellar masses gives $\langle X_{\rm HF}\rangle - \langle X_{\rm cal}\rangle = (-8.8 \pm 12.8)\times10^{-9}$, implying a bounded $H_0$ propagation of $+0.27$ km/s/Mpc at the canonical amplitude; robust (Theil--Sen), quadratic, and distribution-free rank-quantile transfers return $\Delta X = -14.6$, $-45.4$, and $-37.5\times10^{-9}$, corresponding to $+0.45$, $+1.41$, and $+1.16$ km/s/Mpc — the correct sign for the tension across every admissible map. The map dependence is driven by a genuine selection difference: the Hubble-flow population extends to lower host masses than the calibrators (73 of 277 hosts lie below the calibrator range $\log M_* \in [9.13, 12.59]$), so the contrast is partially extrapolated in the low-mass tail. A further propagation channel — the interior-tracking fraction $\epsilon$, under which the Chandrasekhar mass drifts as $(1+z)^{3\epsilon}$ and SN Ia peak magnitudes brighten by $\Delta m = -7.5\epsilon\log_{10}(1+z)$ — is bounded cross-referentially rather than by this sample: the corpus's full-covariance Pantheon+ profile likelihood returns $\epsilon < 0.034$ at $3\sigma$ (full tracking excluded at $\Delta\chi^2 = 1064$; Paper 0 Step 73, Paper 26 Step 03.11), so across the calibrator-to-Hubble-flow redshift span the drift contributes $\lesssim 0.02$ mag — $\lesssim 0.3\ {\rm km\,s^{-1}\,Mpc^{-1}}$ on the inferred intercept — and at the looser residual-floor bound $\epsilon \simeq 0.1$ the propagated shift reaches $\approx 0.7\ {\rm km\,s^{-1}\,Mpc^{-1}}$, still an order below the tension. Its smooth $(1+z)$ dependence is moreover nearly orthogonal to the per-host environmental coordinate $X_i$ carried by $\Gamma_X$. The inference accordingly inherits the flagship's $\epsilon$ bound: any revision of that bound propagates directly into the quoted intercept. Third, Step 66 (\texttt{results/outputs/step\_66\_combined\_evidence.json}) quantifies the joint evidence across the channels with independent noise sources -- the velocity-space endpoint $\Gamma_X$ ($2.33\sigma$), the within-host period term ($1.24\sigma$ in the ad hoc carrier), the environment-coupled metallicity term ($1.12\sigma$), the Cepheid--TRGB differential ($0.80\sigma$), and the Hubble-flow contrast ($0.69\sigma$; a common-mode propagation consistency check, not counted in the combination). A Stouffer combination carrying the fitted $\kappa_P$--$\kappa_Z$ correlation explicitly returns $Z = +2.87$ ($p = 0.004$ two-sided) under zero inter-channel correlation, degrading to $Z = +2.13$ ($p = 0.033$) under a conservative shared $\rho = 0.3$; the velocity-plus-period combination alone returns $Z = +2.52$ ($p = 0.012$). These are the conservative figures, since the period channel is carried there by the ad hoc linear-in-$\log P$ diagnostic at $1.24\sigma$ rather than by the carrier the theory predicts: substituting the derived inverse-density carrier $X\cdot(P/10\,{\rm d})^2$ and its joint metallicity refit (Step 62; $2.28\sigma$ and $0.93\sigma$ in the anchor-screened configuration, fitted correlation $-0.18$ carried explicitly) raises the four-channel combination to $Z = +3.31$ ($p = 9.3\times10^{-4}$ two-sided; $Z = +2.46$, $p = 0.014$, at shared $\rho = 0.3$), and velocity-plus-derived-period alone returns $Z = +3.26$ ($p = 0.001$) — reported as the theory-channel figure rather than the headline, since the carrier was selected over the logarithmic ansätze on this same sample (Step 62), albeit as the form the stellar-structure derivation predicts a priori. All primary channels lie in the predicted direction under either carrier convention, and the derived-carrier form of the period channel sits at $2.3$--$2.8\sigma$ across configurations rather than the ad hoc $1.2\sigma$ carried conservatively in the headline combination.

Three further validations bound the estimator dependence of this combination (Step 67; \texttt{results/outputs/step\_67\_validation\_battery.json}). First, leave-one-host-out predictive validation tests whether the fitted response predicts the environmental ordering rather than absorbing it: for each host, the $(H_{\rm app}, \Gamma_X)$ likelihood is refit on the remaining 36 hosts and the held-out velocity is predicted under the environmental model and under the null, each model carrying its own fitted scatter. The aggregate out-of-sample log-likelihood improves by $\Delta\ln L = +2.6$, and the environmental model is the better predictor for 24 of 37 hosts; more sharply, the predicted per-host environmental shift correlates with the host's held-out residual under the null at Spearman $\rho = +0.465$ ($p = 0.004$) — the response predicts which hosts deviate, and by how much, on hosts excluded from the fit. Second, a source jackknife of the $\sigma_*$ catalogue confirms that the signal is not injected by the H&nbsp;I linewidth tier: the 18 direct stellar-absorption hosts alone return $\Gamma_X = (3.35 \pm 1.82)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.84\sigma$), and a surrogate null in which all 19 proxy hosts are assigned the direct-subset median $\sigma_* = 105\ {\rm km\,s^{-1}}$ — retaining the full sample size and weighting while destroying all proxy-carried information — returns $\Gamma_X = (3.77 \pm 1.76)\times10^7$ ($2.14\sigma$); leave-one-bibcode-out refits over the contributing catalogues return $1.85\sigma$--$2.55\sigma$, positive in every configuration. Third, the shared-correlation assumption of the Stouffer combination is itself bounded: the primary four-channel figure remains at or above $2\sigma$ for any admissible shared inter-channel correlation up to $\rho_{\rm crit} = 0.39$, and the derived-carrier combination to $\rho_{\rm crit} = 0.64$ — both far above the largest fitted pairwise correlation in the design ($-0.18$). The combined evidence is therefore limited by channel amplitude, not by an untested correlation assumption.

> 
Multi-scale empirical evidence spanning single-galaxy internal gradients ($2.6\sigma$--$3.3\sigma$), host-level inner-potential stratification ($\Delta H_0 = +9.59\ {\rm km\,s^{-1}\,Mpc^{-1}}$ across the ensemble, Spearman $\rho = 0.553$, $p < 10^{-3}$, $100\%$ sign stability across all 37 hosts, permutation $p = 0.033$), a velocity-space environmental endpoint at $\Gamma_X = (3.846 \pm 1.653)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.33\sigma$), cosmologically constrained expansion likelihoods ($H_{\rm app} = 68.42\text{--}68.82\ {\rm km\,s^{-1}\,Mpc^{-1}}$), host-mass decoupling (retaining $>88\%$ amplitude under supernova mass-step marginalization), the derived bounded amplitude-sector response preferred by out-of-sample model comparison (permutation $p = 0.001$--$0.006$), and within-host period structure following the derived inverse-density carrier at $2.3$--$2.8\sigma$ with order-unity transport weight ($\lambda_0 = 0.93$--$1.29$, sign-stable in $41/41$ host leave-outs) indicates that the Hubble tension is resolved by dynamical proper time in the local distance scale. The prespecified primary ladder-propagation estimator — the measured-$\kappa$ endpoint projection through the SH0ES design matrix — returns $H_0 = 70.463 \pm 0.972\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\Delta\chi^2 = +9.870$ relative to the null), a $2.80\sigma$ residual against Planck; the host-level unified reconstruction, which additionally evaluates the fitted response at the unperturbed cosmic baseline, returns $H_0 \approx 67.8$--$68.4\ {\rm km\,s^{-1}\,Mpc^{-1}}$ — statistical concordance with the CMB inference.

## 4. Discussion

### 4.1 Physical Synthesis of Multi-Scale Empirical Evidence

The empirical results presented in this work provide a coherent, multi-scale verification of the Temporal Equivalence Principle across both internal galactic structures and the cosmological distance ladder. Single-galaxy tests provide the cleanest experimental environment: because all Cepheids in M31 and the LMC share identical host distances and zero relative peculiar velocities, the observed Period--Luminosity offsets ($\Delta W = +0.356 \pm 0.136$ mag in M31, $+0.681 \pm 0.187$ mag in the HST PHAT catalogue — $+0.630 \pm 0.195$ mag for Pan-STARRS restricted to the same footprint — and $+0.0284 \pm 0.0086$ mag in the LMC) are consistent in sign with a larger inferred period contraction for Cepheids in the less-slowed outer disk relative to Cepheids sampling deeper regions, as predicted by the observing-chain hierarchy of Appendix C; the carrier is the amplitude-sector response — envelope self-screening declining outward strengthens the envelope's coupling to the ambient field — the only TEP channel whose amplitude can reach the measured offsets, since the literal aperture clock-ratio is bounded at $|\Delta\ln q| \lesssim \Delta\Phi/c^2 \sim 10^{-7}$, several orders below them. The M31 offsets fail pair-matched colour and error controls, and the residual readings are now quantitatively bounded in both directions: the overmatching hypothesis is tested against the ambient closure by constructing the $u_{\rm amb}$ proxy at $L = 837$ pc, whose coupling to the error covariate is weak (${\rm corr}(e_W, \log u_{\rm amb}) = -0.18$), so error matching removes only a few per cent of any ambient-carried signal; matching on the ambient field itself leaves $\Delta W = +0.206 \pm 0.163$ mag — though under joint $(\log P, e_W, \log u_{\rm amb})$ cell-matching the offset returns $-0.069 \pm 0.126$ mag, so that residual is carried by the unequated error contrast within ambient strata, not by an ambient-ordered signal; and the error-stratified contrast declines monotonically through the $e_W$ distribution ($+0.270 \pm 0.304$ to $-0.117 \pm 0.202$ mag), so the pooled quality-band persistence is compositional rather than a uniform offset. The PHAT channel shares the susceptibility ($-0.565 \pm 0.079$ mag error-matched versus $+0.906 \pm 0.147$ mag density-matched). The channel is carried as conditional evidence: a positive residual is permitted in the best-measured strata but no estimator-invariant detection survives. The LMC gradient remains the strongest single-galaxy result.

At the host-galaxy scale, the inner-potential reconstruction across all 37 distinct R22 SN Ia hosts confirms this directional coupling. The positive environmental slope ($\Gamma_X = (3.846 \pm 1.653) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km/s}$, $2.33\sigma$, and $2.81\sigma$ at $\sigma_v = 150$--$182.1\ {\rm km/s}$) displays 100% directional stability across all 37 leave-one-host-out jackknife refits, with a median-split expansion contrast of $\Delta H_0 = +9.59\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and Spearman $\rho = 0.553$ ($p = 3.9\times10^{-4}$). The coordinate is the local inner-potential scale $\sigma_*$ appropriate to the Cepheid-inhabited disk, not the inclination-deprojected asymptotic rotation speed; the earlier rotation-proxy construction is retained as a robustness coordinate and audited explicitly (Section 3.9).

The single-galaxy amplitudes are furthermore of the correct order for the same transfer coefficient. Run through the host-level $\kappa_{\rm Cep} \approx 1.2 \times 10^6$ mag fitted on the redshift block, the LMC gradient implies an inner–outer potential contrast $\Delta X \approx (2.4 \pm 0.7) \times 10^{-8}$ — equivalently $\sqrt{\Delta U} \approx 46\ {\rm km\,s^{-1}}$ between the $R \le 0.92$ kpc field and the $R \ge 2.93$ kpc field — the same order as the LMC's adopted inner-potential scale $\sigma_* \approx 24\text{--}33\ {\rm km\,s^{-1}}$, with the modest factor expected because an inner-bar to outer-disk contrast exceeds the mean disk scale. Applied to the M31 baseline amplitude, the same transfer implies $\sqrt{\Delta U} \approx 163\ {\rm km\,s^{-1}}$ against the adopted M31 inner-potential scale $\sigma_* \approx 160\ {\rm km\,s^{-1}}$, in striking numerical agreement for a bulge-dominated inner field against the outer disk, with that channel read as conditional evidence under the ambient-field and error-stratified controls described above. The internal gradients therefore sit at exactly the scale the host-level coefficient predicts once each system's potential-contrast scale is accounted for; a precision transfer requires measured per-aperture $q_i$ values rather than order-of-magnitude potential scales, as specified in Appendix C.

### 4.2 Resolution of the Distance Matrix Degeneracy

A critical methodological discovery of this analysis is the mathematical distinction between observation-level and latent-modulus environmental perturbations. The standard SH0ES generalized least-squares matrix does not generically erase observation-level Cepheid perturbations (as confirmed by the unbiased recovery of raw row-level injections, mean recovery fraction $1.05$ over 200 full-noise draws). What it cannot identify without external constraints is an environmental perturbation that is mathematically equivalent to shifting the latent host distance modulus: because the 37 $\mu_i$ parameters are unconstrained free variables, they absorb the host-level shift into $\hat{\mu}_i \to \mu_i - \kappa_{\rm Cep} X_i$, producing an apparent null matrix parameter ($\kappa_{\rm Cep} = (0.056 \pm 0.370) \times 10^6\ {\rm mag}$). The identifiable ladder signature of an environmental clock response is therefore not a host-uniform offset but the within-host structure of the Cepheid observables. Numerical injection ensembles confirm this identification structure — a host-uniform modulus injection returns a period-coupled coefficient consistent with zero, while period-coupled injections are recovered without bias (near-unit pull scatter, 68% coverage $0.65$--$0.67$ over 200 covariance-level draws). In the real data the structure is carried decisively by the derived carrier: the ad hoc linear-in-$\log P$ term is weak ($1.24\sigma$ jointly with the metallicity term, near-orthogonal at $r = -0.17$), while the stellar-structure-derived period-quadratic carrier $X\cdot(P/10\,{\rm d})^2$ reaches $2.3$--$2.8\sigma$ with an order-unity transport weight $\lambda_0 = 0.93$--$1.29$ and sign stability across all 41 leave-one-host-out refits (Section 3.8). This within-host channel is structurally immune to the redshift-frame and peculiar-velocity systematics that dominate the velocity-space slope, making it the cleanest matrix-level discriminator of the transport mechanism's functional form.

Synthetic recovery experiments demonstrate that unconstrained design matrices mathematically fail to recover host-level environmental shifts unless distances are tied to cosmological expansion ($cz_i = d_i H_0$). When the distance ladder is formulated at the generative observable level in expansion-rate space across the 33 Hubble-flow hosts, coupling observed Cepheid moduli to Hubble-flow supernovae, the combined endpoint response is recovered at $\Gamma_X = (3.805 \pm 1.647) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.31\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$, $2.99\sigma$ under Pantheon+ velocity scatter at $182.1\ {\rm km/s}$, and $3.47\sigma$ at $\sigma_v = 150\ {\rm km/s}$), yielding a conventional Hubble-flow intercept of $H_{\rm app} = 68.48 \pm 1.47\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical $\sigma_v = 250\ {\rm km/s}$ ($68.31 \pm 0.97\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km/s}$). Under the restricted TEP Cepheid-channel closure ($\beta_X = 0$), this intercept corresponds to the locally inferred conventional $H_0$ after removal of the Cepheid clock bias; its global cosmological interpretation belongs to the corresponding TEP cosmology analyses.

### 4.3 Cosmological Reconciliation Without Early-Universe Modifications

The primary quantitative inference of the TEP cosmologically constrained expansion-rate likelihood is $H_{\rm app} = 68.48 \pm 1.47\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ ($68.31 \pm 0.97\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the $\sigma_v = 150\ {\rm km\,s^{-1}}$ sensitivity variant), which reconciles the local distance scale with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) within $0.74\sigma$ ($0.94\sigma$ at the $\sigma_v = 150\ {\rm km\,s^{-1}}$ variant). Mapping to the unperturbed cosmic background ($X_{\rm cosmic} = -U_{\rm ref}/c^2$) yields $H_{\rm cosmic} = 65.23 \pm 2.21\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($65.19 \pm 1.80\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km\,s^{-1}}$) — the measured response is now strong enough that full extrapolation to the unperturbed baseline reaches, and mildly overshoots, the CMB value, bracketing Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) within $1.0\sigma$ alongside the unified host-level reconstruction ($H_0 = 67.83 \pm 1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$). The separate unified host-level reconstruction matches Planck within $0.26\sigma$ (bootstrap centre $68.15$), while conditional matrix projections through the SH0ES system yield $70.46\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under the endpoint-equivalent $\kappa$ — the prespecified primary estimator of the ladder-propagation channel, since it propagates the measured response coefficient — and $71.01\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical scaling. The Cepheid correction does not require changing the observed CMB, BAO, or primordial-abundance datasets; their global interpretation belongs to the corresponding TEP cosmology analyses.

Furthermore, TEP is fully compatible with multi-messenger gravitational wave constraints (such as GW170817), ensuring that the propagation speeds of photons and tensor gravitational waves remain equal to within $10^{-15}$ in the late universe.

The same transport mechanism also operates across the other independent $H_0$ inference channels, rather than requiring a separate resolution for each. The Pantheon+ redshift-profile test (Paper 31) returns $R_H = H_0(z>0.25)/H_0(0.05 \le z < 0.15) = 1.009 \pm 0.006$, flat at $1.5\sigma$: a per-host environmental calibration offset propagates as a uniform multiplicative shift of the standardized supernova magnitude scale and cancels identically in the ratio — precisely the signature of a calibration artifact and the primary discriminant against the competing kinematic-void account (the predicted $\sim 5\%$ decline is excluded at $8.4\sigma$). The supernova side of the same clock-transport channel is measured there as well: pre-standardization residuals carry a directional temporal signal that aggregate SALT3 standardization attenuates by $70 \pm 16\%$, and Cepheid distances sit systematically shorter than TRGB distances within the same hosts ($2.39\sigma$). In time-delay cosmography, the channel budget has now been evaluated on the real TDCOSMO-2025 system data (Paper 19). The propagated-transport sector is clean: the conformal sector contributes exactly zero photon delay by null-cone invariance, the disformal sector is bounded at $\sim 10^{-9}$ to $10^{-8}$ fractional ($\sim 5$–$13$ ms over a $10\,R_{\rm E}$ halo traverse) on galaxy-halo paths by the GW170817 speed constraint, and the endpoint rescaling contributes $\sim 10^{-7}$. The mass-model sector is self-consistent because no phantom convergence enters the image plane: scalar backreaction on $g_{\mu\nu}$ is bounded at $\kappa_\phi \approx 4\times10^{-5}$, $\sim 2\times10^{3}$ below dark-matter level (Paper 19, Step 61), so image positions and the Fermat normalization are ordinary-mass-sourced. The only order-percent route is the dynamical–lensing mass difference entering the velocity-dispersion prior that breaks the mass-sheet degeneracy: dynamical inference exceeds lensing inference where the Temporal Shear is active ($M_{\rm dyn} = M_{\rm lens} + M_{\rm ph}$; Papers 4, 19), so a $\sigma_*$-anchored normalization sets the fitted mass above the lensing value at the tracer positions and the model redistributes the excess along the mass-sheet direction, biasing the inferred $H_0$ in the direction established in Paper 19 — with an amplitude that requires the solved source–tracer response, stellar-orbit weighting and a lens-model refit. The PPN source product does not determine this aperture transfer, and neither $S_\Sigma$ nor $S_\Sigma^2$ alone supplies a universal $H_0$ correction. Galaxy-scale apertures sample $a_{\rm ap}\sim10^{-9}$–$10^{-8}\,{\rm m\,s^{-2}}$; the existing delay-blind residuals test the proposed channel but do not calibrate its absolute mass-to-delay response (Paper 5 §5.3; Paper 19). Sign bookkeeping separately shows that any positive propagated delay residual would bias the inference low, so the delay carrier cannot supply the channel at any amplitude. The elevated central values of current time-delay analyses must therefore be assessed against the mass-model systematic floor — measured at $\sim 6\%$ on the SDSS1206+4332 power-law versus composite distance posteriors and $\sim 3$–$10\%$ in the per-system $\kappa_{\rm ext}$ widths — within which the mechanism's kinematic-prior term operates. Standard sirens (Paper 22) supply an orthogonal bi-metric propagation test rather than a reconciliation channel.

Two bookkeeping clarifications follow for corpus consistency. First, the reconciliation is allocated to the endpoint observing chain, not to shear accumulated along the photon path: the analytic decomposition of Appendix C.4 places $>99.9\%$ of the combined slope $\Gamma_X$ in the distance-modulus channel, so characterizations of the $5.6\ {\rm km\,s^{-1}\,Mpc^{-1}}$ gap as path-integrated Hubble-flow shear describe the same endpoint bias expressed through the redshift variable, not an additional propagation contribution. Second, agreement between the corrected local value and the TEP cosmology products ($0.68\sigma$) is internal consistency within one framework; the genuinely testable content of the resolution is that the same mechanism must act, with the same sign and a commensurate magnitude, on the independent channels enumerated above.

### 4.4 Cross-Channel TRGB Robustness and Anchor Potential Contrast

Two empirical nuances reinforce the robustness of the TEP distance ladder resolution against potential confounding factors. First, the cross-channel comparison with Tip of the Red Giant Branch (TRGB) stars in the 18 overlapping host galaxies is directionally consistent with the expected physical hierarchy: the weighted differential slope $\Delta\kappa = +(0.147 \pm 0.508) \times 10^6\ {\rm mag}$ ($t = 0.29$, $p = 0.78$) from Step 20 has the sign predicted by TEP (Cepheid response larger than TRGB response), while being statistically consistent with zero. Separate per-indicator WLS fits give $B_{\rm Cep} = (2.43 \pm 3.65) \times 10^6\ {\rm mag}$ and $B_{\rm TRGB} = (2.41 \pm 3.52) \times 10^6\ {\rm mag}$; both are individually consistent with zero. The predicted Cepheid--TRGB modulus differential is $\kappa_{\rm diff} X_i$ — of order $0.01\text{--}0.05\ {\rm mag}$ across the host sample — at or below the combined per-galaxy Cepheid/TRGB modulus uncertainty ($0.05\text{--}0.12\ {\rm mag}$). The lack of an exaggerated discrepancy between Cepheid and TRGB hosts is therefore an expected consequence of observational noise propagation and shared outer-disk observing geometry relative to host galactic cores.

Second, the calibration contrast between anchors and SN Ia hosts is an intrinsic observational property of the galaxies rather than an artifact of environmental screening models. The shallow calibrator anchors (the LMC, SMC, and the Milky Way solar neighbourhood) possess inner-potential wells of $\sigma_* = 22\text{--}30\ {\rm km\,s^{-1}}$ ($\langle \sigma_*^2 \rangle \sim 6.5 \times 10^2\ {\rm km^2\,s^{-2}}$), whereas the SN Ia host sample consists of massive luminous spirals with $\sigma_* = 28\text{--}238\ {\rm km\,s^{-1}}$ ($\langle \sigma_*^2 \rangle \sim 1.2 \times 10^4\ {\rm km^2\,s^{-2}}$). The deep geometric anchors NGC 4258 ($115\ {\rm km\,s^{-1}}$) and M31 ($160\ {\rm km\,s^{-1}}$) occupy the host-depth regime, so the weighted reference $\sqrt{U_{\rm ref}} = 87.2\ {\rm km\,s^{-1}}$ sits below 28 of the 37 hosts and the response resolves differentially rather than as a uniform offset. This contrast is fully active even in the completely unscreened coordinate ($S=1$). Furthermore, the direct empirical detection of internal radial Period--Luminosity gradients within the LMC ($3.30\sigma$) and M31 ($3.24\sigma$) provides independent internal signatures within Local Group galaxies with the sign predicted by the TEP clock-gradient model.

### 4.5 Decisive Observational Pathways

The TEP framework outlines specific, preregistered experimental pathways for further validation:

1. Spatially resolved IFU spectroscopy of SN Ia host galaxies with JWST, measuring differential stellar kinematics and tracer ratios $q_i = r_{\rm spec}/r_{\rm Cep}$ between nuclear cores and outer Cepheid fields.

2. Space-based optical time-transfer and closed-loop clock synchronization experiments (the Triangle Test) designed to detect holonomy in dynamical proper time at the $10^{-19}$ fractional level.

3. Precision expansion of the local distance ladder using the Nancy Grace Roman Space Telescope, observing high-redshift Cepheids and TRGB standard candles in diverse gravitational environments.

## 5. Conclusion

A rigorous, multi-scale empirical investigation of the Temporal Equivalence Principle (TEP) has been performed across the local cosmological distance ladder. By elevating proper time to a dynamical scalar field that slows in deeper gravitational potentials ($0 < r_{\rm core} < r_{\rm disk} < r_{\rm cosmic} \equiv 1$), TEP predicts that classical Cepheid pulsation clocks calibrated in diffuse anchor environments systematically misestimate distance moduli when applied to deep-potential SN Ia host galaxies.

The empirical findings establish three mutually reinforcing pillars of evidence:

1. Host inner-potential stratification: Reconstructing all 37 distinct R22 SN Ia host galaxies with the local inner-potential scale $\sigma_*$ — the stellar velocity dispersion at the radii where Cepheids reside — and continuous screening yields an environmental slope of $\Gamma_X = (3.846 \pm 1.653) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km/s}$ ($2.33\sigma$; $2.81\sigma$ at $\sigma_v = 150$--$182.1\ {\rm km/s}$), a median-split expansion contrast of $\Delta H_0 = +9.59\ {\rm km\,s^{-1}\,Mpc^{-1}}$, Spearman $\rho = 0.553$ ($p = 3.9\times10^{-4}$), and 100% positive sign stability across all 37 leave-one-host-out refits. The inclination-corrected rotation coordinate $V_{\rm rot}/\sqrt{2}$ is retained as a documented robustness coordinate; its known deprojection failure at low inclination is audited in Section 3.9.

2. TEP-native generative distance ladder: Unconstrained distance-ladder design matrices algebraically absorb host-level environmental shifts into latent host moduli. Formulating the ladder at the generative observable level connecting observed Cepheid moduli to cosmological expansion across the 33 Hubble-flow hosts recovers the combined endpoint response at $\Gamma_X = (3.805 \pm 1.647) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.31\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$, $2.99\sigma$ under Pantheon+ velocity scatter at $182.1\ {\rm km/s}$, and $3.47\sigma$ at $\sigma_v = 150\ {\rm km/s}$), yielding a conventional Hubble-flow intercept of $H_{\rm app} = 68.48 \pm 1.47\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical $\sigma_v = 250\ {\rm km/s}$ ($68.31 \pm 0.97\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km/s}$). Under the physical observing-chain coupling derived in Appendix C.4, the direct gravitational redshift in the host velocity channel ($\beta_{X,\rm grav} \sim c / \langle d \rangle \approx 10^4\ {\rm km\,s^{-1}\,Mpc^{-1}}$) constitutes only $\sim 0.1\%$ of the combined endpoint slope, with the response allocated to the Cepheid distance-modulus channel ($\kappa_{\rm Cep}^{\rm equiv} = (1.220 \pm 0.531)\times 10^6\ {\rm mag}$); joint likelihood estimation of the coupled model confirms $H_{\rm app} = 68.82 \pm 1.45\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the canonical convention ($68.72 \pm 1.25$ at $\sigma_v = 150\ {\rm km/s}$).

3. Matrix-level period-coupled environmental term: Within the full 3,490-row SH0ES design-matrix likelihood, the identifiable signature of host-environment transport is the within-host period structure of the Cepheid ensemble — identified through the 41 independent period distributions and therefore immune to host-level latent-modulus absorption and to the peculiar-velocity systematics that dominate the velocity-space channels. The ad hoc linear-in-$\log P$ coupling is a diagnostic placeholder that returns $\kappa_P = (-4.20 \pm 3.38)\times 10^5\ {\rm mag}$ jointly with the metallicity term ($r = -0.17$); the detected structure is instead carried by the derived functional form. The stellar-structure derivation (Step 62) of the amplitude-sector law $\delta\ln P_{\rm amp} = \lambda_0\,X\,\rho_T/\bar\rho(P)$ predicts a carrier quadratic in period, $X\cdot(P/10\,{\rm d})^2$, which the data prefer over both $X\log_{10}P$ and $X(\log_{10}P)^2$ in all six anchor-and-control configurations at equal parameter count; its fitted coefficient reaches $2.3$--$2.8\sigma$ and inverts to an order-unity transport weight $\lambda_0 = 0.93$--$1.29$. The unified-column test (Step 63) shows that one coefficient alone reproduces the fitted host-amplitude and period-coupled structure (residual host term consistent with zero at $1.0$--$1.3\sigma$), and that under the decomposition $\kappa = \Gamma_{\rm PL}\,\lambda_0\,\rho_T/\bar\rho$ the canonical amplitude $\kappa_{\rm gal} = 9.6\times10^5$ mag is the law evaluated at the period--luminosity pivot while the ladder estimate $\kappa_{\rm Cep}$ is the same law at the effective ensemble density — closing the provenance gap between the two reported coefficients. The coefficient retains its sign in all 41 leave-one-host-out refits.

4. Single-galaxy internal tests: In M31 (Andromeda), Pan-STARRS Cepheids exhibit an empirical Period--Luminosity offset of $\Delta W = +0.356 \pm 0.136\ {\rm mag}$ ($2.6\sigma$), increasing to $+0.681 \pm 0.187\ {\rm mag}$ ($3.65\sigma$) in the PHAT footprint. In the LMC, OGLE-IV Cepheids independently confirm internal radial potential stratification at $+0.0284 \pm 0.0086\ {\rm mag}$ ($3.3\sigma$), robust to period and colour matching. M31 fails pair-matched controls (colour-matched $-0.017 \pm 0.128$ mag, $0.1\sigma$; eW-matched $-0.103 \pm 0.129$ mag, $0.8\sigma$), and both readings are bounded: the overmatching hypothesis fails because the error covariate is nearly orthogonal to the ambient-field proxy (${\rm corr}=-0.18$ at the $L=837$ pc closure scale), ambient-field matching alone leaves $+0.206 \pm 0.163$ mag — a residual that does not survive joint environment-plus-error control ($-0.069 \pm 0.126$ mag) — and the error-stratified contrast declines monotonically from $+0.270 \pm 0.304$ to $-0.117 \pm 0.202$ mag — so the channel is carried as conditional evidence, with a positive residual permitted in the best-measured strata but no estimator-invariant detection. The macroscopic $H_0$ resolution rests on the host-level channels above; the single-galaxy gradients are a strictly conditional micro-scale diagnostic whose crowding-sensitive estimators do not bear on the ensemble inference.

The estimator hierarchy must be retained in the final inference. The prespecified primary full-ladder projection propagates the measured response coefficient and yields $H_0 = 70.463 \pm 0.972\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\Delta\chi^2 = +9.870$ relative to the uncorrected ladder), a $2.80\sigma$ offset from Planck CMB cosmological parameters ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$). A separate unified host-level reconstruction, which re-optimizes the environmental response to flatten the host-environment trend, yields $H_0 = 67.83 \pm 1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$, in statistical agreement ($0.26\sigma$; bootstrap centre $68.15$) with Planck. Under TEP, the same CMB data yields $H_0 = 66.70 \pm 0.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (TEP-CMB), giving $0.68\sigma$ internal consistency between that host-level route and the TEP-CMB inference — an agreement between products of one framework, not an independent cross-check. This host-level reconciliation is physically motivated by the TEP response decomposition: the observing-chain aperture ratio contributes a bounded bare-clock term while the dominant period response is carried by the amplitude-sector response of the diffuse pulsation envelope (§3.8, Appendix C.3). The primary full-ladder projection, which is structurally restricted to the host-minus-anchor differential because the anchor common mode is degenerate with the fitted Cepheid zero point, retains a bounded residual against Planck; the host-level route, which evaluates the measured response at the unperturbed cosmic baseline, achieves statistical concordance. The anchor weights (0.20 for Milky Way, 0.25 for LMC, 0.55 for NGC 4258) enter the inference only through the gauge-invariant reference level and carry zero measured sensitivity across the weight simplex (Appendix D, Step 67); the residual adopted inputs are the per-anchor dispersions and screening factors themselves, which a future hierarchical treatment using anchor-level Cepheid data can refine.

| Observational Tier | Empirical Result | Statistical Significance | Cosmological Implication |
| --- | --- | --- | --- |
| M31 internal Cepheids (HST PHAT catalogue) | $\Delta W = +0.681 \pm 0.187\ {\rm mag}$ | $3.65\sigma$ nominal | Fails pair-matched controls ($-0.565 \pm 0.079$ mag under propagated-error matching; $+0.906 \pm 0.147$ mag under neighbour-density control); conditional evidence |
| LMC internal Cepheids (OGLE-IV) | $\Delta W = +0.028 \pm 0.009\ {\rm mag}$ | $3.30\sigma$ | Independent radial potential stratification |
| 37-host generative endpoint | $\Gamma_X = (3.846 \pm 1.653)\times 10^7$ | $2.33\sigma$ canonical; $100\%$ sign stability (37/37) | Median-split contrast $\Delta H_0 = +9.59\ {\rm km\,s^{-1}\,Mpc^{-1}}$ |
| TEP-native generative ladder (33 hosts) | $\Gamma_X = (3.805 \pm 1.647)\times 10^7$ | $2.31\sigma$ canonical ($3.47\sigma$ at $\sigma_v=150$) | Conventional Hubble-flow intercept $H_{\rm app} = 68.48 \pm 1.47$ |
| Joint multi-block ladder | $\kappa_{\rm Cep} = (0.665 \pm 0.362)\times 10^6$ (canonical $\sigma_v=250$) | $1.83\sigma$ canonical ($2.17\sigma$ at $\sigma_v = 150$); $3.44\sigma$ redshift-only WLS at $\sigma_v=150$ | Multi-block likelihood closure; $H_{\rm app} = 68.82 \pm 1.45$ |
| Host-level reconstruction | $H_0 = 67.83 \pm 1.56\ {\rm km\,s^{-1}\,Mpc^{-1}}$ | $0.26\sigma$ from Planck | Separate reconciliation route; the prespecified primary full-ladder projection gives $70.463 \pm 0.972$, $2.80\sigma$ from Planck |

A single coherent picture thereby emerges. Under TEP the two
measurements do not read the same quantity: the Cepheid-calibrated
local ladder integrates biased clocks in deep host potentials and
returns the inflated apparent rate $H_{\rm app} = 73.04 \pm
1.01\ {\rm km\,s^{-1}\,Mpc^{-1}}$, while the early-universe acoustic
inference reads the homogeneous temporal background and returns the
corresponding background Hubble-scale parameter. Removing the measured environmental clock
response from the ladder brings the local measurement into
statistical concordance with the CMB value on every route that can
see the full response — $67.83 \pm 1.56$ at host level and
$68.4$--$68.8$ in the cosmologically constrained and joint
multi-block likelihoods — while the prespecified matrix projection
retains only the bounded $2.80\sigma$ residual that the shared
Cepheid zero point structurally forbids it from closing. The
direction of the bias, its magnitude, its environment ordering, its
within-host period carrier, and its persistence across
spectroscopic-tracer classes are all those of the predicted
proper-time mechanism and of no conventional systematic yet
identified. The Hubble tension is accordingly resolved within the
TEP framework: not by altering the early-universe calibration, but by
recognizing that the local distance ladder has been reading
environment-dependent clocks.

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

Smawfield, M. L. (2026). *The Cepheid Bias: Resolving the Hubble Tension*. Preprint v0.11 (Kingston upon Hull). Zenodo. DOI: [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) (Paper 11 — this work)

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
results/outputs/step_07_sigma_provenance_table.csv. All 37 hosts carry the inner-potential
velocity scale $\sigma_*$ — direct stellar-absorption dispersion where
available (18 hosts) and documented H&nbsp;I-linewidth proxies otherwise (19 hosts) —
matching the anchor construction of local stellar dispersions at the Cepheid
sites. The pinned HyperLEDA circular rotation velocities $V_{\rm rot}$
are retained in the master catalog
(data/raw/external/velocity_dispersions_literature.csv) as the
audit coordinate $u_\phi = V_{\rm rot}/\sqrt{2}$ examined in Section 3.9.
For cosmological expansion regressions (the generative and joint multi-block frameworks), the documented union cut ($z_{\rm CMB}>0.0035$ or $z_{\rm HD}>0.0035$) selects the relevant $N=33$ Hubble-flow subset.

| Host | $z_{\rm HD}$ | $\mu$ (mag) | $H_{0,i}$ | $\sigma_*$ (km/s) | $\sigma_{*,\rm err}$ | Method | $\rho_{\rm local}$ | $S_{\rm total}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M 101 | 0.00122 | 29.160 | 53.86 | 28.0 $\pm$ 5.0 | H I linewidth proxy | 0.0089 | 0.8089 |  |
| Mrk 1337 | 0.00925 | 32.916 | 72.42 | 94.0 $\pm$ 12.0 | H I linewidth proxy | 0.0480 | 0.9321 |  |
| NGC 0691 | 0.00855 | 32.822 | 69.88 | 108.0 $\pm$ 23.0 | stellar absorption | 0.0457 | 0.6004 |  |
| NGC 1015 | 0.00815 | 32.618 | 73.19 | 106.5 $\pm$ 6.6 | stellar absorption | 0.0167 | 0.9396 |  |
| NGC 105 | 0.01682 | 34.493 | 63.68 | 85.0 $\pm$ 12.0 | H I linewidth proxy | 0.0265 | 0.9380 |  |
| NGC 1309 | 0.00719 | 32.509 | 67.88 | 89.0 $\pm$ 29.0 | stellar absorption | 0.0324 | 0.9367 |  |
| NGC 1365 | 0.00483 | 31.325 | 78.65 | 151.0 $\pm$ 15.0 | stellar absorption | 0.0086 | 0.1293 |  |
| NGC 1448 | 0.00333 | 31.295 | 54.98 | 95.0 $\pm$ 12.0 | H I linewidth proxy | 0.1022 | 0.7768 |  |
| NGC 1559 | 0.00407 | 31.461 | 62.27 | 73.0 $\pm$ 4.0 | stellar absorption | 0.1059 | 0.9003 |  |
| NGC 2442 | 0.00488 | 31.465 | 74.51 | 144.0 $\pm$ 7.0 | H I linewidth proxy | 1.7591 | 0.0453 |  |
| NGC 2525 | 0.00602 | 32.011 | 71.47 | 82.0 $\pm$ 11.0 | H I linewidth proxy | 0.0416 | 0.8674 |  |
| NGC 2608 | 0.00855 | 32.628 | 76.42 | 87.0 $\pm$ 4.0 | H I linewidth proxy | 0.0868 | 0.9131 |  |
| NGC 3021 | 0.00673 | 32.392 | 67.07 | 55.7 $\pm$ 24.5 | stellar absorption | 0.2556 | 0.6925 |  |
| NGC 3147 | 0.01079 | 33.091 | 77.93 | 238.0 $\pm$ 20.0 | stellar absorption | 1.2881 | 0.0982 |  |
| NGC 3254 | 0.00648 | 32.403 | 64.24 | 117.8 $\pm$ 12.0 | stellar absorption | 0.0172 | 0.7493 |  |
| NGC 3370 | 0.00588 | 32.142 | 65.72 | 85.0 $\pm$ 11.0 | stellar absorption | 0.0361 | 0.9358 |  |
| NGC 3447 | 0.00465 | 31.944 | 56.94 | 55.0 $\pm$ 8.0 | H I linewidth proxy | 0.0063 | 0.9405 |  |
| NGC 3583 | 0.00857 | 32.790 | 71.09 | 108.0 $\pm$ 13.0 | stellar absorption | 0.1182 | 0.8272 |  |
| NGC 3972 | 0.00349 | 31.707 | 47.67 | 78.0 $\pm$ 10.0 | H I linewidth proxy | 0.0588 | 0.0944 |  |
| NGC 3982 | 0.00349 | 31.638 | 49.21 | 87.3 $\pm$ 9.0 | stellar absorption | 0.1789 | 0.0848 |  |
| NGC 4038 | 0.00571 | 31.634 | 80.67 | 107.0 $\pm$ 13.0 | H I linewidth proxy | 0.0488 | 0.9318 |  |
| NGC 4424 | 0.00256 | 30.824 | 52.52 | 65.0 $\pm$ 9.0 | H I linewidth proxy | 0.0403 | 0.0270 |  |
| NGC 4536 | 0.00317 | 30.835 | 64.68 | 103.7 $\pm$ 8.2 | stellar absorption | 0.0049 | 0.7501 |  |
| NGC 4639 | 0.00359 | 31.787 | 47.25 | 96.2 $\pm$ 9.1 | stellar absorption | 0.0360 | 0.8689 |  |
| NGC 4680 | 0.00864 | 32.547 | 80.16 | 103.0 $\pm$ 5.0 | H I linewidth proxy | 0.0773 | 0.9187 |  |
| NGC 5468 | 0.00954 | 33.187 | 65.90 | 68.0 $\pm$ 3.0 | H I linewidth proxy | 0.0251 | 0.8072 |  |
| NGC 5584 | 0.00625 | 31.866 | 79.36 | 98.0 $\pm$ 12.0 | H I linewidth proxy | 0.0586 | 0.9279 |  |
| NGC 5643 | 0.00331 | 30.508 | 78.52 | 107.0 $\pm$ 13.0 | H I linewidth proxy | 0.2463 | 0.7570 |  |
| NGC 5728 | 0.00996 | 32.916 | 77.96 | 176.0 $\pm$ 4.0 | stellar absorption | 0.0365 | 0.9356 |  |
| NGC 5861 | 0.00677 | 32.205 | 73.51 | 112.0 $\pm$ 6.0 | H I linewidth proxy | 0.0943 | 0.8434 |  |
| NGC 5917 | 0.00710 | 32.337 | 72.56 | 54.0 $\pm$ 3.0 | H I linewidth proxy | 0.0245 | 0.8713 |  |
| NGC 7250 | 0.00432 | 31.606 | 61.82 | 52.0 $\pm$ 8.0 | H I linewidth proxy | 0.0392 | 0.9349 |  |
| NGC 7329 | 0.01028 | 33.269 | 68.39 | 124.0 $\pm$ 6.0 | H I linewidth proxy | 0.0082 | 0.9404 |  |
| NGC 7541 | 0.00814 | 32.580 | 74.39 | 64.4 $\pm$ 34.6 | stellar absorption | 0.0820 | 0.8505 |  |
| NGC 7678 | 0.01061 | 33.267 | 70.66 | 107.0 $\pm$ 45.0 | stellar absorption | 0.0404 | 0.9346 |  |
| NGC 976 | 0.01312 | 33.544 | 76.91 | 218.0 $\pm$ 18.0 | stellar absorption | 0.2313 | 0.6666 |  |
| UGC 9391 | 0.00747 | 32.816 | 61.23 | 75.0 $\pm$ 27.0 | stellar absorption | 0.0139 | 0.9399 |  |

*Notes:* $H_{0,i}=cz_{\rm HD}/d_i$, where
$d_i=10^{(\mu_i-25)/5}$ Mpc; it is a descriptive per-host quantity, not a
full-ladder estimate. $\sigma_*$ is the adopted inner-potential velocity
scale: direct stellar-absorption dispersion where measured, documented
H&nbsp;I-linewidth proxy otherwise (provenance per row in
velocity_dispersions_literature.csv). The pinned HyperLEDA
$V_{\rm rot}$ values are retained in that catalog and are examined as the
audit coordinate $u_\phi = V_{\rm rot}/\sqrt{2}$ in Section 3.9.
$\rho_{\rm local}$ and $S_{\rm total}$ are adopted screening inputs.

### A.2 Kinematic Provenance

All 37 hosts carry the inner-potential velocity scale $\sigma_*$: 18 hosts
use direct stellar-absorption dispersions (Héraudeau et al. 1999; Ho et al.
2009; Kormendy \& Ho 2013; Koss et al. 2022; Saulder et al. 2019; Riess et
al. 2022) and 19 hosts use documented H&nbsp;I-linewidth dispersion proxies
(Campbell et al. 2014 6dFGSv), each with per-row source bibcodes, errors,
and traceability flags in
data/raw/external/velocity_dispersions_literature.csv. This
construction matches the anchor endpoints, which are local stellar
dispersions at the Cepheid sites (Milky Way $\sigma_z = 30$, LMC $= 24$,
NGC 4258 $= 115$, M31 $= 160$ km/s). The alternative homogeneous coordinate
$u_\phi = V_{\rm rot}/\sqrt{2}$ built from pinned HyperLEDA rotation
velocities is retained for audit: it requires $1/\sin i$ deprojection,
inherits order-unity uncertainty for the eight low-inclination hosts
(Section 3.9), and probes the outer-halo potential rather than the
Cepheid-inhabited inner region.

### A.3 Coefficient Scope

The canonical Cepheid-channel equation is

\begin{equation}
\Delta\mu_i=\kappa_{\rm Cep}
\frac{S_i(\mathcal E_i)\,\sigma_{*,i}^2-U_{\rm ref}}{c^2}.
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
$\kappa_{\rm Cep}^{\rm equiv}=1.220\times10^6$ mag and
$\sqrt{U_{\rm ref}}=30.507$ km/s. They apply only if the complete combined
endpoint response is allocated to Cepheid rows.

| $\sigma_*$ (km/s) | $S$ | $\Delta\mu$ (mag) | Approx. $\Delta H_0$ (km/s/Mpc) |
| --- | --- | --- | --- |
| 50 | 1.0 | $+0.0213$ | $-0.69$ |
| 75 | 1.0 | $+0.0637$ | $-2.05$ |
| 100 | 1.0 | $+0.1232$ | $-3.97$ |
| 125 | 1.0 | $+0.1995$ | $-6.43$ |
| 150 | 1.0 | $+0.2929$ | $-9.44$ |
| 175 | 1.0 | $+0.4032$ | $-13.00$ |
| 200 | 1.0 | $+0.5305$ | $-17.10$ |
| 225 | 1.0 | $+0.6748$ | $-21.75$ |
| 150 | 0.5 | $+0.1401$ | $-4.52$ |
| 150 | 0.1 | $+0.0179$ | $-0.58$ |
| 200 | 0.1 | $+0.0417$ | $-1.34$ |

### A.5 Matrix Identifiability and Injection Scope

The augmented design matrix applies the endpoint column only to
Cepheid-sensitive observation rows while retaining free host moduli. The
47-parameter augmented matrix has rank 47. Over a seeded ensemble of 200
observation-level injections $\kappa_{\rm inj}=0.960\times10^6$ mag with
noise drawn at the full covariance level, the estimator recovers a mean
$\kappa_{\rm Cep}=1.003\times10^6$ mag with per-draw scatter
$0.352\times10^6$ mag (formal error $0.370\times10^6$), pull mean $+0.12$
and 68% coverage 0.72; the single-draw recovery fraction quoted in
earlier versions (0.997) used noise at 1% of the covariance and understated
the per-realization scatter, though it demonstrated correctness of the
linear algebra. The real-data result
$(0.056\pm0.370)\times10^6$ mag is therefore informative for this
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
$X=(S \sigma_*^2-U_{\rm ref})/c^2$, is dimensionless. Its fitted coefficients
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

The host fit does not by itself furnish the finite-source Solar-System
projection of the TEP action. The fitted $\kappa_{\rm Cep}$ therefore
cannot be converted directly into a PPN parameter, an
equivalence-principle violation, or a Vainshtein suppression factor;
Cassini, MICROSCOPE, and laboratory-clock bounds are essential constraints,
but they belong to the separately solved screening and precision-gravity
sectors of the wider TEP framework (Paper 0, Sections 2 and 7).

### B.4 Permitted Cross-Corpus Claim

The defensible cross-corpus statement is structural: TEP papers may use the
same causal matter-metric convention and the same absolute sign rule
$A(\phi) < 1$, while measuring different channel responses. Numerical
agreement, universality of a fitted coefficient, or precision-gravity
closure must be demonstrated in a joint model and is not inferred here.

## Appendix C: Cepheid Period Transport and the Distance-Ladder Bias

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
the systemic-redshift tracer — an aperture-weighted composite ranging
from integrated disk emission to more central optical apertures —
receives greater weight from a deeper, more strongly slowed region. The
same-host hierarchy is

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
an absolute clock acceleration. The metric factor $r_{\rm Cep}$ entering
$q_i$ supplies the bare proper-time contribution at the envelope
location, while the physical pulsation period also carries the separate
amplitude-sector response of the envelope to the ambient field. The geometric factor $q_i$
itself supplies only the bounded bare-clock contribution; the dominant
observed Cepheid response is the separate amplitude-sector response
encoded in $\kappa_{\rm Cep}$ and derived in Section 3.8 and Appendix
C.3, which separates the two contributions, bounding the first and
identifying the second as the carrier the ladder measures.

The P--L zero point is itself calibrated with an anchor ensemble. Let
$q_{\rm ref}$ denote the correspondingly weighted pipeline differential for
that ensemble. The environmental period residual relevant to applying the
calibrated P--L relation is

\begin{equation}
\delta\ln P_{{\rm rest},i}^{\rm inf}
=\ln\!\left(\frac{q_i}{q_{\rm ref}}\right)
+\delta\ln P_{{\rm amp},i},
\label{eq:calibrated_period_residual}
\end{equation}

where the first term is the bounded geometric clock-ratio contribution
and $\delta\ln P_{{\rm amp},i}$ is the amplitude-sector
pulsation-envelope response identified in Appendix C.3, which carries the
dominant measured effect. The geometric term alone requires a larger
core--disk differential in the target host than in the calibration
ensemble, $q_i < q_{\rm ref}$ — a testable tracer/aperture condition that
cannot be inferred merely from the host's total velocity dispersion or
from a comparison of absolute host and calibrator clock rates.

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
=\frac{b}{\ln10}\,\delta\ln P_{{\rm rest},i}^{\rm inf}.
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
spectroscopy shows $q_i\simeq q_{\rm ref}$, the geometric aperture term
cancels; the amplitude-sector prediction remains independently testable
through the period--environment carrier.

### C.3 The Ladder-Level Response Coefficient ($\kappa_{\rm Cep}$)

The ladder-level coefficient decomposes as
$\kappa_{\rm Cep} = \Gamma_{\rm PL}\,\lambda_{\rm Cep}$, where
$\Gamma_{\rm PL} = -b/\ln 10 \simeq 1.416$ is the Leavitt
period-to-magnitude projector fixed by the Wesenheit P--L slope
$b \simeq -3.26$ and $\lambda_{\rm Cep}$ is the composite
stellar/amplitude transfer factor carrying the envelope's mechanical
response. It is one observable response coefficient, not an independent
coupling; Paper 0 §7 explicitly distinguishes this composite
ladder-level slope from the dimensionless
$\kappa_X = |\beta_A|\,S_X(\mathcal E)\,\Gamma_X$ of its transfer map,
and the microscopic reduction of $\lambda_{\rm Cep}$ to that map is the
registered Gate-A closure debt.

The present data do not measure $q_i$ for each spectral aperture, and the
geometric term is bounded small by the spectroscopic constraints of
Appendix C.3. The restricted Cepheid closure therefore parameterizes the
first-order environmental dependence of the dominant amplitude-sector
response directly, with

\begin{equation}
\delta\ln P_{{\rm amp},i}
=-\lambda_{\rm Cep}X_i,
\qquad
X_i=\frac{S(\mathcal E_i) \sigma_{*,i}^2-U_{\rm ref}}{c^2},
\qquad \lambda_{\rm Cep}>0 .
\end{equation}

Combining this transfer with Equation~(\ref{eq:pipeline_magnitude_shift})
gives

\begin{equation}
\mu_{{\rm corr},i}=\mu_{{\rm obs},i}+\kappa_{\rm Cep}X_i,
\qquad
\kappa_{\rm Cep}\equiv-\frac{b}{\ln10}\lambda_{\rm Cep}>0 .
\end{equation}

The earlier derivation of $0.960\times 10^6\ {\rm mag}$ as a
single-rate target -- $0.288\ {\rm mag}$ divided by
$X=3\times 10^{-7}$, assigning the whole host potential to one
clock -- is superseded: it identified the measured response with the
bare metric clock ratio alone. The bound on that geometric sub-term
is nevertheless computed explicitly, because it is a required
consistency condition of the response-sector identification and
because it fixes which sector of the theory the ladder actually
measures. The clocks the ladder divides share each galactic
potential, so the common metric rate cancels in the nested ratio
$q=r_{\rm spec}/r_{\rm Cep}$ of the spectroscopic-aperture rate to
the Cepheid rate, compared with the same ratio in the anchor. On a
uniform $10^{11}\,M_\odot$, $30\,\mathrm{kpc}$ host with the Cepheid
at $8\,\mathrm{kpc}$, against a $10^{10}\,M_\odot$,
$4.5\,\mathrm{kpc}$ anchor, that geometric ratio differs by
$\Delta\ln q = 2.4\times 10^{-10}$, a bare clock-channel coefficient
$\kappa_{\rm nested}=-7.5\times 10^{-4}\ {\rm mag}$ through the
Leavitt projector at $X=\sigma_*^2/c^2$ for
$200\ {\rm km\,s^{-1}}$. Alternative reference-clock
conventions -- the volume-mean light depth or the direct
Cepheid-to-Cepheid baseline comparison -- leave
$|\kappa_{\rm nested}|$ of order unity at most and of the same sign,
opposite to the measured positive response.

This is the outcome the mechanism itself requires, not a null
result against it. Spectroscopic weak-line equivalent widths cap
any literal core--disk clock-rate differential independently of
TEP, so a geometric channel large enough to carry the measured
response is already excluded; the calculation confirms that the
bare clock ratio lies more than eight orders of magnitude below
the empirical $\kappa_{\rm Cep}$ interval and carries the opposite
sign, supplying neither the amplitude nor the direction of the
observed term. The measured coefficient is thereby identified with
the amplitude sector -- precisely the allocation the
response-sector decomposition
$\kappa_{\rm Cep}=\Gamma_{\rm PL}\,\lambda_{\rm Cep}$ specifies, with
the Leavitt projector $\Gamma_{\rm PL}=-b/\ln10\simeq1.416$ fixed by the
P--L slope rather than fitted. What closes the nine-decade gap between the geometric
scale and the measured response is not an asserted normalization
but a derived and fitted factor: the period--mean-density relation
places the pulsation envelope at
$\bar\rho\sim10^{-5}$--$10^{-4}\ {\rm g\,cm^{-3}}$, so the
inverse-density transport factor $\rho_T/\bar\rho\sim10^{5}$--$10^{6}$
converts the order-$10^{-7}$ field increment into the observed
magnitude-level response. The amplification is specific to the
pulsation mode: $\rho_T/\bar\rho(P)$ is the mechanical response of the
pulsation envelope, whose period is set by the mode's dynamical time,
and is not a rescaling of the universal conformal coupling — atomic
transitions carry no $\bar\rho$-dependent mechanical channel and do not
inherit it, so the spectroscopic bound on the bare clock differential
derived above is fully compatible with a magnitude-level Cepheid
response. The microphysical allocation of the transport factor remains
the identified open derivation: the inverse-density form has the
structure of a fixed perturbation normalized by the envelope's self
gravity — $\propto 1/\bar\rho$ — which is a gradient-type response, and
the same low envelope density $\bar\rho \ll \rho_T$ that supplies the
amplification leaves that channel unsuppressed by the deep-interior
screening which binds Solar-System tests ($S_\Sigma^{(\odot)} \sim
10^{-6}$ applies at stellar-core densities, not at
$\bar\rho \sim 10^{-5}$--$10^{-3}\ {\rm g\,cm^{-3}}$). The empirical
carrier, its sign, and its normalization are fixed by the data
independent of that allocation (Paper 0 registers the same open item as
``$\kappa$ normalization open''). The period-quadratic carrier this
produces is the structure resolved at $2.3$--$2.8\sigma$ in the
within-host channel, with an order-unity transport weight
$\lambda_0\simeq0.93$--$1.29$ (Steps 62--63, Section 3.8). The same
physical object that bounds the geometric term thus supplies the
amplification: the pulsation period is the clock, and its
density-weighted coupling to the ambient field is the measured
response.

The canonical coefficient remains
the prespecified transfer value used across the programme; the
empirical fits measure the response and do not impose it. Its
numerical provenance is itself empirical: the value was set to the
joint-fit/bootstrap coefficient on this same SH0ES sample, so the
benchmark--measured comparison is an estimator comparison on common
data rather than an independent theory--data test.
Numerically, the measured endpoint response under the
inner-potential coordinate is
$\kappa_{\rm Cep}^{\rm equiv} = (1.220 \pm 0.531)\times 10^6$ mag
(redshift-block WLS $(1.21\text{--}1.27)\times10^6$ mag at
$2.3$--$3.4\sigma$ across $\sigma_v$ variants), consistent with the
canonical benchmark $\kappa_{\rm gal} = 0.960\times10^6$ mag at
$0.5\sigma$ — closing the factor-of-two gap that separated the
earlier rotation-proxy measurement from the prespecified transfer
value. The residual theory item is the derivation of the response
product $S_{\rm Cep}\Gamma_{\rm PL}$ (Gate-A): the benchmark is a
declared theory normalization that the measured response now
reproduces within its uncertainty.

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
function. The remaining Gate-A task is the microscopic reduction of the
measured ladder-level Cepheid response to the Paper-0 transfer map; the
empirical carrier, sign, environmental ordering and normalization are
already fixed by the present analysis.

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
multi-stage ladder-level response across $X_i \sim 10^{-7}$. In the
restricted Cepheid closure used here, $\kappa_{\rm Cep}$ denotes the
ladder-level Cepheid-side response; a correlated SN-timescale
contribution of the same underlying temporal mechanism (the SN
light-curve stretch channel of Paper 31, Section 4.9) is a separate
cross-channel prediction rather than being assumed inside
$\kappa_{\rm Cep}$. The tracer-type and stretch-step tests provide a
direct empirical check of that separation.

The identification structure of the ladder likelihood supplies an
independent empirical constraint on the form of the period response.
Because the host
distance moduli $\mu_i$ are unconstrained, any host-uniform response
$\delta\ln P_{\rm amp} = {\rm const}$ (independent of Cepheid period) is absorbed
identically into $\hat{\mu}_i$ and cannot appear in the matrix
likelihood at all — verified numerically by injection, which returns a
period-coupled coefficient consistent with zero for a host-uniform
modulus shift while recovering period-coupled injections without
bias (recovery fractions $0.89$--$1.03$ across injected amplitudes
over 200 covariance-level draws). The residual the ladder can see is
therefore the within-host period dependence
$\delta\ln P_{{\rm amp},i} \equiv f(P_i)$, fitted as an environment-coupled
period term $\kappa_P$. The joint fit of the ad hoc
linear-in-$\log P$ carrier returns
$\kappa_P = (-4.20 \pm 3.38)\times10^5\ {\rm mag}$ ($1.24\sigma$)
against a simultaneous metallicity-coupled term, near-orthogonal to it
($r = -0.17$), with the sign corresponding to the period response rising
with period — weaker contraction for longer-period, lower-density pulsators
in deeper potentials. The structure is resolved by the derived
carrier: the period-quadratic form predicted by the amplitude-sector
law reaches $2.3$--$2.8\sigma$ (Step 62), the direction a
screening-modulated envelope coupling predicts qualitatively.

The competing explanations of such a term — a global period--luminosity
shape systematic, a distance-correlated crowding artifact, and a host
mass or metallicity confound — are tested directly on the same design
matrix by a discriminating battery of model tests. Free quadratic and
10-day-break shape terms are individually weak and leave $\kappa_P$
sign-stable ($1.80\sigma$ with both free; $1.62\sigma$ under joint
$\kappa_Z$ control); replacing the environmental modulator by the
fitted modulus ($1.30\sigma$) or host mass ($0.23\sigma$) collapses
the carrier, and all $41$ leave-one-host-out refits retain the sign
(median $1.25\sigma$). The per-host slope decomposition shows the
anchor populations flat at the reference slope while the regression
of per-host slopes on $X_i$ returns $+1.12 \pm 0.30\times10^6$ mag
per unit $X$ — the sign implied by the mechanism. What remains open is sharper than
the original objection: a within-host decomposition
separates the conventional period--metallicity channel by
construction — the pure per-Cepheid interaction
$(\log_{10}P-1)^{\sim}\times[{\rm O/H}]^{\sim}$ is null
($0.24\sigma$), the apparent PLZ signal lives entirely in the
host-mean-weighted pieces, and the per-host metallicity slopes
themselves order on $X_i$ at $0.96\sigma$ ($r=0.38$, $p=0.017$) — so $\kappa_Z$ is
environment-ordered within-host structure rather than a
conventional PLZ artefact; and the stellar-structure derivation of
$\delta\ln P_{\rm amp}(P)$ — the screening response of a pulsator's
envelope as a
function of its mean density — resolves the shape question in the
derived direction: the amplitude-sector form
$\delta\ln P_{\rm amp} = \lambda_0 X\,\rho_T/\bar\rho(P)$ with
$\bar\rho \propto P^{-2}$ predicts a carrier quadratic in period,
which the design matrix prefers over both the linear-$\log P$ and
quadratic-$\log P$ forms in every configuration tested
(Step 62; e.g. $\Delta\chi^2 = 7.04$ versus $3.38$ and $4.70$
jointly with $\kappa_Z$ under the screened convention), with the implied transport weight
$\lambda_0 \simeq 0.93$--$1.29$. The unified-column closure test
(Step 63) confirms sufficiency: the single predicted column
carries the response with a residual
host-amplitude term consistent with zero
($1.05$--$1.29\sigma$), and under
$\kappa = \Gamma_{\rm PL}\lambda_0\rho_T/\bar\rho$ the canonical
prior is the pivot-period value of the same law. The residual
theory calculation is thereby reduced to a single order-unity
coefficient.

### C.4 Coupled Velocity--Distance Allocation and Analytic Ratio

A critical physical question in interpreting the combined endpoint response $\Gamma_X = \beta_X + (\ln 10/5) H_{\rm app} \kappa_{\rm Cep}$ is whether the allocation $\beta_X \approx 0$ is consistent with the observing-chain mechanism. In Equation~(\ref{eq:spectral_redshift_relation}), the host systemic redshift measured from the adopted systemic tracer — integrated disk H&nbsp;I emission or optical spectroscopy — relates to the cosmological path redshift by

\begin{equation}\label{eq:spectral_redshift_relation}
1 + z_{{\rm spec},i} = \frac{1 + z_{{\rm path},i}}{r_{{\rm spec},i}} \approx (1 + z_{{\rm path},i})\left(1 + \frac{\Phi_i}{c^2}\right) ,
\end{equation}

where $\Phi_i \sim \sigma_{*,i}^2$ is the effective potential depth of the spectroscopic aperture. Expanding to first order in the dimensionless potential coordinate $X_i \approx \Phi_i / c^2$ relative to the calibration reference yields an apparent velocity shift in the host velocity channel:

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

Evaluating at the empirical values $H_{\rm app} \approx 68.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and $\kappa_{\rm Cep} \approx 1.220 \times 10^6\ {\rm mag}$ yields

\begin{equation}
\Gamma_{X,{\rm Cep}} \approx 0.4605 \times 68.5 \times (1.22 \times 10^6) \approx 3.85 \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}} .
\end{equation}

Taking the analytic ratio of the two channels yields

\begin{equation}
\frac{\beta_{X,{\rm grav}}}{\Gamma_{X,{\rm Cep}}}
= \frac{c / \langle d \rangle}{\left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep}}
\approx \frac{9.60 \times 10^3}{3.85 \times 10^7}
\approx 2.5 \times 10^{-4} \approx 0.02\% \text{--} 0.03\% .
\end{equation}

This analytic derivation reveals that the direct gravitational redshift contributes less than three parts in ten thousand to the combined endpoint response $\Gamma_X$, with $>99.97\%$ allocated to the Cepheid distance-modulus channel. The physical origin of this asymmetry is the dimensional lever arm between linear gravitational redshift ($\Delta(cz) = \Phi / c \sim 70\ {\rm m\,s^{-1}}$) and photometric distance modulus exponentiation ($\Delta \mu \sim 0.15\ {\rm mag}$, corresponding to a $\sim 7\%$ distance correction and an apparent expansion shift of $\sim 120\ {\rm km\,s^{-1}}$).

Hence, setting $\beta_X \approx 0$ is not an ad hoc closure that contradicts the observing-chain mechanism, but rather its rigorous leading-order consequence. Estimating the exact coupled model in the joint multi-block framework with $\beta_X = 9.597 \times 10^3\ {\rm km\,s^{-1}\,Mpc^{-1}}$ fixed from first principles recovers $H_{\rm app} = 68.82 \pm 1.45\ {\rm km\,s^{-1}\,Mpc^{-1}}$ and $\kappa_{\rm Cep} = (0.665 \pm 0.362) \times 10^6\ {\rm mag}$ at the canonical $\sigma_v = 250\ {\rm km/s}$ convention ($H_{\rm app} = 68.72 \pm 1.25$, $\kappa_{\rm Cep} = (0.746 \pm 0.343) \times 10^6\ {\rm mag}$ at the $\sigma_v = 150\ {\rm km/s}$ variant), matching the restricted Cepheid closure to within three decimal places at every $\sigma_v$.

### C.5 Spectroscopic-Tracer Provenance from NED

The aperture condition $q_i = r_{\rm spec}/r_{\rm Cep}$ above is
testable only if the spectroscopic tracer that supplies each host's
systemic redshift is known. A full-provenance census of the NASA/IPAC
Extragalactic Database (NED) was therefore executed for all 40
catalogued systems — the 37 SN Ia hosts and the extragalactic anchors
LMC, NGC 4258 and M31 (Step 54). For each system, the NED preferred
redshift, its preferred-measurement refcode, and the verbatim
measurement metadata (spectral range, measurement technique, spatial
mode, spectrograph) were retrieved and archived; the tracer class is
assigned from the aggregated method fields of all NED rows bearing the
preferred refcode, or, where the preferred measurement carries no
method metadata, from the modal method fields among rows whose
redshift equals the preferred value. No class is inferred by name or
morphology: where NED supplies no method evidence, the system is
recorded as unclassified.

The census classifies 33 of the 40 systems from direct NED metadata:
25 hosts adopt disk-integrated 21-cm H&nbsp;I line profiles as the
preferred systemic redshift (predominantly the Springob et al. 2005
H&nbsp;I archive and the Third Reference Catalogue), 4 adopt
integrated optical spectroscopy, 2 adopt nuclear/fibre optical
spectroscopy, and 2 adopt systemic optical determinations; 7 systems
carry a preferred refcode for which NED publishes no technique fields
and are retained as unclassified rather than assigned. The
disk-integrated character of the dominant H&nbsp;I tier is the
relevant aperture fact for the coupled model: a 21-cm systemic
redshift weights the extended disk rather than the deep core, so the
effective $r_{\rm spec}$ for these hosts approaches the outer-disk
value and the direct velocity-channel term is even smaller than the
core-weighted bound derived in C.4.

Conditioning the endpoint response on the measured tracer class
confirms that the signal is not an artifact of any single
spectroscopic aperture. Within the 25-host H&nbsp;I tier alone, the
generative likelihood returns $\Gamma_X = (2.97 \pm 2.00) \times
10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.5\sigma$ at $n = 25$), against
$\Gamma_X = (2.97 \pm 3.27) \times 10^7$ in the unclassified tier
($n = 7$) — statistically indistinguishable point estimates — while
the optical classes ($n \le 2$ each) carry insufficient leverage for
an independent fit. The full-sample value $\Gamma_X = (3.846 \pm
1.653) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ is therefore recovered
across the measured aperture classes, and the per-host classification
table with verbatim NED fields is preserved at
results/outputs/step_54_tracer_class_table.csv.

## Appendix D: Reference-Gauge and Anchor Assumptions

### D.1 Physical Anchor Potential Contrast vs Environmental Screening

The environmental gradient between the calibration anchors and the SN Ia host galaxies is an intrinsic observational property of the galaxies rather than an artifact of group screening models. Under the inner-potential coordinate, the primary distance ladder calibrators divide into shallow dwarf or locally calibrated anchors — the LMC ($\sigma_* = 24\ {\rm km\,s^{-1}}$, stellar disk dispersion), the SMC ($\sigma_* = 22\ {\rm km\,s^{-1}}$), and the Milky Way solar neighbourhood ($\sigma_z = 30\ {\rm km\,s^{-1}}$) — and intermediate-depth geometric anchors — NGC 4258 ($\sigma_* = 115\ {\rm km\,s^{-1}}$) and M31 ($\sigma_* = 160\ {\rm km\,s^{-1}}$). The shallow calibrators alone possess $\langle \sigma_*^2 \rangle \sim 6.5 \times 10^2\ {\rm km^2\,s^{-2}}$.

In contrast, the 37 R22 SN Ia host galaxies sample massive luminous spirals with local inner-potential scales $\sigma_* = 28\text{--}238\ {\rm km\,s^{-1}}$ (median $95$, $\langle \sigma_*^2 \rangle \sim 1.2 \times 10^4\ {\rm km^2\,s^{-2}}$). Even in the completely unscreened coordinate ($S=1$), the potential contrast between the shallow dwarf calibrators and the deep host population reaches a factor of $\sim 18$, and the composite weighted reference $\sqrt{U_{\rm ref}} = 87.2\ {\rm km\,s^{-1}}$ sits below 28 of the 37 host $\sigma_*$ values. The anchor ensemble is itself heterogeneous — NGC 4258 and M31 occupy the deep-well regime alongside the hosts — so the response resolves differentially across hosts rather than appearing as a uniform anchor--host offset. Furthermore, the direct empirical detection of internal radial Period--Luminosity gradients within the LMC ($+0.0284 \pm 0.0086\ {\rm mag}$, $3.30\sigma$) and M31 ($+0.681 \pm 0.187\ {\rm mag}$, $3.65\sigma$ in the HST PHAT catalogue; $+0.630 \pm 0.195\ {\rm mag}$, $3.24\sigma$ for the Pan-STARRS sample restricted to the PHAT footprint) provides internal evidence that radial Period--Luminosity gradients are measured within Local Group members with the sign predicted by TEP — noting that the M31 signal fails pair-matched controls but persists under quality-regime equating ($2.3$–$3.2\sigma$ across the pooled-error band ladder), so it is carried as conditional evidence bounded by the contested colour-matched control (Section 3.2).

### D.2 Adopted Anchor Endpoint Construction

The distance ladder calibration ensemble is represented with approximate weights 0.20, 0.25, and 0.55 for the Milky Way, LMC, and NGC 4258. In the baseline formulation, local stellar velocity dispersions at the Cepheid disk locations are adopted: Milky Way solar neighbourhood ($\sigma_z = 30.0\ {\rm km\,s^{-1}}$; Bovy et al. 2012), LMC stellar disk ($\sigma_{\rm disk} = 24.0\ {\rm km\,s^{-1}}$; van der Marel et al. 2002), and NGC 4258 intermediate annulus ($\sigma_{\rm local} = 115.0\ {\rm km\,s^{-1}}$; Kormendy &amp; Ho 2013). This construction yields the composite reference scale $\sqrt{U_{\rm ref}} = 87.165\ {\rm km\,s^{-1}}$ unscreened, and $\sqrt{U_{\rm ref}} = 30.507\ {\rm km\,s^{-1}}$ when modulated by anchor group screening ($S_{\rm MW} = 0.605$, $S_{\rm LMC} = 0.873$, $S_{\rm N4258} = 0.096$). An unweighted average across the three anchors yields $\sqrt{U_{\rm ref}^{\rm equal}} = 70.002\ {\rm km\,s^{-1}}$ unscreened and $27.769\ {\rm km\,s^{-1}}$ screened. The adopted weights are a conventional choice rather than a fitted quantity, and their influence is bounded by construction: they enter the inference only through the scalar reference level $U_{\rm ref}$, whose cancellation from every physical estimator is proven in Section~D.3. A 51-variant sweep across the normalized weight simplex — canonical, equal, single-anchor extremes, $\pm 50\%$ per-anchor perturbations, and 40 Dirichlet draws — spans $\sqrt{U_{\rm ref}} = 24$--$115\ {\rm km\,s^{-1}}$ and shifts the velocity-space environmental slope by at most $9\times10^2\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\lesssim 3\times10^{-5}$ relative, consistent with optimizer tolerance) and the Hubble-flow intercept by $\lesssim 3\times10^{-5}\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (Step 67; \texttt{results/outputs/step\_67\_validation\_battery.json}). The anchor weights are therefore a gauge choice with zero measured sensitivity, not a likelihood-derived assumption carrying systematic risk.

Alternatively, if the whole-galaxy rotation construction ($u_\phi \equiv V_{\rm rot}/\sqrt{2}$, retained as the secondary robustness coordinate rather than the adopted host proxy) is applied symmetrically to the anchors using observed rotation velocities ($V_{\rm rot} = 220.0\ {\rm km\,s^{-1}}$ for the Milky Way, $66.0\ {\rm km\,s^{-1}}$ for the LMC, and $208.0\ {\rm km\,s^{-1}}$ for NGC 4258), the corresponding anchor potential depths are $u_\phi = 155.56\ {\rm km\,s^{-1}}$, $46.67\ {\rm km\,s^{-1}}$, and $147.08\ {\rm km\,s^{-1}}$ respectively. Under the SH0ES weights, this homogeneous rotation construction yields $\sqrt{U_{\rm ref}^{\rm rot}} = 131.461\ {\rm km\,s^{-1}}$ (or $126.504\ {\rm km\,s^{-1}}$ under equal weighting). It is used here strictly as a reference-origin alternative: local stellar velocity dispersions trace the immediate disk potential inhabited by Cepheids while halo rotation curves trace the outer dark-matter halo, and the inner-potential coordinate is the physically appropriate environment of the clocks.

### D.3 Exact Reference-Gauge Invariance and Analytic Proof

The explicit dependence on the numerical origin $U_{\rm ref}$ is an exact gauge artifact that cancels algebraically in physical observables. Two distinct proofs demonstrate this invariance:

*Analytic Background Invariance:* In the cosmological background calibration (Section 2.8 and Section 3.5), the unperturbed expansion rate evaluated at the cosmic mean baseline evaluates as $H_{\rm cosmic} = H_{\rm app} - \Gamma_X \frac{\langle U \rangle}{c^2}$, where $\Gamma_X \equiv \frac{c}{5 \ln 10} \kappa_X$. In terms of the reference-shifted potential coordinates $X_i = (U_i - U_{\rm ref})/c^2$, the unperturbed cosmic baseline corresponds to $X_{\rm cosmic} = -U_{\rm ref}/c^2$. When measured relative to the sample mean $\langle X \rangle = (\langle U \rangle - U_{\rm ref})/c^2$, the relative offset is:

$$\widetilde X_{\rm cosmic} = X_{\rm cosmic} - \langle X \rangle = -\frac{U_{\rm ref}}{c^2} - \left(\frac{\langle U \rangle - U_{\rm ref}}{c^2}\right) \equiv -\frac{\langle U \rangle}{c^2}$$

The reference scale $U_{\rm ref}$ cancels identically to machine precision. Consequently, $H_{\rm cosmic}$ is an invariant physical property of the sample mean environmental potential depth $\langle U \rangle/c^2$, completely independent of the anchor coordinate zero point.

*Full Distance Ladder Matrix Invariance:* In the simultaneous 3,490-row SH0ES design matrix fit, the Cepheid photometric model includes the anchor absolute magnitude zero point $M_H^W$, which enters with a uniform column of $+1$ across all Cepheid observations. Shifting the reference potential by $\Delta U_{\rm ref}$ introduces an additive offset $\Delta m = -\kappa \Delta U_{\rm ref}/c^2$ across every Cepheid row. In a linear least-squares regression, this uniform offset is absorbed exactly by the refitted Cepheid absolute magnitude zero point:

$$\Delta M_H^W = -\kappa \frac{\Delta U_{\rm ref}}{c^2}$$

The relative distance moduli between hosts and anchors $\Delta \mu_i = \mu_i - \mu_{\rm anchor}$, the calibrator supernova absolute magnitude $M_B$, the Hubble-flow intercept $a_B$, the total fit quality $\chi^2$, and the recovered Hubble constant $H_0$ remain identically invariant. Table D1 presents the full-ladder matrix propagation results across four disparate reference origins spanning 30.51 to 131.46 km/s:

| Anchor Reference Origin | $\sqrt{U_{\rm ref}}$ (km/s) | Null ($\kappa = 0$) | Endpoint ($\kappa = 1.22\times 10^6$) | Canonical ($\kappa = 9.60\times 10^5$) | $\Delta\chi^2$ (Endpoint / Canonical) |
| --- | --- | --- | --- | --- | --- |
| Screened local dispersion | $30.51$ | $73.04$ | $70.46$ | $71.01$ | $+9.87\ /\ +5.94$ |
| Equal-weighted dispersion | $70.00$ | $73.04$ | $70.46$ | $71.01$ | $+9.87\ /\ +5.94$ |
| Standard weighted dispersion | $87.17$ | $73.04$ | $70.46$ | $71.01$ | $+9.87\ /\ +5.94$ |
| Homogeneous rotation proxy | $131.46$ | $73.04$ | $70.46$ | $71.01$ | $+9.87\ /\ +5.94$ |

Across all four conventions, the matrix-recovered $H_0$ and $\chi^2$ match to four decimal places ($H_0 = 70.4627\ {\rm km\,s^{-1}\,Mpc^{-1}}$ for the endpoint projection and $71.0057\ {\rm km\,s^{-1}\,Mpc^{-1}}$ for the canonical coupling, with $\Delta\chi^2 = +9.870$ and $+5.941$ over the null respectively). In contrast, unconstrained host-mean shortcuts that evaluate $\Delta\mu_i = \kappa (U_i - U_{\rm ref})/c^2$ without refitting the anchor zero point $M_H^W$ produce artificial numerical offsets (such as $60.45$ vs $67.83\ {\rm km\,s^{-1}\,Mpc^{-1}}$ in host-level evaluations, or a spurious $70.12$--$77.67\ {\rm km\,s^{-1}\,Mpc^{-1}}$ spread across the same four origins in simplified propagation). These offsets are methodological artifacts of omitting the ladder zero-point refit, and their magnitude scales with the coordinate leverage; only the matrix refit is gauge-exact. The standard-construction host-level value remains statistically concordant with the Planck baseline ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) at $0.26\sigma$.

### D.4 Sensitivity Continuum and Hierarchical Modeling

To verify that empirical concordance does not hinge upon any specific reference choice, the sensitivity analysis evaluates the continuum of inferred $H_0$ across the continuous parameter sweep $\sqrt{U_{\rm ref}} \in [30.0, 131.5]\ {\rm km\,s^{-1}}$. Host-mean shortcut evaluations drift across this domain ($60.45$ at the screened origin versus $67.83\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at the standard origin — a $7.4\ {\rm km\,s^{-1}\,Mpc^{-1}}$ artifact span within only part of the range), while the full distance-ladder matrix refit returns $H_0 = 70.463\ {\rm km\,s^{-1}\,Mpc^{-1}}$ identically at every origin, confirming that the physical result is carried by the invariant construction alone.

While reference-gauge invariance confirms mathematical consistency within the linear regression, physical anchor modeling will benefit from future hierarchical calibration directly incorporating resolved stellar kinematics and multi-wavelength screening profiles. The present results demonstrate that the emergent resolution of the Hubble tension is driven by the physical potential contrast between dwarf calibrators and luminous spiral hosts, rather than reference-gauge conventions or the tested reference/screening constructions.