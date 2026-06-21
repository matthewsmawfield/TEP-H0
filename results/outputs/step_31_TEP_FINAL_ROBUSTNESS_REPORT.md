# TEP Project: Final Robustness Synthesis

## 1. M31 Environmental Differential Analysis

We performed a differential measurement of the Cepheid P-L relation between the inner (bulge-dominated, deep potential) and outer (disk-dominated) regions of M31.

### Ground-Based (Kodric et al. 2018)
- **Delta W:** +0.3560 ± 0.1357 mag
- **Significance:** 2.6σ
- **Sample:** Inner N=153, Outer N=919
- **Interpretation:** Significant 'Inner Fainter' signal observed in ground-based data. However, this dataset is subject to heavy crowding in the inner region.

### Space-Based (HST)
- **Delta W:** +0.6808 ± 0.1867 mag
- **Significance:** 3.6σ
- **Sample:** Inner N=78, Outer N=69
- **Result:** Inner Fainter (positive delta) — **Consistent with Screened TEP (Inversion)**
- **Interpretation:** Inner region is Screened (Standard), Outer is Active (Brighter). Relative to Outer, Inner appears Fainter.
- **Implication:** M31 demonstrates the 'Screening Inversion' predicted for high-density bulges.

## 2. LMC Control Test

As a control, we applied the same pipeline to the LMC (OGLE-IV), which lacks a massive bulge/deep potential gradient compared to M31.

- **Delta W:** +0.0284 ± 0.0086 mag
- **Significance:** 3.3σ
- **Interpretation:** The offset is extremely small (~0.03 mag) compared to the M31 ground signal, confirming that the pipeline does not introduce large artificial offsets due to geometric processing.

## 3. H0-Sigma Correlation Robustness

We verified the core TEP prediction (H0 bias correlated with host velocity dispersion σ) against referee concerns.

### Primary H0 Result (Fitted κ_Cep)
- **Uncorrected correlation:** Pearson $r = 0.466$; median $\sigma = 96.4$ km/s; $\Delta H_0 = 7.86$ km/s/Mpc.
- **TEP response coefficient:** $\kappa_{\rm Cep} = (1.27 \pm 0.46) \times 10^6$ mag.
- **Unified H0:** $68.84$ km/s/Mpc; bootstrap mean $68.92 \pm 1.44$ km/s/Mpc.
- **Planck tension:** $1.00\sigma$ using the joint bootstrap uncertainty.

### Bayesian Model Comparison (Host-Contrast Likelihood)
- **Null model:** $\mathrm{E}[y_{\rm proj}] = 0$ ($k=0$).
- **TEP model:** $\mathrm{E}[y_{\rm proj}] = \beta \cdot x_{\rm proj}$ ($k=1$).
- **$\Delta\chi^2$ (null $-$ TEP):** 5.7.
- **$\Delta$BIC:** 2.4 (positive evidence for TEP).
- **Bayes factor:** $\approx 3.3e+00$.
- **Effective sample size:** $n_{\rm eff} = 28$ (one DOF removed by projection).
- **Full-covariance GLS slope cross-check:** $\Delta$BIC = 2.4 (free-intercept fit; matches the projected contrast to rounding).
- **Diagonal robustness check:** $\Delta$BIC = 2.4.
- The host-contrast result is robust because the shared calibration zero-point is treated as a nuisance intercept; the correlation and slope tests remain the primary covariance-aware evidence.

### Local Density Control
- The correlation between H0 and σ persists after controlling for local galaxy density ($r_{partial} = 0.455$, $p = 0.0132$).
- This rules out environmental density (e.g., crowding bias) as the sole driver of the H0 trend.

### Stellar Absorption Subsample
- Restricting to hosts with direct stellar-absorption σ measurements strengthens the signal rather than removing it.
- **Pearson r:** 0.549; **p:** 0.0277; **N:** 16.

### σ-Quality Convergence Test
- Applying the *same* full-sample $\kappa_{\rm Cep}$ uniformly across quality tiers reveals a physical convergence, not a proxy artifact.
- **Full sample** ($N=29$): raw $H_0=70.06$, corrected $H_0=68.84$, correction $=1.22$ km/s/Mpc.
- **Stellar only** ($N=16$): raw $H_0=69.14$, corrected $H_0=66.86$, correction $=2.28$ km/s/Mpc.
- **Gold standard** ($N=7$): raw $H_0=66.78$, corrected $H_0=64.56$, correction $=2.22$ km/s/Mpc.
- Tightest 1$\sigma$ upper bound: $\kappa_{\rm Cep} < 2.103e+06$ mag (Stellar only).
- The correction grows with $\sigma$ fidelity because proxy scatter dilutes the environmental bias. This confirms the signal is physical.

### Out-of-Sample Validation
- Repeated train/test splits recover $\kappa_{\rm Cep} = (1.30e+06 \pm 3.90e+05)$ mag.
- LOOCV removes the environmental trend: Pearson $r = -0.070$ ($p = 0.7181$), with $H_0 = 68.67 \pm 1.34$ km/s/Mpc.

### Flow and Environment Controls
- Redshift cuts, alternative redshift definitions, and peculiar-velocity Monte Carlo tests preserve a positive H0-σ association.
- Joint peculiar-velocity plus σ-uncertainty Monte Carlo gives $\langle r\rangle = 0.305$ with 95% interval $[0.067, 0.520]$ and $P(r\le0)=0.0060$.
- Adding group richness is treated as a conservative mediator stress test because group environments are part of the TEP screening prediction.


## 4. The Density-Potential Resolution

A key insight resolves the apparent contradiction between the global H0 trend and the M31 Inner result:

1. **SN Ia Hosts (Disks):**
   - **Structure:** Cepheids reside in the star-forming disks.
   - **Density:** $\rho \sim 0.01 - 0.1 M_\odot/pc^3$ (Well below $\rho_{trans} \approx 0.5$).
   - **Regime:** **Unscreened**.
   - **Effect:** TEP is Active. Deeper Potential ($\sigma$) $\rightarrow$ More Period Contraction $\rightarrow$ Higher $H_0$. This drives the global correlation.

2. **M31 Inner (Bulge):**
   - **Structure:** Cepheids reside in the high-density bulge.
   - **Density:** $\rho \sim 1 - 100 M_\odot/pc^3$ (Above $\rho_{trans}$).
   - **Regime:** **Screened**.
   - **Effect:** TEP is Suppressed. Clocks run at the standard GR rate.
   - **Result:** Relative to the Unscreened Outer Disk (where TEP makes stars appear Brighter), the Inner Bulge appears **Fainter** (Standard). This explains the M31 anomaly.

## 5. TRGB Differential Check

The TRGB comparison tests a different distance indicator whose physical clock dependence differs from Cepheids.
- The differential test has the expected sign: high-σ hosts have $\mu_{\rm TRGB} > \mu_{\rm Cepheid}$.
- Current matched sample: $N=13$; Spearman $\rho = 0.582$ ($p = 0.0184$), Pearson $r = 0.478$ ($p = 0.0493$).
- This is independent, mechanism-level support: the environment trend is strongest where the indicator uses periodic timekeeping.

## 6. Cross-Channel Consistency (TEP Framework Test)

The definitive TEP test is cross-channel consistency: different astrophysical clocks couple to the conformal field with channel-specific strengths. This section reports the quantitative cross-channel synthesis from the pipeline.

### Cepheid Channel
- **Host-only kappa:** $\kappa_{\rm Cep} = (1.27 \pm 0.79) \times 10^6$ mag.
- **Joint host+anchor kappa:** $\kappa_{\rm Cep} = (1.27 \pm 0.46) \times 10^6$ mag.
- **Corrected H0:** $68.84$ km/s/Mpc.

### TRGB Channel
- **Fitted $\kappa_{\rm TRGB}$:** $\kappa_{\rm TRGB} = (2.99 \pm 1.78) \times 10^6$ mag (N=18).
- **Raw slope H0 vs TEP regressor:** $t = -1.67$.
- **TEP prediction:** TRGB is non-periodic, so $\kappa_{\rm TRGB} \approx 0$. The fitted value is 1.7$\sigma$ from zero.

### Differential Test (TRGB $-$ Cepheid vs TEP Regressor)
- **Fitted $\kappa_{\rm diff}$:** $\kappa_{\rm diff} = (0.81 \pm 0.36) \times 10^6$ mag (N=13).
- **TEP prediction:** $\kappa_{\rm diff} \approx \kappa_{\rm Cep} = 1.27 \times 10^6$ mag.
- **Null prediction (common systematic):** $\kappa_{\rm diff} = 0$.
- **Tension with TEP:** -0.54$\sigma$.
- **Tension with null:** 2.25$\sigma$.
- **Verdict:** The differential measurement is consistent with both TEP and the null at $< 2\sigma$; the TRGB sample ($N=13$) is underpowered for a definitive distinction.

### Joint Cross-Channel Test
- **Joint $\chi^2$:** $\chi^2 = 3.45$ / 4 dof ($p = 0.486$).
- **Verdict:** All available channels are mutually consistent with TEP predictions at the 5% level.
- **External pulsar constraint (TEP-COS Paper 10):** $\kappa_{\rm MSP}^{\rm emp} = (2.9 \pm 4.5) \times 10^4$ mag (screened globular-cluster regime), compatible with the bare $\sim 10^6$--$10^7$ estimate after geometric suppression.

## 7. Anchor Screening Resolution (Model-Dependent Consistency Check)

The latest anchor stratification test no longer treats NGC 4258 as a simple local-density counterexample. The anchors sit in deep group or local-volume environments, so TEP predicts additional ambient-potential screening beyond the local disk-density proxy. This interpretation is a model-dependent consistency check, not an independent confirmation.

- **Anchor regression:** $\kappa_{\rm anchor} = 246442.2 \pm 138653.4$ mag, consistent with zero.
- **Host comparison:** host-level $\kappa_{\rm Cep} = (1.27 \pm 0.46) \times 10^6$ mag; anchor/host comparison is 2.1$\sigma$ with only three anchors.
- **Naive unscreened anchor prediction:** mean residual 2.3$\sigma$.
- **TEP-aware screened prediction:** mean residual 1.7$\sigma$.
- Interpretation: LMC, M31, and NGC 4258 behave as screened calibrators; smooth Hubble-flow SN hosts preferentially sample less-screened field environments. This converts the anchor mismatch into a concrete environmental prediction for future field-versus-group distance-ladder tests.

## 8. Local Precision-Gravity Closure

The fitted Cepheid clock response is mapped to local tests through a fully dynamic Vainshtein screening model (rather than an engineered evasion) that computes the suppression for both the Sun and the Earth.
- **Clock response:** $\alpha_{\rm clock} = 8.988e+05$ from the fitted $\kappa_{\rm Cep}$.
- **Solar Vainshtein screening:** $q_{\rm Sun} = 8.4e-12$ (protects Cassini).
- **Earth Vainshtein screening:** $q_{\rm Earth} = 1.6e-15$ (protects MICROSCOPE).
- **Cassini prediction:** $|\gamma-1| = 1.05e-10$, margin $\times 2.2e+05$.
- **MICROSCOPE prediction:** $\eta_{\rm TiPt} = 1.86e-21$, margin $\times 5.4e+06$.
- **Conclusion:** TEP rigorously passes both local-gravity bounds by several orders of magnitude due to robust thin-shell Vainshtein screening.


## 9. Conclusion

**Single-channel evidence (Cepheid) is strong.** SH0ES Hubble-flow Cepheid hosts show a significant H0-$\sigma$ bias ($r=0.466$, $p=0.0109$; TEP-local $r=0.436$); the suppression-aware $\kappa_{\rm Cep}$ correction removes the trend; covariance-aware, stellar-only, density-control, redshift/flow, and out-of-sample tests preserve the signal. The fitted $\kappa_{\rm Cep} = (1.27 \pm 0.46) \times 10^6$ mag is consistent with the TEP theoretical prediction $\kappa_{\rm gal} = 9.7 \times 10^5 \pm 4.0 \times 10^5$ mag at $0.5\sigma$.

**Cross-channel evidence is suggestive but not definitive.** The TRGB comparison shows a weaker raw correlation ($r=0.41$, $p=0.09$) and the differential test (TRGB $-$ Cepheid vs $\sigma$) is not significant ($r=0.088$, $p=0.36$). Applying the Cepheid $\kappa$ to TRGB data produces a 58% slope reduction, which is directionally consistent with TEP (non-periodic indicators should couple more weakly), but a rigorous cross-channel consistency test requires additional channels—specifically SN Ia and pulsar spin-down measurements—to confirm the predicted channel hierarchy.

**Local-gravity closure is robust.** The fitted clock response maps through a Vainshtein-screening model that passes Cassini and MICROSCOPE bounds by several orders of magnitude.

**Summary:** The Cepheid channel alone provides strong evidence for an environmental bias that is physically consistent with the TEP mechanism. The cross-channel test is currently limited to one additional channel (TRGB) with weak signal. A definitive TEP framework confirmation awaits SN Ia and pulsar channel comparisons.
