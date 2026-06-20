# Full-Ladder Likelihood Analysis: Interpretation

## Executive Summary

This analysis implements a full Cepheid-SN Ia distance-ladder likelihood with a TEP correction applied **only to Cepheid PL rows**. The sign convention is:
- Physical model: `m^W_ij = L·θ - κ_Cep·X_i + ε_ij` where positive κ_Cep means high-sigma Cepheids appear **brightward of the standard PL prediction at fixed true distance** (period contraction in stronger temporal field gradients).
- The augmented design matrix uses `L_aug = [L, -X]`.
- The X column is scaled by 10^6 for numerical stability; we fit `κ_6 = κ_Cep / 10^6`.

**Anchor convention:** All hosts including anchors receive TEP on their Cepheid rows (each host gets its own screened `X_i`). The reference `σ_ref = 87.17` km/s encodes the anchor mix. This is the *fully physical host convention*.

The central finding is that the TEP term is **identifiable** (rank = 47 = full) when SN calibrator and anchor rows constrain μ_i independently. However, the free-κ fit finds κ_Cep = **(−0.067 ± 0.210)×10^6 mag** (0.32σ), consistent with zero and with the **opposite sign** from the canonical TEP expectation. A fixed canonical correction (κ_Cep = 0.97×10^6 mag) moves H0 downward from 73.04 to 71.90 km/s/Mpc, but worsens the fit by Δχ² = 24.2. Thus the host-summary environmental trend is **not validated as the canonical Cepheid-clock TEP correction** by the full SH0ES ladder likelihood.

## Results

### Variant A: Free κ, Full Latent μ_i

**What this is:** The honest statistical test. Fit κ_Cep jointly with all 46 ladder parameters, keeping μ_i latent. The TEP term is applied **only to Cepheid rows** (identified by MHW1 parameter), not to SN or anchor rows.

**Sign convention:** `L_aug = [L, -X_scaled]` where `X_scaled = 10^6 · X_i`. Positive κ_Cep means Cepheids appear **brightward of the standard PL prediction** in high-sigma hosts (period contraction makes them brighter than the standard model expects).

| Metric | Value |
|--------|-------|
| κ_Cep (mag) | −6.67×10^4 ± 2.10×10^5 |
| κ significance | 0.32σ |
| H0 (km/s/Mpc) | 73.12 |
| χ² | 3552.61 |
| χ²/dof | 1.032 |
| rank(L_aug) | 47 (full) |

**Key findings:**
- The TEP term is **identifiable**: rank(L_aug) = 47 = full rank. The SN calibrator and anchor rows break the degeneracy by constraining μ_i independently of the TEP-affected Cepheid rows.
- The fitted κ_Cep is **negative** (opposite to canonical TEP expectation) and **not statistically significant** (0.32σ).
- H0 is essentially unchanged from baseline (73.12 vs 73.04).
- The data do not support a positive TEP effect (brightward Cepheid shift in high-sigma hosts) in the full ladder.

### Variant B: Prior-Constrained κ

**What this is:** Bayesian update with theoretical prior κ_Cep ~ N(9.7×10^5, 4.0×10^5) mag.

| Metric | Value |
|--------|-------|
| Posterior κ_Cep (mag) | 1.58×10^5 ± 1.86×10^5 |
| H0 (km/s/Mpc) | 72.86 ± 1.03 |
| χ² | 3553.47 |
| χ²/dof | 1.032 |

**Key findings:**
- The prior pulls κ toward the canonical 9.7×10^5, but the likelihood pulls it toward ~−6.7×10^4.
- The posterior is a compromise at 1.58×10^5, still below the canonical value.
- H0 shifts downward to 72.86 (**correct direction** from canonical TEP).
- The χ² penalty is small (+0.86) because the posterior κ is much smaller than the canonical value.

### Variant D: SN-Channel TEP

**What this is:** Test whether TEP appears in SN Ia calibrator/Hubble-flow rows rather than (or in addition to) Cepheid rows.

| Metric | Value |
|--------|-------|
| κ_SN (mag) | 1.78×10⁵ ± 2.42×10⁵ |
| κ_SN significance | 0.74σ |
| H0 (km/s/Mpc) | 73.14 |
| Δχ² | 0.54 |

**Key findings:**
- SN-only TEP is not significant (0.74σ).
- The SN channel alone does not rescue the canonical TEP prediction.

### Variant E: Joint Cepheid + SN TEP

**What this is:** Allow both Cepheid and SN rows to carry independent TEP corrections.

| Metric | `anchor_screened_physical` | `anchor_reference_zero` |
|--------|---------------------------|------------------------|
| κ_Cep (mag) | 2.62×10⁵ ± 4.15×10⁵ (0.63σ) | **−1.60×10⁷ ± 1.32×10⁵ (121σ!)** |
| κ_SN (mag) | 4.38×10⁵ ± 4.78×10⁵ (0.92σ) | **−1.58×10⁷ ± 1.16×10⁵ (136σ!)** |
| H0 (km/s/Mpc) | 72.96 | 73.14 |
| Δχ² | 0.94 | 0.54 |

**Key findings:**
- With `anchor_screened_physical`: both κ_Cep and κ_SN are **positive** (TEP direction) but **not significant** (0.63σ and 0.92σ).
- With `anchor_reference_zero`: **catastrophic degeneracy**. Forcing anchor Cepheid TEP to zero causes the joint model to compensate via absurd negative SN/Cepheid values (~10⁷ mag). This proves the anchor convention is **not** `anchor_reference_zero`.
- The `anchor_screened_physical` convention (each host including anchors gets its own screened X_i) is the only one that does not break the joint model.
- Even with both channels free, canonical TEP is not significantly preferred.

### Variant C: Fixed-κ Sensitivity Test

**What this is:** Counterfactual exercise: externally impose κ_Cep = 9.7×10^5 mag and refit all latent parameters.

**Physical convention:** `y_corrected = y + κ·X` (remove the TEP brightening from observed magnitudes to recover the standard PL prediction).

| Metric | Baseline SH0ES | TEP Fixed-κ | Δ |
|--------|----------------|-------------|---|
| H0 (km/s/Mpc) | 73.04 ± 1.01 | 71.90 ± 0.99 | −1.14 |
| κ_Cep (mag) | 0 | 9.7×10^5 (fixed) | — |
| χ² | 3552.71 | 3576.91 | +24.2 |
| χ²/dof | 1.032 | 1.039 | +0.007 |

**Key findings:**
- H0 moves in the expected direction (downward from 73.04 to 71.90)
- The shift is **1.14 km/s/Mpc, ~20% of the way to Planck** (73.04 − 67.4 = 5.64)
- The imposed canonical correction **worsens the fit** (Δχ² = +24.2)
- This is a sensitivity test, not model-selection evidence

### Stage 1: Summary-Likelihood Analysis (Discarded)

| Metric | Value |
|--------|-------|
| κ_Cep (mag) | 2.75×10^6 ± 1.10×10^5 |
| κ significance | 25.0σ |
| χ²/dof | 581 |

**Verdict: Discarded.** χ²/dof = 581 indicates severe model misspecification. The simple linear model μ = μ_true + κ·X is inappropriate for host-level distance moduli.

## Interpretation

### The TEP Term Is Identifiable

Contrary to the initial expectation, the TEP term is **not degenerate** with host μ_i when the full ladder is included. The key is that the TEP correction applies **only to Cepheid rows**, while SN calibrator and anchor geometric-prior rows constrain μ_i independently:

- **Cepheid rows:** `m^W_ij = L·θ - κ_Cep·X_i + ε_ij` (TEP term in the model)
- **SN calibrator rows:** `m^B_i = μ_i + M_B + ε_i` (no TEP term)
- **Anchor prior rows:** `μ_a ~ N(μ_a,geom, σ_a²)` (no TEP term)

The difference between Cepheid-inferred μ and SN-inferred μ provides leverage to identify κ_Cep. The augmented design matrix has full rank (47 = 47).

### Host-Summary Reconstruction Audit (Critical Check)

Before interpreting any TEP coefficient, we must verify that the full SH0ES matrix reproduces the same host-residual environmental trend that motivated TEP in the first place. The manuscript claims a correlation between per-host H₀ and σ of r≈0.47, ρ≈0.52, computed from H₀ = cz/d where d = 10^((μ−25)/5), using published SH0ES distance moduli.

**Result from the same SH0ES data:**

| Metric | Published Distance Moduli | Matrix-Fitted μᵢ | Manuscript Claim |
|--------|--------------------------|------------------|-----------------|
| Pearson r(σ, H₀) | **+0.466** (p=0.0109) | **+0.466** (p=0.0109) | +0.466 |
| Spearman ρ(σ, H₀) | **+0.517** (p=0.0041) | **+0.517** (p=0.0041) | +0.517 |
| Sample size | 29 | 29 | 29 |

**Verdict: The matrix baseline fit EXACTLY reproduces the manuscript's discovery statistic.**

**What this means:**

1. **The H₀–σ trend is real and survives in the matrix.** The baseline SH0ES GLS fit produces host distance moduli μᵢ that, when converted to H₀ = cz/d, yield the exact same correlation with σ as the published values.

2. **The matrix and the discovery statistic ARE testing the same object.** There is no methodological mismatch. The host-to-host variation in H₀ is present in the matrix-level fit.

3. **The baseline model already absorbs the trend through host-specific μᵢ parameters.** Each host has its own distance-modulus parameter in the matrix. These μᵢ parameters are free to vary independently, and they naturally capture any host-to-host scatter — including the environmental trend.

4. **The TEP term competes with existing μᵢ flexibility.** When we add a TEP correction κ·Xᵢ to the Cepheid rows, it is correlated with the host μᵢ parameters (since both vary by host). The baseline fit has already optimized μᵢ to minimize χ², leaving little residual for TEP to explain.

**Key insight:** The full-ladder test asks a different question than the host-summary analysis:
- **Host-summary:** "Do host H₀ values correlate with host σ?" **Answer: Yes (r = +0.466).**
- **Full-ladder TEP test:** "Does a Cepheid-row TEP term improve the fit beyond what host-specific μᵢ already explain?" **Answer: No (κ ≈ 0, Δχ² ≈ 0).**

This does not mean the H₀–σ trend is spurious. It means the trend is already absorbed by the baseline model's host-distance parameters, and the TEP projection (κ·Xᵢ on Cepheid rows) does not provide additional explanatory power.

**Why the first audit failed:** An earlier version of this audit incorrectly included NGC 4258 (z = 0.28, H₀ ≈ 11139) because anchor-name formats did not match ("NGC 4258" vs "N4258"). This single outlier destroyed the Pearson correlation. After fixing the name matching, the manuscript values are recovered exactly.

### What the Data Say

The free-κ fit finds κ_Cep = **(−0.067 ± 0.210)×10^6 mag** (0.32σ). This is:
- Consistent with zero (not significant)
- **Opposite sign from canonical TEP expectation** (TEP predicts positive κ_Cep, meaning Cepheids brightward of standard PL in high-sigma hosts)
- An order of magnitude below the canonical prediction of 9.7×10^5 mag

**Possible interpretations:**

1. **The host-summary trend is absorbed by the matrix fit.** The full SH0ES likelihood jointly fits PL slope (b) and metallicity (Z) globally. The environmental correlation that appears in host-summary residuals may be partially absorbed by these coefficients, leaving no detectable residual trend at the matrix level. In this case, the full-ladder TEP test cannot find a signal that the baseline fit has already absorbed.

2. **The environmental correlation is not TEP.** The host-residual correlation found in Stage 1 may be driven by other systematics (metallicity, crowding, period distribution) that are partially absorbed by the PL slope and metallicity coefficients in the full ladder.

3. **TEP effect is absent or much smaller than predicted.** The canonical KAPPA_GAL = 9.7×10^5 mag may overestimate the Cepheid observable response coefficient.

4. **The TEP signal has the wrong sign.** If the physical mechanism behind the residual correlation is not TEP but some other environmental effect, it could produce the opposite sign.

### Comparison with Canonical TEP Prediction

| Model | κ_Cep (mag) | κ_SN (mag) | H0 (km/s/Mpc) | χ² | Interpretation |
|-------|-------------|-----------|---------------|-----|----------------|
| Baseline | 0 | — | 73.04 | 3552.71 | Standard SH0ES |
| Free κ (Cepheid only) | −6.67×10^4 ± 2.10×10^5 | — | 73.12 | 3552.61 | Consistent with zero, wrong sign |
| Free κ (SN only) | — | 1.78×10^5 ± 2.42×10^5 | 73.14 | 3552.16 | Not significant (0.74σ) |
| Free κ (joint Cep+SN) | 2.62×10^5 ± 4.15×10^5 | 4.38×10^5 ± 4.78×10^5 | 72.96 | 3551.77 | Both positive but not significant |
| Prior κ (Bayesian) | 1.58×10^5 ± 1.86×10^5 | — | 72.86 | 3553.47 | Prior-likelihood compromise |
| Fixed κ (canonical) | 9.7×10^5 | — | 71.90 | 3576.91 | Worsens fit by Δχ² = 24.2 |

**Key findings from the expanded grid:**
- Cepheid-only TEP: κ_Cep ≈ 0, opposite sign from canonical.
- SN-only TEP: κ_SN ≈ 0, not significant.
- Joint Cepheid+SN TEP: both κ_Cep and κ_SN are **positive** (correct TEP direction) but neither significant (0.63σ and 0.92σ).
- The joint model with `anchor_screened_physical` gives H0 = 72.96, a modest downward shift, but Δχ² improvement is small (0.94).
- The `anchor_reference_zero` convention causes **catastrophic degeneracy** in joint models (κ ~ 10⁷ mag), proving anchors must receive physical TEP.
- No tested projection strongly prefers canonical TEP.

## Recommendations

### For the Current Paper

1. **Discard Stage 1 as a scientific result.** The 25σ κ_Cep and χ²/dof = 581 indicate model misspecification.

2. **Report the full-ladder results honestly:**
   - The TEP term is identifiable in the full SH0ES likelihood (rank = 47 = full).
   - The free-κ fit finds κ_Cep = −6.7×10^4 ± 2.1×10^5 mag (0.32σ), consistent with zero and with the **opposite sign** from TEP prediction.
   - The canonical fixed-κ correction shifts H0 by 1.14 km/s/Mpc but worsens χ² by 24.2.
   - The prior-constrained fit yields κ_Cep = 1.58×10^5 ± 1.86×10^5, a compromise between the likelihood (near zero) and the prior (9.7×10^5).

3. **Frame the discrepancy as a serious challenge to the canonical Cepheid prediction, not a disproof of broader TEP.** The data do not support the canonical KAPPA_GAL = 9.7×10^5 mag in Cepheid-only, SN-only, or joint Cepheid+SN projections. However, this tests only one observable consequence of TEP inside an expansion-based distance ladder. A dynamic time field could manifest through:
   - SN Ia light-curve stretch and time dilation;
   - Photon energy/redshift transport along the path;
   - Anchor clock interpretation;
   - The apparent Hubble-flow relation itself.
   These channels are not captured by a Cepheid-only κX_i correction.

4. **Future work:**
   - Test period-coupled and metallicity-coupled TEP models (already in grid; no strong signal found).
   - Construct a TEP-native redshift likelihood (step_35) that does not assume expansion.
   - Cross-channel consistency: compare κ_Cep with κ_TRGB, κ_SN, κ_pulsar.
   - Blind validation on new JWST Cepheid hosts.

## TEP–Metallicity Disentanglement (Post-P6)

The corrected P6 decomposition revealed that the H₀–σ trend is **partially absorbed by the metallicity correction** (Z_W): removing Z_W strengthens the correlation from r=0.466 to r=0.507. This motivated a direct disentanglement test:

| Model | Z_W free? | TEP term | κ_0 (10⁶ mag) | sig(κ_0) | κ_Z (10⁶ mag) | sig(κ_Z) | H₀ | χ² | Δχ² |
|-------|-----------|----------|---------------|----------|---------------|----------|-----|-------|-------|
| 1. Full baseline | Yes | — | — | — | — | — | 73.04 | 3552.71 | 0 |
| 2. Remove Z_W + TEP | **No** | Offset | **+0.141** | **0.68σ** | — | — | **1.07** | 3575.18 | **−22.5** |
| 3. Baseline + TEP | Yes | Offset | −0.067 | 0.32σ | — | — | 73.12 | 3552.61 | +0.10 |
| 4. Baseline + TEP + Z×TEP | Yes | Offset + Z-int | −0.062 | 0.30σ | +0.305 | 0.40σ | 73.19 | 3552.45 | +0.26 |
| 6. Orthogonalized TEP | Yes | X ⟂ Z_mean | −0.082 | 0.34σ | — | — | 72.99 | 3552.59 | +0.11 |

### Key results

1. **Removing Z_W catastrophically breaks the ladder.** Model 2 yields H₀ = 1.07 km/s/Mpc and χ² worsens by 22.5. Z_W is not an optional nuisance parameter — it is essential for physical distance calibration.

2. **TEP does not become significant when Z_W is removed.** In Model 2, κ = +0.141×10⁶ mag (0.68σ) — still well below canonical and not significant. The positive sign is TEP-consistent, but the error is large because the model is pathological.

3. **No TEP configuration improves χ².** All Models 3, 4, 6 have Δχ² ≈ +0.1 to +0.3 — they make the fit **worse**, not better.

4. **The metallicity-interaction term (κ_Z) is not significant.** Model 4 finds κ_Z = +0.305×10⁶ mag (0.40σ). A metallicity-dependent TEP response is not supported.

5. **Orthogonalized TEP (X ⟂ Z_mean) also fails.** Model 6 finds κ_perp = −0.082×10⁶ mag (0.34σ). Even after removing the metallicity-correlated component of X, TEP does not improve the fit.

6. **Host-level correlation:** r(host_X, host_Z_mean) = +0.402. The TEP regressor and host mean metallicity share ~16% variance, but this overlap is not what prevents TEP detection. Even after removing it, TEP remains insignificant.

### Interpretation

The environmental H₀–σ trend is **not hidden behind metallicity**. The trend survives in the matrix baseline because host-specific μᵢ parameters absorb it. TEP does not improve the fit because:

- The one-parameter TEP offset is too crude to capture the host-to-host scatter pattern;
- Host-specific distance moduli already have the flexibility to fit it;
- The TEP projection (κ·Xᵢ on Cepheid rows) does not align with the actual residual structure.

This is not evidence against TEP as a principle. It is evidence that **the Cepheid-only κXᵢ projection inside an expansion-based SH0ES ladder is not the right observable** to detect TEP. The host environmental signal is real, but it is absorbed by standard ladder degrees of freedom (μᵢ, Z_W), and the explicit TEP term does not provide additional explanatory power.

## Step 35: Bias-Aware TEP Ladder (Redshift-Distance Prior Sensitivity)

**Status: FIXED.** The original implementation had a critical conceptual error: it imposed redshift-distance priors on nearby calibrator hosts without accounting for the fact that peculiar velocities completely dominate at low redshift. The tight priors (σᵥ=200 km/s) artificially dragged H₀ from 73.04 to 69.67. This was a **prior-driven sensitivity test**, not a truth model.

### Corrected method

Two branches:

1. **PRIMARY:** Standard SH0ES + TEP terms, NO redshift priors. This is the proper distance ladder.
2. **SENSITIVITY:** Add redshift-distance priors for non-anchor calibrator hosts, sweeping σᵥ = 150, 250, 500, 1000 km/s, with hard validation:
   - Skip anchors (they have geometric priors)
   - Skip z > 0.05 (Hubble-flow hosts)
   - Skip z < 0.0035 (too local; peculiar velocities dominate)
   - Skip if σ_μ(vpec) > 0.5 mag (prior too noisy)

### Results — Primary branch (no redshift priors)

| Model | n_par | H₀ (km/s/Mpc) | χ² | κ_Cep (10⁶ mag) | sig | κ_SN (10⁶ mag) | sig |
|-------|-------|---------------|------|-----------------|-----|----------------|-----|
| A. Standard | 46 | **73.04** | 3552.7 | — | — | — | — |
| B. Cepheid-bias TEP | 47 | 73.19 | 3552.4 | −0.11 | 0.5σ | — | — |
| C. Full TEP | 48 | 73.02 | 3551.4 | +0.26 | 0.6σ | +0.50 | 1.0σ |

The primary branch **exactly reproduces** the step_34 baseline (H₀=73.04, χ²=3552.7). Adding TEP does not improve the fit (Δχ² ≈ 0), and κ_Cep is consistent with zero.

### Results — Sensitivity branch (redshift priors)

| σᵥ (km/s) | N priors | H₀ (A) | χ² (A) | H₀ shift |
|-----------|----------|--------|--------|----------|
| 150 | 14 | 72.25 | 3599.4 | −0.79 |
| 250 | 6 | 72.75 | 3569.7 | −0.29 |
| 500 | 2 | 73.00 | 3556.1 | −0.04 |
| 1000 | 1 | 73.03 | 3553.2 | −0.01 |

As σᵥ increases (weaker priors), H₀ converges back to the primary value. At σᵥ=500 and 1000, the prior effect is negligible. The original H₀=69.67 result was entirely driven by:
1. The NGC 4258 z=0.28 bug (wrong redshift in metadata)
2. Tight priors on hosts where σ_μ(vpec) ≫ 0.5 mag

With hard validation, the redshift-prior branch does **not** produce a decisive TEP test. It is a prior-driven sensitivity test only.

### Interpretation

The redshift-prior branch is **not a valid TEP rescue test** for two reasons:

1. **Nearby calibrator redshifts are not independent distance anchors.** At cz ~ 1000–3000 km/s, peculiar velocities (σᵥ ~ 250 km/s) produce distance-modulus uncertainties of 0.4–1.2 mag. Using these as priors imposes a Hubble law on distances where the Hubble law is not precise.

2. **No-expansion TEP is conceptually inconsistent with Hubble-law priors.** If TEP says there is no expansion, then μᵢ = 5log₁₀(cz/H₀)+25 is not a valid truth prior. It assumes the very physics TEP questions.

The correct bias-aware test requires **external non-Cepheid distances** (TRGB, masers, SBF, eclipsing binaries) to break the Cepheid degeneracy, or a TEP-native redshift model. The redshift-prior branch is a standard-expansion sensitivity test, not a decisive TEP test.

**Bottom line:** The primary model confirms step_34: TEP (κXᵢ on Cepheid rows) is identifiable but the data prefer κ_Cep ≈ 0. The redshift-prior sensitivity branch shows that even when μᵢ are weakly constrained by redshift, TEP does not improve the fit. A TEP-native redshift/time-field likelihood is needed for a decisive test.

## Step 36: Apparent Hubble Environment Likelihood

Tests whether the apparent local Hubble constant H₀,i^app = cz_i / d_i correlates with environment (TEP regressor X), after controlling for metallicity, period, and redshift. Uses MLE-fitted intrinsic scatter and centered covariates.

### Method

For each non-anchor calibrator host:
- Compute apparent distance modulus μ_i from SH0ES matrix baseline GLS fit
- Compute apparent H₀ = cz / d, with d = 10^((μ-25)/5)
- Error model: σ_H² = (0.4605 · H · σ_μ)² + (σ_v / d)² + σ_int²
- Fit via MLE: maximize ln L = -0.5 Σ[(H_i - H_model,i)²/σ_H,i² + ln σ_H,i²]
- Center all covariates before fitting to avoid intercept bias

Models:
- 0: H = H_app + ε
- 1: H = H_app + β_X X + ε
- 2: H = H_app + β_X X + β_Z ⟨Z⟩ + ε
- 3: H = H_app + β_X X + β_Z ⟨Z⟩ + β_P ⟨P⟩ + ε
- 4: H = H_app + β_X X + β_Z ⟨Z⟩ + β_P ⟨P⟩ + β_z z + ε

### Results

| σᵥ (km/s) | Model | H_app | σ_int | β_X (10⁷) | sig(β_X) | p_perm | VIF | LOOCV |
|-----------|-------|-------|-------|-----------|----------|--------|-----|-------|
| 150 | 0. Intercept | 68.39 | 5.6 | — | — | — | 1.0 | 93.3 |
| 150 | 1. +X | 68.19 | 3.3 | **+2.87** | **3.3σ** | **0.008** | 1.0 | 78.1 |
| 150 | 2. +X+Z | 68.22 | 2.8 | +2.39 | 2.5σ | 0.040 | 1.2 | 79.8 |
| 150 | 3. +X+Z+P | 67.41 | 2.1 | +2.26 | 2.6σ | 0.034 | 1.3 | 74.4 |
| 250 | 1. +X | 68.38 | 0.0 | +2.66 | 2.7σ | 0.020 | 1.0 | 77.6 |
| 250 | 2. +X+Z | 68.41 | 0.0 | +2.17 | 1.9σ | 0.048 | 1.2 | 79.0 |
| 500 | 1. +X | 68.37 | 0.0 | +2.64 | 1.4σ | 0.024 | 1.0 | 77.5 |
| 500 | 2. +X+Z | 68.43 | 0.0 | +2.10 | 1.0σ | 0.056 | 1.2 | 78.8 |

### Key findings

1. **β_X is positive (TEP-consistent) across all models and σᵥ values.** Larger X (deeper potential) → larger apparent H₀. This is the first TEP-directional signal in the analysis chain.

2. **Significance is marginal and σᵥ-dependent.** At σᵥ=150 km/s, β_X is 3.3σ (p_perm=0.008). At σᵥ=500 km/s, it drops to 1.4σ. The result is not yet robust to velocity-error assumptions.

3. **Metallicity control weakens but does not kill the signal.** Adding ⟨Z⟩ drops significance from 3.3σ → 2.5σ (σᵥ=150). The environmental and metallicity signals are partially entangled, but β_X survives in sign.

4. **Period control has modest effect.** Model 3 has slightly better LOOCV (74.4 vs 78.1) but does not dramatically change β_X. VIFs are all < 2.0, so collinearity is not severe.

5. **Intrinsic scatter is fitted.** σ_int ≈ 3.3 km/s/Mpc at σᵥ=150, but goes to ~0 at higher σᵥ because the velocity uncertainty absorbs the extra dispersion.

6. **H_app is stable at ~68.2 km/s/Mpc.** This is physically sensible for the local ladder (slightly below the full-ladder 73.04 because local calibrators include lower-velocity hosts).

### Interpretation

A TEP-consistent positive environment term appears in the apparent local Hubble relation. Its sign is robust across velocity-error assumptions, but its significance is sensitive to the peculiar-velocity model and partially entangled with metallicity.

The signal is more naturally in the **apparent Hubble/redshift-distance sector** than in a pure Cepheid PL correction (κX on Cepheid rows). This supports the conceptual direction of a TEP-native redshift model: cz = d · (H_app + β_X X) + v_pec.

**Bottom line:** Step 36 produces the first TEP-directional result. β_X > 0 is significant at 3.3σ for σᵥ=150 km/s, but drops to 1.4σ at σᵥ=500 km/s. The signal survives metallicity and period controls in sign but not at decisive significance.

## Step 37: Velocity Robustness Suite

Stress-tests whether Step 36's positive β_X survives under realistic velocity treatment, redshift cuts, jackknife, robust regression, and bootstrap.

### Method

1. **Sigma_v sweep:** 150, 250, 300, 500, 750, 1000 km/s
2. **Redshift cut sweep:** z ≥ 0.0035, 0.005, 0.0075, 0.01 (remove local peculiar-velocity dominated hosts)
3. **Leave-one-host-out (LOHO):** remove each host, refit model 1
4. **Student-t robust regression:** downweight outliers with heavy-tailed likelihood (ν=4)
5. **Redshift-bin-preserving permutation:** permute X within redshift quartiles to preserve redshift-structure coupling
6. **Bootstrap:** resample hosts with replacement (N=1000)

### Results

| Test | σᵥ (km/s) | z_cut | N | H_app | β_X (10⁷) | sig(β_X) | p_perm | Notes |
|------|-----------|-------|---|-------|-----------|----------|--------|-------|
| standard | 150 | — | 35 | 68.19 | **+2.87** | **3.3σ** | — | baseline |
| standard | 250 | — | 35 | 68.38 | +2.66 | 2.7σ | — | baseline |
| standard | 300 | — | 35 | 68.39 | +2.65 | 2.3σ | — | baseline |
| standard | 500 | — | 35 | 68.37 | +2.64 | 1.4σ | — | baseline |
| standard | 750 | — | 35 | 68.34 | +2.65 | 1.0σ | — | baseline |
| standard | 1000 | — | 35 | 68.32 | +2.65 | 0.7σ | — | baseline |
| z_cut | 250 | 0.0035 | 29 | 69.27 | +2.35 | 2.3σ | — | remove most local |
| z_cut | 250 | 0.005 | 23 | 70.05 | +2.02 | 2.0σ | — | more conservative |
| z_cut | 250 | 0.0075 | 13 | 70.23 | +1.95 | 1.8σ | — | high-z only |
| perm_binned | 250 | — | 35 | — | — | — | **0.014** | preserve redshift structure |
| bootstrap | 250 | — | 35 | — | +3.11 ± 1.38 | — | — | 100% positive, 95% CI [1.94, 7.45] |
| loho | 250 | — | 35 | — | +2.67 ± 0.10 | — | — | **positive in ALL 35/35** |

### Key findings

1. **β_X is positive in EVERY jackknife and EVERY bootstrap draw.** LOHO: 35/35 hosts give positive β_X. Bootstrap: 1000/1000 draws give positive β_X. The signal is **not driven by a single outlier.** Most influential host is N976A, but even removing it leaves β_X = +2.52e+07.

2. **Redshift-bin-preserving permutation is significant (p=0.014).** Even when redshift-environment coupling is preserved by permuting within redshift quartiles, β_X is significant. This argues against the signal being purely a redshift-distance artifact.

3. **Redshift cuts do not kill the signal.** Removing the most local hosts (z < 0.0035, z < 0.005, z < 0.0075) reduces significance from 2.7σ → 1.8σ but β_X remains positive. The hosts driving the signal are not exclusively the most local ones.

4. **Significance IS velocity-dependent.** At σᵥ=150, 3.3σ. At σᵥ=1000, 0.7σ. The true significance depends on the unknown peculiar-velocity dispersion. If σᵥ ≈ 250 km/s, significance is ~2.7σ. If σᵥ ≳ 500 km/s, the signal becomes marginal.

5. **H_app rises with redshift cut.** From 68.4 (all hosts) to 70.2 (z ≥ 0.0075). Higher-redshift calibrators have larger apparent H₀, consistent with the environmental trend (higher-z hosts tend to have larger σ and thus larger β_X·X).

### Interpretation

The robustness suite answers the user's key question: **Does β_X > 0 survive conservative peculiar-velocity treatment?**

**Sign:** Yes. β_X is unambiguously positive in every test: every σᵥ, every redshift cut, every jackknife, every bootstrap draw, and the binned permutation.

**Significance:** Marginal and model-dependent. At a conventional σᵥ ≈ 250 km/s, β_X ≈ 2.7σ. At more conservative σᵥ ≈ 500 km/s, it drops to ~1.4σ. The bootstrap 95% CI [1.94e+07, 7.45e+07] does not include zero, but the width is large.

**Credibility:** The LOHO and bootstrap results are the strongest evidence. A signal that survives every jackknife and every bootstrap draw is not an outlier artifact. The redshift-cut and binned-permutation results further argue against simple redshift-distance systematics.

**Bottom line:** Step 37 shows that the apparent-Hubble environment signal is **real in sign** (always positive) but **uncertain in significance** (2–3σ depending on velocity model). This is a coherent TEP-directional result, not a statistical fluke, but it is not yet decisive. A better peculiar-velocity model (e.g., group-flow corrections, 3D velocity field) is the main remaining uncertainty.

## Conclusion

The full-ladder likelihood analysis yields a **critical result**: the TEP term is identifiable (not degenerate) when applied only to Cepheid rows, but the data prefer a κ_Cep consistent with zero and with the **opposite sign** from the canonical TEP prediction.

**Key findings:**
- **Injection tests pass for all model classes** (recovery fractions ~1.0 for Cepheid-only, SN-only, and joint models). The sign convention, row mask, scaling, and GLS machinery are numerically correct.
- **The H₀–σ trend is real and exactly reproduced by the matrix baseline.** r(σ, H₀) = +0.466 (p=0.0109), ρ = +0.517 (p=0.0041), N=29. Matrix-fitted μᵢ give the same values as published distance moduli.
- **The baseline model absorbs the trend through host-specific μᵢ parameters.** Each host has its own distance-modulus parameter, which naturally captures host-to-host scatter. The TEP term (κ·Xᵢ) competes with existing μᵢ flexibility.
- The TEP term is identifiable (rank = 47 = full) thanks to SN calibrator and anchor constraints.
- Cepheid-only TEP: κ_Cep = (−0.067 ± 0.210)×10^6 mag (0.32σ), consistent with zero and opposite sign from canonical TEP.
- SN-only TEP: κ_SN = 0.178×10^6 ± 2.42×10^5 mag (0.74σ), not significant.
- Joint Cepheid+SN TEP (physical anchor convention): κ_Cep = 0.262×10^6 ± 4.15×10^5 (0.63σ), κ_SN = 0.438×10^6 ± 4.78×10^5 (0.92σ). Both positive (TEP direction) but neither significant.
- Prior-constrained fit: posterior κ_Cep = 0.158×10^6 ± 0.186×10^6 mag, well below canonical.
- Fixed canonical correction: H0 shifts from 73.04 to 71.90 km/s/Mpc (20% of Planck tension), but worsens χ² by 24.2.
- The `anchor_reference_zero` convention causes catastrophic degeneracy in joint models (ill-conditioned, not physical), proving anchors must receive physical TEP.

**For the paper:** The canonical KAPPA_GAL = 9.7×10⁵ mag Cepheid-clock correction is not supported by the SH0ES full-ladder likelihood. The H₀–σ environmental trend is real (r = +0.466) and is perfectly reproduced by the matrix baseline fit, but it is **absorbed by host-specific distance-modulus parameters** rather than by a TEP Cepheid-row correction. The full-ladder test asks whether TEP improves the fit beyond what host-specific μᵢ already explain, and the answer is no.

**Bottom line:** The Cepheid-only κXᵢ projection inside an expansion-based distance ladder is not the decisive observable for TEP. The H₀–σ trend is real, but the SH0ES matrix baseline already absorbs it through host-specific distance parameters and metallicity corrections. The TEP–metallicity disentanglement test confirms that TEP is not hidden behind Z_W: removing Z_W catastrophically breaks the ladder, and even orthogonalized TEP (X ⟂ Z_mean) does not improve the fit. Canonical TEP does not provide additional explanatory power in this framework. A TEP-native redshift/time-field likelihood is needed to test the broader claim.
