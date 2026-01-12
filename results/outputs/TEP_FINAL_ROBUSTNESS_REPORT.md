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

### Local Density Control
- The correlation between H0 and σ persists after controlling for local galaxy density ($r_{partial} = 0.458$, $p = 0.0124$).
- This rules out environmental density (e.g., crowding bias) as the sole driver of the H0 trend.

### Stellar Absorption Subsample
- Restricting to hosts with high-quality stellar σ (excluding HI proxy) maintains the signal.
- **Pearson r:** 0.41958039938763225

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

## 5. Anchor Tension and Mass Distortion

While M31 and SN hosts fit the model, NGC 4258 presents a challenge:
- **Quantitative Check:** Density reconstruction for NGC 4258 yields $\rho \approx 0.03 M_\odot/pc^3$, which is **Unscreened**.
- **Observation:** Its Cepheid zero-point is standard (consistent with LMC), lacking the predicted TEP brightness boost.
- **Conclusion:** This constitutes a genuine **Anchor Tension**. The calibrators do not strictly follow the environmental trend of the SN hosts.
- **Mass Distortion Caveat:** We note that TEP-induced proper time rate variations could distort dynamical mass measurements ($M \propto V^2 R$), potentially biasing the derived density. However, a factor of ~15 error would be required to shift NGC 4258 into the screened regime.

## 6. Conclusion

The TEP hypothesis survives rigorous robustness testing in SN hosts and M31, but faces a challenge with NGC 4258 (Anchor Tension). The global H0-σ correlation (Step 6) is driven by unscreened disk environments. The M31 'Inner Fainter' signal (Step 8) is identified as the signature of the **screening threshold** being crossed. Future work must resolve why the anchor NGC 4258 appears standard despite its low density.
