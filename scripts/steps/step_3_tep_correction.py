import json
import os
import shutil
import sys
from pathlib import Path

# Make direct execution (`python scripts/steps/step_3_tep_correction.py`)
# behave the same as module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# Import TEP Logger
try:
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )
except ImportError:
    # Add project root to path if needed
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )

from scripts.utils.tep_correction import (
    tep_correction,
    C_SQUARED_KM_S,
    ANCHOR_SCREENING,
    ANCHOR_NMB,
    group_screening_factor,
)


class Step3TEPCorrection:
    r"""
    Step 3: TEP Correction and Unification
    ======================================

    This step applies the Temporal Equivalence Principle (TEP) correction to the
    Cepheid distance moduli to resolve the Hubble Tension.

    The Physics (TEP framework, Jakarta v0.8 §5–§7; Istanbul v0.3 §1.3, §2.4):
    Matter and clocks couple universally to the matter metric
    $\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$
    with conformal factor $A(\phi)$. In freely-falling local labs, c is exactly
    invariant (null cones preserved); the scalar $\phi$ rescales both clocks
    and rulers uniformly so local physics is identical to GR. Observable
    departures arise from the continuous spatial profile of $A(\phi)$
    (Temporal Topology) and its gradient (Temporal Shear $\Sigma_\mu = \nabla_\mu \ln A$).

    Cepheids are standard clocks (Leavitt Law). Their pulsation period sets
    the inferred luminosity, hence the distance modulus. In environments with
    a different proper-time rate $A(\phi)$ relative to the calibrator
    environment, the inferred distance modulus is shifted. The shift is
    parameterized at the channel level (Jakarta Eq. 228) as

    $$ \Delta O_X = \kappa_X \cdot \mathcal{S}_X(\mathcal{E}) \cdot
                    \mathcal{F}_X[\Delta\ln A, \Sigma_\mu, C_A; \Phi, \rho, z]. $$

    For the Cepheid channel, with $\mathcal{F}_{\rm Cep}\propto(\sigma^2 - \sigma_{\rm ref}^2)/c^2$
    via the virial relation $\Phi\propto\sigma^2$, this reduces to:

    $$ \mu_{\rm corr} = \mu_{\rm obs} + \kappa_{\rm Cep} \cdot S(\rho) \cdot \frac{\sigma^2 - \sigma_{\rm ref}^2}{c^2} $$

    Critical TEP framing (do not confuse with standard-GR thinking):
    -   $\kappa_{\rm Cep}$ is an OBSERVABLE channel response coefficient, NOT
        the microscopic conformal coupling $\beta$ and NOT a PPN parameter.
        It absorbs the virial proportionality, the P-L slope, and $1/\ln 10$;
        with $\sigma^2/c^2\sim10^{-7}$ it is naturally of order $10^6$ mag.
    -   $S(\rho)$ as implemented here uses LOCAL stellar density at the typical
        Cepheid disk radius. This captures only one component of TEP screening
        ($\mathcal{S}_X(\mathcal{E})$); the full environmental state $\mathcal{E}$
        in TEP includes group/cluster membership and cosmological-scale density
        which are NOT modelled here. As a result, screening is largely inactive
        ($S\approx 1$) for the SH0ES disk Cepheid sample, consistent with their
        Hubble-flow location (UNSCREENED regime).
    -   The geometric anchors LMC, M31, N4258 reside in DEEP cosmological
        potential wells (Local Group / Local Volume) and are expected to be
        SCREENED in TEP. Absence of $\sigma$-correlation across the anchors
        is therefore the predicted density-regime screening transition,
        not a refutation of TEP. See step_10_anchor_stratification.
    -   The fitted $\kappa_{\rm Cep}$ is a measurement in the unscreened
        Hubble-flow regime. The TEP test is CROSS-CHANNEL consistency
        ($\kappa_{\rm Cep}$ vs $\kappa_{\rm TRGB}$, $\kappa_{\rm SN}$,
        $\kappa_{\rm pulsar}$), not single-channel significance from zero.

    Where:
    -   $\kappa_{\rm Cep}$: Observable Response Coefficient (units: magnitude).
    -   $S(\rho) \in [0,1]$: Local-density shear-suppression factor (partial
        proxy for the full TEP $\mathcal{S}_\Sigma(\mathcal{E})$).
    -   $\sigma_{\rm ref}$: Effective velocity dispersion of the calibrator
        sample (MW, LMC, N4258).

    Objective:
    Fit $\kappa_{\rm Cep}$ to minimise the residual H0–σ slope across the
    SH0ES Hubble-flow hosts. The corrected unified $H_0$ is the channel
    estimate after this correction.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.root_dir / "logs"
        self.figures_dir = self.root_dir / "results" / "figures"
        self.outputs_dir = self.root_dir / "results" / "outputs"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Logger
        self.logger = TEPLogger(
            "step_3_correction", log_file_path=self.logs_dir / "step_3_correction.log"
        )
        set_step_logger(self.logger)

        # Inputs
        self.input_path = self.outputs_dir / "stratified_h0.csv"

        # Outputs
        self.corrected_output_path = self.outputs_dir / "tep_corrected_h0.csv"
        self.json_output_path = self.outputs_dir / "tep_correction_results.json"
        self.plot_path = self.figures_dir / "figure_03_tep_correction_comparison.png"

        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        self.public_figures_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Loads the stratified dataset."""
        print_status("Loading Data...", "SECTION")

        if not self.input_path.exists():
            print_status("Input file missing. Please run Step 2 first.", "ERROR")
            sys.exit(1)

        df = pd.read_csv(self.input_path)
        # Ensure shear_suppression exists; default to 1.0 (fully active) if missing
        if "shear_suppression" not in df.columns:
            df["shear_suppression"] = 1.0
            print_status(
                "shear_suppression column missing; defaulting all hosts to S=1.0 (fully active).",
                "WARNING",
            )
        print_status(f"Loaded {len(df)} hosts for correction.", "INFO")
        return df

    def calculate_effective_calibrator_sigma(self):
        """
        Calculates the effective velocity dispersion of the anchor sample.

        WEIGHTING RATIONALE (SH0ES-motivated):
        The weights reflect each anchor's contribution to the P-L zero-point calibration,
        NOT simply the distance precision. From Riess et al. (2022), the contributions are:

        - MW (~20%): Many Cepheids with individual Gaia parallaxes, but higher scatter
        - LMC (~25%): ~70 Cepheids with excellent HST photometry, precise DEB distance
        - N4258 (~55%): ~139 Cepheids, gold-standard maser distance, and critically:
          it is the ONLY anchor that is a GALAXY (like the SN hosts being corrected)

        NGC 4258 is weighted most heavily because it provides the most relevant
        calibration environment for extragalactic distance measurements.

        CRITICAL: ANCHOR SIGMA VALUES
        These are DISK velocity dispersions at Cepheid locations, NOT central bulge values.
        Cepheids reside in galactic disks at R ~ 4-8 kpc, where the local potential is
        shallower than the nuclear region. Using central bulge dispersions (e.g., Ho+2009
        gives NGC 4258 = 148 km/s) would overestimate the effective potential.

        Sources:
        - MW: Bovy+2012 thin disk σ_z at solar neighborhood = 30 km/s
        - LMC: van der Marel+2002 disk dispersion = 24 km/s
        - N4258: Kormendy & Ho 2013 intermediate-aperture value = 115 km/s
          (NOT the central bulge value of 148 km/s from Ho+2009)

        The resulting σ_ref ≈ 75 km/s is validated by the empirical result that
        low-σ SN hosts (σ < 90 km/s) yield H0 = 67.8 km/s/Mpc, matching Planck.

        SCREENED-EFFECTIVE VARIANT:
        Because the TEP framework argues the geometric anchors reside in deep
        cosmological potential wells (Local Group / Local Volume) where Temporal
        Shear is suppressed, their effective contribution to the *active-shear*
        reference should be weighted by the environmental screening factor S.
        The correction uses σ², so the screened reference is
        σ_ref,scr² = Σ w_i S_i σ_i² / Σ w_i.
        Both standard and screened σ_ref are returned; the headline H₀ is
        required to be stable under both definitions.
        """
        print_status("Calculating Effective Calibrator Sigma...", "SECTION")

        # Anchor velocity dispersions: DISK values at Cepheid locations
        # These are intentionally different from the central bulge values in the
        # regenerated sigma catalog, which uses Ho+2009 nuclear apertures.
        anchors = [
            {
                "ID": "MW",
                "Sigma": 30.0,
                "Desc": "Milky Way Disk (Bovy+2012)",
                "Weight": 0.20,  # ~270 Cepheids, individual parallaxes, higher scatter
            },
            {
                "ID": "LMC",
                "Sigma": 24.0,
                "Desc": "LMC Disk (vdMarel+2002)",
                "Weight": 0.25,  # ~70 Cepheids, excellent photometry
            },
            {
                "ID": "NGC 4258",
                "Sigma": 115.0,
                "Desc": "NGC 4258 Disk (K&H2013)",
                "Weight": 0.55,  # ~139 Cepheids, gold anchor, galaxy environment
            },
        ]

        print_status(
            "Using SH0ES-motivated weights (based on P-L contribution).", "INFO"
        )

        # Display Anchor Table
        headers = ["Anchor", "Sigma (km/s)", "Nmb", "S", "Weight", "Description"]
        rows = []
        numerator = 0.0
        denominator = 0.0
        numerator_scr = 0.0

        for a in anchors:
            S = ANCHOR_SCREENING.get(a["ID"], 1.0)
            nmb = ANCHOR_NMB.get(a["ID"], 1)
            rows.append(
                [a["ID"], f"{a['Sigma']:.1f}", f"{nmb}", f"{S:.3f}", f"{a['Weight']:.2f}", a["Desc"]]
            )
            numerator += a["Sigma"] * a["Weight"]
            denominator += a["Weight"]
            numerator_scr += a["Weight"] * S * (a["Sigma"] ** 2)

        print_table(headers, rows, title="Geometric Anchor Sample (S from Nmb formula)")

        sigma_ref = numerator / denominator
        sigma_ref_screened = np.sqrt(numerator_scr / denominator)
        print_status(
            f"Standard Reference Sigma (σ_ref):       {sigma_ref:.2f} km/s", "SUCCESS"
        )
        print_status(
            f"Screened-Effective Sigma (σ_ref,scr):   {sigma_ref_screened:.2f} km/s  "
            f"(NGC 4258 contribution down-weighted by S={ANCHOR_SCREENING['NGC 4258']:.2f})",
            "SUCCESS",
        )

        return sigma_ref, sigma_ref_screened

    def optimize_correction(self, df, sigma_ref):
        """Finds the optimal correction parameter kappa_cep."""
        print_status("Optimizing TEP Coupling (κ_Cep)...", "SECTION")

        # Shear suppression factor S(rho) from Temporal Topology (TEP v0.8)
        S = df["shear_suppression"].values

        sigma_vals = df["sigma_inferred"].values

        # Objective function: minimize H0 vs σ correlation
        # We want the corrected H0 to be independent of environment (slope ~ 0)
        def objective(params):
            kappa_cep = params[0]

            # Physics-derived correction: mu_corr = mu_obs + kappa_cep * S * (sigma^2 - sigma_ref^2)/c^2
            correction = tep_correction(sigma_vals, sigma_ref, kappa_cep, S)
            mu_corr = df["value"].values + correction

            d_corr = 10 ** ((mu_corr - 25) / 5)
            h0_corr = df["velocity"].values / d_corr

            # Minimize squared slope of H0 vs Sigma
            slope, _ = np.polyfit(sigma_vals, h0_corr, 1)

            return slope**2

        # Optimize. κ_Cep is now ~1e6 in this convention (sigma^2/c^2 ~ 1e-7),
        # so seed the optimizer in the physically expected range.
        initial_guess = [1.0e6]
        res = minimize(
            objective,
            x0=initial_guess,
            method="Nelder-Mead",
            options={"xatol": 1.0, "fatol": 1e-8, "maxiter": 2000},
        )
        best_kappa = res.x[0]

        print_status(f"Optimization converged: {res.success}", "INFO")
        print_status(
            f"Optimal Observable Response Coefficient (κ_Cep): {best_kappa:.3e} mag",
            "SUCCESS",
        )
        print_status(
            f"Mean effective coupling (κ_Cep·⟨S⟩): {(best_kappa * S.mean()):.3e}",
            "INFO",
        )

        return best_kappa

    def apply_correction(self, df, kappa_cep, sigma_ref):
        """Applies the correction and calculates stats."""
        print_status("Applying Conformal Correction...", "SECTION")

        # Continuous shear-suppression factor (TEP v0.8)
        S = df["shear_suppression"].values
        sigma_vals = df["sigma_inferred"].values
        df["effective_coupling"] = kappa_cep * S

        correction = tep_correction(sigma_vals, sigma_ref, kappa_cep, S)
        df["mu_corrected"] = df["value"].values + correction
        df["dist_corrected"] = 10 ** ((df["mu_corrected"] - 25) / 5)
        df["h0_corrected"] = df["velocity"] / df["dist_corrected"]

        # Sample Correction Table
        headers = [
            "Host",
            "Sigma",
            "S(rho)",
            "H0 (Raw)",
            "Corr (mag)",
            "H0 (TEP)",
        ]
        rows = []
        sample = df.sample(5, random_state=42).sort_values("sigma_inferred")
        for _, row in sample.iterrows():
            rows.append(
                [
                    row["normalized_name"],
                    f"{row['sigma_inferred']:.1f}",
                    f"{row['shear_suppression']:.3f}",
                    f"{row['h0_derived']:.2f}",
                    f"{row['effective_coupling'] * (row['sigma_inferred']**2 - sigma_ref**2) / C_SQUARED_KM_S:+.4f}",
                    f"{row['h0_corrected']:.2f}",
                ]
            )
        print_table(headers, rows, title="Sample Corrections (Suppression-Aware)")

        h0_mean = df["h0_corrected"].mean()
        # Standard error of the mean (simple)
        h0_sem = df["h0_corrected"].std() / np.sqrt(len(df))

        print_status("-" * 60, "INFO")
        print_status(
            f"UNIFIED H0 (Statistical): {h0_mean:.2f} +/- {h0_sem:.2f} km/s/Mpc",
            "SUCCESS",
        )
        print_status(
            "Note: This error (SEM) assumes κ_Cep is fixed/known perfectly.", "INFO"
        )
        print_status("-" * 60, "INFO")

        return df, h0_mean, h0_sem

    def bootstrap_analysis(self, df, sigma_ref, n_boot=1000):
        """Joint bootstrap of kappa_Cep and H0 to estimate honest uncertainties.

        For each bootstrap resample of host galaxies, kappa_Cep is RE-OPTIMIZED
        using the same objective as the main fit (minimize squared slope of
        H0 vs sigma). This correctly propagates BOTH:
          (a) Host-to-host sampling variance, and
          (b) kappa_Cep parameter uncertainty,
        into the unified H0 estimate, without circular reasoning or double
        counting. The previous version forced H0 = 68.0 in the inner loop,
        which artificially suppressed bootstrap_h0_std to ~3e-6; this is now
        replaced with the consistent slope^2 objective.

        Returns:
            dict with bootstrap statistics for H0 and kappa_Cep.
        """
        print_status(f"Joint Bootstrap (kappa refit + H0): N={n_boot}...", "SECTION")

        # Set seed for reproducibility
        np.random.seed(42)

        # Suppress optimization warnings
        import warnings
        warnings.filterwarnings("ignore")

        h0s = []
        kappas = []
        slopes = []
        n_samples = len(df)
        n_failed = 0

        for _ in range(n_boot):
            # Resample hosts with replacement
            sample = df.sample(n=n_samples, replace=True)
            S_sample = sample["shear_suppression"].values
            sigma_sample = sample["sigma_inferred"].values
            mu_sample = sample["value"].values
            v_sample = sample["velocity"].values

            # Re-optimize kappa using SAME objective as main fit (slope^2)
            def obj(k):
                corr = tep_correction(sigma_sample, sigma_ref, k[0], S_sample)
                mc = mu_sample + corr
                dc = 10 ** ((mc - 25) / 5)
                hc = v_sample / dc
                slope_b, _ = np.polyfit(sigma_sample, hc, 1)
                return slope_b ** 2

            res = minimize(
                obj,
                x0=[1.0e6],
                method="Nelder-Mead",
                options={"xatol": 100.0, "fatol": 1e-8, "maxiter": 500},
            )
            if not res.success:
                n_failed += 1
                continue

            kappa_b = res.x[0]
            corr = tep_correction(sigma_sample, sigma_ref, kappa_b, S_sample)
            mc = mu_sample + corr
            dc = 10 ** ((mc - 25) / 5)
            hc = v_sample / dc
            h0_b = float(np.mean(hc))
            slope_b, _ = np.polyfit(sigma_sample, hc, 1)

            h0s.append(h0_b)
            kappas.append(kappa_b)
            slopes.append(slope_b)

        warnings.resetwarnings()

        h0s = np.array(h0s)
        kappas = np.array(kappas)
        slopes = np.array(slopes)

        # The kappa bootstrap distribution is positively skewed (heavy right tail
        # from optimizer behavior on under-determined resamples), so we report
        # both standard moments AND robust order statistics. Median is the
        # preferred central estimator; (q84 - q16)/2 approximates 1-sigma robustly.
        kappa_q16 = float(np.percentile(kappas, 16))
        kappa_q50 = float(np.percentile(kappas, 50))
        kappa_q84 = float(np.percentile(kappas, 84))
        h0_q16 = float(np.percentile(h0s, 16))
        h0_q50 = float(np.percentile(h0s, 50))
        h0_q84 = float(np.percentile(h0s, 84))

        metrics = {
            "bootstrap_h0_mean": float(np.mean(h0s)),
            "bootstrap_h0_std": float(np.std(h0s)),
            "bootstrap_h0_median": h0_q50,
            "bootstrap_h0_robust_std": float((h0_q84 - h0_q16) / 2.0),
            "bootstrap_h0_ci_lower": float(np.percentile(h0s, 2.5)),
            "bootstrap_h0_ci_upper": float(np.percentile(h0s, 97.5)),
            "bootstrap_kappa_mean": float(np.mean(kappas)),
            "bootstrap_kappa_std": float(np.std(kappas)),
            "bootstrap_kappa_median": kappa_q50,
            "bootstrap_kappa_robust_std": float((kappa_q84 - kappa_q16) / 2.0),
            "bootstrap_kappa_ci_lower": float(np.percentile(kappas, 2.5)),
            "bootstrap_kappa_ci_upper": float(np.percentile(kappas, 97.5)),
            "bootstrap_kappa_skewness": float(
                ((kappas - kappas.mean()) ** 3).mean() / (kappas.std() ** 3 + np.finfo(float).eps)
            ),
            "bootstrap_kappa_n_negative": int((kappas < 0).sum()),
            "bootstrap_residual_slope_mean": float(np.mean(np.abs(slopes))),
            "bootstrap_n_converged": int(len(h0s)),
            "bootstrap_n_failed": int(n_failed),
        }

        print_status("Joint Bootstrap Results:", "SUBTITLE")
        headers = ["Parameter", "Mean", "Std (Uncertainty)", "95% CI"]
        rows = [
            [
                "H0 (Unified)",
                f"{metrics['bootstrap_h0_mean']:.2f}",
                f"{metrics['bootstrap_h0_std']:.2f}",
                f"[{metrics['bootstrap_h0_ci_lower']:.2f}, {metrics['bootstrap_h0_ci_upper']:.2f}]",
            ],
            [
                "kappa_Cep",
                f"{metrics['bootstrap_kappa_mean']:.3e}",
                f"{metrics['bootstrap_kappa_std']:.3e} ({metrics['bootstrap_kappa_std']/metrics['bootstrap_kappa_mean']*100:.1f}%)",
                f"[{metrics['bootstrap_kappa_ci_lower']:.3e}, {metrics['bootstrap_kappa_ci_upper']:.3e}]",
            ],
        ]
        print_table(headers, rows, title="Bootstrap (kappa refit per sample)")
        print_status(
            f"Converged: {metrics['bootstrap_n_converged']}/{n_boot}, "
            f"residual |slope| = {metrics['bootstrap_residual_slope_mean']:.2e}",
            "INFO",
        )
        print_status(
            "Bootstrap H0 std combines host scatter AND kappa uncertainty.",
            "INFO",
        )

        return metrics

    def sensitivity_analysis(self, df, fixed_kappa_cep=None):
        """Analyzes sensitivity of H_0 to sigma_ref."""
        print_status("Sensitivity Analysis (Sigma Ref Scan)...", "PROCESS")

        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style

            colors = apply_tep_style()
        except ImportError:
            colors = {
                "blue": "#395d85",
                "accent": "#b43b4e",
                "green": "#4a2650",
                "dark": "#301E30",
                "light_blue": "#4b6785",
            }

        sigma_refs = np.linspace(30, 130, 20)
        h0_results_refit = []
        h0_results_fixed = []
        h0_err_refit = []
        h0_err_fixed = []

        planck_h0 = 67.4

        for sr in sigma_refs:
            # Diagnostic curve: re-optimize kappa_cep for this sr.
            kappa_refit = self.optimize_correction(df, sr)

            S = df["shear_suppression"].values
            correction = tep_correction(
                df["sigma_inferred"].values, sr, kappa_refit, S
            )
            mu_corr = df["value"].values + correction
            dist_corr = 10 ** ((mu_corr - 25) / 5)
            h0_corr = df["velocity"].values / dist_corr
            h0_results_refit.append(pd.Series(h0_corr).mean())
            h0_err_refit.append(pd.Series(h0_corr).sem())

            # Stronger robustness curve: keep the primary fitted kappa fixed and
            # vary only the externally defined calibrator reference.
            if fixed_kappa_cep is not None:
                fixed_correction = tep_correction(
                    df["sigma_inferred"].values, sr, fixed_kappa_cep, S
                )
                fixed_mu = df["value"].values + fixed_correction
                fixed_dist = 10 ** ((fixed_mu - 25) / 5)
                fixed_h0 = df["velocity"].values / fixed_dist
                h0_results_fixed.append(pd.Series(fixed_h0).mean())
                h0_err_fixed.append(pd.Series(fixed_h0).sem())

        # Plot
        plt.figure(figsize=(14, 9))
        if fixed_kappa_cep is not None:
            plt.plot(
                sigma_refs,
                h0_results_fixed,
                marker="o",
                color=colors["blue"],
                label=r"Fixed $\kappa_{\rm Cep}$",
                linewidth=2.5,
                zorder=3,
            )
            plt.fill_between(
                sigma_refs,
                np.array(h0_results_fixed) - np.array(h0_err_fixed),
                np.array(h0_results_fixed) + np.array(h0_err_fixed),
                color=colors["blue"],
                alpha=0.15,
                zorder=1,
            )
        plt.plot(
            sigma_refs,
            h0_results_refit,
            marker="s",
            color=colors["dark"],
            label=r"$\kappa_{\rm Cep}$ refit at each $\sigma_{\rm ref}$",
            linewidth=2.0,
            alpha=0.75,
            zorder=3,
        )
        plt.fill_between(
            sigma_refs,
            np.array(h0_results_refit) - np.array(h0_err_refit),
            np.array(h0_results_refit) + np.array(h0_err_refit),
            color=colors["dark"],
            alpha=0.10,
            zorder=1,
        )
        plt.axhline(
            planck_h0,
            color=colors["accent"],
            linestyle="--",
            label="Planck CMB",
            linewidth=2.5,
            zorder=2,
        )
        plt.fill_between(
            sigma_refs,
            planck_h0 - 0.5,
            planck_h0 + 0.5,
            color=colors["accent"],
            alpha=0.15,
            zorder=0,
        )
        
        # Add vertical marker at primary architectural sigma_ref
        sigma_ref_primary = 75.25
        plt.axvline(
            sigma_ref_primary,
            color=colors["green"],
            linestyle="-",
            linewidth=2.0,
            alpha=0.8,
            label="Primary ($\\sigma_{\\rm ref}=75.25$ km/s)",
            zorder=4,
        )

        plt.xlabel(r"Reference $\sigma_{ref}$ (km/s)")
        plt.ylabel(r"Unified $H_0$ (km/s/Mpc)")
        plt.title(r"Sensitivity of $H_0$ to Calibrator Reference (Suppression-Aware)")
        plt.legend()
        plt.tight_layout()

        path = self.figures_dir / "supplement_01_sensitivity_h0_vs_sigmaref.png"
        plt.savefig(path, dpi=300)
        print_status(f"Saved sensitivity plot to {path}", "SUCCESS")
        plt.close()

        # Copy to public
        public_path = self.public_figures_dir / "supplement_01_sensitivity_h0_vs_sigmaref.png"
        shutil.copy(path, public_path)
        print_status(f"Copied sensitivity plot to {public_path}", "SUCCESS")

        grid = pd.DataFrame({
            "sigma_ref": sigma_refs,
            "h0_refit_kappa": h0_results_refit,
            "h0_refit_err": h0_err_refit,
        })
        if fixed_kappa_cep is not None:
            grid["h0_fixed_kappa"] = h0_results_fixed
            grid["h0_fixed_err"] = h0_err_fixed
        grid_path = self.outputs_dir / "sensitivity_h0_vs_sigmaref.csv"
        grid.to_csv(grid_path, index=False)
        print_status(f"Saved sensitivity grid to {grid_path}", "SUCCESS")

        return grid

    def plot_comparison(self, df, h0_mean):
        """Generates comparison plots."""
        print_status("Generating Comparison Plots...", "PROCESS")

        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style

            colors = apply_tep_style()
        except ImportError:
            colors = {
                "blue": "#395d85",
                "accent": "#b43b4e",
                "green": "#4a2650",
                "dark": "#301E30",
                "light_blue": "#4b6785",
            }

        # Use a wide figure for side-by-side
        plt.figure(figsize=(14, 9))

        # Propagate mu uncertainty to H0: sigma_H0 = H0 * ln(10)/5 * sigma_mu
        h0_err = (
            df["h0_derived"] * (np.log(10) / 5) * df["error"]
            if "error" in df.columns
            else None
        )
        h0c_err = (
            df["h0_corrected"] * (np.log(10) / 5) * df["error"]
            if "error" in df.columns
            else None
        )

        eb_kw = dict(
            fmt="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=1.2,
            capsize=3,
            alpha=0.8,
        )

        # Original
        plt.subplot(1, 2, 1)
        if h0_err is not None:
            plt.errorbar(
                df["sigma_inferred"],
                df["h0_derived"],
                yerr=h0_err,
                color=colors.get("light_blue", "#4b6785"),
                ecolor=colors.get("light_blue", "#4b6785"),
                label="Original",
                **eb_kw,
            )
        else:
            plt.scatter(
                df["sigma_inferred"],
                df["h0_derived"],
                alpha=0.8,
                s=80,
                color=colors.get("light_blue", "#4b6785"),
                label="Original",
                edgecolor="white",
                linewidth=0.5,
            )

        if len(df) > 1:
            # Linear empirical fit to highlight the raw correlation baseline
            z = np.polyfit(df["sigma_inferred"], df["h0_derived"], 1)
            p = np.poly1d(z)
            x = np.linspace(df["sigma_inferred"].min(), df["sigma_inferred"].max(), 100)
            plt.plot(
                x,
                p(x),
                color=colors["dark"],
                linestyle="--",
                linewidth=3,
                label="Trend",
            )

        # Highlight NGC 4639 outlier in orange (consistent across figure deck)
        outlier_mask = df["normalized_name"].str.contains("4639", na=False)
        if outlier_mask.any():
            outlier = df[outlier_mask].iloc[0]
            plt.scatter(
                outlier["sigma_inferred"],
                outlier["h0_derived"],
                color="#E67E22",
                s=120,
                edgecolor="white",
                linewidth=1.5,
                zorder=5,
            )
            plt.annotate(
                "NGC 4639",
                xy=(outlier["sigma_inferred"], outlier["h0_derived"]),
                xytext=(outlier["sigma_inferred"] - 14, outlier["h0_derived"] - 4),
                fontsize=9,
                color="#E67E22",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1.0),
            )

        plt.title(rf"Original Data" + "\n" + rf"Mean $H_0$: {df['h0_derived'].mean():.2f}")
        plt.xlabel(r"Velocity Dispersion $\sigma$ (km/s)")
        plt.ylabel(r"$H_0$ (km/s/Mpc)")
        plt.ylim(55, 85)
        plt.legend()

        # Corrected
        plt.subplot(1, 2, 2)
        if h0c_err is not None:
            plt.errorbar(
                df["sigma_inferred"],
                df["h0_corrected"],
                yerr=h0c_err,
                color=colors["blue"],
                ecolor=colors["blue"],
                label="TEP Corrected",
                **eb_kw,
            )
        else:
            plt.scatter(
                df["sigma_inferred"],
                df["h0_corrected"],
                alpha=0.8,
                s=80,
                color=colors["blue"],
                label="TEP Corrected",
                edgecolor="white",
                linewidth=0.5,
            )

        if len(df) > 1:
            z2 = np.polyfit(df["sigma_inferred"], df["h0_corrected"], 1)
            p2 = np.poly1d(z2)
            slope_corrected = z2[0]
            
            # Calculate correlation and p-value for corrected data
            r_corr, p_corr = stats.pearsonr(df["sigma_inferred"], df["h0_corrected"])
            
            plt.plot(
                x,
                p2(x),
                color=colors["blue"],
                linestyle="--",
                linewidth=3,
                label=r"Trend, $r \simeq 0$ (fitted correction)",
            )

        # Highlight NGC 4639 outlier in orange (consistent across figure deck)
        if outlier_mask.any():
            plt.scatter(
                outlier["sigma_inferred"],
                outlier["h0_corrected"],
                color="#E67E22",
                s=120,
                edgecolor="white",
                linewidth=1.5,
                zorder=5,
            )

        plt.axhspan(
            66.9,
            67.9,
            alpha=0.12,
            color=colors["accent"],
            label="Planck CMB $1\\sigma$ band",
        )
        plt.axhline(
            67.4,
            color=colors["accent"],
            linestyle=":",
            linewidth=2.5,
            label="Planck CMB",
        )
        plt.title(rf"TEP Corrected" + "\n" + rf"Mean $H_0$: {h0_mean:.2f}")
        plt.xlabel(r"Velocity Dispersion $\sigma$ (km/s)")
        plt.ylim(55, 85)
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved comparison plot to {self.plot_path}", "SUCCESS")
        plt.close()

        # Copy to public
        public_path = self.public_figures_dir / "figure_03_tep_correction_comparison.png"
        shutil.copy(self.plot_path, public_path)
        print_status(f"Copied comparison plot to {public_path}", "SUCCESS")

    def screened_variant_analysis(self, df, sigma_ref_screened):
        """Re-optimise κ_Cep and compute H₀ using the screened-effective σ_ref.

        Returns dict with screened-variant kappa, H0 mean, and H0 SEM.
        """
        print_status(
            "Screened-Effective Variant (σ_ref,scr = "
            f"{sigma_ref_screened:.2f} km/s)...",
            "SECTION",
        )
        kappa_scr = self.optimize_correction(df, sigma_ref_screened)
        _, h0_mean_scr, h0_sem_scr = self.apply_correction(
            df.copy(), kappa_scr, sigma_ref_screened
        )
        print_status(
            f"Screened Variant: κ_Cep = {kappa_scr:.3e} mag, "
            f"H0 = {h0_mean_scr:.2f} km/s/Mpc",
            "INFO",
        )
        return {
            "kappa_cep_screened": float(kappa_scr),
            "sigma_ref_screened": float(sigma_ref_screened),
            "unified_h0_screened": float(h0_mean_scr),
            "h0_sem_screened": float(h0_sem_scr),
        }

    def slope_convention_audit(self, df, kappa_cep, sigma_ref):
        """Sign-convention audit to prevent old negative-slope language.

        Checks three sign relationships that must hold for the TEP
        correction to be physically consistent:
        1. raw H0 vs σ slope is positive (deep potential → high H0)
        2. correction (H0_corr − H0_raw) vs σ slope is negative
           (high-σ hosts are corrected downward)
        3. distance-modulus correction Δμ is positive for high-σ hosts
           (period contraction makes stars appear brighter → add to μ)
        """
        sigma_vals = df["sigma_inferred"].values.astype(float)
        h0_raw = df["h0_derived"].values.astype(float)
        h0_corr = df["h0_corrected"].values.astype(float)
        correction_delta = h0_corr - h0_raw

        raw_slope, _ = np.polyfit(sigma_vals, h0_raw, 1)
        corr_delta_slope, _ = np.polyfit(sigma_vals, correction_delta, 1)

        S = (
            df["shear_suppression"].values.astype(float)
            if "shear_suppression" in df.columns
            else np.ones(len(df))
        )
        from scripts.utils.tep_correction import tep_correction
        dmu = tep_correction(sigma_vals, sigma_ref, kappa_cep, S)
        high_sigma_mask = sigma_vals > sigma_ref
        dmu_high_mean = np.mean(dmu[high_sigma_mask]) if np.any(high_sigma_mask) else np.nan

        audit = {
            "raw_slope_H0_sigma_positive": bool(raw_slope > 0),
            "correction_slope_H0_sigma_negative": bool(corr_delta_slope < 0),
            "distance_modulus_correction_high_sigma_positive": bool(dmu_high_mean > 0),
            "raw_slope_value": float(raw_slope),
            "correction_delta_slope_value": float(corr_delta_slope),
            "dmu_high_sigma_mean": float(dmu_high_mean),
        }
        all_ok = all([
            audit["raw_slope_H0_sigma_positive"],
            audit["correction_slope_H0_sigma_negative"],
            audit["distance_modulus_correction_high_sigma_positive"],
        ])

        print_status("Slope Convention Audit", "SECTION")
        headers = ["Check", "Value", "Status"]
        rows = [
            ["raw slope H0–σ > 0", f"{raw_slope:+.4f}", "PASS" if audit["raw_slope_H0_sigma_positive"] else "FAIL"],
            ["correction slope < 0", f"{corr_delta_slope:+.4f}", "PASS" if audit["correction_slope_H0_sigma_negative"] else "FAIL"],
            ["Δμ(high σ) > 0", f"{dmu_high_mean:+.4f} mag", "PASS" if audit["distance_modulus_correction_high_sigma_positive"] else "FAIL"],
        ]
        print_table(headers, rows)
        if not all_ok:
            print_status("SLOPE CONVENTION AUDIT FAILED — sign inconsistency detected!", "CRITICAL")
            raise RuntimeError("Slope convention audit failed: inconsistent signs.")
        print_status("Slope convention audit PASSED.", "SUCCESS")
        return audit

    def run(self):
        print_status("Starting Step 3: TEP Correction", "TITLE")

        df = self.load_data()

        # 1. Dynamic Sigma Ref (both standard and screened-effective)
        sigma_ref, sigma_ref_screened = self.calculate_effective_calibrator_sigma()

        # 2. Optimize (standard σ_ref — primary headline)
        kappa_cep = self.optimize_correction(df, sigma_ref)

        # 3. Apply (standard)
        final_df, h0_mean, h0_sem = self.apply_correction(df, kappa_cep, sigma_ref)

        # 4. Slope Convention Audit (must pass before proceeding)
        audit = self.slope_convention_audit(final_df, kappa_cep, sigma_ref)

        # 5. Generate Comparison Plot
        self.plot_comparison(final_df, h0_mean)

        # 6. Joint Bootstrap (host resampling + kappa refit per resample)
        # This combines host-to-host sampling variance AND kappa parameter
        # uncertainty into a single honest H0 uncertainty.
        boot_metrics = self.bootstrap_analysis(final_df, sigma_ref)

        # 7. Sensitivity
        self.sensitivity_analysis(final_df, fixed_kappa_cep=kappa_cep)

        # 8. Screened-Effective Variant
        screened_results = self.screened_variant_analysis(df.copy(), sigma_ref_screened)

        # Stability check
        delta_h0_screened = abs(h0_mean - screened_results["unified_h0_screened"])
        print_status(
            f"Screened-Reference Stability: ΔH0 = {delta_h0_screened:.2f} km/s/Mpc "
            f"(standard {h0_mean:.2f} vs screened {screened_results['unified_h0_screened']:.2f})",
            "SUCCESS" if delta_h0_screened < 1.0 else "WARNING",
        )

        # Error Budget Summary:
        #   h0_sem        : statistical SEM assuming kappa exactly known (too small)
        #   bootstrap_h0_std: full uncertainty including host scatter + kappa fit (primary)
        primary_error = boot_metrics["bootstrap_h0_std"]

        # Planck Comparison
        planck_h0 = 67.4
        planck_err = 0.5

        # Tension Calculations
        tension_stat = abs(h0_mean - planck_h0) / np.sqrt(h0_sem**2 + planck_err**2)
        tension_primary = abs(h0_mean - planck_h0) / np.sqrt(
            primary_error**2 + planck_err**2
        )

        print_status("Final Tension Analysis", "SECTION")
        print_status(f"Planck 2018 Value: {planck_h0} +/- {planck_err}", "INFO")
        print_status("-" * 60, "INFO")
        print_status(f"TEP Unified Value: {h0_mean:.2f}", "INFO")
        print_status(f"  +/- {h0_sem:.2f} (Statistical SEM, kappa fixed)", "INFO")
        print_status(
            f"  +/- {primary_error:.2f} (Joint Bootstrap: host scatter + kappa)",
            "INFO",
        )
        print_status(
            f"  kappa_Cep = ({boot_metrics['bootstrap_kappa_mean']:.2e}) +/- "
            f"({boot_metrics['bootstrap_kappa_std']:.2e})  "
            f"[{boot_metrics['bootstrap_kappa_std']/boot_metrics['bootstrap_kappa_mean']*100:.0f}%]",
            "INFO",
        )
        print_status("-" * 60, "INFO")
        print_status(f"Tension (Statistical):     {tension_stat:.2f} sigma", "INFO")
        print_status(f"Tension (Joint Bootstrap): {tension_primary:.2f} sigma", "RESULT")

        if tension_primary < 1.0:
            print_status(
                "CONCLUSION (Cepheid channel, unscreened Hubble-flow regime): "
                "H0 consistent with Planck CMB after κ_Cep correction.",
                "SUCCESS",
            )
        elif tension_primary < 2.0:
            print_status(
                "CONCLUSION (Cepheid channel): marginal tension < 2σ; plausible "
                "consistency with Planck after κ_Cep correction.",
                "WARNING",
            )
        else:
            print_status(
                "CONCLUSION (Cepheid channel): significant residual tension after "
                "κ_Cep correction.",
                "WARNING",
            )
        print_status(
            "Note: Universal-TEP validation requires CROSS-CHANNEL consistency "
            "(κ_Cep vs κ_TRGB vs κ_SN vs κ_pulsar). Single-channel significance "
            "of κ_Cep from zero is NOT the TEP test (Jakarta §7; Istanbul §1.3).",
            "INFO",
        )

        # Compile and Save Results
        results = {
            "optimal_kappa_cep": float(kappa_cep),
            "sigma_ref": float(sigma_ref),
            "unified_h0": float(h0_mean),
            "h0_sem": float(h0_sem),
            "bootstrap_h0_mean": float(boot_metrics["bootstrap_h0_mean"]),
            "bootstrap_h0_std": float(boot_metrics["bootstrap_h0_std"]),
            "bootstrap_h0_ci_lower": float(boot_metrics["bootstrap_h0_ci_lower"]),
            "bootstrap_h0_ci_upper": float(boot_metrics["bootstrap_h0_ci_upper"]),
            "bootstrap_kappa_mean": float(boot_metrics["bootstrap_kappa_mean"]),
            "bootstrap_kappa_std": float(boot_metrics["bootstrap_kappa_std"]),
            "bootstrap_kappa_median": float(boot_metrics["bootstrap_kappa_median"]),
            "bootstrap_kappa_robust_std": float(boot_metrics["bootstrap_kappa_robust_std"]),
            "bootstrap_kappa_ci_lower": float(boot_metrics["bootstrap_kappa_ci_lower"]),
            "bootstrap_kappa_ci_upper": float(boot_metrics["bootstrap_kappa_ci_upper"]),
            "bootstrap_kappa_skewness": float(boot_metrics["bootstrap_kappa_skewness"]),
            "bootstrap_kappa_n_negative": int(boot_metrics["bootstrap_kappa_n_negative"]),
            "bootstrap_h0_median": float(boot_metrics["bootstrap_h0_median"]),
            "bootstrap_h0_robust_std": float(boot_metrics["bootstrap_h0_robust_std"]),
            "bootstrap_residual_slope_mean": float(
                boot_metrics["bootstrap_residual_slope_mean"]
            ),
            "bootstrap_n_converged": int(boot_metrics["bootstrap_n_converged"]),
            "bootstrap_n_failed": int(boot_metrics["bootstrap_n_failed"]),
            "planck_h0": float(planck_h0),
            "tension_sigma": float(tension_primary),
            "tension_statistical": float(tension_stat),
            "is_consistent": bool(tension_primary < 2.0),
            "n_hosts": len(final_df),
            # Slope convention audit
            "slope_audit": audit,
            # Screening formula metadata
            "screening_formula": "S_group(N_mb) = [1 + (N_mb / N_crit)^gamma]^{-1}",
            "screening_n_crit": 10.0,
            "screening_gamma": 1.2,
            "anchor_nmb": dict(ANCHOR_NMB),
            "anchor_screening": dict(ANCHOR_SCREENING),
            # Screened-effective variant
            "sigma_ref_screened": screened_results["sigma_ref_screened"],
            "optimal_kappa_cep_screened": screened_results["kappa_cep_screened"],
            "unified_h0_screened": screened_results["unified_h0_screened"],
            "h0_screened_sem": screened_results["h0_sem_screened"],
            "delta_h0_screened": float(delta_h0_screened),
        }

        with open(self.json_output_path, "w") as f:
            json.dump(results, f, indent=4)
        print_status(f"Saved results JSON to {self.json_output_path}", "SUCCESS")

        final_df.to_csv(self.corrected_output_path, index=False)
        print_status(f"Saved corrected data to {self.corrected_output_path}", "SUCCESS")


def main():
    step = Step3TEPCorrection()
    step.run()


if __name__ == "__main__":
    main()
