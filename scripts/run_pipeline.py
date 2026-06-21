#!/usr/bin/env python3
"""
TEP-H0 Analysis Pipeline Master Script
======================================
Orchestrates the full analysis pipeline for Paper 11: "The Cepheid Bias: Resolving the Hubble Tension".

This script serves as the central controller for the TEP-H0 analysis.
It executes the scientific workflow in a strictly ordered sequence, ensuring
data integrity and dependency management between steps.

Workflow Steps:
1.  **Data Ingestion**: Downloads raw data (SH0ES, Pantheon+), reconstructs catalogs, 
    and cross-matches hosts with external databases (Simbad, HyperLEDA).
2.  **Stratification**: Calculates H0 for each host, stratifies the sample by 
    gravitational potential (velocity dispersion), and detects the environmental bias.
3.  **TEP Correction**: Optimizes the Observable Response Coefficient (kappa_cep), applies the 
    conformal time correction, and unifies the Hubble Constant.
4.  **Robustness Checks**: Performs rigorous statistical tests (Jackknife, Bivariate 
    Analysis, Sensitivity Analysis) to validate the results against systematics.
5.  **M31 Analysis**: Executes a differential test on M31 Cepheids to verify the 
    environmental P-L dependence in a controlled setting.

Usage:
    python scripts/run_pipeline.py

Author: Matthew Lukin Smawfield
Date: January 2026
"""

import sys
import time
from pathlib import Path
import traceback
import argparse

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
from scripts.steps.step_00_sigma_catalog import Step0SigmaCatalog
from scripts.steps.step_01_data_ingestion import Step1DataIngestion
from scripts.utils.fetch_metadata import fetch_galaxy_metadata
from scripts.steps.step_02_aperture_correction import Step1bApertureCorrection
from scripts.steps.step_03_stratification import Step2Stratification
from scripts.steps.step_04_tep_correction import Step3TEPCorrection
from scripts.steps.step_05_prespecified_predictions import Step14FrozenPredictions
from scripts.steps.step_06_shear_suppression_viz import main as step2b_viz
from scripts.steps.step_07_aperture_sensitivity import Step4bApertureSensitivity
from scripts.steps.step_08_robustness_checks import Step4RobustnessChecks
from scripts.steps.step_09_hierarchical_sigma import Step09HierarchicalSigma
from scripts.steps.step_10_m31_analysis import Step5M31Analysis
from scripts.steps.step_11_m31_radial_suppression import main as step_11_m31_radial_suppression_main
from scripts.steps.step_12_multivariate_analysis import Step6MultivariateAnalysis
from scripts.steps.step_13_enhanced_robustness import Step6EnhancedRobustness
from scripts.steps.step_14_lmc_replication import Step7LMCReplication
from scripts.steps.step_15_trgb_comparison import Step7TRGBComparison
from scripts.steps.step_16_trgb_reanalysis import Step7TRGBReanalysis
from scripts.steps.step_17_host_mass_residual import Step16HostMassResidual
from scripts.steps.step_18_regressor_audit import Step17RegressorAudit
from scripts.steps.step_19_group_env_models import Step18GroupEnvModels
from scripts.steps.step_20_joint_indicator_model import Step19JointIndicatorModel
from scripts.steps.step_21_stratified_validation import Step20StratifiedValidation
from scripts.steps.step_22_exact_sigma_ref import Step21ExactSigmaRef
from scripts.steps.step_23_sn_residual_test import Step22SNResidualTest
from scripts.steps.step_24_synthetic_injection import Step23SyntheticInjection
from scripts.steps.step_25_leave_one_out import Step24LeaveOneOut
from scripts.steps.step_26_m31_phat_analysis import Step8M31PHATAnalysis
from scripts.steps.step_27_anchor_stratification import AnchorStratificationStep
from scripts.steps.step_28_local_gravity_closure import Step10bLocalGravityClosure
from scripts.steps.step_29_cross_channel import Step12CrossChannel
from scripts.steps.step_30_cosmology_inference import main as step_12_cosmology_inference_main
from scripts.steps.step_31_final_synthesis import Step9FinalSynthesis
from scripts.steps.step_32_comprehensive_audit import Step11ComprehensiveAudit
from scripts.steps.step_33_stellar_validation import Step13StellarValidation
from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood
from scripts.steps.step_35_bias_aware_tep_ladder import run as step_35_run
from scripts.steps.step_36_apparent_hubble_environment_likelihood import run as step_36_run
from scripts.steps.step_37_velocity_robustness import run as step_37_run
from scripts.steps.step_38_hierarchical_timefield_ladder import run as step_38_run
from scripts.steps.step_39_environment_slope_decomposition import run as step_39_run
from scripts.steps.step_40_flow_sky_controls import run as step_40_run
from scripts.steps.step_41_external_distance_breakers import run as step_41_run
from scripts.steps.step_42_tep_native_ladder import run as step_42_run
from scripts.steps.step_43_toy_recovery_experiment import run as step_43_run
from scripts.utils.pipeline_audit import audit

def regression_gates(project_root):
    """Hard regression gates for key known outputs. Raises RuntimeError on failure."""
    import json
    from pathlib import Path
    import numpy as np

    results_dir = Path(project_root) / "results" / "outputs"
    gates_passed = 0
    gates_total = 0

    def check(label, condition, msg=""):
        nonlocal gates_passed, gates_total
        gates_total += 1
        if condition:
            gates_passed += 1
            print_status(f"  [PASS] {label}", "SUCCESS")
        else:
            print_status(f"  [FAIL] {label}: {msg}", "ERROR")

    print_status(">>> REGRESSION GATES", "TITLE")

    # Step 34: Full ladder likelihood
    try:
        s34 = json.loads((results_dir / "step_34_full_ladder_likelihood_results.json").read_text())
        H0 = s34.get("baseline", {}).get("H0", 0)
        stage2 = s34.get("stage2", {})
        variant_a = stage2.get("variant_a_free_kappa", {})
        kappa = variant_a.get("kappa_Cep", 0)
        kappa_err = variant_a.get("kappa_err", 1)
        comparison = s34.get("comparison", {})
        delta_chi2 = comparison.get("delta_chi2_vs_baseline", {})
        fixed_chi2 = delta_chi2.get("4", 0) if isinstance(delta_chi2, dict) else 0
        check("S34 H0 ≈ 73.04", abs(H0 - 73.04) < 1.0, f"H0={H0}")
        check("S34 κ_Cep ≈ −0.067e6 ± 0.210e6", abs(kappa + 0.067e6) < 0.5e6 and abs(kappa_err - 0.210e6) < 0.5e6,
              f"κ={kappa:.3e} ± {kappa_err:.3e}")
        check("S34 fixed canonical χ² penalty ≈ +24.2", abs(fixed_chi2 - 24.2) < 10.0, f"penalty={fixed_chi2:.1f}")
    except Exception as e:
        check("S34 outputs exist", False, str(e))

    # Step 37: Velocity robustness
    try:
        s37 = json.loads((results_dir / "step_37_velocity_robustness.json").read_text())
        primary = [r for r in s37 if r.get("test") == "standard" and r.get("sigma_v") == 250 and r.get("model") == "1_X"]
        if primary:
            r = primary[0]
            beta = r.get("beta_X", 0)
            N = r.get("N_hosts", 0)
            check("S37 primary N=29", N == 29, f"N={N}")
            check("S37 β_X > 0 at σ_v=250", beta > 0, f"β={beta:.3e}")
        else:
            check("S37 primary results found", False, "no standard/250/1_X result")
    except Exception as e:
        check("S37 outputs exist", False, str(e))

    # Step 39: Environment slope decomposition
    try:
        s39 = json.loads((results_dir / "step_39_environment_slope_decomposition.json").read_text())
        if isinstance(s39, list):
            s39_rec = [r for r in s39 if r.get("sigma_v") == 250 and r.get("sample") == "primary" and r.get("z_cut", 0) == 0]
            s39_rec = s39_rec[0] if s39_rec else {}
        gamma = s39_rec.get("Gamma_X", 0)
        kappa_eq = s39_rec.get("kappa_equiv", 0)
        # LOHO is in separate statistical tests file
        s39_tests = json.loads((results_dir / "step_39_statistical_tests.json").read_text())
        loho_pos = s39_tests.get("loho", {}).get("n_positive", 0)
        check("S39 Γ_X ≈ +2.3e7 at σ_v=250", abs(gamma - 2.3e7) < 0.5e7, f"Γ={gamma:.3e}")
        check("S39 κ_equiv ≈ +7.2e5", abs(kappa_eq - 7.2e5) < 2.0e5, f"κ_eq={kappa_eq:.3e}")
        check("S39 LOHO positive 29/29", loho_pos == 29, f"positive={loho_pos}/29")
    except Exception as e:
        check("S39 outputs exist", False, str(e))

    # Step 40: Flow/sky controls
    try:
        s40 = json.loads((results_dir / "step_40_flow_sky_controls.json").read_text())
        primary_250 = [r for r in s40 if r.get("sample") == "primary" and r.get("sigma_v") == 250 and r.get("model") != "M0"]
        all_positive = all(r.get("Gamma_X", 0) > 0 for r in primary_250)
        m1 = [r for r in primary_250 if r.get("model") == "M1"]
        m1_aic = m1[0].get("AIC", 999) if m1 else 999
        other_aic = [r.get("AIC", 0) for r in primary_250 if r.get("model") != "M1"]
        m1_best = all(m1_aic < a for a in other_aic)
        check("S40 Γ_X > 0 in all controlled models", all_positive, "")
        check("S40 M1 has best AIC", m1_best, f"M1 AIC={m1_aic:.1f}")
    except Exception as e:
        check("S40 outputs exist", False, str(e))

    # Step 41: External distance breakers
    try:
        s41 = json.loads((results_dir / "step_41_external_distance_breakers.json").read_text())
        N_merged = s41.get("N_merged_hosts", 0)
        diff = s41.get("differential_kappa", {})
        kappa_diff = diff.get("kappa_Cep", 0)
        kappa_sig = diff.get("kappa_Cep_sig", 0)
        vel = s41.get("velocity_beta", [])
        # Check Γ_X stability across κ assumptions at sigma_v=250
        gamma_250 = []
        for r in vel:
            if r.get("sigma_v") == 250:
                kappa_label = r.get("kappa_label", "")
                beta = r.get("beta_X", 0)
                kappa_used = r.get("kappa_used", 0)
                # Reconstruct Γ_X ≈ beta + (ln10/5)*70*kappa
                gamma_est = beta + (np.log(10) / 5) * 70 * kappa_used
                gamma_250.append(gamma_est)
        gamma_stable = all(abs(g - 2.3e7) < 0.5e7 for g in gamma_250) if gamma_250 else False
        check("S41 TRGB overlap N=13", N_merged == 13, f"N={N_merged}")
        check("S41 differential κ consistent with zero", abs(kappa_diff) < 1.0e6, f"κ={kappa_diff:.3e} ({kappa_sig:.1f}σ)")
        check("S41 Γ_X stable across κ assumptions", gamma_stable, f"Γ range={gamma_250}")
    except Exception as e:
        check("S41 outputs exist", False, str(e))

    # Step 42: TEP-native generative model
    try:
        s42 = json.loads((results_dir / "step_42_tep_native_ladder.json").read_text())
        tgamma_250 = [r for r in s42 if r.get("sigma_v") == 250 and r.get("model") == "TGamma"]
        if tgamma_250:
            r = tgamma_250[0]
            gamma = r.get("Gamma_X", 0)
            gamma_sig = r.get("Gamma_X_sig", 0)
            beta_A = r.get("beta_A", 0)
            kappa_B = r.get("kappa_B", 0)
            beta_C = r.get("beta_C", 0)
            kappa_D = r.get("kappa_D", np.nan)
            beta_D = r.get("beta_D", np.nan)
            check("S42 Γ_X positive at σ_v=250", gamma > 0, f"Γ={gamma:.3e}")
            check("S42 Γ_X in expected range", 2.0e7 < gamma < 2.7e7, f"Γ={gamma:.3e}")
            check("S42 Γ_X > 1.5σ", gamma_sig > 1.5, f"sig={gamma_sig:.1f}σ")
            check("S42 Gauge A β == Γ_X", abs(beta_A - gamma) < 1e3, f"β_A={beta_A:.3e}, Γ={gamma:.3e}")
            check("S42 Gauge B κ_equiv ≈ 7.3e5", 6.0e5 < kappa_B < 9.0e5, f"κ_B={kappa_B:.3e}")
            check("S42 Gauge C β < 0 (canonical κ)", beta_C < 0, f"β_C={beta_C:.3e}")
            check("S42 Gauge D κ_ext ≈ 3.2e5", not np.isnan(kappa_D) and 0 < kappa_D < 8.0e5, f"κ_D={kappa_D:.3e}")
            check("S42 Gauge D β > 0", not np.isnan(beta_D) and beta_D > 0, f"β_D={beta_D:.3e}")
        else:
            check("S42 TGamma results found", False, "no TGamma/250 result")
    except Exception as e:
        check("S42 outputs exist", False, str(e))

    # Step 43: Toy recovery experiment
    try:
        s43 = json.loads((results_dir / "step_43_toy_recovery_experiment.json").read_text())
        N_primary = s43.get("N_primary", 0)
        inj = s43.get("injection", {})
        k_inj = float(inj.get("kappa_Cep_injected", np.nan))
        g_inj = float(inj.get("Gamma_X_injected", np.nan))
        dm = s43.get("design_matrix", {})
        vs = s43.get("velocity_space", {})
        k_hat = float(dm.get("kappa_Cep_recovered", np.nan))
        g_hat = float(vs.get("Gamma_X_recovered", np.nan))
        check("S43 primary N=29", N_primary == 29, f"N={N_primary}")
        check("S43 injected κ_Cep positive", np.isfinite(k_inj) and k_inj > 0, f"κ_inj={k_inj:.3e}")
        check("S43 injected Γ_X positive", np.isfinite(g_inj) and g_inj > 0, f"Γ_inj={g_inj:.3e}")
        check("S43 Step-34 κ_Cep absorbed (≈0)", np.isfinite(k_hat) and abs(k_hat) < 0.2 * abs(k_inj), f"κ_hat={k_hat:.3e}")
        check("S43 velocity-space recovers Γ_X", np.isfinite(g_hat) and abs(g_hat - g_inj) < 0.35 * abs(g_inj), f"Γ_hat={g_hat:.3e}")
    except Exception as e:
        check("S43 outputs exist", False, str(e))

    print_status(f"Regression gates: {gates_passed}/{gates_total} passed", "INFO")
    if gates_passed < gates_total:
        raise RuntimeError(f"Regression gates failed: {gates_total - gates_passed} failures.")
    print_status("All regression gates passed.", "SUCCESS")


def run_pipeline():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--skip-sigma-step", action="store_true")
    ap.add_argument("--rebuild-sigma", action="store_true")
    ap.add_argument("--use-lit-overrides", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--run-stellar-validation", action="store_true",
                    help="Run Step 33: MESA/RSP stellar validation (optional, post-pipeline)")
    args = ap.parse_args()

    # Setup Global Logger
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # We use a distinct logger for the pipeline orchestration
    pipeline_logger = TEPLogger("pipeline_master", log_file_path=logs_dir / "pipeline_master.log")
    set_step_logger(pipeline_logger)
    
    print_status("TEP-H0 analysis pipeline initiated", "TITLE")
    print_status(f"Project Root: {PROJECT_ROOT}", "INFO")
    print_status("Starting execution sequence...", "INFO")
    
    start_time = time.time()
    step_times = {}
    
    try:
        # --- Step 01 (Pre-flight): Prepare Coordinates ---
        # Needed for Step 00 (Sigma Catalog Build) to know which galaxies to query
        print_status(">>> Pre-flight: Coordinate preparation", "TITLE")
        step1_pre = Step1DataIngestion()
        step1_pre.prepare_coordinates()

        # --- Step 00: Sigma Catalog Build (Provenance) ---
        if not args.skip_sigma_step:
            print_status(">>> STEP 00: Sigma Catalog (provenance build)", "TITLE")
            t0 = time.time()
            Step0SigmaCatalog().run(rebuild=bool(args.rebuild_sigma))
            step_times['Step 00'] = time.time() - t0
            set_step_logger(pipeline_logger)
            print_status("Step 00 (Sigma Catalog) completed successfully.", "SUCCESS")

        # --- Step 01: Data Ingestion ---
        print_status(">>> STEP 01: Data Ingestion", "TITLE")
        t0 = time.time()
        step1 = Step1DataIngestion()
        step1.run()
        step_times['Step 01'] = time.time() - t0
        
        # Reset logger to master after step completion (step scripts set their own)
        set_step_logger(pipeline_logger)
        print_status("Step 01 (Ingestion) completed successfully.", "SUCCESS")
        
        # --- Step 02: Aperture Correction ---
        print_status(">>> STEP 02: Aperture Correction", "TITLE")
        t0 = time.time()
        
        # Sub-task: Fetch Metadata (RC3 Sizes)
        print_status("Fetching host metadata (RC3) for aperture normalization...", "PROCESS")
        fetch_galaxy_metadata()
        
        # Sub-task: Apply Correction
        step1b = Step1bApertureCorrection()
        step1b.run()
        step_times['Step 02'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 02 (Aperture Correction) completed successfully.", "SUCCESS")
        
        # --- Step 03: Stratification ---
        print_status(">>> STEP 03: Stratification", "TITLE")
        t0 = time.time()
        step2 = Step2Stratification()
        step2.run()
        step_times['Step 03'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 03 (Stratification) completed successfully.", "SUCCESS")

        # --- Step 04: TEP Correction ---
        print_status(">>> STEP 04: TEP Correction", "TITLE")
        t0 = time.time()
        step3 = Step3TEPCorrection()
        step3.run()
        step_times['Step 04'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 04 (Optimization) completed successfully.", "SUCCESS")

        # --- Step 05: Frozen TEP Prediction Table ---
        print_status(">>> STEP 05: Frozen TEP Prediction Table", "TITLE")
        t0 = time.time()
        Step14FrozenPredictions().run()
        step_times['Step 05'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 05 (Frozen Predictions) completed successfully.", "SUCCESS")

        # --- Step 06: Shear Suppression Visualization ---
        print_status(">>> STEP 06: Shear Suppression Visualization", "TITLE")
        t0 = time.time()
        step2b_viz()
        step_times['Step 06'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 06 (Shear Suppression Viz) completed successfully.", "SUCCESS")

        # --- Step 07: Aperture Sensitivity Analysis ---
        print_status(">>> STEP 07: Aperture Sensitivity Analysis", "TITLE")
        t0 = time.time()
        Step4bApertureSensitivity().run()
        step_times['Step 07'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 07 (Aperture Sensitivity) completed successfully.", "SUCCESS")

        # --- Step 08: Robustness Checks ---
        print_status(">>> STEP 08: Robustness Checks", "TITLE")
        t0 = time.time()
        Step4RobustnessChecks().run()
        step_times['Step 08'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 08 (Robustness) completed successfully.", "SUCCESS")

        # --- Step 09: Hierarchical Sigma Measurement-Error Model ---
        print_status(">>> STEP 09: Hierarchical Sigma Measurement-error Model", "TITLE")
        t0 = time.time()
        Step09HierarchicalSigma().run()
        step_times['Step 09'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 09 (Hierarchical Sigma) completed successfully.", "SUCCESS")

        # --- Step 10: M31 Analysis ---
        print_status(">>> STEP 10: M31 Analysis", "TITLE")
        t0 = time.time()
        step5 = Step5M31Analysis()
        step5.run()
        step_times['Step 10'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 10 (M31 Differential Test) completed successfully.", "SUCCESS")

        # --- Step 11: M31 Radial Suppression ---
        print_status(">>> STEP 11: M31 Radial Suppression", "TITLE")
        t0 = time.time()
        step_11_m31_radial_suppression_main()
        step_times['Step 11'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 11 (M31 Radial Suppression) completed successfully.", "SUCCESS")

        # --- Step 12: Multivariate Analysis ---
        print_status(">>> STEP 12: Multivariate Analysis", "TITLE")
        t0 = time.time()
        step6 = Step6MultivariateAnalysis()
        step6.run()
        step_times['Step 12'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 12 (Multivariate Analysis) completed successfully.", "SUCCESS")

        # --- Step 13: Enhanced Robustness (Referee-Facing) ---
        print_status(">>> STEP 13: Enhanced Robustness", "TITLE")
        t0 = time.time()
        Step6EnhancedRobustness().run()
        step_times['Step 13'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 13 (Enhanced Robustness) completed successfully.", "SUCCESS")

        # --- Step 14: LMC Replication ---
        print_status(">>> STEP 14: LMC Replication", "TITLE")
        t0 = time.time()
        step7 = Step7LMCReplication()
        step7.run()
        step_times['Step 14'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 14 (LMC Replication) completed successfully.", "SUCCESS")

        # --- Step 15: TRGB Comparison ---
        print_status(">>> STEP 15: TRGB Comparison", "TITLE")
        t0 = time.time()
        Step7TRGBComparison().run()
        step_times['Step 15'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 15 (TRGB Comparison) completed successfully.", "SUCCESS")

        # --- Step 16: TRGB Differential Reanalysis ---
        print_status(">>> STEP 16: TRGB Differential Reanalysis", "TITLE")
        t0 = time.time()
        Step7TRGBReanalysis().run()
        step_times['Step 16'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 16 (TRGB Differential Reanalysis) completed successfully.", "SUCCESS")

        # --- Step 17: Host-Mass Residual Test ---
        print_status(">>> STEP 17: Host-mass Residual Test", "TITLE")
        t0 = time.time()
        Step16HostMassResidual().run()
        step_times['Step 17'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 17 (Host-Mass Residual) completed successfully.", "SUCCESS")

        # --- Step 18: Regressor Audit ---
        print_status(">>> STEP 18: TEP Regressor Audit", "TITLE")
        t0 = time.time()
        Step17RegressorAudit().run()
        step_times['Step 18'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 18 (Regressor Audit) completed successfully.", "SUCCESS")

        # --- Step 19: Group Environment Model Comparison ---
        print_status(">>> STEP 19: Group Environment Model Comparison", "TITLE")
        t0 = time.time()
        Step18GroupEnvModels().run()
        step_times['Step 19'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 19 (Group Env Models) completed successfully.", "SUCCESS")

        # --- Step 20: Joint Cepheid+TRGB Indicator Model ---
        print_status(">>> STEP 20: Joint Cepheid+trgb Indicator Model", "TITLE")
        t0 = time.time()
        Step19JointIndicatorModel().run()
        step_times['Step 20'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 20 (Joint Indicator) completed successfully.", "SUCCESS")

        # --- Step 21: Physically Stratified Validation ---
        print_status(">>> STEP 21: Physically Stratified Validation", "TITLE")
        t0 = time.time()
        Step20StratifiedValidation().run()
        step_times['Step 21'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 21 (Stratified Validation) completed successfully.", "SUCCESS")

        # --- Step 22: Exact Anchor-Leverage Sigma_ref ---
        print_status(">>> STEP 22: Exact Anchor-leverage Sigma_ref", "TITLE")
        t0 = time.time()
        Step21ExactSigmaRef().run()
        step_times['Step 22'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 22 (Exact Sigma_ref) completed successfully.", "SUCCESS")

        # --- Step 23: SN Ia Downstream Residual Test ---
        print_status(">>> STEP 23: Sn Ia Downstream Residual Test", "TITLE")
        t0 = time.time()
        Step22SNResidualTest().run()
        step_times['Step 23'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 23 (SN Residual) completed successfully.", "SUCCESS")

        # --- Step 24: Synthetic Injection Recovery ---
        print_status(">>> STEP 24: Synthetic Injection Recovery", "TITLE")
        t0 = time.time()
        Step23SyntheticInjection().run()
        step_times['Step 24'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 24 (Synthetic Injection) completed successfully.", "SUCCESS")

        # --- Step 25: Leave-One-Out Influence Analysis ---
        print_status(">>> STEP 25: Leave-one-out Influence Analysis", "TITLE")
        t0 = time.time()
        Step24LeaveOneOut().run()
        step_times['Step 25'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 25 (Leave-One-Out) completed successfully.", "SUCCESS")

        # --- Step 26: M31 PHAT Analysis ---
        print_status(">>> STEP 26: M31 Phat Analysis", "TITLE")
        t0 = time.time()
        step8 = Step8M31PHATAnalysis()
        step8.run()
        step_times['Step 26'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 26 (M31 PHAT Analysis) completed successfully.", "SUCCESS")

        # --- Step 27: Anchor Stratification Test ---
        print_status(">>> STEP 27: Anchor Stratification Test", "TITLE")
        t0 = time.time()
        step10 = AnchorStratificationStep()
        step10.run()
        step_times['Step 27'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 27 (Anchor Stratification) completed successfully.", "SUCCESS")

        # --- Step 28: Local Gravity Closure ---
        print_status(">>> STEP 28: Local Gravity Closure", "TITLE")
        t0 = time.time()
        Step10bLocalGravityClosure().run()
        step_times['Step 28'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 28 (Local Gravity Closure) completed successfully.", "SUCCESS")

        # --- Step 29: Cross-Channel Consistency ---
        print_status(">>> STEP 29: Cross-channel Consistency", "TITLE")
        t0 = time.time()
        Step12CrossChannel().run()
        step_times['Step 29'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 29 (Cross-Channel) completed successfully.", "SUCCESS")

        # --- Step 30: Cosmology Inference ---
        print_status(">>> STEP 30: Cosmology Inference", "TITLE")
        t0 = time.time()
        step_12_cosmology_inference_main()
        step_times['Step 30'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 30 (Cosmology Inference) completed successfully.", "SUCCESS")

        # --- Step 31: Final Synthesis ---
        print_status(">>> STEP 31: Final Synthesis", "TITLE")
        t0 = time.time()
        step9 = Step9FinalSynthesis()
        step9.run()
        step_times['Step 31'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 31 (Final Synthesis) completed successfully.", "SUCCESS")

        # --- Step 32: Comprehensive Audit & Integrity Verification ---
        if not args.skip_audit:
            print_status(">>> STEP 32: Comprehensive Audit & Integrity Verification", "TITLE")
            t0 = time.time()
            Step11ComprehensiveAudit().run()
            step_times['Step 32'] = time.time() - t0
            set_step_logger(pipeline_logger)
            print_status("Step 32 (Comprehensive Audit) completed successfully.", "SUCCESS")
            
            # Lightweight pipeline self-check (legacy)
            print_status(">>> STEP 32b: PIPELINE SELF-CHECK", "TITLE")
            t0 = time.time()
            report = audit(project_root=PROJECT_ROOT, write_report=True)
            if not report.get('summary', {}).get('ok', False):
                n_fail = report.get('summary', {}).get('n_failed', -1)
                raise RuntimeError(f"Pipeline audit failed with {n_fail} errors. See results/outputs/step_32_pipeline_audit_report.json")
            step_times['Step 32b'] = time.time() - t0
            set_step_logger(pipeline_logger)
            print_status("Step 32b (Self-Check) passed: all outputs consistent.", "SUCCESS")

        # --- Step 33: Stellar Validation (Optional) ---
        if args.run_stellar_validation:
            print_status(">>> STEP 33: Stellar Validation of Scalar-boundary Transport", "TITLE")
            t0 = time.time()
            Step13StellarValidation().run()
            step_times['Step 33'] = time.time() - t0

            set_step_logger(pipeline_logger)
            print_status("Step 33 (Stellar Validation) completed successfully.", "SUCCESS")

        # --- Step 34: Full Ladder Likelihood ---
        print_status(">>> STEP 34: Full Ladder Likelihood", "TITLE")
        t0 = time.time()
        FullLadderLikelihood().run()
        step_times['Step 34'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 34 (Full Ladder Likelihood) completed successfully.", "SUCCESS")

        # --- Step 35: Bias-Aware TEP Ladder ---
        print_status(">>> STEP 35: Bias-aware TEP Ladder", "TITLE")
        t0 = time.time()
        step_35_run()
        step_times['Step 35'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 35 (Bias-Aware TEP Ladder) completed successfully.", "SUCCESS")

        # --- Step 36: Apparent Hubble Environment Likelihood ---
        print_status(">>> STEP 36: Apparent Hubble Environment Likelihood", "TITLE")
        t0 = time.time()
        step_36_run()
        step_times['Step 36'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 36 (Apparent Hubble Environment) completed successfully.", "SUCCESS")

        # --- Step 37: Velocity Robustness ---
        print_status(">>> STEP 37: Velocity Robustness", "TITLE")
        t0 = time.time()
        step_37_run()
        step_times['Step 37'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 37 (Velocity Robustness) completed successfully.", "SUCCESS")

        # --- Step 38: Hierarchical Joint Model ---
        print_status(">>> STEP 38: Hierarchical Joint Model", "TITLE")
        t0 = time.time()
        step_38_run()
        step_times['Step 38'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 38 (Hierarchical Joint Model) completed successfully.", "SUCCESS")

        # --- Step 39: Environment Slope Decomposition ---
        print_status(">>> STEP 39: Environment Slope Decomposition", "TITLE")
        t0 = time.time()
        step_39_run()
        step_times['Step 39'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 39 (Environment Slope Decomposition) completed successfully.", "SUCCESS")

        # --- Step 40: Flow / Sky Controls ---
        print_status(">>> STEP 40: Flow / Sky Controls", "TITLE")
        t0 = time.time()
        step_40_run()
        step_times['Step 40'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 40 (Flow / Sky Controls) completed successfully.", "SUCCESS")

        # --- Step 41: External Distance Breakers ---
        print_status(">>> STEP 41: External Distance Breakers", "TITLE")
        t0 = time.time()
        step_41_run()
        step_times['Step 41'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 41 (External Distance Breakers) completed successfully.", "SUCCESS")

        # --- Step 42: TEP-Native Generative Model ---
        print_status(">>> STEP 42: Tep-native Generative Model", "TITLE")
        t0 = time.time()
        step_42_run()
        step_times['Step 42'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 42 (TEP-Native Generative Model) completed successfully.", "SUCCESS")

        # --- Step 43: Toy Recovery Experiment ---
        print_status(">>> STEP 43: Toy Recovery Experiment", "TITLE")
        t0 = time.time()
        step_43_run()
        step_times['Step 43'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 43 (Toy Recovery Experiment) completed successfully.", "SUCCESS")

        # --- Regression Gates ---
        regression_gates(PROJECT_ROOT)

    except Exception as e:
        print_status(f"Pipeline failed: {str(e)}", "CRITICAL")
        print_status("Traceback:", "ERROR")
        pipeline_logger.error(traceback.format_exc())
        sys.exit(1)
        
    total_time = time.time() - start_time
    
    # --- Final Summary ---
    print_status("Pipeline execution summary", "TITLE")
    
    # Execution Times Table
    headers = ["Step", "Duration (s)", "Status"]
    rows = []
    for step, duration in step_times.items():
        rows.append([step, f"{duration:.2f}", "COMPLETED"])
    rows.append(["TOTAL", f"{total_time:.2f}", "SUCCESS"])
    
    print_table(headers, rows, title="Execution Timing")
    
    print_status(f"Total Execution Time: {total_time:.2f} seconds", "SUCCESS")
    print_status(f"Results Directory: {PROJECT_ROOT}/results/", "INFO")
    print_status(f"Logs Directory:    {PROJECT_ROOT}/logs/", "INFO")
    print_status("Pipeline finished.", "SUCCESS")

if __name__ == "__main__":
    run_pipeline()
