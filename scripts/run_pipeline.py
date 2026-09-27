#!/usr/bin/env python3
"""
TEP-H0 Analysis Pipeline Master Script
======================================
Orchestrates the analysis pipeline for Paper 11: "Paper 11: The Cepheid Bias: Resolving the Hubble Tension".

This script serves as the central controller for the TEP-H0 analysis.
It executes the scientific workflow in a strictly ordered sequence, ensuring
data integrity and dependency management between steps.

Workflow Steps:
1.  **Data Ingestion**: Reads or prepares the SH0ES/Pantheon+ inputs, reconstructs catalogs,
    and cross-matches hosts with external databases (Simbad, HyperLEDA).
2.  **Stratification**: Calculates H0 for each host, stratifies the sample by 
    gravitational potential (velocity dispersion), and detects the environmental bias.
3.  **Endpoint Models**: Fits the environmental redshift--distance slope and
    evaluates restricted Cepheid-channel projections.
4.  **Robustness Checks**: Performs rigorous statistical tests (Jackknife, Bivariate 
    Analysis, Sensitivity Analysis) to validate the results against systematics.
5.  **Differential Diagnostics**: Runs M31 and TRGB analyses without treating
    them as independent confirmation when their current power is insufficient.

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
from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood
from scripts.steps.step_37_velocity_robustness import run as step_37_run
from scripts.steps.step_39_environment_slope_decomposition import run as step_39_run
from scripts.steps.step_40_flow_sky_controls import run as step_40_run
from scripts.steps.step_42_tep_native_ladder import run as step_42_run
from scripts.steps.step_43_toy_recovery_experiment import run as step_43_run
from scripts.steps.step_44_joint_distance_redshift_likelihood import run as step_44_run
from scripts.steps.step_45_full_ladder_h0_propagation import run as step_45_run
from scripts.steps.step_47_anchor_double_counting_audit import main as step_47_run
from scripts.steps.step_48_pantheon_velocity_covariance import main as step_48_run
from scripts.steps.step_51_recent_velocity_sensitivity import run as step_51_run
from scripts.steps.step_52_bounded_response_comparison import run as step_52_run
from scripts.steps.step_53_within_host_structure_battery import run as step_53_run
from scripts.steps.step_55_within_host_plz_decomposition import run as step_55_run
from scripts.steps.step_61_nested_kappa import main as step_61_run
from scripts.utils.pipeline_audit import audit

def regression_gates(project_root):
    """Fail when an authoritative artifact regresses to a superseded result."""
    import json
    from pathlib import Path

    import numpy as np

    outputs = Path(project_root) / "results" / "outputs"
    checks = []

    def load(name):
        return json.loads((outputs / name).read_text())

    def check(label, condition, detail=""):
        checks.append((label, bool(condition), detail))
        level = "SUCCESS" if condition else "ERROR"
        state = "PASS" if condition else "FAIL"
        print_status(f"  [{state}] {label}" + (f": {detail}" if detail else ""), level)

    print_status(">>> REGRESSION GATES", "TITLE")

    s34 = load("step_34_full_ladder_likelihood_results.json")
    baseline = s34["baseline"]
    free = s34["stage2"]["variant_a_free_kappa"]
    injection = s34["injection_test"]
    check("S34 baseline H0", np.isclose(baseline["H0"], 73.0434, atol=0.01))
    check(
        "S34 real-data Cepheid coefficient remains weak",
        abs(free["kappa_Cep"] / free["kappa_err"]) < 1.0,
        f"kappa/error={free['kappa_significance']:.3f}",
    )
    check(
        "S34 row-level injection is recovered",
        abs(injection["recovery_fraction"] - 1.0) < 0.01
        and injection["rank_aug"] == 47,
    )

    rows39 = load("step_39_environment_slope_decomposition.json")
    primary39 = next(
        row for row in rows39
        if row["sigma_v"] == 250 and row["sample"] == "all_r22_hosts"
        and row["z_cut"] == 0.0
    )
    tests39 = load("step_39_statistical_tests.json")
    check("S39 complete R22 sample has 37 hosts", primary39["n_hosts"] == 37)
    check(
        "S39 endpoint likelihood is finite",
        np.isfinite(primary39["Gamma_X"])
        and primary39["Gamma_X_err"] > 0
        and np.isfinite(primary39["delta_2logL_vs_null"]),
        (
            f"Gamma={primary39['Gamma_X']:.3e}, "
            f"err={primary39['Gamma_X_err']:.3e}, "
            f"LRT={primary39['Gamma_X_lrt_sig']:.3f}"
        ),
    )
    perm39 = tests39["permutation"]
    check(
        "S39 finite-sample permutation uses plus-one correction",
        perm39["n_permutations"] == 5000
        and np.isclose(
            perm39["permutation_p"],
            (perm39["permutation_exceedances"] + 1) / 5001,
        ),
    )
    check("S39 leave-one-host-out covers all hosts", tests39["loho"]["N_hosts"] == 37)

    rows40 = load("step_40_flow_sky_controls.json")
    primary40 = {
        row["model"]: row for row in rows40
        if row["sample"] == "primary" and row["sigma_v"] == 250
    }
    check(
        "S40 parameter counts include full traceless quadrupole",
        primary40["M1"]["n_params"] == 3
        and primary40["M4"]["n_params"] == 11
        and primary40["M6"]["n_params"] == 12,
    )
    check(
        "S40 M1 reproduces S39 endpoint",
        np.isclose(primary40["M1"]["Gamma_X"], primary39["Gamma_X"], rtol=1e-5),
    )
    check(
        "S40 null BIC is retained in model comparison",
        primary40["M0"]["BIC"]
        == min(primary40[name]["BIC"] for name in ("M0", "M1", "M2", "M3", "M4", "M6")),
    )

    s20 = load("step_20_joint_indicator_model.json")
    check("S20 current matched indicator sample has 18 hosts", s20["N_hosts"] == 18)
    check(
        "S20 weighted Cepheid-TRGB differential is not detected",
        abs(s20["kappa_diff_t"]) < 1.0 and s20["kappa_diff_p"] > 0.3,
        f"t={s20['kappa_diff_t']:.3f}, p={s20['kappa_diff_p']:.3f}",
    )

    s45 = load("step_45_full_ladder_h0_propagation.json")
    standard = {
        row["kappa_name"]: row for row in s45["matrix_propagation"]
        if row["sigma_ref_label"] == "standard"
    }
    screened = {
        row["kappa_name"]: row for row in s45["matrix_propagation"]
        if row["sigma_ref_label"] == "screened"
    }
    gauge_ok = standard.keys() == screened.keys() and all(
        np.isclose(standard[name]["H0"], screened[name]["H0"], atol=1e-10)
        and np.isclose(standard[name]["chi2"], screened[name]["chi2"], atol=1e-8)
        for name in standard
    )
    check("S45 full-ladder result is reference-gauge invariant", gauge_ok)
    endpoint45 = standard["kappa_endpoint_equiv"]
    check(
        "S45 endpoint projection is linked and reported with its penalty",
        np.isclose(endpoint45["kappa"], primary39["kappa_equiv"], rtol=1e-12)
        and endpoint45["delta_chi2"] > 0.0,
        f"H0={endpoint45['H0']:.4f}, delta_chi2={endpoint45['delta_chi2']:.3f}",
    )

    s47 = load("step_47_anchor_double_counting_audit.json")
    check(
        "S47 Cepheid rows are owned once and priors are untouched",
        s47["pseudo_replication"]["no_replication"]
        and s47["pseudo_replication"]["unique_cepheid_rows"] == 3132
        and s47["tep_correction"]["n_prior_modified"] == 0,
    )

    s51 = load("step_51_recent_velocity_sensitivity.json")
    check(
        "S51 recent Manticore table is complete and labelled diagnostic",
        s51["models"]["manticore_screened"]["n_hosts"] == 35
        and s51["source"]["arxiv_version"] == "2509.09665v3"
        and "not an independent" in s51["interpretation"],
    )

    failed = [label for label, ok, _ in checks if not ok]
    print_status(f"Regression gates: {len(checks) - len(failed)}/{len(checks)} passed", "INFO")
    if failed:
        raise RuntimeError(f"Regression gates failed: {', '.join(failed)}")
    print_status("All regression gates passed.", "SUCCESS")

def run_pipeline():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--skip-sigma-step", action="store_true")
    ap.add_argument("--rebuild-sigma", action="store_true")
    ap.add_argument("--use-lit-overrides", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
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

        # --- Step 34: Full Ladder Likelihood ---
        print_status(">>> STEP 34: Full Ladder Likelihood", "TITLE")
        t0 = time.time()
        FullLadderLikelihood().run()
        step_times['Step 34'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 34 (Full Ladder Likelihood) completed successfully.", "SUCCESS")

        # --- Step 37: Velocity Robustness ---
        print_status(">>> STEP 37: Velocity Robustness", "TITLE")
        t0 = time.time()
        step_37_run()
        step_times['Step 37'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 37 (Velocity Robustness) completed successfully.", "SUCCESS")

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

        # --- Step 42: TEP Native Ladder ---
        print_status(">>> STEP 42: TEP Native Ladder", "TITLE")
        t0 = time.time()
        step_42_run()
        step_times['Step 42'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 42 (TEP Native Ladder) completed successfully.", "SUCCESS")

        # --- Step 43: Toy Recovery Experiment ---
        print_status(">>> STEP 43: Toy Recovery Experiment", "TITLE")
        t0 = time.time()
        step_43_run()
        step_times['Step 43'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 43 (Toy Recovery Experiment) completed successfully.", "SUCCESS")

        # --- Step 44: Joint Distance-Redshift Likelihood ---
        print_status(">>> STEP 44: Joint Distance-Redshift Likelihood", "TITLE")
        t0 = time.time()
        step_44_run()
        step_times['Step 44'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 44 (Joint Distance-Redshift Likelihood) completed successfully.", "SUCCESS")

        # --- Step 45: Full-ladder H_0 propagation ---
        print_status(">>> STEP 45: Full-Ladder H_0 Propagation Test", "TITLE")
        t0 = time.time()
        step_45_run()
        step_times['Step 45'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 45 (Full-Ladder Propagation) completed successfully.", "SUCCESS")

        # --- Step 47: Anchor double-counting audit ---
        print_status(">>> STEP 47: Hierarchical Anchor Model Audit", "TITLE")
        t0 = time.time()
        step_47_run()
        step_times['Step 47'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 47 (Anchor Audit) completed successfully.", "SUCCESS")

        # --- Step 48: Pantheon+ peculiar-velocity covariance ---
        print_status(">>> STEP 48: Pantheon+ Peculiar-Velocity Covariance Analysis", "TITLE")
        t0 = time.time()
        step_48_run()
        step_times['Step 48'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 48 (Pantheon+ Velocity Covariance) completed successfully.", "SUCCESS")

        # --- Step 51: recent Manticore posterior-summary sensitivity ---
        print_status(">>> STEP 51: Recent Manticore Velocity Sensitivity", "TITLE")
        t0 = time.time()
        step_51_run()
        step_times['Step 51'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 51 (Recent Velocity Sensitivity) completed successfully.", "SUCCESS")

        # --- Step 52: bounded-response model comparison and kappaP disentanglement ---
        print_status(">>> STEP 52: Bounded-Response Model Comparison", "TITLE")
        t0 = time.time()
        step_52_run()
        step_times['Step 52'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 52 (Bounded-Response Model Comparison) completed successfully.", "SUCCESS")

        # --- Step 53: within-host period-structure discriminating battery ---
        print_status(">>> STEP 53: Within-Host Period-Structure Battery", "TITLE")
        t0 = time.time()
        step_53_run()
        step_times['Step 53'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 53 (Within-Host Structure Battery) completed successfully.", "SUCCESS")

        # --- Step 55: within-host PLZ decomposition (metallicity channel) ---
        print_status(">>> STEP 55: Within-Host PLZ Decomposition", "TITLE")
        t0 = time.time()
        step_55_run()
        step_times['Step 55'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 55 (Within-Host PLZ Decomposition) completed successfully.", "SUCCESS")

        # --- Step 61: nested clock-ratio bound on the raw conformal channel ---
        print_status(">>> STEP 61: Nested Clock-Ratio Bound", "TITLE")
        t0 = time.time()
        step_61_run()
        step_times['Step 61'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 61 (Nested Clock-Ratio Bound) completed successfully.", "SUCCESS")

        # --- Regression Gates ---
        regression_gates(PROJECT_ROOT)

        # Run the manuscript/output audit only after all inferential steps have
        # produced their current artifacts. Assumption-dependent and superseded
        # legacy steps are deliberately excluded from this authoritative path.
        if not args.skip_audit:
            print_status(">>> FINAL PIPELINE SELF-CHECK", "TITLE")
            t0 = time.time()
            report = audit(project_root=PROJECT_ROOT, write_report=True)
            if not report.get('summary', {}).get('ok', False):
                n_fail = report.get('summary', {}).get('n_failed', -1)
                raise RuntimeError(
                    f"Pipeline audit failed with {n_fail} errors. "
                    "See results/outputs/step_32_pipeline_audit_report.json"
                )
            step_times['Final audit'] = time.time() - t0
            set_step_logger(pipeline_logger)
            print_status("Final self-check passed: all audited outputs are consistent.", "SUCCESS")

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
