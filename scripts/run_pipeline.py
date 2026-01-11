#!/usr/bin/env python3
"""
TEP-H0 Analysis Pipeline Master Script
======================================
Orchestrates the full analysis pipeline for Paper 12: "The Cepheid Bias: Resolving the Hubble Tension".

This script serves as the central controller for the TEP-H0 analysis.
It executes the scientific workflow in a strictly ordered sequence, ensuring
data integrity and dependency management between steps.

Workflow Steps:
1.  **Data Ingestion**: Downloads raw data (SH0ES, Pantheon+), reconstructs catalogs, 
    and cross-matches hosts with external databases (Simbad, HyperLEDA).
2.  **Stratification**: Calculates H0 for each host, stratifies the sample by 
    gravitational potential (velocity dispersion), and detects the environmental bias.
3.  **TEP Correction**: Optimizes the TEP screening parameter (alpha), applies the 
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
from scripts.steps.step_0_sigma_catalog import Step0SigmaCatalog
from scripts.steps.step_1_data_ingestion import Step1DataIngestion
from scripts.utils.fetch_metadata import fetch_galaxy_metadata
from scripts.steps.step_1b_aperture_correction import Step1bApertureCorrection
from scripts.steps.step_2_stratification import Step2Stratification
from scripts.steps.step_3_tep_correction import Step3TEPCorrection
from scripts.steps.step_4_robustness_checks import Step4RobustnessChecks
from scripts.steps.step_4b_aperture_sensitivity import Step4bApertureSensitivity
from scripts.steps.step_5_m31_analysis import Step5M31Analysis
from scripts.steps.step_6_multivariate_analysis import Step6MultivariateAnalysis
from scripts.steps.step_6_enhanced_robustness import Step6EnhancedRobustness
from scripts.steps.step_7_lmc_replication import Step7LMCReplication
from scripts.steps.step_7_trgb_comparison import Step7TRGBComparison
from scripts.steps.step_8_m31_phat_analysis import Step8M31PHATAnalysis
from scripts.steps.step_9_final_synthesis import Step9FinalSynthesis
from scripts.steps.step_10_anchor_stratification import AnchorStratificationStep
from scripts.utils.pipeline_audit import audit

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
    
    print_status("TEP-H0 ANALYSIS PIPELINE INITIATED", "TITLE")
    print_status(f"Project Root: {PROJECT_ROOT}", "INFO")
    print_status("Starting execution sequence...", "INFO")
    
    start_time = time.time()
    step_times = {}
    
    try:
        # --- Step 1 (Pre-flight): Prepare Coordinates ---
        # Needed for Step 0 (Sigma Catalog Build) to know which galaxies to query
        print_status(">>> STEP 1 (PRE-FLIGHT): COORDINATE PREPARATION", "TITLE")
        step1_pre = Step1DataIngestion()
        step1_pre.prepare_coordinates()

        # --- Step 0: Sigma Catalog Build (Provenance) ---
        if not args.skip_sigma_step:
            print_status(">>> STEP 0: SIGMA CATALOG (PROVENANCE BUILD)", "TITLE")
            t0 = time.time()
            Step0SigmaCatalog().run(
                rebuild=bool(args.rebuild_sigma),
                use_ledacat=True,
                use_ho2007=True,
                use_bassdr2=True,
                use_apj929=True,
                use_mnras482=True,
                use_lit_overrides=bool(args.use_lit_overrides),
            )
            step_times['Step 0'] = time.time() - t0
            set_step_logger(pipeline_logger)
            print_status("Step 0 (Sigma Catalog) completed successfully.", "SUCCESS")

        # --- Step 1: Data Ingestion ---
        print_status(">>> STEP 1: DATA INGESTION", "TITLE")
        t0 = time.time()
        step1 = Step1DataIngestion()
        step1.run()
        step_times['Step 1'] = time.time() - t0
        
        # Reset logger to master after step completion (step scripts set their own)
        set_step_logger(pipeline_logger)
        print_status("Step 1 (Ingestion) completed successfully.", "SUCCESS")
        
        # --- Step 1b: Aperture Correction ---
        print_status(">>> STEP 1b: APERTURE CORRECTION", "TITLE")
        t0 = time.time()
        
        # Sub-task: Fetch Metadata (RC3 Sizes)
        print_status("Fetching host metadata (RC3) for aperture normalization...", "PROCESS")
        fetch_galaxy_metadata()
        
        # Sub-task: Apply Correction
        step1b = Step1bApertureCorrection()
        step1b.run()
        step_times['Step 1b'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 1b (Aperture Correction) completed successfully.", "SUCCESS")
        
        # --- Step 2: Stratification ---
        print_status(">>> STEP 2: STRATIFICATION", "TITLE")
        t0 = time.time()
        step2 = Step2Stratification()
        step2.run()
        step_times['Step 2'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 2 (Stratification) completed successfully.", "SUCCESS")
        
        # --- Step 3: TEP Correction ---
        print_status(">>> STEP 3: TEP CORRECTION", "TITLE")
        t0 = time.time()
        step3 = Step3TEPCorrection()
        step3.run()
        step_times['Step 3'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 3 (Optimization) completed successfully.", "SUCCESS")

        # --- Step 4: Robustness Checks ---
        print_status(">>> STEP 4: ROBUSTNESS CHECKS", "TITLE")
        t0 = time.time()
        
        print_status("Running Jackknife and Bivariate Analysis...", "PROCESS")
        Step4RobustnessChecks().run()
        
        print_status("Running Aperture Sensitivity Analysis...", "PROCESS")
        Step4bApertureSensitivity().run()
        
        step_times['Step 4'] = time.time() - t0
        set_step_logger(pipeline_logger)
        print_status("Step 4 (Robustness) completed successfully.", "SUCCESS")

        # --- Step 5: M31 Analysis ---
        print_status(">>> STEP 5: M31 ANALYSIS", "TITLE")
        t0 = time.time()
        step5 = Step5M31Analysis()
        step5.run()
        step_times['Step 5'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 5 (M31 Differential Test) completed successfully.", "SUCCESS")

        # --- Step 6: Multivariate Analysis ---
        print_status(">>> STEP 6: MULTIVARIATE ANALYSIS", "TITLE")
        t0 = time.time()
        step6 = Step6MultivariateAnalysis()
        step6.run()
        step_times['Step 6'] = time.time() - t0
        
        set_step_logger(pipeline_logger)
        print_status("Step 6 (Multivariate Analysis) completed successfully.", "SUCCESS")

        # --- Step 6b: Enhanced Robustness (Referee-Facing) ---
        print_status(">>> STEP 6b: ENHANCED ROBUSTNESS", "TITLE")
        t0 = time.time()
        Step6EnhancedRobustness().run()
        step_times['Step 6b'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 6b (Enhanced Robustness) completed successfully.", "SUCCESS")

        # --- Step 7: LMC Replication ---
        print_status(">>> STEP 7: LMC REPLICATION", "TITLE")
        t0 = time.time()
        step7 = Step7LMCReplication()
        step7.run()
        step_times['Step 7'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 7 (LMC Replication) completed successfully.", "SUCCESS")

        # --- Step 7b: TRGB Comparison ---
        print_status(">>> STEP 7b: TRGB COMPARISON", "TITLE")
        t0 = time.time()
        Step7TRGBComparison().run()
        step_times['Step 7b'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 7b (TRGB Comparison) completed successfully.", "SUCCESS")

        # --- Step 8: M31 PHAT Analysis ---
        print_status(">>> STEP 8: M31 PHAT ANALYSIS", "TITLE")
        t0 = time.time()
        step8 = Step8M31PHATAnalysis()
        step8.run()
        step_times['Step 8'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 8 (M31 PHAT Analysis) completed successfully.", "SUCCESS")

        # --- Step 9: Final Synthesis ---
        print_status(">>> STEP 9: FINAL SYNTHESIS", "TITLE")
        t0 = time.time()
        step9 = Step9FinalSynthesis()
        step9.run()
        step_times['Step 9'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 9 (Final Synthesis) completed successfully.", "SUCCESS")

        # --- Step 10: Anchor Stratification Test ---
        print_status(">>> STEP 10: ANCHOR STRATIFICATION TEST", "TITLE")
        t0 = time.time()
        step10 = AnchorStratificationStep()
        step10.run()
        step_times['Step 10'] = time.time() - t0

        set_step_logger(pipeline_logger)
        print_status("Step 10 (Anchor Stratification) completed successfully.", "SUCCESS")

        # --- Step 11: Pipeline Audit (Self-Check) ---
        if not args.skip_audit:
            print_status(">>> STEP 11: PIPELINE AUDIT", "TITLE")
            t0 = time.time()
            
            # Run audit; fail if checks fail
            report = audit(project_root=PROJECT_ROOT, write_report=True)
            if not report.get('summary', {}).get('ok', False):
                n_fail = report.get('summary', {}).get('n_failed', -1)
                raise RuntimeError(f"Pipeline Audit FAILED with {n_fail} errors. See results/outputs/pipeline_audit_report.json")
            
            step_times['Step 11'] = time.time() - t0
            set_step_logger(pipeline_logger)
            print_status("Step 11 (Audit) PASSED: All outputs consistent.", "SUCCESS")
        
    except Exception as e:
        print_status(f"PIPELINE FAILED: {str(e)}", "CRITICAL")
        print_status("Traceback:", "ERROR")
        pipeline_logger.error(traceback.format_exc())
        sys.exit(1)
        
    total_time = time.time() - start_time
    
    # --- Final Summary ---
    print_status("PIPELINE EXECUTION SUMMARY", "TITLE")
    
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
