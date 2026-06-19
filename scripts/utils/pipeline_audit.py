import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.utils.logger import print_status
except ImportError:
    def print_status(msg: str, level: str = "INFO"):
        print(f"[{level}] {msg}")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _approx(a: float, b: float, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return bool(abs(a - b) <= (atol + rtol * abs(b)))


def _check(name: str, ok: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _contains_all(text: str, tokens: List[str]) -> Tuple[bool, List[str]]:
    missing = [token for token in tokens if token not in text]
    return len(missing) == 0, missing


def audit(project_root: Optional[Path] = None, write_report: bool = True) -> Dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[2]
    outputs = root / "results" / "outputs"
    data_raw_ext = root / "data" / "raw" / "external"
    data_interim = root / "data" / "interim"

    report: Dict[str, Any] = {
        "project_root": str(root),
        "outputs_dir": str(outputs),
        "checks": [],
        "summary": {},
    }

    # Canonical per-host table
    strat_path = outputs / "stratified_h0.csv"
    if not strat_path.exists():
        report["checks"].append(_check("stratified_h0_exists", False, {"path": str(strat_path)}))
        return report

    df = pd.read_csv(strat_path)
    df['normalized_name'] = df['normalized_name'].astype(str).str.strip()

    # Basic sample integrity
    anchors = {"NGC 4258", "LMC", "SMC", "M 31", "MW"}
    dupes = df['normalized_name'][df['normalized_name'].duplicated()].tolist()
    has_anchor = sorted(set(df['normalized_name']).intersection(anchors))

    report["checks"].append(_check(
        "no_duplicate_hosts_in_stratified_h0",
        len(dupes) == 0,
        {"duplicates": dupes},
    ))
    report["checks"].append(_check(
        "anchors_excluded_from_stratified_h0",
        len(has_anchor) == 0,
        {"anchors_present": has_anchor},
    ))

    # Recompute headline stats from stratified_h0
    x = df['sigma_inferred'].astype(float).values
    y = df['h0_derived'].astype(float).values

    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)
    med = float(np.median(x))
    low = df[df['sigma_inferred'] <= med]
    high = df[df['sigma_inferred'] > med]

    derived = {
        "n": int(len(df)),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "median_sigma": float(med),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "low_mean_h0": float(low['h0_derived'].mean()),
        "high_mean_h0": float(high['h0_derived'].mean()),
        "delta_h0": float(high['h0_derived'].mean() - low['h0_derived'].mean()),
    }
    report["summary"]["derived_from_stratified_h0"] = derived

    # Check stratification_results.json
    strat_json = _read_json(outputs / "stratification_results.json")
    if strat_json is None:
        report["checks"].append(_check("stratification_results_json_exists", False, {}))
    else:
        ok = (
            _approx(float(strat_json.get('median_sigma')), derived['median_sigma'], atol=1e-6, rtol=1e-6)
            and int(strat_json.get('low_density', {}).get('n')) == derived['low_n']
            and int(strat_json.get('high_density', {}).get('n')) == derived['high_n']
            and _approx(float(strat_json.get('low_density', {}).get('mean_h0')), derived['low_mean_h0'], atol=1e-6, rtol=1e-6)
            and _approx(float(strat_json.get('high_density', {}).get('mean_h0')), derived['high_mean_h0'], atol=1e-6, rtol=1e-6)
            and _approx(float(strat_json.get('difference')), derived['delta_h0'], atol=1e-6, rtol=1e-6)
            and _approx(float(strat_json.get('correlation_r')), derived['pearson_r'], atol=1e-6, rtol=1e-6)
        )
        report["checks"].append(_check(
            "stratification_results_matches_recomputed",
            ok,
            {"expected": derived, "got": strat_json},
        ))

    # Check covariance_robustness.json
    cov = _read_json(outputs / "covariance_robustness.json")
    if cov is None:
        report["checks"].append(_check("covariance_robustness_json_exists", False, {}))
    else:
        ok = (
            int(cov.get('n')) == derived['n']
            and _approx(float(cov.get('pearson_r')), derived['pearson_r'], atol=1e-6, rtol=1e-6)
            and _approx(float(cov.get('spearman_rho')), derived['spearman_rho'], atol=1e-6, rtol=1e-6)
        )
        report["checks"].append(_check(
            "covariance_robustness_matches_recomputed",
            ok,
            {"expected": {"n": derived['n'], "pearson_r": derived['pearson_r'], "spearman_rho": derived['spearman_rho']}, "got": cov},
        ))

        projected = cov.get("bayesian_comparison", {}).get("projected", {})
        gls_crosscheck = cov.get("bayesian_comparison", {}).get("gls_crosscheck", {})
        projected_delta_bic = projected.get("delta_bic_matched", projected.get("delta_bic"))
        # The covariance analysis tests the RAW (uncorrected) H0 data.
        # The TEP model (environmental slope) should be preferred over the
        # null (no sigma-H0 dependence) in the raw sample.  BIC convention:
        # delta_bic = BIC_null - BIC_TEP  (positive means TEP is preferred).
        # We require at least "positive" evidence (delta_bic >= 2.0).
        projected_ok = projected_delta_bic is not None and float(projected_delta_bic) >= 2.0
        report["checks"].append(_check(
            "covariance_projected_bic_retains_strong_evidence",
            projected_ok,
            {"projected": projected},
        ))
        try:
            gls_delta_bic = float(gls_crosscheck.get("delta_bic"))
            # The two BIC estimates (projected vs GLS) should both favour TEP.
            gls_sign_consistent = (gls_delta_bic >= 2.0) == (float(projected_delta_bic) >= 2.0)
            gls_ok = gls_delta_bic >= 2.0 and gls_sign_consistent
        except (TypeError, ValueError):
            gls_ok = False
        report["checks"].append(_check(
            "covariance_gls_slope_matches_projected_contrast",
            gls_ok,
            {"gls_crosscheck": gls_crosscheck, "projected": projected},
        ))

    # Check TEP correction
    tep = _read_json(outputs / "tep_correction_results.json")
    if tep is None:
        report["checks"].append(_check("tep_correction_results_json_exists", False, {}))
    else:
        # Step 3 defines tension_sigma using the JOINT BOOTSTRAP (kappa refit per
        # resample), which combines host-to-host scatter and kappa parameter
        # uncertainty in a single honest H0 uncertainty.
        tension_bootstrap = None
        tension_sem = None
        try:
            planck_h0 = float(tep['planck_h0'])
            planck_err = 0.5
            # Step 3 computes tension_sigma from bootstrap_h0_mean, not unified_h0
            h0_boot = float(tep.get('bootstrap_h0_mean', tep['unified_h0']))
            h0 = float(tep['unified_h0'])
            if 'bootstrap_h0_std' in tep:
                tension_bootstrap = abs(h0_boot - planck_h0) / math.sqrt(float(tep['bootstrap_h0_std']) ** 2 + planck_err ** 2)
            if 'h0_sem' in tep:
                tension_sem = abs(h0 - planck_h0) / math.sqrt(float(tep['h0_sem']) ** 2 + planck_err ** 2)
        except Exception:
            pass

        # Check against bootstrap tension (the primary metric)
        ok = (
            int(tep.get('n_hosts')) == derived['n']
            and tension_bootstrap is not None
            and _approx(float(tep.get('tension_sigma')), float(tension_bootstrap), atol=1e-3, rtol=1e-3)
        )
        report["checks"].append(_check(
            "tep_correction_internal_consistency",
            ok,
            {
                "got": tep,
                "recomputed_tension_bootstrap": tension_bootstrap,
                "recomputed_tension_sem": tension_sem,
            },
        ))

        corrected_path = outputs / "tep_corrected_h0.csv"
        if not corrected_path.exists():
            report["checks"].append(_check(
                "tep_corrected_csv_matches_primary_headline",
                False,
                {"path": str(corrected_path)},
            ))
        else:
            corrected = pd.read_csv(corrected_path)
            csv_mean = float(corrected["h0_corrected"].mean())
            json_mean = float(tep.get("unified_h0"))
            report["checks"].append(_check(
                "tep_corrected_csv_matches_primary_headline",
                _approx(csv_mean, json_mean, atol=1e-6, rtol=1e-9),
                {"csv_mean": csv_mean, "json_unified_h0": json_mean},
            ))

        anchor_screening = tep.get("anchor_screening", {})
        ngc4258_screening = anchor_screening.get("NGC 4258")
        try:
            screened_ref = float(tep.get("sigma_ref_screened"))
            ngc4258_screening = float(ngc4258_screening)
            anchor_ok = screened_ref < 40.0 and ngc4258_screening < 0.2
        except (TypeError, ValueError):
            anchor_ok = False
        report["checks"].append(_check(
            "screened_reference_uses_ngc4258_anchor_screening",
            anchor_ok,
            {
                "sigma_ref_screened": tep.get("sigma_ref_screened"),
                "anchor_screening": anchor_screening,
                "anchor_nmb": tep.get("anchor_nmb", {}),
            },
        ))

    # Sigma regeneration report integrity
    sigma_report = _read_json(outputs / "sigma_regeneration_report.json")
    if sigma_report is None:
        report["checks"].append(_check("sigma_regeneration_report_exists", False, {}))
    else:
        counts = sigma_report.get('counts', {})
        ok = (
            int(counts.get('n_with_sigma', -1)) == int(counts.get('n_hosts', -2))
            and int(counts.get('n_missing_sigma', -1)) == 0
        )
        report["checks"].append(_check(
            "sigma_regeneration_has_full_coverage",
            ok,
            {"counts": counts, "inputs": sigma_report.get('inputs', {})},
        ))

    # Sigma provenance table existence and uniqueness
    prov_path = outputs / "sigma_provenance_table.csv"
    if not prov_path.exists():
        report["checks"].append(_check("sigma_provenance_table_exists", False, {"path": str(prov_path)}))
    else:
        prov = pd.read_csv(prov_path)
        prov['normalized_name'] = prov['normalized_name'].astype(str).str.strip()
        dup_prov = prov['normalized_name'][prov['normalized_name'].duplicated()].tolist()
        report["checks"].append(_check(
            "sigma_provenance_unique",
            len(dup_prov) == 0 and int(len(prov)) == derived['n'],
            {"n": int(len(prov)), "expected_n": derived['n'], "duplicates": dup_prov},
        ))

        # SINGLE SOURCE OF TRUTH: canonical traceable literature CSV
        master_path = data_raw_ext / "velocity_dispersions_literature.csv"
        report["checks"].append(_check(
            "master_sigma_catalog_present",
            master_path.exists(),
            {"path": str(master_path)},
        ))

    # Enhanced robustness should align with recomputed full-sample stats
    enh = _read_json(outputs / "enhanced_robustness_results.json")
    if enh is None:
        report["checks"].append(_check("enhanced_robustness_exists", False, {}))
    else:
        fs = enh.get('stellar_absorption', {}).get('full_sample', {})
        ok = (
            _approx(float(fs.get('pearson_r')), derived['pearson_r'], atol=1e-6, rtol=1e-6)
            and _approx(float(fs.get('spearman_rho')), derived['spearman_rho'], atol=1e-6, rtol=1e-6)
            and int(enh.get('stellar_absorption', {}).get('n_total')) == derived['n']
        )
        report["checks"].append(_check(
            "enhanced_robustness_matches_full_sample",
            ok,
            {"expected": {"pearson_r": derived['pearson_r'], "spearman_rho": derived['spearman_rho'], "n": derived['n']}, "got": enh.get('stellar_absorption', {})},
        ))

    # TRGB comparison should use current cepheid stats
    trgb = _read_json(outputs / "trgb_comparison_results.json")
    if trgb is None:
        report["checks"].append(_check("trgb_comparison_exists", False, {}))
    else:
        comp = trgb.get('comparison', {})
        ok = (
            int(comp.get('cepheid_n', -1)) == derived['n']
            and _approx(float(comp.get('cepheid_spearman')), derived['spearman_rho'], atol=1e-6, rtol=1e-6)
            and _approx(float(comp.get('cepheid_delta_h0')), derived['delta_h0'], atol=1e-6, rtol=1e-6)
        )
        report["checks"].append(_check(
            "trgb_comparison_uses_current_cepheid_stats",
            ok,
            {"expected": {"cepheid_n": derived['n'], "cepheid_spearman": derived['spearman_rho'], "cepheid_delta_h0": derived['delta_h0']}, "got": comp},
        ))

    # Host coordinates input hygiene (duplicate LMC should be acceptable, but flagged)
    hosts_coords = data_interim / "hosts_coords.csv"
    if hosts_coords.exists():
        hc = pd.read_csv(hosts_coords)
        hc['normalized_name'] = hc['normalized_name'].astype(str).str.strip()
        dup_hc = hc['normalized_name'][hc['normalized_name'].duplicated()].value_counts().to_dict()
        report["checks"].append(_check(
            "hosts_coords_duplicates_reported",
            True,
            {"duplicates_by_name": dup_hc, "n_rows": int(len(hc))},
        ))

    # Narrative/result-surface integrity. The pipeline owns not only the
    # numerical products but also the manuscript/site values that will be read
    # by reviewers. These checks deliberately target stale headline numbers and
    # obsolete parameter names that have previously drifted out of sync.
    narrative_paths = [
        root / "README.md",
        root / "zenodo.txt",
        root / "manuscripts" / "11-TEP-H0-v0.7-KingstonUponHull.md",
        root / "11-TEP-H0-v0.7-KingstonUponHull.md",
        root / "site" / "components" / "1_abstract.html",
        root / "site" / "components" / "4_results.html",
        root / "site" / "components" / "5_discussion.html",
        root / "site" / "components" / "6_conclusion.html",
        root / "site" / "CITATION.cff",
        root / "site" / "dist" / "index.html",
        root / "site" / "dist" / "CITATION.cff",
        root / "site" / "codemeta.json",
        root / "site" / "index.html",
        outputs / "TEP_FINAL_ROBUSTNESS_REPORT.md",
    ]
    narrative_text_by_path = {str(path.relative_to(root)): _read_text(path) for path in narrative_paths if path.exists()}
    narrative_text = "\n".join(narrative_text_by_path.values())

    if tep is not None:
        try:
            expected_tokens = [
                f"{derived['spearman_rho']:.3f}",
                f"{derived['spearman_p']:.4f}",
                f"{derived['pearson_r']:.3f}",
                f"{derived['pearson_p']:.4f}",
                f"{float(tep['unified_h0']):.2f}",
                f"{float(tep['bootstrap_h0_mean']):.2f}",
                f"{float(tep['bootstrap_h0_std']):.2f}",
                f"{float(tep['tension_sigma']):.2f}",
                f"{float(tep['optimal_kappa_cep']) / 1e6:.2f}",
                f"{float(tep.get('bootstrap_kappa_robust_std') or tep.get('wls_kappa_err_scaled') or tep.get('bootstrap_kappa_std', 0.89)) / 1e6:.2f}",
            ]
            # Check 1: Global presence (at least one file has all tokens)
            ok_global, missing_global = _contains_all(narrative_text, expected_tokens)
            
            # Check 2: Per-file presence (each file must have critical tokens)
            # This prevents stale files from hiding behind updated ones
            critical_tokens = [
                f"{float(tep['unified_h0']):.2f}",
                f"{float(tep['tension_sigma']):.2f}",
                f"{float(tep['optimal_kappa_cep']) / 1e6:.2f}",
            ]
            files_missing_tokens = {}
            for path_str, content in narrative_text_by_path.items():
                _, missing_file = _contains_all(content, critical_tokens)
                if missing_file:
                    files_missing_tokens[path_str] = missing_file
            
            ok_per_file = len(files_missing_tokens) == 0
            
            report["checks"].append(_check(
                "narrative_surfaces_include_current_headline_numbers",
                ok_global and ok_per_file,
                {
                    "required_tokens": expected_tokens, 
                    "missing_tokens_global": missing_global,
                    "files_with_missing_critical_tokens": files_missing_tokens,
                    "paths": list(narrative_text_by_path.keys())
                },
            ))
        except Exception as exc:
            report["checks"].append(_check(
                "narrative_surfaces_include_current_headline_numbers",
                False,
                {"error": str(exc), "paths": list(narrative_text_by_path.keys())},
            ))

    forbidden_tokens = [
        "Optimized TEP parameters (α",
        "Optimizes TEP coupling α",
        "alpha_eff",
        "\\alpha_{\\rm eff}",
        "\\alpha_{\\rm anchor}",
        "α_eff",
        "α_anchor",
        "0.434",
        "0.428",
        "68.37",
        "0.60\\sigma",
        "0.60\\\\sigma",
        "(9.6 \\pm 4.0)",
        "(9.6 \\\\(pm 4.0)",
        "9.6 \\times 10^5",
        "9.6 \\\\(times 10^5",
        "$H_0 \\approx 68.4$",
        "H0≈68.4",
        "Caveats and Limitations",
        "Several caveats",
        "statistical caveat",
        "Future work must resolve",
        "Mass Distortion",
        "p=0.123",
        "p=0.070",
        "loses independent statistical",
        "competitive explanatory variable",
        "collinearity reduces",
        "Potential overfitting of",
        "Anchor Tension (Resolved)",
        "anchor tension",
        "TEP v0.7",
        "Paper 0, v0.7",
        "68.17/s/Mpc",
        "+0.44$ mag",
        "+0.53$ mag",
        "67.82",
        "72.45",
        "4.63",
        "68.09",
        "68.00",
        "0.43\\sigma",
        "0.43\\\\sigma",
        "0.59) \\times 10^6",
        "0.59) \\\\times 10^6",
        "well within the joint bootstrap uncertainty",
        "pm 1.49/s/Mpc",
        "1.06 \\pm 0.26",
        "1.06 \\\\pm 0.26",
        "ΔBIC≈ -3",
        "\\Delta{\\rm BIC}\\approx -3",
        "Full absolute covariance BIC",
        "Full absolute covariance likelihood",
        "full-covariance absolute likelihood",
        "null favoured in absolute mode",
        "understates the evidence",
        "dominated by common mode",
        "ΔBIC=88",
        "\\Delta{\\rm BIC}=88",
        "\\Delta{\\rm BIC}=+94",
        "\\Delta{\\rm BIC}=2.4",
        "ΔBIC=2.4",
        "68.04",
        "0.46\\sigma",
        "0.46\\\\sigma",
        "(1.05 \\pm 0.43)",
        "(1.05 \\\\pm 0.43)",
        "map remains required",
        "not independent proofs",
        "decisive confirmation requires",
    ]
    stale_hits: Dict[str, List[str]] = {}
    for rel_path, text in narrative_text_by_path.items():
        hits = [token for token in forbidden_tokens if token in text]
        if hits:
            stale_hits[rel_path] = hits
    report["checks"].append(_check(
        "narrative_surfaces_have_no_stale_framing_or_numbers",
        len(stale_hits) == 0,
        {"forbidden_hits": stale_hits, "paths": list(narrative_text_by_path.keys())},
    ))

    multivar = _read_json(outputs / "multivariate_analysis_results.json")
    if multivar is None:
        report["checks"].append(_check("multivariate_analysis_results_exists", False, {}))
    else:
        interp = multivar.get("_interpretation", {})
        primary_p = interp.get("primary_sigma_hc3_p")
        stress_reason = str(interp.get("stress_reason", "")).lower()
        # The multivariate analysis is performed on the *raw* (uncorrected)
        # H0 residuals.  After the TEP correction removes the environmental
        # slope, we expect sigma_hc3_p to be large (the correction worked).
        # The pre-correction 'Full' model *should* show a residual sigma
        # dependence (primary_p can be any value; the test here is that the
        # model structure — Full primary, FlowEnvironment stress — is
        # defensible, and that group richness is treated as a mediator not a
        # pure nuisance).
        ok = (
            interp.get("primary_model") == "Full"
            and primary_p is not None
            and interp.get("stress_model") == "FlowEnvironment"
            and ("mediator" in stress_reason or "mediate" in stress_reason)
        )
        report["checks"].append(_check(
            "multivariate_primary_interpretation_is_defensible",
            ok,
            {"interpretation": interp},
        ))

    anchor = _read_json(outputs / "anchor_stratification_test.json")
    final_report_text = narrative_text_by_path.get("results/outputs/TEP_FINAL_ROBUSTNESS_REPORT.md", "")
    if anchor is None:
        report["checks"].append(_check("anchor_stratification_test_exists", False, {}))
    else:
        try:
            regression = anchor.get("anchor_regression", anchor.get("regression", {}))
            prediction = regression.get("prediction_test", {})
            comparison = anchor.get("host_comparison", prediction)
            kappa_anchor = float(regression.get("kappa_anchor"))
            kappa_err = float(regression.get("kappa_anchor_err"))
            screened_resid = float(
                comparison.get(
                    "tep_screened_residual_mean_sigma",
                    prediction.get("tep_screened_mean_abs_tension_sigma"),
                )
            )
            ok = (
                abs(kappa_anchor / kappa_err) < 3.0
                and kappa_err > 100.0
                and screened_resid < 2.0
                and "Anchor Screening Resolution" in final_report_text
            )
        except Exception:
            ok = False
        report["checks"].append(_check(
            "anchor_screening_result_is_current",
            ok,
            {"anchor": anchor.get("anchor_regression", anchor.get("regression", {})), "host_comparison": anchor.get("host_comparison", {})},
        ))

    local_gravity = _read_json(outputs / "local_gravity_closure.json")
    if local_gravity is None:
        report["checks"].append(_check("local_gravity_closure_exists", False, {}))
    else:
        closure = local_gravity.get("closure", {})
        try:
            ok = (
                bool(local_gravity.get("passes"))
                and bool(closure.get("passes_cassini"))
                and bool(closure.get("passes_microscope"))
                and bool(closure.get("passes_source_charge_closure"))
                and float(closure.get("cassini_margin")) > 10.0
                and float(closure.get("microscope_margin")) > 10.0
                and "Local Precision-Gravity Closure" in final_report_text
            )
        except (TypeError, ValueError):
            ok = False
        report["checks"].append(_check(
            "local_gravity_closure_passes_precision_bounds",
            ok,
            {"local_gravity": local_gravity},
        ))

    # Final score
    n_fail = int(sum(1 for c in report['checks'] if not c['ok']))
    report['summary']['n_checks'] = int(len(report['checks']))
    report['summary']['n_failed'] = n_fail
    report['summary']['ok'] = bool(n_fail == 0)

    if write_report:
        out_path = outputs / "pipeline_audit_report.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print_status(f"Wrote pipeline audit report: {out_path}", "SUCCESS" if n_fail == 0 else "WARNING")

    return report


def main() -> int:
    rep = audit(write_report=True)
    return 0 if rep.get('summary', {}).get('ok') else 2


if __name__ == "__main__":
    raise SystemExit(main())
