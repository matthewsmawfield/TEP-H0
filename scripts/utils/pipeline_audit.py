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
except Exception:
    def print_status(msg: str, level: str = "INFO"):
        print(f"[{level}] {msg}")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _approx(a: float, b: float, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return bool(abs(a - b) <= (atol + rtol * abs(b)))


def _check(name: str, ok: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


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

    # Check TEP correction
    tep = _read_json(outputs / "tep_correction_results.json")
    if tep is None:
        report["checks"].append(_check("tep_correction_results_json_exists", False, {}))
    else:
        # Step 3 defines tension_sigma using the ROBUST bootstrap uncertainty (bootstrap_h0_std)
        # rather than the SEM (h0_sem). We recompute both for transparency.
        tension_robust = None
        tension_sem = None
        try:
            planck_h0 = float(tep['planck_h0'])
            planck_err = 0.5
            h0 = float(tep['unified_h0'])
            if 'bootstrap_h0_std' in tep:
                tension_robust = abs(h0 - planck_h0) / math.sqrt(float(tep['bootstrap_h0_std']) ** 2 + planck_err ** 2)
            if 'h0_sem' in tep:
                tension_sem = abs(h0 - planck_h0) / math.sqrt(float(tep['h0_sem']) ** 2 + planck_err ** 2)
        except Exception:
            tension_robust = None
            tension_sem = None

        ok = (
            int(tep.get('n_hosts')) == derived['n']
            and tension_robust is not None
            and _approx(float(tep.get('tension_sigma')), float(tension_robust), atol=1e-6, rtol=1e-6)
        )
        report["checks"].append(_check(
            "tep_correction_internal_consistency",
            ok,
            {"got": tep, "recomputed_tension_robust": tension_robust, "recomputed_tension_sem": tension_sem},
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

        # Ensure provenance is from regenerated catalog when it exists
        regen_path = data_raw_ext / "velocity_dispersions_literature_regenerated.csv"
        legacy_path = data_raw_ext / "velocity_dispersions_literature.csv"
        report["checks"].append(_check(
            "regenerated_sigma_catalog_present",
            regen_path.exists(),
            {"path": str(regen_path)},
        ))
        report["checks"].append(_check(
            "legacy_sigma_catalog_present",
            legacy_path.exists(),
            {"path": str(legacy_path)},
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
