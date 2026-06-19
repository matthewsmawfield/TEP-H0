#!/usr/bin/env python3
"""Quick sync of narrative surface headline numbers to current pipeline outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "results" / "outputs"

# Load current headline numbers
tep = json.load(open(OUTPUTS / "tep_correction_results.json"))
oos = json.load(open(OUTPUTS / "out_of_sample_validation.json"))
strat = json.load(open(OUTPUTS / "stratification_results.json"))

# Derived headline numbers
unified_h0 = f"{tep['unified_h0']:.2f}"
boot_h0_mean = f"{tep['bootstrap_h0_mean']:.2f}"
boot_h0_std = f"{tep['bootstrap_h0_std']:.2f}"
tension = f"{tep['tension_sigma']:.2f}"
kappa_mill = f"{tep['optimal_kappa_cep'] / 1e6:.2f}"
kappa_err = tep.get('bootstrap_kappa_robust_std') or tep.get('wls_kappa_err_scaled') or tep.get('bootstrap_kappa_std', 0.89e6)
kappa_std_mill = f"{kappa_err / 1e6:.2f}"
sigma_ref = f"{tep['sigma_ref']:.2f}"
loocv_h0 = f"{oos.get('loocv', {}).get('pred_h0_mean', 0):.2f}"
loocv_sem = f"{oos.get('loocv', {}).get('pred_h0_sem', 0):.2f}"

print(f"Current headline numbers:")
print(f"  unified_h0 = {unified_h0}")
print(f"  bootstrap_h0_mean = {boot_h0_mean}")
print(f"  bootstrap_h0_std = {boot_h0_std}")
print(f"  tension_sigma = {tension}")
print(f"  kappa_mill = {kappa_mill}")
print(f"  kappa_std_mill = {kappa_std_mill}")
print(f"  sigma_ref = {sigma_ref}")
print(f"  loocv_h0 = {loocv_h0}")
print(f"  loocv_sem = {loocv_sem}")

# Ordered replacements: specific/long first
REPLACEMENTS = [
    # Abstract / README / zenodo specific strings
    ("TEP-corrected H0 = 65.22 km/s/Mpc (bootstrap 65.09 ± 1.70 km/s/Mpc, 1.23 sigma Planck tension, kappa_Cep = 0.97 ± 0.41e6 mag joint fit chi2-scaled, host-only optimal 1.62e6)",
     f"TEP-corrected H0 = {unified_h0} km/s/Mpc (bootstrap {boot_h0_mean} ± {boot_h0_std} km/s/Mpc, {tension} sigma Planck tension, kappa_Cep = {kappa_mill} ± {kappa_std_mill}e6 mag)"),
    ("TEP-corrected H0 = 65.22 km/s/Mpc (bootstrap 65.09 ± 1.70 km/s/Mpc, 1.23 sigma Planck tension, kappa_Cep = 0.97 ± 0.41e6 mag joint fit chi2-scaled)",
     f"TEP-corrected H0 = {unified_h0} km/s/Mpc (bootstrap {boot_h0_mean} ± {boot_h0_std} km/s/Mpc, {tension} sigma Planck tension, kappa_Cep = {kappa_mill} ± {kappa_std_mill}e6 mag)"),
    # README abstract
    ("κ_Cep = (1.62 ± 0.89) × 10⁶ mag and effective calibrator reference σ_ref = 75.25 km/s yields a unified local Hubble constant. Out-of-sample validation (leave-one-out cross-validation, LOOCV) predicts H0^LOOCV = 64.99 ± 1.50 km/s/Mpc, corresponding to a Planck tension of 1.53σ; this is the stress test confirms the correction generalises out of sample because the response coefficient is trained on 35 hosts and tested on the held-out host. The in-sample corrected mean is H0 = 65.22 km/s/Mpc (bootstrap mean 65.09 ± 1.70), with the corrected r ≈ 0 a fitted-correction diagnostic rather than an independent validation statistic.",
     f"κ_Cep = ({kappa_mill} ± {kappa_std_mill}) × 10⁶ mag and effective calibrator reference σ_ref = {sigma_ref} km/s yields a unified local Hubble constant. Out-of-sample validation (leave-one-out cross-validation, LOOCV) predicts H0^LOOCV = {loocv_h0} ± {loocv_sem} km/s/Mpc; this stress test confirms the correction generalises out of sample because the response coefficient is trained on 35 hosts and tested on the held-out host. The in-sample corrected mean is H0 = {unified_h0} km/s/Mpc (bootstrap mean {boot_h0_mean} ± {boot_h0_std}), with the corrected r ≈ 0 a fitted-correction diagnostic rather than an independent validation statistic."),
    # Key findings
    ("yielding a unified H₀ = 65.22 km/s/Mpc (bootstrap mean 65.09 ± 1.70), reducing Planck tension from 5σ to 1.23σ.",
     f"yielding a unified H₀ = {unified_h0} km/s/Mpc (bootstrap mean {boot_h0_mean} ± {boot_h0_std}), reducing Planck tension from 5σ to {tension}σ."),
    ("κ_Cep ≈ 1.62 × 10⁶ mag) eliminates this trend, yielding a unified H₀ = 65.22 km/s/Mpc (bootstrap mean 65.09 ± 1.70), reducing Planck tension from 5σ to 1.23σ.",
     f"κ_Cep ≈ {kappa_mill} × 10⁶ mag) eliminates this trend, yielding a unified H₀ = {unified_h0} km/s/Mpc (bootstrap mean {boot_h0_mean} ± {boot_h0_std}), reducing Planck tension from 5σ to {tension}σ."),
    ("κ_Cep ≈ 1.62 × 10⁶ mag) is resolved by group halo screening", 
     f"κ_Cep ≈ {kappa_mill} × 10⁶ mag) is resolved by group halo screening"),
    # site/index.html meta description
    ("TEP-corrected H0 = 65.22 km/s/Mpc (bootstrap 65.09 ± 1.70 km/s/Mpc, 1.23 sigma Planck tension, kappa_Cep = 0.97 ± 0.41e6 mag joint fit chi2-scaled, host-only optimal 1.62e6).",
     f"TEP-corrected H0 = {unified_h0} km/s/Mpc (bootstrap {boot_h0_mean} ± {boot_h0_std} km/s/Mpc, {tension} sigma Planck tension, kappa_Cep = {kappa_mill} ± {kappa_std_mill}e6 mag)."),
    # CITATION.cff / codemeta.json
    ("unified H_0 = 65.22 km/s/Mpc (bootstrap mean 65.09 ± 1.70)",
     f"unified H_0 = {unified_h0} km/s/Mpc (bootstrap mean {boot_h0_mean} ± {boot_h0_std})"),
    ("reducing Planck tension from 5σ to 1.23σ",
     f"reducing Planck tension from 5σ to {tension}σ"),
    # Conclusion / Abstract / Discussion — careful with these
    ("bootstrap mean $65.09 \\pm 1.70$, Planck tension $1.23\\sigma$)",
     f"bootstrap mean ${boot_h0_mean} \\pm {boot_h0_std}$, Planck tension ${tension}\\sigma$)"),
    ("$H_0^{\\rm LOOCV} = 64.99 \\pm 1.50$ km/s/Mpc, corresponding to a Planck tension of $1.53\\sigma$",
     f"$H_0^{{\\rm LOOCV}} = {loocv_h0} \\pm {loocv_sem}$ km/s/Mpc, corresponding to a Planck tension of ${tension}\\sigma$"),
    ("The in-sample corrected mean is $H_0 = 65.22$ km/s/Mpc",
     f"The in-sample corrected mean is $H_0 = {unified_h0}$ km/s/Mpc"),
    ("response coefficient $\\kappa_{\\rm Cep} = (0.97 \\pm 0.41)\\times10^6$ mag",
     f"response coefficient $\\kappa_{{\\rm Cep}} = ({kappa_mill} \\pm {kappa_std_mill})\\times10^6$ mag"),
    ("host-only optimal $1.62\\times10^6$, WLS scaled $1.57 \\pm 0.60$",
     f"host-only optimal ${kappa_mill}\\times10^6$, WLS scaled $1.57 \\pm 0.60$"),
    # zenodo
    ("unified H_0 = 65.22 km/s/Mpc (bootstrap mean 65.09 ± 1.70), reducing Planck tension from 5σ to 1.23σ",
     f"unified H_0 = {unified_h0} km/s/Mpc (bootstrap mean {boot_h0_mean} ± {boot_h0_std}), reducing Planck tension from 5σ to {tension}σ"),
]

FILES = [
    ROOT / "README.md",
    ROOT / "zenodo.txt",
    ROOT / "manuscripts" / "11-TEP-H0-v0.7-KingstonUponHull.md",
    ROOT / "11-TEP-H0-v0.7-KingstonUponHull.md",
    ROOT / "site" / "components" / "1_abstract.html",
    ROOT / "site" / "components" / "6_conclusion.html",
    ROOT / "site" / "CITATION.cff",
    ROOT / "site" / "codemeta.json",
    ROOT / "site" / "index.html",
]


def update_file(path: Path):
    if not path.exists():
        print(f"SKIP (missing): {path}")
        return
    text = path.read_text()
    original = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        print(f"UPDATED: {path}")
    else:
        print(f"UNCHANGED: {path}")


if __name__ == "__main__":
    for f in FILES:
        update_file(f)
    print("\nDone. Rebuild site: cd site && npm run build")
