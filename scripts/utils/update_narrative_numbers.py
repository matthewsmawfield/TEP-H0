#!/usr/bin/env python3
"""
DEPRECATED — use scripts/utils/sync_narrative_numbers.py instead,
which reads directly from results/outputs/*.json and does not require
manual hardcoding of target values.

This script is kept only for historical reference. Running it with the
hardcoded REPLACEMENTS below may inject intermediate stale values if
the narrative still contains very old patterns (pre-N=29-sync).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# New headline numbers from latest pipeline run (TEP-H0 N=29 Hubble-flow-safe, v0.7 Kingston upon Hull)
NEW = {
    "spearman_rho": "0.517",
    "spearman_p": "0.0041",
    "pearson_r": "0.466",
    "pearson_p": "0.0109",
    "unified_h0": "68.75",
    "bootstrap_h0_mean": "68.80",
    "bootstrap_h0_std": "1.46",
    "tension_sigma": "0.91",
    "kappa_million": "1.05",
    "kappa_std_million": "0.41",
    "median_sigma": "96.4",
    "low_mean_h0": "66.26",
    "low_std_err": "1.05",
    "high_mean_h0": "74.12",
    "high_std_err": "1.05",
    "delta_h0": "7.86",
}

# Ordered replacements: longer/more specific strings first to avoid partial matches
REPLACEMENTS = [
    # Abstract old numbers (very old, from earlier version)
    ("Spearman $\\rho = 0.634$, $p = 0.0002$; Pearson $r = 0.514$, $p = 0.0043$",
     "Spearman $\\rho = 0.517$, $p = 0.0041$; Pearson $r = 0.466$, $p = 0.0109$"),
    ("$\\sigma_{\\rm med} \\approx 88$ km/s",
     "$\\sigma_{\\rm med} \\approx 96$ km/s"),
    ("$H_0 = 67.94 \\pm 0.94$ km/s/Mpc (low-$\\sigma$; $N=15$) versus $77.28 \\pm 0.99$ km/s/Mpc (high-$\\sigma$; $N=14$), implying $\\Delta H_0 = 9.34$ km/s/Mpc",
     "$H_0 = 66.26 \\pm 2.10$ km/s/Mpc (low-$\\sigma$; $N=15$) versus $74.12 \\pm 1.30$ km/s/Mpc (high-$\\sigma$; $N=14$), implying $\\Delta H_0 = 7.86$ km/s/Mpc"),
    # kappa
    ("$\\kappa_{\\rm Cep} = (1.10 \\pm 0.62) \\times 10^6$ mag",
     "$\\kappa_{\\rm Cep} = (0.99 \\pm 0.56) \\times 10^6$ mag"),
    ("$\\kappa_{\\rm Cep} = (1.10 \\pm 0.62)\\times10^6$ mag",
     "$\\kappa_{\\rm Cep} = (0.99 \\pm 0.56)\\times10^6$ mag"),
    ("$\\kappa_{\\rm Cep} \\approx 1.10\\times10^6$ mag",
     "$\\kappa_{\\rm Cep} \\approx 0.99\\times10^6$ mag"),
    ("$\\langle \\kappa_{\\rm Cep} \\cdot S \\rangle = 1.10\\times10^6$",
     "$\\langle \\kappa_{\\rm Cep} \\cdot S \\rangle = 0.99\\times10^6$"),
    ("$(1.10 \\pm 0.12) \\times 10^6$ mag",
     "$(0.82 \\pm 0.09) \\times 10^6$ mag"),
    # H0 and tension
    ("$H_0^{\\rm LOOCV} = 67.87 \\pm 1.54$ km/s/Mpc",
     "$H_0^{\\rm LOOCV} = 67.95 \\pm 1.32$ km/s/Mpc"),
    ("$0.29\\sigma$", "$0.47\\sigma$"),
    ("$H_0 = 67.87$ km/s/Mpc (bootstrap mean\n        $67.75 \\pm 1.54$)",
     "$H_0 = 68.13$ km/s/Mpc (bootstrap mean\n        $68.06 \\pm 1.49$)"),
    ("$H_0 = 67.87$ km/s/MPc", "$H_0 = 68.13$ km/s/Mpc"),  # typo-safe
    ("$H_0 = 67.87$ km/s/Mpc", "$H_0 = 68.13$ km/s/Mpc"),
    ("bootstrap mean $67.75 \\pm 1.54$", "bootstrap mean $68.06 \\pm 1.49$"),
    ("bootstrap mean $67.75 \\pm 1.54$", "bootstrap mean $68.06 \\pm 1.49$"),
    ("Planck tension $0.29\\sigma$)", "Planck tension $0.47\\sigma$)"),
    # Results / Discussion / Conclusion older numbers
    ("$r = 0.493$ vs. 0.462", "$r = 0.493$ vs. 0.465"),
    ("$r = 0.493$ vs. 0.462", "$r = 0.493$ vs. 0.465"),
    ("Spearman $\\rho = 0.511$, $p = 0.0046$; $N=29$",
     "Spearman $\\rho = 0.517$, $p = 0.0041$; $N=29$"),
    ("Spearman $\\rho = 0.511$, $p = 0.0046$",
     "Spearman $\\rho = 0.517$, $p = 0.0041$"),
    ("Pearson $r=0.462$, $p=0.0116$", "Pearson $r=0.466$, $p=0.0109$"),
    ("Pearson $r = 0.462$, $p = 0.0116$", "Pearson $r = 0.466$, $p = 0.0109$"),
    ("$68.09 \\pm 1.52$", "$68.13 \\pm 1.49$"),
    ("$68.09$ km/s/Mpc", "$68.13$ km/s/Mpc"),
    ("$72.45 \\pm 2.32$ km/s/Mpc", "$74.12 \\pm 1.30$ km/s/Mpc"),
    ("$\\Delta H_0 = 4.63$ km/s/Mpc", "$\\Delta H_0 = 7.86$ km/s/Mpc"),
    ("$\\Delta H_0 = 4.63$", "$\\Delta H_0 = 7.86$"),
    ("68.09 km/s/Mpc", "68.13 km/s/Mpc"),
    ("68.09 km/s/Mpc", "68.13 km/s/Mpc"),
    # index.html specific
    ("TEP-corrected H0 = 67.87 km/s/Mpc (0.29 sigma Planck tension, kappa_Cep = 1.10e6 mag)",
     "TEP-corrected H0 = 68.13 km/s/Mpc (0.47 sigma Planck tension, kappa_Cep = 0.99e6 mag)"),
    ("Spearman $\\rho = 0.511$, $p = 0.0046$; Pearson $r = 0.462$, $p = 0.0116$",
     "Spearman $\\rho = 0.517$, $p = 0.0041$; Pearson $r = 0.466$, $p = 0.0109$"),
    ("$\\sigma_{\\rm med} \\approx 90$ km/s",
     "$\\sigma_{\\rm med} \\approx 96$ km/s"),
    ("$H_0 = 67.82 \\pm 1.62$ km/s/Mpc (low-$\\sigma$; $N=15$) versus $72.45 \\pm 2.32$ km/s/Mpc (high-$\\sigma$; $N=14$), implying $\\Delta H_0 = 4.63$ km/s/Mpc",
     "$H_0 = 66.26 \\pm 2.10$ km/s/Mpc (low-$\\sigma$; $N=15$) versus $74.12 \\pm 1.30$ km/s/Mpc (high-$\\sigma$; $N=14$), implying $\\Delta H_0 = 7.86$ km/s/Mpc"),
    # README / zenodo specific
    ("unified H_0 = 67.87 km/s/Mpc (bootstrap mean 67.75 ± 1.54)",
     "unified H_0 = 68.13 km/s/Mpc (bootstrap mean 68.06 ± 1.49)"),
    ("reducing Planck tension from 5σ to 0.29σ", "reducing Planck tension from 5σ to 0.47σ"),
    ("κ_Cep ≈ 1.10 × 10⁶ mag", "κ_Cep ≈ 0.99 × 10⁶ mag"),
    ("κ_Cep ≈ 1.10 × 10^6 mag", "κ_Cep ≈ 0.99 × 10^6 mag"),
    ("κ_Cep = (1.10 ± 0.62) × 10^6 mag", "κ_Cep = (0.99 ± 0.56) × 10^6 mag"),
    ("κ_Cep = (1.10 ± 0.62) × 10⁶ mag", "κ_Cep = (0.99 ± 0.56) × 10⁶ mag"),
    ("H0^LOOCV = 67.87 ± 1.54 km/s/Mpc", "H0^LOOCV = 67.95 ± 1.32 km/s/Mpc"),
    ("Planck tension of 0.29σ", "Planck tension of 0.47σ"),
    ("in-sample corrected mean is H0 = 67.87 km/s/Mpc", "in-sample corrected mean is H0 = 68.13 km/s/Mpc"),
    ("bootstrap mean 67.75 ± 1.54", "bootstrap mean 68.06 ± 1.49"),
    ("ρ = 0.634, p = 0.0002", "ρ = 0.517, p = 0.0041"),
    ("Pearson r = 0.514, p = 0.0043", "Pearson r = 0.466, p = 0.0109"),
    ("High-σ hosts yield H_0 = 77.28 km/s/Mpc while low-σ hosts yield 67.94 km/s/Mpc—a 9.34 km/s/Mpc environmental bias",
     "High-σ hosts yield H_0 = 74.12 km/s/Mpc while low-σ hosts yield 66.26 km/s/Mpc—a 7.86 km/s/Mpc environmental bias"),
    ("$\\kappa_{\\rm Cep} \\approx 1.10\\times10^6$ mag",
     "$\\kappa_{\\rm Cep} \\approx 0.99\\times10^6$ mag"),
]

# Also need to handle plain number replacements in results tables
PLAIN_REPLACEMENTS = [
    # Be very careful with these — only replace in table cells / specific contexts
    (">67.82<", ">66.26<"),
    (">72.45<", ">74.12<"),
    (">4.63<", ">7.86<"),
    (">0.462<", ">0.465<"),
    (">0.0116<", ">0.0111<"),
    (">0.511<", ">0.512<"),
    (">0.0046<", ">0.0045<"),
    # codemeta.json description field - comprehensive block replacement
    ("Spearman $\\rho = 0.634$, $p = 0.0002$; Pearson $r = 0.514$, $p = 0.0043$). A median-split stratification at $\\sigma_{\\rm med} \\approx 88$ km/s yields $H_0 = 67.94 \\pm 0.94$ km/s/Mpc (low-$\\sigma$; $N=15$) versus $77.28 \\pm 0.99$ km/s/Mpc (high-$\\sigma$; $N=14$), implying $\\Delta H_0 = 9.34$ km/s/Mpc. The TEP correction yields a unified Hubble constant $H_0^{\\rm TEP} = 67.87$ km/s/Mpc (bootstrap $67.75 \\pm 1.54$), reducing the Planck tension to $0.29\\sigma$. The Observable Response Coefficient is $\\kappa_{\\rm Cep} = (1.10 \\pm 0.62) \\times 10^6$ mag.",
     "Spearman $\\rho = 0.517$, $p = 0.0041$; Pearson $r = 0.466$, $p = 0.0109$). A median-split stratification at $\\sigma_{\\rm med} \\approx 96$ km/s yields $H_0 = 66.26 \\pm 2.10$ km/s/Mpc (low-$\\sigma$; $N=15$) versus $74.12 \\pm 1.30$ km/s/Mpc (high-$\\sigma$; $N=14$), implying $\\Delta H_0 = 7.86$ km/s/Mpc. The TEP correction yields a unified Hubble constant $H_0^{\\rm TEP} = 68.13$ km/s/Mpc (bootstrap $68.06 \\pm 1.49$), reducing the Planck tension to $0.47\\sigma$. The Observable Response Coefficient is $\\kappa_{\\rm Cep} = (0.99 \\pm 0.56) \\times 10^6$ mag."),
]

FILES = [
    ROOT / "site" / "components" / "1_abstract.html",
    ROOT / "site" / "components" / "4_results.html",
    ROOT / "site" / "components" / "5_discussion.html",
    ROOT / "site" / "components" / "6_conclusion.html",
    ROOT / "site" / "index.html",
    ROOT / "README.md",
    ROOT / "zenodo.txt",
    ROOT / "site" / "codemeta.json",
]


def update_file(path: Path):
    if not path.exists():
        print(f"SKIP (missing): {path}")
        return
    text = path.read_text()
    original = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    for old, new in PLAIN_REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        print(f"UPDATED: {path}")
    else:
        print(f"UNCHANGED: {path}")


if __name__ == "__main__":
    for f in FILES:
        update_file(f)
    print("\nDone. Remember to rebuild the site: cd site && npm run build")
