#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# zenodo.txt replacements
zenodo_path = ROOT / "zenodo.txt"
if zenodo_path.exists():
    text = zenodo_path.read_text()
    text = text.replace("κ_Cep = (1.62 ± 0.89) × 10⁶ mag", "κ_Cep = (1.61 ± 0.80) × 10⁶ mag")
    text = text.replace("σ_ref = 75.25 km/s", "σ_ref = 87.17 km/s")
    text = text.replace("H0^LOOCV = 64.99 ± 1.50 km/s/Mpc, corresponding to a Planck tension of 1.53σ;", "H0^LOOCV = 65.92 ± 1.52 km/s/Mpc;")
    text = text.replace("H0 = 65.22 km/s/Mpc (bootstrap mean 65.09 ± 1.70, Planck tension 1.23σ)", "H0 = 66.14 km/s/Mpc (bootstrap mean 66.22 ± 1.61, Planck tension 0.70σ)")
    zenodo_path.write_text(text)
    print("UPDATED: zenodo.txt")
else:
    print("MISSING: zenodo.txt")

# codemeta.json replacements
codemeta_path = ROOT / "site" / "codemeta.json"
if codemeta_path.exists():
    text = codemeta_path.read_text()
    text = text.replace("$H_0^{\\rm TEP} = 65.22$ km/s/Mpc", "$H_0^{\\rm TEP} = 66.14$ km/s/Mpc")
    text = text.replace("bootstrap $65.09 \\pm 1.70$", "bootstrap $66.22 \\pm 1.61$")
    text = text.replace("$1.23\\sigma$", "$0.70\\sigma$")
    text = text.replace("$\\kappa_{\\rm Cep} = (1.62", "$\\kappa_{\\rm Cep} = (1.61")
    text = text.replace("0.89) \\times 10^6$ mag", "0.80) \\times 10^6$ mag")
    codemeta_path.write_text(text)
    print("UPDATED: site/codemeta.json")
else:
    print("MISSING: site/codemeta.json")

print("Done.")
