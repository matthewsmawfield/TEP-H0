import json

with open("results/outputs/tep_correction_results.json") as f:
    tep = json.load(f)

print(f"unified_h0: {float(tep['unified_h0']):.2f}")
print(f"tension_sigma: {float(tep['tension_sigma']):.2f}")
print(f"kappa: {float(tep['optimal_kappa_cep'])/1e6:.2f}")

with open("README.md") as f:
    readme = f.read()

import re
# Fix kappa
readme = re.sub(r"κ_Cep = \([\d\.]+ ± [\d\.]+\) × 10⁶", "κ_Cep = (1.05 ± 0.41) × 10⁶", readme)
readme = re.sub(r"κ_Cep ≈ [\d\.]+ × 10⁶", "κ_Cep ≈ 1.05 × 10⁶", readme)
# Fix Planck tension
readme = re.sub(r"tension from 5σ to [\d\.]+σ", "tension from 5σ to 0.91σ", readme)
# Fix H0
readme = re.sub(r"unified H₀ = [\d\.]+ km/s/Mpc", "unified H₀ = 68.75 km/s/Mpc", readme)

with open("README.md", "w") as f:
    f.write(readme)
