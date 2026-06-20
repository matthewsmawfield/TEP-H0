import re
from pathlib import Path

files_to_sync = [
    "manuscripts/11-TEP-H0-v0.7-KingstonUponHull.md",
    "site/components/4_results.html",
    "site/components/5_discussion.html",
    "site/components/6_conclusion.html",
    "site/CITATION.cff",
    "site/codemeta.json",
    "zenodo.txt"
]

for fpath in files_to_sync:
    p = Path(fpath)
    if not p.exists():
        continue
    content = p.read_text()
    content = re.sub(r"κ_Cep = \([\d\.]+ ± [\d\.]+\) × 10⁶", "κ_Cep = (1.05 ± 0.41) × 10⁶", content)
    content = re.sub(r"κ_Cep ≈ [\d\.]+ × 10⁶", "κ_Cep ≈ 1.05 × 10⁶", content)
    content = re.sub(r"1\.46 \\times 10\^6", r"1.05 \\times 10^6", content)
    content = re.sub(r"1\.62 \\times 10\^6", r"1.05 \\times 10^6", content)
    content = re.sub(r"tension from 5σ to [\d\.]+σ", r"tension from 5σ to 0.91σ", content)
    content = re.sub(r"tension from 5\\sigma to [\d\.]+\\sigma", r"tension from 5\\sigma to 0.91\\sigma", content)
    content = re.sub(r"unified H₀ = [\d\.]+ km/s/Mpc", "unified H₀ = 68.75 km/s/Mpc", content)
    
    # In citation and metadata
    content = re.sub(r"1\.46", "1.05", content) # just for kappa if it's there
    content = re.sub(r"0\.70", "0.91", content) # for tension
    
    p.write_text(content)
    print(f"Synced {fpath}")
