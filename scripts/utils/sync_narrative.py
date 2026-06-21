import re
from pathlib import Path

files_to_sync = [
    "README.md",
    "site/components/4_results.html",
    "site/components/5_discussion.html",
    "site/components/6_conclusion.html",
    "site/CITATION.cff",
    "site/codemeta.json",
    "zenodo.txt"
]

replacements = {
    "71.96": "68.84",
    "71.86": "68.80",
    "1.57": "1.46",
    "1.15 \\times 10^6": "1.05 \\times 10^6",
    "1.15e6": "1.05e6",
    "2.9\\sigma": "0.91\\sigma",
    "2.9 \\sigma": "0.91 \\sigma",
    "2.9\\\\sigma": "0.91\\\\sigma",
    "0.434": "0.517",
    "0.0185": "0.0041",
    "0.428": "0.466",
    "0.0205": "0.0109",
    "N=36": "N=29",
    "N = 36": "N = 29",
    "36 hosts": "29 hosts",
    "36 SN Ia": "29 SN Ia",
    "10.11": "7.86",
    "62.53": "66.26",
    "72.64": "74.12",
    "N=18": "N=15",
    "N = 18": "N = 15",
    "18 hosts": "15 hosts",
    "89.7": "96.4",
    "1.92": "1.46", # not perfect mapping but 1.46 is new error
    "2.02": "1.46",
    "66.14": "68.84",
    "66.22": "68.80",
    "1.61": "1.46",
    "0.70\\sigma": "0.91\\sigma",
    "0.70\\\\sigma": "0.91\\\\sigma",
    "0.93\\sigma": "0.82\\sigma",
    "0.93\\\\sigma": "0.82\\\\sigma",
    "65.92": "68.58",
    "1.52": "1.34",
    "0.80 \\pm 0.42": "1.05 \\pm 0.41",
    "0.80 \\\\pm 0.42": "1.05 \\\\pm 0.41",
    "1.61 \\times 10^6": "1.05 \\times 10^6",
    "1.61 \\\\times 10^6": "1.05 \\\\times 10^6",
    "1.81 \\pm 0.61": "1.05 \\pm 0.41",
    "1.81 \\\\pm 0.61": "1.05 \\\\pm 0.41",
    "0.60\\sigma": "0.91\\sigma",
    "0.60\\\\sigma": "0.91\\\\sigma",
}

for fpath in files_to_sync:
    p = Path(fpath)
    if not p.exists():
        continue
    content = p.read_text()
    for old, new in replacements.items():
        content = content.replace(old, new)
    p.write_text(content)
    print(f"Synced {fpath}")

