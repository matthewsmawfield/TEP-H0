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

for fpath in files_to_sync:
    p = Path(fpath)
    if not p.exists():
        continue
    content = p.read_text()
    # Handle the specific remaining missing tokens
    content = content.replace(r"2.9\sigma", r"0.91\sigma")
    content = content.replace("2.9\\sigma", "0.91\\sigma")
    content = content.replace("1.15 \\times", "1.05 \\times")
    content = content.replace("1.15 \times", "1.05 \times")
    content = content.replace("1.15", "1.05")
    content = content.replace("2.9", "0.91")
    p.write_text(content)
    print(f"Synced {fpath}")

