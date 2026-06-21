from pathlib import Path

fpath = "site/index.html"
p = Path(fpath)
content = p.read_text()

# Meta description replacements
content = content.replace("66.14", "68.84")
content = content.replace("66.22", "68.80")
content = content.replace("± 1.61 km/s/Mpc", "± 1.46 km/s/Mpc")
content = content.replace("0.70 sigma", "0.91 sigma")
content = content.replace("0.97 ± 0.41e6 mag", "1.05 ± 0.41e6 mag")
content = content.replace("optimal 1.61e6", "optimal 1.05e6")

p.write_text(content)
print(f"Synced {fpath}")
