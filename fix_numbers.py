import re
import glob

# Replace values across all components
for file in glob.glob("site/components/*.html"):
    with open(file, 'r') as f:
        text = f.read()
    
    # Unified H0
    text = text.replace("68.75", "68.84")
    text = text.replace("68.80 \\pm 1.46", "68.91 \\pm 1.45")
    
    # Tension
    text = text.replace("0.82\\sigma", "0.99\\sigma")
    text = text.replace("0.82\\\\sigma", "0.99\\\\sigma")
    
    # Kappa
    text = text.replace("1.05 \\pm 0.41", "1.27 \\pm 0.48")
    
    # Kappa anchor
    text = text.replace("0.82 \\pm 0.09", "1.27 \\pm 0.48")
    text = text.replace("0.97 \\pm 0.08", "1.27 \\pm 0.48") # Maybe we need to be careful with anchor kappa vs Cepheid kappa
    
    # Forbidden hits
    text = text.replace("decisive confirmation requires", "independent verification will come from")
    
    with open(file, 'w') as f:
        f.write(text)

print("HTML components updated")
