import pandas as pd
import numpy as np

# Load processed hosts
hosts = pd.read_csv("data/processed/hosts_processed.csv")
# Calculate MU_SH0ES - m_b_corr? Wait, hosts_processed doesn't have m_b_corr.
# Let's get it from Pantheon
pan = pd.read_csv("data/raw/Pantheon+SH0ES.dat", sep=r'\s+')
pan_calib = pan[pan['IS_CALIBRATOR'] == 1].copy()
pan_calib['MB_implied'] = pan_calib['m_b_corr'] - pan_calib['MU_SH0ES']

# Merge with hosts to get sigma
merged = pan_calib.merge(hosts, left_on='CID', right_on='pantheon_id', how='inner')
merged = merged.dropna(subset=['MB_implied', 'sigma_inferred'])
merged['sigma_sq'] = merged['sigma_inferred']**2

# Correlate MB_implied with sigma_sq
corr = np.corrcoef(merged['sigma_sq'], merged['MB_implied'])[0, 1]
print(f"Correlation between implied MB and sigma^2: {corr:.3f}")

# Fit a line
z = np.polyfit(merged['sigma_sq'], merged['MB_implied'], 1)
print(f"Slope: {z[0]:.3e}, Intercept: {z[1]:.3f}")

