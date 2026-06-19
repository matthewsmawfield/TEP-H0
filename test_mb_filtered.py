import pandas as pd
import numpy as np

hosts = pd.read_csv("data/processed/hosts_processed.csv")
pan = pd.read_csv("data/raw/Pantheon+SH0ES.dat", sep=r'\s+')
pan_calib = pan[pan['IS_CALIBRATOR'] == 1].copy()
pan_calib['MB_implied'] = pan_calib['m_b_corr'] - pan_calib['MU_SH0ES']

merged = pan_calib.merge(hosts, left_on='CID', right_on='pantheon_id', how='inner')
merged = merged.dropna(subset=['MB_implied', 'sigma_inferred', 'z_cmb'])

MIN_REDSHIFT = 0.0035
merged = merged[merged['z_cmb'] >= MIN_REDSHIFT].copy()
anchors = ["NGC 4258", "LMC", "SMC", "M 31", "MW"]
merged = merged[~merged["normalized_name"].isin(anchors)].copy()

merged['sigma_sq'] = merged['sigma_inferred']**2

corr = np.corrcoef(merged['sigma_sq'], merged['MB_implied'])[0, 1]
print(f"Correlation between implied MB and sigma^2: {corr:.3f}")
z = np.polyfit(merged['sigma_sq'], merged['MB_implied'], 1)
print(f"Slope: {z[0]:.3e}")

