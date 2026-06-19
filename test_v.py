import pandas as pd
import numpy as np

hosts = pd.read_csv("data/processed/hosts_processed.csv")
hosts = hosts.dropna(subset=['sigma_inferred', 'z_hd', 'source_id'])

# Compute velocity
c = 299792.458
hosts['velocity'] = c * hosts['z_hd']

# Compute distance from R22
r22 = pd.read_csv("data/interim/r22_distances.csv")
hosts = hosts.merge(r22[['source_id', 'value']], on='source_id', how='inner')
hosts['distance_mpc'] = 10**((hosts['value'] - 25)/5)

hosts['h0_derived'] = hosts['velocity'] / hosts['distance_mpc']

MIN_REDSHIFT = 0.0035
hosts = hosts[hosts['z_hd'] >= MIN_REDSHIFT].copy()
anchors = ["NGC 4258", "LMC", "SMC", "M 31", "MW"]
hosts = hosts[~hosts["normalized_name"].isin(anchors)].copy()

hosts['sigma_sq'] = hosts['sigma_inferred']**2

corr_v = np.corrcoef(hosts['sigma_sq'], hosts['velocity'])[0, 1]
corr_d = np.corrcoef(hosts['sigma_sq'], hosts['distance_mpc'])[0, 1]
corr_h0 = np.corrcoef(hosts['sigma_sq'], hosts['h0_derived'])[0, 1]

print(f"Correlation v vs sigma^2: {corr_v:.3f}")
print(f"Correlation d vs sigma^2: {corr_d:.3f}")
print(f"Correlation h0 vs sigma^2: {corr_h0:.3f}")

