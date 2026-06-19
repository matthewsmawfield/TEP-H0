import pandas as pd
import numpy as np

hosts = pd.read_csv("data/processed/hosts_processed.csv")
hosts = hosts.dropna(subset=['sigma_inferred', 'z_cmb', 'source_id', 'vpec'])

# Compute velocity
c = 299792.458
hosts['v_cmb'] = c * hosts['z_cmb']
hosts['v_corr'] = hosts['v_cmb'] - hosts['vpec']

# Compute distance from R22
r22 = pd.read_csv("data/interim/r22_distances.csv")
hosts = hosts.merge(r22[['source_id', 'value']], on='source_id', how='inner')
hosts['distance_mpc'] = 10**((hosts['value'] - 25)/5)

hosts['h0_raw'] = hosts['v_cmb'] / hosts['distance_mpc']
hosts['h0_corr'] = hosts['v_corr'] / hosts['distance_mpc']

MIN_REDSHIFT = 0.0035
hosts = hosts[hosts['z_cmb'] >= MIN_REDSHIFT].copy()
anchors = ["NGC 4258", "LMC", "SMC", "M 31", "MW"]
hosts = hosts[~hosts["normalized_name"].isin(anchors)].copy()

hosts['sigma_sq'] = hosts['sigma_inferred']**2

corr_h0_raw = np.corrcoef(hosts['sigma_sq'], hosts['h0_raw'])[0, 1]
corr_h0_corr = np.corrcoef(hosts['sigma_sq'], hosts['h0_corr'])[0, 1]

print(f"Correlation h0_raw vs sigma^2: {corr_h0_raw:.3f}")
print(f"Correlation h0_corr vs sigma^2: {corr_h0_corr:.3f}")

z_raw = np.polyfit(hosts['sigma_sq'], hosts['h0_raw'], 1)
z_corr = np.polyfit(hosts['sigma_sq'], hosts['h0_corr'], 1)

print(f"Slope raw: {z_raw[0]:.3e}")
print(f"Slope corr: {z_corr[0]:.3e}")

