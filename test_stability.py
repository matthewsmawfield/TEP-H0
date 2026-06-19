import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load data
df = pd.read_csv("results/outputs/tep_corrected_h0.csv") # Let's assume this exists
cov = np.load("results/outputs/h0_covariance.npy")
with open("results/outputs/h0_covariance_labels.json", 'r') as f:
    cov_labels = json.load(f)

label_to_idx = {str(lbl): i for i, lbl in enumerate(cov_labels)}
idx = [label_to_idx[str(lbl)] for lbl in df["source_id"].astype(str)]
cov_sub = cov[np.ix_(idx, idx)]
cov_sub = 0.5 * (cov_sub + cov_sub.T)

sigma = df["sigma_inferred"].values.astype(float)
y = df["h0_derived"].values.astype(float)
n = len(y)

S = df["shear_suppression"].values.astype(float) if "shear_suppression" in df.columns else np.ones(n)
C_SQUARED_KM_S = 89875517873.68176
sigma_ref = 75.25

x = S * (sigma**2 - sigma_ref**2) / C_SQUARED_KM_S

# Scale x to avoid numerical issues
scale_factor = 1e7
x_scaled = x * scale_factor

# cov_inv
cov_inv = np.linalg.inv(cov_sub + np.eye(n) * (1e-12 * np.trace(cov_sub) / n))

def gls_fit_reg(X, y, cov_inv):
    XtCi = X.T @ cov_inv
    fisher = XtCi @ X
    reg = 1e-10 * np.trace(fisher) / fisher.shape[0]
    fisher_reg = fisher + reg * np.eye(fisher.shape[0])
    fisher_inv = np.linalg.inv(fisher_reg)
    beta = fisher_inv @ (XtCi @ y)
    return beta

X_unscaled = np.column_stack([np.ones(n), x])
beta_unscaled = gls_fit_reg(X_unscaled, y, cov_inv)
resid_tep_unscaled = y - (beta_unscaled[0] + beta_unscaled[1] * x)
chi2_tep_unscaled = resid_tep_unscaled @ cov_inv @ resid_tep_unscaled

print(f"Chi2 TEP (unscaled X with regularization): {chi2_tep_unscaled}")
