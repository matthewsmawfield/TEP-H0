import numpy as np
import pandas as pd
df = pd.read_csv("results/outputs/stratified_h0.csv")
y = df['h0_derived'].values
mu_err = df['error'].values

h0_diag_err = y * (np.log(10) / 5.0) * mu_err
w = 1.0 / h0_diag_err**2
mu0 = np.average(y, weights=w)
chi2_null_diag = np.sum(w * (y - mu0)**2)
print(f"chi2_null_diag: {chi2_null_diag}")
print(f"Mean H0 error: {np.mean(h0_diag_err)}")
print(f"H0 scatter: {np.std(y)}")

cov = np.load("results/outputs/h0_covariance.npy")
cov_diag_err = np.sqrt(np.diag(cov))
print(f"cov diagonal H0 error mean: {np.mean(cov_diag_err)}")
