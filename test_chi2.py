import numpy as np

n = 29
np.random.seed(42)

# Create a symmetric positive definite matrix
A = np.random.randn(n, n)
C = A @ A.T + np.eye(n) * 10
cov_inv = np.linalg.inv(C)

y = np.random.randn(n) * 10 + 73
x = np.random.randn(n) * 1e-7

# Project out common mode
ones = np.ones(n)
denom = float(ones @ cov_inv @ ones)
Pmat = np.eye(n) - np.outer(ones, ones @ cov_inv) / denom

y_proj = Pmat @ y
x_proj = Pmat @ x

chi2_null_proj = float(y_proj @ cov_inv @ y_proj)
xPx = float(x_proj @ cov_inv @ x_proj)
xPy = float(x_proj @ cov_inv @ y_proj)

beta_proj = xPy / xPx
chi2_tep_proj = chi2_null_proj - (xPy ** 2) / xPx

print(f"Projected Delta Chi2: {chi2_null_proj - chi2_tep_proj}")

# Now via direct GLS
def gls_fit(X, y, C):
    cov_inv = np.linalg.inv(C)
    XtCi = X.T @ cov_inv
    fisher = XtCi @ X
    fisher_inv = np.linalg.inv(fisher)
    beta = fisher_inv @ (XtCi @ y)
    return beta

X0 = np.ones((n, 1))
beta0 = gls_fit(X0, y, C)
resid0 = y - beta0[0]
chi2_null_gls = resid0 @ cov_inv @ resid0

X = np.column_stack([np.ones(n), x])
beta = gls_fit(X, y, C)
resid = y - X @ beta
chi2_tep_gls = resid @ cov_inv @ resid

print(f"GLS Delta Chi2: {chi2_null_gls - chi2_tep_gls}")
print(f"GLS chi2_null: {chi2_null_gls}, chi2_tep: {chi2_tep_gls}")
print(f"Proj chi2_null: {chi2_null_proj}, chi2_tep: {chi2_tep_proj}")

