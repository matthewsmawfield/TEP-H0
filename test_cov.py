import numpy as np
import pandas as pd
cov = np.load("results/outputs/h0_covariance.npy")
print(f"H0 cov mean diagonal: {np.mean(np.diag(cov))}")
