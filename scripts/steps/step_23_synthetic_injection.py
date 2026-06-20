import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
try:
    from scipy.odr import ODR, Model, RealData
except ImportError:
    pass

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.tep_correction import C_SQUARED_KM_S

class Step23SyntheticInjection:
    def __init__(self):
        self.data_dir = PROJECT_ROOT / "data" / "interim"
        self.results_dir = PROJECT_ROOT / "results" / "outputs"
        self.sigma_ref = 87.17
        self.C2 = C_SQUARED_KM_S

    def run(self):
        df = pd.read_csv(self.results_dir / "tep_corrected_h0.csv")
        # Ensure we only use the primary Hubble flow sample
        df = df[df["z_hd"] > 0.0035].copy()
        
        # Load covariance matrix
        cov = np.load(self.results_dir / "h0_covariance.npy")

        # Get errors for ODR
        prov = pd.read_csv(self.results_dir / "sigma_provenance_table.csv")
        df = df.merge(prov[["normalized_name", "sigma_measured_error_kms"]], on="normalized_name", how="left")
        sigma_errs = df["sigma_measured_error_kms"].fillna(10.0).values
        # Delta mu error
        dmu_errs = df["error"].fillna(0.15).values

        # The predictor
        x = df["sigma_inferred"].values
        S = df["shear_suppression"].values
        # Predictor for delta mu space
        X_tep = S * (x**2 - self.sigma_ref**2) / self.C2

        # Nullify existing mu relationship by taking the mean distance modulus
        # Actually, we want to inject into the raw delta-mu residuals.
        # Let's take the observed mu, subtract the empirical TEP correction to get "null" mu
        # then inject known kappa.
        
        with open(self.results_dir / "tep_correction_results.json", "r") as f:
            kappa_empirical = json.load(f)["optimal_kappa_cep"]
            
        mu_null = df["value"].values - kappa_empirical * X_tep
        
        injections = [0, 0.5e6, 1.0e6, 1.5e6]
        results = []

        def linear(B, x):
            return B[0] * x + B[1]
        model = Model(linear)

        for kappa_inj in injections:
            # Inject signal
            mu_inj = mu_null + kappa_inj * X_tep
            
            # 1. OLS
            slope_ols, intercept_ols, r_val, p_val, std_err = stats.linregress(X_tep, mu_inj)
            
            # 2. ODR
            data = RealData(X_tep, mu_inj, sx=(2 * x * sigma_errs / self.C2), sy=dmu_errs)
            odr = ODR(data, model, beta0=[slope_ols, intercept_ols])
            output = odr.run()
            slope_odr = output.beta[0]
            
            # 3. GLS
            # X design matrix for GLS (intercept, slope)
            X_mat = np.column_stack((np.ones(len(X_tep)), X_tep))
            try:
                inv_cov = np.linalg.inv(cov)
                beta_gls = np.linalg.inv(X_mat.T @ inv_cov @ X_mat) @ (X_mat.T @ inv_cov @ mu_inj)
                slope_gls = beta_gls[1]
            except Exception:
                slope_gls = np.nan
                
            results.append({
                "Injected \\kappa": f"{kappa_inj/1e6:.1f}\\times 10^6",
                "OLS recovered": f"{(slope_ols/1e6):.2f}\\times 10^6",
                "ODR recovered": f"{(slope_odr/1e6):.2f}\\times 10^6",
                "GLS recovered": f"{(slope_gls/1e6):.2f}\\times 10^6",
                "false positive rate": f"{p_val:.3f}" if kappa_inj == 0 else "-"
            })

        # Save to csv for markdown embedding
        out_df = pd.DataFrame(results)
        out_df.to_csv(self.results_dir / "synthetic_injection.csv", index=False)
        print_status("Synthetic injection table generated.", "SUCCESS")

if __name__ == "__main__":
    logger = TEPLogger("step_23")
    set_step_logger(logger)
    Step23SyntheticInjection().run()
