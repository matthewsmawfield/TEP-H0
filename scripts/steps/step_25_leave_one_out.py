import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.tep_correction import tep_correction, C_SQUARED_KM_S

class Step24LeaveOneOut:
    def __init__(self):
        self.data_dir = PROJECT_ROOT / "data" / "interim"
        self.results_dir = PROJECT_ROOT / "results" / "outputs"
        self.sigma_ref = 87.17
        self.C2 = C_SQUARED_KM_S

    def run(self):
        df = pd.read_csv(self.results_dir / "step_04_tep_corrected_h0.csv")
        df = df[df["z_hd"] > 0.0035].copy().reset_index(drop=True)
        
        # Load baseline full sample values
        sigma_all = df["sigma_inferred"].values
        h0_all = df["h0_derived"].values
        r_full, p_full = stats.pearsonr(sigma_all**2, h0_all)
        
        with open(self.results_dir / "step_04_tep_correction_results.json", "r") as f:
            full_results = json.load(f)
            kappa_full = full_results["optimal_kappa_cep"]
            h0_corr_full = full_results["unified_h0"]

        results = []
        
        for i, row in df.iterrows():
            host_dropped = row["normalized_name"]
            
            # Create LOO sample
            df_loo = df.drop(index=i).copy()
            
            sigma_loo = df_loo["sigma_inferred"].values
            h0_loo = df_loo["h0_derived"].values
            S_loo = df_loo["shear_suppression"].values
            mu_loo = df_loo["value"].values
            v_loo = df_loo["velocity"].values
            
            # 1. Recompute r and p
            r_loo, p_loo = stats.pearsonr(sigma_loo**2, h0_loo)
            
            # 2. Re-optimize kappa
            def obj(k):
                corr = tep_correction(sigma_loo, self.sigma_ref, k[0], S_loo)
                mc = mu_loo + corr
                mu_fid = 5 * np.log10(v_loo) + 25 - 5 * np.log10(70.0)
                delta_mu = mc - mu_fid
                slope_b, _ = np.polyfit(sigma_loo, delta_mu, 1) # Note: polyfit on sigma, not sigma**2 in step 3
                return slope_b ** 2

            res = minimize(
                obj,
                x0=[kappa_full],
                method="Nelder-Mead",
                options={"xatol": 100.0, "fatol": 1e-8, "maxiter": 500},
            )
            
            kappa_loo = res.x[0] if res.success else np.nan
            
            # 3. Recompute H0 corr
            if not np.isnan(kappa_loo):
                corr = tep_correction(sigma_loo, self.sigma_ref, kappa_loo, S_loo)
                mc = mu_loo + corr
                dc = 10 ** ((mc - 25) / 5)
                hc = v_loo / dc
                h0_corr_loo = float(np.mean(hc))
            else:
                h0_corr_loo = np.nan
                
            results.append({
                "Dropped Host": host_dropped,
                "$\\Delta r$": f"{(r_loo - r_full):.3f}",
                "$\\Delta p$": f"{(p_loo - p_full):.4f}",
                "$\\Delta \\kappa$": f"{((kappa_loo - kappa_full)/1e6):.2f}\\times 10^6",
                "$\\Delta H_0^{\\rm corr}$": f"{(h0_corr_loo - h0_corr_full):.2f}"
            })

        out_df = pd.DataFrame(results)
        # Sort by impact on kappa
        def get_kappa_val(x):
            try:
                return float(x.split("\\times")[0])
            except:
                return 0.0
        out_df["kappa_val"] = out_df["$\\Delta \\kappa$"].apply(get_kappa_val)
        out_df = out_df.sort_values(by="kappa_val", ascending=False).drop(columns=["kappa_val"])
        
        out_df.to_csv(self.results_dir / "step_25_leave_one_out.csv", index=False)
        print_status("Leave-One-Out table generated.", "SUCCESS")

if __name__ == "__main__":
    logger = TEPLogger("step_24")
    set_step_logger(logger)
    Step24LeaveOneOut().run()
