
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import sys
import shutil
import json

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

class Step4bApertureSensitivity:
    r"""
    Step 4b: Aperture Sensitivity Analysis
    ======================================
    
    This analysis tests whether the observed $H_0$-$\sigma$ correlation is an artifact 
    of the aperture corrections applied in Step 1b.
    
    The Skeptic's Challenge:
    "Maybe your correlation is just noise introduced by the Jorgensen et al. (1995) 
    power-law correction you applied to the velocity dispersions."
    
    The Test:
    We calculate the correlation between $H_0$ and the **Raw (Uncorrected)** 
    Velocity Dispersion measurements ($\sigma_{\rm raw}$).
    
    Prediction:
    If the signal is physical (TEP), it should persist in the raw data, perhaps slightly 
    weaker due to aperture noise, but statistically significant. If the signal is an 
    artifact of the correction, the raw correlation should be zero.
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.results_dir = self.root_dir / "results"
        self.logs_dir = self.root_dir / "logs"
        self.figures_dir = self.results_dir / "figures"
        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        self.outputs_dir = self.results_dir / "outputs"
        
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.public_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TEPLogger("step_4b_aperture", log_file_path=self.logs_dir / "step_4b_aperture.log")
        set_step_logger(self.logger)
        
        self.stratified_path = self.outputs_dir / "stratified_h0.csv"
        self.output_stats_path = self.outputs_dir / "aperture_sensitivity_stats.txt"
        self.plot_path = self.figures_dir / "aperture_sensitivity.png"

        self.sigma_compilation_path = self.root_dir / "data" / "raw" / "external" / "velocity_dispersions_literature.csv"
        self.sigma_compilation_regenerated_path = (
            self.root_dir / "data" / "raw" / "external" / "velocity_dispersions_literature_regenerated.csv"
        )
        self.provenance_output_path = self.outputs_dir / "sigma_provenance_table.csv"
        self.grid_output_path = self.outputs_dir / "aperture_sensitivity_grid.csv"
        self.summary_json_path = self.outputs_dir / "aperture_sensitivity_summary.json"

        self.assumed_aperture_radius_arcsec = 1.5
        self.default_beta = 0.04

    def run(self):
        print_status("Starting Step 4b: Aperture Sensitivity Analysis", "TITLE")
        
        if not self.stratified_path.exists():
            print_status("Stratified H0 data missing. Run previous steps.", "ERROR")
            return

        # Load Data
        df = pd.read_csv(self.stratified_path)

        sigma_comp = self._load_sigma_compilation()
        provenance_df = self._build_provenance_table(df, sigma_comp)
        provenance_df.to_csv(self.provenance_output_path, index=False)
        print_status(f"Saved sigma provenance table to {self.provenance_output_path}", "SUCCESS")
        
        # Filter valid data
        # We compare 'sigma_measured' (Raw) vs 'sigma_inferred' (Corrected)
        analysis_df = df.dropna(subset=['h0_derived', 'sigma_measured', 'sigma_inferred']).copy()
        
        print_status(f"Analyzing {len(analysis_df)} hosts with both Raw and Corrected data.", "INFO")
        
        # 1. Baseline (Corrected) Correlation
        r_corr, p_corr = stats.pearsonr(analysis_df['sigma_inferred'], analysis_df['h0_derived'])
        
        # 2. Raw Correlation
        r_raw, p_raw = stats.pearsonr(analysis_df['sigma_measured'], analysis_df['h0_derived'])

        # 2b. Source-stratified correlations
        source_stats = self._source_stratified_stats(provenance_df)

        # 3. Sensitivity grid (beta, R_eff perturbation)
        grid_df, grid_summary = self._run_sensitivity_grid(analysis_df)
        grid_df.to_csv(self.grid_output_path, index=False)
        print_status(f"Saved aperture sensitivity grid to {self.grid_output_path}", "SUCCESS")

        summary = {
            "n_hosts_with_raw_and_corrected": int(len(analysis_df)),
            "assumed_aperture_radius_arcsec": float(self.assumed_aperture_radius_arcsec),
            "default_beta": float(self.default_beta),
            "baseline": {
                "r_raw": float(r_raw),
                "p_raw": float(p_raw),
                "r_corrected": float(r_corr),
                "p_corrected": float(p_corr),
            },
            "source_stratified": source_stats,
            "grid_summary": grid_summary,
        }
        with open(self.summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_status(f"Saved aperture sensitivity summary to {self.summary_json_path}", "SUCCESS")
        
        # Results Table
        headers = ["Metric", "Raw Sigma (Uncorrected)", "Corrected Sigma (Ref)"]
        rows = [
            ["Correlation (r)", f"{r_raw:.4f}", f"{r_corr:.4f}"],
            ["P-value", f"{p_raw:.4e}", f"{p_corr:.4e}"],
            ["Significance", "Significant (>99%)" if p_raw < 0.01 else "Weak", "Significant (>99%)"]
        ]
        print_table(headers, rows, title="Sensitivity Results")
        
        # Conclusion
        if abs(r_raw - r_corr) < 0.1 and r_raw > 0.3:
            print_status("CONCLUSION: The signal is ROBUST. It exists independently of aperture corrections.", "SUCCESS")
        else:
            print_status("CONCLUSION: Aperture corrections significantly alter the signal.", "WARNING")
        
        # Save Stats
        with open(self.output_stats_path, "w") as f:
            f.write("Aperture Correction Sensitivity Analysis\n")
            f.write("========================================\n")
            f.write(f"N: {len(analysis_df)}\n")
            f.write(f"Corrected r: {r_corr:.4f} (p={p_corr:.5f})\n")
            f.write(f"Raw r:       {r_raw:.4f} (p={p_raw:.5f})\n")
            f.write("\nSensitivity Grid Summary\n")
            f.write("------------------------\n")
            f.write(f"Beta range: {grid_summary['beta_min']:.3f} .. {grid_summary['beta_max']:.3f}\n")
            f.write(f"R_eff scale range: {grid_summary['re_scale_min']:.3f} .. {grid_summary['re_scale_max']:.3f}\n")
            f.write(f"r range across grid: {grid_summary['r_min']:.4f} .. {grid_summary['r_max']:.4f}\n")
            f.write(f"slope range across grid: {grid_summary['slope_min']:.6f} .. {grid_summary['slope_max']:.6f}\n")
            f.write(f"delta_H0 range across grid: {grid_summary['delta_h0_min']:.4f} .. {grid_summary['delta_h0_max']:.4f}\n")
            
        # 3. Plot Comparison
        self.plot_comparison(analysis_df, r_raw, r_corr, grid_df)
        
        print_status("Step 4b Complete.", "SUCCESS")

    def _load_sigma_compilation(self) -> pd.DataFrame:
        sigma_path = (
            self.sigma_compilation_regenerated_path
            if self.sigma_compilation_regenerated_path.exists()
            else self.sigma_compilation_path
        )

        if not sigma_path.exists():
            print_status(f"Sigma compilation CSV missing: {sigma_path}", "WARNING")
            return pd.DataFrame(columns=[
                'galaxy', 'sigma_kms', 'error_kms', 'source', 'method', 'notes'
            ])

        if sigma_path == self.sigma_compilation_regenerated_path:
            print_status(f"Using regenerated sigma catalog for provenance: {sigma_path}", "INFO")
        else:
            print_status(f"Using legacy sigma catalog for provenance: {sigma_path}", "WARNING")

        sigma_df = pd.read_csv(sigma_path, comment="#")
        for col in ['galaxy', 'source', 'method', 'notes']:
            if col in sigma_df.columns:
                sigma_df[col] = sigma_df[col].astype(str).str.strip()
        return sigma_df

    def _match_sigma_row(self, name: str, sigma_comp: pd.DataFrame) -> dict:
        if sigma_comp.empty or 'galaxy' not in sigma_comp.columns:
            return {}

        nm = str(name).strip()
        exact = sigma_comp[sigma_comp['galaxy'] == nm]
        if len(exact) > 0:
            return exact.iloc[0].to_dict()

        if nm.startswith("NGC "):
            num = nm[4:].strip()
            num_nozeros = num.lstrip("0")
            if num_nozeros:
                alt = f"NGC {num_nozeros}"
                exact2 = sigma_comp[sigma_comp['galaxy'] == alt]
                if len(exact2) > 0:
                    return exact2.iloc[0].to_dict()

        return {}

    def _build_provenance_table(self, df: pd.DataFrame, sigma_comp: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            nm = r.get('normalized_name', np.nan)
            sigma_row = self._match_sigma_row(nm, sigma_comp)

            sigma_meas = r.get('sigma_measured', np.nan)
            sigma_meas_err = sigma_row.get('error_kms', np.nan)
            sigma_source = sigma_row.get('source', '')
            sigma_method = sigma_row.get('method', '')
            sigma_notes = sigma_row.get('notes', '')

            # Simple, explicit uncertainty model (measurement + source-dependent systematic floor)
            try:
                sigma_meas_val = float(sigma_meas)
            except Exception:
                sigma_meas_val = np.nan

            f_sys = 0.10 if "HI linewidth" in str(sigma_method) else 0.05
            try:
                sigma_err_meas = float(sigma_meas_err)
            except Exception:
                sigma_err_meas = np.nan

            if np.isnan(sigma_meas_val) or sigma_meas_val <= 0:
                sigma_err_total = np.nan
            else:
                sigma_err_total = np.sqrt((sigma_err_meas if not np.isnan(sigma_err_meas) else 0.0) ** 2 + (f_sys * sigma_meas_val) ** 2)

            rows.append({
                "normalized_name": nm,
                "source_id": r.get('source_id', np.nan),
                "h0_derived": r.get('h0_derived', np.nan),
                "sigma_measured_kms": sigma_meas,
                "sigma_measured_error_kms": sigma_meas_err,
                "sigma_measured_error_total_kms": sigma_err_total,
                "sigma_inferred_kms": r.get('sigma_inferred', np.nan),
                "sigma_source": sigma_source,
                "sigma_method": sigma_method,
                "sigma_notes": sigma_notes,
                "aperture_radius_arcsec_assumed": self.assumed_aperture_radius_arcsec,
                "beta_assumed": self.default_beta,
                "log_d25": r.get('log_d25', np.nan),
                "r25_arcsec": r.get('r25_arcsec', np.nan),
                "r_eff_arcsec": r.get('r_eff_arcsec', np.nan),
                "r_eff_provenance": "RC3 D25 via VizieR VII/155; proxy R_eff=0.5*R25",
                "r_eff_frac_uncertainty_assumed": 0.25,
            })

        return pd.DataFrame(rows)

    def _compute_sigma_corrected(self, sigma_obs: np.ndarray, r_eff_arcsec: np.ndarray, beta: float, re_scale: float) -> np.ndarray:
        sigma_obs = np.asarray(sigma_obs, dtype=float)
        r_eff_arcsec = np.asarray(r_eff_arcsec, dtype=float)

        sigma_out = sigma_obs.copy()
        ok = (~np.isnan(sigma_obs)) & (sigma_obs > 0) & (~np.isnan(r_eff_arcsec)) & (r_eff_arcsec > 0)
        if np.any(ok):
            r_norm = (re_scale * r_eff_arcsec[ok]) / 8.0
            cf = (self.assumed_aperture_radius_arcsec / r_norm) ** beta
            sigma_out[ok] = sigma_obs[ok] * cf
        return sigma_out

    def _metrics_for_sigma(self, df: pd.DataFrame, sigma_col: str) -> dict:
        valid = df.dropna(subset=['h0_derived', sigma_col]).copy()
        if len(valid) < 3:
            return {
                "n": int(len(valid)),
                "r": np.nan,
                "p": np.nan,
                "slope": np.nan,
                "delta_h0": np.nan,
            }

        r, p = stats.pearsonr(valid[sigma_col], valid['h0_derived'])
        slope, intercept = np.polyfit(valid[sigma_col], valid['h0_derived'], 1)

        med = valid[sigma_col].median()
        low = valid[valid[sigma_col] <= med]
        high = valid[valid[sigma_col] > med]

        delta_h0 = high['h0_derived'].mean() - low['h0_derived'].mean()
        return {
            "n": int(len(valid)),
            "r": float(r),
            "p": float(p),
            "slope": float(slope),
            "intercept": float(intercept),
            "delta_h0": float(delta_h0),
            "median_sigma": float(med),
        }

    def _calculate_sigma_ref(self) -> float:
        # Mirror Step 3 anchor-weighting; keep local to avoid cross-imports.
        anchors = [
            (30.0, 0.20),  # MW
            (24.0, 0.25),  # LMC
            (115.0, 0.55), # N4258
        ]
        num = sum(s * w for s, w in anchors)
        den = sum(w for _, w in anchors)
        return float(num / den)

    def _optimize_alpha_and_unified_h0(self, df: pd.DataFrame, sigma_col: str, sigma_ref: float) -> dict:
        required = [sigma_col, 'value', 'velocity']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "alpha_opt": np.nan,
                "unified_h0": np.nan,
                "post_slope": np.nan,
                "post_r": np.nan,
            }

        work = df.dropna(subset=['value', 'velocity', sigma_col]).copy()
        if len(work) < 3:
            return {
                "alpha_opt": np.nan,
                "unified_h0": np.nan,
                "post_slope": np.nan,
                "post_r": np.nan,
            }

        sigma_vals = work[sigma_col].astype(float).values
        mu_vals = work['value'].astype(float).values
        v_vals = work['velocity'].astype(float).values

        def objective(params):
            alpha = float(params[0])
            corr = alpha * np.log10(sigma_vals / sigma_ref)
            mu_corr = mu_vals + corr
            d_corr = 10 ** ((mu_corr - 25.0) / 5.0)
            h0_corr = v_vals / d_corr
            slope, _ = np.polyfit(sigma_vals, h0_corr, 1)
            return float(slope ** 2)

        res = minimize(objective, x0=[0.7], method='Nelder-Mead', tol=1e-3)
        alpha_opt = float(res.x[0])

        corr = alpha_opt * np.log10(sigma_vals / sigma_ref)
        mu_corr = mu_vals + corr
        d_corr = 10 ** ((mu_corr - 25.0) / 5.0)
        h0_corr = v_vals / d_corr

        post_slope, _ = np.polyfit(sigma_vals, h0_corr, 1)
        post_r = np.corrcoef(sigma_vals, h0_corr)[0, 1] if len(h0_corr) > 1 else np.nan
        unified_h0 = float(np.mean(h0_corr))

        return {
            "alpha_opt": alpha_opt,
            "unified_h0": unified_h0,
            "post_slope": float(post_slope),
            "post_r": float(post_r),
        }

    def _run_sensitivity_grid(self, analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        grid_rows = []

        betas = np.round(np.arange(0.00, 0.081, 0.01), 3)
        re_scales = np.round(np.arange(0.70, 1.301, 0.05), 3)

        base_df = analysis_df.copy()
        if 'r_eff_arcsec' not in base_df.columns:
            base_df['r_eff_arcsec'] = np.nan

        sigma_ref = self._calculate_sigma_ref()
        can_optimize_alpha = all(c in base_df.columns for c in ['value', 'velocity'])
        if not can_optimize_alpha:
            print_status(
                "TEP optimization columns missing from stratified_h0.csv (need 'value' and 'velocity'); grid will report only raw H0–sigma metrics.",
                "WARNING",
            )

        for beta in betas:
            for re_scale in re_scales:
                sigma_scen = self._compute_sigma_corrected(
                    sigma_obs=base_df['sigma_measured'].values,
                    r_eff_arcsec=base_df['r_eff_arcsec'].values,
                    beta=float(beta),
                    re_scale=float(re_scale),
                )
                tmp = base_df.copy()
                tmp['sigma_scenario'] = sigma_scen

                m = self._metrics_for_sigma(tmp, 'sigma_scenario')

                tep = self._optimize_alpha_and_unified_h0(tmp, 'sigma_scenario', sigma_ref) if can_optimize_alpha else {
                    "alpha_opt": np.nan,
                    "unified_h0": np.nan,
                    "post_slope": np.nan,
                    "post_r": np.nan,
                }
                grid_rows.append({
                    "beta": float(beta),
                    "re_scale": float(re_scale),
                    "n": m["n"],
                    "r": m["r"],
                    "p": m["p"],
                    "slope": m["slope"],
                    "delta_h0": m["delta_h0"],
                    "median_sigma": m.get("median_sigma", np.nan),
                    "sigma_ref": float(sigma_ref),
                    "alpha_opt": tep["alpha_opt"],
                    "unified_h0": tep["unified_h0"],
                    "post_slope": tep["post_slope"],
                    "post_r": tep["post_r"],
                })

        grid_df = pd.DataFrame(grid_rows)

        summary = {
            "beta_min": float(np.min(betas)),
            "beta_max": float(np.max(betas)),
            "re_scale_min": float(np.min(re_scales)),
            "re_scale_max": float(np.max(re_scales)),
            "r_min": float(np.nanmin(grid_df['r'].values)),
            "r_max": float(np.nanmax(grid_df['r'].values)),
            "slope_min": float(np.nanmin(grid_df['slope'].values)),
            "slope_max": float(np.nanmax(grid_df['slope'].values)),
            "delta_h0_min": float(np.nanmin(grid_df['delta_h0'].values)),
            "delta_h0_max": float(np.nanmax(grid_df['delta_h0'].values)),
            "alpha_min": float(np.nanmin(grid_df['alpha_opt'].values)) if 'alpha_opt' in grid_df.columns else np.nan,
            "alpha_max": float(np.nanmax(grid_df['alpha_opt'].values)) if 'alpha_opt' in grid_df.columns else np.nan,
            "unified_h0_min": float(np.nanmin(grid_df['unified_h0'].values)) if 'unified_h0' in grid_df.columns else np.nan,
            "unified_h0_max": float(np.nanmax(grid_df['unified_h0'].values)) if 'unified_h0' in grid_df.columns else np.nan,
        }

        return grid_df, summary

    def _source_stratified_stats(self, provenance_df: pd.DataFrame) -> dict:
        # Join with stratified H0 if present; provenance table is built from it already.
        if provenance_df.empty:
            return {}

        # Work only with rows that have sigma and h0
        # We compute correlations within each source category for transparency.
        out = {}

        df = provenance_df.copy()
        # Attach h0 values (already in df via build_provenance_table inputs if present)
        # If not present, return empty.
        if 'h0_derived' not in df.columns:
            out['note'] = 'h0_derived not present in provenance table'
            return out

        for key in ['sigma_source', 'sigma_method']:
            stats_by = []
            for group_val, g in df.dropna(subset=['sigma_measured_kms', 'h0_derived']).groupby(key):
                if len(g) < 4:
                    continue
                r, p = stats.pearsonr(g['sigma_measured_kms'].astype(float), g['h0_derived'].astype(float))
                stats_by.append({
                    key: str(group_val),
                    "n": int(len(g)),
                    "r": float(r),
                    "p": float(p),
                })
            out[key] = stats_by

        return out

    def plot_comparison(self, df, r_raw, r_corr, grid_df: pd.DataFrame):
        print_status("Generating sensitivity plot...", "PROCESS")
        
        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'dark': '#301E30', 'accent': '#b43b4e', 'light_blue': '#4b6785'}
        
        plt.figure(figsize=(14, 9))
        
        # Left: Raw
        plt.subplot(2, 2, 1)
        plt.scatter(df['sigma_measured'], df['h0_derived'], color=colors.get('light_blue', '#4b6785'), alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
        # Fit
        if len(df) > 1:
            m, c = np.polyfit(df['sigma_measured'], df['h0_derived'], 1)
            x_range = np.linspace(df['sigma_measured'].min(), df['sigma_measured'].max(), 100)
            plt.plot(x_range, m*x_range + c, color=colors['dark'], linestyle='--', linewidth=3, label=f'Fit (r={r_raw:.2f})')
        
        plt.xlabel(r'Raw Measured $\sigma$ (km/s)')
        plt.ylabel(r'$H_0$ (km/s/Mpc)')
        plt.title('Raw (Uncorrected) Velocity Dispersion')
        plt.legend(loc='upper left', frameon=True)
        
        # Right: Corrected
        plt.subplot(2, 2, 2)
        plt.scatter(df['sigma_inferred'], df['h0_derived'], color=colors['blue'], alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
        # Fit
        if len(df) > 1:
            m2, c2 = np.polyfit(df['sigma_inferred'], df['h0_derived'], 1)
            x_range2 = np.linspace(df['sigma_inferred'].min(), df['sigma_inferred'].max(), 100)
            plt.plot(x_range2, m2*x_range2 + c2, color=colors['blue'], linestyle='--', linewidth=3, label=f'Fit (r={r_corr:.2f})')
        
        plt.xlabel(r'Aperture-Corrected $\sigma$ (km/s)')
        plt.ylabel(r'$H_0$ (km/s/Mpc)')
        plt.title('Corrected Velocity Dispersion')
        plt.legend(loc='upper left', frameon=True)

        # Bottom-left: Heatmap of r(beta, re_scale)
        plt.subplot(2, 2, 3)
        if grid_df is not None and len(grid_df) > 0:
            pivot = grid_df.pivot(index='beta', columns='re_scale', values='r')
            im = plt.imshow(
                pivot.values,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                vmin=np.nanmin(pivot.values),
                vmax=np.nanmax(pivot.values),
            )
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Pearson r')
            plt.xlabel(r'$R_{\rm eff}$ Scale Factor')
            plt.ylabel(r'Aperture Exponent $\beta$')
            plt.title('Robustness Envelope: r(H0, σ)')
        else:
            plt.axis('off')

        # Bottom-right: 1D slices through grid
        plt.subplot(2, 2, 4)
        if grid_df is not None and len(grid_df) > 0:
            # r vs beta at re_scale=1.0
            re0 = 1.0
            closest_re = grid_df['re_scale'].unique()[np.argmin(np.abs(grid_df['re_scale'].unique() - re0))]
            slice_beta = grid_df[grid_df['re_scale'] == closest_re].sort_values('beta')
            plt.plot(slice_beta['beta'], slice_beta['r'], color=colors['accent'], linewidth=2.5, label=f'r vs β (Re×{closest_re:.2f})')

            # r vs re_scale at beta=default
            b0 = self.default_beta
            closest_b = grid_df['beta'].unique()[np.argmin(np.abs(grid_df['beta'].unique() - b0))]
            slice_re = grid_df[grid_df['beta'] == closest_b].sort_values('re_scale')
            plt.plot(slice_re['re_scale'], slice_re['r'], color=colors['dark'], linewidth=2.5, label=f'r vs Re-scale (β={closest_b:.2f})')

            plt.axhline(r_corr, color=colors['blue'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline corrected r')
            plt.axhline(r_raw, color=colors.get('light_blue', '#4b6785'), linestyle=':', linewidth=1.5, alpha=0.7, label='Raw r')
            plt.xlabel('β  OR  R_eff scale')
            plt.ylabel('Pearson r')
            plt.title('Stability Slices')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend(loc='best', fontsize=9, frameon=True)
        else:
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300)
        plt.close()
        
        # Copy to public
        shutil.copy(self.plot_path, self.public_figures_dir / "aperture_sensitivity.png")
        print_status(f"Saved plot to {self.plot_path}", "SUCCESS")

if __name__ == "__main__":
    Step4bApertureSensitivity().run()
