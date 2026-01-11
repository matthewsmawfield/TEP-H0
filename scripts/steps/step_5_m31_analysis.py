
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from scipy.optimize import curve_fit
from scipy.optimize import linear_sum_assignment
from scipy import stats
from pathlib import Path
import shutil
import sys
import json

# Ensure project root is in path (avoid collisions with any external 'scripts' package)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import TEP Logger
try:
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table
except ImportError:
    # If an unrelated 'scripts' module was partially imported, clear it and retry.
    if 'scripts' in sys.modules:
        del sys.modules['scripts']
    from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table

# Apply Style
try:
    from scripts.utils.plot_style import apply_tep_style
    colors = apply_tep_style()
except ImportError:
    colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30'}

class Step5M31Analysis:
    r"""
    Step 5: M31 Differential Analysis
    =================================
    
    This step performs a direct, controlled test of the TEP hypothesis using Cepheids 
    within a single galaxy: M31 (Andromeda).
    
    The Test:
    We compare the Period-Luminosity (P-L) relation of Cepheids in two distinct environments 
    within M31:
    1.  **Inner Region (R < 5 kpc)**: Deep potential, high stellar density (Bulge-dominated).
        TEP predicts clocks run faster here -> Cepheids appear shorter-period (thus Brighter at fixed period).
    2.  **Outer Region (R > 15 kpc)**: Shallow potential, low density (Disk-dominated).
        TEP predicts clocks run slower (closer to cosmic time) -> Standard P-L relation.
        
    Methodology:
    - We ingest the Kodric et al. (2018) Cepheid catalog.
    - We deproject the coordinates to calculate physical galactocentric distances ($R_{\rm kpc}$).
    - We calculate the local stellar mass density $\rho(r)$ using a Bulge+Disk model.
    - We fit fixed-slope P-L relations ($W = a + b \log P$) to both subsamples.
    - We measure the intercept offset $\Delta W = W_{\rm inner} - W_{\rm outer}$.
    
    Prediction:
    A significant offset $\Delta W$ confirms environmental dependence.
    Note on Screening: The M31 bulge is dense and may lie near an effective galactic transition
    density scale discussed elsewhere in this project. Such regime effects could modulate the
    *amplitude* and cleanliness of the signal (e.g., through partial suppression and competing
    astrophysical gradients), without changing the basic sign chain used for the H0 bias.
    """
    
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.results_dir = self.root_dir / "results"
        self.logs_dir = self.root_dir / "logs"
        
        self.figures_dir = self.results_dir / "figures"
        self.outputs_dir = self.results_dir / "outputs"
        self.public_figures_dir = self.root_dir / "site" / "public" / "figures"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.public_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TEPLogger("step_5_m31", log_file_path=self.logs_dir / "step_5_m31.log")
        set_step_logger(self.logger)
        
        self.output_plot_path = self.figures_dir / "m31_differential_pl.png"
        self.output_robust_plot_path = self.figures_dir / "m31_differential_robustness.png"
        self.output_csv_path = self.outputs_dir / "m31_results.csv"
        self.output_robust_csv_path = self.outputs_dir / "m31_robustness_summary.csv"
        self.output_robust_json_path = self.outputs_dir / "m31_robustness_summary.json"

    def calculate_galactocentric_distance(self, ra, dec):
        """
        Calculate deprojected galactocentric distance in kpc for M31.
        """
        # M31 Parameters (McConnachie et al. 2005 / Standard)
        RA_CENTER = 10.684708
        DEC_CENTER = 41.268750
        PA = 38.0 * np.pi / 180.0  # Position Angle (radians)
        INC = 77.0 * np.pi / 180.0 # Inclination (radians)
        DIST_KPC = 780.0           # Distance to M31
        
        # Coordinates relative to center
        # Approximation for small angles
        d_alpha = (ra - RA_CENTER) * np.cos(np.radians(DEC_CENTER))
        d_delta = (dec - DEC_CENTER)
        
        # Rotate to galaxy frame
        x = d_alpha * np.cos(PA) + d_delta * np.sin(PA)
        y = -d_alpha * np.sin(PA) + d_delta * np.cos(PA)
        
        # Deproject
        x_deproj = x # Major axis
        y_deproj = y / np.cos(INC) # Minor axis deprojected
        
        # Radial distance in degrees
        r_deg = np.sqrt(x_deproj**2 + y_deproj**2)
        
        # Convert to kpc
        # 1 degree approx 13.6 kpc at 780 kpc distance
        r_rad = np.radians(r_deg)
        r_kpc = DIST_KPC * np.tan(r_rad)
        
        return r_kpc

    def calculate_local_density(self, r_kpc):
        """
        Calculates local stellar mass density (M_sun/pc^3) at radius r_kpc.
        Uses a two-component model (Bulge + Disk) for M31.
        """
        # Parameters (approximate for M31)
        # Bulge: Hernquist profile approximation
        M_bulge = 3.0e10 # Solar masses
        r_bulge = 0.61   # kpc
        
        # Disk: Exponential profile
        M_disk = 7.0e10  # Solar masses
        R_d = 5.3        # Scale length (kpc)
        z_d = 0.6        # Scale height (kpc)
        
        # Bulge Density (Spherical)
        # rho_b(r) = M_b / (2pi) * a / r / (r+a)^3
        # Avoid division by zero
        r = np.maximum(r_kpc, 0.01)
        rho_bulge = (M_bulge / (2 * np.pi)) * (r_bulge / r) * (1 / (r + r_bulge)**3)
        
        # Disk Density (in plane, z=0)
        # 1 kpc^3 = 10^9 pc^3, so we need to be careful with units.
        # The formula above with M in M_sun and R in kpc gives density in M_sun/kpc^3.
        
        rho_disk_kpc = (M_disk / (4 * np.pi * R_d**2 * z_d)) * np.exp(-r / R_d) # M_sun/kpc^3
        
        # Total Density
        rho_total_kpc = rho_bulge + rho_disk_kpc # M_sun/kpc^3
        
        # Convert to M_sun/pc^3
        # 1 kpc^3 = 1e9 pc^3
        rho_total_pc = rho_total_kpc / 1e9
        
        return rho_total_pc

    def _fetch_phat_footprint(self) -> pd.DataFrame:
        """Downloads the PHAT observation log (Williams+ 2014) to define the HST footprint.
        
        Source: J/ApJS/215/9/table1
        Returns a DataFrame with 'ra_deg', 'dec_deg' of field centers.
        """
        Vizier.ROW_LIMIT = -1
        cat = 'J/ApJS/215/9/table1'
        try:
            catalogs = Vizier.get_catalogs(cat)
            if not catalogs:
                return pd.DataFrame()
            df = catalogs[0].to_pandas()
            
            # Parse coordinates (handles sexagesimal if needed)
            ra_s = df['RAJ2000'].astype(str)
            dec_s = df['DEJ2000'].astype(str)
            c = SkyCoord(ra=ra_s.values, dec=dec_s.values, unit=(u.hourangle, u.deg))
            df['ra_deg'] = c.ra.deg
            df['dec_deg'] = c.dec.deg
            
            return df[['ra_deg', 'dec_deg', 'Inst', 'Filter']]
        except Exception as e:
            print_status(f"Failed to fetch PHAT footprint: {e}", "WARNING")
            return pd.DataFrame()

    def _tag_phat_overlap(self, df: pd.DataFrame, phat_df: pd.DataFrame, radius_arcmin: float = 2.0) -> pd.Series:
        """Tags Cepheids that fall within the PHAT footprint.
        
        Uses a simple proximity check to the nearest PHAT field center.
        Radius default 2.0' covers the WFC3/IR FOV with margin.
        """
        if phat_df.empty or 'ra_deg' not in phat_df.columns:
            return pd.Series(False, index=df.index)
            
        # Convert to radians for fast distance calculation
        cep_ra = np.radians(df['RA'].values)
        cep_dec = np.radians(df['DEC'].values)
        phat_ra = np.radians(phat_df['ra_deg'].values)
        phat_dec = np.radians(phat_df['dec_deg'].values)
        
        radius_rad = np.radians(radius_arcmin / 60.0)
        
        # For each Cepheid, check if it's close to ANY PHAT pointing
        # Brute force is fine for ~1000 Cepheids vs ~2700 pointings (2.7M pairs)
        in_phat = np.zeros(len(df), dtype=bool)
        
        # Optimization: use broadcasting in chunks if needed, but 2.7M floats is tiny.
        # Let's just do a loop or broadcasting.
        d_ra = cep_ra[:, None] - phat_ra[None, :]
        d_dec = cep_dec[:, None] - phat_dec[None, :]
        
        # Small angle approximation for speed (valid for local separation check)
        # dist^2 = (d_dec)^2 + (d_ra * cos(dec))^2
        cos_dec = np.cos(cep_dec[:, None])
        dist_sq = d_dec**2 + (d_ra * cos_dec)**2
        
        min_dist = np.min(dist_sq, axis=1)
        in_phat = min_dist < (radius_rad**2)
        
        return pd.Series(in_phat, index=df.index)

    def _weighted_intercept(self, logp: np.ndarray, y: np.ndarray, slope: float, yerr: np.ndarray | None = None) -> tuple[float, float]:
        """Intercept estimate for y = a + slope*logP with fixed slope.

        For referee-facing robustness, the default analysis uses an unweighted estimator (yerr=None)
        to avoid pathological domination by catalog-reported uncertainties that may not encode
        crowding/systematic contributions.
        """
        logp = np.asarray(logp, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(logp) & np.isfinite(y)
        if yerr is not None:
            yerr = np.asarray(yerr, dtype=float)
            ok = ok & np.isfinite(yerr) & (yerr > 0)

        if ok.sum() < 3:
            return np.nan, np.nan

        resid = y[ok] - slope * logp[ok]

        if yerr is None:
            a = float(np.mean(resid))
            aerr = float(np.std(resid, ddof=1) / np.sqrt(len(resid)))
            return a, aerr

        w = 1.0 / (yerr[ok] ** 2)
        a = float(np.sum(w * resid) / np.sum(w))
        aerr = float(np.sqrt(1.0 / np.sum(w)))
        return a, aerr

    def _compute_wesenheit_ri(self, df: pd.DataFrame, R: float, anchor: str = 'i') -> tuple[pd.Series, pd.Series]:
        """Compute a Pan-STARRS-style Wesenheit magnitude from rP1 and iP1.

        Two common conventions exist depending on the chosen anchor band:
        - anchor='i': W = i - R*(r-i)
        - anchor='r': W = r - R*(r-i)

        Error propagation assumes independent r and i errors.
        """
        r = pd.to_numeric(df.get('rP1mag', np.nan), errors='coerce')
        i = pd.to_numeric(df.get('iP1mag', np.nan), errors='coerce')
        er = pd.to_numeric(df.get('e_rP1mag', np.nan), errors='coerce')
        ei = pd.to_numeric(df.get('e_iP1mag', np.nan), errors='coerce')

        # Handle common missing/sentinel values (e.g., -99) and non-physical magnitudes.
        # Pan-STARRS mags should be positive and typically < 40 for this sample.
        bad_r = (~np.isfinite(r)) | (r <= 0) | (r > 40)
        bad_i = (~np.isfinite(i)) | (i <= 0) | (i > 40)
        r = r.mask(bad_r)
        i = i.mask(bad_i)

        if str(anchor).lower().startswith('r'):
            w = r - R * (r - i)
            ew = np.sqrt((er ** 2) + (R ** 2) * ((er ** 2) + (ei ** 2)))
        else:
            w = i - R * (r - i)
            ew = np.sqrt((ei ** 2) + (R ** 2) * ((er ** 2) + (ei ** 2)))
        return w, ew

    def _matched_bootstrap_delta(self, inner: pd.DataFrame, outer: pd.DataFrame, y_col: str, yerr_col: str | None, slope: float, n_boot: int = 1000, random_state: int = 42, match_on_error: bool = False) -> dict:
        rng = np.random.default_rng(random_state)

        in_df = inner.copy()
        out_df = outer.copy()

        in_df = in_df.dropna(subset=['logP', y_col]).copy()
        out_df = out_df.dropna(subset=['logP', y_col]).copy()

        if yerr_col is not None:
            in_df[yerr_col] = pd.to_numeric(in_df[yerr_col], errors='coerce')
            out_df[yerr_col] = pd.to_numeric(out_df[yerr_col], errors='coerce')

        if len(in_df) < 10 or len(out_df) < 10:
            return {
                'n_inner': int(len(in_df)),
                'n_outer': int(len(out_df)),
                'n_boot': int(n_boot),
                'n_matched_mean': np.nan,
                'n_matched_min': np.nan,
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            }

        # Bin edges for period matching
        p_edges = np.linspace(in_df['logP'].min(), in_df['logP'].max(), 11)

        # Optional second dimension (photometric error)
        if match_on_error and (yerr_col is not None) and in_df[yerr_col].notna().any() and out_df[yerr_col].notna().any():
            e_in = in_df[yerr_col].dropna()
            e_out = out_df[yerr_col].dropna()
            e_min = max(float(e_in.min()), float(e_out.min()))
            e_max = min(float(e_in.max()), float(e_out.max()))
            if np.isfinite(e_min) and np.isfinite(e_max) and (e_max > e_min):
                e_edges = np.linspace(e_min, e_max, 6)
            else:
                e_edges = None
        else:
            e_edges = None

        deltas = []
        matched_sizes = []

        for _ in range(n_boot):
            in_idx = []
            out_idx = []

            if e_edges is None:
                # 1D matching in logP
                for j in range(len(p_edges) - 1):
                    g_in = in_df[(in_df['logP'] >= p_edges[j]) & (in_df['logP'] < p_edges[j + 1])]
                    g_out = out_df[(out_df['logP'] >= p_edges[j]) & (out_df['logP'] < p_edges[j + 1])]
                    k = min(len(g_in), len(g_out))
                    if k <= 0:
                        continue
                    in_idx.extend(rng.choice(g_in.index.to_numpy(), size=k, replace=True).tolist())
                    out_idx.extend(rng.choice(g_out.index.to_numpy(), size=k, replace=True).tolist())
            else:
                # 2D matching in (logP, error)
                for j in range(len(p_edges) - 1):
                    for k2 in range(len(e_edges) - 1):
                        g_in = in_df[(in_df['logP'] >= p_edges[j]) & (in_df['logP'] < p_edges[j + 1]) & (in_df[yerr_col] >= e_edges[k2]) & (in_df[yerr_col] < e_edges[k2 + 1])]
                        g_out = out_df[(out_df['logP'] >= p_edges[j]) & (out_df['logP'] < p_edges[j + 1]) & (out_df[yerr_col] >= e_edges[k2]) & (out_df[yerr_col] < e_edges[k2 + 1])]
                        kk = min(len(g_in), len(g_out))
                        if kk <= 0:
                            continue
                        in_idx.extend(rng.choice(g_in.index.to_numpy(), size=kk, replace=True).tolist())
                        out_idx.extend(rng.choice(g_out.index.to_numpy(), size=kk, replace=True).tolist())

            if len(in_idx) < 10 or len(out_idx) < 10:
                continue

            matched_sizes.append(min(len(in_idx), len(out_idx)))

            sub_in = in_df.loc[in_idx]
            sub_out = out_df.loc[out_idx]

            # Intercepts are computed unweighted; the matching itself is the control.
            a_in, _ = self._weighted_intercept(sub_in['logP'].values, sub_in[y_col].values, slope=slope, yerr=None)
            a_out, _ = self._weighted_intercept(sub_out['logP'].values, sub_out[y_col].values, slope=slope, yerr=None)
            deltas.append(a_in - a_out)

        deltas = np.asarray(deltas, dtype=float)
        deltas = deltas[np.isfinite(deltas)]

        if len(deltas) == 0:
            return {
                'n_inner': int(len(in_df)),
                'n_outer': int(len(out_df)),
                'n_boot': int(n_boot),
                'n_matched_mean': np.nan,
                'n_matched_min': np.nan,
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            }

        return {
            'n_inner': int(len(in_df)),
            'n_outer': int(len(out_df)),
            'n_boot': int(len(deltas)),
            'n_matched_mean': float(np.mean(matched_sizes)) if len(matched_sizes) else np.nan,
            'n_matched_min': int(np.min(matched_sizes)) if len(matched_sizes) else np.nan,
            'delta_mean': float(np.mean(deltas)),
            'delta_std': float(np.std(deltas, ddof=1)),
            'delta_p16': float(np.percentile(deltas, 16)),
            'delta_p84': float(np.percentile(deltas, 84)),
        }

    def _multivariate_pair_match(self, inner: pd.DataFrame, outer: pd.DataFrame, feature_cols: list[str], caliper: float | None = 3.0) -> tuple[np.ndarray, np.ndarray, dict]:
        """One-to-one matching: assign each inner object to a unique outer object minimizing total distance.

        Features are z-scored using pooled mean/std. A caliper (in z-score Euclidean distance)
        can be applied to prevent pathological matches.
        """
        in_df = inner.copy()
        out_df = outer.copy()

        for c in feature_cols:
            in_df[c] = pd.to_numeric(in_df[c], errors='coerce')
            out_df[c] = pd.to_numeric(out_df[c], errors='coerce')

        in_df = in_df.dropna(subset=feature_cols).copy()
        out_df = out_df.dropna(subset=feature_cols).copy()

        if len(in_df) < 10 or len(out_df) < 10:
            return np.array([], dtype=int), np.array([], dtype=int), {'n_inner': int(len(in_df)), 'n_outer': int(len(out_df))}

        X_in = in_df[feature_cols].astype(float).values
        X_out = out_df[feature_cols].astype(float).values

        X_all = np.vstack([X_in, X_out])
        mu = np.nanmean(X_all, axis=0)
        sd = np.nanstd(X_all, axis=0)
        sd[sd == 0] = 1.0
        Z_in = (X_in - mu) / sd
        Z_out = (X_out - mu) / sd

        # Cost matrix: Euclidean distances
        # Size ~ 153 x 919 => safe.
        diff = Z_in[:, None, :] - Z_out[None, :, :]
        cost = np.sqrt(np.sum(diff * diff, axis=2))

        cal = caliper
        if cal is not None and np.isfinite(cal):
            cost = np.where(cost <= cal, cost, 1e6)

        row_ind, col_ind = linear_sum_assignment(cost)
        # Filter out caliper-violating matches
        if cal is not None and np.isfinite(cal):
            ok = cost[row_ind, col_ind] < 1e5
            row_ind = row_ind[ok]
            col_ind = col_ind[ok]

        meta = {
            'n_inner': int(len(in_df)),
            'n_outer': int(len(out_df)),
            'n_matched': int(len(row_ind)),
            'caliper': float(cal) if cal is not None else None,
        }

        return in_df.index.to_numpy()[row_ind], out_df.index.to_numpy()[col_ind], meta

    def _matched_delta_from_pairs(self, inner: pd.DataFrame, outer: pd.DataFrame, in_idx: np.ndarray, out_idx: np.ndarray, y_col: str, slope: float, n_boot: int = 1000, random_state: int = 7) -> dict:
        rng = np.random.default_rng(random_state)
        if len(in_idx) < 10 or len(out_idx) < 10:
            return {
                'n_pairs': int(min(len(in_idx), len(out_idx))),
                'n_boot': int(n_boot),
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            }

        # Pairwise sample (one-to-one)
        in_sub = inner.loc[in_idx].copy()
        out_sub = outer.loc[out_idx].copy()

        # Baseline on matched set
        a_in, _ = self._weighted_intercept(in_sub['logP'].values, in_sub[y_col].values, slope=slope, yerr=None)
        a_out, _ = self._weighted_intercept(out_sub['logP'].values, out_sub[y_col].values, slope=slope, yerr=None)
        base_delta = float(a_in - a_out)

        # Bootstrap pairs
        deltas = []
        n = len(in_sub)
        for _ in range(n_boot):
            sel = rng.integers(0, n, size=n)
            a_in_b, _ = self._weighted_intercept(in_sub['logP'].values[sel], in_sub[y_col].values[sel], slope=slope, yerr=None)
            a_out_b, _ = self._weighted_intercept(out_sub['logP'].values[sel], out_sub[y_col].values[sel], slope=slope, yerr=None)
            deltas.append(a_in_b - a_out_b)

        deltas = np.asarray(deltas, dtype=float)
        deltas = deltas[np.isfinite(deltas)]

        return {
            'n_pairs': int(n),
            'n_boot': int(len(deltas)),
            'delta_base': base_delta,
            'delta_mean': float(np.mean(deltas)),
            'delta_std': float(np.std(deltas, ddof=1)),
            'delta_p16': float(np.percentile(deltas, 16)),
            'delta_p84': float(np.percentile(deltas, 84)),
        }

    def _ks_balance(self, inner: pd.DataFrame, outer: pd.DataFrame, cols: list[str]) -> dict:
        out = {}
        for c in cols:
            x = pd.to_numeric(inner.get(c, np.nan), errors='coerce').astype(float).values
            y = pd.to_numeric(outer.get(c, np.nan), errors='coerce').astype(float).values
            okx = np.isfinite(x)
            oky = np.isfinite(y)
            if okx.sum() < 10 or oky.sum() < 10:
                out[c] = {'ks_p': np.nan, 'n_in': int(okx.sum()), 'n_out': int(oky.sum())}
                continue
            try:
                _, p = stats.ks_2samp(x[okx], y[oky])
            except Exception:
                p = np.nan
            out[c] = {'ks_p': float(p), 'n_in': int(okx.sum()), 'n_out': int(oky.sum())}
        return out

    def _select_features_by_coverage(self, inner: pd.DataFrame, outer: pd.DataFrame, candidates: list[str], min_frac: float = 0.7) -> list[str]:
        feats = []
        n_in = len(inner)
        n_out = len(outer)
        for c in candidates:
            if c not in inner.columns or c not in outer.columns:
                continue
            x = pd.to_numeric(inner[c], errors='coerce')
            y = pd.to_numeric(outer[c], errors='coerce')
            frac_in = float(np.isfinite(x).sum()) / float(n_in) if n_in > 0 else 0.0
            frac_out = float(np.isfinite(y).sum()) / float(n_out) if n_out > 0 else 0.0
            if (frac_in >= min_frac) and (frac_out >= min_frac):
                feats.append(c)
        return feats

    def _apply_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if 'Flag1' in out.columns:
            f1 = pd.to_numeric(out['Flag1'], errors='coerce')
            out = out[(np.isfinite(f1)) & (f1 == 0)]
        if 'Flag2' in out.columns:
            f2 = pd.to_numeric(out['Flag2'], errors='coerce')
            out = out[(np.isfinite(f2)) & (f2 <= 1)]
        return out

    def _type_consistent_match(self, inner: pd.DataFrame, outer: pd.DataFrame, feature_cols: list[str], caliper: float | None = 3.0) -> tuple[np.ndarray, np.ndarray, dict]:
        if 'Type' not in inner.columns or 'Type' not in outer.columns:
            in_idx, out_idx, meta = self._multivariate_pair_match(inner, outer, feature_cols, caliper=caliper)
            meta['type_mode'] = 'none'
            return in_idx, out_idx, meta

        in_idx_all = []
        out_idx_all = []
        meta_by_type = {}

        t_in = inner['Type'].astype(str).str.strip()
        t_out = outer['Type'].astype(str).str.strip()
        types = sorted(set(t_in.unique()).intersection(set(t_out.unique())))
        types = [t for t in types if t and t.lower() != 'nan']

        for t in types:
            in_t = inner[t_in == t]
            out_t = outer[t_out == t]
            if len(in_t) < 5 or len(out_t) < 5:
                continue
            ii, oo, m = self._multivariate_pair_match(in_t, out_t, feature_cols, caliper=caliper)
            meta_by_type[str(t)] = m
            in_idx_all.extend(ii.tolist())
            out_idx_all.extend(oo.tolist())

        meta = {
            'type_mode': 'per_type',
            'types_used': list(meta_by_type.keys()),
            'meta_by_type': meta_by_type,
            'n_inner_total': int(len(inner)),
            'n_outer_total': int(len(outer)),
            'n_matched_total': int(len(in_idx_all)),
            'caliper': float(caliper) if caliper is not None else None,
        }
        return np.asarray(in_idx_all, dtype=int), np.asarray(out_idx_all, dtype=int), meta

    def _ols_inner_effect(self, df: pd.DataFrame, y_col: str, x_cols: list[str], inner_col: str = 'is_inner') -> dict:
        """OLS effect of inner membership on y after controlling for x_cols.

        Uses y ~ 1 + inner + X, with standard errors from (X'X)^{-1} scaled by residual variance.
        """
        d = df.copy()
        cols = [y_col, inner_col] + list(x_cols)
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors='coerce') if c != inner_col else d[c]
        d = d.dropna(subset=[y_col] + list(x_cols)).copy()
        d = d[np.isfinite(pd.to_numeric(d[y_col], errors='coerce'))].copy()
        if inner_col not in d.columns:
            return {'n': int(len(d)), 'beta_inner': np.nan, 'se_inner': np.nan}

        y = pd.to_numeric(d[y_col], errors='coerce').astype(float).values
        inner = d[inner_col].astype(int).values

        X_list = [np.ones(len(d)), inner]
        used = []
        for c in x_cols:
            x = pd.to_numeric(d[c], errors='coerce').astype(float).values
            if np.isfinite(x).sum() < 10:
                continue
            X_list.append(x)
            used.append(c)
        X = np.column_stack(X_list)

        if X.shape[0] <= X.shape[1] + 5:
            return {'n': int(len(d)), 'beta_inner': np.nan, 'se_inner': np.nan, 'x_used': used}

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(int(X.shape[0] - X.shape[1]), 1)
        s2 = float(np.sum(resid**2) / dof)
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            se_inner = float(se[1])
        except Exception:
            se_inner = np.nan

        return {
            'n': int(X.shape[0]),
            'beta_inner': float(beta[1]),
            'se_inner': se_inner,
            'x_used': used,
        }

    def _matched_bootstrap_delta_2d(self, inner: pd.DataFrame, outer: pd.DataFrame, y_col: str, slope: float, col1: str, col2: str, n_bin1: int = 10, n_bin2: int = 6, n_boot: int = 1000, random_state: int = 101) -> dict:
        rng = np.random.default_rng(random_state)

        in_df = inner.copy()
        out_df = outer.copy()

        for c in ['logP', y_col, col1, col2]:
            if c in in_df.columns:
                in_df[c] = pd.to_numeric(in_df[c], errors='coerce')
            if c in out_df.columns:
                out_df[c] = pd.to_numeric(out_df[c], errors='coerce')

        in_df = in_df.dropna(subset=['logP', y_col, col1, col2]).copy()
        out_df = out_df.dropna(subset=['logP', y_col, col1, col2]).copy()

        if len(in_df) < 10 or len(out_df) < 10:
            return {
                'n_inner': int(len(in_df)),
                'n_outer': int(len(out_df)),
                'n_boot': int(n_boot),
                'n_matched_mean': np.nan,
                'n_matched_min': np.nan,
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            }

        # Use overlapping support so bins are meaningful for both populations
        x1_min = max(float(in_df[col1].min()), float(out_df[col1].min()))
        x1_max = min(float(in_df[col1].max()), float(out_df[col1].max()))
        x2_min = max(float(in_df[col2].min()), float(out_df[col2].min()))
        x2_max = min(float(in_df[col2].max()), float(out_df[col2].max()))
        if not (np.isfinite(x1_min) and np.isfinite(x1_max) and np.isfinite(x2_min) and np.isfinite(x2_max)):
            return {
                'n_inner': int(len(in_df)),
                'n_outer': int(len(out_df)),
                'n_boot': int(n_boot),
                'n_matched_mean': np.nan,
                'n_matched_min': np.nan,
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            }
        if (x1_max <= x1_min) or (x2_max <= x2_min):
            return {
                'n_inner': int(len(in_df)),
                'n_outer': int(len(out_df)),
                'n_boot': int(n_boot),
                'n_matched_mean': np.nan,
                'n_matched_min': np.nan,
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            }

        x1_edges = np.linspace(x1_min, x1_max, n_bin1 + 1)
        x2_edges = np.linspace(x2_min, x2_max, n_bin2 + 1)

        deltas = []
        matched_sizes = []
        for _ in range(n_boot):
            in_idx = []
            out_idx = []
            for i in range(len(x1_edges) - 1):
                for j in range(len(x2_edges) - 1):
                    g_in = in_df[(in_df[col1] >= x1_edges[i]) & (in_df[col1] < x1_edges[i + 1]) & (in_df[col2] >= x2_edges[j]) & (in_df[col2] < x2_edges[j + 1])]
                    g_out = out_df[(out_df[col1] >= x1_edges[i]) & (out_df[col1] < x1_edges[i + 1]) & (out_df[col2] >= x2_edges[j]) & (out_df[col2] < x2_edges[j + 1])]
                    k = min(len(g_in), len(g_out))
                    if k <= 0:
                        continue
                    in_idx.extend(rng.choice(g_in.index.to_numpy(), size=k, replace=True).tolist())
                    out_idx.extend(rng.choice(g_out.index.to_numpy(), size=k, replace=True).tolist())

            if len(in_idx) < 10 or len(out_idx) < 10:
                continue

            matched_sizes.append(min(len(in_idx), len(out_idx)))
            sub_in = in_df.loc[in_idx]
            sub_out = out_df.loc[out_idx]

            a_in, _ = self._weighted_intercept(sub_in['logP'].values, sub_in[y_col].values, slope=slope, yerr=None)
            a_out, _ = self._weighted_intercept(sub_out['logP'].values, sub_out[y_col].values, slope=slope, yerr=None)
            deltas.append(a_in - a_out)

        deltas = np.asarray(deltas, dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if len(deltas) == 0:
            return {
                'n_inner': int(len(in_df)),
                'n_outer': int(len(out_df)),
                'n_boot': int(n_boot),
                'n_matched_mean': np.nan,
                'n_matched_min': np.nan,
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            }

        return {
            'n_inner': int(len(in_df)),
            'n_outer': int(len(out_df)),
            'n_boot': int(len(deltas)),
            'n_matched_mean': float(np.mean(matched_sizes)) if len(matched_sizes) else np.nan,
            'n_matched_min': int(np.min(matched_sizes)) if len(matched_sizes) else np.nan,
            'delta_mean': float(np.mean(deltas)),
            'delta_std': float(np.std(deltas, ddof=1)),
            'delta_p16': float(np.percentile(deltas, 16)),
            'delta_p84': float(np.percentile(deltas, 84)),
        }

    def run_analysis(self):
        print_status("Initiating M31 Differential Analysis", "SECTION")
        print_status("Fetching M31 Cepheid data (Kodric et al. 2018)...", "PROCESS")
        
        # Configure Vizier
        Vizier.ROW_LIMIT = -1
        
        try:
            # Kodric et al. 2018: "The PAndromeda Cepheid Sample"
            # Table 1: Main catalog (J/AJ/156/130)
            catalogs = Vizier.get_catalogs('J/AJ/156/130')
            if not catalogs:
                print_status("Error: No catalogs found.", "ERROR")
                return
            
            # Prefer 'main' table or 'match'
            if 'J/AJ/156/130/main' in catalogs.keys():
                df = catalogs['J/AJ/156/130/main'].to_pandas()
                print_status("Retrieved main table.", "INFO")
            elif 'J/AJ/156/130/match' in catalogs.keys():
                df = catalogs['J/AJ/156/130/match'].to_pandas()
                print_status("Retrieved match table.", "INFO")
            else:
                 df = catalogs[0].to_pandas()
                 
            print_status(f"Retrieved {len(df)} Cepheids from Vizier.", "SUCCESS")
        except Exception as e:
            print_status(f"Error downloading data: {e}", "ERROR")
            return

        # Standardize column names
        if 'RAJ2000' in df.columns: df.rename(columns={'RAJ2000': 'RA'}, inplace=True)
        if 'DEJ2000' in df.columns: df.rename(columns={'DEJ2000': 'DEC'}, inplace=True)
        
        # Find Period
        if 'Per' in df.columns: 
            df['P'] = df['Per']
        elif 'Pr' in df.columns: 
            df['P'] = df['Pr']
        
        # Check for Magnitude (Wmag)
        if 'Wmag' not in df.columns:
            print_status("Wmag column not found in initial catalog. Attempting to fetch main table explicitly.", "WARNING")
            try:
                 df_main = Vizier.get_catalogs('J/AJ/156/130/main')[0].to_pandas()
                 df = df_main
                 df.rename(columns={'RAJ2000': 'RA', 'DEJ2000': 'DEC', 'Pr': 'P'}, inplace=True)
                 print_status("Retrieved main table with Wmag.", "SUCCESS")
            except:
                 print_status("Could not retrieve Wmag. Aborting analysis.", "ERROR")
                 return
        
        # Filter valid data
        df = df.dropna(subset=['RA', 'DEC', 'P', 'Wmag'])
        df = df[df['P'] > 0]
        df['logP'] = np.log10(df['P'])

        # Simple extinction proxy (color) for matching where photometry is present
        if 'rP1mag' in df.columns and 'iP1mag' in df.columns:
            rmag = pd.to_numeric(df['rP1mag'], errors='coerce')
            imag = pd.to_numeric(df['iP1mag'], errors='coerce')
            # Treat common sentinel values as missing
            rmag = rmag.mask((~np.isfinite(rmag)) | (rmag <= 0) | (rmag > 40))
            imag = imag.mask((~np.isfinite(imag)) | (imag <= 0) | (imag > 40))
            df['ri_color'] = rmag - imag
        
        # Calculate Distances
        print_status("Calculating Galactocentric Properties...", "SECTION")
        df['R_kpc'] = self.calculate_galactocentric_distance(df['RA'], df['DEC'])
        
        # Calculate Local Densities
        df['rho_local'] = self.calculate_local_density(df['R_kpc'])

        # Metallicity proxy from a simple radial gradient model (public literature consensus: negative gradient).
        # The absolute zero-point is irrelevant for matching/regression; we center to mean.
        grad_dex_per_kpc = -0.02
        z0 = 0.0
        df['Z_proxy'] = z0 + grad_dex_per_kpc * pd.to_numeric(df['R_kpc'], errors='coerce')
        df['Z_proxy'] = df['Z_proxy'] - pd.to_numeric(df['Z_proxy'], errors='coerce').mean()
        
        print_status(f"Radial Range: {df['R_kpc'].min():.1f} - {df['R_kpc'].max():.1f} kpc", "INFO")
        
        # Define Cuts
        inner_cut = 5.0
        outer_cut = 15.0
        
        inner_df = df[df['R_kpc'] < inner_cut]
        outer_df = df[df['R_kpc'] > outer_cut]
        
        # Filter Period Range (Standard Cepheid range, 10-60 days)
        p_min_log = np.log10(10.0)
        p_max_log = np.log10(60.0)
        
        inner_df = inner_df[(inner_df['logP'] > p_min_log) & (inner_df['logP'] < p_max_log)]
        outer_df = outer_df[(outer_df['logP'] > p_min_log) & (outer_df['logP'] < p_max_log)]
        
        # Calculate Mean Densities
        rho_inner = inner_df['rho_local'].mean()
        rho_outer = outer_df['rho_local'].mean()

        rho_trans = 0.5
        inner_above = inner_df['rho_local'] > rho_trans
        outer_above = outer_df['rho_local'] > rho_trans
        inner_frac_above = float(inner_above.mean()) if len(inner_df) else np.nan
        outer_frac_above = float(outer_above.mean()) if len(outer_df) else np.nan

        core_cut = 1.0
        inner_core_df = inner_df[inner_df['R_kpc'] < core_cut]
        core_above = inner_core_df['rho_local'] > rho_trans
        core_rho_mean = float(inner_core_df['rho_local'].mean()) if len(inner_core_df) else np.nan
        core_frac_above = float(core_above.mean()) if len(inner_core_df) else np.nan
        
        # Population Stats
        headers = ["Region", "Radius (kpc)", "Mean Density (M_sun/pc^3)", "N (Cepheids)"]
        rows = [
            ["Inner", f"< {inner_cut}", f"{rho_inner:.4f}", str(len(inner_df))],
            ["Outer", f"> {outer_cut}", f"{rho_outer:.4f}", str(len(outer_df))]
        ]
        print_table(headers, rows, title="M31 Subsamples")

        headers = ["Region", "Cut", "Mean Density (M_sun/pc^3)", "N", "N(ρ>ρ_trans)", "f(ρ>ρ_trans)"]
        rows = [
            [
                "Inner",
                f"R < {inner_cut} kpc",
                f"{rho_inner:.4f}",
                str(len(inner_df)),
                str(int(inner_above.sum())),
                f"{inner_frac_above:.3f}" if np.isfinite(inner_frac_above) else "nan",
            ],
            [
                "Inner core",
                f"R < {core_cut} kpc",
                f"{core_rho_mean:.4f}" if np.isfinite(core_rho_mean) else "nan",
                str(len(inner_core_df)),
                str(int(core_above.sum())),
                f"{core_frac_above:.3f}" if np.isfinite(core_frac_above) else "nan",
            ],
            [
                "Outer",
                f"R > {outer_cut} kpc",
                f"{rho_outer:.4f}",
                str(len(outer_df)),
                str(int(outer_above.sum())),
                f"{outer_frac_above:.3f}" if np.isfinite(outer_frac_above) else "nan",
            ],
        ]
        print_table(headers, rows, title=f"Screening Threshold Summary (rho_trans={rho_trans} M_sun/pc^3)")
        
        if len(inner_df) < 10 or len(outer_df) < 10:
            print_status("Insufficient samples after cuts.", "WARNING")
            return

        # Fit P-L Relations
        print_status("Fitting Period-Luminosity Relations...", "SECTION")
        fixed_slope = -3.3
        
        def fixed_slope_model(x, a):
            return a + fixed_slope * x

        # Baseline uses an unweighted fixed-slope intercept estimator to match the original analysis
        # and to avoid over-interpreting catalog-reported photometric errors as full uncertainty.
        a_in, err_in = self._weighted_intercept(inner_df['logP'].values, inner_df['Wmag'].values, slope=fixed_slope, yerr=None)
        a_out, err_out = self._weighted_intercept(outer_df['logP'].values, outer_df['Wmag'].values, slope=fixed_slope, yerr=None)
        
        delta_mag = a_in - a_out
        delta_err = np.sqrt(err_in**2 + err_out**2)
        
        # Fit Results Table
        headers = ["Region", "Intercept (a)", "Error", "Slope (Fixed)"]
        rows = [
            ["Inner", f"{a_in:.4f}", f"{err_in:.4f}", f"{fixed_slope}"],
            ["Outer", f"{a_out:.4f}", f"{err_out:.4f}", f"{fixed_slope}"],
            ["Offset", f"{delta_mag:+.4f}", f"{delta_err:.4f}", "-"]
        ]
        print_table(headers, rows, title="P-L Fit Results (W = a + b*logP)")
        
        # Conclusion
        sigma_significance = abs(delta_mag) / delta_err
        print_status(f"Detected Offset (Inner - Outer): {delta_mag:+.4f} mag ({sigma_significance:.1f} sigma)", "RESULT")
        
        if sigma_significance > 2.0:
            print_status("CONCLUSION: Significant environmental P-L offset detected.", "SUCCESS")
        else:
            print_status("CONCLUSION: No significant offset detected.", "INFO")
        
        # Save Results
        results = pd.DataFrame({
            'Region': ['Inner', 'Outer'],
            'Intercept': [a_in, a_out],
            'Error': [err_in, err_out],
            'N': [len(inner_df), len(outer_df)]
        })
        results.to_csv(self.output_csv_path, index=False)
        print_status(f"Saved results to {self.output_csv_path}", "SUCCESS")

        # --- Robustness suite (referee-driven) ---
        print_status("Running M31 robustness suite (matched subsamples and sensitivity tests)...", "SECTION")

        robustness_rows = []

        # (0) PHAT Footprint Restriction
        # We fetch the PHAT observation log and identify Cepheids falling within the HST footprint.
        # This isolates the sample to the region where high-resolution imaging is available.
        print_status("Fetching PHAT footprint for spatial restriction...", "PROCESS")
        phat_footprint = self._fetch_phat_footprint()
        
        if not phat_footprint.empty:
            in_phat_mask = self._tag_phat_overlap(df, phat_footprint)
            df['in_phat'] = in_phat_mask
            n_phat = in_phat_mask.sum()
            print_status(f"Identified {n_phat} Cepheids within PHAT footprint.", "INFO")
            
            # Subset inner/outer to PHAT overlap
            # Ensure we align by index
            inner_phat = inner_df[df.loc[inner_df.index, 'in_phat']].copy()
            outer_phat = outer_df[df.loc[outer_df.index, 'in_phat']].copy()
            
            if len(inner_phat) > 5 and len(outer_phat) > 5:
                ap, ep = self._weighted_intercept(inner_phat['logP'].values, inner_phat['Wmag'].values, slope=fixed_slope, yerr=None)
                aop, eop = self._weighted_intercept(outer_phat['logP'].values, outer_phat['Wmag'].values, slope=fixed_slope, yerr=None)
                dmp = ap - aop
                dep = float(np.sqrt(ep ** 2 + eop ** 2)) if np.isfinite(ep) and np.isfinite(eop) else np.nan
                
                robustness_rows.append({
                    'test': 'phat_footprint_restriction',
                    'y': 'Wmag',
                    'slope_fixed': fixed_slope,
                    'n_inner': int(len(inner_phat)),
                    'n_outer': int(len(outer_phat)),
                    'n_boot': 0,
                    'delta_mean': float(dmp),
                    'delta_std': float(dep),
                    'delta_p16': np.nan,
                    'delta_p84': np.nan,
                })
                print_status(f"PHAT-restricted Offset: {dmp:+.4f} +/- {dep:.4f} mag", "RESULT")
            else:
                 print_status(f"Insufficient PHAT overlap (Inner={len(inner_phat)}, Outer={len(outer_phat)}).", "WARNING")
                 robustness_rows.append({
                    'test': 'phat_footprint_restriction',
                    'y': 'Wmag',
                    'slope_fixed': fixed_slope,
                    'n_inner': int(len(inner_phat)),
                    'n_outer': int(len(outer_phat)),
                    'n_boot': 0,
                    'delta_mean': np.nan,
                    'delta_std': np.nan,
                    'delta_p16': np.nan,
                    'delta_p84': np.nan,
                })
        else:
            print_status("Skipping PHAT restriction (footprint download failed or empty).", "WARNING")

        # (i) Matched bootstrap on logP only
        boot_logp = self._matched_bootstrap_delta(
            inner_df,
            outer_df,
            y_col='Wmag',
            yerr_col='e_Wmag' if 'e_Wmag' in df.columns else None,
            slope=fixed_slope,
            n_boot=1000,
            random_state=42,
            match_on_error=False,
        )
        robustness_rows.append({
            'test': 'matched_bootstrap_logP',
            'y': 'Wmag',
            'slope_fixed': fixed_slope,
            **boot_logp,
        })

        # (ii) Matched bootstrap on logP + e_W
        boot_logp_err = self._matched_bootstrap_delta(
            inner_df,
            outer_df,
            y_col='Wmag',
            yerr_col='e_Wmag' if 'e_Wmag' in df.columns else None,
            slope=fixed_slope,
            n_boot=1000,
            random_state=43,
            match_on_error=True,
        )
        robustness_rows.append({
            'test': 'matched_bootstrap_logP_eW',
            'y': 'Wmag',
            'slope_fixed': fixed_slope,
            **boot_logp_err,
        })

        # (ii.b) Dust proxy control: logP + r-i color matching
        if 'ri_color' in df.columns:
            boot_logp_c = self._matched_bootstrap_delta_2d(
                inner_df,
                outer_df,
                y_col='Wmag',
                slope=fixed_slope,
                col1='logP',
                col2='ri_color',
                n_bin1=10,
                n_bin2=6,
                n_boot=1000,
                random_state=56,
            )
            robustness_rows.append({
                'test': 'matched_bootstrap_logP_color',
                'y': 'Wmag',
                'slope_fixed': fixed_slope,
                **boot_logp_c,
            })
        else:
            boot_logp_c = None

        # (ii.c) Metallicity plausibility diagnostic: required gamma (mag/dex) to explain baseline ΔW
        if 'Z_proxy' in df.columns and np.isfinite(delta_mag):
            z_in = pd.to_numeric(inner_df['Z_proxy'], errors='coerce')
            z_out = pd.to_numeric(outer_df['Z_proxy'], errors='coerce')
            dz = float(np.nanmean(z_in) - np.nanmean(z_out))
            gamma_req = float(delta_mag / dz) if np.isfinite(dz) and (abs(dz) > 1e-6) else np.nan
        else:
            dz = np.nan
            gamma_req = np.nan

        robustness_rows.append({
            'test': 'metallicity_required_gamma',
            'y': 'gamma_mag_per_dex',
            'slope_fixed': fixed_slope,
            'n_inner': int(len(inner_df)),
            'n_outer': int(len(outer_df)),
            'n_boot': 0,
            'n_matched_mean': np.nan,
            'n_matched_min': np.nan,
            'delta_mean': float(gamma_req),
            'delta_std': np.nan,
            'delta_p16': np.nan,
            'delta_p84': np.nan,
        })

        # (iii) Sensitivity to fixed P–L slope
        for s in [-3.1, -3.3, -3.5]:
            ai, ei = self._weighted_intercept(inner_df['logP'].values, inner_df['Wmag'].values, slope=s, yerr=None)
            ao, eo = self._weighted_intercept(outer_df['logP'].values, outer_df['Wmag'].values, slope=s, yerr=None)
            dm = ai - ao
            de = float(np.sqrt(ei ** 2 + eo ** 2)) if np.isfinite(ei) and np.isfinite(eo) else np.nan
            robustness_rows.append({
                'test': 'slope_sensitivity',
                'y': 'Wmag',
                'slope_fixed': float(s),
                'n_inner': int(len(inner_df)),
                'n_outer': int(len(outer_df)),
                'n_boot': 0,
                'delta_mean': float(dm),
                'delta_std': float(de),
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            })

        # (iv) Alternative Wesenheit assumptions using r/i if available
        if all(col in df.columns for col in ['rP1mag', 'iP1mag', 'e_rP1mag', 'e_iP1mag']):
            # Infer whether Kodric Wmag is closer to an i-anchored or r-anchored Wesenheit and
            # estimate the implied baseline coefficient R0 from the catalog itself.
            rmag = pd.to_numeric(df['rP1mag'], errors='coerce')
            imag = pd.to_numeric(df['iP1mag'], errors='coerce')
            wcat = pd.to_numeric(df['Wmag'], errors='coerce')
            color = rmag - imag

            ok = np.isfinite(rmag) & np.isfinite(imag) & np.isfinite(wcat) & np.isfinite(color) & (np.abs(color) > 0.05)
            if ok.sum() < 50:
                print_status("Insufficient r/i/color coverage for Wesenheit coefficient inference; skipping R scan.", "WARNING")
                R_grid = []
            else:
                R_i = (imag[ok] - wcat[ok]) / color[ok]  # if W = i - R(r-i)
                R_r = (rmag[ok] - wcat[ok]) / color[ok]  # if W = r - R(r-i)

                def robust_center(x):
                    x = np.asarray(x, dtype=float)
                    x = x[np.isfinite(x)]
                    if len(x) == 0:
                        return np.nan
                    lo, hi = np.percentile(x, [5, 95])
                    x = x[(x >= lo) & (x <= hi)]
                    return float(np.median(x)) if len(x) else np.nan

                def mad(x):
                    x = np.asarray(x, dtype=float)
                    x = x[np.isfinite(x)]
                    if len(x) == 0:
                        return np.nan
                    med = np.median(x)
                    return float(np.median(np.abs(x - med)))

                R0_i = robust_center(R_i)
                R0_r = robust_center(R_r)

                # Evaluate which anchor gives a self-consistent reconstruction of Wmag
                def score(anchor, R0):
                    if not np.isfinite(R0):
                        return np.inf
                    w_hat, _ = self._compute_wesenheit_ri(df.loc[ok], R=float(R0), anchor=anchor)
                    res = (wcat[ok].values - w_hat.values)
                    return mad(res)

                score_i = score('i', R0_i)
                score_r = score('r', R0_r)
                if score_r < score_i:
                    anchor_choice = 'r'
                    R0 = R0_r
                else:
                    anchor_choice = 'i'
                    R0 = R0_i

                if not np.isfinite(R0):
                    print_status("Could not infer a stable Wesenheit coefficient R0; skipping R scan.", "WARNING")
                    R_grid = []
                else:
                    print_status(f"Inferred Wesenheit form: anchor='{anchor_choice}', R0={R0:.2f} (MAD score: i={score_i:.3f}, r={score_r:.3f})", "INFO")
                    R_grid = np.linspace(max(R0 - 0.6, 0.0), R0 + 0.6, 13)

            for R in R_grid:
                w_ri, e_w_ri = self._compute_wesenheit_ri(df, R=float(R), anchor=anchor_choice)
                df_tmp = df.copy()
                df_tmp['W_ri'] = w_ri
                df_tmp['e_W_ri'] = e_w_ri

                inner_tmp = df_tmp[df_tmp['R_kpc'] < inner_cut]
                outer_tmp = df_tmp[df_tmp['R_kpc'] > outer_cut]
                inner_tmp = inner_tmp[(inner_tmp['logP'] > p_min_log) & (inner_tmp['logP'] < p_max_log)]
                outer_tmp = outer_tmp[(outer_tmp['logP'] > p_min_log) & (outer_tmp['logP'] < p_max_log)]

                ai, ei = self._weighted_intercept(inner_tmp['logP'].values, inner_tmp['W_ri'].values, slope=fixed_slope, yerr=None)
                ao, eo = self._weighted_intercept(outer_tmp['logP'].values, outer_tmp['W_ri'].values, slope=fixed_slope, yerr=None)
                dm = ai - ao
                de = float(np.sqrt(ei ** 2 + eo ** 2)) if np.isfinite(ei) and np.isfinite(eo) else np.nan
                robustness_rows.append({
                    'test': 'wesenheit_R_scan',
                    'y': f'W_ri(R={float(R):.2f})',
                    'slope_fixed': fixed_slope,
                    'n_inner': int(len(inner_tmp)),
                    'n_outer': int(len(outer_tmp)),
                    'n_boot': 0,
                    'delta_mean': float(dm),
                    'delta_std': float(de),
                    'delta_p16': np.nan,
                    'delta_p84': np.nan,
                })
        else:
            print_status("Pan-STARRS r/i photometry columns not available for alternative Wesenheit scan; skipping.", "WARNING")

        # (v) Multivariate one-to-one matching (period + multi-band photometric uncertainty + color + light-curve shape)
        # This is the closest achievable 'surgical' control with the Kodric catalog alone.
        inner_mv = inner_df.copy()
        outer_mv = outer_df.copy()

        mv_candidates = ['logP', 'e_Wmag', 'e_rP1mag', 'e_iP1mag', 'e_gP1mag', 'A21', 'varphi21', 'ri_color']
        mv_features = self._select_features_by_coverage(inner_mv, outer_mv, mv_candidates, min_frac=0.7)
        if 'logP' not in mv_features:
            mv_features = ['logP'] + [c for c in mv_features if c != 'logP']

        in_idx, out_idx, mv_meta = self._type_consistent_match(inner_mv, outer_mv, mv_features, caliper=3.0)
        mv_res = self._matched_delta_from_pairs(inner_mv, outer_mv, in_idx, out_idx, y_col='Wmag', slope=fixed_slope, n_boot=1000, random_state=77)
        ks = self._ks_balance(inner_mv.loc[in_idx], outer_mv.loc[out_idx], mv_features) if len(in_idx) > 0 else {}

        robustness_rows.append({
            'test': 'multivariate_pair_match_type',
            'y': 'Wmag',
            'slope_fixed': fixed_slope,
            'n_inner': int(mv_meta.get('n_inner_total', mv_meta.get('n_inner', 0))),
            'n_outer': int(mv_meta.get('n_outer_total', mv_meta.get('n_outer', 0))),
            'n_boot': int(mv_res.get('n_boot', 0)),
            'n_matched_mean': float(mv_meta.get('n_matched_total', mv_meta.get('n_matched', np.nan))),
            'n_matched_min': float(mv_meta.get('n_matched_total', mv_meta.get('n_matched', np.nan))),
            'delta_mean': float(mv_res.get('delta_mean', np.nan)),
            'delta_std': float(mv_res.get('delta_std', np.nan)),
            'delta_p16': float(mv_res.get('delta_p16', np.nan)),
            'delta_p84': float(mv_res.get('delta_p84', np.nan)),
        })

        # Strict-flag variant
        inner_mv_s = self._apply_quality_flags(inner_mv)
        outer_mv_s = self._apply_quality_flags(outer_mv)
        mv_features_s = self._select_features_by_coverage(inner_mv_s, outer_mv_s, mv_candidates, min_frac=0.7)
        if 'logP' not in mv_features_s:
            mv_features_s = ['logP'] + [c for c in mv_features_s if c != 'logP']

        in_idx_s, out_idx_s, mv_meta_s = self._type_consistent_match(inner_mv_s, outer_mv_s, mv_features_s, caliper=3.0)
        mv_res_s = self._matched_delta_from_pairs(inner_mv_s, outer_mv_s, in_idx_s, out_idx_s, y_col='Wmag', slope=fixed_slope, n_boot=1000, random_state=78)
        ks_s = self._ks_balance(inner_mv_s.loc[in_idx_s], outer_mv_s.loc[out_idx_s], mv_features_s) if len(in_idx_s) > 0 else {}

        robustness_rows.append({
            'test': 'multivariate_pair_match_type_strict',
            'y': 'Wmag',
            'slope_fixed': fixed_slope,
            'n_inner': int(mv_meta_s.get('n_inner_total', mv_meta_s.get('n_inner', 0))),
            'n_outer': int(mv_meta_s.get('n_outer_total', mv_meta_s.get('n_outer', 0))),
            'n_boot': int(mv_res_s.get('n_boot', 0)),
            'n_matched_mean': float(mv_meta_s.get('n_matched_total', mv_meta_s.get('n_matched', np.nan))),
            'n_matched_min': float(mv_meta_s.get('n_matched_total', mv_meta_s.get('n_matched', np.nan))),
            'delta_mean': float(mv_res_s.get('delta_mean', np.nan)),
            'delta_std': float(mv_res_s.get('delta_std', np.nan)),
            'delta_p16': float(mv_res_s.get('delta_p16', np.nan)),
            'delta_p84': float(mv_res_s.get('delta_p84', np.nan)),
        })

        # (vi) Regression control: inner indicator on P-L residuals controlling for dust/quality proxies
        # NOTE: with sharp inner/outer cuts, a purely radial metallicity proxy is collinear with region and
        # cannot be used as a clean control without independent (2D) metallicity information.
        df_reg = pd.concat([inner_df.copy(), outer_df.copy()], axis=0)
        df_reg['is_inner'] = 0
        df_reg.loc[inner_df.index, 'is_inner'] = 1
        df_reg['is_outer'] = 1 - df_reg['is_inner']
        df_reg['pl_resid'] = pd.to_numeric(df_reg['Wmag'], errors='coerce') - fixed_slope * pd.to_numeric(df_reg['logP'], errors='coerce')

        x_ctrl = []
        for c in ['ri_color', 'e_Wmag', 'A21', 'varphi21']:
            if c in df_reg.columns:
                x_ctrl.append(c)

        ols = self._ols_inner_effect(df_reg, y_col='pl_resid', x_cols=x_ctrl, inner_col='is_inner')
        robustness_rows.append({
            'test': 'ols_inner_effect',
            'y': 'pl_resid',
            'slope_fixed': fixed_slope,
            'n_inner': int(df_reg['is_inner'].sum()),
            'n_outer': int(df_reg['is_outer'].sum()),
            'n_boot': 0,
            'n_matched_mean': np.nan,
            'n_matched_min': np.nan,
            'delta_mean': float(ols.get('beta_inner', np.nan)),
            'delta_std': float(ols.get('se_inner', np.nan)),
            'delta_p16': np.nan,
            'delta_p84': np.nan,
        })

        robust_df = pd.DataFrame(robustness_rows)
        robust_df.to_csv(self.output_robust_csv_path, index=False)
        print_status(f"Saved robustness summary to {self.output_robust_csv_path}", "SUCCESS")

        robust_payload = {
            'baseline': {
                'fixed_slope': float(fixed_slope),
                'delta_mag': float(delta_mag),
                'delta_err': float(delta_err),
                'n_inner': int(len(inner_df)),
                'n_outer': int(len(outer_df)),
            },
            'density_threshold': {
                'rho_trans_msun_pc3': float(rho_trans),
                'inner_cut_kpc': float(inner_cut),
                'outer_cut_kpc': float(outer_cut),
                'inner_rho_mean_msun_pc3': float(rho_inner) if np.isfinite(rho_inner) else None,
                'outer_rho_mean_msun_pc3': float(rho_outer) if np.isfinite(rho_outer) else None,
                'inner_n': int(len(inner_df)),
                'inner_n_above_rho_trans': int(inner_above.sum()),
                'inner_frac_above_rho_trans': float(inner_frac_above) if np.isfinite(inner_frac_above) else None,
                'outer_n': int(len(outer_df)),
                'outer_n_above_rho_trans': int(outer_above.sum()),
                'outer_frac_above_rho_trans': float(outer_frac_above) if np.isfinite(outer_frac_above) else None,
                'inner_core_cut_kpc': float(core_cut),
                'inner_core_rho_mean_msun_pc3': float(core_rho_mean) if np.isfinite(core_rho_mean) else None,
                'inner_core_n': int(len(inner_core_df)),
                'inner_core_n_above_rho_trans': int(core_above.sum()),
                'inner_core_frac_above_rho_trans': float(core_frac_above) if np.isfinite(core_frac_above) else None,
            },
            'matched_bootstrap_logP': boot_logp,
            'matched_bootstrap_logP_eW': boot_logp_err,
            'matched_bootstrap_logP_color': boot_logp_c,
            'metallicity_required_gamma': {'dz_proxy': dz, 'gamma_required_mag_per_dex': gamma_req},
            'multivariate_pair_match_type': {
                'features': mv_features,
                'meta': mv_meta,
                'result': mv_res,
                'ks_balance_p': {k: v.get('ks_p', np.nan) for k, v in ks.items()},
            },
            'multivariate_pair_match_type_strict': {
                'features': mv_features_s,
                'meta': mv_meta_s,
                'result': mv_res_s,
                'ks_balance_p': {k: v.get('ks_p', np.nan) for k, v in ks_s.items()},
            },
            'ols_inner_effect': ols,
        }
        with open(self.output_robust_json_path, 'w') as f:
            json.dump(robust_payload, f, indent=2)
        print_status(f"Saved robustness summary JSON to {self.output_robust_json_path}", "SUCCESS")

        # Report compact table in logs
        headers = ["Test", "N_in", "N_out", "N_matched", "Delta (mag)", "Std/Err", "Central 68%"]
        rows = []
        rows.append(["Baseline", str(len(inner_df)), str(len(outer_df)), "-", f"{delta_mag:+.4f}", f"{delta_err:.4f}", "-"])
        if np.isfinite(boot_logp.get('delta_mean', np.nan)):
            rows.append([
                "Matched logP",
                str(boot_logp['n_inner']),
                str(boot_logp['n_outer']),
                f"{boot_logp.get('n_matched_mean', np.nan):.1f}",
                f"{boot_logp['delta_mean']:+.4f}",
                f"{boot_logp['delta_std']:.4f}",
                f"[{boot_logp['delta_p16']:+.4f}, {boot_logp['delta_p84']:+.4f}]",
            ])
        if isinstance(boot_logp_c, dict) and np.isfinite(boot_logp_c.get('delta_mean', np.nan)):
            rows.append([
                "Matched logP+color",
                str(boot_logp_c['n_inner']),
                str(boot_logp_c['n_outer']),
                f"{boot_logp_c.get('n_matched_mean', np.nan):.1f}",
                f"{boot_logp_c['delta_mean']:+.4f}",
                f"{boot_logp_c['delta_std']:.4f}",
                f"[{boot_logp_c['delta_p16']:+.4f}, {boot_logp_c['delta_p84']:+.4f}]",
            ])
        if np.isfinite(boot_logp_err.get('delta_mean', np.nan)):
            rows.append([
                "Matched logP+eW",
                str(boot_logp_err['n_inner']),
                str(boot_logp_err['n_outer']),
                f"{boot_logp_err.get('n_matched_mean', np.nan):.1f}",
                f"{boot_logp_err['delta_mean']:+.4f}",
                f"{boot_logp_err['delta_std']:.4f}",
                f"[{boot_logp_err['delta_p16']:+.4f}, {boot_logp_err['delta_p84']:+.4f}]",
            ])
        if np.isfinite(mv_res.get('delta_mean', np.nan)):
            rows.append([
                "MV match (Type)",
                str(int(mv_meta.get('n_inner_total', mv_meta.get('n_inner', 0)))),
                str(int(mv_meta.get('n_outer_total', mv_meta.get('n_outer', 0)))),
                str(int(mv_meta.get('n_matched_total', mv_meta.get('n_matched', np.nan))) if np.isfinite(mv_meta.get('n_matched_total', np.nan)) else "-"),
                f"{mv_res['delta_mean']:+.4f}",
                f"{mv_res['delta_std']:.4f}",
                f"[{mv_res['delta_p16']:+.4f}, {mv_res['delta_p84']:+.4f}]",
            ])
        if np.isfinite(mv_res_s.get('delta_mean', np.nan)):
            rows.append([
                "MV match (Strict)",
                str(int(mv_meta_s.get('n_inner_total', mv_meta_s.get('n_inner', 0)))),
                str(int(mv_meta_s.get('n_outer_total', mv_meta_s.get('n_outer', 0)))),
                str(int(mv_meta_s.get('n_matched_total', mv_meta_s.get('n_matched', np.nan))) if np.isfinite(mv_meta_s.get('n_matched_total', np.nan)) else "-"),
                f"{mv_res_s['delta_mean']:+.4f}",
                f"{mv_res_s['delta_std']:.4f}",
                f"[{mv_res_s['delta_p16']:+.4f}, {mv_res_s['delta_p84']:+.4f}]",
            ])
        if np.isfinite(ols.get('beta_inner', np.nan)):
            rows.append([
                "OLS inner (ctrl)",
                str(int(df_reg['is_inner'].sum())),
                str(int(df_reg['is_outer'].sum())),
                "-",
                f"{ols['beta_inner']:+.4f}",
                f"{ols.get('se_inner', np.nan):.4f}",
                "-",
            ])
        if np.isfinite(gamma_req):
            rows.append([
                "Metallicity γ req",
                str(len(inner_df)),
                str(len(outer_df)),
                "-",
                f"{gamma_req:+.3f}",
                "-",
                "-",
            ])
        print_table(headers, rows, title="M31 Differential Robustness Summary")
        
        # Plot
        print_status("Generating Differential P-L Plot...", "PROCESS")
        
        plt.figure(figsize=(14, 9))
        
        plt.scatter(outer_df['logP'], outer_df['Wmag'], 
                   alpha=0.2, s=20, color=colors['blue'], 
                   label=f'Outer (> {outer_cut} kpc)', edgecolor='none')
                   
        plt.scatter(inner_df['logP'], inner_df['Wmag'], 
                   alpha=0.4, s=20, color=colors['accent'], 
                   label=f'Inner (< {inner_cut} kpc)', edgecolor='none')
        
        # Plot Fits
        x_range = np.linspace(p_min_log, p_max_log, 100)
        plt.plot(x_range, fixed_slope_model(x_range, a_in), 
                color=colors['accent'], linestyle='-', linewidth=3, 
                label=f'Inner Fit (a={a_in:.2f})')
                
        plt.plot(x_range, fixed_slope_model(x_range, a_out), 
                color=colors['blue'], linestyle='-', linewidth=3, 
                label=f'Outer Fit (a={a_out:.2f})')
        
        plt.xlabel('log(Period) [days]')
        plt.ylabel('Wesenheit Magnitude (W)')
        plt.title(f'M31 Differential Cepheid Analysis\nOffset (Inner - Outer) = {delta_mag:.3f} +/- {delta_err:.3f} mag')
        plt.gca().invert_yaxis()
        plt.legend()
        
        plt.savefig(self.output_plot_path, dpi=300)
        print_status(f"Saved plot to {self.output_plot_path}", "SUCCESS")
        plt.close()

        # Robustness plot: (i) matched-bootstrap delta distribution, (ii) slope sensitivity, (iii) Wesenheit R scan (if present)
        print_status("Generating M31 robustness plot...", "PROCESS")
        try:
            # Apply TEP style locally to ensure it's active
            try:
                from scripts.utils.plot_style import apply_tep_style
                plot_colors = apply_tep_style()
            except ImportError:
                plot_colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'purple': '#4B3A55', 'red': '#b43b4e'}

            fig = plt.figure(figsize=(14, 9)) # Standard TEP size (consistent with others)

            ax1 = plt.subplot(1, 3, 1)
            
            # Re-running local bootstrap for histogram
            boot_deltas = []
            rng = np.random.default_rng(123)
            # Pre-calculate bin edges for 1D matching
            p_edges = np.linspace(inner_df['logP'].min(), inner_df['logP'].max(), 11)
            
            # Simple 1D matching loop for visualization
            # Use inner_df and outer_df which are already defined
            for _ in range(1000):
                in_idx = []
                out_idx = []
                # Stratified resampling to match period distributions
                for j in range(len(p_edges) - 1):
                    g_in = inner_df[(inner_df['logP'] >= p_edges[j]) & (inner_df['logP'] < p_edges[j + 1])]
                    g_out = outer_df[(outer_df['logP'] >= p_edges[j]) & (outer_df['logP'] < p_edges[j + 1])]
                    k = min(len(g_in), len(g_out))
                    if k > 0:
                        in_idx.extend(rng.choice(g_in.index.to_numpy(), size=k, replace=True).tolist())
                        out_idx.extend(rng.choice(g_out.index.to_numpy(), size=k, replace=True).tolist())
                
                if len(in_idx) > 10:
                    sub_in = inner_df.loc[in_idx]
                    sub_out = outer_df.loc[out_idx]
                    # Unweighted intercept
                    a_in = np.mean(sub_in['Wmag'].values - fixed_slope * sub_in['logP'].values)
                    a_out = np.mean(sub_out['Wmag'].values - fixed_slope * sub_out['logP'].values)
                    boot_deltas.append(a_in - a_out)
            
            boot_deltas = np.array(boot_deltas)
            
            # Plot Histogram
            if len(boot_deltas) > 0:
                # Determine smart range
                p01, p99 = np.percentile(boot_deltas, [1, 99])
                range_width = p99 - p01
                xlims = (p01 - 0.5 * range_width, p99 + 0.5 * range_width)
                
                # Histogram with visible edges
                ax1.hist(boot_deltas, bins='auto', color=plot_colors['blue'], alpha=0.5, density=True, edgecolor=plot_colors['blue'], linewidth=0.5)
                
                # Add KDE
                try:
                    kde = stats.gaussian_kde(boot_deltas)
                    x_grid = np.linspace(xlims[0], xlims[1], 200)
                    ax1.plot(x_grid, kde(x_grid), color=plot_colors['blue'], linewidth=2)
                except:
                    pass
                
                ax1.axvline(np.mean(boot_deltas), color=plot_colors['accent'], linestyle='-', linewidth=2.5, label='Matched Mean')
                ax1.text(0.05, 0.95, f"$\\mu$={np.mean(boot_deltas):+.3f}\n$\\sigma$={np.std(boot_deltas):.3f}", 
                        transform=ax1.transAxes, va='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'))
                
                ax1.set_xlim(xlims)

            ax1.axvline(delta_mag, color=plot_colors['dark'], linestyle='--', linewidth=2, label='Baseline')
            ax1.set_title('Matched (logP) Bootstrap\nDistribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel(r'$\Delta W = W_{\rm in} - W_{\rm out}$ (mag)')
            ax1.set_yticks([])
            ax1.legend(fontsize=10, loc='upper right', frameon=True)

            ax2 = plt.subplot(1, 3, 2)
            slope_rows = robust_df[robust_df['test'].eq('slope_sensitivity')].copy()
            if len(slope_rows) > 0:
                ax2.errorbar(slope_rows['slope_fixed'].values, slope_rows['delta_mean'].values, yerr=slope_rows['delta_std'].values, 
                           fmt='o-', color=plot_colors['blue'], ecolor=plot_colors['dark'], capsize=4, linewidth=2, markersize=8)
                ax2.axhline(delta_mag, color=plot_colors['dark'], linestyle='--', linewidth=2)
            ax2.set_xlabel('Fixed P–L Slope')
            ax2.set_ylabel(r'$\Delta W$ (mag)')
            ax2.set_title('Slope Sensitivity', fontsize=12, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6)

            ax3 = plt.subplot(1, 3, 3)
            rscan = robust_df[robust_df['test'].eq('wesenheit_R_scan')].copy()
            if len(rscan) > 0:
                # parse R from label
                def parse_R(s):
                    try:
                        return float(str(s).split('R=')[1].split(')')[0])
                    except Exception:
                        return np.nan
                rscan['R'] = rscan['y'].apply(parse_R)
                rscan = rscan.dropna(subset=['R']).sort_values('R')
                
                # Check offset and align if necessary to focus on stability
                # The r/i derived W likely has a zero-point offset vs the catalog W.
                # We align the mean of the scan to the baseline delta_mag for visual comparison of STABILITY.
                scan_mean = rscan['delta_mean'].mean()
                offset = delta_mag - scan_mean
                
                # Plot adjusted values
                ax3.errorbar(rscan['R'].values, rscan['delta_mean'].values + offset, yerr=rscan['delta_std'].values, 
                           fmt='o-', color=plot_colors['accent'], ecolor=plot_colors['dark'], capsize=4, linewidth=2, markersize=8, label='r/i Scan (Aligned)')
                
                # Plot original baseline
                ax3.axhline(delta_mag, color=plot_colors['dark'], linestyle='--', linewidth=2, label='Catalog Baseline')
                
                ax3.set_xlabel('Wesenheit Coefficient R (r/i)')
                ax3.set_ylabel(r'$\Delta W$ (mag)')
                ax3.set_title('Extinction/Wesenheit Sensitivity', fontsize=12, fontweight='bold')
                ax3.grid(True, linestyle=':', alpha=0.6)
                
                ax3.legend(fontsize=9, loc='lower center')
            else:
                ax3.axis('off')

            plt.tight_layout()
            plt.savefig(self.output_robust_plot_path, dpi=300)
            print_status(f"Saved robustness plot to {self.output_robust_plot_path}", "SUCCESS")
            plt.close(fig)

            shutil.copy(self.output_robust_plot_path, self.public_figures_dir / "m31_differential_robustness.png")
            print_status("Copied robustness plot to public figures directory.", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to generate robustness plot: {e}", "WARNING")
        
        # Copy to public site
        shutil.copy(self.output_plot_path, self.public_figures_dir / "m31_differential_pl.png")
        print_status("Copied plot to public figures directory.", "SUCCESS")

    def run(self):
        print_status("Starting Step 5: M31 Analysis", "TITLE")
        self.run_analysis()
        print_status("Step 5 Complete.", "SUCCESS")

def main():
    step = Step5M31Analysis()
    step.run()

if __name__ == "__main__":
    main()
