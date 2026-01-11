import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from scipy.optimize import linear_sum_assignment
from scipy import stats
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status, print_table


class Step7LMCReplication:
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

        self.logger = TEPLogger("step_7_lmc", log_file_path=self.logs_dir / "step_7_lmc_replication.log")
        set_step_logger(self.logger)

        self.output_plot_path = self.figures_dir / "lmc_differential_pl.png"
        self.output_robust_plot_path = self.figures_dir / "lmc_differential_robustness.png"
        self.output_csv_path = self.outputs_dir / "lmc_results.csv"
        self.output_robust_csv_path = self.outputs_dir / "lmc_robustness_summary.csv"
        self.output_robust_json_path = self.outputs_dir / "lmc_robustness_summary.json"

        # Default selection for a high-statistics LMC replication.
        # (A long-period-only variant can be added as an explicit robustness slice.)
        self.period_min_days = 2.5
        self.period_max_days = 100.0
        self.lmc_sky_radius_deg = 12.0
        self.inner_quantile = 0.2
        self.outer_quantile = 0.8
        self.min_region_n = 50

    def _angsep_rad(self, ra1_deg: np.ndarray, dec1_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> np.ndarray:
        ra1 = np.radians(np.asarray(ra1_deg, dtype=float))
        dec1 = np.radians(np.asarray(dec1_deg, dtype=float))
        ra0 = float(np.radians(ra0_deg))
        dec0 = float(np.radians(dec0_deg))
        cosang = np.sin(dec1) * np.sin(dec0) + np.cos(dec1) * np.cos(dec0) * np.cos(ra1 - ra0)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang)

    def _parse_ra_dec_deg(self, ra: pd.Series, dec: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Robust RA/Dec parsing to degrees.

        VizieR tables sometimes provide RAJ2000/DEJ2000 as sexagesimal strings.
        We first attempt numeric casting; if coverage is poor, we fall back to
        astropy SkyCoord parsing.
        """
        ra_num = pd.to_numeric(ra, errors='coerce')
        dec_num = pd.to_numeric(dec, errors='coerce')
        ok = np.isfinite(ra_num) & np.isfinite(dec_num)
        if int(ok.sum()) >= max(50, int(0.5 * len(ra))):
            return ra_num, dec_num

        ra_s = ra.astype(str).str.strip()
        dec_s = dec.astype(str).str.strip()
        mask = (ra_s.str.len() > 0) & (dec_s.str.len() > 0) & (~ra_s.str.lower().eq('nan')) & (~dec_s.str.lower().eq('nan'))
        if int(mask.sum()) < 10:
            return ra_num, dec_num

        ra_deg = pd.Series(np.nan, index=ra.index, dtype=float)
        dec_deg = pd.Series(np.nan, index=dec.index, dtype=float)

        # Try hourangle+deg first (common for RAJ2000 strings), then deg+deg.
        for unit in [(u.hourangle, u.deg), (u.deg, u.deg)]:
            try:
                c = SkyCoord(ra=ra_s[mask].values, dec=dec_s[mask].values, unit=unit, frame='icrs')
                ra_deg.loc[mask] = c.ra.deg
                dec_deg.loc[mask] = c.dec.deg
                ok2 = np.isfinite(ra_deg) & np.isfinite(dec_deg)
                if int(ok2.sum()) >= max(50, int(0.5 * len(ra_deg))):
                    return ra_deg, dec_deg
            except Exception:
                continue

        # Last resort: return whatever we managed to parse.
        return ra_deg, dec_deg

    def _weighted_intercept(self, logp: np.ndarray, y: np.ndarray, slope: float, yerr: np.ndarray | None = None) -> tuple[float, float]:
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

    def _matched_bootstrap_delta(self, inner: pd.DataFrame, outer: pd.DataFrame, y_col: str, slope: float, n_boot: int = 1000, random_state: int = 42) -> dict:
        rng = np.random.default_rng(random_state)

        in_df = inner.copy()
        out_df = outer.copy()

        for c in ['logP', y_col]:
            if c in in_df.columns:
                in_df[c] = pd.to_numeric(in_df[c], errors='coerce')
            if c in out_df.columns:
                out_df[c] = pd.to_numeric(out_df[c], errors='coerce')

        in_df = in_df.dropna(subset=['logP', y_col]).copy()
        out_df = out_df.dropna(subset=['logP', y_col]).copy()

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

        p_min = max(float(in_df['logP'].min()), float(out_df['logP'].min()))
        p_max = min(float(in_df['logP'].max()), float(out_df['logP'].max()))
        if not (np.isfinite(p_min) and np.isfinite(p_max)) or (p_max <= p_min):
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

        p_edges = np.linspace(p_min, p_max, 11)

        deltas = []
        matched_sizes = []
        for _ in range(n_boot):
            in_idx = []
            out_idx = []

            for j in range(len(p_edges) - 1):
                g_in = in_df[(in_df['logP'] >= p_edges[j]) & (in_df['logP'] < p_edges[j + 1])]
                g_out = out_df[(out_df['logP'] >= p_edges[j]) & (out_df['logP'] < p_edges[j + 1])]
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

    def _multivariate_pair_match(self, inner: pd.DataFrame, outer: pd.DataFrame, feature_cols: list[str], caliper: float | None = 3.0) -> tuple[np.ndarray, np.ndarray, dict]:
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

        diff = Z_in[:, None, :] - Z_out[None, :, :]
        cost = np.sqrt(np.sum(diff * diff, axis=2))

        cal = caliper
        if cal is not None and np.isfinite(cal):
            cost = np.where(cost <= cal, cost, 1e6)

        row_ind, col_ind = linear_sum_assignment(cost)
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

        in_sub = inner.loc[in_idx].copy()
        out_sub = outer.loc[out_idx].copy()

        a_in, _ = self._weighted_intercept(in_sub['logP'].values, in_sub[y_col].values, slope=slope, yerr=None)
        a_out, _ = self._weighted_intercept(out_sub['logP'].values, out_sub[y_col].values, slope=slope, yerr=None)
        base_delta = float(a_in - a_out)

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

    def _type_consistent_match(self, inner: pd.DataFrame, outer: pd.DataFrame, feature_cols: list[str], type_col: str = 'Type', caliper: float | None = 3.0) -> tuple[np.ndarray, np.ndarray, dict]:
        if type_col not in inner.columns or type_col not in outer.columns:
            in_idx, out_idx, meta = self._multivariate_pair_match(inner, outer, feature_cols, caliper=caliper)
            meta['type_mode'] = 'none'
            return in_idx, out_idx, meta

        in_idx_all = []
        out_idx_all = []
        meta_by_type = {}

        t_in = inner[type_col].astype(str).str.strip()
        t_out = outer[type_col].astype(str).str.strip()
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

    def _fetch_ogle_lmc(self) -> pd.DataFrame:
        Vizier.ROW_LIMIT = -1
        cat_code = 'J/AcA/65/297'

        # The OGLE Magellanic Cepheid collection is split across sub-tables.
        # For the LMC sample, the coordinates live in `lmcident` while the periods
        # and mean magnitudes live in `lmccep`. We merge them on (Star, Mode).
        ident = Vizier.get_catalogs(f'{cat_code}/lmcident')
        cep = Vizier.get_catalogs(f'{cat_code}/lmccep')
        if not ident or not cep:
            raise RuntimeError(f"Could not retrieve required LMC tables from VizieR: {cat_code}/lmcident and {cat_code}/lmccep")

        df_id = ident[0].to_pandas()
        df_c = cep[0].to_pandas()

        # Standardize coordinate columns
        if 'RAJ2000' in df_id.columns:
            df_id = df_id.rename(columns={'RAJ2000': 'RA'})
        if 'DEJ2000' in df_id.columns:
            df_id = df_id.rename(columns={'DEJ2000': 'DEC'})

        # Standardize period column
        if 'Per' in df_c.columns:
            df_c = df_c.rename(columns={'Per': 'P'})

        # Standardize mean magnitude columns when present
        if '<Vmag>' in df_c.columns:
            df_c = df_c.rename(columns={'<Vmag>': 'Vmag'})
        if '<Imag>' in df_c.columns:
            df_c = df_c.rename(columns={'<Imag>': 'Imag'})

        # Standardize mode/type
        if 'Mode' in df_id.columns:
            df_id = df_id.rename(columns={'Mode': 'Type'})
        if 'Mode' in df_c.columns:
            df_c = df_c.rename(columns={'Mode': 'Type'})

        if 'Star' not in df_id.columns or 'Star' not in df_c.columns:
            raise RuntimeError("OGLE LMC tables missing 'Star' column; cannot merge")
        if 'Type' not in df_id.columns or 'Type' not in df_c.columns:
            raise RuntimeError("OGLE LMC tables missing 'Mode/Type' column; cannot merge")

        df = pd.merge(
            df_id[['Star', 'Type', 'RA', 'DEC']].copy(),
            df_c.copy(),
            on=['Star', 'Type'],
            how='inner',
        )

        df['P'] = pd.to_numeric(df.get('P', np.nan), errors='coerce')
        ra_deg, dec_deg = self._parse_ra_dec_deg(df.get('RA', pd.Series(np.nan, index=df.index)), df.get('DEC', pd.Series(np.nan, index=df.index)))
        df['RA'] = ra_deg
        df['DEC'] = dec_deg

        df = df.dropna(subset=['RA', 'DEC', 'P']).copy()
        df = df[df['P'] > 0].copy()
        df['logP'] = np.log10(df['P'])

        return df

    def _infer_mag_columns(self, df: pd.DataFrame) -> tuple[str | None, str | None]:
        cols = set(df.columns)

        if 'I' in cols and 'V' in cols:
            return 'V', 'I'

        v_candidates = ['Vmag', 'Vmean', 'VMean', 'V0']
        i_candidates = ['Imag', 'Imean', 'IMean', 'I0']

        v_col = None
        i_col = None
        for c in v_candidates:
            if c in cols:
                v_col = c
                break
        for c in i_candidates:
            if c in cols:
                i_col = c
                break

        return v_col, i_col

    def run_analysis(self):
        print_status("Initiating LMC replication (OGLE classical Cepheids)", "SECTION")
        print_status("Fetching OGLE-IV Cepheid catalog from VizieR...", "PROCESS")

        try:
            df = self._fetch_ogle_lmc()
        except Exception as e:
            print_status(f"Failed to download OGLE catalog: {e}", "ERROR")
            return

        print_status(f"Retrieved {len(df)} OGLE objects (raw table).", "SUCCESS")

        # --- LMC-only spatial restriction (important: the VizieR collection also contains SMC/Bridge tables) ---
        # This cut is done before any period/type slicing so the subsequent quantiles are well-defined.
        if not {'RA', 'DEC'}.issubset(df.columns):
            print_status("OGLE table missing RA/DEC after merge; aborting.", "ERROR")
            return

        RA0 = 80.8939
        DEC0 = -69.7561
        theta0 = self._angsep_rad(df['RA'].values, df['DEC'].values, RA0, DEC0)
        theta0_deg = np.degrees(theta0)
        df = df[(np.isfinite(theta0_deg)) & (theta0_deg <= float(self.lmc_sky_radius_deg))].copy()
        print_status(f"After LMC sky cut (<= {self.lmc_sky_radius_deg:.1f} deg): N={len(df)}", "INFO")

        v_col, i_col = self._infer_mag_columns(df)
        if v_col is not None and i_col is not None:
            df['Vmag'] = pd.to_numeric(df[v_col], errors='coerce')
            df['Imag'] = pd.to_numeric(df[i_col], errors='coerce')
            df['VI_color'] = df['Vmag'] - df['Imag']
            df['W_I'] = df['Imag'] - 1.55 * (df['Vmag'] - df['Imag'])
        elif 'W_I' in df.columns:
            df['W_I'] = pd.to_numeric(df['W_I'], errors='coerce')
        elif 'W' in df.columns:
            df['W_I'] = pd.to_numeric(df['W'], errors='coerce')
        else:
            print_status("Could not locate (V,I) or an OGLE-provided Wesenheit column; aborting LMC replication.", "ERROR")
            return

        df = df.dropna(subset=['W_I', 'logP']).copy()

        if 'Type' in df.columns:
            t = df['Type'].astype(str).str.upper().str.strip()
            # Prefer a strict fundamental-mode selection when mode labels are OGLE-style.
            # Common labels include: 'F', '1O', '2O', 'F/1O', ...
            keep = t.isin(['F', 'FU', 'FUND'])
            if int(keep.sum()) >= 100:
                df = df[keep].copy()
                df['Type'] = 'FU'
                print_status(f"After fundamental-mode filter: N={len(df)}", "INFO")
            else:
                print_status("Mode labels not recognized for a strict FU filter; proceeding without type filtering.", "WARNING")

        p_min_log = float(np.log10(self.period_min_days))
        p_max_log = float(np.log10(self.period_max_days))
        df = df[(df['logP'] > p_min_log) & (df['logP'] < p_max_log)].copy()
        print_status(f"After period cut ({self.period_min_days:.1f} < P < {self.period_max_days:.1f} d): N={len(df)}", "INFO")

        if len(df) < 200:
            print_status(f"After basic filters, sample is small (N={len(df)}). Consider widening the period range.", "WARNING")

        D_kpc = 49.6
        theta = self._angsep_rad(df['RA'].values, df['DEC'].values, RA0, DEC0)
        df['R_kpc'] = D_kpc * theta

        inner_cut = float(np.nanpercentile(df['R_kpc'], 100.0 * float(self.inner_quantile)))
        outer_cut = float(np.nanpercentile(df['R_kpc'], 100.0 * float(self.outer_quantile)))

        inner_df = df[df['R_kpc'] <= inner_cut].copy()
        outer_df = df[df['R_kpc'] >= outer_cut].copy()

        headers = ["Region", "Radius selection", "N (Cepheids)"]
        rows = [
            ["Inner", f"R_kpc <= {inner_cut:.2f} (20th pct)", str(len(inner_df))],
            ["Outer", f"R_kpc >= {outer_cut:.2f} (80th pct)", str(len(outer_df))],
        ]
        print_table(headers, rows, title="LMC Subsamples")

        if len(inner_df) < int(self.min_region_n) or len(outer_df) < int(self.min_region_n):
            print_status("Insufficient samples after inner/outer selection; aborting.", "WARNING")
            return

        fixed_slope = -3.3

        a_in, err_in = self._weighted_intercept(inner_df['logP'].values, inner_df['W_I'].values, slope=fixed_slope, yerr=None)
        a_out, err_out = self._weighted_intercept(outer_df['logP'].values, outer_df['W_I'].values, slope=fixed_slope, yerr=None)

        delta_mag = a_in - a_out
        delta_err = np.sqrt(err_in**2 + err_out**2)

        sigma_significance = abs(delta_mag) / delta_err if np.isfinite(delta_err) and delta_err > 0 else np.nan
        print_status(f"Detected Offset (Inner - Outer): {delta_mag:+.4f} mag ({sigma_significance:.1f} sigma)", "RESULT")

        results = pd.DataFrame({
            'Region': ['Inner', 'Outer'],
            'Intercept': [a_in, a_out],
            'Error': [err_in, err_out],
            'N': [len(inner_df), len(outer_df)],
            'inner_cut_kpc': [inner_cut, inner_cut],
            'outer_cut_kpc': [outer_cut, outer_cut],
        })
        results.to_csv(self.output_csv_path, index=False)
        print_status(f"Saved results to {self.output_csv_path}", "SUCCESS")

        print_status("Running LMC robustness suite...", "SECTION")

        robustness_rows = []

        boot_logp = self._matched_bootstrap_delta(
            inner_df,
            outer_df,
            y_col='W_I',
            slope=fixed_slope,
            n_boot=1000,
            random_state=42,
        )
        robustness_rows.append({
            'test': 'matched_bootstrap_logP',
            'y': 'W_I',
            'slope_fixed': fixed_slope,
            **boot_logp,
        })

        if 'VI_color' in df.columns:
            boot_logp_c = self._matched_bootstrap_delta_2d(
                inner_df,
                outer_df,
                y_col='W_I',
                slope=fixed_slope,
                col1='logP',
                col2='VI_color',
                n_bin1=10,
                n_bin2=6,
                n_boot=1000,
                random_state=56,
            )
            robustness_rows.append({
                'test': 'matched_bootstrap_logP_color',
                'y': 'W_I',
                'slope_fixed': fixed_slope,
                **boot_logp_c,
            })
        else:
            boot_logp_c = None

        for s in [-3.1, -3.3, -3.5]:
            ai, ei = self._weighted_intercept(inner_df['logP'].values, inner_df['W_I'].values, slope=s, yerr=None)
            ao, eo = self._weighted_intercept(outer_df['logP'].values, outer_df['W_I'].values, slope=s, yerr=None)
            dm = ai - ao
            de = float(np.sqrt(ei ** 2 + eo ** 2)) if np.isfinite(ei) and np.isfinite(eo) else np.nan
            robustness_rows.append({
                'test': 'slope_sensitivity',
                'y': 'W_I',
                'slope_fixed': float(s),
                'n_inner': int(len(inner_df)),
                'n_outer': int(len(outer_df)),
                'n_boot': 0,
                'delta_mean': float(dm),
                'delta_std': float(de),
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            })

        if v_col is not None and i_col is not None:
            R0 = 1.55
            R_grid = np.linspace(max(R0 - 0.6, 0.0), R0 + 0.6, 13)
            for R in R_grid:
                w = inner_df['Imag'] - float(R) * (inner_df['Vmag'] - inner_df['Imag'])
                w_out = outer_df['Imag'] - float(R) * (outer_df['Vmag'] - outer_df['Imag'])

                ai, ei = self._weighted_intercept(inner_df['logP'].values, w.values, slope=fixed_slope, yerr=None)
                ao, eo = self._weighted_intercept(outer_df['logP'].values, w_out.values, slope=fixed_slope, yerr=None)
                dm = ai - ao
                de = float(np.sqrt(ei ** 2 + eo ** 2)) if np.isfinite(ei) and np.isfinite(eo) else np.nan
                robustness_rows.append({
                    'test': 'wesenheit_R_scan',
                    'y': f'W_I(R={float(R):.2f})',
                    'slope_fixed': fixed_slope,
                    'n_inner': int(len(inner_df)),
                    'n_outer': int(len(outer_df)),
                    'n_boot': 0,
                    'delta_mean': float(dm),
                    'delta_std': float(de),
                    'delta_p16': np.nan,
                    'delta_p84': np.nan,
                })

        # Explicit long-period slice (P > 10 days) to mirror M31 cuts
        p_10d_log = np.log10(10.0)
        inner_long = inner_df[inner_df['logP'] > p_10d_log]
        outer_long = outer_df[outer_df['logP'] > p_10d_log]
        
        if len(inner_long) > 10 and len(outer_long) > 10:
            al, el = self._weighted_intercept(inner_long['logP'].values, inner_long['W_I'].values, slope=fixed_slope, yerr=None)
            ao, eo = self._weighted_intercept(outer_long['logP'].values, outer_long['W_I'].values, slope=fixed_slope, yerr=None)
            dm = al - ao
            de = float(np.sqrt(el ** 2 + eo ** 2)) if np.isfinite(el) and np.isfinite(eo) else np.nan
            robustness_rows.append({
                'test': 'long_period_slice_10d',
                'y': 'W_I',
                'slope_fixed': fixed_slope,
                'n_inner': int(len(inner_long)),
                'n_outer': int(len(outer_long)),
                'n_boot': 0,
                'delta_mean': float(dm),
                'delta_std': float(de),
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            })
        else:
             robustness_rows.append({
                'test': 'long_period_slice_10d',
                'y': 'W_I',
                'slope_fixed': fixed_slope,
                'n_inner': int(len(inner_long)),
                'n_outer': int(len(outer_long)),
                'n_boot': 0,
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_p16': np.nan,
                'delta_p84': np.nan,
            })

        mv_candidates = ['logP', 'VI_color', 'Vmag', 'Imag']
        mv_features = self._select_features_by_coverage(inner_df, outer_df, mv_candidates, min_frac=0.7)
        if 'logP' not in mv_features:
            mv_features = ['logP'] + [c for c in mv_features if c != 'logP']

        in_idx, out_idx, mv_meta = self._type_consistent_match(inner_df, outer_df, mv_features, type_col='Type', caliper=3.0)
        mv_res = self._matched_delta_from_pairs(inner_df, outer_df, in_idx, out_idx, y_col='W_I', slope=fixed_slope, n_boot=1000, random_state=77)
        ks = self._ks_balance(inner_df.loc[in_idx], outer_df.loc[out_idx], mv_features) if len(in_idx) > 0 else {}

        robustness_rows.append({
            'test': 'multivariate_pair_match_type',
            'y': 'W_I',
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

        robust_df = pd.DataFrame(robustness_rows)
        robust_df.to_csv(self.output_robust_csv_path, index=False)
        print_status(f"Saved robustness summary to {self.output_robust_csv_path}", "SUCCESS")

        robust_payload = {
            'catalog': 'OGLE-IV classical Cepheids (Soszynski+ 2015)',
            'vizier_source': 'J/AcA/65/297',
            'selection': {
                'type': 'FU (if available)',
                'logP_min': float(p_min_log),
                'logP_max': float(p_max_log),
                'inner_cut_kpc': float(inner_cut),
                'outer_cut_kpc': float(outer_cut),
                'inner_quantile': 0.2,
                'outer_quantile': 0.8,
            },
            'baseline': {
                'fixed_slope': float(fixed_slope),
                'delta_mag': float(delta_mag),
                'delta_err': float(delta_err),
                'n_inner': int(len(inner_df)),
                'n_outer': int(len(outer_df)),
            },
            'matched_bootstrap_logP': boot_logp,
            'matched_bootstrap_logP_color': boot_logp_c,
            'multivariate_pair_match_type': {
                'features': mv_features,
                'meta': mv_meta,
                'result': mv_res,
                'ks_balance_p': {k: v.get('ks_p', np.nan) for k, v in ks.items()},
            },
        }
        with open(self.output_robust_json_path, 'w') as f:
            json.dump(robust_payload, f, indent=2)
        print_status(f"Saved robustness summary JSON to {self.output_robust_json_path}", "SUCCESS")

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
        print_table(headers, rows, title="LMC Differential Robustness Summary")

        print_status("Generating LMC differential P–L plot...", "PROCESS")
        
        try:
            from scripts.utils.plot_style import apply_tep_style
            colors = apply_tep_style()
        except ImportError:
            colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'light_blue': '#4b6785'}
        
        plt.figure(figsize=(14, 9))
        
        plt.scatter(outer_df['logP'], outer_df['W_I'], 
                   alpha=0.3, s=30, color=colors['blue'], 
                   label=f'Outer (> {outer_cut:.1f} kpc)', edgecolor='none')
                   
        plt.scatter(inner_df['logP'], inner_df['W_I'], 
                   alpha=0.5, s=30, color=colors['accent'], 
                   label=f'Inner (< {inner_cut:.1f} kpc)', edgecolor='none')
        
        # Plot Fits
        x_range = np.linspace(p_min_log, p_max_log, 100)
        
        def fixed_slope_model(x, a):
            return a + fixed_slope * x
            
        plt.plot(x_range, fixed_slope_model(x_range, a_in), 
                color=colors['accent'], linestyle='-', linewidth=3, 
                label=f'Inner Fit (a={a_in:.3f})')
                
        plt.plot(x_range, fixed_slope_model(x_range, a_out), 
                color=colors['blue'], linestyle='--', linewidth=3, 
                label=f'Outer Fit (a={a_out:.3f})')
        
        plt.xlabel('log(Period) [days]')
        plt.ylabel('Wesenheit Magnitude (W_I)')
        plt.title(f'LMC Differential Cepheid Analysis\nOffset (Inner - Outer) = {delta_mag:+.4f} +/- {delta_err:.4f} mag')
        plt.gca().invert_yaxis()
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_plot_path, dpi=300)
        print_status(f"Saved plot to {self.output_plot_path}", "SUCCESS")
        plt.close()

        # Robustness plot
        print_status("Generating LMC robustness plot...", "PROCESS")
        try:
            fig = plt.figure(figsize=(16, 9))

            ax1 = plt.subplot(1, 2, 1)
            ax1.axvline(delta_mag, color=colors['dark'], linestyle='--', linewidth=2, label='Baseline')
            if np.isfinite(boot_logp.get('delta_mean', np.nan)):
                ax1.axvline(boot_logp['delta_mean'], color=colors['accent'], linestyle='-', linewidth=2, label='Matched mean')
                ax1.set_title('Matched (logP) bootstrap\nΔW distribution summary')
                ax1.text(0.05, 0.95, f"μ={boot_logp['delta_mean']:+.3f}\nσ={boot_logp['delta_std']:.3f}", transform=ax1.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax1.set_xlabel('ΔW = W_inner − W_outer (mag)')
            ax1.set_yticks([])
            ax1.legend(fontsize=10)

            ax2 = plt.subplot(1, 2, 2)
            slope_rows = pd.DataFrame([r for r in robustness_rows if r['test'] == 'slope_sensitivity'])
            if len(slope_rows) > 0:
                ax2.errorbar(slope_rows['slope_fixed'].values, slope_rows['delta_mean'].values, yerr=slope_rows['delta_std'].values, 
                           fmt='o-', color=colors['blue'], ecolor=colors['dark'], capsize=4, linewidth=2, markersize=8)
                ax2.axhline(delta_mag, color=colors['dark'], linestyle='--', linewidth=2)
            ax2.set_xlabel('Fixed P–L Slope')
            ax2.set_ylabel(r'$\Delta W$ (mag)')
            ax2.set_title('Slope Sensitivity')
            ax2.grid(True, linestyle=':', alpha=0.6)

            plt.tight_layout()
            plt.savefig(self.output_robust_plot_path, dpi=300)
            plt.close(fig)
            print_status(f"Saved robustness plot to {self.output_robust_plot_path}", "SUCCESS")

            shutil.copy(self.output_robust_plot_path, self.public_figures_dir / "lmc_differential_robustness.png")
            print_status("Copied robustness plot to public figures directory.", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to generate robustness plot: {e}", "WARNING")

        shutil.copy(self.output_plot_path, self.public_figures_dir / "lmc_differential_pl.png")
        print_status("Copied plot to public figures directory.", "SUCCESS")

    def run(self):
        print_status("Starting Step 7: LMC Replication", "TITLE")
        self.run_analysis()
        print_status("Step 7 Complete.", "SUCCESS")


def main():
    step = Step7LMCReplication()
    step.run()


if __name__ == "__main__":
    main()
