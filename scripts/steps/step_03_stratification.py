import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Astronomy imports (environment catalog)
try:
    from astroquery.vizier import Vizier
except Exception:
    Vizier = None

# Import TEP Logger
try:
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )
except ImportError:
    # Add project root to path if needed
    sys.path.append(str(PROJECT_ROOT))
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )


class Step2Stratification:
    r"""
    Step 2: Host Stratification and H0 Analysis
    ===========================================

    This step performs the core phenomenological test of the TEP hypothesis:
    Does the inferred Hubble Constant ($H_0$) depend on the gravitational potential
    depth ($\sigma$) of the host galaxy?

    Methodology:
    1.  **Calculate Individual H0**: For each SN Ia host, we calculate $H_0 = v / d$,
        where $v = cz$ (Pantheon+ CMB frame redshift) and $d$ is the distance derived
        from the SH0ES Cepheid distance modulus ($\mu$).
    2.  **Stratification**: We split the sample into two bins based on the median
        velocity dispersion ($\sigma_{\rm med}$), which proxies for gravitational
        potential depth (Temporal Topology).
        - **Low Potential ($\sigma < \sigma_{\rm med}$)**: Shallow potentials, similar to calibrators.
        - **High Potential ($\sigma > \sigma_{\rm med}$)**: Deep potentials, predicted to show TEP bias.
    3.  **Bias Quantification**: We calculate the mean $H_0$ in each bin and the
        Pearson correlation coefficient $r$ between $\sigma$ and $H_0$.

    A significant difference between the bins ($\Delta H_0 > 0$) or a strong positive correlation
    confirms the presence of an environmental bias consistent with TEP Period Contraction.

    Note on Screening (TEP v0.8):
    Rather than a binary screened/unscreened dichotomy, the TEP framework now employs
    a continuous geometric profile (Temporal Topology) in which the scalar field gradient
    (Temporal Shear) is progressively suppressed by ambient density. A dimensionless
    shear-suppression factor $S(\rho) \in [0,1]$ is estimated for each host to quantify
    the residual coupling strength.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.root_dir / "logs"
        self.figures_dir = self.root_dir / "results" / "figures"
        self.outputs_dir = self.root_dir / "results" / "outputs"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Logger
        self.logger = TEPLogger(
            "step_2_stratification",
            log_file_path=self.logs_dir / "step_03_stratification.log",
        )
        set_step_logger(self.logger)

        # Inputs
        self.hosts_path = self.data_dir / "processed" / "hosts_processed.csv"
        self.distances_path = self.data_dir / "interim" / "r22_distances.csv"

        self.mu_cov_path = self.data_dir / "interim" / "r22_mu_covariance.npy"
        self.mu_cov_labels_path = (
            self.data_dir / "interim" / "r22_mu_covariance_labels.json"
        )

        self.env_catalog_cache_path = (
            self.data_dir / "raw" / "external" / "tully2015_2mrs_groups_table5.csv"
        )

        # Stiskalek et al. 2026 (Manticore) Bayesian peculiar-velocity catalog.
        # Provides posterior-mean vpec from a coherent-flow model that handles
        # group environments better than Pantheon+'s simple flow correction.
        self.manticore_path = (
            self.data_dir / "raw" / "external" / "stiskalek2026_manticore_hosts.csv"
        )

        # Outputs
        self.stratified_output_path = self.outputs_dir / "step_03_stratified_h0.csv"
        self.json_output_path = self.outputs_dir / "step_03_stratification_results.json"
        self.plot_path = self.figures_dir / "step_03_figure_01_h0_vs_sigma.png"

        self.h0_cov_path = self.outputs_dir / "step_03_h0_covariance.npy"
        self.h0_cov_labels_path = self.outputs_dir / "step_03_h0_covariance_labels.json"

    def load_and_merge(self):
        """Loads data and merges host properties with distances."""
        print_status("Loading and Merging Datasets...", "SECTION")

        if not self.hosts_path.exists() or not self.distances_path.exists():
            print_status("Input files missing. Please run Step 1 first.", "ERROR")
            sys.exit(1)

        hosts_df = pd.read_csv(self.hosts_path)
        dists_df = pd.read_csv(self.distances_path)

        # Merge on source_id
        merged = pd.merge(dists_df, hosts_df, on="source_id", how="inner")
        print_status(f"Merged {len(merged)} galaxies (Distances + Properties).", "INFO")
        return merged

    def calculate_h0(self, df):
        """Calculates H0 for each host using the 3-rung distance ladder."""
        print_status("Calculating Individual H0 Values (3-Rung Ladder)...", "SECTION")

        # We keep distance_mpc and velocity just for display/reference
        df["distance_mpc"] = 10 ** ((df["value"] - 25) / 5)
        df["velocity"] = 299792.458 * df["z_hd"]

        # H0 = v / d
        df["h0_derived"] = df["velocity"] / df["distance_mpc"]

        # Filter valid entries (require Sigma, H0, and SN photometry)
        valid = df.dropna(subset=["h0_derived", "sigma_inferred", "m_b_corr"]).copy()

        # Ensure normalized_name is stripped and string
        valid["normalized_name"] = valid["normalized_name"].astype(str).str.strip()

        # Exclude Anchors/Calibrators from H0 sample (they are not SN hosts in this context)
        # N4258, LMC, SMC, M31, MW
        anchors = ["NGC 4258", "LMC", "SMC", "M 31", "MW"]

        valid = valid[~valid["normalized_name"].isin(anchors)].copy()
        
        # Retain the complete R22 SN-host set.  The endpoint likelihood carries
        # an explicit peculiar-velocity variance, so deleting nearby galaxies
        # by a post-hoc redshift threshold would discard information and make
        # the sample definition depend on the chosen flow correction.  Plain
        # z_HD cuts remain prespecified sensitivity analyses in Step 39.
        valid = valid[valid["z_hd"].notna() & (valid["z_hd"] > 0)].copy()
        print_status(
            f"Final Sample Size: {len(valid)} R22 SN Ia host galaxies "
            "(no redshift deletion; peculiar velocities enter the likelihood).",
            "SUCCESS",
        )

        # Display Sample
        headers = ["Host", "z_HD", "mu (mag)", "D (Mpc)", "H0 (km/s/Mpc)"]
        rows = []
        for _, row in valid.head(5).iterrows():
            rows.append(
                [
                    row["normalized_name"],
                    f"{row['z_hd']:.4f}",
                    f"{row['value']:.3f}",
                    f"{row['distance_mpc']:.1f}",
                    f"{row['h0_derived']:.2f}",
                ]
            )
        print_table(headers, rows, title="Sample H0 Calculations")

        # Compute vpec-corrected H0 using Stiskalek et al. 2026 (Manticore)
        # Bayesian peculiar velocities where available, falling back to the
        # Pantheon+ flow model otherwise.  The Manticore model handles group
        # environments coherently and resolves large discrepancies for nearby
        # group-host galaxies (e.g. NGC 4639).
        self._add_manticore_vpec_h0(valid)

        return valid

    def _add_manticore_vpec_h0(self, valid):
        """Add a vpec-corrected H0 column using Stiskalek et al. 2026."""
        c_kms = 299792.458

        valid["h0_vpec_corrected"] = valid["h0_derived"].copy()
        valid["vpec_source"] = "pantheon_plus"

        if not self.manticore_path.exists():
            print_status(
                "Manticore vpec catalog not found; using Pantheon+ vpec for all hosts.",
                "INFO",
            )
            return

        manticore = pd.read_csv(self.manticore_path, comment="#")

        # Map Stiskalek host labels to our normalized_name conventions.
        name_map = {}
        for _, r in manticore.iterrows():
            h = str(r["host"]).strip()
            if h == "M101":
                name_map[h] = "M 101"
            elif h.startswith("M1337"):
                name_map[h] = "Mrk 1337"
            elif h.startswith("M") and len(h) > 1 and h[1:].isdigit():
                name_map[h] = "M " + h[1:]
            elif h.startswith("N"):
                name_map[h] = f"NGC {h[1:]}"
            elif h.startswith("U"):
                name_map[h] = f"UGC {h[1:]}"
        manticore["our_name"] = manticore["host"].map(name_map)
        mant_lookup = dict(
            zip(manticore["our_name"], manticore["vpec_manticore_kms"])
        )

        n_corrected = 0
        for idx, row in valid.iterrows():
            name = row["normalized_name"]
            if name in mant_lookup and pd.notna(mant_lookup[name]):
                vpec_mant = float(mant_lookup[name])
                z_cmb = float(row.get("z_cmb", np.nan))
                if pd.notna(z_cmb) and z_cmb > 0:
                    h0_corr = (c_kms * z_cmb - vpec_mant) / row["distance_mpc"]
                    valid.at[idx, "h0_vpec_corrected"] = h0_corr
                    valid.at[idx, "vpec_source"] = "manticore"
                    n_corrected += 1

        print_status(
            f"Applied Manticore vpec correction to {n_corrected}/{len(valid)} hosts; "
            f"{len(valid) - n_corrected} retain Pantheon+ vpec.",
            "INFO",
        )

    def _load_tully_2015_table5(self):
        if Vizier is None:
            return None

        self.env_catalog_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.env_catalog_cache_path.exists():
            try:
                return pd.read_csv(self.env_catalog_cache_path)
            except Exception:
                return None

        v = Vizier(
            columns=[
                "PGC",
                "Nest",
                "Nmb",
                "logpK",
                "Mvir",
                "Mlum",
            ],
            row_limit=-1,
        )

        tables = v.get_catalogs("J/AJ/149/171/table5")
        if not tables:
            return None

        t = tables[0].to_pandas()
        t.to_csv(self.env_catalog_cache_path, index=False)
        return t

    def annotate_large_scale_environment(self, df):
        print_status(
            "Annotating Large-Scale Environment (Tully 2015 groups)...", "SECTION"
        )

        if "pgc" not in df.columns:
            df["tully_nest"] = np.nan
            df["tully_nmb"] = np.nan
            df["tully_logpK"] = np.nan
            df["tully_mvir_tmsun"] = np.nan
            df["tully_mlum_tmsun"] = np.nan
            df["tully_is_group"] = False
            df["tully_is_cluster"] = False
            print_status(
                "No PGC identifiers available in merged data; skipping group crossmatch.",
                "WARNING",
            )
            return df

        t = self._load_tully_2015_table5()
        if t is None or len(t) == 0:
            df["tully_nest"] = np.nan
            df["tully_nmb"] = np.nan
            df["tully_logpK"] = np.nan
            df["tully_mvir_tmsun"] = np.nan
            df["tully_mlum_tmsun"] = np.nan
            df["tully_is_group"] = False
            df["tully_is_cluster"] = False
            print_status(
                "Could not load Tully 2015 group catalog; skipping group crossmatch.",
                "WARNING",
            )
            return df

        cols = ["PGC", "Nest", "Nmb", "logpK", "Mvir", "Mlum"]
        missing = [c for c in cols if c not in t.columns]
        if missing:
            df["tully_nest"] = np.nan
            df["tully_nmb"] = np.nan
            df["tully_logpK"] = np.nan
            df["tully_mvir_tmsun"] = np.nan
            df["tully_mlum_tmsun"] = np.nan
            df["tully_is_group"] = False
            df["tully_is_cluster"] = False
            print_status(
                f"Tully 2015 table missing columns: {missing}. Skipping.", "WARNING"
            )
            return df

        t = t[cols].copy()
        t = t.dropna(subset=["PGC"]).copy()
        try:
            t["PGC"] = t["PGC"].astype(int)
        except Exception:
            t["PGC"] = pd.to_numeric(t["PGC"], errors="coerce").astype("Int64")

        df = df.copy()
        df["pgc_int"] = pd.to_numeric(df["pgc"], errors="coerce").astype("Int64")

        merged = df.merge(t, left_on="pgc_int", right_on="PGC", how="left")

        merged["tully_nest"] = merged["Nest"]
        merged["tully_nmb"] = merged["Nmb"]
        merged["tully_logpK"] = merged["logpK"]
        merged["tully_mvir_tmsun"] = merged["Mvir"]
        merged["tully_mlum_tmsun"] = merged["Mlum"]

        merged["tully_is_group"] = merged["tully_nmb"].fillna(1) >= 2
        merged["tully_is_cluster"] = merged["tully_nmb"].fillna(1) >= 5

        merged = merged.drop(
            columns=["PGC", "Nest", "Nmb", "logpK", "Mvir", "Mlum"], errors="ignore"
        )

        n_tagged = int(merged["tully_nest"].notna().sum())
        print_status(
            f"Matched {n_tagged}/{len(merged)} hosts to Tully 2015 group catalog.",
            "INFO",
        )
        return merged

    def _load_mu_covariance(self):
        if not self.mu_cov_path.exists() or not self.mu_cov_labels_path.exists():
            return None, None

        mu_cov = np.load(self.mu_cov_path)
        with open(self.mu_cov_labels_path, "r") as f:
            mu_labels = json.load(f)

        return mu_cov, mu_labels

    def _subset_covariance(self, cov, cov_labels, target_labels):
        label_to_idx = {str(lbl): i for i, lbl in enumerate(cov_labels)}
        idx = []
        missing = []
        for lbl in target_labels:
            key = str(lbl)
            if key not in label_to_idx:
                missing.append(key)
                continue
            idx.append(label_to_idx[key])

        if missing:
            raise KeyError(
                f"Missing {len(missing)} labels in covariance: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        sub = cov[np.ix_(idx, idx)]
        sub = 0.5 * (sub + sub.T)
        return sub

    def _mu_to_h0_covariance(self, mu_values, h0_values, mu_cov, vpec_err=None, distances=None):
        # Original exact mapping: cov_H0 = (dH0/dmu) * cov_mu * (dH0/dmu)^T
        dH_dmu = -(np.log(10.0) / 5.0) * h0_values
        h0_cov = dH_dmu[:, None] * mu_cov * dH_dmu[None, :]
        h0_cov = 0.5 * (h0_cov + h0_cov.T)
        
        # Add vpec uncertainty to the diagonal
        if vpec_err is not None and distances is not None:
            vpec_variance = (vpec_err / distances) ** 2
            np.fill_diagonal(h0_cov, h0_cov.diagonal() + vpec_variance)
            
        return h0_cov

    def calculate_densities(self, df):
        r"""
        Estimates the local stellar mass density for SH0ES hosts and computes a
        continuous shear-suppression factor consistent with TEP v0.8.

        Physics:
        SH0ES Cepheids typically reside in the disks of spiral galaxies at
        radii r ~ 0.3 - 0.8 R25.
        We model the disk as an exponential profile:
        rho(r) ~ (M / 4 pi Rd^2 zd) * exp(-r/Rd)

        Assumptions:
        - Rd ~ R25 / 3.2
        - zd ~ 0.1 Rd (Thin disk scale height)
        - Typical Cepheid Radius r_cep ~ 0.55 R25 ~ 1.8 Rd

        Shear Suppression:
        In the continuous-screening framework (Temporal Topology), the scalar
        field gradient (Temporal Shear) is progressively suppressed by ambient
        density. We model this with a smooth suppression factor:
            S(rho) = 1 / (1 + (rho / rho_half)^n)
        where rho_half ~ 0.5 M_sun/pc^3 is the density at which suppression
        reaches 50% and n ~ 2 controls the steepness. S = 1 means fully active
        shear; S = 0 means fully suppressed.
        """
        print_status(
            "Estimating Host Environmental Densities & Shear Suppression...", "SECTION"
        )

        # Filter hosts with Mass and Size info
        valid = df.dropna(subset=["host_logmass", "r25_arcsec", "distance_mpc"]).copy()

        if len(valid) == 0:
            print_status(
                "Insufficient data for density estimation (missing RC3 radii or Mass).",
                "WARNING",
            )
            # Ensure columns exist for downstream steps even when density unknown
            df["rho_local"] = np.nan
            df["shear_suppression"] = 1.0  # default: fully active
            return np.nan, np.nan, np.nan, df

        # Calculate Physical Radius R25 in kpc
        # theta = R / D -> R = D * theta
        # r25_arcsec to radians: r25 / 206265
        valid["r25_kpc"] = (
            valid["distance_mpc"] * 1000 * (valid["r25_arcsec"] / 206265.0)
        )

        # Calculate Scale Length Rd (kpc)
        valid["rd_kpc"] = valid["r25_kpc"] / 3.2

        # Scale Height zd (kpc)
        valid["zd_kpc"] = 0.1 * valid["rd_kpc"]

        # Mass in Solar Masses
        valid["mass_sol"] = 10 ** valid["host_logmass"]

        # Evaluate Density at typical Cepheid radius (1.8 Rd)
        # rho = (M / (4 pi Rd^2 zd)) * exp(-1.8)
        # Note: M_disk is roughly M_total for these spirals

        def get_rho(row):
            if row["rd_kpc"] <= 0 or row["zd_kpc"] <= 0:
                return np.nan
            prefactor = row["mass_sol"] / (
                4 * np.pi * (row["rd_kpc"] ** 2) * row["zd_kpc"]
            )
            density = prefactor * np.exp(-1.8)
            # Convert kpc^-3 to pc^-3 (1 kpc^3 = 1e9 pc^3)
            return density / 1e9

        valid["rho_local"] = valid.apply(get_rho, axis=1)

        # Compute continuous shear-suppression factor S(rho) using Universal Two-Factor model
        from scripts.utils.tep_correction import total_screening_factor
        valid["shear_suppression"] = valid.apply(
            lambda r: total_screening_factor(r["rho_local"], r.get("tully_nmb", np.nan)), axis=1
        )

        # Merge back to main df for export
        df["rho_local"] = np.nan
        df["shear_suppression"] = np.nan
        df.update(valid["rho_local"])
        df.update(valid["shear_suppression"])

        # Stats
        rho_mean = valid["rho_local"].mean()
        rho_min = valid["rho_local"].min()
        rho_max = valid["rho_local"].max()
        s_mean = valid["shear_suppression"].mean()
        s_min = valid["shear_suppression"].min()
        s_max = valid["shear_suppression"].max()

        print_status(f"Calculated densities for {len(valid)} hosts.", "INFO")
        if not np.isnan(rho_mean):
            print_status(f"Mean Host Density: {rho_mean:.4f} M_sun/pc^3", "RESULT")
            print_status(f"Density Range: [{rho_min:.4f}, {rho_max:.4f}]", "INFO")
            print_status(f"Mean Shear Suppression (S): {s_mean:.3f}", "RESULT")
            print_status(f"Suppression Range: [{s_min:.3f}, {s_max:.3f}]", "INFO")

            # Identify hosts with appreciable suppression (S < 0.8)
            suppressed = valid[valid["shear_suppression"] < 0.8]
            if len(suppressed) > 0:
                print_status(
                    f"Hosts with appreciable shear suppression (S < 0.8, N={len(suppressed)}):",
                    "INFO",
                )
                for _, row in suppressed.iterrows():
                    print_status(
                        f"  - {row['normalized_name']}: rho={row['rho_local']:.3f}, S={row['shear_suppression']:.3f}",
                        "INFO",
                    )
            else:
                print_status(
                    "No hosts show strong shear suppression (all S > 0.8). Sample is effectively unscreened.",
                    "SUCCESS",
                )

        return rho_mean, rho_min, rho_max, df

    def stratify_and_analyze(self, df):
        """Stratifies by Sigma and analyzes H0 bias."""
        print_status("Stratification Analysis (Low vs High Potential)", "SECTION")

        df = df.reset_index(drop=True)

        median_sigma = df["sigma_inferred"].median()

        low_sigma = df[df["sigma_inferred"] <= median_sigma]
        high_sigma = df[df["sigma_inferred"] > median_sigma]

        mean_low = low_sigma["h0_derived"].mean()
        err_low = low_sigma["h0_derived"].std() / np.sqrt(len(low_sigma))

        mean_high = high_sigma["h0_derived"].mean()
        err_high = high_sigma["h0_derived"].std() / np.sqrt(len(high_sigma))

        diff = mean_high - mean_low

        # Calculate Correlation
        corr = df["sigma_inferred"].corr(df["h0_derived"])

        # Pearson p-value (two-tailed)
        n_sample = len(df)
        if n_sample > 2 and abs(corr) < 1.0:
            t_pearson = corr * np.sqrt((n_sample - 2) / (1.0 - corr**2))
            p_pearson = float(2.0 * scipy_stats.t.sf(np.abs(t_pearson), df=n_sample - 2))
        else:
            p_pearson = float("nan")

        # Spearman rank correlation (robust to outliers)
        spearman_r, spearman_p = scipy_stats.spearmanr(
            df["sigma_inferred"], df["h0_derived"]
        )

        # Covariance-aware uncertainties (if available)
        cov_low_err = None
        cov_high_err = None
        cov_all_mean_err = None
        cov_diff_err = None
        cov_available = False
        weighted_r = None
        weighted_p = None
        weighted_n_eff = None
        try:
            mu_cov, mu_labels = self._load_mu_covariance()
            if mu_cov is not None and mu_labels is not None:
                sample_labels = df["source_id"].astype(str).tolist()
                mu_cov_sub = self._subset_covariance(mu_cov, mu_labels, sample_labels)
                h0_vals = df["h0_derived"].values
                mu_vals = df["value"].values
                vpec_err = pd.to_numeric(df.get("vpecerr", pd.Series(np.full(len(df), 250.0))), errors="coerce").fillna(250.0).values
                distances = df["distance_mpc"].values
                h0_cov = self._mu_to_h0_covariance(mu_vals, h0_vals, mu_cov_sub, vpec_err=vpec_err, distances=distances)

                np.save(self.h0_cov_path, h0_cov)
                with open(self.h0_cov_labels_path, "w") as f:
                    json.dump(sample_labels, f, indent=2)

                cov_all_mean_err = float(np.sqrt(np.sum(h0_cov)) / len(df))

                pos_low = low_sigma.index.to_numpy(dtype=int)
                pos_high = high_sigma.index.to_numpy(dtype=int)

                cov_low = h0_cov[np.ix_(pos_low, pos_low)]
                cov_high = h0_cov[np.ix_(pos_high, pos_high)]
                ones_low = np.ones(len(pos_low))
                ones_high = np.ones(len(pos_high))
                cov_low_err = float(np.sqrt(np.sum(cov_low)) / len(pos_low))
                cov_high_err = float(np.sqrt(np.sum(cov_high)) / len(pos_high))

                w = np.zeros(len(df))
                w[pos_high] = 1.0 / len(pos_high)
                w[pos_low] = -1.0 / len(pos_low)
                cov_diff_err = float(np.sqrt(np.einsum("i,ij,j->", w, h0_cov, w)))
                cov_available = True

                # Heteroscedasticity-weighted Pearson correlation.
                # Weights = 1/diag(h0_cov), so that nearby galaxies with large
                # peculiar-velocity noise are down-weighted relative to
                # Hubble-flow hosts.  This is the appropriate descriptive
                # statistic for the full sample.
                h0_var_diag = np.diag(h0_cov)
                h0_var_diag = np.maximum(h0_var_diag, np.finfo(float).tiny)
                weights = 1.0 / h0_var_diag

                x = df["sigma_inferred"].values
                y = df["h0_derived"].values

                x_mean_w = np.average(x, weights=weights)
                y_mean_w = np.average(y, weights=weights)

                cov_xy_w = np.sum(weights * (x - x_mean_w) * (y - y_mean_w)) / np.sum(weights)
                var_x_w = np.sum(weights * (x - x_mean_w) ** 2) / np.sum(weights)
                var_y_w = np.sum(weights * (y - y_mean_w) ** 2) / np.sum(weights)

                if var_x_w > 0 and var_y_w > 0:
                    weighted_r = float(cov_xy_w / np.sqrt(var_x_w * var_y_w))
                    weighted_n_eff = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
                    if weighted_n_eff > 2 and abs(weighted_r) < 1.0:
                        t_w = weighted_r * np.sqrt(
                            (weighted_n_eff - 2) / (1.0 - weighted_r ** 2)
                        )
                        weighted_p = float(
                            2.0 * scipy_stats.t.sf(np.abs(t_w), df=weighted_n_eff - 2)
                        )
        except Exception as e:
            print_status(
                f"Could not compute covariance-aware uncertainties: {e}", "WARNING"
            )

        # Vpec-corrected correlation using Stiskalek et al. 2026 (Manticore)
        # Bayesian peculiar velocities.  The Manticore coherent-flow model
        # resolves large group-environment discrepancies that bias the raw
        # Pantheon+ vpec for nearby hosts (e.g. NGC 4639).
        vpec_corr_r = None
        vpec_corr_p = None
        vpec_corr_weighted_r = None
        vpec_corr_weighted_p = None
        vpec_corr_n_eff = None
        if "h0_vpec_corrected" in df.columns:
            h0_corr = df["h0_vpec_corrected"].values
            sigma_vals = df["sigma_inferred"].values
            valid_mask = np.isfinite(h0_corr) & np.isfinite(sigma_vals)
            if valid_mask.sum() > 3:
                vpec_corr_r, vpec_corr_p = scipy_stats.pearsonr(
                    sigma_vals[valid_mask], h0_corr[valid_mask]
                )
                vpec_corr_r = float(vpec_corr_r)
                vpec_corr_p = float(vpec_corr_p)

                # Weighted version using the same H0 covariance diagonal
                if cov_available and h0_cov is not None:
                    h0_var_diag_v = np.diag(h0_cov)
                    h0_var_diag_v = np.maximum(
                        h0_var_diag_v, np.finfo(float).tiny
                    )
                    w_v = 1.0 / h0_var_diag_v
                    xv = sigma_vals[valid_mask]
                    yv = h0_corr[valid_mask]
                    wv = w_v[valid_mask]
                    x_mean_v = np.average(xv, weights=wv)
                    y_mean_v = np.average(yv, weights=wv)
                    cov_xy_v = np.sum(wv * (xv - x_mean_v) * (yv - y_mean_v)) / np.sum(wv)
                    var_x_v = np.sum(wv * (xv - x_mean_v) ** 2) / np.sum(wv)
                    var_y_v = np.sum(wv * (yv - y_mean_v) ** 2) / np.sum(wv)
                    if var_x_v > 0 and var_y_v > 0:
                        vpec_corr_weighted_r = float(
                            cov_xy_v / np.sqrt(var_x_v * var_y_v)
                        )
                        vpec_corr_n_eff = float(
                            np.sum(wv) ** 2 / np.sum(wv ** 2)
                        )
                        if vpec_corr_n_eff > 2 and abs(vpec_corr_weighted_r) < 1:
                            t_v = vpec_corr_weighted_r * np.sqrt(
                                (vpec_corr_n_eff - 2)
                                / (1.0 - vpec_corr_weighted_r ** 2)
                            )
                            vpec_corr_weighted_p = float(
                                2.0 * scipy_stats.t.sf(
                                    np.abs(t_v), df=vpec_corr_n_eff - 2
                                )
                            )

        # Results Table
        headers = ["Bin", "Sigma Range", "N", "Mean H0", "Std Err"]
        rows = [
            [
                "Low Potential",
                f"<= {median_sigma:.1f}",
                str(len(low_sigma)),
                f"{mean_low:.2f}",
                f"{cov_low_err:.2f}"
                if cov_available and cov_low_err is not None
                else f"{err_low:.2f}",
            ],
            [
                "High Potential",
                f"> {median_sigma:.1f}",
                str(len(high_sigma)),
                f"{mean_high:.2f}",
                f"{cov_high_err:.2f}"
                if cov_available and cov_high_err is not None
                else f"{err_high:.2f}",
            ],
            ["Difference", "-", "-", f"+{diff:.2f}", "-"],
        ]
        print_table(headers, rows, title="Stratified H0 Results")

        print_status(f"Median Velocity Dispersion: {median_sigma:.2f} km/s", "INFO")
        print_status(
            f"Correlation (u_phi vs H0): r = {corr:.3f} (p = {p_pearson:.3f})", "TEST"
        )
        print_status(
            f"Spearman rank correlation: rho = {spearman_r:.3f} (p = {spearman_p:.3f})",
            "TEST",
        )
        if weighted_r is not None:
            print_status(
                f"Weighted Pearson r = {weighted_r:.3f} (p = {weighted_p:.3f}, "
                f"n_eff = {weighted_n_eff:.1f})",
                "TEST",
            )
        if vpec_corr_r is not None:
            print_status(
                f"Vpec-corrected Pearson r = {vpec_corr_r:.3f} (p = {vpec_corr_p:.3f})",
                "TEST",
            )
            if vpec_corr_weighted_r is not None:
                print_status(
                    f"Vpec-corrected weighted r = {vpec_corr_weighted_r:.3f} "
                    f"(p = {vpec_corr_weighted_p:.3f}, n_eff = {vpec_corr_n_eff:.1f})",
                    "TEST",
                )

        if diff > 3.0:
            print_status(
                f"Significant Environmental Bias Detected (+{diff:.2f} km/s/Mpc)",
                "SUCCESS",
            )

        metrics = {
            "median_sigma": float(median_sigma),
            "low_density": {
                "n": int(len(low_sigma)),
                "mean_h0": float(mean_low),
                "std_err": float(err_low),
                "cov_err": float(cov_low_err) if cov_low_err is not None else None,
            },
            "high_density": {
                "n": int(len(high_sigma)),
                "mean_h0": float(mean_high),
                "std_err": float(err_high),
                "cov_err": float(cov_high_err) if cov_high_err is not None else None,
            },
            "difference": float(diff),
            "correlation_r": float(corr),
            "correlation_p": float(p_pearson),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "weighted_correlation_r": float(weighted_r) if weighted_r is not None else None,
            "weighted_correlation_p": float(weighted_p) if weighted_p is not None else None,
            "weighted_n_eff": float(weighted_n_eff) if weighted_n_eff is not None else None,
            "h0_mean_cov_err": float(cov_all_mean_err)
            if cov_all_mean_err is not None
            else None,
            "difference_cov_err": float(cov_diff_err)
            if cov_diff_err is not None
            else None,
            "h0_covariance_saved": bool(cov_available),
            "vpec_corrected_correlation_r": float(vpec_corr_r) if vpec_corr_r is not None else None,
            "vpec_corrected_correlation_p": float(vpec_corr_p) if vpec_corr_p is not None else None,
            "vpec_corrected_weighted_r": float(vpec_corr_weighted_r) if vpec_corr_weighted_r is not None else None,
            "vpec_corrected_weighted_p": float(vpec_corr_weighted_p) if vpec_corr_weighted_p is not None else None,
            "vpec_corrected_n_eff": float(vpec_corr_n_eff) if vpec_corr_n_eff is not None else None,
        }

        return df, metrics

    def plot_results(self, df, corr):
        """Generates analysis plots."""
        print_status("Generating Diagnostic Plots...", "PROCESS")

        # Apply Style
        try:
            from scripts.utils.plot_style import apply_tep_style

            colors = apply_tep_style()
        except ImportError:
            # Fallback if style file missing
            colors = {"blue": "#395d85", "accent": "#b43b4e", "dark": "#301E30", "light_blue": "#4b6785", "green": "#4a2650"}

        plt.figure(figsize=(14, 9))

        # Theoretical alignment: x-axis is sigma^2 (potential depth proxy)
        sigma_sq = df["sigma_inferred"] ** 2

        # Propagate mu uncertainty to H0: sigma_H0 = H0 * ln(10)/5 * sigma_mu
        h0_err = (
            df["h0_derived"] * (np.log(10) / 5) * df["error"]
            if "error" in df.columns
            else None
        )

        # Data
        if h0_err is not None:
            plt.errorbar(
                sigma_sq,
                df["h0_derived"],
                yerr=h0_err,
                fmt="o",
                color=colors["blue"],
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.5,
                ecolor=colors["blue"],
                elinewidth=1.2,
                capsize=3,
                alpha=0.8,
                label="SN Ia Hosts",
                zorder=3,
            )
        else:
            plt.scatter(
                sigma_sq,
                df["h0_derived"],
                alpha=0.8,
                color=colors["blue"],
                s=100,
                edgecolor="white",
                label="SN Ia Hosts",
                zorder=3,
            )

        # Regression line against sigma^2 (theoretical regressor)
        if len(df) > 1:
            z = np.polyfit(sigma_sq, df["h0_derived"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(sigma_sq.min(), sigma_sq.max(), 100)
            # Pearson correlation for the displayed sigma^2 axis
            corr_sq = np.corrcoef(sigma_sq, df["h0_derived"])[0, 1]
            plt.plot(
                x_range,
                p(x_range),
                color=colors["accent"],
                linestyle="--",
                linewidth=2.5,
                alpha=0.9,
                label=f"Pearson r={corr_sq:.2f}",
                zorder=4,
            )

        # Annotate the outlier NGC 4639 (jackknife-damped correlation)
        ngc4639_mask = df["normalized_name"].str.strip() == "NGC 4639"
        if ngc4639_mask.any():
            row = df[ngc4639_mask].iloc[0]
            plt.annotate(
                "NGC 4639",
                xy=(row["sigma_inferred"] ** 2, row["h0_derived"]),
                xytext=(row["sigma_inferred"] ** 2 + 6000, row["h0_derived"] + 12),
                fontsize=10,
                fontweight="bold",
                color=colors["accent"],
                arrowprops=dict(arrowstyle="->", color=colors["accent"], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=colors["accent"], alpha=0.9),
                zorder=6,
                clip_on=False,
            )

        plt.xlabel(r"Velocity Dispersion Squared $\sigma^2$ (km$^2$/s$^2$)")
        plt.ylabel(r"Derived $H_0$ (km/s/Mpc)")
        plt.title(r"$H_0$ vs Host Potential Depth ($\sigma^2 \propto |\Phi|$)")
        plt.legend(loc="upper left", frameon=True)
        # Grid handled by style
        plt.tight_layout()

        plt.savefig(self.plot_path, dpi=300)
        print_status(f"Saved plot to {self.plot_path}", "SUCCESS")
        plt.close()

        # Copy to public figures
        public_path = self.root_dir / "site" / "public" / "figures" / "step_03_figure_01_h0_vs_sigma.png"
        public_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.plot_path, public_path)
        print_status(f"Copied plot to {public_path}", "SUCCESS")

    def run(self):
        print_status("Starting Step 2: Stratification", "TITLE")

        merged = self.load_and_merge()
        analyzed = self.calculate_h0(merged)

        # Annotate large-scale environment FIRST so tully_nmb is available
        # for group screening in the density/shear-suppression calculation
        analyzed = self.annotate_large_scale_environment(analyzed)

        # Add density calculation (requires tully_nmb for S_group)
        rho_mean, rho_min, rho_max, analyzed = self.calculate_densities(analyzed)

        final_df, metrics = self.stratify_and_analyze(analyzed)

        # Add density metrics
        metrics["sh0es_density_mean"] = (
            float(rho_mean) if not np.isnan(rho_mean) else None
        )
        metrics["sh0es_density_min"] = float(rho_min) if not np.isnan(rho_min) else None
        metrics["sh0es_density_max"] = float(rho_max) if not np.isnan(rho_max) else None

        final_df.to_csv(self.stratified_output_path, index=False)
        print_status(
            f"Saved stratified data to {self.stratified_output_path}", "SUCCESS"
        )

        # Save JSON
        with open(self.json_output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print_status(f"Saved analysis metrics to {self.json_output_path}", "SUCCESS")

        self.plot_results(final_df, metrics["correlation_r"])

        print_status("Step 2 Complete.", "SUCCESS")


def main():
    step = Step2Stratification()
    step.run()


if __name__ == "__main__":
    main()
