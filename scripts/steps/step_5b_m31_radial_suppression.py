#!/usr/bin/env python3
"""
Step 5b: M31 Radial Shear-Suppression Model
==========================================

Tests the TEP v0.7 prediction that Temporal Shear is progressively suppressed
by ambient density, using M31 Cepheid P-L data as a controlled single-galaxy
laboratory. Three competing models are compared:

  Model A (Null):          Standard P-L, no environment dependence.
  Model B (Step/ Binary):  Inner/outer intercept step at R = 5 kpc.
  Model C (Continuous):    Intercept varies smoothly with suppression S(ρ),
                           consistent with TEP v0.7 Temporal Topology.

The continuous model is strongly preferred if the scalar field gradient is
attenuated smoothly rather than switching at a discrete threshold.
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from astroquery.vizier import Vizier
except Exception:
    Vizier = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )
except ImportError:
    from scripts.utils.logger import (
        TEPLogger,
        print_status,
        print_table,
        set_step_logger,
    )

try:
    from scripts.utils.plot_style import apply_tep_style

    colors = apply_tep_style()
except ImportError:
    colors = {"blue": "#395d85", "accent": "#b43b4e", "dark": "#301E30"}


# =============================================================================
# Geometry helpers (same as step_5)
# =============================================================================


def galactocentric_distance(ra, dec):
    RA_CENTER = 10.684708
    DEC_CENTER = 41.268750
    PA = 38.0 * np.pi / 180.0
    INC = 77.0 * np.pi / 180.0
    DIST_KPC = 780.0

    d_alpha = (ra - RA_CENTER) * np.cos(np.radians(DEC_CENTER))
    d_delta = dec - DEC_CENTER

    x = d_alpha * np.cos(PA) + d_delta * np.sin(PA)
    y = -d_alpha * np.sin(PA) + d_delta * np.cos(PA)

    x_deproj = x
    y_deproj = y / np.cos(INC)

    r_deg = np.sqrt(x_deproj**2 + y_deproj**2)
    r_rad = np.radians(r_deg)
    r_kpc = DIST_KPC * np.tan(r_rad)
    return r_kpc


def local_density(r_kpc):
    M_bulge = 3.0e10
    r_bulge = 0.61
    M_disk = 7.0e10
    R_d = 5.3
    z_d = 0.6

    r = np.maximum(r_kpc, 0.01)
    rho_bulge = (M_bulge / (2 * np.pi)) * (r_bulge / r) * (1 / (r + r_bulge) ** 3)
    rho_disk_kpc = (M_disk / (4 * np.pi * R_d**2 * z_d)) * np.exp(-r / R_d)
    rho_total_kpc = rho_bulge + rho_disk_kpc
    return rho_total_kpc / 1e9


def shear_suppression(rho_local, rho_half=0.5, n_steep=2.0):
    return 1.0 / (1.0 + (rho_local / rho_half) ** n_steep)


# =============================================================================
# Model fitting
# =============================================================================


def fit_null(logp, w):
    """Model A: W = a + b*logP (standard P-L, no environment)."""
    X = np.column_stack([np.ones_like(logp), logp])
    beta, *_ = np.linalg.lstsq(X, w, rcond=None)
    resid = w - X @ beta
    chi2 = float(np.sum(resid**2))
    return {"a": float(beta[0]), "b": float(beta[1]), "chi2": chi2, "k": 2}


def fit_step(logp, w, r_kpc, r_cut=5.0):
    """Model B: W = a_outer + b*logP + delta_a * I(R < r_cut)."""
    is_inner = (r_kpc < r_cut).astype(float)
    X = np.column_stack([np.ones_like(logp), logp, is_inner])
    beta, *_ = np.linalg.lstsq(X, w, rcond=None)
    resid = w - X @ beta
    chi2 = float(np.sum(resid**2))
    return {
        "a_outer": float(beta[0]),
        "b": float(beta[1]),
        "delta_a": float(beta[2]),
        "chi2": chi2,
        "k": 3,
    }


def fit_continuous(logp, w, s):
    """Model C: W = a_active + b*logP + delta_a * (1 - S)."""
    one_minus_s = 1.0 - s
    X = np.column_stack([np.ones_like(logp), logp, one_minus_s])
    beta, *_ = np.linalg.lstsq(X, w, rcond=None)
    resid = w - X @ beta
    chi2 = float(np.sum(resid**2))
    return {
        "a_active": float(beta[0]),
        "b": float(beta[1]),
        "delta_a": float(beta[2]),
        "chi2": chi2,
        "k": 3,
    }


def aic_bic(chi2, n, k):
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)
    return float(aic), float(bic)


# =============================================================================
# Main
# =============================================================================


def main():
    root_dir = PROJECT_ROOT
    results_dir = root_dir / "results"
    figures_dir = results_dir / "figures"
    outputs_dir = results_dir / "outputs"
    public_dir = root_dir / "site" / "public" / "figures"

    for d in [figures_dir, outputs_dir, public_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logs_dir = root_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger(
        "step_5b_m31_radial", log_file_path=logs_dir / "step_5b_m31_radial.log"
    )
    set_step_logger(logger)

    print_status("=" * 70, "INFO")
    print_status("STEP 5b: M31 RADIAL SHEAR-SUPPRESSION MODEL", "SECTION")
    print_status("=" * 70, "INFO")

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    print_status("Fetching M31 Cepheid data (Kodric et al. 2018)...", "PROCESS")
    if Vizier is None:
        print_status("astroquery not available; aborting.", "ERROR")
        return

    Vizier.ROW_LIMIT = -1
    try:
        catalogs = Vizier.get_catalogs("J/AJ/156/130")
        if not catalogs:
            print_status("No catalogs found.", "ERROR")
            return
        if "J/AJ/156/130/main" in catalogs.keys():
            df = catalogs["J/AJ/156/130/main"].to_pandas()
        else:
            df = catalogs[0].to_pandas()
        print_status(f"Retrieved {len(df)} Cepheids.", "SUCCESS")
    except Exception as e:
        print_status(f"Download failed: {e}", "ERROR")
        return

    if "RAJ2000" in df.columns:
        df.rename(columns={"RAJ2000": "RA"}, inplace=True)
    if "DEJ2000" in df.columns:
        df.rename(columns={"DEJ2000": "DEC"}, inplace=True)

    if "Per" in df.columns:
        df["P"] = df["Per"]
    elif "Pr" in df.columns:
        df["P"] = df["Pr"]

    df = df.dropna(subset=["RA", "DEC", "P", "Wmag"]).copy()
    df = df[df["P"] > 0].copy()
    df["logP"] = np.log10(df["P"].astype(float))

    # Period cuts
    p_min_log = np.log10(10.0)
    p_max_log = np.log10(60.0)
    df = df[(df["logP"] > p_min_log) & (df["logP"] < p_max_log)].copy()

    # ------------------------------------------------------------------
    # 2. Geometry and suppression
    # ------------------------------------------------------------------
    df["R_kpc"] = galactocentric_distance(
        df["RA"].astype(float), df["DEC"].astype(float)
    )
    df["rho_local"] = local_density(df["R_kpc"].values)
    df["S"] = shear_suppression(df["rho_local"].values)

    logp = df["logP"].values.astype(float)
    w = df["Wmag"].values.astype(float)
    r = df["R_kpc"].values.astype(float)
    s = df["S"].values.astype(float)
    n = len(df)

    print_status(f"Sample after cuts: N = {n}", "INFO")
    print_status(f"Radial range: {r.min():.1f} - {r.max():.1f} kpc", "INFO")
    print_status(
        f"Density range: {df['rho_local'].min():.4f} - {df['rho_local'].max():.4f} M_sun/pc^3",
        "INFO",
    )
    print_status(f"Suppression range: {s.min():.3f} - {s.max():.3f}", "INFO")

    # ------------------------------------------------------------------
    # 3. Fit models
    # ------------------------------------------------------------------
    mA = fit_null(logp, w)
    mB = fit_step(logp, w, r, r_cut=5.0)
    mC = fit_continuous(logp, w, s)

    for m, name in [(mA, "A"), (mB, "B"), (mC, "C")]:
        aic, bic = aic_bic(m["chi2"], n, m["k"])
        m["aic"] = aic
        m["bic"] = bic

    # Model comparison
    aics = np.array([mA["aic"], mB["aic"], mC["aic"]])
    bics = np.array([mA["aic"], mB["aic"], mC["aic"]])
    best_aic = aics.min()
    delta_aic = aics - best_aic
    weights = np.exp(-delta_aic / 2)
    weights /= weights.sum()

    headers = ["Model", "k", "chi2", "AIC", "ΔAIC", "w"]
    rows = []
    for label, m, da, wi in zip(
        ["A (Null)", "B (Step)", "C (Continuous)"], [mA, mB, mC], delta_aic, weights
    ):
        rows.append(
            [
                label,
                str(m["k"]),
                f"{m['chi2']:.2f}",
                f"{m['aic']:.2f}",
                f"{da:.2f}",
                f"{wi:.3f}",
            ]
        )
    print_table(headers, rows, title="Model Comparison (Lower AIC = Better)")

    # ------------------------------------------------------------------
    # 4. Predictions for plotting
    # ------------------------------------------------------------------
    predA = mA["a"] + mA["b"] * logp
    predB = mB["a_outer"] + mB["b"] * logp + mB["delta_a"] * (r < 5.0).astype(float)
    predC = mC["a_active"] + mC["b"] * logp + mC["delta_a"] * (1.0 - s)

    residA = w - predA
    residB = w - predB
    residC = w - predC

    # ------------------------------------------------------------------
    # 5. Visualization
    # ------------------------------------------------------------------
    print_status("Generating radial suppression figure...", "PROCESS")

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    tep_blue = colors.get("blue", "#395d85")
    tep_red = colors.get("accent", "#b43b4e")
    tep_dark = colors.get("dark", "#301E30")

    # Panel 1: Residuals vs Radius
    ax = axs[0]
    sc = ax.scatter(
        r,
        residA,
        c=s,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=15,
        alpha=0.6,
        edgecolor="none",
    )
    # Sort for smooth line
    order = np.argsort(r)
    ax.plot(
        r[order], residC[order], color=tep_dark, linewidth=2, label="Model C prediction"
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(
        5, color=tep_red, linestyle=":", linewidth=1.5, alpha=0.7, label="R = 5 kpc cut"
    )
    ax.set_xlabel(r"Galactocentric Radius $R$ (kpc)")
    ax.set_ylabel(r"P-L Residual $\Delta W$ (mag)")
    ax.set_title("(a) Residuals vs. Radius")
    ax.legend(fontsize=9)
    fig.colorbar(sc, ax=ax, label=r"$S(\rho)$")

    # Panel 2: Residuals vs S
    ax = axs[1]
    ax.scatter(s, residA, c=r, cmap="viridis", s=15, alpha=0.6, edgecolor="none")
    # Model C predicts linear trend in (1-S)
    s_smooth = np.linspace(0, 1, 200)
    pred_smooth = mC["delta_a"] * (1.0 - s_smooth)
    ax.plot(s_smooth, pred_smooth, color=tep_dark, linewidth=2, label="Model C trend")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Shear Suppression $S(\rho)$")
    ax.set_ylabel(r"P-L Residual $\Delta W$ (mag)")
    ax.set_title("(b) Residuals vs. Suppression")
    ax.legend(fontsize=9)

    # Panel 3: Binned by radial annuli
    ax = axs[2]
    bins = [0, 2, 5, 10, 15, 25, 50]
    bin_centers = []
    bin_means = []
    bin_errs = []
    bin_ns = []

    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.sum() < 5:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_means.append(float(np.mean(residA[mask])))
        bin_errs.append(float(np.std(residA[mask]) / np.sqrt(mask.sum())))
        bin_ns.append(int(mask.sum()))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_errs = np.array(bin_errs)

    ax.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_errs,
        fmt="o",
        color=tep_blue,
        capsize=4,
        capthick=2,
        markersize=8,
        label="Binned residuals",
    )

    # Overplot Model C binned predictions
    predC_binned = []
    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.sum() < 5:
            predC_binned.append(np.nan)
        else:
            predC_binned.append(float(np.mean(predC[mask] - predA[mask])))
    ax.plot(
        bin_centers,
        np.array(predC_binned),
        "s--",
        color=tep_red,
        markersize=6,
        label="Model C (binned)",
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Galactocentric Radius $R$ (kpc)")
    ax.set_ylabel(r"Mean P-L Residual $\Delta W$ (mag)")
    ax.set_title("(c) Radially Binned Residuals")
    ax.legend(fontsize=9)

    plt.tight_layout()

    fig_path = figures_dir / "m31_radial_suppression.png"
    fig.savefig(fig_path, dpi=300)
    print_status(f"Saved figure to {fig_path}", "SUCCESS")
    plt.close(fig)

    public_path = public_dir / "m31_radial_suppression.png"
    shutil.copy(fig_path, public_path)
    print_status(f"Copied figure to {public_path}", "SUCCESS")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    results = {
        "n_cepheids": int(n),
        "rho_half": 0.5,
        "n_steep": 2.0,
        "model_A_null": {k: v for k, v in mA.items()},
        "model_B_step": {k: v for k, v in mB.items()},
        "model_C_continuous": {k: v for k, v in mC.items()},
        "model_comparison": {
            "delta_aic": [float(x) for x in delta_aic],
            "aic_weights": [float(x) for x in weights],
            "preferred_model": "C (Continuous)"
            if weights[2] > weights[0] and weights[2] > weights[1]
            else "TBD",
        },
    }

    json_path = outputs_dir / "m31_radial_suppression.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print_status(f"Saved results to {json_path}", "SUCCESS")

    print_status("Step 5b Complete.", "SUCCESS")


if __name__ == "__main__":
    main()
