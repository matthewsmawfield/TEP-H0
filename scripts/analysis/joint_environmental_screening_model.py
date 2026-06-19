#!/usr/bin/env python3
"""
Joint Host + Anchor Environmental-Screening Model

Fits a single Observable Response Coefficient κ_Cep to both
SH0ES Hubble-flow hosts and geometric-anchor calibrators,
using environment-specific screening factors S_k.

For each object k:
    Δμ_k = κ · S_k · (σ_k² - σ_ref²) / c²

Hosts (N=29): Δμ_k is inferred from the H0 shift relative to the
TEP-unified mean.

Anchors (N=3): Δμ_k is the zero-point shift relative to LMC.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_correction import C_SQUARED_KM_S, ANCHOR_SCREENING
from scripts.utils.plot_style import apply_tep_style


def load_host_data():
    """Load SH0ES host data with raw and corrected H0."""
    raw_path = PROJECT_ROOT / "results" / "outputs" / "stratified_h0.csv"
    corr_path = PROJECT_ROOT / "results" / "outputs" / "tep_corrected_h0.csv"
    raw = pd.read_csv(raw_path)
    corr = pd.read_csv(corr_path)
    # Merge on source_id
    df = raw.merge(
        corr[["source_id", "h0_corrected", "mu_corrected"]],
        on="source_id",
        how="left",
    )
    return df


def load_anchor_data():
    """Load anchor stratification results."""
    path = PROJECT_ROOT / "results" / "outputs" / "anchor_stratification_test.json"
    with open(path) as f:
        data = json.load(f)
    return data


def build_joint_dataset(host_df, anchor_json, sigma_ref_screened_sq=30.51**2):
    """Build unified dataset in the common Δμ frame."""
    c2 = C_SQUARED_KM_S
    h0_tep = anchor_json["regression"].get("kappa_host", 1.049548e6)
    # Use mean corrected H0 from host data as TEP baseline
    h0_base = host_df["h0_corrected"].mean()

    # ---------- HOSTS ----------
    sigma_host = host_df["sigma_inferred"].values
    S_host = host_df["shear_suppression"].values
    H0_raw = host_df["h0_derived"].values

    # Linearised conversion: Δμ ≈ +(5/ln 10) · (H0_raw - H0_base) / H0_base
    # H0_raw > H0_base for high-sigma hosts → needs positive Δμ correction
    ln10 = np.log(10)
    delta_mu_host = (5.0 / ln10) * (H0_raw - h0_base) / h0_base

    # Propagate uncertainty: σ_H0/H0 ≈ (ln10/5)·σ_μ, and
    # σ_Δμ = (5/ln10)·σ_H0/H0 ≈ σ_μ.  Use per-host distance-modulus error.
    sigma_mu = host_df["error"].values
    sigma_delta_mu_host = sigma_mu.copy()

    # Regressor x = (S·σ² - σ_ref_screened_sq)/c²
    x_host = (S_host * sigma_host**2 - sigma_ref_screened_sq) / c2

    # ---------- ANCHORS ----------
    anchor_names = ["LMC", "NGC 4258", "M31"]
    sigma_anchor = np.array([anchor_json[n]["sigma"] for n in anchor_names])
    M_W = np.array([anchor_json[n]["M_W_absolute"] for n in anchor_names])
    M_W_err = np.array([anchor_json[n]["M_W_err"] for n in anchor_names])
    S_anchor = np.array([ANCHOR_SCREENING.get(n, 1.0) for n in anchor_names])

    # Use LMC as reference: Δμ_j = M_W,j - M_W,LMC
    ref_idx = anchor_names.index("LMC")
    delta_mu_anchor = M_W - M_W[ref_idx]
    sigma_delta_mu_anchor = np.sqrt(M_W_err**2 + M_W_err[ref_idx]**2)
    # Zero out reference uncertainty (it's the pivot)
    sigma_delta_mu_anchor[ref_idx] = 1e-6

    # For anchors, use reference-subtracted regressor to match step_10
    # x_j = (S_j·σ_j² - S_ref·σ_ref_anchor²)/c²  (sigma_ref_screened_sq cancels)
    x_anchor = (
        S_anchor * sigma_anchor**2
        - S_anchor[ref_idx] * sigma_anchor[ref_idx]**2
    ) / c2

    # ---------- COMBINE ----------
    x_all = np.concatenate([x_host, x_anchor])
    y_all = np.concatenate([delta_mu_host, delta_mu_anchor])
    y_err_all = np.concatenate([sigma_delta_mu_host, sigma_delta_mu_anchor])
    is_anchor = np.array([False] * len(x_host) + [True] * len(x_anchor))
    labels = list(host_df["source_id"].values) + anchor_names

    return {
        "x": x_all,
        "y": y_all,
        "y_err": y_err_all,
        "is_anchor": is_anchor,
        "labels": labels,
        "x_host": x_host,
        "y_host": delta_mu_host,
        "x_anchor": x_anchor,
        "y_anchor": delta_mu_anchor,
        "y_err_anchor": sigma_delta_mu_anchor,
        "sigma_ref": np.sqrt(sigma_ref_screened_sq),
        "h0_base": h0_base,
        "anchor_names": anchor_names,
    }


def fit_joint_model(data):
    """Weighted least-squares fit of y = κ·x."""
    x = data["x"]
    y = data["y"]
    y_err = data["y_err"]

    weights = 1.0 / y_err**2

    # Weighted linear regression through origin: y = κ·x
    # κ = Σ w_i x_i y_i / Σ w_i x_i²
    kappa = np.sum(weights * x * y) / np.sum(weights * x**2)
    kappa_err = np.sqrt(1.0 / np.sum(weights * x**2))

    # Predictions and residuals
    y_pred = kappa * x
    residuals = y - y_pred
    chi2 = np.sum((residuals / y_err) ** 2)
    dof = len(x) - 1

    # Pearson r
    r, p = stats.pearsonr(x, y)

    # Per-subset statistics
    mask_host = ~data["is_anchor"]
    mask_anchor = data["is_anchor"]

    chi2_host = np.sum((residuals[mask_host] / y_err[mask_host]) ** 2)
    chi2_anchor = np.sum((residuals[mask_anchor] / y_err[mask_anchor]) ** 2)

    return {
        "kappa": float(kappa),
        "kappa_err": float(kappa_err),
        "chi2": float(chi2),
        "dof": int(dof),
        "chi2_per_dof": float(chi2 / dof) if dof > 0 else float("nan"),
        "r_pearson": float(r),
        "p_pearson": float(p),
        "n_total": int(len(x)),
        "n_hosts": int(mask_host.sum()),
        "n_anchors": int(mask_anchor.sum()),
        "chi2_host": float(chi2_host),
        "chi2_anchor": float(chi2_anchor),
        "residuals": residuals,
        "y_pred": y_pred,
    }


def fit_screened_vs_unscreened(data):
    """Compare joint fit with S_host=1, S_anchor=1 (naive) vs screened."""
    # Naive: all S = 1
    c2 = C_SQUARED_KM_S
    # sigma_ref_screened_sq = data["sigma_ref_screened_sq"]

    x_host_naive = (data["x_host"] / data.get("S_host", np.ones_like(data["x_host"])))
    # Reconstruct: x_host = S·(σ²-σ_ref²)/c², so naive x = (σ²-σ_ref²)/c²
    # We need sigma_host and S_host. Let me recompute from the original.
    # Actually, the simplest is to reload or store S. For now, skip naive
    # comparison in the function and rely on the step_10 output which already
    # has naive vs screened chi2.
    return {}


def create_figure(data, fit_results, output_path):
    """Create joint-model visualization."""
    colors = apply_tep_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    mask_host = ~data["is_anchor"]
    mask_anchor = data["is_anchor"]

    x_host = data["x"][mask_host]
    y_host = data["y"][mask_host]
    x_anchor = data["x"][mask_anchor]
    y_anchor = data["y"][mask_anchor]
    y_err_anchor = data["y_err"][mask_anchor]

    # Plot hosts
    ax.scatter(
        x_host,
        y_host,
        c=colors["blue"],
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        label=f"SN Ia hosts (N={len(x_host)})",
        zorder=3,
    )

    # Plot anchors
    ax.errorbar(
        x_anchor,
        y_anchor,
        yerr=y_err_anchor,
        fmt="s",
        markersize=14,
        c=colors["accent"],
        ecolor=colors["accent"],
        capsize=6,
        capthick=2,
        label=f"Geometric anchors (N={len(x_anchor)})",
        zorder=4,
    )

    for i, name in enumerate(data["anchor_names"]):
        # Offset anchor labels to avoid overlap
        if name == "LMC":
            xytext = (15, 15)
        elif name == "NGC 4258":
            xytext = (15, -20)
        else:
            xytext = (15, 12)
        ax.annotate(
            name,
            (x_anchor[i], y_anchor[i]),
            xytext=xytext,
            textcoords="offset points",
            fontweight="bold",
            color=colors["accent"],
        )

    # Regression line through origin
    x_line = np.linspace(data["x"].min(), data["x"].max(), 200)
    y_line = fit_results["kappa"] * x_line
    ax.plot(
        x_line,
        y_line,
        "--",
        color="black",
        linewidth=2,
        label=rf"Joint fit: $\kappa_\mathrm{{Cep}} = ({fit_results['kappa']/1e6:.2f} \pm {fit_results['kappa_err']/1e6:.2f}) \times 10^6$ mag",
        zorder=2,
    )

    # Shade: host-only fitted κ band (from step 3)
    kappa_host = 1.049548e6
    kappa_host_err = 0.427260e6
    ax.fill_between(
        x_line,
        (kappa_host - kappa_host_err) * x_line,
        (kappa_host + kappa_host_err) * x_line,
        color=colors["blue"],
        alpha=0.15,
        label=r"Host-only $\kappa_\mathrm{Cep}$ ($1.05 \pm 0.43) \times 10^6$ mag)",
        zorder=1,
    )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.4)

    ax.set_xlabel(
        r"Screened regressor $S \cdot (\sigma^2 - \sigma_{\rm ref}^2)/c^2$ ($\times 10^{-7}$)"
    )
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*1e7:.2f}"))
    ax.set_ylabel(r"Observed shift $\Delta\mu$ (mag)")
    ax.set_title(
        "Joint Environmental-Screening Model: Hosts + Anchors\n"
        r"Single $\kappa_\mathrm{Cep}$ with environment-specific $S$",
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure saved: {output_path}")


def main():
    print("=" * 70)
    print("JOINT HOST + ANCHOR ENVIRONMENTAL-SCREENING MODEL")
    print("=" * 70)

    host_df = load_host_data()
    anchor_json = load_anchor_data()

    data = build_joint_dataset(host_df, anchor_json)
    fit = fit_joint_model(data)

    print(f"\nJoint fit (N={fit['n_total']}: {fit['n_hosts']} hosts + {fit['n_anchors']} anchors)")
    print(f"  κ_Cep = {fit['kappa']:.3e} ± {fit['kappa_err']:.3e} mag")
    print(f"  Pearson r = {fit['r_pearson']:.3f} (p = {fit['p_pearson']:.4g})")
    print(f"  χ² = {fit['chi2']:.2f} / {fit['dof']} dof")
    print(f"  χ²/dof = {fit['chi2_per_dof']:.3f}")
    print(f"  Host contribution: χ² = {fit['chi2_host']:.2f}")
    print(f"  Anchor contribution: χ² = {fit['chi2_anchor']:.2f}")

    # Compare with host-only fit
    kappa_host = anchor_json["regression"]["kappa_host"]
    kappa_host_err = anchor_json["regression"]["kappa_host_err"]
    tension = abs(fit["kappa"] - kappa_host) / np.sqrt(fit["kappa_err"] ** 2 + kappa_host_err**2)
    print(f"\nComparison with host-only κ_Cep = {kappa_host:.3e} ± {kappa_host_err:.3e}")
    print(f"  Tension: {tension:.2f}σ")

    # Save results
    results = {
        "joint_kappa_cep": fit["kappa"],
        "joint_kappa_err": fit["kappa_err"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "chi2_per_dof": fit["chi2_per_dof"],
        "r_pearson": fit["r_pearson"],
        "p_pearson": fit["p_pearson"],
        "n_hosts": fit["n_hosts"],
        "n_anchors": fit["n_anchors"],
        "chi2_host": fit["chi2_host"],
        "chi2_anchor": fit["chi2_anchor"],
        "kappa_host_only": kappa_host,
        "kappa_host_only_err": kappa_host_err,
        "tension_with_host_only": tension,
        "sigma_ref": data["sigma_ref"],
        "h0_base": data["h0_base"],
    }

    out_dir = PROJECT_ROOT / "results" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "joint_environmental_screening_model.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    fig_path = PROJECT_ROOT / "results" / "figures" / "figure_05_joint_screening_model.png"
    create_figure(data, fit, fig_path)

    # Also copy to site/public/figures
    public_fig = PROJECT_ROOT / "site" / "public" / "figures" / "figure_05_joint_screening_model.png"
    shutil.copy(fig_path, public_fig)
    print(f"Copied to: {public_fig}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
