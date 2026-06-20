#!/usr/bin/env python3
"""
Figure 8: Continuous Shear-Suppression Framework Visualization.

This script creates a four-panel figure visualizing the TEP v0.8 continuous
shear-suppression framework for the 29 SH0ES host galaxies. It demonstrates
how local environmental density modulates the temporal response via
the suppression factor S(rho), and how this environment-dependent modulation
propagates into corrected distance moduli and H0 estimates.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    csv_path = project_root / "results" / "outputs" / "step_04_tep_corrected_h0.csv"
    json_path = project_root / "results" / "outputs" / "step_04_tep_correction_results.json"
    fig_dir = project_root / "results" / "figures"
    public_dir = project_root / "site" / "public" / "figures"

    fig_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Load fitted kappa_Cep from correction results
    with open(json_path) as f:
        tep_results = json.load(f)
    kappa_cep = float(tep_results["optimal_kappa_cep"])
    
    rho_half = 0.5

    df["delta_mu"] = np.abs(df["value"] - df["mu_corrected"])

    annot_panel_a = ["NGC 2442", "NGC 3021"]
    annot_panel_d = ["NGC 2442", "NGC 976"]

    try:
        from scripts.utils.plot_style import apply_tep_style

        style_colors = apply_tep_style()
    except Exception:
        style_colors = {
            "blue": "#395d85",
            "accent": "#b43b4e",
            "dark": "#301E30",
        }
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 12,
                "figure.dpi": 300,
                "savefig.dpi": 300,
            }
        )

    tep_red = style_colors.get("accent", "#b43b4e")
    tep_blue = style_colors.get("blue", "#395d85")
    tep_dark = style_colors.get("dark", "#301E30")

    cmap = LinearSegmentedColormap.from_list(
        "tep_shear", [tep_red, "#d0c5cc", tep_blue]
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # -- Panel (a): sigma vs rho_local, colored by S -------------------
    ax = axs[0, 0]
    ax.scatter(
        df["sigma_inferred"],
        df["rho_local"],
        c=df["shear_suppression"],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=80,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.axhline(
        rho_half,
        color=tep_dark,
        linestyle="--",
        linewidth=1.5,
        label=r"$\rho_{\rm half}$ (50% suppression)",
    )
    for name in annot_panel_a:
        row = df[df["normalized_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            # Move NGC 2442 label down to avoid clipping
            offset = (8, -15) if name == "NGC 2442" else (8, 8)
            ax.annotate(
                name,
                (r["sigma_inferred"], r["rho_local"]),
                textcoords="offset points",
                xytext=offset,
                fontsize=9,
                color=tep_dark,
            )
    ax.set_xlabel(r"Velocity Dispersion $\sigma$ (km/s)")
    ax.set_ylabel(r"Local Density $\rho$ (M$_\odot$/pc$^3$)")
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.98,
        "(a)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )

    # -- Panel (b): Suppression curve S(rho) with rug marks ------------
    ax = axs[0, 1]
    rho_smooth = np.logspace(-3, np.log10(2.0), 500)
    s_smooth = 1.0 / (1.0 + (rho_smooth / rho_half) ** 2)
    ax.plot(
        rho_smooth,
        s_smooth,
        color=tep_dark,
        linewidth=2.0,
        zorder=3,
    )

    # Rug plot at bottom instead of vertical lines to reduce clutter
    ax.plot(
        df["rho_local"],
        np.zeros_like(df["rho_local"]) - 0.03,
        '|',
        color=tep_dark,
        markersize=8,
        alpha=0.5,
        markeredgewidth=1.5,
    )
    ax.scatter(
        df["rho_local"],
        df["shear_suppression"],
        c=df["shear_suppression"],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=30,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )

    ax.axhspan(0.0, 0.2, color=tep_red, alpha=0.12, label="Strong suppression")
    ax.axhspan(0.8, 1.0, color=tep_blue, alpha=0.12, label="Active shear")
    ax.set_xlabel(r"Local Density $\rho$ (M$_\odot$/pc$^3$)")
    ax.set_xscale("log")
    ax.set_ylabel(r"Shear Suppression $S(\rho)$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")
    ax.text(
        0.02,
        0.98,
        "(b)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )

    # -- Panel (c): Effective coupling kappa*S vs sigma ----------------
    ax = axs[1, 0]
    ax.scatter(
        df["sigma_inferred"],
        df["effective_coupling"] / 1e6,
        c=df["shear_suppression"],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=80,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    # Reference line for unscreened coupling (S=1)
    kappa_mantissa = kappa_cep / 1e6
    ax.axhline(
        kappa_cep / 1e6,
        color=tep_dark,
        linestyle="--",
        linewidth=1.5,
        label=rf"Unscreened coupling $\kappa_{{\rm Cep}} = {kappa_mantissa:.2f} \times 10^6$ mag",
    )
    ax.set_xlabel(r"Velocity Dispersion $\sigma$ (km/s)")
    ax.set_ylabel(r"Effective Coupling $\kappa_{\rm Cep} \cdot S$ [$\times 10^6$ mag]")
    ax.legend(loc="upper right")
    ax.set_ylim(top=kappa_cep / 1e6 * 1.15)  # Add 15% headroom for legend
    ax.text(
        0.02,
        0.98,
        "(c)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )

    # -- Panel (d): H0 correction magnitude vs S -----------------------
    ax = axs[1, 1]
    ax.scatter(
        df["shear_suppression"],
        df["delta_mu"],
        c=df["shear_suppression"],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=80,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for name in annot_panel_d:
        row = df[df["normalized_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            # Move NGC 2442 label down to avoid overlap
            xytext = (8, -15) if name == "NGC 2442" else (8, 8)
            ax.annotate(
                name,
                (r["shear_suppression"], r["delta_mu"]),
                textcoords="offset points",
                xytext=xytext,
                fontsize=9,
                color=tep_dark,
            )
    # Label highest correction point
    max_idx = df["delta_mu"].idxmax()
    r = df.loc[max_idx]
    ax.annotate(
        r["normalized_name"],
        (r["shear_suppression"], r["delta_mu"]),
        textcoords="offset points",
        xytext=(8, 12),
        fontsize=9,
        color=tep_dark,
    )
    ax.set_xlabel(r"Shear Suppression $S$")
    ax.set_ylabel(r"Absolute Correction Magnitude $|\Delta\mu|$ [mag]")
    ax.text(
        0.02,
        0.98,
        "(d)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )

    # ------------------------------------------------------------------
    # Shared colorbar and layout
    # ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    
    # Create colorbar axis explicitly on the far right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Shear Suppression $S$", rotation=270, labelpad=20)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.07, top=0.95, wspace=0.28, hspace=0.30)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_path = fig_dir / "step_06_supplement_05_shear_suppression.png"
    # fig.savefig(save_path, dpi=300)
    # print(f"Saved figure to {save_path}")

    # public_path = public_dir / "step_06_supplement_05_shear_suppression.png"
    # shutil.copy(save_path, public_path)
    # print(f"Copied figure to {public_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
