#!/usr/bin/env python3
"""
Figure 8: Continuous Shear-Suppression Framework Visualization.

This script creates a four-panel figure visualizing the TEP v0.7 continuous
shear-suppression framework for the 29 SH0ES host galaxies. It demonstrates
how local environmental density modulates the effective shear coupling via
the suppression factor S(rho), and how this density-dependent modulation
propagates into corrected distance moduli and H0 estimates.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    csv_path = project_root / "results" / "outputs" / "tep_corrected_h0.csv"
    fig_dir = project_root / "results" / "figures"
    public_dir = project_root / "site" / "public" / "figures"

    fig_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    alpha_bare = 0.641
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
            ax.annotate(
                name,
                (r["sigma_inferred"], r["rho_local"]),
                textcoords="offset points",
                xytext=(8, 8),
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
        np.log10(rho_smooth),
        s_smooth,
        color=tep_dark,
        linewidth=2.0,
        zorder=3,
    )

    ax.vlines(
        np.log10(df["rho_local"]),
        ymin=0.0,
        ymax=df["shear_suppression"],
        colors=df["shear_suppression"].apply(lambda s: cmap(s)),
        alpha=0.7,
        linewidth=1.5,
        zorder=2,
    )
    ax.scatter(
        np.log10(df["rho_local"]),
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
    ax.set_xlabel(r"$\log_{10}(\rho)$")
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

    # -- Panel (c): Effective coupling alpha*S vs sigma ----------------
    ax = axs[1, 0]
    ax.scatter(
        df["sigma_inferred"],
        df["effective_coupling"],
        c=df["shear_suppression"],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=80,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.axhline(
        alpha_bare,
        color=tep_dark,
        linestyle="--",
        linewidth=1.5,
        label=r"Bare $\alpha = 0.641$",
    )
    ax.set_xlabel(r"Velocity Dispersion $\sigma$ (km/s)")
    ax.set_ylabel(r"Effective Coupling $\alpha \cdot S$")
    ax.legend(loc="upper right")
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
            ax.annotate(
                name,
                (r["shear_suppression"], r["delta_mu"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                color=tep_dark,
            )
    ax.set_xlabel(r"Shear Suppression $S$")
    ax.set_ylabel(r"$|\Delta\mu| = \alpha S \, |\log_{10}(\sigma/\sigma_{\rm ref})|$")
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
    cbar = fig.colorbar(sm, ax=axs, shrink=0.8, pad=0.02)
    cbar.set_label(r"Shear Suppression $S$", rotation=270, labelpad=20)

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_path = fig_dir / "figure_08_shear_suppression.png"
    fig.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")

    public_path = public_dir / "figure_08_shear_suppression.png"
    shutil.copy(save_path, public_path)
    print(f"Copied figure to {public_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
