#!/usr/bin/env python3
"""
plot_stellar_validation.py
==========================

Plotting utilities for the TEP-H0 stellar validation module.

Produces:
1. Transport-grid figure (2-panel): P_obs vs sigma and fractional
   period shift vs sigma for several rho ratios.
2. Closure-test figure (2-panel): DeltaMu_transport vs DeltaMu_direct
   scatter with 1:1 + fitted-slope lines, and zoomed residual histogram.
3. Higher-order stress-test figure: DeltaMu_general curves vs sigma
   for scanned (q_P, chi_L) combinations, showing sign survival.

Author: Matthew Lukin Smawfield
Date: June 2026
License: CC-BY-4.0
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utils.stellar_validation_core import (
    KAPPA_CEP,
    SIGMA_REF,
    B_PL,
    transport_period,
    delta_mu_from_transport,
    delta_mu_direct,
    delta_mu_general,
    S_rho,
)


def plot_period_transport_grid(
    P_mesa_days: float,
    sigmas: np.ndarray,
    rho_ratios: list[float],
    output_path: str | Path,
) -> None:
    """
    Two-panel figure illustrating the scalar-boundary period transport.

    Top panel: P_obs(sigma) in days.
    Bottom panel: fractional shift (P_obs / P_mesa - 1) in percent,
    making the conformal transport visible even for small changes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    colors = plt.cm.viridis(np.linspace(0, 1, len(rho_ratios)))

    for rho, color in zip(rho_ratios, colors):
        P_obs = transport_period(P_mesa_days, sigmas, rho)
        frac = (P_obs / P_mesa_days - 1.0) * 100.0
        axes[0].plot(sigmas, P_obs, label=f"$\\rho/\\rho_{{1/2}}={rho}$", color=color, lw=2)
        axes[1].plot(sigmas, frac, color=color, lw=2)

    axes[0].axvline(SIGMA_REF, color="gray", ls="--", lw=1)
    axes[1].axvline(SIGMA_REF, color="gray", ls="--", lw=1, label=f"$\\sigma_{{ref}}={SIGMA_REF:.2f}$ km/s")
    axes[1].axhline(0, color="black", ls="-", lw=0.5)

    axes[0].set_ylabel("Observed period $P_{\\rm obs}$ (days)", fontsize=12)
    axes[0].set_title(
        f"TEP Scalar-Boundary Period Transport  ($P_{{\\rm MESA}}={P_mesa_days:.2f}$ d)",
        fontsize=13,
    )
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)

    axes[1].set_xlabel("Velocity dispersion $\\sigma$ (km/s)", fontsize=12)
    axes[1].set_ylabel("Fractional shift $(P_{\\rm obs}/P_{\\rm MESA}-1)$ (%)", fontsize=12)
    axes[1].set_xlim(sigmas.min(), sigmas.max())

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_closure_test(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Two-panel closure-test figure.

    Left: scatter of DeltaMu_transport vs DeltaMu_direct with 1:1 line
    and a least-squares fitted line (should coincide with 1:1).
    Right: zoomed residual histogram with numerical annotation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter with both 1:1 and fitted lines
    ax = axes[0]
    diff = df["DeltaMu_transport"] - df["DeltaMu_direct"]
    ax.scatter(
        df["DeltaMu_direct"],
        df["DeltaMu_transport"],
        c=df["S_rho"],
        cmap="viridis",
        edgecolor="k",
        s=80,
        zorder=3,
    )
    lim = [
        min(df["DeltaMu_direct"].min(), df["DeltaMu_transport"].min()),
        max(df["DeltaMu_direct"].max(), df["DeltaMu_transport"].max()),
    ]
    # 1:1 reference
    ax.plot(lim, lim, "k--", lw=1.5, label="1:1", zorder=2)
    # Fitted line (least-squares through origin on x = S*(sigma^2-sigma_ref^2)/c^2)
    x = df["S_rho"].values * ((df["sigma_km_s"].values ** 2 - SIGMA_REF**2) / (299792.458**2))
    y = df["DeltaMu_transport"].values
    kappa_hat = np.sum(x * y) / np.sum(x * x)
    y_fit = kappa_hat * x
    ax.plot(df["DeltaMu_direct"], y_fit, "r-", lw=2, label=f"Fit  ($\\kappa_{{\\rm hat}}={kappa_hat:.3e}$)", zorder=1)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("$\\Delta\\mu_{\\rm direct}$ (mag)", fontsize=12)
    ax.set_ylabel("$\\Delta\\mu_{\\rm transport}$ (mag)", fontsize=12)
    ax.set_title("Closure Test: Transport vs Direct", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("$S(\\rho)$")

    # Right: residual histogram (zoomed)
    ax = axes[1]
    ax.hist(diff, bins=20, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("$\\Delta\\mu_{\\rm transport} - \\Delta\\mu_{\\rm direct}$ (mag)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residuals  (max = {diff.abs().max():.2e} mag)", fontsize=13)
    # Add text box with key numbers
    textstr = (
        f"$\\kappa_{{\\rm hat}} = {kappa_hat:.3e}$ mag\n"
        f"RMS = {np.sqrt(np.mean(diff**2)):.2e} mag\n"
        f"$\\kappa_{{\\rm Cep}} = {KAPPA_CEP:.3e}$ mag"
    )
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_higher_order_stress_test(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot DeltaMu_general vs sigma for each scanned (q_P, chi_L) pair.

    The key visual message is that the sign of DeltaMu_general matches
    the headline prediction for all physically plausible combinations,
    because the falsifier |b|*q_P + 2.5*chi_L never crosses zero in the
    scanned domain.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    q_P_vals = sorted(df["q_P"].unique())
    chi_L_vals = sorted(df["chi_L"].unique())
    linestyles = ["-", "--", ":"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(chi_L_vals)))

    for i, q_P in enumerate(q_P_vals):
        for j, chi_L in enumerate(chi_L_vals):
            sub = df[(df["q_P"] == q_P) & (df["chi_L"] == chi_L)].sort_values("sigma_km_s")
            # Only plot the fully active case (rho=0) for clarity
            sub0 = sub[sub["rho_over_rhohalf"] == 0.0]
            if len(sub0) == 0:
                continue
            label = f"$q_P={q_P}$, $\\chi_L={chi_L}$"
            falsifier = sub0["falsifier"].iloc[0]
            ax.plot(
                sub0["sigma_km_s"],
                sub0["DeltaMu_general"],
                ls=linestyles[i % len(linestyles)],
                color=colors[j],
                lw=2,
                label=f"{label}  (f={falsifier:.2f})",
            )

    ax.axvline(SIGMA_REF, color="gray", ls="--", lw=1, label=f"$\\sigma_{{ref}}={SIGMA_REF:.2f}$ km/s")
    ax.axhline(0, color="black", ls="-", lw=0.5)
    ax.set_xlabel("Velocity dispersion $\\sigma$ (km/s)", fontsize=12)
    ax.set_ylabel("$\\Delta\\mu_{\\rm general}$ (mag)", fontsize=12)
    ax.set_title("Higher-Order Stress Test: $\\Delta\\mu$ vs $\\sigma$ ($\\rho=0$)", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


__all__ = [
    "plot_period_transport_grid",
    "plot_closure_test",
    "plot_higher_order_stress_test",
]
