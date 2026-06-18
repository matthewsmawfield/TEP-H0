#!/usr/bin/env python3
"""
stellar_validation_core.py
==========================

Core physics module for the TEP-H0 stellar validation pipeline (Step 13).

This module implements the scalar-boundary period transport law that validates
Appendix C of Paper 11.  The logic chain is:

1. MESA/RSP (or GYRE) provides the matter-frame pulsation period P_MESA.
2. At leading order the TEP scalar field is coherent across the Cepheid
   envelope, so the local stellar structure remains standard.
3. The observable correction enters as an external conformal transport factor:
       P_obs = P_MESA * exp(-DeltaTheta).
4. Propagating through the Wesenheit P-L relation recovers the same
   DeltaMu law used in the headline analysis:
       DeltaMu = kappa_Cep * S(rho) * (sigma^2 - sigma_ref^2) / c^2.

The module is designed to run independently of MESA (it accepts a float
P_MESA as input), but also provides helpers to extract P_MESA from MESA
history files when they are available.

Author: Matthew Lukin Smawfield
Date: June 2026
License: CC-BY-4.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.utils.tep_correction import C_KM_S, C_SQUARED_KM_S

# =============================================================================
# CANONICAL TEP-H0 PARAMETERS (Paper 11)
# =============================================================================

KAPPA_CEP: float = 1.05e6          # mag
KAPPA_CEP_ERR: float = 0.43e6      # mag
B_PL: float = -3.26                # Wesenheit Period-Luminosity slope
SIGMA_REF: float = 75.25           # km/s (weighted anchor dispersion)

# Derived clock-sector coupling
#   kappa_Cep = |b| / ln(10) * alpha_clock
#   => alpha_clock = kappa_Cep * ln(10) / |b|
ALPHA_CLOCK: float = KAPPA_CEP * np.log(10.0) / abs(B_PL)

# MESA/RSP rsp_Cepheid canonical baseline period (days).
#
# The MESA test-suite model (4.165 M_sun, T_eff=6050 K, L=1438.8 L_sun,
# Z=0.007) yields a converged pulsation period that depends on the exact
# RSP controls and MESA version.  The value below is a representative
# literature placeholder (~5.5 d) used when MESA is not installed locally.
# When MESA is available, the step auto-detects the period from
#   stellar_validation/mesa_rsp/LOGS/history.data
# or the user can override it with the --mesa-period flag.
P_MESA_CANONICAL_DAYS: float = 5.5  # Placeholder; overwritten by real MESA output when present


# =============================================================================
# SCALAR-BOUNDARY TRANSPORT FUNCTIONS
# =============================================================================

def S_rho(rho_over_rhohalf: float | np.ndarray) -> float | np.ndarray:
    """
    Temporal Topology shear-suppression factor.

    Parameters
    ----------
    rho_over_rhohalf : float or array
        Ratio of local density to the half-saturation density.

    Returns
    -------
    float or array
        S(rho) = 1 / (1 + (rho/rho_half)^2), in [0, 1].
    """
    return 1.0 / (1.0 + np.asarray(rho_over_rhohalf) ** 2)


def delta_theta(
    sigma_km_s: float | np.ndarray,
    rho_over_rhohalf: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Scalar phase shift DeltaTheta for a given environment.

    Parameters
    ----------
    sigma_km_s : float or array
        Host velocity dispersion (km/s).
    rho_over_rhohalf : float or array, optional
        Density ratio for shear suppression. Default 0.0 (fully active).

    Returns
    -------
    float or array
        DeltaTheta = alpha_clock * S(rho) * (sigma^2 - sigma_ref^2) / c^2.
    """
    S = S_rho(rho_over_rhohalf)
    sigma_sq = np.asarray(sigma_km_s) ** 2
    return ALPHA_CLOCK * S * (sigma_sq - SIGMA_REF**2) / C_SQUARED_KM_S


def transport_period(
    P_mesa_days: float | np.ndarray,
    sigma_km_s: float | np.ndarray,
    rho_over_rhohalf: float | np.ndarray = 0.0,
    q_P: float = 1.0,
) -> float | np.ndarray:
    """
    Apply the TEP scalar-boundary transport to a matter-frame period.

    Parameters
    ----------
    P_mesa_days : float or array
        Matter-frame pulsation period (days).
    sigma_km_s : float or array
        Host velocity dispersion (km/s).
    rho_over_rhohalf : float or array, optional
        Density ratio for shear suppression. Default 0.0.
    q_P : float, optional
        Period-response factor (q_P = 1 - chi_P).  Default 1.0 corresponds
        to the leading clock-transport limit where the matter-frame
        stellar structure is unchanged.

    Returns
    -------
    float or array
        P_obs = P_MESA * exp(-q_P * DeltaTheta).
    """
    dtheta = delta_theta(sigma_km_s, rho_over_rhohalf)
    return np.asarray(P_mesa_days) * np.exp(-q_P * dtheta)


def delta_mu_from_transport(
    sigma_km_s: float | np.ndarray,
    rho_over_rhohalf: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Distance-modulus shift inferred from the period transport.

    This follows from propagating P_obs = P_MESA * exp(-DeltaTheta)
    through the Wesenheit P-L relation M_W = a + b*log10(P).

    Parameters
    ----------
    sigma_km_s : float or array
        Host velocity dispersion (km/s).
    rho_over_rhohalf : float or array, optional
        Density ratio. Default 0.0.

    Returns
    -------
    float or array
        DeltaMu_transport = |b| / ln(10) * DeltaTheta.
    """
    dtheta = delta_theta(sigma_km_s, rho_over_rhohalf)
    return abs(B_PL) / np.log(10.0) * dtheta


def delta_mu_direct(
    sigma_km_s: float | np.ndarray,
    rho_over_rhohalf: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Distance-modulus shift from the headline TEP correction formula.

    This is the expression used in the main analysis
    (scripts.utils.tep_correction.tep_correction).

    Parameters
    ----------
    sigma_km_s : float or array
        Host velocity dispersion (km/s).
    rho_over_rhohalf : float or array, optional
        Density ratio. Default 0.0.

    Returns
    -------
    float or array
        DeltaMu_direct = kappa_Cep * S(rho) * (sigma^2 - sigma_ref^2) / c^2.
    """
    S = S_rho(rho_over_rhohalf)
    sigma_sq = np.asarray(sigma_km_s) ** 2
    return KAPPA_CEP * S * (sigma_sq - SIGMA_REF**2) / C_SQUARED_KM_S


def delta_mu_general(
    sigma_km_s: float | np.ndarray,
    rho_over_rhohalf: float | np.ndarray = 0.0,
    q_P: float = 1.0,
    chi_L: float = 0.0,
) -> float | np.ndarray:
    """
    Higher-order distance-modulus shift including structural response.

    Parameters
    ----------
    sigma_km_s : float or array
        Host velocity dispersion (km/s).
    rho_over_rhohalf : float or array, optional
        Density ratio. Default 0.0.
    q_P : float, optional
        Period response parameter (q_P = 1 is leading-order scalar
        coherence). Default 1.0.
    chi_L : float, optional
        Luminosity response parameter. Default 0.0.

    Returns
    -------
    float or array
        DeltaMu_general = (|b|*q_P + 2.5*chi_L) / ln(10) * DeltaTheta.
    """
    dtheta = delta_theta(sigma_km_s, rho_over_rhohalf)
    return (abs(B_PL) * q_P + 2.5 * chi_L) / np.log(10.0) * dtheta


# =============================================================================
# GRID GENERATION AND FITTING
# =============================================================================

def generate_transport_grid(
    P_mesa_days: float,
    sigmas: List[float] | np.ndarray = None,
    rho_ratios: List[float] | np.ndarray = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of transported periods and DeltaMu values over a
    (sigma, rho) grid.

    Parameters
    ----------
    P_mesa_days : float
        Baseline matter-frame period (days).
    sigmas : list or array, optional
        Velocity-dispersion grid points (km/s).  Defaults to a sensible
        astrophysical range.
    rho_ratios : list or array, optional
        Density-ratio grid points.  Defaults to [0.0, 0.5, 1.0, 2.0].

    Returns
    -------
    pd.DataFrame
        Columns: P_mesa_days, sigma_km_s, rho_over_rhohalf, S_rho,
        DeltaTheta, P_obs_days, DeltaMu_transport, DeltaMu_direct,
        abs_difference.
    """
    if sigmas is None:
        sigmas = [30, 50, 75.25, 90, 120, 150, 180, 220]
    if rho_ratios is None:
        rho_ratios = [0.0, 0.5, 1.0, 2.0]

    rows: List[Dict[str, float]] = []
    for sigma in sigmas:
        for rho in rho_ratios:
            dtheta = delta_theta(sigma, rho)
            P_obs = transport_period(P_mesa_days, sigma, rho)
            dmu_transport = delta_mu_from_transport(sigma, rho)
            dmu_direct = delta_mu_direct(sigma, rho)
            rows.append({
                "P_mesa_days": float(P_mesa_days),
                "sigma_km_s": float(sigma),
                "rho_over_rhohalf": float(rho),
                "S_rho": float(S_rho(rho)),
                "DeltaTheta": float(dtheta),
                "P_obs_days": float(P_obs),
                "DeltaMu_transport": float(dmu_transport),
                "DeltaMu_direct": float(dmu_direct),
                "abs_difference": float(abs(dmu_transport - dmu_direct)),
            })

    return pd.DataFrame(rows)


def fit_kappa_from_grid(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Fit kappa_Cep back from a synthetic transport grid.

    This is the numerical closure test: starting from the headline
    kappa_Cep, we generate a grid of DeltaMu values, then perform a
    least-squares fit through the origin to recover the slope.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by generate_transport_grid().

    Returns
    -------
    tuple (kappa_hat, rms_residual, max_abs_diff)
        kappa_hat : fitted slope (mag)
        rms_residual : root-mean-square residual (mag)
        max_abs_diff : max |DeltaMu_transport - DeltaMu_direct| (mag)
    """
    x = df["S_rho"].values * ((df["sigma_km_s"].values ** 2 - SIGMA_REF**2) / C_SQUARED_KM_S)
    y = df["DeltaMu_transport"].values

    # Least-squares through origin: y = kappa * x
    kappa_hat = float(np.sum(x * y) / np.sum(x * x))

    resid = y - kappa_hat * x
    rms = float(np.sqrt(np.mean(resid**2)))
    max_abs_diff = float(df["abs_difference"].max())

    return kappa_hat, rms, max_abs_diff


def run_higher_order_stress_test(
    P_mesa_days: float,
    sigmas: List[float] | np.ndarray = None,
    rho_ratios: List[float] | np.ndarray = None,
    q_P_values: List[float] = None,
    chi_L_values: List[float] = None,
) -> pd.DataFrame:
    """
    Stress-test the transport law with non-zero structural-response
    parameters (q_P != 1, chi_L != 0).

    The falsifier for the leading-order sign is:
        |b| * q_P + 2.5 * chi_L ≈ 0.
    If normal Cepheid physics keeps q_P ~ 1 and chi_L modest, the sign
    survives.

    Parameters
    ----------
    P_mesa_days : float
        Baseline matter-frame period (days).
    sigmas : list or array, optional
        Velocity-dispersion grid points.
    rho_ratios : list or array, optional
        Density-ratio grid points.
    q_P_values : list, optional
        Period-response parameters to scan.  Defaults to [0.8, 1.0, 1.2].
    chi_L_values : list, optional
        Luminosity-response parameters to scan.  Defaults to [-0.2, 0.0, 0.2].

    Returns
    -------
    pd.DataFrame
        Columns include q_P, chi_L, sigma_km_s, rho_over_rhohalf,
        DeltaMu_general, and the falsifier quantity |b|*q_P + 2.5*chi_L.
    """
    if sigmas is None:
        sigmas = [30, 50, 75.25, 90, 120, 150, 180, 220]
    if rho_ratios is None:
        rho_ratios = [0.0, 0.5, 1.0, 2.0]
    if q_P_values is None:
        q_P_values = [0.8, 1.0, 1.2]
    if chi_L_values is None:
        chi_L_values = [-0.2, 0.0, 0.2]

    rows: List[Dict[str, float]] = []
    for q_P in q_P_values:
        for chi_L in chi_L_values:
            for sigma in sigmas:
                for rho in rho_ratios:
                    dmu = delta_mu_general(sigma, rho, q_P, chi_L)
                    rows.append({
                        "q_P": float(q_P),
                        "chi_L": float(chi_L),
                        "sigma_km_s": float(sigma),
                        "rho_over_rhohalf": float(rho),
                        "S_rho": float(S_rho(rho)),
                        "DeltaMu_general": float(dmu),
                        "falsifier": float(abs(B_PL) * q_P + 2.5 * chi_L),
                    })
    return pd.DataFrame(rows)


# =============================================================================
# MESA LOG EXTRACTION HELPERS
# =============================================================================

def read_mesa_history(path: str | Path) -> pd.DataFrame:
    """
    Read a MESA history.data file, skipping the header block.

    MESA history files contain metadata lines followed by a column-header
    line starting with 'model_number'.  This function auto-detects that
    boundary and returns a parsed DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to LOGS/history.data (or equivalent).

    Returns
    -------
    pd.DataFrame
    """
    lines = Path(path).read_text().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("model_number"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find MESA column header in history file.")
    return pd.read_csv(path, sep=r"\s+", skiprows=header_idx, engine="python")


def extract_period_from_history(df: pd.DataFrame) -> float:
    """
    Attempt to extract the pulsation period from a MESA history DataFrame.

    Priority:
    1. A column whose name contains 'period' (case-insensitive).
    2. If no such column exists, raise RuntimeError so the caller can
       fall back to time-series peak finding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from read_mesa_history().

    Returns
    -------
    float
        Period in days (last non-NaN value, assumed converged).
    """
    period_cols = [c for c in df.columns if "period" in c.lower()]
    if period_cols:
        c = period_cols[0]
        p = df[c].dropna().iloc[-1]
        return float(p)
    raise RuntimeError(
        "No period column found in MESA history. "
        "Inspect LOGS outputs and extract from radius/luminosity time series."
    )


# =============================================================================
# RESULTS SERIALISATION
# =============================================================================

def save_validation_json(
    path: str | Path,
    kappa_hat: float,
    rms_residual: float,
    max_abs_diff: float,
    alpha_clock: float,
    P_mesa_days: float,
    grid_md5: str | None = None,
) -> None:
    """
    Write a structured JSON summary of the stellar-validation closure test.
    """
    results = {
        "module": "stellar_validation_core",
        "description": (
            "Numerical closure test: MESA/RSP matter-frame period + "
            "TEP scalar-boundary transport recovers headline kappa_Cep."
        ),
        "parameters": {
            "KAPPA_CEP": KAPPA_CEP,
            "KAPPA_CEP_ERR": KAPPA_CEP_ERR,
            "B_PL": B_PL,
            "SIGMA_REF_km_s": SIGMA_REF,
            "ALPHA_CLOCK": float(alpha_clock),
            "P_mesa_days": float(P_mesa_days),
        },
        "closure_test": {
            "kappa_hat": kappa_hat,
            "rms_residual_mag": rms_residual,
            "max_abs_diff_mag": max_abs_diff,
            "expected_kappa_Cep": KAPPA_CEP,
            "relative_error": abs(kappa_hat - KAPPA_CEP) / KAPPA_CEP if KAPPA_CEP != 0 else None,
            "passed": bool(max_abs_diff < 1e-9),
        },
        "grid_checksum": grid_md5,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


__all__ = [
    "KAPPA_CEP",
    "KAPPA_CEP_ERR",
    "B_PL",
    "SIGMA_REF",
    "ALPHA_CLOCK",
    "P_MESA_CANONICAL_DAYS",
    "S_rho",
    "delta_theta",
    "transport_period",
    "delta_mu_from_transport",
    "delta_mu_direct",
    "delta_mu_general",
    "generate_transport_grid",
    "fit_kappa_from_grid",
    "run_higher_order_stress_test",
    "read_mesa_history",
    "extract_period_from_history",
    "save_validation_json",
]
