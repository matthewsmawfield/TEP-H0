"""
Hubble-flow sample selection utility.

This module provides the single, authoritative definition of the Hubble-flow
redshift cut used throughout the TEP-H0 pipeline.  All pipeline steps that
need to select the primary SN Ia host sample must import and use
``hubble_flow_mask`` rather than hard-coding a threshold, ensuring that a
future adjustment propagates automatically and reproducibly.

Provenance
----------
Prior to v0.9.1 the pipeline used a single criterion

    z_hd > 0.0035

where ``z_hd`` is the peculiar-velocity-corrected CMB-frame redshift from
Pantheon+ (Carrick et al. 2015 flow model; see Scolnic et al. 2022).

This criterion excluded four SH0ES Cepheid hosts whose *observed* CMB-frame
redshift (``z_cmb``) places them comfortably in the Hubble flow but whose
flow-model-corrected ``z_hd`` falls marginally below 0.0035 because of large
model-inferred peculiar-velocity corrections:

    ===========  ===========  ===========  ===================
    Host          z_hd         z_cmb        vpec (km/s)
    ===========  ===========  ===========  ===================
    NGC 1448      0.00333      0.00357       +72
    NGC 3972      0.00349      0.00369       +62
    NGC 3982      0.00349      0.00369       +62
    NGC 5643      0.00331      0.00478      +441
    ===========  ===========  ===========  ===================

For NGC 5643 the flow-model correction is particularly large (441 km/s),
reflecting the well-known sensitivity of the Carrick et al. model in the
local volume (D < 10 Mpc).  Excluding these hosts on the basis of a
model-dependent correction—when their *observed* recession velocities
(1070-1430 km/s) are above the Hubble-flow threshold—removes physically
valid calibrators and reduces the statistical power of the TEP test.

Revised criterion (v0.9.1)
-------------------------
A host is retained in the primary Hubble-flow sample if **either** its
observed CMB-frame redshift **or** its flow-corrected redshift exceeds the
threshold:

    (z_cmb > Z_CUT)  OR  (z_hd > Z_CUT)

with ``Z_CUT = 0.0035`` (corresponding to v ≈ 1050 km/s).

This union criterion:
  * keeps every host that was previously included (no regressions);
  * adds the four borderline hosts whose observed velocities are above
    threshold;
  * remains conservative—hosts with *both* z_cmb and z_hd below 0.0035
    (e.g. NGC 4536, M101, NGC 4424) are still excluded.

The H0 velocity for each retained host continues to be computed from
``z_hd`` (the best available peculiar-velocity-corrected redshift),
preserving the existing distance-ladder methodology.

References
----------
  * Carrick et al. 2015, MNRAS, 451, 4369 — flow model used for z_hd.
  * Scolnic et al. 2022, ApJ, 938, 113 — Pantheon+ redshift definitions.
  * Riess et al. 2022, ApJ, 934, L7 — SH0ES R22 Cepheid host sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Single source of truth ──────────────────────────────────────────────
Z_CUT: float = 0.0035
"""Hubble-flow redshift threshold (v ≈ 1050 km/s)."""

# Hosts added by the union criterion (for documentation / audit)
ADDED_HOSTS: tuple[str, ...] = (
    "NGC 1448",
    "NGC 3972",
    "NGC 3982",
    "NGC 5643",
)


def hubble_flow_mask(
    df: pd.DataFrame,
    z_cut: float = Z_CUT,
    z_hd_col: str = "z_hd",
    z_cmb_col: str = "z_cmb",
) -> pd.Series:
    """Return a boolean mask selecting Hubble-flow hosts.

    Parameters
    ----------
    df : DataFrame
        Must contain ``z_hd_col``.  If ``z_cmb_col`` is also present the
        union criterion ``(z_cmb > z_cut) | (z_hd > z_cut)`` is used;
        otherwise the fallback is ``z_hd > z_cut``.
    z_cut : float
        Redshift threshold (default 0.0035).
    z_hd_col, z_cmb_col : str
        Column names for flow-corrected and observed CMB-frame redshifts.

    Returns
    -------
    pd.Series[bool]
        True for rows that satisfy the Hubble-flow criterion.
        Rows with NaN in both columns return False.
    """
    z_hd = pd.to_numeric(df[z_hd_col], errors="coerce")
    if z_cmb_col in df.columns:
        z_cmb = pd.to_numeric(df[z_cmb_col], errors="coerce")
        return (z_cmb > z_cut) | (z_hd > z_cut)
    return z_hd > z_cut


def apply_hubble_flow_cut(
    df: pd.DataFrame,
    z_cut: float = Z_CUT,
    z_hd_col: str = "z_hd",
    z_cmb_col: str = "z_cmb",
) -> pd.DataFrame:
    """Return a copy of *df* filtered by the Hubble-flow criterion.

    Convenience wrapper around :func:`hubble_flow_mask`.
    """
    mask = hubble_flow_mask(df, z_cut=z_cut, z_hd_col=z_hd_col, z_cmb_col=z_cmb_col)
    return df[mask].copy()


def describe_cut(
    df: pd.DataFrame,
    z_cut: float = Z_CUT,
    z_hd_col: str = "z_hd",
    z_cmb_col: str = "z_cmb",
) -> dict:
    """Return a metadata dict describing the cut for provenance logging."""
    mask = hubble_flow_mask(df, z_cut=z_cut, z_hd_col=z_hd_col, z_cmb_col=z_cmb_col)
    n_before = len(df)
    n_after = int(mask.sum())
    n_removed = n_before - n_after
    has_cmb = z_cmb_col in df.columns
    criterion = f"(z_cmb > {z_cut}) | (z_hd > {z_cut})" if has_cmb else f"z_hd > {z_cut}"
    return {
        "criterion": criterion,
        "z_cut": z_cut,
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_removed,
        "added_hosts": list(ADDED_HOSTS),
        "note": (
            "Union of observed (z_cmb) and flow-corrected (z_hd) redshifts. "
            "See scripts/utils/sample_selection.py for full provenance."
        ),
    }
