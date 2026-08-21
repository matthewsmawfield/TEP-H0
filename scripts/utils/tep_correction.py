"""Temporal Equivalence Principle endpoint response for the Cepheid channel.

The canonical observable coordinate is the active environmental potential
proxy

    U_i = S_i(E) * sigma_i**2,

not a screened difference of unscreened dispersions.  Relative to the anchor
ensemble, the Cepheid distance-modulus response is therefore

    Delta_mu_i = kappa_cep * (U_i - U_ref) / c**2,
    U_ref = sum_a(w_a * S_a * sigma_a**2) / sum_a(w_a).

This ordering matters.  ``S_i * (sigma_i**2 - U_ref)`` would screen the
anchor subtraction by the host environment, force fully screened hosts back
to zero response, and make a centered host slope depend spuriously on the
anchor convention.  The endpoint form above preserves the TEP distinction
between a screened host endpoint and a screened anchor endpoint.

``kappa_cep`` is an observable channel-response coefficient.  It absorbs the
transfer from the conformal clock/redshift sector into the measured Cepheid
period-luminosity observable; it is not the microscopic coupling and this
module does not assume a particular stellar-envelope transfer mechanism.
"""

from __future__ import annotations

import numpy as np

# Speed of light in km/s (matches units of sigma).
C_KM_S: float = 299792.458
C_SQUARED_KM_S: float = C_KM_S**2

# Adopted continuous group-halo screening parameters (not fit here).
N_CRIT: float = 10.0
GAMMA: float = 1.2


def group_screening_factor(n_mb: float, n_crit: float = N_CRIT, gamma: float = GAMMA) -> float:
    """Continuous group-halo screening factor from Tully group richness.

    S_group(N_mb) = [1 + (N_mb / N_crit)^gamma]^{-1}

    Parameters
    ----------
    n_mb : float
        Tully group membership count (richness proxy).
    n_crit : float
        Structural transition scale to group-dominated halos.
    gamma : float
        Suppression steepness.

    Returns
    -------
    float
        Screening factor in [0, 1].
    """
    if np.isnan(n_mb) or n_mb < 1:
        n_mb = 1.0
    return 1.0 / (1.0 + (n_mb / n_crit) ** gamma)


# Anchor N_mb values used to compute screening factors via the formula.
# M31 and NGC 4258 are from the Tully 2015 2MRS catalog (actual catalog values).
# MW and LMC are not in the catalog; they use representative Local Group values.
ANCHOR_NMB = {
    "MW": 7,         # Local Group typical (PGC 2, 18, 82, 304 in catalog)
    "LMC": 2,        # Local Group satellite typical (PGC 23, 31, 65, 77 in catalog)
    "M31": 11,       # PGC 224 in Tully 2015 2MRS catalog
    "SMC": 2,        # Local Group satellite environment
    "NGC 4258": 65,  # PGC 39600 in Tully 2015 2MRS catalog
}

# Disk velocity dispersions at Cepheid locations and SH0ES-motivated anchor
# weights used by the H0 paper's Cepheid-channel reference construction.
ANCHOR_SIGMA = {
    "MW": 30.0,
    "LMC": 24.0,
    "M31": 160.0,
    "SMC": 22.0,
    "NGC 4258": 115.0,
}

ANCHOR_WEIGHTS = {
    "MW": 0.20,
    "LMC": 0.25,
    "NGC 4258": 0.55,
}

# Canonical TEP environmental screening factors for geometric calibrators.
# These are now derived from the continuous N_mb formula, not hand-tuned.
ANCHOR_SCREENING = {
    name: group_screening_factor(nmb)
    for name, nmb in ANCHOR_NMB.items()
}


def total_screening_factor(rho_local: float, n_mb: float, rho_half: float = 0.5, n_steep: float = 2.0, anchor_name: str = None) -> float:
    """
    Universal Two-Factor Screening model: S_total = S_local * S_group.
    Applies the same deterministic formula to anchors and SN hosts.
    """
    # Local density screening (M31 bulge test mechanism)
    if np.isnan(rho_local) or rho_local < 0:
        s_local = 1.0
    else:
        s_local = 1.0 / (1.0 + (rho_local / rho_half) ** n_steep)
        
    # Group richness screening applies equitably to ALL galaxies based on N_mb
    if anchor_name and anchor_name in ANCHOR_NMB:
        n_mb_effective = ANCHOR_NMB[anchor_name]
    else:
        n_mb_effective = 1.0 if (n_mb is None or np.isnan(n_mb)) else n_mb
        
    s_group = group_screening_factor(n_mb_effective)
        
    return float(s_local * s_group)


def compute_anchor_sigma_ref(
    screened: bool = False,
    weights: dict[str, float] | None = None,
    renormalize_screened: bool = False,
) -> float:
    """Compute ``sqrt(U_ref)``, the anchor active-response scale.

    For ``screened=True``, the squared return value is
    ``sum(w_i S_i sigma_i^2) / sum(w_i)``.  It is a compact representation of
    the anchor endpoint ``U_ref``, not a physical velocity dispersion.  It is
    intentionally not renormalized by ``sum(w_i S_i)`` unless requested for a
    diagnostic, because that would erase absolute endpoint suppression.
    """
    if weights is None:
        weights = ANCHOR_WEIGHTS

    numerator = 0.0
    denominator = 0.0
    for name, weight in weights.items():
        sigma = ANCHOR_SIGMA.get(name)
        if sigma is None:
            continue
        screening = ANCHOR_SCREENING.get(name, 1.0) if screened else 1.0
        numerator += weight * screening * sigma**2
        denominator += weight * (screening if screened and renormalize_screened else 1.0)

    if denominator <= 0:
        raise ValueError("anchor weights must contain at least one positive weighted anchor")
    return float(np.sqrt(numerator / denominator))


def build_host_screening_map(hosts_df=None, project_root=None) -> dict[str, float]:
    """Load host shear-suppression factors from Step 03 with Tully fallback.

    Several late-stage scripts operate on ``hosts_processed.csv``, which does
    not itself carry ``shear_suppression``.  This helper keeps the TEP
    screening layer attached by reading the Step 03 host table first, then
    deriving group-only screening from the Tully catalog for any remaining
    processed hosts with a PGC identifier.
    """
    from pathlib import Path

    import pandas as pd

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    else:
        project_root = Path(project_root)

    screening_map: dict[str, float] = {}
    step3_path = project_root / "results" / "outputs" / "step_03_stratified_h0.csv"
    if step3_path.exists():
        df_step3 = pd.read_csv(step3_path)
        for _, row in df_step3.iterrows():
            screening = row.get("shear_suppression", np.nan)
            if pd.notna(screening):
                screening_map[str(row["normalized_name"])] = float(screening)
                source_id = row.get("source_id", None)
                if pd.notna(source_id):
                    screening_map[str(source_id)] = float(screening)

    if hosts_df is None:
        hosts_path = project_root / "data" / "processed" / "hosts_processed.csv"
        if not hosts_path.exists():
            return screening_map
        hosts_df = pd.read_csv(hosts_path)

    tully_path = project_root / "data" / "raw" / "external" / "tully2015_2mrs_groups_table5.csv"
    if tully_path.exists():
        df_tully = pd.read_csv(tully_path)
        for _, row in hosts_df.iterrows():
            name = str(row["normalized_name"])
            if name in screening_map:
                continue
            pgc = row.get("pgc", None)
            if pd.notna(pgc):
                match = df_tully[df_tully["PGC"] == int(pgc)]
                if len(match) > 0:
                    screening_map[name] = group_screening_factor(float(match.iloc[0]["Nmb"]))

    return screening_map

def tep_environment_coordinate(
    sigma: np.ndarray | float,
    sigma_ref: float,
    S: np.ndarray | float = 1.0,
) -> np.ndarray | float:
    """Return the dimensionless TEP endpoint contrast ``(S sigma^2-U_ref)/c^2``.

    ``sigma_ref`` is the square root of the chosen anchor endpoint ``U_ref``.
    Screening applies to the host endpoint only; the already-computed anchor
    endpoint is subtracted as a constant.
    """
    sigma_sq = np.asarray(sigma) ** 2
    sigma_ref_sq = sigma_ref ** 2
    return (np.asarray(S) * sigma_sq - sigma_ref_sq) / C_SQUARED_KM_S


def tep_correction(
    sigma: np.ndarray | float,
    sigma_ref: float,
    kappa_cep: float,
    S: np.ndarray | float = 1.0,
) -> np.ndarray | float:
    """Physics-derived TEP correction to the distance modulus, in mag.

    Parameters
    ----------
    sigma : array or float
        Host velocity dispersion, km/s.
    sigma_ref : float
        Square root of the effective calibrator endpoint ``U_ref``, km/s.
    kappa_cep : float
        Observable Response Coefficient (units: magnitude; ~10^6 expected).
    S : array or float, optional
        Universal shear-suppression factor S_total in [0, 1]. Default 1.0.

    Returns
    -------
    array or float
        Additive correction Delta_mu such that mu_corr = mu_obs + Delta_mu.
    """
    return kappa_cep * tep_environment_coordinate(sigma, sigma_ref, S)


__all__ = [
    "C_KM_S",
    "C_SQUARED_KM_S",
    "N_CRIT",
    "GAMMA",
    "group_screening_factor",
    "total_screening_factor",
    "compute_anchor_sigma_ref",
    "build_host_screening_map",
    "tep_environment_coordinate",
    "ANCHOR_NMB",
    "ANCHOR_SIGMA",
    "ANCHOR_WEIGHTS",
    "ANCHOR_SCREENING",
    "tep_correction",
]
