"""
Physics-derived TEP conformal correction.

The Temporal Equivalence Principle predicts that Cepheid periods are
contracted in deep gravitational potentials:

    P_obs = P_true * (1 - S(rho) * |Phi|/c^2)^kappa_0
    
where kappa_0 is the intrinsic coupling (distinct from the Observable Response Coefficient kappa_cep).

where S(rho) is the continuous shear-suppression factor from Temporal
Topology (TEP v0.8). Taylor-expanding for |Phi|/c^2 << 1 and invoking
the virial relation |Phi| ~ sigma^2, the distance-modulus correction
becomes linear in sigma^2/c^2:

    Delta_mu = kappa_cep * S(rho) * (sigma^2 - sigma_ref^2) / c^2

where kappa_cep (Observable Response Coefficient) absorbs the virial
proportionality, the P-L slope, and the 1/ln(10) from the log-expansion.
In this convention kappa_cep has units of magnitude and, given sigma^2/c^2 ~
10^-7, takes values of order 10^6, placing it on the same footing as the
response coefficient inferred from millisecond pulsar spin-down (Paper 10).
Note: kappa_cep is an astrophysical response parameter, distinct from the
bare coupling beta which is constrained by Cassini PPN bounds.

This functional form replaces the earlier phenomenological
log10(sigma/sigma_ref) scaling, which was empirical rather than
derived from the stated TEP mechanism.
"""

from __future__ import annotations

import numpy as np

# Speed of light in km/s (matches units of sigma).
C_KM_S: float = 299792.458
C_SQUARED_KM_S: float = C_KM_S**2

# Continuous group-halo screening parameters (fixed before any fit).
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
    "NGC 4258": 65,  # PGC 39600 in Tully 2015 2MRS catalog
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
    Applies equitably to both anchors and SN hosts to prevent p-hacking.
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
        Effective calibrator velocity dispersion, km/s.
    kappa_cep : float
        Observable Response Coefficient (units: magnitude; ~10^6 expected).
    S : array or float, optional
        Universal shear-suppression factor S_total in [0, 1]. Default 1.0.

    Returns
    -------
    array or float
        Additive correction Delta_mu such that mu_corr = mu_obs + Delta_mu.
    """
    sigma_sq = np.asarray(sigma) ** 2
    sigma_ref_sq = sigma_ref ** 2
    return kappa_cep * S * (sigma_sq - sigma_ref_sq) / C_SQUARED_KM_S


__all__ = [
    "C_KM_S",
    "C_SQUARED_KM_S",
    "N_CRIT",
    "GAMMA",
    "group_screening_factor",
    "total_screening_factor",
    "ANCHOR_NMB",
    "ANCHOR_SCREENING",
    "tep_correction",
]
