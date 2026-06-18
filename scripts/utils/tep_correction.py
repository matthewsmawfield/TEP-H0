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
        This is the astrophysical response parameter, not the bare coupling.
    S : array or float, optional
        Continuous shear-suppression factor S(rho) in [0, 1]. Default 1.0
        (fully active regime).

    Returns
    -------
    array or float
        Additive correction Delta_mu such that mu_corr = mu_obs + Delta_mu.
    """
    sigma_sq = np.asarray(sigma) ** 2
    return kappa_cep * S * (sigma_sq - sigma_ref**2) / C_SQUARED_KM_S


# Canonical TEP environmental screening factors for geometric calibrators.
# These are coarse estimates from cosmological environment depth (Local Group,
# Local Volume, group-halo membership) used consistently across the pipeline.
# They match the values in step_10_anchor_stratification.py and the manuscript.
ANCHOR_SCREENING = {
    "MW": 0.10,      # Local Group thin disk; deeply screened by MW halo
    "LMC": 0.10,     # Local Group satellite; deeply screened
    "M31": 0.20,     # Local Group core; strongly screened
    "NGC 4258": 0.50,   # Canes Venatici I Group; partially screened
}

__all__ = ["C_KM_S", "C_SQUARED_KM_S", "KAPPA_CEP_PAPER10", "ANCHOR_SCREENING", "tep_correction"]
