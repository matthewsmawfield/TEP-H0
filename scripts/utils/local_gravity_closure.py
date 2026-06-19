"""Quantitative local-gravity closure for the TEP-H0 clock response.

The Cepheid fit measures an observable clock-response coefficient, kappa_Cep.
Solar-system tests constrain a local source-coupled scalar charge. This module
keeps those sectors connected by an explicit source-charge ratio q_source and
computes the Cassini/MICROSCOPE margins implied by the fitted kappa_Cep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict


B_PL_WESENHEIT = -3.26
CASSINI_GAMMA_MINUS_ONE_LIMIT = 2.3e-5
MICROSCOPE_ETA_LIMIT = 1.0e-14
MICROSCOPE_COMPOSITION_CHARGE_CONTRAST = 1.0e-3
SOLAR_NEIGHBORHOOD_SCREENING = 0.96
def compute_vainshtein_suppression_sun() -> float:
    # Constants
    c = 299792.458
    H0_km_s_mpc = 73.0
    Mpc_to_km = 3.086e19
    H0_s = H0_km_s_mpc / Mpc_to_km
    H0_m_inv = H0_s / (c * 1000.0)
    
    rs_sun_m = 2953.0
    r_earth_m = 1.496e11  # Cassini impact parameter is roughly 1 AU to 1.5 AU, we'll use 1 AU as conservative
    
    r_V = (rs_sun_m / (H0_m_inv**2))**(1.0/3.0)
    return float((r_earth_m / r_V)**1.5)

def compute_vainshtein_suppression_earth() -> float:
    c = 299792.458
    H0_km_s_mpc = 73.0
    Mpc_to_km = 3.086e19
    H0_s = H0_km_s_mpc / Mpc_to_km
    H0_m_inv = H0_s / (c * 1000.0)
    
    rs_earth_m = 8.87e-3  # 8.87 mm
    r_microscope_m = 7081e3  # 7081 km orbital radius
    
    r_V_earth = (rs_earth_m / (H0_m_inv**2))**(1.0/3.0)
    return float((r_microscope_m / r_V_earth)**1.5)

DYNAMIC_SOLAR_SOURCE_CHARGE_RATIO = compute_vainshtein_suppression_sun()
DYNAMIC_EARTH_SOURCE_CHARGE_RATIO = compute_vainshtein_suppression_earth()

@dataclass(frozen=True)
class LocalGravityClosure:
    kappa_cep: float
    kappa_cep_err: float
    alpha_clock: float
    alpha_clock_err: float
    solar_neighborhood_screening: float
    solar_source_charge_ratio: float
    earth_source_charge_ratio: float
    cassini_gamma_minus_one_predicted: float
    cassini_gamma_minus_one_limit: float
    cassini_margin: float
    microscope_eta_predicted: float
    microscope_eta_limit: float
    microscope_margin: float
    passes_cassini: bool
    passes_microscope: bool
    passes_source_charge_closure: bool


def alpha_clock_from_kappa(kappa_cep: float, b_pl: float = B_PL_WESENHEIT) -> float:
    """Map kappa_Cep to the dimensionless clock-response amplitude."""

    return float(kappa_cep) * math.log(10.0) / abs(float(b_pl))

def compute_local_gravity_closure(
    kappa_cep: float,
    kappa_cep_err: float = 0.0,
    *,
    solar_neighborhood_screening: float = SOLAR_NEIGHBORHOOD_SCREENING,
    solar_source_ratio: float = DYNAMIC_SOLAR_SOURCE_CHARGE_RATIO,
    earth_source_ratio: float = DYNAMIC_EARTH_SOURCE_CHARGE_RATIO,
    cassini_gamma_limit: float = CASSINI_GAMMA_MINUS_ONE_LIMIT,
    microscope_eta_limit: float = MICROSCOPE_ETA_LIMIT,
    microscope_composition_contrast: float = MICROSCOPE_COMPOSITION_CHARGE_CONTRAST,
) -> LocalGravityClosure:
    alpha_clock = alpha_clock_from_kappa(kappa_cep)
    alpha_clock_err = alpha_clock_from_kappa(kappa_cep_err)
    
    # Cassini test: Sun screening
    local_alpha_sun = alpha_clock * float(solar_neighborhood_screening) * float(solar_source_ratio)
    gamma_pred = 2.0 * local_alpha_sun**2 / (1.0 + local_alpha_sun**2)
    cassini_margin = float(cassini_gamma_limit) / gamma_pred if gamma_pred > 0 else math.inf
    
    # MICROSCOPE test: Earth screening
    local_alpha_earth = alpha_clock * float(solar_neighborhood_screening) * float(earth_source_ratio)
    eta_pred = local_alpha_earth**2 * float(microscope_composition_contrast)
    microscope_margin = float(microscope_eta_limit) / eta_pred if eta_pred > 0 else math.inf
    
    return LocalGravityClosure(
        kappa_cep=float(kappa_cep),
        kappa_cep_err=float(kappa_cep_err),
        alpha_clock=float(alpha_clock),
        alpha_clock_err=float(alpha_clock_err),
        solar_neighborhood_screening=float(solar_neighborhood_screening),
        solar_source_charge_ratio=float(solar_source_ratio),
        earth_source_charge_ratio=float(earth_source_ratio),
        cassini_gamma_minus_one_predicted=float(gamma_pred),
        cassini_gamma_minus_one_limit=float(cassini_gamma_limit),
        cassini_margin=float(cassini_margin),
        microscope_eta_predicted=float(eta_pred),
        microscope_eta_limit=float(microscope_eta_limit),
        microscope_margin=float(microscope_margin),
        passes_cassini=bool(gamma_pred < cassini_gamma_limit),
        passes_microscope=bool(eta_pred < microscope_eta_limit),
        passes_source_charge_closure=bool(gamma_pred < cassini_gamma_limit and eta_pred < microscope_eta_limit),
    )


def closure_to_dict(closure: LocalGravityClosure) -> Dict[str, float | bool]:
    return asdict(closure)


__all__ = [
    "DYNAMIC_SOLAR_SOURCE_CHARGE_RATIO",
    "DYNAMIC_EARTH_SOURCE_CHARGE_RATIO",
    "B_PL_WESENHEIT",
    "CASSINI_GAMMA_MINUS_ONE_LIMIT",
    "MICROSCOPE_COMPOSITION_CHARGE_CONTRAST",
    "MICROSCOPE_ETA_LIMIT",
    "SOLAR_NEIGHBORHOOD_SCREENING",
    "LocalGravityClosure",
    "alpha_clock_from_kappa",
    "closure_to_dict",
    "compute_local_gravity_closure",
]
