
import numpy as np

def jorgensen_aperture_correction(sigma_obs: float, r_ap_arcsec: float, r_eff_arcsec: float, beta: float = 0.04) -> float:
    r"""
    Applies the aperture correction formula from Jorgensen et al. (1995) to normalize
    velocity dispersion measurements to a standard physical radius.
    
    Physics:
    Velocity dispersion profiles in early-type galaxies (and bulges) typically fall off 
    with radius. Measurements taken with a fixed angular aperture (e.g., a fiber or slit) 
    sample different physical fractions of the galaxy depending on its distance and size ($R_{\rm eff}$).
    
    To compare $\sigma$ values fairly (as proxies for central potential depth), we must 
    normalize them to a consistent physical scale, typically $R_{\rm eff}/8$.
    
    Formula:
    $$ \log \sigma_{\rm norm} = \log \sigma_{\rm obs} + \beta \log \left( \frac{r_{\rm ap}}{r_{\rm norm}} \right) $$
    
    Equivalent to:
    $$ \sigma_{\rm norm} = \sigma_{\rm obs} \times \left( \frac{r_{\rm ap}}{r_{\rm norm}} \right)^\beta $$
    
    Parameters:
    -----------
    sigma_obs : float
        Observed velocity dispersion [km/s].
    r_ap_arcsec : float
        Radius of the observation aperture [arcsec].
        (e.g., for SDSS 3" fiber, r_ap = 1.5").
    r_eff_arcsec : float
        Effective radius (half-light radius) of the galaxy [arcsec].
    beta : float, optional
        Power-law slope of the dispersion profile. Default is 0.04 (Jorgensen et al. 1995).
        
    Returns:
    --------
    sigma_norm : float
        Velocity dispersion corrected to the normalization radius ($R_{\rm eff}/8$).
        Returns NaN if inputs are invalid.
    """
    # Validation
    if any(x is None or np.isnan(x) for x in [sigma_obs, r_ap_arcsec, r_eff_arcsec]):
        return np.nan
        
    if sigma_obs <= 0 or r_ap_arcsec <= 0 or r_eff_arcsec <= 0:
        return np.nan
        
    # Standard normalization radius (central)
    r_norm_arcsec = r_eff_arcsec / 8.0
    
    # Calculate Correction Factor
    # If r_ap < r_norm (aperture is smaller than normalization region), we are probing 
    # the very core. Since sigma drops with radius, core sigma > norm sigma.
    # The term (r_ap / r_norm) < 1.
    # Log term is negative.
    # Correction factor < 1.
    # Sigma_norm < Sigma_obs. Correct.
    
    correction_factor = (r_ap_arcsec / r_norm_arcsec) ** beta
    
    return sigma_obs * correction_factor
