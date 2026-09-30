#!/usr/bin/env python3
"""
step_62_q_structure_derivation.py

Stellar-structure scope for the within-host kappa_P channel:
what q_i(P) must look like for the fitted coefficient to be a prediction.

Chain
-----
The fitted coefficient multiplies X_i * (log10 P - 1) on Cepheid rows:

    Delta mu = kappa_P * X_i * (log10 P - 1),    kappa_P = -5.79e5 mag
                                                       (step_52, joint fit)

With the Wesenheit slope b (mag per dex), the equivalent transport-ratio
statement is

    d ln q / (dX d log10 P) = kappa_P * ln(10) / b  =  +4.1e5

i.e. the transport ratio's response to the environmental coordinate has a
period slope of ~4e5 per dex. The bare clock-ratio channel is bounded at
|kappa| <~ 1 (step_61), so the slope must be carried by the amplitude
sector: the canonical envelope response S_A(rho_bar) = min[1,
(rho_bar/rho_T)^{1/3}] evaluated at the Cepheid mean density.

Part A assembles the stellar-structure side from adopted scalings:
    <rho>(P) = rho_sun * (Q/P)^2        (period-mean-density relation)
    log10(R/R_sun) = a_R log10 P + b_R  (classical-Cepheid period-radius)
and shows the family

    ln q(P, X) = lambda_0 * X * (rho_T / <rho>(P))^alpha

reproduces the measured slope with an O(0.1) dimensionless weight at
alpha = 1 — i.e. ln q proportional to the inverse envelope mean density.
Since <rho> ∝ P^{-2}, that form predicts the environmental carrier is
quadratic in period: X * P^2, distinct from both the fitted X*(logP-1)
and the previously tested X*(logP-1)^2.

Part B then tests that prediction directly: the same SH0ES design matrix
and covariance as step_52 are refit with three competing carriers
(X*(logP-1), X*(P^2/100 - 1), X*(logP-1)^2), alone and jointly with
kappa_Z, under both anchor conventions.  A data preference for the
quadratic carrier is the empirical signature that the inverse-density
form is realized; a preference for logP over P^2 would bound it.

Inputs
------
results/outputs/step_52_kappaP_kappaZ_disentanglement.csv  (measured kappa_P)
step_34 FullLadderLikelihood machinery                     (design matrix)

Adopted stellar-structure scalings (literature inputs, not fitted):
    Q = 0.041 d   fundamental-mode pulsation constant
    period-radius:  log10(R/R_sun) = 0.68 log10 P + 1.146
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from core.constants import RHO_T, M_SUN, R_SUN, G_NEWTON, C_LIGHT

RHO_SUN = 1.408            # g cm^-3
Q_PULS = 0.041             # days, fundamental-mode pulsation constant (adopted)
PR_A, PR_B = 0.68, 1.146   # log10(R/Rsun) = PR_A log10 P + PR_B  (adopted)
B_WESENHEIT = -3.26        # mag per dex (Appendix C slope, as used in step_61)
X_SCALE = 1e6

STEP52_CSV = (
    Path(__file__).resolve().parents[2]
    / "results" / "outputs" / "step_52_kappaP_kappaZ_disentanglement.csv"
)


# ---------------------------------------------------------------------------
# Part A: structure trend and transport-form inversion
# ---------------------------------------------------------------------------

def structure_grid(p_days):
    """Envelope mean density and related quantities vs period."""
    p = np.asarray(p_days, dtype=float)
    rho = RHO_SUN * (Q_PULS / p) ** 2                    # g cm^-3
    log_r = PR_A * np.log10(p) + PR_B                    # log10 R/Rsun
    r = 10.0 ** log_r                                  # Rsun
    # implied mass from the mean density and radius (self-consistent)
    m_g = (4.0 / 3.0) * np.pi * rho * (r * R_SUN * 100.0) ** 3  # g
    m_sun = m_g / (M_SUN * 1000.0)
    g_surf = G_NEWTON * (m_sun * M_SUN) / (r * R_SUN) ** 2       # m s^-2
    compact = G_NEWTON * (m_sun * M_SUN) / (C_LIGHT ** 2 * r * R_SUN)
    s_a = np.minimum(1.0, (rho / RHO_T) ** (1.0 / 3.0))
    return {
        "P_d": p, "rho_g_cm3": rho, "R_Rsun": r, "M_Msun": m_sun,
        "g_surf_m_s2": g_surf, "compactness": compact, "S_A": s_a,
        "rhoT_over_rho": RHO_T / rho,
    }


def part_a(df52):
    """Invert the measured kappa_P into the required transport form."""
    row = df52[(df52.anchor_convention == "anchor_screened_physical")
               & (df52.model == "kappaP_plus_kappaZ")].iloc[0]
    kappa_p = float(row["kappaP_6"])               # mag per unit X per dex
    row_iso = df52[(df52.anchor_convention == "anchor_screened_physical")
                   & (df52.model == "kappaP_only")].iloc[0]
    kappa_p_iso = float(row_iso["kappaP_6"])

    lam1 = kappa_p * math.log(10.0) / B_WESENHEIT   # d ln q / dX per dex
    lam1_iso = kappa_p_iso * math.log(10.0) / B_WESENHEIT

    # required lambda_0 for  ln q = lambda_0 X (rho_T/<rho>)^alpha ,
    # evaluated at the pivot P0 = 10 d:
    #   slope/dex = lambda_0 X (rho_T/rho0)^alpha * alpha * 2 ln10
    p0 = 10.0
    rho0 = RHO_SUN * (Q_PULS / p0) ** 2
    ratio0 = RHO_T / rho0
    alphas = [1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 4.0 / 3.0, 2.0]
    lam0_vs_alpha = {
        f"{a:.4f}": lam1 / (a * 2.0 * math.log(10.0) * ratio0 ** a)
        for a in alphas
    }

    grid = structure_grid([3, 5, 10, 20, 30, 50, 80])

    out = {
        "kappa_P_fitted_mag": kappa_p,
        "kappa_P_fitted_iso_mag": kappa_p_iso,
        "b_wesenheit": B_WESENHEIT,
        "lambda1_required_per_dex_per_X": lam1,
        "lambda1_required_isolated": lam1_iso,
        "pivot_P_d": p0,
        "pivot_rho_g_cm3": rho0,
        "pivot_S_A": float(min(1.0, (rho0 / RHO_T) ** (1.0 / 3.0))),
        "pivot_rhoT_over_rho": ratio0,
        "S_A_period_slope_dex": (1.0 / 3.0) * (-2.0),  # d ln S_A / d ln P
        "lambda0_by_alpha": lam0_vs_alpha,
        "alpha_1_lambda0": lam0_vs_alpha["1.0000"],
        "predicted_carrier": "X * (P^2 - const): ln q ∝ rho_T/<rho> ∝ P^2",
        "structure_grid": {
            k: ([float(vv) for vv in v] if hasattr(v, "__len__") else float(v))
            for k, v in grid.items()
        },
        "interpretation": (
            "alpha = 1 (ln q proportional to rho_T/<rho>) supplies the "
            "required slope with an O(0.1-1) dimensionless transport "
            "weight; shallower density powers need order-10^2-10^3 "
            "prefactors, steeper powers need 10^-3-10^-8. The "
            "inverse-density carrier is therefore the distinguished "
            "natural-coefficient point. Since <rho> ∝ P^-2 it predicts "
            "the environmental carrier X*(P/10)^2, tested against "
            "X*logP and X*(logP)^2 in Part B; the implied lambda_0 "
            "from the fitted P2 coefficient is recorded there."
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Part B: carrier discrimination on the identical design matrix
# ---------------------------------------------------------------------------

def part_b():
    from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood

    fl = FullLadderLikelihood()
    L, y, C, q, y_source = fl.load_sh0es_data()
    host_sigma, host_screening = fl.load_host_metadata()
    sigma_ref = fl.calculate_effective_sigma_ref()

    b_idx = np.where(q == "bW")[0][0]
    z_idx = np.where(q == "ZW")[0][0]
    logP = L[:, b_idx]                       # log10 P on Cepheid rows
    z_term = L[:, z_idx]
    P = 10.0 ** logP
    n_rows = len(y)

    # Raw design-column conventions (logP, not logP-1) to reproduce step_52
    # exactly: without a kappa0 column, carrier offsets are not absorbed.
    carriers = {
        "logP": logP,                                # step_52 fitted form
        "P2": (P / 10.0) ** 2,                       # derived form ∝ rho^-1
        "logP2": logP ** 2,                          # prior quadratic alt
    }

    rows = []
    for anchor_conv in ["anchor_screened_physical", "anchor_reference_zero"]:
        x_c, x_s, _, _ = fl.build_tep_columns(
            L, q, host_sigma, host_screening, sigma_ref,
            x_mode="centered", anchor_convention=anchor_conv,
            y_source=y_source,
        )
        xc = x_c * X_SCALE
        col_z = -xc * z_term
        col_0 = -xc

        theta_base, _, chi2_base, _, _ = fl.fit_gls(L, y, C)
        h0_idx = np.where(q == "5logH0")[0][0]

        def fit(tag, cols, names):
            L_aug = np.column_stack([L] + cols)
            theta, cov, chi2, rank, _ = fl.fit_gls(L_aug, y, C)
            k = L_aug.shape[1]
            row = {
                "anchor_convention": anchor_conv,
                "model": tag, "n_params": k,
                "chi2": float(chi2),
                "delta_chi2": float(chi2_base - chi2),
                "AIC": float(chi2 + 2 * k),
                "BIC": float(chi2 + k * np.log(n_rows)),
                "H0": float(10 ** (theta[h0_idx] / 5)),
            }
            for j, nm in enumerate(names):
                jdx = len(q) + j
                row[nm] = float(theta[jdx] * X_SCALE)
                row[nm + "_err"] = (
                    float(np.sqrt(cov[jdx, jdx]) * X_SCALE)
                    if cov[jdx, jdx] > 0 else float("nan")
                )
                row[nm + "_sig"] = (
                    float(abs(theta[jdx]) / np.sqrt(cov[jdx, jdx]))
                    if cov[jdx, jdx] > 0 else float("nan")
                )
            return row

        for cname, carr in carriers.items():
            col_p = -xc * carr
            rows.append(fit(f"{cname}_alone", [col_p], [f"kappa_{cname}_6"]))
            rows.append(fit(
                f"{cname}_plus_kappaZ", [col_p, col_z],
                [f"kappa_{cname}_6", "kappaZ_6"],
            ))
            rows.append(fit(
                f"kappa0_{cname}_kappaZ", [col_0, col_p, col_z],
                ["kappa0_6", f"kappa_{cname}_6", "kappaZ_6"],
            ))

    # implied transport weight lambda_0 for the derived carrier:
    #   ln q = lambda_0 X rho_T/<rho> = lambda_0 X [rho_T/(rho_sun Q^2)] (P/10)^2 * (100/100)
    # with carrier (P/10)^2 the fitted kappa_P2 satisfies
    #   kappa_P2 = (b/ln10) lambda_0 [rho_T/(rho_sun Q^2)] * 100
    pref = (RHO_T / (RHO_SUN * Q_PULS ** 2)) * 100.0
    for r in rows:
        for k in list(r):
            if k.startswith("kappa_P2") and k.endswith("_6"):
                r["implied_lambda0"] = float(
                    r[k] * math.log(10.0) / (B_WESENHEIT * pref)
                )
    return rows


def main():
    df52 = pd.read_csv(STEP52_CSV)
    pa = part_a(df52)
    try:
        pb = part_b()
        pb_status = "ok"
    except Exception as exc:  # pragma: no cover - keeps Part A on disk
        pb = {"error": str(exc)}
        pb_status = f"failed: {exc}"

    out = {
        "step": "step_62_q_structure_derivation",
        "part_A_inversion": pa,
        "part_B_carrier_discrimination": pb,
        "part_B_status": pb_status,
        "adopted_inputs": {
            "Q_pulsation_d": Q_PULS,
            "period_radius": "log10(R/Rsun) = 0.68 log10 P + 1.146",
            "rho_T_g_cm3": RHO_T,
            "b_wesenheit": B_WESENHEIT,
        },
    }
    dest = (Path(__file__).resolve().parents[2]
            / "results" / "outputs" / "step_62_q_structure_derivation.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
