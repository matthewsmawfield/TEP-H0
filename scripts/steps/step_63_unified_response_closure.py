#!/usr/bin/env python3
"""
step_63_unified_response_closure.py

Gate-A closure test for the Cepheid response amplitude: does the derived
single response law

    ln q_i(P, X_i) = lambda_0 * X_i * rho_T / <rho>(P_i)
                   = lambda_0 * X_i * 8.45e5 * (P_i/10)^2
                   (rho_T/rho_bar = rho_T P^2/(rho_sun Q^2); step_62)

reproduce BOTH the host-constant response coefficient (kappa_0 / kappa_gal)
and the period-coupled coefficient (kappa_P) as ONE column on the full
SH0ES design matrix, rather than requiring two independent coefficients?

If the law is the true response, then:

  (i)   a single column -X_i*(P_i/10)^2 should capture essentially the
        full delta-chi^2 of the (kappa_0, kappa_P) pair with one fewer
        parameter, and the residual kappa_0 term (constant-in-period
        excess) should be small / sub-sigma;
  (ii)  the fitted coefficient of that column should equal the transport
        weight lambda_0 directly when the carrier is normalised by the
        pivot inverse density  --  with the carrier normalised as
        K(P) = rho_T/rho_bar(P), the fitted coefficient in mag is
        kappa = (b/ln10) * lambda_0, so lambda_0 is read off the fit;
  (iii) the canonical coefficient decomposes as
            kappa_canonical = Gamma_Cep * lambda_0 * rho_T/rho_bar(P_ref),
            Gamma_Cep = |b|/ln10 = 1.416   (Wesenheit period projection)
        at the PL pivot P_ref = 10 d:  Gamma * rho_T/rho_bar(10d)
        = 1.196e6, so kappa_canonical = 9.6e5 corresponds to lambda_0
        = 0.80 -- the same weight the P^2-carrier fit returns.
  (iv)  the ladder-referenced coefficient kappa_Cep = (3.65 +/- 3.04)e5
        corresponds to the law evaluated at an effective reference
        density rho_bar_eff = rho_T * Gamma * lambda_0 / kappa_Cep
        (~6e-5 g/cm^3, P_eff ~ 6 d), i.e. the anchor-referenced
        reading of the same response law.

Model comparison on the identical design matrix as step_52/step_62:

  baseline                        L
  unified_alone                   L + cU                    (1 param)
  unified_plus_kappaZ             L + cU + cZ               (2)
  kappa0_plus_unified             L + c0 + cU               (2)
  kappa0_unified_kappaZ           L + c0 + cU + cZ          (3)
  reference: kappa0_kappaP_kappaZ L + c0 + cP + cZ          (3, step_52)
  reference: kappaP_plus_kappaZ   L + cP + cZ               (2, step_52)

with c0 = -xc, cP = -xc*logP (step_52 convention), cU = -xc*(P/10)^2.

Also reported: the per-host response-weighted carrier means
<X_i rho_T/rho_bar(P)>_host, used to predict the between-host
coefficient kappa_gal under the same law, and the implied lambda_0
map across anchor conventions.

Inputs: step_34 FullLadderLikelihood design matrix; step_52/62 CSVs.
Output: results/outputs/step_63_unified_response_closure.json/.csv
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

from core.constants import RHO_T

RHO_SUN = 1.408            # g cm^-3 (step_62 convention)
Q_PULS = 0.041             # days, fundamental-mode pulsation constant
B_WESENHEIT = -3.26        # mag per dex
X_SCALE = 1e6
RHO_REF_10D = RHO_SUN * (Q_PULS / 10.0) ** 2       # 2.366e-5 g/cm3
INVDENS_10D = RHO_T / RHO_REF_10D                  # 8.453e5
GAMMA_CEP = abs(B_WESENHEIT) / math.log(10.0)      # 1.4159 mag projection


def part_b_design():
    """Unified-column fit on the step_52 design matrix."""
    from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood

    fl = FullLadderLikelihood()
    L, y, C, q, y_source = fl.load_sh0es_data()
    host_sigma, host_screening = fl.load_host_metadata()
    sigma_ref = fl.calculate_effective_sigma_ref()

    b_idx = np.where(q == "bW")[0][0]
    z_idx = np.where(q == "ZW")[0][0]
    logP = L[:, b_idx]
    z_term = L[:, z_idx]
    P = 10.0 ** logP
    n_rows = len(y)
    cep_mask = np.abs(logP) > 0          # Cepheid rows carry logP

    carrier_P2 = (P / 10.0) ** 2
    carrier_K = (RHO_T / (RHO_SUN * Q_PULS ** 2)) * P ** 2  # rho_T/rho_bar

    rows = []
    for anchor_conv in ["anchor_screened_physical", "anchor_reference_zero"]:
        x_c, x_s, _, _ = fl.build_tep_columns(
            L, q, host_sigma, host_screening, sigma_ref,
            x_mode="centered", anchor_convention=anchor_conv,
            y_source=y_source,
        )
        xc = x_c * X_SCALE
        col_0 = -xc
        col_P = -xc * logP
        col_Z = -xc * z_term
        col_U = -xc * carrier_P2            # unified derived law
        col_UK = -xc * carrier_K            # raw rho_T/rho_bar carrier

        theta_base, _, chi2_base, _, _ = fl.fit_gls(L, y, C)
        h0_idx = np.where(q == "5logH0")[0][0]

        def fit(tag, cols, names):
            L_aug = np.column_stack([L] + cols)
            theta, cov, chi2, rank, _ = fl.fit_gls(L_aug, y, C)
            k = L_aug.shape[1]
            row = {
                "anchor_convention": anchor_conv, "model": tag,
                "n_params": k, "chi2": float(chi2),
                "delta_chi2": float(chi2_base - chi2),
                "AIC": float(chi2 + 2 * k),
                "BIC": float(chi2 + k * np.log(n_rows)),
                "H0": float(10 ** (theta[h0_idx] / 5)),
            }
            for j, nm in enumerate(names):
                jdx = len(q) + j
                row[nm] = float(theta[jdx] * X_SCALE)
                row[nm + "_err"] = (float(np.sqrt(cov[jdx, jdx]) * X_SCALE)
                                    if cov[jdx, jdx] > 0 else float("nan"))
                row[nm + "_sig"] = (
                    float(abs(theta[jdx]) / np.sqrt(cov[jdx, jdx]))
                    if cov[jdx, jdx] > 0 else float("nan"))
            return row

        rows.append(fit("unified_alone", [col_U], ["kappaU_6"]))
        rows.append(fit("unified_plus_kappaZ", [col_U, col_Z],
                        ["kappaU_6", "kappaZ_6"]))
        rows.append(fit("kappa0_plus_unified", [col_0, col_U],
                        ["kappa0_6", "kappaU_6"]))
        rows.append(fit("kappa0_unified_kappaZ", [col_0, col_U, col_Z],
                        ["kappa0_6", "kappaU_6", "kappaZ_6"]))
        # raw inverse-density carrier: coefficient should equal
        # (b/ln10)*lambda_0 if the law is exact -> lambda_0 read-off
        rows.append(fit("invdens_carrier", [col_UK], ["kappaK_6"]))
        rows.append(fit("invdens_carrier_kappaZ", [col_UK, col_Z],
                        ["kappaK_6", "kappaZ_6"]))

    # ---- implied lambda_0 map -------------------------------------------------
    # unified carrier (P/10)^2: kappaU = (b/ln10) * lambda_0 * rho_T/rho_10d
    #   => lambda_0 = kappaU * ln10 / (b * rho_T/rho_10d)
    for r in rows:
        if "kappaU_6" in r:
            r["implied_lambda0"] = float(
                r["kappaU_6"] * math.log(10.0)
                / (B_WESENHEIT * INVDENS_10D))
        if "kappaK_6" in r:
            # coefficient of rho_T/rho_bar carrier = (b/ln10) lambda_0
            r["implied_lambda0"] = float(
                r["kappaK_6"] * math.log(10.0) / B_WESENHEIT)
    return rows, fl, L, q, cep_mask, carrier_K


def part_c_kappa_composition(rows):
    """Predicted coefficients under the law with lambda_0 from the fit."""
    lam_map = {}
    for r in rows:
        if r["model"] in ("unified_plus_kappaZ", "kappa0_unified_kappaZ",
                          "unified_alone") and "implied_lambda0" in r:
            lam_map[r["model"] + "/" + r["anchor_convention"]] = \
                r["implied_lambda0"]

    lam0 = np.median(list(lam_map.values()))
    # predicted canonical-scale coefficient at the PL pivot:
    #   kappa(P_ref=10d) = Gamma_Cep * lambda_0 * rho_T/rho_bar(10d)
    kappa_pred_pivot = GAMMA_CEP * lam0 * INVDENS_10D
    # canonical prior and ladder-measured values
    kappa_canonical = 9.6e5
    kappa_cep_meas, kappa_cep_err = 3.65e5, 3.04e5
    lam0_from_canonical = kappa_canonical / (GAMMA_CEP * INVDENS_10D)
    rho_eff_cep = RHO_T * GAMMA_CEP * lam0 / kappa_cep_meas
    p_eff = Q_PULS * math.sqrt(RHO_SUN / rho_eff_cep)
    return {
        "lambda0_map": lam_map,
        "lambda0_median": float(lam0),
        "Gamma_Cep_mag_per_ln": GAMMA_CEP,
        "inv_density_at_pivot": INVDENS_10D,
        "kappa_pred_at_pivot_mag": float(kappa_pred_pivot),
        "kappa_canonical": kappa_canonical,
        "kappa_canonical_over_pred": float(
            kappa_canonical / kappa_pred_pivot),
        "lambda0_implied_by_canonical": float(lam0_from_canonical),
        "kappa_Cep_measured": kappa_cep_meas,
        "kappa_Cep_err": kappa_cep_err,
        "rho_eff_for_kappaCep_g_cm3": float(rho_eff_cep),
        "P_eff_for_kappaCep_days": float(p_eff),
        "composition": (
            "kappa = Gamma_Cep * lambda_0 * rho_T/rho_bar(P_eff); the "
            "canonical 9.6e5 mag corresponds to lambda_0 = %.2f at the "
            "PL pivot (rho_bar=2.37e-5 g/cm3); the ladder value "
            "3.65e5 corresponds to rho_eff = %.1e g/cm3 (P_eff ~ %.0f d)" %
            (lam0_from_canonical, rho_eff_cep, p_eff)),
    }


def main():
    rows, fl, L, q, cep_mask, carrier_K = part_b_design()

    # per-host and global response-weighted carrier moments for the
    # between-host coefficient prediction (informational)
    hostcol = np.where(q == "host")[0][0] if "host" in q else None
    stats = {
        "n_cepheid_rows": int(np.count_nonzero(cep_mask)),
        "inv_density_mean_all_cep_rows": float(
            np.mean(carrier_K[cep_mask])) if cep_mask.any() else None,
        "P2_carrier_mean": float(np.mean(((10.0 ** L[cep_mask,
                  np.where(q == "bW")[0][0]]) / 10.0) ** 2))
            if cep_mask.any() else None,
    }

    comp = part_c_kappa_composition(rows)

    out = {
        "step": "step_63_unified_response_closure",
        "claim": ("the single derived response law ln q = lambda_0 X "
                  "rho_T/rho_bar(P) reproduces the (kappa_0, kappa_P) "
                  "pair on the full ladder matrix with one coefficient"),
        "models": rows,
        "carrier_moments": stats,
        "kappa_composition": comp,
        "adopted_inputs": {
            "Q_pulsation_d": Q_PULS, "rho_sun": RHO_SUN,
            "rho_T": float(RHO_T), "b_wesenheit": B_WESENHEIT,
            "rho_bar_at_10d": float(RHO_REF_10D),
            "inv_density_at_pivot": INVDENS_10D,
        },
    }
    dest = (Path(__file__).resolve().parents[2]
            / "results" / "outputs" / "step_63_unified_response_closure.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(dest.with_suffix(".csv"), index=False)

    print(f"{'model':32s} {'anchor':26s} {'dchi2':>7s} {'kapU':>12s} "
          f"{'kap0':>12s} {'sig':>5s} {'lam0':>6s}")
    for r in rows:
        print(f"{r['model']:32s} {r['anchor_convention']:26s} "
              f"{r['delta_chi2']:7.2f} "
              f"{r.get('kappaU_6', float('nan')):12.4g} "
              f"{r.get('kappa0_6', float('nan')):12.4g} "
              f"{r.get('kappaU_6_sig', r.get('kappaK_6_sig', float('nan'))):5.2f} "
              f"{r.get('implied_lambda0', float('nan')):6.3f}")
    print(json.dumps(comp, indent=2)[:2500])
    print("wrote", dest)


if __name__ == "__main__":
    main()
