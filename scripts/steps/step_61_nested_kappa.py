#!/usr/bin/env python3
"""Bound on the raw clock channel from the nested rate ratio.

The observable is q = r_spec / r_Cep, then q/q_ref. Both rates are the
conformal factor on the same galactic potential, so the common mode
cancels. The magnitude shift uses the Leavitt slope, b/ln(10), not the
distance-modulus factor 5/ln(10).

    u(r) = (GM/c^2 R) * (3/2 - r^2/(2 R^2))     inside a uniform sphere
    ln q = u(r_cep) - u(r_refclock)
    kappa = (b / ln 10) * ln(q_host/q_ref) / X

X is the host environmental coordinate (V_rot^2/c^2), the regressor the
fit actually uses. The retired 0.960e6 mag figure was 0.288 mag / 3e-7, a
single-rate modulus target that assigned the whole host potential to one
clock. It is not a ratio.

The reference clock is a convention. This step evaluates three choices --
the nuclear spectroscopic aperture, the volume-mean light depth, and the
direct Cepheid-to-Cepheid comparison that retains the cross-galaxy
baseline difference -- and shows the raw conformal clock channel stays
|kappa| <~ 1 under all of them. It cannot supply the measured
kappa_Cep ~ 4e5 mag, which is therefore a property of the response and
screening sector, not a clock-ratio artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

C = 2.99792458e8
G = 6.67430e-11
MSUN = 1.98847e30
PC = 3.085677581e16
B_LEAVITT = -3.26          # Appendix C Wesenheit slope
# Host: the step_05 galaxy. Cepheids at 8 kpc.
M_HOST = 1.0e11 * MSUN
R_HOST = 30.0e3 * PC
R_CEP_HOST = 8.0e3 * PC
# Anchor ensemble, LMC-like, Cepheids across the inner disk.
M_REF = 1.0e10 * MSUN
R_REF = 4.5e3 * PC
R_CEP_REF = 1.5e3 * PC
V_ROT = 200.0e3            # m/s, the host coordinate used as X


def u_s(mass, radius):
    return G * mass / (C * C * radius)


def u_inside(mass, radius, r):
    x2 = (r / radius) ** 2
    return u_s(mass, radius) * (1.5 - 0.5 * x2)


def u_light(mass, radius):
    """Volume-weighted mean clock depth, for the convention in which the
    geometric anchor is a light-travel measurement sampling the galaxy."""
    return 1.2 * u_s(mass, radius)


def main():
    X = (V_ROT / C) ** 2
    # delta mu = (b/ln 10) * delta ln P and kappa = delta mu / X
    projector = B_LEAVITT / math.log(10.0)

    u_cep_host = u_inside(M_HOST, R_HOST, R_CEP_HOST)
    u_cep_ref = u_inside(M_REF, R_REF, R_CEP_REF)

    variants = {}

    # (a) Nuclear spectroscopic aperture as the reference clock:
    #     ln q = u(r_cep) - u(0) within each galaxy, then host - anchor.
    dln = (u_cep_host - u_inside(M_HOST, R_HOST, 0.0)) - (
        u_cep_ref - u_inside(M_REF, R_REF, 0.0)
    )
    variants["nuclear_aperture"] = {
        "ln_q_host": u_cep_host - u_inside(M_HOST, R_HOST, 0.0),
        "ln_q_ref": u_cep_ref - u_inside(M_REF, R_REF, 0.0),
        "dln": dln,
        "kappa_mag": projector * dln / X,
    }

    # (b) Volume-mean light depth as the reference clock: geometric
    #     anchors (maser, parallax, DEB light curves) are light-travel
    #     measurements that sample the galaxy rather than the nucleus.
    dln = (u_cep_host - u_light(M_HOST, R_HOST)) - (
        u_cep_ref - u_light(M_REF, R_REF)
    )
    variants["volume_mean_light"] = {
        "ln_q_host": u_cep_host - u_light(M_HOST, R_HOST),
        "ln_q_ref": u_cep_ref - u_light(M_REF, R_REF),
        "dln": dln,
        "kappa_mag": projector * dln / X,
    }

    # (c) Direct Cepheid-to-Cepheid comparison retaining the cross-galaxy
    #     baseline difference: delta ln P = u_cep_host - u_cep_ref.
    dln = u_cep_host - u_cep_ref
    variants["cepheid_to_cepheid_baseline"] = {
        "ln_q_host": u_cep_host,
        "ln_q_ref": u_cep_ref,
        "dln": dln,
        "kappa_mag": projector * dln / X,
    }

    kappa_bound = max(abs(v["kappa_mag"]) for v in variants.values())
    nuclear = variants["nuclear_aperture"]
    old = 0.960e6
    out = {
        "step": "step_61_nested_kappa",
        "ln_q_host": nuclear["ln_q_host"],
        "ln_q_ref": nuclear["ln_q_ref"],
        "ln_q_ratio": nuclear["dln"],
        "X_host": X,
        "b": B_LEAVITT,
        "projector_b_over_ln10": projector,
        "kappa_nested_mag": nuclear["kappa_mag"],
        "delta_mu_mag": nuclear["kappa_mag"] * X,
        "reference_variants": variants,
        "kappa_bound_mag": kappa_bound,
        "withdrawn_single_rate_benchmark_mag": old,
        "ratio_old_over_nested": (
            old / nuclear["kappa_mag"] if nuclear["kappa_mag"] else None
        ),
        "note": (
            "The single-rate benchmark divided 0.288 mag by X=3e-7. "
            "Under every reference-clock convention the raw conformal "
            "clock channel yields |kappa| <~ 1 mag, so the measured "
            "kappa_Cep ~ 4e5 mag is a response-sector coefficient, not "
            "a clock-ratio artifact."
        ),
    }
    dest = (
        Path(__file__).resolve().parents[2]
        / "results" / "outputs" / "step_61_nested_kappa.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
