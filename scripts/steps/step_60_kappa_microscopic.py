#!/usr/bin/env python3
"""Action-level Cepheid coefficient, distinct from the ladder fit.

The core-disk closure in Appendix C is

    ln(q_i / q_ref) = -λ_Cep X_i
    κ_Cep = -(b / ln 10) λ_Cep

with b the Leavitt slope. The bare conformal identification is
λ_Cep = 1: the period ratio is the clock ratio, and X is already
that dimensionless clock contrast. No stellar-envelope density is
inserted to manufacture a 10^6 coefficient. The ladder fit is a
different number and is not this derivation.
"""
from __future__ import annotations

import json
from pathlib import Path

import math

B_LEAVITT = -3.30          # magnitude per log10(period), corpus benchmark
LAMBDA_BARE = 1.0          # ln q = -X on the conformal clock


def main():
    kappa = -(B_LEAVITT / math.log(10.0)) * LAMBDA_BARE
    # A host contrast X ~ V^2/c^2 for V = 200 km/s.
    X = (200.0 / 299792.458) ** 2
    delta_mu = kappa * X
    out = {
        "step": "step_60_kappa_microscopic",
        "b_leavitt": B_LEAVITT,
        "lambda_Cep": LAMBDA_BARE,
        "kappa_Cep_mag": kappa,
        "example_X_200kms": X,
        "example_delta_mu_mag": delta_mu,
        "not_this_derivation": (
            "The fitted ladder coefficient ~10^6 mag is not produced by "
            "an envelope density. It is not κ_micro."
        ),
    }
    dest = Path(__file__).resolve().parents[2] / "results" / "step_60_kappa_microscopic.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
