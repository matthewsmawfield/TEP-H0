#!/usr/bin/env python3
"""
Generate frozen TEP predictions for prospective Cepheid-SN hosts.

Uses the pipeline-frozen parameters from step_04_tep_correction_results.json:
    kappa_Cep = optimal_kappa_cep
    sigma_ref = sigma_ref
    S(rho, N_mb) from total_screening_factor

The correction for a prospective host is:
    Delta_mu = kappa_Cep * S(rho, N_mb) * (sigma^2 - sigma_ref^2) / c^2

This table is a falsification tool: new hosts should obey the precomputed
Delta_mu without refitting kappa_Cep.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.tep_correction import (
    C_SQUARED_KM_S,
    tep_correction,
    total_screening_factor,
)

# Load frozen parameters from pipeline output
with open(ROOT / "results/outputs/step_04_tep_correction_results.json") as f:
    tep_json = json.load(f)

KAPPA_CEP = float(tep_json["optimal_kappa_cep"])
SIGMA_REF = float(tep_json["sigma_ref"])
C2 = C_SQUARED_KM_S

# Print summary
print("=" * 70)
print("FROZEN TEP PREDICTION TABLE")
print("=" * 70)
print(f"kappa_Cep (frozen): {KAPPA_CEP:.3e} mag")
print(f"sigma_ref (frozen): {SIGMA_REF:.2f} km/s")
print(f"c^2:                {C2:.3f} km^2/s^2")
print()

# Existing sample: print their predicted corrections (for verification)
strat = pd.read_csv(ROOT / "results/outputs/step_03_stratified_h0.csv")
print("Verification: existing N=29 hosts")
print("-" * 70)
print(f"{'Host':<15s} {'sigma':>8s} {'S':>6s} {'Delta_mu':>10s} {'H0_raw':>8s}")
for _, row in strat.iterrows():
    s = row["sigma_inferred"]
    S = row["shear_suppression"]
    dmu = KAPPA_CEP * S * (s**2 - SIGMA_REF**2) / C2
    print(
        f"{row['normalized_name']:<15s} {s:>8.1f} {S:>6.3f} {dmu:>+10.4f} "
        f"{row['h0_derived']:>8.2f}"
    )

# Prospective hosts template
print()
print("=" * 70)
print("PROSPECTIVE HOST TEMPLATE (fill in and verify)")
print("=" * 70)
print(
    "To validate the TEP correction as a resolution mechanism, observe a new\n"
    "Cepheid-SN host, measure its sigma and estimate its local density /\n"
    "group environment, then compare the observed distance modulus residual\n"
    "to the predicted correction below."
)
print()

# Create a grid of prospective host parameters
sigma_grid = [50, 75, 100, 125, 150, 175, 200, 225, 250]
S_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

rows = []
for s in sigma_grid:
    for S in S_grid:
        dmu = KAPPA_CEP * S * (s**2 - SIGMA_REF**2) / C2
        rows.append(
            {
                "sigma_kms": s,
                "S": S,
                "Delta_mu_mag": dmu,
                "Delta_H0_approx_kms_mpc": -dmu * np.log(10) * 70 / 5,
            }
        )

pred_df = pd.DataFrame(rows)
print(pred_df.to_string(index=False))

# Save to CSV
out_path = ROOT / "results/outputs/step_05_prespecified_tep_predictions.csv"
pred_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# Also save the frozen parameter manifest
manifest = {
    "kappa_cep_frozen": KAPPA_CEP,
    "sigma_ref_frozen": SIGMA_REF,
    "c_km_s": np.sqrt(C2),
    "c_squared": C2,
    "screening_formula": "S_group(N_mb) = [1 + (N_mb / N_crit)^gamma]^{-1}",
    "screening_n_crit": 10.0,
    "screening_gamma": 1.2,
    "local_screening_formula": "S_local(rho) = [1 + (rho / rho_half)^n_steep]^{-1}",
    "prediction_criterion": (
        "A new Cepheid-SN host validates the TEP correction if its observed\n"
        "distance-modulus residual agrees with the predicted Delta_mu within\n"
        "the quoted uncertainty (distance-modulus uncertainty ~ 0.1-0.2 mag).\n"
        "If the residual is systematically offset, the model is falsified."
    ),
}

manifest_path = ROOT / "results/outputs/step_05_frozen_tep_prediction_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved manifest to {manifest_path}")
