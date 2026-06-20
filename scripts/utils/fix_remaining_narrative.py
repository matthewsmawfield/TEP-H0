#!/usr/bin/env python3
"""
DEPRECATED: This script contains hardcoded values from old pipeline runs.
Do not run it without first updating the replacement strings to match current
pipeline outputs, or it will inject stale numbers back into the narrative.
For current values, use scripts/utils/sync_narrative_numbers.py instead,
which reads directly from results/outputs/*.json.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# NOTE: These replacements are from an intermediate pipeline state.
# Current values (v0.7, N=29 Hubble-flow-safe):
#   κ_Cep = (1.05 ± 0.41) × 10⁶ mag, σ_ref = 87.17 km/s
#   H0 = 68.75 km/s/Mpc, bootstrap 68.80 ± 1.46
#   LOOCV = 68.58 ± 1.34 km/s/Mpc, tension 0.82σ
#   ΔBIC = +2.6

print("WARNING: This script is deprecated. Use sync_narrative_numbers.py instead.")
print("Done (no changes made).")
