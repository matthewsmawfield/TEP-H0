#!/usr/bin/env python3
"""Write the exclusion record for the former external-scale confirmation.

Agreement between a coefficient optimized on Cepheid residuals and selected
published distance-scale summaries is not an independent kappa measurement.
The supported matched-host differential is produced by Step 20.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs" / "step_49_independent_kappa_estimation.json"


def main():
    result = {
        "validity": "excluded_not_inference",
        "description": (
            "Published TRGB/JAGB H0 summaries are not used as an independent "
            "confirmation of the in-sample Cepheid endpoint coefficient."
        ),
        "exclusion_reasons": [
            "The former TEP H0 value came from a superseded residual-optimized coefficient.",
            "The comparison assumed zero TRGB/JAGB response rather than estimating it.",
            "Agreement among summary H0 values is not a raw joint likelihood.",
        ],
        "replacement_artifact": "results/outputs/step_20_joint_indicator_model.json",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote excluded-status artifact: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    main()
