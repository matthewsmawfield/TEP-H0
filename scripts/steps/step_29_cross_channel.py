#!/usr/bin/env python3
"""Write the exclusion record for the former cross-channel consistency step.

The old calculation compared coefficients with different units and inserted a
numerical pulsar response that had no source artifact in this repository. A
cross-channel constraint requires raw data and a derived transfer model for
every observable, so no such constraint is reported by TEP-H0.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs" / "step_29_cross_channel_consistency.json"


class Step12CrossChannel:
    """Compatibility wrapper that records why no joint coefficient is fitted."""

    def run(self):
        result = {
            "validity": "excluded_not_inference",
            "description": (
                "No numerical pulsar--Cepheid--TRGB coefficient consistency "
                "test is supported by the inputs in this repository."
            ),
            "exclusion_reasons": [
                "No pulsar catalog or pulsar likelihood is an input to TEP-H0.",
                "The channel coefficients have different units and transfer functions.",
                "The former residual-flattening estimates were in-sample diagnostics.",
                "The current 16-host Cepheid--TRGB endpoint differential is reported by Step 20.",
            ],
            "replacement_artifact": "results/outputs/step_20_joint_indicator_model.json",
            "theory_prior": None,
            "consistency_test": None,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote excluded-status artifact: {OUTPUT_PATH}")
        return result


def main():
    Step12CrossChannel().run()


if __name__ == "__main__":
    main()
