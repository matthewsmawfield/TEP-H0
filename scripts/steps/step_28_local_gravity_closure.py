#!/usr/bin/env python3
"""Write the exclusion record for the former local-gravity closure.

The host-level observable coefficient cannot be mapped to Solar-System PPN or
equivalence-principle observables without specifying the microscopic coupling
functions and solving the relevant screened source profiles. The previous step
inserted a DGP-style Vainshtein ansatz that was not derived from the TEP action;
it is therefore excluded from inference.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs" / "step_28_local_gravity_closure.json"


class Step10bLocalGravityClosure:
    """Compatibility wrapper that records why no closure is reported."""

    def run(self):
        result = {
            "validity": "excluded_not_inference",
            "description": (
                "No numerical local-gravity closure is derivable from the "
                "phenomenological host response used in TEP-H0."
            ),
            "exclusion_reasons": [
                "A(phi), B(phi), and V(phi) are not specified by this analysis.",
                "No Solar or terrestrial screened field profile is solved.",
                "The former DGP-style Vainshtein map was an imported assumption, not a TEP derivation.",
                "The observable Cepheid coefficient is not a bare scalar charge.",
            ],
            "passes": None,
            "closure": None,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote excluded-status artifact: {OUTPUT_PATH}")
        return result


def main():
    Step10bLocalGravityClosure().run()


if __name__ == "__main__":
    main()
