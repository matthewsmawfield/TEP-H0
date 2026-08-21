#!/usr/bin/env python3
"""Compatibility record for the retired narrative synthesis step.

Manuscript prose is maintained in site/components and generated to Markdown
from those HTML sources. This step no longer assembles scientific claims from
heterogeneous historical outputs.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs" / "step_31_TEP_FINAL_ROBUSTNESS_REPORT.md"


class Step9FinalSynthesis:
    def run(self):
        text = """# Step 31 Narrative Synthesis: Retired

**Validity:** excluded from inference

The former synthesis mixed historical residual-flattening, unsupported
cross-channel coefficients, and superseded sample definitions. It is not an
authoritative result surface.

Current manuscript prose is maintained in site/components/*.html. Current
machine-audited headline artifacts are:

- step_39_environment_slope_decomposition.json
- step_39_statistical_tests.json
- step_40_flow_sky_controls.json
- step_20_joint_indicator_model.json
- step_34_full_ladder_likelihood_results.json
- step_45_full_ladder_h0_propagation.json
- step_47_anchor_double_counting_audit.json
"""
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(text)
        print(f"Wrote retired-status report: {OUTPUT_PATH}")
        return {"validity": "excluded_not_inference", "path": str(OUTPUT_PATH)}


def main():
    Step9FinalSynthesis().run()


if __name__ == "__main__":
    main()
