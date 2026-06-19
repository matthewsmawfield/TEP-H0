#!/usr/bin/env python3
"""Step 10b: Local-gravity closure for the TEP-H0 clock response."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.local_gravity_closure import closure_to_dict, compute_local_gravity_closure
from scripts.utils.logger import TEPLogger, print_status, set_step_logger


class Step10bLocalGravityClosure:
    def __init__(self):
        self.root_dir = PROJECT_ROOT
        self.outputs_dir = self.root_dir / "results" / "outputs"
        self.logs_dir = self.root_dir / "logs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.tep_path = self.outputs_dir / "tep_correction_results.json"
        self.output_path = self.outputs_dir / "local_gravity_closure.json"
        self.logger = TEPLogger(
            "step_10b_local_gravity_closure",
            log_file_path=self.logs_dir / "step_10b_local_gravity_closure.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Starting Step 10b: Local-Gravity Closure", "TITLE")
        if not self.tep_path.exists():
            raise FileNotFoundError(f"Missing Step 3 output: {self.tep_path}")

        with open(self.tep_path, "r") as f:
            tep = json.load(f)

        closure = compute_local_gravity_closure(
            kappa_cep=float(tep["optimal_kappa_cep"]),
            kappa_cep_err=float(tep.get("bootstrap_kappa_std", 0.0)),
        )
        result = {
            "description": (
                "Quantitative source-charge closure mapping the fitted Cepheid "
                "clock-response coefficient to local solar-system scalar charge."
            ),
            "inputs": {
                "tep_correction_results": str(self.tep_path.relative_to(self.root_dir)),
                "cassini_reference": "Bertotti, Iess & Tortora 2003, Nature 425, 374",
                "microscope_reference": "Touboul et al. 2022, Physical Review Letters 129, 121102",
                "source_charge_definition": (
                    "alpha_local = alpha_clock * S_solar * q_source, with q_source "
                    "fixed before checking precision-gravity bounds."
                ),
            },
            "closure": closure_to_dict(closure),
            "passes": bool(
                closure.passes_cassini
                and closure.passes_microscope
                and closure.passes_source_charge_closure
            ),
        }

        with open(self.output_path, "w") as f:
            json.dump(result, f, indent=2)

        print_status(
            f"alpha_clock={closure.alpha_clock:.3e}",
            "INFO",
        )
        print_status(
            f"Cassini margin={closure.cassini_margin:.2e}; MICROSCOPE margin={closure.microscope_margin:.2e}",
            "INFO",
        )
        print_status(
            f"q_sun={closure.solar_source_charge_ratio:.1e}; q_earth={closure.earth_source_charge_ratio:.1e}",
            "INFO",
        )

        if not result["passes"]:
            raise RuntimeError("Local-gravity closure failed precision-gravity bounds")

        print_status(f"Saved local-gravity closure to {self.output_path}", "SUCCESS")
        return result


def main():
    Step10bLocalGravityClosure().run()


if __name__ == "__main__":
    main()
