from pathlib import Path

import pandas as pd

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step0SigmaCatalog:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.logs_dir = self.root_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_0_sigma_catalog", log_file_path=self.logs_dir / "step_00_sigma_catalog.log")
        set_step_logger(self.logger)

    def run(self, rebuild: bool = False):
        # SINGLE SOURCE OF TRUTH: literature inner-potential (stellar-dispersion) catalog.
        lit_csv = self.root_dir / "data" / "raw" / "external" / "velocity_dispersions_literature.csv"
        report_json = self.root_dir / "results" / "outputs" / "step_00_sigma_regeneration_report.json"

        # The master file is the only source of truth. It stores the local
        # inner-potential scale sigma_* (stellar dispersion or documented
        # HI-linewidth proxy) at the Cepheid-site scale, with the pinned
        # HyperLEDA Vrot retained alongside for provenance and audit.
        if not lit_csv.exists():
            raise FileNotFoundError(
                f"Master velocity dispersion file not found: {lit_csv}. "
                "This file is required for the pipeline and must contain traceable "
                "catalog values with source identifiers."
            )

        print_status(f"Using master literature catalog: {lit_csv}", "INFO")

        catalog = pd.read_csv(lit_csv, comment="#")
        required = {"galaxy", "pgc", "vrot_kms", "vrot_error_kms", "sigma_kms", "error_kms"}
        missing = sorted(required.difference(catalog.columns))
        if missing:
            raise ValueError(f"Kinematic catalog missing required columns: {missing}")
        if catalog["pgc"].duplicated().any():
            raise ValueError("Kinematic catalog contains duplicate PGC identifiers")
        n_hosts = int(len(catalog))
        n_complete = int(catalog[list(required)].notna().all(axis=1).sum())

        import json
        report_json.parent.mkdir(parents=True, exist_ok=True)
        with open(report_json, 'w') as f:
            json.dump({
                "mode": "literature_inner_potential_sigma",
                "source_file": str(lit_csv.name),
                "note": "Potential scale is the local inner-potential sigma_* (stellar dispersion / HI-linewidth proxy); Vrot is retained only as provenance.",
                "counts": {
                    "n_hosts": n_hosts,
                    "n_with_sigma": n_complete,
                    "n_missing_sigma": n_hosts - n_complete,
                },
            }, f, indent=2)

        print_status("Step 0 complete: master literature catalog loaded", "SUCCESS")
