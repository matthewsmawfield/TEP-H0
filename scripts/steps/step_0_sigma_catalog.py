import sys
import time
import subprocess
from pathlib import Path

from scripts.utils.logger import TEPLogger, set_step_logger, print_status


class Step0SigmaCatalog:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.logs_dir = self.root_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger("step_0_sigma_catalog", log_file_path=self.logs_dir / "step_0_sigma_catalog.log")
        set_step_logger(self.logger)

    def run(
        self,
        rebuild: bool = True,
        use_ledacat: bool = True,
        use_ho2007: bool = True,
        use_bassdr2: bool = True,
        use_apj929: bool = True,
        use_mnras482: bool = True,
        use_lit_overrides: bool = False,
    ):
        out_csv = self.root_dir / "data" / "raw" / "external" / "velocity_dispersions_literature_regenerated.csv"
        report_json = self.root_dir / "results" / "outputs" / "sigma_regeneration_report.json"

        if out_csv.exists() and not rebuild:
            print_status(f"Sigma catalog already exists; skipping rebuild: {out_csv}", "INFO")
            return

        cmd = [sys.executable, str(self.root_dir / "scripts" / "utils" / "build_sigma_catalog.py"), "--verbose"]
        if use_ledacat:
            cmd.append("--use-ledacat")
        if use_ho2007:
            cmd.append("--use-ho2007")
        if use_bassdr2:
            cmd.append("--use-bassdr2")
        if use_apj929:
            cmd.append("--use-apj929")
        if use_mnras482:
            cmd.append("--use-mnras482")
        if use_lit_overrides:
            cmd.append("--use-lit-overrides")

        print_status("STEP 0: Building sigma catalog with full provenance", "TITLE")
        print_status(f"Command: {' '.join(cmd)}", "INFO")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0

        if proc.stdout:
            for line in proc.stdout.splitlines():
                if line.strip():
                    print_status(line.strip(), "INFO")
        if proc.stderr:
            for line in proc.stderr.splitlines():
                if line.strip():
                    print_status(line.strip(), "WARNING")

        if proc.returncode != 0:
            raise RuntimeError(f"Sigma catalog build failed (exit {proc.returncode})")

        if not out_csv.exists():
            raise RuntimeError(f"Sigma catalog build reported success but output CSV missing: {out_csv}")
        if not report_json.exists():
            raise RuntimeError(f"Sigma catalog build reported success but report JSON missing: {report_json}")

        print_status(f"Step 0 complete in {dt:.2f}s", "SUCCESS")
