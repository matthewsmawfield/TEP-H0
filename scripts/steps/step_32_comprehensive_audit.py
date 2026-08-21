#!/usr/bin/env python3
"""Compatibility wrapper for the current pipeline audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.utils.pipeline_audit import audit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "results" / "outputs"


class Step11ComprehensiveAudit:
    def run(self):
        report = audit(project_root=PROJECT_ROOT, write_report=True)
        OUTPUTS.mkdir(parents=True, exist_ok=True)

        json_path = OUTPUTS / "step_32_audit_master_report.json"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        csv_path = OUTPUTS / "step_32_audit_master_table.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "ok", "details"])
            writer.writeheader()
            for check in report.get("checks", []):
                writer.writerow({
                    "name": check.get("name"),
                    "ok": check.get("ok"),
                    "details": json.dumps(check.get("details", {}), sort_keys=True),
                })

        if not report.get("summary", {}).get("ok", False):
            raise RuntimeError(
                f"Pipeline audit failed with {report['summary'].get('n_failed')} failures"
            )
        return report


def main():
    Step11ComprehensiveAudit().run()


if __name__ == "__main__":
    main()
