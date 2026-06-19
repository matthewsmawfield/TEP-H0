import json
from pathlib import Path
outputs = Path("results/outputs")
local_gravity = json.loads((outputs / "local_gravity_closure.json").read_text())
final_report_text = (outputs / "TEP_FINAL_ROBUSTNESS_REPORT.md").read_text()
try:
    ok = bool(
        local_gravity.get("passes", False)
        and "Local Precision-Gravity Closure" in final_report_text
    )
except (TypeError, ValueError):
    ok = False
print(f"ok = {ok}")
