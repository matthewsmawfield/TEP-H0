import json
from pathlib import Path
outputs = Path("results/outputs")
local_gravity = json.loads((outputs / "local_gravity_closure.json").read_text())
final_report_text = (outputs / "TEP_FINAL_ROBUSTNESS_REPORT.md").read_text()
closure = local_gravity.get("closure", {})
try:
    ok = (
        bool(local_gravity.get("passes"))
        and bool(closure.get("passes_cassini"))
        and bool(closure.get("passes_microscope"))
        and "Local Precision-Gravity Closure" in final_report_text
    )
except Exception as e:
    print(f"Exception: {e}")
    ok = False
print(f"ok = {ok}")
