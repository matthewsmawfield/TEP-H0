#!/usr/bin/env bash
# run_gyre_baseline.sh
# ====================
# Wrapper to run a GYRE radial-mode check on a MESA profile.
# This script is *optional*; step_13_stellar_validation.py does not
# require GYRE.
#
# Usage:
#   bash stellar_validation/run_gyre_baseline.sh
#
# Prerequisites:
#   - GYRE installed and on PATH
#   - A MESA profile exported to GYRE format (e.g. profile.data.GYRE)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check GYRE
if ! command -v gyre &> /dev/null; then
    echo "ERROR: gyre command not found."
    echo "Please install GYRE and add it to your PATH."
    exit 1
fi

# Prepare working directory
WORK_DIR="${REPO_ROOT}/stellar_validation/gyre"
mkdir -p "${WORK_DIR}"

# The GYRE input template lives in templates/
TEMPLATE="${REPO_ROOT}/stellar_validation/templates/gyre_radial.in"
if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: GYRE template not found: $TEMPLATE"
    exit 1
fi

cd "${WORK_DIR}"
echo "Running GYRE radial-mode analysis..."
gyre "$TEMPLATE"

echo ""
echo "GYRE run complete.  Summary output should be in:"
echo "  ${WORK_DIR}/summary.h5"
echo ""
echo "After extracting the radial-mode period/frequency, apply the same"
echo "TEP transport logic from scripts/utils/stellar_validation_core.py."
