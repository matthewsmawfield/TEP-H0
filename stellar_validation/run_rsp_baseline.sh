#!/usr/bin/env bash
# run_rsp_baseline.sh
# ===================
# Wrapper to copy the MESA/RSP Cepheid test suite into the repo and
# run it.  This script is *optional*; step_13_stellar_validation.py
# will fall back to the canonical baseline period if MESA is absent.
#
# Usage:
#   bash stellar_validation/run_rsp_baseline.sh
#
# Prerequisites:
#   - MESA installed ($MESA_DIR set)
#   - MESA SDK initialised ($MESASDK_ROOT set)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Verify MESA environment
if [[ -z "${MESA_DIR:-}" ]]; then
    echo "ERROR: MESA_DIR is not set."
    echo "Please install MESA and set MESA_DIR in your shell profile."
    exit 1
fi

if [[ ! -d "$MESA_DIR/star/test_suite/rsp_Cepheid" ]]; then
    echo "ERROR: MESA rsp_Cepheid test suite not found at:"
    echo "  $MESA_DIR/star/test_suite/rsp_Cepheid"
    exit 1
fi

# Copy test suite into repo
TARGET_DIR="${REPO_ROOT}/stellar_validation/mesa_rsp"
echo "Copying MESA rsp_Cepheid test suite to: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
cp -R "${MESA_DIR}/star/test_suite/rsp_Cepheid/"* "${TARGET_DIR}/"

# Build and run
cd "${TARGET_DIR}"
echo "Building..."
./mk

echo "Running RSP Cepheid model..."
./rn

echo ""
echo "Run complete.  History file should be at:"
echo "  ${TARGET_DIR}/LOGS/history.data"
echo ""
echo "Extract the period with:"
echo "  python scripts/utils/extract_rsp_period.py ${TARGET_DIR}/LOGS/history.data"
