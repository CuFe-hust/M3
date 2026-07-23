#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT}" "${PYTHON_BIN:?Set PYTHON_BIN}" "${RUN_ID:?Set RUN_ID}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m spacers_agent.cli resume-run --run-id "${RUN_ID}"
