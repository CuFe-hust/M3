#!/usr/bin/env bash
set -euo pipefail
: "${DATASET_NAME:?Set DATASET_NAME}" "${DATASET_PATH:?Set DATASET_PATH}" "${RUN_ID:?Set RUN_ID}"
source "${AGENT_VENV:-$HOME/.venvs/agent-env}/bin/activate"
cd "${REPO_ROOT:-$HOME/spacers-agent}"
set -a; source "${ENV_FILE:-.env}"; set +a
python -m spacers_agent.cli health qwen --live
args=(run-dataset --dataset "${DATASET_NAME}" --root "${DATASET_PATH}" --split "${DATASET_SPLIT:-test}" --task "${TASKS:-counting}" --run-id "${RUN_ID}" --resume)
if [[ -n "${ENABLE_DEEPSEEK:-}" ]]; then args+=(--evaluate); fi
python -m spacers_agent.cli "${args[@]}"
