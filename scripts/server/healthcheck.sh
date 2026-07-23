#!/usr/bin/env bash
set -euo pipefail
: "${QWEN_HOST:?Set QWEN_HOST}" "${QWEN_PORT:?Set QWEN_PORT}" "${QWEN_API_KEY:?Set QWEN_API_KEY}"
curl --fail --silent --show-error "http://${QWEN_HOST}:${QWEN_PORT}/v1/models" -H "Authorization: Bearer ${QWEN_API_KEY}"
source "${AGENT_VENV:-$HOME/.venvs/agent-env}/bin/activate"
python -m spacers_agent.cli health qwen --live
