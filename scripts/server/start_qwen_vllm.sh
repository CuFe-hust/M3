#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH}" "${SERVED_MODEL_NAME:?Set SERVED_MODEL_NAME}" "${QWEN_API_KEY:?Set QWEN_API_KEY}"
source "${VLLM_VENV:-$HOME/.venvs/vllm-env}/bin/activate"
case "${VLLM_PROFILE:-bf16-24g}" in
  bf16-24g) : ;; low-memory) MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" ;; fp8) : ;; *) echo "Unknown VLLM_PROFILE" >&2; exit 2 ;; esac
exec vllm serve "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-8000}" --api-key "${QWEN_API_KEY}" --dtype auto --max-model-len "${MAX_MODEL_LEN:-16384}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" --max-num-seqs "${MAX_NUM_SEQS:-2}" --limit-mm-per-prompt '{"image":2}'
