#!/usr/bin/env bash
set -euo pipefail

# Run the Qwen3.5-9B visual-planner LoRA job inside a named Conda environment.
# 在指定 Conda 环境中运行 Qwen3.5-9B visual-planner LoRA 训练。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-m3}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_COMMAND="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_COMMAND="$(command -v conda)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_COMMAND="${HOME}/miniconda3/bin/conda"
else
  echo "conda executable not found" >&2
  exit 2
fi

exec "${CONDA_COMMAND}" run --no-capture-output -n "${CONDA_ENV_NAME}" \
  python "${SCRIPT_DIR}/finetune_qwen35_9b_visual_planner_lora.py" \
  --model-path "${MODEL_PATH:-${REPO_ROOT}/models/Qwen3.5-9B}" \
  --base-model-id "${BASE_MODEL_ID:-Qwen/Qwen3.5-9B}" \
  --dataset-root "${DATASET_ROOT:-${REPO_ROOT}/data/phase2-train-visualplanning-refined-v4}" \
  --output-dir "${OUTPUT_DIR:-${REPO_ROOT}/outputs/finetune/qwen35-9b-visual-planner-lora}" \
  --local-files-only \
  "$@"
