#!/usr/bin/env bash
set -euo pipefail

# Run GeneralVQAAgent answer-only LoRA SFT in the existing m3 environment.
# 在既有 m3 环境中运行 GeneralVQAAgent answer-only LoRA SFT。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Force GPU mode for YOLO/SegFormer unless caller explicitly overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_COMMAND="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"

exec "${CONDA_COMMAND}" run --no-capture-output -n "${CONDA_ENV_NAME:-m3}" \
  python "${SCRIPT_DIR}/finetune_qwen35_9b_general_vqa_agent_lora.py" \
  --model-path "${MODEL_PATH:-${REPO_ROOT}/models/Qwen3.5-9B}" \
  --config "${CONFIG_PATH:-${REPO_ROOT}/configs/local.yaml}" \
  --annotation-root "${ANNOTATION_ROOT:-${REPO_ROOT}/data/2026-08-24_vqa-agent-io}" \
  --vrsbench-root "${VRSBENCH_ROOT:-${REPO_ROOT}/data/vrsbench}" \
  --large-image-root "${LARGE_IMAGE_ROOT:-${REPO_ROOT}/data/phase2-train-visualplanning-refined-v4}" \
  --supplement-root "${SUPPLEMENT_ROOT:-${REPO_ROOT}/data/20260824-visual-planner-supplement}" \
  --output-dir "${OUTPUT_DIR:-${REPO_ROOT}/outputs/finetune/qwen35-9b-general-vqa-agent-lora}" \
  "$@"
