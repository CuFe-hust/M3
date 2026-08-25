#!/usr/bin/env bash
set -euo pipefail

# Launch the offline Qwen3.5 visual-planner LoRA evaluation through the
# repository's model and planner seams. 通过仓库模型入口与 planner seam 启动
# 离线 Qwen3.5 visual-planner LoRA 评测。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${M3_CONDA_BIN:-/home/lijia/miniconda3/bin/conda}"
CONDA_ENV="${M3_CONDA_ENV:-m3}"
ADAPTER_PATH="${M3_VISUAL_PLANNER_ADAPTER:-${REPO_ROOT}/outputs/finetune/qwen35-9b-visual-planner-lora/final_adapter}"
OUTPUT_DIR="${M3_VISUAL_PLANNER_EVAL_OUTPUT:-${REPO_ROOT}/outputs/tests/visual-planner-lora-lrsvqa-$(date +%Y%m%d-%H%M%S)}"
LIMIT="${M3_VISUAL_PLANNER_EVAL_LIMIT:-50}"
QUESTIONS_PER_IMAGE="${M3_VISUAL_PLANNER_QUESTIONS_PER_IMAGE:-10}"
MAX_TOKENS="${M3_VISUAL_PLANNER_MAX_TOKENS:-768}"
REPO_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/scripts/evaluate_qwen35_visual_planner_lora.py" \
  --repo "${REPO_ROOT}" \
  --adapter "${ADAPTER_PATH}" \
  --output "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --questions-per-image "${QUESTIONS_PER_IMAGE}" \
  --max-tokens "${MAX_TOKENS}" \
  --repo-head "${REPO_HEAD}" \
  "$@"
