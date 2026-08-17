#!/usr/bin/env bash
set -euo pipefail
# Default phase-2 launch configuration: seed the base weights from the
# phase-1 merger LoRA adapter (merged at load time), then train LLM LoRA
# + merger base parameters. All data/model paths are specific to the qi2
# host; this file is intentionally NOT committed to git.
# phase2 默认启动配置：从 phase1 merger LoRA adapter 初始化 base 权重（加载
# 时合并），随后训练 LLM LoRA + merger base 参数。所有路径针对 qi2 主机；
# 本文件刻意不提交到 git。
if pgrep -f "finetune_qwen3vl_phase2.py" >/dev/null 2>&1; then
  echo "phase2 finetune already running; not starting a second job  phase2 微调已在运行，未重复启动" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3_ROOT=/home/lijia/M3
PHASE2_REPO=/home/lijia/M3-phase2-20260813
OUTPUT_DIR="${M3_ROOT}/outputs/finetune/qwen3-vl-8b-phase2-$(date +%Y%m%d)"
LOG_FILE="${OUTPUT_DIR}.log"
nohup /home/lijia/miniconda3/envs/m3/bin/python "${PHASE2_REPO}/scripts/finetune_qwen3vl_phase2.py" \
  --model-id "${M3_ROOT}/models/qwen3_vl_8b/weights" \
  --merger-lora-adapter "${M3_ROOT}/outputs/finetune/qwen3-vl-8b-merger-lora" \
  --train-file "${M3_ROOT}/data/phase2-train-processed/train.jsonl" \
  --eval-file "${M3_ROOT}/data/phase2-train-processed/validation.jsonl" \
  --image-root vrsbench="${M3_ROOT}/data/vrsbench" geochat="${M3_ROOT}/data/GeoChat/images_extracted/share/softwares/kartik/GeoChat_finetuning/final_images_llava" \
  --output-dir "${OUTPUT_DIR}" \
  --preflight-limit 32 \
  --eval-steps 10000 \
  "$@" > "${LOG_FILE}" 2>&1 &
echo "started PID=$! output=${OUTPUT_DIR} log=${LOG_FILE}"
