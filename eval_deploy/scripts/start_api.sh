#!/usr/bin/env bash
# Start the participant API on qi1 (GPU1). Run inside tmux: bash start_api.sh
# 在 qi1 上启动评测 API（GPU1）；建议在 tmux 会话中运行。
set -euo pipefail

# .bashrc exports a local mihomo proxy for interactive shells; the API must
# never inherit it. .bashrc 会导出本机代理；评测 API 绝不能继承。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# Reduce CUDA allocator fragmentation across long evaluation runs.
# 降低长评测期间的 CUDA 分配器碎片化。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export M3_ROOT="${M3_ROOT:-/home/tongyi/M3}"
export M3_CONFIG="${M3_CONFIG:-/home/tongyi/eval/m3_bridge/m3_eval.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/tongyi/eval/runs}"
export MODEL_MODE=real
export BRIDGE_TASK_MODE="${BRIDGE_TASK_MODE:-constraint}"
export BRIDGE_BBOX_PICK="${BRIDGE_BBOX_PICK:-largest_area}"
export RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

MODEL_API_KEY="$(cat /home/tongyi/eval/secrets/model_api_key)"
export MODEL_API_KEY

echo "[start_api] run_tag=${RUN_TAG} package=${EVAL_PACKAGE:-/home/tongyi/eval/packages/rsu-dev100-v2}"
exec /home/tongyi/.conda/envs/m3/bin/python /home/tongyi/eval/scripts/run_api.py \
  --package "${EVAL_PACKAGE:-/home/tongyi/eval/packages/rsu-dev100-v2}" \
  --host 127.0.0.1 --port "${EVAL_PORT:-9001}"
