#!/usr/bin/env bash

# Run the VQA evaluation with timestamped system and model-call GPU telemetry.
# 运行 VQA 评测，并记录带时间戳的系统级与模型调用级 GPU 遥测。

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RUN_PREFIX="${VQA_TEST_RUN_PREFIX:-vqa-test-$(date +%Y%m%d-%H%M%S)}"
MONITOR_ROOT="${VQA_GPU_MONITOR_ROOT:-$REPO_ROOT/outputs/gpu-monitor/$RUN_PREFIX}"
MONITOR_INTERVAL="${VQA_GPU_MONITOR_INTERVAL:-1}"
PROGRESS_LOG="$MONITOR_ROOT/evaluation.log"
GPU_CSV="$MONITOR_ROOT/gpu_timeline.csv"
SESSION_JSON="$MONITOR_ROOT/session.json"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_vqa_test_with_gpu_monitor.sh

This wrapper runs run_vqa_test_collection.sh and records:
  gpu_timeline.csv   Periodic nvidia-smi memory/utilization snapshots
  model_calls.jsonl  YOLO/SegFormer before/after CUDA memory events
  evaluation.log     Evaluation stdout/stderr
  session.json       Start/end timestamps and exit status

Environment overrides:
  VQA_GPU_MONITOR_ROOT      Output directory
  VQA_GPU_MONITOR_INTERVAL  Sampling interval in seconds (default: 1)
  VQA_TEST_RUN_PREFIX       Shared run/monitor identity
  PYTHON                    Project Python executable

All VQA_TEST_* and M3_CONFIG options accepted by the evaluation script are
forwarded through the environment.
EOF
  exit 0
fi

if [[ $# -ne 0 ]]; then
  printf 'unexpected positional arguments; use --help\n' >&2
  exit 2
fi

if ! "$PYTHON" - "$MONITOR_INTERVAL" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
then
  printf 'VQA_GPU_MONITOR_INTERVAL must be a positive finite number\n' >&2
  exit 2
fi

mkdir -p "$MONITOR_ROOT"

started_at="$(date --iso-8601=seconds)"
monitor_pid=""

stop_monitor() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap stop_monitor EXIT INT TERM

"$PYTHON" "$REPO_ROOT/scripts/gpu_memory_monitor.py" \
  --interval "$MONITOR_INTERVAL" \
  --progress-log "$PROGRESS_LOG" \
  --output "$GPU_CSV" &
monitor_pid=$!

set +e
M3_GPU_MONITOR=1 \
M3_GPU_MONITOR_DIR="$MONITOR_ROOT" \
VQA_TEST_RUN_PREFIX="$RUN_PREFIX" \
bash "$REPO_ROOT/scripts/run_vqa_test_collection.sh" \
  > >(tee "$PROGRESS_LOG") 2>&1
evaluation_status=$?
set -e

stop_monitor
monitor_pid=""
finished_at="$(date --iso-8601=seconds)"

"$PYTHON" - "$SESSION_JSON" "$RUN_PREFIX" "$started_at" "$finished_at" \
  "$evaluation_status" "$MONITOR_INTERVAL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "run_prefix": sys.argv[2],
    "started_at": sys.argv[3],
    "finished_at": sys.argv[4],
    "evaluation_exit_code": int(sys.argv[5]),
    "gpu_sampling_interval_seconds": float(sys.argv[6]),
    "artifacts": {
        "gpu_timeline": "gpu_timeline.csv",
        "model_call_events": "model_calls.jsonl",
        "evaluation_log": "evaluation.log",
    },
}
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

printf 'GPU monitoring artifacts: %s\n' "$MONITOR_ROOT"
exit "$evaluation_status"
