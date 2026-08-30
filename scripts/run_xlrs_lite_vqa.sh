#!/usr/bin/env bash

# Run the local sharded XLRS-Bench-lite VQA release through DatasetRunner.
# 通过 DatasetRunner 运行本地分片版 XLRS-Bench-lite VQA 数据集。

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
DATASET_ROOT="${XLRS_VQA_ROOT:-$REPO_ROOT/data/XLRS-Bench-lite_VLM}"
CONFIG="${M3_CONFIG:-$REPO_ROOT/configs/local.yaml}"
RUN_ID="${XLRS_VQA_RUN_ID:-xlrs-lite-vqa-$(date +%Y%m%d-%H%M%S)}"
CACHE_PARENT="${XLRS_VQA_CACHE_PARENT:-${TMPDIR:-/tmp}}"
LIMIT="${XLRS_VQA_LIMIT:-}"
START_INDEX="${XLRS_VQA_START_INDEX:-0}"
SAMPLE_CONCURRENCY="${XLRS_VQA_SAMPLE_CONCURRENCY:-1}"
SHARD_INDEX="${XLRS_VQA_SHARD_INDEX:-0}"
SHARD_COUNT="${XLRS_VQA_SHARD_COUNT:-1}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_xlrs_lite_vqa.sh

Environment overrides:
  M3_CONFIG                   Settings YAML (default: configs/local.yaml)
  XLRS_VQA_ROOT               Sharded JSONL dataset directory
  XLRS_VQA_RUN_ID             Run identity
  XLRS_VQA_CACHE_PARENT       Parent for m3-xlrs-image-cache (needs about 48 GB)
  XLRS_VQA_LIMIT              Optional selected-sample limit
  XLRS_VQA_START_INDEX        Stable source-order start index (default: 0)
  XLRS_VQA_SAMPLE_CONCURRENCY Concurrent samples (default: 1)
  XLRS_VQA_SHARD_INDEX        Stable shard index (default: 0)
  XLRS_VQA_SHARD_COUNT        Stable shard count (default: 1)
  PYTHON                      Python executable (default: python3)
EOF
  exit 0
fi

if [[ $# -ne 0 ]]; then
  printf 'unexpected positional arguments; use --help\n' >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'settings file is missing: %s\n' "$CONFIG" >&2
  exit 2
fi
if ! compgen -G "$DATASET_ROOT/XLRS-Bench-lite_part*.jsonl" >/dev/null; then
  printf 'XLRS-lite JSONL partitions are missing under: %s\n' "$DATASET_ROOT" >&2
  exit 2
fi

mkdir -p "$CACHE_PARENT"
common_args=(
  --dataset XLRS-Bench-lite
  --root "$DATASET_ROOT"
  --split train
  --task multiple_choice_vqa
  --run-id "$RUN_ID"
  --evaluate
  --judge-policy none
  --start-index "$START_INDEX"
  --sample-concurrency "$SAMPLE_CONCURRENCY"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
)
if [[ -n "$LIMIT" ]]; then
  common_args+=(--limit "$LIMIT")
fi

TMPDIR="$CACHE_PARENT" "$PYTHON" "$REPO_ROOT/main.py" --config "$CONFIG" \
  run-dataset "${common_args[@]}"
