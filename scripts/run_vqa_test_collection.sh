#!/usr/bin/env bash

# Run the prepared VQA collection through the public DatasetRunner entry point.
# 通过公开 DatasetRunner 入口运行已准备的 VQA 评测集。

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
COLLECTION_ROOT="${VQA_TEST_ROOT:-$REPO_ROOT/data/20260830_232720_VQA_test}"
CONFIG="${M3_CONFIG:-$REPO_ROOT/configs/local.yaml}"
RUN_PREFIX="${VQA_TEST_RUN_PREFIX:-vqa-test-$(date +%Y%m%d-%H%M%S)}"
LIMIT="${VQA_TEST_LIMIT:-}"
START_INDEX="${VQA_TEST_START_INDEX:-0}"
SAMPLE_CONCURRENCY="${VQA_TEST_SAMPLE_CONCURRENCY:-1}"
SHARD_INDEX="${VQA_TEST_SHARD_INDEX:-0}"
SHARD_COUNT="${VQA_TEST_SHARD_COUNT:-1}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_vqa_test_collection.sh

Environment overrides:
  M3_CONFIG                    Settings YAML (default: configs/local.yaml)
  VQA_TEST_ROOT                Prepared collection directory
  VQA_TEST_RUN_PREFIX          Prefix for the two run ids
  VQA_TEST_LIMIT               Optional per-dataset sample limit
  VQA_TEST_START_INDEX         Stable source-order start index (default: 0)
  VQA_TEST_SAMPLE_CONCURRENCY  Concurrent samples (default: 1)
  VQA_TEST_SHARD_INDEX         Stable shard index (default: 0)
  VQA_TEST_SHARD_COUNT         Stable shard count (default: 1)
  PYTHON                       Python executable (default: python3)
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

# Resolve the declared image roots and verify that the exact prepared
# annotations still match their source releases. No data is copied or changed.
# 解析 manifest 声明的图片根目录，并校验已准备标注仍与源发布一致；
# 全过程不复制、不修改数据。
roots="$($PYTHON - "$COLLECTION_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

collection = Path(sys.argv[1]).expanduser().resolve()


def load_manifest(name: str) -> tuple[Path, dict]:
    dataset_dir = collection / name
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    return dataset_dir, json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_digest(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"annotation checksum mismatch: {path}")


vrs_dir, vrs_manifest = load_manifest("VRSBench")
vrs_annotation = vrs_dir / vrs_manifest["files"]["aggregate_annotation"]
verify_digest(vrs_annotation, vrs_manifest["files"]["aggregate_annotation_sha256"])
vrs_root = (vrs_dir / vrs_manifest["files"]["image_reference_root"]).resolve()
source_vrs_annotation = vrs_root.parent / vrs_annotation.name
verify_digest(source_vrs_annotation, vrs_manifest["files"]["aggregate_annotation_sha256"])

mme_dir, mme_manifest = load_manifest("MME-RealWorld")
mme_annotation = mme_dir / mme_manifest["files"]["annotation"]
verify_digest(mme_annotation, mme_manifest["files"]["annotation_sha256"])
mme_root = (mme_dir / mme_manifest["files"]["image_reference_root"]).resolve()
source_mme_annotation = mme_root / "MME_RealWorld.json"
source_rows = json.loads(source_mme_annotation.read_text(encoding="utf-8"))
prepared_rows = json.loads(mme_annotation.read_text(encoding="utf-8"))
selected_rows = [row for row in source_rows if row.get("Subtask") == "Remote Sensing"]
if selected_rows != prepared_rows:
    raise SystemExit("prepared MME-RealWorld selection differs from its source release")

if not vrs_root.is_dir() or not mme_root.is_dir():
    raise SystemExit("one or more manifest image roots are missing")
print(vrs_root.parent)
print(mme_root)
PY
)"
mapfile -t dataset_roots <<<"$roots"
if [[ ${#dataset_roots[@]} -ne 2 ]]; then
  printf 'failed to resolve exactly two dataset roots\n' >&2
  exit 2
fi

common_args=(
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

# An explicit source task defines only the run namespace. Fresh samples still
# receive exactly one visual-task-plan-v5 call; its validated task controls
# routing, and VQA tasks route to general_vqa_agent.
# 显式 source task 只定义 run namespace。fresh 样本仍恰好调用一次
# visual-task-plan-v5；其校验后的 task 决定路由，VQA 任务进入
# general_vqa_agent。
"$PYTHON" "$REPO_ROOT/main.py" --config "$CONFIG" run-dataset \
  --dataset VRSBench \
  --root "${dataset_roots[0]}" \
  --split validation \
  --task general_vqa \
  --run-id "${RUN_PREFIX}-vrsbench" \
  "${common_args[@]}"

"$PYTHON" "$REPO_ROOT/main.py" --config "$CONFIG" run-dataset \
  --dataset MME-RealWorld \
  --root "${dataset_roots[1]}" \
  --split test \
  --task multiple_choice_vqa \
  --run-id "${RUN_PREFIX}-mme-realworld" \
  "${common_args[@]}"
