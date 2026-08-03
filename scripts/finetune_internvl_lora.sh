#!/usr/bin/env bash
# ============================================================
# LoRA fine-tuning of InternVL with LLaMA-Factory.
# 使用 LLaMA-Factory 对 InternVL 进行 LoRA 参数微调。
#
# Usage:
#   bash scripts/finetune_internvl_lora.sh \
#     --model /home/lijia/models/InternVL3_5-8B \
#     [--data-dir /path/to/merged] [--output-dir /path/out] \
#     [--max-samples 200] [--llamafactory-cli "llamafactory-cli"]
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/configs/llamafactory}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/llamafactory}"
TRAIN_YAML="$CONFIG_DIR/train_internvl_lora.yaml"
DATASET_INFO="$CONFIG_DIR/dataset_info.json"

# Overridable paths. 可通过命令行参数或环境变量覆盖的路径。
MODEL_DIR="${MODEL_DIR:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/微调数据集/merged}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/finetune/internvl_lora}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-}"

usage() {
    cat <<'EOF'
Usage: bash scripts/finetune_internvl_lora.sh --model <MODEL_DIR> [options]

Options:
  --model <DIR>          InternVL model directory (required)  模型目录（必填）
  --data-dir <DIR>       merged dataset root (default: <repo>/data/微调数据集/merged)
                         合并数据集根目录
  --output-dir <DIR>     LoRA output dir (default: <repo>/outputs/finetune/internvl_lora)
                         输出目录
  --max-samples <N>      limit training samples for a smoke run (default: unlimited)
                         冒烟测试样本数上限
  --llamafactory-cli <S> LLaMA-Factory CLI command (default: auto-detect)
                         llamafactory-cli 命令（默认自动探测）
  -h, --help             show this help  显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_DIR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --llamafactory-cli) LLAMAFACTORY_CLI="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Validate inputs before launching training. 启动训练前校验输入。
if [[ -z "$MODEL_DIR" ]]; then
    echo "ERROR: --model is required (e.g. --model /home/lijia/models/InternVL3_5-8B)" >&2
    usage
    exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory not found: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! -f "$DATA_DIR/train.json" ]]; then
    echo "ERROR: train.json not found under data dir: $DATA_DIR" >&2
    exit 1
fi

# Locate the LLaMA-Factory CLI. 定位 llamafactory-cli 可执行文件。
if [[ -z "$LLAMAFACTORY_CLI" ]]; then
    if command -v llamafactory-cli >/dev/null 2>&1; then
        LLAMAFACTORY_CLI="llamafactory-cli"
    elif python3 -c "import llamafactory" >/dev/null 2>&1; then
        LLAMAFACTORY_CLI="python3 -m llamafactory.cli"
    else
        echo "ERROR: llamafactory-cli not found. Install it or pass --llamafactory-cli." >&2
        echo "  Hint: pip install llamafactory, or activate the conda env that has it first." >&2
        echo "  提示：请先安装 llamafactory 或激活包含它的 conda 环境。" >&2
        exit 1
    fi
fi

# Warn early if TensorBoard is missing so curve logging does not fail later.
# 若缺少 TensorBoard 提前提示，避免训练启动后才报错。
if ! python3 -c "import tensorboard" >/dev/null 2>&1; then
    echo "WARNING: tensorboard is not importable in the current python3, but"
    echo "         train_internvl_lora.yaml sets report_to: tensorboard."
    echo "         Install it with: pip install tensorboard"
    echo "         警告：当前环境缺少 tensorboard，训练配置启用了曲线记录，请先安装。"
fi

# Regenerate dataset_info.json with an absolute media_dir so that the
# relative image paths in train/val/test.json resolve wherever the data lives.
# 重新生成 dataset_info.json：media_dir 改写为绝对路径，确保相对图像路径始终可解析。
mkdir -p "$CACHE_DIR"
sed "s|\"media_dir\": \"[^\"]*\"|\"media_dir\": \"$DATA_DIR\"|" \
    "$DATASET_INFO" > "$CACHE_DIR/dataset_info.json"

# Assemble the training command. 组装训练命令。
train_args=(
    train "$TRAIN_YAML"
    --model_name_or_path "$MODEL_DIR"
    --dataset_dir "$CACHE_DIR"
    --output_dir "$OUTPUT_DIR"
)
if [[ -n "$MAX_SAMPLES" ]]; then
    train_args+=(--max_samples "$MAX_SAMPLES")
fi

mkdir -p "$OUTPUT_DIR"
# LLAMAFACTORY_CLI may contain arguments, so expand it unquoted on purpose.
# LLAMAFACTORY_CLI 可能包含参数，此处有意不做引号包裹地展开。
# shellcheck disable=SC2086
exec $LLAMAFACTORY_CLI "${train_args[@]}"
