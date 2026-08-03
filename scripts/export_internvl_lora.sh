#!/usr/bin/env bash
# ============================================================
# Export the merged base + LoRA model with LLaMA-Factory.
# 使用 LLaMA-Factory 导出基座 + LoRA 合并后的完整模型。
#
# Usage:
#   bash scripts/export_internvl_lora.sh \
#     --model /home/lijia/models/InternVL3_5-8B \
#     --adapter /path/to/internvl_lora \
#     [--output-dir /path/to/internvl_lora_merged]
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/configs/llamafactory}"
EXPORT_YAML="$CONFIG_DIR/export_internvl_lora.yaml"

# Overridable paths. 可通过命令行参数或环境变量覆盖的路径。
MODEL_DIR="${MODEL_DIR:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
EXPORT_DIR="${EXPORT_DIR:-$PROJECT_ROOT/outputs/finetune/internvl_lora_merged}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-}"

usage() {
    cat <<'EOF'
Usage: bash scripts/export_internvl_lora.sh --model <MODEL_DIR> --adapter <ADAPTER_DIR> [options]

Options:
  --model <DIR>          base InternVL model directory (required)  基座模型目录（必填）
  --adapter <DIR>        LoRA adapter directory from training (required)  LoRA 适配器目录（必填）
  --output-dir <DIR>     merged full-model dir (default: <repo>/outputs/finetune/internvl_lora_merged)
                         合并后完整模型输出目录
  --llamafactory-cli <S> LLaMA-Factory CLI command (default: auto-detect)
                         llamafactory-cli 命令（默认自动探测）
  -h, --help             show this help  显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_DIR="$2"; shift 2 ;;
        --adapter) ADAPTER_DIR="$2"; shift 2 ;;
        --output-dir) EXPORT_DIR="$2"; shift 2 ;;
        --llamafactory-cli) LLAMAFACTORY_CLI="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Validate inputs before exporting. 导出前校验输入。
if [[ -z "$MODEL_DIR" || -z "$ADAPTER_DIR" ]]; then
    echo "ERROR: both --model and --adapter are required" >&2
    usage
    exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory not found: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "ERROR: adapter_config.json not found under: $ADAPTER_DIR" >&2
    echo "  Hint: point --adapter at the LLaMA-Factory training output_dir.  提示：指向训练输出目录。" >&2
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

mkdir -p "$EXPORT_DIR"
# LLAMAFACTORY_CLI may contain arguments, so expand it unquoted on purpose.
# LLAMAFACTORY_CLI 可能包含参数，此处有意不做引号包裹地展开。
# shellcheck disable=SC2086
exec $LLAMAFACTORY_CLI export "$EXPORT_YAML" \
    --model_name_or_path "$MODEL_DIR" \
    --adapter_name_or_path "$ADAPTER_DIR" \
    --export_dir "$EXPORT_DIR"
