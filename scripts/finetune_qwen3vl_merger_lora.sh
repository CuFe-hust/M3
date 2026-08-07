#!/usr/bin/env bash
# Qwen3-VL-8B merger-layer LoRA SFT training entry for a single 4090/48G node.
# 单节点 4090/48G 的 Qwen3-VL-8B merger 层 LoRA SFT 训练入口。
#
# Override paths through environment variables instead of editing this file:
#   DATA_ROOT=/data/vrsbench MODEL_ID=/path/to/Qwen3-VL-8B-Instruct \
#   OUTPUT_DIR=outputs/finetune/qwen3-vl-8b-merger-lora bash scripts/finetune_qwen3vl_merger_lora.sh
# 可通过环境变量覆盖路径，无需修改本文件：
#   DATA_ROOT=/data/vrsbench MODEL_ID=/path/to/Qwen3-VL-8B-Instruct \
#   OUTPUT_DIR=outputs/finetune/qwen3-vl-8b-merger-lora bash scripts/finetune_qwen3vl_merger_lora.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/vrsbench}"
MODEL_ID="${MODEL_ID:-/home/lijia/M3/models/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/finetune/qwen3-vl-8b-merger-lora}"
PYTHON="${PYTHON:-python}"

SFT_DIR="${OUTPUT_DIR}/data"
mkdir -p "${SFT_DIR}"

# Convert local/remote JSONL annotations into the reference SFT JSON format.
# English train records are written twice to emphasize English; val stays balanced.
# 将本地/远端 JSONL 标注转换为参考 SFT JSON 格式。
# 英文训练记录写两遍以侧重英文；验证集保持均衡。
"${PYTHON}" scripts/prepare_vrsbench_sft.py \
  --root "${DATA_ROOT}" \
  --output-dir "${SFT_DIR}" \
  --languages en zh \
  --tasks caption vqa \
  --splits train val \
  --english-multiplier 2 \
  --seed 42

# Pure LoRA on the four merger modules; every other weight stays frozen.
# Hyperparameters are tuned for the ~148k-record VRSBench SFT set on one
# 4090/48G: rank 32 / alpha 64 (only eight target linears, so capacity, not
# overfitting, is the bottleneck); eval/save every 250 of ~4.6k steps per epoch.
# 对四个 merger 模块做纯 LoRA，其余权重全部冻结。
# 超参按约 14.8 万条 VRSBench SFT 记录、单卡 4090/48G 调优：rank 32 / alpha 64
# （目标线性层仅八个，瓶颈是容量而非过拟合）；每 epoch 约 4600 步，每 250 步验证/保存。
"${PYTHON}" scripts/finetune_qwen3vl_merger_lora.py \
  --model_id "${MODEL_ID}" \
  --local_files_only True \
  --train_file "${SFT_DIR}/vrsbench_sft_train.json" \
  --eval_file "${SFT_DIR}/vrsbench_sft_val.json" \
  --image_folder "${DATA_ROOT}" \
  --image_min_pixels $((256 * 32 * 32)) \
  --image_max_pixels $((1280 * 32 * 32)) \
  --max_seq_length 4096 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --freeze_merger_base True \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --fp16 False \
  --gradient_checkpointing True \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 3 \
  --eval_strategy steps \
  --eval_steps 250 \
  --report_to tensorboard \
  --remove_unused_columns False \
  --dataloader_num_workers 4

echo "Training finished. LoRA adapter is at ${OUTPUT_DIR}"
echo "训练完成。LoRA 适配器位于 ${OUTPUT_DIR}"
