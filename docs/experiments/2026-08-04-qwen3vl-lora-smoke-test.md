# Experiment Record: Qwen3-VL-4B LoRA Smoke Test on Remote Server

## Time

2026-08-04 00:16:22 CST (+0800)（训练、验证与导出完成时刻）

## Dataset

服务器 M3 仓库的 `data/merged`（VRSBench + LEVIR-CC 合并 ShareGPT 数据集）；
冒烟测试用 `--max-samples 3` 截取 train/val 各 3 条，仅用于链路验证，不产生
有效训练结论。

## Model

服务器 M3 仓库的 `models/Qwen3-VL-4B-Instruct`（本地完整权重快照，
`Qwen3VLForConditionalGeneration`，总参数 4,437,815,808，原始权重未修改）。

## Configuration File

运行时生成：

- 服务器 M3 仓库的 `.cache/llamafactory/dataset_info.json`
- 服务器 M3 仓库的 `.cache/llamafactory/train_qwen3-vl-4b.yaml`
- 服务器 M3 仓库的 `.cache/llamafactory/export_qwen3-vl-4b.yaml`

关键参数：`template: qwen3_vl`、`trust_remote_code: false`、
`finetuning_type: lora`、`lora_target: all`、`freeze_vision_tower: true`、
`freeze_multi_modal_projector: true`、`lora_rank 16 / alpha 32`、
`per_device_train_batch_size 2`、`gradient_accumulation_steps 16`、
`learning_rate 1e-4`、`bf16`、`gradient_checkpointing`、
`save_steps 1 / eval_steps 1 / logging_steps 1`、`max_samples 3`。

## Run Command

```bash
cd <server-repo>
nohup <server-python> scripts/finetune_vlm_lora.py \
  --model qwen3-vl-4b \
  --model-name-or-path <server-repo>/models/Qwen3-VL-4B-Instruct \
  --data-dir <server-repo>/data/merged \
  --output-dir <server-repo>/outputs/smoke/qwen3vl_lora_smoke1 \
  --max-samples 3 --save-steps 1 --eval-steps 1 --logging-steps 1 \
  --save-total-limit 2 \
  --llamafactory-cli <server-env>/bin/llamafactory-cli \
  > <server-repo>/outputs/smoke/qwen3vl_smoke1.log 2>&1 &
```

## Metric Results

- 训练 1 步（max_samples=3，等效 batch 32）：train loss 11.1347。
- 验证：eval loss 8.1542（1 步结果，不代表真实效果）。
- 最佳 checkpoint：`checkpoint-1`（按最低 `eval_loss` 自动选择）。

## Resource Consumption

- GPU：NVIDIA GeForce RTX 4090（48GB），训练使用 bf16 + gradient
  checkpointing；未记录峰值显存。
- 磁盘：`checkpoint-1/` 约 390MB；最佳 LoRA `best_lora/` 约 127MB；
  合并导出 `merged/` 2 个分片约 8.3GB；曲线 PNG 46KB。
- 合并导出使用 `export_device: cpu`；合并模型可用
  `AutoModelForImageTextToText` 加载为 `Qwen3VLForConditionalGeneration`
  （`Qwen3VLProcessor` 可正常加载）。

## Conclusion

`scripts/finetune_vlm_lora.py` 在 Qwen3-VL 远端服务器上完成端到端冒烟验证：
数据注册、LoRA 训练（1 步）、验证、最佳 checkpoint 选择、LoRA 适配器导出、
基座+LoRA 合并导出、训练曲线绘制与断点状态记录全部成功。相同命令重跑时，
“训练已完成/最佳 LoRA 已导出/合并模型已导出”均正确跳过，仅重绘训练曲线，
断点语义符合预期。

## Reproducibility Statement

本次冒烟产物位于服务器
`<server-repo>/outputs/smoke/qwen3vl_lora_smoke1/`
（状态文件 `.finetune_state.json`、`trainer_log.jsonl`、`all_results.json`、
`best_lora/`、`merged/`、`train_curves.png`），不在 Git 仓库内。运行环境为
服务器 `py311` conda 环境（Python 3.11.15、LLaMA-Factory 0.9.5、
torch 2.13.0+cu130、transformers 5.6.0）。冒烟仅验证链路，不代表正式微调
结果；完整训练需按 `scripts/finetune_vlm_lora.py --help` 的参数在正式输出
目录运行。服务器地址与凭据不写入仓库。
