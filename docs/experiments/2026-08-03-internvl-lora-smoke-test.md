# Experiment Record: InternVL3.5-8B LoRA Smoke Test on Dedicated Server

## Time

2026-08-03 23:22:46 CST (+0800)（训练与导出完成时刻；此前 23:05–23:21 完成
三次失败排查与模型格式转换）

## Dataset

服务器 M3 仓库的 `data/merged`（VRSBench + LEVIR-CC 合并 ShareGPT 数据集）；
冒烟测试用 `--max-samples 3` 截取 train/val 各 3 条，仅用于链路验证，不产生
有效训练结论。

## Model

服务器 M3 仓库的
`models/OpenGVLab--InternVL3_5-8B-HF/snapshots/master`
（由服务器上原有 GitHub 格式快照按官方键名映射转换而来，
`InternVLForConditionalGeneration`，总参数 8,528,318,464）。

## Configuration File

运行时生成：

- 服务器 M3 仓库的 `.cache/llamafactory/dataset_info.json`
- 服务器 M3 仓库的 `.cache/llamafactory/train_internvl3.5-8b.yaml`
- 服务器 M3 仓库的 `.cache/llamafactory/export_internvl3.5-8b.yaml`

关键参数：`template: intern_vl`、`trust_remote_code: true`、
`finetuning_type: lora`、`lora_target` 为 LLM 注意力/MLP 投影层、
`lora_rank 16 / alpha 32`、`per_device_train_batch_size 2`、
`gradient_accumulation_steps 16`、`learning_rate 1e-4`、`bf16`、
`save_steps 1 / eval_steps 1 / logging_steps 1`、`max_samples 3`。

## Run Command

```bash
cd <server-repo>
nohup <server-python> scripts/finetune_vlm_lora.py \
  --model internvl3.5-8b --server \
  --model-name-or-path <server-repo>/models/OpenGVLab--InternVL3_5-8B-HF/snapshots/master \
  --data-dir <server-repo>/data/merged \
  --output-dir <server-repo>/outputs/smoke/internvl_lora_smoke5 \
  --max-samples 3 --save-steps 1 --eval-steps 1 --logging-steps 1 \
  --save-total-limit 2 \
  --llamafactory-cli <server-env>/bin/llamafactory-cli \
  > outputs/smoke/smoke_run5.log 2>&1 &
```

## Metric Results

- 训练 1 步（max_samples=3，等效 batch 32）：train loss 13.2499。
- 验证：eval loss 9.8823（1 步结果，不代表真实效果）。
- 最佳 checkpoint：`checkpoint-1`（按最低 `eval_loss` 自动选择）。

## Resource Consumption

- GPU：NVIDIA GeForce RTX 4090（48GB），训练使用 bf16 + gradient
  checkpointing；未记录峰值显存。
- 磁盘：HF 格式模型目录约 16.6GB；合并导出 `merged/` 4 个分片约 16.6GB；
  最佳 LoRA `best_lora/` 约 167MB；曲线 PNG 47KB。
- 合并导出使用 `export_device: cpu`。

## Conclusion

`scripts/finetune_vlm_lora.py` 在专用服务器上完成端到端冒烟验证：数据注册、
LoRA 训练（1 步）、最佳 checkpoint 选择、LoRA 适配器导出、基座+LoRA 合并
导出、训练曲线绘制与断点状态记录全部成功。期间修复了三个环境/配置问题：
模板应为 `intern_vl`、tokenizer 需补 HF 格式特殊 token 配置、训练配置需顶层
`media_dir`，并将 GitHub 格式权重转换为 LLaMA-Factory 0.9.5 要求的 HF 兼容
格式。

## Reproducibility Statement

本次冒烟产物位于服务器
`<server-repo>/outputs/smoke/internvl_lora_smoke5/`
（状态文件 `.finetune_state.json`、`trainer_log.jsonl`、`all_results.json`、
`best_lora/`、`merged/`、`train_curves.png`），不在 Git 仓库内。模型格式转换
命令与失败排查记录见
`docs/changes/2026-08-03-troy-fix-internvl-smoke-issues.md`。冒烟仅验证链路，
不代表正式微调结果；完整训练需按 `scripts/finetune_vlm_lora.py --help`
的参数在正式输出目录运行。
