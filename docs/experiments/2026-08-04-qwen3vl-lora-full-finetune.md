# Experiment Record: Qwen3-VL-4B Full-Dataset LoRA Fine-Tuning

## Time

2026-08-04 03:03:38 CST (+0800)（训练、验证与导出完成时刻；训练于
2026-08-04 00:19 启动）

## Dataset

服务器 M3 仓库的 `data/merged`（VRSBench + LEVIR-CC 合并 ShareGPT 数据集）：
train 60,262 条、val 7,453 条，全部样本参与训练，未做样本截取。

## Model

服务器 M3 仓库的 `models/Qwen3-VL-4B-Instruct`（本地完整权重快照，
`Qwen3VLForConditionalGeneration`，总参数 4,437,815,808，原始权重未修改）。

## Configuration File

运行时生成：

- 服务器 M3 仓库的 `.cache/llamafactory/dataset_info.json`
- 服务器 M3 仓库的 `.cache/llamafactory/train_qwen3-vl-4b.yaml`
- 服务器 M3 仓库的 `.cache/llamafactory/export_qwen3-vl-4b.yaml`

关键参数（脚本默认值，未传 `--max-samples`）：`template: qwen3_vl`、
`trust_remote_code: false`、`finetuning_type: lora`、`lora_target: all`、
`freeze_vision_tower: true`、`freeze_multi_modal_projector: true`、
`lora_rank 16 / alpha 32 / dropout 0.05`、
`per_device_train_batch_size 2`、`gradient_accumulation_steps 16`、
`learning_rate 1e-4`、`num_train_epochs 1`、`cutoff_len 2048`、
`cosine` 调度 + `warmup_ratio 0.03`、`bf16`、`gradient_checkpointing`、
`save_steps 1000 / eval_steps 1000 / logging_steps 10`、
`save_total_limit 2`。

## Run Command

```bash
cd <server-repo>
nohup <server-python> scripts/finetune_vlm_lora.py \
  --model qwen3-vl-4b \
  --model-name-or-path <server-repo>/models/Qwen3-VL-4B-Instruct \
  --data-dir <server-repo>/data/merged \
  --llamafactory-cli <server-env>/bin/llamafactory-cli \
  > <server-repo>/outputs/finetune/qwen3vl_full_train.log 2>&1 &
```

输出目录为脚本默认的
`<server-repo>/outputs/finetune/qwen3-vl-4b_lora/`。

## Metric Results

- 总步数：1,884 步（等效 batch 32，train 60,262 条）。
- train loss：**0.6109**（全程平均，train_runtime 8,927s ≈ 2h28m46s）。
- eval loss：**0.5025**（val 7,453 条，eval_runtime 364s；
  该值来自 LLaMA-Factory 验证损失，不代表官方任务指标）。
- 最佳 checkpoint：`checkpoint-1884`（按最低 `eval_loss` 自动选择；
  `checkpoint-1000` 保留作为中间断点）。

## Resource Consumption

- GPU：NVIDIA GeForce RTX 4090（48GB），训练使用 bf16 + gradient
  checkpointing，训练期间显存约 17–20GB，峰值未单独记录。
- 磁盘：`checkpoint-1000/` 与 `checkpoint-1884/` 各约 390MB；
  最佳 LoRA `best_lora/` 约 127MB；合并导出 `merged/` 2 个分片约 8.3GB；
  曲线 PNG 88KB；输出目录合计约 9.3GB。
- 合并导出使用 `export_device: cpu`；合并模型可用
  `AutoModelForImageTextToText` 加载为 `Qwen3VLForConditionalGeneration`
  （参数 4,437,815,808），`Qwen3VLProcessor` 可正常加载。

## Conclusion

`scripts/finetune_vlm_lora.py` 在 Qwen3-VL 远端服务器上完成全量数据集的
LoRA SFT：数据注册与 tokenize、1 epoch 训练、step 1000 与最终验证、最佳
checkpoint 选择、LoRA 适配器导出、基座+LoRA 合并导出与训练曲线绘制全部
成功，断点状态文件记录完整。

## Reproducibility Statement

本次训练产物位于服务器
`<server-repo>/outputs/finetune/qwen3-vl-4b_lora/`
（状态文件 `.finetune_state.json`、`trainer_log.jsonl`、`all_results.json`、
`checkpoint-1000/`、`checkpoint-1884/`、`best_lora/`、`merged/`、
`train_curves.png`），不在 Git 仓库内。运行环境为服务器 `py311` conda
环境（Python 3.11.15、LLaMA-Factory 0.9.5、torch 2.13.0+cu130、
transformers 5.6.0）。eval loss 仅用于 checkpoint 选择，正式效果需使用
仓库评测入口按统一评测流程验证。服务器地址与凭据不写入仓库。
