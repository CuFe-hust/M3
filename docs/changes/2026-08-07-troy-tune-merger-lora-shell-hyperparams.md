# Modification Note: Tune Merger-LoRA Shell Hyperparameters for the VRSBench SFT Set - 2026-08-07 10:41:38 CST

## Modification Time

2026-08-07 10:41:38 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Retune the default training hyperparameters in
`scripts/finetune_qwen3vl_merger_lora.sh` for the actual VRSBench SFT data
volume (~148k train records after the 2x English multiplier: 18,237 caption +
31,054 VQA records per language). With only eight merger linear layers as LoRA
targets, capacity rather than overfitting is the bottleneck, and one epoch is
only ~4.6k optimizer steps at effective batch 32, so the previous eval/save
cadence was too sparse to read the loss curve.
按实际 VRSBench SFT 数据量（英文 2 倍复制后约 14.8 万条训练记录：每语言
caption 18,237 条、VQA 31,054 条）重新调整
`scripts/finetune_qwen3vl_merger_lora.sh` 的默认训练超参。LoRA 目标仅八个
merger 线性层，瓶颈是容量而非过拟合；有效 batch 32 时一个 epoch 约 4600 步，
原验证/保存间隔过疏，loss 曲线点数不足。

## Modified Files

- `scripts/finetune_qwen3vl_merger_lora.sh`

## Core Changes

- `--lora_rank` 16 -> 32 and `--lora_alpha` 32 -> 64, keeping the alpha/rank
  ratio at 2. The eight target linears make capacity the limiting factor on a
  148k-record set; dropout stays 0.05 and `--freeze_merger_base True` is
  unchanged, so the run remains pure LoRA.
- `--save_steps` and `--eval_steps` 500 -> 250 so each ~4.6k-step epoch yields
  ~18 eval points instead of ~9; `--save_total_limit 3` is unchanged.
- Added a bilingual comment above the training invocation recording the tuning
  rationale. All other parameters (batch 2 x accumulation 16, lr 1e-4 cosine
  with 3% warmup, weight decay 0.05, 1 epoch, bf16, gradient checkpointing,
  pixel limits 256..1280 x 32 x 32, max_seq_length 4096, 2x English
  multiplier) are intentionally left at their previous values.
- `--lora_rank` 由 16 改为 32，`--lora_alpha` 由 32 改为 64，保持
  alpha/rank = 2 的比例。目标线性层仅八个，14.8 万条数据下容量才是瓶颈；
  dropout 维持 0.05，`--freeze_merger_base True` 不变，仍为纯 LoRA 训练。
- `--save_steps` 与 `--eval_steps` 由 500 改为 250，使每 epoch（约 4600 步）
  的验证点从约 9 个增至约 18 个；`--save_total_limit 3` 不变。
- 在训练调用上方新增中英双语注释说明调参依据。其余参数（batch 2 × 累积 16、
  lr 1e-4 cosine + 3% warmup、weight decay 0.05、1 epoch、bf16、gradient
  checkpointing、像素上下限 256..1280 × 32 × 32、max_seq_length 4096、英文
  2 倍复制）均有意保持不变。

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

Yes: default training hyperparameters of the standalone shell run entry changed
(rank/alpha/save_steps/eval_steps). No configuration file schema or existing
configuration field outside this script was touched; the Python script's CLI
defaults are unchanged.
是：独立 shell 运行入口的默认训练超参变化（rank/alpha/save_steps/eval_steps）。
未触及任何配置文件模式或脚本之外的既有配置字段；Python 脚本的 CLI 默认值不变。

## Whether Evaluation Was Affected

No. No metric, dataset split, reference-answer reading, or `eval/` logic was
touched. The SFT validation set produced by `prepare_vrsbench_sft.py` is only
used for training-time loss monitoring, as before.

## Whether Deployment Was Affected

No. The merge script and the merged-checkpoint loading path are unchanged.

## Whether pytest Was Updated

No. Only shell hyperparameter values changed; no Python code or argument
parsing was modified, so the existing tests
(`tests/test_finetune_qwen3vl_merger_lora.py`,
`tests/test_prepare_vrsbench_sft.py`) remain valid without changes.

## Whether .gitignore Was Updated

No. No new file types are generated.

## Validation Method

- `bash -n scripts/finetune_qwen3vl_merger_lora.sh` passed.
- Record counts were verified by `wc -l` on the eight local
  `datasets/vrsbench/VRSBench_*` JSONL files (train caption 18,237, train VQA
  31,054 per language; val 2,027 caption / 3,416 VQA per language).
- No GPU training run was performed in this environment; the numbers are
  tuning defaults, not measured results.

## Risks and Follow-up TODOs

- Rank 32 doubles the LoRA parameter count versus rank 16; monitor GPU memory
  and val loss on the first real run. If val loss plateaus, next levers are a
  second epoch or `--freeze_merger_base False`, not a blind rank increase.
- If high-resolution images push sequences against `--max_seq_length 4096`,
  lower `--image_max_pixels` before raising the sequence limit on a 48G card.
- No real GPU training was run in this environment; hyperparameters remain
  unvalidated beyond the static syntax check.
