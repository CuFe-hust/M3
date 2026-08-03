# Modification Note: Remove LLaMA-Factory LoRA Fine-Tuning Scripts - 2026-08-03 22:41:21 CST

## Modification Time

2026-08-03 22:41:21 CST (+0800)

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Remove the current LLaMA-Factory based LoRA parameter fine-tuning entry points
from the repository. The two launcher scripts and their exclusively-used
LLaMA-Factory configs are deleted so the repo no longer carries a runnable
LLaMA-Factory LoRA fine-tune/export workflow.

## Modified Files

- `scripts/finetune_internvl_lora.sh`（删除）：LLaMA-Factory LoRA 微调启动脚本
  （`llamafactory-cli train`）。
- `scripts/export_internvl_lora.sh`（删除）：LLaMA-Factory LoRA 合并导出脚本
  （`llamafactory-cli export`）。
- `configs/llamafactory/dataset_info.json`（删除）：注册
  `merged_train` / `merged_val` / `merged_test` 的 ShareGPT 数据集配置。
- `configs/llamafactory/train_internvl_lora.yaml`（删除）：InternVL3.5-8B
  SFT + LoRA 训练配置。
- `configs/llamafactory/export_internvl_lora.yaml`（删除）：基座 + LoRA
  合并导出配置。
- `DETAILS.md`（更新）：第 15 节去掉 LLaMA-Factory 专属表述，删除 15.1
  小节（微调入口、训练曲线与合并导出说明）。

## Core Changes

- 删除全部调用 `llamafactory-cli` 的脚本（训练与导出），以及仅被这两个脚本
  使用的 `configs/llamafactory/` 目录（`dataset_info.json` 与两个 YAML）。
- `DETAILS.md` 第 15 节仅保留 `data/微调数据集/` 数据目录的说明（ShareGPT
  格式、合并拆分、路径约定与溯源），不再引用 LLaMA-Factory 或已删除的脚本。
- 保留 `scripts/plot_train_curves.py`：它是独立的训练曲线绘图工具，不调用
  LLaMA-Factory，不参与 LoRA 微调，本次不删除。

## Whether the Canonical Sample Format Was Changed

No. `data/schema.py` 的 `CanonicalSample` / `CanonicalPrediction` 未改动。

## Whether the Model Interface Was Changed

No. 未改动 `models/` 或任何加载逻辑；删除的是 LLaMA-Factory 外部微调入口，
不涉及模型加载接口。

## Whether the Configuration Was Changed

Yes. 删除 `configs/llamafactory/` 下全部 LLaMA-Factory 配置；仓库其余配置
（`configs/default.yaml` 等）未改动。

## Whether Evaluation Was Affected

No. `eval/`、指标、数据集划分与参考答案读取逻辑未改动；微调数据
`data/微调数据集/` 保留。

## Whether Deployment Was Affected

No. `deploy/` 未涉及；LoRA 合并导出脚本删除后不再通过 LLaMA-Factory 生成
合并模型。

## Whether pytest Was Updated

No. `tests/` 中没有任何测试引用被删除的脚本或配置（已用 `rg` 确认）。

## Whether .gitignore Was Updated

No. `data/微调数据集/`、`.cache/`、`outputs/` 等忽略规则仍适用于现存目录，
无需改动。

## Validation Method

- `rg -n -i "llamafactory|finetune_internvl|export_internvl"` 确认除
  `docs/changes/` 历史记录外，仓库其余文件（`main.py`、`README.md`、
  `configs/default.yaml`、`tests/`、`requirements.txt`）已无残留引用。
- `git status --short` 确认 5 个目标文件均处于 deleted 状态。
- 未运行 `pytest`：本次仅删除脚本与配置，无 Python 代码改动，相关测试不受
  影响。

## Risks and Follow-up TODOs

- 若后续仍需要 LLaMA-Factory LoRA 微调能力，可从 Git 历史恢复本次删除的
  文件（上次新增提交 `9447d3e`）。
- `data/微调数据集/merged/` 仍为 ShareGPT 格式，后续若采用其他微调框架，
  需按新框架重新编写入口并同步 `DETAILS.md`。
