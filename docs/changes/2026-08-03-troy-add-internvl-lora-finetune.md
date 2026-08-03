# Modification Note: Add LLaMA-Factory InternVL LoRA Fine-Tuning Script - 2026-08-03 17:10:00 +0800

## Modification Time

2026-08-03 17:10:00 +0800

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Provide a runnable LLaMA-Factory entry point that LoRA-fine-tunes
`models/InternVL3_5-8B/` on the merged ShareGPT dataset
(`data/微调数据集/merged/`). The dataset and model already exist on the target
server; the launcher script takes the model/data/output paths as arguments so
no absolute server path is hard-coded.

## Modified Files

- `configs/llamafactory/dataset_info.json`（新增）：注册
  `merged_train` / `merged_val` / `merged_test` 三个 ShareGPT 数据集。
- `configs/llamafactory/train_internvl_lora.yaml`（新增）：InternVL3.5-8B 的
  SFT + LoRA 训练配置。
- `scripts/finetune_internvl_lora.sh`（新增）：启动脚本，注入模型路径、数据
  目录、输出目录，并生成绝对 `media_dir` 的 `dataset_info.json`。
- `.gitignore`（更新）：新增忽略 `.cache/`（脚本运行时生成的配置缓存）。
- `DETAILS.md`（更新）：第 15 节新增 15.1 小节，记录微调入口与运行命令。

## Core Changes

- 数据集结构为 ShareGPT 格式：`conversations`（`from`/`value`，human/gpt）+
  `images`（相对 `merged/` 的路径）；vrsbench 每条 1 图，levir-cc 每条 2 图
  （`<image>` 占位符数量与图像数一致）。`dataset_info.json` 用
  `columns: {messages: conversations, images: images}` 与
  `tags: {role_tag: from, content_tag: value, user_tag: human, assistant_tag: gpt}`
  对齐该结构。
- `train_internvl_lora.yaml`：`stage: sft` + `finetuning_type: lora`，
  LoRA 目标为语言模型注意力与 MLP 投影层（`q/v/k/o/gate/up/down_proj`），
  rank 16 / alpha 32，冻结视觉塔，bf16 + gradient checkpointing；
  `template: qwen3`（InternVL3.5 使用 Qwen 风格对话模板），
  `trust_remote_code: true`（模型目录自带 modeling 代码）。
- `scripts/finetune_internvl_lora.sh`：必填 `--model`，可选
  `--data-dir` / `--output-dir` / `--max-samples` / `--llamafactory-cli`；
  自动探测 `llamafactory-cli`（或 `python3 -m llamafactory.cli`）；用 sed 把
  模板 `dataset_info.json` 中的 `media_dir` 改写为运行时数据的绝对路径，写入
  `.cache/llamafactory/` 后通过 `--dataset_dir` 传入，保证相对图像路径在服务
  器上任意数据位置都能解析。

## Whether the Canonical Sample Format Was Changed

No. `data/schema.py` 的 `CanonicalSample` / `CanonicalPrediction` 未改动。

## Whether the Model Interface Was Changed

No. 未改动 `models/` 或任何加载逻辑；模型仍按原样加载，仅通过 LLaMA-Factory
做 LoRA 微调，原始权重未修改。

## Whether the Configuration Was Changed

新增 `configs/llamafactory/` 配置（LLaMA-Factory 使用），仓库既有配置
（`configs/default.yaml` 等）未改动。

## Whether Evaluation Was Affected

No. 未改动任何评测指标、split 或参考答案读取逻辑；`merged_test.json` 与
`*_references.json` 仍仅用于最终评测。

## Whether Deployment Was Affected

No. 仅新增训练入口；导出/部署路径不受影响。

## Whether pytest Was Updated

No. 新增内容为外部工具（LLaMA-Factory）的启动脚本与配置，不涉及仓库内部
配置解析或接口；本机完成静态校验（见 Validation Method），未新增 pytest。

## Whether .gitignore Was Updated

Yes. 新增 `.cache/`（脚本运行时生成的 `dataset_info.json` 缓存目录）。
`outputs/`、`data/微调数据集/`、`models/InternVL3_5-8B/`、`*.safetensors`
已在既有规则中忽略。

## Validation Method

- `bash -n scripts/finetune_internvl_lora.sh`：语法检查通过。
- YAML 配置用 PyYAML 解析通过；`dataset_info.json` 用 `json` 解析通过。
- 本地数据集抽查：train/val/test.json 的 ShareGPT 字段
  （`conversations`、`images`）与 `dataset_info.json` 映射一致；
  `<image>` 数量与图像数一致（此前数据集合并变更中已全量校验）。
- 用 `LLAMAFACTORY_CLI=echo` 模拟启动脚本的参数解析与
  `dataset_info.json` 生成逻辑，检查生成的 JSON 可解析、`media_dir` 为传入的
  绝对路径。
- 未执行真实训练：本机未安装 LLaMA-Factory 且无模型权重加载环境；
  LLaMA-Factory 对 InternVL3.5（`internvl_chat` + qwen3）的端到端加载与
  `template` 匹配需在服务器上以实际环境验证。

## Risks and Follow-up TODOs

- InternVL3.5 较新，需要较新版本的 LLaMA-Factory（≥ 0.9，支持
  `eval_dataset`）并确认其能通过 `trust_remote_code` 加载该模型；若加载失败，
  可尝试调整 `template`（`qwen2`）或改用官方 InternVL 训练仓库。
- `lora_target` 仅覆盖语言模型投影层；如需同时微调视觉塔注意力（
  `vision_model.*.self_attn.qkv`），可扩展该字段并评估显存与效果。
- 服务器上运行前需确认数据与模型路径（`--model`、`--data-dir`），以及
  LLaMA-Factory 已安装（`conda activate m3` 后 `pip install llamafactory`）。
