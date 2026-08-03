# Modification Note: Fix InternVL Smoke-Test Issues (template, processor, media_dir) - 2026-08-03 23:18:00 CST

## Modification Time

2026-08-03 23:18:00 CST (+0800)

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Fix three issues discovered while running the authorized smoke test of
`scripts/finetune_vlm_lora.py` on the dedicated InternVL server
(the server IP/port is kept out of the repository): the InternVL3.5 template,
the missing processor support
files in the server-side model snapshot, and the LLaMA-Factory 0.9.x
`media_dir` argument.

## Modified Files

- `scripts/finetune_vlm_lora.py`（修改）：InternVL3.5 模板由 `qwen3` 改为
  `intern_vl`；训练配置顶层新增 `media_dir`。
- `tests/test_finetune_vlm_lora.py`（修改）：断言 `intern_vl` 模板与
  `media_dir` 字段。
- `DETAILS.md`（修改）：第 16 节补充 `media_dir` 顶层参数说明。
- `docs/changes/2026-08-03-troy-add-finetune-vlm-lora-python.md`（修改）：
  同步模板名称为 `intern_vl`。

服务器侧（不进入仓库，模型目录已有 `.bak` 备份）：

- `models/OpenGVLab--InternVL3_5-8B/snapshots/master/`：用官方
  `OpenGVLab/InternVL3_5-8B-HF` 的 `tokenizer_config.json`、
  `added_tokens.json`、`special_tokens_map.json`、`tokenizer.json` 替换了
  缺失 `start_image_token` / `end_image_token` / `context_image_token` /
  `video_token` 配置的快照版本；原文件保留为 `*.bak`，模型权重未改动。
- 新增 `models/OpenGVLab--InternVL3_5-8B-HF/snapshots/master/`：LLaMA-Factory
  0.9.5 拒绝 GitHub 格式（`InternVLChatModel`），因此使用官方
  `internvl_custom2hf.py` 的键名转换逻辑（并修正 InternVL3.5 顶层键不带
  `model.` 前缀的差异）把现有 GitHub 格式权重转换为 HF 兼容格式
  （`InternVLForConditionalGeneration`），按官方 `-HF` 的
  `model.safetensors.index.json` 重新分片；键集合与官方索引完全一致
  （841/841），`AutoModelForImageTextToText` 加载参数总数
  8,528,318,464 与官方一致。原 GitHub 格式目录未删除。

## Core Changes

- 模板：LLaMA-Factory 0.9.5 将 InternVL3.5 系列注册为 `intern_vl` 模板
  （`InternVLPlugin`，支持 `<image>` 输入）；`qwen3` 是纯文本模板，会导致
  “This model does not support image input”。
- Processor：transformers 5.6 的内置 `InternVLProcessor` 依赖 tokenizer 的
  `start_image_token` / `end_image_token` / `context_image_token` /
  `video_token` 属性；服务器上的 GitHub 格式快照缺少这些配置，已用官方
  `-HF` 的 tokenizer 文件补齐并验证 `AutoProcessor` 可加载。
- `media_dir`：LLaMA-Factory 0.9.5 只读取训练配置顶层的 `media_dir`（默认
  等于 `dataset_dir`），`dataset_info.json` 内每个数据集的 `media_dir` 字段
  不再生效；脚本生成的 train YAML 现在显式写入数据集绝对路径。

## Whether the Canonical Sample Format Was Changed

No. `data/schema.py` 的 `CanonicalSample` / `CanonicalPrediction` 未改动。

## Whether the Model Interface Was Changed

No. 仓库内未改动任何模型加载逻辑；服务器模型快照仅补齐 tokenizer 配置文件
（已备份，权重不变）。

## Whether the Configuration Was Changed

仅新增/修改脚本运行时生成的 LLaMA-Factory 配置（`intern_vl` 模板、
顶层 `media_dir`）；仓库既有配置未改动。

## Whether Evaluation Was Affected

No.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. `tests/test_finetune_vlm_lora.py` 增加对应断言。

## Whether .gitignore Was Updated

No. 无新增产物类型。

## Validation Method

- 本地 `python3 -m compileall -q` 通过。
- 服务器实测：`AutoProcessor.from_pretrained(快照路径, trust_remote_code=True)`
  加载成功，tokenizer 的 image/video 特殊 token 属性齐全。
- 冒烟测试三次运行记录：
  1. `qwen3` 模板 → 报 “This model does not support image input”；
  2. `intern_vl` 模板 → 报 “Processor was not found”（tokenizer 缺
     `start_image_token` 等属性）；
  3. 补齐 tokenizer 后 → 报图像相对路径无法解析（缺少顶层 `media_dir`）。
- 第 4 次运行：GitHub 格式被 LLaMA-Factory 0.9.5 明确拒绝（要求 HF 兼容
  格式），随后完成格式转换。
- 第 5 次运行（`max_samples=3`，1 个优化步）：训练成功（train loss
  13.25，eval loss 9.88），自动导出最佳 LoRA
  （`best_lora/adapter_model.safetensors`，约 167MB）、合并模型
  （`merged/`，4 个 safetensors 分片约 16.6GB）与训练曲线
  （`train_curves.png`），断点状态文件记录完整。
- 重跑验证：相同命令再次执行时，“训练已完成/最佳 LoRA 已导出/合并模型已
  导出”均正确跳过，仅重绘训练曲线，断点语义符合预期。
- 服务器 m3 conda 环境执行 `pytest -q tests/test_finetune_vlm_lora.py`：
  14 passed。

## Risks and Follow-up TODOs

- 服务器原 GitHub 格式目录的 tokenizer 文件已替换为官方 `-HF` 版本（原文件
  在 `*.bak`），另生成了 `-HF` 格式新目录；若后续确认必须使用旧文件可回退。
- 本地仓库 `models/InternVL3_5-8B/` 仍是 GitHub 格式且缺 processor 配置，
  若要在本地用 LLaMA-Factory 0.9.5 微调，需要同样处理或改用 `-HF` 权重。
- 冒烟测试仅验证 3 个样本/1 步；完整数据集的训练、评估与导出仍需正式运行。
