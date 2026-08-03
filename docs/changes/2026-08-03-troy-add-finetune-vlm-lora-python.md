# Modification Note: Add Python LLaMA-Factory LoRA Fine-Tune/Resume/Export Entry - 2026-08-03 23:00:00 CST

## Modification Time

2026-08-03 23:00:00 CST (+0800)

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Provide a single Python entry point for LLaMA-Factory LoRA fine-tuning that
supports both `InternVL3.5-8B` (local `models/InternVL3_5-8B/`) and
`Qwen3-VL-4B-Instruct`, runs on the merged ShareGPT dataset
(`data/微调数据集/merged/`), supports checkpoint resume, and automatically
exports the best LoRA adapter, the merged base+LoRA full model, and training
curves after training.

## Modified Files

- `scripts/finetune_vlm_lora.py`（新增）：Python 主脚本，生成 LLaMA-Factory
  训练/导出配置并调用 `llamafactory-cli`。
- `tests/test_finetune_vlm_lora.py`（新增）：模型归一化、服务器限制、
  dataset_info 生成、训练配置、最佳 checkpoint 选择、适配器拷贝与指纹的
  单元测试。
- `DETAILS.md`（更新）：新增第 16 节，记录微调工作流、断点行为、导出产物
  与服务器约束。
- `README.md`（更新）：新增 LLaMA-Factory LoRA 微调运行说明。

## Core Changes

- 双模型适配：`internvl3.5-8b`（template `intern_vl`、`trust_remote_code: true`）
  与 `qwen3-vl-4b`（template `qwen3_vl`）；模型路径、数据目录、输出目录均可
  通过命令行覆盖，仓库内不硬编码服务器绝对路径。
- 运行时生成 `dataset_info.json`（`merged_train` / `merged_val` /
  `merged_test`，`messages=conversations`、`images=images`、绝对 `media_dir`）
  与训练/导出 YAML 到 `.cache/llamafactory/`。
- 断点支持：输出目录存在 `checkpoint-*` 或 `trainer_log.jsonl` 时自动从最新
  checkpoint 续训（写入 `resume_from_checkpoint`）；`all_results.json` 或状态
  文件表明已完成时跳过训练；`--force-restart` 要求空输出目录且不自动删除；
  状态文件指纹不一致时拒绝续训。
- 自动导出：从 `trainer_log.jsonl` 按指标（默认最低 `eval_loss`，可用
  `--metric` / `--higher-is-better` 调整）选择最佳 checkpoint，仅复制适配器
  权重到 `<output_dir>/best_lora/`，调用 `llamafactory-cli export` 将基座与
  最佳 LoRA 合并导出到 `<output_dir>/merged/`，并复用
  `scripts/plot_train_curves.py` 绘制 `<output_dir>/train_curves.png`。
- 服务器约束：`--server` 标志在专用服务器上拒绝 `qwen3-vl-4b`，仅允许
  InternVL 微调；Qwen3-VL 在本地或其他环境运行。

## Whether the Canonical Sample Format Was Changed

No. `data/schema.py` 的 `CanonicalSample` / `CanonicalPrediction` 未改动。

## Whether the Model Interface Was Changed

No. 未改动 `models/` 或任何加载逻辑；训练与合并导出均由 LLaMA-Factory 外部
工具完成，原始权重文件不被修改。

## Whether the Configuration Was Changed

未改动仓库既有配置（`configs/default.yaml`、`config/baseline.example.json`
等）。新增的 LLaMA-Factory 配置由脚本运行时生成到已忽略的 `.cache/` 目录。

## Whether Evaluation Was Affected

No. `eval/`、指标、数据集划分与参考答案读取逻辑未改动；`merged_test.json`
仍仅用于最终评测。

## Whether Deployment Was Affected

No. 合并导出产物（`<output_dir>/merged/`）为后续部署提供输入，但本变更未
修改 `deploy/` 或任何部署路径。

## Whether pytest Was Updated

Yes. 新增 `tests/test_finetune_vlm_lora.py`，覆盖脚本的纯逻辑分支。

## Whether .gitignore Was Updated

No. 新增产物（`.cache/llamafactory/`、`outputs/finetune/*/`、状态文件）均
位于已有忽略规则内，未引入新文件类型。

## Validation Method

- `python3 -m compileall -q scripts/finetune_vlm_lora.py tests/test_finetune_vlm_lora.py`：
  通过。
- 函数级行为验证（临时目录驱动，未调用真实训练）：dataset_info 生成、
  双模型训练配置、`resume_from_checkpoint` 注入、最佳 checkpoint 选择、
  适配器文件拷贝、导出配置与 `--dry-run` 计划输出均通过。
- `--dry-run` 使用真实 `data/微调数据集/merged/` 生成并打印了 InternVL 与
  Qwen3-VL 的完整 LLaMA-Factory 配置，可被 YAML/JSON 正常解析。
- 完整流程模拟（`/usr/bin/true` 作为假 CLI，`--skip-train` 基于已有产物）：
  跳过训练、导出 best LoRA、调用 export、绘制曲线失败时告警但流程继续，
  状态文件记录正确。
- 未运行 `pytest`：本机未安装 pytest/matplotlib；也未执行真实训练与合并
  导出（本机无 LLaMA-Factory 与对应环境）。
- 服务器补充验证（见 2026-08-03-troy-fix-internvl-smoke-issues.md 与
  docs/experiments/2026-08-03-internvl-lora-smoke-test.md）：单元测试
  14/14 通过；InternVL3.5-8B 冒烟训练、最佳 LoRA/合并模型导出、曲线绘制
  全部成功；重跑时训练与导出正确跳过。

## Risks and Follow-up TODOs

- InternVL3.5 与 `qwen3_vl` 模板需要较新版本 LLaMA-Factory（>= 0.9，且支持
  `qwen3_vl`）；若服务器版本较旧，需先升级或调整 `--extra-yaml`。
- `report_to: tensorboard` 依赖 tensorboard；脚本会在训练前探测并提示安装。
- 训练曲线 PNG 依赖 matplotlib；缺失时脚本告警并继续，TensorBoard 事件仍
  会保留在 `<output_dir>/runs/`。
- 真实训练、断点续训与合并导出的端到端验证需在服务器/训练机上完成。
