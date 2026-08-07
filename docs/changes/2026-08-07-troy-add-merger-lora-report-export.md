# Modification Note: Add Merger-LoRA Training/Test Report Export Script - 2026-08-07 10:57:25 CST

## Modification Time

2026-08-07 10:57:25 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Provide one standalone script that automatically exports the merger-LoRA
training curves and the VRSBench test result statistics after a training and
evaluation run, without requiring TensorBoard or manual copy of logs.
新增一个独立脚本，在训练与评测完成后自动导出 merger-LoRA 训练曲线和
VRSBench 测试结果统计，无需启动 TensorBoard 或手工复制日志。

## Modified Files

- `scripts/export_finetune_report.py` (new)
- `tests/test_export_finetune_report.py` (new)
- `requirements-finetune.txt` (added optional matplotlib)
- `README.md` (added export usage)
- `DETAILS.md` (added the script to section 3.10)

## Core Changes

- `scripts/export_finetune_report.py` reads `trainer_state.json` under
  `--train-dir`, splits the log history into train/eval rows, merges them per
  step, and writes a per-step CSV plus a compact loss summary.
  `scripts/export_finetune_report.py` 读取 `--train-dir` 下的
  `trainer_state.json`，将日志拆分为训练/验证行并按 step 合并，输出每步 CSV
  与紧凑 loss 汇总。
- It scans `--eval-dir` for every `*.summary.json`, re-analyzes the referenced
  sample/prediction JSONL for answer-length statistics and failure reasons,
  and renders a per-task Markdown table (total/succeeded/failed/exact match/
  mean latency/empty predictions/mean answer length).
  扫描 `--eval-dir` 下所有 `*.summary.json`，重新分析其引用的
  sample/prediction JSONL 得到答案长度与失败原因统计，并按任务渲染
  Markdown 表（总数/成功/失败/精确匹配/平均耗时/空预测/平均答案长度）。
- Outputs share the `--report-path` stem: `report.md`, `report.json`,
  `report.csv`, and (when matplotlib is installed and `--no-charts` is not
  set) `report_training_curves.png`. The chart shows train loss, learning
  rate, and eval loss versus step.
  输出文件共用 `--report-path` 的主文件名：`report.md`、`report.json`、
  `report.csv`，以及（装有 matplotlib 且未设置 `--no-charts` 时）
  `report_training_curves.png`。曲线图包含 train loss、learning rate 与
  eval loss 对 step 的变化。
- `requirements-finetune.txt` adds `matplotlib>=3.8` as an optional chart
  dependency; the script degrades to MD/JSON/CSV when it is absent.
  `requirements-finetune.txt` 增加可选的 `matplotlib>=3.8` 用于绘图；
  matplotlib 缺失时脚本自动降级为 MD/JSON/CSV 输出。

## Whether the Canonical Sample Format Was Changed

No. The script only reads existing canonical `{"sample", "prediction"}`
JSONL and evaluation summaries; it never writes sample/prediction records.
否。脚本只读取已有的规范化 `{"sample", "prediction"}` JSONL 与评测摘要，
不写出 sample/prediction 记录。

## Whether the Model Interface Was Changed

No. No model loading, weight, adapter, or inference interface was touched.
否。未改动任何模型加载、权重、适配器或推理接口。

## Whether the Configuration Was Changed

Yes, only inside the new script: its CLI exposes `--train-dir`,
`--eval-dir`, `--report-path`, `--title`, `--no-charts`, and `--chart-dpi`.
No existing configuration schema or field was modified.
是，仅限新脚本内部：CLI 暴露 `--train-dir`、`--eval-dir`、`--report-path`、
`--title`、`--no-charts` 与 `--chart-dpi`。未改动既有配置模式或字段。

## Whether Evaluation Was Affected

No. No metric, dataset split, reference-answer reading, or `eval/` logic was
modified; the export script only aggregates outputs already written by
`scripts/evaluate_qwen3vl_merger_lora.py`.
否。未修改任何指标、数据划分、参考答案读取或 `eval/` 逻辑；导出脚本只汇总
`scripts/evaluate_qwen3vl_merger_lora.py` 已写出的结果。

## Whether Deployment Was Affected

No. The deployment/merge paths are untouched.
否。部署/合并路径未改动。

## Whether pytest Was Updated

Yes: added `tests/test_export_finetune_report.py` with 9 tests covering log
split/merge, summaries, prediction analysis, Markdown rendering, the full CLI
write path, missing-state errors, and optional PNG rendering.
是：新增 `tests/test_export_finetune_report.py`，9 个测试覆盖日志拆分/合并、
汇总、预测分析、Markdown 渲染、完整 CLI 写路径、缺失 state 报错与可选 PNG
渲染。

## Whether .gitignore Was Updated

No. Reports land under `outputs/`, which is already ignored.
否。报告输出在 `outputs/` 下，已被现有 `.gitignore` 覆盖。

## Validation Method

- `python -m compileall -q scripts/export_finetune_report.py
  tests/test_export_finetune_report.py` passed.
- Relevant tests passed:
  `python -m pytest -q tests/test_export_finetune_report.py
  tests/test_prepare_vrsbench_sft.py tests/test_finetune_qwen3vl_merger_lora.py
  tests/test_evaluate_qwen3vl_merger_lora.py` -> 23 passed.
- An end-to-end smoke run against a fabricated `trainer_state.json` and
  evaluation summary in `/tmp` produced `report.md`, `report.json`,
  `report.csv`, and `report_training_curves.png`.
- Full `pytest -q` could not be collected in the local M3 conda environment
  because unrelated modules require `cv2` (ModuleNotFoundError); the
  LoRA/evaluation-related test set above is the feasible scope.
  本地 M3 环境缺少无关模块依赖 `cv2`，无法完整收集全量测试；以上
  LoRA/评测相关测试集为可执行范围。

## Risks and Follow-up TODOs

- The training-curve source is `trainer_state.json` (logged every
  `--logging_steps`); TensorBoard event files are not parsed, so sub-logging
  resolution is not available from this script.
  训练曲线来源为 `trainer_state.json`（按 `--logging_steps` 记录）；本脚本
  不解析 TensorBoard event 文件，因此无法获得比日志更细的粒度。
- If multiple `*.summary.json` files exist under `--eval-dir`, all are
  included in one report; name files by run when comparing variants.
  若 `--eval-dir` 下存在多个 `*.summary.json`，报告会全部包含；比较不同变体
  时请按运行分别命名文件。
- No real GPU training/evaluation data exists yet, so the report content was
  only validated with synthetic fixtures.
  尚无真实 GPU 训练/评测数据，报告内容仅用合成数据验证。
