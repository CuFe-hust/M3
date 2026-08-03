# Modification Note: Add LoRA Merged-Model Export and Training Curves - 2026-08-03 17:35:00 +0800

## Modification Time

2026-08-03 17:35:00 +0800

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

基于已完成的 InternVL LoRA 微调入口，补充两个能力：1) 将基座模型与 LoRA
适配器合并导出为完整模型；2) 展示训练曲线（TensorBoard 事件 + 离线 PNG 绘制）。

## Modified Files

- `configs/llamafactory/export_internvl_lora.yaml`（新增）：`llamafactory-cli
  export` 合并导出配置（safetensors、约 5GB 分片、CPU 合并）。
- `scripts/export_internvl_lora.sh`（新增）：导出启动脚本，注入
  `model_name_or_path` / `adapter_name_or_path` / `export_dir`。
- `scripts/plot_train_curves.py`（新增）：从训练输出目录的
  `trainer_log.jsonl` 绘制训练损失、验证损失与学习率曲线 PNG。
- `configs/llamafactory/train_internvl_lora.yaml`（更新）：`report_to` 由
  `none` 改为 `tensorboard`，训练曲线事件写入 `<output_dir>/runs/`。
- `scripts/finetune_internvl_lora.sh`（更新）：启动前检测 tensorboard 是否可
  导入，缺失时给出中文提示（不中断）。
- `DETAILS.md`（更新）：15.1 小节补充训练曲线查看方式与完整模型导出命令。

## Core Changes

- 导出：`scripts/export_internvl_lora.sh --model <基座> --adapter <LoRA目录>`
  调用 `llamafactory-cli export`，将 LoRA 权重合并回基座并写出完整模型目录
  （`outputs/finetune/internvl_lora_merged/`）。校验 `adapter_config.json`
  存在；`template` 与训练配置保持一致；`export_device: cpu` 减少显存占用。
- 曲线：训练配置启用 `report_to: tensorboard`，事件写入
  `<output_dir>/runs/`，可用 `tensorboard --logdir <output_dir>/runs` 查看；
  同时 LLaMA-Factory 会在输出目录写 `trainer_log.jsonl`，
  `scripts/plot_train_curves.py <output_dir>` 用 matplotlib（Agg 后端）绘制
  损失与学习率曲线 PNG，适配无显示环境的服务器。
- 两个启动脚本沿用与训练脚本一致的模式：路径均可通过命令行参数或环境变量
  覆盖，`llamafactory-cli` 自动探测，无硬编码绝对路径。

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. 导出仅生成新目录（合并后的完整模型），不修改基座模型与适配器原文件。

## Whether the Configuration Was Changed

是，仅限新增/修改 LLaMA-Factory 配置：新增 `export_internvl_lora.yaml`，
`train_internvl_lora.yaml` 的 `report_to` 改为 `tensorboard`。仓库既有配置
（`configs/default.yaml` 等）未改动。

## Whether Evaluation Was Affected

No.

## Whether Deployment Was Affected

No. 导出后的完整模型为后续部署/评测提供输入，但本变更未改部署路径本身。

## Whether pytest Was Updated

No. 新增内容仍为外部工具（LLaMA-Factory）启动脚本与绘图脚本，不涉及仓库
内部接口；本机完成静态与模拟校验（见 Validation Method）。

## Whether .gitignore Was Updated

No. 新增产物（`outputs/finetune/internvl_lora_merged/`、`<输出>/runs/`、
曲线 PNG）均位于已忽略的 `outputs/` 下，未引入新文件类型。

## Validation Method

- `bash -n`：`scripts/export_internvl_lora.sh` 与
  `scripts/finetune_internvl_lora.sh` 语法检查通过。
- PyYAML：`export_internvl_lora.yaml` 与 `train_internvl_lora.yaml` 解析通过。
- 导出脚本模拟：`LLAMAFACTORY_CLI=echo` 运行，确认参数组装正确
  （`export` 子命令 + 三个路径注入）。
- 绘图脚本：在 `/tmp` 构造示例 `trainer_log.jsonl`，用临时安装的
  matplotlib（`pip install --target /tmp/... matplotlib`）实际渲染 PNG 成功，
  验证损失/验证损失/学习率曲线与无日志、日志为空等错误分支。
- 未执行真实训练与导出：本机未安装 LLaMA-Factory 且无法加载权重；
  合并导出与 TensorBoard 事件的端到端验证需在服务器实际环境进行。

## Risks and Follow-up TODOs

- `report_to: tensorboard` 需要训练环境中安装 `tensorboard`（否则训练启动会
  报错）；已在脚本中提前提示。
- 导出合并需要与训练相同的 LLaMA-Factory 版本与 `template`，若版本不一致
  可能出现权重名不匹配。
- 导出耗时与磁盘占用取决于模型规模（约 16GB+），服务器需预留空间。
