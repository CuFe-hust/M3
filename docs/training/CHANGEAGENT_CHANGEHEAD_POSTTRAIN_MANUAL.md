# ChangeAgent ChangeHead 后训练手册

> 文档状态：数据整理与训练前 Gate 已完成；正式训练、校准、独立测试评测和发布 Gate 尚未启动。
>
> 适用范围：`change_head` 多专家 Siamese ChangeHead 的后训练流程。本文不包含 Qwen/VQA 阶段。

## 1. 当前结论

当前数据可以进入后训练准备阶段，但不能把当前 manifest 当作已训练模型使用。

- 最终 Gate：`PASS_WITH_EXCLUSIONS`
- `ready_for_training=true`
- `formal_training_started=false`
- 最终训练记录：9566 条
- train / val / test：6465 / 1242 / 1859
- 排除样本：511 条（train 350、val 91、test 70）
- 额外预检隔离：0 条
- 未解析历史 hard-case 标识：6 个；仅跳过 hard-case 标签，不影响样本进入当前数据集

完整问题样本清单见：
[LEVIR-MCI 数据集问题说明](../changes/LEVIR_MCI_DATASET_ISSUES_FOR_EVENT_ORGANIZER.md)

## 2. 产物位置

以下路径位于 Spark 机器，不是本机路径。

### 2.1 Spark 数据根目录

```text
/home/user/cooper/datasets/levir_mci_changehead
```

源数据和下载归档：

```text
/home/user/cooper/datasets/levir_mci_changehead/source/LEVIR-MCI-dataset
/home/user/cooper/datasets/levir_mci_changehead/downloads/LEVIR-MCI-dataset.zip
```

### 2.2 可用于后训练的核心文件

```text
/home/user/cooper/datasets/levir_mci_changehead/prepared/change_head_training_manifest.json
/home/user/cooper/datasets/levir_mci_changehead/prepared/change_head_records.jsonl
/home/user/cooper/datasets/levir_mci_changehead/prepared/train_records.jsonl
/home/user/cooper/datasets/levir_mci_changehead/prepared/val_records.jsonl
/home/user/cooper/datasets/levir_mci_changehead/prepared/test_records.jsonl
/home/user/cooper/datasets/levir_mci_changehead/prepared/pretrain_ready.json
```

最终特征缓存：

```text
/home/user/cooper/datasets/levir_mci_changehead/cache/final_train
/home/user/cooper/datasets/levir_mci_changehead/cache/final_val
```

缓存审计：

```text
/home/user/cooper/datasets/levir_mci_changehead/prepared/train_cache_audit.json
/home/user/cooper/datasets/levir_mci_changehead/prepared/val_cache_audit.json
```

排除记录：

```text
/home/user/cooper/datasets/levir_mci_changehead/prepared/excluded_samples.jsonl
/home/user/cooper/datasets/levir_mci_changehead/prepared/excluded_sample_ids.txt
/home/user/cooper/datasets/levir_mci_changehead/prepared/exclusion_summary.json
```

### 2.3 本机已整理的说明文档

```text
C:\Users\TZDEZACR\Desktop\spacers-agent\code\docs\changes\LEVIR_MCI_DATASET_ISSUES_FOR_EVENT_ORGANIZER.md
C:\Users\TZDEZACR\Desktop\changeagent_pretrain_stop_report_2026-08-24.md
C:\Users\TZDEZACR\Desktop\spacers-agent\code\docs\training\CHANGEAGENT_CHANGEHEAD_POSTTRAIN_MANUAL.md
```

本次数据传输使用的本机临时文件已清理；本机不保留 LEVIR 数据副本、缓存或 Spark 凭据副本。

## 3. 数据合同与排除规则

ChangeHead 训练使用原始 split，不重新随机划分：

- train 只用于参数训练；
- val 只用于模型选择和校准；
- test 保持独立，不能参与训练、早停或校准。

发现的 511 条冲突样本必须保持原始数据不变，并从 ChangeHead 的 records、cache、training、calibration 和 evaluation 中排除。禁止通过改写 `changeflag`、改写 mask、人工补 mask 或模型生成 mask 来“修复”这些样本。

冲突规则为：

- `CHANGEFLAG_NOCHANGE_MASK_NONEMPTY`：changeflag 表示无变化，但 mask 非空；
- `CHANGEFLAG_CHANGED_MASK_EMPTY`：changeflag 表示有变化，但 mask 为空。

排除动作统一记录为 `EXCLUDED_FROM_CHANGEHEAD_POSTTRAIN`。排除明细以 `excluded_samples.jsonl` 为准。

6 个历史 hard-case 标识不是可唯一映射的真实样本 ID，因此只跳过标签，不做模糊匹配：

```text
06e58013632e752a9ef4
3a91a479f21a3c97729a
812e7fc6aa79d94dfc67
839cd4c0b76a379fdbbd
f582bcf6b67f1d89f685
fb49f46ea205c5096ada
```

## 4. 训练前必须复核的 Gate

在 Spark 上执行后训练前，先检查 Gate，不要直接启动训练：

```bash
cd /home/user/cooper/M3
/home/user/miniconda3/envs/Cooper_tryagents/bin/python3 - <<'PY'
import json
from pathlib import Path

p = Path('/home/user/cooper/datasets/levir_mci_changehead/prepared/pretrain_ready.json')
x = json.loads(p.read_text())
print(json.dumps({
    'status': x.get('status'),
    'ready_for_training': x.get('ready_for_training'),
    'formal_training_started': x.get('formal_training_started'),
    'records_sha256': x.get('records_sha256'),
    'pipeline_fingerprint': x.get('pipeline_fingerprint'),
    'workspace_dirty': x.get('workspace_dirty'),
    'tests': x.get('gates', {}).get('tests'),
}, indent=2))
PY
```

必须同时满足：

1. `ready_for_training` 为 `true`；
2. `formal_training_started` 仍为 `false`，除非确实要开始新一轮训练；
3. train/val cache audit 均为 `PASS`；
4. manifest、records SHA256 和 pipeline fingerprint 与 Gate 文件一致；
5. 训练脚本和测试脚本的变更已提交或已形成可追溯备份。

当前 Gate 记录的 targeted tests 为 31 passed；全量 pytest 为 2243 passed、3 failed。3 个失败属于既有的无关失败，已在 Gate 中记录，不应静默忽略：

```text
tests/application/test_runtime.py::test_evaluate_run_e2_families_zero_qwen
tests/evaluation/test_caption_metrics.py::test_caption_module_import_has_no_network_side_effects
tests/models/test_change_head_runtime.py::test_invalid_pixels_are_zero_after_runtime
```

正式训练前建议先提交 Spark 仓库中新增的准备脚本和测试，避免 dirty workspace 造成结果不可复现。提交前不要提交数据、模型权重、凭据或缓存。

## 5. 正式训练步骤

以下命令是执行模板。当前任务只完成了数据准备，不能在未确认时执行正式训练。

### 5.1 检查配置

代码仓库：

```text
/home/user/cooper/M3
```

训练配置模板：

```text
/home/user/cooper/M3/configs/change_head_train.example.yaml
```

当前默认配置包括：30 epochs、batch size 4、learning rate `3e-4`、AMP、可选专家 dropout `0.20`、主选择指标 `val_pixel_f1`、early-stop patience 6。正式运行前应复制为带日期/实验名的不可变配置并记录 SHA256，不要直接修改 example 文件。

### 5.2 启动训练

```bash
cd /home/user/cooper/M3
RUN=/home/user/cooper/datasets/levir_mci_changehead/posttrain/change_head_run_YYYYMMDD_HHMMSS
/home/user/miniconda3/envs/Cooper_tryagents/bin/python3 scripts/train_change_head.py \
  --config /home/user/cooper/M3/configs/change_head_train.example.yaml \
  --train-cache /home/user/cooper/datasets/levir_mci_changehead/cache/final_train \
  --val-cache /home/user/cooper/datasets/levir_mci_changehead/cache/final_val \
  --manifest /home/user/cooper/datasets/levir_mci_changehead/prepared/change_head_training_manifest.json \
  --output-dir "$RUN" \
  --device cuda
```

训练输出至少应包含：

```text
$RUN/resolved_config.yaml
$RUN/train_log.jsonl
$RUN/summary.json
$RUN/best/
$RUN/last/
```

`best/` 和 `last/` 中的 checkpoint 必须保存 manifest contract、pipeline fingerprint 和真实的 ChangeHead 权重 SHA256。训练前 manifest 中的全零模型权重 SHA 是“未训练模型”哨兵值，不能作为运行时模型发布。

### 5.3 训练后验收

检查：

- `summary.json` 的 train/val 样本数仍为 6465/1242；
- best checkpoint 的选择指标、epoch 和配置可追溯；
- loss/metric 无 NaN 或 Inf；
- 必需专家缺失时训练应失败，而不是静默补零；
- 可选专家缺失只能按 `zero_with_presence_mask` 处理；
- checkpoint fingerprint 与训练时 manifest 一致；
- 不生成或覆盖 test cache。

## 6. 校准步骤

校准只能使用未参与参数选择的验证输出；不要使用 test 输出。需要准备与 val records 对齐的 logits、targets 和 valid mask，且数组顺序必须能由 sample ID 复核。

```bash
cd /home/user/cooper/M3
/home/user/miniconda3/envs/Cooper_tryagents/bin/python3 scripts/calibrate_change_head.py \
  --logits <VAL_LOGITS.npy> \
  --targets <VAL_TARGETS.npy> \
  --valid <VAL_VALID.npy> \
  --tags <VAL_TAGS.json> \
  --checkpoint <RUN>/best/model.safetensors \
  --validation-fingerprint <VAL_FINGERPRINT> \
  --output <RUN>/calibration.json
```

`calibration.json` 至少要记录 temperature、rescue threshold、validation fingerprint、校准前后 NLL/ECE、Brier score 和 checkpoint SHA256。若校准输入与 val records 数量、顺序或 fingerprint 不一致，应停止，不得继续发布。

## 7. 独立 test 评测

校准完成后，重新对 test split 做一次独立推理。test 只用于最终报告，不回写训练配置或阈值。基础像素评测脚本的接口为：

```bash
cd /home/user/cooper/M3
/home/user/miniconda3/envs/Cooper_tryagents/bin/python3 scripts/eval_change_head.py \
  --probabilities <TEST_PROBABILITIES.npy> \
  --targets <TEST_TARGETS.npy> \
  --valid <TEST_VALID.npy> \
  --output <RUN>/test_metrics.json
```

同时保留 baseline 和 ChangeHead assist/candidate 的同一批 test 样本结果，用于后续 release gate。报告中必须明确：test 中仍有 70 条原始冲突样本被排除，不能把它们算作模型误报或漏报。

## 8. Release Gate

发布前必须完成 shadow parity、关键 no-change、normal changed、building-edge、残余 hard-case 和 broad validation 检查。Gate 配置：

```text
/home/user/cooper/M3/configs/change_head_release_gate.yaml
```

执行模板：

```bash
cd /home/user/cooper/M3
/home/user/miniconda3/envs/Cooper_tryagents/bin/python3 scripts/eval_change_head_release.py \
  --baseline <RUN>/baseline_metrics.json \
  --assist <RUN>/assist_metrics.json \
  --gate-config /home/user/cooper/M3/configs/change_head_release_gate.yaml \
  --shadow-parity \
  --hard-case-comparison <RUN>/hard_case_comparison.json \
  --output <RUN>/release_gate.json
```

`release_gate.json` 的 `passed` 必须为 `true`，并且需要人工确认没有数据泄漏、没有用 test 调参、没有把 excluded samples 混入结果。任何 release gate 失败都应保留完整输出并停止发布。

## 9. 停止与回滚条件

遇到以下任一情况立即停止当前阶段：

- Gate 文件显示 `ready_for_training=false`；
- manifest、cache audit、records SHA256 或 pipeline fingerprint 不一致；
- 训练出现 NaN/Inf、OOM 后改变 batch/config 但未重新记录实验；
- 必需专家缺失却被静默补零；
- val/test 样本顺序无法复核；
- excluded sample 出现在训练、校准或评测输入中；
- calibration 使用了 test；
- shadow parity、critical no-change 或 broad validation 失败；
- checkpoint 没有真实权重 SHA、配置、环境和输入 manifest 记录。

停止时只保留日志、配置、summary、checkpoint 元数据和失败报告；不要删除原始数据或 prepared 目录。若要重跑，使用新的 run 目录，不覆盖旧实验。

## 10. 最终交付清单

一次可发布的后训练结果应包含：

- 训练配置及 SHA256；
- `pretrain_ready.json` 和 cache audit；
- manifest、records SHA256、pipeline fingerprint；
- best/last checkpoint 及真实权重 SHA256；
- `train_log.jsonl`、`summary.json`；
- calibration JSON 及 validation fingerprint；
- 独立 test metrics；
- baseline/candidate 对比；
- hard-case comparison；
- `release_gate.json`；
- 环境版本、Git commit、workspace 状态；
- 数据集问题说明和 excluded sample 清单。

当前可交付状态到此为止：数据已整理、缓存已审计、训练前 Gate 已记录；正式训练及其后续产物尚不存在。
