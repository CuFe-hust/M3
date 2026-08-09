# Evaluation and Reporting Integration

本说明记录 `try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868`
中仍有价值的评测/报告能力如何进入当前 `new_structure`。它描述可观察契约，
不声称新旧实现“完全对齐”，也不恢复旧包结构或旧 report artifacts。

## 单一 Evaluation Family Contract

Runtime task 决定执行路径，evaluation family 决定评分方式。唯一生产映射位于
`evaluation.records.RUNTIME_TASK_TO_EVALUATION_TASK`：

| Runtime task | Family | Deterministic artifact | VQA semantic Judge |
|---|---|---|---|
| `counting` | `counting` | `counting_evaluation.json` | no；保留 counting evidence Judge |
| `fine_grained_counting` | `counting` | `counting_evaluation.json` | no；保留 counting evidence Judge |
| `general_vqa` | `general_vqa` | `vqa_evaluation.json` | policy 允许时 yes |
| `multiple_choice_vqa` | `general_vqa` | `vqa_evaluation.json` | policy 允许时 yes |
| `scene_classification` | `general_vqa` | `vqa_evaluation.json` | policy 允许时 yes |
| `spatial_relation` | `general_vqa` | `vqa_evaluation.json` | policy 允许时 yes |
| `change_qa` | `general_vqa` | `vqa_evaluation.json` | policy 允许时 yes |
| `grounding` | `grounding` | `grounding_evaluation.json` | no |
| `caption` | `caption` | `caption_evaluation.json` | no |
| `change_caption` | `caption` | `caption_evaluation.json` | no |

family 与 artifact 映射均不可变。workflow、offline application command 和
report reader 不再维护私有 task set；generic metric 也不含 dataset-specific
runtime task 分支。

## Fresh、Resume 与 Offline

三条路径共用 `workflows.sample_runner.build_deterministic_evaluation(...)`：

```text
fresh:
  run-dataset -> SampleRunner -> deterministic -> optional JudgeService

resume:
  status.task (authoritative)
    -> missing deterministic supplement
    -> missing/failed Judge supplement
    -> no repeated Qwen for succeeded inference

offline:
  evaluate-run -> persisted result -> same deterministic helper
    -> optional DeepSeek -> refreshed report bundle
```

`EvaluationRecord.task` 保存 canonical family，不保存 runtime task。实际执行
task 保留在 status/index/trace 中，避免 candidate fallback 后评错 family。

## 保持的旧能力

- counting deterministic 公式及 failure accounting；
- general VQA strict exact-match；
- caption BLEU/METEOR/ROUGE/CIDEr corpus 公式；
- 外部 standard evaluator seam 与官方输出只读原则。

这些行为可以通过现有 Golden/parity 测试审计，但 Golden fixture 不因当前代
新增能力而重写。

## 当前代有意改进

- 10 个 runtime task 统一映射到 4 个 canonical family；
- spatial/change tasks 获得与 family 一致的 deterministic coverage；
- DeepSeek Judge 为可选、纯文本、结构化结果；
- deterministic、Judge、official 三类指标严格分离；
- unified report 读取持久化结果，不在 report-time 重新推理；
- path、secret、resume 与 artifact contract 均 fail-closed。

这些是当前架构契约，不应描述成 strict legacy parity。

## Exact 与 Semantic 同时可审计

VQA semantic Judge 只处理 strict mismatch。例如 reference=`"2"`、
candidate=`"There are two airplanes."` 时：

```text
deterministic exact_match = false
semantic Judge score      = 1
```

Judge 不会改写 `exact_match`。报告把 strict deterministic score 放在
`metrics`，把 semantic coverage、equivalent mismatch、failure、lower bound
和完整性放在 `judge_metrics`。Judge 未覆盖全部 mismatch 时，完整 semantic
score 为 `null`，只展示 confirmed lower bound。

保留 DeepSeek 是为了审计遥感 VQA 中“文本不同但语义一致”的情况，以及现有
counting evidence Judge；它不是新的视觉 Agent，也不参与 routing。

## Caption 与 Grounding 边界

`caption` 和 `change_caption` 共用 caption family。报告在 pycocoevalcap 可用
时展示 BLEU_1..4、METEOR、ROUGE_L、CIDEr；依赖缺失时输出稳定的
`metric_status=dependency_missing`，仍能生成整个报告。可选依赖保持惰性，
不加入核心运行依赖。

generic grounding 只计算双方均为 `normalized_0_999_top_left` 且均为
4-value xyxy 时的轴对齐 IoU。source pixels、未知 frame、8-value oriented
polygon、missing prediction/GT 均为 not applicable，不记作 IoU=0。benchmark
oriented metric 留在 `evaluation/datasets/*` 或 external standard evaluator。

## Reporting 与 Official Evaluation

统一 report bundle 为：

```text
runs/<run_id>/report/
  report.json
  report.html
  samples.csv
  samples.jsonl
  metadata.json
  deepseek_audit.jsonl
  [external_standard.json]
```

命名空间含义固定：

- `metrics`：local deterministic/corpus aggregate；
- `judge_metrics`：optional Judge quality 与 coverage；
- `external_standard`：独立 official/external evaluator 输出。

`evaluate-run` 负责 local canonical evaluation；`standard-evaluate` 负责调用
外部 `evaluate.py` 并生成 `*.standard.json`/`external_standard.json`。官方
参数不塞入 `evaluate-run`，官方分数也不并入 local metric 名称。

## Secret 与可复现性

DeepSeek key value 只存在于本机环境或 secret manager。仓库和持久化产物只
保存 `api_key_env="DEEPSEEK_API_KEY"` 这样的环境变量名。RequestMeta、cache
metadata、snapshot、manifest、report、audit 和稳定错误均不保存 key value；
Authorization header 只在内存请求构造阶段使用。

active VQA Judge prompt 为版本化 asset。PromptCatalog 注入 prompt text/version，
request hash 覆盖 model、prompt text/version、sample、payload 和 response
schema；run 同时保存 prompt snapshot/hash。历史 v1 prompt 文件保留不变，v2
使用不同 version/hash，不会与历史 cache key 冲突。

## 为什么不恢复旧 Report Artifacts

旧 report 文件名、目录和控制流混合了评测、Judge 与展示职责。当前实现保留
有价值的 metric/official 行为，但以统一 `EvaluationRecord`、只读 reporting、
明确 namespace 和安全 artifact contract 重建。恢复旧 artifacts 会重新引入
双重事实来源、模糊指标来源并削弱 resume/secret/path 安全，因此不属于迁移
目标。
