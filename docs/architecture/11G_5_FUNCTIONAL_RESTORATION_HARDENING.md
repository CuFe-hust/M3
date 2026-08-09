# 11G.5 — Functional Restoration Contract Hardening

## 目标

Task 11A–11G 恢复了大部分公开/运维/评估/报告面后，本硬化任务关闭剩余行为
缺口。仅硬化；不涉及 Task 11H / Task 12 / Spark 真机验收。

## 修复清单

| 修复 | 内容 | 关键文件 |
|---|---|---|
| Fix A | 新增 `runs/<run_id>/run_request.json` 运行调用产物：真实 dataset_root（实际 options.root，非配置默认）、task_mode（explicit/adapter_default/auto）、tasks、auto_task、sample_ids、limit/start/shard、evaluate、judge_policy、judge_sample_rate、render_errors、fail_fast；fresh 在身份确立后、样本/模型执行前原子写入（失败即推理前失败）；resume-run 从 run_request 重建（缺失/损坏稳定失败，绝不猜测） | `workflows/schema.py`（RunRequest）、`workflows/run_store.py`（write/read_run_request）、`application/runtime.py`、`application/commands/resume_run.py` |
| Fix B | judge_policy 与 judge_sample_rate 一并持久化（run_request 为权威来源）；resume 原样恢复（evaluate 门控由 runtime 重应用），绝不静默回退 none | 同上 |
| Fix C | render-errors 用 `status.task`（执行任务）判定 counting 族，存储命名空间（run_task="auto"）绝不决定执行语义 | `application/runtime.py` |
| Fix D | render-errors 源图经 `UnifiedSample.model_validate` + `resolve()` + `is_relative_to(root)` canonical containment；`../`、`../../`、`C:/`、UNC 全部稳定 skip note，外部文件绝不打开 | `application/runtime.py` |
| Fix E | count-image 冻结身份：fresh 无 id 恒 RunStore 唯一 id（不用 sample_id）；fresh 显式重复稳定失败；`--resume` 必须显式 run-id 且校验 manifest + run_request 调用身份（command/sample_id/image_identity 内容哈希）；force 只在 resume 内重跑 | `application/commands/count_image.py` |
| Fix F | count-image `--evaluate` 真实开关：`SampleRunner.run_one(evaluate=True)` 窄开关（数据集执行默认不变）；无标志不写 counting_evaluation.json | `workflows/sample_runner.py` |
| Fix G | summarize-evaluations 双模式：`--input FILE`（EvaluationRecord JSONL，损坏行稳定失败）+ `--run-id`（保留扫描模式）；互斥；`--output` 可选精确输出 | `application/commands/summarize_evaluations.py` |
| Fix H | evaluate-run `--deepseek` 按执行任务分派：counting 族经 `JudgeService.judge_counting`（精确持久化 CountTargetSpec 优先，否则 canonical 标签稳定中性重建）；VQA 不变；judge 失败保留确定性记录、稳定 judge_error、零 Qwen | `application/commands/evaluate_run.py` |
| Fix I/J | 统一报告 bundle 持久化 `persist_report_bundle`（report.html/report.json/samples.csv/samples.jsonl/metadata.json/deepseek_audit.jsonl/external_standard.json 按需）到 `runs/<run_id>/report/`；`Runtime.run_dataset` 终态后自动持久化（失败=稳定命令失败）；standard-evaluate 关联 run 时持久化 bundle + external_standard 命名空间 | `reporting/exporters.py`、`application/runtime.py`、`application/commands/standard_evaluate.py` |
| Fix K | deepseek audit 的 request_id/request_hash/prompt_version 来自实际持久化 RequestMeta（`deepseek_vqa_judge/` 与 `deepseek/` 产物目录，按任务选择）；缺失 → null 身份字段，绝不从判决合成哈希；路径经冻结存储身份推导 | `reporting/exporters.py` |

## 冻结不变式（保持）

- run identity：fresh 唯一 / 显式重复失败 / resume 显式 id + 匹配 run；绝不静默转 resume。
- 样本存储 `runs/<run_id>/tasks/<run_task>/samples/<sha256[:24]>`；sample_id 绝不直接作目录名。
- task 语义：sample.json.task=解析任务、status.json.task=执行任务、prediction.run_task=存储命名空间、prediction.task=执行任务。
- result_path：status 样本相对纯 basename；prediction run 相对；绝不主机绝对路径。
- Reporting 只读、零 Qwen/DeepSeek/重推理、不修改源产物。
- Qwen 仅在 composition root 创建；维护/报告/评估命令零 Qwen。
- MME 官方导出行为冻结：源记录全保留、预测→Output、缺失→""、未关联字段原样、源文件不修改。

## 验证

- 全量 pytest：1550 passed（含 18 个新增 11G.5 测试）
- compileall 全绿；`git diff --check` 干净
- 架构守卫：`allowed_python_files.txt` / `import_rules.json` 未修改；无 spacers_agent/eval 导入；main.py 仍薄分发
- 各套件：architecture 40 / contracts 136 / workflows 212 / evaluation 79 / reporting 24 / application 120 / integration 41 / test_main 35

## 验收

```text
TASK_11G_5_FUNCTIONAL_HARDENING_REPORT
RUN_REQUEST_ARTIFACT=PASS
ACTUAL_DATASET_ROOT_PERSISTED=PASS
RESUME_FROM_RUN_REQUEST=PASS
JUDGE_POLICY_PERSISTED=PASS
JUDGE_SAMPLE_RATE_PERSISTED=PASS
RESUME_JUDGE_POLICY_IDENTICAL=PASS
AUTO_TASK_RENDER_ERRORS_EXECUTION_TASK=PASS
RENDER_ERRORS_DATASET_ROOT_CONTAINMENT=PASS
COUNT_IMAGE_FRESH_UNIQUE_RUN=PASS
COUNT_IMAGE_EXPLICIT_DUPLICATE_REJECTED=PASS
COUNT_IMAGE_RESUME_REQUIRES_RUN_ID=PASS
COUNT_IMAGE_RESUME_ZERO_QWEN=PASS
COUNT_IMAGE_FORCE_REEXECUTES=PASS
COUNT_IMAGE_EVALUATE_FLAG_EFFECTIVE=PASS
SUMMARIZE_FILE_MODE=PASS
SUMMARIZE_RUN_MODE=PASS
EVALUATE_RUN_COUNTING_DEEPSEEK=PASS
EVALUATE_RUN_GENERAL_VQA_DEEPSEEK=PASS
EVALUATE_RUN_ZERO_QWEN=PASS
DETERMINISTIC_METRICS_SURVIVE_JUDGE_FAILURE=PASS
STANDARD_EVALUATE_WRITES_REPORT_BUNDLE=PASS
RUN_DATASET_WRITES_REPORT_BUNDLE=PASS
REPORTING_ADDS_ZERO_QWEN_CALLS=PASS
DEEPSEEK_AUDIT_REAL_REQUEST_ID=PASS
DEEPSEEK_AUDIT_REAL_REQUEST_HASH=PASS
DEEPSEEK_AUDIT_NO_SYNTHETIC_HASH=PASS
DEEPSEEK_AUDIT_NO_SECRET_LEAK=PASS
MME_OFFICIAL_EXPORT=PASS
ARCHITECTURE_ALLOWLIST_UNCHANGED=PASS
IMPORT_RULES_UNCHANGED=PASS
NO_LEGACY_IMPORTS=PASS
COMPILEALL=PASS
ARCHITECTURE_TESTS=PASS
CONTRACT_TESTS=PASS
WORKFLOW_TESTS=PASS
EVALUATION_TESTS=PASS
REPORTING_TESTS=PASS
APPLICATION_TESTS=PASS
INTEGRATION_TESTS=PASS
MAIN_CLI_TESTS=PASS
FULL_PYTEST=PASS
GIT_DIFF_CHECK=PASS
GITHUB_FOUNDATION_TESTS=PASS
READY_FOR_11H
```
