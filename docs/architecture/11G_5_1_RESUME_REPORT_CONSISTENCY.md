# 11G.5.1 — Resume / Reporting Consistency Finalization

## 目标

关闭 11G.5 评审后发现的跨路径一致性缺口：run-dataset resume、resume-run、
count-image resume/force、evaluate-run、judge-vqa-run、报告持久化与 counting
DeepSeek 目标重建。仅一致性硬化；不涉及 11H / 12 / Spark 真机验收。

## 修复清单

| 修复 | 内容 |
|---|---|
| Fix A | **run-dataset --resume 也以 run_request.json 为权威**：中央化 `reconstruct_dataset_resume_options(run_dir, run_id, run_store)`（application/runtime.py），resume-run 与 run-dataset --resume 共用；`_validate_resume_match` 严格比较 CLI 提供值与持久化调用（dataset/root/split/task 模式/tasks/auto_task/sample_ids/limit/start/shard/concurrency/evaluate/judge_policy/rate/render_errors/fail_fast），任何偏离在模型执行前稳定失败（零 Qwen），绝不静默覆盖原调用 |
| Fix B | **数据集根一次性 canonicalize**：`Runtime.run_dataset` 开头 `options.root.expanduser().resolve()`（身份确立/持久化之前），执行、adapter probe/迭代、render-errors data_root 与 `run_request.dataset_root` 恒为同一主机解析路径；相对 `--root ./data` 持久化为绝对路径（host-path-preserving，不声称跨机器可移植） |
| Fix C | **count-image 零 Qwen resume 需要合法匹配 CountingResult**：缺失/损坏/schema 非法/sample_id 不匹配一律视为不完整 → 重跑；绝不输出 `status=resumed, final_count=null` |
| Fix D | **count-image resume 用持久化 evaluate 意图**（`request.evaluate` 权威，CLI --evaluate 忽略；事后评估变更归 evaluate-run）；resume+force 且意图为 false 时在已校验样本目录内窄范围移除过期 counting_evaluation.json；合法零 Qwen resume 不改评估产物 |
| Fix E | **evaluate-run 持久化刷新报告 bundle**（build_report → persist_report_bundle）；deepseek_audit.jsonl 自然反映新 judge 的 counting/VQA 样本；持久化失败 = 稳定命令失败 |
| Fix F | **judge-vqa-run 持久化刷新报告 bundle**（同 helper），report.json/samples.csv/deepseek_audit.jsonl 反映新 judge 状态与 force 重判 |
| Fix G | **中性 CountTargetSpec 重建**：无精确持久化 hint 时只陈述已知 canonical 标签（`"Persisted inclusion rule unavailable."` / `"Persisted exclusion rule unavailable."`），绝不虚构 `exclude none`/`count all` 等未持久化规则；精确 hint 仍优先 |

## 权威矩阵（终态）

```text
fresh dataset run        -> 用户选项权威 -> canonicalize -> 持久化 run_request
resume-run               -> run_request 权威
run-dataset --resume     -> run_request 权威；偏离的 CLI 调用被拒绝
fresh count-image        -> 用户调用权威 -> 持久化
count-image --resume     -> 持久化 count-image 调用权威（含 evaluate 意图）
```

报告持久化唯一权威 helper：`reporting.exporters.persist_report_bundle`（
run-dataset / standard-evaluate / evaluate-run / judge-vqa-run 全部经它刷新；
只读命令仍只构建/读取）。

## 冻结不变式（保持）

- fresh 唯一 run / 显式重复失败 / resume 显式 id；绝不静默转 resume。
- 样本存储 `tasks/<run_task>/samples/<sha256[:24]>`；status.result_path 纯
  basename、prediction.result_path run 相对。
- Reporting 只读、零 Qwen/DeepSeek/重推理；只写报告输出目录。
- 路径安全：resolve + is_relative_to containment；公共错误无密钥/原始异常。
- 架构文件未改；无新生产文件；main.py 薄；evaluate-run/judge-vqa-run 零 Qwen。

## 验证

- 全量 pytest：1557 passed（新增 7 个 11G.5.1 测试）
- compileall 全绿；`git diff --check` 干净
- 各套件：architecture 40 / contracts 136 / workflows 212 / evaluation 79 /
  reporting 24 / application 127 / integration 41 / test_main 35
- Foundation 子集（architecture+contracts+parity+workflows+evaluation）全绿

## 验收

```text
TASK_11G_5_1_RESUME_REPORT_CONSISTENCY
DATASET_ROOT_CANONICALIZED=PASS
DATASET_EXECUTION_ROOT_MATCHES_PERSISTED=PASS
RESUME_RUN_USES_RUN_REQUEST=PASS
RUN_DATASET_RESUME_USES_RUN_REQUEST=PASS
RUN_DATASET_RESUME_ROOT_MISMATCH_REJECTED=PASS
RUN_DATASET_RESUME_TASK_MISMATCH_REJECTED=PASS
RUN_DATASET_RESUME_JUDGE_MISMATCH_REJECTED=PASS
COUNT_IMAGE_VALID_RESULT_REQUIRED_FOR_ZERO_QWEN_RESUME=PASS
COUNT_IMAGE_MISSING_RESULT_REEXECUTES=PASS
COUNT_IMAGE_CORRUPT_RESULT_REEXECUTES=PASS
COUNT_IMAGE_RESULT_SAMPLE_ID_MISMATCH_REEXECUTES=PASS
COUNT_IMAGE_RESUME_EVALUATE_FROM_RUN_REQUEST=PASS
COUNT_IMAGE_FORCE_EVALUATE_TRUE_REFRESHES_EVAL=PASS
COUNT_IMAGE_FORCE_EVALUATE_FALSE_REMOVES_STALE_EVAL=PASS
EVALUATE_RUN_PERSISTS_REPORT_BUNDLE=PASS
EVALUATE_RUN_REFRESHES_DEEPSEEK_AUDIT=PASS
EVALUATE_RUN_ZERO_QWEN=PASS
JUDGE_VQA_RUN_PERSISTS_REPORT_BUNDLE=PASS
JUDGE_VQA_RUN_REFRESHES_DEEPSEEK_AUDIT=PASS
JUDGE_VQA_RUN_ZERO_QWEN=PASS
COUNTING_TARGET_HINT_EXACT_PRESERVED=PASS
COUNTING_TARGET_FALLBACK_NEUTRAL=PASS
COUNTING_TARGET_FALLBACK_NO_INVENTED_EXCLUSION=PASS
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
