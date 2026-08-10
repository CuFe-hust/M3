# 11G.5.2 — Final Invocation / Judge Fidelity Hardening

## 目标

关闭二轮评审发现的最后三个正确性缺口：count-image 未冻结全部行为影响调用、
离线 counting DeepSeek Judge 忽略 canonical `sample.normalization.count_target_hint`、
run-dataset `--resume` 在 Runtime.create() 之后才校验调用偏离（非法 resume 可能
先加载 Qwen 权重）。仅硬化；不涉及 11H / 12 / Spark 真机验收。

## 修复清单

| 修复 | 内容 |
|---|---|
| Fix A | RunRequest 扩展 count-image 保真字段：`count_target_spec`（结构化快照，**绝非主机路径**）+ `count_target_spec_hash`（canonical JSON 的 SHA256，sort_keys + 稳定分隔符）、`count_seam_verify`、`count_max_qwen_calls`、`count_max_deepseek_calls`、`count_render`；validator 约束（count-image 需样本身份、dataset run 不得携带保真字段、类型/边界校验）；**旧当前代 run_request 保持可读**（缺字段 = None） |
| Fix B | fresh 未提供的预算归一化为显式配置默认值（`settings.router.default_*`），使持久化快照恒为**完整可重跑**调用（可与旧运行区分）；resume/force 一律以持久化调用为权威（target/seam/预算/render/evaluate），CLI 行为标志被忽略（文档化）；旧 run 无保真快照：非 force 零 Qwen 复用仍允许（合法结果即可），force 重跑稳定失败 `COUNT_IMAGE_INVOCATION_METADATA_INCOMPLETE`（绝不猜测） |
| Fix C | overlay 意图一致性：持久化 render=true 时 force 重跑用新结果重渲染 overlay；render=false 时 force 重跑窄范围移除 stale overlay（已校验样本目录内）；零 Qwen resume 绝不改动渲染/评估产物 |
| Fix D/E/F | 离线 counting Judge 目标解析优先级对齐 CountingAgent：`sample.normalization.count_target_hint`（VRSBench 审计 hint 权威）→ legacy `metadata["count_target_hint"]` → 中性回退（只陈述已知标签，绝不虚构规则）；VRSBench 风格 normalization 的 canonical label/aliases/inclusion/exclusion 原样传给 `judge_counting`；无效持久化 hint 降级中性回退（稳定处理） |
| Fix G | 新增 `preflight_dataset_resume(options, runs_root, project_root)`（复用中央化 `reconstruct_dataset_resume_options` + `_validate_resume_match`）：run-dataset `--resume` 在 **Runtime.create() 之前**完成全部调用校验（root/task/judge 等偏离 → 稳定失败、零模型构造）；Runtime 内保留同一校验作为纵深防御 |

## 权威矩阵（终态）

```text
fresh dataset run        -> 用户选项权威 -> canonicalize -> 持久化 run_request
resume-run               -> run_request 权威
run-dataset --resume     -> run_request 权威；CLI 预检在 Runtime.create 前拒绝偏离
fresh count-image        -> 用户调用权威 -> 完整保真快照持久化（含结构化 target/hash/seam/预算/render/evaluate）
count-image --resume     -> 持久化调用权威（含 evaluate/render 意图）
count-image --resume --force -> 持久化调用权威 -> 同一语义调用重跑
旧当前代 count-image run -> 非 force 零 Qwen 复用允许；force 重跑稳定拒绝（保真元数据不完整）
```

fresh 专用调用选项（resume 时忽略，事后评估归 evaluate-run、事后渲染可用
render-count）：`--target-spec` / `--no-seam-verify` / `--max-qwen-calls` /
`--max-deepseek-calls` / `--evaluate` / `--render`。

## 冻结不变式（保持）

- fresh 唯一 run / 显式重复失败 / resume 显式 id；count-image 零 Qwen 复用需
  status=succeeded **且** 合法匹配 CountingResult。
- 报告持久化唯一 helper `persist_report_bundle`；Reporting 只读、零模型。
- 路径安全（resolve + containment、纯 basename、run 相对）与公共错误契约不变。
- 架构文件未改；无新生产文件；main.py 薄；evaluate-run/judge-vqa-run 零 Qwen。

## 验证

- 全量 pytest：1563 passed（新增 6 个 11G.5.2 测试）
- compileall 全绿；`git diff --check` 干净
- 各套件：architecture 40 / contracts 136 / workflows 212 / evaluation 79 /
  reporting 24 / application 133 / integration 41 / test_main 35
- Foundation 子集（architecture+contracts+parity+workflows+evaluation）全绿

## 验收

```text
TASK_11G_5_2_FINAL_INVOCATION_JUDGE_FIDELITY
COUNT_IMAGE_TARGET_SPEC_PERSISTED=PASS
COUNT_IMAGE_TARGET_SPEC_PATH_NOT_AUTHORITATIVE=PASS
COUNT_IMAGE_TARGET_SPEC_HASH_STABLE=PASS
COUNT_IMAGE_SEAM_VERIFY_PERSISTED=PASS
COUNT_IMAGE_QWEN_BUDGET_PERSISTED=PASS
COUNT_IMAGE_DEEPSEEK_BUDGET_PERSISTED=PASS
COUNT_IMAGE_RENDER_INTENT_PERSISTED=PASS
COUNT_IMAGE_EVALUATE_INTENT_PRESERVED=PASS
COUNT_IMAGE_FORCE_USES_PERSISTED_TARGET=PASS
COUNT_IMAGE_FORCE_USES_PERSISTED_SEAM_MODE=PASS
COUNT_IMAGE_FORCE_USES_PERSISTED_BUDGETS=PASS
COUNT_IMAGE_FORCE_USES_PERSISTED_RENDER_INTENT=PASS
COUNT_IMAGE_TARGET_SOURCE_FILE_DELETE_SAFE=PASS
COUNT_IMAGE_TARGET_SOURCE_FILE_CHANGE_IGNORED_ON_RESUME=PASS
COUNT_IMAGE_FORCE_RENDER_TRUE_REFRESHES_OVERLAY=PASS
COUNT_IMAGE_FORCE_RENDER_FALSE_REMOVES_STALE_OVERLAY=PASS
COUNT_IMAGE_ZERO_QWEN_RESUME_DOES_NOT_MUTATE_OVERLAY=PASS
OLD_COUNT_IMAGE_RUN_ZERO_QWEN_RESUME_SUPPORTED=PASS
OLD_COUNT_IMAGE_RUN_FORCE_RESUME_REJECTED_IF_METADATA_INCOMPLETE=PASS
COUNTING_JUDGE_NORMALIZATION_HINT_FIRST=PASS
COUNTING_JUDGE_METADATA_HINT_SECOND=PASS
COUNTING_JUDGE_NEUTRAL_FALLBACK_THIRD=PASS
VRSBENCH_COUNTING_JUDGE_CANONICAL_LABEL_PRESERVED=PASS
VRSBENCH_COUNTING_JUDGE_INCLUSION_RULE_PRESERVED=PASS
VRSBENCH_COUNTING_JUDGE_EXCLUSION_RULE_PRESERVED=PASS
EVALUATE_RUN_ZERO_QWEN=PASS
RUN_DATASET_RESUME_PREFLIGHT_BEFORE_RUNTIME_CREATE=PASS
RUN_DATASET_RESUME_ROOT_MISMATCH_ZERO_MODEL_LOAD=PASS
RUN_DATASET_RESUME_TASK_MISMATCH_ZERO_MODEL_LOAD=PASS
RUN_DATASET_RESUME_JUDGE_MISMATCH_ZERO_MODEL_LOAD=PASS
RUN_DATASET_RESUME_MATCHING_INVOCATION_PROCEEDS=PASS
RUNTIME_RESUME_DEFENSE_IN_DEPTH=PASS
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
