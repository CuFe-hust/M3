# 11G.5.2.1 — Seam Verify Effective-Value Fidelity

## 目标

修复 count-image 持久化 CLI 原始 seam 意图（`not --no-seam-verify`）而非实际
生效值的单一复现性缺陷：`settings.counting.seam_verify` 可被配置修改，导致
快照与实际 fresh 执行不一致；force resume 可能因 config 漂移改变 seam 模式。

## 修复

| 修复 | 内容 |
|---|---|
| Fix A | fresh 只计算一次**有效值**：`effective_seam_verify = settings.counting.seam_verify and not args.no_seam_verify`（config 可禁用，CLI 只能进一步禁用，绝不能在 config 禁用时启用）——同一值持久化到 `RunRequest.count_seam_verify` 并用于执行 |
| Fix B | fresh 执行无条件用 `settings.model_copy(counting.seam_verify=effective_seam_verify)` 重建 runtime settings（不再只在 false 时覆盖） |
| Fix C | resume/force 在完整保真元数据校验后**无条件**以持久化值为权威重建 seam（`bool(request.count_seam_verify)`，true/false 两个方向都覆盖当前 config）——config 漂移绝不可能改变原始 seam 模式；不再依赖当前 config 作为回退 |

单一事实来源：fresh 一个 `effective_seam_verify` 值同时用于持久化与执行；
resume/force 一个 `request.count_seam_verify` 权威值用于重建。无两条独立
seam 推导路径。

## 真值表（冻结）

```text
config seam_verify | CLI --no-seam-verify | effective/persisted/executed
true               | false                | true
true               | true                 | false
false              | false                | false
false              | true                 | false
```

## 验证

- 全量 pytest：1569 passed（新增 6 个测试：4 个 fresh 真值表 + 2 个
  config-drift force resume——persisted false + config true → 执行 false；
  persisted true + config false → 执行 true，均经 Runtime.create spy 捕获
  `settings.counting.seam_verify` 验证，非仅断言 run_request.json）
- compileall 全绿；`git diff --check` 干净
- 架构文件未改；无新生产文件；count-image 其余保真（target/预算/render/
  evaluate）、counting judge 保真、dataset resume 预检均无回归

## 验收

```text
TASK_11G_5_2_1_SEAM_VERIFY_EFFECTIVE_VALUE
FRESH_CONFIG_FALSE_CLI_DEFAULT_PERSISTS_FALSE=PASS
FRESH_CONFIG_FALSE_CLI_DEFAULT_EXECUTES_FALSE=PASS
FRESH_CONFIG_TRUE_CLI_DEFAULT_PERSISTS_TRUE=PASS
FRESH_CONFIG_TRUE_CLI_DEFAULT_EXECUTES_TRUE=PASS
FRESH_CONFIG_TRUE_NO_SEAM_PERSISTS_FALSE=PASS
FRESH_CONFIG_TRUE_NO_SEAM_EXECUTES_FALSE=PASS
FRESH_CONFIG_FALSE_NO_SEAM_PERSISTS_FALSE=PASS
FRESH_CONFIG_FALSE_NO_SEAM_EXECUTES_FALSE=PASS
FORCE_PERSISTED_FALSE_CURRENT_CONFIG_TRUE_EXECUTES_FALSE=PASS
FORCE_PERSISTED_TRUE_CURRENT_CONFIG_FALSE_EXECUTES_TRUE=PASS
RUN_REQUEST_SEAM_EQUALS_FRESH_EXECUTION_SEAM=PASS
RESUME_SEAM_USES_PERSISTED_VALUE_UNCONDITIONALLY=PASS
CONFIG_DRIFT_CANNOT_CHANGE_COUNT_SEAM_MODE=PASS
COUNT_IMAGE_TARGET_FIDELITY_REGRESSION=PASS
COUNT_IMAGE_BUDGET_FIDELITY_REGRESSION=PASS
COUNT_IMAGE_RENDER_FIDELITY_REGRESSION=PASS
COUNTING_JUDGE_FIDELITY_REGRESSION=PASS
RUN_DATASET_RESUME_PREFLIGHT_REGRESSION=PASS
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
