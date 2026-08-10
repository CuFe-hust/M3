# 11I — Approve LEVIR Harmonization Evaluator（架构变更）

## 目的

仅架构批准——不实现任何行为。批准 LEVIR 协调评估脚本未来路径进入冻结
白名单，供后续 11I2 实施。

## 批准路径

```text
scripts/evaluate_levir_harmonization.py
```

## 边界声明（批准时的契约）

允许 import：

- 当前 change harmonizer/settings（`agents.change.harmonizer` /
  `agents.change.settings` 等现有实现）；
- 只读数据工具（`data.*` 只读路径）；
- stdlib；
- 可选 numpy / cv2 / PIL。

禁止 import：

- application runtime（`application.*`）；
- models / Qwen（`models.*`）；
- Judge / DeepSeek（`evaluation.judges.*`、`workflows.judge_service`）；
- legacy 包（`spacers_agent/`、`eval/`）。

## 依赖 DAG 说明

`scripts/` 不在 `architecture/import_rules.json` 的包 DAG 规则内
（`test_package_discovery` 显式排除 scripts）；`generate_migration_fixtures.py`
同属此先例。因此本任务无需修改 import_rules——上述 import 边界由文档声明，
11I2 实现时遵守并以测试验证。

## 状态

- `architecture/allowed_python_files.txt`：`scripts/evaluate_levir_harmonization.py`
  已加入（Migration tooling 段）。
- `architecture/implementation_status.json`：`pending_files` 声明该路径
  （已批准未来路径，按 AGENTS.md 不得预先创建空壳，缺失合法）。

## 验收

```text
TASK_11I_ARCH_REPORT
ALLOWLIST_CHANGE_ISOLATED=PASS
LEVIR_EVAL_PATH_APPROVED=PASS
NO_RUNTIME_MODEL_DEPENDENCY=PASS
ARCHITECTURE_TESTS=PASS
READY_FOR_11I2
```
