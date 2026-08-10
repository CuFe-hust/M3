# 11J.1 — Parity Fixture Environment Stability

## 根因

`tests/fixtures/parity/run_init_manifest.json` 在 dirty 工作区生成，冻结了
`"git_dirty": true`；GitHub Actions 使用 clean checkout（`git_dirty: false`），
导致 golden 比较跨环境失败（`test_parity_run_init_manifest`）。运行时行为
正确——manifest 真实记录 Git 状态；错的是把环境 provenance 当作稳定 parity
键。

## 修复（仅测试与 fixture，零生产代码）

| 项 | 变更 |
|---|---|
| `_parity_normalize` | `_PARITY_DROP_KEYS` 增加 `git_dirty`（执行/环境 provenance，不属于稳定 functional parity；仅该字段，不扩大范围——git_commit/config_hash 等既有规则保持现状） |
| `test_parity_run_init_manifest` | 改为两层断言：①字段契约——`"git_dirty" in manifest` 且 `isinstance(manifest["git_dirty"], bool)`；②稳定 parity——`_parity_normalize(manifest) == fixture`（排除 git_dirty） |
| `tests/fixtures/parity/run_init_manifest.json` | 移除 `git_dirty`（方案 A：fixture 只保存稳定 functional facts） |
| 新增回归测试 | `test_parity_normalization_ignores_git_dirty_environment_state`：仅 git_dirty 不同的两个 manifest 规范化后相等；且原对象不被原地修改（纯函数契约） |

## 验证

- `test_parity_run_init_manifest` PASS
- `tests/application/test_runtime.py -k parity` PASS
- `tests/parity`（含 baseline golden fixtures）PASS
- architecture / application / 全量 pytest PASS（1595）
- compileall、`git diff --check` 干净
- GitHub Foundation Tests 对精确最新远端 HEAD 通过后恢复
  `FULL_FUNCTIONAL_PARITY=PASS`；`FINAL_LIVE_GATE` 保持
  BLOCKED(ENVIRONMENT_BLOCKER)（Spark/Qwen checkpoint/真实数据集未就位）。

## 禁止事项遵守

- 未修改生产代码（run_init/runtime/run_store/schema 均未动）
- 未把 git_dirty 固定为常量、未删除 manifest 字段
- 未改架构白名单/import 规则；未新增生产文件
