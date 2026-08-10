# 11H — Approve Explicit Dataset Utility Paths（架构变更）

## 目的

仅架构批准——不实现任何行为。批准三个显式数据集工具未来路径进入冻结
白名单，供后续 11H2 实施。

## 批准路径

```text
data/downloader.py
data/loader.py
application/commands/download_data.py
```

明确不添加 `data/loaders.py`（旧版别名，永久禁止）。

## 边界声明（批准时的契约）

### data/downloader.py

- 依赖：stdlib + 可选 huggingface_hub；
- 不得 import application / workflows / agents / models；
- 网络只发生在显式调用（绝不隐式下载、绝不导入副作用联网）；
- 失败为稳定错误，绝不泄露密钥/路径。

### data/loader.py

- 当前 Registry/Adapters 之上的便捷 API（不重复适配器逻辑）；
- 无隐式下载/网络；
- 返回当前 UnifiedSample（绝非旧 CanonicalSample）；
- 不选择 Agent、不调用模型、不修改任务语义。

### application/commands/download_data.py

- downloader 之上的薄适配命令（main.py 保持薄分发）；
- 公共错误只输出稳定类型名。

## 依赖 DAG

- `data` 包规则 `allow: ["data"]` 已覆盖 downloader/loader 的包内依赖；
  huggingface_hub 为第三方（不涉及内部 DAG），无需修改
  `architecture/import_rules.json`。
- `application`（composition root 宽权限）已允许 command 导入 downloader。

## 状态

- `architecture/allowed_python_files.txt`：3 个路径已加入（data 段 +
  application/commands 段）。
- `architecture/implementation_status.json`：`pending_files` 已声明 3 个路径
  （implemented ∪ pending 与既有架构测试语义一致；pending 为已批准未来
  路径，按 AGENTS.md 不得预先创建空壳，缺失合法）。
- 架构测试 `test_implementation_status_exact_coverage` 相应收窄 missing
  检查至 implemented 文件（pending 缺失合法），undeclared 检查保持
  （实际文件必须被 declared 覆盖）。

## 验收

```text
TASK_11H_ARCH_REPORT
ALLOWLIST_CHANGE_ISOLATED=PASS
ONLY_APPROVED_PATHS_ADDED=PASS
IMPORT_DAG=PASS
LEGACY_PACKAGES_STILL_FORBIDDEN=PASS
ARCHITECTURE_TESTS=PASS
READY_FOR_11H2
```
