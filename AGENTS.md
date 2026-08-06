# AGENTS.md

本文件约束编码代理在本仓库的工作方式。本仓库正在从零重建新架构（分支
`new_structure`），行为参考是 `try_yolo` 分支的锁定提交
`ec962eb87c3ad0b8c1502efcbd08db0daec48868`。`new_structure` 与 `try_yolo`
没有共同祖先，功能是否遗漏必须依赖基线文档、Golden fixtures 与行为测试判断，
不能依赖普通 Git diff。

## 硬性规则

1. 新代码不得 import `spacers_agent` 或旧 `eval`（本分支永久禁止旧包目录存在）。
2. 每次只执行一个任务；任务完成并验证通过后再进入下一个任务。
3. `architecture/allowed_python_files.txt` 是**最终架构白名单**（冻结文件），
   普通实施任务**禁止修改**；当前任务需要的文件不在白名单时必须停止并报告，
   等待用户批准独立架构变更任务（见 `architecture/ALLOWLIST_CHANGE_POLICY.md`）。
   白名单中尚未创建的文件是已批准的未来路径，不得预先创建空壳。
4. `architecture/implementation_status.json` 记录当前实际实现状态；
   实际存在的生产 .py 必须被它精确声明（implemented ∪ pending）。
5. 不得修改 Golden fixtures（`tests/fixtures/migration/`）来迁就新实现；
   行为变化必须以"有意变化"记录在 `docs/migration/`。
6. 不得默认联网：不下载模型、不下载数据集、不调用云 API。
7. 必须运行任务要求的全部测试，并如实报告结果；未运行的验证必须说明原因。
8. 不得删除或弱化既有测试；不得用 `pytest.skip` 掩盖实现缺失。
9. `__init__.py` 只做导出（docstring / import / __all__ / TYPE_CHECKING），
   不得定义函数或类，不得有条件注册或副作用。
10. 不得使用 `sys.path` 修改、动态 import 绕过或 `try/except ImportError`
    在新旧实现间回退。
11. 所有文本文件 UTF-8；注释使用英文 + 中文双语（与既有文件一致）。

## 工作流

- 参考基线只读：`git show ec962eb87c3ad0b8c1502efcbd08db0daec48868:<path>`
- 每个任务开始前：`git status --short`、`git rev-parse HEAD`、运行任务要求的
  修复前测试并记录结果。
- 完成后运行：`git diff --check`、`git status --short`、任务指定 pytest，
  并按全局契约的固定格式汇报。
