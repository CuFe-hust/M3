# DETAILS.md — 当前有效的模块所有权与接口契约

本文件记录当前已实现模块的所有权与契约；未列出的领域**尚未实现**。

## 当前模块所有权

| 文件 | 职责 |
|---|---|
| `architecture/allowed_python_files.txt` | **最终架构批准路径**（冻结）：整个新架构最终允许出现的全部 Python 路径；未创建的未来路径不代表已实现 |
| `architecture/implementation_status.json` | **当前实际实现状态**：implemented_files（存在且非空）与 pending_files；实际生产 .py 必须被精确声明 |
| `architecture/ALLOWLIST_CHANGE_POLICY.md` | 白名单变更政策（普通任务禁止修改白名单） |

| 模块 | 拥有类型/函数 | 说明 |
|---|---|---|
| `data/schema.py` | `TaskName` | 10 个公开任务名（Literal） |
| `data/schema.py` | `ImageRef` | 不可变图像引用；path 统一 posix 序列化；sha256 严格 64 位 hex |
| `data/schema.py` | `GroundTruth` | answers/count/boxes(4|8)/points(2)/labels/raw/coordinate_frame |
| `data/schema.py` | `TaskNormalization` | 一等规范化字段（结构化 spatial_query/answer_constraints/count_target_hint） |
| `data/schema.py` | `UnifiedSample` | 主样本契约；时相角色、question、normalization 一致性校验 |
| `data/schema.py` | `ValidationIssue` | 只读审计问题记录 |
| `data/schema.py` | `stable_sample_id` | 多图稳定样本 ID；source ID 目录名安全检查 |
| `data/__init__.py` | 重导出 | 仅导出上述稳定类型 |

## 关键约定

- 所有自由字段（`metadata`、`raw`、normalization 结构化字段）仅允许 JSON-safe
  值（含有限数值）；拒绝 Path、PIL 对象、set、bytes、callable。
- 所有模型 `extra="forbid"`；`ImageRef` 为 `frozen=True`。
- 变化任务（change_caption/change_qa）：images 必须 [t1, t2, context*]；
  非变化任务：首图 image，其后只允许 context。
- `question` 语义：caption/change_caption 可为空，其余任务必须非空。
- `UnifiedSample.normalization.normalized_task` 必须等于 `sample.task`。
- 依赖方向：`data` 不依赖任何其他业务包；新代码不得 import `spacers_agent`/`eval`。

## 尚未实现

`models`、`agents`、`routing`、`workflows`、`evaluation`、`reporting`、
`application`、`main.py` 及对应目录尚未创建/实现；任务推进时逐层创建并更新本文件。
