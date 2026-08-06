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
| `models/base.py` | `RequestMeta`、`VisionLanguageClient`、`build_request_hash`、`sanitize_messages` | 模型客户端协议与请求哈希/脱敏 |
| `models/cache.py` | `CacheEntry`、`JsonResponseCache`、`ModelCacheError`、`CorruptCacheEntryError` | 原子文件缓存；损坏条目稳定错误 |
| `models/images.py` | `read_normalized_image`、`guess_image_mime`、`image_to_data_url`、`image_sha256` | 模型输入图像工具 |
| `models/settings.py` | `QwenSettings`、`DeepSeekSettings`、`ModelSettings` | 配置声明；默认离线（allow_download=False） |
| `models/entry.py` | `create_model`、`register`、`list_models` | 统一惰性模型工厂 |
| `models/qwen_transformers.py` | `QwenTransformersClient`、`QWEN_CLIENT_VERSION` | 本地 Transformers Qwen 客户端（一次加载/结构化 JSON/修复/缓存/产物/并发锁） |
| `models/qwen3_vl/baseline.py` | `Qwen3VLBaseline`、`Qwen3VLSettings` | Qwen3-VL 多模态文本基线封装 |
| `agents/schema.py` | `AgentName`、`VisualEvidence`、`AgentResult` | 通用 Agent 输出契约 |
| `agents/base.py` | `AgentContext`、`AgentExecution`、`Agent`、`CallBudget` | 执行上下文（含 data_root）与严格执行校验 |
| `agents/registry.py` | `AgentRegistry` | 纯注册/查询/supports/coverage |
| `agents/errors.py` | 9 个稳定错误类型 | 重复/未注册/任务不匹配/执行失败/检测器/可选依赖 |
| `agents/visual_base.py` | `VisualAgentBase`、`PromptBinding` | 数据集无关视觉 Agent 基类（返回 AgentExecution） |
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

## 依赖方向

- `data` 不依赖任何其他业务包；`models` 不依赖 data/agents；`agents` 可依赖
  `data.schema` 与 `models`；`VisualAgentBase` 返回 `AgentExecution`（payload 为
  `AgentResult`）。
- 新代码不得 import `spacers_agent`/`eval`。

## 尚未实现

具体领域 Agents（counting/spatial/change/grounding/caption/general_vqa）、
`routing`、`workflows`、`evaluation`、`reporting`、`application`、`main.py`
及对应目录尚未创建/实现；任务推进时逐层创建并更新本文件。
