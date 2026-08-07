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
| `models/base.py` | `ModelCacheIdentity`、`CacheIdentifiedClient`、`MissingModelCacheIdentityError`、`require_model_cache_identity`、`VisionLanguageClient`、`RequestMeta`、`build_request_hash`、`sanitize_messages` | 模型客户端协议、缓存身份校验（全仓唯一权威）与请求哈希/脱敏 |
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
| `agents/general_vqa/agent.py` | `GeneralVQAAgent` | 通用/场景分类/多选题 VQA；postprocess 强制 MCQ 输出符合 choices 约束 |
| `agents/caption/agent.py` | `CaptionAgent` | 仅 caption；trace 含稳定 agent class/route |
| `agents/grounding/agent.py` | `GroundingAgent` | 仅 grounding；completed 必须携带合法定位证据（0..999） |
| `agents/counting/schema.py` | 计数域全部契约 | PixelRect/TileSpec/观测/目标/结果；final_count 强绑定 accepted points |
| `agents/counting/settings.py` | `CountingSettings`、`AgentCountingSettings`、`YoloDetectorSettings`、`YoloCountingSettings` | 计数确定性默认；YOLO 声明纯结构校验（含 require_cuda/allow_cpu_fallback） |
| `agents/counting/geometry.py` | 切片/坐标换算/owner 规则/`cores_are_neighbours` | 纯确定性；crop_for_tile 接收已打开图片 |
| `agents/counting/evidence.py` | box→点/去重/边界残片/数量解析 | 通用标签归一化，无数据集分支 |
| `agents/counting/point_pipeline.py` | `PointCountingOrchestrator`、`find_boundary_conflicts`、`decide_seam_pairs`、`finalize_representatives` | tile 回调协议、递归分割、seam 最终化接入主流程 |
| `agents/counting/target_parser.py` | `CountTargetParser`、`InvalidCountTargetHintError` | 目标解析优先级：normalization hint → legacy metadata → Qwen |
| `agents/counting/agent.py` | `CountingAgent` | task gate / target parse / plan / request / 调用 executor / 打包 AgentExecution 与公共 trace；主 payload 恒为 CountingResult；AgentResult 仅附加 |
| `agents/counting/executor.py` | `CountingPlanExecutor`、`CountingExecutionResult`、`CountingExecutionPolicy` | BackendPlan 运行时执行：primary 调用、unavailable/runtime 回退、zero review；结构化执行状态，无 8 元 tuple |
| `agents/counting/backends/base.py` | 协议与类型 | CountingRequest/BackendPlan/Outcome；is_enabled/is_available 分离；稳定错误；`require_model_cache_identity`/`MissingModelCacheIdentityError` 重导出 models.base 同一对象 |
| `agents/counting/backends/registry.py` | `BackendRegistry` | 稳定注册顺序、重复检测、数据集中性命名 |
| `agents/counting/backends/selector.py` | `BackendSelector` | 仅按 mode/task/target/hints 计划；计划期不隐藏不可用检测器 |
| `agents/counting/backends/qwen_point.py` | `QwenPointCountingBackend` | tile 点计数；完整 cache identity；response schema 入 hash |
| `agents/counting/backends/quantity_proposal.py` | `QuantityProposalBackend` | 提议+定位；无可靠 hint 拒绝 supports；恢复仅限当前请求 hash 目录 |
| `agents/counting/backends/yolo_model_store.py` | `YoloModelStore` | per-key 并发加载一次；hash/task/class map 校验 |
| `agents/counting/backends/yolo_obb.py` | `YoloOBBCountingBackend` | OBB 计数；alias/composite；边界去重；provider 审计 |
| `agents/counting/backends/yolov5_obb_onnx.py` | `YoloV5ObbOnnxModel` | 惰性 ONNX；require_cuda/allow_cpu_fallback；device 校验 |
| `agents/counting/backends/yolo_adapter.py` | `OBBDetection`、`UltralyticsOBBModelAdapter` | 统一 OBB 输出 |
| `routing/schema.py` | `SampleCapabilities`、`RoutePolicy`、`RoutingDecision`、`TaskResolutionRequest`、`TaskResolution`、`ResolutionSource` | 路由契约 + 样本前任务解析契约；Router 不读 question、不调用模型；TaskResolution* 仅供 TaskResolver 使用 |
| `routing/policies.py` | `POLICIES`、`policy_for` | 固定 task→policy 表；未知 task 显式失败 |
| `routing/router.py` | `TaskRouter` | 同步确定性路由；requires_tiling 为策略字段 |
| `workflows/task_resolver.py` | `TaskResolver`、`TaskResolutionError` | 样本前任务解析：explicit/rule/model 三路径；显式 task 不调用模型；空问题仅两条窄规则；低置信度只返回结构化候选，不执行 Agent |
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

- `data` 不依赖任何其他业务包；`models` 不依赖 data/agents。
- `agents` 只依赖 `models.base`/`models.images`（模型协议与纯工具），
  禁止 `models.entry`/`models.qwen_transformers`/`models.qwen3_*`——
  Agent 不得自己创建具体模型。
- `routing` 不依赖 models（可依赖 `data.schema` 与 `agents.schema`）；
  `TaskResolver` 在 workflows，不需要放宽 routing。
- `workflows` 只依赖模型协议（`models.base`），禁止具体 Qwen 实现；
  具体模型创建属于 composition root（`application/bootstrap.py`、
  `application/runtime.py`）。
- `evaluation.metrics`（path rule）无模型依赖；`evaluation.judges`（path
  rule）可依赖批准的 `models.base`/`models.cache`/`models.settings`；
  path rule 优先于 package 规则，无匹配时回退。
- `reporting` 只依赖 schema/result 层，不执行 Agent。
- `application` 是 composition root，可继续拥有宽 import 权限（settings、
  model factory、registry/workflow assembly、command wiring）。
- `VisualAgentBase` 返回 `AgentExecution`（payload 为 `AgentResult`）。
- 新代码不得 import `spacers_agent`/`eval`。

## 关键运行契约

- `TaskRouter.route` 为同步方法，绝不读取 question 或调用模型；
  `TaskResolver`（workflows）与 `TaskRouter`（routing）职责严格分离：
  Resolver 回答“这是什么任务”，Router 回答“这个已知任务交给哪个 Agent”。
- `TaskResolver` 三路径：显式 task（不调用模型、不消费 budget，非法显式
  task 以 `UNKNOWN_EXPLICIT_TASK` 稳定失败）；空问题仅两条确定性规则
  （1 图→`caption`、2 图→`change_caption`，其余图像数以
  `EMPTY_UNRESOLVABLE_REQUEST` 失败，绝不猜 general_vqa）；缺失 task 且
  有 question 才调用模型（完整 `ModelCacheIdentity` + response schema 入
  hash，`MODEL_IDENTITY_REQUIRED`/`MODEL_RESOLUTION_FAILED` 稳定错误）。
- `TaskResolver` 低置信度只返回结构化候选（`needs_candidate_fallback` +
  含 `general_vqa` 的最多 3 个候选），多 Agent 兜底执行留给 Task 34 的
  SampleRunner，Resolver 自身绝不执行任何业务 Agent。
- `CountingAgent` 主 payload 恒为 `CountingResult`，主文件名恒为
  `counting_result.json`；`AgentResult`（来自后端或 `answer_as_agent_result`
  开关）只写入 `additional_results["agent_result.json"]`（JSON-safe dict）。
- 计数后端使用显式 `BackendKind`（`qwen_point`/`quantity_proposal`/`yolo_obb`）；
  绝不通过 name/类名/模块路径推断类型。只有 `yolo_obb` 进入 detector plan、
  zero-review 与 detector fallback；`quantity_proposal` 不是 detector。
- `CountingBackendUnavailableError` 全仓唯一权威类位于 `agents.errors.py`；
  `agents` 顶层导出与 `agents.counting.backends.base` 导入为同一对象。
- 计数目标解析优先级：`sample.normalization.count_target_hint` →
  `metadata["count_target_hint"]`（兼容）→ Qwen target parser；无效 hint
  抛出 `InvalidCountTargetHintError`，绝不静默吞掉。
- Counting 公共入口只抛稳定错误（`AgentTaskMismatchError`、
  `CountingBackendUnavailableError`、`AgentExecutionError` + 稳定 cause code
  `DATA_ROOT_REQUIRED`/`IMAGE_PATH_ESCAPE`/`IMAGE_NOT_FOUND`/`IMAGE_READ_FAILED`/
  `TARGET_PARSE_FAILED`/`PRIMARY_BACKEND_FAILED`/`FALLBACK_BACKEND_FAILED`/
  `INVALID_BACKEND_KIND`）；trace/warnings 不含原始异常文本、绝对路径、密钥
  或 Base64（`fallback_reason_code` + `fallback_error_type`）。
- Backend plan 与 runtime fallback 职责分离：`BackendSelector.plan` 只基于
  配置/支持性规划；运行时权重/依赖就绪由 count 时验证，不可用经
  `CountingPlanExecutor` 显式回退（fallback 只能由 Executor 执行，
  Backend 不自行切换）。
- seam finalization（`find_boundary_conflicts` → `decide_seam_pairs` →
  `finalize_representatives`）由 `PointCountingOrchestrator.count_image` 执行，
  `CountingSettings.seam_verify` 控制开关。
- YOLO 模型按 (path, sha256) per-key 并发加载一次；ONNX provider 可配置：
  `require_cuda=True` 要求非负整数 `device`（映射到 CUDA `device_id`），
  `require_cuda=False` 要求 `device="cpu"` 且 Session 只请求 CPU；trace 记录
  requested/resolved provider 与 device。
- 所有 Counting 模型调用使用真实 `ModelCacheIdentity`
  （`require_model_cache_identity` helper 权威定义于 `models/base.py`，鸭子
  类型身份明确失败）并把 response schema 纳入 request hash。
- YOLO tile 失败只写稳定 warning（code/tile_id/exception type，无原始异常文本）；
  全部 tile 失败时 backend 抛 `DetectorInferenceError("ALL_YOLO_TILES_FAILED")`，由
  CountingPlanExecutor 决定显式 fallback；部分 tile 成功返回 partial 并保留证据。
- ONNX 显式 CPU-only（`require_cuda=False, device="cpu"`）与 CUDA fallback 语义分离：
  predict 校验与初始化 device 一致，CPU-only 不受 allow_cpu_fallback 门控。
- 未知 backend kind 以固定公共错误失败（`INVALID_BACKEND_KIND`），绝不回显原始
  name/kind 值。

## Task 26–33 运行时完整性（33.5）契约

- **Change 可选依赖**：`agents.change` 基础导入不加载 cv2/numpy（惰性
  `_require_cv2`/`_require_numpy`，缺失抛 `OptionalDependencyMissingError`）；
  `[change]` extra 声明 `numpy>=1.26` + `opencv-python-headless>=4.10`；
  `PairValidator` 导入不要求 cv2。
- **Spatial MIME**：candidate review 从真实内容 `detect_image_mime` 检测 MIME，
  绝不按后缀猜测；损坏/未知图像导致 review 失败时保留初次结果、
  `status=partial`、只记录稳定 error type；缺失 `ModelCacheIdentity` 仍直接失败。
- **Spatial canonical label**：`canonical_answer` 只做归一化，不做英语单复数
  猜测（`bus`→`bus`、`glass`→`glass`、`trucks`→`trucks`）。
- **Change invalid pair**：`preprocess.validation.valid == False` 时
  `ChangeAgent` 在构建证据/消费 budget/调用模型前抛
  `AgentExecutionError(cause="INVALID_CHANGE_PAIR")`；单图/乱序/错误角色已在
  `data.schema` 层拒绝。
- **Workflow 文件名安全**：`write_evaluation` 复用 `agents.base` 的 POSIX+Windows
  basename 契约并额外拒绝控制字符，路径类文件名在 I/O 前拒绝；`additional_results`
  文件名由 `AgentExecution` 构造时同规则校验。
- **JSONL 并发**：`events.jsonl` / `predictions.jsonl` 的
  read-compose-write-replace 在按路径进程内锁内执行——单进程并发 writer 安全；
  **跨进程并发追加不受当前工作流层支持**（文档契约，不声称通用原子追加）。
- **递归敏感扫描**：`workflows/events._reject_secrets` 是唯一实现（`RunStore`
  复用），递归拒绝 10 个敏感键（含 `image_data_url`）与 4 个值前缀
  （`sk-`/`Bearer `/`data:image/`/`-----BEGIN PRIVATE KEY-----`），错误消息不回显
  违规值。
- **统一 EvaluationRecord**：`EvaluationTask = counting|general_vqa|grounding|caption`；
  typed deterministic metrics（`Count`/`VQA`/`Grounding`/`Caption`）；`VQAEvaluationRecord`
  保留为兼容包装（归入 `general_vqa`）；`aggregate()` 覆盖四个已实现任务且
  mixed-task 显式失败；judge 只能旁路记录，永不覆盖 deterministic 指标。
- **CI/打包**：compileall 覆盖 workflows/evaluation；CI 安装 `[dev,migration,change]`
  并运行 `tests/workflows`、`tests/evaluation`；clean wheel smoke 在源码树外验证
  `agents.spatial`/`agents.change`（无 cv2）/`workflows`/`evaluation` 导入。

## Task 33.6 运行时不变式

- **run_id 路径安全**：`RunStore.create_run` 在 `_validate_run_id` 中校验用户
  提供的 run_id 为跨平台 plain identifier（字符集 `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`，
  拒绝 `.`/`..`/绝对路径/drive/UNC/NUL/CR/LF），任何 mkdir/write 前失败；
  `_new_run_id()` 生成值同样满足该校验；错误消息固定且不回显原始输入。
- **密钥扫描泛化**：`workflows/events._reject_secrets` 递归处理任意
  `Mapping`/`Sequence`（str/bytes/bytearray 不作为 generic sequence），tuple
  嵌套不再绕过；`EventWriter` 与 `RunStore` 共用同一实现。
- **EvaluationRecord task/metrics 不变式**：构造时强制
  `deterministic_metrics` 类型与 `task` 匹配（`EXPECTED_METRICS` 映射），
  `metrics=None` 仍合法；错误消息不 dump 指标载荷；counting/VQA/grounding/
  caption 聚合器全部 fail-closed，不再用 `getattr(..., default)` 静默降级。
- **VQA canonical merge 收敛**：`merge_vqa_evaluation` 返回统一
  `EvaluationRecord(task="general_vqa", deterministic_metrics=VQADeterministicMetrics(...))`；
  `VQAEvaluationRecord` 仅作 legacy 兼容（读取旧产物/显式 `to_evaluation_record`
  转换），`aggregate_vqa` 显式兼容两者；judge 仍只旁路记录。
- **Grounding 阈值单次判定**：`grounding_deterministic_metrics` 存储未舍入的
  原始 IoU，`iou_at_0_5` 为唯一阈值权威；`aggregate_grounding` 的 accuracy
  直接复用存储标志，绝不二次比较；0.5 邻域边界（0.4999994/0.4999996/0.5/
  0.5000004）测试保证 record 与 aggregate 自洽。
- **JSONL 锁路径身份**：per-path 锁键使用 `resolve(strict=False)` 的规范化
  路径身份，词法别名（`a/../events.jsonl` 与 `events.jsonl`）共享同一把锁；
  解析仅用于锁身份，不改变实际存储位置；并发范围仍为 single Python
  process only；原子替换对 Windows 瞬态文件锁做有限重试。

## Task 33.7 安全收敛

- **Windows 保留 run id**：`_validate_run_id` 显式拒绝 Windows 保留设备名
  （`_WINDOWS_RESERVED_STEMS`：CON/PRN/AUX/NUL/COM1-9/LPT1-9，含
  `CON.txt`/`COM1.json`/`LPT9.log` 等带扩展名形式，按 stem 匹配、大小写
  不敏感）与尾点/尾空格形式；所有平台确定性拒绝（纯 Python 单测验证，
  不依赖 CI 平台）；pre-I/O 失败顺序不变，错误消息固定且不回显输入。
- **set/frozenset 密钥扫描**：`_reject_secrets` 增加 `Set` 分支（set 与
  frozenset 均覆盖），set→tuple→string 完整递归；EventWriter 与 RunStore
  同一实现；含密钥的 set 在任何 JSON 序列化错误之前以 `sensitive value`
  失败（不依赖序列化报错作为防线）。
- **指标注册表不可变**：`EXPECTED_METRICS` 改为 `MappingProxyType` 且不再
  从 `evaluation` 顶层导出（`__all__` 移除）；`EvaluationRecord` validator
  行为不变。
- **Wheel 公共 API 契约**：clean wheel smoke 验证 Task 33.6 新公共 API
  （`VQADeterministicMetrics`/`GroundingDeterministicMetrics`/
  `merge_vqa_evaluation`），并断言 canonical VQA merge 返回统一
  `EvaluationRecord`（task="general_vqa"、类型化 deterministic metrics）。

## 尚未实现

`reporting`、`application`、`main.py` 尚未创建/实现；Task 34 尚未开始；
`SampleRunner`/`DatasetRunner` 尚未实现（`TaskResolver` 尚未被 dataset
runner 使用）；任务推进时逐层创建并更新本文件。
