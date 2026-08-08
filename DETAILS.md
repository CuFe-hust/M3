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
| `workflows/task_resolver.py` | `TaskResolver`、`TaskResolutionError`、`materialize_sample`、`SampleMaterializationError` | 样本前任务解析：explicit/rule/model 三路径；显式 task 不调用模型；空问题仅两条窄规则；低置信度只返回结构化候选，不执行 Agent；`materialize_sample` 将 draft 物化为 UnifiedSample（角色重建、normalization=None、稳定不兼容错误） |
| `data/schema.py` | `ImageRef` | 不可变图像引用；path 统一 posix 序列化；sha256 严格 64 位 hex |
| `data/schema.py` | `GroundTruth` | answers/count/boxes(4|8)/points(2)/labels/raw/coordinate_frame |
| `data/schema.py` | `TaskNormalization` | 一等规范化字段（结构化 spatial_query/answer_constraints/count_target_hint） |
| `data/schema.py` | `UnifiedSample` | 主样本契约；时相角色、question、normalization 一致性校验 |
| `data/schema.py` | `ValidationIssue` | 只读审计问题记录 |
| `data/schema.py` | `stable_sample_id` | 多图稳定样本 ID；source ID 目录名安全检查 |
| `data/__init__.py` | 重导出 | 仅导出上述稳定类型 |
| `data/schema.py` | `SampleDraft` | pre-sample 契约（无 task 角色校验）；task 保持必填于 UnifiedSample |
| `evaluation/judges/base.py` | `DeepSeekJudgeResult`、`VQAAnswerJudgeResult`、`JudgeClient`、`CountEvidence`/`CountTarget`（结构子集协议）、`build_*_judge_payload`、`build_*_judge_request_hash`、`stable_error_label` | judge Schema/协议/纯载荷与稳定哈希；载荷绝不包含图像数据或路径；judges 层不导入 `agents.counting.schema`（结构协议消费计数证据） |
| `evaluation/judges/deepseek.py` | `DeepSeekJudgeClient`、`DeepSeekJudgeError`、`JudgeTransportError`、`urllib_judge_transport` | 标准库 HTTP 仅文本客户端：缓存/修复一次/退避重试/产物；api_key 注入不读 env；公共错误只含固定 code |
| `workflows/judge_service.py` | `JudgeService` | 策略（none/errors-only/all）+ 预算（真正发起时才 `reserve_deepseek`）+ 合并（judge 永不覆盖确定性指标）；`judge_vqa_resume` 已成功不重复、缺失/损坏/failed 可补 |
| `workflows/sample_runner.py` | `SampleRunner`、`sample_state_from_payload`、`failed_sample_status` | 单样本执行内核：attempt plan（低置信度候选 ≤3、AgentName 稳定去重）、routing fallback、partial 策略、共享逐样本预算（可外部注入）、确定性评估（VQA/counting/grounding/caption 四类产物，fail-closed 不伪造）、可选 VQA judge（`asyncio.to_thread` 不阻塞 loop）、trace（`resolved_task`/`execution_task`）、失败只记录稳定 code |
| `data/adapters/manifest.py` | `ManifestDraftAdapter`、`iter_manifest_drafts`、`load_manifest_mapping` | manifest 驱动 draft 适配器（`spacers_adapter.json` 显式字段映射、JSON/JSONL、task 列可选、不猜字段、不调模型、不 import workflows/models、绝不写 run artifacts）；`samples_file` 经 `resolve_dataset_relative_path` 限制在 dataset root 内；失败均为稳定 DatasetProbeError |
| `workflows/dataset_runner.py` | `DatasetRunner`、`select_samples`、`storage_key`、`ResumeSupplementError` | 数据集编排：probe 经 `ArtifactWriter.write_dataset_probe` 独立写 `dataset_probe.json`（manifest.json 绝不触碰）、固定 selection 顺序（SHA256 分片）、resume 只跳过 succeeded 并按 `status.task`（执行任务）补判缺失确定性评估/缺失或失败 VQA judge（异常→skipped 稳定 code）、单进程 asyncio 并发、fail-fast 取消不遗留 running（已取消 `FAIL_FAST_CANCELLED`、未启动 `FAIL_FAST_NOT_STARTED`，全部计入 predictions 与终态）、`DatasetRunSummary` 计数强制闭合；`task=None` 为内部显式 auto-task mode；目录 `tasks/<task>/samples/<sha256[:24]>` |

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

- **Judge 层（Task 03）**：`DeepSeekJudgeClient` 为同步客户端（标准库
  urllib，阻塞式——可选评测层，默认并发 1 可接受）；api_key 由
  composition root 注入，客户端与 JudgeService 绝不读取环境变量；
  `judge_json(..., system_prompt=...)` 按调用显式传 prompt（VQA 判卷使用
  vqa judge prompt，修复旧实现误用计数 prompt 的问题）；只有 `judge_vqa`
  参与逐样本预算（`reserve_deepseek` 仅在真正发起 Judge 时），
  `judge_counting` 为事后路径不设预算；预算耗尽/任何 judge 异常一律转
  稳定 `judge_error`（仅类名），返回 judge_status=failed 记录，绝不抛出、
  绝不覆盖确定性指标、绝不保存原始异常文本；`judge_vqa_resume` 对
  `vqa_evaluation.json`（统一或 legacy 形状）succeeded 原样返回，其余
  （缺失/损坏/failed/not_requested）读取 `agent_result.json` 的持久化
  answer 重判，受 judge_policy 约束。
- **SampleRunner（Task 04）**：`run_one(sample, sample_dir, *, resolution=None, judge_policy="none")`
  只执行单样本（不迭代数据集）；attempt plan 规则——resolution 缺省或高置信度
  只跑 top task，低置信度按 `candidate_tasks`（≤3）路由后按 AgentName 稳定去重，
  候选与 base task 不同时经 model_copy 重建（task 替换、normalization 清空、
  图像角色重建：变化任务 t1/t2/context，其余 image/context），不兼容候选稳定
  跳过（`INCOMPATIBLE_SAMPLE`/`UNROUTABLE_TASK`/`AGENTS_DEDUPLICATED`），绝不
  原地修改 UnifiedSample；routing fallback（决策声明 fallback_agents 时 primary
  异常触发）与 `fallback_on_partial` 策略只在本 task 内生效，候选兜底在上一
  候选完全失败后进入下一候选；共享逐样本 `CallBudget`（attempts + judge）；
  确定性评估：general_vqa 写 `vqa_evaluation.json`（可选 judge 旁路，judge 失败
  绝不让样本失败），counting 写 `counting_evaluation.json`（新增产物名）；
  失败只记录稳定 code（显式 code 或错误类名），status/trace 绝不包含原始异常
  文本、绝对路径或密钥；产物顺序：sample.json → status=running →
  routing_decision.json → result → evaluation → agent_trace.json → status=final；
  trace 字段含 resolution_source（dataset_task/explicit/rule/model）、
  low_confidence、candidate_tasks、attempt_agents、skipped_candidates、failure_code。
- **DatasetRunner（Task 05）**：`run(*, root, split, task, resume, limit, shard_index,
  shard_count, start_index, sample_ids, fail_fast, sample_concurrency)` 只编排一个
  task；selection 固定顺序（adapter 稳定顺序 → start_index → shard → sample_ids →
  limit），shard 用 `sha256(sample_id) % shard_count`（非 Python hash，跨进程稳定）；
  目录布局 `runs/<run_id>/tasks/<task>/samples/<sha256(sample_id)[:24]>`（不直接使用
  sample_id，Windows 危险名与多 task 同 id 不冲突），`predictions.jsonl` 在 run 根、
  `dataset_summary.json` 在 task 目录；每次运行前 `adapter.probe` 经
  `ArtifactWriter.write_dataset_probe` 独立写 `dataset_probe.json`（数据层绝不
  触碰 manifest.json——manifest 始终可被 RunManifest schema 重新解析、绝不
  动态扩 schema）；resume：succeeded
  默认不重新推理，只补缺失的 vqa/counting 确定性评估与缺失/失败的 VQA judge
  （补判异常 → state=skipped + 稳定 code，重判失败保留 succeeded；补判类型按
  `status.task` 执行任务决定，绝不按 sample.json 的解析任务），
  partial/failed/running/pending/缺失/损坏状态一律重跑 SampleRunner；并发只承诺
  单进程 asyncio（Semaphore 限流 + FIRST_COMPLETED 批次）；fail-fast 后不再提交
  新任务、cancel/await 已启动任务、被取消样本写 skipped（FAIL_FAST_CANCELLED）、
  未启动样本写 skipped（FAIL_FAST_NOT_STARTED）并全部计入 predictions 与终态，
  绝不遗留永久 running；`DatasetRunSummary` 强制
  total == succeeded + partial + failed + skipped；补判 judge 不设逐样本预算
  （call_budget=None）且经 `asyncio.to_thread` 不阻塞事件循环。
- **无 task 数据集 seam（Task 06）**：`SampleDraft` 是 pre-sample 契约（无角色
  校验）；draft 路径固定为 SampleDraft → TaskResolver → `materialize_sample` →
  SampleRunner，`UnifiedSample.task` 保持必填；`DraftDatasetAdapter`（iter_drafts）
  由 `ManifestDraftAdapter`（`spacers_adapter.json` 显式字段映射）实现——task 列
  可选、JSON/JSONL、不猜字段、不调模型、不 import workflows/models、samples_file
  经 `resolve_dataset_relative_path` 限制在 dataset root 内；DatasetRunner
  `run(task=None)` 是**内部显式 auto-task mode**（未来外部入口经
  `DatasetRunOptions.auto_task=True` 选择，auto_task=True 要求 tasks 为空、
  False 要求非空），目录 `tasks/auto/samples/<sha256[:24]>`，resume
  查找无需重新解析：共享默认 CallBudget 贯穿 resolver 与 agent attempts
  （`SampleRunner.run_one(budget=...)` 注入），显式 task 零 resolver 调用，空问题
  只走两条确定性规则；未知 task 绝不冒充 general_vqa——预 task 失败以稳定 code +
  诚实 `unknown` 任务标签（`RunTaskName` 类型）记录 failed 状态；单样本意外异常
  收敛为稳定 failed 状态、绝不终止整个数据集运行；低置信度候选（≤3、top first、
  general_vqa 槽位）由 SampleRunner 既有 attempt plan 执行。
- **06.5 收口契约**：`data` 层绝不写 run artifacts（probe 由
  `ArtifactWriter.write_dataset_probe` 写 `dataset_probe.json`，manifest.json
  始终可被 RunManifest schema 重新解析、绝不动态扩 schema）；judge canonical
  hash 覆盖 model/prompt 文本与版本/sample_id/完整 payload/response schema；
  async 边界的 judge 调用一律 `asyncio.to_thread`；resume 补判按
  `status.task`（执行任务）而非 sample.json 的解析任务；`fallback_used` 涵盖
  candidate index > 0；fail-fast 后所有 selected 样本必有终态（
  succeeded/partial/failed/skipped）且 summary 计数闭合；`auto_task` 是显式
  运行选项（DatasetRunOptions 校验 tasks 空/非空），`task=None` 仅为内部
  auto-task mode。
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

`reporting`、`application`、`main.py` 尚未创建/实现；新计划 Task 07
（Reporting）尚未开始；任务推进时逐层创建并更新本文件。
