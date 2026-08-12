# 14A — First-Qwen Visual Workflow Implementation Plan

> Audience: coding agents implementing the design in staged tasks.
> 面向对象：按阶段实施本方案的编码代理。
> Status: implementation plan only; no production behavior is implemented by this document.
> 状态：仅实施计划；本文档本身不代表生产行为已经实现。

> VQA 子工作流的输入、坐标、实际输出筛选回退和多 ROI 规则已在
> [`14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md`](./14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md)
> 中进一步冻结。实施 VQA 相关阶段时，14B 覆盖本文中的独立 mask 主分支、按类别
> 调用模型、编号框、置信度暴露、SegFormer 候选框、`valid_empty` 及 ROI 坐标等
> 早期设想；本文其余分阶段门禁继续有效。
>
> Grounding 子工作流已在
> [`14C_GROUNDING_AGENT_SUBWORKFLOW.md`](./14C_GROUNDING_AGENT_SUBWORKFLOW.md)
> 中进一步冻结。实施 Grounding 相关阶段时，14C 覆盖本文中的编号标注图、
> SegFormer 候选、全部候选直接输出以及坐标形态等早期设想；架构门禁和分阶段实施
> 要求继续有效。

## 1. Objective

Implement the approved first-Qwen visual planning contract in
[`14_FIRST_QWEN_VISUAL_WORKFLOW_PLAN.md`](./14_FIRST_QWEN_VISUAL_WORKFLOW_PLAN.md)
without breaking the repository's sample, routing, evaluation, artifact, resume,
offline, model-identity, or package-boundary contracts.

最终目标：

```text
UnifiedSample（保留数据集任务/答案协议）
  -> first-Qwen visual plan
  -> deterministic protocol-owner routing
  -> execution-family-specific evidence preparation
  -> final protocol owner / final Qwen answer
  -> existing deterministic evaluation family
```

必须同时保持：

```text
sample.task
    = 数据集任务身份、答案协议和评测身份

visual_plan.execution_family
    = 内部完成路径，不改写 sample.task
```

## 2. Non-negotiable architecture decisions

编码代理在任何阶段都不得重新讨论或静默改变以下结论：

1. `TaskRouter` 继续只读取已知 task，保持同步、确定性、无模型调用；
2. 第一次 Qwen 不是 Router，也不直接选择 YOLO/SegFormer/backend/checkpoint；
3. dataset adapter 不调用第一次 Qwen；
4. `UnifiedSample.task` 始终必填且不被 visual plan 原地修改；
5. 数据集答案格式和确定性评测族不因内部执行工作流改变；
6. 第一次 Qwen 只输出封闭组合类别，去重后最多三个；
7. YOLO 只扫描经校验并增加 halo 后的规划 ROI；区域不确定时使用全图；
8. SegFormer 在该路径中只提供候选区域或存在性证据，不做精确实例计数或
   细粒度属性判断；
9. 所有语义候选筛选和最终回答由最终 Qwen 完成；工作流只做机械处理；
10. 当前最终视觉材料只有普通 ROI 整体图或带编号框的 ROI 整体图；
11. 具体模型只在 `application` composition root 创建一次，领域层只依赖协议；
12. Judge 永远不覆盖 deterministic metrics。

## 3. Why this must be multi-stage

当前实现存在四个必须解耦的事实：

- `TaskResolver` 是不看图片的 pre-sample task resolver；新 visual planner 看图片，
  二者不能合并；
- `TaskRouter` 目前将 task 直接映射到 protocol-owning Agent；visual plan 只能影响
  Agent 内部路径，不能把 Router 改成模型路由；
- YOLO 运行层当前位于 counting 子系统，Grounding/VQA 不能复制第二套 loader；
- 当前 Python allowlist 没有 visual planner 和共享 object-evidence 子系统路径。

因此禁止在一个任务中同时完成 allowlist、模型抽取、Schema、Agent 集成、
resume 和默认值切换。每个阶段必须能够独立审查和验证。

## 4. Target ownership model

### 4.1 Protocol owner remains deterministic

最终输出 Schema、答案约束和评测族仍由外部 task 决定：

| External task | Protocol-owning Agent | Final deterministic family |
|---|---|---|
| `caption` | `caption_agent` | caption |
| `grounding` | `grounding_agent` | grounding |
| `change_caption` / `change_qa` | `change_agent` | caption / VQA（按现有契约） |
| `counting` / `fine_grained_counting` | `counting_agent` | counting |
| VQA task family | `general_vqa_agent` | VQA |

### 4.2 Visual plan selects an internal path

`execution_family` 选择的是 protocol owner 内部可调用的证据/执行路径，不直接
替换 protocol-owning Agent。例如：

```text
multiple_choice_vqa + execution_family=counting
    -> counting evidence path
    -> final GeneralVQAAgent/Qwen 输出合法选项
    -> VQA evaluation
```

禁止把上述样本直接作为普通 `counting` 样本持久化或按 counting metric 评测。

## 5. Proposed new architecture paths

以下是建议的职责路径，不代表已获 allowlist 批准：

```text
workflows/visual_planner.py
    first-Qwen request、hash、budget、strict response validation

agents/object_evidence/__init__.py
    exports only

agents/object_evidence/schema.py
    visual-plan、ROI、检测记录、逐类别状态等严格契约

agents/object_evidence/catalog.py
    组合类别到模型标签/capability 的确定性映射

agents/object_evidence/geometry.py
    preview/ROI/halo/local-to-global 几何

agents/object_evidence/rendering.py
    稳定编号框 ROI 图生成

agents/object_evidence/executor.py
    按类别执行 YOLO -> SegFormer -> unresolved 的确定性状态机
```

建议测试路径：

```text
tests/workflows/test_visual_planner.py
tests/agents/object_evidence/__init__.py
tests/agents/object_evidence/test_schema.py
tests/agents/object_evidence/test_catalog.py
tests/agents/object_evidence/test_geometry.py
tests/agents/object_evidence/test_rendering.py
tests/agents/object_evidence/test_executor.py
```

非 Python 资产：

```text
prompts/first_qwen_visual_plan_v1.md
agents/object_evidence/catalog.json
```

不得新建 `utils.py`、`helpers.py`、`manager.py`、`compat.py`、`legacy.py` 或
第二套 YOLO loader。

## 6. Phase 0 — Architecture and allowlist gate

### Goal

先批准职责边界，不实施业务行为。

### Required changes

1. 审核第 5 节路径是否为最小充分集合；
2. 独立修改：
   - `architecture/allowed_python_files.txt`；
   - `architecture/implementation_status.json`，新增路径先标记 pending；
   - 必要时更新 `architecture/import_rules.json`，但不得放宽顶层 DAG；
3. 记录为什么现有 `workflows/task_resolver.py`、`agents/general_vqa/agent.py`
   和 counting backend 不能容纳全部新职责；
4. allowlist 变更使用独立 commit，不得夹带生产实现。

### Acceptance

```bash
pytest -q \
  tests/architecture/test_allowed_python_files.py \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py
```

### Stop condition

如果任一新 Python 路径未获批准，停止后续实现，不得把职责挤入无关现有文件。

## 7. Phase 1 — Freeze contracts and category catalog

### Goal

只实现可验证契约和离线 catalog，不调用模型、不改变 runtime。

### Scope

实现严格 `extra="forbid"` Schema，至少覆盖：

```text
FirstQwenVisualPlan
ExecutionFamily
ObjectEvidenceRequest
CompositeCategory
RoiPlan
RoiRegion
FinalAnswerImageMode
CategoryEvidenceState
DetectionEvidenceRecord
```

Schema 必须校验：

- `schema_version == "first-qwen-plan-v1"`；
- `confidence` 在 `[0, 1]`；
- 组合类别来自封闭 catalog、去重且最多三个；
- `full_image` 恰好使用每张目标图的 `[0,0,999,999]`；
- `attention_regions` 至少一个非退化框；
- region `image_id` 必须引用当前样本中的图像；
- 坐标固定为 `normalized_0_999_top_left`；
- `object_evidence.required=False` 时不得携带检测类别；
- `numbered_roi_overview` 与需要物体证据保持一致；
- 所有字段严格 JSON-safe、finite、secret-free。

Catalog 必须：

- 使用系统级组合类别作为 key；
- 明确映射到 YOLO 与 SegFormer 的已验证标签；
- 不从 `LABEL_N` 或模型类名猜语义；
- 支持一个组合类别展开为多个细分类别；
- 保留 logical model capability，不保存主机绝对路径；
- 加入 wheel/package-data 测试。

### Tests

```bash
pytest -q \
  tests/agents/object_evidence/test_schema.py \
  tests/agents/object_evidence/test_catalog.py \
  tests/contracts/test_agent_result_contract.py \
  tests/contracts/test_data_schema_contract.py
```

### Acceptance

- 无 runtime 行为变化；
- `UnifiedSample`、TaskName、metrics 不变；
- catalog 不读取权重、不触发网络、不 import torch/transformers/onnxruntime。

## 8. Phase 2 — Unify small-model interfaces without behavior change

### Goal

让 Qwen、YOLO、SegFormer 都通过单一模型无关接口被工作流消费，同时保持 counting
现有行为和唯一 YOLO loader。

### Required protocol work

在 `models/base.py` 中增加最小协议和模型无关输出契约，例如：

```text
ObjectDetectionClient.detect(...)
    -> ObjectDetectionOutput

SemanticSegmentationClient.predict(...)
    -> SemanticSegmentationOutput（已有）

VisionLanguageClient.complete_json(...)
    -> validated schema（已有）
```

`ObjectDetectionOutput` 至少保留：

```text
input width/height
specific model label
confidence
local pixel xyxy
optional OBB polygon
logical model identity
weights SHA
provider/device audit（不含绝对路径）
```

### YOLO migration rule

- 复用当前 `YoloModelStore`、ONNX/Ultralytics adapter 和已验证设置；
- 禁止复制 loader 或重新加载同一 checkpoint；
- composition root 创建/共享一次 detector client/store；
- counting backend 改为消费同一 detection seam，但最终仍生成 `CountingResult`；
- Grounding/VQA 未来只看到 `ObjectDetectionClient` 协议；
- 若需要移动现有 YOLO 文件，必须先另开 allowlist 架构任务，不能在本阶段暗移。

### Tests

```bash
pytest -q \
  tests/models/test_request_sanitization.py \
  tests/agents/counting/test_yolo_adapter.py \
  tests/agents/counting/test_yolo_runtime.py \
  tests/agents/counting/test_backend_selector.py \
  tests/agents/counting/test_executor.py \
  tests/agents/counting/test_agent.py
```

### Acceptance

- counting Golden/现有测试行为不变；
- import 基础包不加载权重；
- 无可选 YOLO/ONNX 依赖时普通 import 不崩溃；
- detector 失败只暴露稳定错误类型/code；
- logical model identity 与物理权重路径继续分离。

## 9. Phase 3 — Deterministic preview, ROI, geometry, and rendering

### Goal

先完成所有不依赖模型的图像与坐标原语。

### Decision checkpoint before coding

以下尚未在设计讨论中冻结，编码前必须由用户批准具体值：

```text
max attention ROIs
halo expansion rule
minimum ROI size
multi-ROI overlap merge rule
box numbering order
annotated image format/quality
full-image internal tiling threshold
```

编码代理不得自行用“常见默认值”填充这些行为参数。

### Required behavior

1. 应用 EXIF orientation 后转 RGB；
2. 最长边超过 1080 才等比缩小，绝不放大小图；
3. Qwen ROI 从规范 0..999 坐标确定性映射回原图像素；
4. halo 在原图坐标上扩张并 clamp；
5. crop 始终来自原图；
6. YOLO 局部 xyxy/OBB 映射回原图全局坐标；
7. 边界、奇数尺寸和舍入规则固定且跨平台一致；
8. 编号顺序确定性，不按并发完成顺序；
9. overlay 不改变原始图像，不覆盖输入文件；
10. 图片 artifact 使用临时文件 + replace 发布，避免半成品。

### Tests

```bash
pytest -q \
  tests/agents/object_evidence/test_geometry.py \
  tests/agents/object_evidence/test_rendering.py \
  tests/models/test_request_sanitization.py
```

测试至少覆盖横图、竖图、方图、小图、超大图、边缘 ROI、多 ROI、空 detection、
OBB 外接框、Windows/POSIX 相同序列化。

## 10. Phase 4 — Implement the first-Qwen visual planner in isolation

### Goal

实现一个可单测但尚未接入 SampleRunner 的 `VisualPlanner`。

### Required behavior

```text
UnifiedSample
  -> safe preview(s)
  -> original question + answer constraints + closed categories
  -> one schema-validated Qwen call
  -> FirstQwenVisualPlan
```

Planner 必须：

- 接收 `VisionLanguageClient` 协议和注入的 Prompt 文本/version；
- 不读取 Ground Truth；
- 不读取 dataset-specific 原始 JSON；
- 不选择具体模型；
- 调用前校验真实 `ModelCacheIdentity`；
- 恰好消费一次 Qwen budget；
- request hash 覆盖 prompt、schema、messages、图片摘要、generation、client version、
  logical model identity 和 revision；
- request/artifact 中不保存 Base64 图片或本机绝对路径；
- 使用 `prompts/first_qwen_visual_plan_v1.md` 并加入 `PromptCatalog` 和 run prompt snapshot；
- 对非法类别、退化 ROI、错误 image id、额外字段稳定失败。

### TaskResolver boundary

```text
SampleDraft（无 task）
  -> TaskResolver（解析外部 task protocol）
  -> materialize UnifiedSample
  -> VisualPlanner（解析内部 execution/evidence plan）
```

显式 task 不再触发 TaskResolver 模型调用，但仍可按启用策略触发 VisualPlanner。
不得让两个服务互相调用或互相覆盖结果。

### Blocking policy decision

Planner 的以下失败策略尚未冻结，接入 runtime 前必须批准：

```text
schema invalid
low confidence
Qwen unavailable/error
budget exhausted
preview decode failure
```

推荐的保守候选策略是“保留外部 task、全图、直接沿现有 Agent 路径”，但在用户
批准前不得把该建议写成生产默认值。

### Tests

```bash
pytest -q \
  tests/workflows/test_visual_planner.py \
  tests/workflows/test_task_resolver.py \
  tests/models/test_response_cache.py \
  tests/workflows/test_call_budget.py
```

## 11. Phase 5 — Implement per-category object-evidence execution

### Goal

用 fake clients 先实现确定性状态机，不接 Agent、不调用真实权重。

### Confirmed state-machine shape

```text
for each composite category independently:
    YOLO supported and successful
        -> retain YOLO evidence
    otherwise
        -> try approved SegFormer capability for this category
    still unresolved
        -> mark for final-Qwen visual fallback
```

一个类别缺失时，其他已成功类别不得重复执行 fallback。

### Decision checkpoint before coding

必须先冻结：

1. `unsupported`、`unavailable`、`error`、`valid_empty` 的精确定义；
2. `valid_empty` 是否进入 SegFormer/Qwen 验证；
3. YOLO 与 SegFormer 都返回证据时的去重/优先级；
4. 某类别 unresolved 时是否保留其他类别的部分成功证据；
5. SegFormer component 生成候选框的阈值和 morphology policy；
6. detector 输出过多时的上限策略；
7. 单 ROI 失败和多 ROI 部分成功的最终状态。

### Executor constraints

- executor 只按 catalog 和状态执行，不重新解释 question；
- YOLO 只接收扩 halo 后 ROI；
- SegFormer候选区域不得伪装成实例检测或精确 count；
- 逐类别保留 backend、specific label、局部/全局坐标、confidence 和稳定状态；
- 不记录 raw exception、完整 mask/tensor、checkpoint 绝对路径；
- 最终 overlay 使用稳定编号并与 JSON detection id 一一对应。

### Tests

```bash
pytest -q \
  tests/agents/object_evidence/test_executor.py \
  tests/agents/object_evidence/test_geometry.py \
  tests/agents/object_evidence/test_rendering.py
```

测试矩阵至少覆盖 1/2/3 个类别、按类别部分成功、unsupported、valid empty、
runtime error、多 ROI、SegFormer候选区域和全部 unresolved。

## 12. Phase 6 — Extract reusable counting evidence seam

### Goal

支持“外部是 VQA 协议、内部使用 counting 工作流”的交叉路径，同时不把
`CountingAgent` 当作 VQA finalizer。

### Required design

从 `CountingAgent` 中提取或显式暴露一个领域服务 seam：

```text
question / approved target hint / image / context
  -> CountingResult
```

约束：

- `CountingAgent` 仍是 public counting task 的协议包装；
- VQA 内部 counting path 只消费 `CountingResult` 作为证据；
- VQA 最终仍由 `GeneralVQAAgent` 输出 `AgentResult`/合法 choices；
- VQA evaluation 仍是 VQA，不落 `counting_evaluation.json`；
- target parse 优先级和 counting fallback 不得复制；
- 共享同一 CallBudget、模型实例、cache identity 和 artifact root；
- 不允许调用 `CountingAgent.run()` 后伪造 sample.task 或改写 persisted sample。

如果现有职责无法在批准路径中容纳这一 seam，必须暂停并申请独立 allowlist
变更，例如职责明确的 `agents/counting/service.py`；不得新建通用 manager/helper。

### Tests

```bash
pytest -q \
  tests/agents/counting/test_agent.py \
  tests/agents/counting/test_executor.py \
  tests/agents/counting/test_target_parser.py \
  tests/evaluation/test_counting_metrics.py \
  tests/evaluation/test_vqa_metrics.py
```

## 13. Phase 7 — Integrate protocol owners under a disabled feature flag

### Goal

把 planner/evidence 接入 Agent 与 SampleRunner，但默认保持当前生产路径。

### Configuration

在 `AppSettings` 增加严格配置组，建议至少包含：

```text
visual_planning.enabled = false
visual_planning.prompt_version
visual_planning.max_rois
visual_planning.halo policy
visual_planning.failure policy
visual_planning.max_detections
```

第一次接入必须默认关闭。默认开启只能在 Phase 10 的独立 rollout 任务中完成。

### Runtime sequence

```text
UnifiedSample ready
  -> if enabled: VisualPlanner
  -> persist visual_plan.json
  -> deterministic TaskRouter selects protocol owner from sample.task
  -> protocol owner consumes typed plan
  -> optional evidence/counting preparation
  -> final Qwen/protocol result
  -> evaluation dispatch from original protocol task
```

推荐通过 `AgentContext` 增加 typed、轻量、JSON-safe 的 visual plan/evidence
依赖；不得把完整 AppSettings、PromptCatalog、权重或 Base64 放进 context。

### Agent-specific integration

#### `GeneralVQAAgent`

- `execution_family=vqa` + 无物体证据：普通 ROI/全图最终 Qwen；
- 需要 object evidence：编号 ROI + detection records + 原问题；
- `execution_family=counting`：CountingResult 作为证据，最终输出仍遵循 VQA/choices；
- multiple-choice postprocess 继续强制合法选项。

#### `GroundingAgent`

- detector/segmenter只提供候选；
- 最终 Qwen选择满足颜色、极值、关系、状态等条件的框；
- completed 仍必须满足现有 grounding geometry contract；
- 不得把全部 YOLO 候选直接复制为最终 grounding boxes。

#### `CountingAgent`

- public counting 继续输出 `CountingResult` 和 `counting_result.json`；
- visual plan 只能提供已验证 target/ROI hint，不能选择 backend；
- 不改变现有 point-derived invariant。

#### `CaptionAgent` / `ChangeAgent`

- 第一轮只接入 planning artifact 和 family validation；
- caption 仍走主 Qwen caption；
- change 保持双时相角色、现有 proposal/semantic 事实边界；
- 不在本阶段重写 change SegFormer 工作流。

### Compatibility matrix gate

接入前必须冻结每个 external task 允许哪些 `execution_family`。不兼容组合必须
稳定 fallback 或失败，不能临时改写 sample.task。尤其需要用户批准：

```text
caption + execution_family=counting/change
grounding + execution_family=vqa/counting
change_* + execution_family=vqa
counting + execution_family=vqa
VQA + execution_family=change
```

### Tests

```bash
pytest -q \
  tests/workflows/test_sample_runner.py \
  tests/agents/general_vqa/test_agent.py \
  tests/agents/grounding/test_agent.py \
  tests/agents/counting/test_agent.py \
  tests/agents/change/test_agent.py \
  tests/agents/caption/test_agent.py \
  tests/evaluation/test_vqa_metrics.py \
  tests/evaluation/test_grounding_metrics.py \
  tests/evaluation/test_counting_metrics.py
```

Feature flag 关闭时，现有测试产物和调用次数必须保持不变。

## 14. Phase 8 — Artifacts, trace, budget, cache, and resume fidelity

### Goal

在启用新路径前完成可复现性与恢复语义。

### Proposed sample artifacts

```text
visual_plan.json
object_evidence.json
roi_000_annotated.png
```

具体文件名必须在本阶段冻结并由 `ArtifactWriter` 统一拥有。JSON 使用现有原子
写入；图片也必须使用同等安全的临时发布原语。任何 result/trace 索引路径不得
信任模型输出。

### Persistence rules

- `visual_plan.json` 保存 validated schema，不保存 raw Qwen body；
- `object_evidence.json` 保存逐类别状态、模型逻辑身份、编号和全局坐标；
- trace 保存 planner prompt version/request hash、fallback codes、执行 family；
- trace 不保存 token、Base64、绝对路径、完整 mask、模型内部 tensor；
- prompt 文件进入 `prompts.snapshot/`；
- config snapshot 包含实际生效的 visual-planning settings；
- request hash 覆盖所有影响规划/最终答案的语义输入。

### Resume rules to implement and test

```text
succeeded
    -> 不重复 planner、不重复 detector、不重复 final Qwen

partial/failed/running/pending/missing status
    -> 按现有明确契约重跑

succeeded + 只缺 evaluation/judge/report
    -> 只补对应阶段，不重新规划或推理
```

必须定义并测试 visual plan/evidence artifact 损坏时的行为；不得依据当前 Prompt、
CLI 默认值或新 config 猜原调用。任何影响 resume 的新选项必须进入权威
`run_request.json` 或冻结 config 契约。

### Budget accounting

至少区分并记录：

```text
optional TaskResolver call
+ exactly one VisualPlanner Qwen call
+ optional counting target/tile/review calls
+ final-answer Qwen call
+ optional Judge call
```

所有阶段共享单样本 budget，不能因 fallback 创建新 budget。默认预算是否需要
调整必须通过配置测试和显式文档变更完成。

### Tests

```bash
pytest -q \
  tests/workflows/test_artifact_writer.py \
  tests/workflows/test_run_store.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/workflows/test_call_budget.py \
  tests/models/test_response_cache.py \
  tests/contracts/test_artifact_contract.py
```

## 15. Phase 9 — Application assembly, packaging, CLI, and reporting

### Goal

由唯一 composition root 完成真实组件组装，并让所有公共入口行为一致。

### Required work

- `application/bootstrap.py` 创建一次 Qwen、YOLO store/client 和 SegFormer clients；
- 注入 `VisualPlanner`、object-evidence executor 和 counting evidence seam；
- `application/prompts.py` 绑定新 Prompt；
- `application/settings.py` 严格解析新配置；
- `pyproject.toml` 打包 object-evidence catalog 与新 Prompt；
- CLI/HTTP 继续只通过 `Runtime`，不得自行复制 planner；
- reporting 只读取已持久化 plan/evidence，可选展示 execution family/fallback；
- reporting 不重新推理、不重跑 detector、不修改执行产物。

除非公开 CLI 参数确有必要，不新增第二套命令；默认通过现有 `ask`、
`run-dataset`、`resume-run`、HTTP `/ask` 复用同一 runtime。

### Documentation

实现事实变化后同步更新：

```text
DETAILS.md
README.md（仅当公开使用方式变化）
configs/default.yaml
configs/*.example.yaml
```

不要修改 migration Golden fixtures 来迎合新行为。新行为应使用独立 feature flag
测试和新的明确 fixtures；只有确认有意改变历史行为后，才按 migration 规则处理。

### Tests

```bash
pytest -q \
  tests/test_main.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_init_side_effects.py
```

还必须构建 wheel 并验证新 catalog/Prompt 已包含、基础 import 不加载模型。

## 16. Phase 10 — Offline integration, live calibration, and rollout

### 10.1 Offline integration gate

先运行全部任务相关测试与架构测试：

```bash
pytest -q \
  tests/agents \
  tests/workflows \
  tests/models \
  tests/evaluation \
  tests/contracts \
  tests/integration \
  tests/architecture

git diff --check
```

然后运行完整离线 pytest；如果环境导致某项不能运行，必须如实报告命令、原因、
替代检查和剩余风险。

### 10.2 Live calibration gate

仅在用户明确授权联网/真实模型/真实数据后执行。至少比较：

```text
baseline answer accuracy
planned answer accuracy
planner schema-valid rate
full-image fallback rate
per-category YOLO/SegFormer hit rate
valid-empty/error rate
Qwen calls per sample
latency and peak memory
ROI miss audit
```

不得通过过滤 failed/skipped 样本提高结果。

### 10.3 Rollout sequence

建议按以下顺序独立开启：

1. VQA direct-Qwen planning only；
2. VQA object-evidence path；
3. Grounding candidate-evidence path；
4. VQA internal counting path；
5. public counting plan hints；
6. caption/change family validation；
7. 全局默认开启。

每一步都应保留 feature flag 和一组固定样本 A/B 审计结果。全局默认值切换必须
是独立任务，不能夹在实现 commit 中。

## 17. Required task/commit decomposition

建议 coding agent 将实施拆成以下独立任务或 PR：

```text
C0  Architecture allowlist approval only
C1  Strict schemas + composite-category catalog
C2  Shared object-detection protocol + YOLO seam parity
C3  Preview/ROI/halo/global-coordinate/rendering primitives
C4  Isolated VisualPlanner + Prompt + cache/budget tests
C5  Per-category object-evidence executor with fake clients
C6  Reusable counting evidence seam
C7  Feature-flagged Agent/SampleRunner integration
C8  Artifact/resume/budget/report fidelity
C9  Bootstrap/settings/package/CLI integration
C10 Offline suite + authorized live calibration
C11 Staged default rollout
```

一个任务不得顺带完成后续任务。每个任务开始前必须重新执行：

```bash
git status --short
git rev-parse HEAD
```

每个任务结束至少执行：

```bash
git diff --check
git status --short
```

并如实记录运行过和未运行的测试。

## 18. Global acceptance criteria

最终完成必须同时满足：

- first-Qwen plan 严格符合 v1 Schema；
- 组合类别封闭、最多三个、无自由文本模型标签；
- 无空间约束默认全图，难以确定的语义区域回退全图；
- ROI/halo/local-global 坐标在测试中确定性且跨平台一致；
- YOLO/SegFormer/Qwen 都只通过单一协议调用，模型只组装一次；
- 不存在第二套 YOLO loader；
- 每个类别独立 fallback，成功类别不重复执行；
- 最终 Qwen看到编号 ROI 和匹配的具体标签/全局坐标；
- external task、答案协议、deterministic evaluation family 不变；
- caption/change/counting 现有契约没有被 VQA 改造破坏；
- planner、final call、fallback、cache、budget、artifact、resume 可审计；
- 默认离线，不下载模型/数据，不调用云 API；
- trace/artifacts 不含 secret、Base64、绝对内部路径或 raw sensitive errors；
- feature flag 关闭时保持现有行为；
- architecture tests、任务相关测试和最终声明的完整测试集合真实通过。

## 19. Explicit stop conditions for coding agents

遇到以下任一情况必须停止当前实现并报告，不得自行猜测：

1. 所需 Python 路径不在 allowlist；
2. 需要改变 deterministic metric、GT 解释、split 或样本纳入规则；
3. 需要让 Router 读取 question 或调用模型；
4. 需要复制 YOLO/SegFormer loader；
5. 需要修改主模型/checkpoint/processor/tokenizer 加载语义；
6. 下层 fallback/valid-empty/partial-evidence policy 尚未获批准；
7. execution-family compatibility matrix 尚未获批准；
8. resume 无法从持久化调用身份确定性重建；
9. 需要新增未批准依赖；
10. 发现当前工作区已有修改与本阶段目标文件冲突。

## 20. Final reporting template for every phase

每个 coding agent 完成阶段任务后，最终汇报必须包含：

```text
Phase / task id:
What changed:
Why:
Files changed:
Tests/checks actually run:
Results:
Not run and why:
Impact on UnifiedSample:
Impact on task/routing:
Impact on model interfaces:
Impact on evaluation:
Impact on reporting:
Impact on CLI/HTTP:
Impact on resume/artifacts:
Known risks / next gate:
```

不得在单一阶段完成时声称整个 first-Qwen pipeline 已完成。
