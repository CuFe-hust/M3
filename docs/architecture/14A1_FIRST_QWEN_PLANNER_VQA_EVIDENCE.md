# 14A1 — Model Seams, ROI Geometry, VisualPlanner, and VQA Evidence

> Superseded note (doc 15): the joint flow (`visual_planning.enabled=True`)
> replaces the independent VisualPlanner gate with one joint Qwen call
> (task + visual plan) and overrides the
> `SampleDraft -> TaskResolver -> UnifiedSample -> VisualPlanner` order for
> that flow. §6's frozen order stays valid for the disabled (legacy) path.
> 取代注记（doc 15）：联合流程（`visual_planning.enabled=True`）用单次联合
> Qwen 调用（task + 视觉计划）替代独立 VisualPlanner 门禁，并在该流程中
> 覆盖 `SampleDraft -> TaskResolver -> UnifiedSample -> VisualPlanner` 顺序；
> §6 冻结顺序对关闭（legacy）路径仍然有效。

> Execute only after the 14A0 allowlist change is approved and complete.
> 仅在 14A0 白名单架构变更已获批准并完成后执行。

## 1. Session context and preflight

必读：根 `AGENTS.md`、当前 `DETAILS.md`、14A 索引、14A0 交接、14B 全文，以及
与本包直接相关的 `models/base.py`、`models/images.py`、counting YOLO adapter/store、
`agents/general_vqa/agent.py`、`workflows/call_budget.py` 和现有测试。

14C 只需阅读其“与 VQA 共用的类别/YOLO 基础规则”；本包不实现 Grounding finalizer。

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

确认 14A0 新路径已在 allowlist、implemented/pending 状态与磁盘事实一致。若不一致，
停止，不得在普通实现中修改白名单。

## 2. Outcome and decomposition

本包按以下顺序实施，保持独立 review unit：

```text
C1  shared strict contracts + versioned evidence catalog
C2  model-independent object-detection seam; counting parity
C3  VQA preview/ROI/local-global geometry and rendering primitives
C4  isolated VisualPlanner + prompt + cache/budget tests
C5  VQA evidence executor with fake model clients
```

本包结束时仍不接入 SampleRunner/真实 Agent 路径，feature 行为默认不存在；真实权重
和网络均不需要。

## 3. C1 — Strict contracts and category catalog

### 3.1 Shared plan schema

在现有 `agents/schema.py` 中定义跨 protocol owner 可消费的最小严格类型，避免 Agent
反向 import `workflows`：

```text
FirstQwenVisualPlan
ExecutionFamily
ObjectEvidenceRequest
RoiPlan
RoiRegion
```

要求：Pydantic `extra="forbid"`；字段 JSON-safe、finite、secret-free；版本固定为
`first-qwen-plan-v1`；confidence 位于 `[0,1]`；组合类别来自同版本封闭目录且最多三个；
规划 ROI 使用 `[0,1]` top-left xyxy；full-image 为 `[0,0,1,1]`；attention ROI 最多
三个且非退化；required=False 不携带类别；不含 backend/checkpoint/device/最终答案；
不修改 `UnifiedSample.task`、不读取 Ground Truth。

shared plan schema 不放 VQA mask、Grounding `box_id` 或 CountingResult。

### 3.2 Shared catalog

实现 `agents/evidence_catalog.py`、`agents/evidence_catalog.json`：

```text
catalog version
composite -> ordered leaf categories
leaf -> verified YOLO labels
leaf -> optional verified SegFormer labels
logical capability identity
```

prompt、plan 校验、展开、能力判断和筛选读取同一版本；不从 `LABEL_N`、类名或模型路径
猜语义；不保存物理路径，不 import 重依赖。Grounding 只消费 YOLO mapping，VQA 可消费
两类 mapping。未校准能力保持 disabled；目录外/部分非法类别若未批准容错则严格失败。

### 3.3 VQA evidence schema

在 `agents/general_vqa/evidence/schema.py` 定义严格 VQA 类型：state、detection、
segmentation、ROI record 和 bundle。不存在 `valid_empty`；成功但筛选为空是 `missing`；
detection 保留 local/global 几何但不向 final Qwen/公共 trace 暴露 confidence；SegFormer
mask 不转框/计数；持久化字段路径安全，不允许 tensor/PIL/bytes/NaN/Base64/raw
exception/物理路径。`__init__.py` 只 re-export。

测试覆盖 strict fields、版本/类别/ROI、required 联动、catalog 顺序/能力/安全、
missing 语义、confidence 隔离、mask 不转框和基础 import 无重依赖：

```bash
pytest -q \
  tests/agents/test_evidence_catalog.py \
  tests/agents/general_vqa/evidence/test_schema.py \
  tests/contracts/test_agent_result_contract.py \
  tests/contracts/test_data_schema_contract.py
```

## 4. C2 — Shared object-detection protocol

在 `models/base.py` 增加最小模型无关协议与输出契约：

```text
ObjectDetectionClient.detect(image, request metadata) -> ObjectDetectionOutput
```

输出至少包含：模型实际输入宽高、specific output label、内部 confidence、ROI-local
pixel xyxy、可选 OBB polygon、logical model identity、weights SHA、provider/device audit。
不得包含 checkpoint 绝对路径、原始 tensor 或 backend 私有对象。

复用现有唯一 `YoloModelStore` 和 ONNX/Ultralytics adapter：

- loader 保持惰性且单次组装；普通 import 缺可选依赖时不崩溃；
- counting backend 改为消费同一 detection seam，但继续输出完全相同的
  `CountingResult`、状态、fallback、artifact 和 point-derived count；
- VQA 后续只依赖 `ObjectDetectionClient`，不 import counting backend/具体 YOLO；
- 若实现需要移动现有 YOLO 文件，立即停止并另开 allowlist 任务；
- detector 错误对外只暴露稳定 code/type，不写 raw exception。

验证：

```bash
pytest -q \
  tests/models/test_request_sanitization.py \
  tests/agents/counting/test_yolo_adapter.py \
  tests/agents/counting/test_yolo_runtime.py \
  tests/agents/counting/test_backend_selector.py \
  tests/agents/counting/test_executor.py \
  tests/agents/counting/test_agent.py
```

必须证明 counting parity、逻辑身份与物理路径分离、基础 import 不加载权重。

## 5. C3 — VQA ROI geometry and rendering

实现位置限定为 `agents/general_vqa/evidence/geometry.py` 与 `rendering.py`。任务无关的
EXIF/RGB/裁切复用 `models/images.py`；若 HEAD 已有 `crop_image_region(...)`，先运行其
测试并直接使用，不复制实现。

### 4.1 Frozen geometry

- 规划预览：应用 EXIF 后转 RGB，最长边超过 1080 才等比缩到 1080，不放大小图；
- VQA ROI：`normalized_0_1_top_left` 的 `[x1,y1,x2,y2]`；最多三个；
- 无可靠空间约束使用整图 `[0,0,1,1]`；任一 ROI 计划整体非法则回退唯一整图，
  不截断前三个、不重试 Planner；
- 多个分离 ROI 分别裁切，不合并外接矩形，也不绑定目标类别；
- crop 从 EXIF 后原始分辨率图获取，绝不从 1080 预览二次裁切；
- 左上 floor、右下 ceil；映射后每边按 ROI 自身宽/高默认扩张 `0.10` 并 clamp；
- 每个 ROI 记录原图 id/尺寸、core 和 expanded pixel xyxy、crop 尺寸、local-global
  变换；模型 resize/letterbox 逆变换必须显式；
- 最终 Qwen 图像若最长边超过 1080 才缩小，不放大；模型证据仍基于原分辨率 ROI。

ROI 最小像素尺寸、超高分辨率内部切片阈值仍未冻结；实现触及二者时必须停止请求
用户决策，不得使用常见默认值。

### 4.2 VQA rendering

14B 已覆盖原 14A 的“编号框 ROI”设想：

- YOLO 证据以文本记录提供，clean ROI 不画检测框；
- SegFormer 证据保留每个 ROI 的独立半透明 overlay 和稳定颜色图例；
- 同一叶子类别跨 ROI/样本使用稳定调色表；重叠 ROI mask 不融合；
- final input 顺序固定为每个 `roi_id` 的 clean ROI 在前、可选 mask overlay 在后；
- overlay 不修改/覆盖源图；图片发布使用临时文件 + replace；
- 具体调色表和图片格式/质量若尚未批准，先用纯内存结构与测试 seam，停止持久化
  格式实现，不自行选择 JPEG/PNG 参数。

测试覆盖横/竖/方/奇数/1px/超大图、边缘 halo、多 ROI、invalid-plan full-image
fallback、resize/letterbox 逆变换、local-global 坐标、跨平台稳定顺序、mask 不转框。

```bash
pytest -q \
  tests/agents/general_vqa/evidence/test_geometry.py \
  tests/agents/general_vqa/evidence/test_rendering.py \
  tests/models/test_request_sanitization.py
```

## 6. C4 — Isolated VisualPlanner

实现 `workflows/visual_planner.py`，但不接 SampleRunner：

```text
UnifiedSample
  -> safe preview(s)
  -> question + answer-domain constraints + same-version closed catalog
  -> exactly one schema-validated Qwen call
  -> FirstQwenVisualPlan
```

要求：

- 只依赖 `VisionLanguageClient`、注入 prompt/version、共享 plan schema/catalog；
- 不读 Ground Truth 或 dataset-specific JSON，不选择具体模型，不改变 sample.task；
- prompt 使用 `prompts/first_qwen_visual_plan_v1.md`，并接入 `PromptCatalog` snapshot seam；
- 调用前验证真实 `ModelCacheIdentity`，恰好消费一次 Qwen budget；
- request hash 覆盖 prompt/schema/messages/图片摘要/generation/client version/logical
  model identity/revision/catalog version；
- artifact/request 不保存 Base64、secret、raw model body 或绝对图像路径；
- 严格拒绝额外字段、目录外类别、退化 ROI、错误 image id、非 finite 值；
- `SampleDraft -> TaskResolver -> UnifiedSample -> VisualPlanner` 顺序不可改变。

Planner 的 schema invalid、low confidence、client unavailable/error、budget exhausted、
preview decode failure 策略必须在接 runtime 前由用户冻结。若当前任务没有明确批准，
C4 只实现 typed failure result/error code seam，不写“现有 Agent + 全图”为生产默认值。

```bash
pytest -q \
  tests/workflows/test_visual_planner.py \
  tests/workflows/test_task_resolver.py \
  tests/models/test_response_cache.py \
  tests/workflows/test_call_budget.py
```

## 7. C5 — VQA evidence executor with fake clients

实现限定在 `agents/general_vqa/evidence/executor.py`；不得接 `GeneralVQAAgent.run()`、
SampleRunner 或真实模型。

### 6.1 Frozen state machine

```text
expand requested composite categories to ordered leaf categories
for each ROI: run YOLO once, then filter all requested leaves
for each still-missing leaf:
    if catalog has approved SegFormer capability:
        run required SegFormer inference once per ROI, then filter
    if still missing:
        leave the leaf for the single final-Qwen visual fallback
preserve all successful evidence from other leaves
```

状态定义：

- `hit`: 成功推理且实际筛选到请求叶子类别；
- `missing`: 成功推理但筛选后为空；不存在 `valid_empty`；
- `unsupported`: catalog 无该模型 capability；
- `unavailable`: 已批准 capability 的客户端/资产不可用；
- `error`: 调用发生稳定分类的运行错误；
- `not_run`: 该叶子类别已在上层命中，无需执行后续模型。

`missing/unsupported/unavailable/error` 可以进入下一批准 fallback；已 `hit` 的叶子类别
不得重跑或被覆盖。若用户尚未批准单 ROI error 与多 ROI 部分成功的原子性，executor
保留逐 ROI/逐类别状态但不得擅自决定 sample 最终成功状态。

### 6.2 VQA-only constraints

- YOLO/SegFormer 调用次数按 ROI，不按类别增长；模型输入只有 ROI 图片；
- 只保留目录请求标签，未请求输出全部丢弃；
- YOLO confidence 仅供阈值/NMS/跨 ROI 去重/冲突裁决，不能进入 final Qwen 或公共 trace；
- SegFormer 只保留 mask/存在性证据，不转 box、不生成 instance count；
- 跨 ROI 重复 YOLO 目标在 whole-image 坐标去重，内部置信度高者胜出；
- mask 在各 ROI 内独立保留；不跨 ROI 像素融合或比较置信度；
- 最终 bundle 按 `roi_id` 和稳定 leaf order 组装，不按并发完成顺序；
- 不记录 raw exception、tensor、完整 raw response、Base64、secret 或物理模型路径。

阈值、NMS/冲突规则、maximum detections、mask 调色表、ROI 部分失败状态若未被用户
明确冻结，只实现可注入 policy 和 fake-client tests；不得填写任意默认值。

```bash
pytest -q \
  tests/agents/general_vqa/evidence/test_executor.py \
  tests/agents/general_vqa/evidence/test_geometry.py \
  tests/agents/general_vqa/evidence/test_rendering.py
```

测试矩阵至少覆盖 1/2/3 类别、1/2/3 ROI、按叶子部分命中、YOLO empty、SegFormer
empty、unsupported/unavailable/error、未请求标签过滤、跨 ROI 去重、全部 unresolved、
调用次数与稳定输出顺序。

## 8. Final checks and handoff

```bash
pytest -q tests/agents/counting tests/agents/general_vqa/evidence tests/workflows/test_visual_planner.py
pytest -q \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py
git diff --check
git status --short
```

验收：feature 尚未接 runtime；existing Agent 调用次数/产物不变；counting parity；VQA
evidence 全部位于 `agents/general_vqa/evidence/`；没有 `agents/object_evidence/`；基础
import 不加载权重；明确列出所有仍未冻结 policy，供 14A2 开始前批准。
