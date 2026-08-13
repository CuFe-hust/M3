# 15 — Joint Task Resolution and Visual Planning

> Status: approved design plan; implementation has not started.
> 状态：已确定的修改方案；尚未开始实现。

## 1. 目标

把当前串行的两次决策：

```text
文本 TaskResolver
  -> task
  -> 图像 VisualPlanner
  -> visual plan
```

合并为一次有视觉输入的 Qwen 调用：

```text
缩略图 + 文本
  -> one schema-validated Qwen call
  -> task + visual plan
```

联合调用同时完成：

1. 根据问题和图像内容判断真实任务；
2. 判断后续完成路径；
3. 为超过模型直接处理尺寸的图像给出注意力 ROI；
4. 在需要时请求封闭目录内的视觉证据类别。

模型输出的 `task` 是后续执行 task，并直接交给确定性的 `TaskRouter`。数据集或调用方
提供的 task 不再决定 Agent 路由；它只能作为来源信息保留，不能覆盖联合调用结果。

本计划只修改运行时联合决策，不包含 Qwen3-VL 的 LoRA/SFT/RL 训练方案。

## 2. 已冻结的设计决定

### 2.1 单次联合视觉调用

每条进入主执行链路的样本只进行一次联合规划 Qwen 调用。不得先进行纯文本 task
解析，再进行视觉规划；也不得在联合调用后再次调用模型复核 task。

联合调用输入至少包括：

```text
question / user text
ordered image descriptors: image_id + source order
safe preview images
allowed task names
catalog version + closed composite categories
answer constraints（若存在）
```

不得向模型提供 Ground Truth、绝对路径、Base64 artifact、secret、具体 backend、
checkpoint、processor 或 device。

### 2.2 缩略图规则

复用当前确定性预览实现：

- EXIF transpose 后转 RGB；
- 最长边超过 `1080` 时等比例缩小到最长边 `1080`；
- 最长边不超过 `1080` 时不放大；
- 传给模型的预览以实际传输字节计算 digest；
- 不写临时预览文件，不把机器绝对路径写入请求或产物。

ROI 使用原图归一化的 `[0,1]`、左上原点 `xyxy` 坐标。小图后续可以直接使用整图；
大图后续使用联合计划给出的 ROI 调用共享裁切工具，裁切几何必须能够确定性映射回
原图坐标。

### 2.3 联合输出

新增版本化严格输出契约，语义结构为：

```json
{
  "version": "joint-qwen-plan-v1",
  "task": "<TaskName>",
  "visual_plan": {
    "execution_family": "direct_vqa | object_evidence_vqa",
    "confidence": 0.0,
    "roi_plan": {
      "rois": [
        {
          "roi_id": "r1",
          "image_id": "img1",
          "xyxy": [0.0, 0.0, 1.0, 1.0]
        }
      ]
    },
    "evidence_request": null,
    "reason_codes": ["..."]
  }
}
```

约束：

- `task` 必须属于 `data.schema.TaskName` 的封闭集合；
- `task` 是后续物化、路由、执行和状态记录使用的 task；
- `visual_plan` 不得包含最终答案；
- `object_evidence_vqa` 必须携带合法 `evidence_request`；
- `direct_vqa` 不得携带 `evidence_request`；
- evidence category 必须来自同版本封闭 catalog；
- ROI 的 `image_id` 必须引用输入图像；
- ROI 数量、有限值、范围、非退化性与稳定去重继续严格校验；
- schema `extra="forbid"`，不得容忍未声明字段；
- 输出解析或校验失败时使用稳定错误码，不持久化原始模型正文或异常全文。

`FirstQwenVisualPlan` 中可复用的 ROI、evidence request 和 family 子结构继续复用；新增
的是包裹 `task + visual_plan` 的版本化联合响应，不创建平行的 ROI/坐标实现。

### 2.4 运行顺序

目标主链路：

```text
adapter source record
  -> pre-routing sample view
  -> joint Qwen planner: preview(s) + text -> task + visual plan
  -> materialize UnifiedSample with model-selected task
  -> deterministic TaskRouter
  -> selected Agent + injected visual plan
  -> result artifact
  -> deterministic evaluation when compatible
  -> optional Judge
  -> status / trace / prediction index
```

长期不变量：

- `UnifiedSample.task` 仍然必填；
- `UnifiedSample` 只在联合决策之后进入 Agent/Router；
- `TaskRouter` 仍同步、确定性、无模型调用，只消费联合输出中的已知 task；
- adapter 不调用模型、不执行 Agent、不写 run artifact；
- source Ground Truth 只读保留，不因联合 task 判断而改写；
- 具体 Qwen 仍只由 `application` composition root 创建一次并共享；
- Judge 不覆盖确定性指标。

### 2.5 来源 task 与执行 task

若来源记录已有 task：

- 来源 task 不作为路由权威；
- 联合 Qwen 返回的 task 成为新的 `UnifiedSample.task`；
- 来源 task 仅以 JSON-safe、路径无关的审计字段持久化；
- 不修改 adapter 原始记录或 Ground Truth；
- 不把 task 冲突写成原始异常文本。

确定性评测按实际执行 task 分派。若现有 Ground Truth 与该指标族不兼容，继续
fail-closed，不猜测、不转换、不伪造指标。任何改变 GT 解释或官方 evaluator 映射的
需求必须作为独立评测任务审批，不包含在本计划中。

## 3. 对当前架构的覆盖关系

本计划实施后，覆盖以下旧决定：

1. `SampleDraft -> TaskResolver -> UnifiedSample -> VisualPlanner` 的固定顺序；
2. `TaskResolver` 不看图像的模型解析路径；
3. VisualPlanner 只能处理已经物化 `UnifiedSample` 的限制；
4. visual plan 永远不得影响 `sample.task` 的限制；
5. 显式 task 可以零模型调用直接执行的行为；
6. auto-task 才调用 TaskResolver 的行为。

以下边界不变：

1. `TaskResolver` 与 `TaskRouter` 的概念不得重新混淆；联合视觉模型负责产生已知 task，
   `TaskRouter` 仍只做确定性 task -> Agent 映射；
2. 不允许 Router 直接读取 question、图像或调用模型；
3. 不允许 Agent 自己重新分类全局 task；
4. 不允许 dataset adapter 调模型；
5. 不允许 workflow import `models.entry` 或具体 Qwen 实现；
6. 不允许新增 `spacers_agent/`、旧 `eval/` 或动态兼容回退。

旧的 14A/14B/14C 文档继续作为已实现历史和证据子流程参考。代码落地并通过门禁后，
应在这些文档顶部增加 superseded 注记；实施前不得把本计划写入 `DETAILS.md` 作为当前
事实。

## 4. 实施范围与文件计划

### 4.1 Schema

修改：

```text
agents/schema.py
routing/schema.py
data/schema.py（仅在联合调用前置视图/物化确有必要时）
```

工作项：

1. 新增 `joint-qwen-plan-v1` 严格响应模型；
2. 复用现有 `RoiRegion`、`RoiPlan`、`ObjectEvidenceRequest`；
3. 保持 `TaskName` 为唯一合法 task 集合；
4. 提供从联合响应到运行时 resolution/trace 字段的纯转换；
5. 扩展物化逻辑，使模型 task 决定 `image/t1/t2/context` 角色；
6. 保证 metadata、normalization 和 Ground Truth 仍满足 JSON-safe 与一致性校验；
7. 不创建第二套 Prediction 或全局 sample schema。

不得新增未在 `architecture/allowed_python_files.txt` 中批准的 Python 文件。若现有职责
无法容纳联合契约，必须先停止并单独申请 allowlist 变更。

### 4.2 Prompt 与 prompt catalog

修改：

```text
application/prompts.py
```

新增版本化 Markdown prompt：

```text
prompts/joint_qwen_task_visual_plan_v1.md
```

新 prompt 必须明确：只判断 task 和规划，不回答、计数或描述；输出严格 JSON；task
来自封闭集合；ROI 使用 `[0,1]`；类别来自同版本 catalog；不得输出路径、GT、backend
或最终答案。

旧 prompt 文件保留用于旧 run snapshot 和审计，不原地改变旧版本语义。`PromptCatalog`
改为绑定新的逻辑 key/version；snapshot 同时保持可恢复性。

### 4.3 联合 Planner

主要修改：

```text
workflows/visual_planner.py
workflows/task_resolver.py
```

工作项：

1. `VisualPlanner` 改为在物化前接收统一的 pre-routing view；
2. 构造安全预览、文本载荷和同版本 catalog；
3. 恰好调用一次 `VisionLanguageClient.complete_json(...)`；
4. 返回已验证的 `task + visual_plan`；
5. 校验 task、图像 id、ROI、family/category 联动与 confidence；
6. 保留真实 `ModelCacheIdentity` 前置校验和稳定错误码；
7. 删除 `TaskResolver` 的独立模型调用路径；
8. `materialize_sample(...)` 若继续留在 `task_resolver.py`，只承担无模型的物化职责；
9. 不让联合 Planner 执行 Agent、detector、segmenter 或裁切后的最终问答。

### 4.4 Dataset、单样本和手动调用链路

修改：

```text
workflows/dataset_runner.py
workflows/sample_runner.py
application/runtime.py
application/bootstrap.py
```

工作项：

1. dataset 的 explicit/default/auto 三种入口统一在路由前经过联合 Planner；
2. `DatasetRunner` 不再先调用文本 `TaskResolver`；
3. 联合输出 task 物化为 `UnifiedSample` 后再交给 `SampleRunner`；
4. `SampleRunner` 不再对同一样本重复调用旧 VisualPlanningGate；
5. `SampleRunner` 使用联合 task 构建确定性路由与 attempt plan；
6. 将同一次调用产生的 visual plan 注入 `AgentContext`；
7. `application.runtime.ask(task="auto")` 与显式 task 路径统一使用联合调用；
8. Qwen 客户端保持单次 runtime assembly、共享实例、惰性模型加载；
9. feature flag 关闭时保留现有行为，完成离线与 live gate 后再单独批准默认开启。

### 4.5 ROI 消费

修改现有 Agent，不新增通用第二 Agent：

```text
agents/general_vqa/
agents/grounding/
agents/counting/
agents/change/
agents/caption/
```

分阶段接入：

1. 小图直接把整图交给后续 Agent；
2. 大图根据 plan ROI 调用 `models.images.crop_image_region(...)`；
3. 裁切必须保留 local -> whole-image 的确定性几何映射；
4. Agent 只能消费联合计划，不能修改全局 task；
5. VQA/Grounding 继续复用现有 evidence service；
6. Counting 继续输出唯一 `CountingResult`，plan 不选择 backend/checkpoint；
7. Change 保持 T1/T2 权威原图与时相顺序；
8. Caption 不引入对象检测作为语义真值；
9. ROI、检测框和 mask 都只是证据/注意力输入，不得改写 Ground Truth。

ROI 消费应按 Agent 家族逐包实现和验证，不在一个提交中重写所有 Agent。

### 4.6 Artifact、预算、缓存与 resume

修改：

```text
workflows/artifact_writer.py
workflows/call_budget.py
workflows/run_store.py（仅当冻结请求字段需要扩展）
workflows/schema.py
reporting/（仅增加只读展示时）
```

工作项：

1. 新 run 持久化验证后的联合响应，建议使用独立 basename：
   `joint_visual_plan.json`；
2. 不保存 raw Qwen body、Base64、绝对路径、secret 或原始异常全文；
3. 单样本联合规划只消费一次 Qwen budget；
4. 删除“resolver 一次 + planner 一次”的双调用预算假设；
5. request hash 覆盖 logical model identity、revision、generation、prompt/schema/catalog
   version、完整 messages、预览 digest、client version 和影响输出的配置；
6. fresh run 的联合规划配置写入权威 `run_request.json`/config snapshot；
7. `succeeded` resume 不重新调用联合 Planner 或 Agent；
8. partial/failed/running/pending/缺失或损坏联合产物按明确 rerun 契约处理；
9. 旧 run 的 `task_resolution`/`visual_plan.json` 只读兼容，不伪装成新联合响应；
10. `predictions.jsonl` 保持 append-only，路径保持 run-relative 且安全。

## 5. 实施顺序

### 阶段 A：契约与离线 Schema

- 新增联合响应模型与 prompt；
- 冻结版本号、字段、坐标和 stable error code；
- 完成纯 schema、prompt、request-hash 测试；
- 此阶段不接入真实 DatasetRunner/SampleRunner。

### 阶段 B：隔离联合 Planner

- 在 fake Qwen client 下完成缩略图 + 文本 -> task + visual plan；
- 验证恰好一次调用、一次 budget、完整 cache identity；
- 验证输出不含 GT、路径、secret 或 raw response；
- 此阶段不执行任何 Agent。

### 阶段 C：运行链路替换

- 先接手动 `ask` 和小型 fake dataset；
- 再统一 explicit/default/auto dataset 路径；
- 删除同一样本的旧 Resolver + VisualPlanner 双调用；
- 保持 feature flag 默认关闭，验证 disabled path 不变。

### 阶段 D：各 Agent 的 ROI 消费

- 按 General VQA/Grounding、Counting、Change、Caption 分包接入；
- 每包单独验证大图裁切、原图坐标回映和结果 schema；
- 不借机改变指标、GT、split、样本过滤或 backend 选择。

### 阶段 E：artifact/resume/rollout

- 完成联合 artifact、run request、resume 与 reporting 只读适配；
- 运行完整离线门禁；
- 使用目标 Qwen3-VL checkpoint 做小切片 live gate；
- 只有在调用数、任务路由、ROI 和结果质量均验证后，才单独批准默认开启。

## 6. 必须新增或更新的测试

### 6.1 联合契约

- 合法 task + direct plan；
- 合法 task + object-evidence plan；
- 非法 task、extra field、非法 category、错误 image id；
- ROI 非 finite、越界、退化、重复和超限；
- 输出不得携带 final answer/backend/checkpoint/device/path/GT；
- schema/version 变化进入 request hash。

### 6.2 预览和调用次数

- 横图、竖图、方图、奇数尺寸；
- 最长边 `1080` 不放大；
- 最长边大于 `1080` 等比缩小；
- 多图稳定顺序与 digest；
- 每样本恰好一次联合 Qwen 调用和一次 budget 消耗；
- 不再发生独立 TaskResolver 模型调用或第二次 VisualPlanner 调用。

### 6.3 物化与路由

- 联合 task 决定 `UnifiedSample.task`；
- change task 重建 `t1/t2/context`；
- 非 change task 重建 `image/context`；
- TaskRouter 只消费模型 task，仍保持确定性；
- 来源 task 与模型 task 不同时，路由采用模型 task，且审计字段保留来源 task；
- 未知/非法 task fail-closed，不猜 `general_vqa`。

### 6.4 Agent 与 ROI

- 小图整图路径；
- 大图 ROI 裁切路径；
- ROI local/global 坐标零漂移；
- 多图 ROI 绑定正确 image id；
- 各 Agent 输出仍满足原领域 schema；
- Counting backend、Grounding box、Change 时相和 Caption 语义边界不漂移。

### 6.5 Artifact 与 resume

- 联合响应原子写入且严格 JSON-safe；
- result path/basename 跨 POSIX/Windows 安全；
- succeeded resume 零模型调用；
- 损坏/缺失联合 artifact 按冻结契约处理；
- 旧 run 不被新 schema 错读；
- prediction index append-only，summary 闭合；
- trace/report 不泄漏 secret、Base64、绝对路径或 raw exception。

建议至少运行：

```bash
pytest -q \
  tests/workflows/test_task_resolver.py \
  tests/workflows/test_visual_planner.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/integration/test_dataset_runner_resume.py \
  tests/application/test_bootstrap.py \
  tests/application/test_runtime.py \
  tests/application/test_prompts.py \
  tests/models/test_response_cache.py \
  tests/models/test_request_sanitization.py
```

涉及 Schema、文件职责和 import 后，补充：

```bash
pytest -q \
  tests/architecture/test_allowed_python_files.py \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py
```

## 7. 文档同步

代码完成并通过测试后同步更新：

```text
DETAILS.md
README.md（若公开 CLI/运行语义变化）
docs/architecture/14A*.md（增加 superseded 注记）
docs/migration/（记录任务权威性与历史结果可比性变化）
```

`DETAILS.md` 至少更新：联合 Schema、运行顺序、task 权威来源、prompt/catalog、预算、
artifact、resume、manual ask、dataset 三种任务模式和已知限制。

## 8. 明确不在本计划内

- Qwen3-VL merger/LLM/vision encoder 的 LoRA 范围；
- 遥感 SFT 数据配比；
- DPO、GRPO 或其他 RL 方案；
- 新 detector/segmenter/checkpoint；
- metric、GT、split、aggregation 或 official evaluator 改造；
- Judge 替代确定性指标；
- 新公共 CLI；
- 默认联网、自动下载模型或数据集；
- 新增未批准 Python 路径或第三方依赖。

## 9. 完成标准

只有同时满足以下条件，才能声称联合调用修改完成：

1. 所有入口均形成一次 `preview(s) + text -> task + visual plan` 调用；
2. 同一样本不再发生独立模型 TaskResolver + VisualPlanner 双调用；
3. 模型 task 成为物化、路由和实际执行 task；
4. TaskRouter 仍确定性且无模型依赖；
5. 大图 ROI 能通过共享裁切原语进入后续 Agent，且坐标可确定性回映；
6. 小图保持整图处理；
7. cache/budget/artifact/resume 契约有专门测试；
8. 旧 run 可安全识别，不被新 schema 静默误读；
9. UnifiedSample、路径安全、JSON-safe、secret 和模型单次组装契约未破坏；
10. 相关离线测试、架构测试与目标 checkpoint live gate 的实际结果如实记录。
