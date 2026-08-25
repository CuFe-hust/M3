# Visual Agent 最小模型 Payload 与 VQA SFT v2 迁移计划

## 1. 目标与边界

本次迁移的判断标准是：**每个模型调用只接收完成当前决策所需的
充足信息，不接收重复事实、空占位字段或只供运行时审计使用的信息**。

信息分为两类：

- 模型可见信息：完成当前任务确实需要的文本、图像、候选项、ROI 和候选几何；
- 运行时身份与审计信息：版本号、catalog identity、原图回映几何、
  detector 内部信息等，继续进入 request hash、trace 或 artifact，但除非
  模型决策确实依赖，不进入模型 payload。

本次生产代码范围：

- `GeneralVQAAgent`：`general_vqa`、`scene_classification`、
  `spatial_relation`、`multiple_choice_vqa`；
- `CaptionAgent`；
- `GroundingAgent` direct path 与 grounding evidence final-Qwen path。

明确不在本次范围：

- `ChangeAgent` 及其 initial/adjudication/building-rescue payload；
- `CountingAgent` 及其 quantity proposal、Qwen point、seam/disagreement
  review payload；
- VisualTaskPlanner 的首次规划输入契约。

以上不在范围的 payload 不做字段重命名、删除或结构收口，只运行
回归测试确认未发生行为漂移。

### 1.1 VQA SFT 数据目标

将 VQA Agent 顶层训练记录升级为 `vqa-agent-sft-v2`，明确区分以下职责：

- `sample`：完整、规范化的 `UnifiedSample`；
- `visual_task_plan`：通过 `AgentContext.visual_task_plan` 控制执行路径；
- `base_user_payload`：由生产代码实际构造的基础文本载荷；
- evidence path 的检测结果、ROI 和最终 multimodal messages 不伪装成
  `base_user_payload`；
- `output.agent_result`：顶层 Agent 输出；
- `supervision`：训练目标及 loss 范围。

目标记录结构：

```json
{
  "schema_version": "vqa-agent-sft-v2",
  "input": {
    "visual_task_plan": {},
    "agent_input": {
      "sample": {},
      "base_user_payload": {}
    }
  },
  "output": {
    "agent_result": {}
  },
  "supervision": {}
}
```

## 2. 每个 Agent 实际给模型的信息

本节是本次迁移后的权威模型可见契约。图像作为 multimodal content
独立传入，下面的 JSON 只表示同一 user message 中的文本 payload。

### 2.1 GeneralVQAAgent

#### General VQA

```json
{
  "question": "What color is the roof?",
  "task": "general_vqa",
  "semantic_subtype": "attribute",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

本次迁移不设计或传递通用答案约束；开放问答不增加闭集答案字段。

#### Scene Classification

```json
{
  "question": "What type of scene is shown?",
  "task": "scene_classification",
  "semantic_subtype": "scene_classification",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

#### Spatial Relation

```json
{
  "question": "Where is the bridge relative to the river?",
  "task": "spatial_relation",
  "semantic_subtype": "spatial_relation",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

#### Multiple Choice VQA

```json
{
  "question": "Where is the bridge relative to the river?",
  "task": "multiple_choice_vqa",
  "choices": [
    "(A) Above",
    "(B) Below",
    "(C) Left",
    "(D) Right"
  ],
  "allow_multiple": false,
  "semantic_subtype": "spatial_relation",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

约束规则：

- 多选题只使用顶层 `choices` 和 `allow_multiple`；
- 本次迁移完全不在模型 payload 中使用 `answer_constraints`；
- 多选题的结构化事实改由 `sample.normalization.choices` 和
  `sample.normalization.allow_multiple` 保存；
- 不允许同一个 payload 同时出现两份 choices 或
  allow_multiple；
- `semantic_subtype=None` 时可以省略该字段。

GeneralVQA direct path 传入按 `sample.images` 稳定顺序读取的规范图像，
再传入上述基础 payload。不传入 Ground Truth、答案、数据集名、
source task 或完整 VisualTaskPlan JSON。

#### GeneralVQA evidence final-Qwen path

GeneralVQA evidence path 先构造同一份基础 payload，再向最终 Qwen 请求
增加实际派生图像和 evidence extension。最终 Qwen 的 system message
继续使用当前 GeneralVQA 版本化 prompt 及唯一份结构化输出契约；
不将 VisualTaskPlan JSON 拼入 system 或 user message。

逐 ROI 模型可见图像分支为：

```text
YOLO only
    -> annotated_roi

SegFormer only
    -> segformer_pure_mask
    -> clean_roi

YOLO + SegFormer
    -> yolo_on_segformer_pure_mask
    -> clean_roi

neither
    -> clean_roi
```

一次完整最终请求的模型可见 message envelope 为：

```json
[
  {
    "role": "system",
    "content": "<versioned GeneralVQA system prompt + AgentResult JSON output contract>"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<roi-1 annotated image>"}
      },
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<roi-2 pure mask>"}
      },
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<roi-2 clean image>"}
      },
      {
        "type": "text",
        "text": "<the JSON payload below>"
      }
    ]
  }
]
```

其中完整文本 payload 为：

```json
{
  "task": "general_vqa",
  "question": "Are there bridges near the river?",
  "semantic_subtype": "proximity",
  "coordinate_frame": "roi_normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "evidence": {
    "visual_inputs": [
      {
        "content_image_index": 0,
        "roi_id": "roi-1",
        "role": "annotated_roi"
      },
      {
        "content_image_index": 1,
        "roi_id": "roi-2",
        "role": "segformer_pure_mask"
      },
      {
        "content_image_index": 2,
        "roi_id": "roi-2",
        "role": "clean_roi"
      }
    ],
    "rois": [
      {
        "roi_id": "roi-1",
        "image_id": "image-1",
        "crop_size": [512, 512]
      },
      {
        "roi_id": "roi-2",
        "image_id": "image-1",
        "crop_size": [512, 512]
      }
    ],
    "requested_categories": ["bridge", "water"],
    "detections": [
      {
        "category": "bridge",
        "roi_id": "roi-1",
        "box": [120, 180, 720, 680]
      }
    ],
    "segmentation_hits": [
      {
        "category": "water",
        "roi_id": "roi-2"
      }
    ],
    "missing_categories": [],
    "mask_legend": [
      {
        "category": "water",
        "color_rgb": [0, 128, 255]
      }
    ]
  }
}
```

对 `multiple_choice_vqa` evidence path，同一结构还必须在顶层保留
唯一份 `choices` 和 `allow_multiple`。该 evidence extension 不属于
`base_user_payload`。最终 response schema 仍为 `AgentResult`；request hash 必须
覆盖基础 payload、extension、完整 messages 与实际派生图像摘要。

最终 GeneralVQA Qwen 不看：

- 完整 VisualTaskPlan、`needs_visual_assistance`、plan reason codes；
- Ground Truth、答案、dataset/split/source task；
- detector confidence、本地路径、model/cache identity；
- catalog/preprocessing/palette 版本号；这些仍必须进入
  request hash、trace 或 evidence artifact。

### 2.2 CaptionAgent

Caption 模型接收按 `sample.images` 顺序读取的图像，文本 payload 为：

```json
{
  "task": "caption",
  "question": "Describe the scene."
}
```

`question` 保留，因为它是 `UnifiedSample` 的显式用户输入，且不同数据或
manual ask 可以提供不同的 caption 指令。不传入：

- `coordinate_frame`；
- `box_format`；
- 空 `answer_constraints`；
- `semantic_subtype=None`。

Caption 的输出格式要求继续由版本化 system prompt 与 response schema 约束，
不在 user payload 中重复。

### 2.3 GroundingAgent

#### Direct path

Grounding direct path 接收原图和：

```json
{
  "task": "grounding",
  "question": "Locate the bridge.",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

不传入空 `answer_constraints` 或 `semantic_subtype=None`。

#### Evidence final-Qwen path

Grounding evidence 的最终 Qwen 接收 clean ROI 图像与结构化 evidence
payload。

该调用必须使用版本化 Grounding final-Qwen system prompt，明确告知模型：

- 对已给 candidates 的 category，只能选择已有 `candidate_id`；
- 对 `missing_categories` 才允许生成 ROI-local fallback box；
- 坐标为 ROI-local `0..999` top-left integer xyxy；
- 只输出匹配 `GroundingQwenResponse` 的 JSON，不输出 confidence 或
  hidden reasoning。

一次完整最终请求的模型可见 message envelope 为：

```json
[
  {
    "role": "system",
    "content": "<versioned Grounding final-Qwen prompt + GroundingQwenResponse contract>"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<roi-1 clean image>"}
      },
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<roi-2 clean image>"}
      },
      {
        "type": "text",
        "text": "<the JSON payload below>"
      }
    ]
  }
]
```

完整文本 payload 为：

```json
{
  "task": "grounding",
  "question": "Locate the bridge.",
  "coordinate_frame": "roi_normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "evidence": {
    "visual_inputs": [
      {
        "content_image_index": 0,
        "roi_id": "roi-1",
        "role": "clean_roi"
      },
      {
        "content_image_index": 1,
        "roi_id": "roi-2",
        "role": "clean_roi"
      }
    ],
    "rois": [
      {
        "roi_id": "roi-1",
        "image_id": "image-1",
        "crop_size": [512, 512]
      },
      {
        "roi_id": "roi-2",
        "image_id": "image-1",
        "crop_size": [512, 512]
      }
    ],
    "candidates": [
      {
        "candidate_id": "box-1",
        "category": "bridge",
        "roi_id": "roi-1",
        "box": [120, 210, 640, 720]
      }
    ],
    "missing_categories": []
  }
}
```

最终 response schema 为：

```json
{
  "selected_box_ids": ["box-1"],
  "fallback_boxes": [
    {
      "leaf_category": "bridge",
      "roi_id": "roi-2",
      "xyxy": [250, 160, 710, 680]
    }
  ]
}
```

`selected_box_ids` 与 `fallback_boxes` 的互斥授权由 response schema 和
确定性 postprocess 验证。最终 AgentResult 中的整图坐标由生产代码
根据已持久化 ROI 几何回映，不由 Qwen 计算。

图像顺序与 `evidence.rois` 顺序必须一致，并使用明确的 ROI
content 绑定/清单测试防止错位。以下信息继续进入 request hash、
trace 或 evidence artifact，但不给最终 Qwen：

- `catalog_version`；
- 原图 `source_size` 及 ROI 回映几何；
- detector confidence 和未选中候选；
- 只用于运行时审计的内部 catalog leaf identity。

删除上述字段前必须先证明它们不参与 prompt 语义解释或确定性坐标
回映；坐标回映仍由生产代码使用已持久化的安全几何完成。

### 2.4 不修改的 Agent

Change 和 Counting 给各自模型/backend 的所有 messages、文本 payload、图像顺序、
response schema 和 request-hash 输入完全保持现状。本次不借机引入统一
envelope，也不移动其 prompt instruction。

## 3. VisualTaskPlan 的位置与作用

`visual_task_plan` 不进入 `base_user_payload`，而是作为 Agent 的运行上下文：

```text
Agent 输入
├── sample: UnifiedSample
├── context.visual_task_plan: VisualTaskPlan
└── context 中的运行依赖
        ↓
Agent 根据 plan 决定执行路径
        ↓
构造最终 Qwen 输入
```

### 3.1 `payload.task` 的权威来源

Agent payload builder 不读 dataset name、source task 或 question 来重新推断
task。它只读取已验证的：

```python
payload["task"] = sample.task
```

在 fresh manual/dataset runtime 中，权威链路必须是：

```text
dataset/source task
    │ 只用于 run namespace 与审计
    ▼
VisualTaskPlanner
    │ plan.task 是 fresh task 决策权威
    ▼
materialize/rebuild UnifiedSample
    │ sample.task = plan.task
    ▼
Agent.build_user_payload(sample)
    │ payload.task = sample.task
    ▼
final Qwen
```

因此 fresh path 必须满足：

```text
payload.task == sample.task == visual_task_plan.task
```

dataset/source/adapter task 不得覆盖 Planner 结果。对于不经 fresh
Planner 的显式 direct compatibility/test 入口，已验证的
`UnifiedSample.task` 是权威；Agent 仍不自行重新分类。

### 3.2 Direct Path

```text
sample
+ visual_task_plan（只决定 direct path）
→ base_user_payload
→ Qwen
```

当 `needs_visual_assistance=false` 时，VisualTaskPlan 已经完成路径选择职责，
Qwen 不需要再次接收完整 Plan JSON。

### 3.3 Evidence Path

```text
sample
+ visual_task_plan
→ ROI / object categories / visual assistance
→ base_user_payload + evidence content + derived images
→ Qwen
```

需要保持：

- `base_user_payload` 不冒充完整 evidence messages；
- evidence path 的最终 request hash 使用实际 messages 和实际图像摘要；
- payload 改变后旧缓存自然失效；
- deterministic metrics 和 Judge 不受影响。

### 3.4 非 direct 时最终模型看到什么

`needs_visual_assistance=true` 时，最终 Agent Qwen 不看完整
VisualTaskPlan JSON。Planner 的控制信息必须先被物化成当前调用的
真实视觉证据：

| Planner 信息 | 最终 Agent 模型可见形式 |
|---|---|
| `task` | 通过重建后的 `sample.task` 进入 `payload.task` |
| `needs_visual_assistance` | 不作为字段传入；通过选择 evidence path 体现 |
| `object_categories` | 实际 requested categories、candidates/detections、mask hits 和 missing categories |
| `region_request` | 实际 ROI 图像、ROI ID 和 image-content binding |
| `reason_codes` | 不给最终模型；只保留在 plan artifact/trace |
| `version` | 不给最终模型；继续进入 snapshot/artifact/hash identity |

GeneralVQA evidence final-Qwen 完整 user content 顺序为：

```text
derived ROI image 0
derived ROI/mask image 1
...
JSON text:
    base VQA payload
    + actual visual input binding
    + ROI records
    + detections/segmentation hits
    + missing categories/mask legend
```

Grounding evidence final-Qwen 完整 user content 顺序为：

```text
clean ROI image 0
clean ROI image 1
...
JSON text:
    Grounding base payload
    + ROI/image binding
    + candidate IDs/categories/ROI-local boxes
    + missing categories
```

每个图像块都必须能通过稳定 `content_image_index` / `roi_id` 清单与
JSON evidence 对应，禁止只靠未声明的偶然顺序。最终 request hash 必须
覆盖完整 messages、实际图像摘要、response schema 和不给模型但会
影响证据语义的协议/catalog/preprocessing identity。

## 4. 生产代码修改

主要修改：

- `data/schema.py`；
- `agents/visual_base.py`；
- `agents/general_vqa/agent.py`；
- `agents/caption/agent.py`；
- `agents/grounding/agent.py`；
- `agents/grounding/evidence.py`；
- `application/prompts.py` 与新的版本化 Grounding final-Qwen prompt asset；
- 所有能产生 `multiple_choice_vqa` 的 dataset adapter 及
  `SampleDraft` materialization/rebuild 路径。

实施内容：

1. `TaskNormalization` 增加专用的 `choices: list[str]` 和
   `allow_multiple: bool` 字段，并对 `multiple_choice_vqa` 做结构校验；
2. 共享基类不再无条件生成同一套 payload；由 task-aware 的权威
   builder 只生成第 2 节声明的字段；
3. GeneralVQAAgent、CaptionAgent 和 GroundingAgent 各自有一个权威的
   `build_user_payload(sample)` 构造入口，共享仅限于真正共同的纯构造
   原语，不再为了形式统一输出无关字段；
4. `multiple_choice_vqa` 从 `sample.normalization.choices` 和
   `sample.normalization.allow_multiple` 读取多选题信息，并只在 payload 顶层输出一次；
5. VisualTaskPlan 保持在 `AgentContext.visual_task_plan`，不复制到
   payload；
6. direct path 和 evidence path 共用同一个基础 payload 构造结果；
7. 不为训练数据创建一套平行的 payload 构造实现；
8. 本次迁移不引入 closed-vocabulary、values 或其他通用答案约束协议；
9. Caption 不输出坐标、box format、空 constraints 或 null subtype；
10. Grounding direct path 复用 Grounding 权威基础 builder；grounding evidence
    从该基础事实构造 ROI 扩展，不手写第二份 question/task 契约；
11. Grounding evidence 的 catalog version、source geometry 等审计信息仍必须
    被 request hash 和持久化 artifact 覆盖，不得因为不给模型而丢失；
12. Change 和 Counting 不修改任何模型可见输入或缓存身份；
13. Grounding evidence final-Qwen 使用独立、版本化且由
    PromptCatalog 绑定的 prompt，不再借用未声明 candidate/fallback
    输出协议的通用 VQA prompt。

Change 和 Counting 必须运行相关回归测试，避免共享代码修改导致
行为漂移。

## 5. 生产测试

需要更新或新增：

- `tests/agents/general_vqa/test_agent.py`；
- `tests/agents/caption/test_agent.py`；
- `tests/agents/grounding/test_agent.py`；
- grounding evidence service 的 request/payload/hash 单元测试；
- PromptCatalog 中 Grounding final-Qwen prompt 的绑定、snapshot 与版本测试；
- `tests/integration/test_general_vqa_vertical_slice.py`；
- `tests/integration/test_auto_task_dataset_vertical_slice.py`；
- 相关 request-hash 和 model-cache 测试。

核心断言：

```python
payload == {
    "question": "...",
    "task": "multiple_choice_vqa",
    "choices": ["..."],
    "allow_multiple": False,
    # common coordinate and semantic fields
}
assert "answer_constraints" not in payload
```

覆盖场景：

- 两选项和四选项；
- 单选和多选；
- 字母答案和选项文本答案；
- 无 choices 时稳定失败；
- 非法答案降级为 `partial`；
- choices 不泄漏 ground truth；
- `semantic_subtype` 正确保留；
- fresh direct/evidence 两条路径都满足
  `payload.task == sample.task == visual_task_plan.task`；
- dataset/source task 与 Planner task 不同时，payload 使用 Planner task，
  source task 只保留在审计信息；
- direct/evidence 路径使用同一个基础 payload；
- request hash 覆盖修改后的真实 messages；
- Caption payload 只包含 task 和非空 question；
- Grounding direct payload 只包含 task、question 和坐标输出契约；
- Grounding evidence payload 包含 task、question、ROI 坐标契约与实际
  evidence，不包含 catalog/source-geometry 审计字段；
- Grounding ROI 图像顺序与 ROI binding 一致；
- 非 direct payload 不包含完整 VisualTaskPlan、reason codes 或
  `needs_visual_assistance`；
- GeneralVQA/Grounding evidence 每个图像块都有唯一且稳定的
  content-index/ROI binding；
- Grounding 坐标回映结果与迁移前确定性结果一致；
- Change 和 Counting 的捕获 messages/request hash golden 不变。

## 6. 数据 Schema 升级

修改：

- `data/2026-08-24_vqa-agent-io/record.schema.json`；
- schema version 从 `vqa-agent-sft-v1` 升到
  `vqa-agent-sft-v2`；
- `user_payload` 重命名为 `base_user_payload`。

不在单条记录中同时保留：

```json
{
  "user_payload": {},
  "base_user_payload": {}
}
```

如需兼容 v1，应通过独立、显式的离线转换处理，而不是在 v2 记录中保存两份字段。

## 7. 数据重新生成

需要重新生成 `data/2026-08-24_vqa-agent-io/` 下所有 train 和 validation：

- VRSBench；
- DOTA；
- HRSCD；
- MiniFrance；
- LRS-VQA-Supplement。

生成流程必须复用生产入口：

```python
sample = UnifiedSample.model_validate(...)
payload = production_payload_builder(sample)
```

并逐条验证：

```python
record["input"]["agent_input"]["base_user_payload"] \
    == production_payload_builder(sample)
```

禁止在转换脚本中手写一份平行的 payload 逻辑。
该入口必须是纯构造且不需要加载模型、读取图像、消耗 budget 或联网。

迁移前必须先收口 choices 的生产来源：

- 所有 `multiple_choice_vqa` adapter 将 choices/allow_multiple 写入
  canonical `TaskNormalization`；
- metadata-only 历史路径不得成为 Agent 的新权威读取路径；
- `SampleDraft -> materialize_sample(...)` 和 planner task rebuild 必须明确保留
  已验证的选项事实，或在 Qwen 调用、图像读取和 budget 消耗前
  以稳定 error code 失败；
- 不得从 Ground Truth 或答案反向推导 choices。

DOTA、HRSCD、MiniFrance 继续保持：

```text
task = multiple_choice_vqa
semantic_subtype = 原 general/scene/spatial 语义
```

答案、split、样本数量、sample ID 和图片映射不得变化。

## 8. Visual Planner 数据

`data/2026-08-25_visual-planner-multiple-choice/` 不包含 Agent payload，
因此无需加入 `base_user_payload`。

只做交叉校验：

- Planner target 是 `multiple_choice_vqa`；
- Planner 输入包含完整选项；
- Planner 输入不包含答案；
- VQA Agent sample task 与 Planner task 一致；
- `visual_task_plan` 在 Agent 数据中独立保存；
- train/val 与图片分组不变。

原始 `data/phase2-train-visualplanning-refined-v4/` 保持不变。

## 9. 查看器与文档

更新：

- `data/2026-08-24_vqa-agent-io/train_annotation_viewer.html`；
- VQA Agent I/O README；
- `DETAILS.md` 中 Agent 输入与 payload 契约；
- 必要的架构说明。

查看器分开显示：

1. 源数据集原始记录；
2. `UnifiedSample`；
3. `VisualTaskPlan`；
4. 生产生成的 `base_user_payload`；
5. `AgentResult`；
6. supervision。

## 10. 验收标准

必须全部满足：

- 所有记录通过 `UnifiedSample` 校验；
- 所有 `base_user_payload` 与生产构造方法逐条相等；
- payload 内不存在重复 choices/allow_multiple；
- Caption 和 Grounding direct payload 不存在空 constraints、null subtype 或
  与当前任务无关的坐标字段；
- Grounding evidence 的模型可见 payload 与运行时审计信息已分离；
- fresh direct/evidence 的 `payload.task` 与已持久化 Planner task、实际
  execution task 一致；
- source/adapter task 冲突用例证明其无法覆盖 Planner task；
- 非 direct 模型输入只包含物化 evidence，不包含完整 Plan JSON；
- Grounding evidence 仍可确定性回映到整图坐标，且结果不变；
- VisualTaskPlan 不进入基础文本 payload；
- direct/evidence 路径测试通过；
- request hash 覆盖修改后的真实 messages；
- train/validation 数量不变；
- sample ID、答案、图片和 split 不变；
- DOTA/HRSCD/MiniFrance 仍为 1,275 条多选题；
- Planner 输入无答案泄漏；
- 源数据目录未修改；
- manifest 数量和 SHA256 全部重新计算并验证；
- Change/Counting 模型 messages、response schema 和 request hash 捕获值不变；
- 无法运行的测试或验证项如实记录。

## 11. 推荐实施顺序

1. 冻结 GeneralVQA、Caption、Grounding direct/evidence 的精确模型可见
   payload 与图像顺序；
2. 收口全部 multiple-choice adapter、draft materialization 和 rebuild 中的
   choices 传播；
3. 修改 task-aware 生产 builder，并确保缺失 choices 在模型调用前
   fail closed；
4. 修改 Grounding evidence 最终请求，验证 ROI binding、坐标回映、
   artifact 和 request hash；
5. 运行 Change/Counting 不变 golden 回归；
6. 升级 `record.schema.json` 到 v2；
7. 使用生产构造入口离线、原子地重新生成全部 VQA Agent I/O；
8. 执行 Planner 与 Agent 数据交叉校验；
9. 更新查看器、README、`DETAILS.md` 和架构文档；
10. 运行目标测试、集成测试、架构测试和全量数据验证。
