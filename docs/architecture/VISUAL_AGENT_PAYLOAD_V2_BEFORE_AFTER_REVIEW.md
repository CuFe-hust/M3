# Visual Agent Payload v2 修改前后对照审阅稿

## 1. 文档目的

本文档供实施前审阅，对照当前生产实现（Before）与拟议的
Visual Agent 最小模型 Payload / VQA SFT v2 契约（After）。

本文档只描述拟议变更，不表示生产代码已经实施。实施顺序、测试和
数据验收见 `VQA_AGENT_PAYLOAD_V2_MIGRATION_PLAN.md`。

本次范围：

- GeneralVQAAgent；
- CaptionAgent；
- GroundingAgent direct path；
- Grounding evidence final-Qwen path；
- VQA Agent SFT 记录 v1 → v2。

不在范围：

- ChangeAgent；
- CountingAgent；
- VisualTaskPlanner 首次规划调用；
- 评测指标、Ground Truth 解释和 Judge。

---

## 2. 总体责任变化

### Before

```text
UnifiedSample
    ↓
VisualAgentBase.build_user_payload()
    ↓ 为 Caption / Grounding / VQA 生成同一套字段
Agent 可选追加字段
    ↓
Evidence path 可能另外手写一套 payload
    ↓
Qwen
```

问题：

- 共享基类为不同任务输出无关字段；
- Multiple Choice 同时传入两份 choices/allow_multiple；
- direct 和 evidence 路径的基础任务契约可能漂移；
- 模型可见信息与审计/缓存身份混在同一 payload。

### After

```text
UnifiedSample（已经 Planner 物化/重建）
    ↓
具体 Agent.build_user_payload(sample)
    ↓ 只选择当前任务需要的 canonical 事实
Evidence path（如启用）
    ↓ 只追加当前调用的派生图像和动态证据
VisualAgentBase / service
    ↓ 组装 messages、hash、budget 和 response schema
Qwen
```

在 fresh runtime 中：

```text
payload.task == sample.task == visual_task_plan.task
```

dataset/source task 只用于 run namespace 与审计，不得覆盖 Planner task。

---

## 3. GeneralVQAAgent direct path

### 3.1 Before：General VQA

图像按 `sample.images` 顺序传入，随后是文本 JSON：

```json
{
  "question": "What color is the roof?",
  "task": "general_vqa",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "answer_constraints": {},
  "semantic_subtype": "attribute"
}
```

当 subtype 不存在时，当前实现仍可输出：

```json
{
  "answer_constraints": {},
  "semantic_subtype": null
}
```

### 3.2 After：General VQA

```json
{
  "question": "What color is the roof?",
  "task": "general_vqa",
  "semantic_subtype": "attribute",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

`semantic_subtype=None` 时省略该字段。不再传入空
`answer_constraints`。

### 3.3 Before：Multiple Choice VQA

```json
{
  "question": "Where is the bridge relative to the river?",
  "task": "multiple_choice_vqa",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "answer_constraints": {
    "type": "multiple_choice",
    "choices": ["(A) Above", "(B) Below", "(C) Left", "(D) Right"],
    "values": ["A", "B", "C", "D"],
    "closed": true,
    "allow_multiple": false
  },
  "semantic_subtype": "spatial_relation",
  "choices": ["(A) Above", "(B) Below", "(C) Left", "(D) Right"],
  "allow_multiple": false
}
```

同一份 choices 和 allow_multiple 被表达两次。

### 3.4 After：Multiple Choice VQA

```json
{
  "question": "Where is the bridge relative to the river?",
  "task": "multiple_choice_vqa",
  "choices": ["(A) Above", "(B) Below", "(C) Left", "(D) Right"],
  "allow_multiple": false,
  "semantic_subtype": "spatial_relation",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

选项事实在 canonical sample 中唯一保存为：

```text
sample.normalization.choices
sample.normalization.allow_multiple
```

缺失或非法 choices 必须在读图、消耗 budget 和调用 Qwen 前 fail closed。

### 3.5 Direct path 完整 message envelope（After）

```json
[
  {
    "role": "system",
    "content": "<versioned GeneralVQA prompt + one AgentResult JSON output contract>"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<normalized source image>"}
      },
      {
        "type": "text",
        "text": "<one of the JSON payloads above>"
      }
    ]
  }
]
```

response schema 为 `AgentResult`。

---

## 4. GeneralVQAAgent evidence path

### 4.1 Before

当 `needs_visual_assistance=true` 时，当前实现从基础 payload 出发，
再在顶层追加：

```json
{
  "images": [],
  "rois": [],
  "requested_leaves": [],
  "rendered_yolo_leaves": [],
  "rendered_segformer_leaves": [],
  "missing_leaves": [],
  "yolo_detections": [],
  "segformer_hits": [],
  "visual_inputs": [],
  "mask_legend": [],
  "evidence_identity": {
    "catalog_version": "...",
    "preprocessing_version": "...",
    "palette_version": "...",
    "visual_content_version": "..."
  }
}
```

主要问题：

- evidence 字段与基础任务字段混在顶层；
- `source_size`、`crop_xyxy` 等回映/审计几何会给最终模型；
- catalog/preprocessing/palette 版本是缓存与协议身份，不是问答事实；
- 内部 `leaf` 命名泄漏到模型协议。

### 4.2 After：图像内容

逐 ROI 图像分支：

```text
YOLO only              -> annotated_roi
SegFormer only         -> segformer_pure_mask + clean_roi
YOLO + SegFormer       -> yolo_on_segformer_pure_mask + clean_roi
neither                -> clean_roi
```

图像传入顺序必须与 `visual_inputs.content_image_index` 一致。

### 4.3 After：完整 message envelope

```json
[
  {
    "role": "system",
    "content": "<versioned GeneralVQA prompt + one AgentResult JSON output contract>"
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
        "text": "<the complete JSON payload below>"
      }
    ]
  }
]
```

```json
{
  "task": "general_vqa",
  "question": "Are there bridges near the river?",
  "semantic_subtype": "proximity",
  "coordinate_frame": "roi_normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "evidence": {
    "visual_inputs": [
      {"content_image_index": 0, "roi_id": "roi-1", "role": "annotated_roi"},
      {"content_image_index": 1, "roi_id": "roi-2", "role": "segformer_pure_mask"},
      {"content_image_index": 2, "roi_id": "roi-2", "role": "clean_roi"}
    ],
    "rois": [
      {"roi_id": "roi-1", "image_id": "image-1", "crop_size": [512, 512]},
      {"roi_id": "roi-2", "image_id": "image-1", "crop_size": [512, 512]}
    ],
    "requested_categories": ["bridge", "water"],
    "detections": [
      {"category": "bridge", "roi_id": "roi-1", "box": [120, 180, 720, 680]}
    ],
    "segmentation_hits": [
      {"category": "water", "roi_id": "roi-2"}
    ],
    "missing_categories": [],
    "mask_legend": [
      {"category": "water", "color_rgb": [0, 128, 255]}
    ]
  }
}
```

Multiple Choice evidence 仅在顶层额外包含唯一份：

```json
{
  "choices": ["(A) Yes", "(B) No"],
  "allow_multiple": false
}
```

最终 Qwen 不看完整 VisualTaskPlan、reason codes、Ground Truth、
dataset/split/source task、detector confidence、本地路径或协议版本号。
这些必要审计身份继续进入 request hash、trace 或 evidence artifact。

---

## 5. CaptionAgent

### 5.1 Before

Caption 复用共享基类 payload：

```json
{
  "question": "Describe the scene.",
  "task": "caption",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "answer_constraints": {},
  "semantic_subtype": null
}
```

坐标、box format、空 constraints 和 null subtype 对 caption 模型无用。

### 5.2 After

```json
{
  "task": "caption",
  "question": "Describe the scene."
}
```

`question` 保留，因为 manual ask 和数据适配可以提供不同 caption 指令。

### 5.3 完整 message envelope（After）

```json
[
  {
    "role": "system",
    "content": "<versioned Caption prompt + AgentResult JSON output contract>"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<normalized source image>"}
      },
      {
        "type": "text",
        "text": "{\"task\":\"caption\",\"question\":\"Describe the scene.\"}"
      }
    ]
  }
]
```

response schema 仍为 `AgentResult`，其中 `boxes` 和 `evidence_items` 必须为空。

---

## 6. GroundingAgent direct path

### 6.1 Before

```json
{
  "question": "Locate the bridge.",
  "task": "grounding",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "answer_constraints": {},
  "semantic_subtype": null
}
```

### 6.2 After

```json
{
  "task": "grounding",
  "question": "Locate the bridge.",
  "coordinate_frame": "normalized_0_999_top_left",
  "box_format": "integer_xyxy_json"
}
```

### 6.3 完整 message envelope（After）

```json
[
  {
    "role": "system",
    "content": "<versioned Grounding direct prompt + AgentResult JSON output contract>"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<normalized source image>"}
      },
      {
        "type": "text",
        "text": "{\"task\":\"grounding\",\"question\":\"Locate the bridge.\",\"coordinate_frame\":\"normalized_0_999_top_left\",\"box_format\":\"integer_xyxy_json\"}"
      }
    ]
  }
]
```

response schema 为 `AgentResult`。

---

## 7. Grounding evidence final-Qwen path

### 7.1 Before

当前 final-Qwen 接收 clean ROI 图像和：

```json
{
  "question": "Locate the bridge.",
  "catalog_version": "...",
  "coordinate_frame": "roi_normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "rois": [
    {
      "roi_id": "roi-1",
      "image_id": "image-1",
      "source_size": [4096, 4096],
      "crop_size": [512, 512]
    }
  ],
  "candidates": [
    {
      "box_id": "box-1",
      "leaf_category": "bridge",
      "roi_id": "roi-1",
      "xyxy": [120, 210, 640, 720]
    }
  ],
  "missing_leaves": []
}
```

当前 evidence final-Qwen 复用通用 VQA prompt，没有独立声明 candidate
selection 和 missing-category fallback 契约。

### 7.2 After：System prompt 职责

新增独立、版本化的 Grounding final-Qwen prompt，必须明确：

- 已有 candidates 的 category 只能选择现有 candidate ID；
- 只有 missing category 允许生成 fallback box；
- fallback box 使用 ROI-local `0..999` integer xyxy；
- 不输出 confidence 或 hidden reasoning；
- 只输出匹配 `GroundingQwenResponse` 的 JSON。

### 7.3 After：完整 message envelope

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
        "text": "<the complete JSON payload below>"
      }
    ]
  }
]
```

```json
{
  "task": "grounding",
  "question": "Locate the bridge.",
  "coordinate_frame": "roi_normalized_0_999_top_left",
  "box_format": "integer_xyxy_json",
  "evidence": {
    "visual_inputs": [
      {"content_image_index": 0, "roi_id": "roi-1", "role": "clean_roi"},
      {"content_image_index": 1, "roi_id": "roi-2", "role": "clean_roi"}
    ],
    "rois": [
      {"roi_id": "roi-1", "image_id": "image-1", "crop_size": [512, 512]},
      {"roi_id": "roi-2", "image_id": "image-1", "crop_size": [512, 512]}
    ],
    "candidates": [
      {
        "candidate_id": "box-1",
        "category": "bridge",
        "roi_id": "roi-1",
        "box": [120, 210, 640, 720]
      }
    ],
    "missing_categories": ["water"]
  }
}
```

### 7.4 After：Response schema

```json
{
  "selected_box_ids": ["box-1"],
  "fallback_boxes": [
    {
      "leaf_category": "water",
      "roi_id": "roi-2",
      "xyxy": [250, 160, 710, 680]
    }
  ]
}
```

`selected_box_ids` 与 `fallback_boxes` 按 category 互斥。Qwen 只输出
ROI-local 选择/候选几何；生产代码使用持久化 ROI 几何确定性回映
到整图坐标，再构造最终 `AgentResult`。

### 7.5 After：不再给最终 Qwen 的信息

```text
catalog_version
source_size / core_xyxy / expanded_xyxy
detector confidence
unselected detector-internal candidates
weights/model/cache identity
local paths
full VisualTaskPlan / reason_codes / needs_visual_assistance
```

这些信息中，会影响 evidence 语义、回映或可复现性的部分仍必须
进入 request hash、trace 或 evidence artifact，不得丢失。

---

## 8. VQA Agent SFT 记录

### 8.1 Before：`vqa-agent-sft-v1`

```json
{
  "schema_version": "vqa-agent-sft-v1",
  "input": {
    "visual_task_plan": {},
    "agent_input": {
      "sample": {},
      "user_payload": {}
    }
  },
  "output": {
    "agent_result": {}
  },
  "supervision": {}
}
```

`user_payload` 容易被理解为 evidence path 最终完整 messages，而实际上
它只代表生产基础 payload。

### 8.2 After：`vqa-agent-sft-v2`

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

每条记录必须满足：

```python
record["input"]["agent_input"]["base_user_payload"] \
    == production_payload_builder(validated_sample)
```

VQA train/validation 会重新生成，但不改变：

- sample ID；
- question 和答案；
- Ground Truth；
- split；
- 图片映射；
- 样本数量和纳入规则。

原始数据集与图片保持只读。

---

## 9. 不变的路径

### ChangeAgent

initial、adjudication 和 building-rescue 的 system prompt、user content、
payload、图像顺序、response schema 和 request-hash 输入保持不变。

### CountingAgent

quantity proposal、localization、Qwen point、empty review、seam review 和
detector disagreement review 的所有模型输入保持不变。

---

## 10. 审阅决策表

| 审阅点 | Before | After | 需确认 |
|---|---|---|---|
| VQA constraints | `answer_constraints` + MC 顶层选项 | 只保留一份 MC 顶层选项 | 是否接受旧 cache 自然失效 |
| Caption | 共享坐标/constraints payload | 只保留 task + question | 是否保留 question |
| Grounding direct | 包含空 constraints/null subtype | 只保留必要任务和坐标字段 | 无 |
| VQA evidence | 证据、基础字段和版本身份混合 | `base payload + evidence` | 是否采用 `evidence` 嵌套 |
| Grounding evidence prompt | 复用通用 VQA prompt | 独立版本化 prompt | prompt 内容和版本号 |
| Grounding evidence identity | catalog/source geometry 给 Qwen | 只进入 hash/trace/artifact | 确认 hash 额外输入 |
| Change/Counting | 当前契约 | 不变 | golden 回归通过 |

## 11. 审阅通过后的实施门禁

实施必须同时证明：

1. fresh path 的 `payload.task == sample.task == visual_task_plan.task`；
2. direct/evidence 共用同一权威基础 payload builder；
3. 每个 evidence image block 有稳定 content-index/ROI binding；
4. 不给模型的审计身份仍完整进入 hash/trace/artifact；
5. Grounding ROI-local 坐标回映的确定性结果不变；
6. 无 choices 的 MC 样本在读图/budget/Qwen 前失败；
7. VQA v2 数据与生产 builder 逐条相等；
8. sample ID、答案、split、图片和样本数量不变；
9. Change/Counting 捕获的 messages、response schema 和 request hash 不变。
