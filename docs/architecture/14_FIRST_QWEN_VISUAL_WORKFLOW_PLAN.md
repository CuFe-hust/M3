# 14 — 第一次 Qwen 视觉工作流规划输出

> Status: discussion decision draft; no production implementation yet.
> 状态：讨论结论草案；尚未进入生产实现。

> VQA 子工作流的后续冻结结论见
> [`14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md`](./14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md)。
> 若本文关于 VQA 分支、按类别调用模型、编号框、置信度暴露、有效空结果、ROI
> 坐标或 SegFormer 输出形态的早期设想与 14B 冲突，以 14B 为准。
>
> Grounding 子工作流的后续冻结结论见
> [`14C_GROUNDING_AGENT_SUBWORKFLOW.md`](./14C_GROUNDING_AGENT_SUBWORKFLOW.md)。
> 若本文关于 Grounding 的编号标注图、SegFormer 回退、候选选择或坐标形态的早期
> 设想与 14C 冲突，以 14C 为准。

## 1. 文档目的

本文冻结当前讨论中已经明确的“第一次 Qwen 调用”职责和结构化输出标签，
作为后续 Schema、Prompt、工作流和测试设计的输入。

本文不是当前代码事实说明，不表示以下能力已经实现，也不修改现有
`UnifiedSample`、TaskRouter、评测或 resume 契约。

## 2. 核心职责分离

数据集任务身份与内部执行规划必须分离：

```text
dataset task / answer protocol
    -> 保留数据集的答案格式、选项约束与评测身份

first Qwen visual plan
    -> 决定完成问题所采用的内部工作流和所需视觉证据
```

例如，一道询问飞机数量的选择题仍按多选题协议输出和评测，但第一次 Qwen
可以把内部执行工作流规划为 counting 或 object-assisted VQA。

第一次 Qwen 对内部执行规划具有优先权，但不得改写 Ground Truth、数据集
评测身份或答案协议。最终答案格式由对应任务 Prompt 约束。

## 3. 第一次 Qwen 的输入

第一次 Qwen 接收：

```text
原始文本问题
+ 图片预览
+ 必要的外部答案协议信息（例如 choices）
```

图片预处理规则为：

```text
最长边 > 1080：等比例缩小到最长边 1080
最长边 <= 1080：保持原尺寸，不放大
```

预览图只用于规划。后续 ROI 必须从应用 EXIF 方向后的原始图像裁切，不能从
1080 预览图二次裁切。

## 4. 第一次 Qwen 的职责边界

第一次 Qwen 只负责声明：

1. 内部工作流族；
2. 是否需要物体位置证据；
3. 需要检测的系统级组合类别；
4. YOLO 应观察的全图或核心注意区域；
5. 最终 Qwen 应接收普通 ROI 图还是带编号检测框的 ROI 图；
6. 简短、可审计的置信度与 reason codes。

第一次 Qwen 不负责：

- 选择 YOLO、SegFormer checkpoint 或具体 backend；
- 输出模型路径、processor、device 或权重信息；
- 判断某个组合类别由 YOLO 还是 SegFormer 执行；
- 根据 `largest`、`closest`、颜色、状态等语义筛选检测候选；
- 输出最终答案；
- 修改数据集任务身份或评测方式。

具体模型选择和 fallback 由确定性工作流根据 capability catalog 执行。

## 5. 结构化输出契约

当前讨论结论建议冻结为以下逻辑 Schema：

```json
{
  "schema_version": "first-qwen-plan-v1",
  "execution_family": "vqa",
  "confidence": 0.93,
  "object_evidence": {
    "required": true,
    "composite_categories": ["aircraft", "vessel"]
  },
  "roi_plan": {
    "scope": "attention_regions",
    "regions": [
      {
        "image_id": "image-0",
        "box": [0, 0, 500, 500],
        "coordinate_frame": "normalized_0_999_top_left"
      }
    ]
  },
  "final_answer_input": {
    "image_mode": "numbered_roi_overview"
  },
  "reason_codes": [
    "object_location_evidence_required",
    "explicit_top_left_constraint"
  ]
}
```

### 5.1 `execution_family`

封闭枚举：

```text
caption
grounding
change
counting
vqa
```

它表示内部工作流族，不是新的 public `TaskName`，也不覆盖
`UnifiedSample.task`。通用 Qwen 是共享模型客户端和 fallback，不是第六个
业务 Agent。

### 5.2 `object_evidence.required`

```text
true
    最终回答需要小模型提供物体位置证据

false
    直接由 Qwen 观察普通预览图或 ROI 作答
```

场景分类、城乡判断、环境条件、住宅区/工业区等高层场景语义在当前方案中
不使用独立 SegFormer 场景分类路径，默认直接交给 Qwen。

### 5.3 `object_evidence.composite_categories`

约束：

- 只能从系统提供的封闭组合类别表中选择；
- 不允许输出自由文本类别；
- 去重后最多三个类别；
- 输出的是高召回组合类别，不是模型原始细分类别；
- 类别没有语义角色，最终 Qwen 根据原始问题理解类别之间的比较、参照和关系。

示意映射：

```text
aircraft
    -> plane + helicopter

vehicle
    -> small vehicle + large vehicle
```

因此对于“飞机是否比直升机多”，第一次 Qwen 可以只输出 `aircraft`；小模型
检测结果仍保留 `plane`、`helicopter` 等具体标签，供最终 Qwen 判断。

组合类别表的最终取值必须根据经过验证的 YOLO 与 SegFormer class maps 单独
校准。本文件只冻结“封闭、组合、高召回、最多三个”的契约，不提前猜测完整
类别表。

### 5.4 `roi_plan.scope`

封闭枚举：

```text
full_image
attention_regions
```

规则：

- 没有明确空间约束时，必须尽量保持全图；
- “左上角”“中心”“右侧”等明确位置可以输出核心注意区域；
- “城市里”“道路附近”等语义区域只有在预览图中可可靠确定时才能输出局部区域；
- 语义区域难以确定时必须回退 `full_image`，不得勉强猜测 ROI；
- YOLO 只扫描 `roi_plan` 指定的范围，因此 ROI 是实际扫描边界，不只是提示文本。

`full_image` 的规范表示为：

```json
{
  "scope": "full_image",
  "regions": [
    {
      "image_id": "image-0",
      "box": [0, 0, 999, 999],
      "coordinate_frame": "normalized_0_999_top_left"
    }
  ]
}
```

每个核心 ROI 在执行前由工作流确定性增加上下文 halo。halo 后的边界必须限制
在原图范围内，并保留从 ROI 局部坐标到原图全局坐标的确定性变换。

### 5.5 `final_answer_input.image_mode`

当前阶段只保留两种材料形态：

```text
plain_roi_overview
    普通全图预览或普通 ROI 整体图，用于直接 Qwen 路径

numbered_roi_overview
    ROI 整体图叠加编号检测框，用于物体证据辅助路径
```

当前阶段不生成逐目标 crop、不生成 contact sheet，也不通过工作流按
`largest`、`closest`、颜色、状态等语义筛选候选。最终 Qwen 接收：

```text
原始问题
+ ROI 整体图或编号框 ROI 整体图
+ 编号对应的具体类别、置信度和原图全局坐标
+ 原任务的答案格式约束
```

全部语义筛选和最终判断都由最终 Qwen 完成。

## 6. VQA 的当前顶层划分

原先的三类 VQA 收缩为两类：

```text
A. 需要物体位置证据
   -> Qwen 输出组合类别和 ROI
   -> 确定性小模型工作流
   -> 编号框 ROI 整体图 + 检测记录
   -> 最终 Qwen

B. 不需要或无法由当前小模型可靠提供证据
   -> 普通全图预览或 ROI
   -> 最终 Qwen
```

SegFormer 不再承担独立场景分类工作流。在物体证据路径中，SegFormer只可提供
候选区域或存在性证据，不能直接承担：

- 精确实例计数；
- 相接实例拆分；
- 细粒度属性判断；
- 颜色、状态或复杂空间语义判断。

## 7. Grounding 的当前原则

Grounding 可以使用组合类别和 ROI 获取候选位置，但 YOLO 只负责检测位置。
属性合取、极值、相对关系、状态、颜色、排除等最终目标选择仍交给最终 Qwen，
不能仅因为类别命中 YOLO 就把所有检测框直接当成最终 grounding 答案。

## 8. 确定性工作流消费规则

工作流必须机械执行第一次 Qwen 的规划，而不能重新解释问题语义：

```text
validated first-Qwen plan
  -> expand core ROI with deterministic halo
  -> map composite categories to model capabilities
  -> run supported small-model paths
  -> restore every local detection to original-image coordinates
  -> draw stable numbered boxes on the ROI overview
  -> provide records and image to final Qwen
```

具体模型接口应只暴露模型无关协议。模型加载、processor、权重、device 和缓存
留在模型实现内部；工作流不得直接依赖具体 checkpoint 类。

## 9. 已确认但属于下层工作流的规则

以下结论已经确认，但不属于第一次 Qwen 输出 Schema：

- YOLO 与 SegFormer 的支持情况由确定性 capability catalog 判断；
- 多个组合类别分别维护执行状态；
- 只有缺失或未成功取得证据的类别进入下一层 fallback；
- 已成功取得证据的类别不重复进入 fallback；
- YOLO 的 ROI 局部框必须转换成原图全局框，不能丢失相对位置；
- 具体细分类别标签必须随检测结果保留给最终 Qwen。

## 10. 尚未冻结的下层问题

以下问题留给后续工作流讨论，不阻塞第一次 Qwen 输出标签的确定：

1. `unsupported`、`unavailable`、`error`、`valid_empty` 的精确状态与回退条件；
2. 某类别回退 Qwen 时，其他类别的部分成功证据如何合并；
3. 一个组合类别同时映射到多个小模型时的优先级和去重；
4. 多 ROI 的数量上限、重叠处理、halo 比例和框编号顺序；
5. 全图 ROI 在超高分辨率原图上的确定性内部切片方式；
6. 检测结果过多时的上下文容量限制；
7. 第一次 Qwen 规划失败、低置信度或 Schema 校验失败时的稳定 fallback；
8. 每样本 Qwen 调用预算以及 planning/final-answer 缓存身份；
9. `change` 工作流的 SegFormer 细节；
10. 封闭组合类别表的最终内容和 YOLO/SegFormer映射。

## 11. 非目标与安全边界

本方案不得：

- 让第一次 Qwen 输出模型路径、backend 名或 checkpoint；
- 让 dataset adapter 调用 Qwen 决定业务工作流；
- 把第一次 Qwen 的内部规划结果当成新的 Ground Truth；
- 改变 deterministic metrics 或让 Judge 覆盖确定性结果；
- 把本机绝对路径、Base64、模型权重或 secret 写入规划、trace 或 artifact；
- 用不安全 ROI 或局部框创建路径逃逸；
- 因实现该方案而重新引入 `spacers_agent/` 或旧 `eval/`。

## 12. 进入实现前的门禁

实现阶段至少需要另行完成：

1. 选定符合 Python allowlist 的 Schema 与 workflow 落点；
2. 审计 `TaskResolver` 与新视觉规划阶段的职责关系，避免两次冲突的任务判断；
3. 定义组合类别 capability catalog；
4. 为 Qwen 输出建立 `extra="forbid"` 的严格 Schema；
5. 将 prompt、response schema、图片摘要、逻辑模型身份和 generation settings
   纳入 request hash；
6. 为 ROI 安全、坐标恢复、类别上限、全图 fallback 和模型选择边界建立测试；
7. 明确本方案对 Qwen 调用预算、run artifacts、resume 与报告的持久化影响；
8. 实现完成后再同步更新 `DETAILS.md`，不得在实现前把提案写成现行事实。
