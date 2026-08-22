# Visual Planner 补充数据审计摘要

## 1. 审计对象

本次重点检查以下派生数据集：

```text
data/phase2-train-visualplanning-refined-v4/
```

其中新增补充数据来自：

- `VRSBenchSupplement`：3,600 条；
- `LEVIR_CC`：1,800 条；
- 合计：5,400 条。

重点字段为：

```text
task
needs_visual_assistance
object_categories
count_target
region_request
```

本次仅执行只读审计，没有修改数据集。

## 2. 当前 task 与 Agent 对应关系

当前确定性路由表如下：

| Task | 主 Agent | Fallback | Requires tiling |
|---|---|---|---|
| `counting` | `counting_agent` | 无 | 是 |
| `fine_grained_counting` | `counting_agent` | 无 | 是 |
| `change_caption` | `change_agent` | 无 | 是 |
| `change_qa` | `change_agent` | `general_vqa_agent` | 是 |
| `grounding` | `grounding_agent` | 无 | 是 |
| `spatial_relation` | `general_vqa_agent` | 无 | 否 |
| `scene_classification` | `general_vqa_agent` | 无 | 否 |
| `general_vqa` | `general_vqa_agent` | 无 | 否 |
| `caption` | `caption_agent` | 无 | 否 |
| `multiple_choice_vqa` | `general_vqa_agent` | 无 | 否 |

`change_qa` 使用 `general_vqa_agent` fallback 时，`SampleRunner` 会将本次执行的任务契约显式映射为 `general_vqa`。原样本的路由任务仍为 `change_qa`。

TaskRouter 只根据已经确定的 task 查固定策略，不读取问题、不调用模型，也不会把未知任务猜测为 `general_vqa`。Visual Planner 是否启用以及是否提供物体证据，不参与 task 到 Agent 的确定性路由。

## 3. 新增数据字段分布

| 新增任务 | 数量 | `needs_visual_assistance=true` | 类别非空 | `count_target` 非空 | Explicit ROI |
|---|---:|---:|---:|---:|---:|
| `caption` | 900 | 900 | 900 | 0 | 0 |
| `fine_grained_counting` | 900 | 900 | 900 | 900 | 0 |
| `grounding` | 900 | 900 | 900 | 0 | 0 |
| `multiple_choice_vqa` | 900 | 810 | 810 | 0 | 0 |
| `change_caption` | 900 | 449 | 449 | 0 | 0 |
| `change_qa` | 900 | 449 | 449 | 0 | 0 |

按来源汇总：

| 来源 | 数量 | Assistance true | 类别非空 | Explicit ROI |
|---|---:|---:|---:|---:|
| `VRSBenchSupplement` | 3,600 | 3,510 | 3,510 | 0 |
| `LEVIR_CC` | 1,800 | 898 | 898 | 0 |

基础字段联动检查没有发现结构错误：

- 不存在 assistance 为 true 但类别为空的新增样本；
- 不存在 assistance 为 false 但类别非空的新增样本；
- 不存在非计数任务错误填写 `count_target` 的新增样本；
- 900 条 `fine_grained_counting` 均具有 `count_target`；
- 不存在 explicit ROI 缺少 `image_index` 或 `roi_xyxy` 的情况。

## 4. 主要物体类别

新增数据中出现频率最高的类别如下：

| 类别 | 出现次数 |
|---|---:|
| `small-vehicle` | 1,421 |
| `large-vehicle` | 1,356 |
| `ship` | 1,046 |
| `building` | 800 |
| `harbor` | 771 |
| `road` | 735 |
| `bareland` | 512 |
| `plane` | 434 |
| `tree` | 423 |
| `tennis-court` | 371 |
| `swimming-pool` | 293 |
| `bridge` | 209 |

最常见的非空类别组合包括：

```text
["ship", "harbor"]
["ship"]
["small-vehicle", "large-vehicle"]
["bareland", "road", "building"]
["plane", "small-vehicle", "large-vehicle"]
```

## 5. ROI 审计结论

新增 5,400 条数据中没有任何一条满足：

```json
"region_request": {
  "explicit": true,
  "image_index": 0,
  "roi_xyxy": [620, 610, 980, 980]
}
```

事实上，全部新增样本均使用：

```json
"region_request": {
  "explicit": false,
  "image_index": null,
  "roi_xyxy": null
}
```

这对新增 `grounding` 样本是合理的。Grounding 中的源 bounding box 是任务要求模型预测的答案，不能把该框作为输入 ROI，否则会直接泄漏 Ground Truth。

但是，当前补充数据没有增加任何“问题已经给出框或区域，要求在框内检测、计数、描述或判断”的训练样本。如果需要训练这种规划能力，应从源数据中单独选择原始问题文本明确提供 bounding box 或区域约束的样本，而不能使用 Grounding 的答案框代替。

全量 v4 中满足以下条件的样本只有 2 条：

```text
needs_visual_assistance = true
object_categories = ["small-vehicle", "large-vehicle"]
count_target = "vehicle"
region_request.explicit = true
```

两条均来自原有 `VRSBench/counting`，不是本次新增数据：

1. `How many vehicles are visible in the center of the service area?`
2. `How many vehicles are parked in the top line?`

两者的 ROI 都是 `[0, 0, 999, 999]`，实质上是整图范围，不是真正的局部 ROI。`[620, 610, 980, 980]` 在 refined-v3 和 refined-v4 中均未出现。

## 6. 需要重点 review 的语义问题

### 6.1 Generic caption 被注入源标注类别

新增的 900 条 `caption` 全部启用了 visual assistance，并从 VRSBench 源物体标注生成 `object_categories`。例如，输入问题仅为：

```text
Write a detailed caption for this image.
```

但 target 可能是：

```json
{
  "task": "caption",
  "needs_visual_assistance": true,
  "object_categories": [
    "baseball-diamond",
    "small-vehicle",
    "large-vehicle"
  ],
  "count_target": null
}
```

这些具体类别无法从 generic caption 问题文本中确定。当前生成脚本直接读取 VRSBench 的源对象类别来构造 Planner target，与最新训练计划中的以下设计意图冲突：

- `object_categories` 不是图中所有可见物体的枚举；
- generic caption 应关闭类别 assistance；
- 不应使用 source annotation 向 Planner target 注入 Planner 输入无法安全确定的信息。

建议将这 900 条 generic caption 修订为：

```json
{
  "needs_visual_assistance": false,
  "object_categories": []
}
```

除非后续明确修改 Planner 契约，允许 Planner 根据图像内容主动选择开放式 caption 所需的子模型类别。

### 6.2 LEVIR change 类任务被注入参考 caption 信息

`change_caption` 和 `change_qa` 的类别来自 LEVIR 参考 caption，而输入问题通常只是：

```text
Are the two temporally ordered scenes visually different?
```

模型不能仅根据这个通用问题预先确定需要 `building`、`road`、`tree` 或其他类别。当前共有 898 条 change 样本因此启用了 assistance。

这些类别来自 Ground Truth/reference caption，而不是问题中明确给出的已知对象，存在隐藏标注注入风险。建议将这 898 条 generic change 样本修订为：

```json
{
  "needs_visual_assistance": false,
  "object_categories": []
}
```

如果未来构造的问题明确写出需要检查的对象，例如“建筑是否发生变化”，则可以从问题文本安全地得到 `building` 并启用辅助证据。

### 6.3 Fine-grained counting 基本符合当前设计

Fine-grained counting 的对象类别和计数范围直接出现在问题文本中。例如：

```text
Count the baseball diamonds and vehicles separately by fine-grained category.
Report one count for each category.
```

对应：

```json
{
  "task": "fine_grained_counting",
  "needs_visual_assistance": true,
  "object_categories": [
    "baseball-diamond",
    "small-vehicle",
    "large-vehicle"
  ],
  "count_target": "separate counts for baseball diamonds and vehicles"
}
```

这类标注能够从问题中确定类别和完整计数语义，原则上可以保留。

### 6.4 Grounding 类别可以保留，但答案框不能作为 ROI

Grounding 问题中的 referring expression 会明确给出待定位对象，例如 baseball diamond。将对应 canonical leaf 写入 `object_categories` 不会泄漏目标坐标，因此可以启用相关检测证据。

但是 source annotation 中的目标框是最终答案，必须继续保持：

```json
"region_request": {
  "explicit": false,
  "image_index": null,
  "roi_xyxy": null
}
```

### 6.5 Multiple-choice VQA 需要继续检查答案泄漏

新增 900 条 `multiple_choice_vqa` 中有 810 条启用了 assistance，类别主要通过问题文本提取。当前脚本对类别识别型问题进行了防答案泄漏处理，但仍建议人工抽查：

- 类别是否是题干中的已知条件；
- 类别是否实际属于待预测答案；
- 选项是否间接暴露了待识别类别；
- parent 类别是否被完整展开；
- 颜色、数量等限定是否被错误扩大为基础物体类别问题。

## 7. 当前建议

按风险和可保留程度，建议如下：

| 新增任务 | 当前建议 |
|---|---|
| `fine_grained_counting` | 原则上保留，抽查 `count_target` 精确范围 |
| `grounding` | 保留类别；继续禁止将答案框写入 ROI |
| `multiple_choice_vqa` | 保留经过问题文本推导的样本，重点抽查答案泄漏 |
| `caption` | 重新修订 generic caption 的 assistance/categories |
| `change_caption` | 删除由参考 caption 注入的隐藏类别 |
| `change_qa` | 删除由参考 caption 注入的隐藏类别 |

如果下一轮需要补充 ROI 数据，应单独构造或筛选以下类型：

```text
给定 bounding box，描述框内内容
给定 bounding box，判断框内对象属性
给定局部区域，统计区域内已知类别
给定两个或多个 ROI，判断空间关系
给定 ROI，对其中内容生成局部 caption
```

ROI 必须来自用户问题中已给出的输入条件，而不能来自任务答案或隐藏源标注。

## 8. 相关实现位置

- 固定 task 路由：`routing/policies.py`
- `change_qa` fallback task remap：`workflows/sample_runner.py`
- 补充数据生成规则：`scripts/supplement_visual_planner_dataset.py`
- 最新训练设计计划：`docs/training/PHASE2_VISUAL_PLANNER_DATA_REFINEMENT_PLAN.md`
- 新增数据运行审计：`data/phase2-train-visualplanning-refined-v4/audit/supplement_run.json`
- 新增数据选择记录：`data/phase2-train-visualplanning-refined-v4/audit/supplement_selections.jsonl`

## 9. 问题 2/3 修复记录（2026 完成）

上述问题 2（generic caption 注入源类别）与问题 3（change 注入参考 caption 类别）
已在 `data/phase2-train-visualplanning-refined-v4` 上就地修复，并同步修正生成脚本：

- 生成脚本改为策略 `visual-planner-structured-supplement-v2`：caption/change 的
  `object_categories` 只允许由问题文本驱动（复用 `_question_evidence_categories`），
  删除从 VRSBench 源物体标注 / LEVIR 参考 caption 抽取类别的 `levir_categories` 路径；
- v4 数据共修复 1,798 条：900 条 caption + 449 条 change_caption + 449 条 change_qa
  全部改为 `needs_visual_assistance=false`、`object_categories=[]`；
- `target_text`、`training/` 展开消息、`manifest.json` 哈希、`audit/distribution.json`、
  `audit/supplement_run.json` 已同步更新；全部补充行 provenance 的
  `supplement_policy_version` 更新为 v2；
- 修复明细见 `data/phase2-train-visualplanning-refined-v4/audit/supplement_assistance_fix.json`；
- `audit/label_decisions.jsonl` 与 `audit/supplement_selections.jsonl` 保留为 v1 原始
  运行历史，不改写；
- 问题 1（无显式 ROI）未在本轮修复：需要从问题文本自带 bounding box 的源数据单独
  筛选构造，属于后续数据采集任务。

## 10. Grounding 类别过度展开修复记录（2026 完成）

进一步 review 发现 grounding 的 `object_categories` 仍从 VRSBench 源 `obj_cls`
展开：源类别 `vehicle` 固定展开成 `["small-vehicle", "large-vehicle"]`，而
referring sentence 已明确限定 small/large 时仍保留两个叶子，共 376 条无法由
问题文本类别提取器完整复现（含 209 条 vehicle 过度展开）。

修复内容（已就地应用到 refined-v4，并同步修正生成脚本）：

- grounding 类别改由 referring sentence 文本推导（`_question_evidence_categories`，
  task=grounding），隐藏源 `obj_cls` 不再注入；
- `_TEXT_CATEGORY_ALIASES` 新增显式文本 alias：`baseball field`、
  `track and field`、`truck`；
- 共更新 376 条（train 345 / val 31），其中 2 条无可推导类别关闭 assistance；
  修复后 grounding 900 条与文本提取结果 mismatch=0；
- 明细见 `data/phase2-train-visualplanning-refined-v4/audit/supplement_grounding_categories_fix.json`。

