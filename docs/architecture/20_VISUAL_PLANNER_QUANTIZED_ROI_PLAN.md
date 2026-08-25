# 20 — Visual Planner 量化 ROI 输出改造计划

> Status: **implemented in the current working tree**
>
> 状态：**已在当前工作树实施；真实模型 live gate 仍需在目标环境执行**
>
> Baseline inspected: `aa18b14869e5402a0374b07723044135b51b8255`。

## 1. 背景

本改造实施前，fresh 运行使用 `visual-task-plan-v4`。模型在问题明确描述区域时输出
`focus_xy_norm`，workflow 再以该焦点为中心物化一个固定的 `1024 x 1024` ROI：

```text
ordered previews + raw question
  -> VisualTaskPlanner
  -> RegionRequest(explicit, image_index, focus_xy_norm)
  -> fixed 1024 x 1024 MaterializedVisualView
  -> Agent / evidence executor
```

固定 `1024 x 1024` 对遥感大图可能过小，而且单个焦点不能表达模型希望重点查看的区域
范围。本任务把模型输出改为 `0..999` 整数 `xyxy` 注意力框，并由确定性后处理把它扩展为
以 `1024` 为量化单位的 ROI。

本计划只改变 planner ROI 契约及其物化、持久化和 resume 语义，不改变 task 判定、
`UnifiedSample`、模型客户端协议、Agent 路由、Ground Truth 或评测定义。

## 2. 已冻结的产品语义

### 2.1 ROI 仍依赖问题中的显式区域描述

模型只有在问题明确描述了需要关注的区域时，才应设置：

```text
region_request.explicit = true
```

并输出目标图片索引和注意力框。问题没有显式区域描述时，必须输出：

```json
{
  "explicit": false,
  "image_index": null,
  "roi_xyxy": null
}
```

runtime 不新增关键词匹配、规则分类、第二次模型调用或其他启发式方法来判断问题是否包含
显式区域描述。模型对显式区域描述的理解和区域选择质量属于训练应解决的能力；本次 runtime
实现只做严格 schema 校验和确定性几何物化。

### 2.2 不要求模型输出正方形

prompt 和 schema 都不得要求：

```text
x1 - x0 == y1 - y0
```

模型只需输出与问题中显式区域描述相符的合法注意力矩形。它可以是横向、纵向或近似方形。
正方形扩展是后处理职责，不由模型执行，也不通过 prompt 要求模型学习像素量化数学。

这也避免了一个错误假设：在非正方形原图中，`0..999` 归一化坐标下等宽等高的框，映射
到原图像素后通常并不是正方形。

### 2.3 正方形由后处理尝试生成，越界部分直接截断

后处理先按模型框映射后的最长边构造理想正方形，并把边长向上量化到 `1024` 的整数倍。
如果理想正方形超出图片边界：

- 不平移正方形来保住完整尺寸；
- 不缩小到另一个 `1024` 倍数；
- 不回退全图；
- 直接将理想正方形与原图边界求交，截断溢出部分。

因此必须区分：

```text
ideal square
    后处理得到的理想正方形，边长严格是 1024 的整数倍，允许越界

materialized crop
    ideal square 与原图求交后的真实裁片，始终在原图内，但允许不是正方形，
    实际宽高也允许不是 1024 的整数倍
```

狭长图像中，纵向注意力框扩展后的理想正方形可能在水平方向越界。此时只裁去水平越界
部分，保留仍在原图内的纵向范围；不能因为最终裁片变成长方形而改成全图。

### 2.4 继续保持单 ROI

每个样本最多选择一个 `image_index` 和一个 ROI。多图样本中，未被选中的图片继续使用
全图。此次不恢复旧 multi-ROI、halo 或跨 ROI 搜索语义。

## 3. 新 planner schema

这次变更必须升级为 `visual-task-plan-v5`，不能原地改变 v4 中 `focus_xy_norm` 的含义。

建议的新结构为：

```json
{
  "version": "visual-task-plan-v5",
  "task": "general_vqa",
  "needs_visual_assistance": false,
  "object_categories": [],
  "count_target": null,
  "region_request": {
    "explicit": true,
    "image_index": 0,
    "roi_xyxy": [120, 180, 760, 820]
  },
  "reason_codes": []
}
```

`roi_xyxy` 的冻结契约：

- 坐标顺序为 `[x0, y0, x1, y1]`；
- 四个值必须是严格整数，布尔值和浮点数不得被宽松转换；
- 每个值都必须在闭区间 `0..999`；
- 必须满足 `x0 < x1`、`y0 < y1`；
- 左上角为坐标原点；
- `999` 表示对应轴的远端图像边界；
- 不校验框的宽高比，不要求正方形；
- `explicit=false` 时，`image_index` 与 `roi_xyxy` 必须同时为 `null`；
- `explicit=true` 时，`image_index` 与 `roi_xyxy` 必须同时存在；
- `image_index` 必须指向当前样本的有效图片。

模型响应仍不得包含原图尺寸、像素坐标、答案、GT、路径、backend、checkpoint、device、
secret 或主观 confidence。

## 4. 确定性 ROI 物化算法

### 4.1 坐标映射

物化必须使用 EXIF transpose、RGB 规范化后的真实原图尺寸：

```text
source_size = (width, height)
requested = (x0, y0, x1, y1) in 0..999
```

先将模型框向外取整到原图的半开像素边界：

```text
left   = floor(x0 / 999 * width)
top    = floor(y0 / 999 * height)
right  = ceil(x1 / 999 * width)
bottom = ceil(y1 / 999 * height)
```

映射后必须满足：

```text
0 <= left < right <= width
0 <= top < bottom <= height
```

`right` 和 `bottom` 是 Pillow crop 使用的半开边界。预览图只用于模型观察；坐标始终直接
映射到规范化原图，不经过多次 resize，也不依赖本地绝对路径。

### 4.2 最长边和 1024 量化

```text
requested_width  = right - left
requested_height = bottom - top
longest_side     = max(requested_width, requested_height)
quantized_side   = ceil(longest_side / 1024) * 1024
```

`quantized_side` 必须至少为 `1024`，并严格是 `1024` 的正整数倍。采用向上量化，保证理想
正方形不会因为量化而比模型请求框更小。

### 4.3 生成理想正方形

理想正方形使用模型框的中心，不重新解释问题语义：

```text
center_x = (left + right) / 2
center_y = (top + bottom) / 2

ideal_left   = floor(center_x - quantized_side / 2)
ideal_top    = floor(center_y - quantized_side / 2)
ideal_right  = ideal_left + quantized_side
ideal_bottom = ideal_top + quantized_side
```

理想框允许出现负坐标，也允许 `ideal_right > width` 或 `ideal_bottom > height`。不得为了让
理想框保持完整正方形而沿边界平移，因为这会改变模型指定区域相对于裁片中心的位置。

### 4.4 截断越界部分

真实裁片是理想框和原图范围的交集：

```text
crop_left   = max(0, ideal_left)
crop_top    = max(0, ideal_top)
crop_right  = min(width, ideal_right)
crop_bottom = min(height, ideal_bottom)
```

最终：

```text
crop_xyxy = (crop_left, crop_top, crop_right, crop_bottom)
crop_size = (crop_right - crop_left, crop_bottom - crop_top)
```

必须保证交集非空。由于原始模型框合法且位于原图内，正确实现生成的理想正方形必然与原图
存在非空交集；如果出现空交集，应视为 `ROI_MATERIALIZATION_FAILED`，不能伪造裁片。

以下性质是预期行为，不是错误：

- `crop_size[0] != crop_size[1]`；
- 某一实际边长不是 `1024` 的整数倍；
- 实际裁片覆盖原图的完整短边；
- 实际裁片恰好等于整张原图。

只要存在显式合法 ROI 请求，即使截断后恰好覆盖整图，也应在 artifact 中保留这是由
quantized ROI 物化得到的事实，不得丢失模型请求框和截断审计信息。

## 5. MaterializedVisualView 契约调整

当前 `view_mode="fixed_roi"` 和“裁片严格为 `1024 x 1024`”不再准确。建议 v5 使用：

```text
view_mode = "quantized_roi"
```

`MaterializedVisualView` 对 v5 ROI 只强制：

- `crop_xyxy` 是原图内非退化的整数半开框；
- `crop_size` 与 `crop_xyxy` 完全一致；
- 请求 ROI 的实际裁片允许为长方形；
- actual crop 不需要满足 1024 倍数；
- full-image view 仍必须覆盖整个 source。

为保证审计和问题定位，`visual_task_plan.json` 应同时保存：

```text
requested_roi_xyxy_0_999
requested_pixel_xyxy
roi_quantum = 1024
quantized_side
ideal_square_xyxy
crop_xyxy
crop_size
was_clipped
```

这些字段只保存 JSON-safe 数值和稳定布尔值，不保存图像字节、绝对路径或原始模型响应。
下游 Agent 仍只消费最终 `crop_xyxy/crop_size`，不能各自重新计算 ROI。

## 6. 代码修改范围

### 6.1 Schema 与 prompt

- `agents/schema.py`
  - 新增 `visual-task-plan-v5`；
  - `RegionRequest` 从 `focus_xy_norm` 改为严格整数 `roi_xyxy`；
  - 保留 explicit/all-or-nothing/image index 校验；
  - 删除固定 `1024 x 1024` materialized view 限制。
- `prompts/visual_task_plan_v5.md`
  - 只有问题显式描述区域时才请求 ROI；
  - 要求输出相关注意力矩形；
  - 不要求模型输出正方形；
  - 不要求模型计算 1024 倍数；
  - 不把 runtime 后处理数学塞入模型决策。

### 6.2 几何与 workflow

- `models/images.py`
  - 增加或替换为共享的 `0..999 xyxy -> quantized/clipped ROI` 确定性原语；
  - 保持纯几何、无模型、无数据集依赖；
  - 不复用带 halo 的旧任意 ROI 路径。
- `agents/general_vqa/evidence/rendering.py`
  - 通过现有 agents seam 暴露共享原语，不复制算法；
  - 裁切器继续校验真实图片尺寸与 materialized geometry 一致。
- `workflows/visual_planner.py`
  - 消费 v5 `roi_xyxy`；
  - 每张图最多生成一个 `quantized_roi`；
  - 未选中的图保持 full image；
  - artifact 保存请求框、理想框和截断结果；
  - 使用稳定错误码处理 schema、index、decode 和 materialization 失败。

### 6.3 下游消费者

- `agents/visual_base.py`
- `agents/general_vqa/evidence/**`
- `agents/grounding/evidence.py`

这些路径必须消费同一个实际 `crop_xyxy`，并允许截断后的长方形裁片。local/global 坐标转换
继续使用实际裁片宽高，不能假设宽高相等或为 1024。

detector、segmenter 和最终 Qwen 看到的必须是同一裁片；不得由不同消费者分别扩方或截断。

### 6.4 Settings、run identity 与 resume

fresh identity 升级为：

```text
planning_mode = "visual-task-plan-v5"
task_prompt_version = "v5"
roi_coordinate_frame = "normalized_0_999_top_left"
roi_quantum = 1024
roi_materialization_policy = "longest-side-ceil-quantum-center-clip"
```

当前 `roi_size=1024` 表示固定裁片边长；v5 中 `1024` 表示量化单位。建议使用新的
`roi_quantum` 字段，避免同一持久化字段跨版本改变含义。

需要同步更新：

- `application/settings.py`；
- `application/bootstrap.py`；
- `application/runtime.py`；
- `workflows/schema.py`；
- 受影响的 CLI/runtime 测试与配置快照。

`run_request.json` 必须冻结以上实际参数。resume 规则建议为：

- v5 succeeded：不重复 planner/Agent 推理；
- v5 非终态：按持久化 v5 参数恢复；
- v4 及更早 succeeded：只允许现有无模型补评测/报告路径；
- v4 及更早非终态：稳定拒绝重新推理，不能把 `focus_xy_norm` 猜成 v5 `roi_xyxy`；
- 新 resume 请求与持久化 ROI policy 冲突时稳定拒绝。

## 7. 明确非目标

本任务不做：

- 不修改或实现训练流程；
- 不在 runtime 中判断问题是否“真的”显式描述了区域；
- 不要求模型输出正方形或 1024 倍数尺寸；
- 不增加多 ROI；
- 不增加 halo；
- 不把越界 ROI 回退成全图；
- 不通过移动正方形避免越界；
- 不修改 `UnifiedSample` / `SampleDraft`；
- 不修改 task 集合、Router 或 AgentRegistry；
- 不修改主模型选择、checkpoint 加载或模型客户端协议；
- 不修改 GT、deterministic metrics、Judge 或报告聚合；
- 不新增未批准的 Python 路径；
- 不修改 Golden fixture 来迁就新行为。

## 8. 测试计划

### 8.1 Schema

- explicit false 只接受 `image_index=null, roi_xyxy=null`；
- explicit true 要求 image index 与 ROI 同时存在；
- 接受横向、纵向和方形矩形；
- 明确验证非正方形 ROI 被接受；
- 拒绝浮点数、布尔值、缺失坐标、五元素框；
- 拒绝小于 0、大于 999、退化和反向框；
- 拒绝无效多图 index；
- v5 schema 不再包含 `focus_xy_norm`；
- v4 artifact 不被当作 v5 校验。

### 8.2 几何

- 小于 1024 的最长边向上量化为 1024；
- 刚超过 1024 的最长边向上量化为 2048；
- 精确 1024/2048/3072 不额外扩大；
- 横向长框和纵向长框都按最长边扩方；
- 中心区域未越界时得到严格 1024 倍数正方形；
- 左、上、右、下和四个角越界时分别截断；
- 狭长竖图 + 纵向长框得到被水平截断的长方形裁片；
- 狭长横图 + 横向长框得到被垂直截断的长方形裁片；
- 理想正方形同时超过两轴时按两轴求交，不回退全图；
- 截断后宽高不是 1024 倍数仍合法；
- 截断结果恰好覆盖整图时仍保留 ROI 审计信息；
- 0 和 999 正确映射到原图半开边界；
- EXIF transpose 后尺寸是唯一几何 authority；
- 同一输入重复物化得到完全一致结果。

### 8.3 Agent 与 evidence

- direct Agent 使用实际截断裁片；
- VQA YOLO/SegFormer 使用完全相同的实际裁片；
- Grounding 使用完全相同的实际裁片；
- rectangular crop 的 local/global 坐标往返正确；
- 最终 evidence 文本报告实际 `crop_size`，不声称它是正方形；
- 多图中只裁选中的图，其余图保持全图；
- 不发生第二次 planner 调用或 Agent 自行重新选择 ROI。

### 8.4 Artifact、resume 与 reporting

- `visual_task_plan.json` 同时记录请求框、理想正方形和截断框；
- artifact 严格 JSON-safe，不含路径、图像字节或 secret；
- request hash 覆盖 v5 prompt/schema/ROI policy；
- `run_request.json` 冻结 `roi_quantum` 和 materialization policy；
- v5 succeeded resume 零模型调用；
- v4 非终态不会按 v5 重跑；
- reporting 能只读展示 v4/v5，不重新计算或修改 ROI；
- predictions/result path 语义不变。

### 8.5 建议验证命令

实施时至少运行：

```text
pytest -q tests/workflows/test_visual_planner.py
pytest -q tests/agents/general_vqa/evidence
pytest -q tests/agents/general_vqa/test_agent.py
pytest -q tests/agents/grounding
pytest -q tests/workflows/test_sample_runner.py tests/workflows/test_artifact_writer.py
pytest -q tests/integration/test_sample_runner_vertical_slice.py
pytest -q tests/integration/test_auto_task_dataset_vertical_slice.py
pytest -q tests/integration/test_dataset_runner_resume.py
pytest -q tests/application/test_settings.py tests/application/test_runtime.py tests/application/test_prompts.py
```

涉及 schema、prompt、依赖或文件布局时还应运行：

```text
pytest -q \
  tests/architecture/test_allowed_python_files.py \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py
python3 -m compileall agents models workflows application
git diff --check
```

真实模型、真实遥感数据和大图内存/时延表现需要单独 live gate；离线 fake 测试通过不能替代
真实 Qwen ROI 选择质量验证。

## 9. 建议实施顺序

1. 冻结 v5 schema、prompt 和量化/截断公式；
2. 先实现共享纯几何原语及完整边界测试；
3. 调整 `MaterializedVisualView` 和统一裁切 consumer；
4. 接入 `VisualTaskPlanner` 与安全 artifact；
5. 升级 settings、composition、run identity 和 resume；
6. 更新 Agent/evidence/runtime/integration 测试；
7. 更新 `DETAILS.md`、相关架构文档和已知限制；
8. 运行目标回归、架构门和 diff 检查；
9. 最后执行真实遥感样本 live gate，记录模型请求框、理想正方形、截断结果、失败样本与
   资源占用，不过滤失败样本。

## 10. 完成判据

实施完成必须同时满足：

1. fresh planner 使用 `visual-task-plan-v5` 和严格整数 `0..999 xyxy`；
2. 只有问题显式区域描述才请求 ROI，runtime 不新增语义判断规则；
3. 模型可以输出任意合法矩形，schema/prompt 不要求正方形；
4. 后处理按最长边向上量化到 1024 整数倍并生成理想正方形；
5. 理想正方形越界时直接截断，不平移、不缩小、不回退全图；
6. 截断后的实际 ROI 可以是非正方形且边长可以不是 1024 倍数；
7. direct、VQA evidence、Grounding 消费完全相同的实际裁片；
8. artifact 可以审计模型框、理想框和实际截断框；
9. v4 历史产物和 resume 不被重新解释；
10. `UnifiedSample`、routing、模型协议、评测、报告聚合和路径安全契约不发生漂移。
