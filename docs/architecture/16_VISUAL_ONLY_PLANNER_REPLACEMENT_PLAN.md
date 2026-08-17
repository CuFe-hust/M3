# 16 — 唯一视觉规划器替换方案

> Status: implemented; the v2 planner, fresh-entry wiring, durable prompt identity, and
> offline verification are complete. The remaining live-Qwen gate is intentionally not run.
>
> 状态：已实施；v2 规划器、新鲜入口接线、持久化 Prompt 身份与离线验证均已完成。
> 真实 Qwen live gate 按默认离线规则未运行。
>
> Cleanup follow-up: doc 17 removed the remaining executable resolver/gate/joint
> implementations; this document keeps the migration baseline and v2 design only.
>
> 后续收口：doc 17 已删除剩余可执行 resolver/gate/joint 实现；本文仅保留迁移基线与
> v2 设计记录。

## 1. 背景与问题

实施前，`visual_planning.enabled` 关闭时系统使用独立文本 `TaskResolver`、显式
task 直通或 adapter task 直通，开启时才使用 doc 15 的 `JointVisualPlanner`；旧
`VisualPlanner` / `VisualPlanningGate`、旧 prompt、旧 artifact writer 和相应测试也
仍然存在。该段描述的是迁移基线，而不是当前 fresh execution 契约。

当前实现已将所有 fresh manual/dataset 入口统一到 doc 16 的 `VisualTaskPlanner`；旧
planner/resolver 只保留在 reporting 的历史只读 seam 与迁移文档中，不由 composition
root 组装。

这不符合本方案冻结的目标：第一次 Qwen 调用必须始终是唯一的视觉规划调用，而不是
可选 feature flag。该调用只接收经过确定性预处理的图像和原始问题文本，一次性输出
任务类型、是否需要视觉辅助、对象类别和显式区域请求等规划信息。系统不得在它之前
调用文本 TaskResolver，也不得在它之后再次调用另一个规划模型。

当前联合规划器还会把 `image_id`、role、catalog、allowed tasks、answer constraints 等
字段序列化为 JSON 后附加到 user message。这也不符合本方案要求。

## 2. 目标

fresh inference 的唯一主链路必须是：

```text
source record / manual request
  -> deterministic image preprocessing
  -> first Qwen: image(s) + raw question text -> one visual task plan
  -> materialize/rebuild UnifiedSample with the planned task
  -> deterministic TaskRouter
  -> optional visual-assistance execution
  -> selected Agent / final task inference
  -> deterministic evaluation
  -> optional Judge
  -> durable artifacts and trace
```

必须满足：

1. 每条 fresh 样本恰好进行一次视觉规划 Qwen 调用；
2. 规划调用同时决定 task 与视觉执行所需信息；
3. 所有入口统一，包括 manual ask 的 auto/显式 task，以及 dataset 的
   explicit/default/auto 三种模式；
4. 调用方或 dataset 提供的 source task 仅用于审计，不发送给规划模型，也不覆盖模型
   规划出的 task；
5. `TaskRouter` 继续同步、确定性、无模型调用；
6. `UnifiedSample.task` 继续必填，并在视觉规划之后物化；
7. Ground Truth 只读，不能参与规划输入或因 task 变化而改写；
8. 视觉规划不得选择 backend、checkpoint、processor、device 或最终答案。

## 3. 非目标

本方案不包含：

- 修改确定性 metric、GT 解释、split、样本纳入规则或官方 evaluator 映射；
- 修改主模型、processor/tokenizer 或 checkpoint 加载语义；
- 让 Router、Agent 或 dataset adapter 自己重新判断全局 task；
- 让规划模型选择 YOLO/SegFormer backend 或权重；
- 为提高成功率而在视觉辅助不可用时静默回退；
- 修改 CountingResult、AgentResult 或全局 Prediction schema；
- 下载模型、数据集或调用云端服务。

## 4. 第一次 Qwen 的输入契约

### 4.1 User message 只包含图像与原始问题

规划调用的 user content 必须严格为：

```text
ordered image block(s)
+ raw question text
```

概念结构：

```python
[
    {"type": "image_url", "image_url": {"url": "<in-memory image>"}},
    {"type": "text", "text": question},
]
```

多图时先按 canonical source order 放置全部 image block，最后放一条未经 JSON 包装的
原始问题文本。空问题保留为空文本，由 system prompt 中的冻结规则决定 caption 类 task。

user message 不得包含：

- `image_id`、role、原图 width/height 或 `roi_eligible`；
- source task、metadata hints、normalization 或 answer constraints；
- allowed task/category 的 JSON 列表；
- Ground Truth、reference answer、绝对路径或文件名；
- backend、checkpoint、processor、device；
- Base64 文本字段、secret 或任意运行时对象。

合法 task 集合、输出 schema、类别闭集、禁止事项和规划规则属于 versioned system prompt
与 response schema，不作为附加 user payload。若 system prompt 根据已组装能力生成，其最终
正文必须进入 prompt snapshot 和 request hash。

### 4.2 图像顺序

模型以 image block 顺序识别多图，规划输出使用从 `0` 开始的 `image_index`。工作流在
模型返回后，将 `image_index` 确定性映射回输入视图中的 `ImageRef.image_id`。模型不需要、
也不应接收机器内部 image id 或 role。

图像顺序不得因 task 规划结果而改变。变化任务在物化时才按模型 task 将输入顺序映射为
`t1`、`t2`、`context`。

## 5. 缩略图与整图规则

每张输入图像独立执行以下确定性规则：

```text
max(width, height) <= 1080
    -> 不进行几何缩放，向第一次 Qwen 发送整图

max(width, height) > 1080
    -> 保持宽高比缩小，使最长边恰好为 1080
    -> 另一边按相同比例计算
    -> 不允许放大
```

允许在不改变几何语义的前提下执行 EXIF transpose、RGB 规范化和安全内存编码。模型收到
的实际图像字节必须计算真实 digest；不得写临时缩略图文件。request hash 必须覆盖每张图
实际传输字节的 digest 及稳定顺序。

边界必须通过测试冻结：最长边 `1079`、`1080`、`1081`，以及横图、竖图、方图和奇数尺寸。

## 6. 规划输出契约

建议新增版本，不原地改变 `joint-qwen-plan-v1`：

```json
{
  "version": "visual-task-plan-v2",
  "task": "general_vqa",
  "needs_visual_assistance": true,
  "object_categories": ["vehicle"],
  "region_request": {
    "explicit": true,
    "image_index": 0,
    "focus_xy_norm": [0.25, 0.25]
  },
  "confidence": 0.93,
  "reason_codes": ["question_explicit_top_left"]
}
```

冻结约束：

- `version` 必须精确匹配；
- `task` 必须属于 `data.schema.TaskName` 的闭合集合；
- `needs_visual_assistance=False` 时 `object_categories` 必须为空；
- `needs_visual_assistance=True` 时类别必须非空、稳定去重，并属于同版本可执行的封闭目录；
- `region_request.explicit=False` 时不得携带 image index 或 focus point；
- `region_request.explicit=True` 时必须携带合法 `image_index` 和有限的归一化 focus point；
- 模型只表达问题是否明确指定区域及关注中心，不决定最终 crop 尺寸；
- `extra="forbid"`；不得包含 final answer、backend、checkpoint、device、path、GT 或 secret；
- 解析、schema、类别、图像索引或 confidence 失败使用稳定错误码，不持久化原始异常全文。

`needs_visual_assistance` 与 `region_request` 是两个独立决定：需要对象证据不等于需要
ROI；问题明确指定区域也不等于需要 detector/segmenter。

内部如需继续使用 `FirstQwenVisualPlan` 供现有 Agent 消费，只能通过纯确定性转换生成：

```text
needs_visual_assistance=False -> direct execution family
needs_visual_assistance=True  -> object-evidence execution family
```

该转换不是第二次规划，也不得重新判断 task、类别或区域。

模型响应中的 `region_request` 不是可直接裁切的像素框。规划器 post-validation 必须先按
输入图像顺序解析 `image_index`，再结合 EXIF transpose 后的真实原图尺寸，生成确定性的
materialized view：

```json
{
  "view_mode": "fixed_roi",
  "image_id": "image-0",
  "source_size": [4096, 3072],
  "crop_xyxy": [512, 256, 1536, 1280],
  "crop_size": [1024, 1024]
}
```

`view_mode="full_image"` 时不得伪造一个 1024 ROI。模型响应与确定性 materialized view
必须在 artifact 中保持可区分、可审计；裁切器只消费后者，不重新解释问题或模型语义。

## 7. ROI 规则

### 7.1 触发条件

ROI 只能同时满足以下两个条件时启用：

1. 当前图像被确定性代码判定为大图；
2. 问题文本明确给出了需要查看的区域。

模型可以把以下类型识别为明确区域：

- 左上、右上、左下、右下、顶部、底部、左侧、右侧、中央；
- 明确指向第几张图的某个区域；
- 问题直接给出的坐标、框选区域或标记区域。

以下不构成明确区域，模型不得仅凭缩略图内容主动聚焦：

- “有多少辆车？”；
- “机场在哪里？”；
- “图中是否有建筑？”；
- 仅出现对象名称但没有空间限定；
- 模型在图像中自行发现的目标聚集位置。

### 7.2 全图与固定区域

每张图独立应用：

```text
非大图
    -> 全图

大图 + 问题未明确指定区域
    -> 全图

大图 + 问题明确指定区域
    -> 一个且仅一个 1024 x 1024 ROI
```

小图即使带有明确区域描述，也不裁剪为 ROI。多图请求中，只有被问题明确指向且满足大图
条件的图可以生成 ROI；其他图保持全图。问题给出区域但无法确定对应图片时，所有图片
回退全图，不猜测。

### 7.3 固定 1024 x 1024 几何

Qwen 只输出归一化中心 `(cx, cy)`。工作流以原图尺寸确定性生成区域：

```text
center_x = round(cx * width)
center_y = round(cy * height)
x0 = clamp(center_x - 512, 0, width - 1024)
y0 = clamp(center_y - 512, 0, height - 1024)
x1 = x0 + 1024
y1 = y0 + 1024
```

边界区域只允许平移，不允许缩小、拉伸或改变宽高比。最终持久化 ROI 必须能够确定性映射
回原图坐标；不得使用缩略图像素坐标冒充原图坐标。

每张图最多一个 ROI。删除当前允许模型给出最多三个任意尺寸 ROI、根据视觉内容主动
选择 ROI、或截断多 ROI 的语义。

### 7.4 大图定义

大图定义冻结为：

```text
width > 1024 AND height > 1024
```

必须同时严格大于；任一边等于或小于 `1024` 都不属于大图，统一使用全图。总像素数不
参与大图判断，因此 `4096 x 512` 等窄长图即使总像素超过 `1024 x 1024`，也不得启用
ROI。这保证任何符合条件的图像都能从原图中裁出真实的 `1024 x 1024` 区域，无需补边、
缩小或拉伸。

### 7.5 裁切器改造

当前 `models.images.crop_image_region(...)` 消费任意 `xyxy`，支持 `[0,1]`、`[0,999]`
两种坐标制式，并可通过 `halo_ratio` 扩大区域。当前
`agents.general_vqa.evidence.geometry` 也会把 normalized ROI 映射为 core/expanded box。
这些行为不能直接用于新的固定 ROI：任意 box 与 halo 都可能让最终裁片不再是精确的
`1024 x 1024`。

新 active crop contract 必须按 v2 ROI 分成两个确定性步骤：

```text
region_request(image_index + focus_xy_norm)
  -> planner post-validation resolves image_id and source dimensions
  -> materialize exact integer crop_xyxy
  -> crop_image_region consumes the materialized fixed ROI
  -> return RGB crop + the exact pixel geometry used
```

裁切器及其调用方必须满足：

1. 新路径只接受已经验证的归一化 focus point 或其确定性生成的整数
   `crop_xyxy`，不得继续把模型给出的任意 normalized `xyxy` 当成 v2 ROI；
2. ROI eligibility 与坐标计算使用 EXIF transpose、RGB 规范化后的真实图像尺寸，不能依赖
   可能缺失或过期的 metadata width/height；
3. 输入图像必须满足 `width > 1024 AND height > 1024`；否则拒绝 ROI materialization，
   上游按冻结规则选择全图；
4. focus 必须是两个有限的 `[0,1]` 数值；非法值在裁切前稳定失败；
5. 使用 §7.3 的 round/clamp 算法生成 Pillow half-open 整数框；
6. `x1 - x0 == 1024` 且 `y1 - y0 == 1024` 必须在裁切前后都成立；
7. 新固定 ROI 路径禁止 halo，`core_xyxy == crop_xyxy`，不得读取旧
   `visual_planning.planner.halo_ratio`；
8. 返回的裁片必须是独立 RGB image，尺寸严格为 `(1024, 1024)`，且不修改输入对象；
9. 裁切结果必须同时返回或记录实际 `source_size` 与整数 `crop_xyxy`，作为 local -> global
   坐标转换的唯一偏移依据；
10. `local_to_global` / `global_to_local` 继续只做确定性平移，不得重新缩放或重新推导 ROI；
11. `view_mode="full_image"` 时不调用固定 ROI 裁切器，直接使用规范化整图；
12. 多图按已解析的 `image_id` 分别裁切，不能把一张图的 focus/尺寸应用到另一张图。

可以在现有 `models/images.py` 中调整 `crop_image_region` 的 active signature，或在同一已批准
文件内提供清晰的 fixed-focus 入口；不得新增未批准的通用 helper 文件。无论采用哪种局部
实现，生产调用链只能有一个 v2 几何事实源，不能让 `models.images` 与 evidence geometry
各自用不同舍入方式重复计算。

General VQA、Grounding 和 direct visual Agent 都必须消费同一 materialized view：明确区域
产生的裁片应成为后续最终 Qwen/视觉辅助实际看到的图像；不能只在 object-evidence 分支中
裁切，而让 direct 分支仍悄悄使用全图。对象检测产生的 crop-local 坐标必须通过记录的
`crop_xyxy` 原点映射回整图坐标。

## 8. 视觉辅助能力

模型只输出是否需要视觉辅助及逻辑对象类别。类别仍来自版本化 evidence catalog，但不得
携带具体 detector、segmenter、checkpoint 或模型路径。

规划器只能接受运行时确实可以执行的类别。由于 user message 不允许附加 capability JSON，
可执行类别集合必须在 runtime assembly 时绑定进 versioned system prompt/response contract，
并进入 prompt snapshot 与 request hash。不得把 catalog 中存在但当前没有可用执行器的类别
描述成可执行能力。

模型若返回不可执行类别或请求不可用的辅助能力，应稳定失败；Agent 不得静默改走 direct
路径。默认配置若没有任何可执行对象证据服务，则 system prompt 必须只允许：

```text
needs_visual_assistance = false
object_categories = []
```

Grounding、Counting 等领域边界保持不变。规划类别不选择 counting backend/checkpoint，
也不自动改写 `CountTargetSpec`；如需复用规划类别作为 count target，应另行批准计数契约
变化。

## 9. 旧执行逻辑删除范围

### 9.1 删除生产执行能力

应删除：

- `visual_planning.enabled` 双轨运行开关；
- 独立文本 `TaskResolver` 的模型调用及请求 schema；
- 旧 `VisualPlanner`、`VisualPlanningGate`、`VisualPlanError`；
- composition root 的 old/joint 条件分叉与 `_build_visual_planning()`；
- `SampleRunner.visual_planning` 注入、旧 gate 调用和旧异常处理；
- DatasetRunner 的 resolver/joint 双路径；
- manual ask 的 resolver/joint 双路径；
- `visual_plan.json` 的新写入方法与 filename constant；
- `"task_resolver"`、旧 `"visual_plan"` 的现役 prompt binding；
- 新路径中的任意 normalized xyxy、多 ROI 与 `halo_ratio` 扩张语义；
- 只验证旧执行路径继续可调用的测试。

### 9.2 保留新路径仍需的契约

应保留：

- `materialize_sample()` 与 `SampleMaterializationError`；
- `TaskResolution` 或等价的纯运行时审计结构；
- `joint_plan_to_resolution()` 的纯转换职责，必要时按 v2 schema 重命名；
- ROI/evidence category 的共享 schema 与执行器；
- `AgentContext.visual_plan` / `visual_bindings` 或其等价轻量字段；
- 确定性 `TaskRouter`；
- reporting 对历史 `visual_plan.json`、`joint_visual_plan.json` 和旧 trace 的只读识别。

保留历史 artifact 的读取不等于保留旧推理逻辑。reporting 不得因此获得重新执行旧 Planner
或任意读取磁盘路径的能力。

## 10. Artifact、缓存与预算

新规划建议使用独立 basename：

```text
visual_task_plan.json
```

不得复用旧 `visual_plan.json` 或 `joint_visual_plan.json` 冒充 v2。artifact 只保存经 schema
和 post-validation 后的规划结果与确定性 materialized ROI，不保存 raw response、图像字节、
绝对路径或原始异常全文。

单样本共享一个 `CallBudget`：

```text
first Qwen visual planning call
  + downstream Agent/final-Qwen calls
  + optional Judge budget（独立服务计数）
```

request hash 至少覆盖：

- logical model identity、revision、generation settings、client version；
- system prompt 版本与完整正文；
- raw question text；
- 有序实际输入图像 digest；
- response schema/version；
- evidence catalog/capability binding；
- 影响缩略图和 ROI 输出的冻结参数。

不得通过删除 JSON user payload 而减少上述真实语义输入的 hash 覆盖。

## 11. Fresh run 与 resume

新 fresh run 必须在权威 `run_request.json` 中持久化规划契约，例如：

```text
planning_mode = "visual-task-plan-v2"
preview_max_side = 1080
roi_size = 1024
large_image_policy = "both-dimensions-strictly-greater-than-1024"
```

resume 不得根据当前默认值猜测旧 run 的规划方式：

- v2 succeeded 样本：零模型调用，只允许补缺失/损坏的确定性评测；
- v2 非终态或按契约需重跑的样本：继续使用持久化的 v2 参数；
- 旧 run succeeded 样本：仍可零模型补评测；
- 旧 run 若需要重新推理：删除旧执行逻辑后必须稳定拒绝，例如
  `LEGACY_PLANNING_RESUME_UNSUPPORTED`，不得悄悄用 v2 重跑；
- 旧 artifact 继续只读展示，不转换成 v2 artifact。

如果产品要求旧 run 可以继续推理，就不能同时彻底删除旧执行逻辑；这应作为独立兼容性
决定，不能由实现者暗中保留 fallback。

## 12. 文件级实施计划

预计只修改已批准路径，不新增 Python 文件，不修改 Python allowlist：

```text
agents/schema.py
agents/base.py（仅当 AgentContext 类型需要更新）
agents/visual_base.py
agents/general_vqa/agent.py
agents/general_vqa/evidence/geometry.py
agents/general_vqa/evidence/rendering.py
agents/general_vqa/evidence/executor.py
agents/grounding/agent.py
agents/grounding/evidence.py
models/images.py
routing/schema.py
routing/__init__.py
workflows/visual_planner.py
workflows/task_resolver.py
workflows/sample_runner.py
workflows/dataset_runner.py
workflows/artifact_writer.py
workflows/schema.py（仅持久化 planning mode 时）
workflows/__init__.py
application/settings.py
application/prompts.py
application/bootstrap.py
application/runtime.py
prompts/<new-versioned-visual-task-plan-prompt>.md
reporting/adapters.py（只增加 v2 allowlisted artifact）
DETAILS.md
README.md（若公开运行语义发生变化）
docs/migration/JOINT_TASK_VISUAL_PLANNER.md
相关既有 tests/**
```

`workflows/visual_planner.py` 继续作为批准的 workflow 职责路径；删除旧类不需要新增
`utils.py`、`compat.py` 或其他未批准路径。

## 13. 实施顺序

### 阶段 A：冻结契约

1. 将冻结的大图定义写入 schema、运行配置、文档和边界测试；
2. 冻结 `visual-task-plan-v2` schema；
3. 冻结新 system prompt；
4. 冻结 `visual_task_plan.json` 与 stable error codes；
5. 冻结 run request 中的 planning identity。

### 阶段 B：隔离规划器

1. user content 改为 image block(s) + raw question；
2. 实现 1080 边界图像预处理；
3. 实现 v2 response validation；
4. 实现 focus point -> 固定 1024 ROI 的确定性转换；
5. 修改裁切器和 evidence geometry，使其消费同一 materialized fixed ROI；
6. 将 direct/evidence Agent 图像输入统一接到 materialized view；
7. 在 fake Qwen 下验证每条样本恰好一次调用。

### 阶段 C：统一所有入口

1. manual ask 的 auto 与显式 task；
2. dataset explicit/default/auto；
3. source task 仅审计；
4. 规划 task 后物化 `UnifiedSample`；
5. 保持 Router 确定性与共享 budget。

### 阶段 D：删除旧执行逻辑

1. 删除旧 VisualPlanner/Gate；
2. 删除独立文本 TaskResolver 模型路径；
3. 删除 feature flag 分叉；
4. 删除旧 artifact 写入与 active prompt binding；
5. 清理 exports、测试 helper、注释和当前事实文档；
6. 保留历史 run 的只读 reporting 支持。

### 阶段 E：resume、报告与 rollout

1. 持久化并校验 planning identity；
2. 拒绝旧 run 的新推理；
3. 增加 v2 artifact 只读报告；
4. 运行完整离线门；
5. 使用目标 Qwen3-VL checkpoint 做真实小切片 live gate。

## 14. 测试矩阵

### 14.1 模型输入

- user content 只含有序 image block 和原始 question；
- 不出现 JSON wrapper、image id、role、尺寸、task、metadata、GT 或 path；
- 单图、多图和空问题；
- question 文本逐字保持，不被序列化成对象；
- system prompt/schema/catalog 变化进入 request hash。

### 14.2 缩略图

- 最长边 `1079`、`1080` 不缩放；
- 最长边 `1081` 等比例缩到 `1080`；
- 横图、竖图、方图、奇数尺寸；
- 不放大；
- 实际传输字节 digest 真实且顺序稳定；
- 不生成临时图片文件。

### 14.3 ROI

- `1024 x 1024`、`1025 x 1024`、`1024 x 1025` -> 非大图；
- `1025 x 1025` -> 大图；
- `4096 x 512` 等窄长图 -> 非大图；
- 小图 + 明确区域 -> 全图；
- 大图 + 无明确区域 -> 全图；
- 大图 + 仅对象名称 -> 全图；
- 大图 + 明确区域 -> 恰好一个 `1024 x 1024` ROI；
- 左上、右下、中央与边界 clamp；
- focus `(0,0)`、`(0.5,0.5)`、`(1,1)` 的确定性像素框；
- 裁切前后尺寸都严格为 `1024 x 1024`；
- 新路径不应用 halo，core/crop box 完全一致；
- Path/PIL 输入均不被修改，输出统一 RGB；
- ROI 始终使用原图坐标；
- 多图只裁问题明确指向的图；
- 多图不得交叉使用 image size、focus 或 crop box；
- 无法确定目标图片 -> 全部全图；
- 非有限 focus、非法 image index、extra field -> 稳定失败；
- `local_to_global` / `global_to_local` 在新 crop box 上零漂移互逆；
- direct 与 object-evidence 路径实际看到相同的规划裁片；
- 视觉内容本身不得触发 ROI。

### 14.4 视觉辅助

- `False` + 空类别合法；
- `False` + 非空类别非法；
- `True` + 空类别非法；
- `True` + 合法且可执行类别合法；
- 未知或当前不可执行类别稳定失败；
- 不允许输出 backend/checkpoint/device。

### 14.5 调用链

- 每条 fresh 样本恰好一次规划调用；
- 不再调用独立 TaskResolver；
- 不再调用第二个 VisualPlanner；
- 显式 task 也必须视觉规划；
- source task 与规划 task 冲突时使用规划 task，并保留安全审计；
- Router 零模型调用；
- 规划与 Agent 共用同一 Qwen budget。

### 14.6 Artifact 与 resume

- v2 artifact 原子、JSON-safe、纯 basename；
- 不保存 raw response、图像字节、绝对路径、secret；
- succeeded resume 零模型调用；
- v2 rerun 使用持久化参数；
- 旧 run 需要推理时稳定拒绝；
- 旧 artifact 只读展示；
- predictions append-only，summary 闭合。

## 15. 建议验证命令

实施后至少运行相关单元/集成测试：

```bash
pytest -q \
  tests/workflows/test_visual_planner.py \
  tests/workflows/test_task_resolver.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/workflows/test_artifact_writer.py \
  tests/integration/test_auto_task_dataset_vertical_slice.py \
  tests/integration/test_dataset_runner_resume.py \
  tests/application/test_bootstrap.py \
  tests/application/test_runtime.py \
  tests/application/test_settings.py \
  tests/application/test_prompts.py \
  tests/routing/test_router.py \
  tests/models/test_request_sanitization.py \
  tests/agents/general_vqa/evidence/test_geometry.py \
  tests/agents/general_vqa/evidence/test_rendering.py \
  tests/agents/general_vqa/evidence/test_executor.py \
  tests/agents/grounding/test_evidence.py
```

架构与安全门：

```bash
pytest -q \
  tests/architecture/test_allowed_python_files.py \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py \
  tests/models/test_response_cache.py \
  tests/models/test_request_sanitization.py
```

然后运行完整离线 pytest、`compileall`、`git diff --check`。真实 Qwen live gate 必须单独
记录 checkpoint、数据切片、调用数、task 结果、ROI 坐标与失败样本；离线 fake 测试通过
不得被表述为 live gate 已通过。

## 16. 验收标准

完成必须同时满足（当前实现状态见下方结果）：

1. fresh runtime 中不存在关闭视觉规划的模式；
2. 第一次 Qwen user message 只有图像和原始问题；
3. 每条 fresh 样本恰好一次视觉规划调用；
4. 所有入口使用同一规划器和同一 schema；
5. 不存在可执行的独立 TaskResolver 模型路径或旧 VisualPlanningGate；
6. 缩略图严格遵守最长边 `1080` 边界；
7. ROI 只在“大图 + 问题明确区域”时出现，且恰好 `1024 x 1024`；
8. 裁切器消费 v2 materialized ROI，禁止 halo，并返回与实际裁片一致的整数几何；
9. direct 与 evidence 路径实际消费相同的规划裁片，local/global 坐标零漂移；
10. 未明确区域时，无论图像内容如何都使用全图；
11. visual assistance 与 ROI 独立，类别只来自当前可执行闭集；
12. `UnifiedSample`、Router、GT、evaluation、reporting 与路径安全契约不被破坏；
13. 新 run 可恢复，旧 run 不被新语义静默重跑；
14. 文档、测试和实际生产接线一致。

## 17. 执行结果

本方案已按上述阶段实施。fresh manual ask、dataset explicit/default/auto 三种入口
共用 `VisualTaskPlanner`、`VisualTaskPlan` 与一次共享 `CallBudget`；规划后的 task 才
物化为 `UnifiedSample`，direct/evidence 消费同一组 materialized views。v2 产物为
`visual_task_plan.json`，生成的 capability-bound system prompt 也进入
`prompts.snapshot/` 和 manifest hash。

已验证的离线门包括 v2 planner、ROI/rendering、Agent/evidence、artifact/resume、runtime
与 import-boundary 相关测试，以及 `compileall` 和 `git diff --check`。完整离线 pytest
未宣称全绿：HTTP socket 测试在受限环境中有一项 oversized-body timeout，旧架构白名单
仍列出仓库既有未批准 Python 文件，模型/finetune 测试还受缺失的可选依赖影响。真实
Qwen checkpoint/live gate 未运行，也没有下载模型、数据集或调用云服务。
