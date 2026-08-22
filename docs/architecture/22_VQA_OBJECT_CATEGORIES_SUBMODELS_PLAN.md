# VQA Object Categories and Submodel Evidence Plan

> Status: Reviewed and approved for implementation
> Branch: `feature/vqa-object-categories-submodels`
> Baseline HEAD: `66578ec1b3d951c131b91f720d53bf50914f69f4`
> Scope: VisualTaskPlanner object leaf catalog, VQA YOLO/SegFormer evidence execution,
> final-Qwen image preprocessing, composition, tests, and current-fact documentation.

## 1. 目标

本计划在不改变 `UnifiedSample`、task routing、评测定义、报告聚合和 resume
语义的前提下，完成三项工作：

1. 扩展 VisualTaskPlanner 可判断的 VQA 对象叶子类别，使其覆盖当前 YOLO
   分类头、iSAID SegFormer 分类头和 OEM SegFormer 分类头；
2. 将 GeneralVQAAgent 的对象证据最终输入改成明确的三分支图像协议：
   YOLO only、SegFormer only、YOLO + SegFormer；
3. 在 YOLO/SegFormer 调用前，将任意尺寸 ROI 确定性切分/插值为严格
   `1024×1024` model tiles，并对 tiles 做有界并发标注。

所有模型继续由 `application` composition root 选择和注入。Agent 只依赖
`models.base` 中的模型协议，不读取 checkpoint 路径、不 import 具体
SegFormer/YOLO 实现，也不改变任何 deterministic metric。

## 2. 当前工作区基线

当前分支和提交：

```text
branch: feature/vqa-object-categories-submodels
HEAD: 66578ec1b3d951c131b91f720d53bf50914f69f4
```

实施时必须保留并避开当前已有用户改动：

```text
M  .gitignore
?? data/LRS-VQA_sample10/
?? data/superRS/
?? models/segformer_mitb2_oem/classes.json
```

当前实现事实：

- `agents/evidence_catalog.json` 已有 18 个 YOLO 叶子；
- 其中 15 个叶子同时绑定 iSAID SegFormer；
- OEM 的 8 个非背景语义类别尚未进入 evidence catalog；
- 工作区已存在用户新增、尚未跟踪的 OEM `classes.json`，内容按本任务确认的
  channel `0..8` 顺序提供；实施时只做契约校验、catalog 接线和 package metadata
  更新，不覆盖或猜测另一套映射；
- `ObjectEvidenceExecutor` 只接受一个 SegFormer client；
- `_build_vqa_evidence_service(...)` 当前明确拒绝启用 VQA segmenter；
- 当前 HEAD 已实现 VisualTaskPlan v5 的 quantized ROI：planner 请求先由
  `models.images.materialize_quantized_roi(...)` 物化为
  `MaterializedVisualView`，evidence executor 消费其确定的 `crop_xyxy`；
- 本任务的 1024 tile partition 发生在已物化的 crop 之内，不重新解析
  planner 归一化坐标，不替换 v5 quantization，也不改写其 audit geometry；
- 当前 VQA evidence 会把任意尺寸的 materialized ROI 直接交给 YOLO/SegFormer
  seam，尚无统一的严格 `1024×1024` partition、余块插值和并发调度；
- VQA final-Qwen 当前按三分支接收 annotated ROI、SegFormer 纯色 mask/clean ROI
  或 YOLO-on-mask/clean ROI，图像角色通过 `visual_inputs` 与 image block 顺序对应；
- 当前三份专家大权重均为 Git LFS pointer，不能据此声称已通过真实模型 gate。

计划初稿在旧 HEAD `b3eb7c2a82fe6b43c265fb6e461a34d2e7c599db`
运行过的聚焦离线基线：

```text
227 passed
```

覆盖 catalog、General VQA evidence、VisualTaskPlanner、settings、bootstrap、
SegFormer runtime 和 General VQA vertical slice。

当前 HEAD 已前进到 quantized ROI v5 实现，因此 `227 passed` 不是对
`66578ec...` 的代替验证。实施 Agent 必须在阶段 0 重跑基线并记录精确结果。

## 3. 类别池契约

### 3.1 YOLO 叶子

保持当前 YOLO 分类头的 18 个 canonical leaves：

```text
plane
baseball-diamond
bridge
ground-track-field
small-vehicle
large-vehicle
ship
tennis-court
basketball-court
storage-tank
soccer-ball-field
roundabout
harbor
swimming-pool
helicopter
container-crane
airport
helipad
```

它们与 YOLO 原始标签的映射继续由 evidence catalog 明确保存，不从类别索引、
模型类名或 dataset 名推断。

### 3.2 iSAID SegFormer 叶子

iSAID checkpoint 的权威 channel map 继续来自现有：

```text
models/segformer_mitb2_isaid/classes.json
```

`background` 不进入 planner 叶子池。其余 15 个类别继续映射到已有 canonical
leaves：

```text
storage-tank
large-vehicle
small-vehicle
plane
ship
swimming-pool
harbor
tennis-court
ground-track-field
soccer-ball-field
baseball-diamond
bridge
basketball-court
roundabout
helicopter
```

### 3.3 OEM SegFormer 叶子

本计划假设用户提供的顺序就是 OEM checkpoint 的 channel `0..8` 顺序：

```text
0 background
1 bareland
2 rangeland
3 developed_space
4 road
5 tree
6 water
7 agriculture_land
8 building
```

工作区已经有用户提供、尚未跟踪的：

```text
models/segformer_mitb2_oem/classes.json
```

实施时将校验并正式接入该资产，不重新生成或静默改写其 channel 顺序。

其中 `background` 只存在于 checkpoint class map，不进入 planner object leaf
pool。新增的 8 个 canonical leaves 为：

```text
bareland
rangeland
developed-space
road
tree
water
agriculture-land
building
```

### 3.4 Task capability 范围

本任务只扩展 `general_vqa` 的对象叶子池：

```text
general_vqa = 18 YOLO/iSAID leaves + 8 OEM leaves
```

以下能力保持不变：

```text
counting
fine_grained_counting
grounding
```

OEM expert 即使获得已确认 class map，也不因此自动成为 Counting backend。
在 counting expert catalog 中，它应保持 `enabled=false`、`status="active"`、
`verification.class_map="verified"`，但 `supports` 继续为空；VQA raw-label 映射
只属于 visual evidence catalog。这样不新增 counting target，也不改变 counting
selector、fallback chain 或指标。

## 4. Evidence catalog 变更

将 visual evidence catalog 升级到新版本，并让每个 SegFormer leaf 明确绑定：

```text
canonical leaf
raw SegFormer label
stable segmenter binding
capability enabled state
```

建议使用稳定的 segmenter binding：

```text
segmenter_mitb2_001    -> iSAID
segmenter_oem_001      -> OEM
```

binding 只表达逻辑能力归属，不包含 checkpoint 物理路径、device 或 secret。
VisualTaskPlanner 仍然只能输出 canonical leaf，不能输出 segmenter binding、
checkpoint 或 backend。

Catalog 校验应增加：

- SegFormer labels 与 segmenter binding 必须同时存在或同时缺失；
- enabled SegFormer capability 必须有已确认 class map；
- `background` 和 `LABEL_N` 不得成为 canonical leaf；
- raw labels 必须真实存在于对应 checkpoint class map；
- task capability 中只允许 canonical leaves；
- catalog version 继续进入 planner binding 和 request hash。

`VisualTaskPlan` schema 继续使用 `visual-task-plan-v5`，不增加模型选择字段。
静态 planner prompt 已要求输出 canonical executable leaves，因此预计不需要改变
planner schema；新的类别集合通过版本化 runtime binding 提供。

## 5. 模型协议与 client 复用

### 5.1 模型无关 semantic-mask seam

在 `models.base` 中增加模型无关的 semantic-mask output/protocol，至少表达：

```text
class-id mask
authoritative class names / id mapping
source or ROI size
logical model identity
weights digest
small JSON-safe diagnostics
```

具体 SegFormer client 在 `models/segformer_transformers.py` 中实现该协议。
Agent 不直接消费 torch tensor、Transformers model、checkpoint path 或具体
`SegFormerTransformersClient` 类型。

现有 `DenseSemanticClient` 继续供 Counting/Change 使用，不改变其既有行为。
新的 VQA mask seam 应复用 SegFormer 已有的预处理、argmax、logits 上采样和
class-map 校验能力，避免为了 VQA 再写第二套 SegFormer loader。

### 5.2 单次组装与惰性加载

Composition root 负责：

- 构造 iSAID 和 OEM 两个逻辑 client；
- 同一逻辑模型在一次 runtime assembly 中只创建一次；
- 与 Counting/Change 需要相同模型时共享实例；
- 仅构造 client 不读取权重；
- 首次实际推理时才校验并加载权重；
- 保持 `allow_download=False`。

## 6. VQA evidence 执行流程

### 6.1 Runtime capability 发布

Planner 的 `general_vqa` executable leaves 必须根据实际已组装能力计算，而不是
只要存在一个 evidence service 就发布整个 catalog：

```text
leaf executable = enabled YOLO capability
               OR enabled leaf-specific SegFormer capability
```

允许三种合法装配：

```text
YOLO only
SegFormer only
YOLO + SegFormer
```

如果两类能力都未启用，则 VQA evidence service 保持 `None`，planner 不发布
任何 `general_vqa` object leaf。

### 6.2 共用 1024×1024 ROI 预处理

YOLO 和两份 SegFormer 都只接收严格的 `1024×1024` 图像。Evidence executor
不得把任意尺寸的 materialized ROI 直接交给子模型，而应先通过一套共享、
确定性的预处理生成 model tiles。

预处理顺序：

```text
materialized ROI
  -> greedy non-overlapping 1024×1024 partition
  -> collect every remaining edge/corner region that can no longer form a full tile
  -> interpolate each remainder independently to 1024×1024
  -> bounded concurrent model annotation
```

#### 6.2.1 贪心切割规则

在 ROI-local 像素坐标中，从左上角开始，使用稳定 row-major 顺序切割：

```text
x: [0, 1024), [1024, 2048), ... , [last_full, width)
y: [0, 1024), [1024, 2048), ... , [last_full, height)
```

x/y 区间做 Cartesian product，形成互不重叠的矩形 partition。每个 ROI
像素必须恰好属于一个 partition：

```text
no gap
no overlap
no implicit halo
no shifted tail tile
```

例如 `2000×2000` ROI 产生：

```text
1024×1024
976×1024
1024×976
976×976
```

例如 `2048×1536` ROI 产生：

```text
1024×1024
1024×1024
1024×512
1024×512
```

若 ROI 本身小于 `1024×1024`，它整体就是唯一 remainder。宽或高恰好可被
1024 整除时，对应轴不得额外生成零尺寸 remainder。

每个 partition 生成稳定 `tile_id`，例如：

```text
<roi_id>-r<row>-c<column>
```

不得使用并发完成顺序、Python `hash()` 或临时文件名生成 tile identity。

#### 6.2.2 余块插值

原始尺寸已经是 `1024×1024` 的 full tile 不做 resize。其他 remainder 无论
只有一条边不足还是两条边都不足，都直接插值到严格的：

```text
model_input_size = (1024, 1024)
```

本任务明确采用插值拉伸，不使用 padding、letterbox、中心补边或重叠切片。
因此每个 tile 必须保存独立的 x/y scale：

```text
scale_x = 1024 / source_tile_width
scale_y = 1024 / source_tile_height
```

RGB tile 的插值算法必须冻结并进入运行身份；建议使用 Pillow LANCZOS。
所有 model input 必须在调用前断言尺寸严格等于 `(1024, 1024)`。

#### 6.2.3 YOLO 坐标逆映射与 SegFormer mask 空间恢复

YOLO 与 SegFormer 的输出类型不同，必须分别处理：

```text
YOLO      -> boxes/geometry -> coordinate inverse mapping
SegFormer -> dense class-id mask -> spatial resize and placement
```

SegFormer 不输出目标坐标，因此不存在“对 SegFormer 坐标做逆变换”。它需要恢复
的是被拉伸到 `1024×1024` 之前的离散像素网格。

YOLO 输出先从 model-tile `1024×1024` 坐标逆映射到原 partition，再加上
partition 的 ROI-local offset：

```text
source_x = model_x / scale_x + tile_x0
source_y = model_y / scale_y + tile_y0
```

逆映射结果必须裁剪到对应 partition/ROI 范围，然后再进入现有 ROI-local →
whole-image 转换和确定性去重。不得把拉伸后的 model 坐标直接当成 ROI 坐标。

SegFormer 对 model tile 输出稠密 class-id mask。该 mask 先使用 NEAREST
插值恢复为原 partition 的 `(source_tile_width, source_tile_height)`，再按
`source_tile_xyxy` 的 `(tile_x0, tile_y0)` 偏移写回 ROI-local mask canvas：

```text
model mask 1024×1024
  -> NEAREST inverse resize to source partition size
  -> exact partition placement in ROI canvas
```

mask 逆变换不得使用 bilinear/LANCZOS，以免产生不存在的 class id。因为
partition 无重叠，拼接时每个 ROI 像素只能写入一次；若检测到 hole、重复写入、
尺寸漂移或越界，必须稳定失败，不能猜测修复。

如果具体 SegFormer seam 暴露的是每类 logits/probabilities，而不是已经 argmax
后的 class-id mask，则顺序必须改为：先将每个 channel 的连续值恢复到原
partition 尺寸，再执行 argmax；不能先把连续值转成伪坐标。无论使用哪一种
seam，最终写回 ROI canvas 的都是离散 class-id/boolean mask，不是 box 坐标。

#### 6.2.4 有界并发与确定性聚合

预处理后的 `1024×1024` tiles 并发标注，但并发必须有显式上限：

- YOLO phase：同一 ROI 的 YOLO tiles 有界并发；
- SegFormer phase：按 segmenter binding 分组，组内 tiles 有界并发；
- YOLO → SegFormer 的 fallback 顺序不变，必须先聚合全部 YOLO 结果，再决定
  哪些 leaves/segmenters 需要进入下一阶段；
- 不为了并发而提前运行本来不需要的 SegFormer；
- 同一 runtime assembly 继续复用单一模型 client，不按 tile 重建模型；
- 调度使用标准库有界 executor/semaphore，不新增第三方并发依赖；
- 具体 client 必须声明并满足线程安全/最大并行度契约；若显式并发上限大于
  client 可安全支持的值，应在组装或执行前稳定拒绝，不能静默串行后仍声称
  已完成并发标注；
- 单 tile 失败按稳定 tile/model error 记录，不取消或丢弃其他 tile 的成功结果；
- 最终结果必须按 `(roi order, row, column, segmenter binding)` 稳定排序，绝不
  使用 future 完成顺序；
- 并发上限属于显式配置和 run identity，resume 不从新的默认值猜原参数。

### 6.3 YOLO 阶段

每个 materialized ROI：

1. 仅当请求叶子中至少一个具有已启用 YOLO capability 时进入 YOLO phase；
2. 对该 ROI 的每个 `1024×1024` model tile 调用一次 YOLO；
3. 各 tile 可有界并发，一次 tile 输出过滤全部请求的 YOLO labels；
4. 未请求类别全部丢弃；
5. confidence 只用于 threshold、NMS、top-k 和内部冲突裁决；
6. persisted bundle、Qwen content 和公共 trace 不出现 confidence；
7. tile-local geometry 必须先经过 scale/offset 逆映射，再形成 ROI-local 和
   whole-image geometry；
8. 全部 tile 结果按稳定 tile order 聚合后再执行现有确定性去重。

OEM-only 请求不得为了维持固定顺序而无意义调用 YOLO。

### 6.4 SegFormer 阶段

YOLO 后仍需要语义证据的 leaves 按 segmenter binding 分组：

```text
iSAID leaves -> iSAID client
OEM leaves   -> OEM client
```

每个 client 对每个 ROI 的每个 `1024×1024` model tile 最多调用一次。一个请求
同时包含 iSAID 与 OEM leaves 时，允许两个 SegFormer 子模型分别处理同一组
tiles，但不得按 leaf 重复推理。各 tile 可有界并发，聚合时必须恢复稳定
row-major 顺序。

SegFormer 使用权威 argmax class-id mask 生成逐 leaf presence mask，不把 mask
转成 box 或 count，也不比较两个 checkpoint 的 confidence。每个 tile mask 必须
先逆插值到原 partition 大小并无缝拼回 ROI canvas。YOLO 已命中的 leaf 不被
SegFormer 结果覆盖。

### 6.5 稳定失败与 final visual fallback

子模型 unavailable/error 时：

- 记录稳定 error code 或异常类型名；
- 不保存原始异常文本、物理路径或 tensor；
- 不伪造 box、mask、hit 或 count；
- 未命中的 leaf 保持 missing，交给唯一一次 final Qwen 处理；
- 其他 leaf 的成功证据不被丢弃。

## 7. Final-Qwen 图像预处理协议

分支选择依据是实际模型调用和有效 evidence，不依据 dataset 名或模型类名猜测。

### 7.1 YOLO only

当流程调用 YOLO，且没有调用 SegFormer 时：

```text
ROI original
  -> render retained YOLO xyxy boxes and leaf labels
  -> shrink longest side to at most 1080
  -> final Qwen
```

最终 Qwen 图像中直接包含框和类别标签，不再只通过文本 JSON 告知框位置。

### 7.2 SegFormer only

当流程没有调用 YOLO，但调用 SegFormer 时：

```text
per-leaf boolean masks
  -> compose pure-color semantic mask image
  -> shrink longest side to at most 1080
  -> pure mask
  -> matching clean ROI
  -> mask image + clean ROI + label/color legend + question
  -> final Qwen
```

纯色 mask 图不得混入 ROI 原图像素，也不得使用半透明 overlay。背景使用固定
颜色；leaf 颜色由稳定 deterministic palette 生成。

### 7.3 YOLO + SegFormer

当流程同时调用 YOLO 和 SegFormer 时：

```text
pure-color SegFormer mask
  -> render retained YOLO boxes and labels on the mask
  -> shrink longest side to at most 1080
  -> annotated mask image
  -> clean ROI original
  -> text explanation / legend / geometry
  -> question
  -> final Qwen
```

每个 ROI 的发送顺序固定为：

```text
annotated mask first
clean ROI second
```

YOLO 框与标签必须使用 SegFormer mask palette 中保留不用的高对比标注色。
冻结亮品红 `RGB(255, 0, 255)` 作为 YOLO 主描边色，并增加黑色外描边；
SegFormer leaf palette 在生成时必须排除与该标注色过近的颜色。这样即使框穿过
多个不同 mask 色块，框和标签仍能与语义色块形成明显区别，而不会被误认为某个
SegFormer 类别。

### 7.4 通用渲染规则

- 所有输入先 EXIF transpose 并转 RGB；
- `VisualTaskPlan.object_categories` 是最终 evidence 与渲染的严格 canonical-leaf
  白名单；YOLO/SegFormer 输出中不属于该集合的类别必须在渲染前丢弃；
- YOLO 只渲染请求 leaves 对应的 retained boxes；SegFormer 只为请求 leaves
  生成色块，所有未请求类别像素统一作为 mask background；
- parent、alias 和 raw model label 只用于 planner/catalog 的确定性解析与模型
  输出映射，不能作为额外渲染类别；
- 图例只列出实际渲染且属于请求集合的 leaves；请求但未命中的 leaves 不生成
  虚假色块或框，只进入 `missing_leaves` 文本说明；
- 最终视觉输入最长边超过 1080 时等比缩小，小图不放大；
- box 使用当前 VQA evidence 已保存的 ROI-local axis-aligned `xyxy`；
- 本任务不扩展新的 OBB/polygon 持久化契约；
- box 只画轮廓和 leaf label，不绘制 confidence；
- YOLO-on-mask 使用冻结的高对比标注色与黑色外描边，不复用任何 SegFormer
  leaf color；
- SegFormer palette 必须对 YOLO 标注色保留颜色安全距离，不能仅依赖“当前样本
  恰好没有相近颜色”；
- mask 背景色和 leaf palette 稳定、确定性且跨样本一致；
- 多 leaf mask 重叠只影响展示层，使用冻结 catalog 顺序做确定性覆盖；
- 原始逐 leaf evidence 仍完整保存在内存状态和 JSON-safe bundle 中；
- 渲染函数不修改输入 PIL image 或 mask；
- 不将最终图片、Base64 或 mask tensor写入 `vqa_evidence.json`。

### 7.5 Text payload 与 request identity

最终文本块继续包含：

```text
question
answer constraints
image/ROI identity and geometry
YOLO leaf labels
SegFormer leaf labels
mask color legend
missing leaves
coordinate/box explanation
```

Qwen request hash 必须覆盖：

- 逻辑模型身份与 revision；
- generation settings；
- prompt version/content；
- 最终 messages；
- 实际发送的所有渲染图片 digest；
- response schema；
- client version。

不得通过沿用 clean ROI digest 为渲染图制造错误 cache hit。

## 8. 配置与 composition root

扩展 `VisualSegmenterSettings`，使启用声明至少能够绑定：

```text
enabled
class_map_version
stable segmenter binding
```

增加 VQA evidence preprocessing 配置并写入 run identity：

```text
tile_size = 1024                    # frozen
partition = greedy-row-major-no-overlap
remainder_resize = stretch
rgb_interpolation = lanczos
mask_inverse_interpolation = nearest
max_tile_concurrency = 4
```

`tile_size` 必须冻结为 1024。并发上限不得使用未持久化的进程环境默认值；
fresh run、resume 冲突校验和 config snapshot 必须看到同一个实际值。默认并发
上限冻结为 4，并允许通过严格配置显式覆盖为 `1..32`；`1` 是调试/低显存模式，
不应被描述为并发 live gate。

具体模型路径、device 和 dtype 继续来自 `models.segformer_isaid`、
`models.segformer_oem` 或批准的 runtime profile。visual planning 配置不保存
权重对象或绝对路径形式的逻辑身份。

更新本地/示例配置，显式声明：

- VQA YOLO evidence 使用的已校准 detector policy；
- iSAID segmenter binding；
- OEM segmenter binding；
- 两份已确认 class-map version。

不改变全局离线默认值，不自动下载模型，不在 import 或 assembly 时加载权重。

## 9. Persisted artifact 与兼容性

继续使用：

```text
visual_task_plan.json
vqa_evidence.json
agent_result.json
agent_trace.json
```

`vqa_evidence.json` 继续保存：

- catalog version；
- ROI geometry；
- retained YOLO geometry；
- SegFormer leaf hit records；
- missing leaves；
- per-layer state；
- path-free model-call audit。

新增的 tile audit 只保存 JSON-safe 几何和变换参数：

```text
tile_id
roi_id
row / column
source_tile_xyxy
source_tile_size
model_input_size = [1024, 1024]
scale_x / scale_y
resize_applied
```

不得保存 tile image、mask array、future/thread 对象或临时物理路径。

不持久化：

```text
PIL image
mask/tensor
Base64
confidence
checkpoint physical path
raw exception
secret
```

Catalog 升级会改变新鲜运行的 planner binding 和 request hash。历史 succeeded
run 不重跑、不补写新渲染 evidence；resume 仍只按现有契约补缺失/损坏的
deterministic evaluation。

## 10. 预计修改文件

以下是实施 Agent 的预期变更面，不是“必须全部改动”的配额。如果某个现有
文件已满足契约，只增加测试证明即可，不为形式完整而制造无效 diff。

生产代码与资产：

| 文件 | 预期职责 |
|---|---|
| `agents/evidence_catalog.py` | catalog schema/version 校验、leaf 到 raw label/binding 映射 |
| `agents/evidence_catalog.json` | 26 个 General VQA leaf 及 YOLO/iSAID/OEM capability |
| `agents/counting/expert_catalog.json` | 记录 OEM 资产和已验证 class map，但保持 `enabled=false`/`supports={}` |
| `agents/general_vqa/agent.py` | 组装三种 final-Qwen image content 与文本解释 |
| `agents/general_vqa/evidence/schema.py` | tile/bundle/call-audit 的 JSON-safe 契约 |
| `agents/general_vqa/evidence/geometry.py` | 1024 贪心分割、box 逆变换与坐标裁剪 |
| `agents/general_vqa/evidence/executor.py` | YOLO → still-missing SegFormer 调度、有界并发与确定性聚合 |
| `agents/general_vqa/evidence/rendering.py` | tile 物化、mask 恢复/拼接、纯 mask 及高对比 box 渲染 |
| `models/base.py` | `SemanticMaskClient`/`SemanticMaskOutput` 模型无关协议 |
| `models/__init__.py` | 仅 re-export 新协议，不增加导入副作用 |
| `models/segformer_transformers.py` | 新增固定 1024、`do_resize=False` 的 semantic-mask adapter |
| `models/segformer_mitb2_oem/classes.json` | 用户已创建的 0..8 权威映射；实施时只校验，不覆盖 |
| `application/settings.py` | strict preprocessing/segmenter settings |
| `application/bootstrap.py` | 单次组装、client 复用、三种 VQA capability 组合 |
| `application/runtime.py` | 把冻结的 preprocessing identity 传入 fresh run request |
| `workflows/schema.py` | 定型 run-request preprocessing identity 及 legacy 读取 |
| `workflows/dataset_runner.py` | resume 冲突校验与 legacy fail-closed |
| `configs/local.yaml` | 本地 iSAID/OEM binding 与 preprocessing 配置 |
| `configs/models.example.yaml` | 不含机器绝对路径的配置示例 |
| `reporting/adapters.py` | 仅当新 artifact 字段破坏旧读取时做向后兼容，不重建 evidence |

本方案使用标准库并发与当前 Pillow/NumPy 能力，预计不需要修改
`pyproject.toml`。如果实施时发现必须新增依赖，先停止并向用户单独说明原因、
可选性和离线影响。

聚焦测试：

```text
tests/agents/test_evidence_catalog.py
tests/agents/counting/test_expert_catalog.py
tests/agents/general_vqa/test_agent.py
tests/agents/general_vqa/evidence/test_schema.py
tests/agents/general_vqa/evidence/test_geometry.py
tests/agents/general_vqa/evidence/test_rendering.py
tests/agents/general_vqa/evidence/test_executor.py
tests/workflows/test_visual_planner.py
tests/workflows/test_run_store.py
tests/workflows/test_dataset_runner.py
tests/models/test_segformer_runtime.py
tests/models/test_segformer_transformers.py
tests/models/test_request_sanitization.py
tests/application/test_settings.py
tests/application/test_bootstrap.py
tests/application/test_runtime.py
tests/integration/test_dataset_runner_resume.py
tests/integration/test_general_vqa_vertical_slice.py
```

文档：

```text
DETAILS.md
README.md
models/MODELS.md
```

预计不新增任何 Python 路径，因此不修改：

```text
architecture/implementation_status.json
```

若实施中发现需要新增 `.py`，必须保持清晰单一职责、遵守 import DAG，并补充
相应架构与行为测试。

## 11. 测试计划

### 11.1 Catalog 与 planner

- 精确验证 26 个 `general_vqa` leaves；
- 验证 18 个 YOLO raw labels；
- 验证 iSAID 15 个非背景 raw labels；
- 验证 OEM `0..8` channel 顺序；
- 验证两个 `background` 均不进入 leaf pool；
- 验证 `LABEL_N`、parent、alias、raw model label 不能作为 planner leaf；
- 验证 planner binding 只发布当前 runtime 可执行 leaves；
- 验证 catalog version 进入 planner request identity。

### 11.2 Executor

- ROI 小于、等于和大于 1024 时均生成正确 partitions；
- 非整除 ROI 形成 full tiles、right/bottom strips 和 corner remainder；
- partition 对 ROI 像素严格 no-gap/no-overlap；
- full tile 不 resize，remainder 严格插值为 `1024×1024`；
- 每次 YOLO/SegFormer 调用收到的图像尺寸都严格为 `1024×1024`；
- YOLO tile geometry 经独立 x/y scale 和 offset 正确逆映射到 ROI/whole image；
- SegFormer 离散 class-id mask 使用 NEAREST 恢复原 partition 尺寸并完整拼接，
  无 hole/重复写入；若 seam 返回连续 logits，则先恢复各 channel 再 argmax；
- tile 并发峰值不超过显式上限；
- 故意打乱并发完成顺序时，bundle/audit/渲染结果仍保持稳定；
- 单 tile 失败不丢弃其他 tile 的成功结果；
- YOLO-only leaf：每 tile YOLO 一次、SegFormer 零次；
- OEM-only leaf：YOLO 零次、每 tile OEM SegFormer 一次；
- YOLO miss + iSAID hit：两阶段均严格按 tile 调用；
- 混合 iSAID/OEM leaves：两个 segmenter 每 ROI、每 tile 各最多一次；
- YOLO hit leaf 不被 SegFormer 覆盖；
- 未请求模型 label 被丢弃；
- 未请求类别不进入 box、mask、legend、segments 或 detections artifact；
- 请求但 missing 的类别只出现在 missing 文本/状态中，不产生伪造标注；
- 模型异常不泄露路径、secret 或原始错误文本；
- bundle 严格 JSON-safe 且无 confidence。

### 11.3 Rendering 与 Agent content

- YOLO-only 生成带框 ROI；
- SegFormer-only 依次生成不含原图像素的纯色 mask 与同一 ROI 的 clean ROI；
- 双模型生成带框 mask，并按 mask → clean ROI 顺序发送；
- 双模型 mask 中 YOLO 框/标签颜色与每个 SegFormer leaf color 都满足冻结的
  最小颜色距离，并通过黑色外描边保持跨色块可见；
- mask legend 与 palette 一致；
- 最长边严格不超过 1080，且不放大小图；
- resize 后 box 几何与图像缩放一致；
- 输入 PIL image/mask 不被修改；
- final request hash 使用实际渲染图片 digest；
- direct VQA、multiple-choice 和禁止组合行为保持不变。

### 11.4 Composition 与模型边界

- YOLO-only、SegFormer-only、双模型三种配置均可组装；
- 无能力时 VQA evidence service 为 `None`；
- class map version/mapping 不一致时 fail closed；
- tile size 非 1024、非法 interpolation 或无效并发上限在 settings 边界拒绝；
- preprocessing/concurrency 参数进入 config snapshot 和 run identity；
- composition 不加载 YOLO/SegFormer 权重；
- 相同 logical model client 在一次 assembly 内复用；
- 缺可选视觉依赖不破坏基础 import。

### 11.5 回归与架构门禁

实施后至少运行：

```text
pytest -q tests/agents/test_evidence_catalog.py
pytest -q tests/agents/general_vqa
pytest -q tests/workflows/test_visual_planner.py
pytest -q tests/models/test_segformer_runtime.py tests/models/test_segformer_transformers.py
pytest -q tests/application/test_settings.py tests/application/test_bootstrap.py
pytest -q tests/integration/test_general_vqa_vertical_slice.py

pytest -q tests/architecture/test_implementation_status.py
pytest -q tests/architecture/test_import_boundaries.py
pytest -q tests/architecture/test_init_side_effects.py
pytest -q tests/architecture/test_package_discovery.py
pytest -q tests/architecture/test_no_new_to_legacy_imports.py

python -m compileall agents models application workflows
git diff --check
git status --short
```

真实权重 materialize 后再单独执行：

```text
YOLO-only live VQA
iSAID-only live VQA
OEM-only live VQA
YOLO + iSAID live VQA
YOLO + OEM live VQA
mixed iSAID + OEM leaf live VQA
```

离线 fake-client 测试通过不能替代这些 live gates。

## 12. 明确不改变的契约

```text
UnifiedSample / SampleDraft schema
TaskName 集合
TaskRouter 映射
VisualTaskPlan v5 task/ROI schema
Ground Truth 解释
deterministic evaluation metrics
Judge 与 deterministic 分离
report aggregation
CLI surface
run_request.json 权威性
succeeded resume 零推理语义
result path safety
default offline / no implicit download
```

## 13. 审阅确认项

实施前需要确认本计划中的以下冻结决策：

1. OEM 类别列表的书写顺序即 checkpoint channel `0..8` 的权威顺序；
2. 新增 OEM leaves 只扩展 `general_vqa`，不扩展 counting/grounding；
3. VQA 框渲染继续使用当前 ROI-local axis-aligned `xyxy`，本任务不新增 OBB
   polygon artifact；
4. SegFormer mask 使用纯色背景和稳定 leaf palette，不与原图做半透明叠加；
5. YOLO-on-mask 框使用 SegFormer palette 保留不用的高对比颜色，并带黑色外描边；
6. 双模型路径的图像顺序固定为“带框 mask → clean ROI”；
7. 多 mask 像素重叠仅按 catalog 顺序做确定性展示覆盖，不改变结构化 leaf hit；
8. `VisualTaskPlan.object_categories` 是严格渲染白名单；未请求类别不得进入
   box、mask、legend 或 persisted evidence；
9. 所有 final-Qwen 图像最长边缩至 1080，小图不放大；
10. 模型输入采用无重叠 row-major 贪心 partition；余块直接拉伸为
   `1024×1024`，不使用 padding/letterbox/halo；
11. YOLO 框使用独立 x/y scale 逆映射，SegFormer mask 使用 NEAREST 逆缩放
    后拼回 ROI；
12. 并发发生在同一模型 phase 的 tiles 之间，YOLO → SegFormer 阶段依赖顺序
    不变，且使用显式有界并发；
13. OEM counting expert 保持 disabled/unsupported，本任务不改变 counting 行为。

## 14. Agent 实施步骤

本节是交给编码 Agent 的执行清单。应严格按阶段推进；每个阶段先改契约和测试，
再进入下一阶段。不得把全部改动一次性堆入 executor 后再补边界。

### 14.1 实施总约束

编码 Agent 开始前必须遵守：

1. 不修改或删除用户已有的 `.gitignore`、数据目录和 OEM `classes.json`；
2. 不新增白名单外 Python 文件；本计划所需职责全部放入已有批准文件；
3. 不修改 `UnifiedSample`、TaskName、Router、deterministic evaluation、Judge、
   reporting aggregation 或 CLI surface；
4. 不启用联网、模型下载或云 API；
5. 不把 VQA 类别接入 Counting selector；
6. 不让 `agents/` import `application`、具体 YOLO/SegFormer 实现或模型路径；
7. 不把 PIL、mask、tensor、Base64、confidence、绝对路径或 raw exception 写入
   artifact；
8. 新增/修改的代码注释使用英文在前、中文在后；
9. 每个阶段运行列出的测试；失败时先修复本阶段，不通过跳过或放宽断言继续；
10. 未获得新授权前不修改 Golden migration fixtures。

### 14.2 阶段 0：只读基线与变更保护

执行：

```text
git status --short --branch
git rev-parse HEAD
git diff --check
```

预期分支与起点：

```text
branch: feature/vqa-object-categories-submodels
HEAD: 66578ec1b3d951c131b91f720d53bf50914f69f4
```

如 HEAD 再次变化，先审计新提交是否修改本计划覆盖的 planner ROI、evidence、
model protocol、settings 或 resume 契约；存在重叠时先更新计划和基线，不盲目应用旧步骤。

确认以下用户资产存在且不被覆盖：

```text
.gitignore
data/LRS-VQA_sample10/
data/superRS/
models/segformer_mitb2_oem/classes.json
```

读取：

```text
AGENTS.md
DETAILS.md
architecture/import_rules.json
本计划
相关生产代码与测试
```

按 14.16 的“最小局部集合”重跑当前 HEAD 基线。记录测试数、失败用例和
环境信息；不得将旧 HEAD 的 `227 passed` 写成当前 HEAD 结果，也不得因无关
失败修改本任务外代码。

停止条件：

- 发现必须新增白名单外 `.py`；
- OEM class map 不再是连续 `0..8`；
- 用户现有改动与目标文件发生无法安全合并的重叠。

### 14.3 阶段 1：固定 OEM class map 与 visual evidence catalog

#### 14.3.1 OEM class map

在不改写用户确认顺序的前提下，验证：

```text
num_classes == 9
id2name keys == "0".."8"
name2id == exact inverse of id2name
id 0 == background
id 1..8 == approved OEM labels
checkpoint_sha256 == declared OEM logical asset digest
```

在 `pyproject.toml` 的 `models` package data 中加入：

```text
segformer_mitb2_oem/classes.json
segformer_mitb2_oem/config.json
segformer_mitb2_oem/preprocessor_config.json
```

不得把 `.safetensors` 加入 wheel package data。

#### 14.3.2 Counting expert asset metadata

更新 `agents/counting/expert_catalog.json` 的 `segmenter_oem_001`：

```text
enabled = false
status = active
asset.class_map = models/segformer_mitb2_oem/classes.json
verification.class_map = verified
supports = {}
```

不要新增 OEM counting targets，不要填写 connected-component policy。

#### 14.3.3 Visual evidence catalog v4

更新 `agents/evidence_catalog.json`：

```text
catalog_version = visual-evidence-catalog-v4
```

在每个 SegFormer-enabled leaf 增加稳定 `segformer_binding`。iSAID leaves 绑定
`segmenter_mitb2_001`；OEM leaves 绑定 `segmenter_oem_001`。新增 OEM leaves 的：

```text
yolo_labels = []
yolo_enabled = false
segformer_labels = [exact raw label]
segformer_enabled = true
segformer_binding = segmenter_oem_001
```

只向 `task_capabilities.general_vqa` 加入 OEM 8 leaves。其他 task capability
列表保持原顺序和内容。

修改 `agents/evidence_catalog.py`：

- `LeafCapabilities` 增加可选 `segformer_binding`；
- labels/binding/enabled 做 all-or-none 校验；
- 增加 `leaf_segformer_binding(leaf)` accessor；
- accessor 对未知 leaf 继续稳定失败；
- catalog 不 import application、model implementation 或 counting backend。

#### 14.3.4 阶段测试

更新并运行：

```text
pytest -q tests/agents/test_evidence_catalog.py
pytest -q tests/agents/counting/test_expert_catalog.py
pytest -q tests/application/test_settings.py -k 'catalog or segformer'
```

验收：

- General VQA leaf pool 精确为 26；
- `background`/`LABEL_N` 不进入 leaf pool；
- raw labels 均存在于对应 class map；
- OEM 不是任何 counting candidate；
- catalog JSON 不含物理路径或 secret。

### 14.4 阶段 2：配置与可恢复运行身份

#### 14.4.1 Settings schema

在 `application/settings.py` 中新增严格模型，例如：

```text
VisualEvidencePreprocessSettings
```

冻结字段：

```text
version = greedy-1024-stretch-v1
tile_size = 1024
partition_policy = greedy-row-major-no-overlap
remainder_resize = stretch
rgb_interpolation = lanczos
mask_inverse_interpolation = nearest
max_tile_concurrency = 4
```

校验：

```text
tile_size: Literal[1024]
max_tile_concurrency: 1..32
all string policies: closed Literal values
extra = forbid
```

把它作为 `VisualPlanningSettings` 的一等字段。不要把这些值塞进自由 dict。

扩展 `VisualSegmenterSettings`，由 `visual_planning.segmenters` 的 key 作为稳定
binding，并验证：

- key 非空且安全；
- enabled 时必须有 class-map version；
- binding 必须能在 composition root 映射到一个已验证 logical client；
- 不保存物理 checkpoint path。

#### 14.4.2 本地配置

更新 `configs/local.yaml` 和 `configs/models.example.yaml`：

- 显式声明 iSAID/OEM segmenter binding；
- 使用两份 verified class map；
- 显式声明 VQA detector policy；
- 写入 preprocessing version 和 concurrency 4；
- 保持 `allow_download=false`。

不要改变内建 AppSettings 的“无 detector/segmenter 即能力关闭”原则。

#### 14.4.3 RunRequest / resume identity

在 `workflows/schema.py` 为 dataset/run identity 增加 typed evidence
preprocessing 字段。建议持久化成一个结构化子对象，而不是六个无关联字段。

兼容规则：

- fresh run 必须显式写 `greedy-1024-stretch-v1`；
- 历史缺字段 run 解析为 `None`/`legacy-unversioned`，不得伪装成新版本；
- succeeded resume 继续零推理，只允许既有补评测；
- 历史非终态若需要重跑 VQA evidence，应稳定拒绝
  `LEGACY_VQA_EVIDENCE_PREPROCESSING_UNSUPPORTED`；
- 新 run resume 时，tile size、policies、interpolation 和 concurrency 冲突均拒绝；
- config snapshot 与 run request 都保存无 secret 的实际值。

按当前调用链更新：

```text
application/runtime.py
workflows/schema.py
workflows/dataset_runner.py
application/bootstrap.py
```

不要从当前 config 或新默认值猜旧 run 的 preprocessing 参数。

#### 14.4.4 阶段测试

```text
pytest -q tests/application/test_settings.py
pytest -q tests/workflows/test_run_store.py
pytest -q tests/workflows/test_dataset_runner.py -k resume
pytest -q tests/integration -k 'resume and vqa'
```

验收：fresh round-trip 字段不丢失；old succeeded resume 零模型调用；old
nonterminal evidence rerun fail closed；新参数冲突稳定拒绝。

### 14.5 阶段 3：模型无关 semantic-mask protocol

在 `models/base.py` 增加：

```text
SemanticMaskOutput
SemanticMaskClient
```

建议 output 字段：

```text
class_id_map: Any                 # in-memory only
id_to_label: Mapping[int, str]
original_size: tuple[int, int]
weights_sha256: str | None
diagnostics: Mapping[str, JSON-safe scalar]
```

client protocol：

```python
@property
def cache_identity(self) -> ModelCacheIdentity: ...

def segment(self, image: Any) -> SemanticMaskOutput: ...
```

在 `models/segformer_transformers.py` 为现有 client 增加 `segment(...)` adapter：

1. 输入必须是 RGB `1024×1024`；
2. processor 调用必须 `do_resize=False`；
3. 检查 processor 输出空间尺寸仍是 `1024×1024`；
4. model logits 上采样到 model tile `1024×1024`；
5. argmax 得到离散 class-id map；
6. 返回权威 id-to-label、逻辑身份和权重 digest；
7. 不改变现有 `predict`、`infer`、`infer_pyramid` 的外部行为。

当前两份 `preprocessor_config.json` 声明 `size=512`，因此测试必须证明 VQA
`segment(...)` 路径使用 `do_resize=False`，不会把已准备好的 1024 tile 暗中缩回
512。不能为了通过测试直接改训练资产的 preprocessor metadata。

更新 `models/__init__.py` 仅做 re-export；不得加入业务逻辑或 import-time load。

阶段测试：

```text
pytest -q tests/models/test_segformer_runtime.py
pytest -q tests/models/test_segformer_transformers.py
pytest -q tests/models/test_request_sanitization.py
```

验收：module import 不加载 torch/transformers/权重；1024 输入不被 processor
二次 resize；输出 class map 严格对齐输入 tile。

### 14.6 阶段 4：composition root 与多 SegFormer client

重构 `application/bootstrap.py`，但保持它是唯一具体模型选择位置。

#### 14.6.1 Client inventory

将 client 构造分成两个概念：

```text
verified semantic clients    # 可供 VQA/Change 复用
enabled counting clients     # 仅注册到 Counting backend
```

`segmenter_oem_001` 可进入 verified client inventory，但因为 `enabled=false` 且
`supports={}`，绝不注册成 Counting backend。

同一个 logical model id：

- 一次 runtime assembly 只创建一个 client；
- assets/class map 不一致时稳定失败；
- client 构造不加载权重；
- VQA/Change/Counting 引用同一个已组装对象。

#### 14.6.2 VQA service 三种模式

改写 `_build_vqa_evidence_service(...)`：

```text
no enabled capability -> None
detector only          -> YOLO-only executor
segmenter only         -> SegFormer-only executor
both                   -> combined executor
```

删除“任何 enabled segmenter 都必然 composition failure”的临时门禁，替换为真正
的 class-map/binding/client 校验。

向 `ObjectEvidenceExecutor` 注入：

```text
optional YOLO client + detector policy
mapping[binding, SemanticMaskClient]
EvidenceCatalog
VisualEvidencePreprocessSettings 的定型值
```

#### 14.6.3 Planner executable leaves

新增纯确定性 helper 计算 General VQA leaves：

```text
leaf executable when:
  yolo enabled and leaf has yolo mapping
  OR
  leaf segmenter binding is enabled and verified client exists
```

保持 catalog 顺序。不要因为 service 非 None 就发布全部 26 类。

阶段测试：

```text
pytest -q tests/application/test_bootstrap.py -k visual
pytest -q tests/workflows/test_visual_planner.py
```

必须用 fake stores/clients 证明 composition 期零权重加载。

### 14.7 阶段 5：tile schema 与纯几何

#### 14.7.1 Persisted-safe tile record

在 `agents/general_vqa/evidence/schema.py` 增加严格模型：

```text
EvidenceTileRecord
```

字段：

```text
tile_id
roi_id
row
column
source_tile_xyxy
source_tile_size
model_input_size = (1024, 1024)
scale_x
scale_y
resize_applied
```

校验：

- tile id 为安全稳定标识；
- row/column 非负整数；
- source box 非退化；
- size 与 box 差值一致；
- model input 严格 1024 square；
- scale finite/positive，并与尺寸计算一致；
- full tile 的 scale 为 1 且 `resize_applied=false`；
- record JSON-safe。

`VqaEvidenceBundle` 增加：

```text
preprocessing_version
tiles: list[EvidenceTileRecord]
```

为了读取历史 artifact，旧 bundle 可以缺少这些字段；fresh v1 executor 必须填满。
`ModelCallAudit` 增加安全 `tile_id`，使同 ROI 多次模型调用可审计。

#### 14.7.2 Pure geometry helpers

在已有 `agents/general_vqa/evidence/geometry.py` 实现纯函数：

```text
partition_axis(length, tile_size=1024)
partition_roi(record, tile_size=1024)
model_xyxy_to_roi_xyxy(box, tile_record)
```

几何输入的权威边界是 `MaterializedVisualView.crop_xyxy/crop_size`：

1. v5 planner 的 `region_request.roi_xyxy` 只能由现有 shared materializer 解析；
2. executor 不得再做一次 normalized-to-pixel 转换或量化；
3. tile `source_tile_xyxy` 是相对已物化 crop 的局部坐标；
4. 转 whole-image 时只加 `MaterializedVisualView.crop_xyxy` 的 `(x0, y0)` 偏移；
5. `requested_pixel_xyxy`、`ideal_square_xyxy`、`was_clipped` 等 v5 audit 字段保持不变。

规则严格使用半开整数区间和 row-major Cartesian product。几何模块不 import
PIL、NumPy、torch 或模型实现。

逆映射：

```text
x_roi = clamp(x_model / scale_x, 0, source_width) + tile_x0
y_roi = clamp(y_model / scale_y, 0, source_height) + tile_y0
```

然后验证非退化，再进入 ROI → whole-image 转换。

阶段测试：

```text
pytest -q tests/agents/general_vqa/evidence/test_schema.py
pytest -q tests/agents/general_vqa/evidence/test_geometry.py
```

测试尺寸至少包含：

```text
1×1
600×400
1024×1024
1024×1536
1536×1024
2000×2000
2048×1536
2048×2048
```

对每组 partition 建 coverage bitmap，断言每个 ROI 像素覆盖次数严格为 1。

### 14.8 阶段 6：图像 tile 物化与 mask 恢复

在 `agents/general_vqa/evidence/rendering.py` 增加纯内存函数：

```text
prepare_model_tile(roi_image, tile_record)
restore_class_id_mask(model_mask, tile_record)
stitch_class_id_masks(restored_tiles, roi_size)
```

RGB：

- 精确裁切 `source_tile_xyxy`；
- full tile 保持原字节/尺寸，不 resize；
- remainder 用 Pillow LANCZOS 拉伸到 1024 square；
- 返回新的 RGB image，不修改 ROI source。

Mask：

- 输入必须是 1024 square、整数 class-id grid；
- remainder 用 NEAREST 恢复到 source tile size；
- full tile 不 resize；
- 按 source box 精确写入 ROI canvas；
- 使用 coverage canvas 检查 hole/overlap；
- 输出仍是离散整数/boolean mask。

若 semantic seam 返回 probabilities，则在 model layer 或 executor 的明确分支中
先逐 channel 恢复，再 argmax；不要把连续 probability image 交给 class-id
NEAREST helper。

阶段测试：

```text
pytest -q tests/agents/general_vqa/evidence/test_rendering.py
```

使用带坐标编码的 synthetic images/masks 验证拉伸、逆缩放、offset 和拼接，
不能只断言输出尺寸。

### 14.9 阶段 7：有界并发调度

在 `agents/general_vqa/evidence/executor.py` 内实现，不新建通用 manager/helper
模块。

建议使用一次 evidence execution 生命周期内的：

```text
concurrent.futures.ThreadPoolExecutor(max_workers=max_tile_concurrency)
```

执行顺序：

1. 同步、确定性生成全部 ROI tile records/images；
2. 生成 YOLO jobs，按稳定 job index submit；
3. 等待全部 YOLO jobs 终止；
4. 按 job index 聚合，不按完成顺序聚合；
5. 确定 still-missing leaves；
6. 生成 `(segmenter_binding, tile)` jobs；
7. 有界并发执行并按稳定 key 聚合；
8. context manager 关闭 worker pool；
9. 组装 bundle、masks 和 audits。

不要让 worker 直接 append 共享 list/dict。每个 future 返回不可变小结果，由主线程
按稳定 index 合并，避免 race 和非确定性顺序。

异常策略：

- 捕获单 job 异常并转稳定类型名；
- 其他 job 继续；
- KeyboardInterrupt/Cancel 应正确关闭 pool 后继续向上抛；
- 不把异常正文写入结果；
- failed job 仍有 tile/model audit；
- 不自动重试或扩大模型调用次数。

并发安全：

- `_LazyObjectDetectionClient` 初始化继续使用现有 lock；
- 初始化完成后的 detect/segment 是否支持并行必须由对应 client contract 和 live
  gate 验证；
- 若具体 provider 不支持配置并行度，稳定拒绝，而不是偷偷创建多份模型；
- 不按 worker 数复制 checkpoint/client。

阶段测试使用带 barrier 和随机延迟的 fake clients：

- 证明活动 job 峰值 `>1` 且 `<=4`；
- 证明乱序完成得到相同 bundle JSON；
- 证明单 tile 失败隔离；
- 证明一个 logical client 只构造一次。

### 14.10 阶段 8：YOLO tile 聚合

重构 `_yolo_phase(...)`：

1. 若请求 leaves 没有任何 YOLO capability，返回 zero calls；
2. 每个 prepared tile 调一次 detector，`image_size=1024`；
3. 断言 detector 实际输入/output reference size 为 1024 square；
4. 只保留 plan whitelist 的 raw labels；
5. 对每个 detection 做 model-tile → ROI-local → whole-image 映射；
6. 裁剪后退化框丢弃并记录稳定 outcome，不伪造最小框；
7. 全部 tile 结果按 tile order 合并；
8. 在统一 whole-image 坐标运行现有 deterministic dedup；
9. confidence 在 dedup 后即丢弃，不进入 schema/trace/rendering。

注意：无重叠 partition 可能切开边界目标。本任务按用户批准的 no-overlap 策略
实施，不增加 halo、shifted-tail 或 seam reviewer；应在文档 known limitations 中
明确记录，不用未授权策略偷偷补偿。

### 14.11 阶段 9：SegFormer tile 聚合

重构 `_segformer_phase(...)`：

1. 只处理 YOLO 后 still-missing 且 SegFormer-enabled 的 requested leaves；
2. 按 `segformer_binding` 分组；
3. 每 binding/tile 调一次 semantic mask client；
4. 严格验证 output class map 与 catalog raw labels；
5. 从 class-id map 提取请求 leaves 的 boolean masks；
6. 未请求类别统一为 background，不进入 segments/legend；
7. 每个 boolean/class-id tile 恢复 source partition 尺寸；
8. 每 ROI 拼成完整 leaf masks；
9. `any()` 为 true 才产生 `SegFormerEvidenceRecord(hit)`；
10. 不从 mask 生成 box/count；
11. 不跨 checkpoint 比较 probability/confidence；
12. YOLO-hit leaf 不被 SegFormer 覆盖或重跑。

同一请求含 iSAID/OEM 时，两组 jobs 可以共享有界 pool，但结果顺序固定为：

```text
roi order -> segmenter binding order -> row -> column
```

### 14.12 阶段 10：最终渲染协议

#### 14.12.1 Strict whitelist

最终可渲染 leaf 集合：

```text
rendered_leaves = actual_hits INTERSECT plan.object_categories
```

未请求输出在 executor 过滤，不允许到 rendering 层再“顺便展示”。missing leaf
只进入文本，不生成色块、框或 legend 项。

#### 14.12.2 Palette 与 YOLO 高对比色

冻结：

```text
mask background: RGB(0, 0, 0)
YOLO inner stroke: RGB(255, 0, 255)
YOLO outer stroke: RGB(0, 0, 0)
outer width at <=1080 output: 5 px
inner width at <=1080 output: 3 px
```

SegFormer palette 按 catalog leaf order 确定性生成并满足：

```text
distance(mask_color, YOLO magenta) >= 128
distance(mask_color, black background) >= 96
distance(mask_color_i, mask_color_j) >= 48
```

使用 RGB Euclidean distance；候选不合格时以 `sha256(leaf|attempt)` 稳定重采样，
设置有限但足够的 attempt 上限，耗尽时稳定失败。不要依赖随机数或进程状态。

YOLO label 使用黑色底板、亮品红边框和白色文字；不得把 confidence 写到标签。

#### 14.12.3 三分支 image content

按每个 ROI 稳定输出：

```text
YOLO only:
  annotated ROI

SegFormer only:
  pure mask
  clean ROI

YOLO + SegFormer:
  YOLO-on-pure-mask
  clean ROI
```

所有 final-Qwen images：

- 最长边超过 1080 才缩小；
- 小图不放大；
- 图像发送顺序与 digest 顺序一致；
- 使用实际 PNG transport bytes 计算 SHA；
- 不写临时图片文件。

### 14.13 阶段 11：GeneralVQAAgent content 与 request hash

改写 `GeneralVQAAgent._build_evidence_content(...)`，只负责消费 executor 已完成
的 bundle/masks/images，不重新运行模型或决定 capability。

文本 payload 明确区分：

```text
requested_leaves
rendered_yolo_leaves
rendered_segformer_leaves
missing_leaves
ROI/source geometry
mask legend
question
answer constraints
```

删除旧的 per-leaf semi-transparent overlay image 输出。可以保留安全、必要的
结构化 detection/segment 描述，但不得与图片中的 frame 说明冲突。

Request hash 使用最终实际 content 和 image digests。测试至少改变以下任意一项
都会改变 hash：

```text
tile partition geometry
remainder interpolation result
YOLO box
mask pixels
palette/catalog version
image order
question
```

仍只消费一次 final-Qwen budget；tile expert calls 不计为 Qwen calls。

### 14.14 阶段 12：Artifact、reporting 读取与 resume

`vqa_evidence.json` 新鲜产物保存 preprocessing version、tiles 和逐 tile call
audit。新增字段必须 JSON-safe、path-free。

检查 `reporting/adapters.py` 对结构化 artifact 的读取：

- 如果它只是透传 dict，不增加推理或派生；
- 如果严格解析旧 bundle，为新字段提供向后兼容读取；
- reporting 不读取 tile 图片，不重建 mask，不重新渲染执行 evidence。

Resume 测试必须覆盖：

```text
new succeeded -> zero model calls
new failed/partial -> exact persisted preprocessing identity
old succeeded -> zero model calls, no evidence repair
old nonterminal VQA evidence -> stable legacy rejection
tampered tile policy -> conflict rejection
```

### 14.15 阶段 13：文档同步

更新 `DETAILS.md`：

- GeneralVQAAgent 三分支图像协议；
- 26-leaf General VQA pool；
- OEM class map 已确认；
- 1024 greedy/stretch preprocessing；
- tile concurrency 与 run identity；
- model residency/lazy-loading 事实；
- boundary-object no-overlap limitation；
- live validation 仍 pending。

更新 `README.md`：

- 用户可见配置示例；
- OEM 不再是 unverified class map；
- 本地权重仍需 Git LFS materialize；
- 不声称默认启用或自动下载。

更新 `models/MODELS.md`：

- OEM classes metadata；
- logical identity 与 digest；
- 两份 SegFormer 均可供 VQA semantic-mask seam；
- counting enablement 与 VQA availability 明确分离。

本计划作为设计基线保留，不改写成开发日志。

### 14.16 阶段 14：最终验证顺序

先运行最小局部集合：

```text
pytest -q tests/agents/test_evidence_catalog.py
pytest -q tests/agents/counting/test_expert_catalog.py
pytest -q tests/agents/general_vqa/evidence
pytest -q tests/agents/general_vqa/test_agent.py
pytest -q tests/models/test_segformer_runtime.py tests/models/test_segformer_transformers.py
pytest -q tests/workflows/test_visual_planner.py tests/workflows/test_run_store.py
pytest -q tests/application/test_settings.py tests/application/test_bootstrap.py
pytest -q tests/integration/test_general_vqa_vertical_slice.py
```

再运行架构门禁：

```text
pytest -q tests/architecture/test_implementation_status.py
pytest -q tests/architecture/test_import_boundaries.py
pytest -q tests/architecture/test_init_side_effects.py
pytest -q tests/architecture/test_package_discovery.py
pytest -q tests/architecture/test_no_new_to_legacy_imports.py
```

然后运行：

```text
python -m compileall agents models application workflows
git diff --check
git status --short
```

若环境允许，再运行全量离线 pytest；必须如实报告与本任务无关的既有失败。

真实权重 live gate 只有在三份 LFS pointer 已 materialize 后执行。每个 live case
至少保存安全统计：

```text
model logical ids
tile count and sizes
peak concurrency
per-model call count
output mask/box dimensions
GPU/CPU provider
stable success/error code
```

不得保存绝对 checkpoint path、raw tensor 或模型输入图片 payload。

### 14.17 Definition of Done

只有同时满足以下条件才可汇报完成：

1. Planner 对 General VQA 发布准确的 runtime-available 26-leaf 子集；
2. OEM class map 通过连续性、逆映射和 asset 校验；
3. 任意 ROI 被 no-gap/no-overlap partition，所有专家输入严格 1024 square；
4. remainder 使用 LANCZOS stretch，SegFormer mask 使用正确空间恢复；
5. YOLO/SegFormer tiles 确实有界并发且结果顺序确定；
6. YOLO only、SegFormer only、combined 三条最终图像协议均通过像素级测试；
7. 最终渲染严格遵守 plan leaf whitelist；
8. YOLO-on-mask 框与 mask palette 满足冻结颜色距离；
9. request hash 覆盖实际渲染内容；
10. 新 run/resume identity 完整，历史 succeeded 零推理；
11. Counting、evaluation、reporting、CLI、UnifiedSample 和 routing 无行为漂移；
12. 聚焦测试、架构测试、compileall 和 diff check 真实通过；
13. 未执行的 live gate 及原因被明确记录。

### 14.18 最终汇报模板

实施 Agent 最终汇报必须包含：

```text
修改内容与原因
修改文件清单
catalog/class-map 版本
三分支行为结果
tile/concurrency 参数
实际运行的测试命令与精确结果
未运行的 live tests 与原因
UnifiedSample/task-routing/model-interface/evaluation/report/CLI/resume 影响矩阵
已知风险（尤其 no-overlap 边界目标与显存并发）
用户原有改动是否保持不动
```
