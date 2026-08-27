# 26. VQA Evidence 有界内存与流式物化编码规范

> Status: Normative implementation specification  
> Audience: Coding agents modifying the current M3 architecture  
> Scope: `agents/general_vqa/evidence/**`、`agents/general_vqa/agent.py`、必要的
> `models.images` 图像读取 seam 与对应测试  
> Out of scope: 模型权重、VisualTaskPlanner 决策、task/router、指标、数据集纳入规则、
> checkpoint 与逻辑模型身份

## 0. 编码代理执行协议

本文件是实现任务的约束规范，不是可自由取舍的设计建议。编码代理执行本任务时，
必须同时遵守根目录 `AGENTS.md`、`DETAILS.md`、`architecture/import_rules.json` 及本文件。
发生冲突时，以更严格且不破坏既有数据、评测、运行和模型身份契约的规则为准。

### 0.1 规范词

本文中的规范词具有以下含义：

- **MUST / 必须**：实现和验证不可缺少；
- **MUST NOT / 禁止**：任何情况下不得作为本任务实现的一部分；
- **SHOULD / 应**：除非存在可记录且可验证的技术阻塞，否则必须执行；
- **MAY / 可以**：在不扩大任务范围且满足所有 MUST 的前提下可选择。

编码代理不得把 MUST 降级为建议，也不得用“后续优化”推迟安全、确定性、测试或
cache identity 要求。

### 0.2 授权范围

编码代理只可以为本规范修改下列职责范围：

```text
models/images.py                         # region-read protocol / generic image seam only
agents/general_vqa/evidence/geometry.py  # pure tile and sampling geometry
agents/general_vqa/evidence/schema.py    # persisted-safe geometry, only if strictly needed
agents/general_vqa/evidence/rendering.py # tile/mask preview rendering
agents/general_vqa/evidence/executor.py  # bounded scheduling and evidence execution
agents/general_vqa/agent.py              # final preview consumption only
application/**                           # composition injection only if a new protocol requires it
tests/models/**
tests/agents/general_vqa/**
tests/workflows/test_visual_planner.py    # only affected materialized-view assertions
DETAILS.md                               # final current-state documentation
docs/architecture/26_VQA_EVIDENCE_BOUNDED_MEMORY_SPEC.md
```

修改 `application/**` 前，编码代理 MUST 证明 protocol 无法由现有 composition 注入。
修改 `schema.py` 前，编码代理 MUST 证明纯运行时 dataclass/tuple 无法表达所需状态。

编码代理 MUST NOT 修改：

```text
data/schema.py
routing/**
evaluation/**
reporting/**
dataset adapters
VisualTaskPlanner prompt/schema/task decision
model checkpoints or class maps
tests/fixtures/migration/**
```

如实现确实需要越过该边界，编码代理 MUST 停止修改，报告具体依赖与影响，并请求用户授权。

### 0.3 开始修改前的强制检查

编码代理 MUST 在编辑前完成并记录：

1. `git status --short`，识别用户已有修改；
2. 目标路径附近是否存在更局部的 `AGENTS.md`；
3. 当前 `ObjectEvidenceExecutor`、tile geometry、SegFormer preprocess/restore、最终视觉
   rendering 的实际调用链；
4. 当前 preprocessing、visual-content、preparation identity 版本所在位置；
5. 与目标代码直接相关的现有测试集合；
6. 当前失败样本的稳定错误类型、ROI geometry 和失败位置。

编码代理 MUST 保留所有无关用户修改，不得覆盖远端或本地 dirty worktree 中的既有工作。

### 0.4 强制实施顺序

实现 MUST 严格按以下阶段推进；前一阶段的门禁未通过时不得进入后一阶段：

```text
Gate 0  baseline/parity fixtures
  -> Gate 1 pure geometry and region-read seam
  -> Gate 2 YOLO lazy bounded tile execution
  -> Gate 3 SegFormer preview-space inverse mapping
  -> Gate 4 final evidence rendering integration
  -> Gate 5 targeted large-ROI validation
  -> Gate 6 full relevant tests and documentation
  -> Gate 7 resume full prepare only after explicit user authorization
```

### 0.5 明确禁止的“修复”

编码代理 MUST NOT：

- 永久或进程级设置 `Image.MAX_IMAGE_PIXELS = None`；
- 仅捕获 `DecompressionBombError` 后跳过 ROI、tile、leaf 或 sample；
- 把失败 leaf 猜成 background、missing 或 hit；
- 降低 ROI 分辨率后声称与原协议等价而不做 parity/版本处理；
- 为降低内存而改变 YOLO tile size、重叠策略、NMS、confidence 或 max detections；
- 为降低内存而提前运行本不需要的 SegFormer；
- 让并发 completion order 决定 bundle/order/hash；
- 一次性提交携带全部 PIL tile 的 Future；
- 在 trace、schema 或 Future 中长期保存 PIL、NumPy、tensor、Base64；
- 减少 cache/request hash 输入或复用不兼容旧 identity；
- 修改、删除或跳过失败测试；
- 以新增重量依赖替代本规范的第一阶段实现；
- 未经用户明确授权重启全量训练或 prepare。

### 0.6 每阶段交付格式

编码代理在每个 Gate 完成后 MUST 记录：

```text
changed files
contract preserved
tests executed
test result
unverified item
remaining risk
```

只有实际执行的测试才能报告为 passed。无法运行的测试 MUST 给出命令、原因和替代检查。

## 1. 背景与问题

当前 GeneralVQAAgent 的 object-evidence 路径可以处理超大遥感 ROI，但中间图像的
物化方式没有保持有界内存。

### 1.1 YOLO 当前路径

当前 executor 先裁出完整 ROI，再同步创建全部 `1024×1024` tile，之后才开始模型调用：

```text
source image
  -> full ROI crop (W×H RGB)
  -> all tile images materialized in row-major order
  -> bounded worker pool runs YOLO
  -> detections inverse-mapped to ROI/source coordinates
```

虽然单次 YOLO 输入已经是严格 `1024×1024`，但完整 ROI 副本和所有 tile 会同时驻留。
稳定的 row-major 归并顺序被错误地与“提前物化全部图像”绑定。

### 1.2 SegFormer 当前路径

当前 SegFormer 正向预处理为：

```text
ROI W×H
  -> right/bottom padding to Wp×Hp (minimal multiples of 1024)
  -> LANCZOS resize to 1024×1024
  -> SegFormer class-id map 1024×1024
```

模型输出后，当前实现执行高成本逆恢复：

```text
1024×1024 class-id map
  -> NEAREST resize to Wp×Hp
  -> crop to W×H
  -> one W×H boolean mask per requested leaf
  -> W×H RGBA pure-mask composition
  -> shrink final visual to <=1080
```

这不会增加模型信息，却可能创建多份数亿像素的中间位图。远端样本
`lrs-vqa-supplement-STAR_1424` 的 ROI 为 `207,533,568` 像素，已在完整 mask
恢复阶段触发 Pillow `DecompressionBombError`。

## 2. 目标

1. YOLO tile 图像按需物化，活跃 tile 数受 `max_tile_concurrency` 硬限制。
2. YOLO 不再创建完整 ROI crop，也不在计划中保存 PIL 图像。
3. SegFormer 不再恢复 `Wp×Hp` 或 `W×H` 的完整 class-id/boolean mask。
4. SegFormer 在 `1024×1024` 模型 mask 上排除 padding，并直接映射到最终
   `<=1080` ROI preview。
5. 保持现有 task、planner、catalog、fallback、模型调用顺序、模型输入尺寸、
   YOLO 坐标逆映射、palette、最终视觉分支和持久化 schema 的语义。
6. 保持确定性：并发完成顺序不得影响 bundle、trace、最终请求或 request hash。
7. 对异常继续 fail closed，不跳过失败样本，不伪造 background 或 leaf hit。

## 3. 非目标

- 本阶段不引入 pyvips/libvips、rasterio 或新的强制依赖。
- 不批量转换 JPEG/PNG 为 tiled TIFF。
- 不改变 SegFormer/YOLO checkpoint、阈值、NMS 或 class map。
- 不改变 `VisualTaskPlan`、`MaterializedVisualView`、`UnifiedSample`。
- 不改变 deterministic evaluation 或 Judge。
- 不以降低图像质量、减少样本或吞掉模型错误换取运行成功。

## 4. 必须保持的不变量

### 4.1 YOLO

- 每次 YOLO 调用继续接收严格 `1024×1024 RGB`。
- `EvidenceTileRecord.tile_id`、row、column 和 row-major 顺序不变。
- 完整 tile 不缩放；右/下尾块继续以 LANCZOS 拉伸到 `1024×1024`。
- detection 使用既有独立 `scale_x/scale_y` 与 tile offset 逆映射。
- tile 之间仍不重叠，每个 ROI 像素恰好属于一个 tile。
- 所有 YOLO 结果先聚合，再决定 still-missing SegFormer leaves。

### 4.2 SegFormer

- 输入仍为 ROI 右侧/底部最小 padding 后缩放得到的严格 `1024×1024 RGB`。
- RGB 缩放继续使用 LANCZOS；离散 class ID 只使用 NEAREST 语义。
- padding 不得出现在最终 mask。
- palette、leaf 稳定顺序及后叶覆盖前叶的优先级不变。
- YOLO-only、SegFormer-only、YOLO+SegFormer 三个最终视觉分支不变。
- YOLO hit 不被 SegFormer 覆盖。

### 4.3 安全与持久化

- 不持久化 PIL、tensor、Base64 源图或绝对路径。
- trace 继续只记录稳定几何和错误类型。
- 不修改已成功 sample cache；preparation identity 的版本变更必须显式。
- 如最终模型可见图像发生任何像素变化，必须升级视觉内容/预处理版本并使
  request hash 变化，禁止复用旧 cache。

## 5. 目标架构

```text
MaterializedVisualView / RoiEvidenceRecord
                  |
                  +--> lightweight tile geometry plan
                  |        |
                  |        +--> bounded submit window
                  |                 |
source image -----+------------> read one tile box
                                    -> normalize/resize to 1024
                                    -> YOLO
                                    -> release tile image
                                    -> ordered result slot

source ROI geometry
        |
        +--> one 1024 SegFormer input
                 -> 1024 class-id map
                 -> padding-aware direct sampling
                 -> <=1080 preview class grid
                 -> leaf masks/pure-color preview
```

计划对象只保存几何，不保存像素。

## 6. 阶段 A：图像区域读取 seam

在 `models.images` 增加职责明确的只读 region seam，例如：

```python
class ImageRegionSource(Protocol):
    @property
    def size(self) -> tuple[int, int]: ...

    def read_box(self, box: tuple[int, int, int, int]) -> Image.Image: ...
```

约束：

- application/workflow 负责从已验证的 `ImageRef.path + data_root` 创建 source；
- Agent 只消费 seam，不选择 JPEG/TIFF backend；
- `read_box` 返回独立 RGB 图像，边界严格验证；
- 第一版可使用 Pillow backend，不增加依赖；
- source 生命周期限定为单 sample，并显式关闭。

### 6.1 格式能力说明

- tiled TIFF 可以由后续可选 backend 实现真正的磁盘窗口读取；
- 普通 JPEG/PNG 使用 Pillow 时可能仍需解码完整源文件；
- 即使 Pillow backend 整图解码，本计划仍会消除完整 ROI 副本、全部 tile 副本和
  完整 SegFormer mask，先解决当前主要峰值；
- 不得声称 Pillow JPEG/PNG backend 已实现真实随机窗口 I/O。

## 7. 阶段 B：YOLO 轻量 geometry plan

将 `_materialize_tiles()` 拆为：

```text
_plan_tiles(records) -> tuple[EvidenceTileRecord, ...]
_read_model_tile(source, roi, tile_record) -> 1024×1024 Image
```

`_plan_tiles` 只运行 `partition_roi()`，返回轻量 Pydantic geometry records。
全局 source box 由以下确定性变换得到：

```text
global_x0 = roi.expanded_xyxy.x0 + tile.source_tile_xyxy.x0
global_y0 = roi.expanded_xyxy.y0 + tile.source_tile_xyxy.y0
global_x1 = roi.expanded_xyxy.x0 + tile.source_tile_xyxy.x1
global_y1 = roi.expanded_xyxy.y0 + tile.source_tile_xyxy.y1
```

worker 只在执行前读取该 box。完整 `1024×1024` tile 原样使用，尾块沿用既有
LANCZOS resize。

## 8. 阶段 C：YOLO 有界提交窗口

不能一次性为所有 tile 创建携带 PIL 图像的 Future。采用固定窗口：

1. 预先为每个 tile 分配稳定 sequence index；
2. 最多提交 `max_tile_concurrency` 个任务；
3. 任一任务完成后，释放其 tile 图像并提交下一个 geometry record；
4. 输出写入 `results[index]`，而不是按 completion order append；
5. phase 结束后按 index 顺序执行验证、审计和 detection 聚合。

必须保证：

```text
active materialized YOLO tiles <= max_tile_concurrency
```

错误处理保持每 tile 隔离和稳定 type-name code；不得因为流式调度而吞掉失败。

## 9. 阶段 D：SegFormer 直接生成预览 mask

### 9.1 正向预处理保持不变

设：

```text
ROI size       = W × H
padded size    = Wp × Hp
model size     = M × M, M = 1024
preview size   = Vw × Vh, max(Vw, Vh) <= 1080
```

仍执行：

```text
ROI -> right/bottom padding -> LANCZOS resize to M×M -> SegFormer
```

### 9.2 禁止完整逆恢复

删除运行时中的以下大图路径：

```text
model_mask.resize((Wp, Hp), NEAREST)
restored.crop((0, 0, W, H))
W×H per-leaf boolean masks
W×H RGBA composition
```

### 9.3 padding-aware 直接采样

对最终 preview 中每个像素，按当前 Pillow NEAREST 的像素中心约定，直接求其在
`M×M` class-id map 中的源索引：

```text
preview pixel center
  -> ROI-local continuous coordinate in W×H
  -> same coordinate in top-left of Wp×Hp padded frame
  -> model-mask coordinate in M×M
  -> nearest discrete class id
```

因为 preview 只覆盖 `[0, W) × [0, H)`，右侧 `[W, Wp)` 和底部 `[H, Hp)` 的
padding 永远不会被采样。

实现不应仅使用未经证明的：

```python
round(M * W / Wp)
round(M * H / Hp)
```

直接整数裁边，因为有效边界通常是分数，简单 floor/ceil/round 可能引入一行或一列
漂移。应封装纯函数生成 x/y lookup table，或使用经过 parity 证明的仿射 NEAREST
变换。

### 9.4 preview 尺寸

preview size 必须复用现有 `compute_preview_size(record.crop_size)`，保持最长边 1080、
小图不放大。class-id preview 应一次生成；随后所有 leaf 从这张小 grid 提取 mask。

### 9.5 小尺寸兼容路径

可以统一使用直接采样路径。若保留旧路径作为测试 oracle，只能位于测试代码，生产
runtime 不得用动态 fallback 在两套实现间切换。

## 10. 阶段 E：最终证据渲染改造

`EvidenceExecution.masks` 不再保存 ROI 原分辨率 boolean mask，改为保存 preview-space
mask，或保存一张 preview class-id grid 加 leaf→class-id 映射。优先后者，避免每 leaf
复制一张 mask。

建议运行时表示：

```python
SegFormerPreviewEvidence(
    roi_id,
    preview_size,
    class_id_grid,
    leaf_class_ids,
)
```

这只是 Agent 内部运行时包装，不创建新的全局持久化 Prediction schema。

`render_pure_mask()` 在 preview size 上着色：

- 黑色背景；
- stable leaf order；
- later leaf overwrites earlier leaf；
- palette 不变；
- 不再创建 W×H RGBA canvas。

YOLO-on-mask 分支应将 ROI-local YOLO boxes 直接按 `preview_size / crop_size` 缩放到
preview mask，复用现有 `source_size` 缩放 seam，不恢复大图。

clean ROI 也应只生成一次 preview。第一版 Pillow backend 如必须整图解码，仍不得创建
第二份 W×H ROI 副本用于纯 mask。

## 11. 确定性与 cache/version

### 11.1 需要证明的 parity

对不会触发资源问题的小/中 ROI，建立旧实现测试 oracle：

- YOLO tile 输入逐像素一致；
- YOLO call 顺序的逻辑身份一致；
- detection bundle 顺序与坐标一致；
- SegFormer preview class ID 逐像素一致；
- pure-mask PNG bytes/hash 一致；
- YOLO-on-mask PNG bytes/hash 一致。

### 11.2 版本策略

- 若模型可见 PNG bytes 完全一致，可保持视觉内容版本，但仍升级 executor 内部实现版本
  或记录 rollout 说明；
- 若严格几何修复导致任何模型可见像素变化，升级 preprocessing/visual content version、
  preparation version，并让 request hash/identity 自然失效；
- 禁止手工保留旧 identity 来制造 cache hit。

## 12. 测试计划

### 12.1 geometry

- 多种 ROI 尺寸：`1024`、`1025`、非方形、可整除、单像素尾块；
- tile global/local 坐标往返；
- row-major identity 不变；
- preview→padded→model lookup 边界；
- padding 宽/高为 0 和接近 1023；
- 不采样 padding 区域。

### 12.2 YOLO executor

- 在第一个 YOLO 调用发生前，未读取后续全部 tile；
- 活跃 tile 峰值不超过 concurrency；
- Future completion 乱序时 bundle 仍按 plan 顺序；
- tile 调用异常继续产生稳定 audit/error；
- 尾块输入严格 `1024×1024`；
- 不调用 `render_roi_crop()` 物化完整 ROI。

### 12.3 SegFormer rendering

- production path 不调用 `resize((Wp, Hp))`；
- production path 不创建 `W×H` class-id、boolean、RGB 或 RGBA mask；
- 小尺寸新旧 preview class grid 逐像素 parity；
- 多 leaf overlap precedence parity；
- 最终 mask 最长边不超过 1080；
- 2.08 亿像素的纯 geometry fixture 不触发 Pillow bomb 检查或大内存分配。

### 12.4 集成

- YOLO-only、SegFormer-only、YOLO+SegFormer；
- YOLO hit 后 SegFormer `not_run`；
- 多 ROI、多 binding；
- `STAR_1424` 定向 prepare；
- resume/cache identity；
- JSON-safe bundle、trace 和 prepared request。

建议至少运行：

```text
tests/models/test_images.py
tests/agents/general_vqa/evidence/test_geometry.py
tests/agents/general_vqa/evidence/test_rendering.py
tests/agents/general_vqa/evidence/test_executor.py
tests/agents/general_vqa/test_agent.py
tests/workflows/test_visual_planner.py
tests/architecture/test_import_boundaries.py
```

## 13. 强制 Gate 与退出条件

### Gate 0：冻结基线

编码代理 MUST 在修改生产行为前：

1. 为小/中 ROI 保存或构造可审计的旧实现 oracle；
2. 固定 YOLO tile bytes/digest、tile geometry、detection order；
3. 固定 SegFormer class-id input、pure-mask PNG 和 YOLO-on-mask PNG；
4. 证明 fixture 不包含绝对路径、模型权重、Base64 持久化或敏感信息。

**退出条件：** baseline 测试在未修改生产代码时通过。否则 MUST 先解释现有失败，
不得把红灯基线当作新实现失败或顺手修改 fixture。

### Gate 1：纯 geometry 与 region seam

编码代理 MUST：

1. 先实现/测试 source-box 纯坐标变换；
2. 再实现 region reader seam；
3. 保持具体 backend 选择在批准的 composition 层；
4. 对 path、box、source size 做 fail-closed 校验；
5. 明确 Pillow JPEG/PNG backend 的整图解码限制。

**退出条件：** geometry 单元测试通过；普通 import 不加载模型或新增重依赖；package import
边界不变。

### Gate 2：YOLO lazy bounded execution

编码代理 MUST：

1. 把 tile plan 的类型限制为 geometry records；
2. 在 worker 即将执行时才读取/创建 tile 图像；
3. 保证携带像素的活跃任务数 `<= max_tile_concurrency`；
4. 在 `finally`/任务生命周期结束后释放 tile 引用；
5. 使用稳定 index slot 归并结果；
6. 保持 YOLO phase 完整结束后才计算 SegFormer fallback。

**退出条件：** lazy-read、峰值、乱序完成、异常隔离、尾块和坐标 parity 测试全部通过。

### Gate 3：SegFormer preview-space inverse mapping

编码代理 MUST：

1. 把 preview→model x/y lookup 实现为无 PIL/模型依赖的纯函数；
2. 明确像素中心、边界夹取和 NEAREST tie 规则；
3. 用 exhaustive 小尺寸或充分边界集合对比旧实现 oracle；
4. 直接产生 `Vw×Vh` class grid；
5. 证明任何 lookup index 都不落入 padding-only 区域；
6. 禁止生产代码创建 `Wp×Hp` 或 `W×H` mask。

**退出条件：** class-grid parity、padding exclusion、超大纯 geometry 和无大图分配测试通过。

### Gate 4：最终 evidence 集成

编码代理 MUST：

1. 仅保存一张 preview class grid；
2. 在 preview space 做逐 leaf 选择与 palette composition；
3. 保持 stable leaf overwrite order；
4. 在 YOLO-on-mask 分支按原 ROI geometry 缩放框；
5. 保持 clean preview、visual input role 和内容顺序；
6. 对最终模型可见 PNG 做 byte/hash parity 判断。

**退出条件：** 三种视觉分支集成测试通过，并已作出有证据的 version/cache 决策。

### Gate 5：定向大 ROI 验证

编码代理只有在用户允许远端执行时，才 MAY 定向运行 `STAR_1424`。运行前 MUST 确认：

- 没有同一 output/cache 的并发 writer；
- 原调用参数来自持久化或已确认命令，不从当前默认值猜测；
- 日志不会记录 credential 或原始敏感异常；
- 监控绑定真实 PID，不使用可匹配自身的模糊 `pgrep -f`。

必须记录：

```text
sample terminal state
wall time
peak RSS (or explicitly unavailable)
max active tile count
final request hash
new stable error type, if any
```

**退出条件：** sample 成功，或以新的真实错误 fail closed。出现新错误时不得自动扩大修改
范围；MUST 回到对应 Gate 分析。

### Gate 6：完整相关验证和文档

编码代理 MUST 运行第 12 节列出的相关集合及受影响架构测试，并更新 `DETAILS.md`。
如果 schema、artifact、cache identity 或公开配置未改变，MUST 明确记录“不需要更新”的证据，
而不是静默遗漏。

**退出条件：** 所有已运行检查如实汇报；未运行项、原因和剩余风险完整列出。

### Gate 7：全量 prepare/resume

全量 prepare/resume 是外部状态变更。编码代理 MUST 获得用户明确授权后才能启动。启动时：

1. MUST 使用真实持久化参数或用户确认的完整命令；
2. MUST 复用兼容 identity 的成功 cache；
3. MUST 让不兼容 identity 自然失效；
4. MUST 使用真实 PID 监控；
5. MUST 保留所有失败 sample/status/log；
6. MUST NOT 因失败过滤样本或改写 summary。

**退出条件：** prepare 产生闭合终态与 manifest，或真实失败已报告且进程状态明确。

## 14. 验收标准

以下条件必须全部满足：

1. YOLO 每次只接收严格 `1024×1024` tile。
2. YOLO 活跃 tile 图像数不超过配置并发上限。
3. YOLO 路径不创建完整 ROI crop 或全部 tile 图像列表。
4. SegFormer 路径不创建 `Wp×Hp`/`W×H` class-id mask。
5. SegFormer 路径不保存逐 leaf 的 `W×H` boolean mask。
6. 最终 mask/clean preview 最长边不超过 1080，小图不放大。
7. padding 不出现在最终 mask，class ID 只按 NEAREST 语义采样。
8. 小/中 ROI parity 测试通过；如存在有意像素差异，版本和文档已更新。
9. `STAR_1424` 成功完成且无 `DecompressionBombError`。
10. Dataset prepare 继续满足失败真实性、cache identity 和 JSON-safe 契约。

## 15. 回滚策略

- 代码按阶段独立提交，YOLO streaming 与 SegFormer preview-space restoration 可分别回滚；
- 不修改原始数据和已成功 artifact；
- 若模型可见内容版本已升级，回滚时不得把新旧 cache identity 混用；
- 发生 parity 不明、坐标漂移或 mask 污染时停止 rollout，保留失败状态与审计产物，
  不使用旧运行时动态 fallback 掩盖问题。

---

## 16. 执行记录（Rollout record）

本规范于 2026-08-27 在 `feat/vqa-evidence-bounded-memory` 分支执行（HEAD
`6deef4528a622719083448e719b37718be20ef44`）。

### 16.1 版本与 cache 决策（Gate 4 证据）

parity 测试证明最终模型可见内容逐字节不变：

- YOLO tile 输入经 region seam 读取后与旧 crop-then-tile 路径字节级一致
  （`test_tile_reads_are_byte_identical_to_legacy_crop_then_tile`）；
- SegFormer preview class grid 与旧“恢复+裁切+NEAREST 缩小”逐像素一致
  （`test_preview_direct_sampling_parity_with_legacy_oracle`，含 1024/1025/
  976/1500×800/2000×1024/1×1 等尺寸）；
- 叶子命中判定在 model mask 前缀矩形上计算，与旧整分辨率判定逐点一致
  （`test_class_ids_in_prefix_rect_matches_legacy_restored_grid`）；
- 真实 executor + agent 产出的纯色 mask PNG 与旧管线字节级一致
  （`test_end_to_end_mask_png_is_byte_identical_to_legacy_pipeline`）。

决策：视觉内容版本保持 `v2`，预处理身份字符串不变，request hash 不变，
旧 cache 继续有效；执行器内部实现版本记为 `bounded-streaming-v1`（见
DETAILS.md）。无手工保留旧 identity 制造 cache hit。

### 16.2 语义说明（有意的实现边界）

- 命中/缺失判定与旧路径完全一致（前缀矩形扫描，O(1024×1024)）；
- `EvidenceExecution.masks` 由 W×H boolean mask 改为
  `preview_evidence`（每（ROI，binding）一张 <=1080 class-id grid +
  leaf→class-id 映射），仅内存、绝不持久化；bundle/artifacts JSON 不变；
- 旧恢复路径函数（`restore_segformer_class_id_mask`、
  `stitch_class_id_masks`、`restore_class_id_mask`）已从生产代码移除，只以
  测试 oracle 形式存在于测试文件（26 §9.5）；
- 第一版 Pillow region backend 对 JPEG/PNG 仍整图解码；消除的是完整 ROI
  副本、全部 tile 副本与全分辨率 mask（26 §6.1 说明）。

### 16.3 Gate 完成状态

| Gate | 状态 |
|---|---|
| 0 baseline/parity fixtures | 完成：基线 284 项相关测试通过后开始修改 |
| 1 pure geometry + region seam | 完成（含 PIL NEAREST 机制的逐点复刻证明） |
| 2 YOLO lazy bounded execution | 完成（惰性读取、窗口峰值、乱序归并、隔离、尾块测试通过） |
| 3 SegFormer preview-space inverse mapping | 完成（parity、padding 排除、2.08 亿像素纯几何无大分配） |
| 4 final evidence rendering integration | 完成（三分支集成 + PNG 字节 parity + 版本决策） |
| 5 STAR_1424 定向验证 | 待用户授权后执行（需远端/本地大图与真实权重） |
| 6 完整相关验证与文档 | 完成（全仓 2476 passed；DETAILS.md 与本文档已更新） |
| 7 全量 prepare/resume | 未启动（需用户明确授权） |

### 16.4 已知风险

- Pillow 12.2.0 的 NEAREST resize 语义（ImagingScaleAffine：双精度累加 +
  向零截断）以单元测试锁定；若未来 Pillow 主版本改变该实现，
  `nearest_lookup` parity 测试会失败并需重新评估；
- STAR_1424 的 SegFormer 输入侧仍会瞬时物化 W×H ROI 裁切（Pillow 整图
  解码限制），未做进一步优化；本次消除了触发
  `DecompressionBombError` 的恢复侧大图；
- 超大 ROI 的 SegFormer 输入 LANCZOS 下采样语义与旧协议一致，属于既有
  模型行为（见 DETAILS.md 79.4）。
