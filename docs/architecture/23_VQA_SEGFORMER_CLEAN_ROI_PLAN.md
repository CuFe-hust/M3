# VQA SegFormer-only 补充干净 ROI 实施计划

> 状态：Implemented
>
> 日期：2026-08-21
>
> 范围：General VQA 的 `object_evidence_vqa` 最终 Qwen 图像组装；不修改模型、
> SegFormer 推理、ROI 物化、评测或持久化 bundle schema。

## 1. 目标

修正 General VQA evidence 流程中“仅命中 SegFormer”分支的最终视觉输入。

当前该分支只向最终 Qwen 发送：

```text
pure-color semantic mask
+ mask legend / label explanation
```

目标改为：

```text
pure-color semantic mask
-> clean ROI
-> mask legend / label explanation / question
```

同一 ROI 的纯色语义 mask 与干净原始裁片必须成对发送，顺序固定为 mask 在前、
clean ROI 在后。这样 Qwen 既能读取 SegFormer 给出的类别和空间分布，也能从无框、
无覆盖层的 ROI 原图中判断纹理、颜色、边界和上下文。

## 2. 当前实现与根因

问题位于：

```text
agents/general_vqa/agent.py
  GeneralVQAAgent._build_evidence_content(...)
```

该方法已经为每个 `RoiEvidenceRecord` 完成：

1. `render_roi_crop(...)`：按 `expanded_xyxy` 从已规范化源图提取精确 ROI；
2. `make_preview(raw_crop)`：生成最长边不超过 1080、LANCZOS 缩小且不放大的
   clean ROI；
3. `_pure_mask(...)`：用 executor 已返回的内存 presence masks 和确定性 palette
   合成纯色语义 mask；
4. `_image_block(...)`：用内存 PNG 传输，并记录最终 Qwen 实际收到字节的 SHA256。

根因只是 SegFormer-only 条件分支目前只 append 了 pure mask，没有 append 同一轮
循环中已经生成的 `clean`。YOLO + SegFormer 分支已经采用“mask 图 + clean ROI”
双图协议，可作为直接行为参照。

因此本任务不需要：

- 修改 `ObjectEvidenceExecutor`；
- 重新调用 SegFormer 或 Qwen；
- 修改 mask 拼接、tile、palette 或 ROI 几何；
- 从 mask 反推 box/count；
- 新增 Python 文件或修改架构白名单；
- 修改 `UnifiedSample`、模型协议、评测或 reporting。

## 3. 目标协议

### 3.1 逐 ROI 图像分支

修改后的稳定协议为：

| 实际 evidence | 最终 Qwen 图像，按发送顺序 |
|---|---|
| 仅 YOLO | `annotated_roi` |
| 仅 SegFormer | `segformer_pure_mask` → `clean_roi` |
| YOLO + SegFormer | `yolo_on_segformer_pure_mask` → `clean_roi` |
| 两者均无 | `clean_roi` |

只有“仅 SegFormer”一行改变；其他三行保持当前行为。

### 3.2 图像语义

`segformer_pure_mask`：

- 继续使用黑色背景和当前确定性 leaf palette；
- 继续只渲染实际命中且属于请求白名单的 leaves；
- 最长边超过 1080 时使用 NEAREST 缩小，小图不放大；
- 不混入 ROI 原图像素，不改成半透明 overlay；
- 不持久化 mask 像素或 PNG 文件。

`clean_roi`：

- 必须来自与 pure mask 完全相同的 `RoiEvidenceRecord.expanded_xyxy`；
- 使用当前 `render_roi_crop(...)` 结果，不重新裁切或猜测坐标；
- 最长边超过 1080 时使用 LANCZOS 缩小，小图不放大；
- 不绘制框、标签、mask、置信度或其他覆盖层；
- 只以内存 PNG 传给最终 Qwen，不新增磁盘 artifact。

### 3.3 图像角色说明

保留当前 `mask_legend`，并在最终文本 payload 中增加一个与 `image_url` blocks
顺序严格对应的轻量角色清单，例如：

```json
{
  "visual_inputs": [
    {
      "content_image_index": 0,
      "roi_id": "roi-1",
      "role": "segformer_pure_mask"
    },
    {
      "content_image_index": 1,
      "roi_id": "roi-1",
      "role": "clean_roi"
    }
  ]
}
```

该清单解决多 ROI 或混合 evidence 时“哪张图属于哪个 ROI、哪张是 mask、哪张是
原图”的歧义。它只描述已经组装的 content，不参与能力判断，也不进入
`VqaEvidenceBundle`。

建议同时在 `evidence_identity` 与安全 trace 中加入固定的最终视觉内容协议版本，
例如：

```text
visual_content_version = "v2"
```

该版本只标识最终图像排列协议，不复用或篡改 `palette_version`、
`preprocessing_version`。如果审核认为无需新增显式版本，最低要求仍是保留
`visual_inputs`，并确保新增 clean ROI 的实际 PNG digest 进入 request hash。

## 4. 实施步骤

### 阶段 A：冻结 content 角色与顺序

在 `agents/general_vqa/agent.py` 中为最终 Qwen 图像角色定义封闭字符串集合或模块级
常量，角色限定为：

```text
annotated_roi
segformer_pure_mask
yolo_on_segformer_pure_mask
clean_roi
```

组装每个 image block 时同步追加一条 `visual_inputs` 记录，确保：

- `content_image_index` 从 0 开始，严格对应 user content 中 image blocks 的顺序；
- `roi_id` 来自当前 `RoiEvidenceRecord`；
- role 只描述已生成图像，不能重新决定 YOLO/SegFormer capability；
- image block 与角色记录在同一分支、同一处追加，避免次序漂移。

### 阶段 B：补齐 SegFormer-only 的 clean ROI

在 `_build_evidence_content(...)` 的：

```python
elif seg_leaves and not yolo_boxes:
```

分支中执行：

1. 继续先追加 `make_preview(_pure_mask(...), resample=NEAREST)`；
2. 随后追加当前循环已经生成的 `clean`；
3. 分别记录 `segformer_pure_mask`、`clean_roi` 角色；
4. 两次都通过 `_image_block(...)`，使发送字节及 digest 顺序完全一致；
5. 不增加模型调用、executor 调用或 Qwen budget 消耗。

为避免重复逻辑，可以给现有 `_image_block(...)` 增加一个仅在调用方维护角色记录的
薄包装；如果包装反而扩大 diff，则保持显式两次 append，优先最小修改。

### 阶段 C：协议身份与缓存

新增 clean ROI 后：

- `messages` 中实际 user content 改变；
- `final_hashes` 多出 clean ROI 的真实 PNG SHA256；
- `build_request_hash(...)` 同时覆盖最终 messages、图像 digest 顺序与视觉内容版本；
- 旧的 SegFormer-only 单图请求不会错误命中新双图请求的模型缓存；
- Qwen 仍然只调用一次，planner 与 evidence 子模型预算语义不变。

不修改 `run_request.json` 的调用参数，不修改 `EvidencePreprocessingIdentity`，因为
tile/resize/mask 恢复协议没有变化。Resume 决策保持原样：已成功样本仍不重复推理；
需要重新执行的非终态样本使用新内容协议，其 request hash 与旧请求隔离。

### 阶段 D：文档同步

更新 `DETAILS.md` 中 General VQA 的逐 ROI 协议：

```text
仅 SegFormer -> 纯色 mask + clean ROI
```

并补充：

- 发送顺序是 mask first、clean ROI second；
- `visual_inputs` 描述 image block、ROI 与角色的对应关系；
- mask 使用 NEAREST，clean ROI 使用 LANCZOS；
- 两张图只在内存传输；
- 两张图的实际 digest 均进入 request hash。

同步修正 `docs/architecture/22_VQA_OBJECT_CATEGORIES_SUBMODELS_PLAN.md` 中已经冻结的
SegFormer-only 旧单图描述，使历史设计文档不会继续与批准后的当前事实冲突。只改
相关段落，不重写该文档的其他计划内容。

## 5. 测试计划

主要修改：

```text
tests/agents/general_vqa/test_agent.py
```

### 5.1 四分支内容测试

更新现有四 ROI 测试的图像数量与顺序断言：

```text
YOLO only          1 张
SegFormer only     2 张
combined           2 张
neither            1 张
合计               6 张
```

同时断言 `visual_inputs` 与六个 image blocks 一一对应，SegFormer-only 的相邻角色
必须为：

```text
segformer_pure_mask
clean_roi
```

### 5.2 SegFormer-only 大图测试

扩展当前 `test_segformer_only_agent_image_shrinks_large_mask_with_nearest`：

- image block 数由 1 变为 2；
- 第一张是 1080 上限内、仅包含背景色和 palette 色的 pure mask；
- 第二张尺寸与第一张对应同一 ROI 的等比 preview；
- 第二张包含原 ROI 像素，不包含 mask palette 覆盖或 YOLO 标注；
- 第一张用 NEAREST，第二张沿用照片 LANCZOS 路径；
- 两张图顺序不可交换。

### 5.3 精确 ROI 一致性测试

使用非整图 `expanded_xyxy` 和具有可辨识象限颜色的源图，断言 clean ROI 的像素范围
恰好来自该框，并与 mask 的 `crop_size`/preview size 一致。该测试防止误传整图、
planner preview 或相邻 ROI。

### 5.4 Request hash 测试

新增或扩展 hash 测试：保持 mask、legend、question、ROI 几何全部相同，只改变 clean
ROI 内的一个可见像素，最终 request hash 必须变化。

同时保留：

- 相同输入得到稳定相同 hash；
- mask 像素变化会改变 hash；
- palette/catalog/version 变化会改变 hash；
- 图像顺序变化会改变 hash。

### 5.5 调用次数与持久化边界

断言：

- 最终 Qwen 仍只有一次 `complete_json(...)`；
- 不增加 evidence service/SegFormer 调用；
- Qwen call budget 仍只消耗一次；
- `vqa_evidence.json` 不新增 PIL、mask、Base64 或物理路径；
- `AgentExecution.trace` 只增加安全协议版本，不保存图像或标签数组。

## 6. 验证命令

实施完成后至少运行：

```bash
python -m pytest tests/agents/general_vqa/test_agent.py -q
python -m pytest tests/agents/general_vqa/evidence/test_rendering.py -q
python -m pytest tests/agents/general_vqa/evidence/test_executor.py -q
python -m pytest tests/agents/general_vqa -q
git diff --check
git status --short
```

本变更不增加 Python 文件、不改 import DAG 或 `__init__.py`，因此不要求修改架构
白名单。若实施时实际触及跨包依赖、文件布局或 `__init__.py`，再补跑 AGENTS.md
指定的 architecture tests，不能用本计划预先豁免。

建议在具备本地 Qwen 与 SegFormer 权重的环境增加一次 live smoke gate：选择一条
SegFormer-only 样本，记录最终 Qwen 实际收到的两张图的角色、尺寸和 SHA256，并人工
确认第二张是无覆盖层的正确 ROI。离线测试通过不能替代该 live 检查。

## 7. 预计修改文件

| 文件 | 修改内容 |
|---|---|
| `agents/general_vqa/agent.py` | SegFormer-only 追加 clean ROI；记录 content image 角色；加入视觉内容协议身份 |
| `tests/agents/general_vqa/test_agent.py` | 更新分支数量/顺序/像素/hash/单次调用测试 |
| `DETAILS.md` | 更新当前 General VQA 最终图像协议事实 |
| `docs/architecture/22_VQA_OBJECT_CATEGORIES_SUBMODELS_PLAN.md` | 修正旧的 SegFormer-only 单图冻结描述 |

原则上不修改：

```text
agents/general_vqa/evidence/executor.py
agents/general_vqa/evidence/rendering.py
agents/general_vqa/evidence/schema.py
models/**
workflows/**
evaluation/**
reporting/**
```

若实施中发现必须修改上述文件，应先停止扩展并重新提交原因与影响供审核。

## 8. 验收标准

1. 仅 SegFormer 命中的每个 ROI 向最终 Qwen 发送且只发送两张对应图：pure mask、
   clean ROI，顺序固定；
2. clean ROI 与 mask 使用同一个 `RoiEvidenceRecord`，没有整图/错 ROI/坐标漂移；
3. mask legend 继续只描述实际渲染 leaves，颜色与 mask 像素一致；
4. Qwen 能通过 `visual_inputs` 明确识别每个 image block 的 ROI 和角色；
5. mask 缩放保持 NEAREST，clean ROI 缩放保持 LANCZOS，二者均只缩不放；
6. 新增 clean ROI 的真实 PNG digest 纳入 request hash，不产生旧缓存误命中；
7. 不增加 planner、SegFormer、YOLO 或最终 Qwen 调用次数；
8. 不改变 `VqaEvidenceBundle` 持久化 schema，不落盘 mask/ROI 图片；
9. YOLO-only、combined、neither 三个分支行为不变；
10. 相关离线测试和 `git diff --check` 通过，live gate 未运行时如实注明。

## 9. 影响面与风险

### 明确不影响

- `UnifiedSample` / `SampleDraft`；
- task 解析与 routing；
- model interface、checkpoint 加载与 cache model identity；
- SegFormer class map、mask 生成与 tile 拼接；
- deterministic evaluation / Judge；
- reporting；
- CLI；
- run artifact 路径与 resume 决策规则。

### 已知影响

- SegFormer-only 最终 Qwen 输入从一张图增加到两张图，会增加该分支的视觉 token、
  推理延迟和显存占用；
- request hash 必然变化，这是隔离新旧视觉输入所需的预期行为；
- 真实 Qwen 对“mask first、clean ROI second”的利用效果仍需 live 样本验证，离线测试
  只能证明协议、像素、顺序与缓存输入正确，不能证明答案质量必然提升。

## 10. 审核后执行边界

本文件只给出计划。审核通过前不修改生产代码、测试断言或当前架构事实文档。
实施时仅处理本计划列出的 SegFormer-only clean ROI 补充，不顺带调整其他 evidence
分支、模型配置、类别 catalog、评测或数据集行为。
