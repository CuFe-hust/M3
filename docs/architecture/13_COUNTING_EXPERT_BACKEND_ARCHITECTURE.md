# 13 — Counting Expert Backend Architecture

> Status: approved architecture paths only; no business implementation in C0.
> 状态：C0 仅批准架构路径，不实施业务代码。

## 1. Decision

本项目选择 Task 12 的路线 A：SegFormer 作为 counting 的辅助感知后端，
不新增 public `semantic_segmentation` task，也不创建 `SegmentationAgent`。

固定专家优先级为：

```text
Object Detection
  > Semantic Segmentation
  > Quantity Proposal / Grounded Localization
  > Generic VLM Point Counting
```

该顺序是项目运行策略。Detection 在目标 label 有明确支持时优先；只有
catalog 明确批准某个 semantic label 使用实例近似计数时，SegFormer 才能
成为候选。合法 `count=0` 不是 backend failure，zero review 属于独立策略。

## 2. Approved module boundaries

本次架构变更批准以下生产路径：

| Path | Sole responsibility |
|---|---|
| `models/segformer_transformers.py` | 本地 SegFormer 资产到 semantic segmentation inference；不做 counting |
| `agents/counting/expert_catalog.py` | `expert_catalog.json` 到已校验 capability specs 与确定性 lookup；不运行模型 |
| `agents/counting/backends/semantic_segmentation.py` | semantic result 到 components、points 与 `CountingResult`；不选择专家 |

同时批准对应测试路径：

```text
tests/models/test_segformer_transformers.py
tests/agents/counting/test_expert_catalog.py
tests/agents/counting/test_semantic_segmentation_backend.py
```

`models/segformer_transformers.py` 在本架构变更开始前已存在并已登记为
implemented；本任务不修改它。其余五个不存在的批准路径登记为 pending，
不得预先创建空壳。

## 3. Data-driven expert selection

未来 `ExpertCatalog` 是专家能力的真相源。它负责规范 canonical target、
aliases、neutral hints、stable backend kind/name、logical model identity、
asset reference、class map、supported labels、counting mode、priority 与
verification state。

VLM 的职责仅限于把问题解析为稳定 `CountTargetSpec`。VLM 不得输出 backend
或 checkpoint 名称作为路由决定；dataset 名称也不得进入通用 selector 或
backend 分支。增加同类 checkpoint 应主要通过资产与 catalog 数据完成，
不要求修改 `CountingAgent.run()`。

## 4. Semantic counting limitation

SegFormer 输出 semantic mask，而不是 instance mask。未来 semantic counting
backend 可以采用：

```text
semantic mask
  -> target mask
  -> connected components
  -> component centroids
  -> LocalPointObservation
  -> GlobalPointObservation
  -> owner-core acceptance / seam handling
  -> CountingResult
```

connected-components 可能把相邻实例粘成一个 component，因此 semantic
segmentation 低于 detection。该限制不能被静默隐藏，也不能据此把 SegFormer
私自降到 VLM fallback 之后。不是所有 semantic label 都可计数；catalog 必须
逐 label 明确 `connected_components` 或 `unsupported` 等 counting mode。

## 5. Counting invariant

所有专家后端最终必须生成统一 `CountingResult`，并保持：

```text
final_count == sum(point.accepted for point in global_points)
```

CountingAgent 只负责 task gate、target parse、中性 hints、请求 selector、构造
request、调用 executor 以及打包结果/trace。专家选择、权重读取、class-map
解析、fallback 执行和 dataset-specific routing 不属于 CountingAgent。

## 6. Explicitly rejected paths

本次不批准以下职责或路径：

```text
agents/counting/model_router.py
agents/counting/expert_manager.py
agents/counting/utils.py
agents/counting/helpers.py
models/segformer_manager.py
agents/segmentation/*
evaluation/metrics/segmentation.py
```

本阶段没有 standalone segmentation task。未来若需要 public semantic
segmentation、独立 Agent、mask artifact 或 segmentation metrics，必须另开
RFC，统一审计 TaskName、routing、evaluation、reporting 与 artifact 契约。

## 7. Offline and security boundary

SegFormer 继续遵循默认离线策略：不自动下载权重。逻辑模型身份与物理资产
路径分离；trace/report 不写本机绝对路径、密钥、Base64 或原始异常文本。

## 8. C0 acceptance boundary

C0 只允许架构控制与本文档变化。后续任务必须逐个把 pending 路径实现并在
实现完成后更新 `architecture/implementation_status.json`；普通实现任务不得
再次修改 allowlist。
