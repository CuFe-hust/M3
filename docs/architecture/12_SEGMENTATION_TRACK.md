# 12 — SegFormer / OpenEarthMap / iSAID 独立扩展线

> Task 12 产物。设计定位记录：确认旧 `try_yolo` 的 SegFormer 资产现状，
> 定义 A/B 两条扩展路线的选择框架与资产政策，**不实施任何代码**。
> SegFormer 不阻塞 core runtime。

## 1. 定位声明

`try_yolo` HEAD（`b86d02b`）相对冻结基线 `ec962eb8` 仅有的增量是
SegFormer 资产（权重/config/classes/metrics + 变更文档），**没有任何新
Python runtime**。因此：

```text
SegFormer 不阻塞 core runtime
```

新架构 core runtime（Task 01–09）不依赖、不引用任何分割资产；本扩展线
独立存在，等待交付要求明确后再决定实施。

## 2. 旧资产盘点（try_yolo HEAD，只读参考）

| 路径 | 内容 | 状态 |
|---|---|---|
| `models/segformer_mitb2_isaid/config.json` | `SegformerForSemanticSegmentation`，`num_labels=16`，**id2label 为占位（LABEL_0..15）** | 小文件，可版本化 |
| `models/segformer_mitb2_isaid/classes.json` | **权威 id2name 映射**（background/storage_tank/Large_Vehicle/Small_Vehicle/plane/ship/Swimming_pool/Harbor/tennis_court/Ground_Track_Field/…，16 类）——经本地 iSAID 训练 mask（P0000/P0025/P0111）混淆矩阵验证，与官方 instance 顺序不同 | 权威资产 |
| `models/segformer_mitb2_isaid/metrics.json` | 验证混淆矩阵（7208 行） | 小文件 |
| `models/segformer_mitb2_isaid/model.safetensors` | **Git LFS 指针**（oid sha256:f8e60686…，size 109,487,184）；实际权重在 LFS/外部存储 | 不随 git 对象分发 |
| `models/segformer_mitb2_oem/config.json` | `SegformerForSemanticSegmentation`（无 classes.json） | 小文件 |
| `models/segformer_mitb2_oem/metrics.json` | 验证指标（855 行） | 小文件 |
| `models/segformer_mitb2_oem/model.safetensors` | LFS 指针 | 同上 |
| `docs/changes/2026-08-06-agent-fix-isaid-class-labels.md` | 变更记录：混淆矩阵验证结论 + 映射权威声明 | 模型卡/变更记录 |

关键事实：**config.json 的 id2label 是占位符，classes.json 才是经验证的
权威映射**（本地 iSAID_coco 类别 id 顺序 ≠ 官方 instance 顺序）。任何未来
实现必须以 classes.json 为唯一权威，绝不信任 config 占位标签。

## 3. 选择框架：A vs B

### A. Auxiliary perception backend（辅助感知后端）

为 counting / spatial / grounding 提供语义先验（如道路/车辆/建筑区域掩码
作为 prompt 或证据）。**不新增 public task**，不改 `TaskName`/路由/评估
registry。影响面小：新增 SegFormer 模型 wrapper + 可选后端模块；仍需
架构变更（allowlist/import_rules）但无任务契约扩展。

### B. Standalone semantic segmentation（独立语义分割）

新增：

```text
semantic_segmentation（public task）
SegmentationAgent
SegFormer model wrapper
OEM/iSAID adapters
mask artifact
mIoU / mAcc 指标
report overlay（掩码可视化）
```

影响面大：`TaskName`/`TaskResolutionRequest`/`routing.POLICIES`/
`EvaluationTask`/`EXPECTED_METRICS`/reporting 全部要扩展——属于架构级变更。

### 选择条件

```text
项目交付明确要求语义分割      → 选 B
需要分割先验但不承诺交付任务   → 选 A
无明确要求                    → 都不实施，保持独立线
```

## 4. 本任务结论

当前交付（core runtime：Judge → SampleRunner → DatasetRunner → auto-task
seam → Reporting → Application → `main.py run-dataset`）**不包含语义分割
要求**。因此：

```text
本阶段不实施 A 也不实施 B；资产政策先行定案，扩展线保持待命。
```

若未来选 B，必须先单独 architecture-change 任务（不得直接创建文件），
可能路径（引用任务清单）：

```text
data/adapters/openearthmap.py
data/adapters/isaid.py
models/segformer_transformers.py
agents/segmentation/__init__.py
agents/segmentation/schema.py
agents/segmentation/agent.py
evaluation/metrics/segmentation.py
```

架构注意事项（B 专属）：`TaskName` 增加 `semantic_segmentation` 会牵动
`UnifiedSample`/resolver allowed list/`routing.POLICIES`/`EvaluationTask`/
`EXPECTED_METRICS`/`RunTaskName`/reporting 指标族——需一次性审计所有
task 封闭集；`agents.*.agent` 与 `workflows.sample_runner` 的 reporting
禁止项保持。若选 A，则不需要任何 task 契约变更。

## 5. 模型资产政策（两条路线通用）

- **权重**：Git LFS / 外部目录策略；git 对象只存 LFS 指针（保持现状）。
  新架构下载策略遵循"离线默认"——本地权重就位才可运行，绝不自动下载。
- **小文件可版本化**：config.json / classes.json / metrics.json /
  model card 直接入库。
- **logical model id**：权重引用一律用逻辑 id（如
  `segformer-mitb2-isaid-local`），物理路径只传给加载器；哈希/ trace/
  报告只用逻辑 id（与 Qwen `cache_model_id` 同一约定）。
- **sha256**：权重注册表记录文件摘要（LFS 指针 oid 即 sha256，可直接复用）。
- **offline default**：`allow_download=false` 为唯一默认。
- **iSAID 权威映射**：`classes.json` 保持权威，任何实现不得从 config
  id2label 占位推断类别名。

## 6. 训练代码边界

训练正式进入工程时，另开独立 RFC 与目录：

```text
training/
```

不得塞进 runtime packages（data/models/agents/routing/workflows/evaluation/
reporting/application）。训练产物（checkpoint/config/metrics）按第 5 节
资产政策入库存档。

## 7. 验收

本任务为零代码变更：只新增本设计记录。验证 = 全量测试不受影响
（`python -m pytest -q` 全绿）且工作区仅含本文档与状态登记。
