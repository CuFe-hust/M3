# DETAILS.md — 当前架构与接口契约

本文件记录 M3 新架构**当前有效**的项目结构、模块职责、核心接口、运行产物、评测与报告契约。

它主要面向 AI 编码代理和需要理解系统内部结构的开发者。

- 编码行为边界：见 `AGENTS.md`
- import DAG：见 `architecture/import_rules.json`
- 迁移历史：见 `docs/migration/`
- 重要架构决策：见 `docs/architecture/`

本文件不是开发日志。Task 00、11G.5、11J 等迁移过程编号不在这里作为长期接口维护。

---

## 1. 当前状态

文档整理基线：

```text
repository: CuFe-hust/M3
architecture baseline: repository mainline (`main`)
evaluation/reporting integration baseline:
  E1-E5 through 396a5900411930e16c75ff9fe18af01080c33428
legacy behavior reference:
  try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868
```

长期架构以仓库主线 `main` 为事实来源。`new_structure` 是迁移期的历史名称，不是
当前公共分支，也不是运行时依赖。

Doc 17 离线质量门记录：

```text
core v2/runtime gate: 1228 passed, 39 deselected (HTTP socket cases)
architecture implementation/import/init/package/no-legacy gate: 33 passed
full offline suite excluding missing safetensors export collection: 1953 passed, 33 failed, 1 skipped
compileall: clean
git diff --check: clean
```

剩余失败来自 HTTP socket/超大请求传输、当时仍启用的重构期路径门禁漂移、缺失可选依赖（如
`safetensors`/`peft`/`transformers`）及未参与本任务的模型测试；不能表述为全仓
pytest 全绿。真实模型、真实数据集、云端 API 或目标 Spark/部署环境相关验证仍应按
具体环境单独执行，不应把离线测试通过等同于所有 live gate 已通过。

### Doc 20 当前执行事实

Doc 20 离线验证：v5 planner/schema/geometry、direct/evidence、runtime/resume
目标回归通过；HTTP harness 仍受当前沙箱禁止绑定 `127.0.0.1` 限制。真实模型、
真实数据集、云端 API 或目标
Spark/部署环境相关验证仍应按具体环境单独执行。

新鲜的 manual `ask` 与 dataset（explicit/default/auto）入口统一使用
`workflows.visual_planner.VisualTaskPlanner`。当前配置不含
`visual_planning.enabled` 字段；composition root 只组装 v5 planner。旧 Resolver、旧 gate、联合
planner 和旧产物写入能力已删除；历史产物读取 seam 仅用于 reporting/迁移审计。

第一次规划调用的 user content 严格为按源顺序排列的内存图像 block，随后是未经
包装的原始 question 文本。输出 schema 是 `VisualTaskPlan`，版本为
`visual-task-plan-v5`，表达 task、canonical leaf、计数语义目标和可选严格整数 `roi_xyxy`，不携带
答案、GT、路径、backend、checkpoint、device、secret 或 planner confidence。v2/v3/v4/legacy 只用于
读取历史 run request；历史非终态需要重新推理时稳定拒绝。

图像先做 EXIF transpose/RGB 规范化；规划预览最长边为 1080 且不放大。显式合法区域
在 EXIF/RGB 规范化源尺寸上按 `0..999` 框向外映射，最长边向上量化到 1024
整数倍，生成允许越界的理想正方形，再直接与源图求交；不平移、不缩小、不回退整图。
历史 large-image policy 仍写入 run identity 供兼容审计，但不再阻止 v5 显式 ROI 物化。
因此实际裁片可以是长方形且不必是 1024 倍数。`MaterializedVisualView` 同时记录请求框、
请求像素框、量化边长、理想框、实际裁片和 `was_clipped`，最终 Agent 与 evidence executor
消费同一视图。

样本产物使用 `visual_task_plan.json`，只保存已校验计划与安全几何。dataset
`run_request.json` 保存 `planning_mode`、`task_prompt_version`、preview、坐标制式、
`roi_quantum`、materialization policy 和兼容的 large-image policy；v5 succeeded resume
不调用模型，v4 及更早成功样本同样只允许无模型补评测，v4 及更早需要重新推理时返回
`LEGACY_PLANNING_RESUME_UNSUPPORTED`。`run_request.json` 还保存两个独立冻结身份：
`evidence_preprocessing`（tile 预处理，14.14）与 `vqa_assistance_scope`（VQA 辅助
范围，doc 24）；二者缺失均表示历史运行，绝不从当前默认值回填。

Doc 25 已接入 Qwen3.5 单基座、多 PEFT LoRA 运行时：一次 assembly 只加载一份
base/processor，Planner 与五个 Agent 使用固定逻辑 binding 获得轻量 bound client；
adapter 选择、资产路径解析与注入只发生在 composition root/models 加载边界。
`configs/local.yaml` 首期把全部 binding 指向同一个 visual-planner supplement adapter。
真实 Transformers/PEFT 推理 gate 仍取决于部署环境的可选依赖与 GPU，不能由离线 fake
测试替代。

---

## 2. 文档职责

### `AGENTS.md`

回答：

> “编码代理修改仓库时必须遵守什么？”

例如：

- 架构边界；
- 最小修改；
- 评测保护；
- 测试；
- offline；
- secret；
- architecture tests。

### `DETAILS.md`

回答：

> “当前系统是什么样，模块和接口怎么协作？”

### `architecture/`

机器约束：

```text
import_rules.json
```

### `docs/migration/`

回答：

> “旧 `try_yolo` 行为如何迁移到新架构？”

### `docs/architecture/`

回答：

> “某个重要架构决策为什么这样设计？”

---

# 3. 总体架构

当前主流程可以概括为：

```text
                    +-----------------------+
                    |       main.py         |
                    |  sole public CLI      |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |     application/      |
                    |   composition root    |
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
      +---------------+                   +---------------+
      |   data/       |                   |   models/     |
      | schema/adapter|                   | protocol/model|
      +-------+-------+                   +-------+-------+
              |                                   |
              | UnifiedSample / SampleDraft       | VisionLanguageClient
              v                                   |
      +---------------+                           |
      | workflows/    |<--------------------------+
      | planner/run   |
      +-------+-------+
              |
              | known task
              v
      +---------------+
      |   routing/    |
      | deterministic |
      +-------+-------+
              |
              v
      +---------------+
      |    agents/    |
      | task workflows|
      +-------+-------+
              |
              v
      +---------------+
      | evaluation/   |
      | deterministic |
      | + optional    |
      | judge         |
      +-------+-------+
              |
              v
      +---------------+
      | persisted run |
      | artifacts     |
      +-------+-------+
              |
              v
      +---------------+
      | reporting/    |
      | read-only     |
      +---------------+
```

核心原则：

```text
Data decides how source records become samples.
Resolver decides what task a draft represents.
Router decides which Agent handles a known task.
Agent executes one task workflow.
Workflow coordinates execution and persistence.
Evaluation evaluates persisted/returned results.
Reporting reads persisted results; it does not execute them.
Application wires everything together.
main.py only exposes the public command surface.
```

---

# 4. 当前项目结构

主要生产结构：

```text
M3/
├── main.py
├── application/
│   ├── settings.py
│   ├── prompts.py
│   ├── bootstrap.py
│   ├── runtime.py
│   └── commands/
├── data/
│   ├── schema.py
│   ├── registry.py
│   ├── selection.py
│   ├── validation.py
│   ├── downloader.py
│   ├── loader.py
│   └── adapters/
├── models/
│   ├── base.py
│   ├── cache.py
│   ├── images.py
│   ├── settings.py
│   ├── entry.py
│   ├── qwen_transformers.py
│   ├── segformer_transformers.py
│   ├── qwen3_vl/
│   ├── qwen3_5/
│   ├── segformer_mitb2_isaid/
│   ├── segformer_mitb2_oem/
│   └── yolo_obb/
├── agents/
│   ├── schema.py
│   ├── base.py
│   ├── registry.py
│   ├── errors.py
│   ├── visual_base.py
│   ├── general_vqa/
│   ├── caption/
│   ├── grounding/
│   ├── counting/
│   └── change/
├── routing/
│   ├── schema.py
│   ├── policies.py
│   └── router.py
├── workflows/
│   ├── schema.py
│   ├── call_budget.py
│   ├── events.py
│   ├── run_store.py
│   ├── artifact_writer.py
│   ├── visual_planner.py
│   ├── judge_service.py
│   ├── sample_runner.py
│   └── dataset_runner.py
├── evaluation/
│   ├── records.py
│   ├── metrics/
│   ├── judges/
│   ├── standard/
│   └── datasets/
├── reporting/
│   ├── schema.py
│   ├── adapters.py
│   ├── builder.py
│   ├── html.py
│   ├── exporters.py
│   └── visualization.py
├── prompts/
├── configs/
├── architecture/
├── docs/
├── scripts/
└── tests/
```

完整 Python 文件列表不在文档中重复维护。当前文件结构以仓库实际目录为准；
模块间依赖边界以 `architecture/import_rules.json` 和 architecture tests 为机器约束。

---

# 5. Import DAG

机器权威：

```text
architecture/import_rules.json
tests/architecture/test_import_boundaries.py
```

简化视图：

| 包 | 主要允许的内部依赖 | 关键禁止 |
|---|---|---|
| `data` | `data` | agents/workflows/application |
| `models` | `models` | data |
| `agents` | agents, `data.schema`, `models.base`, `models.images` | application/workflows/concrete models |
| `routing` | routing, `data.schema`, `agents.schema` | models/data.adapters/workflows/application |
| `workflows` | workflows, data, agents, routing, evaluation, `models.base` | application/concrete model implementation |
| `evaluation.metrics` | evaluation/data schema/agent schema/count schema | models |
| `evaluation.judges` | approved model contracts/cache/settings | concrete main model |
| `reporting` | schema/result layers | Agent implementation/model implementation/SampleRunner |
| `application` | 全部新顶层包 | — |
| `main.py` | application | 其他内部包 |

两个特别重要的边界：

```text
routing never imports models
```

以及：

```text
only application may select concrete main-flow models
```

---

# 6. 数据层

## 6.1 `ImageRef`

`data.schema.ImageRef` 表示一个不可变图像引用。

核心字段：

```text
image_id: str
path: Path
role: image | t1 | t2 | context
width: int | None
height: int | None
sha256: str | None
```

路径契约：

- 相对 dataset root；
- 不得绝对；
- 不得包含 `.` / `..` 逃逸段；
- 序列化为 POSIX `/`;
- schema 本身不负责检查文件存在。

---

## 6.2 `GroundTruth`

核心字段：

```text
answers
count
boxes
points
labels
raw
coordinate_frame
label_binding
```

几何：

```text
box:
  4 values -> axis-aligned xyxy
  8 values -> polygon

point:
  exactly 2 values
```

所有坐标必须 finite。

`labels` 与 geometry 的绑定通过：

```text
boxes
points
all_geometry
unbound
```

或仅在无歧义情况下自动确定。

源标注不得为适配内部 metric 而改写。

---

## 6.3 `TaskNormalization`

adapter 对源任务做规范化后的结构化信息：

```text
source_task
normalized_task
semantic_subtype
confidence
normalizer
version
reason_codes
spatial_query
answer_constraints
choices
allow_multiple
count_target_hint
```

这是 `UnifiedSample` 的一等字段，不应把重要 task-normalization 信息重新塞回不透明 metadata。
`multiple_choice_vqa` 的 `choices` 至少包含两个非空且不重复的字符串；
`allow_multiple` 是独立布尔事实。Agent 不再从 `answer_constraints`、metadata 或
Ground Truth 推导选择项。读取 pre-v2 持久化 normalization 时，schema 可将历史
`answer_constraints.choices/allow_multiple` 一次性提升为 canonical 字段；这是读取
兼容 seam，不是生产 adapter 或 Agent 的权威来源。

---

# 7. `UnifiedSample`

内部 canonical sample：

```python
class UnifiedSample:
    sample_id
    dataset
    split
    task
    images
    question
    ground_truth
    metadata
    normalization
```

公开任务集合：

```text
counting
fine_grained_counting
change_caption
change_qa
grounding
spatial_relation
scene_classification
general_vqa
caption
multiple_choice_vqa
```

## 7.1 图像角色

变化任务：

```text
change_caption
change_qa
```

要求：

```text
images[0].role == "t1"
images[1].role == "t2"
remaining roles == "context"
```

其他任务：

```text
images[0].role == "image"
remaining roles == "context"
```

## 7.2 Question

允许空问题：

```text
caption
change_caption
```

其他 UnifiedSample task 要求非空问题。

空问题的 `caption` / `change_caption` 规则现在由 v4 system prompt 约束，并在
`materialize_sample(...)` 边界校验图像角色与数量。

## 7.3 Normalization 一致性

如果：

```text
sample.normalization != None
```

则：

```text
sample.normalization.normalized_task == sample.task
```

必须成立。

---

# 8. `SampleDraft`

`SampleDraft` 用于尚未确定 task 的样本前阶段。

字段类似 UnifiedSample，但没有 mandatory `task`，而是：

```text
explicit_task: TaskName | None
```

图像角色暂时可以使用普通占位角色。

正确流程：

```text
SampleDraft
  -> VisualTaskPlanner
  -> materialize_sample(draft, task)
  -> UnifiedSample
```

物化时重新建立：

```text
change task:
  t1 / t2 / context

other task:
  image / context
```

`UnifiedSample.task` 始终保持必填。

---

# 9. Sample ID

`stable_sample_id(...)` 的目标是跨平台稳定且目录安全。

如果 source id 本身安全，可以保留。

否则使用输入的稳定 SHA-256 摘要前 20 hex。

hash 输入包括：

```text
dataset
split
source_id
ordered relative image paths
question
source_index
```

绝对路径不允许进入 sample id，因此同一个逻辑样本不应因为运行机器目录不同得到不同 ID。

---

# 10. Dataset Adapter

基础协议：

```python
class DatasetAdapter(Protocol):
    name
    supported_tasks

    def probe(root, task=None) -> AdapterProbe
    def iter_samples(root, split, task) -> Iterator[UnifiedSample]
```

无逐样本明确 task 的 adapter 可以实现：

```python
class DraftDatasetAdapter(Protocol):
    name
    supported_tasks

    def probe(root, task=None) -> AdapterProbe
    def iter_drafts(root, split) -> Iterator[SampleDraft]
```

关键原则：

```text
read-only
no model call
no run artifact writes
no implicit network
no dataset-root escape
```

`AdapterProbe` 在真正运行前给出数据布局证据。

## 10.1 VRSBench caption canonical question

VRSBench `caption` 样本的 `UnifiedSample.question` 由 adapter 在样本构造边界
固定为精确字符串：

```text
Describe the image in detail.
```

- 该值由 `VRSBenchAdapter` 统一提供（模块常量 `CAPTION_QUESTION`），registry、
  dataset runtime、caption 评测脚本及其他复用 `iter_samples(...)` 的入口得到
  同一输入；源行是否包含 `question`、是否带句点或内容不同，均不能覆盖该固定值。
- 源行仍完整保留在 `GroundTruth.raw["source_row"]` 供审计。
- `stable_sample_id(...)` 与 `UnifiedSample.question` 使用同一固定问句，因此
  无安全 source ID 的 caption 样本 ID 基于规范化问句哈希生成；旧 caption run
  的 sample ID 不提供隐式 resume 迁移，既有 run/artifact 不被改写。
- XLRS-Bench caption 继续读取并校验自己的行内 `question`，不受本契约影响。

## 10.2 XLRS 惰性加载

`XLRSAdapter` 的本地/远程行加载统一为**惰性容器**（`datasets.Dataset`
或结构等价的 `LazyRows`：廉价 `len()` + 逐行流式迭代），绝不把整表
`[dict(row) for row in dataset]` 转成 list 驻留内存——XLRS 行携带大体积
图片 bytes，整体物化曾导致 RSS 飙升到 147GB。

- HF disk / hub 布局由 `_load_from_disk` / `_load_from_hub` 直接返回
  `datasets.Dataset`；
- 解压后的 XLRS caption 发布也可直接读取
  `XLRS-Bench_caption_en/train/captions.json`，图片路径相对 release 根解析
  （例如 `train/images/000000.jpg`）；该布局不要求安装 `datasets`；
- `probe` 只物化前 20 行用于字段发现，`sample_count` 走 `len()`；
- `iter_samples` 逐行流式 `yield`，消费一行物化一行；
- `dataset_loader` 注入契约保持兼容（list 也满足 `LazyRows` 结构）。

---

# 11. Dataset Registry

`data.registry.DatasetRegistry` 使用显式注册，不扫描模块、不使用 entry point 自动发现。

当前内建 registry：

| Canonical name | Alias |
|---|---|
| `VRSBench` | — |
| `LEVIR-CC` | `LEVIR` |
| `MME-RealWorld` | `MME` |
| `XLRS-Bench` | `XLRS` |
| `XLRS-Bench-lite` | `XLRS-lite` |

`MME-RealWorld` adapter version `official-v2-question-with-choices` 将源 `Text`
与 `Answer choices` 按原始顺序以换行符合并为 canonical
`UnifiedSample.question`。因此 fresh VisualTaskPlanner 与最终 VQA Agent 都能
看到完整题面；Ground Truth 仍仅保留于 `ground_truth`，不进入
question。`TaskNormalization.choices/allow_multiple` 继续作为结构化答案
约束权威。该 v2 改变 sample ID / planner request / model cache 输入身份，
不与 v1 run 混用。

`XLRS-Bench-lite` 当前 registry 配置的 supported task 为：

```text
multiple_choice_vqa
```

除 Hugging Face `save_to_disk` 布局外，adapter 也只读支持本地
`XLRS-Bench-lite_part<N>.jsonl` 分片布局。分片按数字序号稳定迭代，行内
Base64 图片按需解码到 dataset root 之外的内容寻址 cache；带 `(A)`—`(E)`
标签的 `multi-choice options` 字符串被严格解析为结构化 choices。经完整发布
审计确认，`l2-category == "Overall Land use classification"` 是复选子类；其
紧凑源答案（例如 `BCD`）规范为 `B, C, D`，原值仍保留在 Ground Truth raw。
运行入口为：

```text
bash scripts/run_xlrs_lite_vqa.sh
```

可通过 `XLRS_VQA_CACHE_PARENT` 指定图片 cache 的父目录；完整数据运行需要
为解码后的图片预留约 48 GB 空间。

未知数据集显式失败。

---

# 12. Manifest Draft Adapter

`data/adapters/manifest.py` 提供显式 manifest 驱动的 draft 适配能力。

原则：

- 使用版本化 `spacers_adapter.json`;
- `samples_file` 明确指定；
- fields 显式映射；
- JSON / JSONL；
- task 列可选；
- 不猜字段名；
- 不调用模型；
- `samples_file` 必须被限制在 dataset root 内。

它的职责是把没有标准内建 adapter 的显式 manifest 转成 `SampleDraft`，不是变成万能自动猜数据格式工具。

---

# 13. 数据下载与便利加载

## 13.1 下载

显式下载实现：

```text
data/downloader.py
application/commands/download_data.py
```

公共命令：

```text
python main.py download-data --root ... --datasets ...
```

这是受支持的自动下载路径。

普通 adapter / loader 不会因为文件不存在而隐式联网下载。

## 13.2 便利加载

```text
data/loader.py
```

提供 registry → probe → iterator 的便利封装。

样本路径与 draft 路径保持类型分离，draft 不伪装成 UnifiedSample。

---

# 14. 模型层

## 14.1 模型协议

`models/base.py` 定义主流程模型客户端协议、RequestMeta、cache identity、request hash 与消息脱敏基础。

领域 Agent/Workflow 依赖：

```text
VisionLanguageClient
```

而不是具体 Qwen class。

## 14.2 当前主流程模型入口

`models.entry.ModelName`：

```text
qwen_transformers
qwen3_vl_baseline
qwen3_5_transformers
qwen3_5_multi_adapter
segformer_transformers
```

统一构造：

```python
create_model(name, **kwargs)
```

### `qwen_transformers`

共享本地 Transformers Qwen 客户端。

### `qwen3_vl_baseline`

Qwen3-VL baseline wrapper。

### `qwen3_5_transformers`

通过 `models/qwen3_5/model.py` 暴露，同样复用共享 Qwen Transformers 客户端能力。

### `qwen3_5_multi_adapter`

`models/qwen3_5/multi_adapter.py` 提供 `MultiAdapterQwenEngine`。engine 独占一份
Qwen3.5 base、processor、命名 PEFT LoRA inventory、response cache 和 generation lock；
`bind(...)` 返回继续满足 `VisionLanguageClient` 的 `BoundQwenAdapterClient`。
cache hit 不取得生成锁；miss 时 adapter 切换、首次生成和有界 JSON repair 位于同一锁内。
运行态不调用 `merge_and_unload()`，全部参数冻结并保持 eval mode。

## 14.3 专家模型运行层

`models/segformer_transformers.py` 提供两份本地 fine-tuned SegFormer 的共享
运行层：资产完整性校验、惰性单次加载、processor、device/dtype、预处理、
推理、logits 上采样、argmax 和明确的 `SegmentationResult`。默认纯本地，
不会自动下载。

iSAID 的 `classes.json` 是 checkpoint 权威类别映射；`config.json` 中的
`LABEL_0..15` 只是占位。OEM 的 `classes.json` 已于 2026-08-20 由用户针对
本地 OpenEarthMap checkpoint 明确确认（9 类：background + bareland/
rangeland/developed_space/road/tree/water/agriculture_land/building）。OEM
`config.json` 的 id2label 仍是 `LABEL_0..8` 占位，不能作为语义顺序的独立
证据；runtime 只相信已确认的 `classes.json` 与 checkpoint digest 绑定。

YOLO 的新版 Counting 实现已经是旧版的加强版，仍保持唯一实现：
`YoloModelStore` 负责惰性单次加载和完整性校验，ONNX adapter 负责 provider、
预处理与解码，Counting backend 负责目标语义、切片、去重和 fallback。
通用 `models.base.validate_local_model_asset` 在模型 runtime 之前统一拦截 LFS
pointer；没有新增第二套 YOLO loader。

当前本地部署 inventory 包含 DOTA-v2.0 YOLO11m-OBB 与 VRSBench-QA1024
YOLO11m-OBB。两者使用权重内嵌并经 SHA-256 绑定的 DOTA 18 类顺序；DOTA checkpoint
替代旧 YOLOv5m-OBB CSL ONNX 资产，VRSBench checkpoint 替代旧 iSAID YOLO11s detect
资产。为兼容既有 artifact 与外部配置，DOTA deployment slot 仍使用稳定 backend 名
`detector_obb_csl_001`，实际模型身份由新的 logical model ID 与 SHA 决定。Counting
selector 先按 canonical label 检查 detector class capability，
再在同 kind 内按 priority 选择：VRSBench 专家 priority 200，DOTA 专家 priority 100。
`visual-evidence-catalog-v4` 的 YOLO raw labels 同步使用两个 checkpoint 内嵌的连字符
类别名；面向用户的空格写法仍由 canonical alias normalization 处理。

`agents/counting/backends/semantic_segmentation.py` 只消费 ExpertCatalog 明确批准的
`connected_components` capability：按 per-label confidence/area/morphology policy
将 semantic component centroid 转成共享 `GlobalPointObservation`，随后复用 owner-core
和 seam 去重形成 `CountingResult`。最终数量只由 accepted points 导出。该方式是
semantic instance approximation；相接对象会形成单一 component，可能低估数量，当前不做
watershed 或隐式 instance splitting。composition root 已按 `ExpertCatalog` 将 enabled
semantic expert 构造成 lazy client/backend，并接入固定优先级和 ordered fallback；OEM
当前仅因缺少 `connected_components` policy 保持 counting disabled，已确认的 class map
可独立供 VQA semantic-mask 使用。

## 14.4 图像工具（`models/images.py`）

`models/images.py` 除 EXIF 转正、RGB、MIME、data URL 与图片哈希外，提供纯内存
ROI 裁切接口：

```python
def crop_image_region(
    image: Path | Image.Image,
    box: Sequence[float],
    *,
    coordinate_frame: Literal[
        "normalized_0_1_top_left",
        "normalized_0_999_top_left",
    ],
    halo_ratio: float = 0.0,
) -> Image.Image:
```

- 直接消费 Qwen 输出的 xyxy 坐标，在内存中裁切：不写文件、不缩放图片、不接入
  Agent 或 Workflow。
- 输入为 `Path` 或 `PIL.Image.Image`，两者都先应用 EXIF orientation 并转换为
  RGB；不修改原图片对象，返回独立的内存 RGB 图像。
- 坐标制式（左上角原点的 xyxy）：
  - `normalized_0_1_top_left`：`[0, 1]` 归一化坐标（14B 冻结制式），图片边缘对应 `1.0`；
  - `normalized_0_999_top_left`：`[0, 999]` 归一化坐标（14A 制式），图片边缘对应 `999.0`。
- 像素映射与舍入规则（Pillow 半开像素边界）：
  - 左上边界向下取整（floor）；
  - 右下边界向上取整（ceil）；
  - 像素边界裁剪到原图范围；
  - `[0, 0, 1, 1]` 与 `[0, 0, 999, 999]` 均精确返回整张原图。
- `halo_ratio` 默认 `0.0`：按映射后 ROI 的宽高向四边扩张对应比例（`0.10` 即
  14B 默认上下文扩张），扩张结果裁剪到原图范围。
- 严格拒绝：坐标数量错误、非有限值（NaN/Infinity）、越界值、退化/反向框、
  未知坐标系，以及非法 `halo_ratio`（负数或非有限值）。

---

# 15. Qwen Settings

关键字段：

```text
model
cache_model_id
max_tokens
dtype
device_map
use_kernels
allow_download
min_pixels
max_pixels
revision
qwen_adapters
qwen_adapter_bindings
segformer_experts
```

默认：

```text
allow_download = False
```

如果 `model` 是本地 checkpoint 路径，则：

```text
cache_model_id
```

必须显式提供。

原因：缓存 hash、trace 与 RequestMeta 使用逻辑模型身份，不应泄露机器本地路径。

`qwen_adapters` 是受校验 catalog：每项声明物理 `path`、机器无关 `logical_id`、
精确 `adapter_model.safetensors` SHA-256 `revision` 与 `enabled`。`path` 只在模型加载
边界执行 `expanduser()`/项目根解析，不进入 cache identity、trace、prediction 或报告。
`qwen_adapter_bindings` 是固定六键结构：

```text
planner / counting / change / grounding / general_vqa / caption
```

每个值必须是显式 `base` 或已启用 catalog key；未知、disabled、缺失、不兼容、LFS pointer、
非 LoRA、非空 `modules_to_save`、旧 `visual_planner_roi_head` 或权重未完整消费均 fail closed。
adapter logical id、权重 revision 和 PEFT/client version 进入 bound client 的 cache identity，
因此只修改 binding 也会自然产生不同 request hash。

SegFormer 专家配置包含：

```text
model_path
logical_model_id
weights_filename
weights_sha256
classes_filename
processor_path
device
dtype
allow_download
revision
```

`model_path` 是物理路径；`logical_model_id` 是机器无关身份，两者不得混用。

---

# 16. DeepSeek Settings

声明字段：

```text
base_url
model
api_key_env
timeout_seconds
max_retries
```

settings 中不保存 API key value。

默认只声明：

```text
api_key_env = "DEEPSEEK_API_KEY"
```

实际值由 application composition root 获取后直接注入客户端。

---

# 17. Application Settings

`application.settings.AppSettings`：

```text
models
counting
runs
router
paths
backend
agents
visual_planning
```

### `VisualPlanningSettings`（`application.visual_planning`）

v5 视觉规划与可选证据能力配置组；新鲜规划没有 feature flag：

```text
planning_mode = "visual-task-plan-v5"
task_prompt_version = "v5"
catalog_version = "visual-evidence-catalog-v4"
preview_max_side = 1080
roi_coordinate_frame = "normalized_0_999_top_left"
roi_quantum = 1024
roi_materialization_policy = "longest-side-ceil-quantum-center-clip"
large_image_policy = "both-dimensions-strictly-greater-than-1024"
detectors: detector policy 映射；阈值全为 None = 未校准 = YOLO evidence 关闭
segmenters: 按稳定 binding 配置；默认 enabled=false，启用必须携带已验证 class map
```

- detector policy 为零条校准项时关闭 YOLO evidence；存在校准项时三个阈值必须
  完整，且当前实现只接受一个全局校准项；
- segmenter 是否可执行由 `enabled`、`class_map_version` 和已验证 runtime client
  三者共同决定，不能仅因 catalog 中存在 leaf 就宣布可执行；
- planner 的可执行类别是 catalog task capability 与本次 runtime 实际能力的
  task-specific 交集，不等同于 catalog 全量 leaves；
- catalog/prompt 版本绑定规划 prompt 与封闭证据目录为单一版本对，进入 request hash。

主要分组：

### `RunSettings`

```text
root
save_tiles
save_annotated_images
save_raw_responses
```

默认 run root：

```text
outputs/runs
```

### `RouterSettings`

```text
confidence_threshold
default_qwen_calls
default_deepseek_calls
fallback_on_partial
```

### `PathSettings`

```text
dataset_root
```

### `BackendSettings`

计数 backend 配置。

### `AgentsSettings`

领域 Agent 配置。

---

# 18. 配置加载

`load_settings(...)` 顺序：

```text
built-in defaults
  -> optional YAML
  -> supported environment overrides
```

环境变量优先。

当前普通 override 包括：

```text
QWEN_MODEL
SEGFORMER_ISAID_MODEL
SEGFORMER_OEM_MODEL
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL
DATASET_ROOT
OUTPUT_ROOT
```

Secret value 不通过这个普通 config merge 进入 AppSettings。

配置 snapshot：

```text
settings.safe_snapshot()
settings.to_config_payload()
```

保证 JSON-safe 与 secret-free。

注意：snapshot 保留主机路径语义用于复现/调试，不宣称可以无条件在另一台机器直接复用物理路径。

---

# 19. Prompt

`application.prompts.PromptCatalog` 在 runtime assembly 时加载版本化 prompt。

Prompt 是可复现输入的一部分。

run 创建时：

```text
prompts.snapshot/
```

保存 Prompt 副本，并在 `manifest.json` 保存 hash。

涉及模型输出行为的 Prompt 修改必须被视为行为变化，而不只是文案修改。

当前 active planner binding 为 `"visual_task_plan"` → 版本化
`visual_task_plan_v5.md`（v5）。旧 resolver/gate/joint prompt 不在 active
catalog 中；历史 prompt 文件不参与新鲜规划。

---

# 20. Agent 公共契约

## 20.1 `Agent`

```python
class Agent(Protocol):
    name: AgentName
    supported_tasks: frozenset[str]

    async def run(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> AgentExecution:
        ...
```

## 20.2 `AgentContext`

字段：

```text
artifact_dir
qwen_client
call_budget
data_root
judge_client
request_context
visual_task_plan     # v5 VisualTaskPlan；fresh Agent execution 必须有 materialized views
visual_views         # tuple[MaterializedVisualView, ...]
visual_bindings      # 轻量 evidence service bindings
```

fresh path 中，`visual_task_plan` 与 `visual_views` 由规划器物化后注入；direct 与
evidence 只消费同一组视图。历史 artifact 读取不进入 `AgentContext`，也不能作为
新鲜规划 fallback。

它是单样本轻量上下文。

不保存：

```text
API keys
Base64 image data
weights
full AppSettings
full PromptCatalog
```

## 20.3 `AgentExecution`

字段：

```text
agent_name
payload
result_filename
trace
additional_results
```

## 20.4 `AgentResult` 与 `VisualEvidence`

非计数 Agent 的统一结果字段为：

```text
agent_name
answer
boxes
evidence_items
geometry
status
```

`AgentResult` 不包含独立文本 `evidence` 字段。`evidence_items` 中的
`VisualEvidence` 只保存 `label`、恰好一种 `box`/`point`、可选 `image_id` 与
`coordinate_frame`，不公开 `confidence`。检测器或计数流水线内部的置信度仍属于
各自领域契约，不进入公共 `VisualEvidence`。

安全约束：

- result filename 是纯 basename；
- additional result filename 是纯 basename；
- JSON-safe；
- 无 sensitive key/value；
- payload agent_name 与 execution agent_name 一致。

---

# 21. 当前 Agent

## 21.1 GeneralVQAAgent

覆盖：

```text
general_vqa
scene_classification
multiple_choice_vqa
spatial_relation
```

`spatial_relation` 复用 general_vqa_v3 Prompt 与单次 Qwen 调用，输出
`agent_result.json`；保留 `requires_tiling=False` 与 VQA 确定性评测/Judge 族。

多选题 postprocess 会约束最终答案落在 choices 合法范围。

v5 视觉工作流：当 `VisualTaskPlan.needs_visual_assistance` 为 true 时，
GeneralVQAAgent 消费 `VqaEvidenceService` 产出的
`VqaEvidenceBundle`（executor 提供 bundle + preview 空间证据 + 内存调色表），
按冻结三分支协议（14.12.3）组装唯一一次 final Qwen 调用。逐 ROI 稳定输出：

```text
仅 YOLO      -> 标注 ROI
仅 SegFormer -> 纯色 mask + clean ROI
两者都有     -> YOLO-on-pure-mask + clean ROI
均无         -> clean ROI
```

有界流式物化（doc 26）：Agent 通过 `models.images` 的只读 region seam
（`ImageRegionSource`）逐框读取源图像，executor 的 YOLO tile 计划只保存轻量
几何记录，worker 在执行前才读取自己的 tile 框，提交窗口固定为
`max_tile_concurrency`（活跃物化 tile ≤ 并发上限），结果按稳定 index slot
归并、绝不按完成顺序；YOLO 路径全程不创建完整 ROI 裁切或提前物化的 tile
列表。SegFormer 在 1024×1024 model mask 上通过纯几何查找表直接采样
`<=1080` preview class grid（不恢复 W×H/Wp×Hp 的 class-id/boolean/RGBA
mask），叶子命中判定在 model mask 前缀矩形（旧恢复网格的精确来源）上完成，
与旧整分辨率判定逐点一致；最终纯色 mask 在 preview 空间合成。第一版 Pillow
backend 对 JPEG/PNG 仍是整图解码（非真实随机窗口 I/O），但已消除完整 ROI
副本、全部 tile 副本与全分辨率 mask 峰值。最终模型可见 PNG 与旧管线逐字节
一致，因此视觉内容版本、预处理身份与 request hash 均不变，旧 cache 继续
有效（26 §11.2 有证据决策）。

每个 ROI 的图像顺序固定为 mask first、clean ROI second；文本 payload 的
`visual_inputs` 以 `content_image_index`、`roi_id` 和角色描述每个 image block，
视觉内容协议版本为 `v2`。所有 final-Qwen 图像最长边超过 1080 才缩小、小图绝不
放大（掩膜 NEAREST、照片 LANCZOS）。YOLO 框为黑色 5px 外描边 + 品红 3px 内描边，
标签为黑底、品红边框、白色叶子文字，confidence 绝不写入。SegFormer 调色表按 catalog
segformer 叶子顺序确定性生成（与品红 ≥128、与黑底 ≥96、彼此 ≥48 的 RGB
距离约束，`sha256(leaf|attempt)` 重采样），仅存内存、绝不持久化。最终文本
payload 由同一个 task-aware `build_user_payload(sample)` 基础事实加嵌套
`evidence` 构成；`evidence` 只含实际 `visual_inputs`、最小 ROI identity/crop size、
requested/missing categories、ROI-local detections/segmentation hits 与 mask legend。
完整 VisualTaskPlan、source geometry、catalog/preprocessing/palette identity 和 detector
confidence 不给最终 Qwen。渲染图像摘要（包括 mask 与 clean ROI 的实际 PNG digest）、
source geometry 与 evidence protocol identity 仍共同覆盖 request hash（14.13）。两类图像
只以内存 PNG 传输，不新增磁盘 artifact。
`vqa_evidence.json` 作为 additional result 持久化 bundle（严格 JSON-safe，
含预处理 version、YOLO tiles、SegFormer `segformer_preprocess` 几何记录与
逐调用 call audit（SegFormer 按（ROI，binding），`tile_id=None`），无掩膜
数组/无 secret）。
`needs_visual_assistance == false` 或未注入计划时走 direct 路径。模型侧框统一使用
`0..999` 整数 `xyxy` JSON 表示；内部像素/ROI 浮点坐标只保留在确定性几何处理中。

GeneralVQA 基础 payload 不含 `answer_constraints`。非空 `semantic_subtype` 才进入
payload；`multiple_choice_vqa` 从 `TaskNormalization.choices/allow_multiple`
输出结构化答案约束，缺失选项会在读图、budget 与 Qwen 前稳定失败。
MME-RealWorld v2 同时在 canonical question 中保留人类可读的原始
选项行，使首次 Planner 调用也拥有完整题面；顶层字段仍是
Agent postprocess 的权威结构化来源。

Doc 24：GeneralVQAAgent 的四个 supported task（`general_vqa`、
`scene_classification`、`multiple_choice_vqa`、`spatial_relation`）统一由
`VisualTaskPlan.needs_visual_assistance` 决定 direct/evidence 路径，Agent 内部
不再存在 `sample.task == "general_vqa"` 的二次否决；`sample.task` 仍用于路由、
Prompt/answer constraint（选择题 postprocess）、结果语义与评测 dispatch。四
个 task 共享同一个 `general_vqa` catalog capability owner（`agents/schema.py`
的 `GENERAL_VQA_AGENT_TASKS` 单一来源），planner 的
`task_executable_categories` 按四个真实 task 分别列出同一份运行时可执行类别；
组合根（`application/bootstrap.py`）只计算一次 `_vqa_executable_leaves` 并注入
四个 task，VQA evidence 服务不可用时四个 task 的可执行类别一致为空。`counting`
/`fine_grained_counting` 仍由 CountingAgent 拥有，`grounding` 仍由
GroundingAgent 拥有，`caption`/`change_caption`/`change_qa` 不借此接入 VQA
evidence；routing fallback 不得改写 persisted resolved task 或评测 task。该
运行语义变化由 `vqa_assistance_scope = "general-vqa-agent-tasks-v1"` 冻结身份
保护（见 §Resume 的 VQA assistance scope）。

## 21.2 CaptionAgent

覆盖：

```text
caption
```

模型基础 payload 只包含 `task` 与原始 `question`；不发送坐标、box format、
空 constraints 或 null subtype。图像仍按 `sample.images` 稳定顺序传入，输出
schema 为 `AgentResult`。

## 21.3 GroundingAgent

覆盖：

```text
grounding
```

completed 结果需要合法定位证据。

v5 视觉工作流：当 `VisualTaskPlan.needs_visual_assistance` 为 true 时，
GroundingAgent 消费 `GroundingEvidenceService`：C6 executor 在内部完成
唯一一次 final Grounding Qwen 调用（evidence 管线在 Agent 外执行），
返回确定性整图框 `WholeImageBox`。`AgentResult.answer` 为
`, `.join(labels) 文本；final Qwen 绝不产出自由文本坐标（14C §8）。
`needs_visual_assistance == false` 或无计划时走 direct 路径。服务缺失/失败以
`grounding_evidence_failed:<CODE>` 稳定失败。最终 Grounding Qwen 接收和输出的
ROI 局部框统一为 `0..999` 整数 `xyxy` JSON，确定性后处理再转换为整图坐标。

direct 基础 payload 只包含 `task`、`question`、整图 coordinate frame 与 box
format。evidence final-Qwen 使用独立版本化 `grounding_final_v1` prompt；接收按
`content_image_index/roi_id` 显式绑定的 clean ROI，以及嵌套 `evidence` 中的最小
ROI records、candidate IDs/categories/ROI-local boxes 和 missing categories。
catalog version、source/core/expanded geometry 等审计与回映事实不对模型可见，
但继续进入 request hash 和 `grounding_evidence.json`。最终模型输出严格匹配
`GroundingQwenResponse`，整图回映仍由确定性生产代码完成。

## 21.3.1 VQA Agent SFT v2

`data/2026-08-24_vqa-agent-io/` 使用 `vqa-agent-sft-v2`。每条记录分离保存
`visual_task_plan`、完整 `UnifiedSample`、生产 builder 生成的
`base_user_payload`、`AgentResult` 与 supervision。evidence path 的动态 ROI、
检测结果和完整 multimodal messages 不伪装成基础 payload。转换与校验入口是
`scripts/migrate_vqa_agent_sft_v2.py`；它原子写入并校验 sample ID、question、
答案、split 与图片映射不变，同时重算 manifest SHA256。

`scripts/finetune_qwen35_9b_general_vqa_agent_lora.py` 的 preparation v2 继续运行
生产 evidence path，并监督 `AgentResult` 的稳定 JSON 结构、`answer`、`agent_name`、
`status`，以及 executor 已按 VisualTaskPlan 叶子类别、阈值、NMS 和跨 ROI 去重筛选的
YOLO 证据。监督框使用整图 `normalized_0_999_top_left` 坐标并保留 `image_id`；ROI
局部框只作为 final-Qwen 输入证据。SegFormer 命中仍只作为输入 mask/存在性证据，
不得转框；`geometry` 的值由运行时校验/修复拥有，不进入 causal-LM loss。该变更只调整
assistant token labels，不改变 Qwen、视觉编码器或 LoRA 模块结构。

VQA evidence 的生产配置将 YOLO ONNX 与 SegFormer 放入两个使用 `spawn` 的独立
GPU worker；PIL/CPU 输入和协议输出通过 IPC 传递，CUDA tensor 不跨进程。YOLO
CUDA EP 使用显式 `gpu_mem_limit` 与 `kSameAsRequested` arena；两个 worker 分别按
PID 使用 `nvidia-smi` 监控。默认生产阈值为 YOLO `6/8 GiB`、SegFormer
`10/12 GiB`（soft/hard），目标 GPU 空闲显存低于 `8 GiB` 也触发保护。soft
阈值保留已完成结果并在下次调用前重建；hard 阈值或 allocation failure 终止对应
worker，当前调用最多重试一次。任何 evidence worker 回收均不得调用 Qwen 主进程的
CUDA allocator，也不得重建或 offload Qwen。
VQA LoRA 脚本在 preparation 产物原子持久化完成后显式关闭全部 evidence GPU
worker，再加载 Qwen 权重进入优化阶段，避免两个阶段的 CUDA context 重叠驻留。

## 21.4 CountingAgent

覆盖：

```text
counting
fine_grained_counting
```

是当前最复杂的领域 Agent，内部包含独立 backend planning/execution 与 tile/seam 处理。

## 21.5 ChangeAgent

覆盖：

```text
change_caption
change_qa
```

处理有序时相图对，完整流程为：

```text
UnifiedSample
 -> PairValidator
 -> PairRegistrar
 -> PairHarmonizer
 -> ChangePerceptionPipeline
 -> proposal publisher
 -> ChangeAgent evidence builder
 -> Qwen
 -> review_result
```

PairValidator 只确认 temporal roles、文件与 decode 等结构条件；尺寸不同的、但
结构有效的 pair 进入 registration，而不是直接判 invalid。PairRegistrar 只做保守
的 identity/similarity/affine/homography 全局几何配准并经过 quality gate；它不使用
dense optical flow、TPS 或 elastic warp。PairHarmonizer 只做 PIF/LAB 辐射一致化，
不得重新估计几何变换。默认关闭 semantic auxiliary path，仍可退化到 legacy
deterministic perception；启用时由 bootstrap 注入抽象 `DenseSemanticClient` 或
`DenseSemanticPyramidClient`，SegFormer 同时服务 semantic probabilities 与
intermediate/pyramid features。最终 proposal 由 low-level、feature residual、
semantic difference 和 reliability-aware fusion 确定性地产生。

proposal coordinate frame 永远是 T1 reference canvas，normalized box 使用
`0..999` 的 top-left 坐标。所有 dense difference、threshold 统计和 proposal mask
都必须先受 `registration_valid_mask` 约束；无效 warp border 不得进入 PIF、semantic
score 或 proposal。每个 proposal 的 semantic transition 是辅助模型证据，不是
ground truth。

Qwen 是 proposal-driven semantic confirmer：raw full T1/T2 始终是最终语义 authority，
registered/harmonized 图和 mask 仅是辅助证据。raw T2 不能在未做 inverse transform
时伪造为 T1 坐标 crop。每个 proposal 发布三路 score、有效权重、可靠性、semantic
transition、二值 mask、crop-local mask/overlay 与相对路径清单；trace 记录实际
evidence roles/count、registration quality、阈值来源、fallback、SegFormer logical
identity、verified weight SHA 与算法设置，绝不记录 checkpoint 绝对路径或完整
feature/probability tensor。Change runtime prompt 由 bootstrap 从 `PromptCatalog`
注入，运行时文本、版本、request hash 与 run prompt snapshot 共享同一权威资产。

2026-08-10 NVIDIA GB10 离线校准后，semantic path 的基准配置为 MiT-B2 iSAID、
`feature_stage=1`（stride 8）、`tile_size=768`、`tile_overlap=64`、
`local_match_radius=1`、`pif_threshold_k=4.5`，三路融合权重保持 `0.25/0.50/0.25`。
Change V3 额外支持显式的 `feature_stages` pyramid 配置；旧的单 stage 配置仍兼容。
该选择来自既有离线校准，不是通用 benchmark 结论。`semantic.enabled` 仍默认关闭，
live 校准不会强制所有运行加载 SegFormer。

---

# 22. Task → Route Policy

当前 `routing/policies.py`：

| Task | Primary Agent | Fallback | `requires_tiling` |
|---|---|---|---:|
| `counting` | `counting_agent` | — | yes |
| `fine_grained_counting` | `counting_agent` | — | yes |
| `change_caption` | `change_agent` | — | yes |
| `change_qa` | `change_agent` | `general_vqa_agent` | yes |
| `grounding` | `grounding_agent` | — | yes |
| `spatial_relation` | `general_vqa_agent` | — | no |
| `scene_classification` | `general_vqa_agent` | — | no |
| `general_vqa` | `general_vqa_agent` | — | no |
| `caption` | `caption_agent` | — | no |
| `multiple_choice_vqa` | `general_vqa_agent` | — | no |

`change_qa` 的 routing fallback 会在执行边界显式重建为
`general_vqa/general_vqa_agent`：原始 resolved task 仍为 `change_qa`，但
`execution_task`、执行 sample 的 task 与确定性评测族均为 `general_vqa`；原始
`UnifiedSample` 不被修改。

Router 未知 task：

```text
explicit failure
```

而不是：

```text
guess general_vqa
```

---

# 23. Historical TaskResolver（只读迁移参考，非当前实现）

`TaskResolver` 仅是历史 workflow 服务；当前仓库不再提供该实现或调用入口。
doc 17 已用 `VisualTaskPlanner` 统一替代其模型调用；本节只解释旧 run 与迁移
fixture 的行为，不能据此新增 active resolver 路径。

它回答：

```text
What task is this request/sample?
```

而 Router 回答：

```text
Which Agent handles this known task?
```

## 23.1 Resolver 三路径

```text
explicit
rule
model
```

### Explicit

合法 explicit task：

```text
confidence=1.0
source=explicit
zero model call
```

非法 explicit task：

```text
UNKNOWN_EXPLICIT_TASK
```

### Rule

空 question：

```text
1 image -> caption
2 images -> change_caption
otherwise -> EMPTY_UNRESOLVABLE_REQUEST
```

### Model

只有：

```text
no explicit task
AND question non-empty
```

时进入模型解析。

模型返回：

```text
task
confidence
candidate_tasks <= 3
reason_codes
```

必须通过合法 `TaskName` Schema。

模型解析需要完整 model cache identity 和 request hash。

doc 16 的 fresh dataset 与 manual ask 不走本历史解析路径：单次
`VisualTaskPlanner` 调用同时产出 task 与视觉辅助意图，模型选定 task 对
routing/materialization/execution 权威，源 task 只做审计。历史 v2 计划曾将低置信度
作为稳定失败；当前 v5 不输出或评估 planner confidence，也不回退到另一个规划模型。

---

# 24. Historical low-confidence candidate fallback

旧 TaskResolver 模型 task resolution 低于阈值时：

- task 本身保持第一候选；
- 最多 3 个候选；
- 如果 top task 不是 `general_vqa`，候选表为 `general_vqa` 保留 fallback 槽；
- Resolver 只返回结构化候选；
- 真正执行由 SampleRunner 完成。

该节只解释旧 run 的候选审计；doc 16 历史 v2 `VisualTaskPlanner` 的低置信度曾直接
稳定失败，当前 v5 不再输出该分数，也不启动旧候选路径。候选 fallback 不等于“多
Agent 全跑”。

SampleRunner 会按 AgentName 稳定去重，避免多个 candidate task 实际映射同一 Agent 时重复执行。

---

# 25. Counting 子系统

目录：

```text
agents/counting/
```

主要分层：

```text
schema
settings
geometry
evidence
point_pipeline
target_parser
agent
executor
backends/
```

## 25.1 `CountingResult`

CountingAgent 的主结果契约。

最终计数必须与 accepted evidence 一致。

主文件：

```text
counting_result.json
```

## 25.2 Backend

当前显式 backend kind：

```text
qwen_point
quantity_proposal
semantic_segmentation
yolo_obb
```

### `qwen_point`

基于 Qwen 的 tile pointing/counting。

### `quantity_proposal`

提供数量 proposal + grounded localization point 的路径，不等同 detector。它已由
composition root 注册；定位点进入共享 accepted-point truth。

### `semantic_segmentation`

消费 verified semantic class map，并只对 catalog 明确批准为
`connected_components` 的 target 计数。component centroid 进入共享 point pipeline。
这是 semantic instance approximation，不是 instance segmentation；相接对象可能合并成
一个 component 而低估数量。

### `yolo_obb`

面向 OBB detector 的计数路径。

## 25.3 Backend selector 与 executor

职责分离：

```text
BackendSelector
    plan

CountingPlanExecutor
    execute
    unavailable/runtime fallback
    zero review
```

Selector 不负责吞掉实际 runtime error。

固定 kind 优先级是：

```text
yolo_obb > semantic_segmentation > quantity_proposal > qwen_point
```

同 kind 才按 `priority DESC, backend_name ASC` 排序。`BackendPlan` 保存 primary 与完整
ordered fallback chain；executor 逐个尝试并记录 backend、kind、reason code 与 error type。
invalid kind/contract 和最终 Qwen failure 是 terminal。合法 `final_count == 0` 不是失败，
只有显式 zero-review policy 才能复核；复核失败保留原零并记录 warning。

通用执行策略归属 `CountingSettings`：

```text
fallback_on_backend_unavailable
fallback_on_backend_error
verify_empty_detection
verify_empty_semantic
trust_empty_detection
```

旧 YOLO-scoped key 只在 settings load boundary 一次性迁移；同时声明新旧 key 会拒绝。
Detection zero review 使用 ordered chain 中下一位实际支持的专家，trace 记录真实 reviewer，
不再假设 reviewer 必然是 Qwen。

## 25.4 Deterministic count target resolver

fresh counting 的语义 target 来自 VisualTaskPlanner v5；normalization 与 legacy
metadata hint 只作为确定性 verifier。无 plan 的 direct 兼容边界为：

```text
structured normalization.count_target_hint -> normalization_explicit_hint
legacy metadata count_target_hint           -> legacy_direct_hint
no plan + no hint                           -> COUNT_TARGET_SOURCE_REQUIRED
```

invalid hint：

```text
COUNT_TARGET_VERIFIER_INVALID
```

不得静默忽略。

resolver 不调用模型、cache 或预算。解析后的 target 只和 canonical label、显式 aliases、
parent expansion 与 dataset-neutral hints 匹配；planner/VLM 不返回 backend/checkpoint 决策。

## 25.5 YOLO

YOLO 模型存储、Ultralytics adapter 与兼容保留的 ONNX 实现保持惰性和可选。

Python `YoloCountingSettings` 只保留 schema 与通用默认值：默认 `enabled=false`、
`detectors=[]`，不包含具体 checkpoint、labels 或 backend id。`ExpertCatalog` 是 capability、
logical identity、SHA、labels 与 priority 的事实源；`configs/local.yaml` 等部署配置是 enabled、
物理权重路径、provider/device 与阈值的事实源。只有显式部署 inventory 才注册 YOLO。
相对权重路径在 composition root 按 `project_root` canonicalize，绝对外部挂载路径保持不变；
物理路径不参与 catalog identity。权重与可选 runtime 依赖不自动下载，缺失时仍按通用
fallback 策略进入下一专家。当前正式 DOTA/VRSBench inventory 均使用 Ultralytics；
ONNX adapter 仅保留给显式配置的兼容 detector。

硬件策略由 settings 决定，例如：

```text
require_cuda
allow_cpu_fallback
```

不能仅因为 import 包就要求本地一定安装全部 YOLO runtime。

## 25.6 Point、small-object 与 seam 稳定契约

所有 backend 保持：

```text
final_count == sum(point.accepted for point in global_points)
```

tile 使用 core + halo；halo 只提供上下文，centroid/point 落 owner core 才归属该 tile。
small-object 策略来自 catalog hints，可启用 minimum scan depth、empty-tile second pass 和
保持宽高比的 optional upscale，不读取 dataset 名。seam 先做 deterministic strong/clear
判定，仅 ambiguous band 调用可选视觉 reviewer；`uncertain`、异常或预算耗尽均不合并并留下
unresolved warning。YOLO OBB 自身的 overlap 去重不会再被 point seam 重复处理。

## 25.7 ExpertCatalog、资产与 trace

`ExpertCatalog` 是 routing capability truth，记录 canonical target、aliases、neutral hints、
dataset-neutral backend name、显式 kind、priority、logical model id、project-relative assets、
verified class map、model labels 与 counting mode。YOLO settings 只负责 inference/runtime
参数，bootstrap 在注册前校验 backend name、logical model id、SHA256、priority 与 labels
全部和 catalog 一致。catalog 通过 immutable `experts(...)` 公共 API 向 composition root
枚举声明，bootstrap 不访问其私有存储。

`vehicle` 的固定链为 Detection → SegFormer → QuantityProposal → QwenPoint；`aircraft`
为 Detection → SegFormer → QwenPoint。semantic composite capability 完全由 catalog 的多
`model_labels` 声明，backend 逐 label 独立提取 connected components，禁止先 union mask。
QuantityProposal 一旦进入 grounded localization，就用可解析的 localizer answer 与 accepted
points 判断完成性；原 proposal 不一致保留 warning，但不再把自洽定位结果误标为 partial。

SegFormer/YOLO 权重默认只来自本地 Git LFS 或外部资产，不自动下载；loader 校验 SHA256。
SegFormer runtime profile 可以覆盖 physical `model_path`、device 与 dtype，但不能覆盖 catalog
logical identity、expected SHA 或 verified class-map semantics。
公共 trace 不保存 mask、tensor、prompt、base64、secret 或绝对路径，并至少记录 canonical
target、候选/尝试/final backend、fallback history、counting mode 与 accepted count。

wheel 通过 package-data 携带 expert catalog、`agents/evidence_catalog.json`、verified
SegFormer 小型 metadata 与 prompts，不携带 `.safetensors`、`.onnx` 或 `.pt` 大权重。
Prompt root 的顺序是 explicit override → `project_root/prompts` → installed package
`prompts`；显式错误配置 fail closed，公开 composition error 不包含绝对路径。CI 使用真实
wheel 在源码树外组装 runtime，验证 catalog/prompt metadata、QuantityProposal、lazy
SegFormer 与不加载权重的 YOLO 注册。

---

# 26. Spatial 子系统（已移除）

`agents/spatial/` 已从新架构删除，`spatial_relation` 由 `general_vqa_agent`
接管（见第 22 节 Task → Route Policy）。不再存在 SpatialAgent、candidate
review、evidence merge 或 deterministic geometry 专用实现。

`spatial_relation` 仍保留为公开 task，`TaskNormalization.spatial_query` 与
VRSBench normalization 规则不变，评测仍走 VQA 确定性族与可选 Judge。执行
语义变化：原先最多两次 Qwen 调用（候选 + review）并可能做确定性几何改写，
现为单次 general_vqa_v3 Prompt 调用，输出 `agent_result.json`；历史空间 run
结果不直接可比。

---

# 27. Change 子系统

目录：

```text
agents/change/
```

主要组件：

```text
pair_validator
registration
registration_quality
harmonizer
difference_proposal
feature_residual
semantic_difference
proposal_fusion
perception
semantic_transition
preprocess
reviewer
agent
```

变化任务输入必须是合法时相对：

```text
t1
t2
[context...]
```

无效 pair 应尽可能在模型调用前稳定失败。

cv2/numpy 等视觉依赖保持可选边界，不应使基础 import 强制依赖所有变化检测扩展环境。

---

# 28. Composition Root

`application/bootstrap.py` 是具体 runtime assembly 的唯一位置。

一次 assemble：

```text
PromptCatalog
Qwen3.5 multi-adapter engine（或显式测试注入 client）
planner bound client
per-Agent bound clients
ExpertCatalog
lazy SegFormer clients by logical model id
Counting backend registry
AgentRegistry
TaskRouter
VisualTaskPlanner
VisualTaskPlan / MaterializedVisualView
VisualPlanBindings
DeepSeekJudgeClient (optional)
JudgeService
ArtifactWriter
CallBudgetFactory
RunStore
SampleRunner factory
DatasetRunner factory
Reporting functions
```

## 28.1 Qwen

如果没有测试注入 client：

```python
create_model("qwen3_5_multi_adapter", ...)
```

在这里创建一次 engine。VisualTaskPlanner 固定使用 `planner` binding；Counting、Change、
Grounding、GeneralVQA、Caption 及其嵌套 Qwen backend/evidence service 使用各自 binding。
首期本地配置六个 binding 指向同一 supplement adapter；以后只改 catalog/binding 配置即可
切换单个 Agent，无需修改 Agent 或 Router。`RuntimeComponents.qwen_client` 仅保留为 planner
兼容 seam，fresh Agent 接线使用 `qwen_clients[agent_name]`。

## 28.2 DeepSeek

只有提供 `api_key` 时创建 Judge client。

无 key：

```text
judge_client = None
```

系统仍可以执行 deterministic-only 评测。

## 28.3 Agent Registry

当前稳定注册顺序包含：

```text
CountingAgent
ChangeAgent
GroundingAgent
GeneralVQAAgent
CaptionAgent
```

组装后会校验全部 routable task 是否被覆盖。

---

# 29. Call Budget

单样本调用预算通过 `CallBudget` 协议传入 Agent。

至少区分：

```text
reserve_qwen()
reserve_deepseek()
```

每条 fresh manual/dataset 样本的 VisualTaskPlanner 调用消费一次 Qwen budget；规划后
的 Agent/final-Qwen 与可选 Judge 继续共享同一样本预算语义。显式 task 也不跳过规划。

SampleRunner 的多个 attempt 与逐样本 judge 共享样本预算语义，不能让 fallback 无限放大模型调用数。

---

# 30. RunStore

默认 run root：

```text
outputs/runs
```

创建 run 时不构造模型，也不调用模型。

run 创建产物：

```text
manifest.json
config.snapshot.json
prompts.snapshot/
events.jsonl
```

随后具体运行写：

```text
run_request.json
```

## 30.1 `manifest.json`

`RunManifest`：

```text
run_id
created_at
git_commit
git_dirty
config_hash
prompt_hashes
model_ids
dataset
split
sample_filter
```

其中 `git_dirty` 是环境/执行 provenance，不应被误认为稳定 functional parity 字段。

## 30.2 Run ID

显式 run id 需要跨平台安全：

- 非空；
- 无 `/` / `\`;
- 非绝对路径；
- 非 `.` / `..`;
- 无控制字符；
- 不是 Windows reserved device name；
- 长度与字符集受限。

fresh run 指定已存在 id：

```text
fail
```

而不是覆盖。

---

# 31. `RunRequest`

`workflows.schema.RunRequest` 表示一次运行的**具体调用身份**。

Dataset run 关键字段：

```text
dataset
dataset_root
split
task_mode
tasks
auto_task
sample_ids
limit
start_index
shard_index
shard_count
sample_concurrency
evaluate
judge_policy
judge_sample_rate
render_errors
fail_fast
```

task mode：

```text
explicit
adapter_default
auto
```

Count-image 还会保存与该调用相关的冻结身份/行为参数。

`run_request.json` 与 `manifest.json` 职责不同：

```text
manifest.json
    reproducibility / identity metadata

config.snapshot.json
    application config snapshot

run_request.json
    actual user/runtime invocation
    + path-free Qwen base/adapter catalog/binding identity
```

resume 重建实际调用时以 `run_request.json` 为准。

---

# 32. DatasetRunOptions

运行时定型对象：

```text
dataset
root
split
tasks
run_id
resume
limit
start_index
shard_index
shard_count
sample_concurrency
sample_ids
evaluate
judge_policy
judge_sample_rate
render_errors
fail_fast
auto_task
```

三种任务模式：

### Adapter default

```text
tasks = None
auto_task = False
```

运行 adapter.supported_tasks。

每条 fresh sample 仍由统一 VisualTaskPlanner 规划一次；adapter default 本身不
改变 task 集合，也不把 task 选择交给 Router。

### Explicit

```text
tasks = ("...", ...)
auto_task = False
```

### Auto

```text
tasks = ()
auto_task = True
```

走 SampleDraft → VisualTaskPlanner → materialize_sample → UnifiedSample。

---

# 33. SampleRunStatus

字段：

```text
sample_id
task
state
error_code
error_message
result_path
updated_at
```

状态：

```text
pending
running
succeeded
partial
failed
skipped
```

`task` 可以是公开 task，或 pre-task draft 失败时的诚实：

```text
unknown
```

不得为失败样本虚构一个已知 task。

---

# 34. Sample 状态中的 task 含义

三个 task 概念必须区分。

### Resolved task

VisualTaskPlanner / canonical sample 决定的任务。

保存在：

```text
sample.json.task
agent_trace.resolved_task
```

### Execution task

最终真正成功/失败执行的 attempt task。

保存在：

```text
status.json.task
agent_trace.execution_task
routing_decision
evaluation task semantics
```

### Run task

DatasetRunner 的 task namespace：

```text
predictions.jsonl.run_task
tasks/<run_task>/
```

auto-task run 的 run namespace 可以是：

```text
auto
```

这三者不能混为一谈。

---

# 35. SampleRunner

单样本执行内核。

概念流程：

```text
persist sample
persist running status
visual-only planning (one call)
materialize/rebuild task
deterministic TaskRouter
build routing/attempt plan
run Agent
persist result
build deterministic evaluation
optional Judge
persist trace
persist final status
append execution index
```

所有 fresh 样本都由 `VisualTaskPlanner` 先消费一次共享 Qwen budget，产出
`visual-task-plan-v5`，再由纯转换生成 routing decision。planner 失败稳定写入
`state=failed`，不重试、不 legacy 回退；模型 task 在物化后成为执行 task。

`visual_task_plan.json` 保存已校验计划与 `MaterializedVisualView` 几何；
direct、General VQA evidence 与 Grounding evidence 都消费同一视图。历史
`visual_plan.json` / `joint_visual_plan.json` 只由 reporting 只读展示。

职责：

- 单样本；
- routing attempt plan；
- routing fallback；
- partial fallback policy；
- 调用预算；
- deterministic evaluation dispatch；
- optional VQA judge；
- trace；
- durable status。

不负责遍历整个数据集。

---

# 36. DatasetRunner

数据集编排层。

职责：

```text
adapter probe
task-mode handling
selection
start index
sharding
sample-id filtering
limit
resume
sample concurrency
fail-fast
SampleRunner orchestration
predictions index
dataset summary
```

## 36.1 Selection

稳定选择语义按现有实现保持：

```text
adapter stable order
-> start_index
-> shard
-> sample_ids
-> limit
```

shard 使用稳定 SHA-256 逻辑，不能替换为 Python `hash()`。

## 36.2 并发

当前明确支持：

```text
single-process asyncio concurrency
```

JSONL writer 当前只承诺同一 Python 进程内的并发写入安全。

不宣称多进程同时追加同一 run 文件安全。

## 36.3 Fail-fast

fail-fast 后：

- 不再提交新的样本；
- 已启动任务被正确 cancel/await；
- 已选择样本最终仍需要有终态记录；
- summary 计数闭合。

## 36.4 v3 visual-only planning

DatasetRunner 在 resume 检查之后、任何执行之前，对每条 fresh 样本先做一次
视觉规划 Qwen 调用（预览图 + 原始文本 → v3 task plan），再以模型选定 task
物化/路由/执行。三条入口统一接线：

```text
tasks=(...) 显式任务    -> _run_sample_visual：重建成选定 task 后执行
tasks=None   默认任务    -> 同显式任务（重建成选定 task 后执行）
auto_task=True + tasks=() -> _run_draft_visual：SampleDraft 直接走 planner
```

旧 resolver/gate/joint planner 不再参与 fresh 路径；重建成选定 task 失败或所选
task 与样本图像数不兼容时稳定失败。规划调用与后续 Agent 共享同一样本
`CallBudget`；
resume 时 succeeded 样本零模型调用，只补缺失/损坏的确定性评测产物
（§36.5 补评测）。summary 计数闭合规则不变。

## 36.5 补评测 / evaluation supplement

resume 补评测只补缺失或损坏的确定性 EvaluationRecord 产物
（`counting_evaluation.json` / `vqa_evaluation.json` /
`grounding_evaluation.json` / `caption_evaluation.json`，按 `status.task`
的运行时任务族映射）。补评测异常降级为 skipped，绝不重跑 Agent 或覆盖
已有记录。

---

# 37. ArtifactWriter

`workflows/artifact_writer.py` 集中拥有数据集执行产物文件名。

当前固定名：

```text
sample.json
status.json
routing_decision.json
agent_result.json
counting_result.json
counting_attempts.json       # counting only; ordered backend audit
agent_trace.json
visual_task_plan.json        # current versioned plan + materialized view geometry
predictions.jsonl
dataset_summary.json
dataset_probe.json
```

`visual_task_plan.json` 只存已验证 `VisualTaskPlan` 与
`MaterializedVisualView` 几何（原子写入），绝不存原始模型正文。对象证据路径的证据 bundle 作为 Agent additional
result 持久化（`vqa_evidence.json` / grounding candidate JSON），严格
JSON-safe；clean ROI 图像与 SegFormer 纯色 mask 目前仅内存传输，持久化
格式/质量参数未批准（14A2 §5.1 语义占位）。ROI 图像文件、目录结构等
后续阶段冻结前不采用。

Evaluation filename 由 evaluation dispatch 按任务声明，但同样必须是安全纯 basename。

所有结构化 JSON 写入走原子写入。

---

# 38. Run 目录布局

典型 dataset run：

```text
outputs/runs/<run_id>/
├── manifest.json
├── config.snapshot.json
├── run_request.json
├── prompts.snapshot/
├── events.jsonl
├── predictions.jsonl
├── report/
│   ├── report.html
│   ├── report.json
│   ├── samples.csv
│   ├── samples.jsonl
│   ├── metadata.json
│   ├── deepseek_audit.jsonl
│   └── external_standard.json      # when provided
└── tasks/
    └── <run_task>/
        ├── dataset_probe.json
        ├── dataset_summary.json
        └── samples/
            └── <storage_key>/
                ├── sample.json
                ├── status.json
                ├── routing_decision.json
                ├── agent_result.json
                │   or counting_result.json
                ├── counting_attempts.json    # new counting runs only
                ├── visual_task_plan.json     # current versioned plan + exact view geometry
                ├── <task>_evaluation.json
                ├── agent_trace.json
                └── optional model/judge artifacts
```

当前 v5 证据路径（规划请求视觉辅助时）可能另存 additional result：

```text
vqa_evidence.json     # object_evidence_vqa bundle (JSON-safe)
```

实际 evaluation filename 由共享 dispatch 决定，当前主要包括：

```text
vqa_evaluation.json
counting_evaluation.json
grounding_evaluation.json
caption_evaluation.json
```

不是所有 task 都一定有逐样本 deterministic evaluation。

---

# 39. Sample storage key

sample 目录不直接使用未经处理的 sample id。

当前 DatasetRunner/reporting 使用冻结身份推导安全 storage key，从而避免：

- Windows reserved path；
- 非法字符；
- 多 task 同 sample id 冲突；
- 任意 result path 注入。

Reporting 应重新由：

```text
(run_task, sample_id)
```

推导 sample 目录，而不是相信 `predictions.result_path` 去读任意位置。

---

# 40. `status.result_path`

`SampleRunStatus.result_path` 是：

```text
sample-relative plain basename
```

例：

```text
agent_result.json
counting_result.json
```

禁止：

```text
C:\...
/home/...
../...
a/b.json
\\server\...
```

旧绝对路径 status 无法通过当前 schema 时，resume 应把状态视为无效并按契约处理，而不是继续信任。

---

# 41. `predictions.jsonl`

每行：

```text
sample_id
run_task
task
status
result_path
updated_at
```

它是 append-only execution index。

当前状态的定义：

```text
last row for (run_task, sample_id)
```

它不是完整 SampleResult 数据库，也不是唯一 artifact 真相。

报告会用它确定当前行，再读取安全 sample directory 中的持久化 artifact。

---

# 42. Resume

resume 是持久化契约的重要组成。

原则：

### Succeeded

默认不重复推理。

如果缺确定性评测或允许补 Judge，可按共享逻辑补充。

推理期产物（`visual_task_plan.json`、证据 JSON、ROI 视图记录）属于推理期
产物，绝不进入补判范围——succeeded 样本保持原样（不重跑、不修复），即使
损坏或缺失也只可能通过后续 explicit rerun 契约处理。resume 判断 metric family
只用 `status.task`（实际执行任务），计划字段绝不覆盖指标族。

### Partial / Failed / Running / Pending

按现有 DatasetRunner 规则重新进入 SampleRunner。

### Missing / Corrupt status

不信任旧状态，重新执行或稳定失败。

### Evaluation dispatch

resume 判断 metric family 使用：

```text
status.task
```

即实际 execution task。

不能只看：

```text
sample.json.task
```

否则 candidate fallback 后会写错指标族。

### Invocation

resume 具体原始参数以：

```text
run_request.json
```

为权威。

新的 CLI 默认值或 config 漂移不得静默改变原 run 行为。

### Qwen adapter identity（doc 25）

新 run 在 `manifest.model_ids`、`config.snapshot.json` 与 `run_request.qwen_runtime_identity`
冻结 base logical id/revision、adapter logical id/weights digest、六个完整 binding 和 client
version。`run_request` 只含逻辑身份，不含 adapter 物理路径。resume 重跑前要求当前已组装
identity 与持久化 identity 完全一致；冲突稳定拒绝，绝不换用当前默认 adapter。Doc 25 之前
缺少该字段的 run 明确解释为 legacy base-only，不猜成当前 adapter。已经 succeeded 且可直接
复用的 count-image 结果保持零模型调用。

### VQA evidence 预处理身份（14.14 / doc 26）

`evidence_preprocessing` 冻结完整算法组合身份（26 §5.3），两个互斥版本：

```text
greedy-1024-stretch-v1
    YOLO: greedy 1024 tiles + remainder stretch
    SegFormer: greedy 1024 tiles + remainder stretch（仅历史只读解释）

yolo-v1-segformer-pad-v1（fresh 默认）
    YOLO: greedy-1024-stretch-v1（不变）
    SegFormer: pad-multiple-1024-resize-square-v1
```

- 新鲜运行由 application 显式写入 `yolo-v1-segformer-pad-v1` 并携带全部 v2
  必填字段（含 `yolo_version` / `segformer_version` 等 backend-specific
  字段）；v2 对象缺少任一字段解析失败，schema 默认值绝不补齐；
- 旧 v1 JSON 缺少所有 v2 字段时仍是 v1，绝不自动升级为 pad 协议；同一版本
  字符串不得代表两种算法；
- 两个版本的 succeeded 样本 resume 均零模型调用，绝不修复或重写推理期证据
  产物（含 `vqa_evidence.json`）；
- 非终态/明确重跑样本使用当前（新）协议；调用方显式提供的不同身份（含
  篡改的 tile policy）以 `resume evidence preprocessing mismatch` 稳定拒绝，
  v1/v2 双向冲突都在任何模型调用前失败；
- 历史无身份（None）运行的 VQA evidence 非成功重跑以
  `LEGACY_VQA_EVIDENCE_PREPROCESSING_UNSUPPORTED` 稳定失败，绝不静默切换
  成新语义；历史 succeeded 样本 resume 仅补评测、零模型调用。

执行器内部实现版本（doc 26 rollout）：`bounded-streaming-v1` 是有界流式
物化（region seam 逐框读取 + 固定提交窗口 + preview 空间 SegFormer 恢复）
的执行器实现身份，**不改变**上表两个组合版本字符串。parity 测试（含
end-to-end mask PNG 字节比较）证明最终模型可见像素逐字节不变，因此视觉
内容版本保持 `v2`、预处理身份与 request hash 均不变，旧 cache 继续有效；
无手工保留旧 identity 制造 cache hit 的行为。回滚时不得混用新旧 cache
identity（26 §15）。

### VQA assistance scope（doc 24）

`vqa_assistance_scope` 是独立于 tile 预处理身份的第二个冻结运行身份，值为
`general-vqa-agent-tasks-v1`：它冻结“哪些 task 可消费 GeneralVQAAgent 的
共享 evidence 开关”（四个 GeneralVQAAgent task）。规则：

- 新鲜运行把 scope 写入 planner `planning_parameters`（进而进入 system
  prompt 绑定与 prompt snapshot）、`run_request.json` 与手动 ask 的
  `request.json`；planner identity 比较覆盖该字段；
- 历史 run request 缺该字段时解析为 None，绝不从当前默认值回填；
- 历史 succeeded VQA 样本 resume 仍只补缺失的确定性评测/Judge/report，零
  模型调用；
- 历史非终态 VQA 样本需要重新规划/重跑 evidence 时以
  `LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED` 稳定失败，绝不静默采用新
  scope；两个 legacy 门禁都以持久化 `status.json` 的 execution task 为权威
  （planner 可能改写 adapter 的 source task），仅当没有任何持久化状态时才
  回退 source task；持久化 execution task 为 `unknown` 哨兵（预 task 失败）
  时无法证明重规划留在 VQA 族之外，同样 fail-closed；counting、grounding、
  caption/change 的 resume 行为不受影响；
- 不复用或污染 `EvidencePreprocessingIdentity`：task scope 与 tile
  preprocessing 是两个独立身份。

---

# 43. EvaluationRecord

统一评测记录：

```python
EvaluationRecord(
    sample_id,
    task,
    deterministic_metrics,
    judge_status,
    judge_raw,
    judge_parsed,
    judge_inconsistency,
    judge_error,
)
```

`task` canonical metric family：

```text
counting
general_vqa
grounding
caption
```

Runtime task 决定执行什么；evaluation family 决定如何评分。唯一生产映射
位于 `evaluation.records.RUNTIME_TASK_TO_EVALUATION_TASK`：

| Runtime task | Evaluation family | Artifact |
|---|---|---|
| `counting` | `counting` | `counting_evaluation.json` |
| `fine_grained_counting` | `counting` | `counting_evaluation.json` |
| `general_vqa` | `general_vqa` | `vqa_evaluation.json` |
| `multiple_choice_vqa` | `general_vqa` | `vqa_evaluation.json` |
| `scene_classification` | `general_vqa` | `vqa_evaluation.json` |
| `spatial_relation` | `general_vqa` | `vqa_evaluation.json` |
| `change_qa` | `general_vqa` | `vqa_evaluation.json` |
| `grounding` | `grounding` | `grounding_evaluation.json` |
| `caption` | `caption` | `caption_evaluation.json` |
| `change_caption` | `caption` | `caption_evaluation.json` |

generic metric 不维护 dataset-specific runtime task 分支。fresh、resume 和
`evaluate-run` 都调用 `build_deterministic_evaluation(...)`；resume/offline
以持久化的 `status.task` 选择 family，成功推理不会因此重复调用 Qwen。

Judge 与 deterministic metrics 并列。

---

# 44. Counting deterministic metrics

`CountDeterministicMetrics`：

```text
predicted_count
gold_count
exact_match
absolute_error
relative_error
smooth_error_score
```

只有 Ground Truth count 可用时才能形成有意义的 deterministic count comparison。

不得为缺 GT 样本伪造 gold。

运行策略 benchmark 在 deterministic metrics 之外读取持久化 trace，统计 expert usage、
per-backend exact accuracy、fallback/zero-review/seam/Qwen-call 指标；它只能评估当前固定
策略，不能自动重排 backend priority。无真实 checkpoint、数据集或目标硬件时必须标记为
synthetic/offline gate，不能冒充 live 模型质量结果。

---

# 45. VQA deterministic metrics

`VQADeterministicMetrics`：

```text
exact_match: bool
```

当前 task family 映射包括：

```text
general_vqa
multiple_choice_vqa
scene_classification
spatial_relation
change_qa
```

落盘使用 VQA evaluation record。

这些 runtime task 共享严格 exact-match，并在 judge policy/budget 允许时共享
DeepSeek 纯文本 semantic-equivalence Judge。Judge 只处理 deterministic
mismatch；即使语义判定等价，`exact_match=false` 仍原样保留。

semantic aggregate 单独报告 coverage/completeness。存在未 Judge 或 Judge
失败的 mismatch 时，只报告 confirmed lower bound，不伪造完整 semantic
accuracy。

---

# 46. Grounding deterministic metrics

`GroundingDeterministicMetrics`：

```text
iou
iou_at_0_5
```

当前内建 deterministic grounding 是严格的轴对齐 IoU 路径。

只有 prediction 和 GT 坐标契约受支持时计算。

当前长期约束强调：

```text
normalized_0_999_top_left
4-value xyxy
```

兼容时才能生成当前内部 grounding deterministic record。

以下情况应 fail-closed 或交给显式 official evaluator / coordinate conversion：

```text
source_pixels_top_left without conversion
8-point polygon
unknown coordinate frame
```

---

# 47. Caption evaluation

`CaptionDeterministicMetrics` 实际保存逐样本：

```text
candidate
references
```

它是后续 corpus-level metric 的输入契约。

语料级指标例如：

```text
BLEU_1..4
METEOR
ROUGE_L
CIDEr
```

`caption` 与 `change_caption` 共用同一个 caption family 和 corpus aggregate。
Reporting 从持久化的 caption records 调用 `aggregate_caption()`：

```text
pycocoevalcap available -> metric_status=ok + corpus metrics
pycocoevalcap missing   -> metric_status=dependency_missing + record_count
Java/METEOR missing     -> metric_status=partial + not_computed=["METEOR"];
                          BLEU/ROUGE_L/CIDEr remain independently reported
```

可选依赖保持惰性导入；无 caption record 时不导入，缺失时也不阻断 report。
METEOR 还需要 Java；缺少 Java 时只标记 METEOR 未计算，不影响其他已成功
scorer。CHAIR2 当前没有经批准的 scorer，始终明确标记为未计算。这些本地
corpus 指标不等同于 benchmark official score。

---

# 48. Specialized runtime task 的 family 复用

以下 specialized runtime task 不创建平行 metric 实现，而是复用 canonical
family：

```text
fine_grained_counting -> counting
multiple_choice_vqa   -> general_vqa
scene_classification  -> general_vqa
spatial_relation      -> general_vqa
change_qa             -> general_vqa
change_caption        -> caption
```

其中 caption family 的逐样本 record 是 corpus metric 输入；grounding 只在
兼容 geometry 下产生 record。不支持的 runtime task 或不兼容输入返回
not applicable，不伪造分数。

如果以后新增指标，应明确：

- Ground Truth 来源；
- metric 定义；
- coordinate/label 解释；
- aggregation；
- 与历史结果可比性。

---

# 49. Judge

## 49.1 DeepSeekJudgeClient

是纯文本结构化 Judge 客户端。

它不是 Agent，不处理图像。

DeepSeek key value 只从 composition-root 边界读取本机环境变量；settings、
snapshot、manifest、RequestMeta、cache metadata、report、audit 和公共错误只
能保存环境变量名或稳定非敏感元数据，不能保存 key value。

实现特性：

- 标准库 HTTP；
- cache；
- schema validated structured result；
- retry；
- request artifacts；
- secret 不进入公共错误。

## 49.2 JudgeService

负责：

```text
judge policy
budget
deterministic + judge merge
resume supplement
```

Judge policy：

```text
none
errors-only
all
```

Dataset run 还可以使用：

```text
judge_sample_rate
```

进行确定性抽样。

## 49.3 Judge 不覆盖 deterministic

必须始终保持：

```text
EvaluationRecord.deterministic_metrics
```

独立存在。

Judge 可以记录：

```text
judge_status
judge_parsed
judge_inconsistency
judge_error
```

但不能改掉 deterministic exact_match/IoU/count error。

---

# 50. Judge sample rate

`judge_sample_rate` 范围：

```text
0.0 .. 1.0
```

DatasetRunner 使用 run/sample identity 进行确定性抽样，而不是 `random.random()`。

该值持久化，以保证 resume 仍选择相同样本集合。

---

# 51. Offline evaluation commands

当前应用包含执行后评测命令。

### `evaluate-run`

对已有 run 做离线确定性评测补全/刷新，并可选择 DeepSeek pass。

关键设计：

```text
no Qwen inference reconstruction
use persisted results
shared deterministic dispatch
```

三条评测路径保持同一 deterministic helper：

```text
fresh   -> SampleRunner -> deterministic -> optional JudgeService
resume  -> status.task -> missing deterministic/judge supplement
offline -> evaluate-run -> persisted result -> deterministic -> optional DeepSeek
```

offline 路径零 Qwen；Judge 失败保留 deterministic record。

### `judge-vqa-run`

对已有 VQA run 做 Judge pass。

应使用已有 prediction，而不是重新调用 Qwen 获得另一个答案。

### `standard-evaluate`

调用外部标准 evaluator seam。

外部 evaluator 输出与内部 deterministic metric namespace 分离。

---

# 52. Standard / dataset evaluator seam

目录：

```text
evaluation/standard/
evaluation/datasets/
```

当前：

```text
evaluation/standard/adapter.py
evaluation/datasets/vrsbench.py
```

用于把统一结果适配到团队/数据集官方评测。

原则：

- source prediction 只读；
- 外部工具失败明确报告；
- 结果进入独立 external namespace；
- 不为了“统一”而偷偷修改官方 metric 定义。

---

# 53. Reporting

Reporting 纯结果层。

主要流程：

```text
predictions.jsonl
  -> current rows (last-wins)
  -> safe sample dir resolution
  -> load persisted artifacts
  -> ReportSample
  -> TaskSummary
  -> Report
```

它不会：

- 调 Qwen；
- 调 Agent；
- 重新调用 VisualTaskPlanner；
- 修改状态；
- 重新跑 deterministic evaluation；
- 自动纠正模型答案。

---

# 54. Report Schema

## `ReportSample`

主要字段：

```text
sample_id
run_task
task
state
error_code
result_path
updated_at
question
prediction
ground_truth
resolved_task
execution_agent
fallback_used
judge_status
inference_seconds
evaluation
routing_decision
execution_path
backend_stages
model_calls
structured_artifacts
```

`task` 表示 execution task。

`ground_truth` 是 task-neutral 的只读投影，稳定包含 answers、count、boxes、
points、labels 与 coordinate_frame。`backend_stages` 只来自持久化的
`counting_attempts.json`，按真实 attempt 顺序展示；旧 run 缺少该文件时保持
空列表，不从 final result 或 trace 反推中间阶段。`model_calls` 只读取当前
sample 目录内已有的 request/raw/parsed/validation 产物，raw response 最多
8000 字符，且不暴露 artifact_dir、绝对路径、Base64 或 credential。
`execution_path` 是从已持久化 trace、任务与模型调用投影出的顶层模块交接链，
仅用于 HTML 审计，不重新推理。`structured_artifacts` 只读取允许列表中的
`vqa_evidence.json`、`grounding_evidence.json` 与 `visual_task_plan.json`，用于
展示已落盘的结构化 v2/v3 状态；历史 `visual_plan.json` /
`joint_visual_plan.json` 仅按只读兼容规则展示。

## `TaskSummary`

```text
run_task
total
succeeded
partial
failed
skipped
fallback_count
fallback_rate
agent_usage
judge_status_counts
metrics
judge_metrics
```

`metrics` 保存 local deterministic/corpus aggregate；`judge_metrics` 保存
独立的 semantic Judge aggregate。官方外部结果不进入这两个字段。

## `Report`

```text
run_id
dataset
total
succeeded
partial
failed
skipped
samples
tasks
```

---

# 55. Report Bundle

`reporting.exporters.persist_report_bundle(...)` 写：

```text
runs/<run_id>/report/
├── report.html
├── report.json
├── samples.csv
├── samples.jsonl
├── metadata.json
├── deepseek_audit.jsonl
└── external_standard.json   # optional
```

报告严格区分三类来源：

```text
metrics           -> deterministic/local（含 caption corpus aggregate）
judge_metrics     -> optional Judge quality/coverage
external_standard -> official/external evaluator
```

Reporting 只读取持久化产物并聚合，不发网络、不调用模型，也不回写 sample、
status 或 evaluation。

官方 VRSBench grounding 适配同时识别 `VRSBench_EVAL_referring.json` 与
派生的 `VRSBench_EVAL_Det.json`。前者保留官方 8 点 polygon，并显式标记
`normalized_0_1_top_left`；HTML 会按该坐标系叠加 GT 与模型框。当前通用
deterministic IoU 仍对不兼容 polygon fail-closed，不把可视化叠加冒充官方
oriented-box 分数。

### JSON

UTF-8，稳定字段布局。

### CSV

`utf-8-sig`，方便 Windows Excel。

### HTML

离线，无需 CDN。

用户/模型文本需要转义，不能因模型输出注入 HTML。

### DeepSeek audit

只记录稳定 request identity / status / parsed metadata。

不输出：

```text
auth
api key
raw secret
```

---

# 56. External standard namespace

外部评估输出包装为：

```json
{
  "schema_version": "report-v2",
  "external_standard": {
    "...": "..."
  }
}
```

这样不会把外部官方 metric 和内部 deterministic metric 混成同一个无来源的数字。

---

# 57. MME official export

Reporting exporter 还支持 MME-RealWorld 官方提交转换。

原则：

- 原始官方记录只读；
- 按 question id 写预测到官方 `Output`；
- 其他无关字段保持；
- 不修改 source file。

---

# 58. Counting visualization

`reporting.visualization` 根据：

```text
source image
CountingResult
```

生成 overlay。

这是展示能力，不参与 metric。

如果图像尺寸/结果契约不一致，应失败，不应为了画图改变计数结果。

### Report V2 presentation and asset boundary

Report V2 is a typed presentation layer over persisted artifacts. Its stable
models include `RunMetadata`, `LatencySummary`, `RoutingView`, typed task
details, `VisualAssetView`, routing/failure/target aggregates, and bounded
Counting point previews. Missing optional artifacts remain missing; they are
not converted to zero/false or reconstructed from names.

The export lifecycle is:

```text
build read-only Report
  -> materialize every resolvable sample preview
  -> materialize report-relative final/stage WEBP/PNG assets
  -> write report-v2 JSON/CSV/JSONL/metadata
  -> render the pure offline HTML dashboard
```

Visual materialization is the only reporting stage allowed to consume the
private `run_request.dataset_root`, and that value never enters a Report view
model or text export. It does not call an Agent, router, model, backend, judge,
or evaluator. Evidence with an explicit `image_id` is bound strictly; missing
IDs bind only for a single-image sample. Ambiguous multi-image evidence and
unsafe ground-truth coordinate frames are not guessed.

The dashboard is sample-first: Samples is the first/default content, while
Overview, Tasks, Expert Routing, Failures, and Runtime remain secondary,
collapsible aggregate views. A collapsed sample row already shows its
thumbnail, question, prediction, ground truth, result quality, final backend,
fallback state, and latency. The expanded first screen shows visuals and the
answer audit before the real persisted execution stages and sanitized model
calls. Deterministic quality and optional Judge quality remain separate,
including the existing incomplete semantic-Judge lower-bound behavior.

Asset policy:

- default visual-sample budget: none; every resolvable sample is materialized;
- an explicit compatibility limit still uses failed/partial/incorrect/fallback/
  warning priority followed by stable run-task/sample order;
- original preview: WEBP, maximum side 1024, quality 85;
- overlay: PNG, outline/ring only, 1–2 px;
- accepted/prediction green, rejected red, ground truth cyan, unresolved
  amber, reviewer purple;
- real OBB polygons are drawn as polygons, never downgraded to enclosing
  rectangles;
- OBB/box geometry suppresses the point-pipeline helper radius; point-only
  evidence uses a fixed 3–5 px hollow ring and never `point.radius_px`;
- successful persisted counting attempts with geometry receive separate,
  hash-safe stage overlays; failed/unavailable or legacy inferred stages do not;
- dimension mismatch preserves an original preview but suppresses the
  invalid overlay.

All report text exports remove host paths, dataset roots, unsafe raw exception
messages, credentials, authorization values, and network URLs. HTML consumes
only the structured `Report`; it performs no I/O and contains no external
resource or Base64 dependency.

---

# 59. Public CLI

唯一公共入口：

```text
python main.py
```

无子命令：

```text
serve
```

默认：

```text
host=127.0.0.1
port=8000
```

当前公开命令：

```text
serve
ask
run-init
health
list-datasets
smoke-qwen
resume-run
inspect-data
count-image
download-data
evaluate-run
judge-vqa-run
standard-evaluate
render-count
summarize-evaluations
run-dataset
```

具体参数以：

```text
python main.py <command> --help
```

为准。

---

# 60. `serve`

本地 HTTP 服务。

受支持：

```text
GET /health
POST /ask
```

设计：

- stdlib HTTP；
- process runtime 创建一次；
- handler 不创建模型；
- `/health` 不调用模型；
- `/ask` 委托 Runtime；
- 请求输入有大小与格式检查；
- 错误响应只暴露稳定信息。

它不是 dataset evaluation service。

---

# 61. `ask`

对一个本地图像目录执行一次手动请求。

概念参数：

```text
--images-dir
--question
--task
--output
```

`--task auto` 时：

- 空问题规则由 v5 VisualTaskPlanner 的冻结 system prompt 处理；
- 显式 task 也先经过同一个 VisualTaskPlanner，requested task 只作审计；
- 规划 task 后才物化 UnifiedSample；
- 一次手动请求只执行一个主业务路径；
- 不自动做完整 DatasetRunner Judge/Report workflow。

手动图像路径在 artifact 中保持相对/安全表示，不应把本机绝对输入目录写进结果 payload。

---

# 62. `run-dataset`

核心数据集批量运行入口。

关键参数：

```text
--dataset
--root
--split
--task
--auto-task
--sample-ids
--run-id
--resume
--evaluate / --no-evaluate
--judge-policy
--judge-sample-rate
--render-errors
--max-samples / --limit
--start-index
--shard-index
--shard-count / --num-shards
--sample-concurrency
--fail-fast
```

默认：

```text
evaluate = True
judge_policy = none
```

这是默认离线的重要表现：运行数据集不会默认调用 DeepSeek。

---

# 63. `run-init`

只创建 run identity 和 snapshot。

不执行 Qwen。

适用于：

- 检查 run store；
- 提前建立可复现目录；
- 运维测试。

---

# 64. `health`

组件：

```text
qwen
deepseek
```

普通 health 只查看元数据/配置可用性，不应泄露 secret。

`--live` 才允许执行真实 probe。

---

# 65. `smoke-qwen`

直接进行一次 VisionLanguageClient smoke request。

用途是验证模型端，不经过完整 Agent / DatasetRunner。

不要把它的结果误认为完整任务评测。

---

# 66. `count-image`

单图计数维护/验证入口。

支持：

```text
--image
--question
--target-spec
--run-id
--evaluate
--render
--resume
--force
--no-seam-verify
--max-qwen-calls
--max-deepseek-calls
```

为保证 resume/force 行为一致，行为相关调用参数会进入 `run_request.json` 的 count-image fidelity 字段。

---

# 67. `inspect-data`

对数据集根做只读审计。

不调用模型。

用于检查：

- layout；
- adapter probe；
- 文件/字段问题；
- 快速或完整扫描。

---

# 68. `download-data`

显式数据集联网下载入口。

它是“用户明确要求下载”的行为，而不是普通加载的 fallback。

---

# 69. `render-count`

读取：

```text
image
persisted counting result
```

生成 overlay。

不运行模型。

---

# 70. `summarize-evaluations`

聚合已存在 EvaluationRecord / run evaluation。

不重新推理。

---

# 71. Application Runtime

`application/runtime.py` 暴露高层 use case。

核心思想：

```text
runtime owns use cases
bootstrap owns dependency assembly
commands own CLI adaptation
```

不要把三层重新合并成一个“大 application.py”。

---

# 72. Offline 原则

默认本地/离线能力：

- import packages；
- parse config；
- dataset adapter/probe；
- deterministic routing；
- run store；
- report；
- deterministic metrics；
- tests。

可能联网的能力必须是显式路径，例如：

```text
download-data
DeepSeek judge
health --live
显式允许模型下载
```

缺少网络不应让普通模块 import 崩溃。

---

# 73. Secret 原则

Secret value 不进入：

```text
AppSettings
config.snapshot.json
manifest.json
run_request.json
AgentContext.request_context
agent_trace.json
public error
report metadata
DeepSeek audit
```

settings 只存 API key 的环境变量名称。

---

# 74. Path portability 与 path safety

新架构特别强调 Windows/POSIX 一致性。

应区分三类路径。

### Dataset image path

```text
dataset-root relative
```

### Status result path

```text
sample-relative basename
```

### Prediction result path

```text
run-relative display/index path
```

机器本地物理路径可以存在于运行时 `Path` 对象和 host-oriented config snapshot 中，但不应进入逻辑 ID、sample id、可移植 result identity 或模型 cache logical identity。

---

# 75. Error 语义

公共持久化状态尽量使用稳定：

```text
error_code
exception class name
stable domain code
```

避免写完整内部异常。

理由：

- 防 secret；
- 防 host absolute path；
- 防依赖版本差异；
- 提高 parity；
- 方便报告聚合。

---

# 76. Tests

测试分层大致包括：

```text
tests/architecture/
tests/contracts/
tests/data/
tests/models/
tests/agents/
tests/routing/
tests/workflows/
tests/evaluation/
tests/reporting/
tests/application/
tests/integration/
tests/parity/
```

## Architecture

保证：

```text
import DAG
__init__ no side effects
package discovery
no legacy import
```

## Contracts

保证核心 Schema 与 artifact 不变量。

## Unit

按 package 验证具体实现。

## Integration

验证纵向：

```text
sample
dataset
auto-task
resume
run-dataset
```

## Parity

与锁定 `try_yolo` 行为基线比较可观察行为，而不是源码结构。

---

# 77. Golden migration fixtures

位置：

```text
tests/fixtures/migration/
```

作用：

- 冻结旧行为的可观察事实；
- 验证迁移功能没有无意丢失；
- 避免两个无共同祖先分支只能靠普通 diff 对比。

Golden fixture 不属于“修实现时可以跟着一起改”的普通测试数据。

有意差异必须单独说明。

---

# 78. Architecture boundaries

项目不使用 Python 文件路径白名单，也不维护中央 implemented/pending 文件清单。

新增模块是否合法由以下因素决定：

1. 文件所在 package 的职责；
2. `architecture/import_rules.json`；
3. architecture tests；
4. `AGENTS.md` 的长期架构约束。

普通新增 Python 文件无需单独登记。

新增顶层 package、改变既有 package 职责、改变 import DAG 等仍属于架构变更。

---

# 79. 当前已知限制与边界

以下不是 bug，而是当前明确边界，除非后续任务改变设计。

### 79.1 Grounding coordinate coverage

内建 deterministic IoU 不是万能坐标转换器。

source pixel / polygon 等需要 official evaluator 或显式转换。

### 79.2 Spatial/change sample-level metric

当前没有为所有 spatial/change 任务强行定义通用逐样本 deterministic score。

### 79.3 Cross-process append

当前工作流 JSONL 并发安全承诺局限于同一 Python 进程。

### 79.4 v5 visual-only planner；runtime evidence 按 task 严格发布

`VisualTaskPlanner` 对所有 fresh manual/dataset 入口始终启用；当前配置不再
提供 `visual_planning.enabled` 开关，所有 fresh 入口都先经过 v5 规划器。
`visual-evidence-catalog-v4` 已根据当前 DOTA-v2 YOLO、经验证的 iSAID
SegFormer 与 2026-08-20 确认的 OEM class map 声明 26 个 General VQA 叶子：
15 个叶子同时具有 YOLO 与 iSAID SegFormer（`segmenter_mitb2_001`）绑定；
`container-crane`、`airport`、`helipad` 只有 YOLO；`bareland`、`rangeland`、
`developed-space`、`road`、`tree`、`water`、`agriculture-land`、`building`
只有 OEM SegFormer（`segmenter_oem_001`）绑定。OEM class map 由用户针对本地
OpenEarthMap checkpoint 明确确认（`classes.json` 9 类：background + bareland/
rangeland/developed_space/road/tree/water/agriculture_land/building；checkpoint
SHA256 `d2141c79b2fc27ea5505db378b48e90e75e5ee06751df1c5b4028ef662fb2fab`）。
其 `config.json` id2label 仍是 `LABEL_0..8`，只说明 channel 数，不能独立证明
语义顺序。
composition root 在发布 VQA binding 前严格匹配 settings 的
`class_map_version`、`classes.json` 的 checkpoint digest 与 evidence catalog 的全部
SegFormer raw labels；不一致时不会发布对应 planner leaves。
类别映射已验证不等于运行能力已启用。当前 `configs/local.yaml` 已完整校准并启用
DOTA-v2 YOLO detector policy、iSAID SegFormer 与 OEM SegFormer binding，因此
General VQA planner binding 发布 catalog 中全部 26 个 leaves；Counting、
Fine-grained Counting 与 Grounding 各发布 18 个 leaves。这里的数量是当前配置经
composition root 实际组装后的结果，不是仅凭 catalog 声明推断。VQA 没有匹配的
已启用能力时对应 executable leaf 集合为空；模型仍
请求不可用 leaf 会以 `CAPABILITY_UNAVAILABLE` 失败，绝不静默回退 direct。
Grounding 与 VQA 的发布条件不同：Grounding executor 始终组装并使 catalog 中的
18 个 grounding leaves 可规划；detector policy 未校准时只关闭其 YOLO phase，最终
Grounding Qwen 的定位 seam 仍存在。Counting 的可执行 leaves 则由独立的 enabled
expert inventory 决定。检测器一律惰性接线，组合期绝不加载 YOLO 权重。
证据预处理身份冻结为组合版本（doc 26）：YOLO 每个 ROI 按确定性贪心
row-major 无重叠切 1024×1024 tile，余块 LANCZOS 拉伸，掩膜逆变换 NEAREST；
SegFormer 不再共用 tile 路径，fresh 默认协议
`pad-multiple-1024-resize-square-v1` 把整张 ROI 仅在右侧/底部以固定黑色
padding 到 1024 的最小倍数（`padded = ceil(W/1024)*1024 × ceil(H/1024)*1024`，
padding 恒在 `[0, 1023]`），再整体 LANCZOS 缩放到单张 1024×1024 模型输入；
每个（ROI，binding）恰好一次 SegFormer 推理，离散 class-id map 用 NEAREST
恢复到 padded 尺寸后确定性裁切 `[0:W, 0:H]`，最终 mask 与 YOLO ROI-local
框严格同坐标系（doc 26 rollout 之后，该“恢复+裁切”只作为测试 oracle 存在，
生产路径改为 preview 空间直接采样，见 21.1 有界流式物化说明）。几何记录
`SegFormerPreprocessRecord` 强制最小上取整，
过度 padding 稳定失败；tile 并发有界（默认 ≤4，`max_tile_concurrency`
可配置），单次执行生命周期使用单一 worker pool，tile 图像按固定提交窗口
按需物化（活跃 tile ≤ 并发上限），全程不创建完整 ROI 裁切或全部 tile 列表。
无重叠 partition 只保证
每个 ROI 像素属于一个 source tile；它不等价于“对象不会跨 tile”或“不会产生
重复检测”。YOLO 候选逆映射到整图后另做基于 IoU 的全局去重，但边界目标仍
可能被切开或漏检；余块拉伸也会改变纵横比。大 ROI 在新 SegFormer 协议下
整体压缩到 1024×1024：小目标可能因下采样而消失、非正方形 padded canvas
会被拉伸成正方形、tile seam 消失、调用次数与显存/吞吐特征变化、边缘预测
可能受固定 padding 颜色影响——这些是预期的模型行为变化而非几何 bug，
真实精度变化需用已批准本地权重做 live gate 单独验证。v5 显式 ROI 在源图上
生成 1024 整数倍边长的理想正方形后直接与图像求交，不平移、不缩小；边界
裁切后的实际视图可以是长方形，也不保证宽高为 1024 的整数倍。历史
`visual_plan.json` / `joint_visual_plan.json` 仅供 reporting 只读展示。
真实 Qwen3-VL、YOLO 和 SegFormer live gate 仍需单独验证。

### 79.5 Live validation

离线测试通过不代表：

- 本地 Qwen 权重已存在；
- DeepSeek key 可用；
- 真实 VRSBench/XLRS/MME/LEVIR 数据集存在；
- Spark 目标机可运行；
- GPU/ONNX/CUDA detector 实际可用。

必须分别验证。

### 79.6 Optional vision/deployment dependencies

变化检测、YOLO/ONNX 等扩展能力依赖环境配置，基础 import 不等于所有 backend 都 available。

### 79.7 OEM class labels

OEM 9-channel checkpoint 的分类头维度已核验，并以 OpenEarthMap 官方八类顺序
（加 index 0 background）写入版本化 `classes.json`；runtime 只读取该映射，不再
暴露旧的 `LABEL_0..8` 占位名。

### 79.8 Semantic connected-component counting

SegFormer 输出 semantic region 而非 instance mask。相接实例可能形成一个 component 并
低估数量；当前不隐藏加入 watershed 或 instance splitting。此限制应在 benchmark 中单列。

### 79.9 Historical joint task + visual planning（doc 15，仅只读）

doc 15 的单次 Qwen 联合调用先被 doc 17 的 `visual-task-plan-v2` 替代，随后
doc 18 移除 confidence 并升级为 v3、doc 19 增加精确 `count_target` 和 leaf-only
类别并升级为 v4、doc 20 冻结量化 ROI 后升级为当前 `visual-task-plan-v5`。
本节只保留旧 artifact、旧 trace 与迁移结果的只读解释；不得重新接线为
fresh execution fallback。

```text
joint-qwen-plan-v1
    version: "joint-qwen-plan-v1"
    task: 闭合 data.schema.TaskName 集合（extra="forbid"）
    visual_plan: FirstQwenVisualPlan 子结构
```

契约要点：

- `extra="forbid"`；模型正文不含 final answer / backend / checkpoint /
  path / GT；非法输出稳定失败 `JOINT_PLAN_FAILED:CODE`；
- 模型选定 task 对 routing / materialization / execution 权威，源 task
  只做审计，GT 只读；
- `joint_plan_to_resolution` 纯确定性派生 resolution（`source="model"`，
  单一候选，reason codes 透传），TaskRouter 保持确定性；
- 每条样本恰好一次联合调用（shared `CallBudget.max_qwen_calls`），resume
  succeeded 零模型调用（§36.5 只补评测）；
- request hash 覆盖逻辑模型身份 / revision / generation settings / prompt
  版本与正文 / catalog / messages / preview digest / client version；
- 产物 `joint_visual_plan.json`（与 gate 的 `visual_plan.json` 永不冲突），
  trace 增加 `joint_plan` 审计字段；manual ask 以 placeholder-role
  SampleDraft 走联合路径（request.json `"joint_plan": true`）；
- 低置信度候选 fallback 语义不变（SampleRunner 职责，非 Resolver）；
- 联合规划器、联合 prompt binding 与对应 runner 分支均已从当前 runtime 删除。

已知限制：本节只解释历史 run；不得把这些类型、字段或路径重新接线到 fresh
execution。

---

# 80. 修改哪个模块时先看什么

### 修改数据集

先看：

```text
data/schema.py
data/adapters/base.py
对应 adapter
data/registry.py
tests/data/
tests/contracts/test_data_schema_contract.py
```

### 修改 task routing

先看：

```text
data/schema.py
routing/schema.py
routing/policies.py
routing/router.py
tests/routing/
tests/workflows/test_sample_runner.py
```

### 修改 Agent

先看：

```text
agents/base.py
agents/schema.py
agents/registry.py
对应领域目录
tests/agents/
tests/contracts/test_agent_result_contract.py
```

### 修改 counting

先看：

```text
agents/counting/schema.py
agents/counting/settings.py
agents/counting/agent.py
agents/counting/executor.py
agents/counting/backends/
tests/agents/counting/
tests/contracts/test_counting_result_contract.py
```

### 修改模型

先看：

```text
models/base.py
models/settings.py
models/entry.py
application/bootstrap.py
tests/models/
tests/application/test_bootstrap.py
```

### 修改运行产物/resume

先看：

```text
workflows/schema.py
workflows/run_store.py
workflows/artifact_writer.py
workflows/sample_runner.py
workflows/dataset_runner.py
reporting/adapters.py
tests/workflows/
tests/integration/
tests/contracts/test_artifact_contract.py
```

### 修改评测

先看：

```text
evaluation/records.py
evaluation/metrics/
workflows/sample_runner.py
workflows/judge_service.py
evaluation/standard/
evaluation/datasets/
tests/evaluation/
tests/parity/
```

### 修改报告

先看：

```text
reporting/schema.py
reporting/adapters.py
reporting/builder.py
reporting/exporters.py
reporting/html.py
tests/reporting/
```

### 修改 CLI

先看：

```text
main.py
application/commands/
application/runtime.py
application/bootstrap.py
tests/test_main.py
tests/application/
tests/integration/
```

---

# 81. 新功能应该落在哪里

可以使用这个判断表。

| 新需求 | 首选位置 |
|---|---|
| 新数据集解析 | `data/adapters/` |
| 新统一样本字段 | `data/schema.py`（高风险） |
| 新模型 wrapper | `models/` + `models/entry.py` |
| 新业务 task workflow | `agents/<domain>/` |
| 已知 task → Agent 映射 | `routing/` |
| 无 task 样本的任务判定 | `workflows/dataset_runner.py` + `workflows/visual_planner.py` + `data/schema.py` |
| 单样本 orchestration | `workflows/sample_runner.py` |
| 数据集 orchestration | `workflows/dataset_runner.py` |
| deterministic metric | `evaluation/metrics/` |
| Judge | `evaluation/judges/` + `workflows/judge_service.py` |
| external official evaluator | `evaluation/standard/` 或 `evaluation/datasets/` |
| 报表 | `reporting/` |
| CLI use case | `application/commands/` |
| dependency assembly | `application/bootstrap.py` |
| 顶层命令参数 | `main.py` |

新增文件应优先放入职责对应的现有 package；如果需要新增顶层 package 或改变已有职责边界，
再按架构变更处理。

---

# 82. 不应该再恢复的旧架构模式

新架构不再使用：

```text
spacers_agent/
eval/
第二套内部公开 CLI
dataset-specific logic in Agent
Router model call
model construction inside workflows
report-time inference
legacy-path resume trust
```

旧 `try_yolo` 中相应实现只作为迁移行为参考。

---

# 83. Evaluation 与 Reporting 对齐 `try_yolo` 时的原则

后续对齐旧评测和报告时，应区分三种情况。

## 83.1 行为必须保持

例如：

- metric 公式；
- GT 读取；
- official submission format；
- sample inclusion；
- failure accounting。

这类应通过 parity/fixture 证明。

## 83.2 架构实现可以不同

例如旧代码可能：

```text
eval/metrics.py
spacers_agent/reporting.py
```

新架构可以拆成：

```text
evaluation/metrics/
evaluation/records.py
reporting/
```

只要求外部可观察行为与明确的新契约一致。

## 83.3 有意变化

例如新架构强化：

- path safety；
- secret handling；
- append-only execution index；
- deterministic/Judge 分离；
- unknown task fail-closed；
- report read-only。

这类不应为了“源码相似”退回旧行为，应在 migration docs 中明确差异和理由。

---

# 84. 当前架构一句话总结

当前 M3 新架构的主链路是：

```text
Dataset source
  -> explicit Adapter
  -> UnifiedSample / SampleDraft
  -> VisualTaskPlanner (one call)
  -> materialize UnifiedSample
  -> deterministic TaskRouter
  -> task Agent
  -> SampleRunner
  -> persisted result
  -> deterministic EvaluationRecord
  -> optional Judge
  -> DatasetRunner execution index
  -> read-only Reporting
  -> application use cases
  -> main.py public surface
```

其中：

```text
application
```

是唯一 composition root，

```text
UnifiedSample
```

是统一内部样本契约，

```text
run_request.json + persisted artifacts
```

是 resume 与报告的事实基础，

```text
EvaluationRecord
```

是统一评测记录，

```text
reporting
```

只读取结果，不重新执行结果。

这几个边界应被视为后续开发最重要的长期契约。

---

# 85. Analysis tooling

独立分析脚本（非 RunStore run 产物、非公共 CLI）：

```text
scripts/evaluate_vrsbench_counting.py
```

在 VRSBench-counting JSONL 上逐样本调用真实 CountingAgent（经
`Runtime.create` 的完整组合），从 trace 标注最终答案来源
（final_backend/kind、primary/fallback 与 target），复用
`count_deterministic_metrics` / `aggregate_counting` 与生产车辆
`count_target_hint`；输出 `results.jsonl` + `summary.json` +
`unsupported_or_error.json`，无 DeepSeek/Judge 参与。

```text
scripts/prepare_qwen3vl_phase2_sft.py
```

Phase 2 SFT canonical 数据准备器（只读、离线、确定性，任务文档见
`docs/train/01_PREPARE_PHASE2_SFT_DATA.md`）：解析
`data/phase2-train/VRSBench/`（缩进 JSON 块或单行 JSONL 均可）与
`data/phase2-train/GeoChat/GeoChat_Instruct.json`（流式 JSON 数组），
导出与 Transformers/PEFT/Qwen template 解耦的 canonical episodes：
`<output_dir>/{train,validation}.jsonl` + `manifest.json` +
`rejected.jsonl`。契约要点：VRSBench grounding 每合法 object 一条；
VQA 有框视图的 input_boxes 只声明为同图 annotation 上下文，不做
question→object 模糊绑定；train 按 `source_task` 分层、以
`sha256(seed + parent_episode_id)` 排序取前 `round(0.4*N)` 生成
self-attention 无框额外副本（不替代有框版，validation 不增广）；
GeoChat `[refer]`→target 框、`[identify]`→input 框 + 文本、普通对话
保序，坐标统一 `round(c*999/100)` 转换且转换前后校验，拒绝记录以稳定
错误码进 `rejected.jsonl`（含闭合计数）。episode_id 全局唯一；输出无
机器绝对路径；同输入同 seed 字节级稳定。

```text
scripts/refine_visual_planner_dataset.py
```

Visual Planner SFT 标注扩展器。源数据只读，默认只做输入审计和当前 runtime
protocol 组装；只有显式 `--use-api` 且从环境变量或无回显终端提示取得
key 时才调用 DeepSeek。
teacher 的每条 sample payload 严格为 `{"question": raw_question}`，不发送图像、
旧 target、答案、数据集信息、provenance 或路径；response schema 也只允许
`task`、`object_categories`、`needs_visual_assistance`、`count_target` 四个用户授权
字段，禁止 teacher 生成完整 target、ROI、reason code 或最终监督 token。
teacher system message 同时嵌入版本化 text-only task taxonomy/assistance rubric、精确
四字段 JSON Schema、完整 runtime v5 prompt 和当前 planner binding；JSON repair
可显式保留原始 question payload。脚本按精确 question 去重并发
请求，使用内容寻址 cache 和可复现 resume identity。text-teacher v6.1 数据标注
策略按全局可调用子模型 leaf 并集选择证据，不采用当前 runtime 的 task-specific
开关；scene classification、spatial relation 等 task 只要能识别出相关 callable
类别即可启用，限定计数也可调用基础类别子模型。证据始终是交给最终 VLM 的非
权威辅助信息。整图或明确 ROI/局部区域的开放式描述归入 `caption`；框内类别、
颜色、朝向或运动状态等封闭问题仍归入 `general_vqa`。本地确定性层负责 answer
leakage、alias/parent expansion、八类
上限和 `VisualTaskPlan` schema 门控。输出使用带显式 annotation evidence policy
的 content-addressed protocol；`datasets/` 保留紧凑引用，`training/` 展开
为准确的 system、image+raw-question user、compact JSON assistant 消息，
`training_images/` 通过生产 `preview_from_path` seam 物化与推理 planner 完全相同
的确定性 PNG bytes。
`audit/training_contract.json` 区分已验证的消息语义和尚未在真实 processor 上验证的
token/chat-template 一致性。
DeepSeek V4 JSON 请求显式使用 non-thinking 模式，避免把隐藏推理引入标注产物；
复标 identity 同时冻结 timeout、retry 和 concurrency，运行参数漂移时拒绝 resume。

```text
scripts/supplement_visual_planner_dataset.py
```

Visual Planner 稀缺 task 的离线结构化补充编译器。它以 refined-v3 为只读基线，
从 VRSBench 的 caption、object box、QA 与 LEVIR-CC 有序 A/t1、B/t2 图对构造
`caption`、`grounding`、`fine_grained_counting`、`multiple_choice_vqa`、
`change_caption`、`change_qa`。默认每类选择 train 800、val 100，排除 test；
LEVIR 变化与未变化样本等量。VRS 同一图复用为四种 planner episode，LEVIR
同一图对复用为两个 change episode；图片按内容摘要硬链接并用生产 preview seam
生成训练图。MC 仅允许数值计数、Yes/No 存在性和颜色三种同语义答案空间，且不
持久化答案键。输出保持 `visual-planner-compiled-chat-v1`，双图 user content 按
image、image、raw question 排列。选择 seed、源摘要、配额、split 排除和逐条来源
均写入 audit；编译过程不联网、不修改源数据、不改变 `VisualTaskPlan` schema。

```text
scripts/qwen3vl_phase2_data.py
```

Phase 2 数据管线（任务文档见 `docs/train/02_QWEN3VL_PHASE2_DATA_PIPELINE.md`）：
公开 `Phase2EpisodeDataset` / `Phase2DataCollator` / `AugmentationConfig` /
`DatasetRootConfig`，供训练脚本消费，不解析原始标注、不加载主模型。
要点：图片路径经 `image_source -> root` 映射安全解析（拒绝绝对路径/盘符/
UNC/`.`/`..`/符号链接逃逸，错误只带相对路径）；在线增强 seed 为
`sha256(seed|epoch|parent_episode_id)`，成对有框/无框 VQA 同 epoch 图像完全
一致；几何（90°旋转/仿射/透视）与成像退化（对比度/亮度/暗角/失焦或运动
模糊/噪声/JPEG）用独立随机子流，退化不移动坐标；任一必需框未过质量门禁
则整条 episode 回退 identity；`orientation_locked` 永不做几何增强但允许成像
退化；对话渲染统一（grounding/refer 输出 JSON 框、VQA 有框版声明
`Available annotated regions`、无框版不含任何坐标）；labels 来自 chat-template
assistant mask（5.14.1 键 `assistant_masks`，4.x `assistant_tokens_mask`，
不支持时逐 turn 边界编码回退），图像 token 展开按长度差 delta 对齐；截断
只按完整 turn pair、单对超长抛 `episode_too_long`；collator 右填充并返回
`(batch, meta)`，视觉张量沿 dim0 拼接。processor 契约已对照 M3 环境
transformers 5.14.1 验证（pixel_values (G,1176)、image_grid_thw (1,3)、
mm_token_type_ids 默认开启）。

ChangeAgent SFT 接口（任务文档见 `docs/train/05_CHANGE_QWEN_SFT.md`）使用
`scripts/build_change_qwen_sft_corpus.py`、
`scripts/finetune_multimodal_sft.py --data-profile change_agent` 和
`scripts/export_multimodal_sft_checkpoint.py`。canonical episode 固定为有序
`raw_full_t1`、`raw_full_t2`，并以 `ChangeInitialResult` JSON 为唯一
initial-stage target；`change_caption` 可为空问题，`change_qa` 必须非空。
多图 processor 保留所有图像占位符且只对 assistant token 计算 loss；Change
profile 拒绝旧单图随机增强。正式 train 在 source ingestion、split、exclusion、
dedup 后按 manifest 中的 `sha256_episode_id_v1`/seed 1234 确定性排序，validation
保持 builder source order；重复构建须字节级一致。checkpoint manifest 记录
profile、有序多图 contract、数据/ordering identity 和生产 prompt SHA，resume
不允许与 `phase2` profile 或不同数据/ordering identity 交叉。正式 Qwen3.5
checkpoint 通过 generic exporter 导出，并使用
`scripts/build_change_export_fixture.py` 生成的确定性双图 fixture 做真实 forward
验证；Qwen3-VL Phase 2 composite exporter 不属于该路径。

```text
scripts/finetune_qwen3vl_phase2.py
```

Phase 2 Qwen3-VL-8B SFT 训练脚本（任务文档见
`docs/train/03_FINETUNE_QWEN3VL_PHASE2.md`）。策略：Vision Encoder 冻结；
主 `model.visual.merger` 与全部 `deepstack_merger_list.*` 全量训练
（`--merger-lr`，不挂 LoRA）；LLM base 冻结，每层
`self_attn.{q,k,v,o}_proj` 与 `mlp.{gate,up,down}_proj` 挂 LoRA
（`--lora-lr`）。结构定位不依赖模糊字符串匹配：按属性识别视觉根
（merger + deepstack_merger_list）与文本根（layers + embed_tokens），已对照
transformers 5.14.1（视觉根 `model.visual`、文本根 `model.language_model`）。
启动时硬审计：LoRA target 全路径、trainable/冻结计数、可训练参数精确分类为
`merger_base`/`llm_lora` 且两集合不重叠，审计 JSON 写入
`output_dir/parameter_audit.json`。optimizer 显式四组（merger/lora ×
decay/no_decay），cosine + warmup（默认 0.03）使用同一 lambda 保持两套 LR
比值；DeepSpeed/FSDP 被稳定拒绝（四组 optimizer 契约与复合保存要求全量
权重，CLI 传入即报错）。数据只消费 `scripts/qwen3vl_phase2_data.py` 的
`Phase2EpisodeDataset`/`Phase2DataCollator`/`AugmentationConfig`/
`DatasetRootConfig`，不复制数据语义、增强或 prompt 逻辑；支持确定性 group
repeat weight（默认 1；按 episode JSONL 的 group key 稳定展开，每 epoch 至少
遍历全部 Episode 一次；配置与实际 group 计数进 manifest）。唯一产物是可
resume 的复合 checkpoint：`checkpoint-N/{adapter/,
merger_model.safetensors, processor/, phase2_training_manifest.json,
trainer_state.json, optimizer.pt, scheduler.pt, rng_state.pth}`；
manifest 最后写作为完成标记（含 base/processor 逻辑身份与 revision、
train/eval Episode checksum 与上游 manifest checksum、LoRA 配置与完整
target 列表、merger 参数 name/shape/dtype/numel 表、adapter/merger 文件
sha256、optimizer group 摘要与两套 LR、增强 seed 与配置、训练参数、
git HEAD 与 transformers/torch/peft 版本）。resume 按 manifest 与当前显式
请求逐项校验（base/processor 身份、数据 checksum、LoRA rank/alpha/target、
merger 参数表、optimizer group 拓扑、增强 seed/配置、max_seq_length 与图像
像素设置），冲突稳定拒绝；默认自动 resume 最新完整 checkpoint，半成品目录
不视为成功且拒绝覆盖。`--smoke-gradients` 在训练前做小型 forward/backward
梯度检查（peft 将 lora_B 初始化为零，首步 lora_A 梯度按设计为零，检查先做
一步 warm-up 更新）；`--image-min/max-pixels` 仅在处理器声明支持时注入
（5.14.1 Qwen3-VL processor 不支持该参数，manifest 记录
`image_pixels_applied=false`）。默认 `--local-files-only`；模块 import 不加载
权重（torch/transformers/peft 惰性导入）。

```text
scripts/export_qwen3vl_phase2_checkpoint.py
```

Phase 2 完整模型导出器（任务文档见 `docs/train/04_EXPORT_QWEN3VL_PHASE2_CHECKPOINT.md`）：
把第三轮复合训练 checkpoint（base + LLM LoRA adapter + 全量训练后的主
Merger/DeepStack Merger state）导出为可由 `AutoModelForImageTextToText.
from_pretrained()` 与项目 Qwen3-VL 主流程直接加载的完整部署 checkpoint。
导出器只恢复、合并、保存和验证模型；不读取训练集、不执行训练、不重新
解释数据配置；模块 import 不加载权重（torch/transformers/peft 惰性导入）。
固定顺序：校验训练 manifest → 加载 base → 从 base 枚举预期 merger key →
三方严格加载 merger（模型枚举 × manifest 参数表 × safetensors 内容；
missing/unexpected/shape 冲突/非浮点 dtype/数量不符均失败；允许显式记录
的浮点 dtype 转换）→ 通过 PEFT 官方接口挂 LLM LoRA → 校验 adapter 身份/
target 集合与 manifest 完全一致、全部 adapter tensor 被消费、无视觉或
merger LoRA target → `merge_and_unload` → 审计最终模型（无 lora_A/lora_B
或 PEFT wrapper-only key 残留；merger 张量保持训练值）→ 保存模型与
processor（`safe_serialization=True`）→ 复制 `save_pretrained` 未产出的辅助
配置（不覆盖已有文件）→ 从临时目录以 `local_files_only=True` 离线 reload
验证（AutoConfig/AutoProcessor/AutoModelForImageTextToText、model_type、
config 身份与 base 一致、merger 数量、无 LoRA 残留模块、最小 image+text
chat template 渲染、权重分片齐全、全文件 checksum）→ 可选 `--verify-forward`
（程序生成小图 + 固定短 prompt 的无梯度 forward）→ 写
`phase2_export_manifest.json`（base 逻辑身份/revision、训练 manifest
sha256、adapter/merger sha256、LoRA 配置与 target 集合、merger 摘要、
输出 dtype 与转换、版本、文件 size/sha256、reload/forward 结果、git HEAD）
→ 同文件系统原子 rename 发布。`output_path` 已存在即拒绝；失败或中断
（130）不创建最终目录，临时目录安全清理（绝不删除用户给定路径）。
CLI：`--model-id`（默认 `models/qwen3_vl_8b/weights`）/
`--checkpoint-path`/`--output-path`/`--torch-dtype`（默认 bfloat16）/
`--device`（默认 cpu）/`--local-files-only`（默认 true）/
`--verify-forward`。公共 stderr 只输出稳定阶段与异常类型；导出 manifest
不记录 secret、机器绝对路径（作为逻辑身份）或原始异常全文。

```text
scripts/finetune_qwen35_9b_visual_planner_lora.py
scripts/run_qwen35_9b_visual_planner_lora.sh
```

Qwen3.5-9B visual planner 的离线 LLM LoRA SFT 工具（任务文档见
`docs/train/05_FINETUNE_QWEN35_9B_VISUAL_PLANNER_LORA.md`）。直接消费
`phase2-train-visualplanning-refined-v4/training/{train,val}.jsonl` 中已经冻结的
system + ordered image blocks + raw question + assistant target JSON，不重新解释
`datasets/` provenance，也不改变 split/样本纳入规则。processor/chat template 使用同一
Qwen3.5 checkpoint，`enable_thinking=False`；system/user/视觉与 generation prefix 的
labels 为 `-100`，assistant JSON + 结束 token 做标准 autoregressive token-mean CE；
`region_request.roi_xyxy` 坐标作为普通 JSON 数字 token 接受同一 CE，不增加 ROI head，
也不计算 L1/GIoU。所有 task 共享同一 LM head，不做默认 task weighting。

训练冻结完整 vision 与 LLM base，按 Qwen3.5 hybrid decoder 真实结构显式枚举
linear-attention 的 `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`、full-attention 的
`q/k/v/o_proj` 以及每层 MLP `gate/up/down_proj`；当前 32 层 9B checkpoint 共 248 个
LoRA target，PEFT `modules_to_save` 为空且不得出现 auxiliary ROI head。启动前硬审计
完整命中与 trainable 闭合，默认本地加载、BF16、gradient
checkpointing，输出 PEFT checkpoint/final adapter、参数审计和不含机器绝对模型身份的
训练 manifest。adapter config 用完整路径正则而非会被 PEFT 压缩的 projection 叶子名，
保证保存/重载后仍不命中视觉同名模块。Shell wrapper 通过可配置 Conda env 启动，不保存
SSH credential。resume 只接受同一 output dir 下完整的 Trainer/PEFT checkpoint，并按
manifest 比较 base config/data checksum、选择规模、LoRA 与全部权重影响参数，冲突拒绝。
