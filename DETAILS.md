# DETAILS.md — 当前架构与接口契约

本文件记录 M3 新架构**当前有效**的项目结构、模块职责、核心接口、运行产物、评测与报告契约。

它主要面向 AI 编码代理和需要理解系统内部结构的开发者。

- 编码行为边界：见 `AGENTS.md`
- Python 文件最终批准范围：见 `architecture/allowed_python_files.txt`
- 当前实现状态：见 `architecture/implementation_status.json`
- import DAG：见 `architecture/import_rules.json`
- 迁移历史：见 `docs/migration/`
- 重要架构决策：见 `docs/architecture/`

本文件不是开发日志。Task 00、11G.5、11J 等迁移过程编号不在这里作为长期接口维护。

---

## 1. 当前状态

文档整理基线：

```text
repository: CuFe-hust/M3
branch: new_structure
evaluation/reporting integration baseline:
  E1-E5 through 396a5900411930e16c75ff9fe18af01080c33428
legacy behavior reference:
  try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868
```

当前 `architecture/implementation_status.json` 中生产 Python 文件已全部实现，`pending_files` 为空。

E6 最终离线质量门记录：

```text
Full offline suite: 1648 passed
compileall: clean
git diff --check: clean
```

新架构已完成核心离线功能迁移与硬化。真实模型、真实数据集、云端 API 或目标 Spark/部署环境相关验证仍应按具体环境单独执行，不应把离线测试通过等同于所有 live gate 已通过。

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
- allowlist。

### `DETAILS.md`

回答：

> “当前系统是什么样，模块和接口怎么协作？”

### `architecture/`

机器约束：

```text
allowed_python_files.txt
implementation_status.json
import_rules.json
ALLOWLIST_CHANGE_POLICY.md
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
      | resolver/run  |
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
│   ├── task_resolver.py
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

完整 Python 路径不要复制维护在本文档中；以：

```text
architecture/allowed_python_files.txt
```

为唯一白名单来源。

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
count_target_hint
```

这是 `UnifiedSample` 的一等字段，不应把重要 task-normalization 信息重新塞回不透明 metadata。

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

注意：这和 pre-sample TaskResolver 的空问题规则不是同一个层次。

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
  -> TaskResolutionRequest
  -> TaskResolver
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

`XLRS-Bench-lite` 当前 registry 配置的 supported task 为：

```text
multiple_choice_vqa
```

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

## 14.3 专家模型运行层

`models/segformer_transformers.py` 提供两份本地 fine-tuned SegFormer 的共享
运行层：资产完整性校验、惰性单次加载、processor、device/dtype、预处理、
推理、logits 上采样、argmax 和明确的 `SegmentationResult`。默认纯本地，
不会自动下载。

iSAID 的 `classes.json` 是 checkpoint 权威类别映射；`config.json` 中的
`LABEL_0..15` 只是占位。OEM 旧资产没有额外 classes metadata，因此保留其
config 中的 `LABEL_0..8`，不猜测人类可读顺序。

YOLO 的新版 Counting 实现已经是旧版的加强版，仍保持唯一实现：
`YoloModelStore` 负责惰性单次加载和完整性校验，ONNX adapter 负责 provider、
预处理与解码，Counting backend 负责目标语义、切片、去重和 fallback。
通用 `models.base.validate_local_model_asset` 在模型 runtime 之前统一拦截 LFS
pointer；没有新增第二套 YOLO loader。

`agents/counting/backends/semantic_segmentation.py` 只消费 ExpertCatalog 明确批准的
`connected_components` capability：按 per-label confidence/area/morphology policy
将 semantic component centroid 转成共享 `GlobalPointObservation`，随后复用 owner-core
和 seam 去重形成 `CountingResult`。最终数量只由 accepted points 导出。该方式是
semantic instance approximation；相接对象会形成单一 component，可能低估数量，当前不做
watershed 或隐式 instance splitting。composition root 已按 `ExpertCatalog` 将 enabled
semantic expert 构造成 lazy client/backend，并接入固定优先级和 ordered fallback；OEM
因缺少 verified class map 保持 disabled。

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
```

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
```

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

`spatial_relation` 复用 general_vqa_v2 Prompt 与单次 Qwen 调用，输出
`agent_result.json`；保留 `requires_tiling=False` 与 VQA 确定性评测/Judge 族。

多选题 postprocess 会约束最终答案落在 choices 合法范围。

## 21.2 CaptionAgent

覆盖：

```text
caption
```

## 21.3 GroundingAgent

覆盖：

```text
grounding
```

completed 结果需要合法定位证据。

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

处理有序时相图对，包含 pair validation、单次 harmonization、proposal perception、artifact publish 与 rule-only review。默认关闭 semantic auxiliary path，保持 V1 difference proposal；启用时由 bootstrap 单独构造并注入抽象 `DenseSemanticClient`，按 low-level、PIF feature residual、confidence-weighted semantic difference 和 robust fusion 生成 V2 proposal。PIF validity 与 harmonization transform validity 是独立事实：PIF 同时满足原始 pixel-count/ratio gate 才可用于 feature normalization 与 robust threshold；transform 即使被拒绝，valid raw-derived PIF 仍可与 raw comparison pair 进入 V2。Proposal attention evidence 的附加不依赖 harmonized artifacts：rejected transform 的成功 V2 run 向 Qwen 发送 raw full T1/T2 与 proposal evidence，并省略不存在的 harmonized images。每个 V2 proposal 发布三路 score、融合图、二值 mask、crop-local mask/overlay 与相对路径清单；V2 实际消费 PIF 时，无论 `harmonization.save_artifacts` 如何设置，都必须发布 mandatory `pif_mask.png`，该开关只控制 optional harmonized artifacts。trace 记录实际 evidence roles/count、阈值来源、fallback、组件有效权重、PIF validity/usage、SegFormer logical identity、verified weight SHA 与算法设置，绝不记录 checkpoint 绝对路径或完整 feature/probability tensor。SegFormer 不消耗 Qwen budget；失败按显式策略稳定降级或在 Qwen 调用前终止。raw T1/T2 始终是语义事实依据，harmonized 图、SegFormer 输出和 proposal mask 仅用于注意力引导。Change runtime prompt 由 bootstrap 从 `PromptCatalog` 注入，运行时文本、版本、request hash 与 run prompt snapshot 共享同一权威资产。

2026-08-10 NVIDIA GB10 离线校准后，semantic path 的基准配置为 MiT-B2 iSAID、`feature_stage=1`（stride 8）、`tile_size=768`、`tile_overlap=64`、`local_match_radius=1`、`pif_threshold_k=4.5`，三路融合权重保持 `0.25/0.50/0.25`。该选择来自 stage/tile、20 对 LEVIR-CC、5×4 threshold/fusion 联合矩阵与 iSAID/OEM 对照；不是通用 benchmark 结论。`semantic.enabled` 仍默认关闭，live 校准不会改变 V1 行为。

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

# 23. TaskResolver

`TaskResolver` 是 workflow 服务，不是业务 Agent。

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

---

# 24. Low-confidence candidate fallback

当模型 task resolution 低于阈值：

- task 本身保持第一候选；
- 最多 3 个候选；
- 如果 top task 不是 `general_vqa`，候选表为 `general_vqa` 保留 fallback 槽；
- Resolver 只返回结构化候选；
- 真正执行由 SampleRunner 完成。

候选 fallback 不等于“多 Agent 全跑”。

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

## 25.4 Target parser

目标优先级：

```text
normalization.count_target_hint
  -> legacy metadata count_target_hint
  -> Qwen target parsing
```

invalid hint：

```text
InvalidCountTargetHintError
```

不得静默忽略。

解析后的 target 只和 `ExpertCatalog` 的 canonical label、显式 aliases、dataset-neutral
hints 匹配；VLM 不返回 backend/checkpoint 决策。

## 25.5 YOLO

YOLO 模型存储/adapter/ONNX 实现保持惰性和可选。

Python `YoloCountingSettings` 只保留 schema 与通用默认值：默认 `enabled=false`、
`detectors=[]`，不包含具体 checkpoint、labels 或 backend id。`ExpertCatalog` 是 capability、
logical identity、SHA、labels 与 priority 的事实源；`configs/local.yaml` 等部署配置是 enabled、
物理权重路径、provider/device 与阈值的事实源。只有显式部署 inventory 才注册 YOLO。
相对权重路径在 composition root 按 `project_root` canonicalize，绝对外部挂载路径保持不变；
物理路径不参与 catalog identity。权重与 ONNX Runtime 不自动下载，缺失时仍按通用 fallback
策略进入下一专家。

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

wheel 通过 package-data 携带 expert catalog、verified SegFormer 小型 metadata 与 prompts，
不携带 `.safetensors`、`.onnx` 或 `.pt` 大权重。
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
现为单次 general_vqa_v2 Prompt 调用，输出 `agent_result.json`；历史空间 run
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
harmonizer
difference_proposal
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
Qwen client
ExpertCatalog
lazy SegFormer clients by logical model id
Counting backend registry
AgentRegistry
TaskRouter
TaskResolver
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
create_model("qwen_transformers", ...)
```

在这里创建一次。

Agent 和 Workflow 共享这个实例。

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

TaskResolver 如果真的调用模型，会消费 Qwen budget。

显式 task 或 deterministic rule 不消费 resolver 模型调用。

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

不调用 TaskResolver。

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

走 SampleDraft → TaskResolver。

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

TaskResolver / canonical sample 决定的任务。

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
resolve/use task
build routing/attempt plan
run Agent
persist result
build deterministic evaluation
optional Judge
persist trace
persist final status
append execution index
```

职责：

- 单样本；
- attempt plan；
- task candidate fallback；
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
predictions.jsonl
dataset_summary.json
dataset_probe.json
```

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
                ├── <task>_evaluation.json
                ├── agent_trace.json
                └── optional model/judge artifacts
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
```

可选依赖保持惰性导入；无 caption record 时不导入，缺失时也不阻断 report。
这些本地 corpus 指标不等同于 benchmark official score。

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
- 调 TaskResolver；
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
backend_stages
model_calls
```

`task` 表示 execution task。

`ground_truth` 是 task-neutral 的只读投影，稳定包含 answers、count、boxes、
points、labels 与 coordinate_frame。`backend_stages` 只来自持久化的
`counting_attempts.json`，按真实 attempt 顺序展示；旧 run 缺少该文件时保持
空列表，不从 final result 或 trace 反推中间阶段。`model_calls` 只读取当前
sample 目录内已有的 request/raw/parsed/validation 产物，raw response 最多
8000 字符，且不暴露 artifact_dir、绝对路径、Base64 或 credential。

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

- 空问题走 deterministic rule；
- 有问题可以进入 TaskResolver；
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
allowlist
implementation status
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

# 78. Architecture allowlist

`architecture/allowed_python_files.txt`：

```text
final approved paths
```

`architecture/implementation_status.json`：

```text
what is actually implemented now
```

当前：

```text
pending_files = []
```

以后新增 Python 路径必须遵循独立架构批准流程。

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

### 79.4 Live validation

离线测试通过不代表：

- 本地 Qwen 权重已存在；
- DeepSeek key 可用；
- 真实 VRSBench/XLRS/MME/LEVIR 数据集存在；
- Spark 目标机可运行；
- GPU/ONNX/CUDA detector 实际可用。

必须分别验证。

### 79.5 Optional vision/deployment dependencies

变化检测、YOLO/ONNX 等扩展能力依赖环境配置，基础 import 不等于所有 backend 都 available。

### 79.6 OEM class labels

迁移来源只提供 OEM 9-channel checkpoint 和占位 `LABEL_0..8`，没有经训练
语义验证的 `classes.json`。当前 runtime 保留该事实，不用网络资料猜测类别顺序。

### 79.7 Semantic connected-component counting

SegFormer 输出 semantic region 而非 instance mask。相接实例可能形成一个 component 并
低估数量；当前不隐藏加入 watershed 或 instance splitting。此限制应在 benchmark 中单列。

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
workflows/task_resolver.py
tests/routing/
tests/workflows/test_task_resolver.py
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
| 无 task 样本的任务判定 | `workflows/task_resolver.py` |
| 单样本 orchestration | `workflows/sample_runner.py` |
| 数据集 orchestration | `workflows/dataset_runner.py` |
| deterministic metric | `evaluation/metrics/` |
| Judge | `evaluation/judges/` + `workflows/judge_service.py` |
| external official evaluator | `evaluation/standard/` 或 `evaluation/datasets/` |
| 报表 | `reporting/` |
| CLI use case | `application/commands/` |
| dependency assembly | `application/bootstrap.py` |
| 顶层命令参数 | `main.py` |

如果目标文件不在 allowlist，先做架构批准，不直接新建。

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
  -> optional TaskResolver
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
