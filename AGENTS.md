# AGENTS.md

本文件定义 AI 编码代理在本仓库中读取、修改、验证和汇报代码时必须遵守的长期规则。

本仓库是“面向太空智算的多模态遥感大模型应用探索”项目的当前架构实现。长期架构以仓库主线实现及本文件、`DETAILS.md` 和 `architecture/` 中的约束为准；迁移行为参考为 `try_yolo` 的锁定提交 `ec962eb87c3ad0b8c1502efcbd08db0daec48868`。历史分支只用于行为对照、Golden fixtures 与迁移审计，不作为日常开发的第二套架构来源。

本文档主要面向 Claude Code、Codex 等编码代理。当前项目结构、接口与运行契约见 `DETAILS.md`；机器可检查的架构约束见 `architecture/`；迁移历史见 `docs/migration/`。

---

## 0. 最高优先级原则

修改本仓库时，优先保证以下事项：

1. **不得破坏统一内部样本契约 `data.schema.UnifiedSample`。**
2. **不得破坏已持久化运行产物、resume 语义与结果路径安全契约。**
3. **不得未经明确授权改变评测指标、Ground Truth 解释、数据集 split、样本纳入规则或官方评测适配方式。**
4. **不得未经明确授权改变主模型、processor/tokenizer、checkpoint 加载语义或逻辑模型身份。**
5. **保持确定性评测与 Judge 解耦；Judge 永远不得覆盖确定性指标。**
6. **保持新架构包边界；不得重新引入 `spacers_agent/` 或旧 `eval/`。**
7. **只做当前任务所需的最小修改，不顺带进行无关重构。**
8. **默认离线；除显式联网命令或用户明确要求外，不下载模型、不下载数据集、不调用云 API。**
9. **测试、错误状态、失败样本和未验证项必须如实记录，不得通过跳过、过滤或伪造结果使输出看起来更好。**
10. **可复现性、评测可比性、路径安全和敏感信息安全高于形式上的代码整洁。**

如果“更漂亮的实现”和以上契约冲突，优先保留契约。

---

## 1. 指令优先级与事实来源

### 1.1 指令优先级

默认优先级如下：

1. 当前用户对本任务的明确要求；
2. 更靠近目标文件的局部 `AGENTS.md`（如果未来存在）；
3. 根目录 `AGENTS.md`；
4. `DETAILS.md` 中记录的当前接口事实；
5. 其他说明文档。

局部规则可以增加更严格的约束，但不得静默弱化本文件关于数据契约、评测、路径安全、敏感信息、架构边界和验证真实性的规则。

### 1.2 当前事实来源

不同文件承担不同职责：

| 来源 | 职责 |
|---|---|
| `AGENTS.md` | 编码代理的长期行为规则 |
| `DETAILS.md` | 当前有效的架构、接口、运行与评测事实 |
| `architecture/allowed_python_files.txt` | 最终批准的 Python 路径白名单 |
| `architecture/implementation_status.json` | 当前实际实现状态 |
| `architecture/import_rules.json` | 顶层包 import DAG 与 path-specific 依赖规则 |
| `architecture/ALLOWLIST_CHANGE_POLICY.md` | 白名单变更政策 |
| `tests/fixtures/migration/` | 离线 Golden 行为基线 |
| `docs/migration/` | `try_yolo` → 新架构的迁移基线与有意行为差异 |
| `docs/architecture/` | 重要架构决策、收口说明和运行门禁 |
| 生产代码 + 测试 | 最终可执行事实 |

如果文档与代码明显不一致，不要擅自选择一个继续实现。应先确认当前 HEAD、相关测试和机器约束文件，修复文档漂移或按当前任务要求处理。

### 1.3 `try_yolo` 的定位

`try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868` 是只读行为参考，不是新架构的 import 来源。

允许：

```text
git show ec962eb87c3ad0b8c1502efcbd08db0daec48868:<path>
```

禁止：

- 从旧包直接 import 实现；
- 新建兼容层把新代码转发回旧代码；
- 因为旧代码存在某个目录，就在新架构中机械复刻该目录；
- 修改 Golden fixtures 来迁就新实现。

如果有意改变旧行为，应在 `docs/migration/` 中明确记录“为什么改变、是否影响历史结果可比性、哪些测试/fixture 被更新”。

---

## 2. 架构白名单与文件范围

### 2.1 白名单是硬约束

`architecture/allowed_python_files.txt` 冻结整个新架构最终批准的 Python 路径。

必须理解：

- 白名单不是“当前存在文件列表”；
- 当前存在情况以 `architecture/implementation_status.json` 为准；
- 白名单中的未创建路径只是已批准未来路径；
- 不得为“结构看起来完整”提前创建空壳；
- 普通实现任务不得修改白名单。

如果任务需要新增一个不在白名单中的 `.py`：

1. 停止创建该文件；
2. 报告缺失路径以及为什么现有职责无法容纳；
3. 等待用户批准独立架构变更；
4. 白名单变更与业务实现不得混在同一个普通实现步骤中。

不得为了绕过此限制创建或泛化：

```text
utils.py
helpers.py
manager.py
compat.py
legacy.py
common.py
```

除非用户明确批准新的职责边界。

### 2.2 永久禁止旧包

以下路径永久禁止重新出现：

```text
spacers_agent/**
eval/**
```

新代码不得：

- import `spacers_agent`;
- import 旧 `eval`;
- 使用动态 import 回退到旧包；
- 用 `try/except ImportError` 在新旧实现间切换；
- 修改 `sys.path` 绕过包边界。

---

## 3. 顶层包依赖边界

实际机器规则以 `architecture/import_rules.json` 和 `tests/architecture/test_import_boundaries.py` 为准。

长期原则如下。

### 3.1 `data`

职责：数据契约、dataset adapter、数据选择、验证、显式下载与便利加载。

约束：

- `data` 不依赖 `agents`、`routing`、`workflows`、`evaluation`、`reporting`、`application`；
- dataset adapter 不调用模型；
- adapter 不写 run artifacts；
- adapter 不根据问题文本偷偷执行 Agent 路由；
- 数据层只负责把源数据只读转换成 `UnifiedSample` 或 `SampleDraft`。

### 3.2 `models`

职责：模型协议、模型缓存、图像输入工具、配置声明、统一模型 factory 与具体模型实现。

约束：

- `models` 不依赖数据集实现；
- `models.entry.create_model(...)` 是主流程模型统一构造入口；
- import `models.entry` 不得触发权重加载；
- 具体模型实现不得被领域层自行选择。

### 3.3 `agents`

职责：执行具体遥感任务工作流。

允许依赖：

- `data.schema`;
- `models.base`;
- `models.images`;
- `agents` 内部模块。

禁止：

- import `application`;
- import `workflows`;
- import `models.entry`;
- import `models.qwen_transformers`;
- import `models.qwen3_*` 具体实现；
- 读取具体数据集原始 JSON；
- 控制整个 dataset loop；
- 修改评测指标。

Agent 依赖模型协议，而不是具体模型类。

### 3.4 `routing`

职责：对**已知 task** 做确定性 Agent 路由。

约束：

- 不读取 question 决定 task；
- 不调用模型；
- 不读 dataset adapter；
- 不依赖 workflows/application；
- 未知 task 显式失败，不猜 `general_vqa`。

### 3.5 `workflows`

职责：TaskResolver、SampleRunner、DatasetRunner、预算、状态、run store、artifact writer、judge service 等编排。

约束：

- 可以消费 `data` / `agents` / `routing` / `evaluation` 和 `models.base` 协议；
- 不直接 import `models.entry` 或具体 Qwen 实现；
- 不成为第二个 composition root；
- 不把 dataset-specific 原始解析逻辑塞进 workflow。

### 3.6 `evaluation`

职责：统一 EvaluationRecord、确定性指标、Judge 客户端与标准评测 seam。

约束：

- `evaluation.metrics/**` 不依赖任何模型；
- `evaluation.judges/**` 只能依赖批准的模型契约/缓存/配置；
- evaluation 不 import `application`；
- evaluation 不选择具体主模型实现；
- metric 不调用 Agent；
- Judge 结果不覆盖 deterministic metrics。

### 3.7 `reporting`

职责：从已持久化执行索引和产物构建报告及导出。

约束：

- reporting 是只读结果层；
- 不调用模型；
- 不执行 Agent；
- 不 import Agent 实现类；
- 不 import `workflows.sample_runner`;
- 不为了生成报告重新推理；
- 不修改执行产物；
- 不相信不安全的任意 result path。

### 3.8 `application`

`application` 是唯一 composition root。

只有这里可以：

- 读取应用配置；
- 选择具体模型；
- 创建 Qwen/DeepSeek 客户端；
- 组装 registries / workflows / reporting；
- 将 CLI/HTTP 命令接到 use case。

### 3.9 `main.py`

`main.py` 是唯一受支持的顶层 CLI surface。

硬规则：

- 内部包只允许 import `application`;
- 不写模型业务逻辑；
- 不写 Agent fallback；
- 不写 dataset loop；
- 不写评测计算；
- 不写报告聚合；
- 新公共命令应实现于 `application/commands/`，`main.py` 仅解析参数并委托。

---

## 4. `__init__.py` 规则

`__init__.py` 只做导出与类型辅助：

允许：

- module docstring；
- import/re-export；
- `__all__`;
- `TYPE_CHECKING`。

禁止：

- 定义业务函数或业务类；
- 模型加载；
- 数据集注册副作用；
- 条件注册；
- 文件系统访问；
- 网络访问；
- import 时执行昂贵逻辑。

---

## 5. 统一内部样本契约

### 5.1 `UnifiedSample` 是内部 canonical sample

新架构内部跨模块传递的数据样本必须使用：

```text
data.schema.UnifiedSample
```

其核心字段为：

```text
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

不得用 dataset-specific `dict` 绕过该契约。

### 5.2 `SampleDraft`

只有“样本本身没有明确 task，需要在物化前解析”的路径使用：

```text
data.schema.SampleDraft
```

正确流程：

```text
SampleDraft
  -> TaskResolver
  -> materialize_sample(...)
  -> UnifiedSample
```

`UnifiedSample.task` 始终必填。不得把 `task=None` 的半成品伪装成 UnifiedSample。

### 5.3 图像路径

`ImageRef.path`：

- 必须是相对 dataset root 的路径；
- 不允许绝对路径；
- 不允许 `.` / `..` 逃逸；
- 序列化统一使用 `/`;
- 本地机器绝对路径不得进入 sample id、模型逻辑身份或可移植产物。

运行时通过 `AgentContext.data_root` 显式解析相对路径。

### 5.4 图像角色

变化类任务：

```text
change_caption
change_qa
```

必须：

```text
t1, t2, [context...]
```

其他任务必须：

```text
image, [context...]
```

不得在 Agent 内静默纠正错误角色；应在物化/Schema 边界失败。

### 5.5 JSON-safe

以下自由字段必须保持严格 JSON-safe：

- `UnifiedSample.metadata`;
- `GroundTruth.raw`;
- `TaskNormalization` 结构化字段；
- Agent trace；
- additional results；
- request context。

不得放入：

```text
Path
PIL.Image
bytes
bytearray
set
callable
NaN / Infinity
```

### 5.6 Ground Truth

Ground Truth 是源标注的只读保留。

不得：

- 为提高指标修改 GT；
- 改写 source annotations；
- 丢掉失败样本；
- 将不明确的坐标系猜成某个坐标系；
- 将 polygon 静默转换成 xyxy 并声称与官方指标等价。

---

## 6. TaskResolver 与 TaskRouter

这是新架构的核心职责分离，禁止重新合并。

### 6.1 TaskResolver

`workflows.task_resolver.TaskResolver` 回答：

> “这是什么任务？”

解析优先级：

```text
explicit task
  -> deterministic rule
  -> model resolution
```

具体规则：

- 显式合法 task：直接使用，零模型调用，零 resolver budget；
- 显式非法 task：稳定失败；
- 空 question + 1 图：`caption`;
- 空 question + 2 图：`change_caption`;
- 其他空 question：稳定失败；
- 只有缺少 task 且 question 非空时，才允许一次模型解析路径。

低置信度时 Resolver 只返回候选任务信息，不执行 Agent。

### 6.2 TaskRouter

`routing.TaskRouter` 回答：

> “一个已经确定的 task 应交给哪个 Agent？”

Router：

- 同步；
- 确定性；
- 不读 question；
- 不调用模型；
- 不猜未知 task。

### 6.3 候选任务 fallback

低置信度候选的实际执行属于 `SampleRunner`，不属于 Resolver 或 Router。

不得：

- Resolver 自己跑 Agent；
- Router 自己跑模型；
- Agent 自己重新分类任务；
- dataset adapter 调 Qwen 决定业务 task；
- 无依据地“全部 Agent 都跑一遍”。

---

## 7. Task 与 Agent 契约

公开 task 集合以 `data.schema.TaskName` 和 `routing/policies.py` 为准。

当前任务族：

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

Agent 必须实现统一协议：

```python
async def run(
    sample: UnifiedSample,
    context: AgentContext,
) -> AgentExecution:
    ...
```

并声明：

```text
name
supported_tasks
```

`AgentRegistry` 必须覆盖全部可路由任务。

---

## 8. `AgentContext` 与 `AgentExecution`

### 8.1 `AgentContext`

`AgentContext` 只包含单样本执行所需的轻量依赖。

不得保存：

- API key；
- Base64 image；
- 模型权重；
- 完整 `AppSettings`;
- 完整 `PromptCatalog`。

### 8.2 `AgentExecution`

`AgentExecution` 是运行时包装，不是新的持久化全局 Prediction Schema。

必须满足：

- `result_filename` 是跨 POSIX/Windows 安全的纯 basename；
- additional result filename 也是纯 basename；
- 主结果名与附加结果名不得冲突；
- trace 和 additional results 严格 JSON-safe；
- trace 不得包含 token/key/password/authorization/base64/private key 等敏感内容；
- payload 如果暴露 `agent_name`，必须与 execution.agent_name 一致。

---

## 9. 计数子系统的稳定边界

计数是独立领域流水线，不得退化为大量 dataset-specific `if/else`。

长期契约：

- `CountingAgent` 的主 payload 是 `CountingResult`;
- 主结果文件名为 `counting_result.json`;
- 如果需要 `AgentResult`，只能作为附加结果；
- Backend 类型使用显式 `BackendKind`，不得从类名或字符串猜类型；
- `qwen_point`、`quantity_proposal`、`semantic_segmentation`、`yolo_obb` 的职责保持分离；
- selector 只规划，不偷偷吞掉不可用 detector；
- executor 负责运行时 fallback 与执行状态；
- YOLO 模型加载必须保持惰性；
- detector 不得因为依赖缺失而让普通 import 崩溃；
- seam / tile 几何与去重应保持确定性；
- 新增 expert 以 `ExpertCatalog`、资产和 composition 配置为主，不在 `CountingAgent` 增加 model-specific 分支；
- VLM 只解析 `CountTargetSpec`，不得选择 backend、checkpoint 或模型路径；
- 启用的 SegFormer expert 必须使用已验证 class map，不得从 `LABEL_N` 占位标签猜语义。

计数目标解析优先级：

```text
sample.normalization.count_target_hint
  -> metadata["count_target_hint"] 兼容路径
  -> Qwen target parser
```

无效 hint 应明确失败，不得静默忽略后继续猜目标。

---

## 10. 模型构造与模型身份

### 10.1 统一模型入口

主流程模型通过：

```python
models.entry.create_model(name, **kwargs)
```

构造。

当前模型名以 `models.entry.ModelName` 为准。

新增主流程模型：

- 在 `models/entry.py` 注册 builder；
- builder 使用惰性 import；
- 不得让 `import models.entry` 加载 torch/transformers/权重；
- 具体模型的选择只能由 `application` composition root 完成。

### 10.2 单次组装

一次 runtime assembly：

- Qwen 客户端只创建一次或由测试注入；
- 后续 Agent/Workflow 共享该客户端；
- 不允许每个 Agent、每条样本、每个 HTTP 请求重复加载模型。

### 10.3 Cache identity

任何进入可恢复模型缓存的调用都必须有完整逻辑模型身份。

本地 checkpoint 路径不得直接成为可持久化逻辑身份。

如果 `settings.models.qwen.model` 是本地路径，则必须提供安全、与机器路径无关的：

```text
cache_model_id
```

request hash 必须继续覆盖影响结果的关键输入，例如：

- logical model identity；
- generation settings；
- prompt version/content；
- messages；
- image digest；
- response schema；
- client version；
- model revision。

不得通过减少 hash 输入制造错误 cache hit。

---

## 11. 网络、下载与外部服务

默认离线。

### 11.1 Qwen

`QwenSettings.allow_download=False` 是默认值。

除非：

- 用户明确要求联网加载；
- 或使用明确设计的下载/联网命令；

否则不得触发 Hugging Face 自动下载。

### 11.2 数据集

普通 dataset loading 不得隐式联网。

显式自动下载入口是：

```text
python main.py download-data ...
```

不得因为本地数据缺失就在 adapter/loader 中偷偷下载。

### 11.3 DeepSeek Judge

DeepSeek 是可选 Judge，不是主业务模型。

规则：

- 无 API key 时 Judge 禁用并退化为纯确定性评测；
- secret value 不进入 settings、snapshot、trace 或 artifact；
- settings 只声明 `api_key_env` 名称；
- 只有 composition root 读取 key value 并直接注入客户端；
- 非 live/非 judge 任务不得因为 DeepSeek 不可用而失败。

---

## 12. Run、Artifact 与路径安全

### 12.1 RunStore

创建 run 必须先写可复现元数据，且创建 run 本身不得调用模型。

核心 run 级产物包括：

```text
manifest.json
config.snapshot.json
prompts.snapshot/
events.jsonl
run_request.json
predictions.jsonl
report/
tasks/
```

### 12.2 `run_request.json`

`run_request.json` 是 resume 重建**具体调用参数**的权威来源。

不得在 resume 时通过以下信息猜原调用：

- 当前 config；
- CLI 默认值；
- 目录名；
- summary；
- 新代码的默认参数。

如果持久化 run request 与新的 resume 请求冲突，应稳定拒绝，而不是悄悄改成新参数继续跑。

### 12.3 Sample 状态

`SampleRunStatus.state` 的稳定集合：

```text
pending
running
succeeded
partial
failed
skipped
```

所有被选择样本最终必须落入终态：

```text
succeeded / partial / failed / skipped
```

Dataset summary 必须闭合：

```text
total == succeeded + partial + failed + skipped
```

### 12.4 Result path

`status.json.result_path`：

- 是 sample-relative 的纯 basename；
- 不允许 absolute / drive / UNC / `..` / nested path；
- legacy 绝对路径不能继续被信任。

`predictions.jsonl.result_path`：

- 是 run-relative 展示/索引路径；
- 不得包含机器绝对路径；
- reporting 不得把它作为任意磁盘读取权限。

### 12.5 `predictions.jsonl`

这是 append-only execution index。

当前状态：

```text
(run_task, sample_id) 的最后一行
```

resume 历史不得通过覆盖整个索引而丢失。

### 12.6 原子写入

结构化 JSON/JSONL 产物应继续使用统一原子写入原语。

不得让进程中断后留下半个 JSON。

当前 JSONL 并发承诺是**单 Python 进程内**安全；不得未经验证声称支持跨进程并发 append。

---

## 13. SampleRunner 与 DatasetRunner

### 13.1 SampleRunner

`SampleRunner` 只负责一条样本：

```text
routing
-> attempt plan
-> Agent execution
-> result artifact
-> deterministic evaluation
-> optional judge
-> trace
-> final status
```

不得在 SampleRunner 内实现 dataset 读取循环。

低置信度候选最多按现有契约执行有限候选，不能扩成无界多 Agent 搜索。

### 13.2 DatasetRunner

`DatasetRunner` 负责编排数据集：

- adapter probe；
- sample selection；
- shard；
- resume；
- concurrency；
- fail-fast；
- 每样本 SampleRunner；
- prediction index；
- task summary。

选择顺序与 shard 算法属于可复现行为，不得随意变化。

分片必须继续使用稳定 hash，而不是 Python 进程随机 hash。

### 13.3 auto-task

`DatasetRunOptions.auto_task=True` 是显式模式。

语义区分：

```text
tasks=None
    adapter 默认任务集合，不调用 TaskResolver

tasks=(...)
    显式任务模式

auto_task=True + tasks=()
    SampleDraft -> TaskResolver 模式
```

不得把 `tasks=None` 偷偷解释成 auto-task。

---

## 14. Resume 契约

resume 是高风险区域。

默认规则：

- `succeeded` 样本不重复推理；
- 如果允许补评测/Judge，应只补缺失或损坏的对应产物；
- partial/failed/running/pending/缺失或损坏状态按当前明确契约重跑；
- resume 使用实际 `status.task` 判断执行任务指标族；
- 不得用 canonical resolved task 覆盖实际 execution task；
- 旧的危险绝对 result_path 视为无效，不继续信任；
- 原 fresh run 的调用选项以持久化 `run_request.json` 为准。

任何改变 resume 行为的修改都必须有专门测试。

---

## 15. 评测规则

`evaluation/` 是高风险目录。

未经任务明确要求，不得改变：

1. metric 定义；
2. Ground Truth 解释；
3. 坐标系定义；
4. dataset split；
5. 样本过滤与纳入/排除规则；
6. failed/skipped 的统计语义；
7. aggregation；
8. reference answer 读取；
9. external/official evaluator 输入映射；
10. 历史结果可比性。

### 15.1 确定性指标

统一持久化契约是：

```text
evaluation.records.EvaluationRecord
```

当前 canonical deterministic families：

```text
counting
general_vqa
grounding
caption
```

task 与 deterministic metrics 类型必须匹配。

### 15.2 Judge

Judge 只作为附加审计信号。

必须保持：

```text
deterministic metrics
        +
optional judge status/result
```

禁止：

```text
judge result -> 覆盖 deterministic metric
```

Judge 调用失败时：

- 不应抹掉已成功的确定性结果；
- 应记录稳定失败状态；
- 不保存 credential；
- 不把原始内部异常文本传播到公共 artifact。

### 15.3 Grounding

当前确定性 grounding 指标是严格 fail-closed 的。

只有在实现明确支持且 prediction/GT 坐标契约兼容时计算。

不得：

- 未经显式坐标转换就把 source-pixel 坐标当 normalized；
- 把 8 点 polygon 静默当 4 值 xyxy；
- 为了“有个分数”而猜坐标系。

### 15.4 Caption

逐样本 EvaluationRecord 保存 candidate + references。

语料级 BLEU / METEOR / ROUGE / CIDEr 属于 aggregate/可选标准评测能力，不得把逐样本记录伪装成完整语料级分数。

### 15.5 Official / external evaluator

外部标准评估结果必须使用独立 namespace，例如：

```text
external_standard
```

不得无说明地合并进内部 deterministic metric 名称。

---

## 16. Reporting 规则

Reporting 是执行后的只读层。

允许：

- 读取 `predictions.jsonl`;
- 根据冻结身份定位样本目录；
- 读取 status/sample/trace/evaluation/result；
- 聚合已持久化 EvaluationRecord；
- 生成 JSON/CSV/HTML/audit/export。

禁止：

- 调模型；
- 重新执行 Agent；
- 重新决定 task；
- 修改 sample/status/evaluation；
- 通过报告层重新计算一个与持久化指标不同的“更好指标”；
- 信任任意用户/模型提供的绝对路径；
- 输出 secret。

统一报告 bundle 放在：

```text
runs/<run_id>/report/
```

报告生成不应改变执行结果。

---

## 17. Application 与 CLI 规则

### 17.1 唯一公共入口

公共命令从：

```text
python main.py ...
```

进入。

不要新增第二套独立公共 CLI。

### 17.2 新命令

新公共命令：

1. 在 `application/commands/` 中实现薄 use-case adapter；
2. 在 `main.py` 注册参数；
3. 复用现有 Runtime / Workflow / Reporting；
4. 不复制 dataset loop、模型构造或评测逻辑。

### 17.3 HTTP

服务层只负责输入校验、调用 Runtime 与稳定错误响应。

不得：

- 每个 HTTP 请求重新构造模型；
- 在 handler 中复制 Agent 路由；
- 暴露原始异常、API key 或绝对内部路径。

---

## 18. 配置规则

配置入口：

```text
application.settings.AppSettings
```

规则：

- Pydantic `extra="forbid"` 语义不得随意弱化；
- 配置中的行为参数应有稳定默认值；
- 环境变量覆盖由 application settings 统一处理；
- secret value 不属于 settings；
- 本地绝对 checkpoint 路径不得作为逻辑模型身份；
- 不在代码中硬编码个人机器绝对路径；
- 临时实验不得无说明修改默认生产配置。

配置行为改变时，必须更新测试和 `DETAILS.md`。

---

## 19. Dependency 规则

- 不为方便随意添加第三方依赖；
- 优先标准库和仓库现有依赖；
- 可选硬件/视觉依赖保持可选；
- 缺少 YOLO/ONNX/cv2 等可选依赖时，普通 base import 不应崩溃；
- 不未经授权升级 PyTorch、Transformers、Pydantic、ONNX Runtime 等关键依赖；
- 新依赖必须说明：
  - 为什么需要；
  - 哪个模块使用；
  - 是否可选；
  - 对离线测试和部署的影响。

---

## 20. 敏感信息与日志

永远不得提交或持久化：

- API key；
- Authorization header；
- password；
- private key；
- credential；
- Base64 image payload；
- 含 secret 的完整环境变量 dump。

错误 artifact 和 trace 应优先使用：

```text
稳定 error code / 类型名
```

而不是：

```text
原始异常全文
```

尤其不要把可能包含主机路径、HTTP body、token 或模型原始敏感响应的异常字符串直接写入 `status.json`、trace 或报告。

---

## 21. 数据、权重、缓存和大型文件

默认不得提交：

```text
datasets/
dataset/
data/raw/
raw_data/
checkpoints/
weights/
outputs/
runs/
logs/
wandb/
tensorboard/
.cache/
huggingface/
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.onnx
*.om
*.engine
*.npy
*.npz
*.log
*.tar
*.zip
*.7z
```

如果当前任务生成了新类型的本地产物，应检查 `.gitignore` 是否需要更新。

测试 fixture 中明确批准的小型静态数据不适用“盲目全部忽略”，但新增前必须确认用途和体积。

---

## 22. 测试规则

### 22.1 不得弱化测试

禁止：

- 删除失败测试来让 CI 变绿；
- 用 `pytest.skip` 掩盖缺失实现；
- 修改 Golden fixture 迎合错误行为；
- 放宽断言直到错误不再被发现；
- 只运行一个与修改无关的测试然后声称已验证。

### 22.2 修改哪些区域必须有测试

以下修改通常必须新增或更新 pytest：

- `UnifiedSample` / `SampleDraft`;
- dataset adapter；
- Router / Resolver；
- Agent 输入输出；
- counting backend / selector / executor；
- model entry / model cache identity；
- run/artifact/resume；
- deterministic evaluation；
- Judge；
- reporting；
- config parsing；
- CLI；
- bug fix；
- import boundary / allowlist 行为。

### 22.3 架构测试

修改跨模块依赖、文件布局或 `__init__.py` 时，应特别运行：

```text
tests/architecture/test_allowed_python_files.py
tests/architecture/test_implementation_status.py
tests/architecture/test_import_boundaries.py
tests/architecture/test_init_side_effects.py
tests/architecture/test_package_discovery.py
tests/architecture/test_no_new_to_legacy_imports.py
```

### 22.4 迁移 parity

涉及与 `try_yolo` 行为对齐时，应优先使用：

```text
tests/fixtures/migration/
tests/parity/
docs/migration/
```

而不是普通 Git diff。

### 22.5 真实性

无法运行某项验证时必须明确说明：

- 未运行的命令；
- 原因；
- 已执行的替代检查；
- 剩余风险。

绝不写“all tests passed”，除非确实运行了对应集合。

---

## 23. 文本、编码和注释

- 所有文本文件 UTF-8；
- 尽量保留现有换行风格；
- 不因为编辑器自动格式化制造全文件 diff；
- 新增或修改的代码注释使用英文 + 中文双语；
- 英文在前、中文在后；
- 注释解释设计意图，不机械复述每行代码；
- 不要求顺便翻译与当前任务无关的旧注释。

示例：

```python
# Keep the router deterministic and model-free.
# 保持 Router 确定性且不依赖模型。
```

---

## 24. 最小修改原则

每个任务只修改完成需求所必需的内容。

禁止：

- 顺手全仓格式化；
- 顺手重命名大量模块；
- 无关 import sorting；
- 没有需求的“架构优化”；
- 为未来可能需要的功能提前做抽象；
- 同时修复无关 bug；
- 将一个局部修复扩成大规模重写。

如果现有稳定函数可以复用，不创建平行实现。

---

## 25. 文档维护

### 25.1 `DETAILS.md`

`DETAILS.md` 是当前架构与接口事实手册。

当修改以下内容时必须同步更新：

- package 职责；
- `UnifiedSample` / `SampleDraft`;
- task 集合；
- task → Agent 映射；
- Router/Resolver 规则；
- Agent 输入输出；
- model entry；
- run directory；
- artifact filename；
- resume semantics；
- evaluation；
- reporting；
- CLI；
- config；
- dataset registry；
- known limitations。

不要把临时调试过程、Task 编号流水账和逐 commit 历史塞进 `DETAILS.md`。

### 25.2 `docs/migration/`

只记录迁移与 parity：

- 旧行为基线；
- intentional differences；
- Golden fixture 解释；
- 历史结果可比性。

### 25.3 `docs/architecture/`

记录架构决策及其原因，不重复维护完整 `DETAILS.md`。

### 25.4 README

README 面向普通使用者。

如果安装、数据准备、公开运行命令、模型准备方式发生变化，应同步更新 README。

---

## 26. Git 与任务工作流

除非用户明确要求直接在当前分支修改，否则不要自行改变用户的 Git 工作流。

对编码任务，至少应：

### 修改前

```text
git status --short
git rev-parse HEAD
```

然后：

1. 阅读 `AGENTS.md`;
2. 阅读 `DETAILS.md`;
3. 阅读任务相关代码；
4. 查看相关测试；
5. 如果是 parity 任务，再读对应 migration fixture/docs。

### 修改中

- 保持 diff 局部；
- 不触碰用户已有无关修改；
- 不修改 Golden fixture 逃避问题；
- 不擅自改白名单；
- 不引入旧包回退。

### 修改后

至少检查：

```text
git diff --check
git status --short
```

并运行任务相关 pytest。

涉及架构边界时补充 architecture tests；涉及行为 parity 时补充 parity tests。

---

## 27. 最终汇报要求

任务完成后，编码代理的最终汇报应包含：

1. 修改了什么；
2. 为什么这样修改；
3. 修改文件；
4. 实际运行了哪些测试/检查；
5. 测试结果；
6. 未运行什么以及原因；
7. 是否影响：
   - UnifiedSample；
   - task/routing；
   - model interface；
   - evaluation；
   - report；
   - CLI；
   - resume；
8. 已知风险或后续项。

不要声称不存在尚未验证的问题。

---

## 28. 常见禁止模式

以下做法默认视为架构回退：

```text
Router 读取 question 后直接调用 Qwen 决定 Agent
Agent 直接读取 VRSBench/MME 原始 JSON
Workflow import models.qwen_transformers
每条样本 create_model(...)
Reporting 调模型补答案
Evaluation 为了提高分数过滤 failed 样本
Resume 根据当前配置猜原来的参数
Adapter 数据缺失时自动联网下载
将本机绝对路径写进 sample/result/trace
新建 spacers_agent 兼容层
新建 eval 兼容层
用 sys.path 或动态 import 绕过 import DAG
把所有公共命令逻辑塞回 main.py
```

出现上述需求时，应先重新审视模块职责，而不是直接实现。

---

## 29. 文档基线

本版规则按照仓库主线的当前长期架构整理，重构迁移期的 Task 编号不再作为日常开发接口的一部分。

旧行为仍可通过：

```text
try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868
docs/migration/
tests/fixtures/migration/
tests/parity/
```

进行审计。

未来代码改变事实时，应更新 `DETAILS.md`；只有规则本身发生变化时才更新本文件。
