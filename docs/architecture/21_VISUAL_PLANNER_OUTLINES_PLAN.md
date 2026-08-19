# 21 — Visual Planner Outlines 结构化解码接入计划

> Status: **implemented offline / live compatibility gate pending**
>
>
> 状态：**离线实现已完成，真实兼容性门禁待环境具备后验证**
>
> Baseline: `aa18b14869e5402a0374b07723044135b51b8255`，并假定 Doc 20 的
> `visual-task-plan-v5` 量化 ROI 方案已经完成。当前工作树中的 v5 ROI 修改属于用户现有
> 工作，实施本计划时必须保留。
>
> Review decision（2026-08-18）：用户确认方案可行。后续实现必须保持单份 Qwen
> model/processor、Outlines 仅用于第一次 Visual Planner 调用、子 Agent 保持 native
> generation，以及 one-generation/no-repair/fail-closed 语义。

## 1. 目标

为每条样本的第一次 Qwen 调用——`VisualTaskPlanner`——增加 Outlines JSON Schema
constrained decoding，使模型在生成过程中就受到 `VisualTaskPlan v5` 输出格式约束。

当前典型任务至少包含两次主模型调用：

```text
1. VisualTaskPlanner
   -> VisualTaskPlan v5
   -> task / categories / count_target / optional ROI

2. selected Agent
   -> final answer / task-specific result
```

Counting tile、empty review、quantity proposal、Grounding evidence 等路径可能继续产生更多
Qwen 调用。本计划必须保证 Outlines 只作用于第一次 planner 调用，不改变这些子 Agent
主模型调用的解码方式。

## 2. 当前调用边界

application composition root 当前只创建一个 `QwenTransformersClient`，并将同一个 client
注入 planner 和所有需要 Qwen 的 Agent/Backend：

```text
application.bootstrap
  -> one QwenTransformersClient
       ├─ VisualTaskPlanner
       ├─ GeneralVQAAgent
       ├─ GroundingAgent
       ├─ CaptionAgent
       ├─ ChangeAgent
       └─ Counting backends / review callbacks
```

所有结构化调用最终进入：

```text
QwenTransformersClient.complete_json(...)
  -> _generate(...)
  -> model.generate(...)
```

client 内的单个 `_generation_lock` 会序列化实际生成；cache hit 不占用该锁。

因此，若把 Outlines 无条件加入公共 `_generate()`，所有子 Agent 都会被改变。这不是本计划
允许的实现。

## 3. 结论：对子 Agent 的影响

### 3.1 推荐实现下没有直接解码影响

Outlines 必须实现为逐请求选择的 constrained decoding policy：

```text
one shared Qwen model + processor + generation lock
  ├─ planner request
  │    decoding = outlines_json_schema
  │    schema = VisualTaskPlan.model_json_schema()
  │
  └─ Agent requests
       decoding = native
       existing model.generate() behavior
```

在此结构下：

- 子 Agent 不接收 Outlines logits processor；
- 子 Agent 的 prompt、response schema、generation kwargs 和解析流程保持不变；
- planner 与子 Agent 共享同一份 Qwen 权重；
- 不创建第二个主模型 client；
- 不重新执行 `from_pretrained()`；
- Outlines processor 的状态不得泄漏到下一次 native 调用。

### 3.2 仍然存在的间接影响

即使子 Agent 的解码没有改变，Outlines 仍可能通过 planner 结果间接改变下游：

- task 选择可能变化，从而路由到不同 Agent；
- categories/count target 可能变化；
- ROI 是否启用、ROI 坐标和最终裁片可能变化；
- planner schema/业务校验失败时，子 Agent 不再执行；
- constrained decoding 的逐 token 处理可能增加 planner 延迟；
- 所有调用共享 generation lock，planner 延迟会影响 dataset 并发排队。

这些属于上游计划结果和调度影响，不能表述为“Outlines 对子 Agent 完全无影响”；准确表述
应为：**不直接改变子 Agent 解码，但可能通过 planner 输出和共享锁产生间接影响。**

## 4. 模型内存与单次加载硬约束

### 4.1 禁止双模型结构

以下结构会加载两份独立 Qwen 权重：

```text
planner
  -> from_pretrained(...)
  -> Outlines wrapper

sub Agents
  -> another from_pretrained(...)
```

这会使权重相关的 CPU RAM 或 GPU VRAM 占用接近翻倍。总占用不一定精确为两倍，因为还
包括 KV cache、激活、CUDA buffer、allocator fragmentation 和 processor 侧对象，但它在
架构上就是两个主模型实例，违反仓库“一次 runtime assembly 只创建一次 Qwen client”的
契约。

### 4.2 批准的共享结构

唯一批准的结构是：

```text
existing_model, existing_processor = load_once()

QwenTransformersClient
  ├─ native generation path
  └─ lazy Outlines adapter
       -> references existing_model
       -> references existing_processor
       -> does not own or reload weights
```

如果采用 Outlines 高层 adapter，应使用已经加载的对象：

```python
outlines_model = outlines.from_transformers(
    existing_model,
    existing_processor,
)
```

不得向 Outlines 另传 checkpoint 名称后让它加载模型，也不得为 planner 创建第二个
`QwenTransformersClient`。

Outlines schema/FSM/logits processor 和编译缓存会增加少量 CPU/GPU 辅助内存，但不得新增
第二份 Qwen 权重。实施测试必须检查对象 identity 和模型加载次数，而不能只观察类名。

## 5. Outlines 能力与尚未验证的风险

Outlines 官方资料说明：

- 本地 Transformers constrained generation 通过 logits processor 控制每一步 token；
- `from_transformers` 可以接收 Transformers model 与 tokenizer/processor；
- `ProcessorMixin` 会进入 multimodal adapter；
- multimodal Chat 支持 image content；
- Pydantic class 可以直接作为 JSON Schema output type；
- `max_new_tokens` 必须显式提供，默认值可能不足并导致截断。

参考：

- <https://dottxt-ai.github.io/outlines/latest/features/advanced/logits_processors/>
- <https://dottxt-ai.github.io/outlines/latest/features/models/transformers/>
- <https://dottxt-ai.github.io/outlines/latest/features/models/transformers_multimodal/>
- <https://dottxt-ai.github.io/outlines/latest/features/core/output_types/>

但官方多模态示例主要展示 Qwen2.5-VL，不足以证明当前 Qwen3-VL/Qwen3.5、项目固定的
Transformers 版本、Qwen chat template、`enable_thinking=False` 和 v5 Pydantic schema
组合已经兼容。因此真实模型 compatibility spike 是实施前置门，不得跳过。

## 6. 建议的请求级解码契约

在 `models.base` 中增加模型无关的请求选项，例如：

```text
JsonDecodingPolicy
  native
  json_schema_constrained
```

约束：

- 默认值必须为 `native`；
- `VisualTaskPlanner` 由 composition root 注入 `json_schema_constrained`；
- 子 Agent 继续省略该选项，因此使用 `native`；
- workflow/agents 不 import `outlines`；
- workflow 不选择具体 logits processor；
- application 负责选择 Outlines 作为 constrained policy 的具体实现；
- `QwenTransformersClient` 在具体模型层执行 policy dispatch。

不建议增加一个与 `complete_json()` 平行、但复制 cache/artifact/validation 的完整客户端。
应复用当前公共结构，只在实际 generation seam 上做请求级分支。

## 7. Qwen client 实施方案

### 7.1 Native 路径必须保持原样

```text
complete_json(policy=native)
  -> current message conversion
  -> current chat template
  -> current processor inputs
  -> current model.generate(do_sample=False)
  -> current JSON/Pydantic validation
```

所有子 Agent 继续走该路径。

### 7.2 Planner constrained 路径

```text
complete_json(policy=json_schema_constrained)
  -> same sanitized messages
  -> same image decoding and processor
  -> same chat template
  -> same model instance
  -> Outlines constrained generation using VisualTaskPlan v5 schema
  -> Pydantic model_validate
  -> VisualTaskPlanner post-validation
```

实现要求：

1. Outlines 惰性 import，普通 core import 不得加载它；
2. adapter 惰性创建，并引用现有 `self.model/self.processor`；
3. adapter 创建和 constrained generation 均不得再次加载 checkpoint；
4. constrained generation 必须位于现有 `_generation_lock` 内；
5. 每次调用使用全新或已正确 reset 的请求状态，禁止 processor 状态泄漏；
6. schema 直接来自 `VisualTaskPlan.model_json_schema()`，不维护第二份手写 schema；
7. 继续显式设置 `max_new_tokens`；
8. Qwen3.5 继续关闭 thinking 输出；
9. 返回字符串后仍执行 Pydantic 和 planner 业务校验；
10. Outlines 不可用、schema 编译失败或 generation 失败时稳定 fail closed；
11. 不允许静默退回 native generation。

## 8. Schema 能保证什么、不能保证什么

Outlines 只替代生成时的结构约束，不替代业务验证。

预期可约束：

- 顶层 JSON object；
- required fields；
- `additionalProperties=false`；
- task/version 等 Literal；
- array/tuple 长度；
- integer 与数值范围；
- nullable field；
- `region_request.roi_xyxy` 的基本结构。

仍必须由 Pydantic/planner 校验：

- `explicit` 与 `image_index/roi_xyxy` 的 all-or-nothing 关系；
- image index 是否指向当前样本；
- counting task 与 count target 的一致性；
- categories 是否属于当前 catalog 和实际可执行能力；
- parent/leaf 展开一致性；
- runtime capability availability；
- ROI 的确定性像素物化。

不能因为 Outlines 成功生成 JSON 就跳过现有 `_post_validate()`。

## 9. Repair 与调用次数

建议 Outlines planner 路径只执行一次物理主模型 generation：

```text
one constrained generation
  -> Pydantic validation
  -> planner post-validation
  -> success or stable failure
```

不建议继续使用通用 JSON repair：

- JSON 结构已经由 constrained decoding 负责；
- repair 会使“第一次 planner 调用”内部隐藏第二次实际 Qwen generation；
- repair prompt 通常不再携带原始图像，可能错误修改视觉判断；
- budget、延迟和调用审计会变得模糊。

因此建议冻结：

```text
planner Outlines generation = exactly one physical Qwen generation
selected Agent generation   = independent next Qwen budget entry
```

如果 Pydantic 自定义 validator 或 catalog post-validation 失败，应稳定失败，不做无图 repair。

## 10. Cache identity、artifact 与 resume

planner request identity 必须覆盖：

```text
structured_decoding = "outlines-json-schema"
outlines_adapter_version
pinned_outlines_version
schema_sha256
```

这些字段应进入：

- planner request hash；
- `run_request.json`；
- config snapshot；
- planner artifact metadata；
- resume 一致性校验。

建议使用 planner 专属 decoding identity，不因接入 Outlines 就改变 native Agent 的 generation
identity。只有 native `_generate()` 的实际行为也发生变化时，才提升全局
`QWEN_CLIENT_VERSION`；否则无理由提升会导致全部子 Agent cache miss。

artifact 应记录：

- decoding policy；
- Outlines 和 adapter 版本；
- response schema digest；
- schema compilation/cache 状态；
- token usage；
- latency；
- Pydantic/post-validation 状态；
- 稳定错误码。

不得记录 logits、Base64 图片、主机绝对路径、原始敏感异常或模型权重路径。

resume 建议：

- succeeded run 不重复 planner 或 Agent 推理；
- native-planner 与 outlines-planner invocation identity 不可互换；
- persisted structured decoding 与新 resume 请求冲突时稳定拒绝；
- 不得通过当前默认配置猜原调用的 decoding policy；
- reporting 只读展示已持久化信息，不重新生成 planner 输出。

## 11. 依赖策略

Outlines 是新的第三方依赖，必须显式说明并验证：

- 只用于本地 Qwen planner constrained decoding；
- 必须精确 pin 经过验证的版本；
- 不允许隐式升级或降级 Transformers、Torch、Pydantic；
- 默认离线，不自动下载包、权重或 tokenizer；
- base import 没有 Outlines 时仍应成功；
- 配置要求 Outlines 的 fresh inference 在缺依赖时稳定失败；
- Linux/Windows/目标部署 Python 必须有可用 wheel；
- 新 dependency 对容器体积、启动时间和离线部署的影响必须记录。

建议放入明确的 model/structured-generation dependency group，而不是让数据、reporting 或
基础 architecture tests 强制安装 Outlines。

## 12. 实施前 compatibility gate

在修改生产路径前先完成独立 spike：

1. 用依赖解析器确认候选 Outlines 版本与仓库固定依赖兼容；
2. 用真实 Qwen3-VL checkpoint 创建现有 model/processor；
3. 用同一对象创建 Outlines adapter，确认没有第二次模型加载；
4. 用 planner 的单图、多图消息验证 multimodal chat template；
5. 编译完整 `VisualTaskPlan v5` JSON Schema；
6. 验证严格 integer ROI、nullable、Literal、extra-forbid 等字段；
7. 验证 Qwen3.5 non-thinking 路径；
8. 验证 `max_new_tokens`、device/dtype 和实际 token trimming；
9. 记录 CPU RAM、GPU VRAM、首次 schema 编译时间和生成延迟；
10. 对比创建 adapter 前后的 model/processor object identity。

任一失败都应阻止正式接入。不得通过放宽 v5 schema、升级关键依赖或加载第二个模型绕过。

## 13. 测试计划

### 13.1 单模型与内存边界

- `from_pretrained()` 在一次 runtime assembly 中恰好调用一次；
- planner 与 Agent 使用相同 `id(model)`；
- planner 与 Agent 使用相同 `id(processor)`；
- Outlines adapter 不拥有独立 checkpoint/model；
- adapter 初始化前后没有第二份模型参数；
- live smoke 记录峰值 CPU RAM/GPU VRAM，不声称 fake test 等同真实显存验证。

### 13.2 请求隔离

- planner generation 带 Outlines constraint；
- 紧随其后的 Agent generation 不带 Outlines processor；
- native call 的 generation kwargs 与当前基线一致；
- Outlines processor 状态不会进入下一次 native call；
- concurrency 下实际 generation 继续串行；
- planner cache hit 不进入 Outlines 或 generation lock；
- Agent cache hit 行为不变。

### 13.3 Planner 输出

- v5 schema 首次生成即为合法 JSON；
- version/task Literal 不可越界；
- extra field 不可生成；
- `roi_xyxy` 只能是四个 `0..999` 整数；
- 非正方形 ROI 仍合法；
- Pydantic cross-field validator 继续运行；
- catalog/image-index/post-validation 失败阻止 Agent；
- Outlines failure 不进行 native fallback；
- 不发生隐藏 repair generation。

### 13.4 子 Agent 回归

代表性覆盖：

- General VQA direct；
- General VQA object evidence；
- Grounding；
- Caption；
- Change；
- Counting tile；
- Counting empty review；
- Quantity proposal。

需要断言 messages、response model、generation kwargs、request hash 和 budget 语义没有因
Outlines planner 分支发生无关改变。

### 13.5 Cache、artifact 与 resume

- native planner cache 与 Outlines planner cache 不混用；
- planner hash 覆盖 schema 和 adapter identity；
- Agent cache identity 在 native 路径未变时保持稳定；
- artifact 记录 decoding metadata 且不泄漏敏感信息；
- succeeded resume 零模型调用；
- nonterminal resume 不允许切换 decoding policy；
- reporting 不重新运行 Outlines。

## 14. Live 对照验证

使用固定、完整的遥感样本切片比较 native planner 与 Outlines planner：

```text
first-pass schema-valid rate
planner failure rate
task agreement rate
explicit ROI trigger rate
ROI coordinate distribution
planner latency
completion tokens
selected Agent distribution
child Agent call count
end-to-end succeeded / partial / failed / skipped
peak CPU RAM / GPU VRAM
```

所有被选择样本必须计入统计，不过滤 Outlines 失败样本。需要把 planner 输出变化与子 Agent
解码是否变化分开报告。

## 15. 修改范围

预计涉及现有批准路径：

- `models/base.py`：请求级 decoding policy/protocol；
- `models/qwen_transformers.py`：Outlines adapter 与隔离 generation 分支；
- `models/settings.py`：必要的本地 structured-generation 配置声明；
- `workflows/visual_planner.py`：请求 constrained policy；
- `application/settings.py`：planner decoding 配置；
- `application/bootstrap.py`：composition 接线与能力检查；
- `workflows/schema.py`、`application/runtime.py`：run request/resume identity；
- `pyproject.toml`、模型 requirements：精确依赖声明；
- planner/model/runtime/resume/integration 对应测试；
- `DETAILS.md` 和本架构文档的实施状态。

不得新增未在 `architecture/allowed_python_files.txt` 批准的 Python 文件。若现有职责无法容纳
实现，应先单独申请 allowlist 架构变更，不能在普通实现中绕过。

## 16. 建议实施顺序

1. 完成 Outlines/Qwen3-VL/Qwen3.5 compatibility spike；
2. 冻结依赖版本、adapter version 和 decoding identity；
3. 增加模型无关的逐请求 decoding policy；
4. 在同一个 Qwen client 内实现 lazy Outlines adapter；
5. 加入单模型加载、对象 identity 和 processor 不泄漏测试；
6. 仅为 VisualTaskPlanner 打开 constrained policy；
7. 冻结 one-generation/no-repair/fail-closed 语义；
8. 接入 cache、artifact、run request 和 resume；
9. 运行子 Agent native-path 回归和 architecture tests；
10. 执行真实模型内存、兼容性和端到端对照 gate；
11. 更新 `DETAILS.md`、依赖说明和已知限制。

## 17. 完成判据

实施完成必须同时满足：

1. 只有第一次 Visual Planner 调用启用 Outlines；
2. 所有子 Agent 主模型调用保持 native generation；
3. 一次 runtime assembly 只加载一个 Qwen model 和一个 processor；
4. Outlines adapter 引用现有 model/processor，不重新加载权重；
5. 模型加载次数测试严格为 1；
6. planner/Agent 的 model 与 processor object identity 相同；
7. planner constrained state 不泄漏到子 Agent；
8. planner 使用 v5 的精确 Pydantic schema，且保留业务 post-validation；
9. planner 只执行一次物理 generation，不进行无图 repair；
10. Outlines 不可用或失败时 fail closed，不回退 native；
11. planner decoding identity 进入 cache、artifact、run request 和 resume；
12. native Agent cache identity 不发生无依据漂移；
13. 真实模型 gate 证明没有第二份 Qwen 权重和显著异常内存增长；
14. `UnifiedSample`、routing、模型逻辑身份、评测、报告和路径安全契约不发生漂移。

## 18. 本次执行状态

已完成的离线实现：

- `VisualTaskPlanner` 默认携带 `outlines-json-schema` 逐请求身份；所有子 Agent 调用继续使用
  `native`，共享同一个 Qwen model/processor/client 与 generation lock。
- `QwenTransformersClient` 增加惰性 Outlines adapter、精确 `VisualTaskPlan.model_json_schema()`
  约束、processor 状态恢复/processor reset、一次 physical generation、无 repair、fail-closed
  错误码与 JSON-safe decoding metadata。
- planner request hash、Qwen planner artifacts、config snapshot、run request 与 resume seam
  记录 adapter/version/schema identity；native Agent 的全局 client identity 未提升。
- 新依赖声明为 `outlines[transformers]==1.3.3`，位于 `structured-generation` optional group；
  README 已补充显式安装命令。
- 离线 fake-object 测试覆盖共享对象 identity、无第二次生成、native 隔离、processor 状态、无
  repair、cache identity、artifact metadata 与配置/规划器身份。

尚未满足、必须保留为阻塞项的内容：

- 当前执行环境为 Python 3.14，未安装 Outlines；仓库也没有可用的真实 Qwen3-VL/Qwen3.5
  checkpoint，因此第 12 节 compatibility spike、显存/内存测量与第 14 节 live 对照不能诚实
  宣称通过。缺少 Outlines 时生产路径按设计稳定返回 `OUTLINES_UNAVAILABLE`，不回退 native。
- 完成第 12/14 节前，不得把本实现描述为真实 Qwen3-VL/Qwen3.5 已验证或把 fake 测试结果当作
  live compatibility 结论。
