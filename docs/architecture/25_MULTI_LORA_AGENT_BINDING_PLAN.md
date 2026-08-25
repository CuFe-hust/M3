# 25. Qwen3.5-9B 单基座、多 LoRA、按 Agent 绑定实施计划

## 1. 目标

在不改变现有 Agent 业务协议、任务路由、评测语义和 resume 语义的前提下，支持：

1. 每次 Runtime 只加载一次 Qwen3.5-9B 基座模型与 processor；
2. 同一基座上加载一个或多个 PEFT LoRA adapter；
3. VisualTaskPlanner 与不同 Agent 通过稳定逻辑 binding 选择 adapter；
4. 首期所有调用方统一绑定同一个 adapter；
5. 后续只修改配置即可为某个 Agent 切换 adapter，不修改 Agent 代码；
6. adapter 身份完整进入缓存、run snapshot、manifest 和审计信息，避免错误 cache hit；
7. 默认离线，adapter 不存在、不兼容或身份不完整时稳定 fail closed。

首期本地 adapter 输出目录确定为：

```text
outputs/finetune/qwen35-9b-visual-planner-lora-supplement-20260824
```

运行时绑定其中实际包含 `adapter_config.json` 和
`adapter_model.safetensors` 的明确部署目录：

```text
outputs/finetune/qwen35-9b-visual-planner-lora-supplement-20260824/final_adapter
```

该目录已经从远端训练环境复制到当前工作区，并通过远端/本地 SHA-256 对照校验。当前
`adapter_model.safetensors` 摘要为：

```text
e59d7f5afe0f3c75a06e785579a74d4a7a9880a589be8d91197d699569dea170
```

不得依靠扫描 checkpoint 或“自动选择最新目录”猜测 adapter；配置必须最终解析到唯一、明确的
adapter 目录。

## 2. 当前事实与缺口

当前 `application.bootstrap.assemble_runtime(...)` 只创建一个
`QwenTransformersClient`，并把同一个 client 传给 VisualTaskPlanner、Counting、Change、
Grounding、GeneralVQA、Caption 及其 Qwen 子流程。

现有 `QwenTransformersClient`：

- 直接从 `QwenSettings.model` 加载一个完整 checkpoint；
- 没有 adapter catalog、`load_adapter`、`set_adapter` 或 request-level adapter 参数；
- 使用单个 `_generation_lock` 串行化本 client 的生成；
- `ModelCacheIdentity` 当前只表达主模型、generation、client version 和 revision；
- Runtime 与 `AgentContext` 当前都只暴露单个 `qwen_client`。

当前工作区中的 supplement 训练产物是纯 LLM LoRA。已对照远端原始文件核对其
`final_adapter/adapter_config.json` 与训练 manifest：

```text
peft_type = LORA
r = 32
lora_alpha = 64
modules_to_save = null / []
auxiliary_heads = null
```

`visual_planner_roi_head` 属于工作区内旧 visual-planner adapter 的过时契约，不能被带入
supplement adapter 的新运行时。运行时应以每个待加载 adapter 自己的配置和权重清单为准；如果
未来某个 adapter 声明非空 `modules_to_save`，必须显式支持并校验，否则稳定拒绝加载，不能猜测或
自动挂载旧辅助头。

## 3. 架构决策

### 3.1 共享 engine，绑定 client

在 `models` 中引入一个共享的多 adapter engine。engine 独占：

```text
Qwen3.5-9B base model
processor
PEFT adapter inventory
generation lock
response cache access
```

engine 为调用方创建轻量的、只读的 bound client：

```text
MultiAdapterQwenEngine
  ├── bind("planner")     -> VisionLanguageClient
  ├── bind("counting")    -> VisionLanguageClient
  ├── bind("change")      -> VisionLanguageClient
  ├── bind("grounding")   -> VisionLanguageClient
  ├── bind("general_vqa") -> VisionLanguageClient
  └── bind("caption")     -> VisionLanguageClient
```

每个 bound client 继续实现现有 `models.base.VisionLanguageClient`。Agent 只知道模型协议，
不知道 PEFT、adapter 名称、adapter 路径或 checkpoint 路径。

### 3.2 adapter 选择属于 application composition root

adapter catalog、binding 校验和 client 注入只在 `application` 完成：

```text
AppSettings
  -> models.entry.create_model(...)
  -> MultiAdapterQwenEngine
  -> deterministic component/agent binding
  -> VisualTaskPlanner / AgentRegistry / evidence services
```

禁止：

- Agent 根据 question、dataset 或 metadata 自行选择 adapter；
- Router 加载或切换 adapter；
- Workflow 读取 adapter 物理路径；
- 在 `AgentContext.request_context` 中携带 adapter 路径；
- 未知 binding 回退到任意 adapter；
- 为每个 Agent 重新加载一份 Qwen3.5-9B 基座。

### 3.3 首期统一绑定，结构上保留独立切换

首期定义一个逻辑 adapter，例如：

```text
qwen35-9b-visual-planner-supplement-20260824
```

以下 binding 全部指向它：

```text
planner
counting
change
grounding
general_vqa
caption
```

`spatial_relation`、`scene_classification`、`multiple_choice_vqa` 等 task 不新增平行的模型配置；
它们继续先由 TaskRouter 解析到实际 Agent，再使用该 Agent 的 binding。

后续切换只需要把某个 binding 改为另一个已声明 adapter：

```yaml
models:
  qwen_adapter_bindings:
    planner: visual-planner-supplement
    counting: counting-v2
    change: change-v1
    grounding: grounding-v1
    general_vqa: general-vqa-v3
    caption: caption-v2
```

不提供运行中由用户文本任意指定磁盘路径的接口。切换目标必须先进入受校验的配置 catalog。

## 4. 配置设计

在 `models.settings.ModelSettings` 中增加声明式配置，字段名以实施时最终评审结果为准，建议结构：

```yaml
models:
  qwen:
    model: models/Qwen3.5-9B
    cache_model_id: Qwen/Qwen3.5-9B:local
    allow_download: false

  qwen_adapters:
    visual-planner-supplement:
      path: outputs/finetune/qwen35-9b-visual-planner-lora-supplement-20260824/final_adapter
      logical_id: qwen35-9b-visual-planner-supplement-20260824
      revision: e59d7f5afe0f3c75a06e785579a74d4a7a9880a589be8d91197d699569dea170
      enabled: true

  qwen_adapter_bindings:
    planner: visual-planner-supplement
    counting: visual-planner-supplement
    change: visual-planner-supplement
    grounding: visual-planner-supplement
    general_vqa: visual-planner-supplement
    caption: visual-planner-supplement
```

配置规则：

- `path` 是本机物理路径，可以包含 `~`，只允许在 application/models 加载边界执行
  `expanduser()`；
- 物理路径不得进入逻辑模型身份、request hash、trace、prediction index 或可移植 artifact；
- `logical_id` 必须非空、非路径、机器无关；
- `revision` 首期要求为 `adapter_model.safetensors` 的 SHA-256，禁止只依赖目录名；
- binding key 使用固定集合并 `extra="forbid"`；
- binding 指向 disabled/missing/unknown adapter 时组装失败；
- 首期不支持请求参数覆盖 binding；
- run 的 `config.snapshot.json` 保留复现所需配置，但公共 trace/error 不泄漏主机绝对路径。

为了兼容无 adapter 的基座运行，可显式保留一个逻辑 binding 值，例如 `base`；不能用缺失字段
或未知 adapter 隐式表达回退。

## 5. 模型层实施

### 5.1 新增明确职责模块

建议在 `models/qwen3_5/` 下新增职责明确的模块，例如：

```text
models/qwen3_5/multi_adapter.py
```

不要新增 `manager.py`、`helpers.py`、`utils.py` 或 `compat.py`。

主要对象建议为：

- `QwenAdapterSpec`：已校验的物理资产与逻辑身份；
- `MultiAdapterQwenEngine`：共享 base、processor、PEFT model 与切换锁；
- `BoundQwenAdapterClient`：实现 `VisionLanguageClient` 的轻量绑定视图。

`models.entry` 增加或扩展一个惰性 builder。`import models.entry` 仍不得加载 torch、
transformers、PEFT 或模型权重。

### 5.2 启动时资产与兼容性校验

加载顺序必须稳定：

1. 解析并校验基座的逻辑身份；
2. 加载一次 Qwen3.5-9B base model 与 processor；
3. 对每个 enabled adapter 校验 `adapter_config.json` 和权重文件存在；
4. 校验 `peft_type == "LORA"`；
5. 校验 adapter 声明的 base/model type 与实际 Qwen3.5-9B 兼容；
6. 校验 target modules 能在实际模型树中解析；
7. 校验 supplement adapter 的 `modules_to_save` 为空，且不存在过时 auxiliary head；
8. 使用 PEFT 官方接口加载 adapter，并显式命名；
9. 校验所有 LoRA adapter tensor 均被消费；未来遇到非空 `modules_to_save` 时，无明确实现则
   fail closed；
10. 设置 eval mode，并冻结全部参数；
11. 生成不含物理路径的 adapter inventory 审计数据。

任何一步失败都应产生稳定 error code/type，不把完整本地路径或底层异常全文写入 artifact。

### 5.3 不 merge adapter

本功能不能调用 `merge_and_unload()`，否则无法在同一个 base 上切换多个 adapter。部署态保持：

```text
base weights + named PEFT adapters
```

### 5.4 切换与并发安全

首期继续使用 Transformers + PEFT。PEFT `set_adapter(...)` 改变共享 model 的活动状态，因此：

```text
acquire shared generation lock
  -> set_adapter(bound adapter name) 或 disable adapter for base
  -> generate
  -> bounded repair generation（仍在同一锁内）
release lock
```

adapter 切换、首次生成和可能的 JSON repair 必须处于同一个临界区，禁止在两次生成之间被另一个
请求切换 adapter。

这会保证正确性，但会让同一基座上的 live generation 串行执行；cache hit 可以继续不占生成锁。
若后续需要不同 adapter 请求真正并发，再以独立任务评估 vLLM Multi-LoRA，不能在本次改造中同时
替换推理后端。

## 6. 缓存与身份

每个 bound client 必须暴露独立、完整的 `ModelCacheIdentity`。建议将 adapter 信息加入
generation identity 或扩展稳定 schema，使 request hash 至少覆盖：

```text
base logical model id
base revision
adapter logical id（或显式 base）
adapter revision / weights SHA-256
adapter scale（未来支持时）
PEFT/client version
generation settings
prompt version/content
messages
image digest
response schema
```

即使首期所有 binding 指向同一个 adapter，也必须按最终 binding client 构造身份，不能继续仅用
Qwen3.5-9B 基座身份。以后只改 binding 时，缓存必须自然失效，不能误命中旧 adapter 的结果。

adapter 物理路径永远不作为持久化逻辑身份。对现有
`QwenTransformersClient.cache_identity` 的改变要保留旧的严格路径校验和 JSON-safe 契约。

## 7. Application 组装改造

### 7.1 RuntimeComponents

保持 Agent 和 workflow 依赖 `VisionLanguageClient`，但 application 内部建立明确 client inventory：

```text
planner_client
agent_clients[agent_name]
```

`RuntimeComponents.qwen_client` 若继续保留，必须明确其含义（例如默认/base compatibility seam），
不能再被 fresh runtime 无条件注入所有 Agent。更稳妥的方向是增加类型明确的 binding 容器，避免
未来代码重新误用全局 client。

### 7.2 VisualTaskPlanner

Planner 始终接收 `planner` binding。它不能根据 source task、dataset 或尚未确定的 Agent 改选
adapter。首期该 binding 与全部 Agent 相同，后续可独立切换。

### 7.3 AgentRegistry

`_build_agent_registry(...)` 按 Agent 名注入对应 bound client：

```text
CountingAgent      <- counting client
ChangeAgent        <- change client
GroundingAgent     <- grounding client
GeneralVQAAgent    <- general_vqa client
CaptionAgent       <- caption client
```

Counting 的 `qwen_point`、`quantity_proposal`、seam review 等 Qwen 子流程全部使用 counting
client。Grounding evidence 的 final-Qwen 使用 grounding client。不得遗漏嵌套服务而形成“一部分
调用新 adapter、一部分调用旧全局 client”的混合执行。

### 7.4 SampleRunner 与 AgentContext

当前 `SampleRunner` 把单个 client 放入 `AgentContext`。实施时先审计实际消费方，然后二选一：

1. 如果生产 Agent 不消费 `context.qwen_client`，从 context 删除该字段并更新测试；或
2. SampleRunner 在每次已确定 primary/fallback agent 后，把该 Agent 的 bound client 放入
   context。

不能保留一个与 Agent 构造时 client 身份不一致的全局 context client。fallback attempt 必须按
实际执行的 Agent 重新建立一致 context，且继续遵守既有 attempt 硬上限。

## 8. 持久化、resume 与报告

新 fresh run 应冻结：

- base logical model id/revision；
- adapter catalog 的逻辑身份与 revision/digest；
- component/Agent 到 adapter logical id 的完整 binding；
- runtime/client version。

`run_request.json` 仍是 resume 调用参数的权威来源。实现时应明确：

- resume 不得用当前配置悄悄替换原 run 的 adapter binding；
- 原 run binding 与当前请求冲突时稳定拒绝；
- succeeded 样本不因当前 adapter 改变而重新推理；
- partial/failed 等需要重跑时使用原 run 冻结的 binding；
- predictions/status/reporting 只展示逻辑 adapter 身份，不读取或信任 adapter 物理路径；
- reporting 不重新推理，也不重新选择 adapter。

如果旧 run 没有 adapter binding 元数据，应定义明确的 legacy 解释（例如“base-only legacy”），
不能猜成当前默认 adapter。

## 9. 分阶段实施顺序

### 阶段 A：配置与身份契约

1. 增加 adapter spec、catalog 和 binding settings；
2. 增加路径、logical id、digest、固定 binding key 校验；
3. 定义 bound client 的 cache identity；
4. 更新 config snapshot、manifest/run identity 和 `DETAILS.md`；
5. 先用 fake engine/client 完成纯离线测试。

完成标准：不加载 torch/PEFT 也能验证配置、身份、快照和 import 边界。

### 阶段 B：单基座、多 adapter engine

1. 实现 Qwen3.5 base 单次加载；
2. 明确拒绝为 supplement adapter 挂载过时的 `visual_planner_roi_head`；
3. 实现 adapter 资产/兼容性/权重消费校验；
4. 实现 named adapter 加载与 bound clients；
5. 实现锁内 adapter 切换、生成和 repair；
6. 在 `models.entry` 注册惰性 builder。

完成标准：fake Qwen/PEFT 测试证明 base 只加载一次，多个 adapter 各加载一次，切换不会串权重。

### 阶段 C：Application 接线

1. 创建 engine 一次；
2. 为 Planner 和每个 Agent 创建 bound client；
3. 替换 `_build_visual_task_planning`、`_build_agent_registry`、counting backend、grounding
   evidence 的全局 client 注入；
4. 修正 SampleRunner/AgentContext 的 client 一致性；
5. 首期将所有 binding 配置为同一 supplement adapter。

完成标准：每条模型调用的 audit/cache identity 都显示同一 supplement adapter logical id，且基座
只加载一次。

### 阶段 D：可配置切换与 resume

1. 测试单独把某个 Agent binding 改到第二个 fake adapter；
2. 验证 Router 决定 Agent 后使用正确 binding；
3. 验证 fallback agent 改用 fallback Agent 的 binding；
4. 验证不同 adapter 不共享 cache entry；
5. 验证 resume 使用冻结 binding，冲突 fail closed；
6. 验证 report 只读展示逻辑身份。

完成标准：无需修改 Agent 代码，仅修改配置即可切换某个 Agent 的 adapter。

### 阶段 E：真实本地 gate

在明确具备本地 Qwen3.5-9B 与 supplement adapter 文件的环境执行：

1. adapter 文件、SHA-256、base compatibility 和 target-module 审计；
2. 最小 image+text planner 调用；
3. 每个 Agent 至少一个最小真实调用；
4. 并发提交两个不同 binding 的请求，验证锁内无串 adapter；
5. 显存检查，证明只有一份 base 权重；
6. 对比 base-only 与 adapter 输出，确认 adapter 实际生效而非未消费。

真实 gate 未执行前，不得声称多 LoRA live inference 已验证。

## 10. 测试计划

至少新增或更新以下测试族：

### models

- adapter settings `extra="forbid"`；
- `~` 仅在加载边界展开；
- 物理路径不进入 cache identity；
- logical id/path-like/revision/digest 校验；
- missing/corrupt/LFS pointer adapter 权重稳定失败；
- base/model type/target module 不兼容稳定失败；
- supplement adapter 的 `modules_to_save` 为空且不会挂载旧 ROI head；
- 非空 `modules_to_save` 在没有显式实现时稳定失败；
- base 只加载一次；
- named adapters 各加载一次；
- bound clients 暴露不同身份；
- adapter 切换与 repair 共用同一锁；
- cache hit 不触发 adapter 切换或 generation；
- `models.entry` import 不加载 torch/transformers/PEFT。

### application/workflows

- Planner 获取 planner binding；
- 每个 Agent 获取自己的 binding；
- counting 所有 Qwen backend 获取 counting binding；
- grounding evidence 获取 grounding binding；
- primary/fallback Agent context 与构造 client 身份一致；
- 未知/禁用/missing binding 在模型调用前失败；
- config snapshot 和 manifest 冻结完整逻辑 binding；
- resume binding 冲突拒绝；
- succeeded resume 不重复模型调用；
- report 不读取 adapter 物理路径、不重新推理。

### 架构

运行：

```text
tests/architecture/test_repository_hygiene.py
tests/architecture/test_import_boundaries.py
tests/architecture/test_init_side_effects.py
tests/architecture/test_package_discovery.py
tests/architecture/test_no_new_to_legacy_imports.py
```

并运行所有受影响的 model、application、workflow、Agent integration 与 resume 测试。不得通过修改
Golden fixture、跳过测试或放宽断言掩盖 adapter 选择错误。

## 11. 文档同步

实施时同步更新：

- `DETAILS.md`：模型入口、Qwen settings、单次组装、Agent client binding、cache identity、run
  snapshot、resume 和限制；
- README：仅当公开配置或运行命令变化时更新；
- `docs/migration/`：只有该改变影响历史行为或结果可比性时记录 intentional difference；
- `architecture/implementation_status.json`：生产包状态发生变化时更新。

## 12. 非目标

本计划不包括：

- 改变 VisualTaskPlanner 的 task 决策语义；
- 让 Agent 或 Router 自主发现/选择 adapter；
- 按 token 混合多个 LoRA；
- adapter 权重线性融合或 TIES/DARE merge；
- 通过 HTTP/CLI 接受任意 adapter 文件路径；
- 把 Transformers 后端同时迁移到 vLLM；
- 为每个 Agent 加载独立基座；
- 改变指标、GT、split、样本选择或 Judge 语义；
- 修改 supplement adapter 或训练产物来迎合运行时。

## 13. 验收标准

全部满足才视为完成：

1. Qwen3.5-9B base 和 processor 在一次 runtime assembly 中只加载一次；
2. supplement adapter 的全部 LoRA 权重被完整加载，且没有挂载过时的
   `visual_planner_roi_head`；
3. 首期 Planner 与所有 Agent 明确绑定同一 supplement adapter；
4. 单改配置即可把任意一个 Agent 切换到另一已声明 adapter；
5. 并发请求不会发生 adapter 串用；
6. adapter 改变必然改变 cache identity/request hash；
7. run snapshot 和 resume 冻结并遵守原 adapter binding；
8. artifact、trace、status、prediction 和 report 不泄漏 adapter 物理绝对路径；
9. 默认离线，缺失或不兼容资产稳定 fail closed；
10. 架构测试、相关单元测试、integration/resume 测试通过；
11. 真实模型 gate 的执行情况与未验证风险被如实记录。
