# 19 — Counting Target from VisualTaskPlan（讨论后修订 Plan）

> Historical implementation plan: this document introduced `visual-task-plan-v4`.
> Doc 20 subsequently superseded the active protocol; current fresh execution uses
> `visual-task-plan-v5`. The v4 names and examples below are retained as historical
> design records, not as statements of the current runtime contract.
>
> 历史实施计划：本文引入了 `visual-task-plan-v4`。doc 20 随后替代了现役协议；
> 当前 fresh execution 使用 `visual-task-plan-v5`。下文的 v4 名称与示例作为历史设计
> 记录保留，不表示当前 runtime 契约。

> Status: **implemented through entrypoint/resume/artifact plumbing**

> 状态：**已完成入口、resume、artifact 与 trace 接线；后续阶段按实施包继续验证**

## 1. 背景与当前问题

当前 fresh 运行已经先调用一次 `workflows.visual_planner.VisualTaskPlanner`，并把
`VisualTaskPlan` 与 `MaterializedVisualView` 通过 `AgentContext` 传给最终 Agent。

但是 counting 的目标类别仍在 `CountingAgent` 内单独解析：

```text
VisualTaskPlanner
  -> VisualTaskPlan(task / object_categories / region_request)
  -> CountingAgent
       -> normalization.count_target_hint
       -> legacy metadata["count_target_hint"]
       -> 缺失时再次调用 Qwen target_parse_v1
       -> CountTargetSpec
```

这带来两个问题：

1. 第一次 planner 已经看过图像和问题，但 counting 没有消费它的目标判断；
2. hint 未命中时会再执行一次纯文本 Qwen，形成重复的目标解析调用。

同时，不能把现有 `VisualTaskPlan.object_categories[0]` 直接当作 count target。该字段当前
只允许 `EvidenceCatalog` 的**组合父类**，例如：

```text
vehicle -> small_vehicle + large_vehicle
```

因此它能表达 `vehicle`，却不能直接表达可执行叶子类 `small-vehicle`。直接复用会把：

```text
small vehicle -> vehicle
```

从精确子类扩大成父类，正是本任务要避免的问题。

讨论后冻结的新方向是：

```text
模型 head 与 planner 可执行类别池：只包含叶子类
父类：只作为确定性展开规则，不作为 YOLO/SegFormer head，也不进入可执行类别池
用户语义目标：可保留父类含义，但执行前必须展开为叶子集合
```

## 2. 本次目标

本计划建议实现以下目标：

1. 第一次 `VisualTaskPlanner` 对 counting 任务必须输出精确计数目标；
2. `CountingAgent` 优先且必须消费该 planner 产物，不再发起 `target_parse_v1` Qwen 调用；
3. 现有硬编码 hint/ontology 不删除，但职责改为**确定性校验、类别边界核实与必要纠偏**；
4. 当问题明确包含 `small`、`large` 等限定词时，planner 不得静默删除限定词并扩大类别；
5. planner 的 `object_categories` 只输出经 canonical 化且已验证可执行的叶子类；
6. `vehicle`、`aircraft` 等父类不进入模型分类头和 planner 可执行类别池；
7. 新 YOLO 类别契约采用“现有 18 类 + 9 个非重复叶子类”，共 27 类；
8. 当前 SegFormer 只纳入已核实的 iSAID 15 个前景叶子类，OEM 暂不纳入；
9. planner 仍不得选择 counting backend、checkpoint、device 或 detector class；
10. 不改变 `UnifiedSample`、`CountingResult`、评测指标、GT、split、报告聚合和主模型身份；
11. 保持一次 runtime assembly 只创建一个 Qwen client，counting backend 继续共享该 client。

目标流程改为：

```text
thumbnail/full image(s) + raw question
  -> VisualTaskPlanner (唯一目标理解调用)
  -> VisualTaskPlan.count_target                # 用户语义目标
  -> VisualTaskPlan.object_categories           # 可执行叶子集合
  -> deterministic target verifier
       + normalization.count_target_hint
       + legacy metadata hint
       + narrowly scoped lexical specificity guard
       + canonical leaf/parent expansion catalog
  -> reconciled CountTargetSpec + executable leaf set
  -> ExpertCatalog / BackendSelector / Executor
```

这里的“唯一调用”只限定于**任务与计数目标规划**。后续 Qwen point counting、quantity
proposal、zero review 等真正执行 counting 的模型调用不属于重复 target parsing，保持现有策略。

## 3. 非目标

本任务不做以下事情：

- 不让 planner 输出 backend、模型名、权重路径或 class map；
- 不把模型原始 head 字符串直接拼成未经 canonical 化的 planner pool；
- 不把 `EvidenceCatalog` 与 `ExpertCatalog` 粗暴合并成一个大目录；
- 不删除 VRSBench 的官方 small/large vehicle ontology；
- 不把 `vehicle`、`aircraft`、`sports_facility` 等父类训练成与叶子并列的模型 head；
- 不使用模糊字符串相似度自动猜任意类别；
- 不让 dataset adapter 调模型；
- 不让 Router 读取问题或决定 count target；
- 不改变 counting backend 的优先级、fallback 或 zero-review 策略；
- 不修改 detector/point/evidence confidence；
- 不修改 ROI、1080 预览图或大图方案 A；
- 不修改历史 run artifact；
- 不在本计划的 runtime 实现步骤中直接训练、覆盖或提交大型权重；
- 不新增未在 allowlist 中批准的 Python 文件。

## 4. 建议的 planner 输出契约

### 4.1 分离“用户语义目标”与“可执行叶子类别”

修订后保留两个不同语义，但 `object_categories` 不再是组合父类：

```text
count_target
    counting/fine_grained_counting 的用户语义目标；保留问题中的限定词。
    它可以是 generic vehicle，因为这是用户真正提出的问题语义。

object_categories
    最终可执行的 canonical 叶子类别列表；不得包含父类、alias 或模型原始拼写。
    VQA/Grounding/Counting 后续视觉能力都消费这一层。
```

父类作为**可执行类别**时只存在于 catalog 的确定性展开表中；`count_target` 可以保留父类名称
来表达用户原始语义：

```text
vehicle -> [small-vehicle, large-vehicle]
aircraft -> [plane, helicopter]
```

父类不属于 YOLO/SegFormer 分类头，也不属于 planner 的 `object_categories` 允许值。

### 4.2 `count_target` 字段

建议在 `VisualTaskPlan` 增加：

```python
count_target: str | None = None
```

字段约束：

- `task in {"counting", "fine_grained_counting"}` 时必须是非空安全字符串；
- 其他 task 时必须为 `None`；
- 去除首尾空白后长度设置保守上限，例如 80；
- 禁止控制字符、路径样式与空值；
- 表达“要数什么”，不得包含答案或数量；
- 必须保留问题中的语义限定词，例如 `small vehicle`、`large vehicle`、`red car`；
- 可以做拼写归一化，例如把用户输入 `small vehical` 输出为 `small-vehicle`，但不能丢掉
  `small` 后只输出 `vehicle`。

`CountTargetSpec` 仍由 `agents/counting/schema.py` 所有，不移动到通用 schema。下游 resolver
根据 `count_target`、已审计 hint 和叶子展开结果构造完整 spec，避免复制 counting schema 或
让模型自由生成大段 inclusion/exclusion 规则。

### 4.3 `object_categories` 改为叶子集合

`object_categories` 的新约束：

- 只允许 canonical leaf；
- 不允许 `vehicle`、`aircraft`、`sports_facility` 等父类；
- 不允许 `airplane`/`plane` 同时作为两个 canonical 类；alias 必须先去重；
- 不允许把空格、下划线、连字符差异当成多个类别；
- 按 catalog 稳定顺序去重；
- 将当前 schema 的 `max_length=3` 调整为能容纳已批准父类最大展开的保守上限；初版建议
  `max_length=8`，并继续由 post-validation 限制真实 catalog 展开，不能借此无界输出；
- 每个输出叶子必须至少有一个对当前任务真正可执行的能力；
- `needs_visual_assistance=False` 时仍必须为空；
- 已知 counting target 且存在叶子能力时，必须为对应的完整叶子集合；
- 未知或当前无视觉 expert 的 counting target 可以保留 `count_target`，但
  `object_categories=[]`，由现有 generic counting backend 执行，不再二次解析目标。

示例一，精确小型车辆：

```json
{
  "version": "visual-task-plan-v4",
  "task": "counting",
  "needs_visual_assistance": true,
  "object_categories": ["small-vehicle"],
  "count_target": "small-vehicle",
  "region_request": {
    "explicit": false,
    "image_index": null,
    "focus_xy_norm": null
  },
  "reason_codes": ["count_target_explicit"]
}
```

示例二，用户泛指 vehicle：

```json
{
  "version": "visual-task-plan-v4",
  "task": "counting",
  "needs_visual_assistance": true,
  "object_categories": ["small-vehicle", "large-vehicle"],
  "count_target": "vehicle",
  "region_request": {
    "explicit": false,
    "image_index": null,
    "focus_xy_norm": null
  },
  "reason_codes": ["count_target_parent_expanded"]
}
```

示例三，目标不在当前视觉能力池：

```json
{
  "version": "visual-task-plan-v4",
  "task": "counting",
  "needs_visual_assistance": false,
  "object_categories": [],
  "count_target": "building",
  "region_request": {
    "explicit": false,
    "image_index": null,
    "focus_xy_norm": null
  },
  "reason_codes": ["count_target_generic_backend"]
}
```

### 4.4 Planner canonical 叶子类别池

类别池不是模型 head 字符串的简单并集，而是：

```text
verified YOLO labels
  + verified SegFormer labels
  -> canonicalize aliases/separators/case
  -> remove background
  -> remove every parent/composite
  -> stable deduplication
  -> task-specific executable capability filter
```

#### 新 YOLO：27 个叶子类别

保持当前 18 类的顺序与逻辑含义：

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

只追加 9 个不重复叶子类：

```text
chimney
dam
expressway-service-area
expressway-toll-station
golffield
overpass
stadium
trainstation
windmill
```

VRSBench 的 `airplane` 映射到现有 canonical `plane`，不新增重复 head。`vehicle` 是
`small-vehicle + large-vehicle` 的父类，也不新增 head。因此目标分类头总数为 27，而不是
把 VRSBench 字符串机械追加后的数量。

27 类是**新权重交付后的目标状态**。在新权重、class-map、logical model id、SHA256 和逐类
验证全部通过之前，catalog 最多只能声明当前已验证的 18 个 YOLO 叶子类；每个具体 runtime
再按已启用的 task capability 发布其中的可执行子集。不得提前把 9 个新类放入 planner binding
后再让执行阶段失败。新权重启用时必须升级 catalog version，使最终 system prompt、snapshot
和 request hash 同步变化。

#### 当前 SegFormer：15 个已验证前景叶子类别

只允许当前 iSAID `classes.json` 已核实的前景类别进入池：

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

`background` 必须排除。OpenEarthMap SegFormer 当前只有 `LABEL_0..LABEL_8` 占位映射且
`blocked_unverified_class_map`，不得进入 planner 类别池。iSAID 的 15 个前景类都已包含在上述
YOLO 类别中。因此当前 verified catalog union 仍是 18 个叶子类；具体 runtime 广告其启用能力
子集。新 YOLO 交付后 verified union 才扩展为 27 个叶子类。SegFormer 不会为这个 union
额外增加父类或重复 alias。

### 4.5 Catalog 职责

建议由版本化 catalog 明确维护：

```text
canonical leaves
aliases（airplane -> plane 等）
parent expansions（vehicle -> small/large vehicle 等）
per-expert raw label mappings
per-task executable capabilities
catalog version
```

不得让 planner、Agent 或 backend 各自维护一份类别列表。模型原始 label 只存在于 expert
mapping；planner/artifact 的 `object_categories` 只使用 canonical leaf，`count_target` 则可以
保留经安全规范化的用户语义父类。

## 5. Prompt 与 post-validation 规则

新 planner prompt 必须明确：

```text
For counting and fine_grained_counting, return the exact requested target in
count_target. Preserve every scope-changing modifier from the raw question.
Never replace small vehicle or large vehicle with the broader vehicle category.
Return only canonical executable leaf categories in object_categories.
Never return a parent category there; expand a generic parent to all of its
declared executable leaves.
```

中文等价规则也应写入 prompt：

```text
对于 counting/fine_grained_counting，count_target 必须保留会改变范围的限定词；
不得把“小型车辆”或“大型车辆”扩大成泛指“车辆”。
object_categories 只能包含 canonical 可执行叶子类；父类必须按 catalog 展开为完整叶子集合。
```

`VisualTaskPlanner._post_validate()` 只做结构和安全校验，不凭模型主观 confidence 接受或拒绝。
建议增加：

- counting task 缺失 `count_target` -> `SCHEMA_INVALID`；
- 非 counting task 携带 `count_target` -> `SCHEMA_INVALID`；
- count target 含路径/控制字符/明显数量答案 -> `SCHEMA_INVALID`；
- `object_categories` 按 canonical leaf pool 与当前 task capability 校验；
- 任意父类、alias、模型原始拼写或未知 leaf -> `SCHEMA_INVALID`；
- 已知父类语义的 `count_target` 必须对应完整叶子展开，不能只返回其中一部分；
- `vehicle` 对应 `[small-vehicle, large-vehicle]`，不得把 `vehicle` 本身写入
  `object_categories`；
- planner 不在 post-validation 中选择具体 backend；catalog 只核实叶子是否存在可执行能力。

## 6. 硬编码的新职责：Verifier，而不是首选解析器

### 6.1 输入顺序

建议把 `agents/counting/target_parser.py` 从“模型 fallback parser”改为纯确定性 reconciler，
输入固定为：

```text
required planner count_target
planner object_categories leaf set
raw question
normalization.count_target_hint
legacy metadata["count_target_hint"]
versioned catalog aliases/parent expansions/capabilities
```

逻辑顺序：

```text
1. 先读取并校验 planner count_target；
2. 再取得最高优先级的确定性 verifier hint；
3. 把 count_target 与 verifier target 确定性展开为 canonical 叶子集合；
4. 核实 planner object_categories 是否等于该任务实际可执行的完整叶子集合；
5. 必要时只按已批准父子关系纠偏，互不相关类别 fail closed；
6. 生成最终 CountTargetSpec 与 executable leaf set；
7. 全程不调用 Qwen。
```

这与当前“hint 先决定，缺失再问 Qwen”的职责不同。planner proposal 是必需输入，hint 是
独立核验信号。

### 6.2 建议的 reconciliation 表

| Planner `count_target` | Planner leaves | Deterministic verifier | 结果 | 稳定审计状态 |
|---|---|---|---|---|
| `small-vehicle` | `[small-vehicle]` | `small-vehicle` | 使用 verifier 的已审计完整 spec | `matched` |
| `vehicle` | `[small-vehicle, large-vehicle]` | `vehicle` | 保留 generic 语义并执行两个叶子 | `matched_parent_expansion` |
| `vehicle` | `[small-vehicle, large-vehicle]` | `small-vehicle` | 收窄 target 和 leaves 为 `small-vehicle` | `planner_scope_broadened_corrected` |
| `small-vehicle` | `[small-vehicle]` | `vehicle` | 恢复 generic target 和完整两叶集合 | `planner_scope_narrowed_corrected` |
| `vehicle` | `[small-vehicle]` | `vehicle` | 补齐缺失的 `large-vehicle` | `incomplete_parent_expansion_corrected` |
| `ship` | `[ship]` | `small-vehicle` | fail closed，不猜测 | `planner_target_conflict` |
| `building` | `[]` | 无 verifier | 接受语义 target，走 generic backend | `planner_only_no_visual_expert` |
| 缺失/非法 | 任意 | 任意 | planner schema 阶段失败，不进入 Agent | `SCHEMA_INVALID` |

这里建议只自动纠正**已知父子范围关系**；互不相干的类别冲突应稳定失败，避免某一侧的错误被
静默掩盖。

### 6.3 `small vehical` 防扩大规则

不能依赖通用模糊匹配。建议增加一个范围很窄、可审计的 lexical specificity guard：

- 识别 `small` / `large` 等已批准 scope modifier；
- 识别 catalog 中明确声明的 vehicle aliases；
- 对用户已提出的 `vehical` 仅作为显式批准的常见拼写变体处理；
- 比较时先做 case、空格、下划线、连字符和单复数的确定性规范化；
- 不使用 Levenshtein 距离把未知词自动吸附到相似类别；
- guard 只允许保持或收窄范围，绝不据此扩大范围。

必须有回归用例：

```text
question: How many small vehicals are visible?
planner count_target: vehicle
planner object_categories: [small-vehicle, large-vehicle]
final CountTargetSpec.canonical_label: small-vehicle
final executable leaves: [small-vehicle]
target-parse Qwen calls: 0
```

对于 VRSBench，现有 `normalization.count_target_hint` 和 ontology 继续保留。是否把
`vehical` 变体加入 `data/adapters/vrsbench/ontology.py`，应以是否属于数据层可复用事实为准；
通用 counting reconciler 不得 import VRSBench adapter。初版倾向于：

- VRSBench 官方/数据事实仍在 adapter ontology；
- 通用的 scope-preservation guard 留在现有 `agents/counting/target_parser.py`；
- 两者通过结构化 hint/字符串输入协作，不建立反向依赖。

### 6.4 生成 `CountTargetSpec`

当 verifier hint 存在并与 planner 相容时，优先复用 hint 中已经审计的 aliases、
inclusion/exclusion rules。

执行类别始终来自核验后的 canonical leaf set，而不是直接从 `CountTargetSpec.canonical_label`
猜测。父类 target 只用于保留用户语义和选择 catalog 中冻结的展开规则。

当没有 verifier hint 时，根据 planner target 生成保守、中性的 `CountTargetSpec`：

```text
canonical_label = normalized planner target
aliases = [original planner target]（仅在规范化后不同）
required_attributes = []
excluded_attributes = []
spatial_constraints = []
inclusion_rule = 只计数与精确目标类别匹配的独立可见实例
exclusion_rule = 排除其他类别、重复视图和无法确认的碎片
ambiguity = []
```

没有 verifier 且 target 不在 canonical catalog 时，`object_categories` 必须为空，不得虚构叶子。
不得为了命中 detector 而把未知/精确 target 改写成更宽的 ExpertCatalog target。
`ExpertCatalog` 后续只负责 capability lookup；无专用 expert 时继续走已有通用 counting backend
策略。

## 7. YOLO 调整的最简单顶层思路

YOLO 权重调整是一个独立的训练/资产交付，不与 planner runtime 代码修改混成同一步。最简单的
顶层路线如下。

### 7.1 冻结 27 类 class contract

1. 保持现有 18 类的索引顺序不变；
2. 追加本计划 §4.4 的 9 个新叶子类；
3. `airplane` 训练标注统一映射到现有 `plane`；
4. 空格、下划线、连字符和大小写差异统一映射到 canonical class id；
5. `vehicle` 等父类不进入 head；
6. class-map 作为版本化资产与权重一起冻结，不能只在 prompt 中手写。

### 7.2 从当前权重扩展

最简单且稳妥的迁移方式：

```text
current 18-class checkpoint
  -> load compatible backbone / neck / detection features
  -> build 27-class detection head
  -> retain/copy old 18 class channels only when training implementation has an audited mapping
  -> initialize 9 new class channels
  -> fine-tune with old-class replay + new-class data
```

如果当前 YOLO 导出/训练实现不能安全地按 anchor 与输出布局复制分类 channel，就不要手工拼接
ONNX tensor；直接重建 27 类 Detect head、加载其余兼容权重并联合 fine-tune，风险更低。

### 7.3 Generic `vehicle` 标注门禁

VRSBench 中只标成 generic `vehicle` 的框不能直接塞进纯叶子 27 类 head。必须三选一：

1. 有可靠人工或官方 small/large 子类标注时，映射为对应叶子；
2. 有经过单独审计、达到批准门槛的转换流程时，生成子类并保留 provenance；
3. 无法可靠区分时，从这次分类训练中排除该 generic vehicle 框。

禁止：

- 同一框同时标成 `small-vehicle` 与 `large-vehicle`；
- 随机分配子类；
- 把 generic `vehicle` 作为第 28 个并列 head；
- 用旧 detector 的未审核预测直接冒充真实训练标签；
- 修改 VRSBench Ground Truth 来迁就训练 head。

### 7.4 最小训练与导出验证

至少验证：

- 27 类 class-map 的 id/name 唯一且顺序冻结；
- 18 个旧类没有明显灾难性遗忘；
- 9 个新类有逐类 precision/recall 或同等检测指标；
- small/large vehicle 混淆矩阵单独报告；
- `airplane -> plane` 等 alias 不产生重复类别；
- 导出 ONNX 后输出维度与 class-map 完全一致；
- 新权重使用新的 logical model id、revision、SHA256 与 catalog version；
- 大型权重保持本地/部署资产，不提交到普通 Git diff。

这个训练工作不改变 planner 的原则：planner 只输出 canonical leaf，运行时由 catalog 把 leaf
映射到新 YOLO 的物理 label。

## 8. 删除第二次 target Qwen

本任务实现后，fresh counting 目标解析不得再包含以下能力：

- `CountTargetParser._parse_via_qwen()`；
- target parser 的 model cache identity 检查；
- target parser 的 request hash / `RequestMeta`；
- `artifact_dir / "target_parse"` 模型调用目录；
- `target_parse_v1.md` prompt 绑定；
- CountingAgent 构造参数 `target_prompt` / `target_prompt_version`；
- bootstrap 对 `catalog["target"]` 的注入；
- target parser 消费 `CallBudget.reserve_qwen()` 的分支。

建议在原有白名单路径 `agents/counting/target_parser.py` 内改造成确定性 resolver，不新增
`resolver.py`、`utils.py` 或兼容包装文件。类名可以在实现时改为 `CountTargetResolver`；不建议
长期保留会让人误以为仍有模型 fallback 的 `TargetParser` alias。

`CountingAgent` 仍保留 Qwen client，因为 Qwen point/quantity proposal 等执行后端仍需要它。

## 9. 没有 VisualTaskPlan 的直连路径

当前 `SampleRunner` 允许显式 task 的内部直连调用省略 plan，`count-image` 也可能通过显式
`CountTargetSpec` 运行。建议冻结以下边界：

### 9.1 有显式结构化 target

```text
VisualTaskPlan absent
+ valid normalization.count_target_hint / explicit count_target_spec
-> 允许零 target-Qwen 执行
```

这保留离线工具、测试和明确用户输入的可用性。该模式的 target 来源是 caller-provided，不能
伪装成 planner-derived。

### 9.2 无 plan 且无显式 target

```text
VisualTaskPlan absent
+ no structured target
-> COUNT_TARGET_SOURCE_REQUIRED
-> 不调用 Qwen
```

不能恢复旧 `target_parse_v1` 作为静默 fallback。

### 9.3 `count-image` 公共命令

初版推荐：

- 用户显式提供 `count_target_spec`：继续直接执行，零 planner 调用；
- 用户没有提供 target spec：先走同一个 `VisualTaskPlanner`，并要求其 task 为 counting；
- planner 返回非 counting task：稳定拒绝，而不是强制改写为 counting；
- 不在 `count-image` 内复制 planner、图片缩略或 target reconciliation 逻辑。

如果 review 希望 `count-image` 永远是显式 target-only，也可以改为缺 target 直接报错；但不建议
保留第二次独立 target Qwen。

## 10. Schema 版本、cache 与 resume

### 10.1 建议升级为 v4

`count_target` 的 task-linked 必填语义会改变 response schema、prompt、cache 和重推理行为，
不应在已定义的 v3 上原地改义。建议在 doc 18 的 confidence 删除完成后升级：

```text
schema version       visual-task-plan-v4
prompt asset         prompts/visual_task_plan_v4.md
prompt catalog       visual_task_plan -> v4
planning mode        visual-task-plan-v4
prompt snapshot      visual_task_plan_v4.runtime.md
artifact basename    visual_task_plan.json（保持不变）
```

request hash 继续覆盖 prompt version、完整 messages、图像 digest 与
`VisualTaskPlan.model_json_schema()`；不得手工让 v4 命中 v3 cache。

### 10.2 当前未提交 v3 改动

开始实现本计划前，先把当前工作树中的 doc 18/v3 confidence 删除改动收口并验证。不要把 v3
半成品与 v4 target contract 混成无法审计的中间版本。

如果 v3 从未发布且维护者明确决定 squash，也可以在实现前批准直接重定义尚未发布版本；这是
review 决策，不应由 coding agent自行假定。本计划默认采用更安全的 v4。

### 10.3 Resume

建议沿用现有历史模式保护：

- v4 succeeded：零 planner/target 模型调用补评测或 Judge；
- v4 非终态：按冻结 v4 request identity 重跑；
- v3/v2 succeeded：只读历史 plan，可零模型补已有产物；
- v3/v2 任意需要重新推理的 counting 样本：稳定拒绝，不用 v4 语义重跑；
- 不从历史 `object_categories=["vehicle"]` 猜 `small-vehicle`；
- 不从 result、GT 或当前 question 反向伪造历史 `count_target`；
- reporting 继续只读显示 v2/v3/v4，历史 artifact 不迁移、不覆盖。

## 11. Trace 与 artifact

`visual_task_plan.json` 保留同一 basename，并在 v4 中持久化已校验的 `count_target`。

Counting trace 建议增加安全、稳定的审计字段：

```json
{
  "target": "small-vehicle",
  "target_source": "visual_task_plan",
  "planner_target": "vehicle",
  "planner_object_categories": ["small-vehicle", "large-vehicle"],
  "executable_leaf_categories": ["small-vehicle"],
  "target_validation": "planner_scope_broadened_corrected",
  "verifier_source": "normalization.count_target_hint"
}
```

约束：

- 不保存原始异常全文；
- 不把整个 question 重复写入 trace；
- 不保存图像、Base64、绝对路径或 prompt 内部推理；
- `planner_target`/`target` 和叶子列表都必须先经过安全/canonical catalog 校验；
- 直连显式 hint 使用 `target_source="explicit_hint"`；
- 没有发生纠偏时也明确记录 `matched` 或 `planner_only`，便于统计 planner 质量。

`CountingResult.target` 与 `counting_attempts.json.target` 使用**核验后的最终 target**，不使用
纠偏前 planner 值。backend plan 的 `target_classes` 使用核验后的叶子集合；父类不得作为物理
detector/segmenter label。

## 12. 预计修改范围

### 12.1 生产代码与资产

预计至少涉及：

```text
agents/schema.py
agents/evidence_catalog.py
agents/evidence_catalog.json
agents/counting/target_parser.py
agents/counting/agent.py
agents/counting/expert_catalog.py
agents/counting/expert_catalog.json
workflows/visual_planner.py
workflows/schema.py
workflows/sample_runner.py
workflows/dataset_runner.py
application/prompts.py
application/settings.py
application/bootstrap.py
application/runtime.py
application/commands/count_image.py
reporting/builder.py
prompts/visual_task_plan_v4.md
prompts/target_parse_v1.md              # v4 完成后删除
```

新 YOLO 权重交付时另行涉及：

```text
configs/yolo.example.yaml
deployment config（只改实际使用的新 logical model/class map）
models/MODELS.md
local weight / exported ONNX（大型本地资产，不提交普通 Git）
```

只有在回归用例证明数据层也需要识别已批准拼写变体时，才修改：

```text
data/adapters/vrsbench/ontology.py
```

不新增 Python 路径，不修改 `architecture/allowed_python_files.txt`。若实际实现发现必须新增路径，
按仓库规则停止并单独申请架构批准。

### 12.2 文档

实现完成后同步：

```text
DETAILS.md
README.md（仅当 count-image 公共行为发生变化）
docs/architecture/16_VISUAL_ONLY_PLANNER_REPLACEMENT_PLAN.md
docs/architecture/18_REMOVE_VISUAL_TASK_PLAN_CONFIDENCE_PLAN.md
docs/migration/JOINT_TASK_VISUAL_PLANNER.md（只记录历史/迁移差异）
```

doc 16 中“规划类别不自动改写 `CountTargetSpec`，需要另行批准”的保留说明，应指向本 doc 19，
而不是静默删除历史决定。

## 13. 实施阶段

### 阶段 A：冻结契约与失败策略

1. 先完成并验证当前 v3 confidence 删除；
2. 冻结 `count_target` 字段与 leaf-only `object_categories` 语义；
3. 冻结 canonical alias、父类展开和当前 18/目标 27 类能力边界；
4. 批准 reconciliation 表，尤其是“父子范围自动纠正、无关类别冲突 fail closed”；
5. 批准 direct/count-image 无 target 时的行为；
6. 冻结 v4 schema、prompt version、planning mode 和 resume policy；
7. 先写 schema、catalog 与 resolver 回归测试，再改实现。

### 阶段 B：planner v4

1. 新增并绑定 v4 prompt；
2. 给 `VisualTaskPlan` 增加 task-linked `count_target`；
3. 将 `object_categories` 从父类 composite 输出改成 canonical leaf 输出；
4. catalog 增加 alias、父类展开与 per-task executable capability；
5. 更新 planner post-validation 与 capability-bound system prompt；
6. 保持 user message 仍只有缩略图/整图 + 原始问题；
7. 验证 prompt、catalog version、response schema 和 image digest 全部进入 request hash；
8. 更新 planning mode、snapshot、artifact 和 trace version。

### 阶段 C：deterministic target reconciliation

1. 把 `target_parser.py` 改为无模型 resolver；
2. 实现 label 安全规范化、alias 等价、parent-to-leaf 展开和 scope hierarchy 比较；
3. 保留 normalization/legacy hint 校验与 invalid-hint fail closed；
4. 加入窄范围 specificity guard；
5. 生成最终 `CountTargetSpec`、executable leaf set 与稳定 resolution audit；
6. 从 CountingAgent 删除 target prompt/model fallback 接线。

### 阶段 D：入口、budget 与历史保护

1. manual ask、dataset explicit/default/auto 均把 v4 plan 传入 CountingAgent；
2. `count-image` 按已批准策略接入同一 planner 或要求显式 target；
3. 删除 target prompt binding 和 `target_parse_v1.md`；
4. 更新 budget 测试，证明不再保留 `:target` Qwen 请求；
5. 更新 v4 fresh 与 v2/v3 historical resume；
6. reporting 同时识别 v2/v3/v4，不重新推理 target。

### 阶段 E：新 YOLO 独立训练与资产门禁

1. 冻结 27 类 class-map，保留旧 18 类顺序并追加 9 类；
2. 审计 generic vehicle 标注的排除或可靠子类化策略；
3. 从当前 checkpoint 加载兼容权重，构造 27 类 head；
4. 使用旧类 replay 与新类数据联合 fine-tune；
5. 完成逐类、small/large 混淆和 ONNX 输出验证；
6. 注册新 logical identity、digest 与升级后的 catalog version；
7. 只有所有门禁通过后，runtime pool 才从 18 扩展到 27。

### 阶段 F：文档和全量审计

1. 更新当前事实文档；
2. 搜索并删除现役 target-Qwen 接线；
3. 保留其他 counting backend Qwen 调用；
4. 运行目标测试、integration、resume、architecture 和 offline suite；
5. 如实记录未运行的 live/model/dataset 验证。

## 14. 测试计划

### 14.1 Planner schema/prompt

- v4 counting plan 缺 `count_target` 拒绝；
- v4 non-counting plan 携带 `count_target` 拒绝；
- `count_target="small-vehicle"` 合法；
- `object_categories=["small-vehicle"]` 合法；
- `object_categories=["vehicle"]` 因父类进入可执行池而拒绝；
- `object_categories` 的 schema 上限能容纳最大批准父类展开，但仍拒绝超限/无界列表；
- generic `count_target="vehicle"` 必须展开为两个 vehicle 叶子；
- alias `airplane` 确定性 canonicalize 为 `plane`，不会形成两个类别；
- 当前 verified catalog union 是 18 类，runtime 只广告启用子集；只有注入已验证新 YOLO
  catalog 时 union 才是 27 类；
- iSAID background 和 OEM `LABEL_N` 永远不进入 pool；
- 路径样式、控制字符、空字符串和超长 target 拒绝；
- prompt 明确禁止 small/large -> generic vehicle；
- user content 仍严格是 ordered images + raw question；
- v4 hash 使用 v4 prompt 与含 `count_target` 的 response schema；
- v3 response 不能作为 v4 plan 通过。

### 14.2 Deterministic resolver

- planner/hint exact match；
- alias、大小写、空格、下划线、连字符、单复数等价；
- `small-vehicle` 与 `vehicle` 的双向 scope mismatch；
- `large-vehicle` 与 `vehicle` 的双向 scope mismatch；
- generic vehicle 的完整双叶展开；
- 缺一个 vehicle 叶子时确定性补齐并记录纠偏；
- `vehicle`、`aircraft` 等父类永远不作为 executable leaf；
- 用户给出的 `small vehical` 回归；
- 无 verifier 时接受任意安全 planner target；
- 无关类别冲突稳定失败；
- malformed normalization hint 继续稳定失败，不降级；
- legacy metadata 只在 normalization hint 缺失时参与核验；
- 任何 resolver 路径都没有 model/cache/budget 调用。

### 14.3 CountingAgent

- plan-derived target 进入 `BackendSelector`；
- 核验后的 executable leaf set 进入 backend plan；
- generic vehicle 只消费 small/large 物理 label，并进行现有跨类别去重；
- 核验后的 target 进入 `CountingResult`、attempt audit 和 trace；
- 有 planner target 时不产生 `request_id=*":target"`；
- 不创建 `target_parse` 模型 artifact；
- budget 只统计 planner 与真正 counting backend 调用；
- 直连显式 hint 可运行；
- 无 plan 且无 hint 稳定失败；
- unsupported task 仍在读图/解析/budget 前失败。

### 14.4 Integration 与 resume

- manual counting：一次 planner target 理解，后续只执行 counting；
- dataset counting：plan artifact 中 target 与最终 trace 可审计；
- auto task materialization 后仍把同一个 plan 传入 Agent；
- `count-image` 两种 target 来源按批准策略运行；
- v4 succeeded resume 零规划调用；
- v3/v2 succeeded 只读补产物；
- v3/v2 incomplete counting 不用 v4 重推理；
- report 能读取无 `count_target` 的历史 artifact。

### 14.5 YOLO class-contract 与资产门禁

- 旧 18 类 id/order 完全保持；
- 只追加 9 个唯一叶子类，总数 27；
- `airplane` 不形成独立于 `plane` 的重复 head；
- `vehicle` 不进入 head；
- class-map 无 alias/separator/case 重复；
- generic vehicle 标注未被随机或重复分配到子类；
- 导出 ONNX 输出维度、class count 与 class-map 一致；
- 新 logical model id、revision、SHA256 与 catalog version 全部改变；
- 未通过资产验证时 planner 仍只看到当前已验证类别。

### 14.6 必跑命令（runtime 实现阶段）

```text
pytest -q \
  tests/agents/counting/test_target_parser.py \
  tests/agents/counting/test_agent.py \
  tests/agents/counting/test_expert_catalog.py \
  tests/agents/test_evidence_catalog.py \
  tests/workflows/test_visual_planner.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/application/test_prompts.py \
  tests/application/test_settings.py \
  tests/application/test_bootstrap.py \
  tests/application/test_runtime.py \
  tests/integration/test_run_dataset_vertical_slice.py \
  tests/integration/test_auto_task_dataset_vertical_slice.py \
  tests/integration/test_dataset_runner_resume.py \
  tests/reporting/test_builder.py
```

涉及 schema/version/import 接线时追加：

```text
pytest -q \
  tests/architecture/test_allowed_python_files.py \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py
```

最后运行仓库可用的完整 offline suite。真实 Qwen、真实数据集、YOLO/SegFormer 权重和硬件验证
应单独报告，不能用 fake client 测试冒充。

## 15. 静态审计

实现完成后执行有范围的搜索：

```text
rg -n "target_parse_v1|_parse_via_qwen|:target|target_prompt_version|catalog\[\"target\"\]" \
  agents workflows application prompts tests DETAILS.md docs
```

预期：

- 生产代码中无现役 target-Qwen 调用；
- migration/history 文档允许保留历史说明；
- counting backend 自己的 Qwen 调用仍存在；
- `count_target_hint` 仍存在，但描述为 verifier/explicit direct source；
- `object_categories` 只包含 canonical leaf，不再包含 evidence composite 父类；
- `vehicle`、`aircraft` 等父类不进入 head/pool，只可作为语义 target 与确定性展开键；
- `small-vehicle`、`large-vehicle` 保持两个独立物理类别；
- `airplane` 只作为 `plane` alias，不形成重复 head；
- OEM `LABEL_N` 和 background 未进入 planner pool；
- 新 YOLO 未通过资产门禁时，9 个新类未被提前发布为 executable。

还应执行：

```text
git diff --check
git status --short
```

## 16. 验收标准

- [ ] fresh counting 的精确目标来自同一次 `VisualTaskPlanner` 输出；
- [ ] counting/fine_grained_counting plan 必须携带安全的 `count_target`；
- [ ] `count_target` 保存用户语义，`object_categories` 只保存可执行 canonical leaves；
- [ ] 父类不进入 YOLO/SegFormer head 或 planner 可执行类别池；
- [ ] generic vehicle 确定性展开为 small/large vehicle；
- [ ] 当前 verified catalog union 为 18 类且 runtime 只广告启用子集；新 YOLO 交付后目标
  union 为 27 类；
- [ ] iSAID 只贡献 15 个已验证前景类，background/OEM placeholder 被排除；
- [ ] YOLO 旧 18 类顺序保留，只追加 9 个唯一叶子类；
- [ ] `airplane -> plane` 等 alias 不产生重复类别；
- [ ] generic VRSBench vehicle 标注没有被猜测、随机或重复分配；
- [ ] `small vehicle` / `small vehical` 不会被扩大成 generic `vehicle`；
- [ ] 硬编码 ontology/hint 仍保留，但只参与核验、纠偏和显式直连；
- [ ] planner/hint 无关类别冲突 fail closed；
- [ ] planned path 不再调用 `target_parse_v1` Qwen；
- [ ] 没有 `:target` request、target parser cache identity 或 target parse artifact；
- [ ] backend/selector/executor 策略未改变；
- [ ] v4 cache 与 v3 隔离；
- [ ] v2/v3 历史 run 不被改写，需重推理时稳定拒绝；
- [ ] UnifiedSample、GT、evaluation、report aggregation、ROI 与模型加载语义未改变；
- [ ] 目标测试、integration、resume 和 architecture tests 真实通过；
- [ ] live/model/dataset 未验证项被明确报告。

## 17. 已确认方向与剩余 Review 点

### 17.1 已按本轮讨论确认

```text
count_target: 保存 counting 用户语义
object_categories: 只保存 canonical executable leaves
YOLO target head: 当前 18 类 + 新增 9 类 = 27 个唯一叶子类
SegFormer pool: 仅 iSAID 15 个 verified foreground leaves
parents: 只作 deterministic expansion，不进入 head/pool
aliases: 只作 canonical mapping，不形成重复类别
```

### 17.2 仍需在实施前最终确认

1. 已知父子范围冲突按 verifier 纠偏，互不相干类别冲突 fail closed；
2. 无 plan 但有显式结构化 target 的 direct counting 继续允许；
3. `count-image` 无显式 target 时接入同一个 upstream planner；
4. 在 doc 18/v3 完成后使用 `visual-task-plan-v4`，不原地改义 v3；
5. VRSBench generic `vehicle` 训练框采用“可靠子类化或排除”，不增加父类 head。

这些剩余项确认后，coding agent 才进入 runtime 实施；新 YOLO 训练与权重交付仍按独立资产门禁
推进。
