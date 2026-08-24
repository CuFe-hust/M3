# Phase 2 Visual Planner 数据进一步处理计划

## 1. 目标与结论

本计划面向以下只读源数据：

```text
data/phase2-train-visualplanning-dedup/
```

目标是在不改变 Visual Planner v5 输入协议的前提下，进一步核对输入，并补齐或纠正
用户明确授权的四个语义字段：

```text
VisualTaskPlan.task
VisualTaskPlan.object_categories
VisualTaskPlan.needs_visual_assistance
VisualTaskPlan.count_target
```

本轮建议采用以下总体路线：

```text
源数据只读冻结
  -> 全量输入与既有 target 审计
  -> 冻结当前 planner/schema/catalog/runtime capability profile
  -> DeepSeek API 仅根据 raw question 生成四字段受限 proposal
  -> 确定性合并与 VisualTaskPlan 复验
  -> 冲突隔离和人工复核
  -> 生成新的派生数据集、protocol snapshot、展开训练消息与审计报告
```

不建议原地修改 `phase2-train-visualplanning-dedup`。建议将最终结果写到新的版本目录，例如：

```text
data/phase2-train-visualplanning-refined-v3/
```

源数据中的每一条 episode 都应保留可追溯关系。无法可靠确定的样本进入 quarantine，不能静默删除、猜测或伪造标注。

## 2. 当前定义依据

进一步处理必须以当前仓库的可执行事实为准，而不是只依据数据包中历史 protocol 的版本字符串。

主要事实来源：

| 文件 | 本计划使用的事实 |
|---|---|
| `agents/schema.py` | `VisualTaskPlan`、字段联动、长度和格式约束 |
| `workflows/visual_planner.py` | 当前 v5 planner 的 post-validation 语义 |
| `prompts/visual_task_plan_v5.md` | 模型输出规则和禁止项 |
| `agents/evidence_catalog.json` | canonical leaves、aliases、parents、task capabilities |
| `agents/evidence_catalog.py` | leaf、parent expansion 和 task capability 的确定性校验 |
| `application/bootstrap.py` | catalog capability 与当前 runtime 实际可执行能力的交集 |
| `application/settings.py` | `visual-task-plan-v5`、catalog v4、ROI 和 preview 冻结参数 |
| `configs/local.yaml` | 当前本地 detector/segmenter 的实际启用状态 |
| `DETAILS.md` | 当前 visual planner、evidence 和 artifact 契约 |

正式处理不能只记录 Git HEAD；脚本同时记录实际 schema、prompt、catalog、
expert catalog、runtime config、teacher prompt 和 response schema 的内容摘要。
因此未提交的工作区能力变化也会进入 refinement identity，不能在 resume 时静默漂移。

## 3. 已完成的只读基线审计

### 3.1 数据规模

当前数据包共有 4,156 条 episode、300 个唯一图像引用：

| source group | split | episode 数 | 唯一图像数 |
|---|---:|---:|---:|
| DOTA | train | 529 | 14 |
| HRSCD | train | 489 | 14 |
| LRS | train | 1,941 | 60 |
| MiniFrance | train | 399 | 12 |
| VRSBench | train | 713 | 180 |
| VRSBench | val | 85 | 20 |

task 分布：

| task | 数量 |
|---|---:|
| `general_vqa` | 3,315 |
| `scene_classification` | 371 |
| `counting` | 330 |
| `spatial_relation` | 140 |

### 3.2 输入层已通过的基础检查

全量只读检查结果如下：

- 4,156 个 `episode_id` 全部唯一；
- 所有 JSONL 行均可解析；
- 每条样本均为一个 system reference、一个 user message；
- 每条 user content 均严格为一个 image block，随后一个原始 text block；
- `messages` 中不存在 Base64 图像；
- 所有相对图像引用均存在且没有路径逃逸；
- `images/sha256/` 下检查过的内容摘要与文件名一致；
- `provenance.request_meta.image_sha256` 与实际图像内容一致；
- 顶层 `image` 与 user message 中的 image reference 一致；
- `protocol_id` 与 system `content_ref` 一致，protocol 文件均存在；
- `target_text` 与 `target` 的 canonical compact JSON 完全一致；
- manifest 中 JSONL 的行数、字节数和 SHA-256 均一致；
- VRSBench train/val 的图像集合无交集；
- 所有现有 target 均能通过当前 `VisualTaskPlan` Pydantic schema；
- 现有 counting target 均能通过当前 catalog 的 counting leaf-expansion 检查。

这些结果说明输入封装可以作为下一阶段的基础，不需要重做图像去重或消息格式。

### 3.3 当前输出的真实缺口

四个授权字段并非物理缺失，而是需要按当前 task taxonomy、计数语义和 runtime
capability 进一步复核；其中 assistance/categories 的语义覆盖尤其不完整：

| task | `needs_visual_assistance=true` | `false` | 当前类别状态 |
|---|---:|---:|---|
| `counting` | 199 | 131 | 已有部分 canonical leaf 标注 |
| `general_vqa` | 0 | 3,315 | 全部为空，尚未完成 evidence 类别标注 |
| `scene_classification` | 0 | 371 | 按当前 task capability 应保持为空 |
| `spatial_relation` | 0 | 140 | 按当前 task capability 应保持为空 |

因此本阶段的主要语义工作量在 `general_vqa`，同时需要复核 counting 的 target/category 精确范围。

一次保守的纯文本词法扫描在 3,315 条 `general_vqa` 中找到了约 1,310 条包含当前 catalog leaf、alias 或 parent 词面的候选样本。这个数字只能用于估算复核工作量，不能直接作为标注结果，因为：

- 出现某个名词不等于该类别需要 evidence；
- 类别可能正是问题要求模型回答的未知答案；
- 同一句话可能包含参照物、背景物或不可执行类别；
- `needs_visual_assistance` 同时受 task、完整类别集合和 runtime capability 限制。

### 3.4 protocol 漂移

当前数据中存在两个 protocol：

| protocol | episode 数 | catalog | `general_vqa` executable categories |
|---|---:|---|---:|
| `protocol-55b1cf02daec5152` | 3,358 | v4 | 18 |
| `protocol-083481b225a48ee9` | 798 | v3 | 0 |

它们都声明 `protocol_version=visual-task-plan-v5`，但能力绑定不同。特别是 VRSBench 的 798 条样本仍绑定 v3，明确不允许 `general_vqa` object evidence。

此外，当前 `agents/evidence_catalog.json` 的 v4 catalog 包含 26 个
`general_vqa` leaves。按当前工作区 `configs/local.yaml` 实际组装验证，冻结的
runtime profile 为：`general_vqa=26`、`counting=18`、
`fine_grained_counting=18`、`grounding=18`。最终数字不写死在标注逻辑中，
而由脚本调用当前 composition root 组装后取得。

结论：正式复标前必须生成一个新的、内容寻址的 protocol snapshot，不能继续仅凭 `visual-task-plan-v5` 字符串判断语义相同，也不能把 catalog 中存在但当前 runtime 未启用的类别标成可执行 assistance。

### 3.5 重复输入与目标冲突

当前有 6 组完全相同的 `(image, raw question)` 输入重复出现，共多出 6 行：

- 2 组 target 完全一致；
- 4 组 target 不一致；
- 不一致主要来自 `region_request.roi_xyxy` 不同；
- 其中两组 `Is the scene urban or rural?` 被标成非常小的 explicit ROI，与全局场景问题存在明显语义风险。

同一确定性 planner 输入不应具有不同训练目标。处理时应保留所有源 episode 的 provenance，但同一输入组必须共享同一个经复核的 canonical target；无法判定时整组进入 quarantine。

## 4. 需要冻结的最终标注契约

### 4.1 输入契约保持不变

模型每条样本只能看到：

```text
按源顺序排列的 image block(s)
  +
未经包装、未经改写的原始 question 文本
```

不得向标注模型发送或拼入：

- Ground Truth 或最终答案；
- source annotation 内容；
- dataset-specific question type；
- 旧 target；
- 本机绝对路径；
- backend、checkpoint、device 或 detector 阈值；
- provenance 中可能影响判断的隐藏标签。

旧 target 只能在模型标注完成后用于差异审计，不能作为 teacher prompt 的暗示。

### 4.2 四字段修订边界

本轮只允许 DeepSeek proposal 和确定性合并修订：

```text
task
needs_visual_assistance
object_categories
count_target
```

`version`、`region_request`、`reason_codes` 和其他 target 字段保持源值。
`target_text` 是 target 的必需序列化镜像，protocol reference 和
manifest 摘要是派生数据的必需包装同步，不属于新增标注字段。
teacher/request/review 审计信息只写入独立 audit sidecar，不向
episode `provenance` 新增字段。

### 4.3 `object_categories` 的准确含义

`object_categories` 不是“图中所有可见物体”，也不是通用实体抽取结果。它是：

> 当前 task 为完成本问题而请求确定性视觉 evidence 时，允许 evidence executor 实际执行和渲染的完整 canonical leaf 白名单。

最终字段必须满足：

- 只允许当前冻结 catalog 中的 canonical leaf；
- 只允许当前 task 声明支持的 leaf；
- 只允许当前冻结 runtime profile 中真正可执行的 leaf；
- alias 必须先 canonicalize；
- parent 必须完整展开，不能只选部分 children；
- 多个 leaf 按 catalog 顺序稳定排列；
- 不重复；
- 最多 8 个；
- 不放入 raw model label、未知类别、路径、答案或属性词；
- `needs_visual_assistance=false` 时必须为空；
- 无法得到完整可执行 leaf 集合时必须为空，不能截断到前 8 个。

为支持审计，可以在 sidecar 中保存 `semantic_mentions` 和 `unsupported_mentions`，但它们不得进入最终 `VisualTaskPlan.object_categories`。

### 4.4 `needs_visual_assistance` 的准确含义

当前字段名是 `needs_visual_assistance`，不是一个独立的“图像是否有用”字段。
按用户确认的 text-teacher v6 目标，它不再受业务 task gate 限制：只要问题能识别
出相关且可调用的子模型类别，即可提供非权威辅助证据，最终判断仍由 VLM 结合
原图完成。

```text
1. 至少一个相关类别可映射为全局可调用子模型 leaf；
2. 类别数不超过 schema 上限；
3. 对象类别识别题不会因计划字段直接泄漏答案。
```

任一条件不成立时：

```json
{
  "needs_visual_assistance": false,
  "object_categories": []
}
```

为了区分“确实不需要”和“需要但当前能力不可执行”，建议在审计 sidecar 中记录稳定 decision code，而不是改变 planner schema，例如：

```text
assistance_enabled
no_callable_category
category_is_requested_answer
annotation_conflict
```

### 4.5 task-specific 决策表

| task / 问题类型 | `needs_visual_assistance` | `object_categories` |
|---|---|---|
| counting，目标含可调用基础类别，包括颜色/运动等限定目标 | true | 基础 leaf 或 parent 展开；完整保留 `count_target` |
| counting，完全无法识别可调用基础类别 | false | `[]`，保留原 `count_target` |
| general VQA，问题可识别出相关可调用类别 | true | 相关可调用 leaves，允许有用的非完整子集 |
| general VQA，类别本身是问题要求识别的答案 | false | `[]`，不得用 target category 泄漏答案 |
| scene classification | true | 最多八类的场景证据 profile，并优先保留问题显式类别 |
| spatial relation | true（存在可映射实体时） | 所有可映射关系参与者类别；允许部分实体覆盖 |
| caption / change | 由问题文本决定 | 只有文本识别出可调用类别才启用；generic caption 关闭 |
| general VQA，只给颜色/形状/框而没有可安全确定的语义类别 | false | `[]` |
| grounding，已知文本目标完整映射到可执行 leaves | true | 目标 leaves |
| scene classification / spatial relation / caption / change / multiple choice | false | `[]`，除非未来 catalog 和执行路径另行版本化扩展 |

`region_request` 与 assistance 是两个独立决定。`needs_visual_assistance=false` 不代表必须删除合法 ROI；反之，存在 explicit ROI 也不自动意味着需要 object evidence。

### 4.6 counting 的特殊约束

counting 继续遵循当前严格规则：

```text
count_target = 用户请求的精确语义范围
object_categories = 该 target 在当前 task 下的完整可执行 leaf 集合
```

示例：

| question target | `count_target` | `object_categories` | assistance |
|---|---|---|---|
| small vehicles | `small-vehicle` | `["small-vehicle"]` | true |
| vehicles | `vehicle` | `["small-vehicle", "large-vehicle"]` | true |
| blue buildings | `blue building` | `[]` | false |
| terminals | `terminal` | `[]` | false |

不能为了命中 detector 把 `blue building` 扩大成 `building`，也不能丢掉 small、large、color、motion 等改变计数范围的限定词。

### 4.7 general VQA 的防答案泄漏规则

以下问题即使图中真实类别属于 catalog，也不能把真实答案写进 `object_categories`：

```text
Identify the object category within the given bounding box ...
What kind of object is shown ...
Which category does the highlighted object belong to ...
```

因为类别就是需要最终 Agent 回答的未知信息。Visual Planner 禁止输出答案，训练标注也不能通过 evidence category 偷渡答案。

相反，以下类型可以成为正样本候选：

```text
Is there a ship in the upper-right area?
What color is the plane near the runway?
Are the two storage tanks adjacent?
```

但只有在 task、完整类别集合和当前 runtime capability 均合法时才可设为 true。

## 5. 处理阶段

## 5.1 阶段 0：源数据冻结与可恢复运行

1. 将 `data/phase2-train-visualplanning-dedup` 设为只读输入，不原地覆盖。
2. 记录源 `manifest.json` SHA-256、每个 JSONL SHA-256、protocol SHA-256 和图像索引摘要。
3. 记录 Git HEAD、dirty 状态以及 schema/prompt/catalog/config 的实际文件摘要。
4. 为每次处理创建唯一 `refinement_run_id`。
5. 所有结构化输出使用临时文件加原子 replace；JSONL 只在单进程 writer 中提交。
6. API cache、accepted、quarantine 和 audit 状态分离，使中断后可以 resume。
7. resume 只接受完全一致的 source manifest、protocol digest、model identity、规则版本和参数；冲突时稳定拒绝。

## 5.2 阶段 1：全量输入审计

正式处理前重复并固化以下检查：

- JSONL 编码、逐行 JSON 和字段类型；
- `episode_id` 唯一性；
- dataset/source_group/split 与文件路径一致；
- user content 顺序为 image(s) 后 raw text；
- 顶层 `image` 与 message image 一致；
- 图像路径为安全相对路径；
- 图像存在、可解码、EXIF transpose/RGB 规范化可执行；
- canonical image SHA-256 与引用、request meta 一致；
- protocol 引用存在且摘要匹配；
- `target` 通过当前 schema；
- `target_text` 等于重新序列化的 target；
- manifest 计数和摘要闭合；
- train/val 不共享图像或完全相同输入；
- 完全相同输入的 target 是否一致。

输入审计失败的样本先进入 quarantine，不进入 API 请求队列，避免把格式错误伪装成模型标注问题。

## 5.3 阶段 2：冻结新的标注 protocol

生成一个新的 content-addressed protocol snapshot，至少绑定：

```text
planning_mode
task_prompt_version
完整 system prompt
VisualTaskPlan JSON Schema 及摘要
catalog version 及 catalog 文件摘要
aliases
parent expansions
canonical leaves
task catalog capabilities
当前 runtime executable categories by task
preview_max_side
ROI coordinate frame / quantum / materialization policy
teacher annotation schema version
```

建议所有本次复标样本统一引用这个新 protocol。不能继续让 VRSBench 保留 v3 binding、其他数据使用 v4 binding，却在训练时把两者当成同一个 planner 语义。

默认建议冻结“当前本地可执行 profile”：

- YOLO-backed 18 leaves 可进入当前 counting/general_vqa/grounding 的 executable pool；
- disabled SegFormer-only leaves 不进入最终 assistance categories；
- catalog 中仍保留这些 leaf，用于审计 `category_not_runtime_executable`；
- 如果后续启用 `segmenter_mitb2_001` 或 `segmenter_oem_001`，必须生成新的 protocol ID 并单独重标，不能静默复用本次标签。

## 5.4 阶段 3：确定性预标注

规则层只自动接受能够精确证明的结果。

### A. 直接保持 false 的任务

对当前没有 object-evidence task capability 的任务：

```text
scene_classification
spatial_relation
caption
change_caption
change_qa
multiple_choice_vqa
```

固定：

```json
{
  "needs_visual_assistance": false,
  "object_categories": []
}
```

仍要独立检查 task 和 region 是否存在明显语义冲突，但不能因为问题中出现 catalog 名词就绕过 task capability。

### B. counting

1. 从 raw question 重新判断 `count_target`，保留所有 scope-changing
   限定词，并将可精确映射的基础类别规范到 canonical leaf、alias
   target 或 declared parent；旧值只用于 diff 审计；
2. 用 catalog alias/parent expansion 做确定性映射；
3. 与冻结 runtime executable pool 求完整可执行性；
4. 完整可执行时生成精确 leaves 并设 true；
5. 未知、属性范围无法由 specialist 表达或能力不完整时设 false + `[]`；
6. 与旧 target 不一致时记录 diff，不静默覆盖；
7. 模型输出不得推翻更严格的 deterministic scope 规则。

### C. general VQA 候选生成

规则层可以生成候选，但默认不直接把词法命中写入最终 target：

1. 从 raw question 提取 canonical leaf、alias 和 parent mention；
2. 标记类别是否是已知输入条件、参照物，还是问题要求预测的答案；
3. parent 按 catalog 完整展开；
4. 检查 task runtime capability；
5. 检查 leaf 数量是否超过 8；
6. 生成 `candidate_categories`、`negative_gate` 和 decision code；
7. 将候选送入纯文本 DeepSeek teacher；teacher 仍不得读取图像或答案。

规则层不得读取图像 annotation 或最终答案来补类别。

## 5.5 阶段 4：DeepSeek 纯文本 API 标注

### 推荐策略

使用 DeepSeek 作为只看问题文本的 teacher，并要求严格 JSON Schema 输出。
由于它不看图像，任何仅凭图像才能确定的类别都必须 fail closed；输出仍标为
`unreviewed`，在完成人工复核前不能视为最终金标。

API 不直接输出整个可自由变化的 target，而只输出一个四字段受限
annotation proposal：

```json
{
  "task": "general_vqa",
  "needs_visual_assistance": true,
  "object_categories": ["ship"],
  "count_target": null
}
```

随后由本地确定性代码把 proposal 合并到现有 target，再通过完整
`VisualTaskPlan`、catalog 和 runtime post-validation。API 负责判断四个授权
字段；本地代码独立保护旧 `region_request`、`reason_codes` 和其他
未授权字段，并对 task schema、counting exact scope、task capability、
answer leakage、parent 完整展开、runtime 可执行性和 8 类上限做硬门控。

teacher system message 必须同时包含：版本化的 text-teacher rubric、精确
四字段 JSON Schema、完整 runtime v5 prompt 和当前 planner binding。
schema repair 必须保留原始 question payload，不得退化为无样本语义的
纯格式修复。

teacher 输入只包含：

```text
冻结的 annotation system prompt + planner binding
{"question": raw_question}
```

不得发送图像、旧 target、答案、数据集名、provenance、路径或 annotation。

建议参数：

- temperature 设为 0 或供应商提供的最接近确定性的设置；
- 启用 strict JSON Schema；
- 不请求 chain-of-thought；
- 不保存隐藏推理；
- 网络失败、超时、限流和 schema failure 使用稳定状态码；
- 只对可重试的传输/格式错误做有限重试；
- 不因重试而改变 prompt 或模型；
- request identity 覆盖模型逻辑身份、generation settings、prompt/schema/catalog/runtime profile 和 raw question；
- API key 只从 composition root 的环境变量或无回显终端提示读取，绝不进入
  设置快照、命令参数、JSONL、日志或错误文本。

### 调用范围

推荐的完整质量模式：

```text
对全部 3,315 条 general_vqa 做一次 primary teacher 标注；
对全部 positive proposal、全部规则/API 冲突、全部 duplicate conflict
以及分层抽样的 negative proposal 做第二次 verifier 或人工复核。
```

预算受限模式：

```text
优先处理约 1,310 条词法候选；
再从其余 general_vqa negative pool 按 source group、reason code 和问题模板
分层抽取至少 10%，用于发现词法规则漏召回。
```

预算受限模式只能作为 pilot，不能在没有漏召回评估的情况下直接发布最终训练集。

虽然只有 300 个唯一图像，但每个问题必须保持单 episode planner 语义，不能把同图多个问题拼成一个 teacher prompt。可以按 image digest 复用上传或视觉缓存，但输出 cache key 必须包含 raw question。

## 5.6 阶段 5：确定性合并与冲突处理

合并优先级建议固定为：

```text
schema/task hard gate
  -> counting exact catalog rule
  -> answer-leakage negative gate
  -> runtime capability gate
  -> verified teacher proposal
  -> unresolved quarantine
```

具体规则：

1. API 返回 parent 时本地完整展开；返回 alias 时本地 canonicalize；
2. API 返回未知、task 不支持或 runtime 不可执行 leaf 时拒绝 proposal；
3. API 返回的类别多于 8 时不得截断，整条标为不可执行；
4. categories 为空但 assistance=true，或反向组合，一律拒绝；
5. counting 的 API `object_categories` 不得覆盖确定性
   `count_target -> leaves` 规则；`count_target` 可修订，但必须保留所有改变
   计数范围的限定词；
6. 类别识别问题触发 answer-leakage gate 时必须 false + `[]`；
7. 合并后重新构造完整 `VisualTaskPlan`，不能只做字符串替换；
8. `target_text` 必须由最终 target 重新 canonical serialize；
9. 完全相同的 `(protocol, image digest(s), raw question)` 必须共享同一最终 target；
10. 重复 episode 保留 provenance，不因重复而静默丢行；是否在训练采样时降权属于后续训练策略，不在本标注任务中决定。

`task` 和 `count_target` 属于本轮用户明确授权的可修订字段，经确定性门控
后可进入派生 target，但所有变更必须进入人工 review 队列。
`region_request`、`reason_codes`、`version` 与其他字段仍是 protected；teacher
建议修改这些字段时必须拒绝。

## 5.7 阶段 6：人工复核

以下样本要求 100% 人工复核：

- 所有 `needs_visual_assistance=true` 的 general VQA；
- 所有 parent expansion；
- 所有 API 与规则不一致的样本；
- 所有 duplicate-input target conflict；
- 所有 teacher verifier 不一致；
- 所有类别数接近或超过 schema 上限的样本；
- 所有 rare leaf，若某 leaf 少于 50 条则复核该 leaf 的全部正样本；
- 所有 answer-leakage gate 命中但 teacher 给出 positive 的样本；
- 所有 `task` 或 `count_target` 变更；
- 所有 protected-field disagreement。

negative 样本按 source group、task、reason code、问题模板和候选类别分层抽样：

```text
至少抽 5%；
每个 source group / 主要 reason code 组合至少 20 条；
不足 20 条时全量复核。
```

人工界面应同时显示图像、raw question、冻结的可执行类别表、proposal、decision code 和旧/新 target diff，但不显示 Ground Truth 或最终答案。

## 5.8 阶段 7：最终编译

最终 JSONL 保持现有训练核心结构：

```text
schema_version
episode_id
protocol_id / protocol_version
dataset / source_group / split
image / source_image_id
messages
response_model
target
target_text
provenance
```

根据本轮“不增加 episode 字段”的要求，以下安全、可复现的最小信息
只写入 audit sidecar，不新增 `provenance.refinement`：

```text
refinement_run_id
source_protocol_id
label_source = deterministic | teacher_verified | human_reviewed
decision_code
rule_version
teacher logical model id / revision（如实际调用）
teacher request hash（如实际调用）
review_status
```

不写入 API key、Authorization header、Base64、绝对路径、隐藏推理或原始异常全文。

每条 accepted target 都重新生成 `target_text`。最终 protocol reference、manifest 计数和 SHA-256 也全部重算。

## 6. 建议的派生数据目录

```text
data/phase2-train-visualplanning-refined-v3/
├── README.md
├── manifest.json
├── protocols/
│   └── protocol-<content-hash>.json
├── datasets/
│   ├── DOTA/train.jsonl
│   ├── HRSCD/train.jsonl
│   ├── LRS/train.jsonl
│   ├── MiniFrance/train.jsonl
│   └── VRSBench/{train,val}.jsonl
├── training/
│   └── ...  # 已展开的 system/user/assistant chat records
├── training_images/
│   └── sha256/...  # 与推理 planner 完全相同的确定性 PNG previews
├── images/
│   └── ...  # verified hardlinks or copied canonical files; no unsafe symlink
└── audit/
    ├── input_audit.json
    ├── label_decisions.jsonl
    ├── protected_field_disagreements.jsonl
    ├── duplicate_groups.jsonl
    ├── quarantine.jsonl
    ├── distribution.json
    └── refinement_run.json
```

如果使用 hardlink 复用图像，manifest 应明确记录 link policy，并重新验证每个目标文件内容摘要。不要依赖逃逸到源目录的相对 symlink，以免破坏数据包可移植性和路径安全。

## 7. 质量门与验收标准

### 7.1 结构门：必须 100% 通过

- 所有 accepted 行可解析；
- 所有 accepted target 通过当前 `VisualTaskPlan` schema；
- `target_text` 与 target 完全一致；
- 所有 image/protocol ref 存在且安全；
- manifest 的行数、字节数和 SHA-256 闭合；
- `needs_visual_assistance=true` 当且仅当 categories 非空；
- 所有 categories 都是 canonical executable leaves；
- 无重复 leaf，顺序稳定，长度不超过 8；
- counting known target 的 leaf expansion 完整且精确；
- 非 counting task 的 `count_target` 为 null；
- unsupported task 的 assistance 必须为 false；
- duplicate identical inputs 的最终 target 完全一致；
- 不包含 Base64、secret 或机器绝对路径；
- `accepted + quarantine == 4,156`，任何样本都可追踪。

### 7.2 语义门

- 类别识别问题不得通过 `object_categories` 泄漏答案；
- 不把 image 中所有可见对象当成 evidence 请求；
- 不把未知或属性受限 target 扩大成可执行父类；
- parent expansion 必须完整；
- runtime 不可执行类别不能标为 assistance=true；
- `region_request` 与 assistance 独立；
- teacher/规则分歧未解决的样本不得进入 accepted；
- 人工抽检的字段错误率建议小于 1%，且任何单一 leaf 的错误率小于 2%；未达到时回到规则/API 阶段修订并全量重跑，而不是只修抽检样本。

### 7.3 分布报告

最终至少报告：

- source group / split / task 的 accepted、quarantine 数；
- true/false assistance 数与比例；
- 每个 leaf 的 episode 数；
- parent expansion 数；
- decision code 分布；
- deterministic / teacher / human 的标注来源分布；
- old target 与 new target 的变化矩阵；
- protected-field disagreement 数；
- duplicate group 和冲突解决情况；
- train/val 图像及精确输入交集；
- API 调用、cache hit、重试、失败和 verifier disagreement 数。

不能只给总体比例。每个 source group 都应单独查看，避免某个数据源的模板占据全部 positive 或 negative。

## 8. Pilot 建议

在全量 API 标注前先做一个 300 条左右的分层 pilot：

```text
80  条 counting：已知 leaf、parent、unknown/attribute target 混合
160 条 general_vqa：存在、颜色、类别识别、全局推理、局部关系混合
30  条 scene_classification
30  条 spatial_relation
```

pilot 必须覆盖：

- 5 个 source groups；
- v3 和 v4 两种历史 protocol；
- active YOLO leaves；
- disabled SegFormer-only leaves；
- parent aliases；
- answer-leakage cases；
- duplicate/conflicting inputs；
- explicit ROI 和 full-image cases。

pilot 通过条件：

1. 结构门 100% 通过；
2. 两位复核者对 assistance/category 的一致率达到预设门槛，建议不低于 98%；
3. answer leakage 为 0；
4. counting scope broadening 为 0；
5. API 相同输入重复执行的结构化结果一致；
6. 根据 pilot 更新规则版本和 annotation prompt 后，冻结版本再启动全量处理。

## 9. 实现边界与仓库约束

当前实现使用职责明确的专用 visual-planner 数据复标脚本与测试：

```text
scripts/refine_visual_planner_dataset.py
tests/test_refine_visual_planner_dataset.py
```

不应把该职责塞进现有 `scripts/prepare_qwen3vl_phase2_sft.py` 或
`scripts/qwen3vl_phase2_data.py`；后两者具有不同、已冻结的训练数据职责。

实现时还必须保持：

- 默认离线；只有显式 `--use-api` 才允许联网；
- API 模型由参数和环境变量注入，不硬编码 provider secret；
- source dataset、GT、split 和 episode 纳入规则只读；
- 不修改 `UnifiedSample`、Router、Agent、evaluation、reporting、CLI 或 resume 语义；
- 不下载模型或数据集；
- 不修改 Golden fixtures；
- 不在 `data/` 层调用生产 Agent；
- 注释使用英文在前、中文在后；
- 输出通过原子写入并支持稳定 resume；
- 错误记录使用稳定 code，不持久化原始敏感异常。

## 10. 测试与后续门禁

当前单元测试覆盖 runtime protocol、展开训练消息、question-only payload、
answer-leakage、alias/parent、counting scope、unsupported task、ROI 保护、
duplicate conflict 和端到端编译。正式发布数据前还应补齐或实际运行以下门禁：

1. input message/image/protocol/path 安全校验；
2. current schema/catalog/profile snapshot digest；
3. false/categories 联动；
4. canonical leaf、alias、parent expansion 和稳定排序；
5. counting exact scope；
6. answer-leakage negative gate；
7. unsupported task hard gate；
8. runtime unavailable category；
9. category 数量超过 8 时 fail closed；
10. API strict schema、cache identity、有限重试和 secret redaction；
11. protected field 不被 API 自动改写；
12. duplicate input canonicalization；
13. quarantine 闭合；
14. atomic output 与 resume identity；
15. manifest/target_text/content hash 重建；
16. train/val leakage 检查。

并运行现有相关回归：

```text
pytest -q tests/workflows/test_visual_planner.py
pytest -q tests/agents/test_evidence_catalog.py
pytest -q tests/contracts/test_data_schema_contract.py
pytest -q tests/architecture/test_implementation_status.py
pytest -q tests/architecture/test_import_boundaries.py
git diff --check
```

只有实际运行并通过的检查才能在最终报告中写为通过。

## 11. 建议执行顺序与停点

### Checkpoint A：协议确认

在调用任何 API 前确认：

- 使用当前 local runtime 已实际验证的 26-leaf General VQA profile；
- 新 protocol snapshot 内容和摘要；
- assistance 的 task-specific 决策表；
- answer-leakage 规则；
- 派生数据目录名。

这是最重要的停点。profile 未冻结前不应开始大规模标注。

### Checkpoint B：300 条 pilot

完成 deterministic proposal、teacher proposal、人工复核和错误分析。若一致率或 leakage 门未通过，修改规则/prompt 并从 pilot 重新开始。

### Checkpoint C：全量复标

冻结 rule version、teacher identity 和 request identity，处理全量样本；中断后只按完全一致的 run identity resume。

### Checkpoint D：全量验收

运行结构门、语义门、分布报告、duplicate consistency 和 split leakage 检查。quarantine 未清零并不等于失败，但必须逐条可解释，且不能被静默纳入训练集。

### Checkpoint E：发布

只发布新的派生目录和 manifest；源目录保持不变。README 记录：

- 数据来源；
- 当前 planner/catalog/runtime profile；
- 是否使用 API；
- teacher 逻辑身份和 revision；
- accepted/quarantine 数；
- 已知限制；
- 所有实际执行的检查和结果。

## 12. 非目标

本阶段不做以下事项：

- 不生成最终 VQA 答案或 Ground Truth；
- 不改变 task 集合或 Router 映射；
- 不改变 `VisualTaskPlan` schema；
- 不训练或启用 detector/segmenter；
- 不改变 ROI 几何协议；
- 不改变 evaluation 指标；
- 不改变数据 split；
- 不因标签困难而删除源样本；
- 不将 catalog 中存在但 runtime disabled 的类别伪装成可执行 evidence；
- 不把 teacher 的自由文本解释、隐藏推理或答案写入训练 target。

## 13. 完成定义

只有同时满足以下条件，才认为数据进一步处理完成：

1. 输入全量审计闭合；
2. 新 protocol 对当前 schema、prompt、catalog 和 runtime profile 完整绑定；
3. 4,156 条源 episode 全部进入 accepted 或 quarantine；
4. accepted 中 `task`、`count_target`、`object_categories` 和
   `needs_visual_assistance` 满足本计划的 task-specific 契约；
5. duplicate identical inputs 不再存在目标冲突；
6. answer leakage 为 0；
7. 所有 target 通过 schema、全局 callable-leaf 和四字段保护校验；
8. target_text、manifest、文件摘要和路径安全检查通过；
9. 人工复核与分层抽检达到质量门；
10. API、规则、人工修改和失败样本均有可复现审计记录；
11. 源数据目录未被修改；
12. 最终报告如实记录未验证项和剩余风险。

## 14. 2026-08-22 实施结果（待人工 review）

本轮按 text-teacher v5 与当前 runtime binding 生成派生目录
`data/phase2-train-visualplanning-refined-v2`，未覆盖只读源目录。teacher 逻辑
模型为 `deepseek-v4-flash`，冻结 protocol 为
`protocol-4e5f3a925b09112d`。

- source：4,156；accepted：4,146；quarantine：10；
- quarantine：8 条 `DUPLICATE_REGION_CONFLICT`、1 条
  `DEEPSEEK_JUDGE_INVALID_JSON`、1 条
  `COUNT_TARGET_MISSING_AFTER_TASK_GUARD`；
- assistance：374 true / 3,772 false；
- accepted task：counting 332、general_vqa 2,994、
  scene_classification 421、spatial_relation 398、caption 1；
- source manifest SHA-256 在实施前后保持
  `fec2b4f67e0d6d9ab2e9ff42cbb857968640cc748153a789958455e8f8ab2ce6`；
- 本轮产物仍为 `unreviewed`。上文要求的人工复核与分层抽检尚未由人工签字，
  因此不能宣称为最终金标或进入 Checkpoint E 发布状态。

## 15. text-teacher v6 修订

用户进一步确认：数据标注阶段暂不采用当前代码中的 task-specific Visual Plan
开关；只要问题能识别出可调用子模型类别，就允许启用 Visual Plan。检测、分割
结果只是交给最终 VLM 的辅助证据，原图仍是语义权威，漏检不表示目标不存在。

因此 v6 使用全局 callable leaf 并集，允许 scene classification、spatial
relation 和其他 task 输出相关类别，也允许颜色、运动等限定计数调用基础类别
子模型。v5 的 `task_not_supported`、必须完整覆盖全部实体以及限定目标不得调用
基础类别等规则不再适用。新结果写入 v3 派生目录，v2 保留为历史审计版本。

### 15.1 v6.1 区域 caption 边界

人工 review 进一步确认：caption 不限于整图描述。若问题要求对一个明确 ROI、
局部区域或命名目标周边进行开放式描述，也归入 `caption`。例如 “How would you
describe the activity around the bottom-most bridge?” 属于区域 caption。给定框后只问
类别、颜色、朝向或运动状态仍是封闭式 `general_vqa`，不得仅因存在 ROI 改成
caption。本地高精度 task guard 固化该边界，避免 teacher 边界漂移。本次 review
据此将 3 条局部开放描述从 `general_vqa` 调整为 `caption`；v3 的 caption 分布由
1 条变为 4 条，其他三个授权字段保持不变。

## 16. structured supplement v1

为补齐 v3 中缺失或极少的 task，新增完全离线、确定性的结构化补充阶段，输出到
`data/phase2-train-visualplanning-refined-v4`，不覆盖 refined-v3：

- VRSBench：从 train/val 的 caption、objects 与 qa_pairs 选择 900 张未被 v3
  使用的图像，同一图构造 `caption`、`grounding`、
  `fine_grained_counting`、`multiple_choice_vqa` 各 900 条；
- LEVIR-CC：从 `LevirCCcaptions_readable.jsonl` 选择 900 对图像，同一 A/t1 →
  B/t2 图对构造 `change_caption`、`change_qa` 各 900 条；
- 每类配额固定为 train 800、val 100；test split 完全排除；LEVIR 的 changed 与
  unchanged 在每个 split 内等量；
- VRS MC 只使用可严格构造同语义选项的数值计数、Yes/No 存在性和颜色问题，
  数字与英文数字词按数值去重；不持久化答案键；
- 源 caption/object 标注只用于生成 planner 四字段监督和 evidence 类别，不把
  caption、box 或最终 VQA answer 写入训练记录；
- 双图消息保持同一 chat 格式，user content 为两个有序 image block 后接 raw
  question；顶层 `image` 继续引用第一张图以保持现有格式兼容；
- 全程零网络调用，图片按 SHA-256 内容寻址并优先硬链接，源目录保持只读。

实施结果：新增 5,400 条，v4 共 9,546 条 accepted、10 条历史 quarantine；十种
task 全覆盖。最终 task 分布为 general_vqa 2,936、caption 904、
fine_grained_counting/grounding/multiple_choice_vqa/change_caption/change_qa 各
900、scene_classification 463、spatial_relation 403、counting 340。Visual Plan
assistance 为 6,891 true / 2,655 false；唯一图像 2,998，图像 block 11,346。

## 17. structured supplement v2 修正（caption/change 隐藏标注注入）

人工 review 确认 v1 补充逻辑存在两处隐藏标注注入：

- 900 条 generic caption 从 VRSBench 源物体标注直接写入 `object_categories`，
  但问题文本（如 “Write a detailed caption for this image.”）不包含任何类别；
- 898 条 change_caption/change_qa 从 LEVIR 参考 caption 抽取类别，而问题文本
  （如 “Are the two temporally ordered scenes visually different?”）同样不含类别。

v2 修正（已就地应用到 refined-v4，并同步修正生成脚本）：

- caption/change 的 `object_categories` 只允许由问题文本驱动，复用
  `_question_evidence_categories`；删除 `levir_categories` 与
  `_LEVIR_CATEGORY_PATTERNS` 从参考 caption 抽取类别的路径；
- 修复 1,798 条（caption 900 + change_caption 449 + change_qa 449）为
  `needs_visual_assistance=false`、`object_categories=[]`；
- `target_text`、`training/` 展开消息、`manifest.json` 哈希、
  `audit/distribution.json`、`audit/supplement_run.json` 同步更新；全部补充行
  provenance `supplement_policy_version` 更新为
  `visual-planner-structured-supplement-v2`；
- 修复后 v4 总 assistance 为 5,093 true / 4,453 false；补充部分为 2,610 true /
  2,790 false；
- 修复明细与验证结果见
  `data/phase2-train-visualplanning-refined-v4/audit/supplement_assistance_fix.json`；
- `audit/label_decisions.jsonl` 与 `audit/supplement_selections.jsonl` 保留为 v1
  原始运行历史，不改写。

## 18. structured supplement v2 修正（grounding 类别改由文本推导）

人工 review 确认 v1 的 grounding 类别仍从 VRSBench 源 `obj_cls` 展开：源类别
`vehicle` 固定展开成 `["small-vehicle", "large-vehicle"]`，但不少 referring
sentence 已明确限定 small/large，导致 376 条 target 无法由问题文本类别提取器
完整复现（其中 209 条为 vehicle 过度展开）。

v2 修正（已就地应用到 refined-v4，并同步修正生成脚本）：

- grounding 的 `object_categories` 改由 referring sentence 文本推导，复用
  `_question_evidence_categories(task="grounding")`；隐藏的源 `obj_cls` 不再注入
  类别；
- `small vehicle -> ["small-vehicle"]`、`large vehicle -> ["large-vehicle"]`、
  裸 `vehicle -> ["small-vehicle", "large-vehicle"]`；
- 同义表达通过显式文本 alias 映射：`_TEXT_CATEGORY_ALIASES` 新增
  `baseball field -> baseball-diamond`、`track and field -> ground-track-field`、
  `truck -> large-vehicle`；
- 共更新 376 条（train 345 / val 31）；2 条 referring 无可推导类别，关闭为
  `needs_visual_assistance=false`、`object_categories=[]`；
- 修复后 grounding 900 条全部可由问题文本类别提取器完整复现（mismatch=0）；
- 明细见
  `data/phase2-train-visualplanning-refined-v4/audit/supplement_grounding_categories_fix.json`。
