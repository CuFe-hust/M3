# XLRS VQA question-with-choices v2

## Intentional difference / 有意行为差异

XLRS-Bench-lite previously stored only the question stem in
`UnifiedSample.question`, while choices were available only through
`TaskNormalization.choices`. The visual-only planner therefore could not
reliably distinguish `multiple_choice_vqa` from free-form `general_vqa`.

XLRS-Bench-lite 过去只把题干写入 `UnifiedSample.question`，选项仅存在于
`TaskNormalization.choices`。因此纯视觉 Planner 无法可靠区分
`multiple_choice_vqa` 与自由回答 `general_vqa`。

The v2 adapter now builds the canonical question as:

```text
<raw question stem>

Choices:
(A) <choice A>
(B) <choice B>
...
```

Ground Truth is never appended. Structured choices remain authoritative for
final-agent choice normalization. Dataset names, source tasks, metadata, and
answers remain absent from the planner request.

v2 不追加 Ground Truth；结构化 choices 继续作为最终 Agent 答案规范化的权威。
dataset 名、source task、metadata 与答案仍不进入 Planner 请求。

## Comparability / 可比性

The adapter versions the source identity used by `stable_sample_id`, while the
canonical question participates in model request hashes. v2 therefore changes
XLRS sample IDs, planner requests, and cache keys.
Fresh v1 and v2 XLRS runs must not be resumed or compared as identical input
populations. Source annotations and deterministic metric definitions are
unchanged.

adapter 将 version 纳入 `stable_sample_id` 使用的 source identity，同时 canonical
question 参与模型请求哈希，因此 v2 会改变 XLRS sample ID、Planner 请求和 cache
key。v1/v2 fresh XLRS run 不得作为相同输入总体混合 resume 或直接比较。源标注
与确定性指标定义没有变化。

Task-level `dataset_probe.json` freezes the adapter version. Resume validates
the current probe against the persisted version before overwriting artifacts
or scheduling model work; corrupt or drifting probe identity fails closed.
Earlier legacy runs without a task probe do not invent one during resume and
remain subject to the existing per-sample legacy gates. JSONL image paths
resolve against the adapter-declared external cache root, and DatasetRunner
binds that same root to both Planner and Agent.

task 级 `dataset_probe.json` 冻结 adapter version。resume 在覆盖产物或调度模型
前比较当前 probe 与持久化版本；身份损坏或漂移均 fail closed。缺少 task probe
的更早期 legacy run 不在 resume 时猜测或补写 probe，并继续服从既有逐样本
legacy 门禁。JSONL 图片路径相对 adapter 声明的外部 cache 根解析，DatasetRunner
将同一根绑定到 Planner 与 Agent。
