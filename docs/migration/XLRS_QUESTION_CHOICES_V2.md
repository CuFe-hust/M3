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

Because canonical question text participates in stable sample IDs and model
request hashes, v2 changes XLRS sample IDs, planner requests, and cache keys.
Fresh v1 and v2 XLRS runs must not be resumed or compared as identical input
populations. Source annotations and deterministic metric definitions are
unchanged.

由于 canonical question 参与稳定 sample ID 与模型请求哈希，v2 会改变 XLRS
sample ID、Planner 请求和 cache key。v1/v2 fresh XLRS run 不得作为相同输入总体
混合 resume 或直接比较。源标注与确定性指标定义没有变化。
