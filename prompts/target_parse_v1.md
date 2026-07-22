<!-- name: target_parse; version: v1; schema: CountTargetSpec -->

Convert a counting question into a stable target specification. Do not inspect an image and do not count. Preserve only conditions stated by the user. Return JSON only.

将计数问题转换为稳定的目标规格。不要看图，也不要计数。只保留用户明确提出的条件。只返回 JSON。

```json
{"canonical_label":"string","aliases":[],"required_attributes":[],"excluded_attributes":[],"spatial_constraints":[],"inclusion_rule":"string","exclusion_rule":"string","ambiguity":[]}
```
