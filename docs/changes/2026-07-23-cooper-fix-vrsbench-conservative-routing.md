# Modification Note: Fix VRSBench Conservative Routing - 2026-07-23 01:02:46 +08:00

## Modification Time

2026-07-23 01:02:46 +08:00

## Modifier

Cooper <crj31415926@gmail.com>

## Modification Goal

Remove coarse-type and small-sample assumptions from VRSBench VQA routing. Make general VQA the safe fallback, constrain answers only when the question entails a closed vocabulary, and keep workflow status separate from semantic answers.

## Modified Files

- `spacers_agent/routing.py`
- `spacers_agent/vqa_geometry.py`
- `spacers_agent/workflow.py`
- `tests/test_phase5_routing.py`
- `tests/test_vrsbench_vqa_geometry.py`
- `tests/test_multiagent_vqa_pipeline.py`
- `DETAILS.md`
- `README.md`
- `docs/experiments/2026-07-23-vrsbench-conservative-routing-offline.md`

## Core Changes

- Route from explicit question semantics instead of treating the official VRSBench `type` as a dispatch contract.
- Fall back to `GeneralVQAExpert` for open categories, scenes, existence/relation questions, and unknown official types.
- Use closed answer vocabularies only when the question text itself establishes the answer space.
- Preserve valid candidate-review boxes for generic singular position targets with a neutral evidence label.
- Promote deterministic geometry results to completed status when their required evidence is complete.
- Remove `partial`, `failed`, and `completed` placeholders from the semantic answer field.
- Avoid downgrading open/global answers solely because they have no localizable box.

## Whether the Canonical Sample Format Was Changed

No. `UnifiedSample` and persisted source sample fields are unchanged.

## Whether the Model Interface Was Changed

No model, processor, tokenizer, weight-loading, or `ExpertResult` schema interface was changed. `TaskRouter.route_vrsbench_vqa` gained an optional, backward-compatible `question` keyword used for conservative routing.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

Yes. Qwen answers and Agent routes can change, while metric definitions, reference-answer loading, dataset splits, and DeepSeek judging remain unchanged. Results produced before and after this change are not directly comparable without rerunning the same samples.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Router, semantic subtype, vocabulary, status separation, open-answer, GeneralVQA integration, and generic review-box recovery cases were added or updated.

## Whether .gitignore Was Updated

No. The change introduces no new local output, cache, dataset, checkpoint, or secret-file type.

## Validation Method

- `python -m compileall -q spacers_agent tests` — passed.
- `python -m pytest -q tests/test_vrsbench_vqa_geometry.py tests/test_phase5_routing.py tests/test_multiagent_vqa_pipeline.py` — 46 passed.
- `python -m pytest -q` — 131 passed.
- Read-only route audit over 173 persisted questions from the prior 200-sample run — 109 general VQA, 47 explicit counting, and 17 high-confidence spatial routes; reference answers were not read by the audit script.
- `git diff --check` — passed.

## Risks and Follow-up TODOs

- No live Qwen, DeepSeek, Spark, GPU, or server validation was performed in this modification.
- A fresh, disjoint evaluation run is required to measure model-level accuracy and latency after the routing change.
- The existing metric writer still reports canonical records that reached evaluation; dataset run summaries must remain visible alongside metrics when operational failures occur.
