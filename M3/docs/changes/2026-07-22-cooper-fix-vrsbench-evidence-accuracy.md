# Modification Note: Fix VRSBench Evidence-Router Accuracy - 2026-07-22 18:23:34 +08:00

## Modification Time

2026-07-22 18:23:34 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Correct the observed evidence-router regression without removing the Router, accepted-point counting, retained boxes, local Transformers backend, or text-only DeepSeek validation.

## Modified Files

- `spacers_agent/counting.py`, `routing.py`, `workflow.py`, `vqa_geometry.py`, `vqa_report.py`
- `spacers_agent/settings.py`, `spacers_agent/cli.py`, and command prompt loaders
- `configs/default.yaml`
- New versioned counting, empty-review, spatial, and candidate-review prompts
- VRSBench geometry, point-counting, and workflow tests
- `README.md`, `DETAILS.md`, and an offline experiment record

## Core Changes

- Replaced the VRSBench quantity target LLM parse with a fixed vehicle ontology containing small-, large-, and generic-vehicle aliases.
- Normalized harmless space/hyphen/underscore and singular/plural target spellings before strict tile-label validation.
- Added optional minimum scan depth, crop enlargement, and one empty-leaf review while preserving `final_count == accepted point count`.
- Marked an empty review without explicit `confirmed_absent` as `ZERO_COUNT_UNCONFIRMED` and partial.
- Added one bounded spatial candidate-completeness review for extreme and arrangement questions.
- Required at least two vehicle candidates before deterministic top/bottom comparison.
- Added reference-independent answer normalization for declared VRSBench yes/no, vehicle category, grid position, color, and cardinal-direction vocabularies.
- Kept partial VQA artifacts visible in the default report and retained all raw Qwen responses.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No model, processor, tokenizer, backbone, or weight-loading interface was changed. Optional workflow arguments were added to the point-counting orchestrator and thin counting expert.

## Whether the Configuration Was Changed

Yes. `counting.prompt_version` now selects `count-point-v3`; default YAML adds `vrsbench_min_scan_depth`, `vrsbench_zero_review`, and `vrsbench_tile_upscale_max_side`. Pydantic defaults remain backward-compatible and opt out unless a configuration enables them.

## Whether Evaluation Was Affected

Prediction generation is affected because answers may be normalized before the unchanged evaluator runs. Metrics, reference-answer reading, dataset split, sample order, and Judge logic were not changed. Historical and repaired runs must use separate run IDs.

## Whether Deployment Was Affected

No. The local Transformers inference path remains unchanged and vLLM is not required.

## Whether pytest Was Updated

Yes. Tests cover fixed target aliases, answer normalization, incomplete extreme evidence, candidate review, forced scan depth, crop enlargement, recovered empty tiles, and unconfirmed zero status.

## Whether .gitignore Was Updated

No. No new generated artifact class, dataset path, model format, secret file, or output directory was introduced.

## Validation Method

- `python -m compileall -q spacers_agent`
- `python -m compileall -q .`
- Focused pytest: 26 passed
- Full `pytest -q`: 107 passed
- `git diff --check`: passed

## Risks and Follow-up TODOs

- Extra tile and review calls increase local Qwen latency.
- Real Qwen behavior is not validated by offline Mock tests. A separately authorized fresh Spark run is required.
- Cardinal direction remains unmodified when audited north-up metadata is unavailable.
- The current 20 evaluation samples were not used to select prompt wording or thresholds; prompt development needs a separate non-test development set.
