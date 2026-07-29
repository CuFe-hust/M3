# Modification Note: Ground Singular VQA Positions - 2026-07-22 23:39:14 +08:00

## Modification Time

2026-07-22 23:39:14 +08:00

## Modifier

Cooper

## Modification Goal

Prevent singular VRSBench grid-position questions from returning a quadrant or image-corner region in place of the actual vehicle extent, while preserving model-provided boxes and deterministic three-by-three geometry.

## Modified Files

- `prompts/spatial_v5.md`
- `prompts/spatial_candidate_review_v3.md`
- `spacers_agent/cli.py`
- `spacers_agent/commands/common.py`
- `spacers_agent/workflow.py`
- `spacers_agent/vqa_geometry.py`
- `tests/test_multiagent_vqa_pipeline.py`
- `DETAILS.md`
- `README.md`
- `docs/experiments/2026-07-22-vrsbench-position-grounding-v5.md`

## Core Changes

- Grid-position localization no longer receives the nine grid labels before it has localized an object.
- The active spatial prompt requires tight physical object boxes and prohibits quadrant, grid-cell, and corner-region boxes.
- A valid single target box skips candidate review; missing, ambiguous, or corner-anchored evidence receives one independent review.
- A suspicious first-pass corner-region target box is replaced rather than merged when review is required.
- When a review returns valid top-level boxes but omits `evidence_items`, the explicit small/large vehicle class in the question is attached to those unchanged coordinates.
- Geometry audit fields record replaced evidence and boxes labeled from the explicit question target.

## Whether the Canonical Sample Format Was Changed

No. The existing `UnifiedSample` contract is unchanged.

## Whether the Model Interface Was Changed

No public model client or weight-loading interface changed. The active spatial prompt versions changed from `spatial-v4`/`spatial-candidate-review-v2` to `spatial-v5`/`spatial-candidate-review-v3`.

## Whether the Configuration Was Changed

No configuration key or default value changed.

## Whether Evaluation Was Affected

Reference-answer reading, dataset split, exact-match calculation, and DeepSeek judging were not changed. Candidate answers may change because grid positions are now derived from better-grounded object boxes, so affected historical inference runs must be rerun for comparison.

## Whether Deployment Was Affected

No. The Transformers loading path and deployment exports are unchanged.

## Whether pytest Was Updated

Yes. Tests cover vocabulary de-anchoring, direct use of a valid edge-touching object box, replacement of a corner-region placeholder, and recovery of model-provided review boxes lacking labeled evidence items.

## Whether .gitignore Was Updated

No. No new generated artifact type or local secret path was introduced.

## Validation Method

- `python -m compileall -q spacers_agent tests`
- `pytest -q` (`121 passed` before the final documentation-only update)
- Spark targeted live run with local Qwen3-VL and sequential calls: IDs 5 and 16 both completed.
- ID 5: retained box `[0,450,100,600]`, deterministic answer `middle-left`, exact match true, DeepSeek score 1.
- ID 16: retained box `[840,270,920,310]`, deterministic answer `top-right`, exact match true, DeepSeek score 1.
- Spark 20-sample regression: 17/20 exact match and 17/20 DeepSeek score 1; 14 completed, 6 partial, 0 failed.
- The previous report's error set `{2,5,12,16,19}` became `{2,12,19}`. The initially observed ID 7 non-grid regression disappeared after scoping v5/v3 to `grid_position`.

## Risks and Follow-up TODOs

- Corner-anchored physical objects may trigger one unnecessary independent review; their original coordinates are not silently accepted as trusted target evidence.
- The 4B model may still omit structured evidence on genuinely ambiguous images; such samples remain visible as partial rather than receiving invented coordinates.
- Remaining errors at IDs 2, 12, and 19 are outside the two requested grid-position regressions and were not changed opportunistically.
