# Modification Note: Support VRSBench Official Evaluation Fields - 2026-07-18 15:03:01 +08:00

## Modification Time

2026-07-18 15:03:01 +08:00

## Modifier

roxanne517

## Modification Goal

Allow the existing VRSBench captioning, VQA, and grounding adapters to read the field names
used by the downloaded official validation annotations without changing canonical schemas,
reference values, task prompts, dataset splits, or evaluation metrics.

## Modified Files

- `.gitignore`
- `data/loaders.py`
- `tests/test_baseline_adapters.py`
- `DETAILS.md`
- `docs/experiments/qwen3-vl-4b-four-dataset-smoke.md`
- `docs/changes/2026-07-18-agent-support-vrsbench-eval-fields.md`

## Core Changes

Added `image_id` image lookup and `ground_truth` reference lookup for the three VRSBench
task adapters. Question identifiers now take precedence over image identifiers when selecting
a canonical sample ID. Added a regression test based on the released validation field shapes,
recorded the four-dataset smoke experiment, and ignored local pytest cache files.

## Whether the Canonical Sample Format Was Changed

No. Existing official fields are mapped into the unchanged `CanonicalSample` contract.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

Reference-answer reading compatibility was extended for the official VRSBench validation
field names. Reference values, metrics, task prompts, and dataset splits were not modified.
Historical runs that could not load these records should be rerun; successfully loaded runs
using older aliases remain comparable.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Added coverage for caption, VQA, and grounding records using `image_id`, `ground_truth`,
and `question_id`.

## Whether .gitignore Was Updated

Yes. Added `.pytest_cache/`, which was produced by the required local validation run.

## Validation Method

- Ran `python -m compileall data models eval tests` successfully.
- Ran `pytest -q`; all 6 tests passed.
- In Colab, loaded two official validation records for each VRSBench task after applying the
  same compatibility changes; all six canonical samples passed validation.

## Risks and Follow-up TODOs

- The smoke test found that Qwen3-VL may emit grounding coordinates on a 0-1000-like scale
  even when prompted for 0-100. Raw predictions must remain preserved, and any normalization
  must be explicitly reported before it is considered for default evaluation behavior.
- No full-dataset or upstream official VRSBench score was produced in this smoke test.
