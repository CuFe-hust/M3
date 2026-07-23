# Modification Note: Harden Qwen Structured Output - 2026-07-22 18:58:41 +08:00

## Modification Time

2026-07-22 18:58:41 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Prevent otherwise usable VRSBench predictions from failing when local Qwen emits a reversed or zero-area box, emits both a box and point for one observation, or truncates the final JSON member.

## Modified Files

- `spacers_agent/schemas.py`
- `spacers_agent/clients/qwen_transformers.py`
- `tests/test_vrsbench_vqa_geometry.py`
- `tests/test_multiagent_vqa_pipeline.py`
- `DETAILS.md`
- `README.md`
- `docs/changes/2026-07-22-cooper-harden-qwen-structured-output.md`

## Core Changes

- Order reversed box corners and minimally expand zero-width or zero-height axes within normalized `0..999` coordinates.
- Retain a valid box and remove a conflicting point from the same evidence item; retain a valid point when the accompanying box is malformed.
- Conservatively recover end-truncated JSON by closing open containers or removing only the final incomplete member.
- Persist every local normalization in geometry and validation metadata without changing the original raw response.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. The same Transformers checkpoint, processor, generation call, and `ExpertResult` interface remain in use.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

Metric definitions, dataset splits, and reference-answer reading were not changed. More model responses can reach the existing evaluator instead of becoming failed samples, so new run coverage can differ from earlier runs.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Geometry conflict/degeneracy and truncated-tail recovery tests were added.

## Whether .gitignore Was Updated

No. No new generated artifact type or local path was introduced.

## Validation Method

- `pytest -q tests/test_multiagent_vqa_pipeline.py tests/test_vrsbench_vqa_geometry.py`
- Full repository tests and a fresh 20-sample Spark run are required before final handoff.

## Risks and Follow-up TODOs

- A locally recovered truncated response may omit only its final incomplete evidence member; the recovery marker must be considered when auditing completeness.
- Malformed JSON before the end of output still fails or uses the existing single text-only model repair.
