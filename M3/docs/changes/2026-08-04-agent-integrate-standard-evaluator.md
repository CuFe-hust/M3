# Modification Note: Integrate Team Standard Evaluator - 2026-08-04 15:52:18 +08:00

## Modification Time

2026-08-04 15:52:18 +08:00

## Modifier

Cooper

## Modification Goal

Connect the separately maintained `eval_standard` tool to canonical try_yolo outputs and surface its primary result in the existing HTML audit report.

## Modified Files

- `eval/standard_adapter.py`
- `eval/audit_report.py`
- `spacers_agent/cli.py`
- `tests/test_standard_eval_integration.py`
- `README.md`
- `DETAILS.md`

## Core Changes

- Added `standard-evaluate` for invoking `eval_standard/evaluate.py` on an existing canonical JSONL.
- Persisted the external report as `<result-stem>.standard.json` by default.
- Made the HTML audit report automatically display the external `primary_metric`, `primary_value`, and `score`.
- Kept standard rules, LLM configuration, prompts, synonyms, and CHAIR2 outside this repository.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No. Tool directory, output, and Python executable are command-line arguments.

## Whether Evaluation Was Affected

Yes. A new opt-in external standard-evaluation path was added. Existing metrics and their output formats were not changed, and the standard result is displayed separately.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Tests cover external invocation, adjacent report persistence, and HTML discovery.

## Whether .gitignore Was Updated

No. Standard reports are generated under already ignored experiment output directories in normal use, and no new cache, dataset, or weight type was introduced.

## Validation Method

- `C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m pytest -q tests/test_standard_eval_integration.py tests/test_baseline_audit_report.py`
- `python -m compileall -q eval spacers_agent tests/test_standard_eval_integration.py`
- `git diff --check`
- Full `pytest -q` was attempted but collection stopped in the pre-existing YOLO ONNX test because the local `m3` environment cannot import NumPy's `_multiarray_umath` extension.

## Risks and Follow-up TODOs

- The real server copy of `~/eval_standard` was not available in this workspace, so its live dependencies and API-backed Tier 3 path still require server validation.
- The adapter intentionally fails when the external process fails or does not create JSON; it does not silently substitute legacy metrics.
