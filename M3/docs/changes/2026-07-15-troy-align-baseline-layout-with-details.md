# Modification Note: Align Baseline Layout With DETAILS.md - 2026-07-15 22:25:43 +08

## Modification Time

2026-07-15 22:25:43 +08

## Modifier

tRoy

## Modification Goal

Align the current baseline implementation with the minimum project layout defined in `DETAILS.md`.

## Modified Files

- Added `main.py`
- Restored `models/__init__.py`
- Restored `models/qwen3vl.py`
- Deleted `main/run_baseline.py`
- Deleted `main/agent/.gitkeep`
- Deleted `main/model/.gitkeep`
- Deleted `main/pytest/.gitkeep`
- Deleted `route/.gitkeep`
- Deleted `config/.gitkeep`
- Updated `tests/test_baseline_adapters.py`
- Updated `README.md`
- Updated `DETAILS.md`
- Updated `docs/experiments/baseline-qwen3-vl-4b.md`
- Added `docs/changes/2026-07-15-troy-align-baseline-layout-with-details.md`

## Core Changes

Moved the baseline entry point to root-level `main.py` and restored the Qwen3-VL wrapper to
root-level `models/`. Removed the empty `main/` and `route/` placeholders because no agent or
router workflow is implemented. Updated imports and Colab commands while preserving the existing
baseline behavior, dataset adapters, canonical records, and evaluation logic.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. The model wrapper returns to the `models.qwen3vl` module path defined by `DETAILS.md`.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No. Metric definitions, dataset splits, reference answers, and output structures were not changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Updated the model-helper import in `tests/test_baseline_adapters.py`.

## Whether .gitignore Was Updated

No. No new generated artifact type or local configuration was introduced.

## Validation Method

- Ran `python3 -m compileall data models eval tests main.py` successfully.
- Ran all test functions manually because local `pytest` is not installed.
- Ran `python3 main.py --help` successfully.
- Ran `git diff --check` successfully.

## Risks and Follow-up TODOs

- The documented command changes from `python main/run_baseline.py` to `python main.py`.
- No model weights or datasets were loaded locally; Colab inference remains to be run.
