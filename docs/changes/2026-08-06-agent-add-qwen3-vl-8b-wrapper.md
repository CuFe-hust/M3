# Modification Note: Add Qwen3-VL-8B-Instruct Wrapper - 2026-08-06 17:57:43 CST

## Modification Time

2026-08-06 17:57:43 CST

## Modifier

tRoy (`791056216@qq.com`)

## Modification Goal

Add a standalone Qwen3-VL-8B-Instruct model wrapper (not reusing the 4B
wrapper) under `models/`, and download the official 8B Instruct checkpoint
directly on the remote server `100.88.222.9` into `~/M3/models` through the
local network proxy. The wrapper is constructed through the unified model
entry; the repository copy never downloads weight files.

## Modified Files

- `models/qwen3_vl_8b/__init__.py` (new)
- `models/qwen3_vl_8b/model.py` (new)
- `models/qwen3_vl_8b/weights/README.md` (new, local-only placeholder)
- `models/entry.py`
- `tests/test_model_entry.py`
- `tests/test_qwen3vl_local_loading.py`
- `README.md`
- `DETAILS.md`
- Remote `~/M3/models/qwen3_vl_8b/`, `~/M3/models/entry.py`,
  `~/M3/tests/test_model_entry.py`, `~/M3/tests/test_qwen3vl_local_loading.py`
  (synced from this branch, not committed on the remote)

## Core Changes

- Added `Qwen3VL8BSettings`, a self-contained dataclass that defaults
  `model_id` to `Qwen/Qwen3-VL-8B-Instruct`; it does not inherit the 4B
  `Qwen3VLSettings`.
- Added `Qwen3VL8BInstruct`, a standalone wrapper that does not reuse the 4B
  `Qwen3VLBaseline` implementation. Loading, canonical-sample validation,
  prediction output, grounding coordinate conversion, and metadata behavior
  are implemented inside `models/qwen3_vl_8b/model.py`.
- Registered the new builder as `qwen3_vl_8b_baseline` in `models/entry.py`;
  all existing entries and default parameters remain unchanged.
- The wrapper itself never downloads weights. The checkpoint must already be
  cached locally or be supplied through `settings.model_id` together with
  `local_files_only`.
- On the remote, the full `Qwen/Qwen3-VL-8B-Instruct` directory was
  downloaded through a per-command HTTP proxy (`127.0.0.1:7897` via SSH
  reverse tunnel) into `~/M3/models/Qwen3-VL-8B-Instruct`; the wrapper files
  were synced directly from the local branch and verified with the remote
  `py311` conda environment.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes, additively. A new entry name (`qwen3_vl_8b_baseline`) and new wrapper
classes were added; all existing interfaces, class names, and loading logic
remain valid.

## Whether the Configuration Was Changed

No new configuration field was added. The wrapper accepts
`Qwen3VL8BSettings` directly; the existing `models.qwen.model` /
`local_files_only` fields continue to serve the local Transformers backend.

## Whether Evaluation Was Affected

No metric, split, reference answer, or result post-processing rule was changed.
8B predictions are a different checkpoint and must be reported as a separate
experiment when used.

## Whether Deployment Was Affected

No deployment export or hardware path was changed. The wrapper only provides
an additional model entry and requires a locally cached checkpoint before real
inference.

## Whether pytest Was Updated

Yes. `tests/test_model_entry.py` covers the new entry default and explicit
settings; `tests/test_qwen3vl_local_loading.py` covers the 8B settings default,
wrapper independence from the 4B baseline, and local loading with the native
`Qwen3VLForConditionalGeneration` class.

## Whether .gitignore Was Updated

No. The new `models/qwen3_vl_8b/weights/` directory is already covered by the
existing `models/*/weights/` ignore rule, and no new generated file type was
introduced.

## Validation Method

Local: `/opt/miniconda3/envs/M3/bin/python -m pytest -q
tests/test_model_entry.py tests/test_qwen3vl_local_loading.py` passed
(15 tests); `python -m compileall` passed for the changed Python files.
Remote: `/home/lijia/.conda/envs/py311/bin/python -m pytest -q
tests/test_model_entry.py tests/test_qwen3vl_local_loading.py` passed
(15 tests) after syncing; `list_models()` includes
`qwen3_vl_8b_baseline`.
Remote model directory: all 16 files were compared byte-size against the
official Hugging Face repo (`RESULT= ALL_OK`, total ~17GB); a local-only
smoke check with Transformers 5.14.1 loaded `AutoConfig` (`model_type=
qwen3_vl`, architecture `Qwen3VLForConditionalGeneration`) and
`AutoProcessor` (`Qwen3VLProcessor`) from the downloaded directory.

## Risks and Follow-up TODOs

- Real 8B weight loading and inference have not been run yet; only config,
  processor, file-size, and wrapper tests were validated. Loading the full
  model requires enough GPU/CPU memory on the remote server.
- On the remote `py311` env, installing the project requirements upgraded
  `transformers` to 5.14.1, which conflicts with the pre-existing
  `llamafactory 0.9.5` constraint (`transformers<=5.6.0`); LoRA workflows in
  that env may need a separate environment.
- The remote `try_yolo` working tree now contains uncommitted synced changes
  (`models/qwen3_vl_8b/`, `models/entry.py`, two test files) and the
  untracked weight directory; decide later whether to commit them on the
  remote.
- When running inference, use a fresh run ID and keep 8B results separate from
  historical Qwen3-VL-4B results.
