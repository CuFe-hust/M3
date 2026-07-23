# Modification Note: Enable Local Qwen Loading - 2026-07-22 13:30:03 +08:00

## Modification Time

2026-07-22 13:30:03 +08:00

## Modifier

Cooper3516833584

## Modification Goal

Allow the existing Qwen3-VL baseline to use an already-downloaded external checkpoint without falling back to a model-hub network request.

## Modified Files

`models/qwen3vl.py`, `main.py`, `data/loaders.py`, `config/baseline.example.json`, `tests/test_qwen3vl_local_loading.py`, `tests/test_baseline_adapters.py`, `README.md`, and `DETAILS.md`.

## Core Changes

- Added an opt-in `model.local_files_only` configuration field with a backward-compatible `false` default.
- Forwarded the setting to both model and processor loading and used the supported `dtype` loading argument.
- Added focused tests for model/processor loading arguments and baseline configuration parsing.
- Accepted the official VRSBench evaluation fields `image_id`, `ground_truth`, and `question_id`
  after the Spark data-layout audit, preserving unique question-level sample IDs.
- Documented that real server paths belong only in ignored local configuration files.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes, additively. `Qwen3VLSettings` accepts the optional `local_files_only` field; existing callers retain the previous default behavior.

## Whether the Configuration Was Changed

Yes. `model.local_files_only` is an optional Boolean with a default of `false`.

## Whether Evaluation Was Affected

The VRSBench VQA adapter now reads the official evaluation release's `ground_truth` field as
the reference answer. Dataset splits, reference values, prediction records, and evaluation metrics are unchanged.

## Whether Deployment Was Affected

Yes. A server may now explicitly prohibit Hugging Face network fallback when using an external local checkpoint path. No model weights are changed or copied.

## Whether pytest Was Updated

Yes. `tests/test_qwen3vl_local_loading.py` covers the additive loading and configuration behavior,
and `tests/test_baseline_adapters.py` covers the audited official VRSBench VQA field layout.

## Whether .gitignore Was Updated

No. Existing rules already ignore `config/local*.json`, model artifacts, datasets, outputs, logs, caches, and `.env`.

## Validation Method

Run the focused pytest file, the complete offline pytest suite, syntax compilation, and one explicitly authorized Spark smoke inference against the existing local Qwen3-VL-4B-Instruct checkpoint.

## Risks and Follow-up TODOs

The local checkpoint and GPU runtime remain external deployment dependencies. Live benchmark results must report the selected sample limit and keep deterministic metrics separate from optional DeepSeek proxy judgments.
