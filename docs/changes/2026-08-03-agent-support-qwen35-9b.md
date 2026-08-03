# Modification Note: Support Qwen3.5-9B - 2026-08-03 10:15:28 +08:00

## Modification Time

2026-08-03 10:15:28 +08:00

## Modifier

Cooper

## Modification Goal

Allow the existing local Transformers inference paths to load the official multimodal
Qwen3.5-9B checkpoint without breaking Qwen3-VL-4B behavior or the JSON-only Agent contract.

## Modified Files

- `models/qwen3vl.py`
- `spacers_agent/clients/qwen_transformers.py`
- `tests/test_qwen3vl_local_loading.py`
- `tests/test_multiagent_vqa_pipeline.py`
- `README.md`
- `DETAILS.md`

## Core Changes

- Read the checkpoint `model_type` with `AutoConfig` and select the matching native
  Qwen3-VL or Qwen3.5 conditional-generation class.
- Disable Qwen3.5 thinking during chat-template rendering because its default
  `<think>` prefix is incompatible with strict JSON response validation.
- Preserve the previous deterministic generation path, processor path, settings,
  artifact format, and Qwen3-VL rendering behavior.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes, additively. The existing interfaces and class names remain valid, while local
loading now also accepts checkpoints with `model_type: qwen3_5`.

## Whether the Configuration Was Changed

No new configuration field was added. A local configuration may set the existing
model path field to a downloaded Qwen3.5-9B directory.

## Whether Evaluation Was Affected

No metric, split, reference answer, or result post-processing rule was changed.
Predictions from different checkpoints must still be reported as separate experiments.

## Whether Deployment Was Affected

Yes. Spark can use `/home/user/models/Qwen3.5-9B` through the existing in-process
Transformers backend when its installed Transformers version exposes the Qwen3.5 class.

## Whether pytest Was Updated

Yes. Tests cover native Qwen3.5 class selection and non-thinking chat rendering.

## Whether .gitignore Was Updated

No. Model weights and local configurations are already ignored and remain outside Git.

## Validation Method

Run the focused loading and multi-Agent client tests, the repository offline check,
and an explicitly authorized Spark smoke inference against the downloaded checkpoint.

## Risks and Follow-up TODOs

- Qwen3.5 predictions are not directly comparable with historical Qwen3-VL-4B runs.
- Model quality and resource use require a fresh run ID and separate experiment record.
- The local environment must provide `Qwen3_5ForConditionalGeneration`; unsupported
  Transformers releases fail visibly before model weights are loaded.
