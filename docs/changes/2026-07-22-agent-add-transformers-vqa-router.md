# Modification Note: Add Transformers VQA Router - 2026-07-22 15:31:36 +08:00

## Modification Time

2026-07-22 15:31:36 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Run official VRSBench VQA samples through the real `TaskRouter -> GeneralVQAExpert` path with the existing local Qwen3-VL Transformers checkpoint, use DeepSeek only for answer validation, and generate the default per-sample HTML audit report.

## Modified Files

- `spacers_agent/clients/qwen_transformers.py`, `spacers_agent/clients/__init__.py`
- `spacers_agent/dataset_adapters.py`, `spacers_agent/workflow.py`, `spacers_agent/vqa_report.py`
- `spacers_agent/evaluation.py`, `spacers_agent/cli.py`, `spacers_agent/settings.py`
- `prompts/deepseek_vqa_judge_v1.md`, `configs/default.yaml`
- `tests/test_multiagent_vqa_pipeline.py`
- `README.md`, `DETAILS.md`

## Core Changes

- Added an in-process Transformers client that loads the declared local Qwen3-VL checkpoint once, performs deterministic sequential generation, validates structured output, and saves sanitized inputs, raw output, parsed output, duration, and token counts.
- Added a strict read-only adapter for official `VRSBench_EVAL_vqa.json` records.
- Persisted the deterministic TaskRouter decision and explicit Agent trace for every non-counting sample.
- Added versioned text-only DeepSeek VQA validation and separated Judge failure from Qwen inference status.
- Reused the existing audit report component to create canonical JSONL, metrics, DeepSeek audit JSONL, CSV, copied images, and HTML from multi-Agent artifacts.

## Whether the Canonical Sample Format Was Changed

No. The existing canonical baseline sample contract and `UnifiedSample` contract were not changed.

## Whether the Model Interface Was Changed

Additive only. `QwenTransformersClient` implements the existing `VisionLanguageClient.complete_json` protocol; the vLLM client remains available for compatible configurations.

## Whether the Configuration Was Changed

Yes. Qwen settings gained `backend`, `dtype`, `device_map`, `local_files_only`, `min_pixels`, and `max_pixels`. Existing defaults preserve the vLLM path; Spark can explicitly select `backend: transformers`.

## Whether Evaluation Was Affected

The existing metrics, dataset split, and reference-answer reading logic were not modified. Additive VQA records preserve strict exact match and an explicitly non-official, text-only DeepSeek binary validation score.

## Whether Deployment Was Affected

Yes. A local Transformers inference path was added for machines that already hold the checkpoint; no model export or device-specific deployment artifact changed.

## Whether pytest Was Updated

Yes. Tests cover the local Transformers client, official VRSBench mapping, Router/Expert/Judge artifacts, successful status mapping, and HTML Agent trace.

## Whether .gitignore Was Updated

No. Run outputs are already covered by `outputs/`, and local Spark YAML files are already covered by `configs/local*.yaml`.

## Validation Method

- `python -m pytest -q tests/test_multiagent_vqa_pipeline.py tests/test_phase1_foundation.py tests/test_phase2_clients.py tests/test_stage_a_to_g_contracts.py`
- `git diff --check`
- Full pytest and the authorized Spark 20-sample live run are completed after this note is created and reported in the handoff.

## Risks and Follow-up TODOs

- Local generation depends on the installed Transformers version supporting the downloaded Qwen3-VL checkpoint.
- DeepSeek validation is a text-only proxy and is not the official VRSBench GPT metric.
- The official VRSBench adapter currently exposes the audited validation VQA file only; other VRSBench tasks retain their baseline adapters until an exact multi-Agent layout is separately authorized.
