# Modification Note: Unified VLM Interfaces - 2026-08-05 10:41 +08:00

## Modification Time

2026-08-05 10:41:54 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Expose six explicit canonical local-model interfaces, including separate Qwen3.5-4B and Qwen3.5-9B wrappers.

## Modified Files

`models/`, `main.py`, `config/baseline.example.json`, `eval/audit_report.py`, tests, README, DETAILS, and `requirements-models.txt`.

## Core Changes

Added a typed registry, shared canonical helpers, deterministic model wrappers, optional PEFT loading for the two approved models, explicit grounding conversion, and model-neutral audit wording.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes. `load_model_from_config` now selects six explicit model types behind a shared `predict(CanonicalSample)` protocol.

## Whether the Configuration Was Changed

Yes. New configurations use an explicit `model.type`; limited compatibility remains for legacy official Qwen IDs.

## Whether Evaluation Was Affected

No metric, split, reference, or canonical JSONL rule changed. Only audit display wording and optional model fields changed.

## Whether Deployment Was Affected

No deployment path changed. Model dependencies remain optional and lazily imported.

## Whether pytest Was Updated

Yes. Registry and dedicated Qwen3.5 wrapper coverage was added and existing Qwen3-VL/report tests were updated.

## Whether .gitignore Was Updated

No new generated artifact type was introduced.

## Validation Method

Local pytest, full repository checks, and separate Spark smoke processes for each configured model.

## Risks and Follow-up TODOs

Real inference requires compatible server-side Transformers/model snapshots and sufficient memory, particularly for Qwen3.5-9B.
