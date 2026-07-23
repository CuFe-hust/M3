# Modification Note: Runnable Qwen Agent and Spark Handoff - 2026-07-22 12:51:31 +08:00

## Modification Time

2026-07-22 12:51:31 +08:00

## Modifier

Cooper3516833584

## Modification Goal

Complete the local runnable multi-Agent workflow and provide safe, environment-driven Spark deployment handoff assets without contacting a server or cloud service.

## Modified Files

`spacers_agent/` workflow, adapter, schema, counting, client-cache, settings and CLI modules; versioned runtime documentation; server scripts; `.env.example`; and offline tests.

## Core Changes

- Added CLI contracts for live health, adapter listing, Qwen smoke, one-image counting, dataset execution/resume, and persisted-result evaluation.
- Added strict versioned manifest adapters for LEVIR-CC, VRSBench, MME-RealWorld, and XLRS-Bench-lite; field mappings are explicit and probed before use.
- Added confidence-gated acceptance, explicit Qwen seam decisions, atomic JSON writes, persisted sample ground truth for deterministic re-evaluation, per-sample statuses, and non-counting visual expert primitives.
- Added Spark vLLM lifecycle, health, dataset, resume, and systemd examples using environment variables only.
- Split the count command into `spacers_agent.commands`, added stable command exit codes, explicit one-image flags, bounded explicit sample concurrency, stable sample-ID sharding, append-only prediction JSONL, and probe evidence in the run manifest.

## Whether the Canonical Sample Format Was Changed

No. The existing baseline canonical JSONL and `main.py` interfaces remain unchanged. New `UnifiedSample` workflow records are additive.

## Whether the Model Interface Was Changed

No existing model-loading interface changed. The additive OpenAI-compatible Qwen/DeepSeek clients are used by new CLI commands only.

## Whether the Configuration Was Changed

No existing key changed. The ignored `.env` template and documented server environment variables were added; explicit YAML run roots retain precedence over dotenv output-root values.

## Whether Evaluation Was Affected

Existing metrics are unchanged. Persisted count results produce `evaluation.json` and `evaluations.jsonl`, with an optional text-only DeepSeek judgment when a key is available; judge output never replaces deterministic truth.

## Whether Deployment Was Affected

Added unexecuted Spark deployment examples. No real domestic AI chip validation was performed; only static and offline checks were completed.

## Whether pytest Was Updated

Yes. Added parser, explicit-adapter probe, and minimum-confidence acceptance tests.

## Whether .gitignore Was Updated

No. Existing rules already ignore `.env`, outputs, datasets, caches, logs, and model artifacts; no new generated-file category was introduced.

## Validation Method

`C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m compileall spacers_agent` and `C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m pytest -q` completed locally: 76 passed.

## Risks and Follow-up TODOs

No live endpoint, DeepSeek API, Spark host, tunnel, server start/stop, real adapter probe, real dataset sample, or end-to-end model run was performed. Real datasets must supply audited `spacers_adapter.json` mappings before use. Server parameters must be selected only after `bootstrap.sh` reports actual hardware.
