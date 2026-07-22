# Modification Note: Add Phase 1 Local Foundation - 2026-07-22 02:11:55 +08:00

## Modification Time

2026-07-22 02:11:55 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`), the configured local Git identity. No remote push was performed.

## Modification Goal

Add the offline, reproducible foundation required before model clients, geometry, point counting, or dataset adapters can be implemented.

## Modified Files

- `pyproject.toml`
- `.env.example`
- `configs/default.yaml`
- `prompts/count_tile_v1.md`
- `spacers_agent/__init__.py`
- `spacers_agent/errors.py`
- `spacers_agent/settings.py`
- `spacers_agent/events.py`
- `spacers_agent/run_store.py`
- `spacers_agent/cli.py`
- `tests/test_phase1_foundation.py`
- `.gitignore`
- `requirements.txt`
- `DETAILS.md`
- `README.md`
- `docs/implementation_status.md`

## Core Changes

- Added a project package and an offline CLI with `run-init` and metadata-only `health` commands.
- Added validated phase-one settings, non-secret dotenv/process-environment endpoint overrides, and defaults for future point counting.
- Added stable error codes, safe JSONL event persistence, run manifests, configuration snapshots, Prompt snapshots, prompt/config hashes, and Git-state capture.
- Added a real versioned tile-counting Prompt asset while deferring all model calls to later phases.
- Added tests for settings, dotenv handling, sequential concurrency, run artifacts, safe events, and CLI help/health behavior.

## Whether the Canonical Sample Format Was Changed

No. Existing `CanonicalSample` and `CanonicalPrediction` remain unchanged.

## Whether the Model Interface Was Changed

No. The existing local Transformers baseline remains unchanged. The new `health` command intentionally performs no endpoint request.

## Whether the Configuration Was Changed

Yes. Added additive YAML settings in `configs/default.yaml`, local `.env` documentation, and project dependency metadata. Existing JSON baseline configuration remains supported and unchanged.

## Whether Evaluation Was Affected

No. No metrics, dataset splits, references, or evaluation code changed.

## Whether Deployment Was Affected

No live deployment action occurred. The only deployment-adjacent setting is metadata for a future local Qwen endpoint.

## Whether pytest Was Updated

Yes. Added `tests/test_phase1_foundation.py` and configured pytest's repository-root import path in `pyproject.toml`.

## Whether .gitignore Was Updated

Yes. Added `.env` and `configs/local*.yaml` to prevent local secrets and machine-specific configuration from being committed.

## Validation Method

- `python -m compileall -q spacers_agent tests`
- `python -m pytest -q`
- `python -m spacers_agent.cli --help`
- `python -m spacers_agent.cli health qwen`
- `git diff --check`

All commands were run locally without model, dataset, server, SSH tunnel, or cloud API access.

## Risks and Follow-up TODOs

- `pydantic-settings` is declared but not installed locally. Phase 1 uses a narrow local dotenv compatibility reader; replace it with the declared dependency after an authorized local dependency installation.
- No live endpoint, model, DeepSeek, or real dataset validation was performed.
- Phase 2 must add async clients and Mock integration before any live smoke test is considered.
