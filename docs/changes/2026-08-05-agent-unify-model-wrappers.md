# Modification Note: Unify Model Wrappers and Remove QwenVLLMClient - Modification Time

## Modification Time

2026-08-05 11:46:38 CST

## Modifier

tRoy

## Modification Goal

Unify all main-flow model wrappers under `models/` with one folder per model
(wrapper code plus a `weights/` subdirectory for remote-hosted checkpoints), add
`models/entry.py` as the only construction entry (`create_model`), and delete the
remote vLLM path (`QwenVLLMClient`, vLLM server scripts, and vLLM-only settings).
Test/training clients `DeepSeekJudgeClient` and `MockVisionClient` remain in
`spacers_agent/clients/` and are intentionally not registered in the main-flow entry.

## Modified Files

- `models/base.py` (moved from `spacers_agent/clients/base.py`)
- `models/qwen_transformers.py` (moved from `spacers_agent/clients/qwen_transformers.py`)
- `models/qwen3_vl/baseline.py` (moved from `models/qwen3vl.py`), `models/qwen3_vl/__init__.py`,
  `models/qwen3_vl/weights/README.md`
- `models/qwen3_5/__init__.py`, `models/qwen3_5/model.py`, `models/qwen3_5/weights/README.md`
- `models/qwen3vl.py` (backward-compatible re-export alias)
- `models/entry.py` (new unified entry: `register`/`create_model`/`list_models`)
- `models/__init__.py` (exports the entry and base types)
- Deleted `spacers_agent/clients/qwen_vllm.py`; removed `spacers_agent/clients/base.py`
  and `spacers_agent/clients/qwen_transformers.py` after migration
- `spacers_agent/clients/__init__.py`, `deepseek.py`, `mock.py` (base imports now
  from `models.base`; only DeepSeek/Mock remain)
- `spacers_agent/cli.py` (client factory via `create_model`; `health qwen --live`
  now reports local metadata only), `spacers_agent/commands/common.py`,
  `spacers_agent/bootstrap.py`, `main.py`, all `agents/**`/`routing/**`/`workflows/**`
  imports of the shared base
- `spacers_agent/settings.py` (`QwenSettings` drops vLLM-only fields; `QWEN_BACKEND`
  and `QWEN_BASE_URL` overrides removed), `configs/default.yaml`, `.env.example`
- `spacers_agent/counting_report.py`, `spacers_agent/vqa_report.py`,
  `spacers_agent/workflows/sample_runner.py`, `spacers_agent/agents/counting/point_pipeline.py`
  (no `qwen.backend`/`qwen.temperature` reads)
- Deleted `scripts/server/start_qwen_vllm.sh`, `stop_qwen_vllm.sh`, `healthcheck.sh`,
  `systemd/qwen-vllm.service.example`; updated `env.example`, `run_dataset.sh`,
  `systemd/spacers-dataset.service.example`
- Tests: `tests/test_model_entry.py` (new), `tests/test_phase2_clients.py`,
  `tests/test_multiagent_vqa_pipeline.py`, `tests/test_phase1_foundation.py`,
  `tests/test_qwen3vl_local_loading.py`, `tests/architecture/test_legacy_imports.py`,
  and all tests importing the moved base/Transformers modules
- `tests/fixtures/legacy/default_config_without_agents.yaml`
- `DETAILS.md`, `README.md`, `docs/runbook.md`, `docs/implementation_status.md`, `.gitignore`

## Core Changes

- Main-flow models are constructed only through `models.entry.create_model`; the
  registered entries are `qwen_transformers`, `qwen3_vl_baseline`, and
  `qwen3_5_transformers`. Adding a model requires one `@register` builder function.
- Each model folder keeps its wrapper and a `weights/` subdirectory; local development
  does not download the large checkpoints (they exist on the remote server). Existing
  untracked weights (`models/InternVL3_5-8B/`, `models/*.tgz`, `models/*/weights/`)
  are ignored, not deleted.
- The remote vLLM client and all vLLM-specific scripts/config are removed; Qwen always
  runs through the local Transformers backend. `health qwen --live` no longer probes a
  remote endpoint.
- `models/qwen3vl.py` remains as a compatibility alias so historical imports and the
  report metadata string `models.qwen3vl.Qwen3VLBaseline` are unchanged.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes (breaking): `QwenVLLMClient` and the `spacers_agent.clients.base` /
`spacers_agent.clients.qwen_transformers` module paths are removed; the classes
`QwenTransformersClient`, `VisionLanguageClient`, `RequestMeta`, and the baseline wrapper
keep their names and behavior but now live under `models.*`. The new unified entry
`models.entry.create_model` is the supported construction path.

## Whether the Configuration Was Changed

Yes (breaking): `QwenSettings` removes `backend`, `base_url`, `api_key_env`,
`timeout_seconds`, `max_retries`, and `temperature`; `QWEN_BACKEND`/`QWEN_BASE_URL`
environment overrides are removed. `configs/default.yaml`, `.env.example`, and the
legacy fixture were updated accordingly.

## Whether Evaluation Was Affected

No metric, split, or reference-answer logic changed. Counting/VQA report metadata now
records `backend: transformers` instead of reading `qwen.backend`.

## Whether Deployment Was Affected

Yes: vLLM server deployment files were removed; Spark/remote runs now use the local
Transformers backend with `QWEN_MODEL` pointing at the checkpoint directory. No real
domestic-chip or device-side validation was performed.

## Whether pytest Was Updated

Yes: added `tests/test_model_entry.py`; updated imports in affected tests; removed the
four `QwenVLLMClient` cases from `tests/test_phase2_clients.py`; adjusted health-output
and baseline-loading tests.

## Whether .gitignore Was Updated

Yes: added `models/*/weights/`, `models/InternVL3_5-8B/`, and `models/*.tgz`.

## Validation Method

- `python3 -m compileall -q models spacers_agent main.py tests/test_model_entry.py`
- Full offline suite in the `M3` Conda environment: `pytest -q` → 431 passed.
- Import smoke: `import models; models.entry.list_models()` returns the three entries.

## Risks and Follow-up TODOs

- Existing user configs containing removed `QwenSettings` fields will fail validation
  (`extra="forbid"`); update them to the new schema.
- Weights directories are placeholders locally; inference machines must place checkpoints
  under each model folder's `weights/` and point `QWEN_MODEL` there.
- Real Qwen local inference and DeepSeek smoke tests still require explicit authorization
  and hardware/weights; they were not run.
