# Modification Note: Single main.py Entry - 2026-08-05 13:23 +08:00

## Modification Time

2026-08-05 13:23:31 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Add one lightweight, unambiguous public entry point `main.py` at the repository root that
loads the YAML config, loads the local Qwen model once, assembles the existing multi-Agent
Runtime, starts a serial local HTTP service by default, supports one-shot CLI questions,
and reuses the existing dataset run entry. The old Colab baseline `main.py` was removed by
the audit baseline `f96544b`; this change introduces the new single entry in its place.

## Modified Files

- Added `main.py` (root single public entry).
- Added `spacers_agent/application.py` (`RuntimeApplication`, `PublicAnswer`,
  image collection, task resolution, serial HTTP service, `run_dataset_command`).
- Added `tests/entry/__init__.py`, `tests/entry/test_main_parser.py`,
  `tests/entry/test_application.py`, `tests/entry/test_manual_images.py`,
  `tests/entry/test_http_service.py`, `tests/entry/test_single_model_load.py`.
- Updated `README.md`, `DETAILS.md`, `AGENTS.md`, `docs/implementation_status.md`.
- No changes to `models/qwen_transformers.py`, `models/entry.py`,
  `spacers_agent/bootstrap.py`, `spacers_agent/agents/*`, `spacers_agent/routing/*`,
  `spacers_agent/workflows/*`, or `spacers_agent/dataset_adapters.py`.

## Core Changes

- `main.py` only parses arguments, loads settings (`spacers_agent.settings.load_settings`),
  creates `RuntimeApplication`, and dispatches `serve` / `ask` / `run-dataset`
  (no subcommand defaults to `serve`). It contains no model business logic; the top level
  catches exceptions and prints one uniform JSON error, returning exit codes 1/130.
- `RuntimeApplication.create()` calls `create_model("qwen_transformers", ...)` and
  `assemble_runtime(...)` exactly once (no DeepSeek client, no vLLM, no remote endpoint).
- `ask()` collects first-level images (natural sort, `.jpg/.jpeg/.png/.webp/.tif/.tiff/.bmp`,
  max 8, strict failures), resolves the task (explicit task via fixed `ROUTES` without a
  Router model call; `auto` maps 2-image empty-question to `change_caption`, 2-image
  question to `change_qa`, 1-image empty-question to `caption`, and everything else to one
  `route_unknown()` call), builds canonical `UnifiedSample` with `image`/`context` or strict
  `t1`/`t2` roles, runs exactly one primary Agent, maps the payload to a uniform
  `PublicAnswer`, and persists only `request.json` + `result.json` + Agent artifacts under
  `outputs/runs/service/requests/<request-id>/`.
- The serial `HTTPServer` exposes only `GET /health` and `POST /ask`
  (1 MiB body cap, 400/404/413/500 mapping, default `127.0.0.1:8000`).
- `run_dataset_command` validates the common flags and delegates the whole dataset loop to
  the existing CLI implementation (`spacers_agent.cli._run_dataset`), which reuses
  `get_adapter()` / `RunStore()` / `create_model()` / `assemble_runtime()` /
  `build_dataset_runner()`; SampleRunner fallback, Judge, Resume, Artifact, and Report
  behavior are preserved without duplicating the loop.
- Manual/service paths deliberately do not run fallback Agents, Judge, evaluation, reports,
  or a second Router call; failures propagate to the caller.
- Request IDs use `manual-YYYYMMDD-HHMMSS-<6hex>` / `http-YYYYMMDD-HHMMSS-<6hex>`.

## Whether the Canonical Sample Format Was Changed

No. The manual path constructs the existing `UnifiedSample` / `ImageRef` /
`GroundTruth` contracts unchanged.

## Whether the Model Interface Was Changed

No. Model construction still goes through `models.entry.create_model`; the runtime still
goes through `spacers_agent.bootstrap.assemble_runtime`.

## Whether the Configuration Was Changed

No new configuration keys. `--config` (default `configs/default.yaml`), service
`--host`/`--port`, and existing `models.qwen` / `runs.root` are used as-is.

## Whether Evaluation Was Affected

No. Metric calculation, dataset splits, reference-answer reading, and evaluation output
formats are untouched; the manual path performs no evaluation.

## Whether Deployment Was Affected

No deployment export change. The service binds `127.0.0.1` by default and must not be
exposed to untrusted networks; it loads the model once per process and serves requests
serially.

## Whether pytest Was Updated

Yes. Added `tests/entry/` (62 tests): parser contract, image collection, task rules,
image roles, artifact contents, HTTP status codes, no-fallback behavior, and the
model-loads-once architecture acceptance.

## Whether .gitignore Was Updated

No. The new code adds no output directories or file types that are not already ignored
(`outputs/` is already ignored).

## Validation Method

- `python -m compileall main.py spacers_agent models` — passed.
- `pytest tests/entry` — 62 passed (isolated venv, `--basetemp` local to the worktree).
- Full offline suite `pytest -q` — 462 passed; 2 pre-existing environment-specific
  failures unrelated to this change: `test_multiagent_vqa_pipeline.py` needs the `transformers`
  package that the isolated venv does not install, and `test_dataset_validator.py`
  `test_levir_cc_missing_image_side_fails` asserts a `/` path separator that Windows renders
  as `\`.
- `python main.py --help`, `serve --help`, `ask --help`, `run-dataset --help` — all pass.
- Fake-Runtime tests (no real model) cover one create + three consecutive asks with
  `load_count == 1`, HTTP flows, and artifact contents.
- Offline failure paths: default `serve`, `ask`, and `run-dataset` without a model print the
  uniform JSON error to stderr and exit 1 (expected in an environment without model weights).

## Risks and Follow-up TODOs

- No real-device validation was performed: Spark real model loading, CUDA memory stability,
  consecutive multi-request reuse on a live model, real-dataset accuracy, public-network
  exposure safety, and multi-user concurrency are not claimed as verified.
- `run_dataset_command` delegates to the internal `spacers_agent.cli._run_dataset`; if the
  legacy CLI is later removed, the delegation must be replaced by the five public
  composition calls.
- The default config model id `qwen3-vl-4b-instruct` requires a local checkpoint or an
  explicitly authorized local config (`configs/local*.yaml`); the entry never downloads.
