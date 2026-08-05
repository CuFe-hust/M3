# Modification Note: Fix Single main.py Entry for Offline CI - 2026-08-05 14:20 +08:00

## Modification Time

2026-08-05 14:20:41 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Fix the blocking issues in the single `main.py` entry so offline CI passes and the
real-deployment acceptance is prepared: implicit default serve without `--host`/`--port`,
CI dependency and entry verification, and decoupling `run-dataset` from the internal CLI's
private function.

## Modified Files

- `main.py` — root parser defaults (`command=serve`, `host=127.0.0.1`, `port=8000`);
  removed the runtime `args.command = "serve"` fallback.
- `pyproject.toml` — `dev` extra now declares `numpy>=1.26` (test collection needs it).
- `.github/workflows/offline-tests.yml` — rewritten: compile `main.py`/`spacers_agent`/
  `models`; run `tests/entry` first, then the full suite; verify `main.py --help`,
  `serve/ask/run-dataset --help`, and the internal maintenance CLI.
- Added `spacers_agent/commands/run_dataset.py` — public `RunDatasetOptions` +
  `run_dataset(settings, options)`; the single dataset-loop implementation reusing
  `get_adapter` / `RunStore` / `create_model` / `assemble_runtime` /
  `build_dataset_runner`, with SampleRunner fallback, Judge, Resume, Artifact, and Report.
- `spacers_agent/cli.py` — `DEFAULT_PROMPT_PATHS` re-exported from
  `commands.run_dataset`; `_run_dataset` and `resume-run` now delegate to the public
  `run_dataset`; removed the now-unused `get_adapter` / `assemble_runtime` /
  `build_dataset_runner` imports.
- `spacers_agent/application.py` — `run_dataset_command` builds `RunDatasetOptions` and
  calls the public `run_dataset` instead of the private `cli._run_dataset`.
- Added `tests/entry/test_main_execution.py` — four `main()`-level regression tests
  (implicit serve, config-only serve, explicit override, ask never starts HTTP).
- `tests/entry/test_main_parser.py` — `test_no_subcommand_defaults_to_serve` now asserts
  `command == "serve"`, `host == "127.0.0.1"`, `port == 8000`.
- Docs: `README.md` (help verification commands), `DETAILS.md` (root parser defaults and
  the shared public dataset command), `AGENTS.md` (single dataset-loop rule),
  `docs/implementation_status.md` (fix status, CI and Spark boundaries).

## Core Changes

- Default serve: the root parser sets `command=serve`/`host=127.0.0.1`/`port=8000`, so
  `python main.py` and `python main.py --config ...` reach `run_http_server` with real
  attributes (previously `AttributeError: host` after a successful model load).
- CI: `numpy` added to the `dev` extra because `tests/test_yolov5_obb_onnx.py` imports
  numpy during collection; the workflow now treats the new `main.py` as the primary
  acceptance target and the internal CLI as a compatibility check.
- run-dataset decoupling: `spacers_agent.cli` no longer owns the dataset loop;
  `main.py run-dataset`, `cli run-dataset`, and `cli resume-run` all call the public
  `spacers_agent.commands.run_dataset.run_dataset`. The prompt snapshot list
  (`PROMPT_PATHS`) also moved to the shared command module.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. Model construction remains `models.entry.create_model`; runtime composition remains
`spacers_agent.bootstrap.assemble_runtime`.

## Whether the Configuration Was Changed

No new configuration keys. The `dev` extra in `pyproject.toml` gained `numpy>=1.26`
(test-only dependency).

## Whether Evaluation Was Affected

No. Metrics, splits, reference-answer reading, and the dataset-run evaluation/report
behavior are unchanged (the counting evaluation path was moved verbatim into
`commands/run_dataset.py`).

## Whether Deployment Was Affected

No. Service defaults (`127.0.0.1:8000`) are unchanged; the fix only guarantees the
implicit serve path carries them.

## Whether pytest Was Updated

Yes. Added `tests/entry/test_main_execution.py` (4 tests) and strengthened
`test_no_subcommand_defaults_to_serve` in `tests/entry/test_main_parser.py`.

## Whether .gitignore Was Updated

No new file types or output directories were introduced.

## Validation Method

- `python -m compileall main.py spacers_agent models` — passed.
- `python -m pytest -q tests/entry` — passed.
- `python -m pytest -q` (m3 conda env) — 478 passed; the only failure is the pre-existing
  Windows path-separator assertion in `tests/test_dataset_validator.py`.
- `python main.py --help`, `serve --help`, `ask --help`, `run-dataset --help` — all pass.
- `python -m spacers_agent.cli --help` — passes.

## Risks and Follow-up TODOs

- GitHub Actions must be confirmed green on the remote `try_yolo` push; local results do
  not substitute for the remote run.
- Spark real-model verification (default start without AttributeError, `/health`,
  three consecutive `/ask` requests, PID/model-object/memory stability) remains to be
  executed on the server with a local checkpoint; it is not claimed here.
- The internal CLI still exists for maintenance commands; its eventual removal must keep
  `main.py` and `commands/run_dataset.py` intact.
