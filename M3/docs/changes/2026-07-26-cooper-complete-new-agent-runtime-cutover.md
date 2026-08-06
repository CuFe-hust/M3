# Modification Note: Complete New Agent Runtime Cutover - 2026-07-26 16:30:00 +08:00

## Modification Time

2026-07-26 16:30:00 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Remove the retired runtime business paths and make every supported CLI inference entry use the composed Agent Runtime.

## Modified Files

- `spacers_agent/workflow.py`
- `spacers_agent/workflows/artifact_writer.py`
- `spacers_agent/workflows/dataset_runner.py`
- `spacers_agent/commands/count_image.py`
- `spacers_agent/agents/counting/agent.py`
- `spacers_agent/workflows/counting_workflow.py` (removed)
- Parity, architecture, and VQA pipeline tests
- Runtime, status, runbook, README, change, and experiment documentation

## Core Changes

- Replaced the former `workflow.py` DatasetRunner and counting business implementation with a thin import-compatible facade.
- Removed the duplicate `CountingWorkflow`; `count-image` now builds a canonical `UnifiedSample`, calls `assemble_runtime()`, and executes `CountingAgent` through `SampleRunner`.
- Kept `--target-spec`, seam-disable, model-call-budget, evaluation, resume, and overlay behaviors on the new path.
- Moved atomic JSON publication to `ArtifactWriter` ownership so production workflows do not import the retired module.
- Made VQA Judge resume persist its retry result and a visible failure artifact rather than silently leaving stale Judge state.
- Replaced legacy-runtime parity harness imports with frozen fixture checks against the composed runtime and added architecture checks for the direct counting CLI path.

## Whether the Canonical Sample Format Was Changed

No. `count-image` now explicitly constructs the existing `UnifiedSample` schema.

## Whether the Model Interface Was Changed

No. Qwen and VRSBench backend request contracts are retained and covered by frozen fixtures.

## Whether the Configuration Was Changed

No configuration key or default changed. The existing `--no-seam-verify` and call-budget arguments are applied to the injected Runtime configuration.

## Whether Evaluation Was Affected

No metric or reference-answer logic changed. Resume Judge retries now write their actual success or failure record to `vqa_evaluation.json`.

## Whether Deployment Was Affected

No. YOLO remains unregistered and `backend.yolo.enabled: true` is rejected before runtime construction.

## Whether pytest Was Updated

Yes. Runtime-entry and compatibility-boundary checks were added, legacy parity tests were converted to frozen new-runtime fixture checks, and VQA resume coverage now verifies persisted Judge retry output.

## Whether .gitignore Was Updated

No. No new generated artifact type, dataset, model weight, cache, or local configuration file was introduced.

## Validation Method

- `git diff --check`
- `python -m compileall spacers_agent`
- `pytest -q` — 364 passed
- Architecture, agent, routing, workflow, parity, compatibility, and CLI subsets — all passed
- CLI help, dataset listing, and offline Qwen/DeepSeek health metadata commands — passed

## Risks and Follow-up TODOs

- No real Qwen/DeepSeek service or local production dataset was supplied or authorized, so real-dataset request-level and result-level regression remains unexecuted.
- `pip check` in the existing shared Conda environment reports platform-support issues for 18 pre-installed packages unrelated to this repository; it is not a project test failure.
- The offline GitHub Actions workflow is added but cannot be claimed green until GitHub executes it on a push or pull request.
