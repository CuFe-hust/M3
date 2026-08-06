# Modification Note: Complete New-Agent Runtime Composition - 2026-07-26 12:52:16 +08:00

## Modification Time

2026-07-26 12:52:16 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Complete the dependency graph required by the new Agent runtime without switching the CLI or
removing the frozen legacy implementation.

## Modified Files

- `spacers_agent/bootstrap.py`
- `spacers_agent/prompt_catalog.py`
- `spacers_agent/agents/base.py`
- `spacers_agent/agents/grounding/agent.py`
- `spacers_agent/routing/budget.py`
- `spacers_agent/routing/router.py`
- `spacers_agent/routing/__init__.py`
- `spacers_agent/workflows/artifact_writer.py`
- `spacers_agent/workflows/sample_runner.py`
- `spacers_agent/workflows/dataset_runner.py`
- `spacers_agent/workflows/judge_service.py`
- `spacers_agent/workflows/__init__.py`
- `tests/runtime/*`
- Existing Agent/workflow tests updated for the injected runtime contracts
- `DETAILS.md`
- `docs/architecture/agent-runtime.md`
- `docs/architecture/counting-backends.md`

## Core Changes

- Expanded `RuntimeComponents` to contain the Qwen/Judge clients, versioned Prompt catalog, one
  Router, one Agent registry, one Judge service, one Artifact writer, one budget factory, and one
  Sample runner.
- Added `build_dataset_runner()` that reuses the exact SampleRunner and injected dependency graph.
- Added immutable `PromptAsset` bindings for all active Prompt files and request versions without
  changing Prompt text.
- Added `CallBudgetFactory` with the previous centralized limits of 50 Qwen and 10 DeepSeek calls
  per sample by default.
- Replaced the duplicate AgentContext Prompt dictionary with the runtime `PromptCatalog`.
- Added `SampleRunOutcome` and the single `sample_state_from_payload()` mapping. The new
  DatasetRunner consumes `outcome.status` and cannot default partial/failed payloads to success.
- Added a business-neutral `ArtifactWriter` for all declared sample/run artifact operations.
- Made Router-agent failures return an auditable rule-fallback reason code instead of silently
  discarding the error.
- Kept Judge failures visible while preserving the Agent result and payload-derived status.
- Rejected YOLO enablement in the Composition Root and registered only `qwen_point` and
  `vrsbench_qwen_count`.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No model, processor, tokenizer, endpoint, or weight-loading behavior changed. Existing injected
client protocols are reused.

## Whether the Configuration Was Changed

No configuration key or default changed. Existing YOLO enablement is now explicitly rejected by
the new Composition Root during this cutover; the current CLI is not switched in this phase.

## Whether Evaluation Was Affected

No metric, split, reference-answer reader, or benchmark meaning changed. The new Judge service
continues to pass text and structured evidence only. Its errors are persisted rather than swallowed.

## Whether Deployment Was Affected

No. YOLO, `ultralytics`, model weights, exports, and device-specific runtimes were not loaded.

## Whether pytest Was Updated

Yes. Runtime wiring, budget creation, status mapping, ArtifactWriter output, Judge failure
visibility, Router fallback reasons, Prompt bindings, and YOLO isolation are covered.

## Whether .gitignore Was Updated

No. No new generated artifact class or output directory was introduced.

## Validation Method

- `python -m compileall -q spacers_agent tests/runtime tests/parity` -> passed.
- Runtime plus frozen legacy parity tests -> `39 passed`.
- `git diff --check` -> passed.
- Full repository suite -> `334 passed`, `0 failed`, `0 errors`.

## Risks and Follow-up TODOs

- The CLI intentionally remains on the frozen legacy DatasetRunner in this phase.
- Legacy `workflow.py`, `counting.py`, and `experts.py` still contain their implementations and
  must not be removed until later Agent parity stages pass the static fixtures.
- The YOLO draft remains outside acceptance and must not be enabled during the cutover.
