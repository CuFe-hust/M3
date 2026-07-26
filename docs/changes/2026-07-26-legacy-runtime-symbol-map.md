# Modification Note: Freeze Legacy Runtime and Map Legacy Symbols - 2026-07-26 12:27:54 +08:00

## Modification Time

2026-07-26 12:27:54 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Freeze the observable behavior of the current `spacers_agent.workflow.DatasetRunner` before the new-Agent runtime cutover, and classify every remaining import of legacy workflow, counting, and expert symbols so later compatibility-layer changes are reviewable.

## Modified Files

- `tests/parity/legacy_runtime_harness.py`
- `tests/parity/canonicalize.py`
- `tests/parity/fake_clients.py`
- `tests/parity/test_legacy_runtime_parity.py`
- `tests/parity/fixtures/*`
- `tests/test_multiagent_vqa_pipeline.py`
- `docs/changes/2026-07-26-legacy-runtime-symbol-map.md`
- `docs/experiments/2026-07-26-pre-cutover-dataset-baseline.md`

## Core Changes

- Added 20 deterministic legacy runtime scenarios covering 13 task routes plus Qwen failure, partial/failed expert payloads, a failed counting tile, Judge failure, missing-Judge resume, and succeeded-sample resume.
- Recorded response schema, request identity/hash, Prompt version, sample/image identity, sanitized message hash, relative artifact directory, text-only Judge payload, final artifacts, status, trace, and summary.
- Replaced embedded images in call records with SHA256 and byte length; no Base64 is stored in fixtures.
- Added an explicit canonicalization whitelist for timestamps, duration, supplied roots, separators, trace extensions, and class module paths. Unknown fields outside the trace whitelist remain comparable.
- Updated one stale pre-cutover test assertion to the already persisted `primary_agent` routing schema; no production routing behavior changed.
- Ran fixed real-adapter samples from VRSBench, LEVIR-CC, MME-RealWorld, and the local XLRS-Bench-lite Arrow release without network or live inference.

## Legacy Runtime Symbol Map

### Production and CLI dependencies

| Legacy module | Consumer | Symbols | Classification | Required migration destination |
|---|---|---|---|---|
| `workflow` | `spacers_agent/cli.py` | `DatasetRunner`, `TargetParser`, `atomic_write_json` | CLI real path | `workflows.DatasetRunner`, counting target parser, shared atomic I/O |
| `workflow` | `agents/spatial/agent.py` | `apply_vrsbench_geometry` | Production geometry helper | `vqa_geometry.apply_vrsbench_geometry` |
| `workflow` | `agents/spatial/candidate_review.py` | spatial private helpers | Production duplicate helper dependency | `agents/spatial/candidate_review.py` single implementation |
| `workflow` | `agents/counting/agent.py` | `atomic_write_json` | Production persistence helper | shared workflow persistence helper |
| `workflow` | `workflows/{sample_runner,dataset_runner,judge_service,counting_workflow}.py` | `atomic_write_json` | New-runtime reverse dependency | shared workflow persistence helper |
| `counting` | `workflow.py`, `cli.py`, `commands/count_image.py` | `PointCountingOrchestrator` | Current real counting path | new Qwen point backend implementation, legacy re-export |
| `counting` | `agents/counting/backends/qwen_point.py` | `PointCountingOrchestrator` | New backend reverse dependency | backend-owned point-counting implementation |
| `counting` | `routing/router.py` | `PointCountingOrchestrator` | Legacy compatibility class dependency | compatibility-only routing export |
| `counting` | `workflows/counting_workflow.py` | orchestrator and geometry helpers | Duplicate workflow dependency | canonical counting backend/workflow modules |
| `counting` | YOLO draft backend | acceptance/seam helpers | Disabled draft dependency | unchanged during this cutover; never registered |

### Tests and public compatibility

| Legacy module | Consumers | Classification |
|---|---|---|
| `workflow` | `tests/architecture/test_legacy_imports.py`, `tests/cli/test_resume_run_contract.py` | Public compatibility imports that must remain valid |
| `workflow` | `tests/test_multiagent_vqa_pipeline.py` | Legacy behavior tests and private-helper regression coverage; migrate to static parity fixtures/new owners |
| `counting` | phase 4/5/stage tests and architecture compatibility tests | Public compatibility for `PointCountingOrchestrator`, checkpoints, seam/acceptance helpers |
| `experts` | architecture compatibility tests | Public compatibility for the old `Expert` and `ExpertContext` names only |

### Documentation-only references

- `README.md`, `DETAILS.md`, `docs/architecture/*`, `docs/implementation_status.md`, and previous change/experiment notes describe the historical implementation. Historical notes are not import consumers and must not be rewritten as if they were current source code.

### Private helpers requiring one owner

| Current symbol | Current owner(s) | Target single owner |
|---|---|---|
| `_run_vqa_counting`, `_run_vqa_count_proposal`, `_run_vqa_count_localizer` | `workflow.DatasetRunner` plus new VRSBench backend equivalents | `agents/counting/backends/vrsbench_qwen_count.py` |
| `_accepted_count_evidence`, `_parse_count_answer`, proposal recovery helpers | `workflow.py` plus `agents/counting/evidence.py` | `agents/counting/evidence.py` |
| `_merge_visual_evidence` and candidate-review helpers | `workflow.py` plus `agents/spatial/candidate_review.py` | `agents/spatial/candidate_review.py` |
| `atomic_write_json` | `workflow.py` used throughout new runtime | a non-legacy workflow I/O module |

### Public compatibility names to preserve

- `workflow`: `DatasetRunner`, `WorkflowService`, `TargetParser`, `CountTargetParser`, legacy visual expert names, geometry/evidence helper imports currently asserted by compatibility tests.
- `counting`: `PointCountingOrchestrator`, `TileCheckpointStore`, `BoundaryConflict`, `SeamDecision`, acceptance/seam/finalization helpers.
- `experts`: `Expert`, `ExpertContext`, persisted expert-schema aliases currently asserted by compatibility tests.

The final legacy modules may import, re-export, map parameters, and provide deprecation documentation only. This stage intentionally does not remove any implementation.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. Recording clients implement the existing injected test interfaces only.

## Whether the Configuration Was Changed

No repository configuration key or default changed.

## Whether Evaluation Was Affected

No metric, split, reference-answer reader, or benchmark meaning changed. The real-data baseline is an offline runtime/artifact regression using deterministic Fake clients, not a model-quality result.

## Whether Deployment Was Affected

No. YOLO, model export, and device deployment were not loaded or exercised.

## Whether pytest Was Updated

Yes. Twenty static legacy parity cases and two canonicalizer/recording-client contract tests were added. One existing VRSBench integration assertion was aligned with the branch's current `primary_agent` schema.

## Whether .gitignore Was Updated

No. Generated run directories and derived local manifests/images are under the already ignored `outputs/` tree. No new generated-file type was introduced.

## Validation Method

- `python -m compileall -q tests/parity`
- `python -m compileall -q spacers_agent tests/parity` -> passed.
- `git diff --check` -> passed.
- `python -m pytest -q -p no:cacheprovider tests/parity/test_legacy_runtime_parity.py` -> `22 passed`
- Targeted parity plus the updated VRSBench integration assertion -> `23 passed`.
- Full repository suite with a repository-local `--basetemp` -> `317 passed`, `0 failed`, `0 errors`.
- Real VRSBench adapter/images: 20 fixed samples, `20 succeeded`, `0 partial`, `0 failed`, `33` deterministic Fake Qwen calls, `0` network calls.
- Derived read-only LEVIR-CC and MME-RealWorld manifests copied selected source images into ignored `outputs/`; eight samples succeeded with eight deterministic Fake Qwen calls and no network call.
- XLRS-Bench-lite Arrow release: two native counting samples succeeded with 52 Fake Qwen calls, two grounding samples succeeded with two Fake Qwen calls, and two derived caption runtime rows reproduced the legacy pre-request `caption_expert` failure. The source Arrow shards were read only through an already cached CPython 3.10/`pyarrow` environment; no package was installed or changed.

## Risks and Follow-up TODOs

- XLRS-Bench-lite is a VQA-only release. Its two caption cases are explicitly derived runtime rows over official embedded images and must not be reported as an official XLRS caption benchmark subset.
- The frozen caption scenario records the current visible `UNSUPPORTED_EXPERT:caption_expert` failure in the old workflow. Later work must explicitly decide whether this is parity or a cutover defect fix.
- The first LEVIR derived-manifest attempt failed before sample execution because absolute external image paths violate the existing Adapter contract. The successful `-v2` run copied only selected images into ignored output storage and did not modify source data.
