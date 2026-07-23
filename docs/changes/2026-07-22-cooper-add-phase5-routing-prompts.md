# Modification Note: Phase 5 Sparse Routing and Prompt Assets - 2026-07-22 11:40:25 +08:00

## Modification Time

2026-07-22 11:40:25 +08:00

## Modifier

Cooper3516833584 (local implementation; no push was performed).

## Modification Goal

Add deterministic multi-Agent routing, bounded model-call budgeting, an honest counting presentation layer, and versioned prompt assets without changing baseline inference or evaluation.

## Modified Files

- `spacers_agent/routing.py`
- `spacers_agent/counting.py`
- `spacers_agent/settings.py`
- `spacers_agent/cli.py`
- `prompts/router_v1.md`
- `prompts/target_parse_v1.md`
- `prompts/count_repair_v1.md`
- `prompts/seam_verify_v1.md`
- `prompts/missing_point_review_v1.md`
- `prompts/change_v1.md`
- `prompts/spatial_v1.md`
- `prompts/general_vqa_v1.md`
- `tests/test_phase5_routing.py`
- `README.md`
- `DETAILS.md`
- `docs/implementation_status.md`

## Core Changes

- Added fixed rule routes for declared tasks and an explicitly injected text-only fallback for unknown tasks.
- Added `CallBudget`; Qwen and DeepSeek reservations fail before an over-budget request is issued.
- Added `CountingExpert`, which wraps rather than duplicates the Phase 4 point pipeline and never describes a partial result as final.
- Added independent versioned Prompt files and expanded `run-init` Prompt snapshots.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No existing model interface changed. The new router and expert use the existing injected `VisionLanguageClient` protocol.

## Whether the Configuration Was Changed

Yes. The default point-counting prompt version is now `count-point-v2`, matching `count_tile_v2.md`.

## Whether Evaluation Was Affected

No existing metrics, reference-answer readers, dataset splits, or output formats changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Added Mock-based tests for deterministic routing, unknown-task routing, exhausted budgets, global-point counting answers, and resume without another budget charge.

## Whether .gitignore Was Updated

No. No new generated-file type or output root was introduced.

## Validation Method

- `C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m pytest -q tests\test_phase5_routing.py tests\test_phase1_foundation.py tests\test_phase4_point_counting.py`

## Risks and Follow-up TODOs

- No Qwen, DeepSeek, SSH, server, cloud, model, or dataset call was made.
- Change, grounding, spatial, general-VQA, visual critic, and DeepSeek judge prompts are versioned assets only; their live expert workflows require separate implementation and explicit authorization.
- Prompt revisions must use new versioned files and offline regression fixtures; test sets must not be used for prompt development.
