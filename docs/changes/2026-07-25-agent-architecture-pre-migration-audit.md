# Pre-Migration Audit: M3 Multi-Agent Architecture — 2026-07-25 18:15 CST

## Modification Time

2026-07-25 18:15:00 CST

## Modifier

TZDEZACR (task context — pre-migration audit)

## Git Snapshot

| Field | Value |
|---|---|
| Branch | `try_yolo` |
| HEAD | `949365e7723078facd9fe36edabf351075c289cc` |
| HEAD description | Merge fix/vrsbench-conservative-routing into main |
| Upstream | `origin/main` at same commit |
| Dirty | Yes — uncommitted refactoring + new files from prior task |

## Environment

| Tool | Version |
|---|---|
| Python | 3.11.15 |
| pytest | 9.1.1 |
| pydantic | 2.x |

## Class Location Map

### Workflow / Expert (workflow.py:1328 lines)

| Class | Line | Inherits |
|---|---|---|
| `VisualExpert` | 98 | — |
| `CountTargetParser` | 78 | — |
| `ChangeExpert` | 174 | VisualExpert |
| `GroundingExpert` | 181 | VisualExpert |
| `SpatialExpert` | 188 | VisualExpert |
| `GeneralVQAExpert` | 368 | VisualExpert |
| `WorkflowService` | 375 | — |
| `DatasetRunner` | 403 | — |
| `TargetParser` | 171 | alias = CountTargetParser |

### Routing (routing/ package + old routing.py DELETED)

| Class | Module |
|---|---|
| `TaskRouter` | `routing/router.py:22` |
| `RoutingDecision` | `routing/routes.py:56` |
| `ExpertAssignment` | `routing/routes.py:45` |
| `CallBudget` | `routing/budget.py:19` |
| `CallBudgetExceeded` | `routing/budget.py:13` |
| `CountingExpert` | `routing/router.py:120` |
| `CountingExpertAnswer` | `routing/router.py:108` |

### Counting (counting.py:833 lines)

| Class | Line |
|---|---|
| `BoundaryConflict` | 55 |
| `SeamDecision` | 69 |
| `TileCheckpointStore` | 92 |
| `PointCountingOrchestrator` | 229 |

### Experts (experts.py:86 lines)

| Symbol | Line |
|---|---|
| `Expert` (Protocol) | 48 |
| `ExpertContext` (dataclass) | 39 |

### New Agent Wrappers (agents/)

| Agent | File | Delegates to |
|---|---|---|
| `ChangeAgent` | `agents/change.py` | `workflow.ChangeExpert` |
| `GroundingAgent` | `agents/grounding.py` | `workflow.GroundingExpert` |
| `SpatialAgent` | `agents/spatial.py` | `workflow.SpatialExpert` |
| `GeneralVQAAgent` | `agents/general_vqa.py` | `workflow.GeneralVQAExpert` |
| `CaptionAgent` | `agents/caption.py` | `workflow.GeneralVQAExpert` |
| `CountingAgent` | `agents/counting/agent.py` | `BackendRegistry.select()` |

### Counting Backends

| Backend | Priority | File |
|---|---|---|
| `QwenTileCountingBackend` | 0 | `agents/counting/backends/qwen_tile.py` |
| `YoloOBBCountingBackend` | per-weight | `agents/counting/backends/yolo_obb.py` |

## Import Usage Map

### Import `spacers_agent.workflow` from:

- `cli.py` (DatasetRunner, TargetParser, atomic_write_json)
- `workflows/judge_service.py` (atomic_write_json)
- `workflows/sample_runner.py` (CountTargetParser, atomic_write_json)
- `workflows/counting_workflow.py` (atomic_write_json)
- `agents/caption.py` (GeneralVQAExpert)
- `agents/change.py` (ChangeExpert)
- `agents/general_vqa.py` (GeneralVQAExpert)
- `agents/grounding.py` (GroundingExpert)
- `agents/spatial.py` (SpatialExpert)
- `agents/counting/agent.py` (CountTargetParser, atomic_write_json)
- `tests/test_multiagent_vqa_pipeline.py` (DatasetRunner, GeneralVQAExpert, SpatialExpert + private helpers)

### Import `spacers_agent.routing` from:

- `workflow.py` (CallBudget, TaskRouter, CountingExpert)
- `commands/count_image.py` (CountingWorkflow imports routing transitively via counting_workflow)
- `tests/test_phase5_routing.py` (CallBudget, CountingExpert, TaskRouter, attach_qwen_budget)
- `tests/test_routing_package.py` (new test)
- `workflows/sample_runner.py` (TaskRouter)

### Import `spacers_agent.counting` from:

- `cli.py` (PointCountingOrchestrator)
- `workflow.py` (PointCountingOrchestrator)
- `routing/router.py` (PointCountingOrchestrator)
- `agents/counting/backends/qwen_tile.py` (PointCountingOrchestrator)
- `agents/counting/backends/yolo_obb.py` (apply_acceptance_policy, find_boundary_conflicts, finalize_representatives)
- `commands/count_image.py` (PointCountingOrchestrator)
- `tests/test_phase4_point_counting.py`
- `tests/test_phase5_routing.py`
- `tests/test_targeting_and_seam.py`

### Import `spacers_agent.experts` from:

- No current direct import in production code
- `tests/test_stage_a_to_g_contracts.py` may reference it

## Artifact Map

| Artifact | Relative to run_dir | Writer(s) |
|---|---|---|
| `manifest.json` | `/` | `run_store.py:87` |
| `config.snapshot.yaml` | `/` | `run_store.py:88` |
| `prompts.snapshot/` | `/` | `run_store.py` |
| `events.jsonl` | `/` | `events.py` |
| `predictions.jsonl` | `/` | `workflow.py:452` |
| `dataset_summary.json` | `/` | `workflow.py:487` |
| `samples/<id>/sample.json` | `/samples/<id>/` | `workflow.py:494` |
| `samples/<id>/status.json` | `/samples/<id>/` | `workflow.py:496,583` |
| `samples/<id>/routing_decision.json` | `/samples/<id>/` | `workflow.py:511` |
| `samples/<id>/counting_result.json` | `/samples/<id>/` | `workflow.py:516`, `agents/counting/agent.py:78` |
| `samples/<id>/expert_result.json` | `/samples/<id>/` | `workflow.py:555`, `agents/counting/agent.py:89` |
| `samples/<id>/evaluation_record.json` | `/samples/<id>/` | `workflow.py:518` |
| `samples/<id>/agent_trace.json` | `/samples/<id>/` | `workflow.py:562`, `workflows/sample_runner.py:89` |
| `samples/<id>/vqa_evaluation.json` | `/samples/<id>/` | `workflow.py:871` |

## Router Behavior

### Route Mapping

```
counting               → counting_expert (single)
fine_grained_counting   → counting_expert, spatial_expert (multi, only first executed)
change_caption          → change_expert (single)
change_qa               → change_expert, general_vqa_expert (multi, only first executed)
grounding               → grounding_expert (single)
spatial_relation        → spatial_expert (single)
scene_classification    → general_vqa_expert (single)
general_vqa             → general_vqa_expert (single)
multiple_choice_vqa     → general_vqa_expert (single)
```

### KNOWN DEFECT: `caption` has NO ROUTE

`TaskName` includes `caption` but `ROUTES` does not. Any sample with `task="caption"` will raise `KeyError` at `ROUTES[task]`.

### VRSBench Routing (conservative)

`TaskRouter.route_vrsbench_vqa` uses `execution_task_for_vrsbench()` from `vqa_geometry.py`:
- `counting` questions → `counting` task → `counting_expert`
- `extreme_category`, `grid_position`, `orientation`, `arrangement` → `spatial_relation` task → `spatial_expert`
- Everything else → `general_vqa` task → `general_vqa_expert`

### Execution Mode

Only `single` mode implemented — `decision.experts[0]` is always used. Multi-expert routes (e.g., `fine_grained_counting`, `change_qa`) declare fallback experts but never execute them.

## Counting Special Paths

### VRSBench Quantity VQA (in workflow.py DatasetRunner._run_one)

When `sample.dataset == "VRSBench"` AND `task == "general_vqa"` AND routed to `counting_expert`:
1. `vrsbench_count_target()` — fixed vehicle ontology, NO LLM target parse
2. `_run_vqa_count_proposal()` — GeneralVQA v1 whole-image count proposal with boxes
3. If proposal_count ≠ accepted_points → `_run_vqa_count_localizer()` — independent box enumeration
4. `_accepted_count_evidence()` — dedup, drop tiny border fragments, keep centres
5. `_merge_count_evidence()` — near-identical vehicle dedup (IoU ≥ 0.9 or point distance ≤ 12)
6. VRSBench vehicles class aliases used: small-vehicle (car, automobile, motorcycle), large-vehicle (truck, bus, trailer)

### Tile-Based Point Counting (QwenTileCountingBackend)

1. `build_core_halo_tiles()` → row-major non-overlapping owner cores
2. Sequential tile processing with resume via `TileCheckpointStore`
3. `convert_local_point_to_global()` → strict owner-core acceptance
4. `apply_acceptance_policy()` → reject below min_confidence
5. `find_boundary_conflicts()` → adjacent-core near-boundary candidates
6. `finalize_representatives()` → same-instance merges via union-find
7. `final_count == sum(accepted points)` enforced by `CountingResult` validator

### YOLO OBB Path (YoloOBBCountingBackend)

1. Same tile geometry as Qwen
2. OBB inference per tile → box centre → `LocalPointObservation` → global conversion
3. No seam-verify (no DeepSeek for YOLO)
4. Boundary conflicts flagged but not merged

## Judge Paths

### Counting Judge

- `evaluate-run --deepseek` → `cli._evaluate_run()` → `DeepSeekJudgeClient`
- `build_count_judge_payload()` — text + structured evidence, NO image
- `merge_count_evaluation()` — flags `judge_inconsistency` when verdict=correct but count mismatches

### VQA Judge

- `judge-vqa-run` → `cli._judge_vqa_run()` → `DeepSeekJudgeClient`
- `build_vqa_judge_payload()` — question + references + candidate + exact_match flag
- Can also run inline during `run-dataset --evaluate --judge-policy all`

## Semantic Invariants (must survive migration)

1. `final_count == count of accepted points` (CountingResult.check_count)
2. Failed tiles → status ∈ {partial, failed} (CountingResult.check_count)
3. Change samples: `t1` before `t2` (UnifiedSample.validate_temporal_order)
4. VisualEvidence: exactly one of box or point (VisualEvidence.validate_geometry)
5. Coordinates: `0..999` normalized; `PixelRect` half-open (`left < right AND top < bottom`)
6. DeepSeek never sees images, Base64, or file paths (build_count_judge_payload / build_vqa_judge_payload)
7. API keys never in settings models, manifests, or artifacts
8. Tile resume: matching request_hash in TileCheckpointStore.load_success()
9. Prompt versions: versioned files in `prompts/`, snapshotted by `run-init`

## Test Baseline

```
Total: 155 tests (109 passed, 46 errors)
Errors: all due to Windows tmp_path PermissionError in pytest-asyncio (pre-existing)
```

## Risks

1. `caption` has no ROUTE entry → any caption-typed sample will crash
2. Multi-expert routes declared but never executed (only experts[0] used)
3. 46 test errors from Windows permission issue (not code bugs)
4. `counting_workflow.py` in `workflows/` appears to be an orphan/unused file
5. `routing.py` (old flat file) deleted, replaced by `routing/` package — imports must all resolve through package `__init__.py`
