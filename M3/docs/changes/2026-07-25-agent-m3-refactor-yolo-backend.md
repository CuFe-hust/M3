# Modification Note: M3 Multi-Agent Architecture Refactoring + Optional YOLO Backend - 2026-07-25 17:55 CST

## Modification Time

2026-07-25 17:55:00 CST

## Modifier

TZDEZACR (task context — no push authorization)

## Modification Goal

1. Refactor flat `spacers_agent/` into layered multi-Agent architecture with per-Agent modules.
2. Extract `routing/` package from `routing.py`.
3. Extract `SampleRunner` and `JudgeService` from `DatasetRunner`.
4. Encapsulate `CountingAgent` with pluggable `CountingBackend` protocol, registry, and selector.
5. Pull existing Qwen tile counting into `QwenTileCountingBackend`.
6. Add optional `YoloOBBCountingBackend` (disabled by default, local weights only, no network).
7. Preserve all old import paths and VRSBench quantity behavior.

## Modified Files

### NEW
- `spacers_agent/agents/__init__.py`
- `spacers_agent/agents/base.py` — Agent Protocol, AgentContext, AgentExecution, AgentRegistry
- `spacers_agent/agents/change.py` — ChangeAgent
- `spacers_agent/agents/grounding.py` — GroundingAgent
- `spacers_agent/agents/spatial.py` — SpatialAgent
- `spacers_agent/agents/general_vqa.py` — GeneralVQAAgent
- `spacers_agent/agents/caption.py` — CaptionAgent
- `spacers_agent/agents/counting/__init__.py`
- `spacers_agent/agents/counting/agent.py` — CountingAgent
- `spacers_agent/agents/counting/backends/__init__.py` — CountingBackend Protocol, BackendRegistry
- `spacers_agent/agents/counting/backends/qwen_tile.py` — QwenTileCountingBackend
- `spacers_agent/agents/counting/backends/yolo_obb.py` — YoloOBBCountingBackend
- `spacers_agent/routing/__init__.py` — public re-exports
- `spacers_agent/routing/routes.py` — ROUTES, RoutingDecision, ExpertAssignment
- `spacers_agent/routing/budget.py` — CallBudget, CallBudgetExceeded
- `spacers_agent/routing/router.py` — TaskRouter, CountingExpert
- `spacers_agent/workflows/__init__.py`
- `spacers_agent/workflows/sample_runner.py` — SampleRunner
- `spacers_agent/workflows/judge_service.py` — JudgeService
- `spacers_agent/bootstrap.py` — dependency injection

### MODIFIED
- `spacers_agent/schemas.py` — added YoloWeightEntry, YoloConfig, BackendConfig
- `spacers_agent/settings.py` — added backend: BackendConfig to AppSettings
- `spacers_agent/__init__.py` — added agent exports
- `configs/default.yaml` — added commented backend.yolo section
- `.gitignore` — added weight file patterns
- `pyproject.toml` — added `[yolo]` optional dependency group

### DELETED
- `spacers_agent/routing.py` — replaced by `routing/` package

## Core Changes

1. **Agent Protocol**: Every task has an `Agent` wrapper implementing `execute(ctx) -> AgentExecution`.
2. **AgentRegistry**: Centralized, no hand-rolled dicts.
3. **Routing package**: `ROUTES`, `CallBudget`, `TaskRouter` moved to `routing/` submodules.
4. **CountingBackend**: Protocol + registry + selector. Qwen (priority 0, always available). YOLO (priority 10, conditional).
5. **YOLO backend**: Lazy ultralytics import, OBB center → LocalPointObservation → global → CountingResult. No network, no auto-download.
6. **SampleRunner**: Extracted per-sample routing + dispatch from DatasetRunner.
7. **JudgeService**: Extracted DeepSeek judging from DatasetRunner.
8. **bootstrap.py**: Creates AgentRegistry + BackendRegistry with all registered agents.

## Whether the Canonical Sample Format Was Changed

No. Sample format unchanged.

## Whether the Model Interface Was Changed

No. Qwen/DeepSeek client protocols unchanged.

## Whether the Configuration Was Changed

Yes — added `backend.yolo` section (commented, disabled by default).

## Whether Evaluation Was Affected

No. Evaluation metrics unchanged. VRSBench quantity behavior preserved.

## Whether Deployment Was Affected

No. Deployment unchanged.

## Whether pytest Was Updated

New tests added (see tests/). Existing tests not modified.

## Whether .gitignore Was Updated

Yes — added weight file patterns (`weights/`, `*.pt`, `*.pth`, etc.).

## Validation Method

```bash
python -m compileall spacers_agent
pytest -q
python -m spacers_agent.cli --help
python -m spacers_agent.cli list-datasets
python -m spacers_agent.cli health qwen
python -m spacers_agent.cli health deepseek
python -m pip check
```

## Risks and Follow-up TODOs

1. YOLO backend not tested with real weights (no GPU, no weights available).
2. SampleRunner only handles `single` mode; `fallback` (ensemble) not implemented.
3. DatasetRunner still uses inline expert logic for VRSBench VQA counting; should be migrated to SampleRunner + CountingAgent.
4. No live smoke test performed (requires Qwen endpoint + DeepSeek key).
5. ultralytics dependency is optional; must be installed separately for YOLO.
