# Agent Runtime Architecture

## Directory Layout

```
spacers_agent/
├── agents/                 # Per-task Agent implementations
│   ├── base.py             # Agent Protocol, AgentContext, AgentExecution
│   ├── registry.py         # AgentRegistry
│   ├── errors.py           # Stable error types
│   ├── visual_base.py      # VisualAgentBase (shared visual primitive)
│   ├── counting/           # CountingAgent + Backends
│   ├── change/             # ChangeAgent
│   ├── grounding/          # GroundingAgent
│   ├── spatial/            # SpatialAgent + candidate_review
│   ├── general_vqa/        # GeneralVQAAgent
│   └── caption/            # CaptionAgent
├── routing/                # Task routing
│   ├── schemas.py          # RoutingDecision, ExpertName, AgentName
│   ├── policies.py         # ROUTES table, VRSBench rules
│   ├── budget.py           # CallBudget
│   └── router.py           # TaskRouter
├── workflows/              # Dataset-level orchestration
│   ├── sample_runner.py    # Single-sample pipeline
│   ├── dataset_runner.py   # Data iteration, resume, concurrency
│   └── judge_service.py    # DeepSeek text-only judge
├── bootstrap.py            # Composition Root (DI)
├── prompt_catalog.py       # Logical key → versioned prompt files
├── cli.py                  # CLI entry point
├── counting.py             # Compat re-exports
├── workflow.py             # Compat re-exports
├── experts.py              # Compat re-exports
└── ...
```

## Dependency Direction

```
schemas / settings / clients / imaging / evaluation
              ↑
         agents (never import CLI, DatasetRunner, SampleRunner)
    ┌─────────┤
    ↓         ↓
 routing   workflows
    └────┬────┘
         ↓
     bootstrap
         ↓
        cli
```

## Key Contracts

### Agent

```python
class Agent(Protocol):
    name: AgentName
    supported_tasks: frozenset[str]
    async def run(self, sample, context: AgentContext) -> AgentExecution: ...
```

### CountingBackend

```python
class CountingBackend(Protocol):
    name: str
    def is_available(self) -> bool: ...
    def supports(self, target: CountTargetSpec) -> bool: ...
    async def count(self, request: CountingRequest, context) -> CountingResult: ...
```

## Routing

`TaskRouter.route_sample()` → VRSBench semantic → known task → unknown → rule fallback.

ROUTES map task → (primary_agent, fallback_agents...).

## Counting Backend Selection

`BackendSelector.select(target, sample)`:
- VRSBench general_vqa → `vrsbench_qwen_count`
- YOLO enabled + class match → first matching detector by priority
- Default → `qwen_point`

## YOLO Integration

- Default OFF (`backend.yolo.enabled: false`)
- Lazy import of `ultralytics` in `YoloModelStore.get()`
- Weight file checked BEFORE import
- No network, no auto-download
- OBB centre → `LocalPointObservation` → global conversion → `CountingResult`
- Provenance tracked in `PointProvenance`

## Artifacts

Per sample (in `samples/<id>/`):
- `sample.json`, `status.json`
- `routing_decision.json` (new format, old format readable)
- `counting_result.json` or `expert_result.json`
- `agent_trace.json`
- `evaluation_record.json`

Run-level:
- `manifest.json`, `config.snapshot.yaml`, `prompts.snapshot/`
- `events.jsonl`, `predictions.jsonl`, `dataset_summary.json`
