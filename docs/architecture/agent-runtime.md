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
│   ├── spatial/            # SpatialAgent + candidate_review + evidence_merge
│   ├── general_vqa/        # GeneralVQAAgent
│   └── caption/            # CaptionAgent
├── routing/                # Task routing
│   ├── schemas.py          # RoutingDecision, ExpertName, AgentName
│   ├── policies.py         # ROUTES table, VRSBench rules
│   ├── budget.py           # CallBudget + CallBudgetFactory
│   └── router.py           # TaskRouter
├── workflows/              # Dataset-level orchestration
│   ├── sample_runner.py    # Single-sample pipeline
│   ├── dataset_runner.py   # Data iteration, resume, concurrency
│   ├── judge_service.py    # DeepSeek text-only judge
│   └── artifact_writer.py  # Artifact serialization only
├── bootstrap.py            # Composition Root (DI)
├── prompt_catalog.py       # Logical key → versioned prompt files
├── cli.py                  # CLI entry point
├── counting.py             # Compat re-exports
├── workflow.py             # Import-compatible thin adapters only
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

### Runtime composition

`bootstrap.assemble_runtime()` constructs one `RuntimeComponents` object containing the injected
Qwen client, optional DeepSeek client, `PromptCatalog`, `TaskRouter`, `AgentRegistry`,
`JudgeService`, `ArtifactWriter`, `CallBudgetFactory`, and `SampleRunner`. `build_dataset_runner()`
reuses that exact `SampleRunner`; it does not construct replacement Router, Registry, Judge, or
Writer instances. `run-dataset` and `resume-run` use this graph through `build_dataset_runner()`;
`count-image` uses the same graph and invokes its `SampleRunner` directly. No CLI execution enters
the retired workflow implementation.

`AgentContext` carries the same `PromptCatalog` and a fresh per-sample `CallBudget`. It does not
carry a second mutable Prompt dictionary. `SampleRunOutcome.status` is the only status consumed by
the new `DatasetRunner`; `completed` and `completed_with_warnings` map to `succeeded`, while
`partial` and `failed` remain visible.

### Non-counting execution

`CaptionAgent`, `ChangeAgent`, `GroundingAgent`, `GeneralVQAAgent`, and `SpatialAgent` are the
authoritative non-counting implementations. Their constructors receive immutable `PromptAsset`
bindings from the Composition Root. `workflow.WorkflowService` is a transitional compatibility
facade: it normalizes former expert names and delegates through an `AgentRegistry`; it does not own
another expert dictionary or model-request implementation.

The request versions remain `caption-v1`, `change-expert-v1`, `general-vqa-v2`, `spatial-v4`,
`spatial-v5`, and spatial-review v2/v3. Spatial review predicates and evidence merging are pure
functions in `agents/spatial/evidence_merge.py`. A spatial request performs no more than one
candidate-review call, then applies VRSBench geometry once. General VQA applies the same geometry
postprocess once for non-spatial VRSBench questions.

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
- Default → `qwen_point`

## YOLO Isolation During Cutover

YOLO draft modules remain in the repository, but the Composition Root does not import or register
them. `backend.yolo.enabled: true` is rejected with a `RuntimeError`. The only registered counting
backends in this cutover are `qwen_point` and `vrsbench_qwen_count`; no `ultralytics` import, model
load, weight validation, or YOLO inference is part of the runtime acceptance gate.

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
