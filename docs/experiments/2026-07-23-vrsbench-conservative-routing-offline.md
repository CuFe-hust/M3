# Experiment Record: VRSBench Conservative Routing Offline Validation

## Time

2026-07-23 01:02:46 +08:00

## Dataset

Synthetic router/geometry fixtures plus a read-only question-only audit of 173 persisted VRSBench records from the prior 200-sample validation run. Reference answers were not supplied to routing or subtype classification.

## Model

No live model. Tests used local mock clients; the route audit called deterministic Python functions only.

## Configuration File

Not applicable. No runtime model configuration or secret was loaded.

## Run Command

```powershell
python -m compileall -q spacers_agent tests
python -m pytest -q tests/test_vrsbench_vqa_geometry.py tests/test_phase5_routing.py tests/test_multiagent_vqa_pipeline.py
python -m pytest -q
```

The read-only audit loaded each persisted question and prior official type, then called `vrsbench_question_subtype` and `execution_task_for_vrsbench`. It did not load reference answers into either function.

## Metric Results

- Focused regression suite: 46 passed.
- Full pytest suite: 131 passed.
- Audited deterministic route distribution over 173 questions:
  - General VQA: 109
  - Explicit counting: 47
  - High-confidence spatial: 17

These are routing-distribution counts, not model-accuracy metrics.

## Resource Consumption

CPU-only local validation. No GPU memory, Qwen inference, DeepSeek tokens, network traffic, or server time was consumed.

## Conclusion

The deterministic layer no longer forces open scene/category questions into the vehicle vocabulary, unknown official types have a safe GeneralVQA fallback, and valid generic position-review boxes survive into deterministic grid geometry. Status placeholders are kept out of semantic answers.

## Reproducibility Statement

The pytest commands are fully local and deterministic. Model-quality conclusions require a fresh run on a disjoint sample set because this validation intentionally made no live model calls.
