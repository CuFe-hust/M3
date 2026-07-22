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

## Live Spark Validation

### Time

2026-07-23 01:19-01:52 +08:00

### Dataset and Model

- VRSBench official evaluation file, validation split, first 200 `general_vqa` samples selected by the existing dataset adapter.
- Local Qwen3-VL-4B-Instruct Transformers weights at the Spark-local configured path.
- DeepSeek `deepseek-v4-flash` was used only as the configured text-and-structured-evidence semantic judge for all 200 predictions.
- Git commit: `8166c14c611e60546359bfcb403c96ce749f129c`.
- Run ID: `vrsbench-qwen3vl-conservative-8166c14-200-20260723-r1`.

### Configuration and Run Command

Configuration: `configs/local.spark-router.yaml` on Spark. The local configuration and API credentials were not committed.

```bash
python -m spacers_agent.cli \
  --config configs/local.spark-router.yaml \
  run-dataset \
  --dataset VRSBench \
  --root <spark-local-vrsbench-root> \
  --split validation \
  --task general_vqa \
  --run-id vrsbench-qwen3vl-conservative-8166c14-200-20260723-r1 \
  --max-samples 200 \
  --sample-concurrency 1 \
  --evaluate \
  --judge-policy all
```

### Metric Results

- Dataset execution: 200 total, 193 succeeded, 7 partial, 0 failed, 0 skipped.
- Exact-match accuracy: 83/200 = 0.415.
- DeepSeek semantic-match proxy: 104/200 = 0.520; all 200 records were evaluated and no judge call failed.
- The DeepSeek score is a text-only proxy and is not the official VRSBench GPT metric.

### Resource Consumption

The sequential 200-sample run, including local Qwen loading, inference, DeepSeek judging, export, and HTML generation, took approximately 33 minutes wall-clock time. Peak GPU memory and power consumption were not measured, so no device-utilization claim is made.

### Conclusion and Reproducibility

The fresh 200-sample run completed without a hard sample failure and produced a self-contained HTML audit report. The report, predictions, judge audit, metrics, manifest, and configuration snapshot were downloaded outside the repository. The manifest records the Spark worktree as dirty because a local untracked configuration backup was intentionally preserved; the executed source commit is recorded above. Reproduction requires the same local model weights, VRSBench files, Spark-local configuration, and API environment variables.
