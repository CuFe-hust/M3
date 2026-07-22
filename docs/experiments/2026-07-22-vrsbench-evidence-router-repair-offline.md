# Experiment Record: VRSBench Evidence-Router Repair Offline Validation

## Time

2026-07-22 18:23:34 +08:00

## Dataset

Offline synthetic images, mocked structured responses, and fixed geometry fixtures only. The first 20 VRSBench evaluation references were not used for prompt tuning.

## Model

No model weights were loaded. Tests injected `MockVisionClient` responses.

## Configuration File

`configs/default.yaml`, with VRSBench minimum scan depth 1, empty review enabled, and review-crop maximum side 768.

## Run Command

```powershell
pytest -q tests/test_vrsbench_vqa_geometry.py tests/test_phase4_point_counting.py tests/test_multiagent_vqa_pipeline.py
```

## Metric Results

Focused offline result: 26 passed. The complete repository suite also passed 107 tests. This is contract validation, not a VRSBench accuracy result.

## Resource Consumption

CPU-only Mock tests; no GPU, Qwen weight, DeepSeek API, or remote-server consumption.

## Conclusion

The repaired contracts preserve point-derived counting, enforce visible unconfirmed-zero states, normalize declared answer vocabularies without references, and prevent deterministic extreme selection from a single candidate. Real accuracy remains unvalidated.

## Reproducibility Statement

Tests use repository fixtures and deterministic mock responses. A future real-model experiment must use a new run ID, preserve prompt/config snapshots, avoid resume from earlier runs, and report latency increases from added review calls.
