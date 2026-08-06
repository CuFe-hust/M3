# Experiment Record: VRSBench Grounded Inference v4 Offline Validation

## Time

2026-07-22 20:18:00 +08:00

## Dataset

No live dataset was read. Tests use generated images and local mock responses only.

## Model

No model was loaded. The production target remains the configured local Qwen3-VL-4B-Instruct Transformers checkpoint.

## Configuration File

`configs/default.yaml`

## Run Command

```powershell
python -m compileall -q spacers_agent tests
python -m pytest -q
```

## Metric Results

No benchmark accuracy was measured. Offline contract and geometry tests passed; the final test count is recorded in the modification note and commit validation.

## Resource Consumption

CPU-only offline validation; no GPU model inference and no cloud API calls.

## Conclusion

The v4 inference contracts preserve accepted-point counting, remove fabricated one-pixel boxes, support independent candidate review, and trigger finer crops after an unconfirmed overview zero. Visual-recall improvement remains unvalidated until a fresh Spark run.

## Reproducibility Statement

All validation uses committed prompts, configuration, generated fixtures, and Mock clients. Reproduce the actual 20-sample comparison with a new run ID and the same local Qwen checkpoint; do not reuse cached sample artifacts from earlier prompt versions.
