# Experiment Record: VRSBench Hybrid Count v1

## Time

2026-07-22 22:32:20 +08:00

## Dataset

Offline tests use generated images and Mock responses. A fresh 20-sample VRSBench validation run on Spark is pending.

## Model

Production target: the configured local Qwen3-VL-4B-Instruct Transformers checkpoint. DeepSeek remains a text-only answer Judge.

## Configuration File

Spark-local ignored configuration; no committed configuration values changed.

## Run Command

```powershell
python -m compileall -q .
pytest -q
```

## Metric Results

Offline result: 117 pytest tests passed. Live VRSBench metrics will be recorded after the fresh Spark run.

## Resource Consumption

Offline validation was CPU-only. Live GPU inference resource use is pending.

## Conclusion

The workflow contract now prevents an omitted model point list from automatically forcing a zero count. It preserves supporting boxes but still derives `final_count` exclusively from accepted centre points.

## Reproducibility Statement

Offline behavior is reproducible from committed prompts, generated fixtures, and Mock clients. Live validation must use a fresh run ID with the same local checkpoint and official VRSBench validation data; prior cached sample artifacts must not be reused.
