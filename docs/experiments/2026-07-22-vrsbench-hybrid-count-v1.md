# Experiment Record: VRSBench Hybrid Count v1

## Time

2026-07-22 22:32:20 +08:00

## Dataset

Offline tests use generated images and Mock responses. Spark run `vrsbench-qwen3vl-hybrid-count-v1-502d8ed-20-20260722-r1` used the first 20 official VRSBench validation questions.

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

Offline result before the first run: 117 pytest tests passed. The first Spark run restored four of five quantity answers, but ID 14 remained partial because count deduplication collapsed two adjacent coarse boxes at IoU 0.77. The persisted proposal and localizer both reported four, while only three points remained. This directly motivated the count-only near-identical-box threshold and its regression test. Final live metrics remain pending a second fresh run.

## Resource Consumption

Offline validation was CPU-only. Live GPU inference resource use is pending.

## Conclusion

The workflow contract prevents an omitted model point list from automatically forcing a zero count. The first live run confirmed four corrected quantity answers and exposed an overly aggressive count-evidence merge, which is now narrowed without changing general spatial deduplication. Supporting boxes remain visible and `final_count` is still derived exclusively from accepted centre points.

## Reproducibility Statement

Offline behavior is reproducible from committed prompts, generated fixtures, and Mock clients. Live validation must use a fresh run ID with the same local checkpoint and official VRSBench validation data; prior cached sample artifacts must not be reused.
