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

Offline result before the first run: 117 pytest tests passed. The first Spark run restored four of five quantity answers, but ID 14 remained partial because count deduplication collapsed two adjacent coarse boxes at IoU 0.77. The persisted proposal and localizer both reported four, while only three points remained. This directly motivated the count-only near-identical-box threshold and its regression test. After adding that test, the full suite passed 118 tests.

The second fresh Spark run `vrsbench-qwen3vl-hybrid-count-v1-b24024d-20-20260722-r2` completed all 20 samples with 13 succeeded, 7 partial, and 0 failed workflow states. Deterministic exact match was 15/20 (75%). All five quantity questions were exact: IDs 1, 4, 9, 14, and 18 returned 2, 3, 1, 4, and 5 respectively, and every persisted answer matched both its supporting-box count and accepted-point count. ID 14 specifically persisted proposal 4, localizer answer 4, four boxes, four accepted points, and completed status.

DeepSeek remained unchanged, but this shell did not export the existing `.env` key, so Judge status was `not_requested`; no DeepSeek result is claimed for this run. Per the task scope, Qwen was not rerun merely to populate Judge fields.

## Resource Consumption

Offline validation was CPU-only. The final Spark run loaded the local checkpoint in approximately 50 seconds and completed the end-to-end command in approximately 242 seconds. Peak GPU memory was not measured in this run.

## Conclusion

The workflow contract prevents an omitted model point list from automatically forcing a zero count. The first live run exposed an overly aggressive count-evidence merge, and the second live run confirmed all five quantity answers after narrowing only the count-specific merge threshold. General spatial deduplication remains unchanged, supporting boxes remain visible, and `final_count` is still derived exclusively from accepted centre points.

## Reproducibility Statement

Offline behavior is reproducible from committed prompts, generated fixtures, and Mock clients. The final live result is tied to commit `b24024d`, the configured local Qwen3-VL-4B-Instruct checkpoint, the first 20 official VRSBench validation questions, sequential sample execution, and the fresh run ID above. Prior cached sample artifacts were not reused. Because the same validation subset informed the count-deduplication diagnosis, its 75% overall score is an engineering regression result rather than an unbiased held-out benchmark estimate.
