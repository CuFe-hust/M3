# Experiment Record: VRSBench Evidence Router First 20

## Time

2026-07-22 17:19:01 +08:00 to 2026-07-22 17:26:47 +08:00 (wall-clock observation); recorded at 2026-07-22 17:28:54 +08:00.

## Dataset

- Official `VRSBench_EVAL_vqa.json` validation release.
- Source sample count reported by the read-only probe: 37,409.
- Evaluated subset: the first 20 source-ordered samples, question IDs 0 through 19.
- Source annotations and reference answers were not modified.

## Model

- Local `/home/user/lijia/models/Qwen3_vl_4b_instruct` checkpoint.
- In-process Transformers backend, bfloat16, local files only, max 512 new tokens.
- DeepSeek `deepseek-v4-flash` was used only for text/reference answer validation after local Qwen inference.

## Configuration File

- Ignored Spark-local configuration: `configs/local.spark-router.yaml`.
- Git commit: `a210e7f4c78f2bdb7561f2892b06c783f540927c` with a clean tracked worktree.
- Active prompts and hashes are persisted in the run manifest and `prompts.snapshot/`.

## Run Command

```bash
python -m spacers_agent.cli --config configs/local.spark-router.yaml run-dataset \
  --dataset VRSBench --root /home/user/下载/datasets/vrsbench --split validation \
  --task general_vqa --run-id vrsbench-qwen3vl-evidence-router-20-v2-20260722 \
  --max-samples 20 --sample-concurrency 1 --evaluate --judge-policy all
```

The run used a new directory and did not use `--resume` or any prior run cache.

## Metric Results

- Structural completion: 20 succeeded, 0 partial, 0 failed, 0 skipped.
- Existing exact-match accuracy: 5/20 = 25%.
- Text-only DeepSeek semantic proxy: 10/20 = 50%; 20 evaluated and 0 Judge failures.
- Actual Agent distribution: 5 CountingExpert, 12 SpatialExpert, and 3 GeneralVQAExpert samples.
- Persisted evidence: 24 labeled boxes. The five quantity samples produced zero accepted points and therefore answers of zero.
- Quantity subset: 0/5 exact. Saved tile responses visibly contain empty point sets; `final_count` correctly equals the accepted-point count, so the zero results are model/prompt behavior rather than discarded points.
- Deterministic three-by-three position rules made IDs 5 and 16 exact. Cardinal-direction IDs 3 and 13 were not overridden because the ordinary PNG metadata did not provide geographic north.

## Resource Consumption

- Model load: 48.517606 seconds.
- Summed per-sample inference: 371.524058 seconds.
- Observed end-to-end command duration: approximately 466 seconds, including model load, DeepSeek validation, artifact writing, and report generation.
- Final run directory: 7,372,526 bytes across 472 files.
- HTML report directory: 7,043,913 bytes.
- No vLLM process or endpoint was used.

## Conclusion

The official subtype router, retained evidence contract, report overlays, and local JSON compatibility path completed all 20 samples without structural failure. The run is substantially more auditable than the prior direct-GeneralVQA report, but it is not an accuracy improvement: exact match fell to 25% and the accepted-point counting route returned zero on all five quantity questions. A future counting experiment should change the versioned counting inference design and run under a new experiment ID; this result must remain unchanged as the baseline for that comparison.

An earlier fresh preflight run on commit `b884320` completed only 7/20 because common Qwen two-corner box output failed the strict evidence schema. It is retained separately as a failed diagnostic run and is excluded from the metrics above. Commit `a210e7f` added general corner-pair normalization and one image-free JSON repair attempt before this final fresh run.

## Reproducibility Statement

The run manifest records the clean Git commit, configuration hash, model IDs, official dataset probe, prompt hashes, split, sample filter, and run ID. Per-sample directories retain source sample metadata, routing decisions, Qwen requests, raw and parsed responses, validation metadata, duration/token records, expert results, deterministic geometry, DeepSeek audit records, and status. The ignored `.env`, API key value, model weights, dataset files, cache data, and run artifacts are not committed to Git.
