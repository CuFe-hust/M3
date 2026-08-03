# Experiment Record: Qwen3.5-9B Spark Smoke Test

## Time

2026-08-03 10:29 +08:00

## Dataset

No benchmark dataset was used. The smoke input was the committed generated fixture
`tests/fixtures/legacy/test_image.png`, a solid gray image, with the question
`Describe this image briefly.`

## Model

Official `Qwen/Qwen3.5-9B`, downloaded through ModelScope to the external server path
`/home/user/models/Qwen3.5-9B`. The checkpoint contains four safetensors shards and
occupies approximately 19 GiB on disk. No weights were modified or committed.

## Configuration File

Ignored Spark-local `configs/local.spark-router.yaml`, using the Transformers backend,
`bfloat16`, `device_map: auto`, `local_files_only: true`, and the external model path.

## Run Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/time -v \
  /home/user/miniconda3/envs/Cooper_tryagents/bin/python \
  -m spacers_agent.cli --config configs/local.spark-router.yaml \
  smoke-qwen --image tests/fixtures/legacy/test_image.png \
  --question 'Describe this image briefly.'
```

## Metric Results

This was not a quality evaluation and produced no benchmark metric. The command exited
with status 0. The structured result had `status: completed`, no geometry, and the answer
`The image is a solid gray color with no discernible objects or features.` The response
contained no Qwen3.5 thinking prefix and passed the existing `ExpertResult` validation.

## Resource Consumption

- Model weight loading progress completed in approximately 50 seconds.
- End-to-end command wall time: 60.40 seconds.
- Maximum process resident set reported by `/usr/bin/time`: 9,521,604 KiB.
- Filesystem input reported by `/usr/bin/time`: 3,248,152 blocks.
- Swap activity: none.
- The GB10 driver did not report per-process GPU memory through the queried
  `nvidia-smi` fields, so peak accelerator/unified-memory use was not measured.
- No inference process remained after the smoke command exited.

## Conclusion

The deployed native Qwen3.5 model class, Qwen3-VL processor, non-thinking chat-template
setting, deterministic generation path, and strict JSON validation work together for one
real image request on Spark. This smoke result does not establish remote-sensing quality.

## Reproducibility Statement

The run used Git commit `9ad75d0`, the external official checkpoint, Python 3.12.13,
PyTorch 2.13.0+cu130, Transformers 5.14.1, CUDA 13.0, and one NVIDIA GB10. Network access
was disabled for model loading and inference. A fresh run ID and fixed benchmark sample set
are required before comparing Qwen3.5-9B with historical Qwen3-VL-4B results.
