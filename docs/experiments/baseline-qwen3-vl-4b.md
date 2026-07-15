# Experiment Record: Qwen3-VL-4B Zero-Shot Baseline

## Time

2026-07-15 22:51:41 +08

## Dataset

- VRSBench validation: captioning, VQA, and visual grounding.
- MME-RealWorld: Remote Sensing subdomain only.
- XLRS-Bench: full English captioning, full English visual grounding, and Lite VQA reported separately.
- LEVIR-CC test: bi-temporal change captioning.

## Model

`Qwen/Qwen3-VL-4B-Instruct`, original checkpoint, zero-shot inference only.

## Configuration File

`config/baseline.example.json`, copied to ignored `config/local.baseline.json` before execution.

## Run Command

```bash
python main.py --config config/local.baseline.json download
python main.py --config config/local.baseline.json infer --dataset all --overwrite
```

## Metric Results

Pending. No dataset download, model download, or Colab inference was run in this local repository.

## Resource Consumption

Pending Colab measurement. Record GPU type, peak GPU memory, model storage, dataset storage,
per-sample latency, and end-to-end runtime when executing the baseline.

## Conclusion

This record fixes the zero-shot baseline protocol before later LoRA experiments. XLRS full
Caption/Grounding and Lite VQA must remain separate result groups.

## Reproducibility Statement

The command saves model settings, UTC timestamp, task-scope notes, and completed sample count
next to every JSONL result. The optional DeepSeek score requires `DEEPSEEK_API_KEY` in the
Colab session and is a non-official proxy metric.
