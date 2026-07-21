# Experiment Record: Qwen3-VL-4B Four-Dataset Baseline and Agent Smoke Test

## Time

2026-07-17 to 2026-07-20, Asia/Shanghai (UTC+08:00).

## Dataset

- VRSBench validation: two caption, two VQA, and two grounding records.
- MME-RealWorld Lite: two `Perception/Remote Sensing` position records.
- XLRS-Bench Lite: the first two VQA records from official shard
  `train/data-00000-of-00074.arrow`.
- LEVIR-CC test: records `8148` and `8149` for change captioning.

These are smoke-test subsets and must not be reported as official full-dataset scores.

## Model

`Qwen/Qwen3-VL-4B-Instruct`, original checkpoint, deterministic zero-shot inference with no
training, LoRA, quantization, or weight modification. The Agent path used LangGraph 1.2.5 to
run `read_sample`, `call_qwen`, `validate_prediction`, and `save_result` with the same model.

## Configuration File

`config/local.smoke.json`, derived from `config/baseline.example.json`, with
`max_new_tokens=256`. MME-RealWorld and XLRS-Bench inference capped processor input at
1,003,520 pixels to run ultra-high-resolution images safely on a Tesla T4; source images were
preserved.

## Run Command

The smoke subsets were selected and run in a Google Colab notebook through the repository's
`CanonicalSample`, `Qwen3VLBaseline.predict`, and `CanonicalPrediction.validate` interfaces.
Baseline and Agent JSONL files plus metadata and comparison JSON files were saved separately.

## Metric Results

| Dataset and scope | Baseline result | Agent result | Equal predictions |
| --- | --- | --- | --- |
| VRSBench, 6 records | 6/6 completed; VQA 2/2; diagnostic grounding mean IoU 0.4698 and IoU@0.5 accuracy 1/2 | 6/6 completed | Yes |
| MME-RealWorld Lite RS, 2 records | 2/2 completed; exact-match accuracy 1/2 | 2/2 completed; exact-match accuracy 1/2 | Yes |
| XLRS-Bench Lite VQA, 2 records | 2/2 completed; exact-match accuracy 1/2 | 2/2 completed; exact-match accuracy 1/2 | Yes |
| LEVIR-CC test, 2 records | 2/2 completed; both manually marked as change-judgment failures | 2/2 completed | Yes |

The VRSBench grounding diagnostic divided 0-1000-like Qwen outputs by 10 only in the separate
smoke comparison report. Raw predictions were preserved. This diagnostic is not an official
VRSBench score and did not change repository evaluation code. Both VRSBench captions reached
the 256-token generation limit and ended mid-sentence.

## Resource Consumption

| Dataset and scope | Baseline time | Agent time | Peak GPU memory |
| --- | ---: | ---: | ---: |
| VRSBench, 6 records | 39.40 s | 39.16 s | 8.42 GB |
| MME-RealWorld Lite RS, 2 records | 3.61 s | 3.62 s | 8.90 GB |
| XLRS-Bench Lite VQA, 2 records | 6.84 s | 6.18 s | 8.94 GB |
| LEVIR-CC test, 2 records | 12.08 s | 11.98 s | 8.33 GB |

Runtime environment: Tesla T4, PyTorch 2.11.0+cu128. Timings cover model calls and the
documented per-dataset loops, not dataset download time.

On 2026-07-20, an additional server smoke check loaded the same original checkpoint from a
local ModelScope snapshot on NVIDIA GB10 with PyTorch 2.13.0+cu130. Direct baseline inference
completed LEVIR-CC test sample `8148` and wrote its canonical prediction and metadata. The
prediction again described an apparent land-cover change even though the official references
describe no change. This one-record server check verifies model loading, GPU inference, dataset
loading, and result persistence only; it is not a benchmark score and has no Agent comparison.

## LEVIR-CC Full Server Run

The complete official LEVIR-CC test split was subsequently evaluated on NVIDIA GB10 with
PyTorch 2.13.0+cu130 and the same original `Qwen/Qwen3-VL-4B-Instruct` checkpoint loaded from
a local ModelScope snapshot. The server configuration used `max_new_tokens=256` and
`max_pixels=1003520`. No training, LoRA, quantization, or weight modification was performed.

| Run | Completed | Failed | Model-loop time | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Direct baseline | 1929/1929 | 0 | 11393.50 s | 8.36 GB |
| LangGraph Agent | 1929/1929 | 0 | 11186.07 s | 8.36 GB |

All 1929 sample IDs appeared in the same order and all canonical prediction objects were
identical between the two runs. The small timing difference is treated as runtime variation,
not evidence that the Agent accelerates inference.

| Metric | Baseline | Agent |
| --- | ---: | ---: |
| BLEU-1 | 0.0818626893 | 0.0818626893 |
| BLEU-2 | 0.0415929507 | 0.0415929507 |
| BLEU-3 | 0.0154370036 | 0.0154370036 |
| BLEU-4 | 0.0066851443 | 0.0066851443 |
| METEOR | 0.0994615283 | 0.0994615283 |
| ROUGE-L | 0.1020866062 | 0.1020866062 |
| CIDEr | 0.0000207700 | 0.0000207700 |

These are reproducible repository `pycocoevalcap` metrics, not an upstream leaderboard
submission. Generated text contained line breaks, so evaluation folded line breaks and repeated
whitespace only in temporary metric input to avoid breaking METEOR's line protocol. Raw JSONL
predictions and references were preserved. Generated token length was about eight times the
reference token length, and manual smoke review showed false change descriptions on official
no-change examples.

The archived result bundle is named `M3_levir_cc_full_bundle_2026-07-21.tar.gz`; its SHA256 is
`c9f1a9c8992d00606c9b472146e3616c3cf06193aaa9056c41772dd6e469e9f1`. The bundle is retained
outside Git because it contains generated results and logs.

## Conclusion

The original Qwen3-VL baseline and the fixed LangGraph workflow both completed every selected
smoke sample without program failures and produced identical predictions. The same conclusion
was verified across the complete 1929-record LEVIR-CC test split. The results verify the
first-stage baseline and Agent system paths while retaining concrete model failure examples for
later prompt, preprocessing, or model research.

## Reproducibility Statement

Every result group records sample IDs, official source scope, model ID, GPU, elapsed time, peak
GPU memory, raw predictions, references, failures, and whether Agent predictions equal baseline
predictions. Result bundles and downloaded datasets remain outside Git and are not committed.
The full LEVIR-CC result bundle was copied from Spark to local storage and its SHA256 matched.
Full runs for VRSBench, MME-RealWorld, and XLRS-Bench, plus upstream leaderboard submissions,
remain unvalidated follow-up work.
