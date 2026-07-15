# Modification Note: Add Qwen3-VL-4B Zero-Shot Baseline - 2026-07-15 21:54:14 +08

## Modification Time

2026-07-15 21:54:14 +08

## Modifier

tRoy

## Modification Goal

Provide a Colab-ready, reproducible zero-shot Qwen3-VL-4B baseline for the four competition
dataset families without changing model weights, dataset references, or official data splits.

## Modified Files

- `.gitignore`
- `README.md`
- `DETAILS.md`
- `requirements.txt`
- `config/baseline.example.json`
- `data/__init__.py`
- `data/schema.py`
- `data/loaders.py`
- `models/__init__.py`
- `models/qwen3vl.py`
- `eval/__init__.py`
- `eval/metrics.py`
- `main/run_baseline.py`
- `tests/test_schema.py`
- `tests/test_baseline_adapters.py`
- `docs/experiments/baseline-qwen3-vl-4b.md`
- `docs/changes/2026-07-15-troy-add-qwen3-vl-baseline.md`

## Core Changes

Added official Hugging Face download targets, canonical sample/prediction records, Qwen3-VL
inference, JSONL output persistence, deterministic local metrics, and an opt-in DeepSeek VQA
semantic proxy that reads its key only from `DEEPSEEK_API_KEY`. Added Colab commands and a
pre-experiment protocol. XLRS full English Caption/Grounding and Lite VQA have separate targets
and metadata scope notes. MME Real RS additionally writes the upstream evaluator's JSON shape,
with only its `Output` field changed to the generated answer.

## Whether the Canonical Sample Format Was Changed

No pre-existing canonical format was changed. The initial concrete implementation follows the
recommended `id`, `task_type`, text/answer, boxes, masks, labels, scores, count, and meta contract.

## Whether the Model Interface Was Changed

No existing model interface was changed. Added a new wrapper for the original
`Qwen/Qwen3-VL-4B-Instruct` checkpoint only.

## Whether the Configuration Was Changed

Yes. Added `config/baseline.example.json` for model runtime settings and external Colab paths.
No local credentials or absolute user paths are tracked.

## Whether Evaluation Was Affected

Added new baseline evaluation paths. Existing metric code and reference-answer reading logic did
not exist and were not replaced. Oriented grounding metrics remain delegated to upstream official
evaluators. The DeepSeek result is explicitly non-official and separate from VRSBench GPT metrics.

## Whether Deployment Was Affected

No. The implementation targets server-side Colab inference and does not export or quantize models.

## Whether pytest Was Updated

Yes. Added schema, VRS VQA adapter, choice/box parser, and exact-match tests.

## Whether .gitignore Was Updated

Yes. Added dataset caches, model caches, output, local JSON configuration, and Firecrawl artifact patterns.

## Validation Method

- Ran `python3 -m compileall data models eval main tests` successfully.
- Attempted `pytest -q`; it could not run because `pytest` is not installed in the local environment.
- Did not download datasets or model weights locally because execution is explicitly intended for Colab.

## Risks and Follow-up TODOs

- Run each `inspect` command in Colab after download to verify exact released JSON fields and image paths.
- Record Colab GPU, memory, storage, and latency before claiming resource efficiency.
- Use the upstream VRSBench and XLRS-Bench code for official oriented-box grounding metrics.
- DeepSeek proxy calls incur external API cost and are not directly comparable with official GPT evaluation.
