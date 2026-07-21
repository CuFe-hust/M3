# DETAILS.md

This file records the project structure, directory responsibilities, interface contracts, and notes that are currently still valid.

This file is intended primarily for AI coding agents. `AGENTS.md` defines behavioral boundaries and mandatory rules, while `DETAILS.md` records the current facts of the project. Before beginning modifications, the AI must read both files.

`DETAILS.md` is not a development log. Historical modification records are placed in `docs/changes/`, experiment records are placed in `docs/experiments/`, and temporary research or troubleshooting records are placed in `docs/notes/`.

## 1. Recommended Project Structure and Directory Responsibilities

This repository is recommended to use a lightweight layered structure, but the project does not require all directories to be created at once during the initial stage.
The directory structure is a weak constraint, while the canonical sample format, evaluation rules, model interfaces, configuration rules, `.gitignore`, tests, and documentation maintenance are strong constraints.

The AI agent must not create stub files, empty classes, empty directories, meaningless wrappers, or excessive abstraction layers merely to match the recommended structure. A directory should be created or expanded only when it already has a clearly defined responsibility and actual code.

### 2.1 Project Structure

Minimum project skeleton:

```text
AGENTS.md
CLAUDE.md
README.md
DETAILS.md
main.py
requirements.txt
.gitignore
config/
scripts/
data/
models/
eval/
tests/
docs/
```

### 2.2 Add Later as Needed

Do not create the following directories in advance when there is no actual need:

```text
router/    Create it when task types increase and unified dispatch is needed
agents/    Create it when task workflows such as caption/vqa/counting/detection clearly diverge
deploy/    Create it when model export, quantization, pruning, ONNX, or adaptation for domestic chips begins
utils/     Create it when the same utility function is reused by two or more modules
```

Before creating these directories, the AI agent must confirm:

1. Whether a simpler implementation location already exists;
2. Whether there are truly two or more callers or two or more task workflows that require reuse;
3. Whether it would add meaningless layers of indirection;
4. Whether it would cause current functionality to be split apart merely for the sake of directory structure;
5. Whether `DETAILS.md` and `docs/changes/` need to be updated accordingly.

### 2.3 Recommended Mature-Stage Structure

When project functionality gradually stabilizes, task types increase, and deployment work begins, the project can evolve into the following structure:

```text
AGENTS.md
CLAUDE.md
README.md
DETAILS.md
main.py
requirements.txt
.gitignore

config/
  default.yaml
  train.yaml
  infer.yaml
  eval.yaml
  deploy.yaml
  local.example.yaml

router/
  task_router.py
  registry.py

agents/
  base_agent.py
  caption_agent.py
  vqa_agent.py
  counting_agent.py
  detection_agent.py
  classification_agent.py
  segmentation_agent.py
  change_detection_agent.py

models/
  base_model.py
  model_loader.py
  processor.py
  adapters/
  lora/
  quantization/

data/
  schema.py
  validators.py
  datasets/
  transforms/
  collators/

eval/
  metrics/
  evaluators/
  run_eval.py

deploy/
  export.py
  quantize.py
  benchmark.py
  ascend/
  onnx/

utils/
  io.py
  logging.py
  seed.py
  device.py

tests/
  test_schema.py
  test_router.py
  test_data_loading.py
  test_inference_contract.py

scripts/
  check.sh
  train.sh
  infer.sh
  eval.sh
  export.sh

docs/
  changes/
  experiments/
  notes/
```

The actual repository does not need to match this exactly. When modifying code, the AI should prioritize module boundaries and canonical interfaces rather than mechanically pursuing an exact directory match.

### 2.4 Directory Evolution Rules

Directory evolution must comply with the following:

- New directories must have clearly defined responsibilities;
- New directories must not contain only empty classes, empty functions, or meaningless `pass` statements;
- Do not split a simple script into multiple layers of calls merely to make the “architecture look complete”;
- Do not create abstractions such as adapters, registries, factories, or managers in advance merely because they “might be useful in the future”;
- When a function is used in only one place, prefer leaving it in the current module rather than prematurely moving it into `utils/`;
- When multiple task workflows do not yet have clear differences, prefer keeping a simple entry point rather than forcibly splitting them into multiple agents;
- When device-side export or quantization has not yet begun, do not create a complex `deploy/` subsystem in advance;
- Any directory structure change must be recorded in `docs/changes/`, and `DETAILS.md` must be updated when necessary.

## 2. Directory Responsibilities

This section explains the responsibilities that directories should assume once they exist.
If a directory has not yet been created, the AI agent must not proactively create it merely because it is mentioned in this section; it should first determine whether the current task genuinely requires that directory.

### 3.1 `main.py`

`main.py` is the main entry point of the project.

Allowed:

- Parse command-line arguments;
- Load configurations;
- Call the router;
- Start training, inference, evaluation, or deployment workflows.

Prohibited:

- Writing specific model structures in `main.py`;
- Writing specific dataset parsing logic in `main.py`;
- Writing evaluation metrics in `main.py`;
- Writing quantization and chip-adaptation details in `main.py`.

### 3.2 `config/`

`config/` is used to store configuration files.

Content that must be placed in configuration includes:

- Model name;
- Weight path;
- Dataset path;
- Task type;
- Batch size;
- Learning rate;
- Image size;
- Max tokens;
- LoRA parameters;
- Quantization parameters;
- Export parameters;
- Output directory.

Prohibited:

- Hard-coding local absolute paths in code;
- Uploading `local.yaml` files containing personal local paths, accounts, tokens, or keys;
- Changing the meanings of existing configuration fields without authorization;
- Directly modifying default configurations for temporary experiments without explanation.

### 3.3 `router/`

`router/` is responsible only for task dispatch. This directory should be created only when task types increase and unified dispatch is genuinely needed.

Allowed:

- Selecting the corresponding agent according to `task_type`;
- Maintaining mappings from task names to agents;
- Performing lightweight parameter checks.

Prohibited:

- Writing model `forward` logic;
- Directly reading specific dataset files;
- Writing evaluation metrics;
- Writing deployment export logic;
- Writing large numbers of special branches for a particular task in order to bypass the agent.

### 3.4 `agents/`

`agents/` is responsible for orchestrating specific task workflows. This directory should be created only when different task workflows clearly diverge.

Allowed:

- Receiving canonical samples;
- Calling `models/`;
- Organizing inference results;
- Outputting the canonical prediction format.

Prohibited:

- Modifying the model structure without authorization;
- Modifying the processor/tokenizer without authorization;
- Directly modifying evaluation metrics;
- Directly modifying original dataset annotations;
- Writing large amounts of dataset-specific parsing logic inside an agent.

### 3.5 `models/`

`models/` is responsible for model loading, model wrapping, and the canonical inference interface.

Allowed:

- Loading the base multimodal model;
- Loading LoRA, adapters, and projectors;
- Managing the processor, tokenizer, and image processor;
- Providing canonical `generate` / `forward` interfaces;
- Loading and saving lightweight modules required by the project.

Prohibited:

- Directly reading specific dataset JSON files;
- Writing evaluation metrics;
- Writing the main training workflow;
- Replacing the base model without authorization;
- Upgrading model dependencies without authorization;
- Downloading a new model from the network as the default model without authorization;
- Breaking compatibility with existing checkpoints.

### 3.6 `data/`

`data/` is responsible for dataset reading, preprocessing, conversion to the canonical sample format, and validation.

Allowed:

- Reading data such as VRSBench, MME Real RS, XLRS-bench, and LEVIR-CC;
- Converting different datasets into the canonical sample format;
- Image preprocessing;
- Prompt construction;
- Collators;
- Schema validation.

Prohibited:

- Writing model structures;
- Writing evaluation metrics;
- Writing training loops;
- Hard-coding local dataset paths;
- Returning temporary dictionaries incompatible with the canonical sample format.

### 3.7 `eval/`

`eval/` is a high-risk area responsible for evaluation workflows and metrics.

Allowed:

- Calling inference or agents;
- Calculating metrics;
- Saving results;
- Counting failed samples;
- Generating reproducible experiment outputs.

Prohibited:

- Modifying metric definitions without an explicit requirement;
- Filtering failed samples to improve scores;
- Modifying reference answers;
- Modifying dataset splits;
- Modifying result post-processing rules in ways that make historical results incomparable;
- Changing output formats without documentation.

### 3.8 `deploy/`

`deploy/` is responsible for model export, quantization, pruning, adaptation for domestic AI chips, and performance statistics. This directory should be created only when deployment, export, quantization, or device-side adaptation actually begins.

Allowed:

- ONNX export;
- Ascend-related export;
- Adaptation for other domestic AI chips;
- INT8 / INT4 / FP16 quantization;
- Statistics for inference time, GPU memory, and model size;
- Device-side benchmarking.

Prohibited:

- Breaking the original PyTorch inference path;
- Directly modifying training data;
- Directly modifying evaluation metrics;
- Temporarily hacking model outputs for deployment without preserving a comparison path.

### 3.9 `tests/`

`tests/` is used for pytest tests.

After modifying code, if the current change affects any of the following, pytest must be added or updated:

- Canonical sample format;
- Dataset reading;
- Router dispatch;
- Agent inputs and outputs;
- Model wrapper interfaces;
- Evaluation metrics;
- Deployment export parameters;
- Configuration parsing;
- Bug fixes.

If pytest cannot be run in the current environment, the reason must be stated, and at least feasible static checks must be performed.

## 3. Current Core Interface Contracts

This section records the core interfaces that are currently still valid. When modifying code, the AI should use this file as the source of current facts while also complying with the mandatory behavioral rules in `AGENTS.md`.

### 3.1 Canonical Sample Format

All dataset-reading modules, agents, routers, inference, and evaluation components must use the canonical sample format when passing samples between them.

### 3.2 Canonical Prediction Format

The output of a model or agent should be converted into the canonical prediction format whenever possible.

The recommended structure is similar to the following, **which is only an illustration here and is subject to the actual development situation**:

```json
{
  "id": "sample_id",
  "task_type": "vqa",
  "text": "There are 3 ships in the image.",
  "answer": "3",
  "boxes": [],
  "masks": [],
  "labels": [],
  "scores": [],
  "count": null,
  "meta": {}
}
```

Rules:

- `id` must correspond to the input sample;
- `task_type` must remain consistent;
- Textual outputs should be placed in `text` or `answer`;
- Detection boxes should be placed in `boxes`;
- Segmentation results should be placed in `masks`;
- Classification labels should be placed in `labels`;
- Confidence scores should be placed in `scores`;
- Counting results should be placed in `count`;
- Additional information should be placed in `meta`.

Do not allow `eval/` to depend directly on a model’s private output structure. If the model outputs a special structure, it must be converted into the canonical prediction format at the agent or inference layer.

## 4. Configuration and Local File Conventions

All adjustable parameters should preferably be placed in `config/` or command-line arguments, including:

- Model name;
- Weight path;
- Dataset path;
- Task type;
- Batch size;
- Learning rate;
- Image size;
- Max tokens;
- LoRA parameters;
- Quantization parameters;
- Export parameters;
- Output directory.

Do not hard-code local absolute paths in code. Real local configurations should be ignored through `.gitignore`, and only example files such as `local.example.yaml` should be committed to the repository.

---

## 5. Current Known Notes

- The project does not require all recommended directories to be created at once during the initial stage.
- Do not create empty classes, empty functions, empty wrappers, empty managers, or empty factories merely to make the directory structure complete.
- `tests/` is the pytest test directory; do not use `pytest/` as the directory name.
- After adding output directories, cache directories, weight formats, data directories, or local configuration files, check `.gitignore` accordingly.
- If the structures, interfaces, or conventions in this file are modified, a corresponding modification note must also be added in `docs/changes/`.

## 6. Current Qwen3-VL-4B Baseline Interface

The currently implemented zero-shot baseline entry point is `main.py`; its model wrapper is
located in `models/qwen3vl.py` and uses the original `Qwen/Qwen3-VL-4B-Instruct` checkpoint.
It accepts a JSON configuration file
with `model` settings and external `paths.data_root` / `paths.output_root` values. It does
not include model fine-tuning, LoRA loading, quantization, or any server-transfer logic.
`config/baseline.example.json` uses project-relative `datasets/baseline` and `outputs/baseline`
paths. Users copy it to the ignored `config/local.baseline.json` before running Colab commands.

`data/schema.py` defines `CanonicalSample` and `CanonicalPrediction`. All implemented
download adapters, Qwen3-VL inference, persisted JSONL records, and local metrics pass data
through these two structures. A persisted result line contains the serialized source sample
and the corresponding canonical prediction, allowing the original answers to be read without
placing dataset-specific logic in `eval/`.

Supported evaluation targets are `vrsbench_caption`, `vrsbench_vqa`,
`vrsbench_grounding`, `mme_real_rs`, `xlrs_caption_en`, `xlrs_grounding_en`,
`xlrs_vqa_lite`, and `levir_cc`. The MME target filters to the Remote Sensing subdomain.
XLRS-Bench full English captioning and grounding releases are reported separately from the
official Lite VQA release; these scores must not be labelled as a single full-XLRS score.

The VRSBench validation adapter accepts the official `image_id`, `ground_truth`, and
`question_id` fields for captioning, VQA, and grounding annotations. `question_id` takes
precedence over an image identifier when both are present so question-level records remain
uniquely addressable. This compatibility handling does not modify official reference values.

The `evaluate --deepseek-proxy` command reads `DEEPSEEK_API_KEY` only from the runtime
environment and produces a non-official `deepseek_semantic_match_proxy` for VRSBench VQA.
It must not be described as the benchmark's official GPT-based evaluation metric.

Before caption and change-caption records are passed to `pycocoevalcap`, line breaks and
repeated whitespace are folded to single spaces and the METEOR protocol separator `|||` is
removed. Persisted canonical predictions and references remain unchanged. This prevents model
formatting from being interpreted as commands by METEOR's line-oriented Java subprocess.

The `infer` command runs the direct baseline. The `agent-infer` command runs the same model
through the fixed LangGraph nodes `read_sample -> call_qwen -> validate_prediction ->
save_result`; it does not train, replace, or modify the model. Baseline records use
`<dataset>.jsonl`, while Agent records use `<dataset>.agent.jsonl`.

Both inference commands save one canonical JSONL record at a time and flush it immediately.
`--resume` appends to an existing result and skips sample IDs already present; `--overwrite`
deletes that target's previous result, failure, and metadata files before a new run. The two
flags are mutually exclusive. Per-sample exceptions are written to the corresponding
`*.failures.jsonl` file and counted in metadata instead of being silently discarded. Metadata
is written atomically every ten attempts and at normal completion, including elapsed time and
peak allocated CUDA memory. A malformed existing result causes resume to fail visibly rather
than guessing which samples completed.
