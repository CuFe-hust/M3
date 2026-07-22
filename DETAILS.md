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
The optional `model.local_files_only` setting is `false` by default. Set it to `true` together
with an external local `model.id` path to prohibit network fallback during server inference;
the model, processor, tokenizer, checkpoint contents, and prediction contract remain unchanged.
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
The VRSBench VQA adapter accepts the official evaluation release fields `image_id`,
`question`, `ground_truth`, and `question_id` in addition to its existing nested QA-pair form;
the question identifier remains the canonical sample ID when several questions share one image.
XLRS-Bench full English captioning and grounding releases are reported separately from the
official Lite VQA release; these scores must not be labelled as a single full-XLRS score.

The `evaluate --deepseek-proxy` command reads `DEEPSEEK_API_KEY` only from the runtime
environment and produces a non-official `deepseek_semantic_match_proxy` for VRSBench VQA.
It must not be described as the benchmark's official GPT-based evaluation metric.

`eval/audit_report.py` is the baseline visual-report component. Baseline inference enables it by
default and writes `<result-stem>.report/report.html` plus `samples.csv`, a bounded visual
`samples.jsonl`, and content-addressed PNG files without changing the canonical prediction file.
`report.enabled` defaults to `true`; `report.max_samples` defaults to `200` and limits only the
number of displayed visual records, not inference or metric scope. Each inference and evaluation
command prints the absolute report path.

When `evaluate --deepseek-proxy` runs, `eval.metrics.evaluate_records` retains its existing metric
return format and may additionally receive a caller-owned audit list. The baseline CLI writes that
list to `<result-stem>.report/deepseek_audit.jsonl` with request payload, raw response, parsed
result, duration, attempts, token usage, and errors. It never stores the API key. The report is
then regenerated with per-sample Qwen/reference comparison and DeepSeek judgment. DeepSeek remains
text-only and never claims visual verification.

## 7. Phase 1 Local Multi-Agent Foundation

`spacers_agent/` is an additive local package that does not replace the existing `main.py`
baseline. `python -m spacers_agent.cli run-init` loads `configs/default.yaml`, applies
non-secret environment overrides from `.env` or the process environment, and creates a run
directory with `manifest.json`, `config.snapshot.yaml`, `prompts.snapshot/`, and
`events.jsonl`. API-key values are intentionally not read into the settings model or written
to these artifacts.

`spacers_agent.settings.AppSettings` validates the phase-one Qwen, DeepSeek, counting, run,
and path settings. `EnvironmentOverrides` uses `pydantic-settings` to read only non-secret
endpoint and path overrides from `.env` or the process environment. API-key values remain
environment-only and are not placed in the settings model or run artifacts.

The Phase 1 `health` command only prints configured endpoint metadata and explicitly performs
no network check. Live endpoint health checks begin no earlier than Phase 2 and still require
explicit user authorization.

## 8. Phase 2 Structured Client Interfaces

`spacers_agent.clients.VisionLanguageClient` defines async `complete_json` requests shared by
the future Qwen client and the offline `MockVisionClient`. `QwenVLLMClient` is an
OpenAI-compatible implementation that is inert until a caller invokes it with a key supplied
through `QWEN_API_KEY` (or the configured environment-variable name). It supports bounded
429/5xx/timeout retries, JSON fence normalization, one JSON repair request, Pydantic
validation, a request-hash cache, and artifact persistence for raw responses, parsed JSON,
validation errors, latency, and available token counters.

`RequestMeta` and persisted request metadata exclude API keys and Base64 image data. Data URLs
may be sent only by a live caller; `sanitize_messages` replaces them with a hash and encoded
byte count before logging or hashing. No live client call has been made locally.

## 9. Phase 3 Unified Schema, Geometry, and Dataset Audit

`spacers_agent.schemas` adds Pydantic contracts for `UnifiedSample`, `ImageRef`,
`GroundTruth`, half-open `PixelRect`, `TileSpec`, local/global point observations, and
`CountingResult`. This is additive: the existing baseline `CanonicalSample` and
`CanonicalPrediction` remain unchanged. Change samples require `t1` before `t2`; count results
enforce `final_count == accepted point count` and cannot report completion if tiles failed.

`spacers_agent.imaging` contains model-independent EXIF/RGB normalization, row-major owner-core
plus halo planning, no-upscale resizing, `0..999` coordinate conversion, strict owner-core
acceptance, boundary-candidate detection, and clamp provenance. Tile crops are transient in
memory; no adapter writes permanent tiles to datasets.

`python -m spacers_agent.cli inspect-data --root <dataset-root> --output <report.json>` performs
a read-only generic layout audit and writes a separate JSON report. It reports file extensions,
candidate manifests, image samples/damage, discovered JSON fields, split hints, duplicate IDs,
encoding failures, and missing referenced images. It does not infer a dataset-specific Adapter
mapping; that remains blocked until the named local datasets are available for audit.

## 10. Phase 4 Point Counting Orchestration

`spacers_agent.counting` is an additive module; it does not alter the baseline `main.py`, the
existing canonical sample/prediction format, dataset splits, or evaluation metrics. Its
`PointCountingOrchestrator` accepts a normalized image, a stable `CountTargetSpec`, and an
injected `VisionLanguageClient`. It processes tiles row-major and sequentially, with one image per
request. `TileCheckpointStore` writes `spec.json`, `parsed.json`, conversion validation, and a
status checkpoint under `tiles/<tile_id>/`; a matching successful request hash is reused during
resume, while a failed tile remains explicitly visible.

`CountTargetSpec` is shared by all tiles of a sample and is included in request-cache inputs.
`count_tile_v2.md` defines the owner-core-only point protocol. `TileCountResponse` is additionally
checked against both the requested tile ID and the target label before coordinate conversion.
Parents that request a split are stored as `superseded_by_children`; their four child owner cores
are processed sequentially and replace the parent points. The final count is calculated only from
accepted representatives. Boundary conflict discovery considers only matching-label points near
neighbouring owner-core seams; it records candidates rather than applying a global clustering
algorithm. Explicit same-instance decisions may be supplied to `finalize_representatives`; absent
such a decision, the fixed current policy is to retain and flag the conflict for review.

No live seam-verifier call, target-spec LLM parsing call, or missing-point review call is wired into
the default path. These require a separately authorized live client and must preserve the same
point/owner validation rules.

## 11. Phase 5 Sparse Multi-Agent Routing and Prompt Assets

`spacers_agent.routing.ROUTES` defines fixed expert routes for all declared multi-Agent tasks.
`TaskRouter.route_known` is deterministic and makes no model call. `route_unknown` is explicitly
separate, accepts only an injected client, uses a text-only request, records discrete reason codes,
and consumes a Qwen entry from `CallBudget` before it can call the router. The budget is mutable,
validated against its configured limits, and may be attached to `PointCountingOrchestrator` so tile,
recursive, seam, or review calls cannot exceed the same Qwen limit.

`CountingExpert` delegates to the existing `PointCountingOrchestrator`; it contains no duplicate
geometry or counting logic. Its completed answer is derived from the accepted global-point set. If
any tile failed, it reports the completed-tile fraction and confirmed points and marks the answer
non-final. `CallBudget` also exposes a separate DeepSeek reservation method. No default route calls
DeepSeek, and any future judge must receive only text and structured evidence rather than image data.

The versioned prompt assets are `router_v1.md`, `target_parse_v1.md`, `count_tile_v2.md`,
`count_repair_v1.md`, `seam_verify_v1.md`, `missing_point_review_v1.md`, `change_v1.md`,
`spatial_v1.md`, and `general_vqa_v1.md`. `run-init` snapshots each asset. Prompts are not changed
in place when their behavior changes; a new versioned file must be added and selected explicitly.

## 12. Phase 6 DeepSeek Structured Evaluator

`spacers_agent.evaluation` computes deterministic counting metrics before and independently from
LLM evaluation. With known count truth, it records exact match, absolute error, relative error, and
the explicitly named `smooth_error_score` (which is not an accuracy metric). Its compact judge
payload contains question text, target rules, display answer, count consistency, tile/conflict
statistics, ground truth, and deterministic metrics; it deliberately excludes source images, image
paths, Base64 values, and the full point list.

`DeepSeekJudgeResult` hard-codes `judge_scope="text_and_structured_evidence_only"` and
`can_verify_visual_truth=false`. `merge_count_evaluation` preserves the raw judge response and flags
`judge_inconsistency` when a `correct` verdict conflicts with a known count mismatch. It does not
rewrite the judge result. Samples without ground truth retain judge feedback only as internal quality
evidence, separate from benchmark metrics.

`spacers_agent.clients.DeepSeekJudgeClient` uses OpenAI Chat Completions JSON mode, reads its key
only from the configured `DEEPSEEK_API_KEY` environment variable, retries transient and empty-content
responses within `max_retries`, permits one JSON-format repair, and persists raw/parsed/validation,
latency, retry, and token records. Its request hash includes the model, judge-prompt hash, sample ID,
prediction hash, ground-truth hash, and deterministic-metrics hash. No live call is automatic.

## 13. Phase 7 Offline Acceptance and Runbook

`spacers_agent.visualization.render_counting_overlay` creates an explicit local artifact from a
normalized source image, `CountingResult`, and rebuilt tile geometry. It renders blue owner-core
grids, green accepted points, and red rejected points. It is a rendering tool only: it neither
changes counting results nor calls a model. `spacers_agent.cli render-count` validates source/result
dimensions before generating this artifact.

`spacers_agent.reporting.summarize_evaluations` aggregates persisted `EvaluationRecord` values into
an `EvaluationSummary`. It reports deterministic exact-match/MAE/relative-error fields separately
from optional Judge success, failure, inconsistency, and semantic-quality fields. The CLI command
`summarize-evaluations` reads JSONL locally and writes one JSON summary. The complete local command
sequence and live-test safety boundary are recorded in `docs/runbook.md`.

## 14. Runnable Agent Workflow, Explicit Adapters, and Spark Handoff

`spacers_agent.workflow` adds atomic JSON artifacts, automatic structured target parsing, a sequential dataset runner, and reusable Qwen visual primitives for change, grounding, spatial, and general-VQA routes. Counting continues to derive `final_count` only from accepted global points; confidence gating sets low-confidence points to `accepted=False`, and seam merging occurs only after a `SeamDecision` explicitly returns `same_instance` for a local seam crop.

`spacers_agent.dataset_adapters` intentionally does not reuse the baseline heuristics. Its LEVIR-CC, VRSBench, MME-RealWorld, and XLRS-Bench-lite registry entries require a local version-1 `spacers_adapter.json` that declares the samples file and exact field mappings. `probe()` validates those mappings and reports observed fields before any sample runs. The source dataset is only read; no download or fallback inference occurs.

The new CLI preserves `main.py` and adds `health --live`, `smoke-qwen`, `count-image`, `run-dataset`, `resume-run`, and `evaluate-run`. `evaluate-run --deepseek` reads persisted counting results and can invoke the existing text-only judge without issuing Qwen inference. Spark-facing, environment-driven examples are in `scripts/server/`, including vLLM lifecycle, health, dataset/resume, and systemd files. They are not executed by local tests.
