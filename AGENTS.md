# AGENTS.md

This file defines the constraints governing how AI coding agents read, modify, validate, maintain documentation, and report results in this repository.

This repository corresponds to the project “Exploration of Multimodal Remote-Sensing Large Model Applications for Space-Based Intelligent Computing.” Its goal is to build a multimodal large-model system for remote-sensing image interpretation tasks. The project needs to support tasks including remote-sensing image captioning, remote-sensing visual question answering, object counting, object detection, scene classification, semantic segmentation, and change detection, while also addressing constrained computing resources, model lightweighting, inference efficiency, domestic AI chips, and adaptation for onboard/edge deployment.

This file is intended primarily for AI coding agents rather than ordinary human developers. Human-facing project documentation is in `README.md`, current interfaces and notes are in `DETAILS.md`, and records of each modification are in `docs/changes/`.

---

## 0. Highest-Priority Principles

When modifying this repository, prioritize ensuring that:

1. **The canonical sample format is not broken**.
2. **Existing training, inference, evaluation, and deployment entry points are not broken**.
3. **Evaluation metrics, dataset splits, and the method for reading reference answers are not changed without authorization**.
4. **The base model, processor, tokenizer, backbone, or weight-loading logic is not changed without authorization**.
5. **Only the minimum changes required to complete the current task are made**.
6. **No repository-wide formatting, renaming, directory migration, or large-scale refactoring is performed**.
7. **Datasets, model weights, checkpoints, caches, logs, and large files are not committed**.
8. **After modifications, update `DETAILS.md`, `docs/changes/`, `.gitignore`, and pytest tests as needed**.
9. **When adding or modifying code comments, English + Chinese bilingual comments must be used**.
10. **Do not create empty directories, empty classes, shell wrappers, or meaningless abstraction layers in advance merely to match a recommended structure**.
11. **When validation cannot be performed, this must be stated truthfully; never claim that tests not actually run have passed**.

Stability, reproducibility, and comparability of evaluation results take precedence over formal code cleanliness.

---

## 1. Scope and Instruction Priority

This file applies to the entire repository by default.

When a more specific `AGENTS.md` exists in a subdirectory:

- Rules closer to the target file take priority;
- However, they must not weaken the requirements in this file concerning data formats, evaluation, deployment, documentation maintenance, and minimal changes;
- Explicit user requirements in the current task take priority over this file;
- If a user requirement conflicts with project security, reproducibility, or data-leakage risks, first point out the risk, then provide a safer minimal alternative.

If the task description is unclear, do not guess and make large changes to the project. First explain the uncertainty and propose the smallest feasible modification scope.

---

## 2. Project Structure and Source of Current Interfaces

The project structure, directory responsibilities, current valid interfaces, and notes are maintained centrally in `DETAILS.md`.

Before beginning any modification, the AI coding agent must first read `DETAILS.md` and use the current project structure described there as the basis. `AGENTS.md` only defines behavioral boundaries and mandatory rules; it does not maintain the complete directory structure, in order to prevent directory rules from drifting across multiple files.

The directory structure is a weak constraint. The canonical sample format, evaluation rules, model interfaces, configuration rules, `.gitignore`, pytest, and documentation maintenance are strong constraints.

The AI agent must not create shell files, empty classes, empty directories, meaningless wrappers, or excessive abstraction layers merely to match a recommended structure in `DETAILS.md`. A directory should be created or expanded only when it already has a clearly defined responsibility and actual code.


## 3. Mandatory Rules for the Canonical Sample Format

The canonical sample format is the core contract of this project. The AI agent must not bypass, weaken, or arbitrarily extend this format.

All dataset-reading modules, agents, routers, inference, and evaluation components must use the canonical sample format when passing samples between them.

## 4. Mandatory Rules for the Canonical Prediction Format

See `DETAILS.md`.

## 5. Mandatory Modification Rules

### 5.1 Minimal-Change Principle

- Modify only the files and code necessary to complete the current task.
- Do not opportunistically fix unrelated issues.
- Do not perform repository-wide formatting, import sorting, batch renaming, or directory reorganization.
- If a problem can be solved with a local patch, do not rewrite the entire file.
- If an existing function can be reused, do not create a parallel implementation.
- Do not replace a stable implementation merely because the code “does not look elegant enough.”
- If the diff is clearly larger than the requirement itself, first reduce the scope of the solution.

### 5.2 Compatibility Rules

Unless the task explicitly requires a breaking change, preserve:

- Existing command-line entry points;
- Existing configuration key names;
- Existing default parameters;
- Existing model-loading interfaces;
- Existing dataset splits;
- Existing evaluation metrics;
- Existing output directory structure;
- Existing log semantics;
- Existing public function and class names;
- The existing canonical sample format.

If a breaking change is unavoidable, you must:

1. Explain the reason;
2. Update callers;
3. Update tests;
4. Update `DETAILS.md`;
5. Record it in `docs/changes/`;
6. Mark it as a high-risk change in the final result.

### 5.3 Encoding and Line Endings

- All text files must use UTF-8.
- Prefer preserving the target file’s existing line-ending style.
- Do not allow an editor’s automatic save behavior to change the entire file’s line endings or encoding.
- Files containing Chinese comments, Chinese prompts, or Chinese output text must not be rewritten in full.
- Do not batch-polish Chinese wording.
- Do not replace normal Chinese text with escape sequences or garbled text.

### 5.4 Bilingual Code Comment Rules

All newly added or modified code comments must use English + Chinese bilingual text.

Recommended format:

```python
# Validate the canonical remote-sensing sample schema.
# 校验统一遥感样本格式。
```

Or:

```python
def load_sample(path: str) -> dict:
    """Load one canonical remote-sensing sample.
    加载一条统一格式的遥感样本。
    """
```

Rules:

- English first, Chinese second;
- Comments should explain intent rather than restating the code itself;
- Do not add meaningless comments to every line merely to satisfy the bilingual requirement;
- When modifying an old comment, complete it in both languages;
- There is no requirement to batch-add bilingual text to old comments unrelated to the current task;
- Batch-translating comments throughout an entire file and thereby creating a huge diff is prohibited.

### 5.5 Dependency Rules

- Do not add third-party dependencies unless necessary.
- Prefer the standard library and dependencies already present in the repository.
- Do not upgrade PyTorch, Transformers, Datasets, Accelerate, PEFT, OpenCV, NumPy, ONNX, ONNX Runtime, MindSpore, Ascend-related dependencies without authorization.
- If a new dependency is unavoidable, update the dependency file and explain the reason.
- Optional dependencies should remain optional imports or have a clearly defined fallback path.
- Do not allow ordinary development environments lacking deployment hardware dependencies to crash during import.

---

## 6. Data, Weights, Caches, and `.gitignore` Rules

### 6.1 Content Prohibited from Being Committed

Do not commit the following to Git:

```text
datasets/
data/raw/
raw_data/
checkpoints/
weights/
outputs/
runs/
logs/
wandb/
tensorboard/
.cache/
huggingface/
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.onnx
*.om
*.engine
*.npy
*.npz
*.log
*.tar
*.zip
*.7z
```

Unless the user explicitly requests it and confirms that repository policy permits it, model weights, datasets, and large files must not be committed.

### 6.2 Check `.gitignore` After Modifications

After each modification, the AI agent must check whether new file types were generated or introduced.

If any of the following are added, update `.gitignore` as needed:

- New output directories;
- New checkpoint directories;
- New log directories;
- New cache directories;
- New dataset directories;
- New exported-model formats;
- New temporary-file formats;
- Local configuration files.

Example:

```gitignore
# Local configs
config/local.yaml

# Experiment outputs
outputs/
runs/

# Model artifacts
*.pt
*.pth
*.safetensors
*.onnx
*.om
```

Do not add large-model weights or datasets directly to the repository for the sake of “convenient reproducibility.”

---

## 7. Evaluation Rules

`eval/`, evaluation metrics, and reference-answer reading logic are high-risk areas.

Unless explicitly required by the task, do not modify:

- Metric calculation methods;
- Dataset splits;
- Reference-answer reading methods;
- Result post-processing rules;
- Evaluation output formats;
- Failure-sample handling methods.

Prohibited actions:

- Filtering failed samples in order to improve metrics;
- Silently skipping samples that cannot be processed;
- Modifying reference answers;
- Writing a model’s private output format directly into evaluation logic;
- Writing non-reproducible experimental logic into default evaluation scripts.

If evaluation logic must be modified, explain the following in `docs/changes/`:

1. Which metric was modified;
2. Why it was modified;
3. Whether results before and after the modification remain comparable;
4. Whether historical experiments need to be rerun;
5. Which tests were updated.

---

## 8. Deployment and Lightweighting Rules

This project targets constrained computing resources, onboard/edge devices, and deployment on domestic AI chips.

When working with `deploy/`, quantization, pruning, distillation, ONNX, Ascend, or adaptation for other domestic chips, comply with the following:

- Preserve the original PyTorch inference path;
- Export scripts must not break training code;
- Outputs from the quantized model and the original model must be comparable;
- Record model size, GPU memory usage, and inference time;
- Do not pursue accuracy alone while ignoring resource consumption;
- Do not temporarily remove model capabilities merely for deployment;
- Do not make hardware-specific dependencies globally mandatory dependencies;
- Do not claim that device-side validation has been completed in an environment without the required hardware.

If real device-side validation cannot be performed, explicitly write:

```text
No real domestic AI chip validation was performed; only the export script or static checks were completed.
```

---

## 9. Documentation Maintenance Rules

The documentation in this repository is mainly used by AI coding agents to continuously understand the project state. After modifying code, the AI agent must maintain documentation according to the following rules.

### 9.1 `DETAILS.md`

`DETAILS.md` records the currently valid project structure, directory responsibilities, interfaces, conventions, and notes. It is the sole maintenance location for project-structure information and is not a development log.

If the current modification changes any of the following, `DETAILS.md` must be updated accordingly:

- Project structure;
- Directory responsibilities;
- Module boundaries;
- Canonical sample format;
- Canonical prediction format;
- Router dispatch rules;
- Agent inputs and outputs;
- Model interfaces;
- Configuration fields;
- Run commands;
- Dataset path conventions;
- Evaluation rules;
- Deployment export rules;
- Known pitfalls and notes.

Do not write temporary ideas, abandoned solutions, invalid experiments, or chronological logs into `DETAILS.md`.

### 9.2 `docs/changes/`

After every actual change to code, configuration, scripts, or interfaces, a modification note must be added under `docs/changes/`.
Each modification note must record the exact local modification time, including hour and minute at minimum, and the modifier. Do not record only the date.

File naming format:

```text
docs/changes/YYYY-MM-DD-agent-short-task-name.md
```

Example:

```text
docs/changes/2026-07-10-agent-add-vrsbench-loader.md
```

Each modification note must contain:

```md
# Modification Note: Task Name - Modification Time

## Modification Time

YYYY-MM-DD HH:mm:ss TZ

## Modifier

Name, account, or AI agent identity that made the change.

## Modification Goal

## Modified Files

## Core Changes

## Whether the Canonical Sample Format Was Changed

## Whether the Model Interface Was Changed

## Whether the Configuration Was Changed

## Whether Evaluation Was Affected

## Whether Deployment Was Affected

## Whether pytest Was Updated

## Whether .gitignore Was Updated

## Validation Method

## Risks and Follow-up TODOs
```

If the change is only a documentation typo, comment-format adjustment, or small README edit and does not affect code behavior, a new `docs/changes/` file is not required. However, as long as code behavior, configuration behavior, interfaces, or tests are changed, a new file is mandatory.

### 9.3 `docs/experiments/`

Experiment records belong in `docs/experiments/`.

If the current modification involves training, evaluation, model-effect comparisons, quantization effects, or deployment performance, add or update an experiment record.

An experiment record must include at least:

```md
# Experiment Record: Experiment Name

## Time

## Dataset

## Model

## Configuration File

## Run Command

## Metric Results

## Resource Consumption

## Conclusion

## Reproducibility Statement
```

### 9.4 `README.md`

If the current modification changes any of the following, `README.md` must be updated accordingly:

- Installation method;
- Running method;
- Training commands;
- Inference commands;
- Evaluation commands;
- Deployment export commands;
- Data preparation method;
- Directory structure;
- Default configuration description.

### 9.5 Prohibition on Documentation Pollution

Do not write meaningless content merely to satisfy documentation requirements.

Do not write:

```text
The code structure was optimized in this change.
Maintainability was improved in this change.
Some modifications were made in this change.
```

You must clearly describe specific behavioral changes, specific files, and specific risks.

---

## 10. pytest and Validation Rules

### 10.1 When pytest Must Be Updated

If the current modification involves any of the following, pytest must be added or updated:

- Canonical sample format;
- Dataset adapters;
- Schema validators;
- Router dispatch;
- Agent inputs and outputs;
- Model wrapper interfaces;
- Inference output formats;
- Evaluation metrics;
- Configuration parsing;
- Bug fixes;
- Deployment export argument parsing.

Test files must be placed in:

```text
tests/
```

Naming format:

```text
tests/test_xxx.py
```

### 10.2 Default Check Commands

After modifications are complete, prefer running:

```bash
bash scripts/check.sh
```

If `scripts/check.sh` does not exist, run at least:

```bash
python -m compileall .
pytest -q
```

If the change scope is small, the relevant tests may be run:

```bash
pytest -q tests/test_schema.py
pytest -q tests/test_router.py
```

### 10.3 When Tests Cannot Be Run

If the current environment lacks dependencies, model weights, datasets, or hardware and complete tests cannot be run, explicitly state:

- Which command could not be executed;
- Why it could not be executed;
- Which alternative checks were completed;
- Which items require manual or later environment validation.

Do not describe “syntax checks passed” as “complete functional validation passed.”

---

## 11. Pre-Modification Workflow

Before beginning a modification, you must:

1. Read `AGENTS.md`;
2. Read `DETAILS.md`;
3. Read the complete context of code files relevant to the task;
4. Search for similar existing implementations;
5. Determine which module the current task belongs to;
6. Determine whether it affects the canonical sample format;
7. Determine whether it affects comparability of evaluation results;
8. Determine whether it affects model interfaces or weight compatibility;
9. Determine whether it affects deployment export;
10. Design the smallest possible patch.

If the task could trigger large-scale refactoring, first split it into multiple small modifications. Do not modify data, models, training, inference, evaluation, and deployment all at once.

---

## 12. Implementation Requirements During Modifications

### 12.1 Preserve the Existing Style

- Prefer reusing existing repository utility functions;
- Prefer preserving the existing import style;
- Do not add unrelated type annotations;
- Do not introduce large frameworks;
- Do not add complex abstraction layers;
- Do not turn simple functions into over-engineered class systems.

### 12.2 Error Handling

Do not use:

```python
except Exception:
    pass
```

If catching an exception is necessary:

- Record the necessary error information;
- Preserve debuggability;
- Do not swallow critical errors;
- Do not allow evaluation to silently skip large numbers of failed samples;
- Do not allow training to continue producing misleading results when data is corrupted.

### 12.3 Path Handling

- Do not hard-code local absolute paths;
- Do not hard-code paths for only one platform, whether Windows or Linux;
- Pass paths through configuration files or command-line arguments;
- Put example paths in `config/local.example.yaml` or the README;
- Do not commit real private paths.

### 12.4 Logging

- Do not expand logging to include sensitive paths, tokens, keys, or user privacy;
- Do not add large amounts of INFO logging inside high-frequency loops;
- Do not save complete images, arrays, point clouds, or large text by default;
- Debug logging must be possible to disable.

---

## 13. High-Risk Modification Areas

The following are high-risk modification areas:

```text
data/schema.py
data/validators.py
router/
models/
eval/
deploy/
config/
main.py
DETAILS.md
```

Be more conservative when modifying these areas:

- Make only local patches;
- Do not rewrite entire files;
- Do not opportunistically refactor;
- Do not change public interfaces;
- After modifications, check whether pytest needs to be updated;
- After modifications, check whether `DETAILS.md` needs to be updated;
- After modifications, write a record in `docs/changes/`.

Among these, `eval/`, the canonical sample format, and model-loading logic are the highest-risk areas.

---

## 14. Explicitly Prohibited Actions

AI coding agents must not:

- Rewrite an entire file for a small requirement;
- Batch-format the entire repository;
- Batch-modify line endings;
- Batch-translate comments;
- Batch-polish Chinese prompts;
- Modify the canonical sample format without authorization;
- Modify evaluation metrics without authorization;
- Modify dataset splits without authorization;
- Modify reference-answer reading without authorization;
- Replace the base model without authorization;
- Upgrade key dependencies without authorization;
- Download new weights as the default model without authorization;
- Make cloud API calls the default inference path;
- Commit datasets, weights, checkpoints, caches, or logs to Git;
- Remove protective logic merely to make tests pass;
- Claim “completely fixed” without validation;
- Modify code without updating required documentation, tests, or `.gitignore`;
- Create empty directories, empty classes, shell wrappers, or meaningless abstraction layers merely to match a recommended structure;
- Prematurely abstract into `utils/`, `router/`, or `agents/` when there is only one caller;
- Create a temporary data format parallel to the canonical sample format and spread it across multiple modules.

---

## 15. Mandatory Checklist After Every Modification

After every modification, the AI agent must check:

```text
1. Were only necessary files modified?
2. Was the canonical sample format changed?
3. Was the model interface changed?
4. Were configuration fields changed?
5. Were evaluation metrics changed?
6. Was deployment export affected?
7. Does pytest need to be updated?
8. Has the required pytest been updated?
9. Does .gitignore need to be updated?
10. Has .gitignore been updated?
11. Does DETAILS.md need to be updated?
12. Has DETAILS.md been updated?
13. Has a docs/changes/ modification note been added?
14. Were feasible check commands run?
15. Are there any unvalidated items that need to be stated?
```

If any item is incomplete, the reason must be stated in the final result.

---

## 16. Final Output Requirements

After completing the task, the result must include:

```text
Modification Summary:
Modified Files:
Whether the Canonical Sample Format Was Changed:
Whether the Model Interface Was Changed:
Whether the Configuration Was Changed:
Whether Evaluation Was Affected:
Whether Deployment Was Affected:
Whether pytest Was Updated:
Whether .gitignore Was Updated:
Whether DETAILS.md Was Updated:
Whether docs/changes/ Was Added:
Validation Commands and Results:
Unvalidated Items:
Risks and Follow-up TODOs:
```

Prohibited:

- Writing only “Completed”;
- Hiding the fact that tests were not run;
- Hiding interface changes;
- Hiding evaluation changes;
- Hiding newly added dependencies;
- Hiding generated-file or large-file risks.

---

## 17. Default Priorities

When conflicts occur, make decisions according to the following default priority order:

1. **Stability of the canonical sample format**
2. **Comparability of evaluation results**
3. **Model-interface and weight compatibility**
4. **No leakage of data, weights, or privacy**
5. **Correctness of the current task**
6. **Minimal changes and reversibility**
7. **Deployment adaptability**
8. **Test coverage and documentation synchronization**
9. **Code style and formal cleanliness**

Any “more elegant” solution should not be used as the default if it expands the modification scope, breaks format stability, affects evaluation comparability, or increases regression risk.
