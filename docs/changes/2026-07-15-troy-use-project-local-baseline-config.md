# Modification Note: Use Project-Local Baseline Configuration - 2026-07-15 22:51:41 +08

## Modification Time

2026-07-15 22:51:41 +08

## Modifier

troy

## Modification Goal

Keep the Colab baseline configuration and its default data/output paths inside the cloned project
directory instead of requiring a duplicate JSON file under `/content`.

## Modified Files

- `config/baseline.example.json`
- `README.md`
- `DETAILS.md`
- `docs/experiments/baseline-qwen3-vl-4b.md`

## Core Changes

- Changed the example paths to `datasets/baseline` and `outputs/baseline`.
- Documented copying the example to the Git-ignored `config/local.baseline.json`.
- Updated Colab commands to use the project-local configuration path.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

Yes. The default storage paths are now project-relative.

## Whether Evaluation Was Affected

No. Metrics, dataset splits, reference-answer reading, and result formats are unchanged.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. This change updates an example configuration and documentation only.

## Whether .gitignore Was Updated

No. `datasets/`, `outputs/`, and `config/local*.json` were already ignored.

## Validation Method

- Parse `config/baseline.example.json` with `python3 -m json.tool`.
- Check the updated documentation for obsolete `/content` baseline paths.
- Run `git diff --check`.

## Risks and Follow-up TODOs

The revised commands have not been executed in Colab. Users should ensure the cloned repository
is their current working directory before running the commands.
