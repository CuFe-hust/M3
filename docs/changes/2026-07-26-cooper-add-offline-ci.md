# Modification Note: Add Offline CI Gate - 2026-07-26 15:51:59 +08:00

## Modification Time

2026-07-26 15:51:59 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Add a repeatable GitHub Actions gate for the package's offline regression suite.

## Modified Files

- `.github/workflows/offline-tests.yml`
- `spacers_agent/workflow.py`
- `tests/compat/test_legacy_exports.py`
- `DETAILS.md`
- `docs/experiments/2026-07-26-new-agent-runtime-dataset-regression.md`

## Core Changes

- Added an Ubuntu/Python 3.11 workflow for pushes and pull requests.
- The workflow installs only the existing `dev` extra, compiles `spacers_agent`, runs the full pytest suite, and checks CLI help.
- It does not install the optional `yolo` extra, access a model service, or access a dataset.
- Removed one trailing blank line so `git diff --check` remains a strict gate.
- Added a compatibility test group so the required `tests/compat` command checks real legacy-to-new implementation aliases.
- Clarified that the existing regression record is fixture-based and does not represent an unexecuted real-dataset run.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No evaluation metric, split, reference-answer reader, or result post-processing rule changed.

## Whether Deployment Was Affected

No. YOLO, weights, exports, and device-specific runtimes are not installed or invoked.

## Whether pytest Was Updated

Yes. The legacy export aliases are covered by the required compatibility test group.

## Whether .gitignore Was Updated

No. The workflow produces no repository artifacts.

## Validation Method

- `git diff --check`
- `python -m compileall spacers_agent`
- Focused and full offline pytest gates using the repository Python 3.11 environment
- CLI offline commands

## Risks and Follow-up TODOs

- The local Python 3.11 environment reports unrelated pre-existing `pip check` platform/constraint issues; CI installation validation must be observed in GitHub Actions.
- Real Qwen/DeepSeek and real-dataset regression require the separately authorized local runtime, fixed sample IDs/configuration, and the Phase 1 baseline artifacts; they are not executed by this offline workflow.
