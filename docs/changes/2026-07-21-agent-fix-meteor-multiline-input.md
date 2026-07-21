# Modification Note: Fix METEOR Multiline Input - 2026-07-21 11:14:41 +08:00

## Modification Time

2026-07-21 11:14:41 +08:00

## Modifier

roxanne517

## Modification Goal

Prevent multiline model-generated captions from being misinterpreted as commands by
`pycocoevalcap` METEOR's line-oriented Java subprocess while preserving raw saved results.

## Modified Files

- `eval/metrics.py`
- `tests/test_metrics.py`
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-07-21-agent-fix-meteor-multiline-input.md`

## Core Changes

Caption references and predictions now pass through a metric-input-only normalization that
folds repeated whitespace to one space and removes the reserved `|||` protocol separator.
Canonical JSONL files are read-only during evaluation and retain their original text.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

Yes. Only caption/change-caption strings sent to `pycocoevalcap` are normalized. Runs whose
texts contain no line breaks, repeated whitespace, or `|||` remain comparable. Previously,
affected runs produced no completed metric file because METEOR treated continuation lines as
invalid commands; those saved JSONL results can now be reevaluated without rerunning inference.
Raw predictions, raw references, dataset splits, and metric implementations are unchanged.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Added a regression test for newlines, carriage returns, tabs, repeated whitespace, and the
reserved METEOR protocol separator.

## Whether .gitignore Was Updated

No. No new generated-file type or directory was introduced.

## Validation Method

- Run the complete local pytest suite.
- Reevaluate the saved 20-record LEVIR-CC smoke result on Spark before attempting the saved
  1929-record full result.

## Risks and Follow-up TODOs

- Replacing a literal `|||` changes that punctuation sequence only in metric input. The raw
  prediction remains available for audit.
- The fix requires Spark validation with the installed Java METEOR subprocess.
