# Modification Note: Add Change Comparison Report - 2026-07-30 13:17:33 +08:00

## Modification Time

2026-07-30 13:17:33 +08:00

## Modifier

roxanne517

## Modification Goal

Generate a portable, per-sample comparison of the saved LEVIR-CC baseline and two-stage Change Agent artifacts without rerunning inference.

## Modified Files

- `eval/change_comparison_report.py`
- `tests/test_change_comparison_report.py`
- `DETAILS.md`
- `docs/changes/2026-07-30-agent-add-change-comparison-report.md`
- `docs/experiments/qwen3-vl-4b-levir-cc-change-evidence-20.md`

## Core Changes

- Display T1/T2 images, references, baseline, analysis, verification, final answer, selected stage, guard, and inference time.
- Keep official caption metrics separate from the auxiliary binary `changeflag` diagnostic.
- Read the official `changeflag` from a caller-supplied, read-only derived sample manifest.
- Copy report images to a self-contained sibling directory while leaving dataset files unchanged.
- Emit a machine-readable JSON summary beside the HTML report.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No metric, split, reference, prediction, or failure-handling rule was changed. The report copies already persisted caption metrics and labels `changeflag` accuracy as an auxiliary diagnostic.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. A report-generation regression test covers stage rendering, accuracy separation, and portable image copies.

## Whether .gitignore Was Updated

No. Reports are written to caller-selected output paths, and existing output-directory rules already apply.

## Validation Method

- Compile the new report module and test.
- Run the focused report test.
- Run the complete offline pytest suite on Spark.
- Generate the report from the fixed saved 20-sample experiment without loading Qwen.

## Risks and Follow-up TODOs

- The binary diagnostic relies on the official `changeflag` and an explicitly documented phrase-based no-change classifier; it is not an official caption metric.
- The current report is intentionally specialized for persisted LEVIR-CC two-stage Change Agent artifacts.
