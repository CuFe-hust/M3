# Modification Note: Add Default Visual Audit Report - 2026-07-22 14:35:06 +08:00

## Modification Time

2026-07-22 14:35:06 +08:00

## Modifier

Cooper3516833584

## Modification Goal

Turn the validated 20-sample HTML audit layout into the default baseline report and print its absolute save path after inference and evaluation.

## Modified Files

`eval/audit_report.py`, `eval/metrics.py`, `main.py`, `config/baseline.example.json`, `tests/test_baseline_audit_report.py`, `README.md`, and `DETAILS.md`.

## Core Changes

- Added a bounded visual artifact writer that captures source images while inference still owns the image objects and deduplicates them by content hash.
- Added automatic HTML and CSV generation beside each canonical prediction file, enabled by default with a 200-sample visual cap.
- Added additive DeepSeek audit collection for request payloads, raw and parsed responses, duration, attempts, token usage, and errors.
- Regenerate the same report after evaluation and print the absolute report and metric paths.
- Added model-load, total-inference, and per-sample inference durations without changing generation behavior.

## Whether the Canonical Sample Format Was Changed

No. The canonical sample and prediction JSONL contracts are unchanged; report inputs and images are stored in a separate ignored output directory.

## Whether the Model Interface Was Changed

No. The model, processor, tokenizer, weight loading, prompts, and deterministic generation settings are unchanged.

## Whether the Configuration Was Changed

Yes, additively. `report.enabled` defaults to `true` and `report.max_samples` defaults to `200`. Existing local configurations require no edit.

## Whether Evaluation Was Affected

The metric algorithms, reference answers, dataset splits, and metric JSON return format are unchanged. The existing DeepSeek calls now optionally copy auditable call details into a caller-owned list for report generation.

## Whether Deployment Was Affected

Yes. Server inference and evaluation now emit a self-contained report directory and print its absolute location. The feature uses existing Pillow and standard-library dependencies only.

## Whether pytest Was Updated

Yes. Tests cover image deduplication, HTML/CSV content, DeepSeek raw-call audit fields, and absolute report-path output.

## Whether .gitignore Was Updated

No. All generated report directories remain under the already ignored `outputs/` tree.

## Validation Method

Run focused audit-report tests, the complete offline pytest suite, compile checks, `git diff --check`, and one Spark smoke inference/evaluation using existing local weights and the authorized DeepSeek configuration.

## Risks and Follow-up TODOs

The visual cap prevents a full benchmark from generating an unbounded HTML and image set; users who intentionally raise it must account for output size. DeepSeek judgments remain a non-official text-only proxy and must not replace deterministic or upstream benchmark metrics.
