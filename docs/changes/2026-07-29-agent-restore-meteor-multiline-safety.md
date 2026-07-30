# Modification Note: Restore METEOR Multiline Safety - 2026-07-29 17:30:00 +08:00

## Modification Time

2026-07-29 17:30:00 +08:00

## Modifier

roxanne517

## Modification Goal

Restore the already validated metric-input safeguard that prevents multiline captions from
being interpreted as separate commands by pycocoevalcap METEOR.

## Modified Files

- `eval/metrics.py`
- `tests/test_metrics.py`
- `DETAILS.md`
- `docs/changes/2026-07-29-agent-restore-meteor-multiline-safety.md`

## Core Changes

- Fold repeated whitespace and line breaks only in temporary caption metric input.
- Remove METEOR's reserved `|||` separator only from temporary metric input.
- Preserve raw JSONL predictions and references unchanged.
- Add a focused regression test for newlines, carriage returns, tabs, repeated whitespace,
  and the reserved separator.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

Only previously failing multiline caption inputs can now complete evaluation. Caption strings
that already contain one line and no reserved separator remain comparable. Metric
implementations, dataset splits, and reference answers are unchanged.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. `tests/test_metrics.py` covers the metric-protocol normalization.

## Whether .gitignore Was Updated

No. No new generated artifact type was introduced.

## Validation Method

- Run the focused metric regression test.
- Run the complete offline pytest suite on Spark.
- Reevaluate the saved 20-sample baseline and Agent results without rerunning inference.

## Risks and Follow-up TODOs

- A literal `|||` is removed only from temporary metric input; raw artifacts remain available
  for audit.
- Spark validation is required because METEOR uses the installed Java subprocess.
