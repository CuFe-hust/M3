# Modification Note: Make Change Verification Conditional - 2026-07-30 16:04:47 +08:00

## Modification Time

2026-07-30 16:04:47 +08:00

## Modifier

roxanne517

## Modification Goal

Retain an auditable second change-caption review for structurally risky results without paying for
two Qwen calls on every sample.

## Modified Files

- `spacers_agent/agents/change/agent.py`
- `eval/change_comparison_report.py`
- `tests/agents/change/test_change_agent.py`
- `tests/test_change_comparison_report.py`
- `DETAILS.md`
- `docs/architecture/agent-runtime.md`
- `docs/changes/2026-07-30-agent-make-change-verification-conditional.md`
- `docs/experiments/qwen3-vl-4b-levir-cc-change-evidence-20.md`

## Core Changes

- Always run the evidence-analysis pass first.
- Trigger verification only for incomplete status, explicit uncertainty, or a positive answer
  without geometry or a localized T1/T2 comparison.
- Reserve model-call budget when each call is actually issued.
- Persist `verification_triggered` and `verification_reasons` in the Agent trace.
- Stop treating a local clause such as “the rest remains unchanged” as a global no-change answer.
- Allow the portable comparison report to represent samples where verification was not triggered.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. The same structured Qwen client and `ExpertResult` schema are used.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No metric, split, reference, or evaluation output format was changed. Runtime predictions can
change because low-risk samples now keep the first-pass result instead of forcing verification.

## Whether Deployment Was Affected

No deployment interface changed. Conditional execution is intended to reduce average local
inference latency and token use.

## Whether pytest Was Updated

Yes. Tests cover clean no-change skipping, supported-positive skipping, risk-triggered
verification, dynamic budget use, trace fields, and local-versus-global unchanged wording.

## Whether .gitignore Was Updated

No new generated artifact type or local path was introduced.

## Validation Method

- Compile the modified Agent and tests.
- Run focused ChangeAgent, runtime wiring, and parity tests.
- Run the complete offline pytest suite on Spark before any live model test.
- Evaluate the fixed rule on a development or separately selected validation sample set; do not
  tune the rule on the fixed 20 test samples.

Completed validation:

- 32 focused tests passed on Spark.
- 377 full repository tests passed on Spark.
- A fresh balanced 20-sample LEVIR-CC validation run completed 20/20 samples with no failures.
- Verification triggered for 10/20 samples, reducing the average to 1.5 model calls per sample.
- The auxiliary changeflag diagnostic improved from 9/20 for analysis only to 13/20 for the
  conditional final result; all persisted caption metrics also improved.

## Risks and Follow-up TODOs

- Lexical uncertainty detection is intentionally conservative and English-only because the current
  LEVIR-CC prompts return English.
- A confident but visually wrong first pass can still bypass verification.
- The balanced validation run remains only 20 samples; a larger preregistered sample is needed
  before making this workflow the default for a full evaluation.
