# Modification Note: Two-Stage Change Evidence Agent - 2026-07-29 15:54:41 +08:00

## Modification Time

2026-07-29 15:54:41 +08:00

## Modifier

roxanne517 (`3287638548@qq.com`)

## Modification Goal

Add a bounded, reference-free evidence verification stage to the new-runtime ChangeAgent so
LEVIR-CC change captions can be compared against the existing direct-model baseline without
changing model weights or benchmark references.

## Modified Files

- `spacers_agent/agents/change/agent.py`
- `spacers_agent/agents/visual_base.py`
- `spacers_agent/bootstrap.py`
- `spacers_agent/prompt_catalog.py`
- `prompts/change_analysis_v2.md`
- `prompts/change_verification_v2.md`
- `prompts/change_verification_v3.md`
- `prompts/change_verification_v4.md`
- `tests/agents/change/test_change_agent.py`
- `tests/parity/test_visual_base_parity.py`
- `tests/parity/fixtures/change_caption/expected_calls.json`
- `tests/runtime/test_bootstrap_wiring.py`
- `docs/architecture/agent-runtime.md`
- `DETAILS.md`

## Core Changes

- Changed active `change_caption` execution from one Qwen request to two bounded requests:
  evidence analysis followed by independent verification.
- Sent the first pass only as an untrusted hypothesis in the second request. No reference answer
  or ground-truth field is added to either model request.
- Persisted the first pass under `change_expert/analysis/` and the final verification under the
  existing `change_expert/` artifact path.
- Added trace fields for the two stage names, prompt versions, artifact paths, and logical model
  call count.
- Reserved both Qwen calls before inference and failed visibly before the first call when the
  per-sample budget cannot accommodate the complete workflow.
- Kept `change_qa` and compatibility construction on the existing single-pass behavior.
- Added a reference-free evidence guard after a live Spark smoke test showed the verifier
  overturning a correct no-change analysis while returning empty `evidence`, `evidence_items`,
  and `boxes`. The unsupported positive override is rejected, both raw model calls remain
  persisted, and `selected_stage` plus `verification_guard` expose the decision in the trace.
- Versioned the stricter verification contract as `change-verification-v3`; the prompt now
  requires concrete persisted evidence before overturning a no-change first pass.
- A second live Spark run showed that a generic sentence could satisfy the v3 non-empty evidence
  check while merely restating the positive conclusion. Version v4 therefore requires geometry
  or a localized textual comparison that explicitly describes both T1 and T2; generic
  second-image greenness claims are rejected and exposed as
  `rejected_non_contrastive_positive_override`.

## Whether the Canonical Sample Format Was Changed

No. `UnifiedSample`, T1/T2 image roles, task names, and ground-truth handling are unchanged.

## Whether the Model Interface Was Changed

No model, processor, tokenizer, checkpoint, weight-loading path, or `ExpertResult` schema changed.
The active change-caption workflow now makes two structured calls instead of one.

## Whether the Configuration Was Changed

No user-facing configuration key or default changed. The active Prompt Catalog gained two
versioned internal bindings.

## Whether Evaluation Was Affected

Metric implementations, dataset splits, reference-answer readers, and failure denominators were
not changed. Predictions and latency can change, so the new workflow must be reported as a
separate experiment and compared with the same preselected samples.

## Whether Deployment Was Affected

No. YOLO, model weights, export code, and hardware-specific dependencies were not touched.

## Whether pytest Was Updated

Yes. Tests cover two-stage execution, reference isolation, artifact paths, call-budget behavior,
unchanged single-pass `change_qa`, Prompt Catalog bindings, and the frozen change-caption request
contract.

## Whether .gitignore Was Updated

No. The new first-pass artifacts remain below the existing ignored run/output directory.

## Validation Method

- `python -m compileall -q spacers_agent tests` -> passed.
- `git diff --check` -> passed.
- Local pytest was attempted in a project-only virtual environment, but dependency installation
  could not complete because the available Python package connections failed with TLS/HTTP errors.
- Focused and full pytest must therefore be run in the existing Spark
  `yiruoxuan-m3-qwen` environment before any branch push or live comparison.

## Risks and Follow-up TODOs

- Change-caption inference uses approximately twice the logical Qwen calls and must report the
  measured latency increase.
- Run the same fixed 20 LEVIR-CC samples through the direct baseline and this Agent before making
  any quality claim.
- Treat the 20-sample result as a smoke comparison, not an official full-dataset score.
- Do not enable or claim YOLO validation as part of this change.
