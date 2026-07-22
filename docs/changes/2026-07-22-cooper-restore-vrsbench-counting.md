# Modification Note: Restore VRSBench Counting Accuracy - 2026-07-22 22:32:20 +08:00

## Modification Time

2026-07-22 22:32:20 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Fix the regression where VRSBench quantity questions became zero whenever the local Qwen model omitted points, while retaining auditable boxes and the invariant that the final count is derived only from accepted points. DeepSeek behavior is intentionally unchanged.

## Modified Files

- `spacers_agent/workflow.py`
- `spacers_agent/cli.py` and `spacers_agent/commands/common.py`
- `prompts/count_localize_v1.md`
- `tests/test_multiagent_vqa_pipeline.py`
- `DETAILS.md`
- `docs/experiments/2026-07-22-vrsbench-hybrid-count-v1.md`

## Core Changes

- Reuse the previously accurate GeneralVQA v1 whole-image contract for an independent integer count proposal.
- Convert valid proposal boxes to accepted centre points; when boxes are absent or disagree with the proposal, run an independent tight-box localization pass.
- Recover only a complete integer answer header from a malformed or truncated proposal, discard its malformed geometry, and require localization.
- Deduplicate evidence and reject only tiny boundary fragments whose visible centre remains at the image edge.
- Persist the proposal, optional localizer, accepted points, supporting boxes, warnings, raw responses, and route information for HTML debugging.

## Whether the Canonical Sample Format Was Changed

No. `UnifiedSample`, source adapters, and canonical prediction artifacts are unchanged.

## Whether the Model Interface Was Changed

No model, processor, tokenizer, checkpoint, weight-loading path, or public client interface changed. The internal VRSBench quantity workflow now performs a proposal call and an optional localization call.

## Whether the Configuration Was Changed

No configuration field or default value changed. Two versioned prompts are loaded by the existing prompt registry.

## Whether Evaluation Was Affected

Inference behavior and therefore predictions can change. Reference answers, dataset split, exact-match calculation, report format, DeepSeek payload, and Judge role are unchanged.

## Whether Deployment Was Affected

No. The existing local Transformers path remains in use, with no vLLM or additional vision model.

## Whether pytest Was Updated

Yes. Tests cover direct proposal-box counting, localization fallback, accepted-point equality, conservative border-fragment filtering, and malformed proposal header recovery.

## Whether .gitignore Was Updated

No. No new output, cache, secret, dataset, model, or checkpoint file type was introduced.

## Validation Method

- `python -m compileall -q .`
- `pytest -q`
- `git diff --check`

## Risks and Follow-up TODOs

- Mock tests cannot measure Qwen visual recall; a new Spark run with a fresh run ID is required.
- The integer proposal is never accepted as the persisted final count without a matching accepted point set.
- Border-fragment filtering is intentionally conservative and requires both border contact and an edge-adjacent visible centre.
