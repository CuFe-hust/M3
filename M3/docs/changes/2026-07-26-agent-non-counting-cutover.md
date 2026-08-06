# Modification Note: Non-Counting Agent Parity Cutover - 2026-07-26 13:16:40 +08:00

## Modification Time

2026-07-26 13:16:40 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Make Caption, Change, Grounding, GeneralVQA, and Spatial use the standalone Agent implementations
while preserving the frozen non-counting request, result, geometry, status, and artifact contracts.

## Modified Files

- `spacers_agent/agents/visual_base.py`
- `spacers_agent/agents/{caption,change,grounding,general_vqa}/agent.py`
- `spacers_agent/agents/spatial/agent.py`
- `spacers_agent/agents/spatial/candidate_review.py`
- `spacers_agent/agents/spatial/evidence_merge.py`
- `spacers_agent/bootstrap.py`
- `spacers_agent/prompt_catalog.py`
- `spacers_agent/workflow.py`
- `spacers_agent/cli.py`
- `tests/agents/*`
- `tests/parity/test_visual_base_parity.py`
- `tests/parity/test_legacy_runtime_parity.py`
- `tests/parity/fake_clients.py`
- `DETAILS.md`
- `docs/architecture/agent-runtime.md`

## Core Changes

- Replaced mutable Prompt dictionaries in standalone visual Agents with immutable `PromptAsset`
  injection and retained the frozen request versions, including `change-expert-v1`.
- Routed the transitional `WorkflowService` through an `AgentRegistry`; it no longer owns a second
  expert dictionary or non-counting model-request implementation.
- Kept former workflow expert symbols as parameter-compatible adapters.
- Added the dedicated CaptionAgent request path using `caption_v1.md`, `caption-v1`,
  `caption_expert`, and `expert_result.json`.
- Preserved VRSBench subtype payloads, answer vocabularies, evidence repair, and geometry
  postprocessing for GeneralVQA and Spatial.
- Moved spatial review predicates and evidence merging into pure functions and replaced former
  workflow-private implementations with aliases.
- Removed the duplicate Spatial review/postprocess invocation; review failure remains visible and
  partial, while successful review is merged before one geometry postprocess.
- Preserved `change_qa` fallback only after a primary exception; a partial primary result remains
  partial and does not trigger fallback.

## Whether the Canonical Sample Format Was Changed

No. `UnifiedSample`, image roles, task names, metadata, and reference-answer handling are unchanged.

## Whether the Model Interface Was Changed

No. Qwen still receives the same messages, `ExpertResult` response model, request metadata, and
temperature/hash inputs. No model, processor, tokenizer, endpoint, or weight-loading behavior was
changed.

## Whether the Configuration Was Changed

No configuration key or default changed. The legacy CLI Prompt loader now includes the existing
`caption_v1.md` asset so caption requests do not fall back to a different Prompt.

## Whether Evaluation Was Affected

No metric, dataset split, reference-answer reader, Judge payload, or benchmark meaning changed.
VRSBench geometry is applied once in the standalone Agent path.

## Whether Deployment Was Affected

No. YOLO, `ultralytics`, weights, export paths, and hardware runtimes were not loaded or modified.

## Whether pytest Was Updated

Yes. Tests cover frozen request/result parity, dedicated caption behavior, partial/failed status,
primary-only change fallback, Spatial pure functions, single candidate review, artifacts, and legacy
symbol compatibility.

## Whether .gitignore Was Updated

No. No new persistent output, cache, dataset, weight, or model-artifact type was introduced.

## Validation Method

- `python -m compileall -q spacers_agent` -> passed.
- `pytest -q tests/agents` -> `55 passed`, `0 failed`, `0 errors`.
- `pytest -q tests/parity -k "caption or change or grounding or general_vqa or spatial"` ->
  `14 passed`, `22 deselected`, `0 failed`, `0 errors`.
- `pytest -q tests/runtime` -> `17 passed`, `0 failed`, `0 errors`.
- `pytest -q` -> `356 passed`, `0 failed`, `0 errors`.
- `git diff --check` -> passed.

Pytest emitted one Windows cache warning because the pre-existing repository `.pytest_cache`
directory is not writable. Test temporary paths were explicitly placed under ignored `outputs/`;
this warning did not produce a failed or error test.

## Risks and Follow-up TODOs

- The CLI still constructs the legacy DatasetRunner until the explicit CLI cutover phase.
- Legacy counting and VRSBench quantity implementations remain untouched for the later CountingAgent
  parity phase.
- The pre-cutover caption fixture intentionally remains as evidence of the former unsupported gap;
  post-cutover caption behavior is frozen separately in the new-Agent parity test.
