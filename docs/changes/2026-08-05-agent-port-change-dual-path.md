# Modification Note: Port ChangeAgent Dual Path to try_yolo - 2026-08-05 13:48 +08:00

## Modification Time

2026-08-05 13:48:31 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Port the completed LEVIR-CC harmonization / ChangeAgent dual-path work (originally
committed as `2b7f85b` on `feature/unified-vlm-interfaces`) onto the `try_yolo` branch,
adapting it to the `try_yolo` runtime architecture so the feature is preserved and all
offline tests pass on `try_yolo`.

## Modified Files

- Added (ported as-is): `spacers_agent/agents/change/preprocess.py`, `harmonizer.py`,
  `pair_validator.py`, `difference_proposal.py`, `reviewer.py`, `schemas.py`,
  `prompts/change_dual_path_v1.md`, `scripts/evaluate_levir_harmonization.py`,
  `tests/agents/change/test_difference_proposal.py`, `test_harmonizer.py`,
  `test_pair_validator.py`, and the two `docs/changes/` notes plus the experiment record.
- Adapted to `try_yolo`: `spacers_agent/agents/change/agent.py`,
  `spacers_agent/agents/change/reviewer.py`, `tests/agents/change/test_change_agent.py`.
- Merged cleanly from `2b7f85b`: `spacers_agent/bootstrap.py`, `settings.py`,
  `prompt_catalog.py`, `cli.py`, `commands/common.py`, `configs/default.yaml`,
  `pyproject.toml`, `requirements.txt`, `.gitignore`, README, DETAILS,
  `docs/architecture/agent-runtime.md`, `tests/runtime/test_bootstrap_wiring.py`,
  `tests/parity/test_visual_base_parity.py`.
- `spacers_agent/workflow.py`: kept deleted (the compatibility shim was already removed
  on `try_yolo`; the feature-branch edit to it is not needed).
- `spacers_agent/application.py`: fixed the HTTP `404` branch to consume the request
  body before closing the connection (Windows `ConnectionAbortedError` flake).

## Core Changes

- Cherry-picked `2b7f85b` onto `try_yolo` and resolved five conflicts (`workflow.py`
  kept deleted; prompt binding, two tests, and `ChangeAgent` unified to the WIP side).
- Adapted the dual-path ChangeAgent from the feature-branch `ExpertResult` contract to
  the `try_yolo` canonical `AgentResult` contract: response model, prompt instruction
  (`agent_name='change_agent'`), request id `change_agent`, artifact dir, and
  `result_filename="agent_result.json"`; `reviewer.py` now operates on `AgentResult`.
  No `ExpertResult` token remains in `spacers_agent/` or `prompts/`, so the
  `tests/architecture/test_no_legacy_agent_api.py` guard passes.
- The HTTP service `POST` handler now drains the request body on unknown paths before
  replying 404, eliminating a flaky Windows `ConnectionAbortedError` in the entry tests.

## Whether the Canonical Sample Format Was Changed

No. `UnifiedSample` and `t1`/`t2` roles are unchanged.

## Whether the Model Interface Was Changed

No. `models.entry.create_model`, `VisionLanguageClient.complete_json`, and
`assemble_runtime` are unchanged. The ChangeAgent prompt/request contract uses the
existing `AgentResult` schema with prompt version `change-dual-path-v1`.

## Whether the Configuration Was Changed

No new top-level keys. `agents.change.*` typed defaults were ported from the feature
branch and remain optional in older YAML files.

## Whether Evaluation Was Affected

No. Metrics, splits, and reference-answer reading are untouched. The offline LEVIR-CC
harmonization evaluator is ported as-is; its 100-pair run was re-executed and matches
the original experiment record (99 applied, 1 rejected/raw-fallback, 0 failures).

## Whether Deployment Was Affected

No deployment export or weight-loading path changed. OpenCV/NumPy were already declared
by the ported `requirements.txt`/`pyproject.toml` edits.

## Whether pytest Was Updated

Yes. The three ported change-agent test files and the adapted
`test_change_agent.py`/`test_bootstrap_wiring.py`/`test_visual_base_parity.py` run on
`try_yolo`. Full suite result: 473 passed; the only failure is the pre-existing
Windows path-separator assertion in `tests/test_dataset_validator.py`.

## Whether .gitignore Was Updated

No new entry needed; `outputs/` and `tmp/` are already ignored.

## Validation Method

- `python -m compileall -q main.py spacers_agent models` — passed.
- Full `python -m pytest -q` on `try_yolo` (m3 conda env): 473 passed, 1 pre-existing
  environment-specific failure (`test_levir_cc_missing_image_side_fails`, Windows
  backslash path assertion) unrelated to this port.
- `tests/entry` (single-entry tests) — 62 passed, re-run three times after the HTTP 404
  body-drain fix to confirm stability.
- Offline harmonization evaluation re-run on the local LEVIR-CC layout: 100/100 pairs
  processed, 99 applied / 1 rejected, 0 failures; metrics match the original experiment
  record (PIF MAD 41.51→26.44, full MAD 52.57→38.72, PIF coverage 0.4454).

## Risks and Follow-up TODOs

- The ported ChangeAgent now performs real image preprocessing before the model call;
  parity tests and any caller that previously assumed a pure model-only change agent
  must supply readable image files (the parity harness already does).
- No live model A/B/C ablation was run; calibration thresholds still need a versioned
  annotated layout before being treated as permanent acceptance limits.
- The single-entry `main.py` on `try_yolo` is unaffected: it consumes the same
  `AgentResult` payload contract through `PublicAnswer`.
