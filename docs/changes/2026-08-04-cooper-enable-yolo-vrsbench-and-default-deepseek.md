# Enable YOLO for VRSBench counting and default DeepSeek review

Modified: 2026-08-04 00:27 +08:00  
Modifier: Cooper (`crj31415926@gmail.com`)

## Scope

- Route VRSBench quantity questions through the highest-priority enabled YOLO detector when its
  audited class map supports the requested target.
- Retain `vrsbench_qwen_count` as the detector unavailable, runtime-error, and empty-result review
  backend. Explicit `qwen_point` mode continues to use the dedicated VRSBench Qwen backend.
- Make `run-dataset` evaluation and DeepSeek VQA review enabled by default. `--no-evaluate` and
  `--judge-policy none` remain explicit opt-outs.

## Compatibility and risk

Canonical samples, official references, exact-match computation, report metric definitions, prompt
contents, and final counting invariants are unchanged. Default command behavior changes: VQA runs
now make billable DeepSeek API requests and fail visibly when the configured key is unavailable.
Enabling a detector can change VRSBench quantity answers because YOLO becomes the primary backend;
the trace records its attempt, final-use state, and any fallback. YOLO remains an optional dependency
and is still disabled in the repository default configuration.

## Deployment profile

The Spark-local ignored configuration enables `yolo26s_dota_obb` with the verified local weight
`/home/user/models/yolo26/yolo26s-obb.pt` and SHA-256
`38dbd72ef6804f9bbbea7ad20f486e6ca6e093c8cd9bc857207a846565bd6e0b`.

## Validation

- `python -m pytest -q`: 395 passed in 4.17 seconds on Spark in the isolated
  `Cooper_for_qwen9b` environment.
- The 100-sample VRSBench benchmark and deployment evidence are recorded after deployment.
