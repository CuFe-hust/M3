# Enable YOLO for VRSBench counting and default DeepSeek review

Modified: 2026-08-04 00:27 +08:00  
Modifier: Cooper (`crj31415926@gmail.com`)

## Scope

- Route VRSBench quantity questions through the highest-priority enabled YOLO detector when its
  audited class map supports the requested target.
- Retain `vrsbench_qwen_count` as the detector unavailable, runtime-error, and empty-result review
  backend. Explicit `qwen_point` mode continues to use the dedicated VRSBench Qwen backend.
- Reject detector polygons clipped at an image edge when their centres are also within the existing
  25/999 border-fragment band, and merge same-tile OBB duplicates only at the configured strict IoU.
- Adapt accepted YOLO points to a canonical VQA `ExpertResult` while retaining the full
  `counting_result.json`, so Judge and the image-bearing HTML report consume the detector result.
- Make `run-dataset` evaluation and DeepSeek VQA review enabled by default. `--no-evaluate` and
  `--judge-policy none` remain explicit opt-outs.

## Compatibility and risk

Canonical samples, official references, exact-match computation, report metric definitions, prompt
contents, and final counting invariants are unchanged. Default command behavior changes: VQA runs
now make billable DeepSeek API requests and fail visibly when the configured key is unavailable.
Enabling a detector can change VRSBench quantity answers because YOLO becomes the primary backend;
the trace records its attempt, final-use state, and any fallback. YOLO remains an optional dependency
and is still disabled in the repository default configuration. Border rejection can omit a real but
severely clipped object; the centre-plus-polygon conjunction matches the established VRSBench count
policy and avoids rejecting ordinary objects merely located near an edge.

## Deployment profile

The Spark-local ignored configuration enables `yolo26s_dota_obb` with the verified local weight
`/home/user/models/yolo26/yolo26s-obb.pt` and SHA-256
`38dbd72ef6804f9bbbea7ad20f486e6ca6e093c8cd9bc857207a846565bd6e0b`.

## Validation

- `python -m pytest -q`: 398 passed in 3.84 seconds on Spark after installing Ultralytics in the
  isolated `Cooper_for_qwen9b` environment.
- The 100-sample VRSBench benchmark and deployment evidence are recorded after deployment.
