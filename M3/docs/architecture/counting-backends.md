# Counting Backends

## Architecture

```
CountingAgent
  → CountTargetParser (text-only Qwen or rule-based)
  → BackendSelector
     → VRSBenchQwenCountBackend (VRSBench vehicle quantity, proposal+localizer)
     → QwenPointCountingBackend (tile-based owner-core point counting)
     → YoloOBBCountingBackend draft (not registered during cutover)
  → CountingResult (final_count == accepted points)
```

## Backend Protocol

Each backend must implement:
- `is_available()` — weight file exists, client alive (no model load)
- `supports(target)` — class/alias/composite match
- `count(request, context)` — full inference → CountingResult

## QwenPointCountingBackend

Wraps `PointCountingOrchestrator`:
- Non-overlapping owner cores + halo context
- Tile-local 0..999 coordinate system
- Sequential tile processing with checkpoint/resume
- Recursive split for dense regions
- Empty tile review (missing_point_review_v3)
- Seam conflict detection (no DeepSeek for YOLO)
- `final_count == accepted points` enforced by CountingResult validator

## VRSBenchQwenCountBackend

Preserves 200-sample-comparable VRSBench vehicle pipeline:
1. `vrsbench_count_target()` — fixed vehicle ontology
2. Whole-image count proposal (GeneralVQA v1)
3. Proposal mismatch → independent localizer
4. Box evidence → accepted centres → dedup → border fragment rejection
5. `final_count == accepted points`

## YoloOBBCountingBackend

This optional backend is registered only when enabled local detector settings are supplied. It lazy
loads a SHA256-verified OBB model, validates its task and class map, filters every detection to the
requested target classes, and converts OBB centres into owner-core accepted points.

Per-detector backend:
- `YoloDetectorSettings` defines classes, aliases, composites
- `YoloModelStore` — thread-safe lazy load, cache by resolved path
- Weight checked before `ultralytics` import
- OBB polygon centre → local 0..999 → `convert_local_point_to_global`
- Owner-core acceptance (point centre in core, not box intersection)
- Provenance: `PointProvenance(source="yolo_obb_center", ...)`

## Configuration

The default remains disabled. An enabled local configuration is accepted only when it supplies an
enabled detector with its model ID, SHA256, source dataset, and audited class map.

```yaml
backend:
  yolo:
    enabled: false
    fallback_to_qwen_on_unavailable: true
    fallback_to_qwen_on_error: true
    verify_empty_with_qwen: false
    detectors:
      - name: dota_obb
        enabled: true
        weights: weights/yolo11s-obb.pt
        priority: 100
        classes: [plane, ship, ...]
        aliases: {car: small vehicle}
        composite_targets: {vehicle: [small vehicle, large vehicle]}
```

## Safety

- YOLO disabled → no `ultralytics` import
- YOLO enabled → registers configured detectors but still does not load them during runtime assembly
- Weight file missing → `DetectorWeightsMissingError` before import
- No auto-download, no network access
- `ultralytics` is an optional dependency (`[yolo]` extra)
- Empty detection = valid 0 (not a failure)
