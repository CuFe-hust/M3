# Local model asset manifest

Binary weights are local-only by default. The checkpoint files listed below are
explicit Git LFS assets: Git stores pointers while an LFS checkout
materializes the binaries in the worktree. Logical IDs are portable; physical
checkpoint paths are supplied by application configuration.

| Model | Logical ID | Task | Dataset | Local asset | Runtime | SHA256 | Source |
|---|---|---|---|---|---|---|---|
| SegFormer MiT-B2 iSAID | `SegFormer-MiT-B2:iSAID:local` | semantic segmentation | iSAID | `models/segformer_mitb2_isaid/model.safetensors` | Transformers | `f8e60686ec41160b5cbc494e8a3c1d28a92f7afdd41708c7b77e3d5793908b9a` | `try_yolo` Git LFS; Spark hash verified |
| SegFormer MiT-B2 OEM | `SegFormer-MiT-B2:OpenEarthMap:local` | semantic segmentation | OpenEarthMap | `models/segformer_mitb2_oem/model.safetensors` | Transformers | `d2141c79b2fc27ea5505db378b48e90e75e5ee06751df1c5b4028ef662fb2fab` | user-confirmed class map; verified 2026-08-20; active |
| YOLOv5m OBB CSL | `YOLOv5-OBB-CSL:DOTA-v2.0:yolov5m` | OBB detection/counting | DOTA v2.0 | `models/yolo_obb/yolov5m_obb_csl_dotav20.onnx` | ONNX Runtime | `c964985b56ab05bcb679718f3fe5261246fd41f8cf0e4e620ba5b1c68092a81a` | `try_yolo` Git LFS; Spark hash verified |
| YOLO11m OBB VRSBench | `YOLO11m-OBB:VRSBench-QA1024:best` | OBB detection/counting | VRSBench QA1024 | `models/vrsbench-yolo11m-obb-1024/vrsbench_yolo11m_obb_best.pt` | Ultralytics | `4a86331ed43b316e60050c5a890f3149f48bf34ce6f5c0e61a15382969feaf52` | Git LFS; 100 epochs completed; server and local hashes verified 2026-08-28 |
| Qwen3-VL 4B | `Qwen/Qwen3-VL-4B-Instruct` | main-flow VLM | — | configurable external checkpoint | Transformers | revision-based | Spark checkpoint verified |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B` | main-flow VLM | — | configurable external checkpoint | Transformers | revision-based | Spark checkpoint and invocation verified |

The iSAID and OEM `classes.json` mappings are authoritative for their respective
checkpoints. The OEM nine-channel order was confirmed by the user for this local
checkpoint and recorded with its SHA256 and verification date in the class map.
OEM labels are `background`, `bareland`, `rangeland`, `developed_space`, `road`,
`tree`, `water`, `agriculture_land`, and `building` in ID order 0 through 8.

The VRSBench YOLO11m-OBB checkpoint embeds the canonical DOTA 18-class order.
The accepted source export has zero training instances for `large-vehicle` and
`small-vehicle`; this known taxonomy limitation was not repaired during training.

Both SegFormer directories include the canonical NVIDIA MiT-B2
`preprocessor_config.json` pinned from upstream revision
`3a609931044a9a83814802af9d861a47a1397636`. It supplies the audited ImageNet
normalization contract required for fully local `AutoImageProcessor` loading;
the tiled runtime still passes `do_resize=False`, so the upstream 512 default
never resizes inference tiles.

Change V2 is calibration-qualified with Transformers 5.14.1. The SegFormer
extras constrain Transformers to `>=5.14.1,<5.15`; other releases are not
calibration-qualified. Hidden-state geometry remains fail-closed: an
unresolvable token sequence is never reshaped by guessing a square grid.

## Change V3 semantic runtime contract

The SegFormer runtime is an optional deterministic evidence provider. Its logical
model identity, revision and verified weights SHA256 remain the cache/audit
identity; physical checkpoint paths are supplied by application configuration and
must not be written into public trace payloads.

When the backend supports the requested hidden states, the runtime can expose a
`DenseSemanticPyramidClient` result containing:

- class probabilities in the same class order for T1 and T2;
- `features_by_stage` for the explicitly configured feature stages;
- the real probability and feature grid strides;
- original image size, class names and JSON-safe diagnostics.

The existing `DenseSemanticClient.infer()` contract remains supported for legacy
callers. A requested stage that is unavailable is reported as missing/fallback; the
runtime must not fabricate a feature grid or silently claim multi-scale success.
No SegFormer weights listed here have been trained or fine-tuned for this project's
Change task. The current ChangeHead interface is disabled by default and no
post-training implementation is included.

## Expert orchestration

Counting selects enabled experts by declared target-label support, executes at
most five detector experts, and fuses instances with provenance so overlapping
detections are not double-counted. Only unresolved detector disagreements are
sent to the bounded Qwen review hook.

Change may run multiple verified SegFormer experts. Their taxonomies remain
independent: semantic and feature evidence is fused only after each expert has
produced its own score maps and transitions. iSAID supplies object/facility
semantics and OEM supplies `persistent_landcover` semantics with neutral
`background` and persistent land-cover labels. Generic bootstrap discovers both
verified experts automatically. Raw T1/T2 imagery remains authoritative for
every semantic decision.
