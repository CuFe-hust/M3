# Local model asset manifest

Binary weights are local-only by default. The two SegFormer checkpoints listed
below are explicit Git LFS assets: Git stores pointers while an LFS checkout
materializes the binaries in the worktree. Logical IDs are portable; physical
checkpoint paths are supplied by application configuration.

| Model | Logical ID | Task | Dataset | Local asset | Runtime | SHA256 | Source |
|---|---|---|---|---|---|---|---|
| SegFormer MiT-B2 iSAID | `SegFormer-MiT-B2:iSAID:local` | semantic segmentation | iSAID | `models/segformer_mitb2_isaid/model.safetensors` | Transformers | `f8e60686ec41160b5cbc494e8a3c1d28a92f7afdd41708c7b77e3d5793908b9a` | `try_yolo` Git LFS; Spark hash verified |
| SegFormer MiT-B2 OEM | `SegFormer-MiT-B2:OpenEarthMap:local` | semantic segmentation | OpenEarthMap | `models/segformer_mitb2_oem/model.safetensors` | Transformers | `d2141c79b2fc27ea5505db378b48e90e75e5ee06751df1c5b4028ef662fb2fab` | `try_yolo` Git LFS; Spark hash verified |
| YOLOv5m OBB CSL | `YOLOv5-OBB-CSL:DOTA-v2.0:yolov5m` | OBB detection/counting | DOTA v2.0 | `models/yolo_obb/yolov5m_obb_csl_dotav20.onnx` | ONNX Runtime | `c964985b56ab05bcb679718f3fe5261246fd41f8cf0e4e620ba5b1c68092a81a` | `try_yolo` Git LFS; Spark hash verified |
| Qwen3-VL 4B | `Qwen/Qwen3-VL-4B-Instruct` | main-flow VLM | — | configurable external checkpoint | Transformers | revision-based | Spark checkpoint verified |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B` | main-flow VLM | — | configurable external checkpoint | Transformers | revision-based | Spark checkpoint and invocation verified |

The iSAID `classes.json` mapping is the only authoritative class-name source;
placeholder `LABEL_*` values in `config.json` must never replace it. The OEM
`classes.json` (its exact order was confirmed by the user on 2026-08-20 for
the local OpenEarthMap checkpoint, checkpoint SHA256
`d2141c79b2fc27ea5505db378b48e90e75e5ee06751df1c5b4028ef662fb2fab`) declares
the 9-class mapping `background` + bareland / rangeland / developed_space /
road / tree / water / agriculture_land / building. The OEM `config.json`
still contains placeholder `LABEL_0..8` values, so it is not independent
semantic evidence and placeholder labels must not be published as metadata.

Both SegFormer checkpoints (`SegFormer-MiT-B2:iSAID:local` and
`SegFormer-MiT-B2:OpenEarthMap:local`) are addressable by the VQA semantic-mask
catalog. A checkpoint becomes executable in that seam only when its stable
binding is enabled and composition injects the verified client. Availability
to counting is a separate decision (`connected_components` policies in the
counting expert catalog) and must not be conflated with VQA enablement.

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
