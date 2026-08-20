# Local model asset manifest

Binary weights are local-only by default. The two SegFormer checkpoints listed
below are explicit Git LFS assets: Git stores pointers while an LFS checkout
materializes the binaries in the worktree. Logical IDs are portable; physical
checkpoint paths are supplied by application configuration.

| Model | Logical ID | Task | Dataset | Local asset | Runtime | SHA256 | Source |
|---|---|---|---|---|---|---|---|
| SegFormer MiT-B2 iSAID | `SegFormer-MiT-B2:iSAID:local` | semantic segmentation | iSAID | `models/segformer_mitb2_isaid/model.safetensors` | Transformers | `f8e60686ec41160b5cbc494e8a3c1d28a92f7afdd41708c7b77e3d5793908b9a` | `try_yolo` Git LFS; Spark hash verified |
| SegFormer MiT-B2 OEM | `SegFormer-MiT-B2:OpenEarthMap:local` | semantic segmentation | OpenEarthMap | `models/segformer_mitb2_oem/model.safetensors` | Transformers | `d2141c79b2fc27ea5505db378b48e90e75e5ee06751df1c5b4028ef662fb2fab` | checkpoint-specific class map unverified; runtime blocked |
| YOLOv5m OBB CSL | `YOLOv5-OBB-CSL:DOTA-v2.0:yolov5m` | OBB detection/counting | DOTA v2.0 | `models/yolo_obb/yolov5m_obb_csl_dotav20.onnx` | ONNX Runtime | `c964985b56ab05bcb679718f3fe5261246fd41f8cf0e4e620ba5b1c68092a81a` | `try_yolo` Git LFS; Spark hash verified |
| YOLO11s iSAID tiled | `YOLO11s:iSAID:tiles1024-o20:epoch111` | axis-aligned detection/counting | iSAID | `models/isaid-yolo11s-tiles1024-o20/isaid_yolo11s_tiles1024_o20_best_epoch111.pt` | Ultralytics | `f3d741a8f1c6c78d2e3cf2c92392fd2547ef537c25ab4f8da093c2d938369266` | Git LFS; 1024 px tiles with 20% overlap; best epoch 111 |
| Qwen3-VL 4B | `Qwen/Qwen3-VL-4B-Instruct` | main-flow VLM | — | configurable external checkpoint | Transformers | revision-based | Spark checkpoint verified |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B` | main-flow VLM | — | configurable external checkpoint | Transformers | revision-based | Spark checkpoint and invocation verified |

The iSAID `classes.json` mapping is authoritative. The OEM checkpoint exposes
nine output channels, but its checkpoint-specific channel order has not been
verified; its `LABEL_*` placeholders are not semantic metadata and the runtime
remains blocked until a verified map is supplied.

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
