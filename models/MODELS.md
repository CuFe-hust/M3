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
asset has no equivalent authoritative class file, so its class names are
unknown (`None`/empty) even though its output-channel dimension remains usable.
Placeholder labels must not be published as semantic metadata.
