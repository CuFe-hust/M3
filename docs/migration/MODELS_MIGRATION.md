# Model migration map

| `try_yolo` component | `new_structure` equivalent | Result |
|---|---|---|
| `models/qwen_transformers.py` | hardened `models/qwen_transformers.py` + `models.entry.create_model` | New implementation retained; offline/cache identity/single assembly preserved |
| Qwen3-VL/Qwen3.5 wrappers | `models/qwen3_vl/`, `models/qwen3_5/` | New implementation retained; no legacy alias restored |
| `spacers_agent/.../yolo_model_store.py` | `agents/counting/backends/yolo_model_store.py` | New concurrent single-load/hash/task/class validation retained; shared LFS-pointer guard added |
| `spacers_agent/.../yolov5_obb_onnx.py` | `agents/counting/backends/yolov5_obb_onnx.py` | New CUDA/provider/device policy retained |
| iSAID/OEM checkpoint directories | versioned metadata + ignored local weights | Restored from `origin/try_yolo`; Git LFS objects materialized locally |
| SegFormer loading/inference | `models/segformer_transformers.py` | New offline runtime: load/preprocess/infer/upsample/argmax/class mapping |
| Spark absolute checkpoint paths | application YAML/environment settings + logical model IDs | Machine paths remain local and never become cache/trace identities |

The migration intentionally does not restore `spacers_agent/`, change public
tasks, or add a segmentation Agent/evaluator. SegFormer is currently a reusable
expert runtime; any standalone semantic-segmentation task remains a separate
task-contract change.
