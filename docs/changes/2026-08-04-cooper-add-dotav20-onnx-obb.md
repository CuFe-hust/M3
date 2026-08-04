# Modification Note: DOTA-v2 YOLOv5-OBB ONNX Runtime - 2026-08-04 15:21:37 +08:00

## Modification Time

2026-08-04 15:21:37 +08:00

## Modifier

Cooper <crj31415926@gmail.com>

## Modification Goal

Make the verified YOLOv5m-OBB CSL DOTA-v2.0 ONNX artifact the deployable default detector profile while retaining the existing multi-detector configuration interface.

## Modified Files

- `spacers_agent/schemas.py`
- `spacers_agent/agents/counting/backends/yolo_model_store.py`
- `spacers_agent/agents/counting/backends/yolov5_obb_onnx.py`
- `spacers_agent/agents/counting/backends/yolo_obb.py`
- `configs/yolo.example.yaml`
- `configs/default.yaml`
- `pyproject.toml`
- `tests/test_yolov5_obb_onnx.py`
- `README.md`
- `DETAILS.md`

## Core Changes

- Added a lazy GPU ONNX Runtime adapter for YOLOv5-OBB CSL output `[cx, cy, long, short, objectness, classes, 180 theta bins]`.
- Preload NVIDIA site-package CUDA/cuDNN libraries before creating the CUDA execution provider, supporting Spark's isolated `Cooper_for_qwen9b` environment.
- Preserved the `detectors` list and priority-based routing interface; each detector now selects `ultralytics` or `onnx_yolov5_obb` explicitly.
- Restored OBB polygons with the source image-coordinate convention (`-sin(theta)`), including the corresponding rotated-NMS angle sign.
- Declared the DOTA-v2.0 18-class example profile and its verified SHA256.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes. `YoloDetectorSettings.runtime` selects the lazy detector runtime; existing configurations default to `ultralytics`.

## Whether the Configuration Was Changed

Yes. The example/default commented profile now documents the DOTA-v2.0 ONNX detector.

## Whether Evaluation Was Affected

No evaluation metric, split, answer reader, or evaluation output contract was changed.

## Whether Deployment Was Affected

Yes. Deployment needs `.[yolo-onnx]` for this profile and the pre-provisioned ONNX artifact.

## Whether pytest Was Updated

Yes. Added geometry and runtime-schema coverage.

## Whether .gitignore Was Updated

No. The user explicitly requested this single model artifact to be synchronized through Git LFS; `.gitattributes` records that exception.

## Validation Method

- Local CUDA ONNX Runtime inspection: input `[1, 3, 1024, 1024]`, output `[1, 64512, 203]`.
- Local 50-image VRSBench vehicle-count box render, including visual inspection after correcting the angle convention.
- Focused pytest for the ONNX adapter and YOLO backend configuration.

## Risks and Follow-up TODOs

- The ONNX adapter accepts only the audited fixed-size DOTA-v2.0 CSL output contract.
- Full Spark run remains required after deployment to validate its installed GPU ONNX Runtime.
