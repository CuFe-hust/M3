# Experiment Record: YOLOv5m-OBB CSL DOTA-v2.0 VRSBench Vehicle-Count 50

## Time

2026-08-04 15:21:37 +08:00

## Dataset

The established first 50 VRSBench validation `object quantity` questions containing `vehicle`; question IDs are recorded in the output summary.

## Model

`yolov5m_obb_csl_dotav20.onnx`, SHA256 `c964985b56ab05bcb679718f3fe5261246fd41f8cf0e4e620ba5b1c68092a81a`.

## Configuration File

Local detector-only visualization using image size 1024, confidence 0.60, IoU 0.50, CUDA ONNX Runtime.

## Run Command

`C:\Users\TZDEZACR\miniconda3\envs\yolo_env\python.exe tmp\run_dotav20_onnx_vrsbench_boxes.py`

## Metric Results

This run generated detector boxes only; it did not calculate VQA accuracy or call Qwen/DeepSeek. It produced 119 vehicle boxes across 50 images.

## Resource Consumption

Mean inference time was 47.3 ms/image after the corrected run, including one model warm-up image.

## Conclusion

The DOTA-v2.0 ONNX model uses 18 classes and 180 CSL angle bins. Its image-coordinate polygon conversion requires the YOLOv5-OBB `-sin(theta)` convention; OpenCV display/NMS requires the corresponding negated angle.

## Reproducibility Statement

The model artifact is synchronized through Git LFS by explicit user request. The local report contains images and is not committed.
