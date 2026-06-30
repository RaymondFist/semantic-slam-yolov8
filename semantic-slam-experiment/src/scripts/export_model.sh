#!/bin/bash
# =============================================================================
# Model Export: PyTorch YOLOv8 -> ONNX -> TensorRT Engine
# =============================================================================
# Run once to generate models/ from official Ultralytics weights.
# Requires: ultralytics, onnx, onnxruntime, tensorrt (Python packages)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../models"
mkdir -p "${MODEL_DIR}"

echo "==> Exporting YOLOv8-nano-seg model..."

python3 << 'PYEOF'
import os
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

# Step 1: Export to ONNX
print("[1/3] Exporting ONNX (opset=12, dynamic batch=OFF)...")
model.export(
    format="onnx",
    opset=12,
    imgsz=640,
    dynamic=False,
    half=True,
    simplify=True,
)

# Step 2: Create COCO class names file
print("[2/3] Creating coco.names...")
coco_names = model.model.names  # dict {0: 'person', 1: 'bicycle', ...}
coco_path = os.path.join("..", "..", "models", "coco.names")
with open(coco_path, "w") as f:
    for i in range(80):
        name = coco_names.get(i, f"class_{i}")
        f.write(f"{name}\n")
print(f"    Written {len(coco_names)} class names to {coco_path}")

print("[3/3] Done.")
print("\nONNX model:  models/yolov8n-seg.onnx")
print("Class names: models/coco.names")
print("\nTo build TensorRT engine (on target machine):")
print("  trtexec --onnx=models/yolov8n-seg.onnx \\")
print("          --saveEngine=models/yolov8n-seg.trt \\")
print("          --fp16 --optShapes=input:1x3x640x640")
PYEOF

echo "==> Model export complete."