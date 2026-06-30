#!/usr/bin/env python3
"""
YOLOv8 Offline Inference for Semantic-SLAM
===========================================
Runs YOLOv8-seg on all frames of a TUM RGB-D dataset and saves
per-frame detection results as JSON files.

Usage:
    python3 yolov8_offline_inference.py \
        --dataset ~/autodl-tmp/Script/semantic-slam-yolov8/data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --output  ~/autodl-tmp/Script/semantic-slam-yolov8/data/detections/walking_xyz \
        --model   yolov8n-seg.pt \
        --conf    0.45 \
        --iou     0.45

Output format (per frame JSON):
    {
        "frame_id": 0,
        "timestamp": 1305031102.175304,
        "image_path": "rgb/1305031102.175304.png",
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.92,
                "bbox": [x1, y1, x2, y2],
                "is_dynamic": true,
                "mask_bbox_shape": [h, w]
            }
        ],
        "num_detections": 2,
        "num_dynamic": 1
    }
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

# Dynamic COCO class IDs for SLAM filtering
# NOTE: Keep in sync with src/src/YoloDetector.cc: DYNAMIC_COCO_IDS
DYNAMIC_COCO_IDS = {0, 1, 2, 3, 5, 7, 16, 17}
# person, bicycle, car, motorcycle, bus, truck, dog, cat


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 offline inference for Semantic-SLAM")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--dataset_type", default="auto", choices=["auto", "tum", "kitti"],
                        help="Dataset format (auto-detect by default)")
    parser.add_argument("--output", required=True, help="Output directory for JSON detections")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLOv8 model file")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="CUDA device (0, 1, ...) or 'cpu'")
    parser.add_argument("--half", action="store_true", help="Use FP16 half precision")
    return parser.parse_args()


def load_frame_list(dataset_dir, dataset_type="auto"):
    """Load frame list from dataset directory.
    Supports TUM (rgb.txt), KITTI (image_0/ or image_2/), and EuRoC (mav0/cam0/data/) formats.
    """
    # Auto-detect dataset type
    if dataset_type == "auto":
        if os.path.exists(os.path.join(dataset_dir, "rgb.txt")):
            dataset_type = "tum"
        elif os.path.isdir(os.path.join(dataset_dir, "image_0")) or os.path.isdir(os.path.join(dataset_dir, "image_2")):
            dataset_type = "kitti"
        elif os.path.isdir(os.path.join(dataset_dir, "mav0", "cam0", "data")):
            dataset_type = "euroc"
        else:
            print(f"[ERROR] Cannot detect dataset type in {dataset_dir}")
            sys.exit(1)

    frames = []

    if dataset_type == "tum":
        rgb_txt = os.path.join(dataset_dir, "rgb.txt")
        if not os.path.exists(rgb_txt):
            print(f"[ERROR] rgb.txt not found in {dataset_dir}")
            sys.exit(1)
        with open(rgb_txt) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    frames.append((float(parts[0]), parts[1]))
        print(f"[INFO] TUM dataset: {len(frames)} frames from rgb.txt")

    elif dataset_type == "kitti":
        # KITTI: images in image_0/ (grayscale left) or image_2/ (color left)
        img_dir = None
        for d in ["image_2", "image_0"]:
            candidate = os.path.join(dataset_dir, d)
            if os.path.isdir(candidate):
                img_dir = d
                break
        if img_dir is None:
            print(f"[ERROR] No image_0/ or image_2/ found in {dataset_dir}")
            sys.exit(1)

        img_dir_full = os.path.join(dataset_dir, img_dir)
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        files = sorted([f for f in os.listdir(img_dir_full)
                        if os.path.splitext(f)[1].lower() in exts])
        for idx, fname in enumerate(files):
            frames.append((float(idx), f"{img_dir}/{fname}"))
        print(f"[INFO] KITTI dataset: {len(frames)} frames from {img_dir}/")

    elif dataset_type == "euroc":
        # EuRoC: images in mav0/cam0/data/, timestamps in mav0/cam0/data.csv
        cam0_data = os.path.join(dataset_dir, "mav0", "cam0", "data")
        cam0_csv = os.path.join(dataset_dir, "mav0", "cam0", "data.csv")
        if not os.path.isdir(cam0_data):
            print(f"[ERROR] mav0/cam0/data/ not found in {dataset_dir}")
            sys.exit(1)

        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        files = sorted([f for f in os.listdir(cam0_data)
                        if os.path.splitext(f)[1].lower() in exts])

        # Try to read timestamps from CSV
        ts_map = {}
        if os.path.exists(cam0_csv):
            with open(cam0_csv) as f:
                header = f.readline()  # skip header
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        ts_ns = float(parts[0])
                        fname = parts[1].strip()
                        ts_map[fname] = ts_ns * 1e-9

        for fname in files:
            ts = ts_map.get(fname, 0.0)
            frames.append((ts, f"mav0/cam0/data/{fname}"))
        print(f"[INFO] EuRoC dataset: {len(frames)} frames from mav0/cam0/data/")

    return frames


def run_inference(args):
    from ultralytics import YOLO

    # Load model
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # Load frame list
    frames = load_frame_list(args.dataset, args.dataset_type)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Also export coco.names if not exists
    coco_names_path = os.path.join(os.path.dirname(args.output), "coco.names")
    if not os.path.exists(coco_names_path):
        names = model.model.names
        with open(coco_names_path, 'w') as f:
            for i in range(80):
                f.write(f"{names.get(i, f'class_{i}')}\n")
        print(f"[INFO] Exported coco.names to {coco_names_path}")

    # Process frames
    total_dynamic = 0
    total_detections = 0

    for idx, (timestamp, rgb_path) in enumerate(frames):
        img_path = os.path.join(args.dataset, rgb_path)
        if not os.path.exists(img_path):
            continue

        # Run YOLOv8 inference
        results = model(
            img_path,
            verbose=False,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
        )

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box_idx in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[box_idx])
                    conf = float(r.boxes.conf[box_idx])
                    x1, y1, x2, y2 = r.boxes.xyxy[box_idx].cpu().numpy().tolist()

                    mask_bbox_shape = None
                    if r.masks is not None and box_idx < len(r.masks):
                        mask_np = r.masks.data[box_idx].cpu().numpy()
                        mask_bbox_shape = list(mask_np.shape)

                    is_dynamic = cls_id in DYNAMIC_COCO_IDS

                    detections.append({
                        "class_id": cls_id,
                        "class_name": r.names.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "is_dynamic": is_dynamic,
                        "mask_bbox_shape": mask_bbox_shape,
                    })

        num_dynamic = sum(1 for d in detections if d["is_dynamic"])
        total_dynamic += num_dynamic
        total_detections += len(detections)

        result_json = {
            "frame_id": idx,
            "timestamp": timestamp,
            "image_path": rgb_path,
            "detections": detections,
            "num_detections": len(detections),
            "num_dynamic": num_dynamic,
        }

        out_path = os.path.join(args.output, f"{idx:06d}.json")
        with open(out_path, 'w') as f:
            json.dump(result_json, f, indent=2)

        # Progress
        if (idx + 1) % 50 == 0 or idx == len(frames) - 1:
            print(f"  [{idx+1}/{len(frames)}] "
                  f"det={len(detections)} dyn={num_dynamic} "
                  f"img={rgb_path}")

    # Summary
    print(f"\n[DONE] Processed {len(frames)} frames")
    print(f"  Total detections: {total_detections}")
    print(f"  Total dynamic:    {total_dynamic}")
    print(f"  Output dir:       {args.output}")

    # Save summary
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "total_frames": len(frames),
        "total_detections": total_detections,
        "total_dynamic": total_dynamic,
        "dynamic_class_ids": sorted(list(DYNAMIC_COCO_IDS)),
    }
    summary_path = os.path.join(args.output, "_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
