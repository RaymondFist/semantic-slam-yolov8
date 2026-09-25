#!/usr/bin/env python3
"""
Random Mask Control Dataset Generation Script (E1.6 Random-Mask Baseline)
========================================================
Generates a "random mask" control dataset
to rule out the confound that the improvement comes only from fewer features,
so the YOLO-Mask gain can be attributed to semantic localization itself.

Core design:
  - For each frame, read the dynamic object bboxes from YOLOv8 detections and compute their total area and count.
  - Sample a rectangle in the image at a random location with the same area as a dynamic bbox
    (same number of rectangles as dynamic bboxes, matched one-to-one by area) and zero out its pixels.
  - Result: Random-Mask removes the same pixel area and the same number of blocks as YOLO-Mask;
    the only difference is that locations are random (no semantic localization) rather than on dynamic objects.
  - Fixed random seed for reproducibility.

Usage (single sequence):
    python3 scripts/random_mask_dataset.py \
        --detections data/detections/walking_xyz \
        --dataset   data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --output    data/datasets_randmask/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --dataset-type tum --seed 42

Batch (--all mode) works the same as mask_dataset.py.
"""

import os
import sys
import json
import random
import argparse
import shutil

import cv2
import numpy as np

# Dynamic COCO class IDs (kept consistent with mask_dataset.py and the C++ side)
DYNAMIC_COCO_IDS = {0, 1, 2, 3, 5, 7, 16, 17}


def load_detection_map(detection_dir):
    """Build a frame_id -> detections mapping."""
    det_map = {}
    if not os.path.isdir(detection_dir):
        print(f"  [WARN] detection directory not found: {detection_dir}")
        return det_map
    for fname in os.listdir(detection_dir):
        if not fname.endswith('.json'):
            continue
        try:
            frame_id = int(fname.replace('.json', ''))
        except ValueError:
            continue
        fpath = os.path.join(detection_dir, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            det_map[frame_id] = data.get("detections", [])
        except (json.JSONDecodeError, IOError):
            continue
    return det_map


def dynamic_detections(detections):
    """Return the bbox list [x1,y1,x2,y2] of all dynamic objects in this frame."""
    boxes = []
    for det in detections:
        if not det.get("is_dynamic", False):
            continue
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def mask_random_equivalent(image, boxes, rng, dilate_px=3):
    """Place rectangles with the same area as dynamic bboxes at random locations in the image and zero them out.

    Args:
        boxes: dynamic bbox list (same as mask_dataset.py, includes dilated regions)
        dilate_px: same dilation as YOLO-Mask, so the area measure matches
    """
    h, w = image.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        # Dilate first, same as YOLO-Mask, so the area measure aligns
        x1 = max(0, x1 - dilate_px)
        y1 = max(0, y1 - dilate_px)
        x2 = min(w, x2 + dilate_px)
        y2 = min(h, y2 + dilate_px)
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        # Pick a random rectangle of the same area (area = bw*bh, aspect ratio bounded like the bbox)
        # Simple and rigorous: place a bw x bh rectangle at a random location (exact same area).
        rx = rng.randint(0, w - bw) if w > bw else 0
        ry = rng.randint(0, h - bh) if h > bh else 0
        image[ry:ry + bh, rx:rx + bw] = 0
    return image


# ---- Data I/O below is kept consistent with mask_dataset.py; only the mask call changes ----

def mask_tum(dataset_dir, detection_dir, output_dir, rng):
    os.makedirs(output_dir, exist_ok=True)
    det_map = load_detection_map(detection_dir)
    if not det_map:
        return False
    rgb_dir = os.path.join(dataset_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        return False
    out_rgb = os.path.join(output_dir, "rgb")
    os.makedirs(out_rgb, exist_ok=True)
    for item in ["depth", "rgb.txt", "depth.txt", "associate.txt",
                 "groundtruth.txt", "accelerometer.txt"]:
        src = os.path.join(dataset_dir, item)
        if os.path.exists(src):
            dst = os.path.join(output_dir, item)
            if not os.path.exists(dst):
                if os.path.isdir(src):
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    frames = []
    rgb_txt = os.path.join(dataset_dir, "rgb.txt")
    if os.path.isfile(rgb_txt):
        with open(rgb_txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    frames.append((float(parts[0]), parts[1]))
    masked = 0
    for idx, (ts, rgb_path) in enumerate(frames):
        sp = os.path.join(dataset_dir, rgb_path)
        if not os.path.isfile(sp):
            continue
        img = cv2.imread(sp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        boxes = dynamic_detections(det_map.get(idx, []))
        if boxes:
            mask_random_equivalent(img, boxes, rng)
            masked += 1
        dp = os.path.join(output_dir, rgb_path)
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        cv2.imwrite(dp, img)
    print(f"  TUM done: {len(frames)} frames, {masked} with random masks")
    return True


def mask_kitti(dataset_dir, detection_dir, output_dir, rng):
    os.makedirs(output_dir, exist_ok=True)
    det_map = load_detection_map(detection_dir)
    if not det_map:
        return False
    for imgdir in ["image_0", "image_1"]:
        src_dir = os.path.join(dataset_dir, imgdir)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(output_dir, imgdir)
        os.makedirs(dst_dir, exist_ok=True)
        images = sorted(f for f in os.listdir(src_dir) if f.endswith('.png'))
        masked = 0
        for name in images:
            fid = int(name.replace('.png', ''))
            sp = os.path.join(src_dir, name)
            img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            boxes = dynamic_detections(det_map.get(fid, []))
            if boxes:
                mask_random_equivalent(img, boxes, rng)
                masked += 1
            cv2.imwrite(os.path.join(dst_dir, name), img)
        print(f"  KITTI {imgdir} done: {masked} frames with random masks")
    for item in ["times.txt", "calib.txt"]:
        src = os.path.join(dataset_dir, item)
        if os.path.exists(src):
            dst = os.path.join(output_dir, item)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    return True


def mask_euroc(dataset_dir, detection_dir, output_dir, rng):
    os.makedirs(output_dir, exist_ok=True)
    det_map = load_detection_map(detection_dir)
    if not det_map:
        return False
    for cam in ["cam0", "cam1"]:
        src_dir = os.path.join(dataset_dir, "mav0", cam, "data")
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(output_dir, "mav0", cam, "data")
        os.makedirs(dst_dir, exist_ok=True)
        images = sorted(f for f in os.listdir(src_dir) if f.endswith('.png'))
        masked = 0
        for name in images:
            fid = int(name.replace('.png', ''))
            sp = os.path.join(src_dir, name)
            img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            boxes = dynamic_detections(det_map.get(fid, []))
            if boxes:
                mask_random_equivalent(img, boxes, rng)
                masked += 1
            cv2.imwrite(os.path.join(dst_dir, name), img)
        print(f"  EuRoC {cam} done: {masked} frames with random masks")
    for cam in ["cam0", "cam1"]:
        src = os.path.join(dataset_dir, "mav0", cam, "sensor.yaml")
        if os.path.isfile(src):
            dst = os.path.join(output_dir, "mav0", cam, "sensor.yaml")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    for sub in ["imu0", "state_groundtruth_estimate0"]:
        src = os.path.join(dataset_dir, "mav0", sub)
        if os.path.isdir(src):
            dst = os.path.join(output_dir, "mav0", sub)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copytree(src, dst)
    return True


def detect_dataset_type(dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, "mav0", "cam0", "data")):
        return "euroc"
    if os.path.isdir(os.path.join(dataset_dir, "image_0")):
        return "kitti"
    if os.path.isfile(os.path.join(dataset_dir, "rgb.txt")):
        return "tum"
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate Random Mask control datasets (E1.6)")
    parser.add_argument("--detections")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--dataset-type", choices=["tum", "kitti", "euroc", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42, help="random seed (fixed for reproducibility)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--datasets", default="data/datasets")
    parser.add_argument("--detections-root", default="data/detections")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.all:
        sequences = [
            ("tum", "TUM/rgbd_dataset_freiburg3_sitting_static", "sitting_static"),
            ("tum", "TUM/rgbd_dataset_freiburg3_sitting_xyz", "sitting_xyz"),
            ("tum", "TUM/rgbd_dataset_freiburg3_walking_static", "walking_static"),
            ("tum", "TUM/rgbd_dataset_freiburg3_walking_xyz", "walking_xyz"),
            ("tum", "TUM/rgbd_dataset_freiburg3_walking_halfsphere", "walking_halfsphere"),
            ("kitti", "KITTI/00", "kitti_00"),
            ("euroc", "EuRoC/MH_01_easy", "euroc_mh01easy"),
            ("euroc", "EuRoC/MH_03_medium", "euroc_mh03medium"),
            ("euroc", "EuRoC/MH_05_difficult", "euroc_mh05difficult"),
        ]
        out_root = os.path.join(os.path.dirname(args.datasets), "datasets_randmask")
        for ds_type, rel, det_name in sequences:
            ds = os.path.join(args.datasets, rel)
            dp = os.path.join(args.detections_root, det_name)
            op = os.path.join(out_root, rel)
            if not os.path.isdir(ds) or not os.path.isdir(dp):
                print(f"[SKIP] {rel}")
                continue
            print(f"\n== {rel} ==")
            if ds_type == "tum":
                mask_tum(ds, dp, op, rng)
            elif ds_type == "kitti":
                mask_kitti(ds, dp, op, rng)
            elif ds_type == "euroc":
                mask_euroc(ds, dp, op, rng)
        return 0

    if not (args.detections and args.dataset and args.output):
        parser.error("single sequence mode requires --detections --dataset --output")
    ds_type = args.dataset_type
    if ds_type == "auto":
        ds_type = detect_dataset_type(args.dataset)
    if ds_type is None:
        print("[ERROR] could not detect dataset type")
        sys.exit(1)
    if ds_type == "tum":
        ok = mask_tum(args.dataset, args.detections, args.output, rng)
    elif ds_type == "kitti":
        ok = mask_kitti(args.dataset, args.detections, args.output, rng)
    elif ds_type == "euroc":
        ok = mask_euroc(args.dataset, args.detections, args.output, rng)
    else:
        print("[ERROR] unsupported dataset type")
        sys.exit(1)
    print("Done" if ok else "Failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()