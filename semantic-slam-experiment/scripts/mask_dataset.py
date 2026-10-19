#!/usr/bin/env python3
"""
Dynamic Region Masking Script for Datasets (offline dynamic-region masking)
==============================================
Reads YOLOv8 detection JSON and zeros out (blackens) dynamic object regions in the RGB images,
producing a masked copy of the dataset. The masked dataset is run directly with stock ORB-SLAM3,
no SemanticSLAM patch required.

Supports: TUM RGB-D, KITTI, EuRoC

Usage:
    python3 scripts/mask_dataset.py \
        --detections data/detections/walking_xyz \
        --dataset   data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --output    data/datasets_masked/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --dataset-type tum

    # Batch processing:
    python3 scripts/mask_dataset.py --all --datasets data/datasets --detections data/detections --output data/datasets_masked
"""

import os
import sys
import json
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path


# Dynamic COCO class IDs
DYNAMIC_COCO_IDS = {0, 1, 2, 3, 5, 7, 16, 17}

# Bounding-box expansion (pixels) applied when zeroing dynamic regions.
DEFAULT_DILATE_PX = 3


def load_detection_map(detection_dir):
    """Load all detection JSON files and build a frame_id -> detections mapping."""
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
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] failed to read {fpath}: {e}")

    return det_map


def mask_image(image, detections, dilate_px=DEFAULT_DILATE_PX):
    """Zero out dynamic object bbox regions in the image.

    Args:
        image: BGR numpy array
        detections: YOLO detection list
        dilate_px: pixels to expand the bbox outward (avoid leftover edge features)

    Returns:
        masked image (in-place modified)
    """
    h, w = image.shape[:2]

    for det in detections:
        if not det.get("is_dynamic", False):
            continue

        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue

        x1 = max(0, int(bbox[0]) - dilate_px)
        y1 = max(0, int(bbox[1]) - dilate_px)
        x2 = min(w, int(bbox[2]) + dilate_px)
        y2 = min(h, int(bbox[3]) + dilate_px)

        if x2 > x1 and y2 > y1:
            image[y1:y2, x1:x2] = 0

    return image


def mask_dataset_tum(dataset_dir, detection_dir, output_dir, dilate_px=DEFAULT_DILATE_PX):
    """Mask a TUM RGB-D dataset.

    TUM directory layout:
        rgb/              <- RGB images (to be masked)
        depth/            <- depth images (copied/linked directly)
        rgb.txt           <- frame list
        depth.txt
        associate.txt
        groundtruth.txt
        accelerometer.txt (optional)
    """
    os.makedirs(output_dir, exist_ok=True)

    det_map = load_detection_map(detection_dir)
    if not det_map:
        print(f"  [ERROR] no detection data, skipping")
        return False

    rgb_dir = os.path.join(dataset_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"  [ERROR] RGB directory not found: {rgb_dir}")
        return False

    # Create the output RGB directory
    output_rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(output_rgb_dir, exist_ok=True)

    # Copy/link other files
    for item in ["depth", "rgb.txt", "depth.txt", "associate.txt",
                 "groundtruth.txt", "accelerometer.txt"]:
        src = os.path.join(dataset_dir, item)
        if os.path.exists(src):
            dst = os.path.join(output_dir, item)
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    # Depth directory needs no changes; copy as a symlink or fall back to a hard copy
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copytree(src, dst)
            else:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # Read rgb.txt for the frame order
    rgb_txt = os.path.join(dataset_dir, "rgb.txt")
    frames = []
    if os.path.isfile(rgb_txt):
        with open(rgb_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    frames.append((float(parts[0]), parts[1]))

    print(f"  Processing {len(frames)} frames...")

    masked_count = 0
    for idx, (timestamp, rgb_path) in enumerate(frames):
        src_path = os.path.join(dataset_dir, rgb_path)
        if not os.path.isfile(src_path):
            continue

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # frame_id matches the file name number of the detection JSON
        dets = det_map.get(idx, [])

        if dets:
            mask_image(img, dets, dilate_px)
            masked_count += 1

        # Save the masked image
        dst_path = os.path.join(output_dir, rgb_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img)

        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(frames)} frames done ({masked_count} with dynamic objects)")

    print(f"  Done: {len(frames)} frames, {masked_count} with dynamic objects masked")
    return True


def mask_dataset_kitti(dataset_dir, detection_dir, output_dir, dilate_px=DEFAULT_DILATE_PX):
    """Mask a KITTI Odometry dataset.

    KITTI directory layout:
        image_0/    <- left grayscale images (to be masked)
        image_1/    <- right grayscale images (to be masked)
        times.txt
        calib.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    det_map = load_detection_map(detection_dir)
    if not det_map:
        print(f"  [ERROR] no detection data, skipping")
        return False

    for img_dir_name in ["image_0", "image_1"]:
        src_dir = os.path.join(dataset_dir, img_dir_name)
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(output_dir, img_dir_name)
        os.makedirs(dst_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
        print(f"  Processing {img_dir_name}: {len(images)} frames...")

        masked_count = 0
        for img_name in images:
            frame_id = int(img_name.replace('.png', ''))
            src_path = os.path.join(src_dir, img_name)

            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            dets = det_map.get(frame_id, [])
            if dets:
                # Grayscale images would need a BGR conversion for color assignment, but black is 0 in grayscale too
                mask_image(img, dets, dilate_px)
                masked_count += 1

            dst_path = os.path.join(dst_dir, img_name)
            cv2.imwrite(dst_path, img)

        print(f"    {img_dir_name} done: {masked_count} frames with dynamic objects masked")

    # Copy other files
    for item in ["times.txt", "calib.txt"]:
        src = os.path.join(dataset_dir, item)
        if os.path.exists(src):
            dst = os.path.join(output_dir, item)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    return True


def mask_dataset_euroc(dataset_dir, detection_dir, output_dir, dilate_px=DEFAULT_DILATE_PX):
    """Mask an EuRoC MAV dataset.

    EuRoC directory layout:
        mav0/cam0/data/   <- grayscale images (to be masked)
        mav0/cam0/sensor.yaml
        mav0/cam1/data/   <- grayscale images (optional)
        mav0/imu0/data.csv
        ...
    """
    os.makedirs(output_dir, exist_ok=True)

    det_map = load_detection_map(detection_dir)
    if not det_map:
        print(f"  [ERROR] no detection data, skipping")
        return False

    for cam_name in ["cam0", "cam1"]:
        src_dir = os.path.join(dataset_dir, "mav0", cam_name, "data")
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(output_dir, "mav0", cam_name, "data")
        os.makedirs(dst_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
        print(f"  Processing mav0/{cam_name}/data: {len(images)} frames...")

        masked_count = 0
        for img_name in images:
            frame_id = int(img_name.replace('.png', ''))
            src_path = os.path.join(src_dir, img_name)

            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            dets = det_map.get(frame_id, [])
            if dets:
                mask_image(img, dets, dilate_px)
                masked_count += 1

            dst_path = os.path.join(dst_dir, img_name)
            cv2.imwrite(dst_path, img)

        print(f"    mav0/{cam_name}/data done: {masked_count} frames with dynamic objects masked")

    # Copy sensor.yaml
    for cam_name in ["cam0", "cam1"]:
        src = os.path.join(dataset_dir, "mav0", cam_name, "sensor.yaml")
        if os.path.isfile(src):
            dst = os.path.join(output_dir, "mav0", cam_name, "sensor.yaml")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Copy IMU data and ground truth
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
    """Detect the dataset type automatically."""
    if os.path.isdir(os.path.join(dataset_dir, "mav0", "cam0", "data")):
        return "euroc"
    elif os.path.isdir(os.path.join(dataset_dir, "image_0")):
        return "kitti"
    elif os.path.isfile(os.path.join(dataset_dir, "rgb.txt")):
        return "tum"
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Mask dynamic regions in datasets (offline dynamic-region masking)")
    parser.add_argument("--detections", help="detection JSON directory")
    parser.add_argument("--dataset", help="source dataset directory")
    parser.add_argument("--output", help="output directory for the masked dataset")
    parser.add_argument("--dataset-type", choices=["tum", "kitti", "euroc", "auto"],
                        default="auto", help="dataset type (default: auto detect)")
    parser.add_argument("--dilate-px", type=int, default=DEFAULT_DILATE_PX,
                        help="bbox expansion in pixels (default: 3)")
    parser.add_argument("--all", action="store_true",
                        help="batch process all known sequences")
    parser.add_argument("--datasets", default="data/datasets",
                        help="dataset root directory (--all mode)")
    parser.add_argument("--detections-root", default="data/detections",
                        help="detection results root directory (--all mode)")
    args = parser.parse_args()

    if args.all:
        # Batch mode
        return batch_mask_all(args)

    # Single sequence mode
    if not args.detections or not args.dataset or not args.output:
        parser.error("single sequence mode requires --detections, --dataset, --output")

    ds_type = args.dataset_type
    if ds_type == "auto":
        ds_type = detect_dataset_type(args.dataset)

    if ds_type is None:
        print(f"[ERROR] could not detect dataset type: {args.dataset}")
        sys.exit(1)

    print(f"Masking dataset: {args.dataset} -> {args.output}")
    print(f"  Type: {ds_type}")
    print(f"  Detections: {args.detections}")

    if ds_type == "tum":
        ok = mask_dataset_tum(args.dataset, args.detections, args.output, args.dilate_px)
    elif ds_type == "kitti":
        ok = mask_dataset_kitti(args.dataset, args.detections, args.output, args.dilate_px)
    elif ds_type == "euroc":
        ok = mask_dataset_euroc(args.dataset, args.detections, args.output, args.dilate_px)
    else:
        print(f"[ERROR] unsupported dataset type: {ds_type}")
        sys.exit(1)

    if ok:
        print(f"\nMasking complete: {args.output}")
    else:
        print(f"\nMasking failed")
        sys.exit(1)


def batch_mask_all(args):
    """Batch process all known training/evaluation sequences."""
    datasets_root = args.datasets
    detections_root = args.detections_root
    output_root = os.path.join(os.path.dirname(datasets_root), "datasets_masked")

    # Define all sequences: (dataset type, dataset path, detection dir name)
    sequences = [
        # TUM
        ("tum", "TUM/rgbd_dataset_freiburg3_sitting_static", "sitting_static"),
        ("tum", "TUM/rgbd_dataset_freiburg3_sitting_xyz", "sitting_xyz"),
        ("tum", "TUM/rgbd_dataset_freiburg3_walking_static", "walking_static"),
        ("tum", "TUM/rgbd_dataset_freiburg3_walking_xyz", "walking_xyz"),
        ("tum", "TUM/rgbd_dataset_freiburg3_walking_halfsphere", "walking_halfsphere"),
        # KITTI
        ("kitti", "KITTI/00", "kitti_00"),
        # EuRoC
        ("euroc", "EuRoC/MH_01_easy", "euroc_mh01easy"),
        ("euroc", "EuRoC/MH_03_medium", "euroc_mh03medium"),
        ("euroc", "EuRoC/MH_05_difficult", "euroc_mh05difficult"),
    ]

    total = 0
    success = 0
    for ds_type, ds_rel_path, det_name in sequences:
        ds_path = os.path.join(datasets_root, ds_rel_path)
        det_path = os.path.join(detections_root, det_name)
        out_path = os.path.join(output_root, ds_rel_path)

        if not os.path.isdir(ds_path):
            print(f"\n[SKIP] dataset not found: {ds_path}")
            continue
        if not os.path.isdir(det_path):
            print(f"\n[SKIP] detection data not found: {det_path}")
            continue

        total += 1
        print(f"\n{'='*60}")
        print(f"[{total}] {ds_type.upper()}: {ds_rel_path}")
        print(f"{'='*60}")

        if ds_type == "tum":
            ok = mask_dataset_tum(ds_path, det_path, out_path, args.dilate_px)
        elif ds_type == "kitti":
            ok = mask_dataset_kitti(ds_path, det_path, out_path, args.dilate_px)
        elif ds_type == "euroc":
            ok = mask_dataset_euroc(ds_path, det_path, out_path, args.dilate_px)
        else:
            ok = False

        if ok:
            success += 1

    print(f"\n{'='*60}")
    print(f"Batch masking done: {success}/{total} succeeded")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}")

    return 0 if success == total else 1


if __name__ == "__main__":
    main()