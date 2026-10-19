#!/usr/bin/env python3
"""
mask_coverage.py -- report the masked-area fraction implied by a detection dir.
===============================================================================
For every per-frame detection JSON, this computes the fraction of image pixels
that would be zeroed by box masking: the UNION of the dilated dynamic bounding
boxes (dilation identical to mask_dataset.py, default 3 px). It reports the mean
and max fraction over all frames, the share of frames that contain any dynamic
object, and the mean number of dynamic detections per frame.

No SLAM run and no GPU are needed. This is used by the sweep scripts to fill the
"mask %" column of the ablation tables at run time (no value is pre-computed).

Image size is auto-detected from the dataset (TUM rgb/, KITTI image_0/,
EuRoC mav0/cam0/data/) or can be given explicitly with --width/--height.

Usage:
    python mask_coverage.py --detections <det_dir> --dataset <dataset_dir> \
        [--dataset-type auto|tum|kitti|euroc] [--dilate-px 3]

    # only the mean fraction as a bare number (for shell capture):
    python mask_coverage.py --detections <det_dir> --width 640 --height 480 \
        --dilate-px 3 --mean-only
"""

import argparse
import json
import os
import sys

import numpy as np


def detect_dataset_type(dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, "mav0", "cam0", "data")):
        return "euroc"
    if os.path.isdir(os.path.join(dataset_dir, "image_0")):
        return "kitti"
    if os.path.isfile(os.path.join(dataset_dir, "rgb.txt")):
        return "tum"
    return None


def first_image(dataset_dir, ds_type):
    """Return the path of one representative image so its size can be read."""
    if ds_type == "tum":
        d = os.path.join(dataset_dir, "rgb")
    elif ds_type == "kitti":
        d = os.path.join(dataset_dir, "image_0")
    else:  # euroc
        d = os.path.join(dataset_dir, "mav0", "cam0", "data")
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            return os.path.join(d, f)
    return None


def read_size(dataset_dir, ds_type):
    import cv2
    p = first_image(dataset_dir, ds_type)
    if p is None:
        return None
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def load_frames(det_dir):
    frames = []
    for fname in sorted(os.listdir(det_dir)):
        if not (fname.endswith(".json") and fname[:-5].isdigit()):
            continue
        with open(os.path.join(det_dir, fname)) as f:
            frames.append(json.load(f).get("detections", []))
    return frames


def main():
    ap = argparse.ArgumentParser(description="Masked-area coverage from detection JSONs")
    ap.add_argument("--detections", required=True)
    ap.add_argument("--dataset", help="dataset dir (to auto-detect image size)")
    ap.add_argument("--dataset-type", choices=["auto", "tum", "kitti", "euroc"], default="auto")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--dilate-px", type=int, default=3)
    ap.add_argument("--mean-only", action="store_true", help="print only the mean fraction")
    args = ap.parse_args()

    if not os.path.isdir(args.detections):
        raise SystemExit(f"[ERROR] detection dir not found: {args.detections}")

    w, h = args.width, args.height
    if w is None or h is None:
        if not args.dataset:
            raise SystemExit("[ERROR] provide --width/--height or --dataset for size detection.")
        ds_type = args.dataset_type
        if ds_type == "auto":
            ds_type = detect_dataset_type(args.dataset)
            if ds_type is None:
                raise SystemExit(f"[ERROR] could not detect dataset type: {args.dataset}")
        size = read_size(args.dataset, ds_type)
        if size is None:
            raise SystemExit(f"[ERROR] could not read an image from {args.dataset}")
        w, h = size

    frames = load_frames(args.detections)
    if not frames:
        raise SystemExit(f"[ERROR] no per-frame JSONs in {args.detections}")

    total_px = float(w * h)
    fracs = []
    n_dyn = []
    for dets in frames:
        canvas = np.zeros((h, w), dtype=np.uint8)
        dyn = 0
        for det in dets:
            if not det.get("is_dynamic", False):
                continue
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            dyn += 1
            x1 = max(0, int(bbox[0]) - args.dilate_px)
            y1 = max(0, int(bbox[1]) - args.dilate_px)
            x2 = min(w, int(bbox[2]) + args.dilate_px)
            y2 = min(h, int(bbox[3]) + args.dilate_px)
            if x2 > x1 and y2 > y1:
                canvas[y1:y2, x1:x2] = 1
        fracs.append(canvas.sum() / total_px)
        n_dyn.append(dyn)

    fracs = np.array(fracs)
    mean_frac = float(fracs.mean())
    if args.mean_only:
        print(f"{mean_frac:.6f}")
        return

    pct_frames = float((fracs > 0).mean() * 100.0)
    print(f"[INFO] frames: {len(frames)}  image: {w}x{h}  dilate: {args.dilate_px}px")
    print(f"[INFO] mean masked area : {mean_frac * 100:.2f}%")
    print(f"[INFO] max  masked area : {fracs.max() * 100:.2f}%")
    print(f"[INFO] frames w/ dynamic: {pct_frames:.1f}%")
    print(f"[INFO] mean dynamic/frame: {float(np.mean(n_dyn)):.3f}")


if __name__ == "__main__":
    main()
