#!/usr/bin/env python3
"""
mask_dataset_external.py -- apply externally supplied per-frame masks.
========================================================================
Builds a masked image dataset from a directory of per-frame mask files that were
produced by some other component (e.g. motion_mask.py), rather than from detector
output. The masks are applied with the same convention as the rest of the pipeline
(masked pixels set to zero, optional box/dilation handling), so a dataset built
here goes through exactly the same code path as the semantic-mask datasets.

Expected mask format (one file per frame, zero-padded frame index, where the
index matches the frame order used by `yolov8_offline_inference.py`):

  * PNG masks  :  <masks_dir>/<idx:06d>.png
        uint8 image, same size as the frame (or any size; it is resized with
        nearest-neighbour). Any non-zero pixel = region to zero out.
  * JSON boxes :  <masks_dir>/<idx:06d>.json
        {"boxes": [[x1, y1, x2, y2], ...]}  (box regions, expanded by --dilate-px)

Usage:
    python mask_dataset_external.py \
        --masks-dir <mask_dir> \
        --dataset <dataset_dir> --output <masked_out> \
        --dataset-type auto|tum|kitti|euroc \
        --masks-format auto|png|json --dilate-px 3

Environment:
    SEMANTIC_ROOT  repo root, to locate scripts/mask_dataset.py and the frame
                   enumerator in scripts/yolov8_offline_inference.py.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_scripts_dir(explicit):
    if explicit:
        return explicit
    root = os.environ.get("SEMANTIC_ROOT")
    if root and os.path.isfile(os.path.join(root, "scripts", "mask_dataset.py")):
        return os.path.join(root, "scripts")
    rel = os.path.dirname(EXP_DIR)
    if os.path.isfile(os.path.join(rel, "mask_dataset.py")):
        return rel
    raise SystemExit("[ERROR] mask_dataset.py not found; set SEMANTIC_ROOT or --scripts-dir.")


def resolve_format(masks_dir, fmt):
    if fmt != "auto":
        return fmt
    has_png = any(f.endswith(".png") for f in os.listdir(masks_dir))
    has_json = any(f.endswith(".json") for f in os.listdir(masks_dir))
    if has_png:
        return "png"
    if has_json:
        return "json"
    raise SystemExit(f"[ERROR] mask dir {masks_dir} contains neither .png nor .json files.")


def main():
    ap = argparse.ArgumentParser(description="Apply externally supplied per-frame masks")
    ap.add_argument("--masks-dir", required=True,
                    help="directory of per-frame masks (png) or boxes (json), see docstring")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset-type", choices=["auto", "tum", "kitti", "euroc"], default="auto")
    ap.add_argument("--masks-format", choices=["auto", "png", "json"], default="auto")
    ap.add_argument("--dilate-px", type=int, default=3)
    ap.add_argument("--scripts-dir", help="dir containing mask_dataset.py")
    args = ap.parse_args()

    if not os.path.isdir(args.masks_dir) or not os.listdir(args.masks_dir):
        raise SystemExit(
            f"[ERROR] mask dir missing or empty: {args.masks_dir}\n"
            "        Supply per-frame masks, expected either\n"
            "        <dir>/<idx:06d>.png (uint8, non-zero = masked) or\n"
            "        <dir>/<idx:06d>.json ({\"boxes\": [[x1,y1,x2,y2], ...]})\n"
            "        written by the upstream mask generator."
        )

    scripts_dir = find_scripts_dir(args.scripts_dir)
    sys.path.insert(0, scripts_dir)
    import mask_dataset as md
    from yolov8_offline_inference import load_frame_list

    fmt = resolve_format(args.masks_dir, args.masks_format)

    ds_type = args.dataset_type
    if ds_type == "auto":
        ds_type = md.detect_dataset_type(args.dataset)
        if ds_type is None:
            raise SystemExit(f"[ERROR] could not detect dataset type: {args.dataset}")

    frames = load_frame_list(args.dataset, args.dataset_type)

    # Build a synthetic detection directory understood by mask_dataset.*
    tmp_dir = tempfile.mkdtemp(prefix="external_masks_")
    n_masked_frames = 0
    try:
        for idx, _ in enumerate(frames):
            stem = f"{idx:06d}"
            dets = []
            if fmt == "png":
                p = os.path.join(args.masks_dir, stem + ".png")
                if os.path.isfile(p):
                    dets.append({"is_dynamic": True, "confidence": 1.0,
                                 "bbox": [0, 0, 0, 0], "mask_png": p})
            else:
                p = os.path.join(args.masks_dir, stem + ".json")
                if os.path.isfile(p):
                    with open(p) as f:
                        boxes = json.load(f).get("boxes", [])
                    for b in boxes:
                        if len(b) >= 4:
                            dets.append({"is_dynamic": True, "confidence": 1.0,
                                         "bbox": [float(v) for v in b[:4]]})
            if dets:
                n_masked_frames += 1
            with open(os.path.join(tmp_dir, stem + ".json"), "w") as f:
                json.dump({"frame_id": idx, "detections": dets,
                           "num_detections": len(dets),
                           "num_dynamic": len(dets)}, f)

        if n_masked_frames == 0:
            raise SystemExit(
                "[ERROR] mask dir supplied but no frame index matched the dataset "
                f"(checked {len(frames)} frames, format={fmt}). Check the file naming "
                "(<idx:06d>.png/.json) matches the pipeline frame order."
            )

        dilate = args.dilate_px

        def external_mask_image(image, detections, dilate_px=dilate):
            h, w = image.shape[:2]
            for det in detections:
                if not det.get("is_dynamic", False):
                    continue
                mpng = det.get("mask_png")
                if mpng:
                    m = cv2.imread(mpng, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    if dilate_px > 0:
                        k = 2 * dilate_px + 1
                        m = cv2.dilate(m, np.ones((k, k), np.uint8))
                    image[m > 0] = 0
                else:
                    x1 = max(0, int(det["bbox"][0]) - dilate_px)
                    y1 = max(0, int(det["bbox"][1]) - dilate_px)
                    x2 = min(w, int(det["bbox"][2]) + dilate_px)
                    y2 = min(h, int(det["bbox"][3]) + dilate_px)
                    if x2 > x1 and y2 > y1:
                        image[y1:y2, x1:x2] = 0
            return image

        md.mask_image = external_mask_image

        print(f"[INFO] external masking {args.dataset} -> {args.output} "
              f"(type={ds_type}, format={fmt}, {n_masked_frames}/{len(frames)} frames)")
        if ds_type == "tum":
            ok = md.mask_dataset_tum(args.dataset, tmp_dir, args.output, args.dilate_px)
        elif ds_type == "kitti":
            ok = md.mask_dataset_kitti(args.dataset, tmp_dir, args.output, args.dilate_px)
        elif ds_type == "euroc":
            ok = md.mask_dataset_euroc(args.dataset, tmp_dir, args.output, args.dilate_px)
        else:
            ok = False
        if not ok:
            raise SystemExit("[ERROR] external masking failed")
        print("[DONE] masked dataset written")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
