#!/usr/bin/env python3
"""
mask_dataset_instance.py -- instance-mask a dataset from YOLOv8-seg masks.
=========================================================================
Study 5 (box mask vs instance/segmentation mask). The repository's
`scripts/mask_dataset.py` zeroes the whole bounding box of each dynamic object;
this variant zeroes only the pixels inside the object's segmentation mask.

This script combines the two steps of the instance-masking study:

  1. Seg inference. If the detection directory is missing or empty, it runs
     YOLOv8-seg inference (the same logic/classes/behaviour as the repository
     inference script) and, for every dynamic object, writes a uint8 PNG mask
     (0/255, original resolution, from the YOLO polygon `masks.xy`) to
     `<detections>/masks/<frame:06d>_<obj:03d>.png`, storing the relative path
     in the detection JSON under `mask_png`. Existing detection fields
     (frame_id, timestamp, image_path, detections[...], num_detections,
     num_dynamic) are unchanged.
     The dynamic class set is the pipeline dynamic class set {0,1,2,3,5,7,16,17}.
  2. Instance masking. Reuses the dataset-level writers of `scripts/mask_dataset.py`
     with a patched per-image masking function that zeroes only the pixels
     covered by each dynamic object's `mask_png` (not the bbox).

Usage:
    python mask_dataset_instance.py \
        --dataset <dataset_dir> --output <masked_out> [--conf 0.45] [--detections <det_dir>]

    # detections defaults to $DATA_DIR/detections/<dataset_name>_segmasks
    python mask_dataset_instance.py --dataset <ds> --output <masked_out>

Environment:
    DATA_DIR       data root, used for the default --detections location.
    SEMANTIC_ROOT  repo root, to locate scripts/yolov8_offline_inference.py and
                   scripts/mask_dataset.py (or pass --scripts-dir).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DYNAMIC = "0,1,2,3,5,7,16,17"  # the pipeline dynamic class set
DEFAULT_MODEL = "yolov8n-seg.pt"
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "0"
DILATE_PX = 3


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


def parse_ids(token):
    return {int(x) for x in token.split(",") if x.strip()}


def default_detections(dataset):
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        return None
    name = os.path.basename(os.path.normpath(dataset))
    return os.path.join(data_dir, "detections", f"{name}_segmasks")


def has_frames(det_dir):
    if not os.path.isdir(det_dir):
        return False
    return any(f.endswith(".json") and f[:-5].isdigit() for f in os.listdir(det_dir))


def run_seg_inference(dataset, output, conf, dataset_type, dynamic_classes):
    from yolov8_offline_inference import load_frame_list
    from ultralytics import YOLO

    dyn_ids = parse_ids(dynamic_classes)
    print(f"[INFO] loading model: {DEFAULT_MODEL}")
    model = YOLO(DEFAULT_MODEL)

    frames = load_frame_list(dataset, dataset_type)
    os.makedirs(output, exist_ok=True)
    mask_dir = os.path.join(output, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    total_det = total_dyn = 0
    for idx, (timestamp, rgb_path) in enumerate(frames):
        img_path = os.path.join(dataset, rgb_path)
        if not os.path.exists(img_path):
            continue
        results = model(img_path, verbose=False, conf=conf, iou=DEFAULT_IOU,
                        imgsz=DEFAULT_IMGSZ, device=DEFAULT_DEVICE, half=False)

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for bi in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[bi])
                    dconf = float(r.boxes.conf[bi])
                    x1, y1, x2, y2 = r.boxes.xyxy[bi].cpu().numpy().tolist()
                    is_dynamic = cls_id in dyn_ids

                    mask_shape = None
                    if is_dynamic and r.masks is not None and bi < len(r.masks.data):
                        mask_shape = list(r.masks.data[bi].cpu().numpy().shape)

                    detections.append({
                        "class_id": cls_id,
                        "class_name": r.names.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(dconf, 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "is_dynamic": is_dynamic,
                        "mask_bbox_shape": mask_shape,
                        "mask_png": None,  # filled below for dynamic objects
                    })

                    # Save the instance mask (original resolution) for dynamic objects.
                    if is_dynamic and r.masks is not None and bi < len(r.masks.xy):
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        h, w = img.shape[:2]
                        poly = np.asarray(r.masks.xy[bi], dtype=np.float32)
                        if poly.size >= 3:
                            canvas = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(canvas, [poly.astype(np.int32)], 255)
                            rel = os.path.join("masks", f"{idx:06d}_{bi:03d}.png")
                            cv2.imwrite(os.path.join(output, rel), canvas)
                            detections[-1]["mask_png"] = rel

        num_dynamic = sum(1 for d in detections if d["is_dynamic"])
        total_det += len(detections)
        total_dyn += num_dynamic

        with open(os.path.join(output, f"{idx:06d}.json"), "w") as f:
            json.dump({
                "frame_id": idx,
                "timestamp": timestamp,
                "image_path": rgb_path,
                "detections": detections,
                "num_detections": len(detections),
                "num_dynamic": num_dynamic,
            }, f, indent=2)

        if (idx + 1) % 50 == 0 or idx == len(frames) - 1:
            print(f"  [{idx+1}/{len(frames)}] det={len(detections)} dyn={num_dynamic}")

    with open(os.path.join(output, "_summary.json"), "w") as f:
        json.dump({
            "dataset": dataset, "model": DEFAULT_MODEL,
            "conf_threshold": conf, "iou_threshold": DEFAULT_IOU,
            "total_frames": len(frames), "total_detections": total_det,
            "total_dynamic": total_dyn, "dynamic_class_ids": sorted(dyn_ids),
            "saves_instance_masks": True,
        }, f, indent=2)

    print(f"[DONE] {len(frames)} frames; det={total_det} dyn={total_dyn}; masks in {mask_dir}")


def mask_instances(dataset, detections, output, dataset_type):
    import mask_dataset as md

    mask_root = detections

    def instance_mask_image(image, dets, dilate_px=DILATE_PX):
        h, w = image.shape[:2]
        for det in dets:
            if not det.get("is_dynamic", False):
                continue
            rel = det.get("mask_png")
            if not rel:
                continue
            mp = rel if os.path.isabs(rel) else os.path.join(mask_root, rel)
            if not os.path.isfile(mp):
                continue
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            if dilate_px > 0:
                k = 2 * dilate_px + 1
                m = cv2.dilate(m, np.ones((k, k), np.uint8))
            image[m > 0] = 0
        return image

    md.mask_image = instance_mask_image

    print(f"[INFO] instance-masking {dataset} -> {output} "
          f"(type={dataset_type}, dilate={DILATE_PX}px, mask-root={mask_root})")
    if dataset_type == "tum":
        ok = md.mask_dataset_tum(dataset, detections, output, DILATE_PX)
    elif dataset_type == "kitti":
        ok = md.mask_dataset_kitti(dataset, detections, output, DILATE_PX)
    else:
        ok = False
    if not ok:
        raise SystemExit("[ERROR] masking failed")
    print("[DONE] instance-masked dataset written")


def main():
    ap = argparse.ArgumentParser(description="Instance-mask a dataset from YOLOv8-seg masks")
    ap.add_argument("--dataset", required=True, help="source dataset directory")
    ap.add_argument("--output", required=True, help="output directory for the masked dataset")
    ap.add_argument("--detections", default=None,
                    help="seg detection dir (default: $DATA_DIR/detections/<dataset_name>_segmasks)")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--dataset-type", choices=["tum", "kitti"], default="tum")
    ap.add_argument("--scripts-dir", help="dir containing yolov8_offline_inference.py and mask_dataset.py")
    args = ap.parse_args()

    if not args.detections:
        args.detections = default_detections(args.dataset)
        if not args.detections:
            raise SystemExit("[ERROR] --detections omitted and DATA_DIR not set; pass --detections.")

    scripts_dir = find_scripts_dir(args.scripts_dir)
    sys.path.insert(0, scripts_dir)

    if not has_frames(args.detections):
        print(f"[INFO] detections missing/empty at {args.detections}; running YOLOv8-seg inference")
        run_seg_inference(args.dataset, args.detections, args.conf,
                          args.dataset_type, DEFAULT_DYNAMIC)

    mask_instances(args.dataset, args.detections, args.output, args.dataset_type)


if __name__ == "__main__":
    main()
