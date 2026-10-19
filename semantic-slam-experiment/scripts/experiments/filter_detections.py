#!/usr/bin/env python3
"""
filter_detections.py -- offline re-filtering of YOLOv8 detection JSONs (no GPU).
================================================================================
Rewrites a directory of per-frame detection JSONs into a new directory, applying
two independent filters and recomputing the per-detection `is_dynamic` flag and
the per-frame `num_dynamic` count:

  (a) class set   : which COCO class ids are considered dynamic
  (b) conf-min    : drop any detection whose confidence is below the threshold

This is a post-processing step. The detection JSONs store all detected classes,
not only the dynamic ones, so `is_dynamic` can be recomputed offline without
re-running inference. Lowering `--conf-min` below the confidence used at
inference time is not possible here (those boxes were never written); re-run
`yolov8_offline_inference.py` for that case.

Class sets:
  person          = {0}                        (people only)
  vehicle         = {1,2,3,5,7}                (bicycle,car,motorcycle,bus,truck)
  person,vehicle  = {0,1,2,3,5,7}              (people + road vehicles, no animals)
  full            = {0,1,2,3,5,7,16,17}        (the pipeline dynamic class set)

Usage:
    python filter_detections.py \
        --input  <src_detection_dir> \
        --output <dst_detection_dir> \
        --classes full \
        [--conf-min 0.45]

    # or an explicit id list / name list:
    python filter_detections.py --input in --output out --classes 0,1,2

The TUM/KITTI/EuRoC detection directories all share the same JSON schema, so
this script is dataset-agnostic.
"""

import argparse
import json
import os
import sys


# ---- COCO classes relevant to the dynamic set (name <-> id) ----
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 16: "dog", 17: "cat",
}

CLASS_SETS = {
    "person": {0},
    "vehicle": {1, 2, 3, 5, 7},
    "person,vehicle": {0, 1, 2, 3, 5, 7},
    "full": {0, 1, 2, 3, 5, 7, 16, 17},  # the pipeline dynamic class set
}


def parse_classes(token):
    """Resolve --classes into a set of class ids.

    Accepts a preset key from CLASS_SETS, or an explicit comma list of ids
    and/or COCO names (e.g. "0,1,bus").
    """
    token = token.strip()
    if token in CLASS_SETS:
        return set(CLASS_SETS[token])
    name_to_id = {v: k for k, v in COCO_NAMES.items()}
    ids = set()
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            ids.add(int(part))
        elif part in name_to_id:
            ids.add(name_to_id[part])
        else:
            raise SystemExit(
                f"[ERROR] unknown class token '{part}'. Use a preset "
                f"{sorted(CLASS_SETS)} or ids/names from {sorted(COCO_NAMES.values())}."
            )
    if not ids:
        raise SystemExit("[ERROR] --classes resolved to an empty set.")
    return ids


def is_frame_json(fname):
    base = fname[:-5] if fname.endswith(".json") else None
    return base is not None and base.isdigit()


def process_dir(in_dir, out_dir, dyn_ids, conf_min):
    if not os.path.isdir(in_dir):
        raise SystemExit(f"[ERROR] input detection dir not found: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(in_dir) if is_frame_json(f))
    if not frame_files:
        raise SystemExit(f"[ERROR] no per-frame JSON files found in {in_dir}")

    n_frames = 0
    n_det_in = 0
    n_det_out = 0
    n_dyn_out = 0

    for fname in frame_files:
        with open(os.path.join(in_dir, fname)) as f:
            data = json.load(f)

        kept = []
        for det in data.get("detections", []):
            n_det_in += 1
            conf = float(det.get("confidence", 0.0))
            if conf_min is not None and conf < conf_min:
                continue
            cls_id = int(det.get("class_id", -1))
            det["is_dynamic"] = cls_id in dyn_ids
            kept.append(det)

        num_dynamic = sum(1 for d in kept if d.get("is_dynamic"))
        data["detections"] = kept
        data["num_detections"] = len(kept)
        data["num_dynamic"] = num_dynamic

        n_frames += 1
        n_det_out += len(kept)
        n_dyn_out += num_dynamic

        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(data, f, indent=2)

    print(f"[INFO] frames: {n_frames}")
    print(f"[INFO] detections: {n_det_in} in -> {n_det_out} kept "
          f"({n_det_in - n_det_out} dropped by conf-min)")
    print(f"[INFO] dynamic detections (after filter): {n_dyn_out}")
    print(f"[INFO] class set: {sorted(dyn_ids)}  conf-min: {conf_min}")
    print(f"[DONE] wrote filtered detections to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Offline re-filter of YOLOv8 detection JSONs")
    ap.add_argument("--input", required=True, help="source detection JSON directory")
    ap.add_argument("--output", required=True, help="destination detection JSON directory")
    ap.add_argument("--classes", default="full",
                    help="class set: person | vehicle | person,vehicle | full | explicit ids/names")
    ap.add_argument("--conf-min", type=float, default=None,
                    help="drop detections with confidence below this value (upward filter only)")
    args = ap.parse_args()

    dyn_ids = parse_classes(args.classes)
    process_dir(args.input, args.output, dyn_ids, args.conf_min)


if __name__ == "__main__":
    main()
