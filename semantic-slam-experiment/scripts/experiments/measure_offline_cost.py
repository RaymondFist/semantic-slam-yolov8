#!/usr/bin/env python3
"""
measure_offline_cost.py -- end-to-end offline pre-processing cost per sequence.
================================================================================
Times the two offline stages of the delivered image-level pipeline and prints a
summary table. Neither stage runs inside ORB-SLAM3, so this quantifies the
offline overhead separately from the SLAM runtime:

    stage 1  YOLOv8-nano inference      scripts/yolov8_offline_inference.py
    stage 2  dynamic-region masking     scripts/mask_dataset.py

For each sequence the script reports the wall clock of both stages, the combined
wall clock, the per-frame latency, the throughput in frames per second and the
peak resident set size of the child processes. CPU time is taken from
resource.getrusage(RUSAGE_CHILDREN); peak RSS is the largest single child peak
observed so far.

Usage:
    # representative sequences (TUM walking_xyz, KITTI 00), paths from the env
    export SEMANTIC_ROOT=/path/to/semantic-slam-experiment
    export DATA_DIR=/path/to/data
    python3 scripts/experiments/measure_offline_cost.py --batch

    # extra TUM sequences
    python3 scripts/experiments/measure_offline_cost.py --batch \
        --seqs walking_xyz walking_static walking_halfsphere sitting_static sitting_xyz

    # a single sequence with explicit paths
    python3 scripts/experiments/measure_offline_cost.py \
        --dataset    $DATA_DIR/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --detections $DATA_DIR/detections/timing_walking_xyz \
        --masked     $DATA_DIR/datasets_masked/timing_walking_xyz \
        --dataset-type tum
"""

import argparse
import os
import resource
import subprocess
import sys
import time


def peak_rss_gb():
    """Largest single-child peak RSS seen so far, in GB."""
    kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        return kb / 1024.0 / 1024.0 / 1024.0
    return kb / 1024.0 / 1024.0


def count_frames(dataset, dataset_type):
    """Number of frames the offline stages will process."""
    if dataset_type == "tum":
        index = os.path.join(dataset, "rgb.txt")
        if not os.path.isfile(index):
            return 0
        # Mirror load_frame_list() in yolov8_offline_inference.py exactly: skip
        # comments/blanks and any line that does not carry timestamp + filename.
        n = 0
        with open(index) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and len(line.split()) >= 2:
                    n += 1
        return n
    img_dir = os.path.join(dataset, "image_2")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(dataset, "image_0")
    if not os.path.isdir(img_dir):
        return 0
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sum(1 for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts)


def run_stage(label, cmd, log_path, env):
    """Run one stage, echo its output to a log file, and return its timings."""
    print(f"[INFO] {label}: {' '.join(cmd)}")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with open(log_path, "w") as log:
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    wall = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    if ret != 0:
        print(f"[WARN] {label} exited with code {ret}; see {log_path}")
    return {"wall": wall, "cpu": cpu, "ret": ret}


def measure_one(root, dataset, detections, masked, dataset_type, model, conf, imgsz, device, dilate_px):
    """Measure the two offline stages for a single sequence."""
    frames = count_frames(dataset, dataset_type)
    os.makedirs(detections, exist_ok=True)
    os.makedirs(masked, exist_ok=True)
    env = dict(os.environ)

    infer = os.path.join(root, "scripts", "yolov8_offline_inference.py")
    mask = os.path.join(root, "scripts", "mask_dataset.py")
    if not os.path.isfile(infer) or not os.path.isfile(mask):
        raise SystemExit(f"[ERROR] expected scripts under {root}/scripts/")

    det = run_stage(
        "detection",
        # NOTE: yolov8_offline_inference.py spells this flag with an underscore
        # (--dataset_type); mask_dataset.py below uses the hyphen form.
        [sys.executable, infer, "--dataset", dataset, "--output", detections,
         "--dataset_type", dataset_type, "--model", model, "--conf", str(conf),
         "--imgsz", str(imgsz), "--device", str(device)],
        os.path.join(masked, "..", os.path.basename(masked) + "_detection.log"),
        env,
    )
    msk = run_stage(
        "masking",
        [sys.executable, mask, "--dataset", dataset, "--detections", detections,
         "--output", masked, "--dataset-type", dataset_type, "--dilate-px", str(dilate_px)],
        os.path.join(masked, "..", os.path.basename(masked) + "_masking.log"),
        env,
    )

    total = det["wall"] + msk["wall"]
    return {
        "frames": frames,
        "detect": det["wall"],
        "mask": msk["wall"],
        "total": total,
        "cpu": det["cpu"] + msk["cpu"],
        "rss": peak_rss_gb(),
        "ok": det["ret"] == 0 and msk["ret"] == 0,
    }


def default_targets(data_dir, seqs):
    """Build the (label, dataset, detections, masked, type) tuples from DATA_DIR."""
    tum_root = os.path.join(data_dir, "datasets", "TUM")
    targets = []
    for seq in seqs:
        if seq in ("kitti_00", "kitti"):
            continue  # handled below as a KITTI sequence, not rgbd_dataset_freiburg3_kitti_00
        targets.append((
            f"TUM {seq}",
            os.path.join(tum_root, f"rgbd_dataset_freiburg3_{seq}"),
            os.path.join(data_dir, "detections", f"timing_{seq}"),
            os.path.join(data_dir, "datasets_masked", "timing", seq),
            "tum",
        ))
    if "kitti_00" in seqs:
        targets.append((
            "KITTI 00",
            os.path.join(data_dir, "datasets", "KITTI", "00"),
            os.path.join(data_dir, "detections", "timing_kitti_00"),
            os.path.join(data_dir, "datasets_masked", "timing", "kitti_00"),
            "kitti",
        ))
    return targets


def print_summary(rows):
    """Print the offline-cost table in a fixed-width layout."""
    header = ["Sequence", "Frames", "Detect (s)", "Mask (s)", "Total (s)", "ms/frame", "FPS", "CPU (s)", "Peak RSS (GB)"]
    body = [header]
    for label, r in rows:
        if not r["ok"]:
            # A crashed stage yields meaningless startup-time numbers; mark the row
            # instead of printing them.
            body.append([label, str(r["frames"]), "FAILED", "FAILED", "FAILED",
                         "-", "-", "FAILED", "-"])
            continue
        if r["frames"]:
            ms = r["total"] / r["frames"] * 1000.0
            fps = r["frames"] / r["total"] if r["total"] > 0 else 0.0
        else:
            ms = fps = 0.0
        body.append([
            label,
            str(r["frames"]),
            f"{r['detect']:.2f}",
            f"{r['mask']:.2f}",
            f"{r['total']:.2f}",
            f"{ms:.1f}",
            f"{fps:.1f}",
            f"{r['cpu']:.2f}",
            f"{r['rss']:.3f}",
        ])
    widths = [max(len(row[c]) for row in body) for c in range(len(header))]
    line = "-+-".join("-" * w for w in widths)
    print()
    print("Offline pre-processing cost (detection + masking), separately from the SLAM runtime")
    print(" | ".join(cell.ljust(w) for cell, w in zip(body[0], widths)))
    print(line)
    for row in body[1:]:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def main():
    ap = argparse.ArgumentParser(description="Offline pre-processing cost (detection + masking)")
    ap.add_argument("--batch", action="store_true", help="measure the sequences under DATA_DIR")
    ap.add_argument("--seqs", nargs="+", default=["walking_xyz", "kitti_00"],
                    help="batch sequences: TUM short names plus optional kitti_00")
    ap.add_argument("--dataset", help="single-sequence: source dataset directory")
    ap.add_argument("--detections", help="single-sequence: output detection directory")
    ap.add_argument("--masked", help="single-sequence: output masked dataset directory")
    ap.add_argument("--dataset-type", choices=["tum", "kitti"], default="tum",
                    help="single-sequence dataset type")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="YOLOv8 weights")
    ap.add_argument("--conf", type=float, default=0.45, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    ap.add_argument("--dilate-px", type=int, default=3, help="bbox expansion in pixels")
    args = ap.parse_args()

    root = os.environ.get("SEMANTIC_ROOT", "")
    data_dir = os.environ.get("DATA_DIR", "")
    if not root or not os.path.isdir(root):
        raise SystemExit("[ERROR] SEMANTIC_ROOT is unset or not a directory")
    if args.batch and not (data_dir and os.path.isdir(data_dir)):
        raise SystemExit("[ERROR] DATA_DIR is unset or not a directory")

    if args.batch:
        targets = default_targets(data_dir, args.seqs)
    elif args.dataset and args.detections and args.masked:
        label = "TUM " + os.path.basename(args.dataset)
        targets = [(label, args.dataset, args.detections, args.masked, args.dataset_type)]
    else:
        ap.error("use --batch, or provide --dataset, --detections and --masked")

    print(f"[INFO] SEMANTIC_ROOT={root}")
    print(f"[INFO] model={args.model} conf={args.conf} imgsz={args.imgsz} dilate_px={args.dilate_px}")

    rows = []
    for label, dataset, detections, masked, dtype in targets:
        if not os.path.isdir(dataset):
            print(f"[SKIP] {label}: dataset not found ({dataset})")
            continue
        print(f"[INFO] === {label} ({dtype}) ===")
        r = measure_one(root, dataset, detections, masked, dtype,
                        args.model, args.conf, args.imgsz, args.device, args.dilate_px)
        rows.append((label, r))

    if not rows:
        raise SystemExit("[ERROR] no sequence was measured; check DATA_DIR / --seqs")
    print_summary(rows)

    failed = [label for label, r in rows if not r["ok"]]
    if failed:
        raise SystemExit(f"[ERROR] offline stage(s) failed for: {', '.join(failed)}. "
                         "The table above is not usable (see the *_detection.log / *_masking.log files).")


if __name__ == "__main__":
    main()
