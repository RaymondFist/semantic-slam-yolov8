#!/usr/bin/env python3
"""
motion_mask.py -- detector-independent dynamic-region masks from RGB-D geometry.
===============================================================================
No learned detector and no YOLO inference is involved. For every consecutive pair
of frames (a -> b) the registered depth map of frame a, together with the provided
ground-truth camera trajectory, is used to reproject every depth pixel into frame
b. A pixel is flagged as dynamic when the depth it reprojects to disagrees with the
depth actually measured at that image location, i.e. when the surface there did not
obey the rigid camera motion. Each frame collects candidates from both the pair it
opens (i -> i+1) and the pair it closes (i-1 -> i).

Output matches the format consumed by mask_dataset_external.py:
    <out_dir>/<idx:06d>.png     uint8, non-zero = dynamic region
Frame indices follow rgb.txt order, which is the enumerator used by
yolov8_offline_inference.py.

These masks are geometric, not ground truth, so this row is reported as
a "geometric motion mask" comparison and makes no upper-bound claim.

Usage:
    python3 scripts/experiments/motion_mask.py \
        --dataset data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --output  data/motion_masks/walking_xyz \
        --calib   src/config/TUM3_baseline.yaml
"""

import argparse
import bisect
import math
import os

import cv2
import numpy as np


def read_index(path):
    """Read a TUM index file (rgb.txt / depth.txt): timestamp + relative path."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), parts[1]))
    return rows


def read_trajectory(path):
    """Read groundtruth.txt: timestamp tx ty tz qx qy qz qw (camera-to-world)."""
    ts, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                v = [float(x) for x in parts[:8]]
                ts.append(v[0])
                poses.append((np.array(v[1:4], dtype=np.float64),
                              np.array(v[4:8], dtype=np.float64)))
    return ts, poses


def quat_to_matrix(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0, q1, a):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(max(-1.0, min(1.0, d))) * a
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def interpolate_pose(ts, poses, t):
    """Camera-to-world pose at time t: (R, t) with the translation as a 3-vector."""
    idx = bisect.bisect_left(ts, t)
    if idx <= 0:
        i0 = i1 = 0
    elif idx >= len(ts):
        i0 = i1 = len(ts) - 1
    else:
        i0, i1 = idx - 1, idx
    if i0 == i1 or ts[i1] <= ts[i0]:
        return quat_to_matrix(poses[i0][1]), poses[i0][0].copy()
    a = (t - ts[i0]) / (ts[i1] - ts[i0])
    a = max(0.0, min(1.0, a))
    trans = (1.0 - a) * poses[i0][0] + a * poses[i1][0]
    return quat_to_matrix(slerp(poses[i0][1], poses[i1][1], a)), trans


def load_calib(path):
    """Intrinsics and depth scale; falls back to the standard TUM fr3 values."""
    calib = {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6, "depth_factor": 5000.0}
    if path and os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key, val = key.strip(), val.strip()
                if key in ("Camera.fx", "Camera.fy", "Camera.cx", "Camera.cy"):
                    calib[key.split(".")[1]] = float(val)
                elif key == "DepthMapFactor":
                    calib["depth_factor"] = float(val)
    return calib


def nearest_index(times, t):
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def clean_mask(flag, min_area):
    """Close small holes, then drop components below min_area."""
    m = flag.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out[labels == k] = 255
    return out


def main():
    ap = argparse.ArgumentParser(description="Geometric dynamic-region masks (no detector)")
    ap.add_argument("--dataset", required=True, help="TUM sequence directory")
    ap.add_argument("--output", required=True, help="output mask directory")
    ap.add_argument("--calib", default="", help="ORB-SLAM3 TUM yaml (intrinsics, depth factor)")
    ap.add_argument("--tau-abs", type=float, default=0.15, help="absolute depth tolerance (m)")
    ap.add_argument("--tau-rel", type=float, default=0.05, help="relative depth tolerance")
    ap.add_argument("--min-area", type=int, default=300, help="discard components smaller than this (px)")
    ap.add_argument("--min-depth", type=float, default=0.3, help="valid depth lower bound (m)")
    ap.add_argument("--max-depth", type=float, default=8.0, help="valid depth upper bound (m)")
    ap.add_argument("--skip-existing", action="store_true", help="do not overwrite existing masks")
    args = ap.parse_args()

    ds = args.dataset
    rgb = read_index(os.path.join(ds, "rgb.txt"))
    dep = read_index(os.path.join(ds, "depth.txt"))
    gts, gposes = read_trajectory(os.path.join(ds, "groundtruth.txt"))
    if not rgb or not dep or not gts:
        raise SystemExit(f"[ERROR] rgb.txt / depth.txt / groundtruth.txt missing under {ds}")

    calib = load_calib(args.calib)
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    factor = calib["depth_factor"]
    print(f"[INFO] calib fx={fx} fy={fy} cx={cx} cy={cy} depth_factor={factor}")
    print(f"[INFO] frames: rgb={len(rgb)} depth={len(dep)} gt={len(gts)}")

    dep_ts = [t for t, _ in dep]
    os.makedirs(args.output, exist_ok=True)

    cache = {}

    def depth_at(i):
        if i not in cache:
            j = nearest_index(dep_ts, rgb[i][0])
            img = cv2.imread(os.path.join(ds, dep[j][1]), cv2.IMREAD_UNCHANGED)
            cache[i] = None if img is None else img.astype(np.float32) / factor
        return cache[i]

    first = depth_at(0)
    if first is None:
        raise SystemExit("[ERROR] could not read the first depth image")
    h, w = first.shape[:2]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

    def pair_flags(ia, ib):
        """(flags on frame ia, flags on frame ib) for the pair ia -> ib."""
        da, db = depth_at(ia), depth_at(ib)
        if da is None or db is None:
            return None, None
        if da.shape[:2] != (h, w):
            da = cv2.resize(da, (w, h), interpolation=cv2.INTER_NEAREST)
        if db.shape[:2] != (h, w):
            db = cv2.resize(db, (w, h), interpolation=cv2.INTER_NEAREST)
        ra, ta = interpolate_pose(gts, gposes, rgb[ia][0])
        rb, tb = interpolate_pose(gts, gposes, rgb[ib][0])

        valid = (da > args.min_depth) & (da < args.max_depth)
        pts = np.empty((3, h * w), dtype=np.float64)
        pts[0] = ((uu - cx) / fx * da).reshape(-1)
        pts[1] = ((vv - cy) / fy * da).reshape(-1)
        pts[2] = da.reshape(-1)
        # camera a -> world -> camera b
        xc = (rb.T @ (ra @ pts + ta[:, None] - tb[:, None])).reshape(3, h, w)
        z = xc[2]
        zz = np.where(np.abs(z) > 1e-6, z, 1e-6)
        ui = np.rint(fx * xc[0] / zz + cx).astype(np.int32)
        vi = np.rint(fy * xc[1] / zz + cy).astype(np.int32)
        inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 0)
        uc = np.clip(ui, 0, w - 1)
        vc = np.clip(vi, 0, h - 1)
        sampled = db[vc, uc]
        ok = inb & (sampled > args.min_depth) & (sampled < args.max_depth)
        mismatch = np.abs(z - sampled) > (args.tau_abs + args.tau_rel * z)
        src = valid & ok & mismatch
        dst = np.zeros((h, w), dtype=bool)
        dst[vc[src], uc[src]] = True
        return src, dst

    written = 0
    cover = 0.0
    pending_dst = None
    for i in range(len(rgb)):
        mask = np.zeros((h, w), dtype=bool)
        src_cur, dst_next = (None, None)
        if i + 1 < len(rgb):
            src_cur, dst_next = pair_flags(i, i + 1)
        if src_cur is not None:
            mask |= src_cur
        if pending_dst is not None:
            mask |= pending_dst

        m = clean_mask(mask, args.min_area)
        cover += float((m > 0).sum()) / (h * w)
        out_path = os.path.join(args.output, f"{i:06d}.png")
        if args.skip_existing and os.path.exists(out_path):
            pass
        else:
            cv2.imwrite(out_path, m)
            written += 1
        if i % 100 == 0:
            print(f"[INFO] frame {i}/{len(rgb)}")
        pending_dst = dst_next

    cache.clear()
    print(f"[INFO] masks written: {written}/{len(rgb)} -> {args.output}")
    print(f"[INFO] mean dynamic coverage: {100.0 * cover / max(1, len(rgb)):.2f}%")


if __name__ == "__main__":
    main()
