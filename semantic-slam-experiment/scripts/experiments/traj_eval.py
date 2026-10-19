#!/usr/bin/env python3
"""
traj_eval.py -- ATE (SE3) + RPE for one trajectory pair.
========================================================
Evaluates a single estimated trajectory against ground truth:

  * ATE  : SE(3) Umeyama, no scale (delegated to the repo's compute_ate.py, so
           the Umeyama implementation is not re-implemented here).
  * RPE  : translation RPE. For KITTI (12-column pose files) it reports the
           KITTI odometry relative translation error for Delta = 1..10 frames
           and their mean; for TUM it reports consecutive-frame translation RPE.

This supplies the RPE routine used by study 8 (`run_extra_kitti.sh`), which the
repo lacked (only compute_ate.py existed).

Usage:
    # KITTI (12-col matrices; est = ORB-SLAM3 CameraTrajectory.txt, gt = poses/00.txt)
    python traj_eval.py --est <traj.txt> --gt <gt.txt> --format kitti --dt 0.1

    # TUM (timestamp x y z qx qy qz qw)
    python traj_eval.py --est <traj.txt> --gt <gt.txt> --format tum

Environment:
    SEMANTIC_ROOT  repo root, to locate scripts/compute_ate.py (or pass --compute-ate-dir).
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_compute_ate(explicit):
    cands = []
    if explicit:
        cands.append(explicit)
    root = os.environ.get("SEMANTIC_ROOT")
    if root:
        cands.append(os.path.join(root, "scripts"))
    cands.append(os.path.dirname(EXP_DIR))
    for d in cands:
        if os.path.isfile(os.path.join(d, "compute_ate.py")):
            sys.path.insert(0, d)
            import compute_ate
            return compute_ate
    raise SystemExit("[ERROR] compute_ate.py not found; set SEMANTIC_ROOT or --compute-ate-dir.")


def read_kitti_matrices(path):
    mats = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 12:
                continue
            m = np.array([float(x) for x in p[:12]]).reshape(3, 4)
            T = np.eye(4)
            T[:3, :4] = m
            mats.append(T)
    return np.array(mats)


def kitti_to_tum_file(mats, path, dt):
    """Write a KITTI 4x4 sequence as a TUM 8-column file (timestamp = i*dt)."""
    with open(path, "w") as f:
        for i, T in enumerate(mats):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
            f.write(f"{i * dt:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def rpe_kitti(est_m, gt_m, deltas):
    out = {}
    n = min(len(est_m), len(gt_m))
    for d in deltas:
        if n - d <= 0:
            out[d] = None
            continue
        errs = []
        for i in range(n - d):
            e_rel = np.linalg.inv(est_m[i]) @ est_m[i + d]
            g_rel = np.linalg.inv(gt_m[i]) @ gt_m[i + d]
            errs.append(np.linalg.norm(e_rel[:3, 3] - g_rel[:3, 3]))
        out[d] = float(np.mean(errs))
    return out


def rpe_tum(compute_ate, est_file, gt_file, max_diff=0.02):
    est = compute_ate.read_tum_trajectory(est_file)
    gt = compute_ate.read_tum_trajectory(gt_file)
    if len(est) == 0 or len(gt) == 0:
        return None
    pairs = compute_ate.find_closest_timestamps(est[:, 0], gt[:, 0], max_diff)
    if len(pairs) < 10:
        return None
    idx = {e: g for e, g in pairs}
    errs = []
    for e in sorted(idx):
        if e + 1 in idx:
            d_est = est[e + 1, 1:4] - est[e, 1:4]
            d_gt = gt[idx[e + 1], 1:4] - gt[idx[e], 1:4]
            errs.append(np.linalg.norm(d_est - d_gt))
    if not errs:
        return None
    return float(np.sqrt(np.mean(np.square(errs))))


def main():
    ap = argparse.ArgumentParser(description="ATE(SE3) + RPE for one trajectory pair")
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--format", required=True, choices=["tum", "kitti"])
    ap.add_argument("--dt", type=float, default=0.1, help="frame period for KITTI timestamps")
    ap.add_argument("--deltas", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--compute-ate-dir", help="dir containing compute_ate.py")
    ap.add_argument("--json", help="optional path to write results as JSON")
    args = ap.parse_args()

    for p in (args.est, args.gt):
        if not os.path.isfile(p):
            raise SystemExit(f"[ERROR] file not found: {p}")

    compute_ate = load_compute_ate(args.compute_ate_dir)
    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]
    result = {"est": args.est, "gt": args.gt, "format": args.format}

    if args.format == "kitti":
        est_m = read_kitti_matrices(args.est)
        gt_m = read_kitti_matrices(args.gt)
        if len(est_m) < 10 or len(gt_m) < 10:
            raise SystemExit(f"[ERROR] too few poses (est={len(est_m)}, gt={len(gt_m)})")
        with tempfile.TemporaryDirectory() as td:
            e_tum = os.path.join(td, "est_tum.txt")
            g_tum = os.path.join(td, "gt_tum.txt")
            kitti_to_tum_file(est_m, e_tum, args.dt)
            kitti_to_tum_file(gt_m, g_tum, args.dt)
            ate = compute_ate.compute_ate(e_tum, g_tum)
        rpe = rpe_kitti(est_m, gt_m, deltas)
        rpe_vals = [v for v in rpe.values() if v is not None]
        result["ate_se3_rmse"] = None if ate is None else ate["rmse"]
        result["n_matched"] = None if ate is None else ate["n_matched"]
        result["rpe_by_delta"] = rpe
        result["rpe_mean"] = float(np.mean(rpe_vals)) if rpe_vals else None
        print(f"ATE_SE3 RMSE : {result['ate_se3_rmse']}")
        print(f"RPE (mean 1..10): {result['rpe_mean']}")
        print(f"RPE by delta : {rpe}")
    else:
        ate = compute_ate.compute_ate(args.est, args.gt)
        rpe = rpe_tum(compute_ate, args.est, args.gt)
        result["ate_se3_rmse"] = None if ate is None else ate["rmse"]
        result["n_matched"] = None if ate is None else ate["n_matched"]
        result["rpe_consecutive"] = rpe
        print(f"ATE_SE3 RMSE : {result['ate_se3_rmse']}")
        print(f"RPE (consecutive): {rpe}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
