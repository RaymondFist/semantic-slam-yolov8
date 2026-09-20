# -*- coding: utf-8 -*-
"""
ATE (Absolute Trajectory Error) Calculator
==========================================
Pure Python implementation of SE(3) alignment + ATE RMSE computation
Does not depend on evo_traj (avoids installation issues)

Usage:
    python compute_ate.py <estimated_traj.txt> <ground_truth.txt>

Trajectory format (TUM/RGB-D standard format):
    timestamp tx ty tz qx qy qz qw
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import os


def read_tum_trajectory(filepath):
    """Read a TUM format trajectory file, return an (N, 7) array [timestamp, x, y, z, qx, qy, qz, qw]"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                # Skip duplicate rows with repeated timestamps (present in some trajectory files)
                ts = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                data.append([ts, x, y, z, qx, qy, qz, qw])
    return np.array(data)


def find_closest_timestamps(est_ts, gt_ts, max_diff=0.02):
    """Find the closest ground truth timestamp for each estimated timestamp"""
    matched = []
    for i, t in enumerate(est_ts):
        idx = np.argmin(np.abs(gt_ts - t))
        if np.abs(gt_ts[idx] - t) < max_diff:
            matched.append((i, idx))
    return matched


def umeyama_alignment(est_xyz, gt_xyz):
    """
    SE(3) alignment via Umeyama algorithm (without scale)
    Estimate: est_xyz (N,3), ground truth: gt_xyz (N,3)
    Returns: aligned est_xyz_aligned, rotation R, translation t
    """
    assert est_xyz.shape == gt_xyz.shape
    n = est_xyz.shape[0]
    
    # Centroids
    mu_est = est_xyz.mean(axis=0)
    mu_gt = gt_xyz.mean(axis=0)
    
    # Center the point sets
    est_centered = est_xyz - mu_est
    gt_centered = gt_xyz - mu_gt
    
    # Covariance matrix
    H = est_centered.T @ gt_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Handle reflection
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    
    # Rotation matrix (no scale)
    R = Vt.T @ D @ U.T
    
    # Translation
    t = mu_gt - R @ mu_est
    
    # Aligned trajectory
    est_aligned = (R @ est_xyz.T).T + t
    
    return est_aligned, R, t


def compute_ate(est_file, gt_file, max_timestamp_diff=0.1):
    """
    Compute ATE RMSE (after SE(3) alignment)
    """
    # Read trajectories
    est_data = read_tum_trajectory(est_file)
    gt_data = read_tum_trajectory(gt_file)
    
    if len(est_data) == 0 or len(gt_data) == 0:
        print(f"  [WARN] Empty trajectory file")
        return None
    
    # Timestamp matching
    matched = find_closest_timestamps(est_data[:, 0], gt_data[:, 0], max_timestamp_diff)
    
    if len(matched) < 10:
        print(f"  [WARN] Too few matched points: {len(matched)}")
        return None
    
    est_indices = [m[0] for m in matched]
    gt_indices = [m[1] for m in matched]
    
    est_xyz = est_data[est_indices, 1:4]
    gt_xyz = gt_data[gt_indices, 1:4]
    
    # SE(3) alignment
    est_aligned, R, t = umeyama_alignment(est_xyz, gt_xyz)
    
    # ATE computation
    errors = np.linalg.norm(est_aligned - gt_xyz, axis=1)
    
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    max_err = np.max(errors)
    std_err = np.std(errors)
    
    return {
        'rmse': rmse,
        'mean': mean_err,
        'median': median_err,
        'max': max_err,
        'std': std_err,
        'n_matched': len(matched),
        'R': R,
        't': t,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python compute_ate.py <estimated_traj.txt> <ground_truth.txt>")
        sys.exit(1)
    
    est_file = sys.argv[1]
    gt_file = sys.argv[2]
    
    if not os.path.exists(est_file):
        print(f"[ERROR] Estimated trajectory file not found: {est_file}")
        sys.exit(1)
    if not os.path.exists(gt_file):
        print(f"[ERROR] Ground truth file not found: {gt_file}")
        sys.exit(1)
    
    print(f"\nComputing ATE:")
    print(f"  Estimated trajectory: {os.path.basename(est_file)}")
    print(f"  Ground Truth: {os.path.basename(gt_file)}")
    
    result = compute_ate(est_file, gt_file)
    
    if result:
        print(f"\n  ATE RMSE:   {result['rmse']:.6f} m")
        print(f"  ATE Mean:   {result['mean']:.6f} m")
        print(f"  ATE Median: {result['median']:.6f} m")
        print(f"  ATE Max:    {result['max']:.6f} m")
        print(f"  ATE Std:    {result['std']:.6f} m")
        print(f"  Matched:    {result['n_matched']} points")
        return result
    else:
        print("  [FAILED] could not compute")
        return None


if __name__ == '__main__':
    main()
