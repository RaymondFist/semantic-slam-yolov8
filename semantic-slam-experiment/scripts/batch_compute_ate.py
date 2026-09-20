# -*- coding: utf-8 -*-
"""
Batch compute ATE for all sequences
=====================
Compute: EuRoC MH_01/MH_05 + TUM walking sequences
Compare: baseline (ORB-SLAM3) vs mask (YOLO-Mask ORB-SLAM3)
"""

import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from compute_ate import compute_ate

BASE_DIR = os.path.join(SCRIPT_DIR, '..')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'real_trajectories')

results = {}

# ========== EuRoC ==========
print("\n" + "="*70)
print("EuRoC Dataset")
print("="*70)

euroc_sequences = [
    # (seq_name, mask_traj, baseline_traj, gt)
    ('MH_01_easy', 
     'euroc_MH_01_easy_trajectory.txt',
     'baseline_euroc_mh_01_easy.txt',
     os.path.join(DATA_DIR, 'EuRoC', 'MH_01_easy.txt')),
    ('MH_05_difficult',
     'euroc_MH_05_difficult_trajectory.txt',
     'baseline_euroc_mh_05_difficult.txt',
     os.path.join(DATA_DIR, 'EuRoC', 'MH_05_difficult.txt')),
]

for seq_name, mask_file, baseline_file, gt_file in euroc_sequences:
    print(f"\n--- {seq_name} ---")
    
    # Mask trajectory
    mask_path = os.path.join(OUTPUT_DIR, mask_file)
    if os.path.exists(mask_path):
        print(f"\n  [Mask/YOLO] {mask_file}:")
        mask_result = compute_ate(mask_path, gt_file)
    else:
        print(f"  [WARN] Mask file not found: {mask_file}")
        mask_result = None
    
    # Baseline trajectory
    baseline_path = os.path.join(OUTPUT_DIR, baseline_file)
    if os.path.exists(baseline_path):
        print(f"\n  [Baseline/ORB-SLAM3] {baseline_file}:")
        baseline_result = compute_ate(baseline_path, gt_file)
    else:
        print(f"  [WARN] Baseline file not found: {baseline_file}")
        baseline_result = None
    
    results[f'EuRoC_{seq_name}'] = {
        'mask': mask_result,
        'baseline': baseline_result,
    }
    
    # Compute improvement rate
    if mask_result and baseline_result and baseline_result['rmse'] > 0:
        improvement = (baseline_result['rmse'] - mask_result['rmse']) / baseline_result['rmse'] * 100
        print(f"\n  >>> Improvement: {improvement:+.1f}%")

# ========== TUM RGB-D ==========
print("\n" + "="*70)
print("TUM RGB-D Dataset")
print("="*70)

tum_sequences = [
    # (seq_short, baseline_traj, mask_traj, gt_traj)
    ('walking_xyz',
     'baseline_tum_xyz.txt',
     'rgbd_dataset_freiburg3_walking_xyz_trajectory.txt',
     os.path.join(DATA_DIR, 'TUM', 'rgbd_dataset_freiburg3_walking_xyz_groundtruth.txt')),
    ('walking_halfsphere',
     'baseline_tum_half.txt',
     'rgbd_dataset_freiburg3_walking_halfsphere_trajectory.txt',
     os.path.join(DATA_DIR, 'TUM', 'rgbd_dataset_freiburg3_walking_halfsphere_groundtruth.txt')),
    ('walking_static',
     'baseline_tum_walking_static.txt',
     'mask_tum_walking_static.txt',
     os.path.join(DATA_DIR, 'TUM', 'rgbd_dataset_freiburg3_walking_static_groundtruth.txt')),
    ('sitting_xyz',
     'baseline_tum_sitting_xyz.txt',
     'output_tum_sitting_xyz_mask.txt',  # may not exist
     os.path.join(DATA_DIR, 'TUM', 'rgbd_dataset_freiburg3_sitting_xyz_groundtruth.txt')),
    ('sitting_static',
     'baseline_tum_sitting_static.txt',
     'output_tum_sitting_static_mask.txt',  # may not exist
     os.path.join(DATA_DIR, 'TUM', 'rgbd_dataset_freiburg3_sitting_static_groundtruth.txt')),
]

for seq_name, baseline_file, mask_file, gt_file in tum_sequences:
    print(f"\n--- TUM {seq_name} ---")
    
    # Mask trajectory
    mask_path = os.path.join(OUTPUT_DIR, mask_file)
    if not os.path.exists(mask_path):
        # Try other file names
        alt = os.path.join(OUTPUT_DIR, f'parameter_sweep/rgbd_dataset_freiburg3_{seq_name}_trajectory.txt')
        if os.path.exists(alt):
            mask_path = alt
        else:
            print(f"  [WARN] Mask file not found: {mask_file}")
            mask_path = None
    
    if mask_path and os.path.exists(mask_path):
        print(f"\n  [Mask/YOLO]:")
        mask_result = compute_ate(mask_path, gt_file)
    else:
        mask_result = None
    
    # Baseline
    baseline_path = os.path.join(OUTPUT_DIR, baseline_file)
    if os.path.exists(baseline_path):
        print(f"\n  [Baseline/ORB-SLAM3]:")
        baseline_result = compute_ate(baseline_path, gt_file)
    else:
        print(f"  [WARN] Baseline file not found: {baseline_file}")
        baseline_result = None
    
    results[f'TUM_{seq_name}'] = {
        'mask': mask_result,
        'baseline': baseline_result,
    }
    
    if mask_result and baseline_result and baseline_result['rmse'] > 0:
        improvement = (baseline_result['rmse'] - mask_result['rmse']) / baseline_result['rmse'] * 100
        print(f"\n  >>> Improvement: {improvement:+.1f}%")

# ========== Summary table ==========
print("\n" + "="*70)
print("Summary: ATE RMSE (meters)")
print("="*70)
print(f"\n{'Sequence':<30} {'Baseline':>12} {'YOLO-Mask':>12} {'Improvement':>12}")
print("-" * 70)

for key in results:
    r = results[key]
    b = r['baseline']
    m = r['mask']
    b_rmse = f"{b['rmse']:.4f}" if b else "N/A"
    m_rmse = f"{m['rmse']:.4f}" if m else "N/A"
    if b and m and b['rmse'] > 0:
        imp = f"{(b['rmse']-m['rmse'])/b['rmse']*100:+.1f}%"
    else:
        imp = "N/A"
    print(f"{key:<30} {b_rmse:>12} {m_rmse:>12} {imp:>12}")

# Save results as JSON
results_output = {}
for key, val in results.items():
    results_output[key] = {}
    for subkey, subval in val.items():
        if subval:
            results_output[key][subkey] = {
                'rmse': subval['rmse'],
                'mean': subval['mean'],
                'median': subval['median'],
                'max': subval['max'],
                'std': subval['std'],
                'n_matched': subval['n_matched'],
            }
        else:
            results_output[key][subkey] = None

json_path = os.path.join(OUTPUT_DIR, 'ate_results.json')
with open(json_path, 'w') as f:
    json.dump(results_output, f, indent=2)
print(f"\nResults saved: {json_path}")
