#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Run ATE Statistics Aggregator (PRIORITY 1)
================================================
Aggregate ATE across multiple runs of each (sequence, configuration) into:
  - RMSE mean, std, min, max
  - 95% confidence interval (t-distribution, df=n-1)
  - Paired improvement: Baseline vs YOLO-Mask (Wilcoxon signed-rank; falls back to sign test / t-test for fewer than 6 pairs or ties)
Outputs:
  - output/multi_run_statistics.json
  - Console Markdown table

Current data situation: the repo already contains several sets of trajectory files for the same sequence (baseline_*, mask_tum_*,
parameter_sweep/*_traj.txt, output/*_trajectory.txt) treated as independent run draws.
In real use, replace each file with a trajectory from an independent run; this script only aggregates statistics and is agnostic to the data source.
"""
import os
import sys
import json
import math
import numpy as np
import scipy.stats as scipy_t  # t-distribution quantiles + paired tests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from compute_ate import compute_ate  # reuse the existing ATE computation

BASE = os.path.join(SCRIPT_DIR, '..')
OUT = os.path.join(BASE, 'output')

# sequence -> (baseline config, mask config, gt)
# Each config maps to a list of files (independent run draws; default 1 = a single run).
# IMPORTANT: the existing baseline_*/mask_tum_* files in the repo are single-run trajectories;
# tests show that mixing in parameter_sweep/* or output/*_trajectory.txt produces wrong aggregates (they are not repeated runs of the same config).
# So the default is 1 real file per config; once formal multi-run data (independent re-runs, N>=5) is available, append the new trajectory files to the list.
TUM_GT = os.path.join(OUT, 'gt_trajectories', 'TUM')
SEQS = {
    'walking_xyz': {
        'gt': os.path.join(TUM_GT, 'rgbd_dataset_freiburg3_walking_xyz_groundtruth.txt'),
        'baseline': [os.path.join(OUT, 'baseline_tum_xyz.txt')],
        'mask': [os.path.join(OUT, 'mask_tum_walking_xyz.txt')],
    },
    'walking_halfsphere': {
        'gt': os.path.join(TUM_GT, 'rgbd_dataset_freiburg3_walking_halfsphere_groundtruth.txt'),
        'baseline': [os.path.join(OUT, 'baseline_tum_half.txt')],
        'mask': [os.path.join(OUT, 'mask_tum_walking_half.txt')],
    },
    'walking_static': {
        'gt': os.path.join(TUM_GT, 'rgbd_dataset_freiburg3_walking_static_groundtruth.txt'),
        'baseline': [os.path.join(OUT, 'baseline_tum_walking_static.txt')],
        'mask': [os.path.join(OUT, 'mask_tum_walking_static.txt')],
    },
    'sitting_xyz': {
        'gt': os.path.join(TUM_GT, 'rgbd_dataset_freiburg3_sitting_xyz_groundtruth.txt'),
        'baseline': [os.path.join(OUT, 'baseline_tum_sitting_xyz.txt')],
        'mask': [os.path.join(OUT, 'mask_tum_sitting_xyz.txt')],
    },
    'sitting_static': {
        'gt': os.path.join(TUM_GT, 'rgbd_dataset_freiburg3_sitting_static_groundtruth.txt'),
        'baseline': [os.path.join(OUT, 'baseline_tum_sitting_static.txt')],
        'mask': [os.path.join(OUT, 'mask_tum_sitting_static.txt')],
    },
}


def ates(paths, gt):
    rmses, nmatched = [], []
    for p in paths:
        if not os.path.exists(p):
            continue
        r = compute_ate(p, gt)
        if r is not None:
            rmses.append(r['rmse'])
            nmatched.append(r['n_matched'])
    return np.array(rmses) if rmses else np.array([])


def t_ci(a, conf=0.95):
    if len(a) < 2:
        return float('nan'), float('nan')
    m = a.mean()
    se = a.std(ddof=1) / math.sqrt(len(a))
    tcrit = scipy_t.t.ppf(0.5 + conf / 2, len(a) - 1)
    return m - tcrit * se, m + tcrit * se


def paired_test(base, mask):
    """base/mask must have the same length (paired). Returns (p_value, test_name). For n<6 or ties, use the paired sign test; otherwise Wilcoxon Signed-Rank."""
    b = np.asarray(base, float)
    m = np.asarray(mask, float)
    if len(b) != len(m) or len(b) < 1:
        return float('nan'), 'n/a'
    d = b - m  # positive = improvement
    n = len(d)
    import scipy.stats as st
    nonz = d[d != 0]
    if len(d) < 6:
        # Paired sign test
        cnt = int((nonz > 0).sum()) if len(nonz) else n
        tot = len(nonz) if len(nonz) else n
        # Binomial test, H0: sign is equally likely positive or negative
        p = st.binomtest(cnt, tot, 0.5, alternative='two-sided').pvalue if tot else float('nan')
        return p, 'sign'
    try:
        stat, p = st.wilcoxon(b, m)
        return p, 'wilcoxon'
    except ValueError:
        return float('nan'), 'tie'


def main():
    out = {}
    md_rows = []
    for seq, cfg in SEQS.items():
        gt = cfg['gt']
        b = ates(cfg['baseline'], gt)
        m = ates(cfg['mask'], gt)
        row = {'seq': seq, 'baseline': {}, 'mask': {}, 'improvement': {}}
        if len(b):
            row['baseline'] = {'n': int(len(b)), 'rmse_mean': float(b.mean()),
                               'rmse_std': float(b.std(ddof=1)) if len(b) > 1 else None,
                               'rmse_pairs': [float(x) for x in b]}
        if len(m):
            row['mask'] = {'n': int(len(m)), 'rmse_mean': float(m.mean()),
                           'rmse_std': float(m.std(ddof=1)) if len(m) > 1 else None,
                           'rmse_pairs': [float(x) for x in m]}
        # Improvement % is computable for n>=1 (a single run gives a point estimate); statistical tests only make sense for paired samples with n>=2
        if len(b) == len(m) and len(b) > 0:
            imp = (b.mean() - m.mean()) / b.mean() * 100
            if len(b) >= 2:
                pv, tname = paired_test(b, m)
                lo, hi = t_ci(b), t_ci(m)
            else:
                pv, tname = float('nan'), 'needs N>=2'
                lo, hi = (float('nan'), float('nan')), (float('nan'), float('nan'))
            row['improvement'] = {'mean_pct': float(imp),
                                  'p': float(pv) if pv == pv else None, 'test': tname,
                                  'baseline_ci95': lo, 'mask_ci95': hi}
        out[seq] = row
        # markdown row
        def fmt(d):
            if not d: return 'N/A'
            n = d.get('n', 0)
            if n >= 2 and d.get('rmse_std') is not None:
                return f"{d['rmse_mean']:.4f}{chr(0x00B1)}{d['rmse_std']:.4f} (n={n})"
            return f"{d['rmse_mean']:.4f} (n=1)"
        imp = row['improvement']
        imp_s = f"{imp['mean_pct']:+.1f}%" if imp.get('mean_pct') is not None else 'N/A'
        p_s = imp.get('test', '')
        if imp.get('p') is not None:
            p_s = f"p={imp['p']:.3f}({imp['test']})"
        md_rows.append(f"| {seq} | {fmt(row['baseline'])} | {fmt(row['mask'])} | {imp_s} | {p_s} |")

    # Save JSON
    json_path = os.path.join(OUT, 'multi_run_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print("== Multi-Run ATE Statistics (mean +/- std, n runs) ==")
    print("| Sequence | Baseline ATE | YOLO-Mask ATE | Improvement | Paired test |")
    print("|---|---|---|---|---|")
    for r in md_rows:
        print(r)
    print(f"\nSaved: {json_path}")
    print("NOTE: n is currently limited by the trajectory files already in the repo; run independent repeats (N>=5) per sequence and rerun this script.")


if __name__ == '__main__':
    main()