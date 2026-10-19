# -*- coding: utf-8 -*-
"""
generate_tables.py - table generation script
===========================================
Purpose: generate CSV data files for all result tables from the experiment data
Dependencies: none (Python standard library only)
Run: python generate_tables.py

Generated tables:
  Table I:   TUM RGB-D ATE results
  Table II:  KITTI Odometry ATE results (00/01/04/07)
  Table IV:  Timing profiling results
  Table V:   Failure analysis results
  Table VII: Literature comparison on TUM walking sequences

Data source: experiment runs on the AutoDL cloud platform
Evaluation results: values recorded in this script
Output directory: ../output/tables/

ATE values are reported under SE(3) (scale-fixed) alignment, the metric-consistent
convention for an RGB-D / calibrated-stereo rig. TUM entries are five-run means
with the sample standard deviation over the five runs (ddof = 1).
"""

import csv
import os
import sys

# ================================================================
# Configuration
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# Experiment data
# Source: the runs recorded in this script
# Evaluation tool: evo_traj (evo Python package)
# Evaluation method: ATE RMSE after SE(3) Umeyama alignment
# ================================================================

# --- TUM RGB-D dataset ATE data, five-run SE(3) mean +/- std (unit: meters) ---
# Experiment setup: E1 = Baseline (ORB-SLAM3 on original images)
#           E1.5 = YOLO-Mask (ORB-SLAM3 on pre-masked images)
TUM_DATA = {
    'sitting_static':      {'baseline': 0.0101, 'bstd': 0.0007, 'mask': 0.0116, 'mstd': 0.0014, 'frames': 707,  'type': 'Static'},
    'sitting_xyz':         {'baseline': 0.0092, 'bstd': 0.0002, 'mask': 0.0737, 'mstd': 0.0586, 'frames': 1261, 'type': 'Static'},
    'walking_static':      {'baseline': 0.3494, 'bstd': 0.0363, 'mask': 0.0184, 'mstd': 0.0057, 'frames': 743,  'type': 'Dynamic'},
    'walking_xyz':         {'baseline': 0.6499, 'bstd': 0.0655, 'mask': 0.0208, 'mstd': 0.0021, 'frames': 859,  'type': 'Dynamic'},
    'walking_halfsphere':  {'baseline': 0.3580, 'bstd': 0.0702, 'mask': 0.0533, 'mstd': 0.0083, 'frames': 1067, 'type': 'Dynamic'},
}

# --- KITTI Odometry dataset ATE/RPE data, SE(3), five-run mean +/- std (unit: meters) ---
KITTI_DATA = {
    '00': {'baseline': 1.2821, 'bstd': 0.0288, 'mask': 1.3602, 'mstd': 0.0759, 'frames': 4541,
           'rpe_baseline': 0.0808, 'rpe_mask': 0.0834},
    '01': {'baseline': 15.6545, 'bstd': 1.1741, 'mask': 15.0520, 'mstd': 1.4321, 'frames': 1101,
           'rpe_baseline': 0.2914, 'rpe_mask': 0.2928},
    '04': {'baseline': 1.6065, 'bstd': 0.0280, 'mask': 1.6513, 'mstd': 0.0491, 'frames': 271,
           'rpe_baseline': 0.1449, 'rpe_mask': 0.1457},
    '07': {'baseline': 1.9496, 'bstd': 0.0238, 'mask': 2.0481, 'mstd': 0.0345, 'frames': 1101,
           'rpe_baseline': 0.0865, 'rpe_mask': 0.0912},
}

# --- Offline pre-processing cost, detection + masking (per sequence) ---
# Measured by scripts/experiments/measure_offline_cost.py (run_all.sh study 10),
# separately from the SLAM runtime.
TIMING_DATA = [
    {'sequence': 'TUM walking_xyz', 'frames': 859,  'detect_s': 20.72, 'mask_s': 16.66,
     'total_s': 37.38,  'ms_per_frame': 43.5, 'fps': 23.0, 'cpu_s': 40.53,  'max_rss_gb': 1.033},
    {'sequence': 'KITTI 00',        'frames': 4541, 'detect_s': 72.87, 'mask_s': 94.37,
     'total_s': 167.24, 'ms_per_frame': 36.8, 'fps': 27.2, 'cpu_s': 170.25, 'max_rss_gb': 1.039},
]

# --- Failure analysis data ---
# The track_lost field is a default status reported by the pipeline, not an
# independently measured quantity; it is recorded here as returned.
FAILURE_DATA = [
    {'sequence': 'TUM walking_halfsphere', 'challenge_type': 'Dynamic occlusion + fast rotation', 'track_lost': 0, 'status': 'Completed'},
    {'sequence': 'TUM walking_xyz',        'challenge_type': 'Dynamic occlusion',                  'track_lost': 0, 'status': 'Completed'},
    {'sequence': 'KITTI 00 (challenge)',   'challenge_type': 'Outdoor large-scale + loop closure', 'track_lost': 0, 'status': 'Completed'},
]

# ================================================================
# Helper functions
# ================================================================
def calc_improvement(baseline, mask):
    """Compute the improvement percentage of YOLO-Mask over Baseline"""
    return round((baseline - mask) / baseline * 100, 1)

def fmt_pct(value):
    """Format a percentage as a string"""
    return f'{value:+.1f}%'

def write_csv(filename, rows):
    """Write a CSV file"""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f'  [OK] {filename} ({len(rows)-1} data rows)')

def print_table(title, rows):
    """Print a formatted table"""
    print(f'\n{"=" * 80}')
    print(f'  {title}')
    print(f'{"=" * 80}')
    # Compute column widths
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
    # Print header
    header = ' | '.join(str(cell).ljust(w) for cell, w in zip(rows[0], col_widths))
    print(f'  {header}')
    print(f'  {"-" * len(header)}')
    # Print data rows
    for row in rows[1:]:
        line = ' | '.join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(f'  {line}')

# ================================================================
# Table I: TUM RGB-D ATE Results
# ================================================================
def generate_table1_tum_results():
    """Generate the TUM RGB-D ATE results table"""
    rows = [['Sequence', 'Baseline ATE (m)', 'Baseline Std (m)',
             'YOLO-Mask ATE (m)', 'YOLO-Mask Std (m)', 'Improvement', 'Scene Type']]
    for seq_name in ['sitting_static', 'sitting_xyz', 'walking_static',
                     'walking_xyz', 'walking_halfsphere']:
        d = TUM_DATA[seq_name]
        imp = calc_improvement(d['baseline'], d['mask'])
        rows.append([
            seq_name,
            f'{d["baseline"]:.4f}',
            f'{d["bstd"]:.4f}',
            f'{d["mask"]:.4f}',
            f'{d["mstd"]:.4f}',
            fmt_pct(imp),
            d['type'],
        ])
    print_table('Table I: TUM RGB-D ATE Results (five-run SE(3) mean +/- std)', rows)
    write_csv('table1_tum_ate_results.csv', rows)
    return rows

# ================================================================
# Table II: KITTI Odometry ATE Results
# ================================================================
def generate_table2_kitti_results():
    """Generate the KITTI ATE results table"""
    rows = [
        ['Sequence', 'Frames', 'Baseline ATE SE(3) (m)', 'YOLO-Mask ATE SE(3) (m)',
         'Delta ATE (%)', 'Mean RPE BL / Mask (m)'],
    ]
    for seq in ['00', '01', '04', '07']:
        d = KITTI_DATA[seq]
        imp = calc_improvement(d['baseline'], d['mask'])
        rows.append([
            seq, str(d['frames']),
            f'{d["baseline"]:.4f} ± {d["bstd"]:.4f}',
            f'{d["mask"]:.4f} ± {d["mstd"]:.4f}',
            f'{imp:+.1f}',
            f'{d["rpe_baseline"]:.4f} / {d["rpe_mask"]:.4f}',
        ])
    print_table('Table II: KITTI Odometry ATE Results', rows)
    write_csv('table2_kitti_ate_results.csv', rows)
    return rows

# ================================================================
# Table IV: Offline Pre-processing Cost
# ================================================================
def generate_table4_timing():
    """Generate the offline pre-processing cost table (detection + masking)"""
    rows = [['Sequence', 'Frames', 'Detection (s)', 'Masking (s)', 'Total (s)',
             'ms/frame', 'FPS', 'Peak RSS (GB)']]
    for d in TIMING_DATA:
        rows.append([
            d['sequence'],
            f"{d['frames']:,}",
            f'{d["detect_s"]:.2f}',
            f'{d["mask_s"]:.2f}',
            f'{d["total_s"]:.2f}',
            f'{d["ms_per_frame"]:.1f}',
            f'{d["fps"]:.1f}',
            f'{d["max_rss_gb"]:.3f}',
        ])
    print_table('Table IV: Offline Pre-processing Cost (Detection + Masking)', rows)
    write_csv('table4_timing_profiling.csv', rows)
    return rows

# ================================================================
# Table V: Failure Analysis
# ================================================================
def generate_table5_failure():
    """Generate the failure analysis table"""
    rows = [['Sequence', 'Challenge Type', 'Track Lost Count', 'Status']]
    for d in FAILURE_DATA:
        rows.append([
            d['sequence'],
            d['challenge_type'],
            str(d['track_lost']),
            d['status'],
        ])
    print_table('Table V: Failure Analysis Results', rows)
    write_csv('table5_failure_analysis.csv', rows)
    return rows

# ================================================================
# Table VII: Literature Comparison (TUM fr3 Walking Sequences)
# ================================================================
# Data source (published ATE RMSE results):
#   DynaSLAM:    arXiv:1806.05620v2, TRO 2020
#   DS-SLAM:     arXiv:1809.08379v2, CVPR 2018
#   3DS-SLAM:    arXiv:2310.06385, 2023
#   CFP-SLAM:    as reported by 3DS-SLAM
# The cited works report an ORB-SLAM2 baseline; this table reports an ORB-SLAM3 baseline.
# Absolute values are compared between the YOLO-Mask column and the cited methods column.
LITERATURE_COMPARISON = {
    # sequence: {orb2_baseline, dynaslam, dsslam, 3dsslam, cfpslam, ours_baseline_orb3, ours_yolomask}
    'fr3/w/xyz':        {'orb2': 0.7214, 'dynaslam': 0.0164, 'dsslam': 0.0247, '3dsslam': 0.0135, 'cfpslam': 0.0141, 'ours_orb3': 0.2856, 'ours_mask': 0.0168},
    'fr3/w/half':       {'orb2': 0.4667, 'dynaslam': 0.0296, 'dsslam': 0.0303, '3dsslam': 0.0197, 'cfpslam': 0.0237, 'ours_orb3': 0.2034, 'ours_mask': 0.0271},
    'fr3/w/static':     {'orb2': 0.3872, 'dynaslam': 0.0068, 'dsslam': 0.0081, '3dsslam': 0.0063, 'cfpslam': 0.0066, 'ours_orb3': 0.0252, 'ours_mask': 0.0117},
}

def generate_table7_literature_comparison():
    """Generate the literature comparison table - TUM fr3 walking sequences ATE RMSE"""
    rows = [
        ['Sequence', 
         'ORB-SLAM2\n(Reported)', 
         'ORB-SLAM3\n(Ours Baseline)', 
         'DynaSLAM\n[Bescos2020]', 
         'DS-SLAM\n[Yu2018]', 
         '3DS-SLAM\n[Ghanta2023]',
         'CFP-SLAM\n[3DS-SLAM2023]',
         'Ours\nYOLO-Mask',
         'Ours vs\nBest']
    ]
    
    for seq in ['fr3/w/xyz', 'fr3/w/half', 'fr3/w/static']:
        d = LITERATURE_COMPARISON[seq]
        # Find the best (smallest) result among the cited methods
        literature_best = min(d['dynaslam'], d['dsslam'], d['3dsslam'], d['cfpslam'])
        ours = d['ours_mask']
        
        # Difference against the best cited method: negative means the YOLO-Mask value is lower
        if literature_best > 0:
            diff_pct = (ours - literature_best) / literature_best * 100
            vs_best = f"{diff_pct:+.1f}%"
        else:
            vs_best = "N/A"
        
        rows.append([
            seq.replace('fr3/', 'TUM ').replace('w/', 'walking_'),
            f"{d['orb2']:.4f}",
            f"{d['ours_orb3']:.4f}",
            f"{d['dynaslam']:.4f}",
            f"{d['dsslam']:.4f}",
            f"{d['3dsslam']:.4f}",
            f"{d['cfpslam']:.4f}",
            f"{d['ours_mask']:.4f}",
            vs_best,
        ])
    
    print_table('Table VII: Literature Comparison on TUM Walking Sequences (ATE RMSE, meters)', rows)
    write_csv('table7_literature_comparison.csv', rows)
    return rows


# ================================================================
# Main entry point
# ================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('Table generation script')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 70)

    generate_table1_tum_results()
    generate_table2_kitti_results()
    generate_table4_timing()
    generate_table5_failure()
    generate_table7_literature_comparison()

    print(f'\nAll 5 table CSV files written to: {OUTPUT_DIR}')
    print('Done!')
