# -*- coding: utf-8 -*-
"""
generate_figures.py - figure generation script
===========================================
Purpose: generate all figures from the experiment data
Dependencies: matplotlib, numpy
Run: python generate_figures.py

Generated figures:
  Figure 1: System architecture
  Figure 2: TUM RGB-D ATE comparison (Baseline vs YOLO-Mask)
  Figure 3: Mean ATE comparison, static vs dynamic scenes
  Figure 4: KITTI Odometry Sequence 00 ATE comparison
  Figure 5: Offline pre-processing cost (detection + masking)

Data source: experiment runs on the AutoDL cloud platform
Output directory: ../output/figures/

All ATE values are the five-run SE(3) (scale-fixed) means, with the sample
standard deviation over the five runs (ddof = 1).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, BoxStyle
import numpy as np
import os
import sys

# ================================================================
# Configuration
# ================================================================
# Output directory (relative to this script: ../output/figures/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Unified typographic scale: every text element is set at FS_BASE, and only the
# figure/panel titles use FS_TITLE. No other sizes are used anywhere, so labels
# within a figure are the same size and none of them are undersized.
FS_BASE = 10
FS_TITLE = 12
plt.rcParams.update({
    'font.size': FS_BASE,
    'axes.titlesize': FS_TITLE,
    'axes.labelsize': FS_BASE,
    'xtick.labelsize': FS_BASE,
    'ytick.labelsize': FS_BASE,
    'legend.fontsize': FS_BASE,
    'figure.titlesize': FS_TITLE,
})

# ================================================================
# Experiment data
# Source: the runs recorded in this script
# Evaluation tool: evo_traj (evo Python package)
# Evaluation method: ATE RMSE after SE(3) Umeyama alignment
# ================================================================

# --- TUM RGB-D dataset ATE data, five-run SE(3) mean +/- std (unit: meters) ---
TUM_DATA = {
    'sitting_static':     {'baseline': 0.0101, 'bstd': 0.0007, 'mask': 0.0116, 'mstd': 0.0014, 'frames': 707,  'type': 'static'},
    'sitting_xyz':        {'baseline': 0.0092, 'bstd': 0.0002, 'mask': 0.0737, 'mstd': 0.0586, 'frames': 1261, 'type': 'static'},
    'walking_static':     {'baseline': 0.3494, 'bstd': 0.0363, 'mask': 0.0184, 'mstd': 0.0057, 'frames': 743,  'type': 'dynamic'},
    'walking_xyz':        {'baseline': 0.6499, 'bstd': 0.0655, 'mask': 0.0208, 'mstd': 0.0021, 'frames': 859,  'type': 'dynamic'},
    'walking_halfsphere': {'baseline': 0.3580, 'bstd': 0.0702, 'mask': 0.0533, 'mstd': 0.0083, 'frames': 1067, 'type': 'dynamic'},
}

# --- KITTI Odometry dataset ATE data, SE(3) (unit: meters) ---
# Five-run means per configuration (the standard deviation is reported in the paper's
# KITTI table). Sequence 00 is shown here; sequences 01, 04 and 07 are tabulated there
# because sequence 01 has an ATE an order of magnitude larger and would dominate a bar
# chart shared with the others.
KITTI_DATA = {
    '00': {'baseline': 1.2821, 'mask': 1.3602, 'frames': 4541},
}

# --- Offline pre-processing cost, detection + masking (per sequence) ---
# Measured by scripts/experiments/measure_offline_cost.py (run_all.sh study 10),
# separately from the SLAM runtime.
TIMING_DATA = {
    'TUM walking_xyz': {'frames': 859, 'detect_s': 20.72, 'mask_s': 16.66, 'total_s': 37.38,
                        'ms_per_frame': 43.5, 'fps': 23.0, 'cpu_s': 40.53, 'max_rss_gb': 1.033},
    'KITTI 00':        {'frames': 4541, 'detect_s': 72.87, 'mask_s': 94.37, 'total_s': 167.24,
                        'ms_per_frame': 36.8, 'fps': 27.2, 'cpu_s': 170.25, 'max_rss_gb': 1.039},
}

# ================================================================
# Helper functions
# ================================================================
def calc_improvement(baseline, mask):
    """Compute the improvement percentage of YOLO-Mask over Baseline"""
    return round((baseline - mask) / baseline * 100, 1)

def save_figure(fig, filename):
    """Save a figure to the output directory"""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    size_kb = os.path.getsize(path) / 1024
    print(f'  [OK] {filename} ({size_kb:.1f} KB)')
    plt.close(fig)

# ================================================================
# Figure 1: System architecture
# System architecture overview
# ================================================================
def generate_figure1_architecture():
    """Generate the system architecture flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    ax.set_title('System Architecture of Semantic Masking Pipeline',
                 fontsize=FS_TITLE, fontweight='bold', pad=14)

    # Component definitions: (x, y, w, h, label, color, sublabel)
    boxes = [
        (0.3, 2.1, 2.0, 1.2, 'Input Images\n(RGB / RGB-D)', '#4472C4', 'TUM / KITTI'),
        (3.0, 2.1, 2.0, 1.2, 'YOLOv8-nano\nObject Detection', '#ED7D31', 'Offline Detection'),
        (5.7, 2.1, 2.0, 1.2, 'mask_dataset.py\nImage Masking', '#2E7D32', 'BBox -> Zero Out'),
        (8.4, 2.1, 2.0, 1.2, 'ORB-SLAM3\n(Unmodified)', '#7030A0', 'Tracking + Mapping'),
        (8.4, 0.20, 2.0, 0.85, 'ATE Evaluation\n(evo_traj)', '#C00000', 'Trajectory Analysis'),
    ]

    for (x, y, w, h, label, color, sublabel) in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle=BoxStyle("Round", pad=0.18),
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.13, label, ha='center', va='center',
                fontsize=FS_BASE, fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.30, sublabel, ha='center', va='center',
                fontsize=FS_BASE, color='white', alpha=0.9)

    # Arrows
    arrows = [
        (2.3, 2.7, 3.0, 2.7), (5.0, 2.7, 5.7, 2.7),
        (7.7, 2.7, 8.4, 2.7), (9.4, 2.1, 9.4, 1.05),
    ]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2.5,
                                    connectionstyle='arc3,rad=0'))

    # Offline/online boundary
    ax.axvline(x=7.9, ymin=0.09, ymax=0.955, color='#FF0000', linewidth=2,
               linestyle='--', alpha=0.5)
    ax.text(3.9, 5.05, 'OFFLINE (semantic pre-processing)', ha='center', fontsize=FS_BASE,
            color='#FF0000', fontweight='bold', alpha=0.75)
    ax.text(9.7, 5.05, 'ONLINE (SLAM execution)', ha='center', fontsize=FS_BASE,
            color='#2E7D32', fontweight='bold', alpha=0.75)

    # Data flow labels, placed in the free band above the component row so that
    # they never overlap the component boxes.
    ax.text(2.65, 3.75, 'Detection\nJSON', ha='center', va='bottom', fontsize=FS_BASE,
            color='#555555', fontstyle='italic')
    ax.text(5.35, 3.75, 'Masked\nImages', ha='center', va='bottom', fontsize=FS_BASE,
            color='#555555', fontstyle='italic')
    ax.text(8.05, 3.75, 'Original\nRGB-D', ha='center', va='bottom', fontsize=FS_BASE,
            color='#555555', fontstyle='italic')
    # Trajectory label sits to the left of the vertical arrow, inside the free band
    # between the tracking box and the evaluation box.
    ax.text(9.30, 1.57, 'Trajectory\n(KeyFrameTrajectory.txt)', ha='right', va='center',
            fontsize=FS_BASE, color='#555555', fontstyle='italic')
    ax.text(6.0, 0.05, 'Dynamic classes: person, bicycle, car, motorcycle, bus, '
            'truck, dog, cat',
            ha='center', fontsize=FS_BASE, color='#ED7D31', fontstyle='italic')

    plt.tight_layout()
    save_figure(fig, 'fig1_system_architecture.png')

# ================================================================
# Figure 2: TUM RGB-D ATE comparison
# TUM RGB-D sequence results
# ================================================================
def generate_figure2_tum_ate():
    """Generate the TUM RGB-D Baseline vs YOLO-Mask ATE comparison figure"""
    sequences = list(TUM_DATA.keys())
    seq_labels = [s.replace('_', '\n') for s in sequences]
    baseline_ate = [TUM_DATA[s]['baseline'] for s in sequences]
    baseline_std = [TUM_DATA[s]['bstd'] for s in sequences]
    mask_ate = [TUM_DATA[s]['mask'] for s in sequences]
    mask_std = [TUM_DATA[s]['mstd'] for s in sequences]
    improvements = [calc_improvement(b, m) for b, m in zip(baseline_ate, mask_ate)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Absolute ATE values, error bars = std over the five runs
    x = np.arange(len(sequences))
    width = 0.35
    bars1 = ax1.bar(x - width/2, baseline_ate, width, yerr=baseline_std, capsize=3,
                    label='Baseline (ORB-SLAM3)', color='#4472C4',
                    edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, mask_ate, width, yerr=mask_std, capsize=3,
                    label='YOLO-Mask (Ours)', color='#ED7D31',
                    edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('ATE RMSE (m)', fontsize=FS_BASE, fontweight='bold')
    ax1.set_title('(a) Absolute ATE, five-run mean +/- std', fontsize=FS_TITLE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_labels)
    # Headroom is sized so that the rotated value labels never reach the top spine.
    ax1.set_ylim(0, max(baseline_ate) * 1.70)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val, std in zip(bars1, baseline_ate, baseline_std):
        # Offset clears the error-bar cap so the rotated value label never overlaps it.
        off = max(std * 1.2, 0.012)
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + off,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=FS_BASE, rotation=90,
                 color='#333333')
    for bar, val, std in zip(bars2, mask_ate, mask_std):
        off = max(std * 1.2, 0.012)
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + off,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=FS_BASE, rotation=90,
                 color='#333333')

    # (b) Relative improvement. A symmetric-log y-axis keeps the large static-scene
    # degradation and the dynamic-scene gains legible in a single panel.
    colors = ['#C62828' if v < 0 else '#2E7D32' for v in improvements]
    bars3 = ax2.bar(x, improvements, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax2.set_yscale('symlog', linthresh=10, linscale=1.0)
    ax2.set_ylim(-1200, 300)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Improvement over Baseline (%), symlog', fontsize=FS_BASE, fontweight='bold')
    ax2.set_title('(b) Relative Improvement', fontsize=FS_TITLE, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_labels)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axvspan(-0.5, 1.5, alpha=0.07, color='red')
    ax2.axvspan(1.5, 4.5, alpha=0.07, color='green')
    for bar, val in zip(bars3, improvements):
        color = '#2E7D32' if val > 0 else '#C62828'
        # For positive bars the label sits above the bar; for negative bars it sits
        # below the bar end, so it never falls inside the coloured area.
        ax2.text(bar.get_x() + bar.get_width()/2., val, f'{val:+.1f}%',
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=FS_BASE,
                 fontweight='bold', color=color)
    # Region labels sit in the empty half of each shaded band (static bars are
    # negative, dynamic bars are positive), so they never overlap the bars.
    ax2.text(0.5, 25, 'Static: false dynamic', ha='center', va='center',
             fontsize=FS_BASE, color='#C62828', fontstyle='italic')
    ax2.text(3.0, -25, 'Dynamic: masking effective', ha='center', va='center',
             fontsize=FS_BASE, color='#2E7D32', fontstyle='italic')

    plt.suptitle('Figure 2: TUM RGB-D ATE Comparison - Baseline vs YOLO-Mask',
                 fontsize=FS_TITLE, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig2_tum_ate.png')

# ================================================================
# Figure 3: Mean ATE comparison for static vs dynamic scenes
# Scene type analysis over the TUM RGB-D sequences
# ================================================================
def generate_figure3_scene_type():
    """Generate the static vs dynamic mean ATE comparison figure"""
    static_seqs = [s for s, d in TUM_DATA.items() if d['type'] == 'static']
    dynamic_seqs = [s for s, d in TUM_DATA.items() if d['type'] == 'dynamic']

    static_baseline_mean = np.mean([TUM_DATA[s]['baseline'] for s in static_seqs])
    static_mask_mean = np.mean([TUM_DATA[s]['mask'] for s in static_seqs])
    dynamic_baseline_mean = np.mean([TUM_DATA[s]['baseline'] for s in dynamic_seqs])
    dynamic_mask_mean = np.mean([TUM_DATA[s]['mask'] for s in dynamic_seqs])

    static_imp = calc_improvement(static_baseline_mean, static_mask_mean)
    dynamic_imp = calc_improvement(dynamic_baseline_mean, dynamic_mask_mean)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    categories = ['Static scenes\n(sitting_*)', 'Dynamic scenes\n(walking_*)']
    x = np.arange(len(categories))
    width = 0.3

    ax.bar(x - width/2, [static_baseline_mean, dynamic_baseline_mean], width,
           label='Baseline (ORB-SLAM3)', color='#4472C4', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, [static_mask_mean, dynamic_mask_mean], width,
           label='YOLO-Mask (Ours)', color='#ED7D31', edgecolor='white', linewidth=0.5)
    # Log scale: the static means are two orders of magnitude below the dynamic ones.
    ax.set_yscale('log')
    ax.set_ylim(0.004, 2.0)
    ax.set_ylabel('Mean ATE RMSE (m), log scale', fontsize=FS_BASE, fontweight='bold')
    ax.set_title('Figure 3: Static vs Dynamic Scenes - Mean ATE Comparison',
                 fontsize=FS_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotate values (multiplicative offsets keep them above the bars on a log axis)
    for xpos, val, color in [(0 - width/2, static_baseline_mean, '#4472C4'),
                             (0 + width/2, static_mask_mean, '#ED7D31'),
                             (1 - width/2, dynamic_baseline_mean, '#4472C4'),
                             (1 + width/2, dynamic_mask_mean, '#ED7D31')]:
        ax.text(xpos, val * 1.22, f'{val:.4f}', ha='center', va='bottom',
                fontsize=FS_BASE, fontweight='bold', color=color)

    # Improvement percentages are placed in the empty band above the bars (and
    # above the value labels) so that the annotation boxes never sit on a bar.
    ax.annotate(f'{static_imp:+.1f}%',
                xy=(0, 0.13),
                fontsize=FS_BASE, fontweight='bold', color='#C62828', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#C62828', alpha=0.9))
    ax.annotate(f'{dynamic_imp:+.1f}%',
                xy=(1, 1.05),
                fontsize=FS_BASE, fontweight='bold', color='#2E7D32', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#2E7D32', alpha=0.9))

    ax.text(0.25, 0.97, 'YOLO-Mask degrades static scenes', transform=ax.transAxes,
            ha='center', fontsize=FS_BASE, color='#C62828', fontstyle='italic')
    ax.text(0.75, 0.97, 'YOLO-Mask improves dynamic scenes', transform=ax.transAxes,
            ha='center', fontsize=FS_BASE, color='#2E7D32', fontstyle='italic')

    plt.tight_layout()
    save_figure(fig, 'fig3_scene_type.png')

# ================================================================
# Figure 4: KITTI Odometry Sequence 00 ATE comparison
# KITTI Odometry sequence 00 results
# ================================================================
def generate_figure4_kitti_ate():
    """Generate the KITTI 00 ATE comparison figure"""
    kitti_baseline = KITTI_DATA['00']['baseline']
    kitti_mask = KITTI_DATA['00']['mask']
    improvement = calc_improvement(kitti_baseline, kitti_mask)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0]
    width = 0.3
    ax.bar(x[0] - width/2, kitti_baseline, width, label='Baseline (ORB-SLAM3)',
           color='#4472C4', edgecolor='white', linewidth=0.5)
    ax.bar(x[0] + width/2, kitti_mask, width, label='YOLO-Mask (Ours)',
           color='#ED7D31', edgecolor='white', linewidth=0.5)
    ax.set_ylabel('ATE RMSE (m)', fontsize=FS_BASE, fontweight='bold')
    ax.set_title('Figure 4: KITTI Odometry Sequence 00 - ATE Comparison',
                 fontsize=FS_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['KITTI Sequence 00\n(4,541 frames, with loop closure)'])
    ax.set_ylim(0, max(kitti_baseline, kitti_mask) * 1.45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax.text(x[0] - width/2, kitti_baseline + 0.03, f'{kitti_baseline:.4f} m',
            ha='center', fontsize=FS_BASE, fontweight='bold', color='#4472C4')
    ax.text(x[0] + width/2, kitti_mask + 0.03, f'{kitti_mask:.4f} m',
            ha='center', fontsize=FS_BASE, fontweight='bold', color='#ED7D31')

    ax.text(0.5, 0.98,
            f'ATE change: {improvement:+.1f}%\n'
            f'(masking underperforms the baseline; the\n'
            f'present experiments do not isolate the causes)',
            transform=ax.transAxes, ha='center', va='top', fontsize=FS_BASE,
            fontstyle='italic', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3F3',
                      edgecolor='#C62828', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'fig4_kitti_ate.png')

# ================================================================
# Figure 5: Offline pre-processing cost
# Timing analysis
# ================================================================
def generate_figure5_timing():
    """Generate the offline pre-processing cost figure (detection + masking)"""
    datasets = list(TIMING_DATA.keys())
    labels = [f"{d}\n({TIMING_DATA[d]['frames']:,} frames)" for d in datasets]
    detect = [TIMING_DATA[d]['detect_s'] for d in datasets]
    mask = [TIMING_DATA[d]['mask_s'] for d in datasets]
    total = [TIMING_DATA[d]['total_s'] for d in datasets]
    fps = [TIMING_DATA[d]['fps'] for d in datasets]
    rss = [TIMING_DATA[d]['max_rss_gb'] for d in datasets]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(datasets))
    ax2_twin = ax.twinx()

    ax.bar(x, detect, 0.55, label='Detection (s)',
           color='#4472C4', edgecolor='white', linewidth=0.5)
    ax.bar(x, mask, 0.55, bottom=detect, label='Masking (s)',
           color='#ED7D31', edgecolor='white', linewidth=0.5)
    ax2_twin.plot(x, rss, marker='D', markersize=7, color='#2E7D32',
                  linewidth=1.5, label='Peak RSS (GB)')

    ax.set_ylabel('Offline pre-processing time (s)', fontsize=FS_BASE, fontweight='bold')
    ax2_twin.set_ylabel('Memory (GB)', fontsize=FS_BASE, fontweight='bold')
    # Headroom above the tallest bar so the note box never touches the bar labels.
    ax.set_ylim(0, max(total) * 1.32)
    ax2_twin.set_ylim(0, max(rss) * 1.9)
    ax.set_title('Figure 5: Offline Pre-processing Cost Across Representative Sequences',
                 fontsize=FS_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, d in enumerate(datasets):
        # Segment labels are placed inside each stack component.
        ax.text(i, detect[i] / 2, f'{detect[i]:.1f}s', ha='center', va='center',
                fontsize=FS_BASE, fontweight='bold', color='white')
        ax.text(i, detect[i] + mask[i] / 2, f'{mask[i]:.1f}s', ha='center', va='center',
                fontsize=FS_BASE, fontweight='bold', color='white')
        ax.text(i, total[i] + max(total) * 0.035,
                f'total {total[i]:.1f}s\n{fps[i]:.1f} FPS', ha='center', va='bottom',
                fontsize=FS_BASE, fontweight='bold')
    ax2_twin.text(x[0] - 0.12, rss[0], f'{rss[0]:.2f} GB', ha='right', va='center',
                  fontsize=FS_BASE, fontweight='bold', color='#2E7D32')
    ax2_twin.text(x[-1] + 0.12, rss[-1], f'{rss[-1]:.2f} GB', ha='left', va='center',
                  fontsize=FS_BASE, fontweight='bold', color='#2E7D32')

    # The note sits in the empty band between the two bars, clear of the legend
    # (top left), the RSS line and the two bars.
    ax.text(0.30, 0.66, 'Detection and masking run offline,\n'
            'before SLAM (not part of the SLAM runtime).',
            transform=ax.transAxes, ha='center', va='center', fontsize=FS_BASE, fontstyle='italic',
            color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFF0',
                      edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    save_figure(fig, 'fig5_timing.png')

# ================================================================
# Main entry point
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Figure generation script')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)
    print()

    generate_figure1_architecture()
    generate_figure2_tum_ate()
    generate_figure3_scene_type()
    generate_figure4_kitti_ate()
    generate_figure5_timing()

    print()
    print(f'All 5 figures written to: {OUTPUT_DIR}')
    print('Done!')
