# -*- coding: utf-8 -*-
"""
generate_figures.py — 论文图表生成脚本
===========================================
论文: Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments
用途: 根据实验数据生成所有论文图表，用于数据分析和评审提交
依赖: matplotlib, numpy
运行: python generate_figures.py

生成图表列表:
  Figure 1: 系统架构图 (System Architecture)
  Figure 2: TUM RGB-D ATE 对比 (Baseline vs YOLO-Mask)
  Figure 3: 静态场景 vs 动态场景 平均ATE对比
  Figure 4: KITTI Odometry Sequence 00 ATE 对比
  Figure 5: 运行时性能分析 (Timing Profiling)
  Figure 6: 参数敏感性分析 (Parameter Sensitivity)

数据来源: 全部来自 AutoDL 云平台上的真实实验运行结果
评估结果: ../data/experiment_results.json
输出目录: ../output/figures/
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
# 配置
# ================================================================
# 输出目录 (相对于本脚本: ../output/figures/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 实验原始数据 (全部来自真实实验运行)
# 来源: deploylog.txt 第 383598 行 "实验结果导出完成"
# 评估工具: evo_traj (evo Python package)
# 评估方法: ATE RMSE after SE(3) Umeyama alignment
# ================================================================

# --- TUM RGB-D 数据集 ATE 数据 (单位: 米) ---
TUM_DATA = {
    'sitting_static':  {'baseline': 0.007228, 'mask': 0.010217, 'frames': 682,  'type': 'static'},
    'sitting_xyz':     {'baseline': 0.009150, 'mask': 0.031067, 'frames': 864,  'type': 'static'},
    'walking_static':  {'baseline': 0.025241, 'mask': 0.011717, 'frames': 744,  'type': 'dynamic'},
    'walking_xyz':     {'baseline': 0.285579, 'mask': 0.016849, 'frames': 827,  'type': 'dynamic'},
    'walking_halfsphere': {'baseline': 0.203419, 'mask': 0.027066, 'frames': 1020, 'type': 'dynamic'},
}

# --- KITTI Odometry 数据集 ATE 数据 (单位: 米) ---
KITTI_DATA = {
    '00': {'baseline': 0.967078, 'mask': 1.112997, 'frames': 4541},
}

# --- 时序性能分析数据 (来源: deploylog.txt 第 166465 行) ---
TIMING_DATA = {
    'TUM walking_xyz': {'wall_clock_s': 37.42, 'cpu_time_s': 125.87, 'max_rss_gb': 0.895, 'cpu_pct': 402, 'frames': 827},
    'KITTI 00':         {'wall_clock_s': 65.66, 'cpu_time_s': 257.57, 'max_rss_gb': 1.184, 'cpu_pct': 471, 'frames': 4541},
    'EuRoC MH_03':      {'wall_clock_s': 85.54, 'cpu_time_s': 219.76, 'max_rss_gb': 0.741, 'cpu_pct': 336, 'frames': 2700},
}

# --- 参数敏感性分析数据 (来源: deploylog.txt 第 383596 行, 在 TUM walking_xyz 上执行) ---
PARAM_SWEEP = {
    'orb_features': {
        'values': [500, 800, 1000, 1200, 1500, 2000],
        'ate':    [0.280548, 0.281851, 0.273511, 0.280186, 0.284893, 0.276629],
        'label':  'Number of ORB Features',
        'xlabel': 'Number of ORB Features',
    },
    'semantic_weight': {
        'values': [0.0, 0.2, 0.4, 0.5, 0.8, 1.0],
        'ate':    [0.271473, 0.283091, 0.269283, 0.281110, 0.278646, 0.286891],
        'label':  'Semantic Weight $w_{sem}$',
        'xlabel': 'Semantic Weight $w_{sem}$',
    },
    'dynamic_threshold': {
        'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'ate':    [0.283685, 0.271894, 0.279017, 0.277027, 0.278439, 0.275720],
        'label':  'Dynamic Threshold $\\tau_{dyn}$',
        'xlabel': 'Dynamic Threshold $\\tau_{dyn}$',
    },
}

# ================================================================
# 辅助函数
# ================================================================
def calc_improvement(baseline, mask):
    """计算 YOLO-Mask 相对 Baseline 的改进百分比"""
    return round((baseline - mask) / baseline * 100, 1)

def save_figure(fig, filename):
    """保存图表到输出目录"""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    size_kb = os.path.getsize(path) / 1024
    print(f'  [OK] {filename} ({size_kb:.1f} KB)')
    plt.close(fig)

# ================================================================
# Figure 1: 系统架构图
# 对应论文 Section 3.1 System Overview
# ================================================================
def generate_figure1_architecture():
    """生成系统架构流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('System Architecture of Semantic Masking Pipeline',
                 fontsize=14, fontweight='bold', pad=15)

    # 组件定义: (x, y, w, h, label, color, sublabel)
    boxes = [
        (0.3, 2.0, 2.0, 1.2, 'Input Images\n(RGB / RGB-D)', '#4472C4', 'TUM / KITTI'),
        (3.0, 2.0, 2.0, 1.2, 'YOLOv8-nano\nInstance Segmentation', '#ED7D31', 'Offline Detection'),
        (5.7, 2.0, 2.0, 1.2, 'mask_dataset.py\nImage Masking', '#2E7D32', 'BBox \u2192 Zero Out'),
        (8.4, 2.0, 2.0, 1.2, 'ORB-SLAM3\n(Unmodified)', '#7030A0', 'Tracking + Mapping'),
        (8.4, 0.3, 2.0, 0.9, 'ATE Evaluation\n(evo_traj)', '#C00000', 'Trajectory Analysis'),
    ]

    for (x, y, w, h, label, color, sublabel) in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle=BoxStyle("Round", pad=0.3),
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.1, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.35, sublabel, ha='center', va='center',
                fontsize=7, color='white', alpha=0.85)

    # 箭头
    arrows = [
        (2.3, 2.6, 3.0, 2.6), (5.0, 2.6, 5.7, 2.6),
        (7.7, 2.6, 8.4, 2.6), (9.4, 2.0, 9.4, 1.2),
    ]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2.5,
                                    connectionstyle='arc3,rad=0'))

    # 离线/在线边界
    ax.axvline(x=7.9, ymin=0.15, ymax=0.95, color='#FF0000', linewidth=2,
               linestyle='--', alpha=0.5)
    ax.text(7.9, 4.5, 'OFFLINE', ha='center', fontsize=8, color='#FF0000',
            fontweight='bold', alpha=0.7)
    ax.text(4.0, 4.5, 'ONLINE (SLAM Execution)', ha='center', fontsize=8,
            color='#2E7D32', fontweight='bold', alpha=0.7)
    ax.text(9.4, 4.5, 'OFFLINE', ha='center', fontsize=8, color='#FF0000',
            fontweight='bold', alpha=0.7)

    # 数据流标注
    ax.text(2.65, 3.0, 'Detection\nJSON', ha='center', fontsize=6.5,
            color='#555555', fontstyle='italic')
    ax.text(5.35, 3.0, 'Masked\nImages', ha='center', fontsize=6.5,
            color='#555555', fontstyle='italic')
    ax.text(8.05, 3.0, 'Original\nRGB-D', ha='center', fontsize=6.5,
            color='#555555', fontstyle='italic')
    ax.text(9.4, 1.55, 'Trajectory\n(KeyFrameTrajectory.txt)', ha='center',
            fontsize=6.5, color='#555555', fontstyle='italic')
    ax.text(1.3, 1.5, 'Dynamic classes: person, car, bicycle, motorcycle, '
            'bus, truck, bird, cat, dog',
            ha='center', fontsize=7, color='#ED7D31', fontstyle='italic')

    plt.tight_layout()
    save_figure(fig, 'fig1_system_architecture.png')

# ================================================================
# Figure 2: TUM RGB-D ATE 对比
# 对应论文 Section 4.2 TUM RGB-D Results
# ================================================================
def generate_figure2_tum_ate():
    """生成 TUM RGB-D Baseline vs YOLO-Mask ATE 对比图"""
    sequences = list(TUM_DATA.keys())
    seq_labels = [s.replace('_', '\n') for s in sequences]
    baseline_ate = [TUM_DATA[s]['baseline'] for s in sequences]
    mask_ate = [TUM_DATA[s]['mask'] for s in sequences]
    improvements = [calc_improvement(b, m) for b, m in zip(baseline_ate, mask_ate)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) 绝对 ATE 值
    x = np.arange(len(sequences))
    width = 0.35
    bars1 = ax1.bar(x - width/2, baseline_ate, width, label='Baseline (ORB-SLAM3)',
                    color='#4472C4', edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, mask_ate, width, label='YOLO-Mask (Ours)',
                    color='#ED7D31', edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('ATE RMSE (m)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Absolute ATE Values', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_labels, fontsize=9)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars1, baseline_ate):
        offset = 0.004 if val < 0.03 else 0.008
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=90, color='#333333')
    for bar, val in zip(bars2, mask_ate):
        offset = 0.004 if val < 0.03 else 0.008
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=90, color='#333333')

    # (b) 相对改进百分比
    colors = ['#C62828' if v < 0 else '#2E7D32' for v in improvements]
    bars3 = ax2.bar(x, improvements, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Relative Improvement', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_labels, fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axvspan(-0.5, 1.5, alpha=0.07, color='red')
    ax2.axvspan(1.5, 4.5, alpha=0.07, color='green')
    for bar, val in zip(bars3, improvements):
        y_pos = bar.get_height() + 3 if val > 0 else bar.get_height() - 10
        color = '#2E7D32' if val > 0 else '#C62828'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos, f'{val:+.1f}%',
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=9,
                 fontweight='bold', color=color)
    ax2.text(0.5, max(improvements) * 0.85, 'Static:\nFalse Dynamic\nProblem',
             ha='center', fontsize=8, color='#C62828', fontstyle='italic')
    ax2.text(3.0, max(improvements) * 0.85, 'Dynamic:\nSemantic Masking\nEffective',
             ha='center', fontsize=8, color='#2E7D32', fontstyle='italic')

    plt.suptitle('Figure 2: TUM RGB-D ATE Comparison \u2014 Baseline vs YOLO-Mask',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig2_tum_ate.png')

# ================================================================
# Figure 3: 静态场景 vs 动态场景 平均 ATE 对比
# 对应论文 Section 4.2 TUM RGB-D Results (场景类型分析)
# ================================================================
def generate_figure3_scene_type():
    """生成 静态 vs 动态 场景平均 ATE 对比图"""
    static_seqs = [s for s, d in TUM_DATA.items() if d['type'] == 'static']
    dynamic_seqs = [s for s, d in TUM_DATA.items() if d['type'] == 'dynamic']

    static_baseline_mean = np.mean([TUM_DATA[s]['baseline'] for s in static_seqs])
    static_mask_mean = np.mean([TUM_DATA[s]['mask'] for s in static_seqs])
    dynamic_baseline_mean = np.mean([TUM_DATA[s]['baseline'] for s in dynamic_seqs])
    dynamic_mask_mean = np.mean([TUM_DATA[s]['mask'] for s in dynamic_seqs])

    static_imp = calc_improvement(static_baseline_mean, static_mask_mean)
    dynamic_imp = calc_improvement(dynamic_baseline_mean, dynamic_mask_mean)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    categories = ['Static Scenes\n(sitting_*)', 'Dynamic Scenes\n(walking_*)']
    x = np.arange(len(categories))
    width = 0.3

    ax.bar(x - width/2, [static_baseline_mean, dynamic_baseline_mean], width,
           label='Baseline (ORB-SLAM3)', color='#4472C4', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, [static_mask_mean, dynamic_mask_mean], width,
           label='YOLO-Mask (Ours)', color='#ED7D31', edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean ATE RMSE (m)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Static vs Dynamic Scenes \u2014 Mean ATE Comparison',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 标注数值
    ax.text(0, static_baseline_mean + 0.003, f'{static_baseline_mean:.4f}',
            ha='center', fontsize=10, fontweight='bold', color='#4472C4')
    ax.text(0, static_mask_mean + 0.003, f'{static_mask_mean:.4f}',
            ha='center', fontsize=10, fontweight='bold', color='#ED7D31')
    ax.text(1, dynamic_baseline_mean + 0.005, f'{dynamic_baseline_mean:.4f}',
            ha='center', fontsize=10, fontweight='bold', color='#4472C4')
    ax.text(1, dynamic_mask_mean + 0.005, f'{dynamic_mask_mean:.4f}',
            ha='center', fontsize=10, fontweight='bold', color='#ED7D31')

    # 改进百分比标注
    ax.annotate(f'{static_imp:+.1f}%',
                xy=(0, (static_baseline_mean + static_mask_mean)/2),
                fontsize=12, fontweight='bold', color='#C62828', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#C62828', alpha=0.9))
    ax.annotate(f'{dynamic_imp:+.1f}%',
                xy=(1, (dynamic_baseline_mean + dynamic_mask_mean)/2),
                fontsize=12, fontweight='bold', color='#2E7D32', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#2E7D32', alpha=0.9))

    ax.text(0, static_baseline_mean + 0.015, 'YOLO-Mask degrades\nstatic scenes',
            ha='center', fontsize=8, color='#C62828', fontstyle='italic')
    ax.text(1, dynamic_baseline_mean + 0.02, 'YOLO-Mask improves\ndynamic scenes',
            ha='center', fontsize=8, color='#2E7D32', fontstyle='italic')

    plt.tight_layout()
    save_figure(fig, 'fig3_scene_type.png')

# ================================================================
# Figure 4: KITTI Odometry Sequence 00 ATE 对比
# 对应论文 Section 4.3 KITTI Odometry Results
# ================================================================
def generate_figure4_kitti_ate():
    """生成 KITTI 00 ATE 对比图"""
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
    ax.set_ylabel('ATE RMSE (m)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4: KITTI Odometry Sequence 00 \u2014 ATE Comparison',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['KITTI Sequence 00\n(4,541 frames, with loop closure)'], fontsize=11)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax.text(x[0] - width/2, kitti_baseline + 0.02, f'{kitti_baseline:.4f} m',
            ha='center', fontsize=11, fontweight='bold', color='#4472C4')
    ax.text(x[0] + width/2, kitti_mask + 0.02, f'{kitti_mask:.4f} m',
            ha='center', fontsize=11, fontweight='bold', color='#ED7D31')

    ax.text(0, max(kitti_baseline, kitti_mask) * 0.85,
            f'Improvement: {improvement:+.1f}%\n'
            f'(False dynamic masking dominates\n'
            f'in sparse outdoor dynamics)',
            ha='center', fontsize=10, fontstyle='italic', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3F3',
                      edgecolor='#C62828', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'fig4_kitti_ate.png')

# ================================================================
# Figure 5: 运行时性能分析
# 对应论文 Section 4.5 Timing Analysis
# ================================================================
def generate_figure5_timing():
    """生成运行时性能分析图"""
    datasets = list(TIMING_DATA.keys())
    labels = ['TUM\nwalking_xyz\n(827 frames)', 'KITTI 00\n(4,541 frames)',
              'EuRoC\nMH_03\n(2,700 frames)']
    wall_times = [TIMING_DATA[d]['wall_clock_s'] for d in datasets]
    user_times = [TIMING_DATA[d]['cpu_time_s'] for d in datasets]
    max_rss_gb = [TIMING_DATA[d]['max_rss_gb'] for d in datasets]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(datasets))
    width = 0.25
    ax2_twin = ax.twinx()

    bars_wall = ax.bar(x - width, wall_times, width, label='Wall Clock (s)',
                       color='#4472C4', edgecolor='white', linewidth=0.5)
    bars_user = ax.bar(x, user_times, width, label='CPU Time (s)',
                       color='#ED7D31', edgecolor='white', linewidth=0.5)
    bars_mem = ax2_twin.bar(x + width, max_rss_gb, width, label='Max RSS (GB)',
                            color='#2E7D32', edgecolor='white', linewidth=0.5, alpha=0.7)

    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2_twin.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: Runtime Profiling Across Representative Sequences',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val in zip(bars_wall, wall_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{val:.1f}s', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(bars_user, user_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{val:.1f}s', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(bars_mem, max_rss_gb):
        ax2_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                      f'{val:.2f} GB', ha='center', fontsize=8, fontweight='bold')

    ax.text(0.5, 0.98, 'Note: Semantic masking is offline pre-processing;\n'
            'zero runtime overhead during SLAM execution.',
            transform=ax.transAxes, ha='center', fontsize=8, fontstyle='italic',
            color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFF0',
                      edgecolor='#CCCCCC', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'fig5_timing.png')

# ================================================================
# Figure 6: 参数敏感性分析
# 对应论文 Section 4.7 Parameter Sensitivity
# ================================================================
def generate_figure6_parameter_sensitivity():
    """生成参数敏感性分析图 (三组参数扫描)"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#4472C4', '#ED7D31', '#2E7D32']
    markers = ['o', 's', 'D']
    subplot_labels = ['(a) ORB Feature Count', '(b) Semantic Weight', '(c) Dynamic Threshold']

    for idx, (param_name, data) in enumerate(PARAM_SWEEP.items()):
        ax = axes[idx]
        x_vals = data['values']
        y_vals = data['ate']
        color = colors[idx]
        marker = markers[idx]

        ax.plot(x_vals, y_vals, f'{marker}-', color=color, linewidth=2,
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(x_vals, [min(y_vals)-0.005]*len(x_vals), y_vals,
                        alpha=0.1, color=color)
        ax.set_xlabel(data['xlabel'], fontsize=11, fontweight='bold')
        ax.set_ylabel('ATE RMSE (m)', fontsize=11, fontweight='bold')
        ax.set_title(subplot_labels[idx], fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')

        best_idx = np.argmin(y_vals)
        best_val = x_vals[best_idx]
        best_ate = y_vals[best_idx]

        if param_name == 'orb_features':
            label_text = f'Optimal: {best_val} features\nATE = {best_ate:.4f} m'
            xytext_offset = (300, 0.004)
        elif param_name == 'semantic_weight':
            label_text = f'Optimal: $w_{{sem}}$ = {best_val}\nATE = {best_ate:.4f} m'
            xytext_offset = (0.15, 0.004)
        else:
            label_text = f'Optimal: $\\tau_{{dyn}}$ = {best_val}\nATE = {best_ate:.4f} m'
            xytext_offset = (0.08, 0.003)

        ax.annotate(label_text,
                    xy=(best_val, best_ate),
                    xytext=(best_val + xytext_offset[0], best_ate + xytext_offset[1]),
                    fontsize=9, color='#C62828', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#C62828', alpha=0.9))

    plt.suptitle('Figure 6: Parameter Sensitivity Analysis on TUM walking_xyz',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig6_parameter_sensitivity.png')

# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('论文图表生成脚本')
    print('论文: Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments')
    print(f'输出目录: {OUTPUT_DIR}')
    print('=' * 60)
    print()

    generate_figure1_architecture()
    generate_figure2_tum_ate()
    generate_figure3_scene_type()
    generate_figure4_kitti_ate()
    generate_figure5_timing()
    generate_figure6_parameter_sensitivity()

    print()
    print(f'全部 6 张图表已生成到: {OUTPUT_DIR}')
    print('完成!')