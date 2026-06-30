# -*- coding: utf-8 -*-
"""
generate_tables.py — 论文表格生成脚本
===========================================
论文: Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments
用途: 根据实验数据生成所有论文表格的 CSV 数据文件，用于数据分析和评审提交
依赖: 无 (仅 Python 标准库)
运行: python generate_tables.py

生成表格列表:
  Table I:   TUM RGB-D ATE 结果表
  Table II:  KITTI Odometry Sequence 00 ATE 结果表
  Table III: 消融实验 (Ablation Study) 结果表
  Table IV:  运行时性能分析 (Timing Profiling) 表
  Table V:   失败案例分析 (Failure Analysis) 表
  Table VI:  参数敏感性分析 (Parameter Sensitivity) 表

数据来源: 全部来自 AutoDL 云平台上的真实实验运行结果
评估结果: ../data/experiment_results.json
输出目录: ../output/tables/
"""

import csv
import os
import sys

# ================================================================
# 配置
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 实验原始数据 (全部来自真实实验运行)
# 来源: deploylog.txt 第 383598 行 "实验结果导出完成"
# 评估工具: evo_traj (evo Python package)
# 评估方法: ATE RMSE after SE(3) Umeyama alignment
# ================================================================

# --- TUM RGB-D 数据集 ATE 数据 (单位: 米) ---
# 实验配置: E1 = Baseline (ORB-SLAM3 on original images)
#           E1.5 = YOLO-Mask (ORB-SLAM3 on pre-masked images)
TUM_DATA = {
    'sitting_static':      {'baseline': 0.007228, 'mask': 0.010217, 'frames': 682,  'type': 'Static'},
    'sitting_xyz':         {'baseline': 0.009150, 'mask': 0.031067, 'frames': 864,  'type': 'Static'},
    'walking_static':      {'baseline': 0.025241, 'mask': 0.011717, 'frames': 744,  'type': 'Dynamic'},
    'walking_xyz':         {'baseline': 0.285579, 'mask': 0.016849, 'frames': 827,  'type': 'Dynamic'},
    'walking_halfsphere':  {'baseline': 0.203419, 'mask': 0.027066, 'frames': 1020, 'type': 'Dynamic'},
}

# --- KITTI Odometry 数据集 ATE 数据 (单位: 米) ---
KITTI_DATA = {
    '00': {'baseline': 0.967078, 'mask': 1.112997, 'frames': 4541},
}

# --- 时序性能分析数据 (来源: deploylog.txt 第 166465 行) ---
TIMING_DATA = [
    {'sequence': 'TUM walking_xyz', 'frames': 827,  'wall_clock_s': 37.42, 'cpu_time_s': 125.87, 'max_rss_gb': 0.895, 'cpu_pct': 402},
    {'sequence': 'KITTI 00',        'frames': 4541, 'wall_clock_s': 65.66, 'cpu_time_s': 257.57, 'max_rss_gb': 1.184, 'cpu_pct': 471},
    {'sequence': 'EuRoC MH_03',     'frames': 2700, 'wall_clock_s': 85.54, 'cpu_time_s': 219.76, 'max_rss_gb': 0.741, 'cpu_pct': 336},
]

# --- 失败案例分析数据 (来源: deploylog.txt 实验 E4b) ---
FAILURE_DATA = [
    {'sequence': 'TUM walking_halfsphere', 'challenge_type': 'Dynamic occlusion + fast rotation', 'track_lost': 0, 'status': 'Completed'},
    {'sequence': 'TUM walking_xyz',        'challenge_type': 'Dynamic occlusion',                  'track_lost': 0, 'status': 'Completed'},
    {'sequence': 'KITTI 00 (challenge)',   'challenge_type': 'Outdoor large-scale + loop closure', 'track_lost': 0, 'status': 'Completed'},
    {'sequence': 'EuRoC MH_05_difficult',  'challenge_type': 'Low texture + motion blur',          'track_lost': 0, 'status': 'Completed'},
]

# --- 参数敏感性分析数据 (来源: deploylog.txt 第 383596 行, 在 TUM walking_xyz 上执行) ---
PARAM_SWEEP = {
    'orb_features': {
        'values': [500, 800, 1000, 1200, 1500, 2000],
        'ate':    [0.280548, 0.281851, 0.273511, 0.280186, 0.284893, 0.276629],
    },
    'semantic_weight': {
        'values': [0.0, 0.2, 0.4, 0.5, 0.8, 1.0],
        'ate':    [0.271473, 0.283091, 0.269283, 0.281110, 0.278646, 0.286891],
    },
    'dynamic_threshold': {
        'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'ate':    [0.283685, 0.271894, 0.279017, 0.277027, 0.278439, 0.275720],
    },
}

# ================================================================
# 辅助函数
# ================================================================
def calc_improvement(baseline, mask):
    """计算 YOLO-Mask 相对 Baseline 的改进百分比"""
    return round((baseline - mask) / baseline * 100, 1)

def fmt_pct(value):
    """格式化百分比为字符串"""
    return f'{value:+.1f}%'

def write_csv(filename, rows):
    """写入 CSV 文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f'  [OK] {filename} ({len(rows)-1} data rows)')

def print_table(title, rows):
    """打印格式化的表格"""
    print(f'\n{"=" * 80}')
    print(f'  {title}')
    print(f'{"=" * 80}')
    # 计算列宽
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
    # 打印表头
    header = ' | '.join(str(cell).ljust(w) for cell, w in zip(rows[0], col_widths))
    print(f'  {header}')
    print(f'  {"-" * len(header)}')
    # 打印数据行
    for row in rows[1:]:
        line = ' | '.join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(f'  {line}')

# ================================================================
# Table I: TUM RGB-D ATE Results
# ================================================================
def generate_table1_tum_results():
    """生成 TUM RGB-D ATE 结果表"""
    rows = [['Sequence', 'Baseline ATE (m)', 'YOLO-Mask ATE (m)', 'Improvement', 'Scene Type']]
    for seq_name in ['sitting_static', 'sitting_xyz', 'walking_static',
                     'walking_xyz', 'walking_halfsphere']:
        d = TUM_DATA[seq_name]
        imp = calc_improvement(d['baseline'], d['mask'])
        rows.append([
            seq_name,
            f'{d["baseline"]:.4f}',
            f'{d["mask"]:.4f}',
            fmt_pct(imp),
            d['type'],
        ])
    print_table('Table I: TUM RGB-D ATE Results', rows)
    write_csv('table1_tum_ate_results.csv', rows)
    return rows

# ================================================================
# Table II: KITTI Odometry ATE Results
# ================================================================
def generate_table2_kitti_results():
    """生成 KITTI ATE 结果表"""
    d = KITTI_DATA['00']
    imp = calc_improvement(d['baseline'], d['mask'])
    rows = [
        ['Sequence', 'Frames', 'Baseline ATE (m)', 'YOLO-Mask ATE (m)', 'Improvement'],
        ['00', str(d['frames']), f'{d["baseline"]:.4f}', f'{d["mask"]:.4f}', fmt_pct(imp)],
    ]
    print_table('Table II: KITTI Odometry Sequence 00 ATE Results', rows)
    write_csv('table2_kitti_ate_results.csv', rows)
    return rows

# ================================================================
# Table III: Ablation Study
# ================================================================
def generate_table3_ablation():
    """生成消融实验结果表"""
    interpretations = {
        'sitting_static': 'False dynamic: sitting person masked',
        'sitting_xyz': 'False dynamic: person occupies large area',
        'walking_static': 'Dynamic features removed successfully',
        'walking_xyz': 'Strongest improvement: two people walking',
        'walking_halfsphere': 'Rotation + walking handled well',
        'KITTI 00': 'Sparse dynamics + loop closure; false dynamic dominates',
    }

    rows = [['Sequence', 'Baseline ATE (m)', 'YOLO-Mask ATE (m)', 'Improvement', 'Interpretation']]

    for seq_name in TUM_DATA:
        d = TUM_DATA[seq_name]
        imp = calc_improvement(d['baseline'], d['mask'])
        rows.append([
            f'TUM {seq_name}',
            f'{d["baseline"]:.4f}',
            f'{d["mask"]:.4f}',
            fmt_pct(imp),
            interpretations[seq_name],
        ])

    # KITTI
    d = KITTI_DATA['00']
    imp = calc_improvement(d['baseline'], d['mask'])
    rows.append([
        'KITTI 00',
        f'{d["baseline"]:.4f}',
        f'{d["mask"]:.4f}',
        fmt_pct(imp),
        interpretations['KITTI 00'],
    ])

    print_table('Table III: Ablation Study Results', rows)
    write_csv('table3_ablation_study.csv', rows)
    return rows

# ================================================================
# Table IV: Runtime Profiling
# ================================================================
def generate_table4_timing():
    """生成运行时性能分析表"""
    rows = [['Sequence', 'Frames', 'Wall Clock (s)', 'CPU Time (s)', 'Max RSS (GB)', 'CPU %']]
    for d in TIMING_DATA:
        rows.append([
            d['sequence'],
            str(d['frames']),
            f'{d["wall_clock_s"]:.2f}',
            f'{d["cpu_time_s"]:.2f}',
            f'{d["max_rss_gb"]:.3f}',
            f'{d["cpu_pct"]}%',
        ])
    print_table('Table IV: Runtime Profiling Results', rows)
    write_csv('table4_timing_profiling.csv', rows)
    return rows

# ================================================================
# Table V: Failure Analysis
# ================================================================
def generate_table5_failure():
    """生成失败案例分析表"""
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
# Table VI: Parameter Sensitivity
# ================================================================
def generate_table6_parameter_sensitivity():
    """生成参数敏感性分析表"""
    orb_vals = PARAM_SWEEP['orb_features']['values']
    orb_ate = PARAM_SWEEP['orb_features']['ate']
    sem_vals = PARAM_SWEEP['semantic_weight']['values']
    sem_ate = PARAM_SWEEP['semantic_weight']['ate']
    dyn_vals = PARAM_SWEEP['dynamic_threshold']['values']
    dyn_ate = PARAM_SWEEP['dynamic_threshold']['ate']

    rows = [['ORB Features', 'ATE (m)', 'Semantic Weight', 'ATE (m)', 'Dynamic Threshold', 'ATE (m)']]
    for i in range(len(orb_vals)):
        rows.append([
            str(orb_vals[i]),
            f'{orb_ate[i]:.4f}',
            f'{sem_vals[i]:.1f}',
            f'{sem_ate[i]:.4f}',
            f'{dyn_vals[i]:.1f}',
            f'{dyn_ate[i]:.4f}',
        ])

    print_table('Table VI: Parameter Sensitivity Results on TUM walking_xyz', rows)
    write_csv('table6_parameter_sensitivity.csv', rows)
    return rows

# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('论文表格生成脚本')
    print('论文: Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments')
    print(f'输出目录: {OUTPUT_DIR}')
    print('=' * 60)

    generate_table1_tum_results()
    generate_table2_kitti_results()
    generate_table3_ablation()
    generate_table4_timing()
    generate_table5_failure()
    generate_table6_parameter_sensitivity()

    print(f'\n全部 6 张表格 CSV 文件已生成到: {OUTPUT_DIR}')
    print('完成!')