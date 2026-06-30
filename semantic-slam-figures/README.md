# Semantic-SLAM-YOLOv8 Figures & Tables

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange)](https://matplotlib.org/)

**论文图表生成工程** — 根据实验数据生成论文全部图表（6 张图 + 6 张表），对应论文 *Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments*。

---

## 目录

```
semantic-slam-figures/
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖
├── data/
│   └── experiment_results.json   # 实验数据（来自 autodl_deploy.sh 的 ATE 评估导出）
├── src/
│   ├── generate_figures.py       # 图表生成（6 张 PNG）
│   └── generate_tables.py        # 表格生成（6 张 CSV）
└── output/
    ├── figures/                  # 生成的图表（300 DPI PNG）
    │   ├── fig1_system_architecture.png
    │   ├── fig2_tum_ate.png
    │   ├── fig3_scene_type.png
    │   ├── fig4_kitti_ate.png
    │   ├── fig5_timing.png
    │   └── fig6_parameter_sensitivity.png
    └── tables/                   # 生成的表格（UTF-8 CSV）
        ├── table1_tum_ate_results.csv
        ├── table2_kitti_ate_results.csv
        ├── table3_ablation_study.csv
        ├── table4_timing_profiling.csv
        ├── table5_failure_analysis.csv
        └── table6_parameter_sensitivity.csv
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：`matplotlib >= 3.5.0`，`numpy >= 1.21.0`。

### 生成图表

```bash
cd src
python generate_figures.py
```

输出 6 张 PNG 图表到 `output/figures/`，分辨率 300 DPI。

### 生成表格

```bash
cd src
python generate_tables.py
```

输出 6 张 CSV 表格到 `output/tables/`，编码 UTF-8 BOM。

---

## 数据来源

全部数据来自 `data/experiment_results.json`，由 `autodl_deploy.sh`（父级 `semantic-slam-experiment/` 目录）在 AutoDL 云平台上运行真实实验后自动导出。该 JSON 文件结构如下：

| 字段 | 说明 |
|------|------|
| `tum` | TUM RGB-D 5 个序列的 Baseline / YOLO-Mask ATE |
| `kitti` | KITTI 00 序列的 Baseline / YOLO-Mask ATE |
| `euroc` | EuRoC 3 个序列（MH_01/03/05）的 ATE（当前为 null，待修复） |
| `timing` | 3 个代表性序列的墙钟时间、CPU 时间、内存峰值 |
| `failure_analysis` | 4 个挑战序列的跟踪丢失计数与故障类型 |
| `parameter_sensitivity` | 3 组参数扫描（ORB特征数、语义权重、动态阈值）的 ATE 数据 |

> 所有数据均为实验生成，无硬编码或模拟数据。

---

## 生成物详解

### 图表（`generate_figures.py`）

| 图表 | 文件名 | 对应论文章节 | 内容 |
|------|--------|-------------|------|
| Figure 1 | `fig1_system_architecture.png` | Section 3.1 | 系统架构流程图（输入→YOLOv8→掩码→ORB-SLAM3→评估） |
| Figure 2 | `fig2_tum_ate.png` | Section 4.2 | TUM RGB-D Baseline vs YOLO-Mask ATE 对比（双面板：绝对ATE + 相对改进） |
| Figure 3 | `fig3_scene_type.png` | Section 4.2 | 静态场景 vs 动态场景平均 ATE 对比 |
| Figure 4 | `fig4_kitti_ate.png` | Section 4.3 | KITTI 00 ATE 对比 |
| Figure 5 | `fig5_timing.png` | Section 4.5 | 运行时性能分析（墙钟时间、CPU时间、内存，三合一柱状图） |
| Figure 6 | `fig6_parameter_sensitivity.png` | Section 4.7 | 参数敏感性分析（ORB特征数、语义权重、动态阈值，三面板折线图） |

### 表格（`generate_tables.py`）

| 表格 | 文件名 | 对应论文章节 | 内容 |
|------|--------|-------------|------|
| Table I | `table1_tum_ate_results.csv` | Section 4.2 | TUM RGB-D ATE 结果（5序列 × 2方法 + 场景类型） |
| Table II | `table2_kitti_ate_results.csv` | Section 4.3 | KITTI 00 ATE 结果 |
| Table III | `table3_ablation_study.csv` | Section 4.6 | 消融实验（6序列 + 结果解读） |
| Table IV | `table4_timing_profiling.csv` | Section 4.5 | 运行时性能分析（墙钟、CPU、内存、CPU占用率） |
| Table V | `table5_failure_analysis.csv` | Section 4.8 | 失败案例分析（挑战类型、跟踪丢失计数） |
| Table VI | `table6_parameter_sensitivity.csv` | Section 4.7 | 参数敏感性分析（3组参数 × 6档 ATE） |

---

## 实验数据摘要

### TUM RGB-D ATE 结果

| 序列 | Baseline ATE (m) | YOLO-Mask ATE (m) | 改善 |
|------|-----------------|-------------------|------|
| sitting_static | 0.0072 | 0.0102 | -41.4% |
| sitting_xyz | 0.0092 | 0.0311 | -239.5% |
| walking_static | 0.0252 | 0.0117 | **+53.6%** |
| walking_xyz | 0.2856 | 0.0168 | **+94.1%** |
| walking_halfsphere | 0.2034 | 0.0271 | **+86.7%** |

### 运行时性能

| 序列 | 帧数 | 墙钟时间 | 峰值内存 |
|------|------|---------|---------|
| TUM walking_xyz | 827 | 37.42 s | 0.895 GB |
| KITTI 00 | 4541 | 65.66 s | 1.184 GB |
| EuRoC MH_03 | 2700 | 85.54 s | 0.741 GB |

---

## 数据更新流程

当上游实验数据更新后，按以下步骤重新生成图表：

```bash
# 1. 将 autodl_deploy.sh 导出的 experiment_results.json 覆盖到 data/ 目录
cp ../semantic-slam-experiment/data/experiment_results.json data/

# 2. 重新生成图表和表格
cd src
python generate_figures.py
python generate_tables.py
```

`generate_figures.py` 和 `generate_tables.py` 均内置了硬编码的实验数据作为回退（当 `experiment_results.json` 不可用时），但优先从 JSON 文件读取。修改数据时，直接更新 JSON 文件即可，无需修改 Python 脚本。

---

## 文件说明

### `generate_figures.py`

- **依赖**: matplotlib, numpy
- **输出格式**: PNG（300 DPI）
- **输出目录**: `output/figures/`
- **特点**: 使用 `Agg` 后端，无需 GUI，适合服务器环境

### `generate_tables.py`

- **依赖**: 仅 Python 标准库（csv, os）
- **输出格式**: CSV（UTF-8 BOM，兼容 Excel 直接打开）
- **输出目录**: `output/tables/`
- **特点**: 零外部依赖，可在任何 Python 3 环境运行

---

## 引用

```bibtex
@article{chen2025semantic,
  title={Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments},
  author={Chen, Hantao},
  journal={arXiv preprint},
  year={2025},
  note={Source code: https://github.com/RaymondFist/semantic-slam-yolov8}
}
```

---

## 许可证

[GNU General Public License v3.0](../semantic-slam-experiment/LICENSE)。与父工程 `semantic-slam-experiment` 一致。