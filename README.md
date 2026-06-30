# Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![ORB-SLAM3](https://img.shields.io/badge/ORB--SLAM3-v1.0-orange)](https://github.com/UZ-SLAMLab/ORB_SLAM3)

**论文**: *Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments*  
**作者**: Hantao Chen, School of Computing and Artificial Intelligence, Guangzhou Xinhua University  
**代码仓库**: [https://github.com/RaymondFist/semantic-slam-yolov8](https://github.com/RaymondFist/semantic-slam-yolov8)

---

## 概述

本项目提出了一种图像级语义掩码方法，将 YOLOv8-nano 语义信息集成到 ORB-SLAM3 中，用于动态环境下的视觉惯性 SLAM。

核心思路：不修改 ORB-SLAM3 的 C++ 追踪流水线，而是在离线阶段对输入图像进行预处理 —— 使用 YOLOv8-nano 检测动态物体区域并置零，然后将掩码后的数据集传入未修改的 ORB-SLAM3 系统。这种方法避免了在线特征过滤中遇到的 g2o 优化器兼容性问题，并且零运行时开销。

### 关键结果

| 序列 | Baseline ATE (m) | YOLO-Mask ATE (m) | 改善 |
|------|-----------------|-------------------|------|
| TUM walking_xyz | 0.2856 | 0.0168 | **+94.1%** |
| TUM walking_halfsphere | 0.2034 | 0.0271 | **+86.7%** |
| TUM walking_static | 0.0252 | 0.0117 | **+53.6%** |
| TUM sitting_static | 0.0072 | 0.0102 | -41.4% (假动态) |
| KITTI 00 | 0.9671 | 1.1130 | -15.1% |

---

## 仓库结构

```
semantic-slam/
├── semantic-slam-experiment/     # 实验主工程（部署 + 运行 + 数据）
│   ├── autodl_deploy.sh          # AutoDL 云端一键部署脚本（~2700行）
│   ├── deploylog.txt             # 部署日志（部署后自动生成）
│   ├── requirements.txt          # Python 依赖
│   ├── LICENSE                   # GPL-3.0
│   │
│   ├── scripts/                  # Python 工具脚本
│   │   ├── yolov8_offline_inference.py   # YOLOv8 离线推理（生成检测 JSON）
│   │   ├── mask_dataset.py               # 动态区域掩码生成（Plan C 核心）
│   │   ├── associate.py                  # TUM RGB-D 时间戳对齐
│   │   ├── offline_semantic_validation.py # 离线语义验证分析
│   │   └── generate_planb_report.py      # 实验报告生成
│   │
│   ├── src/                  # C++ SemanticSLAM 模块
│   │   ├── CMakeLists.txt     # CMake 构建配置
│   │   ├── include/           # 头文件
│   │   │   ├── YoloDetector.h
│   │   │   ├── DynamicFeatureFilter.h
│   │   │   ├── SemanticWeights.h
│   │   │   ├── SemanticSLAM.h
│   │   │   └── ORBSLAM3Patch.h
│   │   ├── src/               # 源文件
│   │   │   ├── YoloDetector.cc
│   │   │   ├── DynamicFeatureFilter.cc
│   │   │   ├── SemanticWeights.cc
│   │   │   └── SemanticSLAM.cc
│   │   ├── benchmark/         # 基准测试可执行文件
│   │   │   ├── benchmark_main.cc
│   │   │   ├── tum_benchmark.cc
│   │   │   ├── kitti_benchmark.cc
│   │   │   ├── euroc_benchmark.cc
│   │   │   └── ablation_runner.cc
│   │   ├── config/            # YAML 配置文件
│   │   │   ├── TUM3_baseline.yaml
│   │   │   ├── TUM3_semantic.yaml
│   │   │   ├── KITTI00_baseline.yaml
│   │   │   ├── KITTI00_semantic.yaml
│   │   │   ├── EuRoC_baseline.yaml
│   │   │   └── EuRoC_semantic.yaml
│   │   ├── scripts/           # 构建脚本
│   │   │   ├── build.sh
│   │   │   ├── build.bat
│   │   │   └── export_model.sh
│   │   └── tests/             # 单元测试
│   │
│   ├── patches/                  # ORB-SLAM3 源码补丁
│   │   └── patch_orbslam3.py     # 将 SemanticSLAM 模块注入 ORB-SLAM3
│   │
│   ├── models/                   # 预训练模型文件
│   │   ├── yolov8n-seg.pt        # YOLOv8-nano 分割模型 (~6.7MB)
│   │   ├── yolov8n-seg.onnx      # ONNX 导出（可选）
│   │   ├── yolov8n-seg.trt       # TensorRT 引擎（可选）
│   │   ├── coco.names            # COCO 80 类别名称
│   │   └── ORBvoc.txt            # ORB-SLAM3 词袋词汇表 (~138MB)
│   │
│   ├── data/                     # 数据目录（需自行下载或部署生成）
│   │   ├── datasets/             # 原始数据集
│   │   │   ├── TUM/              # TUM RGB-D (fr3 序列)
│   │   │   ├── KITTI/            # KITTI Odometry (sequence 00)
│   │   │   └── EuRoC/            # EuRoC MAV (MH 序列)
│   │   ├── detections/           # YOLO 检测结果（JSON）
│   │   ├── figure_data/          # 图表数据
│   │   ├── gt_trajectories/      # 真值轨迹
│   │   └── real_trajectories/    # 实际轨迹
│   │
│   └── output/                   # 实验结果输出
│       ├── *_trajectory.txt      # 轨迹文件
│       ├── failure_analysis/     # 失败分析日志
│       ├── timing_logs/          # 性能分析日志
│       └── parameter_sweep/      # 参数敏感性数据
│
└── semantic-slam-figures/        # 论文图表生成
    ├── requirements.txt          # matplotlib, numpy
    ├── src/
    │   └── generate_figures.py   # 图表生成脚本
    ├── data/                     # 图表输入数据
    │   └── experiment_results.json  # 实验结果汇总
    └── output/                   # 生成的论文图表
        └── figures/
```

---

## 环境要求

### 硬件
- CPU: Intel Core i7-12700K (或同等性能)
- RAM: 32 GB DDR4
- GPU: NVIDIA RTX 3080 (10 GB VRAM) 或更高
- 磁盘: 至少 50 GB 可用空间（数据集 + 编译产物）

### 软件
- **操作系统**: Ubuntu 20.04/22.04 (推荐) 或 Windows 10/11
- **Python**: 3.10+
- **C++ 编译器**: GCC 9+ (C++17)
- **CUDA**: 11.8+ (用于 GPU 推理)
- **依赖库**:
  - OpenCV 4.4+
  - Eigen3 3.3+
  - Pangolin 0.8
  - ORB-SLAM3 (含 DBoW2, g2o, Sophus)

### Python 依赖
```
matplotlib>=3.5.0
numpy>=1.21.0
ultralytics>=8.0.0
opencv-python>=4.5.0
evo>=1.20.0
```

---

## 快速开始

### 方式一：AutoDL 云端一键部署（推荐）

```bash
# 上传项目到 AutoDL 实例后
cd semantic-slam-experiment
bash autodl_deploy.sh
```

脚本会自动完成：
1. 安装系统依赖（OpenCV 4.10, Eigen3, Pangolin 等）
2. 编译 ORB-SLAM3（含 C++17 兼容修复和 Sophus NaN 保护）
3. 编译 SemanticSLAM C++ 模块
4. 下载 YOLOv8-nano 模型（PT + ONNX + TensorRT）
5. 下载数据集（TUM fr3, KITTI 00, EuRoC MH）
6. 运行 YOLOv8 离线推理
7. 生成掩码数据集
8. 运行全部实验（Baseline, YOLO-Mask, 消融, 参数扫描）
9. 执行 ATE 评估并生成 `experiment_results.json`

**命令行参数**:
```bash
bash autodl_deploy.sh                 # 完整部署（下载数据集 + 运行实验）
bash autodl_deploy.sh --skip-download # 跳过数据集下载（使用已上传数据）
bash autodl_deploy.sh --resume        # 断点续传（跳过已完成的实验）
```

### 方式二：手动部署

```bash
# 1. 安装系统依赖
sudo apt-get install build-essential cmake libeigen3-dev libglew-dev \
    libboost-dev libssl-dev libgtk2.0-dev python3-pip

# 2. 编译 ORB-SLAM3（参考 autodl_deploy.sh [2/8] 步骤）

# 3. 编译 SemanticSLAM
cd src && mkdir build && cd build
cmake .. -DORB_SLAM3_ROOT=/path/to/ORB_SLAM3
make -j$(nproc)

# 4. 安装 Python 依赖
pip install -r requirements.txt

# 5. 下载模型
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"

# 6. 运行 YOLO 推理
python scripts/yolov8_offline_inference.py \
    --dataset data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
    --output data/detections/walking_xyz \
    --model yolov8n-seg.pt

# 7. 生成掩码数据集
python scripts/mask_dataset.py --all \
    --datasets data/datasets \
    --detections data/detections \
    --output data/datasets_masked

# 8. 运行实验
# （参考 autodl_deploy.sh [7/8] 步骤中的具体命令）
```

---

## 实验设计

### 实验矩阵

| 实验 | 名称 | 说明 | 数据集 | 状态 |
|------|------|------|--------|------|
| E1 | Baseline | 原始 ORB-SLAM3（无语义） | 原始数据集 | 可运行 |
| E1.5 | YOLO-Mask | 原始 ORB-SLAM3 + 掩码数据集 | 掩码数据集 | 可运行 |
| E2a | YOLO-Only | C++ 语义特征过滤（无语义权重） | 原始数据集 | 跳过（g2o崩溃） |
| E2b | GeoConst | C++ 语义 + 光流几何验证 | 原始数据集 | 跳过（g2o崩溃） |
| E3 | Full System | C++ 语义 + 光流 + 语义权重 | 原始数据集 | 跳过（g2o崩溃） |
| E4a | Timing | 运行时性能分析 | 代表性序列 | 可运行 |
| E4b | Failure | 失败案例分析 | 挑战序列 | 可运行 |
| E4c | Sweep | 参数敏感性分析 | TUM walking_xyz | 可运行 |

### 数据集覆盖

| 数据集 | 序列 | 帧数 | 特点 |
|--------|------|------|------|
| TUM RGB-D | sitting_static | 711 | 完全静态基线 |
| TUM RGB-D | sitting_xyz | 859 | 小幅运动，静态场景 |
| TUM RGB-D | walking_static | 797 | 相机静，人动 |
| TUM RGB-D | walking_xyz | 827 | 相机和人都在动 |
| TUM RGB-D | walking_halfsphere | 1068 | 大幅旋转 + 行人 |
| KITTI Odom | 00 | 4541 | 回环 + 丰富动态车辆/行人 |
| EuRoC MAV | MH_01_easy | 3682 | 静态机房基线 |
| EuRoC MAV | MH_03_medium | 2700 | 中等难度 |
| EuRoC MAV | MH_05_difficult | 2273 | 困难序列，低纹理 + 运动模糊 |

---

## 评估指标

- **ATE (Absolute Trajectory Error)**: 使用 `evo_ape` 计算，SE(3) 对齐，尺度校正
- **墙钟时间**: `/usr/bin/time -v` 测量
- **峰值内存**: Max RSS (KB)
- **跟踪丢失次数**: 从 ORB-SLAM3 日志统计

---

## 生成论文图表

```bash
cd semantic-slam-figures
pip install -r requirements.txt

# 将 experiment_results.json 复制到 data/ 目录
cp ../semantic-slam-experiment/data/experiment_results.json data/

# 生成全部图表
python src/generate_figures.py
```

生成的图表保存在 `output/figures/` 目录下：
- `fig01_system_architecture.png` — 系统架构图
- `fig02_tum_ate_comparison.png` — TUM ATE 对比
- `fig03_scene_type_comparison.png` — 静态 vs 动态场景
- `fig04_kitti_ate_comparison.png` — KITTI ATE 对比
- `fig05_timing_profiling.png` — 运行时性能
- `fig06_parameter_sensitivity.png` — 参数敏感性

---

## 技术要点

### Plan C: 图像掩码法

核心创新在于**离线语义掩码策略**：

1. YOLOv8-nano 对全部帧进行离线实例分割，生成每帧检测 JSON
2. `mask_dataset.py` 读取检测结果，将动态物体区域（person, car, bicycle 等 COCO 类别）的像素置零
3. 将掩码后的数据集副本传入**未修改的** ORB-SLAM3

**优势**:
- 零运行时开销（掩码完全离线完成）
- 不修改 SLAM 系统，可移植到任何基于特征的 SLAM 框架
- 避免了 ORB-SLAM3 g2o 优化器中 C++ 在线特征过滤的 double-free 崩溃问题

**局限性（假动态问题）**:
- 无法区分运动物体和静态动态类实例（如坐着的人 vs 走路的人）
- 在静态场景中会移除有效静态特征，导致精度下降

### 已知问题

- **g2o 崩溃**: C++ 在线特征过滤（E2a/E2b/E3）在 `LinearSolverDense` 析构时触发 SIGSEGV（double-free），已通过 backtrace 确认。当前 Plan C 方案通过图像掩码法绕过此问题。修复方案见 Future Work。
- **EuRoC ATE 评估**: 低纹理室内工业场景中 ORB-SLAM3 持续跟踪失败，无法可靠计算 ATE。系统完成执行（时序数据见表 IV），但 EuRoC 的 ATE 对比未包含在论文中。
- **YOLOv8 COCO 限制**: 检测类别限于 80 个 COCO 类别，工业设备等域特定动态物体无法检测。

---

## 引用

如果本工作对您的研究有帮助，请引用：

```bibtex
@article{chen2025semantic,
  title={Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments},
  author={Chen, Hantao},
  journal={arXiv preprint},
  year={2025},
  note={Source code available at https://github.com/RaymondFist/semantic-slam-yolov8}
}
```

---

## 致谢

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) — 视觉惯性 SLAM 基础框架
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 实例分割检测器
- [evo](https://github.com/MichaelGrupp/evo) — SLAM 轨迹评估工具
- [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) — 室内 SLAM 基准数据集
- [KITTI](http://www.cvlibs.net/datasets/kitti) — 自动驾驶基准数据集
- [EuRoC MAV](https://projects.asl.ethz.ch/datasets) — 无人机 SLAM 基准数据集

---

## 许可证

本项目基于 [GNU General Public License v3.0](LICENSE) 开源。所依赖的 ORB-SLAM3 同样使用 GPLv3 许可证。