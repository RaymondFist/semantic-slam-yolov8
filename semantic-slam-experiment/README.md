# Semantic-SLAM-YOLOv8 Experiment

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![ORB-SLAM3](https://img.shields.io/badge/ORB--SLAM3-v1.0-orange)](https://github.com/UZ-SLAMLab/ORB_SLAM3)

**论文配套实验工程** — 基于 YOLOv8-nano 语义掩码的动态环境 ORB-SLAM3 实验。  
论文: *Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments*  

---

## 目录

```
semantic-slam-experiment/
├── autodl_deploy.sh              # AutoDL 云端一键部署（~2700行）
├── deploylog.txt                 # 部署日志（部署后自动生成）
├── requirements.txt              # Python 依赖
├── LICENSE                       # GPL-3.0
│
├── scripts/                      # Python 工具脚本
│   ├── yolov8_offline_inference.py      # YOLOv8 离线推理
│   ├── mask_dataset.py                  # 动态区域掩码生成
│   ├── associate.py                     # TUM RGB-D 时间戳对齐
│   ├── offline_semantic_validation.py   # 离线语义过滤分析
│   └── generate_planb_report.py         # 实验报告生成
│
├── src/                          # C++ SemanticSLAM 模块
│   ├── CMakeLists.txt            # CMake 构建配置
│   ├── include/                  # 头文件（5个）
│   ├── src/                      # 源文件（4个）
│   ├── benchmark/                # 基准测试（TUM/KITTI/EuRoC/消融）
│   ├── config/                   # YAML 配置（Baseline/Semantic/YOLO-Only/GeoConst）
│   ├── scripts/                  # 构建脚本（build.sh, build.bat, export_model.sh）
│   └── tests/                    # 单元测试
│
├── patches/                      # ORB-SLAM3 源码补丁
│   └── patch_orbslam3.py         # 注入 SemanticSLAM 模块到 ORB-SLAM3
│
├── models/                       # 预训练模型
│   ├── yolov8n-seg.pt            # YOLOv8-nano 分割模型 (~6.7MB)
│   ├── coco.names                # COCO 80 类别
│   └── ORBvoc.txt                # ORB-SLAM3 词袋 (~138MB)
│
├── data/                         # 数据目录
│   ├── datasets/                 # 原始数据集（TUM/KITTI/EuRoC）
│   ├── detections/               # YOLO 检测结果 JSON
│   ├── figure_data/              # 图表数据
│   ├── gt_trajectories/          # 真值轨迹
│   └── real_trajectories/        # 实际轨迹
│
└── output/                       # 实验结果
    ├── baseline_*.txt            # Baseline 轨迹
    ├── mask_*.txt                # YOLO-Mask 轨迹
    ├── failure_analysis/         # 失败分析
    ├── timing_logs/              # 性能分析
    └── parameter_sweep/          # 参数敏感性
```

---

## 实验方案

| 方案 | 方法 | 说明 | 状态 |
|------|------|------|------|
| **Plan A** | C++ 在线特征过滤 | 修改 ORB-SLAM3 追踪流水线，在 Track() 中按帧过滤动态特征 | 失败（g2o 崩溃） |
| **Plan B** | C++ 语义权重 | 在 g2o 优化前为语义特征赋低权重 | 失败（g2o 崩溃） |
| **Plan C** | 离线图像掩码 | 预处理阶段用 YOLOv8 掩码动态区域，传入未修改的 ORB-SLAM3 | **成功（论文方案）** |

### Plan C 工作流

```
原始图像 → YOLOv8-nano 离线推理 → 检测 JSON
                                        ↓
                               mask_dataset.py → 掩码图像
                                        ↓
                             未修改的 ORB-SLAM3 → 轨迹
```

### g2o 崩溃原因

Plan A/B 在运行时崩溃，backtrace 确认问题出在 `g2o::LinearSolverDense` 析构时的 double-free。根本原因：ORB-SLAM3 的 `Optimizer::PoseOptimization()` 和 `Tracking::Track()` 在 g2o 优化器生命周期与特征指针生命周期之间没有正确同步，在语义特征过滤后已删除的特征指针仍被 g2o 内部引用。完整 backtrace 详见 `autodl_deploy.sh` 中的 Plan B 章节。

---

## 快速开始

### 一键部署（AutoDL 云端）

```bash
bash autodl_deploy.sh                      # 完整部署
bash autodl_deploy.sh --skip-download      # 跳过数据集下载
bash autodl_deploy.sh --resume             # 断点续传
```

脚本自动执行 8 个阶段:
1. **[0/8]** 安装系统依赖（build-essential, cmake, Eigen3, OpenCV 4.10, Pangolin 等）
2. **[1/8]** 编译 ORB-SLAM3（含 C++17 兼容修复和 Sophus NaN 保护）
3. **[2/8]** 编译 SemanticSLAM C++ 模块
4. **[3/8]** 下载 YOLOv8-nano 模型（PT + ONNX + TensorRT）
5. **[4/8]** 下载数据集（TUM fr3 ×5, KITTI 00, EuRoC MH ×3）
6. **[5/8]** 运行 YOLOv8 离线推理（生成检测 JSON）
7. **[6/8]** 生成掩码数据集（Plan C 核心步骤）
8. **[7/8]** 运行全部实验（Baseline, YOLO-Mask, 消融, 参数扫描）
9. **[8/8]** 执行 ATE 评估并生成 `experiment_results.json`

### 手动部署

```bash
# 1. 安装依赖
sudo apt-get install build-essential cmake git libeigen3-dev libglew-dev \
    libboost-dev libssl-dev libgtk2.0-dev python3-pip
pip install ultralytics opencv-python-headless evo scipy

# 2. 编译 ORB-SLAM3（参考 autodl_deploy.sh [1/8] 步骤）

# 3. 编译 SemanticSLAM
cd src && mkdir build && cd build
cmake .. -DORB_SLAM3_ROOT=/path/to/ORB_SLAM3
make -j$(nproc)

# 4. 下载模型
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"

# 5. 运行 YOLO 离线推理
python scripts/yolov8_offline_inference.py \
    --dataset data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
    --output data/detections/walking_xyz \
    --model yolov8n-seg.pt

# 6. 生成掩码数据集
python scripts/mask_dataset.py --all \
    --datasets data/datasets \
    --detections data/detections \
    --output data/datasets_masked

# 7. 运行实验
# Baseline: ORB-SLAM3 原始数据集
# YOLO-Mask: ORB-SLAM3 掩码数据集
# 详见 autodl_deploy.sh [7/8] 步骤
```

---

## 脚本详解

### `yolov8_offline_inference.py`

对指定数据集运行 YOLOv8-nano 实例分割，输出每帧检测结果 JSON。

```bash
python scripts/yolov8_offline_inference.py \
    --dataset <path/to/sequence> \
    --output <path/to/detections> \
    --model yolov8n-seg.pt \
    --conf 0.45 --nms 0.45
```

每帧输出 JSON 格式:
```json
{
  "frame": 0,
  "detections": [
    {"class_id": 0, "class_name": "person", "confidence": 0.92,
     "bbox": [x1, y1, x2, y2], "mask": [[x1,y1,x2,y2,...], ...]},
    ...
  ]
}
```

### `mask_dataset.py`

读取 YOLO 检测结果，将动态物体区域像素置零，生成掩码数据集副本。

**动态类别**（COCO ID）: `person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7), cat(16), dog(17)`

```bash
python scripts/mask_dataset.py --all \
    --datasets data/datasets \
    --detections data/detections \
    --output data/datasets_masked
```

### `offline_semantic_validation.py`

无需运行 ORB-SLAM3，离线分析 YOLO 检测结果，量化语义过滤的预期效果。

输出指标:
- 动态物体覆盖率（占图像面积百分比）
- 预期特征过滤率（基于 bbox 面积估算）
- 每帧统计 + 序列级汇总

```bash
python scripts/offline_semantic_validation.py \
    --detections data/detections \
    --output output/semantic_validation.json
```

### `generate_planb_report.py`

生成综合验证报告，包含 Baseline ATE 对比表、离线语义过滤分析、动态退化分析、理论改进估算。

```bash
python scripts/generate_planb_report.py
# 输出: output/planb_validation_report.md
```

### `associate.py`

TUM RGB-D 数据集的时间戳对齐工具，将 `rgb.txt` 与 `depth.txt` 按时间戳最近邻关联。

```bash
python scripts/associate.py rgb.txt depth.txt > associations.txt
```

---

## C++ 模块

### 目录结构

```
src/
├── CMakeLists.txt              # 构建配置（C++17, OpenCV 4.2+, Eigen3, TensorRT 可选）
├── include/                    # 头文件
│   ├── YoloDetector.h          # ONNX Runtime 推理，加载 YOLOv8-nano 模型
│   ├── DynamicFeatureFilter.h  # 光流几何验证 + 动态特征标记
│   ├── SemanticWeights.h       # 语义类别权重（building=1.0, road=0.7, vegetation=0.5）
│   ├── SemanticSLAM.h          # 主入口，协调 YOLO + 过滤 + 权重
│   └── ORBSLAM3Patch.h         # ORB-SLAM3 接口补丁（特征级别拦截）
├── src/                        # 源文件
│   ├── YoloDetector.cc
│   ├── DynamicFeatureFilter.cc
│   ├── SemanticWeights.cc
│   └── SemanticSLAM.cc
├── benchmark/                  # 基准测试可执行文件
│   ├── benchmark_main.cc       # 入口
│   ├── tum_benchmark.cc        # TUM RGB-D 基准
│   ├── kitti_benchmark.cc      # KITTI 基准
│   ├── euroc_benchmark.cc      # EuRoC MAV 基准
│   ├── ablation_runner.cc      # 消融实验运行器
│   └── benchmark_utils.h       # 公共工具
├── config/                     # YAML 配置文件
│   ├── TUM3_baseline.yaml
│   ├── TUM3_semantic.yaml
│   ├── TUM3_geoconst.yaml
│   ├── TUM3_yolo_only.yaml
│   ├── TUM1_semantic.yaml
│   ├── KITTI00_baseline.yaml
│   ├── KITTI00_semantic.yaml
│   ├── KITTI00_geoconst.yaml
│   ├── KITTI00_yolo_only.yaml
│   ├── EuRoC_baseline.yaml
│   ├── EuRoC_semantic.yaml
│   ├── EuRoC_geoconst.yaml
│   └── EuRoC_yolo_only.yaml
├── scripts/                    # 构建辅助
│   ├── build.sh                # Linux 编译
│   ├── build.bat               # Windows 编译
│   └── export_model.sh         # PT → ONNX → TensorRT 模型导出
└── tests/                      # 单元测试
    ├── CMakeLists.txt
    ├── test_yolo_detector.cc
    ├── test_dynamic_filter.cc
    ├── test_semantic_weights.cc
    ├── test_integration.cc
    └── test_utils.h
```

### 编译

```bash
cd src
mkdir build && cd build
cmake .. -DORB_SLAM3_ROOT=/path/to/ORB_SLAM3
make -j$(nproc)
```

### 配置文件

`config/` 目录下按数据集和实验模式组织:

| 配置文件 | 数据集 | 模式 |
|----------|--------|------|
| `TUM3_baseline.yaml` | TUM fr3 | 原始 ORB-SLAM3 |
| `TUM3_semantic.yaml` | TUM fr3 | 语义 + 几何 |
| `TUM3_geoconst.yaml` | TUM fr3 | 仅几何约束 |
| `TUM3_yolo_only.yaml` | TUM fr3 | 仅语义过滤 |
| `KITTI00_baseline.yaml` | KITTI 00 | 原始 ORB-SLAM3 |
| `KITTI00_semantic.yaml` | KITTI 00 | 语义 + 几何 |
| `EuRoC_baseline.yaml` | EuRoC MH | 原始 ORB-SLAM3 |
| `EuRoC_semantic.yaml` | EuRoC MH | 语义 + 几何 |

### 单元测试

```bash
cd src/build
ctest --output-on-failure
```

测试覆盖:
- `test_yolo_detector` — YOLO 模型加载、推理、边界情况
- `test_dynamic_filter` — 动态特征过滤、光流验证
- `test_semantic_weights` — 语义权重计算、一致性检查
- `test_integration` — 端到端集成测试

---

## 数据集

| 数据集 | 序列 | 帧数 | 大小 | 特点 |
|--------|------|------|------|------|
| TUM fr3 | sitting_static | 711 | ~423MB | 两人坐着聊天（静态基线） |
| TUM fr3 | sitting_xyz | 859 | ~739MB | 咖啡机旁，小幅运动 |
| TUM fr3 | walking_static | 797 | ~452MB | 相机不动，两人走动 |
| TUM fr3 | walking_xyz | 827 | ~503MB | 相机和人都在动 |
| TUM fr3 | walking_halfsphere | 1068 | ~610MB | 大幅旋转 + 行人 |
| KITTI | 00 | 4541 | — | 回环 + 动态车辆/行人 |
| EuRoC | MH_01_easy | 3682 | ~150MB | 明亮机房 |
| EuRoC | MH_03_medium | 2700 | ~105MB | 中等难度 |
| EuRoC | MH_05_difficult | 2273 | ~79MB | 低纹理 + 运动模糊 |

### 数据集下载地址

| 数据集 | 官方下载地址 | 说明 |
|--------|-------------|------|
| **TUM RGB-D** | [https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) | 下载 `freiburg3_sitting_*` 和 `freiburg3_walking_*` 序列（tgz 格式） |
| **KITTI Odometry** | [https://www.cvlibs.net/datasets/kitti/eval_odometry.php](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | 下载 odometry `grayscale` + `calibration` + `ground truth poses`（需注册） |
| **EuRoC MAV** | [https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) | 下载 `Machine Hall 01/03/05`（ASL 格式 zip） |

> `autodl_deploy.sh` 会自动下载数据集到 `data/datasets/` 目录。手动下载时请将数据集解压到对应子目录：`data/datasets/TUM/`、`data/datasets/KITTI/`、`data/datasets/EuRoC/`。

---

## 实验结果

### 关键结果

| 序列 | Baseline ATE (m) | YOLO-Mask ATE (m) | 改善 |
|------|-----------------|-------------------|------|
| TUM walking_xyz | 0.2856 | 0.0168 | **+94.1%** |
| TUM walking_halfsphere | 0.2034 | 0.0271 | **+86.7%** |
| TUM walking_static | 0.0252 | 0.0117 | **+53.6%** |
| TUM sitting_static | 0.0072 | 0.0102 | -41.4% |
| TUM sitting_xyz | 0.0092 | 0.0311 | -239.5% |
| KITTI 00 | 0.9671 | 1.1130 | -15.1% |

### 输出文件

```
output/
├── baseline_tum_walking_xyz.txt          # Baseline 轨迹
├── mask_tum_walking_xyz.txt              # YOLO-Mask 轨迹
├── baseline_kitti00.txt                  # KITTI Baseline
├── mask_kitti00.txt                      # KITTI YOLO-Mask
│
├── failure_analysis/
│   ├── failure_summary.txt               # 失败汇总
│   ├── tum_walking_xyz_failure.log       # 失败帧日志
│   ├── tum_walking_halfsphere_failure.log
│   ├── kitti_00_challenge_failure.log
│   └── euroc_MH_05_difficult_failure.log
│
├── timing_logs/
│   ├── tum_walking_xyz_timing.log        # 墙钟时间: 37.4s
│   ├── kitti_00_timing.log               # 墙钟时间: 65.7s
│   └── euroc_mh_03_timing.log            # 墙钟时间: 85.5s
│
└── parameter_sweep/
    ├── TUM3_sweep.yaml                   # 扫描配置
    ├── tum_orb_500_traj.txt → tum_orb_2000_traj.txt  # ORB 特征数扫描
    ├── tum_semw_0.0_traj.txt → tum_semw_1.0_traj.txt # 语义权重扫描
    └── tum_dyn_0.3_traj.txt → tum_dyn_0.8_traj.txt   # 动态阈值扫描
```

---

## 性能数据

| 序列 | 墙钟时间 | 峰值内存 | 模型 |
|------|---------|---------|------|
| TUM walking_xyz | 37.4 s | <1.2 GB | YOLOv8-nano (3.2M params) |
| KITTI 00 | 65.7 s | <1.2 GB | 同上 |
| EuRoC MH_03 | 85.5 s | <1.2 GB | 同上 |

YOLOv8-nano 离线推理在 RTX 3080 上约 100 FPS，掩码处理完全离线，SLAM 运行时零额外开销。

---

## 已知问题

1. **g2o 崩溃**: Plan A/B 的 C++ 在线特征过滤在 `g2o::LinearSolverDense` 析构时触发 SIGSEGV（double-free），已通过 backtrace 确认。Plan C 方案通过离线图像掩码完全绕过此问题。

2. **EuRoC ATE 评估**: 低纹理室内工业场景中 ORB-SLAM3 持续跟踪失败，无法可靠计算 ATE。系统完成执行（时序数据见 Table IV），但 EuRoC ATE 对比未包含在论文中。

3. **假动态退化**: 静态场景中的动态类实例（如坐着的人）被错误掩码，导致 over-filtering。Plan C 方案无法区分运动/静止的动态类实例。

4. **COCO 类别限制**: 检测类别限于 80 个 COCO 类别，道路标识、工业设备等域特定动态物体无法检测。

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

[GNU General Public License v3.0](LICENSE)。依赖的 ORB-SLAM3 同样使用 GPLv3。