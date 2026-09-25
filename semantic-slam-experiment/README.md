# Semantic-SLAM-YOLOv8 Experiment

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![ORB-SLAM3](https://img.shields.io/badge/ORB--SLAM3-v1.0-orange)](https://github.com/UZ-SLAMLab/ORB_SLAM3)

**Dynamic-environment ORB-SLAM3 experiments built on YOLOv8-nano semantic masking.**  

---

## Table of contents

```
semantic-slam-experiment/
├── autodl_deploy.sh              # One-click cloud deployment for AutoDL (~2700 lines)
├── autodl_p1p2.sh                # P1 multi-run stats + P2 random-mask comparison (AutoDL)
├── autodl_prepare_data.sh        # Fill in re-runnable inputs (extract tgz + associate.txt + masks)
├── requirements.txt              # Python dependencies
├── LICENSE                       # GPL-3.0
│
├── scripts/                      # Python utility scripts
│   ├── yolov8_offline_inference.py      # Offline YOLOv8 inference
│   ├── mask_dataset.py                  # Dynamic-region mask generation
│   ├── random_mask_dataset.py           # Random-mask control datasets (E1.6)
│   ├── associate.py                     # TUM RGB-D timestamp alignment
│   ├── compute_ate.py                   # Per-sequence ATE computation
│   ├── batch_compute_ate.py             # Batch ATE computation (EuRoC + TUM walking)
│   ├── aggregate_multi_run.py           # Multi-run ATE statistics (P1)
│   ├── offline_semantic_validation.py   # Offline semantic filtering analysis
│   └── generate_planb_report.py         # Experiment report generation
│
├── src/                          # C++ SemanticSLAM module
│   ├── CMakeLists.txt            # CMake build configuration
│   ├── include/                  # Header files (5)
│   ├── src/                      # Source files (4)
│   ├── benchmark/                # Benchmarks (TUM/KITTI/EuRoC/ablation)
│   ├── config/                   # YAML configs (Baseline/Semantic/YOLO-Only/GeoConst)
│   ├── scripts/                  # Build scripts (build.sh, build.bat, export_model.sh)
│   └── tests/                    # Unit tests
│
├── patches/                      # ORB-SLAM3 source patches
│   └── patch_orbslam3.py         # Injects the SemanticSLAM module into ORB-SLAM3
│
├── models/                       # Pretrained models
│   ├── models.zip                # Bundled model files for one-file upload, extract to this folder
│   ├── yolov8n-seg.pt            # YOLOv8-nano segmentation model (~6.7MB)
│   ├── coco.names                # COCO 80 classes
│   └── ORBvoc.txt                # ORB-SLAM3 vocabulary / BoW (~138MB)
│
├── data/                         # Data directory
│   ├── datasets/                 # Raw datasets (TUM/KITTI/EuRoC)
│   ├── datasets_masked/          # YOLO semantic-masked datasets (mask_dataset.py)
│   ├── datasets_randmask/        # Random-mask control datasets (random_mask_dataset.py)
│   ├── detections/               # YOLO detection results JSON
│   ├── gt_trajectories/          # Ground-truth trajectories
│   └── real_trajectories/        # Estimated trajectories
│
└── output/                       # Experiment results
    ├── baseline_*.txt            # Baseline trajectories
    ├── mask_*.txt                # YOLO-Mask trajectories
    ├── p1/                       # P1 multi-run statistics (run archives + p1_stats.json)
    ├── p2/                       # P2 random-mask comparison (randmask_raw.txt + p2_compare.json)
    ├── failure_analysis/         # Failure analysis
    ├── timing_logs/              # Performance logs
    └── parameter_sweep/          # Parameter sensitivity
```

---

## Experiment plans

| Plan | Method | Description | Status |
|------|------|------|------|
| **Plan A** | C++ online feature filtering | Modifies the ORB-SLAM3 tracking pipeline, filters dynamic features per frame inside Track() | Failed (g2o crash) |
| **Plan B** | C++ semantic weights | Assigns low weights to semantic features before g2o optimization | Failed (g2o crash) |
| **Plan C** | Offline image masking | Masks dynamic regions with YOLOv8 in a preprocessing stage, feeds unmodified ORB-SLAM3 | **Adopted** |

### Plan C workflow

```
raw images -> YOLOv8-nano offline inference -> detection JSON
                                        v
                               mask_dataset.py -> masked images
                                        v
                          unmodified ORB-SLAM3 -> trajectories
```

### Why g2o crashed

Plans A/B crash at runtime. The backtrace points to a double-free in the destructor of `g2o::LinearSolverDense`. The root cause is that `Optimizer::PoseOptimization()` and `Tracking::Track()` in ORB-SLAM3 never properly synchronize the g2o optimizer lifecycle with the feature-pointer lifecycle. After semantic feature filtering, deleted feature pointers are still referenced internally by g2o. The full backtrace is in the Plan B section of `autodl_deploy.sh`.

---

## Quick start

### One-click deployment (AutoDL cloud)

```bash
bash autodl_deploy.sh                      # full deployment
bash autodl_deploy.sh --skip-download      # skip dataset download
bash autodl_deploy.sh --resume             # resume from checkpoints
```

The script runs 8 stages automatically:
1. **[0/8]** Install system dependencies (build-essential, cmake, Eigen3, OpenCV 4.10, Pangolin, etc.)
2. **[1/8]** Build ORB-SLAM3 (includes C++17 compatibility fixes and Sophus NaN protection)
3. **[2/8]** Build the SemanticSLAM C++ module
4. **[3/8]** Download the YOLOv8-nano model (PT + ONNX + TensorRT)
5. **[4/8]** Download datasets (TUM fr3 x5, KITTI 00, EuRoC MH x3)
6. **[5/8]** Run offline YOLOv8 inference (produces detection JSON)
7. **[6/8]** Generate the masked dataset (core step of Plan C)
8. **[7/8]** Run all experiments (Baseline, YOLO-Mask, ablation, parameter sweep)
9. **[8/8]** Run ATE evaluation and produce `experiment_results.json`

### Manual deployment

```bash
# 1. install dependencies
sudo apt-get install build-essential cmake git libeigen3-dev libglew-dev \
    libboost-dev libssl-dev libgtk2.0-dev python3-pip
pip install ultralytics opencv-python-headless evo scipy

# 2. build ORB-SLAM3 (see the [1/8] step of autodl_deploy.sh)

# 3. build SemanticSLAM
cd src && mkdir build && cd build
cmake .. -DORB_SLAM3_ROOT=/path/to/ORB_SLAM3
make -j$(nproc)

# 4. download the model
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"

# 5. run offline YOLO inference
python scripts/yolov8_offline_inference.py \
    --dataset data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
    --output data/detections/walking_xyz \
    --model yolov8n-seg.pt

# 6. generate the masked dataset
python scripts/mask_dataset.py --all \
    --datasets data/datasets \
    --detections data/detections \
    --output data/datasets_masked

# 7. run experiments
# Baseline: stock ORB-SLAM3 on raw datasets
# YOLO-Mask: stock ORB-SLAM3 on masked datasets
# see the [7/8] step of autodl_deploy.sh for details
```

---

## Script details

### `yolov8_offline_inference.py`

Runs YOLOv8-nano instance segmentation on a given dataset and dumps per-frame detection JSON.

```bash
python scripts/yolov8_offline_inference.py \
    --dataset <path/to/sequence> \
    --output <path/to/detections> \
    --model yolov8n-seg.pt \
    --conf 0.45 --nms 0.45
```

Per-frame JSON format:
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

Reads the YOLO detections, zeros out pixels of dynamic object regions, and produces a masked copy of the dataset.

**Dynamic classes** (COCO IDs): `person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7), cat(16), dog(17)`

```bash
python scripts/mask_dataset.py --all \
    --datasets data/datasets \
    --detections data/detections \
    --output data/datasets_masked
```

### `offline_semantic_validation.py`

Analyzes the YOLO detections offline, without running ORB-SLAM3, to quantify the expected effect of semantic filtering.

Outputs:
- Dynamic object coverage (percentage of image area)
- Expected feature filtering rate (estimated from bbox areas)
- Per-frame stats plus sequence-level summaries

```bash
python scripts/offline_semantic_validation.py \
    --detections data/detections \
    --output output/semantic_validation.json
```

### `generate_planb_report.py`

Builds a combined validation report with the Baseline ATE comparison table, offline semantic filtering analysis, dynamic degradation analysis, and theoretical improvement estimates.

```bash
python scripts/generate_planb_report.py
# output: output/planb_validation_report.md
```

### `associate.py`

Timestamp alignment utility for TUM RGB-D datasets. Pairs `rgb.txt` and `depth.txt` by nearest-neighbor timestamps.

```bash
python scripts/associate.py rgb.txt depth.txt > associations.txt
```

---

## C++ module

### Directory layout

```
src/
├── CMakeLists.txt              # Build config (C++17, OpenCV 4.2+, Eigen3, optional TensorRT)
├── include/                    # Header files
│   ├── YoloDetector.h          # ONNX Runtime inference, loads the YOLOv8-nano model
│   ├── DynamicFeatureFilter.h  # Optical-flow geometric verification + dynamic feature marking
│   ├── SemanticWeights.h       # Semantic class weights (building=1.0, road=0.7, vegetation=0.5)
│   ├── SemanticSLAM.h          # Main entry, coordinates YOLO + filtering + weights
│   └── ORBSLAM3Patch.h         # ORB-SLAM3 interface patch (feature-level interception)
├── src/                        # Source files
│   ├── YoloDetector.cc
│   ├── DynamicFeatureFilter.cc
│   ├── SemanticWeights.cc
│   └── SemanticSLAM.cc
├── benchmark/                  # Benchmark executables
│   ├── benchmark_main.cc       # Entry point
│   ├── tum_benchmark.cc        # TUM RGB-D benchmark
│   ├── kitti_benchmark.cc      # KITTI benchmark
│   ├── euroc_benchmark.cc      # EuRoC MAV benchmark
│   ├── ablation_runner.cc      # Ablation runner
│   └── benchmark_utils.h       # Shared utilities
├── config/                     # YAML configuration files
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
├── scripts/                    # Build helpers
│   ├── build.sh                # Linux build
│   ├── build.bat               # Windows build
│   └── export_model.sh         # PT -> ONNX -> TensorRT model export
└── tests/                      # Unit tests
    ├── CMakeLists.txt
    ├── test_yolo_detector.cc
    ├── test_dynamic_filter.cc
    ├── test_semantic_weights.cc
    ├── test_integration.cc
    └── test_utils.h
```

### Build

```bash
cd src
mkdir build && cd build
cmake .. -DORB_SLAM3_ROOT=/path/to/ORB_SLAM3
make -j$(nproc)
```

### Config files

Files under `config/` are organized by dataset and experiment mode:

| Config file | Dataset | Mode |
|----------|--------|------|
| `TUM3_baseline.yaml` | TUM fr3 | Stock ORB-SLAM3 |
| `TUM3_semantic.yaml` | TUM fr3 | Semantics + geometry |
| `TUM3_geoconst.yaml` | TUM fr3 | Geometry constraints only |
| `TUM3_yolo_only.yaml` | TUM fr3 | Semantic filtering only |
| `KITTI00_baseline.yaml` | KITTI 00 | Stock ORB-SLAM3 |
| `KITTI00_semantic.yaml` | KITTI 00 | Semantics + geometry |
| `EuRoC_baseline.yaml` | EuRoC MH | Stock ORB-SLAM3 |
| `EuRoC_semantic.yaml` | EuRoC MH | Semantics + geometry |

### Unit tests

```bash
cd src/build
ctest --output-on-failure
```

Coverage:
- `test_yolo_detector` : model loading, inference, edge cases
- `test_dynamic_filter` : dynamic feature filtering, optical-flow verification
- `test_semantic_weights` : weight computation, consistency checks
- `test_integration` : end-to-end integration

---

## Datasets

| Dataset | Sequence | Frames | Size | Notes |
|--------|------|------|------|------|
| TUM fr3 | sitting_static | 711 | ~423MB | Two people sitting and talking (static baseline) |
| TUM fr3 | sitting_xyz | 859 | ~739MB | Next to a coffee machine, small motion |
| TUM fr3 | walking_static | 797 | ~452MB | Static camera, two people walking |
| TUM fr3 | walking_xyz | 827 | ~503MB | Camera and people both move |
| TUM fr3 | walking_halfsphere | 1068 | ~610MB | Large rotation + pedestrians |
| KITTI | 00 | 4541 | n/a | Loop closure + dynamic vehicles/pedestrians |
| EuRoC | MH_01_easy | 3682 | ~150MB | Bright machine hall |
| EuRoC | MH_03_medium | 2700 | ~105MB | Medium difficulty |
| EuRoC | MH_05_difficult | 2273 | ~79MB | Low texture + motion blur |

### Dataset download links

| Dataset | Official download | Notes |
|--------|-------------|------|
| **TUM RGB-D** | [https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) | Download the `freiburg3_sitting_*` and `freiburg3_walking_*` sequences (tgz format) |
| **KITTI Odometry** | [https://www.cvlibs.net/datasets/kitti/eval_odometry.php](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | Download odometry `grayscale` + `calibration` + `ground truth poses` (registration required) |
| **EuRoC MAV** | [https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) | Download `Machine Hall 01/03/05` (ASL format zip) |

> `autodl_deploy.sh` downloads the datasets into `data/datasets/` automatically. If you download them manually, extract each dataset into the matching subdirectory: `data/datasets/TUM/`, `data/datasets/KITTI/`, `data/datasets/EuRoC/`.

---

## Experiment results

### Key results

| Sequence | Baseline ATE (m) | YOLO-Mask ATE (m) | Improvement |
|------|-----------------|-------------------|------|
| TUM walking_xyz | 0.2856 | 0.0168 | **+94.1%** |
| TUM walking_halfsphere | 0.2034 | 0.0271 | **+86.7%** |
| TUM walking_static | 0.0252 | 0.0117 | **+53.6%** |
| TUM sitting_static | 0.0072 | 0.0102 | -41.4% |
| TUM sitting_xyz | 0.0092 | 0.0311 | -239.5% |
| KITTI 00 | 0.9671 | 1.1130 | -15.1% |

### Output files

```
output/
├── baseline_tum_xyz.txt                  # Baseline trajectory
├── mask_tum_walking_xyz.txt              # YOLO-Mask trajectory
├── baseline_tum_sitting_static.txt       # Baseline trajectory (static scene)
├── baseline_kitti00.txt                  # KITTI Baseline
├── mask_kitti00.txt                      # KITTI YOLO-Mask
│
├── p1/                                   # P1 multi-run statistics
│   ├── p1_stats.json                     # Multi-run ATE mean/std/improvement
│   ├── raw_{seq}_{cfg}.txt               # Per-run RMSE values
│   └── run_{seq}_{cfg}_{k}.txt           # Per-run trajectory archive
│
├── p2/                                   # P2 random-mask comparison
│   ├── p2_compare.json                   # Random-mask ATE (seed=42)
│   └── randmask_{seq}_run1.txt           # Random-mask trajectories
│
├── failure_analysis/
│   ├── failure_summary.txt               # Failure summary (human readable)
│   ├── failure_summary.json              # Failure summary (machine readable)
│   └── *_failure.log                     # Per-sequence failed-frame logs (runtime only)
│
├── timing_logs/
│   ├── timing_summary.json               # Parsed timing values
│   └── *_timing.log                      # Per-sequence wall-clock logs (runtime only)
│
└── parameter_sweep/
    ├── TUM3_sweep.yaml                   # Sweep configuration
    ├── sweep_orb.txt                     # ORB feature count sweep results
    ├── sweep_semantic.txt                # Semantic weight sweep results
    ├── sweep_dynamic.txt                 # Dynamic threshold sweep results
    ├── tum_orb_500_traj.txt -> tum_orb_2000_traj.txt   # ORB feature count trajectories
    ├── tum_semw_0.0_traj.txt -> tum_semw_1.0_traj.txt  # Semantic weight trajectories
    └── tum_dyn_0.3_traj.txt -> tum_dyn_0.8_traj.txt    # Dynamic threshold trajectories
```

> The `*.log` files are produced at run time and are excluded from version control (see `.gitignore`). They are listed here so the directory layout matches what a fresh run produces.

---

## Performance

| Sequence | Wall-clock time | Peak memory | Model |
|------|---------|---------|------|
| TUM walking_xyz | 37.4 s | <1.2 GB | YOLOv8-nano (3.2M params) |
| KITTI 00 | 65.7 s | <1.2 GB | Same as above |
| EuRoC MH_03 | 85.5 s | <1.2 GB | Same as above |

Offline YOLOv8-nano inference runs at roughly 100 FPS on an RTX 3080. Masking is fully offline, so SLAM has zero extra runtime cost.

---

## Known issues

1. **g2o crash**: the C++ online feature filtering in Plans A/B triggers SIGSEGV (double-free) when `g2o::LinearSolverDense` is destroyed, confirmed by backtrace. Plan C avoids this entirely by masking images offline.

2. **EuRoC ATE evaluation**: ORB-SLAM3 repeatedly loses tracking in the low-texture indoor industrial scenes, so ATE cannot be computed reliably. The pipeline finishes running (timing data is recorded in `table4_timing_profiling.csv`), but the EuRoC ATE comparison is not reported.

3. **False-dynamic degradation**: instances of dynamic classes in static scenes (e.g. sitting people) get masked by mistake, which causes over-filtering. Plan C cannot tell a moving instance of a dynamic class from a static one.

4. **COCO class limit**: detection is restricted to 80 COCO classes, so domain-specific dynamic objects such as road signs and industrial equipment are never detected.

---

## License

[GNU General Public License v3.0](LICENSE). The bundled ORB-SLAM3 is also under GPLv3.