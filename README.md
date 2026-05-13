# Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-KITTI%20%7C%20EuRoC%20%7C%20TUM%20%7C%20TartanAir-orange.svg)]()

**Reproducible figure generation toolkit** for the paper *"Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments"*.

This repository contains all scripts, data, and generated figures used in the paper. Every figure is fully reproducible from source data with a single command.

---

## Paper Abstract

We present a semantic visual-inertial SLAM system that integrates **YOLOv8-nano** instance segmentation with **ORB-SLAM3** to achieve robust localization in dynamic environments. A joint semantic-geometric filtering strategy removes features on moving objects while preserving static scene structure. The system achieves **real-time performance at 25 FPS** on consumer-grade GPU hardware and demonstrates **ATE improvements of 5.9%–19.4%** across four benchmark datasets (KITTI, EuRoC, TUM RGB-D, TartanAir).

---

## Repository Structure

```
semantic-slam-yolov8/
├── src/                          # Figure generation scripts
│   ├── common.py                 # Shared utilities, styling, data I/O
│   ├── generate_all.py           # Master script to generate all figures
│   ├── fig01_system_architecture.py
│   ├── fig02_kitti_trajectory.py
│   ├── fig03_euroc_trajectory.py
│   ├── fig04_yolov8_detection.py
│   ├── fig05_ablation_study.py
│   ├── fig06_timing_analysis.py
│   ├── fig07_sota_comparison.py
│   ├── fig08_qualitative_analysis.py
│   ├── fig09_failure_analysis.py
│   └── fig10_parameter_sensitivity.py
├── data/                         # Benchmark data & ground truth
│   ├── benchmark_data.json       # Published benchmark results with DOIs
│   ├── fig02_kitti_trajectory.json
│   ├── fig03_euroc_trajectory.json
│   ├── fig04_yolov8_detection.json
│   ├── fig05_ablation_study.json
│   ├── fig06_timing_analysis.json
│   ├── fig07_sota_comparison.json
│   ├── fig08_qualitative_analysis.json
│   ├── fig09_failure_analysis.json
│   ├── fig10_parameter_sensitivity.json
│   └── real_trajectories/        # Ground truth trajectories
│       ├── KITTI/                # KITTI Odometry (sequences 00–10)
│       ├── EuRoC/                # EuRoC MAV (MH_01, MH_05)
│       └── TUM/                  # TUM RGB-D (fr3 sequences)
└── output/
    └── figures/                  # Generated figures (PNG, 300 DPI)
        ├── fig01_system_architecture.png
        ├── fig02_kitti_trajectory.png
        ├── fig03_euroc_trajectory.png
        ├── fig04_yolov8_detection.png
        ├── fig05_ablation_study.png
        ├── fig06_timing_analysis.png
        ├── fig07_sota_comparison.png
        ├── fig08_qualitative_analysis.png
        ├── fig09_failure_analysis.png
        └── fig10_parameter_sensitivity.png
```

---

## Figures Overview

| Fig | Title | Description |
|-----|-------|-------------|
| 1 | System Architecture | Overall pipeline: sensor input → ORB extraction → YOLOv8 segmentation → semantic-geometric filtering → pose estimation |
| 2 | KITTI Trajectory | Trajectory comparison on KITTI Odometry sequences (00–10) |
| 3 | EuRoC Trajectory | Trajectory comparison on EuRoC MAV sequences (MH_01, MH_05) |
| 4 | YOLOv8 Detection | Instance segmentation results on dynamic objects |
| 5 | Ablation Study | Component-wise analysis: semantic-only vs. geometric-only vs. joint filtering |
| 6 | Timing Analysis | Runtime breakdown per module (feature extraction, YOLOv8 inference, filtering, optimization) |
| 7 | SOTA Comparison | ATE and FPS comparison against ORB-SLAM3, VINS-Fusion, DynaSLAM II, SG-SLAM, RDS-SLAM, Dynamic-VINS, VIS-SLAM |
| 8 | Qualitative Analysis | Visual comparison of estimated vs. ground truth trajectories |
| 9 | Failure Analysis | Failure cases: extreme motion blur, severe occlusion, low-texture scenes |
| 10 | Parameter Sensitivity | Sensitivity analysis of key hyperparameters (confidence threshold, IoU threshold, feature count) |

---

## Quick Start

### Prerequisites

- **Python** 3.8 or higher
- **pip** (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/RaymondFist/semantic-slam-yolov8.git
cd semantic-slam-yolov8

# Install dependencies
pip install matplotlib numpy
```

### Generate All Figures

```bash
cd src
python generate_all.py
```

All 10 figures will be generated and saved to `output/figures/`.

### Generate Specific Figures

```bash
# Generate only figure 1
python generate_all.py --fig 1

# Generate figures 1, 3, and 5
python generate_all.py --fig 1,3,5

# Generate figures 7–10
python generate_all.py --fig 7,8,9,10
```

---

## Data Sources & Reproducibility

All benchmark comparison data is sourced from **published papers with verified DOIs**. The baseline results are not re-implemented; they are directly cited from the original publications:

| Method | Paper | DOI |
|--------|-------|-----|
| ORB-SLAM3 | Campos et al., IEEE TRO, 2021 | [10.1109/TRO.2021.3075644](https://doi.org/10.1109/TRO.2021.3075644) |
| DynaSLAM | Bescos et al., IEEE RA-L, 2018 | [10.1109/LRA.2018.2860039](https://doi.org/10.1109/LRA.2018.2860039) |
| DynaSLAM II | Bescos et al., IEEE RA-L, 2021 | [10.1109/LRA.2021.3068640](https://doi.org/10.1109/LRA.2021.3068640) |
| SG-SLAM | Cheng et al., IEEE TIM, 2022 | [10.1109/TIM.2022.3228006](https://doi.org/10.1109/TIM.2022.3228006) |
| RDS-SLAM | Liu et al., IEEE Access, 2021 | [10.1109/ACCESS.2021.3050617](https://doi.org/10.1109/ACCESS.2021.3050617) |
| Dynamic-VINS | Song et al., IEEE RA-L, 2023 | [10.1109/LRA.2023.3243805](https://doi.org/10.1109/LRA.2023.3243805) |
| VIS-SLAM | Zhong et al., IEEE TIE, 2023 | [10.1109/TIE.2023.3239921](https://doi.org/10.1109/TIE.2023.3239921) |

Ground truth trajectories are from the official benchmark websites:
- **KITTI**: [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/)
- **EuRoC**: [projects.asl.ethz.ch/datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- **TUM RGB-D**: [cvg.cit.tum.de/data/datasets/rgbd-dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)

---

## Key Results

| Dataset | ORB-SLAM3 | Ours | Improvement |
|---------|-----------|------|-------------|
| KITTI (ATE, m) | 4.17 | **3.59** | **13.9%** |
| EuRoC (ATE, m) | 0.036 | **0.033** | **8.3%** |
| TUM RGB-D (ATE, m) | 0.034 | **0.032** | **5.9%** |
| TartanAir (ATE, m) | — | — | **19.4%** |
| **FPS** | 30 | **25** | Real-time |

---

## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@article{chen2026semantic,
  title   = {Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments},
  author  = {Chen, Hantao and Wang, Yujuan and Ye, Jianying and Gao, Ying and
             Chen, Zhenhe and Xie, Xiaolan and Tang, Jun},
  journal = {[Journal Name]},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Authors

| Author | Affiliation | Role |
|--------|-------------|------|
| **Hantao Chen** | [Institution] | Conceptualization, Methodology, Software, Writing |
| **Yujuan Wang** | [Institution] | Validation, Formal Analysis |
| **Jianying Ye** | [Institution] | Supervision, Project Administration |
| **Ying Gao** | [Institution] | Data Curation, Visualization |
| **Zhenhe Chen** | [Institution] | Software, Investigation |
| **Xiaolan Xie** | [Institution] | Resources, Investigation |
| **Jun Tang** ★ | [Institution] | Corresponding Author, Supervision |

★ Corresponding Author: [email@institution.edu]

---

## Acknowledgments

This work was supported by [Funding Agency / Grant Numbers].

We thank the authors of ORB-SLAM3, YOLOv8, and the benchmark dataset creators for making their work publicly available.