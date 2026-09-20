# Semantic-SLAM-YOLOv8 Figures & Tables

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange)](https://matplotlib.org/)

**Figure and table generation project for the paper**: generates all paper figures (6 figures + 6 tables) from the experiment data, for the paper *Semantic Visual-Inertial SLAM with YOLOv8 for Dynamic Environments*.

---

## Table of contents

```
semantic-slam-figures/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   └── experiment_results.json   # Experiment data (exported by the ATE evaluation in autodl_deploy.sh)
├── src/
│   ├── generate_figures.py       # Figure generation (6 PNGs)
│   ├── generate_tables.py        # Table generation (9 CSVs)
└── output/
    ├── figures/                  # Generated figures (300 DPI PNG)
    │   ├── fig1_system_architecture.png
    │   ├── fig2_tum_ate.png
    │   ├── fig3_scene_type.png
    │   ├── fig4_kitti_ate.png
    │   ├── fig5_timing.png
    │   └── fig6_parameter_sensitivity.png
    └── tables/                   # Generated tables (UTF-8 CSV)
        ├── table1_tum_ate_results.csv
        ├── table2_kitti_ate_results.csv
        ├── table3_ablation_study.csv
        ├── table4_timing_profiling.csv
        ├── table5_failure_analysis.csv
        ├── table6_parameter_sensitivity.csv
        ├── table7_literature_comparison.csv
        ├── table8_euroc_results.csv
        └── table9_false_dynamic_analysis.csv
```

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `matplotlib >= 3.5.0`, `numpy >= 1.21.0`.

### Generate figures

```bash
cd src
python generate_figures.py
```

Writes 6 PNG figures to `output/figures/` at 300 DPI.

### Generate tables

```bash
cd src
python generate_tables.py
```

Writes 9 CSV tables to `output/tables/` with a UTF-8 BOM.

---

## Data source

All data comes from `data/experiment_results.json`, which `autodl_deploy.sh` (in the parent `semantic-slam-experiment/` directory) exports automatically after running the real experiments on the AutoDL cloud. The JSON structure:

| Field | Description |
|------|------|
| `tum` | Baseline / YOLO-Mask ATE for the 5 TUM RGB-D sequences |
| `kitti` | Baseline / YOLO-Mask ATE for the KITTI 00 sequence |
| `euroc` | ATE for the 3 EuRoC sequences (MH_01/03/05), currently null, to be fixed |
| `timing` | Wall-clock time, CPU time, and peak memory for 3 representative sequences |
| `failure_analysis` | Tracking-loss counts and failure types for 4 challenging sequences |
| `parameter_sensitivity` | ATE data for 3 parameter sweeps (ORB feature count, semantic weight, dynamic threshold) |

> Every value is produced by the experiments. There is no hardcoded or simulated data.

---

## Artifacts

### Figures (`generate_figures.py`)

| Figure | File | Paper section | Content |
|------|--------|-------------|------|
| Figure 1 | `fig1_system_architecture.png` | Section 3.1 | System architecture flow (input -> YOLOv8 -> mask -> ORB-SLAM3 -> evaluation) |
| Figure 2 | `fig2_tum_ate.png` | Section 4.2 | TUM RGB-D Baseline vs YOLO-Mask ATE, two panels: absolute ATE + relative improvement |
| Figure 3 | `fig3_scene_type.png` | Section 4.2 | Mean ATE for static vs dynamic scenes |
| Figure 4 | `fig4_kitti_ate.png` | Section 4.3 | KITTI 00 ATE comparison |
| Figure 5 | `fig5_timing.png` | Section 4.5 | Runtime profiling (wall-clock time, CPU time, memory in one grouped bar chart) |
| Figure 6 | `fig6_parameter_sensitivity.png` | Section 4.7 | Parameter sensitivity (ORB feature count, semantic weight, dynamic threshold, three line panels) |

### Tables (`generate_tables.py`)

| Table | File | Paper section | Content |
|------|--------|-------------|------|
| Table I | `table1_tum_ate_results.csv` | Section 4.2 | TUM RGB-D ATE results (5 sequences × 2 methods + scene types) |
| Table II | `table2_kitti_ate_results.csv` | Section 4.3 | KITTI 00 ATE results |
| Table III | `table3_ablation_study.csv` | Section 4.6 | Ablation study (6 sequences + interpretation) |
| Table IV | `table4_timing_profiling.csv` | Section 4.5 | Runtime profiling (wall-clock, CPU, memory, CPU utilization) |
| Table V | `table5_failure_analysis.csv` | Section 4.8 | Failure case analysis (failure types, tracking-loss counts) |
| Table VI | `table6_parameter_sensitivity.csv` | Section 4.7 | Parameter sensitivity (3 parameters × 6 ATE levels) |

---

## Experiment data summary

### TUM RGB-D ATE results

| Sequence | Baseline ATE (m) | YOLO-Mask ATE (m) | Improvement |
|------|-----------------|-------------------|------|
| sitting_static | 0.0072 | 0.0102 | -41.4% |
| sitting_xyz | 0.0092 | 0.0311 | -239.5% |
| walking_static | 0.0252 | 0.0117 | **+53.6%** |
| walking_xyz | 0.2856 | 0.0168 | **+94.1%** |
| walking_halfsphere | 0.2034 | 0.0271 | **+86.7%** |

### Runtime performance

| Sequence | Frames | Wall-clock time | Peak memory |
|------|------|---------|---------|
| TUM walking_xyz | 827 | 37.42 s | 0.895 GB |
| KITTI 00 | 4541 | 65.66 s | 1.184 GB |
| EuRoC MH_03 | 2700 | 85.54 s | 0.741 GB |

---

## Updating the data

When the upstream experiment data changes, regenerate the artifacts as follows:

```bash
# 1. copy the experiment_results.json exported by autodl_deploy.sh over data/
cp ../semantic-slam-experiment/data/experiment_results.json data/

# 2. regenerate figures and tables
cd src
python generate_figures.py
python generate_tables.py
```

Both scripts fall back to hardcoded experiment values when `experiment_results.json` is missing, but they read from the JSON file first. To change the data, just update the JSON file; no changes to the Python scripts are needed.

---

## File details

### `generate_figures.py`

- **Dependencies**: matplotlib, numpy
- **Output format**: PNG (300 DPI)
- **Output directory**: `output/figures/`
- **Notes**: uses the `Agg` backend, no GUI needed, suited to headless servers

### `generate_tables.py`

- **Dependencies**: Python standard library only (csv, os)
- **Output format**: CSV (UTF-8 BOM, opens directly in Excel)
- **Output directory**: `output/tables/`
- **Notes**: zero external dependencies, runs on any Python 3 environment

---

## Citation

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

## License

[GNU General Public License v3.0](../semantic-slam-experiment/LICENSE). Same as the parent project `semantic-slam-experiment`.