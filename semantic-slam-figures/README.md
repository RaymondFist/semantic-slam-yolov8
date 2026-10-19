# Semantic-SLAM-YOLOv8 Figures & Tables

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange)](https://matplotlib.org/)

**Figure and table generation**: produces the 5 figures and 5 CSV tables from the experiment data.

---

## Table of contents

```
semantic-slam-figures/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   └── experiment_results.json   # Experiment data (exported by the ATE evaluation in autodl_deploy.sh)
├── src/
│   ├── generate_figures.py       # Figure generation (5 PNGs)
│   ├── generate_tables.py        # Table generation (5 CSVs)
└── output/
    ├── figures/                  # Generated figures (600 DPI PNG)
    │   ├── fig1_system_architecture.png
    │   ├── fig2_tum_ate.png
    │   ├── fig3_scene_type.png
    │   ├── fig4_kitti_ate.png
    │   └── fig5_timing.png
    └── tables/                   # Generated tables (UTF-8 CSV)
        ├── table1_tum_ate_results.csv
        ├── table2_kitti_ate_results.csv
        ├── table4_timing_profiling.csv
        ├── table5_failure_analysis.csv
        └── table7_literature_comparison.csv
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

Writes 5 PNG figures to `output/figures/` at 600 DPI.

### Generate tables

```bash
cd src
python generate_tables.py
```

Writes 5 CSV tables to `output/tables/` with a UTF-8 BOM.

---

## Data source

All data comes from `data/experiment_results.json`, which `autodl_deploy.sh` (in the parent `semantic-slam-experiment/` directory) exports after the runs on the AutoDL cloud. The JSON structure:

| Field | Description |
|------|------|
| `tum` | Baseline / YOLO-Mask SE(3) ATE (five-run mean ± std) for the 5 TUM RGB-D sequences |
| `kitti` | Baseline / YOLO-Mask SE(3) ATE for the KITTI 00 sequence |
| `timing` | Offline pre-processing cost (detection + masking time, per-frame cost, FPS, peak memory) for 2 representative sequences |
| `failure_analysis` | Tracking-loss counts and failure types for 3 challenging sequences |
| `parameter_sensitivity` | ATE data for the ORB feature-count sweep |

---

## Artifacts

### Figures (`generate_figures.py`)

| Figure | File | Content |
|------|--------|-------------|
| Figure 1 | `fig1_system_architecture.png` | System architecture flow (input -> YOLOv8 -> mask -> ORB-SLAM3 -> evaluation) |
| Figure 2 | `fig2_tum_ate.png` | TUM RGB-D Baseline vs YOLO-Mask ATE, two panels: absolute ATE + relative improvement |
| Figure 3 | `fig3_scene_type.png` | Mean ATE for static vs dynamic scenes |
| Figure 4 | `fig4_kitti_ate.png` | KITTI 00 ATE comparison |
| Figure 5 | `fig5_timing.png` | Offline pre-processing cost (stacked detection + masking time per sequence, with peak memory overlaid) |

### Tables (`generate_tables.py`)

| Table | File | Content |
|------|--------|-------------|
| Table I | `table1_tum_ate_results.csv` | TUM RGB-D ATE results (5 sequences × 2 methods + scene types) |
| Table II | `table2_kitti_ate_results.csv` | KITTI 00/01/04/07 ATE results |
| Table IV | `table4_timing_profiling.csv` | Offline pre-processing cost (detection, masking, total, ms/frame, FPS, peak memory) |
| Table V | `table5_failure_analysis.csv` | Failure case analysis (failure types, tracking-loss counts) |
| Table VII | `table7_literature_comparison.csv` | Literature comparison on TUM walking sequences |

---

## Experiment data summary

### TUM RGB-D ATE results

Five-run SE(3) (scale-fixed) mean ± sample standard deviation (ddof = 1).

| Sequence | Baseline ATE (m) | YOLO-Mask ATE (m) | Improvement |
|------|-----------------|-------------------|------|
| sitting_static | 0.0101 ± 0.0007 | 0.0116 ± 0.0014 | -14.9% |
| sitting_xyz | 0.0092 ± 0.0002 | 0.0737 ± 0.0586 | -701.1% |
| walking_static | 0.3494 ± 0.0363 | 0.0184 ± 0.0057 | **+94.7%** |
| walking_xyz | 0.6499 ± 0.0655 | 0.0208 ± 0.0021 | **+96.8%** |
| walking_halfsphere | 0.3580 ± 0.0702 | 0.0533 ± 0.0083 | **+85.1%** |

### Offline pre-processing cost

Detection plus masking, measured before SLAM. Both stages run once, ahead of the
SLAM run; the SLAM runtime itself is identical across configurations and is not
included here.

| Sequence | Frames | Detection + masking (s) | ms/frame | FPS | Peak memory |
|------|------|---------|---------|------|------|
| TUM walking_xyz | 859 | 37.38 | 43.5 | 23.0 | 1.033 GB |
| KITTI 00 | 4,541 | 167.24 | 36.8 | 27.2 | 1.039 GB |

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

Both scripts carry the values recorded from the experiment runs; they do not read `experiment_results.json` at runtime. To change the data, update the value tables at the top of each script and re-run them.

---

## File details

### `generate_figures.py`

- **Dependencies**: matplotlib, numpy
- **Output format**: PNG (600 DPI)
- **Output directory**: `output/figures/`
- **Notes**: uses the `Agg` backend, no GUI needed, suited to headless servers

### `generate_tables.py`

- **Dependencies**: Python standard library only (csv, os)
- **Output format**: CSV (UTF-8 BOM, opens directly in Excel)
- **Output directory**: `output/tables/`
- **Notes**: zero external dependencies, runs on any Python 3 environment

---

## License

[GNU General Public License v3.0](../semantic-slam-experiment/LICENSE). Same as the parent project `semantic-slam-experiment`.