# Supplemental experiments (scripts/experiments/)

Auxiliary ablation scripts for the semantic-slam-yolov8 study. Numbers are
printed at run time; paths come from the environment (`SEMANTIC_ROOT`,
`DATA_DIR`).

## Scripts

| Script | Purpose |
|--------|---------|
| `run_all.sh` | Orchestrator for the 10 studies; selects studies via `STUDIES`. |
| `sweeps.sh` | Combined ablation sweeps for studies 2/3/4/6; selects via `SWEEP` / `--mode`. |
| `mask_dataset_instance.py` | Study 5: seg inference (if needed) + instance/segmentation masking. |
| `filter_detections.py` | Offline class/conf re-filter of detection JSONs (used by all sweeps). |
| `mask_coverage.py` | Masked-area fraction implied by a detection dir (the `mask%` column). |
| `motion_mask.py` | Study 7: detector-independent dynamic masks from RGB-D geometry. |
| `mask_dataset_external.py` | Applies externally supplied per-frame masks (e.g. from `motion_mask.py`) to build a masked dataset. |
| `traj_eval.py` | ATE (SE(3)) + KITTI/TUM RPE for one trajectory pair. |
| `aggregate_runs.py` | Multi-run ATE aggregation + paired Wilcoxon signed-rank test. |
| `run_multiseed.sh` | N>=10 repeated runs and paired test. |
| `run_extra_kitti.sh` | Extra KITTI Odometry sequences + ATE/RPE (N repeats, default 5). |
| `prepare_extra_kitti.sh` | Fetch + lay out the extra KITTI sequences (skip-if-present) for `run_extra_kitti.sh`; `GRAY_MODE=stream` pipes the archive into `bsdtar` so the ~22 GB zip is never written to disk. |
| `measure_offline_cost.py` | Offline pre-processing cost (detection + masking) per sequence. |
| `common.sh` | Shared bash helpers (`require_env`, `discover_bins`, `yolo_infer`, `filter_dets`, `coverage`, `run_tum`, `run_kitti`, `ate_se3`, `repeat_ate`, ...). |

## Repeats

Sweep settings and the instance/motion studies are each run `N` times (default 5)
and reported as `mean +/- std (n=k)`. This is needed because this ORB-SLAM3 build
is not run-to-run deterministic: identical masked input can differ by >10% ATE,
so a single run cannot separate a real effect from run noise. Override with
`N=<k>`. The class-set sweep (study 3) runs on KITTI 00 by default, since TUM
walking sequences contain only `person` and would give an empty contrast; set
`CLASS_KIND=tum` to run it on `TUM_SEQS` instead.

The `random_mask` control is produced outside these studies by `autodl_p1p2.sh`
(P2 step, `scripts/random_mask_dataset.py`, seed=42).

## Class sets

`filter_detections.py` and the class-set sweep select detections by COCO class id.
The three sets used here are:

| Name | COCO class ids | Classes |
|------|----------------|---------|
| `person` | 0 | person |
| `person,vehicle` | 0, 1, 2, 3, 5, 7 | person, bicycle, car, motorcycle, bus, truck |
| `full` | 0, 1, 2, 3, 5, 7, 16, 17 | the pipeline dynamic class set |

## Study index

Study indices follow the order used by `run_all.sh`.

| Script | Study |
|--------|-------|
| `filter_detections.py` | study 1 (class/conf re-filter, used by all sweeps) |
| `sweeps.sh` (`SWEEP=conf`) | study 2 (confidence sweep) |
| `sweeps.sh` (`SWEEP=class`) | study 3 (class-set ablation) |
| `sweeps.sh` (`SWEEP=model`) | study 4 (detector-size ablation) |
| `mask_dataset_instance.py` | study 5 (box vs instance mask) |
| `sweeps.sh` (`SWEEP=dilation`) | study 6 (bbox-dilation sweep) |
| `motion_mask.py` (uses `mask_dataset_external.py`) | study 7 (geometric motion mask) |
| `run_extra_kitti.sh` (uses `traj_eval.py`) | study 8 (extra KITTI sequences) |
| `run_multiseed.sh` (uses `aggregate_runs.py`) | study 9 (multi-seed / repeats) |
| `measure_offline_cost.py` | study 10 (offline pre-processing cost) |

## Entry point

```bash
export SEMANTIC_ROOT=/path/to/semantic-slam-experiment
export DATA_DIR=/path/to/data

bash scripts/experiments/run_all.sh                 # all 10 studies
```

## Examples

```bash
# Run only studies 2 and 3:
STUDIES="2 3" bash scripts/experiments/run_all.sh

# Run one sweep directly (env override):
SWEEP="conf" TUM_SEQS="walking_xyz" bash scripts/experiments/sweeps.sh
SWEEP="dilation" DELTAS="0 3 6 9" bash scripts/experiments/sweeps.sh
bash scripts/experiments/sweeps.sh --mode all
bash scripts/experiments/sweeps.sh --mode model

# Offline pre-processing cost only (detection + masking), representative set:
STUDIES="10" bash scripts/experiments/run_all.sh

# ...or a custom sequence set / a single sequence:
TIMING_SEQS="walking_xyz walking_static sitting_xyz kitti_00" STUDIES="10" bash scripts/experiments/run_all.sh
python3 scripts/experiments/measure_offline_cost.py --batch --seqs walking_xyz kitti_00
```
