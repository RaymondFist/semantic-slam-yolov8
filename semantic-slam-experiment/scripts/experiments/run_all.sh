#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- orchestrator for the 10 supplemental studies.
# =============================================================================
# Override by exporting these before calling the script, or run run_all.sh with no
# arguments and the standard repo layout is detected automatically.
#     bash scripts/experiments/run_all.sh
#
# Each study is a separate script in this folder; this driver only calls them in
# a sensible order and passes the environment through. No numbers are produced
# here -- every script prints its results at run time.
#
# Studies:
#   1  filter_detections.py     offline class/conf re-filter (used by all sweeps)
#   2  sweeps.sh (SWEEP=conf)      conf in {0.10,0.25,0.50,0.75}
#   3  sweeps.sh (SWEEP=class)     {person} vs {person,vehicle} vs {full}
#   4  sweeps.sh (SWEEP=model)     YOLOv8 n / s / m
#   5  mask_dataset_instance.py    instance (segmentation) masking
#   6  sweeps.sh (SWEEP=dilation)  delta in {0,3,6,9} px
#   7  motion_mask.py            detector-independent RGB-D geometry dynamic masks
#   8  run_extra_kitti.sh       KITTI 01,04,05,07,08,09,10 + ATE/RPE
#   9  run_multiseed.sh         N>=10 repeats + paired Wilcoxon
#  10  measure_offline_cost.py  offline pre-processing cost (detection + masking)
# -----------------------------------------------------------------------------

# ---- PATHS: SEMANTIC_ROOT and DATA_DIR default to the standard repo layout ----
# SEMANTIC_ROOT is auto-detected from this script's location (<repo>/scripts/experiments/),
# so no path needs editing when the repo is laid out as delivered.
export SEMANTIC_ROOT="${SEMANTIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DATA_DIR="${DATA_DIR:-$SEMANTIC_ROOT/data}"
export ORB_SLAM3_ROOT="${ORB_SLAM3_ROOT:-$HOME/ORB_SLAM3}"   # ORB-SLAM3 build root
# Optional explicit binaries (auto-discovered under ORB_SLAM3_ROOT if unset):
# export RGBD_TUM="/path/to/rgbd_tum"
# export STEREO_KITTI="/path/to/stereo_kitti"
# export VOCAB="$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt"

# For study 7, point MOTION_MASK_DIR at the per-frame mask directory written by
# motion_mask.py (one <idx:06d>.png per rgb.txt frame, non-zero = dynamic):
export MOTION_MASK_DIR="${MOTION_MASK_DIR:-}"
# For study 5, which TUM sequence to instance-mask:
export S5_SEQ="${S5_SEQ:-walking_xyz}"
# -----------------------------------------------------------------------------

set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/common.sh"
require_env
discover_bins

# Re-run-study selection (space separated); default runs all.
STUDIES="${STUDIES:-1 2 3 4 5 6 7 8 9 10}"
has() { [[ " $STUDIES " == *" $1 "* ]]; }

# Repeats per setting (default 5) -- the SLAM build is not run-to-run deterministic.
N="${N:-5}"

# run_sweep <mode>  ->  runs the combined ablation sweeps for a single mode.
run_sweep() { SWEEP="$1" bash "$DIR/sweeps.sh"; }

mkdir -p "$OUT"

if has 3; then echo "=== [study 3] class-set ablation ==="; run_sweep class; fi
if has 2; then echo "=== [study 2] confidence sweep ===";   run_sweep conf; fi
if has 6; then echo "=== [study 6] dilation sweep ===";     run_sweep dilation; fi
if has 4; then echo "=== [study 4] detector-size sweep ==="; run_sweep model; fi

if has 5; then
    echo "=== [study 5] box vs instance mask ==="
    ds="$(tum_ds_dir "$S5_SEQ")"
    instmasked="$MASKED/instance/${S5_SEQ}"
    mkdir -p "$OUT/instance_mask"
    if [ -d "$ds" ]; then
        "$PYTHON" "$DIR/mask_dataset_instance.py" --dataset "$ds" --output "$instmasked" --conf "${CONF:-0.45}"
        log "instance-mask ATE_SE3 (${S5_SEQ}): $(repeat_ate tum "$instmasked" "$(tum_gt "$S5_SEQ")" "$OUT/instance_mask" "instance_${S5_SEQ}" "$N")"
    else
        log "[SKIP] study 5: dataset not found ($ds)"
    fi
fi

if has 7; then
    echo "=== [study 7] geometric motion mask (detector-independent) ==="
    if [ -n "$MOTION_MASK_DIR" ] && [ -e "$MOTION_MASK_DIR" ]; then
        ds="$(tum_ds_dir "$S5_SEQ")"
        motion_masked="$MASKED/motion/${S5_SEQ}"
        mkdir -p "$OUT/motion"
        "$PYTHON" "$DIR/mask_dataset_external.py" --masks-dir "$MOTION_MASK_DIR" --dataset "$ds" \
            --output "$motion_masked" --dataset-type tum
        log "motion-mask ATE_SE3 (${S5_SEQ}): $(repeat_ate tum "$motion_masked" "$(tum_gt "$S5_SEQ")" "$OUT/motion" "motion_${S5_SEQ}" "$N")"
    else
        log "[SKIP] study 7: MOTION_MASK_DIR unset or missing. Generate masks with motion_mask.py first."
    fi
fi

if has 8; then echo "=== [study 8] extra KITTI sequences ==="; bash "$DIR/run_extra_kitti.sh"; fi
if has 9; then echo "=== [study 9] multi-seed / repeat runs ==="; bash "$DIR/run_multiseed.sh"; fi
if has 10; then
    echo "=== [study 10] offline pre-processing cost (detection + masking) ==="
    # Representative set for the timing table; override with TIMING_SEQS="<names>".
    "$PYTHON" "$DIR/measure_offline_cost.py" --batch --seqs ${TIMING_SEQS:-walking_xyz kitti_00}
fi
if has 1; then
    echo "=== [study 1] filter_detections.py demo (class/conf re-filter) ==="
    echo "  all sweeps already invoke it; standalone example:"
    echo "  python $DIR/filter_detections.py --input <det_dir> --output <out> --classes full --conf-min 0.45"
fi

echo ""
log "run_all.sh finished. Outputs under: $OUT"
