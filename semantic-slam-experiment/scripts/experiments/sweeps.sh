#!/usr/bin/env bash
# =============================================================================
# sweeps.sh -- combined ablation sweeps (studies 2, 3, 4, 6).
# =============================================================================
# Select one or more sweeps with the SWEEP env var (space separated subset of
# `conf class model dilation`), default "conf class model dilation". A comma
# separated value and `--mode <name>` / `--mode all` are also accepted.
#
#   * conf      Study 2: detection-confidence threshold sweep.
#         Re-runs YOLOv8-seg inference at each `--conf`, RE-FILTERS the
#         detections to the full dynamic class set (applying the full dynamic
#         class set offline), masks, runs stock ORB-SLAM3 and reports a
#         (conf, mask %, ATE SE3) table.
#         Env: TUM_SEQS, CONF_LIST (default "0.10 0.25 0.50 0.75").
#   * class     Study 3: dynamic class-set ablation.
#         Reuses the EXISTING detection JSONs (produced at conf=0.45; no GPU
#         needed) and re-filters them offline to three dynamic class sets:
#             person         = {0}
#             person,vehicle = {0,1,2,3,5,7}
#             full           = {0,1,2,3,5,7,16,17}      (the pipeline dynamic class set)
#         Runs on KITTI Odometry by default (CLASS_KIND=kitti, CLASS_KITTI_SEQS=00),
#         where several dynamic classes coexist, so the factor actually contrasts.
#         CLASS_KIND=tum runs it on TUM_SEQS instead (TUM walking sequences hold
#         only 'person', so person vs person,vehicle is an empty contrast there).
#         Env: CLASS_KIND (kitti|tum), CLASS_KITTI_SEQS, TUM_SEQS,
#              CLASSES (default "person person,vehicle full").
#   * model     Study 4: detector-size ablation.
#         Re-runs inference with three model sizes (n / s / m), re-filters to
#         the full dynamic class set, masks, runs stock ORB-SLAM3 and reports
#         (model, mask %, ATE SE3). The .pt weights must be on disk (ultralytics
#         downloads them on first use if the machine has internet).
#         Env: TUM_SEQS, MODELS (default "yolov8n-seg.pt yolov8s-seg.pt
#         yolov8m-seg.pt"), CONF (default 0.45).
#   * dilation  Study 6: bounding-box expansion (delta) sweep.
#         For delta in {0,3,6,9} px, masks the existing detections (re-filtered
#         to the full dynamic class set) with a delta-pixel bbox expansion via
#         `scripts/mask_dataset.py --dilate-px`, runs stock ORB-SLAM3 and
#         reports (delta, mask %, ATE SE3).
#         Env: TUM_SEQS, DELTAS (default "0 3 6 9").
#
# Values are produced at run time. Each
# setting is run N times and reported as mean +/- std, because this ORB-SLAM3
# build is not run-to-run deterministic (identical input can differ by >10% ATE).
#
# Usage:
#   SEMANTIC_ROOT=<repo> DATA_DIR=<data> bash sweeps.sh
#   SWEEP="conf" bash sweeps.sh
#   SWEEP="class dilation" TUM_SEQS="walking_xyz" bash sweeps.sh
#   bash sweeps.sh --mode all
#   bash sweeps.sh --mode model
#
# Assumes detections already exist at $DATA_DIR/detections/<seq> for the class
# and dilation modes.
# -----------------------------------------------------------------------------
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_env
discover_bins

# Repeats per ablation setting (default 5); see the header note.
N="${N:-5}"

# ---- mode selection ---------------------------------------------------------
SWEEP="${SWEEP:-conf class model dilation}"
SWEEP="${SWEEP//,/ }"
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)
            [ $# -ge 2 ] || die "--mode needs a value (conf|class|model|dilation|all)"
            if [ "$2" = "all" ]; then SWEEP="conf class model dilation"; else SWEEP="${2//,/ }"; fi
            shift 2 ;;
        --mode=*)
            v="${1#*=}"
            if [ "$v" = "all" ]; then SWEEP="conf class model dilation"; else SWEEP="${v//,/ }"; fi
            shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
has() { [[ " $SWEEP " == *" $1 "* ]]; }

# ---- study 2: confidence sweep ---------------------------------------------
run_conf() {
    local TUM_SEQS="${TUM_SEQS:-walking_xyz walking_halfsphere}"
    local CONF_LIST="${CONF_LIST:-0.10 0.25 0.50 0.75}"
    local OUTDIR="$OUT/sweep_confidence"
    mkdir -p "$OUTDIR"

    printf "%-22s %8s %10s %20s\n" "sequence" "conf" "mask%" "ATE_SE3"
    printf -- "--------------------------------------------------------------------\n"

    local seq ds gt conf tag inf_dir fdir masked ate cov covpct
    for seq in $TUM_SEQS; do
        ds="$(tum_ds_dir "$seq")"
        gt="$(tum_gt "$seq")"
        if [ ! -d "$ds" ]; then log "[SKIP] ${seq}: dataset not found ($ds)"; continue; fi
        [ -f "$gt" ] || log "[WARN] ${seq}: groundtruth missing ($gt)"

        for conf in $CONF_LIST; do
            tag="${seq}_conf${conf}"
            inf_dir="$DET/${tag}"
            fdir="$DET/${tag}_full"
            masked="$MASKED/cond_sweep/${seq}_conf${conf}"

            yolo_infer "$ds" "$inf_dir" "$conf"
            filter_dets "$inf_dir" "$fdir" "full"
            "$PYTHON" "$SEMANTIC_ROOT/scripts/mask_dataset.py" \
                --detections "$fdir" --dataset "$ds" --output "$masked" --dataset-type tum \
                >/dev/null 2>&1 || log "[WARN] masking failed for $tag"

            cov="$(coverage "$fdir" "$ds" tum)"
            covpct="$(awk -v c="$cov" 'BEGIN{ if(c=="") print "N/A"; else printf "%.2f", c*100 }')"
            ate="$(repeat_ate tum "$masked" "$gt" "$OUTDIR" "$tag" "$N")"

            printf "%-22s %8s %10s %20s\n" "$seq" "$conf" "$covpct" "${ate:-N/A}"
        done
    done

    echo ""
    log "done. trajectories in $OUTDIR ; per-conf detections in $DET/<seq>_conf<N>_full"
}

# ---- study 3: class-set ablation -------------------------------------------
run_class() {
    local CLASSES="${CLASSES:-person person,vehicle full}"
    local KIND="${CLASS_KIND:-kitti}"
    local OUTDIR="$OUT/sweep_class_set"
    mkdir -p "$OUTDIR"

    printf "%-14s %16s %10s %20s\n" "sequence" "class_set" "mask%" "ATE_SE3"
    printf -- "--------------------------------------------------------------------\n"

    local seq ds gt base_det
    if [ "$KIND" = "kitti" ]; then
        for seq in ${CLASS_KITTI_SEQS:-00}; do
            ds="$(kitti_ds_dir "$seq")"; gt="$(kitti_gt "$seq")"; base_det="$DET/kitti_${seq}"
            if [ ! -d "$ds/image_0" ]; then log "[SKIP] KITTI ${seq}: dataset not found ($ds/image_0)"; continue; fi
            if [ ! -d "$base_det" ]; then log "[SKIP] KITTI ${seq}: detections not found ($base_det)"; continue; fi
            _class_settings "kitti_${seq}" "$ds" "$gt" "$base_det" kitti $CLASSES
        done
    else
        for seq in ${TUM_SEQS:-walking_xyz walking_halfsphere}; do
            ds="$(tum_ds_dir "$seq")"; gt="$(tum_gt "$seq")"; base_det="$DET/${seq}"
            if [ ! -d "$ds" ]; then log "[SKIP] ${seq}: dataset not found ($ds)"; continue; fi
            if [ ! -d "$base_det" ]; then log "[SKIP] ${seq}: detections not found ($base_det)"; continue; fi
            _class_settings "$seq" "$ds" "$gt" "$base_det" tum $CLASSES
        done
    fi

    echo ""
    log "done. trajectories in $OUTDIR ; per-class detections in $DET/<tag>_cls_<slug>"
}

_class_settings() {  # <tag> <ds> <gt> <base_det> <kind> <class...>
    local tag="$1" ds="$2" gt="$3" base_det="$4" dt="$5"; shift 5
    local cls slug fdir masked cov covpct ate
    for cls in "$@"; do
        slug="$(echo "$cls" | tr ',' '_')"
        fdir="$DET/${tag}_cls_${slug}"
        masked="$MASKED/class_set/${tag}_${slug}"

        filter_dets "$base_det" "$fdir" "$cls"
        "$PYTHON" "$SEMANTIC_ROOT/scripts/mask_dataset.py" \
            --detections "$fdir" --dataset "$ds" --output "$masked" --dataset-type "$dt" \
            >/dev/null 2>&1 || log "[WARN] masking failed for ${tag}/${cls}"

        cov="$(coverage "$fdir" "$ds" "$dt")"
        covpct="$(awk -v c="$cov" 'BEGIN{ if(c=="") print "N/A"; else printf "%.2f", c*100 }')"
        ate="$(repeat_ate "$dt" "$masked" "$gt" "$OUT/sweep_class_set" "${tag}_${slug}" "$N")"

        printf "%-14s %16s %10s %20s\n" "$tag" "$cls" "$covpct" "${ate:-N/A}"
    done
}

# ---- study 4: detector-size ablation ---------------------------------------
run_model() {
    local TUM_SEQS="${TUM_SEQS:-walking_xyz walking_halfsphere}"
    local MODELS="${MODELS:-yolov8n-seg.pt yolov8s-seg.pt yolov8m-seg.pt}"
    local CONF="${CONF:-0.45}"
    local OUTDIR="$OUT/sweep_model_size"
    mkdir -p "$OUTDIR"

    printf "%-22s %18s %10s %20s\n" "sequence" "model" "mask%" "ATE_SE3"
    printf -- "--------------------------------------------------------------------\n"

    local seq ds gt model mslug tag inf_dir fdir masked ate cov covpct
    for seq in $TUM_SEQS; do
        ds="$(tum_ds_dir "$seq")"
        gt="$(tum_gt "$seq")"
        if [ ! -d "$ds" ]; then log "[SKIP] ${seq}: dataset not found ($ds)"; continue; fi

        for model in $MODELS; do
            mslug="$(echo "$model" | sed 's/\.pt$//; s/yolov8//; s/-seg//')"
            tag="${seq}_${mslug}"
            inf_dir="$DET/${tag}"
            fdir="$DET/${tag}_full"
            masked="$MASKED/model_size/${tag}"

            yolo_infer "$ds" "$inf_dir" "$CONF" "$model"
            filter_dets "$inf_dir" "$fdir" "full"
            "$PYTHON" "$SEMANTIC_ROOT/scripts/mask_dataset.py" \
                --detections "$fdir" --dataset "$ds" --output "$masked" --dataset-type tum \
                >/dev/null 2>&1 || log "[WARN] masking failed for $tag"

            cov="$(coverage "$fdir" "$ds" tum)"
            covpct="$(awk -v c="$cov" 'BEGIN{ if(c=="") print "N/A"; else printf "%.2f", c*100 }')"
            ate="$(repeat_ate tum "$masked" "$gt" "$OUTDIR" "$tag" "$N")"

            printf "%-22s %18s %10s %20s\n" "$seq" "$model" "$covpct" "${ate:-N/A}"
        done
    done

    echo ""
    log "done. trajectories in $OUTDIR"
}

# ---- study 6: bounding-box expansion (delta) sweep -------------------------
run_dilation() {
    local TUM_SEQS="${TUM_SEQS:-walking_xyz walking_halfsphere}"
    local DELTAS="${DELTAS:-0 3 6 9}"
    local OUTDIR="$OUT/sweep_dilation"
    mkdir -p "$OUTDIR"

    printf "%-22s %8s %10s %20s\n" "sequence" "delta" "mask%" "ATE_SE3"
    printf -- "--------------------------------------------------------------------\n"

    local seq ds gt base_det fdir d tag masked ate cov covpct
    for seq in $TUM_SEQS; do
        ds="$(tum_ds_dir "$seq")"
        gt="$(tum_gt "$seq")"
        base_det="$DET/${seq}"
        fdir="$DET/${seq}_full"
        if [ ! -d "$ds" ]; then log "[SKIP] ${seq}: dataset not found ($ds)"; continue; fi
        if [ ! -d "$base_det" ]; then log "[SKIP] ${seq}: detections not found ($base_det)"; continue; fi

        filter_dets "$base_det" "$fdir" "full"

        for d in $DELTAS; do
            tag="${seq}_d${d}"
            masked="$MASKED/dilation/${tag}"

            "$PYTHON" "$SEMANTIC_ROOT/scripts/mask_dataset.py" \
                --detections "$fdir" --dataset "$ds" --output "$masked" \
                --dataset-type tum --dilate-px "$d" >/dev/null 2>&1 || log "[WARN] masking failed ($seq d=$d)"

            cov="$(coverage "$fdir" "$ds" tum "$d")"
            covpct="$(awk -v c="$cov" 'BEGIN{ if(c=="") print "N/A"; else printf "%.2f", c*100 }')"
            ate="$(repeat_ate tum "$masked" "$gt" "$OUTDIR" "$tag" "$N")"

            printf "%-22s %8s %10s %20s\n" "$seq" "$d" "$covpct" "${ate:-N/A}"
        done
    done

    echo ""
    log "done. trajectories in $OUTDIR"
}

# ---- dispatch (a single invocation runs only the selected modes) ------------
if has conf;     then run_conf;     fi
if has class;    then run_class;    fi
if has model;    then run_model;    fi
if has dilation; then run_dilation; fi
