#!/usr/bin/env bash
# =============================================================================
# common.sh -- shared helpers for the supplemental ablation studies.
# Source this from the sweep scripts. Paths are read from the environment
# (see run_all.sh for defaults).
# =============================================================================
#
# Required environment (export before sourcing, or set at the top of run_all.sh):
#   SEMANTIC_ROOT   repo root: .../semantic-slam-yolov8/semantic-slam-experiment
#   DATA_DIR        autodl data root containing datasets/, detections/, output/
#   ORB_SLAM3_ROOT  ORB-SLAM3 build root            (default: $HOME/ORB_SLAM3)
#   RGBD_TUM        path to the rgbd_tum binary     (auto-discovered if unset)
#   STEREO_KITTI    path to the stereo_kitti binary (auto-discovered if unset)
#   VOCAB           path to ORBvoc.txt              (default: $ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt)
#   PYTHON          python interpreter               (default: python3)
#   SLAM_TIMEOUT    per-run timeout, TUM seconds     (default: 180)
#   KITTI_TIMEOUT   per-run timeout, KITTI seconds   (default: 900)
# -----------------------------------------------------------------------------

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_DIR
: "${PYTHON:=python3}"
: "${ORB_SLAM3_ROOT:=$HOME/ORB_SLAM3}"
: "${SLAM_TIMEOUT:=180}"
: "${KITTI_TIMEOUT:=900}"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# Resolve and export the repo-relative paths; export SEMANTIC_ROOT so the python
# helpers can import compute_ate.py / mask_dataset.py without absolute paths.
require_env() {
    : "${SEMANTIC_ROOT:?SEMANTIC_ROOT not set (repo root: .../semantic-slam-experiment)}"
    : "${DATA_DIR:?DATA_DIR not set (autodl data root)}"
    [ -d "$SEMANTIC_ROOT/scripts" ] || die "no scripts/ under SEMANTIC_ROOT=$SEMANTIC_ROOT"
    export SEMANTIC_ROOT
    export DATA_DIR
    CONFIG="$SEMANTIC_ROOT/src/config"
    DET="$DATA_DIR/detections"
    MASKED="$DATA_DIR/datasets_masked"
    OUT="$SEMANTIC_ROOT/output"
    export CONFIG DET MASKED OUT
}

# The rgbd_tum of some builds is linked against libSemanticSLAM.so (the C++
# semantic module under src/). If that library exists but its directory is not on
# LD_LIBRARY_PATH, the SLAM binary aborts at start-up, so add it automatically.
ensure_lib_path() {
    local cand libdir=""
    for cand in "${BUILD:-}" "$SEMANTIC_ROOT/src/build" "$ORB_SLAM3_ROOT/lib"; do
        if [ -n "$cand" ] && [ -e "$cand/libSemanticSLAM.so" ]; then libdir="$cand"; break; fi
    done
    if [ -z "$libdir" ]; then
        libdir="$(find "$SEMANTIC_ROOT/src" "$ORB_SLAM3_ROOT" -maxdepth 6 \
                 -name 'libSemanticSLAM.so' -printf '%h\n' 2>/dev/null | head -1)"
    fi
    [ -n "$libdir" ] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$libdir:"*) : ;;
        *) export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}" ;;
    esac
    export LD_LIBRARY_PATH
    log "LD_LIBRARY_PATH += $libdir (libSemanticSLAM.so)"
}

discover_bins() {
    [ -n "${VOCAB:-}" ] || VOCAB="$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt"
    [ -n "${RGBD_TUM:-}" ] || RGBD_TUM=$(find "$ORB_SLAM3_ROOT" -type f -name rgbd_tum \
        ! -name '*.cc' ! -name '*.cpp' ! -name '*.h' 2>/dev/null | head -1)
    [ -n "${STEREO_KITTI:-}" ] || STEREO_KITTI=$(find "$ORB_SLAM3_ROOT" -type f -name stereo_kitti \
        ! -name '*.cc' ! -name '*.cpp' ! -name '*.h' 2>/dev/null | head -1)
    export VOCAB RGBD_TUM STEREO_KITTI
    ensure_lib_path
    log "RGBD_TUM=${RGBD_TUM:-NOT FOUND}"
    log "STEREO_KITTI=${STEREO_KITTI:-NOT FOUND}"
    log "VOCAB=${VOCAB:-NOT FOUND}"
}

# Map a short TUM name to its dataset directory name.
tum_ds_dir() { echo "$DATA_DIR/datasets/TUM/rgbd_dataset_freiburg3_$1"; }
tum_gt()     { echo "$DATA_DIR/datasets/TUM/rgbd_dataset_freiburg3_$1/groundtruth.txt"; }

ensure_tum_associate() {
    local ds="$1"
    [ -f "$ds/associate.txt" ] && return 0
    [ -f "$ds/rgb.txt" ] && [ -f "$ds/depth.txt" ] || return 1
    "$PYTHON" "$SEMANTIC_ROOT/scripts/associate.py" "$ds/rgb.txt" "$ds/depth.txt" > "$ds/associate.txt" 2>/dev/null
}

_slam_timeout_run() {  # <timeout> <cmd...>
    local t="$1"; shift
    if command -v xvfb-run >/dev/null 2>&1; then
        xvfb-run -a timeout --signal=TERM --kill-after=5 "$t" "$@"
    else
        DISPLAY="" timeout --signal=TERM --kill-after=5 "$t" "$@"
    fi
    local rc=$?
    pkill -9 -x rgbd_tum 2>/dev/null || true
    pkill -9 -x stereo_kitti 2>/dev/null || true
    sleep 1
    return $rc
}

# run_tum <dataset_dir> <out_traj> [yaml_name]
run_tum() {
    local ds="$1" out="$2" yaml="${3:-TUM3_baseline.yaml}" ypath rundir
    case "$yaml" in /*) ypath="$yaml";; *) ypath="$CONFIG/$yaml";; esac
    [ -d "$ds" ] || { log "[WARN] dataset missing: $ds"; return 1; }
    [ -x "$RGBD_TUM" ] || { log "[WARN] rgbd_tum not found; cannot run TUM"; return 1; }
    ensure_tum_associate "$ds" || log "[WARN] associate.txt unavailable for $ds"
    # ORB-SLAM3 writes CameraTrajectory.txt to the process CWD, so run the binary
    # from a known directory and read the trajectory back from there.
    rundir="$(dirname "$out")"; mkdir -p "$rundir"
    rm -f "$rundir/CameraTrajectory.txt"
    log "ORB-SLAM3 TUM: $ds -> $out (yaml=$ypath)"
    ( cd "$rundir" && _slam_timeout_run "$SLAM_TIMEOUT" "$RGBD_TUM" "$VOCAB" "$ypath" "$ds" "$ds/associate.txt" )
    if [ -s "$rundir/CameraTrajectory.txt" ]; then
        cp "$rundir/CameraTrajectory.txt" "$out"
        rm -f "$rundir/CameraTrajectory.txt" "$rundir/KeyFrameTrajectory.txt"
        log "traj saved: $out"
    else
        log "[WARN] no trajectory produced for $ds (expected $rundir/CameraTrajectory.txt)"
    fi
}

# run_kitti <dataset_dir> <out_traj> [yaml_name]
run_kitti() {
    local ds="$1" out="$2" yaml="${3:-KITTI00_baseline.yaml}" ypath rundir
    case "$yaml" in /*) ypath="$yaml";; *) ypath="$CONFIG/$yaml";; esac
    [ -d "$ds" ] || { log "[WARN] dataset missing: $ds"; return 1; }
    [ -x "$STEREO_KITTI" ] || { log "[WARN] stereo_kitti not found; cannot run KITTI"; return 1; }
    rundir="$(dirname "$out")"; mkdir -p "$rundir"
    rm -f "$rundir/CameraTrajectory.txt"
    log "ORB-SLAM3 KITTI: $ds -> $out"
    ( cd "$rundir" && _slam_timeout_run "$KITTI_TIMEOUT" "$STEREO_KITTI" "$VOCAB" "$ypath" "$ds" )
    if [ -s "$rundir/CameraTrajectory.txt" ]; then
        cp "$rundir/CameraTrajectory.txt" "$out"
        rm -f "$rundir/CameraTrajectory.txt" "$rundir/KeyFrameTrajectory.txt"
        log "traj saved: $out"
    else
        log "[WARN] no trajectory produced for $ds (expected $rundir/CameraTrajectory.txt)"
    fi
}

# ate_se3 <traj> <gt>  -> echoes the SE(3) ATE RMSE (or "N/A")
ate_se3() {
    local traj="$1" gt="$2"
    if [ ! -s "$traj" ] || [ ! -f "$gt" ]; then echo "N/A"; return; fi
    "$PYTHON" "$SEMANTIC_ROOT/scripts/compute_ate.py" "$traj" "$gt" 2>/dev/null \
        | grep -i "ATE RMSE" | awk '{print $(NF-1)}'
}

# KITTI Odometry paths (one calibration, and GT poses shared under poses/).
kitti_ds_dir() { echo "$DATA_DIR/datasets/KITTI/$1"; }
kitti_gt()     { echo "$DATA_DIR/datasets/KITTI/poses/$1.txt"; }

# _ate_one <traj> <gt> <format>  -> SE(3) ATE RMSE (or "N/A"); format = tum|kitti
_ate_one() {
    local traj="$1" gt="$2" fmt="${3:-tum}"
    if [ ! -s "$traj" ] || [ ! -f "$gt" ]; then echo "N/A"; return; fi
    if [ "$fmt" = "kitti" ]; then
        "$PYTHON" "$EXP_DIR/traj_eval.py" --est "$traj" --gt "$gt" --format kitti 2>/dev/null \
            | grep "ATE_SE3 RMSE" | awk '{print $NF}'
    else
        "$PYTHON" "$SEMANTIC_ROOT/scripts/compute_ate.py" "$traj" "$gt" 2>/dev/null \
            | grep -i "ATE RMSE" | awk '{print $(NF-1)}'
    fi
}

# repeat_ate <kind> <masked_dir> <gt> <outdir> <tag> [n] [yaml]
#   Runs the SLAM n times on the same masked dataset (ORB-SLAM3 is not run-to-run
#   deterministic, so a single run cannot separate a real effect from run noise)
#   and echoes "mean +/- std (n=k)" over the successful ATE values, or "N/A".
#   kind = tum | kitti.  Per-run trajectories are kept as <outdir>/<tag>_r<i>.txt.
repeat_ate() {
    local kind="$1" masked="$2" gt="$3" outdir="$4" tag="$5" n="${6:-1}" yaml="${7:-}"
    mkdir -p "$outdir"
    local fmt="tum" i traj a vals=""
    [ "$kind" = "kitti" ] && fmt="kitti"
    for i in $(seq 1 "$n"); do
        traj="$outdir/${tag}_r${i}.txt"
        # Chatter goes to stderr so command substitution captures only the aggregate.
        if [ "$kind" = "kitti" ]; then
            if [ -n "$yaml" ]; then run_kitti "$masked" "$traj" "$yaml" >&2; else run_kitti "$masked" "$traj" >&2; fi
        else
            if [ -n "$yaml" ]; then run_tum "$masked" "$traj" "$yaml" >&2; else run_tum "$masked" "$traj" >&2; fi
        fi
        a="$(_ate_one "$traj" "$gt" "$fmt")"
        if [ -n "$a" ] && [ "$a" != "N/A" ]; then vals="$vals $a"; fi
    done
    printf '%s' "$vals" | awk '{n=0;s=0;q=0;for(i=1;i<=NF;i++){v=$i+0;s+=v;q+=v*v;n++} if(n==0){print "N/A"} else {m=s/n; d=(n>1)?(q-n*m*m)/(n-1):0; if(d<0)d=0; printf "%.6f +/- %.6f (n=%d)", m, sqrt(d), n}}'
}

# coverage <det_dir> <dataset_dir> <dataset_type> [dilate_px] -> mean fraction (0..1)
coverage() {
    local det="$1" ds="$2" dt="${3:-auto}" d="${4:-3}"
    "$PYTHON" "$EXP_DIR/mask_coverage.py" --detections "$det" --dataset "$ds" \
        --dataset-type "$dt" --dilate-px "$d" --mean-only 2>/dev/null
}

# yolo_infer <dataset> <out_det_dir> <conf> [model]
yolo_infer() {
    local ds="$1" out="$2" conf="$3" model="${4:-yolov8n-seg.pt}"
    mkdir -p "$out"
    log "YOLO inference ($model, conf=$conf): $ds"
    "$PYTHON" "$SEMANTIC_ROOT/scripts/yolov8_offline_inference.py" \
        --dataset "$ds" --output "$out" --model "$model" --conf "$conf" \
        >/dev/null 2>&1 || log "[WARN] inference reported an error"
}

# filter_dets <in> <out> <classes> [conf_min]
filter_dets() {
    local in="$1" out="$2" classes="$3" cmin="${4:-}"
    if [ -n "$cmin" ]; then
        "$PYTHON" "$EXP_DIR/filter_detections.py" --input "$in" --output "$out" \
            --classes "$classes" --conf-min "$cmin"
    else
        "$PYTHON" "$EXP_DIR/filter_detections.py" --input "$in" --output "$out" \
            --classes "$classes"
    fi
}
