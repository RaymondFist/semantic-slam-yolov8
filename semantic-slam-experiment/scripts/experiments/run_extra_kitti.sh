#!/usr/bin/env bash
# =============================================================================
# run_extra_kitti.sh -- Study 8: additional KITTI Odometry sequences + ATE/RPE.
# =============================================================================
# Runs the full pipeline (YOLO inference -> full-set class re-filter -> box mask ->
# stock ORB-SLAM3) for KITTI Odometry sequences 01,04,05,07,08,09,10 and reports
# both SE(3) ATE (via compute_ate.py) and KITTI RPE (translation, Delta=1..10
# mean, via traj_eval.py -- the repo lacked an RPE routine).
#
# REQUIRED DATA (the user must supply; not downloaded here):
#   $DATA_DIR/datasets/KITTI/<seq>/image_0/ , image_1/ , times.txt (optional)
#   $DATA_DIR/datasets/KITTI/poses/<seq>.txt          (ground-truth poses)
#   KITTI odometry shares one calibration, so KITTI00_baseline.yaml is reused.
# Sequences whose dataset dir is missing are skipped with a clear message.
#
# Usage:
#   SEMANTIC_ROOT=<repo> DATA_DIR=<data> bash run_extra_kitti.sh
#   KITTI_SEQS="01 04 05" CONF=0.45 bash run_extra_kitti.sh
#
# Each configuration is run N times (default 5) and reported as mean +/- std:
# this ORB-SLAM3 build is not run-to-run deterministic, so a single run cannot
# separate a real effect from run noise.
#
# Env overrides: KITTI_SEQS, CONF (default 0.45), N (default 5).
# -----------------------------------------------------------------------------
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_env
discover_bins

KITTI_SEQS="${KITTI_SEQS:-01 04 05 07 08 09 10}"
CONF="${CONF:-0.45}"
N="${N:-5}"          # repeats per configuration; this ORB-SLAM3 build is not run-to-run deterministic
OUTDIR="$OUT/extra_kitti"
mkdir -p "$OUTDIR"

eval_kitti() {  # <traj> <gt> -> prints "<ate> <rpe>"
    local traj="$1" gt="$2" out ate rpe
    if [ ! -s "$traj" ] || [ ! -f "$gt" ]; then echo "N/A N/A"; return; fi
    out="$("$PYTHON" "$EXP_DIR/traj_eval.py" --est "$traj" --gt "$gt" --format kitti 2>/dev/null)"
    ate="$(echo "$out" | grep "ATE_SE3 RMSE" | awk '{print $NF}')"
    rpe="$(echo "$out" | grep "RPE (mean" | awk '{print $NF}')"
    echo "${ate:-N/A} ${rpe:-N/A}"
}

# Mean +/- std of the per-run RPE over the trajectories saved by repeat_ate.
rpe_meanstd() {  # <gt> <outdir> <tag> <n> -> "mean +/- std (n=k)"
    local gt="$1" outdir="$2" tag="$3" n="$4" i _a v vals=""
    for i in $(seq 1 "$n"); do
        read -r _a v <<<"$(eval_kitti "$outdir/${tag}_r${i}.txt" "$gt")"
        if [ -n "$v" ] && [ "$v" != "N/A" ]; then vals="$vals $v"; fi
    done
    printf '%s' "$vals" | awk '{for(i=1;i<=NF;i++){x=$i+0;s+=x;q+=x*x;n++}} END{if(n==0){print "N/A"} else {m=s/n; d=(n>1)?(q-n*m*m)/(n-1):0; if(d<0)d=0; printf "%.4f +/- %.4f (n=%d)", m, sqrt(d), n}}'
}

printf "%-6s %30s %30s %30s %30s\n" "seq" "base_ATE_SE3(N)" "mask_ATE_SE3(N)" "base_RPE(N)" "mask_RPE(N)"
printf -- "------------------------------------------------------------------------------------------------------\n"

for seq in $KITTI_SEQS; do
    ds="$DATA_DIR/datasets/KITTI/$seq"
    gt="$DATA_DIR/datasets/KITTI/poses/$seq.txt"
    if [ ! -d "$ds/image_0" ]; then
        log "[SKIP] KITTI $seq: dataset not found ($ds/image_0). Download the odometry sequence and place it here."
        continue
    fi
    [ -f "$gt" ] || log "[WARN] KITTI $seq: GT poses missing ($gt)"

    det="$DET/kitti_${seq}"
    fdir="$DET/kitti_${seq}_full"
    masked="$MASKED/KITTI/$seq"

    yolo_infer "$ds" "$det" "$CONF"
    filter_dets "$det" "$fdir" "full"
    "$PYTHON" "$SEMANTIC_ROOT/scripts/mask_dataset.py" \
        --detections "$fdir" --dataset "$ds" --output "$masked" --dataset-type kitti \
        >/dev/null 2>&1 || log "[WARN] masking failed for KITTI $seq"

    b_ate="$(repeat_ate kitti "$ds"     "$gt" "$OUTDIR" "kitti_${seq}_baseline" "$N")"
    m_ate="$(repeat_ate kitti "$masked" "$gt" "$OUTDIR" "kitti_${seq}_mask"     "$N")"
    b_rpe="$(rpe_meanstd "$gt" "$OUTDIR" "kitti_${seq}_baseline" "$N")"
    m_rpe="$(rpe_meanstd "$gt" "$OUTDIR" "kitti_${seq}_mask"     "$N")"

    printf "%-6s %30s %30s %30s %30s\n" "$seq" "$b_ate" "$m_ate" "$b_rpe" "$m_rpe"
done

echo ""
log "done. trajectories in $OUTDIR ; ATE=SE(3) rmse(m), RPE=mean trans error Delta 1..10(m)"
