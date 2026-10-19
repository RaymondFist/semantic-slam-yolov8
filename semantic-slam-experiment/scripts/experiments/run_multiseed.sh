#!/usr/bin/env bash
# =============================================================================
# run_multiseed.sh -- Study 9: N>=10 repeated runs + paired Wilcoxon.
# =============================================================================
# Repeats the TUM baseline and mask pipelines N times each and aggregates ATE as
# mean +/- std with a paired Wilcoxon signed-rank test (n>=10), via
# aggregate_runs.py (glob-discovered run files).
#
# VARIANCE POLICY:
#   This ORB-SLAM3 build is not run-to-run deterministic: repeated runs on the
#   same masked input were measured to differ by >10% ATE, so plain re-runs do
#   yield a paired spread. An optional per-run settings variation is still
#   supported for a stronger stress test:
#     VAR_MODE=none       (default) -> identical settings each run; the measured
#                          spread comes from the SLAM's own non-determinism.
#     VAR_MODE=nfeatures  -> each run r uses a copy of TUM3_baseline.yaml with
#                          ORBextractor.nFeatures = base + (r-1)*NF_STEP. This
#                          injects controlled, documented variation (the choice
#                          is reported in the log/output).
#
# The mask config reuses the already-generated masked dataset
# ($DATA_DIR/datasets_masked/TUM/rgbd_dataset_freiburg3_<seq>); run the masking
# step (deploy / sweep_*) first if it does not exist.
#
# Usage:
#   SEMANTIC_ROOT=<repo> DATA_DIR=<data> bash run_multiseed.sh
#   TUM_SEQS="walking_xyz" N=10 VAR_MODE=nfeatures NF_STEP=20 bash run_multiseed.sh
#
# Env overrides: TUM_SEQS, N (default 10), VAR_MODE (none|nfeatures), NF_STEP (20).
# -----------------------------------------------------------------------------
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_env
discover_bins

TUM_SEQS="${TUM_SEQS:-walking_xyz walking_halfsphere}"
N="${N:-10}"
VAR_MODE="${VAR_MODE:-none}"
NF_STEP="${NF_STEP:-20}"
OUTDIR="$OUT/multiseed"
RUNDIR="$OUTDIR/runs"
YAMLDIR="$OUTDIR/yaml"
mkdir -p "$RUNDIR" "$YAMLDIR"

if [ "$N" -lt 10 ]; then
    log "[NOTE] N=$N < 10 (study requests N>=10)."
fi
if [ "$VAR_MODE" = "none" ]; then
    log "[NOTE] VAR_MODE=none: identical settings each run; the paired spread comes"
    log "       from the SLAM's own non-determinism (measured >10% ATE between reruns)."
    log "       Set VAR_MODE=nfeatures for an additional disclosed nFeatures variation."
fi

make_yaml() {  # <run_index> -> path to a settings yaml (or the default name)
    local r="$1"
    if [ "$VAR_MODE" != "nfeatures" ]; then echo "TUM3_baseline.yaml"; return; fi
    local base nf outf
    base="$CONFIG/TUM3_baseline.yaml"
    [ -f "$base" ] || { log "[WARN] $base missing; cannot build variant yaml"; echo "TUM3_baseline.yaml"; return; }
    nf="$(grep -E 'nFeatures' "$base" | head -1 | grep -oE '[0-9]+' | head -1)"
    nf="${nf:-1000}"
    outf="$YAMLDIR/TUM3_baseline_nf$((nf + (r - 1) * NF_STEP)).yaml"
    sed "s/nFeatures:.*/nFeatures: $((nf + (r - 1) * NF_STEP))/" "$base" > "$outf"
    echo "$outf"
}

for seq in $TUM_SEQS; do
    ds="$(tum_ds_dir "$seq")"
    masked="$MASKED/TUM/rgbd_dataset_freiburg3_$seq"
    gt="$(tum_gt "$seq")"
    if [ ! -d "$ds" ]; then log "[SKIP] ${seq}: dataset not found ($ds)"; continue; fi
    if [ ! -d "$masked" ]; then
        log "[SKIP] ${seq} mask config: masked dataset missing ($masked); run the masking step first."
    fi

    for r in $(seq 1 "$N"); do
        ypath="$(make_yaml "$r")"
        run_tum "$ds"     "$RUNDIR/run_${seq}_baseline_${r}.txt" "$ypath"
        if [ -d "$masked" ]; then
            run_tum "$masked" "$RUNDIR/run_${seq}_mask_${r}.txt" "$ypath"
        fi
    done

    echo ""
    log "=== aggregate ${seq} (VAR_MODE=$VAR_MODE, N=$N) ==="
    if [ -d "$masked" ]; then
        "$PYTHON" "$EXP_DIR/aggregate_runs.py" --gt "$gt" --format tum \
            --baseline-glob "$RUNDIR/run_${seq}_baseline_*.txt" \
            --mask-glob "$RUNDIR/run_${seq}_mask_*.txt" \
            --json "$OUTDIR/stats_${seq}.json"
    else
        log "[WARN] mask runs absent; aggregating baseline only is not supported by aggregate_runs.py."
    fi
done

echo ""
log "done. per-run trajectories in $RUNDIR ; stats in $OUTDIR/stats_*.json"
log "VARIANCE POLICY USED: VAR_MODE=$VAR_MODE $([ "$VAR_MODE" = nfeatures ] && echo "(nFeatures += $((${NF_STEP:-20})) per run)")"
