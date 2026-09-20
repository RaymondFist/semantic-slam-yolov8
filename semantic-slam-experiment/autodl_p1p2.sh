#!/usr/bin/env bash
# =============================================================================
# autodl_p1p2.sh - P1 multi-run statistics + P2 random mask comparison (one-click script for AutoDL cloud)
#
# Prerequisites: environment deployed as in autodl_deploy.sh
#   ${DATA_DIR}/datasets/TUM/{seq}                raw RGB-D data + associate.txt
#   ${DATA_DIR}/datasets_masked/TUM/{seq}         YOLO masks (for the E1.5 / mask experiment)
#   ${DATA_DIR}/detections                        YOLO detections (for P2 random masks)
#   $RGBD_TUM (ORB-SLAM3 rgbd_tum executable) $ORB_SLAM3_VOCAB $CONFIG
#   evo (evo_ape) and python3/numpy installed
#
# Usage:  bash autodl_p1p2.sh [N_RUNS] [--no-p2]
#   N_RUNS   number of repeats per config (baseline/mask), default 5
#   --no-p2  skip the P2 random mask comparison
#
# Output:
#   ${OUTPUT}/p1/run_{seq}_{cfg}_{k}.txt   per-run trajectories (audit archive)
#   ${OUTPUT}/p1/raw_{seq}_{cfg}.txt       per-run RMSE (space separated)
#   ${OUTPUT}/p1/p1_stats.json             mean/std/n (produced by the self-contained stats script)
#   ${DATA_DIR}/datasets_randmask/...       random mask datasets (seed=42)
#   ${OUTPUT}/p2/randmask_raw.txt / p2_compare.json
# =============================================================================
set -u

# ------------------------------ Environment variables (overridable) ------------------------------
# Same as autodl_deploy.sh: SEMANTIC_SLAM_ROOT = the script's directory (experiment package root, i.e. semantic-slam-experiment)
# Note: the script lives in the experiment package root, so SLAM_ROOT = dirname itself; do not append "/.." (avoids climbing one level too far)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMANTIC_SLAM_ROOT="${SEMANTIC_SLAM_ROOT:-$SCRIPT_DIR}"
# Data directory matches deploy: ${SEMANTIC_SLAM_ROOT}/data (contains datasets / datasets_masked / datasets_randmask / detections / experiment_results.json)
DATA_DIR="${DATA_DIR:-$SEMANTIC_SLAM_ROOT/data}"
# Output directory matches deploy: ${SEMANTIC_SLAM_ROOT}/output (p1/ and p2/ subdirectories)
OUTPUT="${OUTPUT:-${SEMANTIC_SLAM_ROOT}/output}"
ORB_SLAM3_ROOT="${ORB_SLAM3_ROOT:-$HOME/ORB_SLAM3}"
: "${ORB_SLAM3_VOCAB:=}"
if [ -z "$ORB_SLAM3_VOCAB" ]; then
  if [ -f "${SEMANTIC_SLAM_ROOT}/models/ORBvoc.txt" ]; then
    ORB_SLAM3_VOCAB="${SEMANTIC_SLAM_ROOT}/models/ORBvoc.txt"
  else
    ORB_SLAM3_VOCAB="$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt"
  fi
fi
BUILD="${BUILD:-$SEMANTIC_SLAM_ROOT/src/build}"
# Config directory uses src/config inside the experiment package (matches the deploy script CONFIG); overridable
CONFIG="${CONFIG:-$SEMANTIC_SLAM_ROOT/src/config}"
# Executable: same as deploy, locate rgbd_tum automatically with find
if [ -z "${RGBD_TUM:-}" ]; then
  RGBD_TUM=$(find "$ORB_SLAM3_ROOT" -path "*/Examples/RGB-D/rgbd_tum" -type f ! -name "*.cc" ! -name "*.cpp" ! -name "*.h" 2>/dev/null | head -1)
fi
if [ -z "${RGBD_TUM:-}" ]; then
  echo "[FATAL] rgbd_tum executable not found. Check that ORB_SLAM3 is built, or set RGBD_TUM=/path/to/rgbd_tum"; exit 1
fi
echo "  Using RGBD_TUM=$RGBD_TUM"
echo "  Using VOCAB=$ORB_SLAM3_VOCAB"
echo "  Using CONFIG_DIR=$CONFIG"

# rgbd_tum is the semantic build linked against our libSemanticSLAM.so, so the semantic library directory must be on the library search path
# (the .so is built from src/; if LD_LIBRARY_PATH was not preset, add it here automatically)
if [ -f "${BUILD}/libSemanticSLAM.so" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${BUILD}:"*) : ;;
    *) export LD_LIBRARY_PATH="${BUILD}:${LD_LIBRARY_PATH:-}" ;;
  esac
  echo "  Added semantic library search path: $BUILD"
fi

N_RUNS="${1:-5}"
DO_P2=1
for a in "$@"; do [ "$a" = "--no-p2" ] && DO_P2=0; done

mkdir -p "${OUTPUT}/p1" "${OUTPUT}/p2"

# ------------------------------ Sequence definitions ------------------------------
declare -A LABEL_DIR
for e in "sitting_static|rgbd_dataset_freiburg3_sitting_static" \
         "sitting_xyz|rgbd_dataset_freiburg3_sitting_xyz" \
         "walking_static|rgbd_dataset_freiburg3_walking_static" \
         "walking_xyz|rgbd_dataset_freiburg3_walking_xyz" \
         "walking_halfsphere|rgbd_dataset_freiburg3_walking_halfsphere"; do
  IFS='|' read -r lbl ds <<< "$e"
  LABEL_DIR[$lbl]="$ds"
done
LABELS=(sitting_static sitting_xyz walking_static walking_xyz walking_halfsphere)

gt_for_seq() { case "$1" in
  sitting_static) echo "${DATA_DIR}/gt_trajectories/TUM/rgbd_dataset_freiburg3_sitting_static_groundtruth.txt";;
  sitting_xyz)    echo "${DATA_DIR}/gt_trajectories/TUM/rgbd_dataset_freiburg3_sitting_xyz_groundtruth.txt";;
  walking_static) echo "${DATA_DIR}/gt_trajectories/TUM/rgbd_dataset_freiburg3_walking_static_groundtruth.txt";;
  walking_xyz)    echo "${DATA_DIR}/gt_trajectories/TUM/rgbd_dataset_freiburg3_walking_xyz_groundtruth.txt";;
  walking_halfsphere) echo "${DATA_DIR}/gt_trajectories/TUM/rgbd_dataset_freiburg3_walking_halfsphere_groundtruth.txt";;
esac; }

# ------------------------------ Single SLAM run + eval ------------------------------
# Headless server has no X display: rgbd_tum's Pangolin Viewer aborts when it cannot open X,
# so use xvfb-run to provide a virtual X (same as autodl_deploy.sh); if xvfb is missing, disable DISPLAY.
USE_XVFB=""
if command -v xvfb-run >/dev/null 2>&1; then USE_XVFB=1; fi
run_slam() {  # $1=dataset_dir  $2=out_traj
  local d="$1" out="$2"
  local vocab="$ORB_SLAM3_VOCAB"; [ -f "$vocab" ] || vocab="$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt"
  local cfg="${CONFIG}/TUM3_baseline.yaml"
  # Keep stdout+stderr per run in <out>.log in the same directory, so instant crashes are easy to diagnose (writing to /dev/null hid everything)
  local log="${out%.txt}.log"
  local ec=0
  if [ "$USE_XVFB" = "1" ]; then
    xvfb-run -a timeout --signal=TERM --kill-after=5 300 \
      "$RGBD_TUM" "$vocab" "$cfg" "$d" "$d/associate.txt" >"$log" 2>&1 || ec=$?
  else
    DISPLAY="" timeout 300 "$RGBD_TUM" "$vocab" "$cfg" "$d" "$d/associate.txt" >"$log" 2>&1 || ec=$?
  fi
  echo "   [run] exit=$ec log=$log"
  # rgbd_tum writes CameraTrajectory.txt to the CWD (the script's working directory), not to $HOME/ORB_SLAM3/
  # The old code only looked for stale files under $HOME/ORB_SLAM3/, so fresh trajectories were never copied (root cause of the empty ATE values).
  # Now prefer the trajectory just written to the CWD, with the stale path as a fallback.
  local camtraj=""
  for cand in "$PWD/CameraTrajectory.txt" "$HOME/ORB_SLAM3/CameraTrajectory.txt"; do
    if [ -s "$cand" ]; then camtraj="$cand"; break; fi
  done
  if [ -n "$camtraj" ]; then
    cp "$camtraj" "$out"
  else
    echo "   [WARN] CameraTrajectory.txt not found after run (log=$log)"
  fi
}

ev_rmse() {  # $1=gt  $2=traj  -> prints the rmse value or empty
  local out
  out=$(eval "evo_ape tum \"$1\" \"$2\" --align --correct_scale" 2>&1)
  echo "$out" | awk '/rmse/{for(i=1;i<=NF;i++) if($i=="rmse"){print $(i+1); exit}}'
}

# =============================================================================
# P1: Multi-run statistics
# =============================================================================
echo "[P1] Multi-run statistics: ${N_RUNS} runs per config, 5 sequences x baseline/mask"
declare -A P1_RAW
for lbl in "${LABELS[@]}"; do
  for cfg in baseline mask; do
    if [ "$cfg" = baseline ]; then
      d="${DATA_DIR}/datasets/TUM/${LABEL_DIR[$lbl]}"
    else
      d="${DATA_DIR}/datasets_masked/TUM/${LABEL_DIR[$lbl]}"
    fi
    gt=$(gt_for_seq "$lbl")
    [ -f "$gt" ] || { echo "  [WARN] $lbl missing GT"; continue; }
    [ -d "$d" ] || { echo "  [WARN] $lbl/$cfg missing data dir=$d"; continue; }

    rms=()
    for k in $(seq 1 "$N_RUNS"); do
      traj="${OUTPUT}/p1/run_${lbl}_${cfg}_${k}.txt"
      run_slam "$d" "$traj"
      if [ -s "$traj" ]; then
        r=$(ev_rmse "$gt" "$traj")
      else
        r=""
      fi
      rms+=("$r")
      echo "    run${k}/${N_RUNS} ${lbl}/${cfg}: ATE=[$r]"
    done
    P1_RAW["$lbl|$cfg"]="${rms[*]}"
  done
done

# ---- Write raw files ----
for lbl in "${LABELS[@]}"; do
  for cfg in baseline mask; do
    echo "${P1_RAW[$lbl|$cfg]:-}" > "${OUTPUT}/p1/raw_${lbl}_${cfg}.txt"
  done
done

# ---- Self-contained statistics ----
cat > "${OUTPUT}/p1/stats.py" <<'PYEOF'
import json, re, os
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
labels = ['sitting_static','sitting_xyz','walking_static','walking_xyz','walking_halfsphere']
def num(s):
    m = re.search(r'[0-9]*\.?[0-9]+', s)
    return float(m.group()) if m else None
def load(name):
    txt = open(f"{OUT}/raw_{name}.txt").read().split()
    return [num(x) for x in txt if num(x) is not None]
res = {}
for lbl in labels:
    b, m = load(f"{lbl}_baseline"), load(f"{lbl}_mask")
    row = {'seq': lbl, 'baseline': {'n': len(b)}, 'mask': {'n': len(m)}}
    for k, a in (('baseline', b), ('mask', m)):
        if a:
            row[k]['rmse_mean'] = float(np.mean(a))
            row[k]['rmse_std'] = float(np.std(a, ddof=1)) if len(a) > 1 else None
            row[k]['rmse_pairs'] = a
    if b and m:
        imp = (np.mean(b) - np.mean(m)) / np.mean(b) * 100
        row['improvement'] = {'mean_pct': float(imp),
                              'baseline_mean': float(np.mean(b)), 'mask_mean': float(np.mean(m))}
    res[lbl] = row
with open(f"{OUT}/p1_stats.json", "w") as f:
    json.dump(res, f, indent=2, default=str)
print("== P1 stats saved:", f"{OUT}/p1_stats.json")
PYEOF
python3 "${OUTPUT}/p1/stats.py" 2>&1 || echo "  [WARN] stats script failed (check numpy)"

# =============================================================================
# P2: Random mask comparison
# =============================================================================
if [ "$DO_P2" = "1" ]; then
  echo "[P2] Random mask comparison (semantic vs random masks, fixed seed=42)"
  # In --all mode random_mask_dataset.py writes to dirname(--datasets)/datasets_randmask
  RAND_DATA_ROOT="$(dirname "${DATA_DIR}/datasets")/datasets_randmask"
  python3 "${SEMANTIC_SLAM_ROOT}/scripts/random_mask_dataset.py" \
      --all --datasets "${DATA_DIR}/datasets" \
      --detections-root "${DATA_DIR}/detections" \
      --seed 42 || echo "  [WARN] random mask generation failed"

  for lbl in "${LABELS[@]}"; do
    d="${RAND_DATA_ROOT}/TUM/${LABEL_DIR[$lbl]}"
    gt=$(gt_for_seq "$lbl")
    [ -f "$gt" ] || continue; [ -d "$d" ] || continue
    traj="${OUTPUT}/p2/randmask_${lbl}_run1.txt"
    run_slam "$d" "$traj"
    if [ -s "$traj" ]; then
      r=$(ev_rmse "$gt" "$traj")
      echo "    random-mask ${lbl}: ATE=[$r]"
      echo "${lbl} ${r}" >> "${OUTPUT}/p2/randmask_raw.txt"
    fi
  done

  python3 - "${OUTPUT}/p2" <<'PY'
import sys, os, json
out = sys.argv[1]
res = {}
try:
    for line in open(os.path.join(out, "randmask_raw.txt")):
        parts = line.split()
        if len(parts) >= 2:
            lbl, r = parts[0], parts[1]
            res[lbl] = {'random_mask_ate': float(r) if r.lower() != 'null' else None}
except Exception as e:
    res = {'error': str(e)}
with open(os.path.join(out, "p2_compare.json"), "w") as f:
    json.dump(res, f, indent=2)
print("== P2 comparison saved:", os.path.join(out, "p2_compare.json"))
PY
fi

echo ""
echo "===== ALL DONE ====="
echo "  P1:  ${OUTPUT}/p1/p1_stats.json"
echo "  P2:  ${OUTPUT}/p2/p2_compare.json"
echo "  Archive: ${OUTPUT}/p1/run_*.txt, ${OUTPUT}/p2/randmask_*.txt"
echo "  Tip: compare the P1 per-run RMSE and the P2 random mask results"
echo "       with data/experiment_results.json (source of paper Table 1, semantic mask ATE);"
echo "       this supports the supplementary experiments in the paper."