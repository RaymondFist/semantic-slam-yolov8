#!/usr/bin/env bash
# =============================================================================
# prepare_extra_kitti.sh -- fetch + lay out KITTI Odometry sequences for Study 8
# (run_extra_kitti.sh).
# =============================================================================
# Downloads the KITTI odometry grayscale archive and the ground-truth poses
# archive, extracts only the requested sequences, and exposes them under the
# flat layout run_extra_kitti.sh expects:
#
#   $DATA_DIR/datasets/KITTI/<seq>/image_0|image_1|times.txt|calib.txt
#   $DATA_DIR/datasets/KITTI/poses/<seq>.txt
#
# The unpacked archives keep the original KITTI layout under
# $DATA_DIR/datasets/KITTI/data_odometry_gray/dataset/ ; flat symlinks are then
# created for the run script (no file duplication). A sequence whose images are
# already present is skipped, and an archive already on disk is never
# re-downloaded (the grayscale archive is ~22 GB). With GRAY_MODE=stream the
# archive is never written to disk at all: the request is piped straight into
# bsdtar and only the requested sequence members are extracted.
#
# Usage:
#   DATA_DIR=<data> bash prepare_extra_kitti.sh
#   KITTI_SEQS="01 05" DATA_DIR=<data> bash prepare_extra_kitti.sh
#   GRAY_MODE=stream KITTI_SEQS="01 04 07" DATA_DIR=<data> bash prepare_extra_kitti.sh
#   RUN=1 N=5 DATA_DIR=<data> bash prepare_extra_kitti.sh    # fetch + lay out + run
#
# Env: DATA_DIR (required), KITTI_SEQS (default "01 04 05 07 08 09 10"),
#      GRAY_MODE (zip|stream, default zip), GRAY_URL / POSES_URL (override),
#      RUN (0|1, default 0), N (default 5).
# -----------------------------------------------------------------------------
set -uo pipefail

: "${DATA_DIR:?DATA_DIR not set (data root containing datasets/)}"
KITTI_SEQS="${KITTI_SEQS:-01 04 05 07 08 09 10}"
GRAY_URL="${GRAY_URL:-https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip}"
POSES_URL="${POSES_URL:-https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip}"
RUN="${RUN:-0}"
N="${N:-5}"
GRAY_MODE="${GRAY_MODE:-zip}"    # zip: keep the ~22 GB archive on disk | stream: curl|bsdtar, no archive stored

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMANTIC_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
export SEMANTIC_ROOT DATA_DIR

KROOT="$DATA_DIR/datasets/KITTI"          # where the run script looks
UNPACK="$KROOT/data_odometry_gray"        # original KITTI layout (unpacked)
DL="$KROOT/downloads"
mkdir -p "$UNPACK" "$DL"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ "$GRAY_MODE" = "stream" ]; then
    command -v bsdtar >/dev/null 2>&1 || { log "[ERROR] bsdtar not found (install libarchive-tools)"; exit 1; }
    command -v curl   >/dev/null 2>&1 || { log "[ERROR] curl not found"; exit 1; }
fi

fetch() {   # <url> <dest> -- skip if already downloaded; download via a .part file
    local url="$1" dest="$2" tmp="$2.part"
    if [ -s "$dest" ]; then log "skip download (exists): $dest"; return 0; fi
    log "downloading $url"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$tmp" "$url" || { log "[ERROR] wget failed"; return 1; }
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "$tmp" "$url" || { log "[ERROR] curl failed"; return 1; }
    else
        log "[ERROR] need wget or curl"; return 1
    fi
    [ -s "$tmp" ] && mv -f "$tmp" "$dest" || { log "[ERROR] download produced no file"; return 1; }
    log "saved: $dest"
}

# Ground-truth poses: fetch the poses archive only if some requested .txt is absent.
need_poses=0
for s in $KITTI_SEQS; do [ -f "$UNPACK/dataset/poses/$s.txt" ] || need_poses=1; done
if [ "$need_poses" = "1" ]; then
    command -v unzip >/dev/null 2>&1 || { log "[ERROR] unzip not found (needed to extract the poses archive)"; exit 1; }
    fetch "$POSES_URL" "$DL/data_odometry_poses.zip"
    for s in $KITTI_SEQS; do
        if [ -f "$UNPACK/dataset/poses/$s.txt" ]; then
            log "skip extract: GT poses $s already present"
        else
            unzip -o -q "$DL/data_odometry_poses.zip" "dataset/poses/$s.txt" -d "$UNPACK" \
                || log "[WARN] GT poses extract failed for $s"
        fi
    done
else
    log "skip poses archive: all requested GT poses already present"
fi

# Grayscale images.
if [ "$GRAY_MODE" = "stream" ]; then
    missing=""
    for s in $KITTI_SEQS; do
        if [ -d "$UNPACK/dataset/sequences/$s/image_0" ] && [ -d "$UNPACK/dataset/sequences/$s/image_1" ]; then
            log "skip stream: sequence $s already present"
        else
            missing="$missing dataset/sequences/$s/*"
        fi
    done
    if [ -n "$missing" ]; then
        log "streaming grayscale archive (no archive written to disk) for:$missing"
        set -f    # hand the globs to bsdtar literally; never let the shell expand them
        curl -sL "$GRAY_URL" | bsdtar -xf - -C "$UNPACK" $missing \
            || log "[WARN] stream extract failed; re-run with GRAY_MODE=zip"
        set +f
    fi
else
    fetch "$GRAY_URL" "$DL/data_odometry_gray.zip"
    for s in $KITTI_SEQS; do
        seqdir="$UNPACK/dataset/sequences/$s"
        if [ -d "$seqdir/image_0" ] && [ -d "$seqdir/image_1" ]; then
            log "skip extract: sequence $s already present"
        else
            log "extracting sequence $s from grayscale archive"
            unzip -o -q "$DL/data_odometry_gray.zip" \
                "dataset/sequences/$s/image_0/*" "dataset/sequences/$s/image_1/*" \
                "dataset/sequences/$s/times.txt" "dataset/sequences/$s/calib.txt" \
                -d "$UNPACK" || log "[WARN] extract failed for sequence $s"
        fi
    done
fi

# Flat symlinks for run_extra_kitti.sh (it expects datasets/KITTI/<seq> and
# datasets/KITTI/poses/<seq>.txt). Guarded so an existing real directory is
# never turned into a nested link, and a pre-existing poses/ is left as is.
for s in $KITTI_SEQS; do
    if [ -L "$KROOT/$s" ] || [ ! -e "$KROOT/$s" ]; then
        ln -sfn "$UNPACK/dataset/sequences/$s" "$KROOT/$s"
    else
        log "[NOTE] $KROOT/$s already exists (not a symlink); left untouched"
    fi
done
if [ -d "$UNPACK/dataset/poses" ]; then
    if [ -e "$KROOT/poses" ] || [ -L "$KROOT/poses" ]; then
        log "[NOTE] $KROOT/poses already present; left untouched"
    else
        ln -sfn "$UNPACK/dataset/poses" "$KROOT/poses"
    fi
fi
log "flat layout ready: $KROOT/<seq> -> unpacked sequences; poses available"

# Light sanity check: frame count vs ground-truth pose count per sequence.
for s in $KITTI_SEQS; do
    n0="$(find "$KROOT/$s/image_0" -type f 2>/dev/null | wc -l)"
    np="$(wc -l < "$KROOT/poses/$s.txt" 2>/dev/null || echo 0)"
    log "check $s: image_0=$n0  gt_poses=$np"
done

if [ "$RUN" = "1" ]; then
    log "running Study 8 (KITTI_SEQS=$KITTI_SEQS, N=$N)"
    KITTI_SEQS="$KITTI_SEQS" N="$N" bash "$EXP_DIR/run_extra_kitti.sh"
else
    log "fetch/layout only. To run Study 8:"
    log "  KITTI_SEQS=\"$KITTI_SEQS\" N=$N KITTI_TIMEOUT=1800 bash $EXP_DIR/run_extra_kitti.sh"
fi
