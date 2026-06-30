#!/bin/bash
# =============================================================================
# Build Script for SemanticSLAM-YOLOv8
# =============================================================================
# Prerequisites:
#   1. ORB-SLAM3 installed at $ORB_SLAM3_ROOT (or set below)
#   2. OpenCV 4.5+ with CUDA DNN support
#   3. Eigen 3.3+, Pangolin, DBoW2, g2o
#   4. TensorRT 8.6+ (optional, for optimized inference)
#   5. CUDA 11.8+ toolkit
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"

# ---- Configuration ----
export ORB_SLAM3_ROOT="${ORB_SLAM3_ROOT:-/root/ORB_SLAM3}"
YOLO_MODEL_DIR="${SCRIPT_DIR}/../../models"
# -----------------------

echo "==> SemanticSLAM-YOLOv8 Build Script"
echo "    ORB_SLAM3 root: ${ORB_SLAM3_ROOT}"
echo "    Build dir:      ${BUILD_DIR}"

mkdir -p "${YOLO_MODEL_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DORB_SLAM3_ROOT="${ORB_SLAM3_ROOT}" \
    -DUSE_TENSORRT=OFF \
    -DUSE_OPENCV_DNN=ON \
    -DENABLE_BENCHMARKS=ON

make -j$(nproc) 2>&1 | tail -20

echo ""
echo "==> Build complete."
echo "    Libraries:  ${BUILD_DIR}/libSemanticSLAM.so"
echo "    Benchmark:  ${BUILD_DIR}/semantic_slam_benchmark"

echo ""
echo "    Usage: ${BUILD_DIR}/semantic_slam_benchmark <mode> [args...]"
echo "      mode: kitti | euroc | tum | ablation"

# ---- Optional: copy to root ----
cp "${BUILD_DIR}/libSemanticSLAM.so" "${SCRIPT_DIR}/../" 2>/dev/null || true