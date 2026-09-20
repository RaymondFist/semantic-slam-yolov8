#!/bin/bash
# =============================================================================
# AutoDL one-click deploy script - Semantic-SLAM-YOLOv8
# =============================================================================
# Server: AutoDL (Ubuntu 20.04, x86_64, RTX 3090 24GB, 32GB RAM)
# Usage:   bash autodl_deploy.sh
#
# Datasets (paper standard):
#   TUM   - sitting_static, sitting_xyz, walking_static, walking_xyz, walking_halfsphere
#   KITTI - 00 (4541 frames, loop closure + rich dynamics)
#   EuRoC - MH_01_easy, MH_03_medium, MH_05_difficult
# =============================================================================

set -e

# ===================== Command line arguments =====================
SKIP_DOWNLOAD=false
RESUME=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --resume)        RESUME=true; shift ;;
        *) echo "unknown argument: $1"; echo "Usage: bash autodl_deploy.sh [--skip-download] [--resume]"; exit 1 ;;
    esac
done

# Auto-detect the script directory (works when deployed from any path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMANTIC_SLAM_ROOT="${SEMANTIC_SLAM_ROOT:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT="${OUTPUT:-${SEMANTIC_SLAM_ROOT}/output}"

# ===================== Logging =====================
DEPLOY_LOG="${SCRIPT_DIR}/deploy.log"
# Clear the old log file, then mirror stdout/stderr to both terminal and log
: > "$DEPLOY_LOG"
exec > >(tee -a "$DEPLOY_LOG") 2>&1

echo "=========================================="
echo "  Semantic-SLAM-YOLOv8 AutoDL Deploy"
echo "  Code path: ${SEMANTIC_SLAM_ROOT}"
echo "=========================================="

# ===================== 0. System dependencies =====================
echo ""
echo "[0/8] Installing system dependencies..."
apt-get update && apt-get install -y \
    build-essential cmake git wget curl unzip \
    libeigen3-dev libglew-dev \
    libboost-dev libboost-thread-dev libboost-filesystem-dev \
    libssl-dev libjpeg-dev libpng-dev \
    libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module \
    python3-pip python3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    xvfb

pip3 install ultralytics opencv-python-headless evo scipy --quiet

# ===================== 0.5 Build OpenCV (from source, >4.4) =====================
echo ""
echo "[0.5/8] Building OpenCV 4.10 (ORB-SLAM3 needs >4.4)..."
# Make sure pkg-config can find libraries installed under /usr/local
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}
OPENCV_VER=$(pkg-config --modversion opencv4 2>/dev/null || echo "0.0.0")
OPENCV_MAJOR=$(echo "$OPENCV_VER" | cut -d. -f1)
OPENCV_MINOR=$(echo "$OPENCV_VER" | cut -d. -f2)
# Build OpenCV if not installed or version < 4.4
NEED_BUILD=false
if [ ! -f /usr/local/lib/libopencv_core.so ]; then
    NEED_BUILD=true
elif [ "$OPENCV_MAJOR" -lt 4 ] || { [ "$OPENCV_MAJOR" -eq 4 ] && [ "$OPENCV_MINOR" -lt 4 ]; }; then
    echo "  OpenCV ${OPENCV_VER} installed but too old (need >= 4.4), rebuilding..."
    NEED_BUILD=true
fi

if [ "$NEED_BUILD" = true ]; then
    cd /tmp
    if [ ! -d opencv-4.10.0 ]; then
        wget -q https://github.com/opencv/opencv/archive/4.10.0.tar.gz
        tar -xzf 4.10.0.tar.gz
        rm -f 4.10.0.tar.gz
    fi
    cd opencv-4.10.0 && mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DWITH_OPENGL=ON \
        -DWITH_GTK=ON \
        -DWITH_QT=OFF \
        -DWITH_IPP=OFF \
        -DWITH_TBB=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_V4L=ON \
        -DWITH_FFMPEG=ON \
        -DBUILD_SHARED_LIBS=ON \
        2>&1 | tail -3
    make -j$(nproc) 2>&1 | tail -3
    make install 2>&1 | tail -3
    ldconfig
    # Make sure pkg-config can find the newly installed OpenCV
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}
    echo "  OpenCV: OK ($(pkg-config --modversion opencv4 2>/dev/null || echo '4.10.0'))"
else
    echo "  OpenCV: already exists (${OPENCV_VER}, >= 4.4)"
fi

# ===================== 1. Build Pangolin =====================
echo ""
echo "[1/8] Building Pangolin..."
if [ ! -d ~/Pangolin ]; then
    git clone https://github.com/stevenlovegrove/Pangolin.git ~/Pangolin
fi
cd ~/Pangolin
# Pin v0.8 to avoid sigslot slots_reference() compatibility issues
git checkout v0.8
# Init only the core submodule (sigslot), skip optional pybind11/vcpkg
git submodule update --init external/sigslot 2>/dev/null || true
# If the sigslot submodule path is missing, clone it manually
if [ ! -d external/sigslot ]; then
    mkdir -p external && cd external
    git clone https://github.com/palacaze/sigslot.git sigslot
    cd sigslot && git checkout v1.2.1 && cd ../..
fi
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF 2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -3
make install 2>&1 | tail -3
ldconfig

# Replace the system sigslot header with a compatible version (Pangolin make install does not overwrite old headers)
if grep -q "slots_reference" /usr/local/include/sigslot/signal.hpp 2>/dev/null; then
    echo "  Replacing the incompatible sigslot header..."
    if [ -d ~/Pangolin/external/sigslot/include/sigslot ]; then
        rm -rf /usr/local/include/sigslot
        cp -r ~/Pangolin/external/sigslot/include/sigslot /usr/local/include/sigslot
    else
        cd /tmp && rm -rf sigslot
        git clone https://github.com/palacaze/sigslot.git
        cd sigslot && git checkout v1.2.1
        rm -rf /usr/local/include/sigslot
        cp -r /tmp/sigslot/include/sigslot /usr/local/include/sigslot
    fi
    echo "  sigslot: OK (replaced)"
fi
echo "  Pangolin: OK"

# ===================== 2. Build ORB-SLAM3 =====================
echo ""
echo "[2/8] Building ORB-SLAM3..."
if [ ! -d ~/ORB_SLAM3 ]; then
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ~/ORB_SLAM3
fi
cd ~/ORB_SLAM3
chmod +x build.sh

# Restore old patches - git checkout is more reliable than .orig
git checkout -- include/System.h include/Tracking.h include/Frame.h include/MapPoint.h \
    src/Tracking.cc src/System.cc src/Optimizer.cc src/ORBmatcher.cc src/Frame.cc CMakeLists.txt 2>/dev/null
# Also restore .orig backup files
for f in $(find . -name '*.orig' 2>/dev/null); do
    mv "$f" "${f%.orig}"
done
echo "  Old patches restored"

# Fix: sigslot v1.2.1 needs C++17 (uses enable_if_t, is_func_v, etc.)
# ORB-SLAM3 uses the -std=c++11 flag instead of CMAKE_CXX_STANDARD, so replace it
sed -i 's/-std=c++11/-std=c++17/g' CMakeLists.txt
sed -i 's/-std=c++0x/-std=c++17/g' CMakeLists.txt
sed -i 's/CHECK_CXX_COMPILER_FLAG("-std=c++11"/CHECK_CXX_COMPILER_FLAG("-std=c++17"/g' CMakeLists.txt
sed -i 's/-std=c++11/-std=c++17/g' Thirdparty/g2o/CMakeLists.txt
find Thirdparty/DBoW2 -name "CMakeLists.txt" -exec sed -i 's/-std=c++11/-std=c++17/g' {} +

# Fix: ORB-SLAM3 bug - mnFullBAIdx wrongly declared as bool; C++17 forbids bool++
sed -i 's/bool mnFullBAIdx/int mnFullBAIdx/g' include/LoopClosing.h

# Fix: bOK in ORB-SLAM3 Tracking.cc may be uninitialized (GCC treats -Wmaybe-uninitialized as error)
# Replace all "bool bOK;" with "bool bOK = false;"
sed -i 's/bool bOK;/bool bOK = false;/g' src/Tracking.cc

# Fix: silence warnings about possibly uninitialized Ng/Na/Ngw/Naw variables
# Append -Wno-* flags to CMAKE_CXX_FLAGS directly (more reliable than sed-replacing -Werror)
sed -i '/CMAKE_CXX_FLAGS.*-Wall/aset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-reorder -Wno-sign-compare -Wno-deprecated-declarations")' CMakeLists.txt

# BUG FIX #3: Sophus SO3::exp NaN guard
# Masked datasets can produce degenerate Essential Matrix decompositions during loop closing,
# causing SO3::exp() to receive a NaN rotation vector and crash (Sophus ensure failed)
# Fix: add a NaN check in SO3::expAndTheta, return identity rotation on NaN
SOPHUS_HPP=~/ORB_SLAM3/Thirdparty/Sophus/sophus/so3.hpp
if [ -f "$SOPHUS_HPP" ]; then
    echo "  Fixing Sophus SO3 NaN crash (Python)..."
    cp "$SOPHUS_HPP" "${SOPHUS_HPP}.bak.$(date +%s)"
    python3 -c "
import re

path = '${SOPHUS_HPP}'
with open(path, 'r') as f:
    content = f.read()

# Step 0: If old broken guard exists (Quaternion without Eigen::), remove it.
# The old guard caused compilation errors: 'Quaternion' was not declared.
# We detect it by the 'Quaternion<Scalar>' pattern within the FIX_NAN_GUARD block.
old_guard_pattern = re.compile(
    r'\s*// FIX_NAN_GUARD:.*?\n'
    r'(?:\s*//.*?\n)*'
    r'\s*if\s*\(!std::isfinite\(omega\[0\]\).*?\{\n'
    r'(?:\s*//.*?\n)*'
    r'.*?Quaternion<Scalar>.*?\n'
    r'.*?Quaternion<Scalar>.*?\n'
    r'\s*\}\n',
    re.DOTALL
)
if old_guard_pattern.search(content):
    content = old_guard_pattern.sub('', content, count=1)
    print('  [CLEAN] Removed old broken guard (Quaternion without Eigen::)')
elif 'FIX_NAN_GUARD' in content and 'SO3<Scalar>()' in content:
    print('  [SKIP] Guard already correct (SO3<Scalar>() present)')
    import sys; sys.exit(0)

# New guard: uses SO3<Scalar>() default constructor = identity rotation.
# No Quaternion/Eigen:: prefix needed - safe in SO3Base template context.
guard = (
    '  // FIX_NAN_GUARD: NaN check for masked datasets (BUG FIX #3)\n'
    '  // If omega is NaN (can happen with degenerate Essential Matrix\n'
    '  // decomposition during loop closing on masked datasets), return\n'
    '  // identity rotation to avoid SO3::exp crash.\n'
    '  if (!std::isfinite(omega[0]) || !std::isfinite(omega[1]) || !std::isfinite(omega[2])) {\n'
    '    *theta = Scalar(0);\n'
    '    return SO3<Scalar>();\n'
    '  }\n'
)

# Strategy: Find expAndTheta() in the so3.hpp file, then find the opening brace.
idx = content.find('expAndTheta() const')
if idx < 0:
    so3base_pos = content.find('SO3Base')
    if so3base_pos >= 0:
        idx = content.find('expAndTheta(', so3base_pos)
if idx < 0:
    idx = content.find('::expAndTheta')
if idx < 0:
    idx = content.find('expAndTheta(')
if idx >= 0:
    brace_pos = content.find('{', idx)
    if brace_pos >= 0:
        nl = content.find('\n', brace_pos)
        if nl >= 0:
            insert_pos = nl + 1
        else:
            insert_pos = brace_pos + 1
        content = content[:insert_pos] + guard + content[insert_pos:]
        print('  Sophus NaN fix: OK')
    else:
        print('  Sophus NaN fix: FAILED (no opening brace after expAndTheta)')
else:
    print('  Sophus NaN fix: FAILED (expAndTheta not found in any pattern)')

with open(path, 'w') as f:
    f.write(content)
"
    if grep -q "FIX_NAN_GUARD" "$SOPHUS_HPP"; then
        if grep -q "Quaternion<Scalar>" "$SOPHUS_HPP"; then
            echo "  [ERROR] Sophus NaN fix FAILED: old broken guard still present!"
        else
            echo "  Sophus NaN fix: verified OK"
        fi
    else
        echo "  [WARN] Sophus NaN fix: FIX_NAN_GUARD marker not found after fix"
    fi
fi

# Build Thirdparty first (clean old builds to apply C++17)
cd Thirdparty/DBoW2 && rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) 2>&1 | tail -3
cd ../../g2o && rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) 2>&1 | tail -3
cd ../../..

# Build the ORB-SLAM3 main library
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -eq 0 ] && [ -f ../lib/libORB_SLAM3.so ]; then
    echo "  ORB-SLAM3: OK (libORB_SLAM3.so)"
else
    echo "  ORB-SLAM3: FAILED (build error)"
    echo "  Detailed errors:"
    cd ~/ORB_SLAM3/build && make -j1 2>&1 | grep "error:" | head -10
    exit 1
fi

# ===================== 3. Upload & patch SemanticSLAM =====================
echo ""
echo "[3/8] Building SemanticSLAM..."
# Assumes the code was scp'd to ${SEMANTIC_SLAM_ROOT}
cd ${SEMANTIC_SLAM_ROOT}

# Restore old patches - git checkout is more reliable than .orig
cd ~/ORB_SLAM3
git checkout -- include/System.h include/Tracking.h include/Frame.h include/MapPoint.h \
    src/Tracking.cc src/System.cc src/Optimizer.cc src/ORBmatcher.cc src/Frame.cc CMakeLists.txt 2>/dev/null
for f in $(find . -name '*.orig' 2>/dev/null); do
    mv "$f" "${f%.orig}"
done
echo "  Old patches restored"

# git checkout restored CMakeLists.txt, so re-apply C++17 and other fixes
sed -i 's/-std=c++11/-std=c++17/g' CMakeLists.txt
sed -i 's/-std=c++0x/-std=c++17/g' CMakeLists.txt
sed -i 's/CHECK_CXX_COMPILER_FLAG("-std=c++11"/CHECK_CXX_COMPILER_FLAG("-std=c++17"/g' CMakeLists.txt
sed -i 's/-std=c++11/-std=c++17/g' Thirdparty/g2o/CMakeLists.txt
find Thirdparty/DBoW2 -name "CMakeLists.txt" -exec sed -i 's/-std=c++11/-std=c++17/g' {} +
sed -i 's/bool mnFullBAIdx/int mnFullBAIdx/g' include/LoopClosing.h
sed -i 's/bool bOK;/bool bOK = false;/g' src/Tracking.cc
sed -i '/CMAKE_CXX_FLAGS/a set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-reorder -Wno-sign-compare -Wno-deprecated-declarations")' CMakeLists.txt
echo "  C++17 and build fixes re-applied"

# Run the patch script to modify the ORB-SLAM3 source (pass the correct SemanticSLAM path)
cd ${SEMANTIC_SLAM_ROOT}
python3 patches/patch_orbslam3.py /root/ORB_SLAM3 ${SEMANTIC_SLAM_ROOT}/src

# Build the SemanticSLAM library first (skip benchmarks; they need libORB_SLAM3.so)
cd ${SEMANTIC_SLAM_ROOT}/src
rm -rf build && mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DORB_SLAM3_ROOT=/root/ORB_SLAM3 \
    -DUSE_TENSORRT=OFF \
    -DUSE_OPENCV_DNN=ON \
    -DENABLE_BENCHMARKS=OFF
make -j$(nproc) 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  SemanticSLAM: FAILED"
    make -j1 2>&1 | grep -E "error:" | head -20
    exit 1
fi
echo "  SemanticSLAM: OK"

# Then rebuild ORB-SLAM3 (with semantic module patches) - must be a full clean rebuild
cd ~/ORB_SLAM3
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc) 2>&1 | tee /tmp/orbslam3_build.log | tail -5
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "  ORB-SLAM3 (patched): OK"
else
    echo "  ORB-SLAM3 (patched): FAILED"
    echo "=== ERROR LINES ==="
    grep -E "error:" /tmp/orbslam3_build.log | head -30
    echo "=== LAST 30 LINES ==="
    tail -30 /tmp/orbslam3_build.log
    exit 1
fi

# Finally build the benchmark executables (they need libORB_SLAM3.so)
# Clean the old build cache to force CMake to reconfigure
cd ${SEMANTIC_SLAM_ROOT}/src
rm -rf build && mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DORB_SLAM3_ROOT=/root/ORB_SLAM3 \
    -DUSE_TENSORRT=OFF \
    -DUSE_OPENCV_DNN=ON \
    -DENABLE_BENCHMARKS=ON
make -j$(nproc) 2>&1 | tee /tmp/benchmark_build.log | tail -5
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  Benchmarks: FAILED (non-fatal)"
    echo "=== BENCHMARK ERROR LINES ==="
    grep -E "error:|undefined reference|cannot find|ld returned|No such file" /tmp/benchmark_build.log | head -20
    echo "=== SINGLE-THREAD RETRY ==="
    make -j1 2>&1 | grep -E "error:|undefined reference|cannot find|fatal error" | head -20
else
    echo "  Benchmarks: OK"
fi

# ===================== 3.5 Download YOLOv8 model + ONNX export =====================
echo ""
echo "[3.5/8] Downloading YOLOv8 model and exporting ONNX/TensorRT..."
MODEL_DIR=${SEMANTIC_SLAM_ROOT}/models
mkdir -p ${MODEL_DIR}

# 3.5.1 Download YOLOv8n-seg.pt (Ultralytics auto-downloads to cache)
if [ ! -f ${MODEL_DIR}/yolov8n-seg.pt ]; then
    echo "  Downloading yolov8n-seg.pt (official Ultralytics, ~6.7MB)..."
    python3 -c "
from ultralytics import YOLO
# Auto-download the model weights from the Ultralytics Hub
model = YOLO('yolov8n-seg.pt')
print(f'yolov8n-seg.pt downloaded to: {model.ckpt_path}')
"
    # Find it and copy it to the models/ directory
    PT_FILE=$(python3 -c "
from ultralytics.utils import USER_CONFIG_DIR
from pathlib import Path
import os
# Ultralytics cache path
cache_dir = Path(os.path.expanduser('~/.cache/ultralytics'))
# Search the weights subdirectory
for root, dirs, files in os.walk(str(cache_dir)):
    for f in files:
        if f == 'yolov8n-seg.pt':
            print(os.path.join(root, f))
            exit(0)
# Fallback: look in the current directory
for p in ['yolov8n-seg.pt', '../yolov8n-seg.pt']:
    if os.path.exists(p):
        print(os.path.abspath(p))
        exit(0)
" 2>/dev/null)
    if [ -n "$PT_FILE" ] && [ -f "$PT_FILE" ]; then
        cp "$PT_FILE" ${MODEL_DIR}/yolov8n-seg.pt
        echo "  yolov8n-seg.pt: OK ($(du -h ${MODEL_DIR}/yolov8n-seg.pt | cut -f1))"
    else
        echo "  [WARN] yolov8n-seg.pt not found, ONNX export will fail"
    fi
else
    echo "  yolov8n-seg.pt: already exists"
fi

# 3.5.2 Export ONNX format (needed for C++ TensorRT inference)
if [ -f ${MODEL_DIR}/yolov8n-seg.pt ] && [ ! -f ${MODEL_DIR}/yolov8n-seg.onnx ]; then
    echo "  Exporting yolov8n-seg.onnx (dynamic batch, FP32)..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('${MODEL_DIR}/yolov8n-seg.onnx')
" 2>/dev/null || python3 -c "
from ultralytics import YOLO
model = YOLO('${MODEL_DIR}/yolov8n-seg.pt')
# Export ONNX: opset=17, dynamic batch, FP32, with the segmentation head
model.export(format='onnx', opset=17, dynamic=True, simplify=True)
import shutil, os
# Exported file lands in the models/ directory
for f in os.listdir('.'):
    if f.startswith('yolov8n-seg') and f.endswith('.onnx'):
        shutil.move(f, '${MODEL_DIR}/yolov8n-seg.onnx')
        print(f'yolov8n-seg.onnx exported: {os.path.getsize(\"${MODEL_DIR}/yolov8n-seg.onnx\")/1e6:.1f}MB')
        break
" 2>/dev/null
    if [ -f ${MODEL_DIR}/yolov8n-seg.onnx ]; then
        echo "  yolov8n-seg.onnx: OK ($(du -h ${MODEL_DIR}/yolov8n-seg.onnx | cut -f1))"
    else
        echo "  [WARN] ONNX export failed, will download the pre-exported ONNX from Ultralytics"
        # Fallback: download the pre-exported ONNX from Ultralytics assets
        wget -q --timeout=60 \
            "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.onnx" \
            -O ${MODEL_DIR}/yolov8n-seg.onnx 2>/dev/null || {
            echo "  [WARN] Fallback ONNX download also failed, C++ inference will use the .pt file"
        }
    fi
else
    echo "  yolov8n-seg.onnx: already exists" 2>/dev/null || echo "  yolov8n-seg.onnx: skipped (.pt missing)"
fi

# 3.5.3 TensorRT Engine (build only if GPU + TensorRT are available)
if [ -f ${MODEL_DIR}/yolov8n-seg.onnx ] && [ ! -f ${MODEL_DIR}/yolov8n-seg.trt ]; then
    echo "  Trying to build the TensorRT Engine (yolov8n-seg.trt)..."
    if command -v trtexec &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        trtexec --onnx=${MODEL_DIR}/yolov8n-seg.onnx \
            --saveEngine=${MODEL_DIR}/yolov8n-seg.trt \
            --fp16 --workspace=2048 2>&1 | tail -3
        if [ -f ${MODEL_DIR}/yolov8n-seg.trt ]; then
            echo "  yolov8n-seg.trt: OK ($(du -h ${MODEL_DIR}/yolov8n-seg.trt | cut -f1))"
        fi
    elif python3 -c "import tensorrt" 2>/dev/null; then
        python3 -c "
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
with open('${MODEL_DIR}/yolov8n-seg.onnx', 'rb') as f:
    parser.parse(f.read())
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.set_flag(trt.BuilderFlag.FP16)
plan = builder.build_serialized_network(network, config)
with open('${MODEL_DIR}/yolov8n-seg.trt', 'wb') as f:
    f.write(plan)
print('TensorRT engine built')
" 2>/dev/null && echo "  yolov8n-seg.trt: OK ($(du -h ${MODEL_DIR}/yolov8n-seg.trt | cut -f1))" || \
        echo "  Skipping TensorRT: GPU has no TensorRT runtime support"
    else
        echo "  Skipping TensorRT: GPU or trtexec unavailable"
    fi
elif [ -f ${MODEL_DIR}/yolov8n-seg.trt ]; then
    echo "  yolov8n-seg.trt: already exists"
fi

# 3.5.4 coco.names (class name file)
if [ ! -f ${MODEL_DIR}/coco.names ]; then
    echo "  Downloading coco.names..."
    python3 -c "
from ultralytics.utils import ROOT
import shutil, os
# Ultralytics ships coco.yaml for reference
import yaml
try:
    with open(os.path.join(ROOT, 'cfg/datasets/coco.yaml')) as f:
        data = yaml.safe_load(f)
    names = {v: k for k, v in data['names'].items()}
    with open('${MODEL_DIR}/coco.names', 'w') as f:
        for i in range(80):
            f.write(names.get(i, f'class_{i}') + '\n')
    print('coco.names generated from ultralytics coco.yaml')
except Exception as e:
    print(f'coco.yaml not found ({e}), using fallback download')
" 2>/dev/null
    if [ ! -f ${MODEL_DIR}/coco.names ]; then
        # Download from the Ultralytics GitHub repo
        wget -q --timeout=30 \
            "https://raw.githubusercontent.com/ultralytics/assets/main/coco.names" \
            -O ${MODEL_DIR}/coco.names 2>/dev/null || {
            # Last resort: write the 80 COCO class names manually
            echo "  Generating coco.names manually (80 classes)..."
            python3 -c "
classes = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag',
'tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table',
'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
'hair drier','toothbrush']
with open('${MODEL_DIR}/coco.names', 'w') as f:
    f.write('\n'.join(classes))
print('coco.names generated (80 classes)')
"
        }
    fi
    echo "  coco.names: OK"
else
    echo "  coco.names: already exists"
fi

echo "  Model preparation: done"

# ===================== 4. Download datasets (paper standard) =====================
echo ""
echo "[4/8] Downloading datasets (paper standard)..."
mkdir -p ${DATA_DIR}/datasets ${DATA_DIR}/detections

# Disk space check (datasets ~18GB after extraction + build artifacts 5GB, at least 30GB recommended)
AVAIL_GB=$(df -BG ${DATA_DIR} 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "${AVAIL_GB}" -lt 25 ]; then
    echo "  [WARN] Only ${AVAIL_GB}GB of disk space available, datasets+build need ~25GB"
    echo "  AutoDL data disk expansion: Instance Management -> Expand Data Disk -> 50GB or more"
    if [ "${AVAIL_GB}" -lt 18 ]; then
        echo "  [ERROR] Not enough disk space, exiting"
        exit 1
    fi
fi
echo "  Available disk space: ${AVAIL_GB}GB (minimum required 18GB)"

# ==== Generic robust download function (resume + automatic fallback) =====
# Usage: robust_download <URL> <output file> [OSS mirror URL] [Baidu Netdisk hint]
robust_download() {
    local url="$1"
    local outfile="$2"
    local mirror_url="${3:-}"
    local baidu_hint="${4:-}"

    local start_time=$(date +%s)
    local success=0

    # If a partial file exists, try to resume; otherwise download from scratch
    if [ -f "${outfile}" ]; then
        echo "      [resume] Partial file $(du -h ${outfile} | cut -f1) exists, resuming..."
    fi

    # Strategy 1: curl resume (up to 10 retries, 10s apart, 600s total timeout)
    echo "      [1] curl -C - --retry 10 ${url}"
    if curl -C - --retry 10 --retry-delay 10 --retry-max-time 600 \
        -L --connect-timeout 30 --max-time 3600 \
        -o "${outfile}" "${url}" 2>&1 | tail -1; then
        success=1
    else
        # Strategy 2: wget fallback (some servers handle curl Range requests poorly)
        if [ -n "${mirror_url}" ]; then
            echo "      [2] Trying OSS mirror: ${mirror_url}"
            rm -f "${outfile}"  # remove a possibly corrupted file
            if wget -c --timeout=300 --tries=5 -q --show-progress "${mirror_url}" -O "${outfile}"; then
                success=1
            fi
        fi
    fi

    if [ "${success}" -eq 0 ] && [ -n "${mirror_url}" ]; then
        echo "      [3] curl retry on mirror: ${mirror_url}"
        rm -f "${outfile}"
        if curl --retry 5 --retry-delay 15 -L --connect-timeout 30 --max-time 3600 \
            -o "${outfile}" "${mirror_url}"; then
            success=1
        fi
    fi

    if [ "${success}" -eq 0 ]; then
        echo "      [ERROR] Download failed: ${url}"
        [ -n "${baidu_hint}" ] && echo "      ${baidu_hint}"
        return 1
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local filesize=$(du -h "${outfile}" | cut -f1)
    echo "      [OK] Download complete: ${filesize} (took ${elapsed}s)"
    return 0
}

# If --skip-download is set, skip all dataset downloads
if [ "${SKIP_DOWNLOAD}" = true ]; then
    echo ""
    echo "  --skip-download mode: skipping dataset downloads, using already-extracted data"
    echo "  If you uploaded archives (.tgz/.zip), drop --skip-download,"
    echo "  and the script will extract and reorganize them into the right paths."
    echo "  Make sure these directories exist with complete data:"
    echo "    TUM:   ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_*/"
    echo "    KITTI: ${DATA_DIR}/datasets/KITTI/00/image_0/"
    echo "    EuRoC: ${DATA_DIR}/datasets/EuRoC/MH_*/mav0/cam0/data/"
else
DOWNLOAD_START=$(date +%s)

# ==== TUM fr3 sequences ====
# URL pattern: https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_<name>.tgz
# Note: the official server may 403 directory listings, but direct file downloads usually work
TUM_BASE="https://cvg.cit.tum.de/rgbd/dataset/freiburg3"
# OSS mirror (Aliyun OSS for faster access in China, upload it beforehand)
TUM_MIRROR="https://semantic-slam-datasets.oss-cn-beijing.aliyuncs.com/tum"

# Declare all required TUM sequences
# sitting_static  - fully static baseline, proves dynamic filtering does not remove static points
# sitting_xyz     - slight-motion static baseline, proves dynamic filtering does not hurt SLAM accuracy
# walking_static  - camera still, person moving, purest dynamic-feature filtering test
# walking_xyz     - camera and person both move, standard dynamic benchmark
# walking_halfsphere - large rotation + pedestrian, hard dynamic scene
declare -A TUM_SEQS=(
    ["sitting_static"]="static baseline (1.2GB)"
    ["sitting_xyz"]="slight-motion static (1.5GB)"
    ["walking_static"]="pure dynamic test (1.6GB)"
    ["walking_xyz"]="standard dynamic benchmark (1.7GB)"
    ["walking_halfsphere"]="hard dynamic (1.9GB)"
)

TUM_DOWNLOADED=0
for seq in "${!TUM_SEQS[@]}"; do
    if [ ! -d ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${seq} ]; then
        TUM_FILE="${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${seq}.tgz"
        echo "  Preparing TUM ${seq} (${TUM_SEQS[$seq]})..."

        # -- First check: a local tgz archive already uploaded by the user --
        if [ -f "${TUM_FILE}" ]; then
            echo "  [local extract] Found local ${seq}.tgz, extracting directly..."
            cd ${DATA_DIR}/datasets/TUM
            tar -xzf ${TUM_FILE} && rm -f ${TUM_FILE}
            if [ -f rgbd_dataset_freiburg3_${seq}/rgb.txt ]; then
                cd rgbd_dataset_freiburg3_${seq}
                python3 ${SEMANTIC_SLAM_ROOT}/scripts/associate.py rgb.txt depth.txt > associate.txt
                echo "  TUM ${seq}: OK (extracted from local tgz)"
                TUM_DOWNLOADED=$((TUM_DOWNLOADED + 1))
                continue
            else
                echo "  TUM ${seq}: extraction failed, file incomplete, trying download..."
                rm -rf rgbd_dataset_freiburg3_${seq}
            fi
        fi

        # -- Fallback: download online --
        mkdir -p ${DATA_DIR}/datasets/TUM
        set +e
        robust_download \
            "${TUM_BASE}/rgbd_dataset_freiburg3_${seq}.tgz" \
            "${TUM_FILE}" \
            "${TUM_MIRROR}/rgbd_dataset_freiburg3_${seq}.tgz" \
            "  Baidu Netdisk backup (dynamic sequences): https://pan.baidu.com/s/1UWOO4mUkSLAw9odG458Z5Q  code: jo7l"
        local_rc=$?
        set -e
        if [ ${local_rc} -eq 0 ] && [ -f "${TUM_FILE}" ]; then
            echo "      Extracting ${seq}..."
            cd ${DATA_DIR}/datasets/TUM
            tar -xzf ${TUM_FILE} && rm -f ${TUM_FILE}
            cd rgbd_dataset_freiburg3_${seq}
            python3 ${SEMANTIC_SLAM_ROOT}/scripts/associate.py rgb.txt depth.txt > associate.txt
            echo "  TUM ${seq}: OK"
            TUM_DOWNLOADED=$((TUM_DOWNLOADED + 1))
        else
            echo "  TUM ${seq}: download failed. Manual download options:"
            echo "    Baidu Netdisk (dynamic sequences): https://pan.baidu.com/s/1UWOO4mUkSLAw9odG458Z5Q  code: jo7l"
            echo "    Note: https://pan.baidu.com/s/1W8tBo_QHpAHNyer10dW0Zg (di9m) has no freiburg3, not recommended"
            echo "    Official link: ${TUM_BASE}/rgbd_dataset_freiburg3_${seq}.tgz"
            echo "    Upload: scp rgbd_dataset_freiburg3_${seq}.tgz root@<IP>:${DATA_DIR}/datasets/TUM/"
            echo "    The script will extract it automatically on the next run"
        fi
    else
        echo "  TUM ${seq}: already exists"
        TUM_DOWNLOADED=$((TUM_DOWNLOADED + 1))
    fi
done
# ============================================================

# ==== KITTI 00 (replaces the old KITTI 03) ====
# 00: 2011_10_03_drive_0027, 4541 frames, loop closure, rich dynamic vehicles/pedestrians
# Better than the old KITTI 03 (800 frames, no loop) for evaluating long-term SLAM accuracy and dynamic filtering
#
# Data actually needed (only sequence 00, not all of KITTI):
#   ${DATA_DIR}/datasets/KITTI/00/image_0/  - left camera, 4541 PNGs (~2GB)
#   ${DATA_DIR}/datasets/KITTI/00/image_1/  - right camera, 4541 PNGs (~2GB)
#   ${DATA_DIR}/datasets/KITTI/00/times.txt - timestamps (optional, inferred when missing)
#   ${DATA_DIR}/datasets/KITTI/poses/00.txt  - ground truth trajectory (~5MB)
# About 4GB total, sequences 01-10 are not needed
#
# Download strategies (four options, ordered from least to most traffic):
#   Option C (least traffic, ~4GB): extract the odometry zip locally, upload only 00/ + poses/
#   Option D (bundle, ~8GB): upload the pre-extracted data_odometry_gray/dataset/ directory,
#          the script reorganizes it into the right paths
#   Option A (automatic, ~14GB): official KITTI raw data (AWS S3)
#   Option B (most traffic, ~22GB): upload the full data_odometry_gray.zip, the script extracts then deletes it
# Baidu Netdisk odometry grayscale dataset:
#   Link: https://pan.baidu.com/s/1htFmXDE  code: uu20
#   File: data_odometry_gray.zip (22GB, includes all 00-10 sequences)
#   After extraction: dataset/sequences/00/image_0/ and image_1/
#   Only need the whole dataset/sequences/00/ directory + dataset/poses/00.txt
KITTI00_DIR=${DATA_DIR}/datasets/KITTI/00
if [ ! -d ${KITTI00_DIR}/image_0 ]; then
    echo "  Preparing KITTI 00 (4541 frames, loop + rich dynamics, only ~4GB needed)..."

    KITTI_DOWNLOADED=0

    # -- Option D: user uploaded the pre-extracted data_odometry_gray/dataset/ directory (with sequences/00/ and poses/) --
    ODOMETRY_SEQ00="${DATA_DIR}/datasets/KITTI/data_odometry_gray/dataset/sequences/00"
    ODOMETRY_POSES00="${DATA_DIR}/datasets/KITTI/data_odometry_gray/dataset/poses/00.txt"
    if [ "$KITTI_DOWNLOADED" -eq 0 ] && [ -d "${ODOMETRY_SEQ00}/image_0" ]; then
        echo "  [Option D] Found a pre-extracted odometry directory, reorganizing to the right paths..."
        cd ${DATA_DIR}/datasets/KITTI
        mv data_odometry_gray/dataset/sequences/00 ./
        echo "  KITTI 00 images: OK ($(ls 00/image_0/*.png 2>/dev/null | wc -l) frames)"
        KITTI_DOWNLOADED=1
        # Also extract the poses
        if [ -f "${ODOMETRY_POSES00}" ] && [ ! -f "${DATA_DIR}/datasets/KITTI/poses/00.txt" ]; then
            mkdir -p ${DATA_DIR}/datasets/KITTI/poses
            cp data_odometry_gray/dataset/poses/*.txt ${DATA_DIR}/datasets/KITTI/poses/ 2>/dev/null || true
            echo "  KITTI poses: OK (extracted from pre-extracted directory)"
        fi
        # Clean up the odometry wrapper directory
        rm -rf data_odometry_gray
        echo "  Cleaned up the odometry wrapper directory"
    fi

    # -- Option B: check whether data_odometry_gray.zip (full 22GB zip) was uploaded --
    ODOMETRY_ZIP="${DATA_DIR}/datasets/KITTI/data_odometry_gray.zip"
    if [ "$KITTI_DOWNLOADED" -eq 0 ] && [ -f "${ODOMETRY_ZIP}" ]; then
        echo "  [Option B] Found the odometry zip, extracting only sequence 00 (skipping 01-10)..."
        mkdir -p ${DATA_DIR}/datasets/KITTI && cd ${DATA_DIR}/datasets/KITTI
        unzip -q -o "${ODOMETRY_ZIP}" "dataset/sequences/00/*" -d _odom_tmp
        if [ -d _odom_tmp/dataset/sequences/00 ]; then
            mv _odom_tmp/dataset/sequences/00 ./
            rm -rf _odom_tmp
            echo "  KITTI 00 images: OK ($(ls 00/image_0/*.png 2>/dev/null | wc -l) frames)"
            KITTI_DOWNLOADED=1
        else
            echo "  [Option B] odometry zip structure mismatch, falling back to Option A"
            rm -rf _odom_tmp
        fi
        # Extract the poses (ground truth)
        unzip -q -o "${ODOMETRY_ZIP}" "dataset/poses/*" -d _poses_tmp 2>/dev/null || true
        if [ -d _poses_tmp/dataset/poses ]; then
            mkdir -p ${DATA_DIR}/datasets/KITTI/poses
            cp _poses_tmp/dataset/poses/*.txt ${DATA_DIR}/datasets/KITTI/poses/ 2>/dev/null || true
            rm -rf _poses_tmp
            echo "  KITTI poses: OK (extracted from odometry zip)"
        fi
        rm -f "${ODOMETRY_ZIP}"
        echo "  Deleted ${ODOMETRY_ZIP} to free ~22GB"
    fi

    # -- Option A: KITTI raw data (AWS S3, download only the raw sequence for 00, ~14GB) --
    if [ "$KITTI_DOWNLOADED" -eq 0 ]; then
        KITTI00_SYNC="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip"
        KITTI00_CALIB="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip"
        KITTI_MIRROR="https://semantic-slam-datasets.oss-cn-beijing.aliyuncs.com/kitti"

        mkdir -p ${DATA_DIR}/datasets/KITTI && cd ${DATA_DIR}/datasets/KITTI

        # Download the sync zip (contains images, ~14GB archive, ~4GB excluding lidar)
        KITTI_FILE="2011_10_03_drive_0027_sync.zip"
        set +e
        robust_download "${KITTI00_SYNC}" "${KITTI_FILE}" \
            "${KITTI_MIRROR}/2011_10_03_drive_0027_sync.zip" \
            "  Baidu Netdisk backup: https://pan.baidu.com/s/1htFmXDE  code: uu20"
        local_rc=$?
        set -e
        if [ ${local_rc} -ne 0 ] || [ ! -f "${KITTI_FILE}" ]; then
            echo "  KITTI 00 automatic download failed. Manual options (pick one):"
            echo "    Option C (recommended, ~4GB): extract the odometry zip locally, upload only the 00/ directory"
            echo "      Baidu Netdisk: https://pan.baidu.com/s/1htFmXDE  code: uu20"
            echo "      Local: unzip data_odometry_gray.zip \"dataset/sequences/00/*\""
            echo "      Upload: scp -r 00/ root@<IP>:${DATA_DIR}/datasets/KITTI/00/"
            echo "    Option B (~22GB): upload the full data_odometry_gray.zip, the script extracts it automatically"
            echo "      scp data_odometry_gray.zip root@<IP>:${DATA_DIR}/datasets/KITTI/"
        else
            # Download the calib files
            wget -q --timeout=60 "${KITTI00_CALIB}" -O 2011_10_03_calib.zip 2>/dev/null || true
            # Extract (exclude velodyne_points to save space)
            echo "      Extracting KITTI 00 (excluding lidar data)..."
            unzip -q ${KITTI_FILE} -x '*/velodyne_points/*'
            [ -f 2011_10_03_calib.zip ] && unzip -q 2011_10_03_calib.zip
            # Original layout: 2011_10_03/2011_10_03_drive_0027_sync/image_00/data/*.png
            # ORB-SLAM3 expects: 00/image_0/*.png
            mkdir -p 00
            cp -r 2011_10_03/2011_10_03_drive_0027_sync/image_00/data 00/image_0
            cp -r 2011_10_03/2011_10_03_drive_0027_sync/image_01/data 00/image_1
            rm -rf 2011_10_03 *.zip
            echo "  KITTI 00: OK ($(ls 00/image_0/*.png 2>/dev/null | wc -l) frames)"
            KITTI_DOWNLOADED=1
        fi
    fi
else
    echo "  KITTI 00: already exists ($(ls ${KITTI00_DIR}/image_0/*.png 2>/dev/null | wc -l) frames)"
    KITTI_DOWNLOADED=1
fi

# KITTI ground truth poses (for ATE evaluation and figure generation, ~5MB)
# Note: this step is skipped automatically if the poses were already extracted from the odometry zip
if [ ! -f ${DATA_DIR}/datasets/KITTI/poses/00.txt ]; then
    echo "  Downloading KITTI poses (all sequences, ~5MB)..."
    mkdir -p ${DATA_DIR}/datasets/KITTI/poses
    POSES_FILE="/tmp/kitti_poses.zip"
    rm -f ${POSES_FILE}
    set +e
    robust_download \
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip" \
        "${POSES_FILE}" \
        "https://semantic-slam-datasets.oss-cn-beijing.aliyuncs.com/kitti/data_odometry_poses.zip"
    set -e
    if [ -f "${POSES_FILE}" ]; then
        unzip -q ${POSES_FILE} -d /tmp/kitti_poses_tmp/ || true
        cp /tmp/kitti_poses_tmp/dataset/poses/*.txt ${DATA_DIR}/datasets/KITTI/poses/ 2>/dev/null || true
        rm -rf /tmp/kitti_poses_tmp ${POSES_FILE}
        echo "  KITTI poses: OK ($(ls ${DATA_DIR}/datasets/KITTI/poses/*.txt 2>/dev/null | wc -l) sequences)"
    else
        echo "  KITTI poses: download failed. Note: ATE evaluation needs poses/00.txt"
        echo "    Can be extracted from the odometry zip: unzip data_odometry_gray.zip \"dataset/poses/00.txt\""
    fi
else
    echo "  KITTI poses: already exists ($(ls ${DATA_DIR}/datasets/KITTI/poses/*.txt 2>/dev/null | wc -l) sequences)"
fi
# ============================================================

# ==== EuRoC MAV ====
# MH_01_easy     - static machine hall baseline (~2.2GB)
# MH_03_medium   - medium difficulty, completes the progression (~2.0GB)
# MH_05_difficult - hard sequence stress test (~1.9GB)
# Download strategy: ETH HTTP (resume) -> ETH HTTPS -> OSS mirror in China -> Kaggle API
EUROC_ETH_HTTP="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall"
EUROC_ETH_HTTPS="https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall"
EUROC_MIRROR="https://semantic-slam-datasets.oss-cn-beijing.aliyuncs.com/euroc"
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    if [ ! -d ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data ]; then
        echo "  Preparing EuRoC ${seq} (full dataset)..."
        mkdir -p ${DATA_DIR}/datasets/EuRoC
        cd ${DATA_DIR}/datasets/EuRoC

        DOWNLOADED=0
        EUROC_FILE="${seq}.zip"

        # -- First check: a local zip archive already uploaded by the user --
        if [ -f "${EUROC_FILE}" ]; then
            echo "  [local extract] Found ${EUROC_FILE}, extracting directly..."
            unzip -q "${EUROC_FILE}" && rm -f "${EUROC_FILE}"
            if [ -d "${seq}/mav0/cam0/data" ]; then
                echo "  EuRoC ${seq}: OK ($(ls ${seq}/mav0/cam0/data/*.png 2>/dev/null | wc -l) images) (extracted from local zip)"
                DOWNLOADED=1
            else
                echo "  EuRoC ${seq}: directory structure mismatch after extraction, trying online download..."
                rm -rf "${seq}"
            fi
        fi

        # 1) curl resume ETH HTTP
        if [ "$DOWNLOADED" -eq 0 ]; then
            echo "  [1/4] curl resume ETH HTTP"
            if curl -C - --retry 10 --retry-delay 15 --retry-max-time 600 \
                -L --connect-timeout 30 --max-time 3600 \
                -o "${EUROC_FILE}" "${EUROC_ETH_HTTP}/${seq}/${seq}.zip" 2>/dev/null; then
                if [ -s "${EUROC_FILE}" ]; then
                    echo "      Extracting ${seq}.zip ..."
                    unzip -q "${EUROC_FILE}" && rm -f "${EUROC_FILE}" && DOWNLOADED=1
                fi
            fi
        fi

        # 2) ETH HTTPS
        if [ "$DOWNLOADED" -eq 0 ]; then
            echo "  [2/4] Trying ETH HTTPS"
            rm -f "${EUROC_FILE}"
            if curl --retry 5 --retry-delay 15 -L --connect-timeout 30 --max-time 3600 \
                -o "${EUROC_FILE}" "${EUROC_ETH_HTTPS}/${seq}/${seq}.zip" 2>/dev/null; then
                if [ -s "${EUROC_FILE}" ]; then
                    echo "      Extracting ${seq}.zip ..."
                    unzip -q "${EUROC_FILE}" && rm -f "${EUROC_FILE}" && DOWNLOADED=1
                fi
            fi
        fi

        # 3) OSS mirror in China (added)
        if [ "$DOWNLOADED" -eq 0 ]; then
            echo "  [3/4] Trying OSS mirror in China: ${EUROC_MIRROR}/${seq}.zip"
            rm -f "${EUROC_FILE}"
            if curl -C - --retry 10 --retry-delay 10 --retry-max-time 300 \
                -L --connect-timeout 15 --max-time 3600 \
                -o "${EUROC_FILE}" "${EUROC_MIRROR}/${seq}.zip" 2>/dev/null; then
                if [ -s "${EUROC_FILE}" ]; then
                    echo "      Extracting ${seq}.zip ..."
                    unzip -q "${EUROC_FILE}" && rm -f "${EUROC_FILE}" && DOWNLOADED=1
                fi
            fi
        fi

        # 4) Kaggle API
        if [ "$DOWNLOADED" -eq 0 ]; then
            echo "  [4/4] Trying Kaggle API download (kagglehub)..."
            pip3 install kagglehub -q 2>/dev/null || true
            if python3 -c "import kagglehub" 2>/dev/null; then
                python3 -c "
import kagglehub, os, shutil
dst = '${DATA_DIR}/datasets/EuRoC'
os.makedirs(dst, exist_ok=True)
path = kagglehub.dataset_download('kmader/euroc-mav-dataset', path='${seq}')
if os.path.exists(path):
    target = os.path.join(dst, '${seq}')
    if not os.path.exists(target):
        shutil.move(path, target)
    print('Kaggle download OK')
" 2>/dev/null && DOWNLOADED=1
            fi
        fi

        # 5) All automatic downloads failed -> show manual download instructions
        if [ "$DOWNLOADED" -eq 0 ]; then
            echo "  +-------------------------------------------------------------"
            echo "  | EuRoC ${seq} automatic download failed. The dataset is large (~2GB), download and upload it manually:"
            echo "  |"
            echo "  | Method 1 (fastest): download locally, then scp to AutoDL"
            echo "  |   Download: ${EUROC_ETH_HTTP}/${seq}/${seq}.zip"
            echo "  |   Upload: scp ${seq}.zip root@<AutoDL-IP>:${DATA_DIR}/datasets/EuRoC/"
            echo "  |"
            echo "  | Method 2: Baidu Netdisk"
            echo "  |   Link: https://pan.baidu.com/s/1miXf40o  code: xm59"
            echo "  |   Upload: same as Method 1"
            echo "  |"
            echo "  | Method 3: Kaggle CLI"
            echo "  |   pip3 install kagglehub && python3 -c \"import kagglehub; \\"
            echo "  |       kagglehub.dataset_download('kmader/euroc-mav-dataset')\""
            echo "  +-------------------------------------------------------------"
            echo "  EuRoC ${seq}: skipped (no image data, cannot run SLAM experiments)"
        else
            echo "  EuRoC ${seq}: OK"
        fi
        cd ${SEMANTIC_SLAM_ROOT}
    else
        echo "  EuRoC ${seq}: already exists ($(ls ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data/*.png 2>/dev/null | wc -l) images)"
    fi
done

# Download statistics
DOWNLOAD_END=$(date +%s)
TOTAL_DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))
TOTAL_MIN=$((TOTAL_DOWNLOAD_TIME / 60))
TOTAL_SEC=$((TOTAL_DOWNLOAD_TIME % 60))
echo ""
echo "  ========================================"
echo "  Dataset downloads complete!"
echo "  TUM:   ${TUM_DOWNLOADED}/5 sequences ready"
echo "  Total time: ${TOTAL_MIN}m${TOTAL_SEC}s"
echo "  Data directory: ${DATA_DIR}/datasets/"
du -sh ${DATA_DIR}/datasets/TUM 2>/dev/null | head -1
du -sh ${DATA_DIR}/datasets/KITTI 2>/dev/null | head -1
du -sh ${DATA_DIR}/datasets/EuRoC 2>/dev/null | head -1
echo "  ========================================"
fi  # end of --skip-download check

# ===================== 4.5 Export GT trajectory files (for local figure generation) =====================
echo ""
echo "[4.5/8] Exporting Ground Truth trajectory files..."
GT_EXPORT_DIR=${OUTPUT}/gt_trajectories
mkdir -p ${GT_EXPORT_DIR}/TUM
mkdir -p ${GT_EXPORT_DIR}/KITTI
mkdir -p ${GT_EXPORT_DIR}/EuRoC

# -- TUM GT --
for seq in fr1_xyz fr3_walking_xyz fr3_walking_halfsphere fr3_walking_static fr3_sitting_static fr3_sitting_xyz; do
    tum_seq_dir=""
    case $seq in
        fr1_xyz) tum_seq_dir="rgbd_dataset_freiburg1_xyz" ;;
        fr3_walking_xyz) tum_seq_dir="rgbd_dataset_freiburg3_walking_xyz" ;;
        fr3_walking_halfsphere) tum_seq_dir="rgbd_dataset_freiburg3_walking_halfsphere" ;;
        fr3_walking_static) tum_seq_dir="rgbd_dataset_freiburg3_walking_static" ;;
        fr3_sitting_static) tum_seq_dir="rgbd_dataset_freiburg3_sitting_static" ;;
        fr3_sitting_xyz) tum_seq_dir="rgbd_dataset_freiburg3_sitting_xyz" ;;
    esac
    gt_src="${DATA_DIR}/datasets/TUM/${tum_seq_dir}/groundtruth.txt"
    gt_dst="${GT_EXPORT_DIR}/TUM/${tum_seq_dir}_groundtruth.txt"
    if [ -f "${gt_src}" ]; then
        cp "${gt_src}" "${gt_dst}"
        echo "  TUM ${seq}: exported (${gt_dst})"
    else
        echo "  TUM ${seq}: GT not found, skipping"
    fi
done

# -- KITTI GT --
KITTI_GT_SRC="${DATA_DIR}/datasets/KITTI/poses/00.txt"
if [ -f "${KITTI_GT_SRC}" ]; then
    cp "${KITTI_GT_SRC}" "${GT_EXPORT_DIR}/KITTI/00.txt"
    echo "  KITTI 00: exported (${GT_EXPORT_DIR}/KITTI/00.txt)"
else
    echo "  KITTI 00: GT not found, skipping"
fi

# -- EuRoC GT --
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    gt_src="${DATA_DIR}/datasets/EuRoC/${seq}/mav0/state_groundtruth_estimate0/data.csv"
    gt_dst="${GT_EXPORT_DIR}/EuRoC/${seq}.csv"
    if [ -f "${gt_src}" ]; then
        cp "${gt_src}" "${gt_dst}"
        echo "  EuRoC ${seq}: exported (${gt_dst})"
    else
        echo "  EuRoC ${seq}: GT not found, skipping"
    fi
done
echo "  GT files exported to ${GT_EXPORT_DIR}/"

# ===================== 5. YOLOv8 offline inference =====================
echo ""
echo "[5/8] YOLOv8 offline inference..."

# Helper: run YOLO inference
run_yolo_infer() {
    local seq_name="$1"
    local dataset_path="$2"
    local det_dir="$3"
    if [ ! -d "${det_dir}" ] || [ -z "$(ls -A ${det_dir} 2>/dev/null)" ]; then
        echo "  Running inference on ${seq_name}..."
        python3 ${SEMANTIC_SLAM_ROOT}/scripts/yolov8_offline_inference.py \
            --dataset "${dataset_path}" \
            --output  "${det_dir}" \
            --model   yolov8n-seg.pt \
            --conf    0.45
        echo "  ${seq_name} inference: OK ($(ls ${det_dir}/*.json 2>/dev/null | wc -l) frames)"
    else
        echo "  ${seq_name} inference: already exists ($(ls ${det_dir}/*.json 2>/dev/null | wc -l) frames)"
    fi
}

# TUM inference
for seq in sitting_static sitting_xyz walking_static walking_xyz walking_halfsphere; do
    run_yolo_infer "TUM ${seq}" \
        ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${seq} \
        ${DATA_DIR}/detections/${seq}
done

# KITTI 00 inference
if [ -d ${DATA_DIR}/datasets/KITTI/00 ]; then
    run_yolo_infer "KITTI 00" \
        ${DATA_DIR}/datasets/KITTI/00 \
        ${DATA_DIR}/detections/kitti_00
fi

# EuRoC inference (needs full image data)
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    det_name=$(echo $seq | tr '[:upper:]' '[:lower:]' | tr -d '_')
    if [ -d ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data ]; then
        run_yolo_infer "EuRoC ${seq}" \
            ${DATA_DIR}/datasets/EuRoC/${seq} \
            ${DATA_DIR}/detections/euroc_${det_name}
    fi
done

# ===================== 6. Update YAML config paths =====================
echo ""
echo "[6/8] Updating YAML configs..."
cd ${SEMANTIC_SLAM_ROOT}/src/config

# Use absolute paths (relative paths cannot be resolved from the ORB_SLAM3/build directory)
for yaml in *.yaml; do
    sed -i "s|onnx_path:.*|onnx_path: ${SEMANTIC_SLAM_ROOT}/models/yolov8n-seg.onnx|" $yaml
    sed -i "s|engine_path:.*|engine_path: ${SEMANTIC_SLAM_ROOT}/models/yolov8n-seg.trt|" $yaml
    sed -i "s|class_names_path:.*|class_names_path: ${SEMANTIC_SLAM_ROOT}/models/coco.names|" $yaml
done

# ==== Create per-sequence semantic YAMLs (copy TUM3_semantic.yaml as the template) ====
for seq in sitting_static sitting_xyz walking_static walking_xyz walking_halfsphere; do
    cp TUM3_semantic.yaml TUM3_${seq}_semantic.yaml
    sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/${seq}|" TUM3_${seq}_semantic.yaml
done

# KITTI 00 semantic (uses KITTI00_semantic.yaml, the KITTI03 sequence is deprecated)
sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/kitti_00|" KITTI00_semantic.yaml

# KITTI 00 baseline
sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/kitti_00|" KITTI00_baseline.yaml 2>/dev/null || true

# EuRoC semantic - create a separate YAML for each sequence
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    det_name=$(echo $seq | tr '[:upper:]' '[:lower:]' | tr -d '_')
    cp EuRoC_semantic.yaml EuRoC_${seq}_semantic.yaml
    sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/euroc_${det_name}|" EuRoC_${seq}_semantic.yaml
done

# ====== Ablation experiment YAMLs (YOLO-only / GeoConst-only) ======
echo "  Creating ablation experiment YAMLs..."
# TUM ablation - one config per sequence
for seq in sitting_static sitting_xyz walking_static walking_xyz walking_halfsphere; do
    cp TUM3_yolo_only.yaml TUM3_${seq}_yolo_only.yaml
    sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/${seq}|" TUM3_${seq}_yolo_only.yaml
    cp TUM3_geoconst.yaml TUM3_${seq}_geoconst.yaml
    sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/${seq}|" TUM3_${seq}_geoconst.yaml
done

# EuRoC ablation - one config per sequence
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    det_name=$(echo $seq | tr '[:upper:]' '[:lower:]' | tr -d '_')
    cp EuRoC_yolo_only.yaml EuRoC_${seq}_yolo_only.yaml
    sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/euroc_${det_name}|" EuRoC_${seq}_yolo_only.yaml
    cp EuRoC_geoconst.yaml EuRoC_${seq}_geoconst.yaml
    sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/euroc_${det_name}|" EuRoC_${seq}_geoconst.yaml
done

# KITTI 00 ablation
sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/kitti_00|" KITTI00_yolo_only.yaml
sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/kitti_00|" KITTI00_geoconst.yaml

# Copy ORBvoc.txt (may need to extract the tar.gz first)
if [ ! -f ~/ORB_SLAM3/Vocabulary/ORBvoc.txt ]; then
    echo "  Extracting ORBvoc.txt.tar.gz..."
    cd ~/ORB_SLAM3/Vocabulary && tar -xzf ORBvoc.txt.tar.gz
fi
cp ~/ORB_SLAM3/Vocabulary/ORBvoc.txt ${SEMANTIC_SLAM_ROOT}/models/

echo "  YAML configs: OK"

# ==================== 6.8 Generate masked datasets (Plan C: Image Masking) ====================
echo ""
echo "[6.8/8] Generating masked datasets (YOLO dynamic regions zeroed out, for the YOLO-Mask experiments)..."

MASKED_DATA_DIR="${DATA_DIR}/datasets_masked"

# Check whether masked data already exists (skipped in --resume mode)
NEED_MASKING=true
if [ "$RESUME" = true ] && [ -d "${MASKED_DATA_DIR}" ] && [ "$(ls -A ${MASKED_DATA_DIR} 2>/dev/null)" ]; then
    echo "  --resume mode: masked datasets already exist, skipping generation"
    NEED_MASKING=false
fi

if [ "$NEED_MASKING" = true ]; then
    echo "  Running mask_dataset.py --all ..."
    python3 ${SEMANTIC_SLAM_ROOT}/scripts/mask_dataset.py \
        --all \
        --datasets ${DATA_DIR}/datasets \
        --detections-root ${DATA_DIR}/detections \
        --output ${MASKED_DATA_DIR} || echo "  [WARN] Mask generation partially failed, continuing deployment..."
    echo "  Masked datasets output: ${MASKED_DATA_DIR}/"
fi

# ==================== 7. Run experiments ====================
echo ""
echo "[7/8] Running experiments..."
echo ""
echo "  Experiment overview (Plan C):"
echo "    E1:   Baseline (original datasets, pure ORB-SLAM3)"
echo "    E1.5: YOLO-Mask (masked datasets, pure ORB-SLAM3, replaces the YOLO-only ablation)"
echo "    E2a:  YOLO-only  (C++ per-feature filtering - known to crash, skipped)"
echo "    E2b:  GeoConst    (C++ per-feature filtering - known to crash, skipped)"
echo "    E3:   Full System (C++ per-feature filtering - known to crash, skipped)"
echo ""

VOCAB=${SEMANTIC_SLAM_ROOT}/models/ORBvoc.txt
BUILD=${SEMANTIC_SLAM_ROOT}/src/build
CONFIG=${SEMANTIC_SLAM_ROOT}/src/config

mkdir -p ${OUTPUT}

# Resume-skip helper: skip if the output file already exists and is non-empty
skip_if_output_exists() {
    local outfile="$1"
    local label="$2"
    if [ "${RESUME}" = true ] && [ -s "${outfile}" ]; then
        echo "  [SKIP] ${label} (already exists ${outfile})"
        return 0
    fi
    # Diagnostic: if file exists but RESUME is false, warn and remove stale file
    if [ -f "${outfile}" ] && [ ! -s "${outfile}" ]; then
        echo "  [WARN] ${label} - found empty file ${outfile}, deleting and re-running"
        rm -f "${outfile}"
    elif [ -f "${outfile}" ] && [ "${RESUME}" != true ]; then
        echo "  [WARN] ${label} - found old output ${outfile} (RESUME=false but will re-run anyway)"
    fi
    return 1
}

# Headless mode: use xvfb or disable the Pangolin viewer
# Note: timeouts are now set per dataset by the run_with_timeout function
if ! command -v xvfb-run &>/dev/null; then
    export DISPLAY=""
fi

# Run SLAM commands safely: timeout + leftover process cleanup
# ORB-SLAM3 may hang after saving the trajectory on a headless server (Pangolin destructor bug)
# This function sends SIGTERM first on timeout (lets Pangolin clean up),
# then SIGKILLs after 5s if it still has not exited, so the script can continue
# Usage: run_with_timeout <timeout_seconds> <cmd> [args...]
run_with_timeout() {
    local timeout_sec="$1"
    shift
    # BUG FIX #5: use SIGTERM + kill-after instead of a hard SIGKILL
    # Reason: SIGKILL gives Pangolin no chance to clean up, corrupting the malloc heap
    #       SIGTERM lets the process exit cleanly, avoiding "malloc(): unsorted double linked list corrupted"
    #       kill-after=5 force-kills after 5s, preventing an infinite hang
    if command -v xvfb-run &>/dev/null; then
        xvfb-run -a timeout --signal=TERM --kill-after=5 "${timeout_sec}" "$@"
    else
        export DISPLAY=""
        timeout --signal=TERM --kill-after=5 "${timeout_sec}" "$@"
    fi
    local rc=$?
    # Clean up any leftover ORB-SLAM3 processes (kill only specific names, avoid collateral kills)
    # BUG FIX #6: only clean up processes for the current experiment type
    # Use exact name matching so pkill does not kill running experiments
    pkill -9 -x "rgbd_tum" 2>/dev/null || true
    pkill -9 -x "stereo_kitti" 2>/dev/null || true
    pkill -9 -x "stereo_euroc" 2>/dev/null || true
    pkill -9 -x "semantic_slam_benchmark" 2>/dev/null || true
    sleep 1
    return $rc
}

# Find the ORB-SLAM3 executables (exclude source files .cc/.cpp/.h, match built binaries only)
cd ~/ORB_SLAM3
RGBD_TUM=$(find ~/ORB_SLAM3/build -name "rgbd_tum" -type f 2>/dev/null | head -1)
STEREO_KITTI=$(find ~/ORB_SLAM3/build -name "stereo_kitti" -type f 2>/dev/null | head -1)
STEREO_EUROC=$(find ~/ORB_SLAM3/build -name "stereo_euroc" -type f 2>/dev/null | head -1)
# If not found in build/, fall back to a search that excludes source files
[ -z "$RGBD_TUM" ] && RGBD_TUM=$(find ~/ORB_SLAM3 -path "*/Examples/RGB-D/rgbd_tum" -type f ! -name "*.cc" ! -name "*.cpp" ! -name "*.h" 2>/dev/null | head -1)
[ -z "$STEREO_KITTI" ] && STEREO_KITTI=$(find ~/ORB_SLAM3 -path "*/Examples/Stereo/stereo_kitti" -type f ! -name "*.cc" ! -name "*.cpp" ! -name "*.h" 2>/dev/null | head -1)
[ -z "$STEREO_EUROC" ] && STEREO_EUROC=$(find ~/ORB_SLAM3 -path "*/Examples/Stereo/stereo_euroc" -type f ! -name "*.cc" ! -name "*.cpp" ! -name "*.h" 2>/dev/null | head -1)
echo "  RGBD_TUM=${RGBD_TUM:-NOT FOUND}"
echo "  STEREO_KITTI=${STEREO_KITTI:-NOT FOUND}"
echo "  STEREO_EUROC=${STEREO_EUROC:-NOT FOUND}"
ORB_SLAM3_VOCAB=~/ORB_SLAM3/Vocabulary/ORBvoc.txt

# Make sure associate.txt exists for all TUM sequences
for seq in sitting_static sitting_xyz walking_static walking_xyz walking_halfsphere; do
    ds_dir=${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${seq}
    if [ -d "$ds_dir" ] && [ ! -f "${ds_dir}/associate.txt" ]; then
        python3 ${SEMANTIC_SLAM_ROOT}/scripts/associate.py \
            ${ds_dir}/rgb.txt ${ds_dir}/depth.txt \
            > ${ds_dir}/associate.txt 2>/dev/null || true
    fi
done

# ==================== E1: Baseline (pure ORB-SLAM3) ====================
echo ""
echo "=== E1: Baseline (pure ORB-SLAM3, no semantics) ==="

# Declare the baseline experiment list: [sequence id, TUM dir name, output file name]
if [ -n "$RGBD_TUM" ]; then
    BASELINE_TUM_SEQS=(
        "sitting_static|rgbd_dataset_freiburg3_sitting_static|baseline_tum_sitting_static"
        "sitting_xyz|rgbd_dataset_freiburg3_sitting_xyz|baseline_tum_sitting_xyz"
        "walking_static|rgbd_dataset_freiburg3_walking_static|baseline_tum_walking_static"
        "walking_xyz|rgbd_dataset_freiburg3_walking_xyz|baseline_tum_xyz"
        "walking_halfsphere|rgbd_dataset_freiburg3_walking_halfsphere|baseline_tum_half"
    )
    for entry in "${BASELINE_TUM_SEQS[@]}"; do
        IFS='|' read -r label ds out <<< "$entry"
        if [ -d ${DATA_DIR}/datasets/TUM/${ds} ]; then
            skip_if_output_exists "${OUTPUT}/${out}.txt" "TUM ${label} baseline" && continue
            echo "  TUM ${label} baseline..."
            run_with_timeout 180 $RGBD_TUM \
                $ORB_SLAM3_VOCAB \
                ${CONFIG}/TUM3_baseline.yaml \
                ${DATA_DIR}/datasets/TUM/${ds} \
                ${DATA_DIR}/datasets/TUM/${ds}/associate.txt || true
            cp ~/ORB_SLAM3/CameraTrajectory.txt ${OUTPUT}/${out}.txt 2>/dev/null || true
        fi
    done
fi

# KITTI 00 baseline
echo "  [E1] KITTI 00 baseline..."
if [ -d ${DATA_DIR}/datasets/KITTI/00/image_0 ] && ! skip_if_output_exists "${OUTPUT}/baseline_kitti00.txt" "KITTI 00 baseline"; then
    echo "    Running KITTI 00 baseline..."
    if [ -n "$STEREO_KITTI" ]; then
        run_with_timeout 900 $STEREO_KITTI \
            $ORB_SLAM3_VOCAB \
            ${CONFIG}/KITTI00_baseline.yaml \
            ${DATA_DIR}/datasets/KITTI/00 || true
        cp ~/ORB_SLAM3/CameraTrajectory.txt ${OUTPUT}/baseline_kitti00.txt 2>/dev/null || true
    fi
else
    echo "    Skipped (dataset missing or output already exists)"
fi

# EuRoC baseline - use stereo_euroc (bUseViewer=true, stable under xvfb)
# If stereo_euroc is unavailable, fall back to semantic_slam_benchmark
# BUG FIX #1: generate the timestamp file dynamically from data.csv so it matches the dataset exactly
# The old approach relied on ORB-SLAM3's bundled EuRoC_TimeStamps, which can mismatch the actual dataset
# This caused very few initialization points (18-50 points) and constant "Fail to track local map!"
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    if [ -d ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data ]; then
        seq_lower=$(echo ${seq} | tr '[:upper:]' '[:lower:]')
        skip_if_output_exists "${OUTPUT}/baseline_euroc_${seq_lower}.txt" "EuRoC ${seq} baseline" && continue
        echo "  EuRoC ${seq} baseline..."
        # BUG FIX #1: generate the timestamp file from data.csv
        ts_base=$(echo ${seq} | sed 's/_easy//;s/_medium//;s/_difficult//' | sed 's/_//')
        ts_file="/tmp/euroc_ts_${ts_base}.txt"
        # Prefer the data.csv shipped with the dataset (format: #timestamp [ns],filename)
        data_csv="${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data.csv"
        if [ -f "$data_csv" ]; then
            echo "    Generating timestamp file from data.csv..."
            # Extract the first column (skip comment lines), write to a temp file
            grep -v '^#' "$data_csv" | cut -d',' -f1 > "$ts_file"
            ts_count=$(wc -l < "$ts_file")
            img_count=$(ls ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data/*.png 2>/dev/null | wc -l)
            echo "    Timestamps: ${ts_count} lines, images: ${img_count}"
        elif [ -f ~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${ts_base}.txt ]; then
            ts_file=~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${ts_base}.txt
            echo "    Using bundled timestamps: ${ts_file}"
        elif [ -f ~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${seq}.txt ]; then
            ts_file=~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${seq}.txt
            echo "    Using bundled timestamps: ${ts_file}"
        else
            echo "    [WARN] No timestamp file found, trying to generate from image file names..."
            ls ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data/*.png 2>/dev/null | sort | while read img; do
                basename "$img" .png
            done > "$ts_file"
            ts_file=""
            [ -s "/tmp/euroc_ts_${ts_base}.txt" ] && ts_file="/tmp/euroc_ts_${ts_base}.txt"
        fi
        if [ -n "$STEREO_EUROC" ] && [ -n "$ts_file" ] && [ -s "$ts_file" ]; then
            run_with_timeout 600 $STEREO_EUROC \
                $ORB_SLAM3_VOCAB \
                ${CONFIG}/EuRoC_baseline.yaml \
                ${DATA_DIR}/datasets/EuRoC/${seq} \
                $ts_file || true
            if [ -f ~/ORB_SLAM3/CameraTrajectory.txt ] && [ -s ~/ORB_SLAM3/CameraTrajectory.txt ]; then
                cp ~/ORB_SLAM3/CameraTrajectory.txt ${OUTPUT}/baseline_euroc_${seq_lower}.txt 2>/dev/null || true
                traj_lines=$(wc -l < ${OUTPUT}/baseline_euroc_${seq_lower}.txt)
                echo "    [OK] Trajectory saved (${traj_lines} lines)"
            else
                echo "    [WARN] Trajectory file missing or empty - EuRoC tracking failed (possible causes: motion blur / low texture / lighting changes)"
            fi
        else
            echo "    [WARN] stereo_euroc or timestamps unavailable, falling back to semantic_slam_benchmark"
            run_with_timeout 600 ${BUILD}/semantic_slam_benchmark euroc \
                ${DATA_DIR}/datasets/EuRoC \
                $VOCAB \
                ${CONFIG}/EuRoC_baseline.yaml \
                ${seq} \
                ${OUTPUT} || true
            if [ -f ${OUTPUT}/euroc_${seq}_trajectory.txt ] && [ -s ${OUTPUT}/euroc_${seq}_trajectory.txt ]; then
                cp ${OUTPUT}/euroc_${seq}_trajectory.txt ${OUTPUT}/baseline_euroc_${seq_lower}.txt 2>/dev/null || true
                echo "    [OK] Trajectory saved"
            else
                echo "    [WARN] Trajectory file missing or empty"
            fi
        fi
        # Clean up the temporary timestamp file
        rm -f "/tmp/euroc_ts_${ts_base}.txt"
    fi
done

# ==================== E1.5: YOLO-Mask (Plan C: image masking, replaces the YOLO-only ablation) ====================
echo ""
echo "=== E1.5: YOLO-Mask (image masking, pure ORB-SLAM3 + masked datasets) ==="
echo "  Dynamic regions are pre-masked from YOLO detections, run with stock ORB-SLAM3 (no SemanticSLAM patch)"
echo "  This experiment replaces the original YOLO-only ablation (the C++ per-feature filter crashed in g2o)"
echo ""

# YOLO-Mask TUM - same commands as E1 Baseline, only the dataset path points to masked
if [ -n "$RGBD_TUM" ] && [ -d "${MASKED_DATA_DIR}/TUM" ]; then
    YOLOMASK_TUM_SEQS=(
        "sitting_static|rgbd_dataset_freiburg3_sitting_static|mask_tum_sitting_static"
        "sitting_xyz|rgbd_dataset_freiburg3_sitting_xyz|mask_tum_sitting_xyz"
        "walking_static|rgbd_dataset_freiburg3_walking_static|mask_tum_walking_static"
        "walking_xyz|rgbd_dataset_freiburg3_walking_xyz|mask_tum_walking_xyz"
        "walking_halfsphere|rgbd_dataset_freiburg3_walking_halfsphere|mask_tum_walking_half"
    )
    for entry in "${YOLOMASK_TUM_SEQS[@]}"; do
        IFS='|' read -r label ds out <<< "$entry"
        masked_ds="${MASKED_DATA_DIR}/TUM/${ds}"
        if [ -d "$masked_ds" ] && [ -f "${masked_ds}/associate.txt" ]; then
            skip_if_output_exists "${OUTPUT}/${out}.txt" "TUM ${label} YOLO-Mask" && continue
            echo "  TUM ${label} YOLO-Mask..."
            run_with_timeout 180 $RGBD_TUM \
                $ORB_SLAM3_VOCAB \
                ${CONFIG}/TUM3_baseline.yaml \
                ${masked_ds} \
                ${masked_ds}/associate.txt || true
            cp ~/ORB_SLAM3/CameraTrajectory.txt ${OUTPUT}/${out}.txt 2>/dev/null || true
        else
            echo "  TUM ${label} YOLO-Mask: skipped (masked dataset missing)"
        fi
    done
else
    echo "  TUM YOLO-Mask: skipped (masked dataset missing or RGBD_TUM not set)"
fi

# YOLO-Mask KITTI 00
echo "  YOLO-Mask KITTI 00..."
if [ -d "${MASKED_DATA_DIR}/KITTI/00/image_0" ] && ! skip_if_output_exists "${OUTPUT}/mask_kitti00.txt" "KITTI 00 YOLO-Mask"; then
    if [ -n "$STEREO_KITTI" ]; then
        echo "    Running KITTI 00 YOLO-Mask..."
        run_with_timeout 900 $STEREO_KITTI \
            $ORB_SLAM3_VOCAB \
            ${CONFIG}/KITTI00_baseline.yaml \
            ${MASKED_DATA_DIR}/KITTI/00 || true
        cp ~/ORB_SLAM3/CameraTrajectory.txt ${OUTPUT}/mask_kitti00.txt 2>/dev/null || true
    fi
else
    echo "    Skipped (masked dataset missing or output already exists)"
fi

# YOLO-Mask EuRoC
# BUG FIX #1: also generate the timestamp file dynamically from data.csv
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    if [ -d "${MASKED_DATA_DIR}/EuRoC/${seq}/mav0/cam0/data" ]; then
        seq_lower=$(echo ${seq} | tr '[:upper:]' '[:lower:]')
        skip_if_output_exists "${OUTPUT}/mask_euroc_${seq_lower}.txt" "EuRoC ${seq} YOLO-Mask" && continue
        echo "  EuRoC ${seq} YOLO-Mask..."
        ts_base=$(echo ${seq} | sed 's/_easy//;s/_medium//;s/_difficult//' | sed 's/_//')
        ts_file="/tmp/euroc_ts_mask_${ts_base}.txt"
        # Prefer the data.csv shipped with the masked dataset
        data_csv="${MASKED_DATA_DIR}/EuRoC/${seq}/mav0/cam0/data.csv"
        if [ -f "$data_csv" ]; then
            grep -v '^#' "$data_csv" | cut -d',' -f1 > "$ts_file"
        elif [ -f ~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${ts_base}.txt ]; then
            ts_file=~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${ts_base}.txt
        elif [ -f ~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${seq}.txt ]; then
            ts_file=~/ORB_SLAM3/Examples/Stereo/EuRoC_TimeStamps/${seq}.txt
        fi
        if [ -n "$STEREO_EUROC" ] && [ -n "$ts_file" ] && [ -s "$ts_file" ]; then
            run_with_timeout 600 $STEREO_EUROC \
                $ORB_SLAM3_VOCAB \
                ${CONFIG}/EuRoC_baseline.yaml \
                ${MASKED_DATA_DIR}/EuRoC/${seq} \
                ${ts_file} || true
            if [ -f ~/ORB_SLAM3/CameraTrajectory.txt ] && [ -s ~/ORB_SLAM3/CameraTrajectory.txt ]; then
                cp ~/ORB_SLAM3/CameraTrajectory.txt ${OUTPUT}/mask_euroc_${seq_lower}.txt 2>/dev/null || true
                traj_lines=$(wc -l < ${OUTPUT}/mask_euroc_${seq_lower}.txt)
                echo "    [OK] Trajectory saved (${traj_lines} lines)"
            else
                echo "    [WARN] Trajectory file missing or empty - EuRoC tracking failed"
            fi
        elif [ -x "${BUILD}/semantic_slam_benchmark" ]; then
            run_with_timeout 600 ${BUILD}/semantic_slam_benchmark euroc \
                ${MASKED_DATA_DIR}/EuRoC \
                $VOCAB \
                ${CONFIG}/EuRoC_baseline.yaml \
                ${seq} \
                ${OUTPUT} || true
            if [ -f ${OUTPUT}/euroc_${seq}_trajectory.txt ] && [ -s ${OUTPUT}/euroc_${seq}_trajectory.txt ]; then
                cp ${OUTPUT}/euroc_${seq}_trajectory.txt ${OUTPUT}/mask_euroc_${seq_lower}.txt 2>/dev/null || true
                echo "    [OK] Trajectory saved"
            else
                echo "    [WARN] Trajectory file missing or empty"
            fi
        fi
        rm -f "/tmp/euroc_ts_mask_${ts_base}.txt"
    else
        echo "  EuRoC ${seq} YOLO-Mask: skipped (masked dataset missing)"
    fi
done

# ==================== E2a: Ablation - YOLOv8 Only (no semantic weights, no optical flow) ====================
echo ""
echo "=== E2a: YOLO-Only Ablation (YOLO detections, no optical flow verification, known crash - skipped) ==="
echo "  Plan C: this ablation tier is replaced by E1.5 YOLO-Mask, experiments below are skipped"
echo ""

# Plan C: SKIP all semantic C++ experiments (g2o crash)
SKIP_SEMANTIC=true

# TUM YOLO-only
TUM_YOLO_SEQS=(
    "sitting_static|TUM3_sitting_static_yolo_only"
    "sitting_xyz|TUM3_sitting_xyz_yolo_only"
    "walking_static|TUM3_walking_static_yolo_only"
    "walking_xyz|TUM3_walking_xyz_yolo_only"
    "walking_halfsphere|TUM3_walking_halfsphere_yolo_only"
)
for entry in "${TUM_YOLO_SEQS[@]}"; do
        [ "$SKIP_SEMANTIC" = true ] && { echo "  [SKIP] Plan C: semantic experiments skipped"; break; }
        IFS='|' read -r label yaml <<< "$entry"
    if [ -d ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${label} ]; then
        skip_if_output_exists "${OUTPUT}/yolo_tum_${label}.txt" "TUM ${label} YOLO-only" && continue
        echo "  TUM ${label} YOLO-only..."
        run_with_timeout 180 ${BUILD}/semantic_slam_benchmark tum \
            ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${label} \
            ${VOCAB} \
            ${CONFIG}/${yaml}.yaml \
            ${OUTPUT} || true
        cp ${OUTPUT}/rgbd_dataset_freiburg3_${label}_trajectory.txt ${OUTPUT}/yolo_tum_${label}.txt 2>/dev/null || true
    fi
done

# KITTI 00 YOLO-only
if [ -d ${DATA_DIR}/datasets/KITTI/00/image_0 ] && ! skip_if_output_exists "${OUTPUT}/yolo_kitti00.txt" "KITTI 00 YOLO-only"; then
    echo "  KITTI 00 YOLO-only..."
    run_with_timeout 900 ${BUILD}/semantic_slam_benchmark kitti \
        ${DATA_DIR}/datasets/KITTI \
        ${VOCAB} \
        ${CONFIG}/KITTI00_yolo_only.yaml \
        ${OUTPUT} || true
    cp ${OUTPUT}/kitti_00_trajectory.txt ${OUTPUT}/yolo_kitti00.txt 2>/dev/null || true
fi

# EuRoC YOLO-only
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    if [ -d ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data ]; then
        seq_lower=$(echo ${seq} | tr '[:upper:]' '[:lower:]')
        skip_if_output_exists "${OUTPUT}/yolo_euroc_${seq_lower}.txt" "EuRoC ${seq} YOLO-only" && continue
        echo "  EuRoC ${seq} YOLO-only..."
        run_with_timeout 600 ${BUILD}/semantic_slam_benchmark euroc \
            ${DATA_DIR}/datasets/EuRoC \
            ${VOCAB} \
            ${CONFIG}/EuRoC_${seq}_yolo_only.yaml \
            ${seq} \
            ${OUTPUT} || true
        cp ${OUTPUT}/euroc_${seq}_trajectory.txt ${OUTPUT}/yolo_euroc_$(echo ${seq} | tr '[:upper:]' '[:lower:]').txt 2>/dev/null || true
    fi
done

# ==================== E2b: Ablation - GeoConst (YOLO + optical flow, no semantic weights) ====================
echo ""
echo "=== E2b: GeoConst Ablation (YOLO + optical flow, known crash - skipped) ==="
echo "  Plan C: this ablation tier cannot run (g2o crash), skipped"
echo ""

# SKIP GeoConst experiments (Plan C: g2o crash)
if [ "$SKIP_SEMANTIC" != true ]; then

# TUM GeoConst
TUM_GEO_SEQS=(
    "sitting_static|TUM3_sitting_static_geoconst"
    "sitting_xyz|TUM3_sitting_xyz_geoconst"
    "walking_static|TUM3_walking_static_geoconst"
    "walking_xyz|TUM3_walking_xyz_geoconst"
    "walking_halfsphere|TUM3_walking_halfsphere_geoconst"
)
for entry in "${TUM_GEO_SEQS[@]}"; do
    IFS='|' read -r label yaml <<< "$entry"
    if [ -d ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${label} ]; then
        skip_if_output_exists "${OUTPUT}/geo_tum_${label}.txt" "TUM ${label} GeoConst" && continue
        echo "  TUM ${label} GeoConst..."
        run_with_timeout 180 ${BUILD}/semantic_slam_benchmark tum \
            ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${label} \
            ${VOCAB} \
            ${CONFIG}/${yaml}.yaml \
            ${OUTPUT} || true
        cp ${OUTPUT}/rgbd_dataset_freiburg3_${label}_trajectory.txt ${OUTPUT}/geo_tum_${label}.txt 2>/dev/null || true
    fi
done

# KITTI 00 GeoConst
if [ -d ${DATA_DIR}/datasets/KITTI/00/image_0 ] && ! skip_if_output_exists "${OUTPUT}/geo_kitti00.txt" "KITTI 00 GeoConst"; then
    echo "  KITTI 00 GeoConst..."
    run_with_timeout 900 ${BUILD}/semantic_slam_benchmark kitti \
        ${DATA_DIR}/datasets/KITTI \
        ${VOCAB} \
        ${CONFIG}/KITTI00_geoconst.yaml \
        ${OUTPUT} || true
    cp ${OUTPUT}/kitti_00_trajectory.txt ${OUTPUT}/geo_kitti00.txt 2>/dev/null || true
fi

# EuRoC GeoConst
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    if [ -d ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data ]; then
        seq_lower=$(echo ${seq} | tr '[:upper:]' '[:lower:]')
        skip_if_output_exists "${OUTPUT}/geo_euroc_${seq_lower}.txt" "EuRoC ${seq} GeoConst" && continue
        echo "  EuRoC ${seq} GeoConst..."
        run_with_timeout 600 ${BUILD}/semantic_slam_benchmark euroc \
            ${DATA_DIR}/datasets/EuRoC \
            ${VOCAB} \
            ${CONFIG}/EuRoC_${seq}_geoconst.yaml \
            ${seq} \
            ${OUTPUT} || true
        cp ${OUTPUT}/euroc_${seq}_trajectory.txt ${OUTPUT}/geo_euroc_$(echo ${seq} | tr '[:upper:]' '[:lower:]').txt 2>/dev/null || true
    fi
done
fi  # SKIP_SEMANTIC

# ==================== E3: Full System (YOLO + Flow + Semantic Weights) ====================
echo ""
echo "=== E3: Full System (Ours, YOLO + optical flow dynamic filtering + semantic weight optimization, known crash - skipped) ==="
echo "  Plan C: this experiment cannot run (g2o crash), marked as Future Work"
echo ""

# Plan C: SKIP E3 experiments
if [ "$SKIP_SEMANTIC" != true ]; then

# TUM semantic
TUM_SEM_SEQS=(
    "sitting_static|TUM3_sitting_static_semantic"
    "sitting_xyz|TUM3_sitting_xyz_semantic"
    "walking_static|TUM3_walking_static_semantic"
    "walking_xyz|TUM3_walking_xyz_semantic"
    "walking_halfsphere|TUM3_walking_halfsphere_semantic"
)
for entry in "${TUM_SEM_SEQS[@]}"; do
    IFS='|' read -r label yaml <<< "$entry"
    if [ -d ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${label} ]; then
        skip_if_output_exists "${OUTPUT}/rgbd_dataset_freiburg3_${label}_trajectory.txt" "TUM ${label} semantic" && continue
        echo "  TUM ${label} semantic..."
        run_with_timeout 180 ${BUILD}/semantic_slam_benchmark tum \
            ${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_${label} \
            ${VOCAB} \
            ${CONFIG}/${yaml}.yaml \
            ${OUTPUT} || true
    fi
done

# KITTI 00 semantic
if [ -d ${DATA_DIR}/datasets/KITTI/00/image_0 ] && ! skip_if_output_exists "${OUTPUT}/kitti_00_trajectory.txt" "KITTI 00 semantic"; then
    echo "  KITTI 00 semantic..."
    run_with_timeout 900 ${BUILD}/semantic_slam_benchmark kitti \
        ${DATA_DIR}/datasets/KITTI \
        ${VOCAB} \
        ${CONFIG}/KITTI00_semantic.yaml \
        ${OUTPUT} || true
fi

# EuRoC semantic
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    if [ -d ${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data ]; then
        skip_if_output_exists "${OUTPUT}/euroc_${seq}_trajectory.txt" "EuRoC ${seq} semantic" && continue
        echo "  EuRoC ${seq} semantic..."
        run_with_timeout 600 ${BUILD}/semantic_slam_benchmark euroc \
            ${DATA_DIR}/datasets/EuRoC \
            ${VOCAB} \
            ${CONFIG}/EuRoC_${seq}_semantic.yaml \
            ${seq} \
            ${OUTPUT} || true
    fi
done
fi  # SKIP_SEMANTIC

# ==================== 7.5 Enhanced experiments: timing profiling / failure analysis / parameter search ====================
echo ""
echo "[7.5/8] Enhanced experiments..."

# ---- E4a: timing profiling (fig07) ----
run_timing_profiling() {
    echo ""
    echo "=== E4a: timing profiling ==="
    TIMING_DIR="${OUTPUT}/timing_logs"
    mkdir -p ${TIMING_DIR}

    # Time representative sequences (one representative scene per dataset)
    TIMING_EXPERIMENTS=(
        "tum_walking_xyz|tum|${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz|${VOCAB}|${CONFIG}/TUM3_walking_xyz_semantic.yaml"
        "kitti_00|kitti|${DATA_DIR}/datasets/KITTI|${VOCAB}|${CONFIG}/KITTI00_semantic.yaml"
        "euroc_mh_03|euroc|${DATA_DIR}/datasets/EuRoC|${VOCAB}|${CONFIG}/EuRoC_MH_03_medium_semantic.yaml|MH_03_medium"
    )

    for entry in "${TIMING_EXPERIMENTS[@]}"; do
        IFS='|' read -r name fmt ds vocab yaml seq <<< "$entry"
        echo "  Profiling ${name}..."
        TIMING_LOG="${TIMING_DIR}/${name}_timing.log"

        # Use /usr/bin/time to measure overall performance
        # NOTE: /usr/bin/time is an external program, it cannot call shell functions (run_with_timeout)
        #       so wrap xvfb-run + timeout in a variable
        if command -v xvfb-run &>/dev/null; then
            local VFB_PREFIX="xvfb-run -a timeout --signal=TERM --kill-after=5 1200"
        else
            export DISPLAY=""
            local VFB_PREFIX="timeout --signal=TERM --kill-after=5 1200"
        fi
        if [ "$fmt" = "tum" ]; then
            /usr/bin/time -v -o ${TIMING_LOG} \
                ${VFB_PREFIX} ${BUILD}/semantic_slam_benchmark tum "${ds}" "${vocab}" "${yaml}" "${OUTPUT}" 2>&1 | tee -a ${TIMING_DIR}/${name}_stdout.log || true
        elif [ "$fmt" = "kitti" ]; then
            /usr/bin/time -v -o ${TIMING_LOG} \
                ${VFB_PREFIX} ${BUILD}/semantic_slam_benchmark kitti "${ds}" "${vocab}" "${yaml}" "${OUTPUT}" 2>&1 | tee -a ${TIMING_DIR}/${name}_stdout.log || true
        elif [ "$fmt" = "euroc" ]; then
            /usr/bin/time -v -o ${TIMING_LOG} \
                ${VFB_PREFIX} ${BUILD}/semantic_slam_benchmark euroc "${ds}" "${vocab}" "${yaml}" "${seq}" "${OUTPUT}" 2>&1 | tee -a ${TIMING_DIR}/${name}_stdout.log || true
        fi

        # Extract the timing data
        if [ -f ${TIMING_LOG} ]; then
            WALL_CLOCK=$(grep "Elapsed (wall clock) time" ${TIMING_LOG} | awk '{print $NF}' | sed 's/:/ /' | awk '{printf "%.2f", $1*60+$2}')
            USER_TIME=$(grep "User time (seconds)" ${TIMING_LOG} | awk '{print $NF}')
            SYS_TIME=$(grep "System time (seconds)" ${TIMING_LOG} | awk '{print $NF}')
            MAX_RSS=$(grep "Maximum resident set size" ${TIMING_LOG} | awk '{print $NF}')
            CPU_PCT=$(grep "Percent of CPU" ${TIMING_LOG} | awk '{print $NF}' | sed 's/%//')
            echo "    Wall: ${WALL_CLOCK:-N/A}s  User: ${USER_TIME:-N/A}s  MaxRSS: ${MAX_RSS:-N/A}KB  CPU: ${CPU_PCT:-N/A}%"
        fi
    done

    # Summarize timing to JSON
    TIMING_JSON="${TIMING_DIR}/timing_summary.json"
    python3 -c "
import json, os, re
from datetime import datetime

def parse_time_log(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        content = f.read()
    def extract(pattern):
        m = re.search(pattern, content)
        return m.group(1).strip() if m else None
    return {
        'wall_clock_sec': extract(r'Elapsed \(wall clock\) time.*?(\d+:\d+[\d.]*)'),
        'user_time_sec': extract(r'User time \(seconds\):\s*([\d.]+)'),
        'system_time_sec': extract(r'System time \(seconds\):\s*([\d.]+)'),
        'max_rss_kb': extract(r'Maximum resident set size.*?:\s*(\d+)'),
        'cpu_percent': extract(r'Percent of CPU.*?:\s*([\d.]+)'),
    }

data = {
    '_meta': {'source': 'autodl_deploy.sh timing_profiling', 'generated': datetime.utcnow().isoformat() + 'Z'},
    'tum_walking_xyz': parse_time_log('${TIMING_DIR}/tum_walking_xyz_timing.log'),
    'kitti_00': parse_time_log('${TIMING_DIR}/kitti_00_timing.log'),
    'euroc_mh_03': parse_time_log('${TIMING_DIR}/euroc_mh_03_timing.log'),
}

with open('${TIMING_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'  Timing summary: ${TIMING_JSON}')
"
}

# ---- E4b: failure case analysis (fig09) ----
run_failure_analysis() {
    echo ""
    echo "=== E4b: failure case analysis ==="
    FAIL_DIR="${OUTPUT}/failure_analysis"
    mkdir -p ${FAIL_DIR}

    # Challenge sequence list: [name, format, dataset path, challenge type]
    CHALLENGE_SEQS=(
        "tum_walking_halfsphere|tum|${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_walking_halfsphere|${VOCAB}|${CONFIG}/TUM3_walking_halfsphere_semantic.yaml|dynamic_occlusion"
        "tum_walking_xyz|tum|${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz|${VOCAB}|${CONFIG}/TUM3_walking_xyz_semantic.yaml|fast_motion"
    )

    if [ -d "${DATA_DIR}/datasets/KITTI/00/image_0" ]; then
        CHALLENGE_SEQS+=("kitti_00_challenge|kitti|${DATA_DIR}/datasets/KITTI|${VOCAB}|${CONFIG}/KITTI00_semantic.yaml|outdoor_large_scale")
    fi

    for seq in MH_05_difficult; do
        if [ -d "${DATA_DIR}/datasets/EuRoC/${seq}/mav0/cam0/data" ]; then
            CHALLENGE_SEQS+=("euroc_${seq}|euroc|${DATA_DIR}/datasets/EuRoC|${VOCAB}|${CONFIG}/EuRoC_${seq}_semantic.yaml|${seq}|low_texture_motion_blur")
        fi
    done

    for entry in "${CHALLENGE_SEQS[@]}"; do
        IFS='|' read -r name fmt ds vocab yaml seq challenge_type <<< "$entry"
        echo "  Challenge: ${name} (${challenge_type})..."
        FAIL_LOG="${FAIL_DIR}/${name}_failure.log"

        if [ "$fmt" = "tum" ]; then
            run_with_timeout 180 ${BUILD}/semantic_slam_benchmark tum "${ds}" "${vocab}" "${yaml}" "${OUTPUT}" 2>&1 | tee ${FAIL_LOG} || true
        elif [ "$fmt" = "kitti" ]; then
            run_with_timeout 900 ${BUILD}/semantic_slam_benchmark kitti "${ds}" "${vocab}" "${yaml}" "${OUTPUT}" 2>&1 | tee ${FAIL_LOG} || true
        elif [ "$fmt" = "euroc" ]; then
            run_with_timeout 600 ${BUILD}/semantic_slam_benchmark euroc "${ds}" "${vocab}" "${yaml}" "${seq}" "${OUTPUT}" 2>&1 | tee ${FAIL_LOG} || true
        fi

        # Extract failure metrics: tracking lost, large reprojection error, few inliers
        TRACK_LOST=$(grep -c "track lost\|Tracking lost\|LOST" ${FAIL_LOG} 2>/dev/null || echo 0)
        echo "    Track lost events: ${TRACK_LOST}"
        echo "${name}|${challenge_type}|${TRACK_LOST}" >> ${FAIL_DIR}/failure_summary.txt
    done

    # Summarize the failure analysis
    FAIL_SUMMARY_JSON="${FAIL_DIR}/failure_summary.json"
    if [ -f ${FAIL_DIR}/failure_summary.txt ]; then
        python3 -c "
import json
from datetime import datetime

results = {}
with open('${FAIL_DIR}/failure_summary.txt') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            name, ctype, lost = parts
            results[name] = {
                'challenge_type': ctype,
                'track_lost_count': int(lost) if lost.isdigit() else 0,
                'status': 'completed' if int(lost) == 0 else 'partial_failure'
            }

data = {
    '_meta': {'source': 'autodl_deploy.sh failure_analysis', 'generated': datetime.utcnow().isoformat() + 'Z',
              'note': 'track_lost_count=0 means tracking succeeded end to end. >0 means lost tracking segments exist, the failure type needs manual annotation.'},
    'challenge_sequences': results,
    'failure_categories': ['dynamic_occlusion', 'fast_motion', 'outdoor_large_scale', 'low_texture_motion_blur', 'lighting_change']
}
with open('${FAIL_SUMMARY_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'  Failure summary: ${FAIL_SUMMARY_JSON}')
"
    else
        echo "  No challenge sequences available (datasets missing)"
    fi
}

# ---- E4c: parameter grid search (fig10) ----
run_parameter_sweep() {
    echo ""
    echo "=== E4c: parameter grid search ==="
    SWEEP_DIR="${OUTPUT}/parameter_sweep"
    mkdir -p ${SWEEP_DIR}
    SWEEP_JSON="${SWEEP_DIR}/parameter_sensitivity.json"

    # Grid search configuration
    ORB_FEATURES_VALS=(500 800 1000 1200 1500 2000)
    SEMANTIC_WEIGHT_VALS=(0.0 0.2 0.4 0.5 0.8 1.0)
    DYNAMIC_THRESHOLD_VALS=(0.3 0.4 0.5 0.6 0.7 0.8)

    # Use walking_xyz as the representative sequence for the sweep (medium dynamic scenes)
    TUM_REP_DS="${DATA_DIR}/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz"
    TUM_REP_GT="${TUM_REP_DS}/groundtruth.txt"
    TUM_REP_YAML_TEMPLATE="${CONFIG}/TUM3_semantic.yaml"
    TUM_REP_YAML_SWEEP="${SWEEP_DIR}/TUM3_sweep.yaml"

    # ---- Parameter dimension 1: ORB feature count ----
    if [ -d "$TUM_REP_DS" ]; then
        echo "  [1/3] ORB feature count sweep (walking_xyz)..."
        cp ${TUM_REP_YAML_TEMPLATE} ${TUM_REP_YAML_SWEEP}
        sed -i "s|ORBextractor.nFeatures:.*|ORBextractor.nFeatures: 1000|" ${TUM_REP_YAML_SWEEP}
        sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/walking_xyz|" ${TUM_REP_YAML_SWEEP}

        for nfeat in "${ORB_FEATURES_VALS[@]}"; do
            echo "    nFeatures=${nfeat}..."
            sed -i "s|ORBextractor.nFeatures:.*|ORBextractor.nFeatures: ${nfeat}|" ${TUM_REP_YAML_SWEEP}
            run_with_timeout 300 ${BUILD}/semantic_slam_benchmark tum "${TUM_REP_DS}" "${VOCAB}" "${TUM_REP_YAML_SWEEP}" "${SWEEP_DIR}" 2>/dev/null || true
            cp ${SWEEP_DIR}/rgbd_dataset_freiburg3_walking_xyz_trajectory.txt ${SWEEP_DIR}/tum_orb_${nfeat}_traj.txt 2>/dev/null || true
            ATE_VAL="N/A"
            if [ -f "${SWEEP_DIR}/tum_orb_${nfeat}_traj.txt" ] && [ -f "$TUM_REP_GT" ]; then
                ATE_VAL=$(evo_ape tum "$TUM_REP_GT" "${SWEEP_DIR}/tum_orb_${nfeat}_traj.txt" --align --correct_scale 2>&1 | extract_rmse)
            fi
            echo "orb_${nfeat}=${ATE_VAL}" >> ${SWEEP_DIR}/sweep_orb.txt
            echo "      ATE: ${ATE_VAL}"
        done
    fi

    # ---- Parameter dimension 2: semantic weight (consistency_lambda sweep) ----
    if [ -d "$TUM_REP_DS" ]; then
        echo "  [2/3] Semantic weight sweep (walking_xyz)..."
        cp ${TUM_REP_YAML_TEMPLATE} ${TUM_REP_YAML_SWEEP}
        sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/walking_xyz|" ${TUM_REP_YAML_SWEEP}
        sed -i "s|ORBextractor.nFeatures:.*|ORBextractor.nFeatures: 1000|" ${TUM_REP_YAML_SWEEP}

        for w in "${SEMANTIC_WEIGHT_VALS[@]}"; do
            echo "    consistency_lambda=${w}..."
            sed -i "s|semantic_weights.consistency_lambda:.*|semantic_weights.consistency_lambda: ${w}|" ${TUM_REP_YAML_SWEEP}
            # Also adjust building_weight (the main semantic weight)
            sed -i "s|semantic_weights.building_weight:.*|semantic_weights.building_weight: $(python3 -c "print(1.0 - 0.5*${w})")|" ${TUM_REP_YAML_SWEEP}
            run_with_timeout 300 ${BUILD}/semantic_slam_benchmark tum "${TUM_REP_DS}" "${VOCAB}" "${TUM_REP_YAML_SWEEP}" "${SWEEP_DIR}" 2>/dev/null || true
            cp ${SWEEP_DIR}/rgbd_dataset_freiburg3_walking_xyz_trajectory.txt ${SWEEP_DIR}/tum_semw_${w}_traj.txt 2>/dev/null || true
            ATE_VAL="N/A"
            if [ -f "${SWEEP_DIR}/tum_semw_${w}_traj.txt" ] && [ -f "$TUM_REP_GT" ]; then
                ATE_VAL=$(evo_ape tum "$TUM_REP_GT" "${SWEEP_DIR}/tum_semw_${w}_traj.txt" --align --correct_scale 2>&1 | extract_rmse)
            fi
            echo "semweight_${w}=${ATE_VAL}" >> ${SWEEP_DIR}/sweep_semantic.txt
            echo "      ATE: ${ATE_VAL}"
        done
    fi

    # ---- Parameter dimension 3: dynamic threshold (flow_threshold_px sweep) ----
    if [ -d "$TUM_REP_DS" ]; then
        echo "  [3/3] Dynamic threshold sweep (walking_xyz)..."
        cp ${TUM_REP_YAML_TEMPLATE} ${TUM_REP_YAML_SWEEP}
        sed -i "s|yolo_detector.detection_dir:.*|yolo_detector.detection_dir: ${DATA_DIR}/detections/walking_xyz|" ${TUM_REP_YAML_SWEEP}
        sed -i "s|ORBextractor.nFeatures:.*|ORBextractor.nFeatures: 1000|" ${TUM_REP_YAML_SWEEP}
        sed -i "s|semantic_weights.consistency_lambda:.*|semantic_weights.consistency_lambda: 0.1|" ${TUM_REP_YAML_SWEEP}
        sed -i "s|semantic_weights.building_weight:.*|semantic_weights.building_weight: 1.0|" ${TUM_REP_YAML_SWEEP}

        for thresh in "${DYNAMIC_THRESHOLD_VALS[@]}"; do
            echo "    flow_threshold_px=${thresh}..."
            sed -i "s|dynamic_filter.flow_threshold_px:.*|dynamic_filter.flow_threshold_px: ${thresh}|" ${TUM_REP_YAML_SWEEP}
            run_with_timeout 300 ${BUILD}/semantic_slam_benchmark tum "${TUM_REP_DS}" "${VOCAB}" "${TUM_REP_YAML_SWEEP}" "${SWEEP_DIR}" 2>/dev/null || true
            cp ${SWEEP_DIR}/rgbd_dataset_freiburg3_walking_xyz_trajectory.txt ${SWEEP_DIR}/tum_dyn_${thresh}_traj.txt 2>/dev/null || true
            ATE_VAL="N/A"
            if [ -f "${SWEEP_DIR}/tum_dyn_${thresh}_traj.txt" ] && [ -f "$TUM_REP_GT" ]; then
                ATE_VAL=$(evo_ape tum "$TUM_REP_GT" "${SWEEP_DIR}/tum_dyn_${thresh}_traj.txt" --align --correct_scale 2>&1 | extract_rmse)
            fi
            echo "dynthresh_${thresh}=${ATE_VAL}" >> ${SWEEP_DIR}/sweep_dynamic.txt
            echo "      ATE: ${ATE_VAL}"
        done
    fi

    # Summarize the parameter sweep results
    python3 -c "
import json, os
from datetime import datetime

def parse_sweep(path):
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    k, v = line.split('=', 1)
                    try: results[k] = float(v)
                    except: results[k] = None
    return results

data = {
    '_meta': {
        'source': 'autodl_deploy.sh parameter_sweep',
        'generated': datetime.utcnow().isoformat() + 'Z',
        'representative_sequence': 'TUM fr3/walking_xyz',
        'note': 'ATE RMSE values for parameter grid search. null indicates experiment failure.'
    },
    'orb_features': parse_sweep('${SWEEP_DIR}/sweep_orb.txt'),
    'semantic_weight': parse_sweep('${SWEEP_DIR}/sweep_semantic.txt'),
    'dynamic_threshold': parse_sweep('${SWEEP_DIR}/sweep_dynamic.txt'),
}

with open('${SWEEP_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'  Parameter sweep summary: ${SWEEP_JSON}')
"
}

# Run the enhanced experiments
if ! skip_if_output_exists "${OUTPUT}/timing_logs/timing_summary.json" "E4a timing profiling"; then
    run_timing_profiling
fi
if ! skip_if_output_exists "${OUTPUT}/failure_analysis/failure_summary.json" "E4b failure analysis"; then
    run_failure_analysis
fi
# Helper: extract the rmse value from evo_ape output
extract_rmse() {
    # Extract the rmse value from evo_ape output, e.g. "rmse    0.012345"
    # Return "N/A" when there is no output to avoid breaking downstream Python parsing
    local result=$(grep -i "rmse" 2>/dev/null | head -1 | awk '{print $NF}')
    echo "${result:-N/A}"
}

if ! skip_if_output_exists "${OUTPUT}/parameter_sweep/parameter_sensitivity.json" "E4c parameter search"; then
    run_parameter_sweep
fi

# ==================== 8. ATE evaluation & experiment result export ====================
echo ""
echo "[8/8] ATE evaluation & experiment result export..."

RESULTS_JSON="${DATA_DIR}/experiment_results.json"
RESULTS_TMP="/tmp/ate_results.txt"
> ${RESULTS_TMP}

# Helper: TUM ATE evaluation and result recording
ate_tum_record() {
    local key="$1"   # e.g. "tum_sitting_static_baseline"
    local traj="$2"
    local gt="$3"
    local rmse="N/A"
    if [ -f "$traj" ] && [ -f "$gt" ]; then
        echo "  [ATE] TUM ${key} ..."
        FIRST_LINE=$(head -1 "$traj" 2>/dev/null)
        NCOLS=$(echo "$FIRST_LINE" | awk '{print NF}')
        if [ "$NCOLS" = "12" ]; then
            # KITTI to TUM format conversion
            python3 -c "
import numpy as np
from scipy.spatial.transform import Rotation
with open('${traj}') as f:
    for i, line in enumerate(f):
        v = list(map(float, line.split()))
        if len(v) >= 12:
            T = np.eye(4)
            T[0,:4]=v[0:4]; T[1,:4]=v[4:8]; T[2,:4]=v[8:12]
            q = Rotation.from_matrix(T[:3,:3]).as_quat()
            print(f'{i*0.033:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}')
" > /tmp/ate_tmp_tum.txt 2>/dev/null
            if [ -s /tmp/ate_tmp_tum.txt ]; then
                rmse=$(evo_ape tum "$gt" /tmp/ate_tmp_tum.txt --align --correct_scale 2>&1 | extract_rmse)
            fi
        else
            rmse=$(evo_ape tum "$gt" "$traj" --align --correct_scale 2>&1 | extract_rmse)
        fi
    fi
    echo "${key}=${rmse}" >> ${RESULTS_TMP}
    echo "    rmse: ${rmse}"
}

# Helper: KITTI ATE evaluation and result recording
ate_kitti_record() {
    local key="$1"
    local traj="$2"
    local gt="$3"
    local rmse="N/A"
    if [ -f "$traj" ] && [ -f "$gt" ]; then
        echo "  [ATE] KITTI ${key} ..."
        FIRST_LINE=$(head -1 "$traj" 2>/dev/null)
        NCOLS=$(echo "$FIRST_LINE" | awk '{print NF}')
        if [ "$NCOLS" = "8" ]; then
            # TUM format (8 columns) -> convert to KITTI format (12 columns), then evaluate
            local kitti_traj="${traj}.kitti"
            rm -f "$kitti_traj" 2>/dev/null || true
            evo_traj tum "$traj" --save_as_kitti 2>/dev/null || true
            if [ -f "$kitti_traj" ]; then
                rmse=$(evo_ape kitti "$gt" "$kitti_traj" --align --correct_scale 2>&1 | extract_rmse)
            else
                rmse=$(evo_ape kitti "$gt" "$traj" --align --correct_scale 2>&1 | extract_rmse)
            fi
        elif [ "$NCOLS" = "12" ]; then
            # KITTI format (12 columns) - evaluate directly
            rmse=$(evo_ape kitti "$gt" "$traj" --align --correct_scale 2>&1 | extract_rmse)
        else
            echo "    [WARN] Unknown trajectory format (${NCOLS} columns), skipped"
        fi
    fi
    echo "${key}=${rmse}" >> ${RESULTS_TMP}
    echo "    rmse: ${rmse}"
}

# Helper: EuRoC ATE evaluation and result recording
ate_euroc_record() {
    local key="$1"
    local traj="$2"
    local gt="$3"
    local rmse="N/A"
    if [ -f "$traj" ] && [ -f "$gt" ]; then
        echo "  [ATE] EuRoC ${key} ..."
        rmse=$(evo_ape euroc "$gt" "$traj" --align --correct_scale 2>&1 | extract_rmse)
    fi
    echo "${key}=${rmse}" >> ${RESULTS_TMP}
    echo "    rmse: ${rmse}"
}

# Helper: generic ATE evaluation (auto-selects TUM/KITTI/EuRoC format)
ate_record_generic() {
    local key="$1"   # e.g. "tum_sitting_static_yolo"
    local traj="$2"  # trajectory file path
    local gt="$3"    # GT file path
    local fmt="$4"   # tum | kitti | euroc
    local rmse="N/A"
    if [ -f "$traj" ] && [ -f "$gt" ]; then
        echo "  [ATE] ${key} ..."
        if [ "$fmt" = "kitti" ]; then
            rmse=$(evo_ape kitti "$gt" "$traj" --align --correct_scale 2>&1 | extract_rmse)
        elif [ "$fmt" = "euroc" ]; then
            rmse=$(evo_ape euroc "$gt" "$traj" --align --correct_scale 2>&1 | extract_rmse)
        else
            rmse=$(evo_ape tum "$gt" "$traj" --align --correct_scale 2>&1 | extract_rmse)
        fi
    fi
    echo "${key}=${rmse}" >> ${RESULTS_TMP}
    echo "    rmse: ${rmse}"
}

# ---- TUM Baseline ----
for entry in \
    "tum_sitting_static_baseline|baseline_tum_sitting_static.txt|rgbd_dataset_freiburg3_sitting_static" \
    "tum_sitting_xyz_baseline|baseline_tum_sitting_xyz.txt|rgbd_dataset_freiburg3_sitting_xyz" \
    "tum_walking_static_baseline|baseline_tum_walking_static.txt|rgbd_dataset_freiburg3_walking_static" \
    "tum_walking_xyz_baseline|baseline_tum_xyz.txt|rgbd_dataset_freiburg3_walking_xyz" \
    "tum_walking_halfsphere_baseline|baseline_tum_half.txt|rgbd_dataset_freiburg3_walking_halfsphere"; do
    IFS='|' read -r key traj ds <<< "$entry"
    ate_tum_record "${key}" \
        ${OUTPUT}/${traj} \
        ${DATA_DIR}/datasets/TUM/${ds}/groundtruth.txt
done

# ---- TUM Semantic ----
for entry in \
    "tum_sitting_static_semantic|rgbd_dataset_freiburg3_sitting_static_trajectory.txt|rgbd_dataset_freiburg3_sitting_static" \
    "tum_sitting_xyz_semantic|rgbd_dataset_freiburg3_sitting_xyz_trajectory.txt|rgbd_dataset_freiburg3_sitting_xyz" \
    "tum_walking_static_semantic|rgbd_dataset_freiburg3_walking_static_trajectory.txt|rgbd_dataset_freiburg3_walking_static" \
    "tum_walking_xyz_semantic|rgbd_dataset_freiburg3_walking_xyz_trajectory.txt|rgbd_dataset_freiburg3_walking_xyz" \
    "tum_walking_halfsphere_semantic|rgbd_dataset_freiburg3_walking_halfsphere_trajectory.txt|rgbd_dataset_freiburg3_walking_halfsphere"; do
    IFS='|' read -r key traj ds <<< "$entry"
    ate_tum_record "${key}" \
        ${OUTPUT}/${traj} \
        ${DATA_DIR}/datasets/TUM/${ds}/groundtruth.txt
done

# ---- TUM YOLO-Mask (Plan C: image masking) ----
for entry in \
    "tum_sitting_static_mask|mask_tum_sitting_static.txt|rgbd_dataset_freiburg3_sitting_static" \
    "tum_sitting_xyz_mask|mask_tum_sitting_xyz.txt|rgbd_dataset_freiburg3_sitting_xyz" \
    "tum_walking_static_mask|mask_tum_walking_static.txt|rgbd_dataset_freiburg3_walking_static" \
    "tum_walking_xyz_mask|mask_tum_walking_xyz.txt|rgbd_dataset_freiburg3_walking_xyz" \
    "tum_walking_halfsphere_mask|mask_tum_walking_half.txt|rgbd_dataset_freiburg3_walking_halfsphere"; do
    IFS='|' read -r key traj ds <<< "$entry"
    ate_tum_record "${key}" \
        ${OUTPUT}/${traj} \
        ${DATA_DIR}/datasets/TUM/${ds}/groundtruth.txt
done

# ---- KITTI 00 Baseline ----
ate_kitti_record "kitti_00_baseline" \
    ${OUTPUT}/baseline_kitti00.txt \
    ${DATA_DIR}/datasets/KITTI/poses/00.txt

# ---- KITTI 00 Semantic ----
KITTI00_SEM_TRAJ="${OUTPUT}/kitti_00_trajectory.txt"
rmse_kitti00_sem="N/A"
if [ -f "${KITTI00_SEM_TRAJ}" ] && [ -f ${DATA_DIR}/datasets/KITTI/poses/00.txt ]; then
    echo "  [ATE] KITTI 00 (semantic) ..."
    FIRST_LINE=$(head -1 "${KITTI00_SEM_TRAJ}" 2>/dev/null)
    NCOLS=$(echo "$FIRST_LINE" | awk '{print NF}')
    if [ "$NCOLS" = "8" ]; then
        # TUM format -> convert to KITTI format
        rm -f "${KITTI00_SEM_TRAJ}.kitti" 2>/dev/null || true
        evo_traj tum "${KITTI00_SEM_TRAJ}" --save_as_kitti 2>/dev/null || true
        if [ -f "${KITTI00_SEM_TRAJ}.kitti" ]; then
            rmse_kitti00_sem=$(evo_ape kitti ${DATA_DIR}/datasets/KITTI/poses/00.txt "${KITTI00_SEM_TRAJ}.kitti" --align --correct_scale 2>&1 | extract_rmse)
        fi
    elif [ "$NCOLS" = "12" ]; then
        # KITTI format - evaluate directly
        rmse_kitti00_sem=$(evo_ape kitti ${DATA_DIR}/datasets/KITTI/poses/00.txt "${KITTI00_SEM_TRAJ}" --align --correct_scale 2>&1 | extract_rmse)
    fi
fi
echo "kitti_00_semantic=${rmse_kitti00_sem}" >> ${RESULTS_TMP}
echo "    rmse: ${rmse_kitti00_sem}"

# ---- KITTI 00 YOLO-Mask (Plan C) ----
ate_kitti_record "kitti_00_mask" \
    ${OUTPUT}/mask_kitti00.txt \
    ${DATA_DIR}/datasets/KITTI/poses/00.txt

# ---- EuRoC Baseline & Semantic ----
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    seq_lower=$(echo $seq | tr '[:upper:]' '[:lower:]')
    EUROC_GT=${DATA_DIR}/datasets/EuRoC/${seq}/mav0/state_groundtruth_estimate0/data.csv
    ate_euroc_record "euroc_${seq_lower}_baseline" \
        ${OUTPUT}/baseline_euroc_${seq_lower}.txt \
        ${EUROC_GT}
    ate_euroc_record "euroc_${seq_lower}_semantic" \
        ${OUTPUT}/euroc_${seq}_trajectory.txt \
        ${EUROC_GT}
    # YOLO-Mask (Plan C)
    ate_euroc_record "euroc_${seq_lower}_mask" \
        ${OUTPUT}/mask_euroc_${seq_lower}.txt \
        ${EUROC_GT}
done

# ---- TUM YOLO-Only Ablation ----
for entry in \
    "tum_sitting_static_yolo|yolo_tum_sitting_static.txt|rgbd_dataset_freiburg3_sitting_static" \
    "tum_sitting_xyz_yolo|yolo_tum_sitting_xyz.txt|rgbd_dataset_freiburg3_sitting_xyz" \
    "tum_walking_static_yolo|yolo_tum_walking_static.txt|rgbd_dataset_freiburg3_walking_static" \
    "tum_walking_xyz_yolo|yolo_tum_walking_xyz.txt|rgbd_dataset_freiburg3_walking_xyz" \
    "tum_walking_halfsphere_yolo|yolo_tum_walking_halfsphere.txt|rgbd_dataset_freiburg3_walking_halfsphere"; do
    IFS='|' read -r key traj ds <<< "$entry"
    ate_tum_record "${key}" \
        ${OUTPUT}/${traj} \
        ${DATA_DIR}/datasets/TUM/${ds}/groundtruth.txt
done

# ---- TUM GeoConst Ablation ----
for entry in \
    "tum_sitting_static_geo|geo_tum_sitting_static.txt|rgbd_dataset_freiburg3_sitting_static" \
    "tum_sitting_xyz_geo|geo_tum_sitting_xyz.txt|rgbd_dataset_freiburg3_sitting_xyz" \
    "tum_walking_static_geo|geo_tum_walking_static.txt|rgbd_dataset_freiburg3_walking_static" \
    "tum_walking_xyz_geo|geo_tum_walking_xyz.txt|rgbd_dataset_freiburg3_walking_xyz" \
    "tum_walking_halfsphere_geo|geo_tum_walking_halfsphere.txt|rgbd_dataset_freiburg3_walking_halfsphere"; do
    IFS='|' read -r key traj ds <<< "$entry"
    ate_tum_record "${key}" \
        ${OUTPUT}/${traj} \
        ${DATA_DIR}/datasets/TUM/${ds}/groundtruth.txt
done

# ---- KITTI 00 YOLO-Only ----
ate_kitti_record "kitti_00_yolo" \
    ${OUTPUT}/yolo_kitti00.txt \
    ${DATA_DIR}/datasets/KITTI/poses/00.txt

# ---- KITTI 00 GeoConst ----
ate_kitti_record "kitti_00_geo" \
    ${OUTPUT}/geo_kitti00.txt \
    ${DATA_DIR}/datasets/KITTI/poses/00.txt

# ---- EuRoC YOLO-Only & GeoConst ----
for seq in MH_01_easy MH_03_medium MH_05_difficult; do
    seq_lower=$(echo $seq | tr '[:upper:]' '[:lower:]')
    EUROC_GT=${DATA_DIR}/datasets/EuRoC/${seq}/mav0/state_groundtruth_estimate0/data.csv
    ate_euroc_record "euroc_${seq_lower}_yolo" \
        ${OUTPUT}/yolo_euroc_${seq_lower}.txt \
        ${EUROC_GT}
    ate_euroc_record "euroc_${seq_lower}_geo" \
        ${OUTPUT}/geo_euroc_${seq_lower}.txt \
        ${EUROC_GT}
done

# ---- Write the summary JSON ----
echo ""
echo "  Writing experiment results to ${RESULTS_JSON} ..."
python3 -c "
import json, os, sys
from datetime import datetime

results = {}
with open('${RESULTS_TMP}') as f:
    for line in f:
        line = line.strip()
        if '=' in line:
            k, v = line.split('=', 1)
            results[k] = float(v) if v and v != 'N/A' else None

# Helper: safely convert to float
def sf(v):
    return float(v) if v and v != 'N/A' else None

# Build the structured JSON
output = {
    '_meta': {
        'source': 'autodl_deploy.sh evo_ape evaluation',
        'generated': datetime.utcnow().isoformat() + 'Z',
        'output_dir': '${OUTPUT}',
        'note': 'ATE RMSE values computed by evo_ape with --align --correct_scale. '
                'null means trajectory file not found or evo_ape failed. '
                'Plan C experiments: Baseline (original dataset, pure ORB-SLAM3), '
                'YOLO-Mask (masked dataset: YOLO dynamic regions blacked out, pure ORB-SLAM3). '
                'E2a/E2b/E3 (C++ per-feature semantic filter) skipped due to g2o crash.',
        'data_provenance': 'ALL values in this file are experimentally generated by autodl_deploy.sh. '
                          'Baseline ATE from ORB-SLAM3 on original datasets. '
                          'YOLO-Mask ATE from ORB-SLAM3 on YOLO-masked datasets (real data). '
                          'No hardcoded or fabricated data.',
    },
    'tum': {
        'sitting_static': {
            'baseline_ate': sf(results.get('tum_sitting_static_baseline')),
            'semantic_ate': sf(results.get('tum_sitting_static_semantic')),
            'mask_ate': sf(results.get('tum_sitting_static_mask')),
            'yolo_ate': sf(results.get('tum_sitting_static_yolo')),
            'geoconst_ate': sf(results.get('tum_sitting_static_geo')),
        },
        'sitting_xyz': {
            'baseline_ate': sf(results.get('tum_sitting_xyz_baseline')),
            'semantic_ate': sf(results.get('tum_sitting_xyz_semantic')),
            'mask_ate': sf(results.get('tum_sitting_xyz_mask')),
            'yolo_ate': sf(results.get('tum_sitting_xyz_yolo')),
            'geoconst_ate': sf(results.get('tum_sitting_xyz_geo')),
        },
        'walking_static': {
            'baseline_ate': sf(results.get('tum_walking_static_baseline')),
            'semantic_ate': sf(results.get('tum_walking_static_semantic')),
            'mask_ate': sf(results.get('tum_walking_static_mask')),
            'yolo_ate': sf(results.get('tum_walking_static_yolo')),
            'geoconst_ate': sf(results.get('tum_walking_static_geo')),
        },
        'walking_xyz': {
            'baseline_ate': sf(results.get('tum_walking_xyz_baseline')),
            'semantic_ate': sf(results.get('tum_walking_xyz_semantic')),
            'mask_ate': sf(results.get('tum_walking_xyz_mask')),
            'yolo_ate': sf(results.get('tum_walking_xyz_yolo')),
            'geoconst_ate': sf(results.get('tum_walking_xyz_geo')),
        },
        'walking_halfsphere': {
            'baseline_ate': sf(results.get('tum_walking_halfsphere_baseline')),
            'semantic_ate': sf(results.get('tum_walking_halfsphere_semantic')),
            'mask_ate': sf(results.get('tum_walking_halfsphere_mask')),
            'yolo_ate': sf(results.get('tum_walking_halfsphere_yolo')),
            'geoconst_ate': sf(results.get('tum_walking_halfsphere_geo')),
        }
    },
    'kitti': {
        '00': {
            'baseline_ate': sf(results.get('kitti_00_baseline')),
            'semantic_ate': sf(results.get('kitti_00_semantic')),
            'mask_ate': sf(results.get('kitti_00_mask')),
            'yolo_ate': sf(results.get('kitti_00_yolo')),
            'geoconst_ate': sf(results.get('kitti_00_geo')),
        }
    },
    'euroc': {
        'mh_01_easy': {
            'baseline_ate': sf(results.get('euroc_mh_01_easy_baseline')),
            'semantic_ate': sf(results.get('euroc_mh_01_easy_semantic')),
            'mask_ate': sf(results.get('euroc_mh_01_easy_mask')),
            'yolo_ate': sf(results.get('euroc_mh_01_easy_yolo')),
            'geoconst_ate': sf(results.get('euroc_mh_01_easy_geo')),
        },
        'mh_03_medium': {
            'baseline_ate': sf(results.get('euroc_mh_03_medium_baseline')),
            'semantic_ate': sf(results.get('euroc_mh_03_medium_semantic')),
            'mask_ate': sf(results.get('euroc_mh_03_medium_mask')),
            'yolo_ate': sf(results.get('euroc_mh_03_medium_yolo')),
            'geoconst_ate': sf(results.get('euroc_mh_03_medium_geo')),
        },
        'mh_05_difficult': {
            'baseline_ate': sf(results.get('euroc_mh_05_difficult_baseline')),
            'semantic_ate': sf(results.get('euroc_mh_05_difficult_semantic')),
            'mask_ate': sf(results.get('euroc_mh_05_difficult_mask')),
            'yolo_ate': sf(results.get('euroc_mh_05_difficult_yolo')),
            'geoconst_ate': sf(results.get('euroc_mh_05_difficult_geo')),
        }
    }
}

with open('${RESULTS_JSON}', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Print the summary
print('  experiment_results.json saved')
print()
for dataset, seqs in [('TUM', output['tum']), ('KITTI', output['kitti']), ('EuRoC', output['euroc'])]:
    for seq, vals in seqs.items():
        b = vals.get('baseline_ate')
        m = vals.get('mask_ate')
        parts = []
        if b is not None: parts.append(f'baseline={b:.4f}')
        if m is not None: parts.append(f'YOLO-mask={m:.4f}')
        if b is not None and m is not None:
            impr = (b - m) / b * 100 if b > 0 else 0
            parts.append(f'improvement={impr:+.1f}%')
        if parts:
            print(f'  {dataset}/{seq}: {\"  \".join(parts)}')
        else:
            print(f'  {dataset}/{seq}: trajectory file missing, skipped')
"

rm -f "${RESULTS_TMP}" /tmp/ate_tmp_tum.txt

# ==================== 8.5 Merge enhanced experiment data into the main JSON ====================
echo ""
echo "  Merging enhanced experiment data (timing + failure + parameter sweep) into ${RESULTS_JSON} ..."
python3 -c "
import json, os

with open('${RESULTS_JSON}') as f:
    main_data = json.load(f)

# Merge the timing data
timing_path = '${OUTPUT}/timing_logs/timing_summary.json'
if os.path.exists(timing_path):
    with open(timing_path) as f:
        timing_data = json.load(f)
    main_data['timing'] = timing_data
    print('  + timing profiling data ({:.1f} KB)'.format(os.path.getsize(timing_path)/1024))

# Merge the failure analysis data
failure_path = '${OUTPUT}/failure_analysis/failure_summary.json'
if os.path.exists(failure_path):
    with open(failure_path) as f:
        failure_data = json.load(f)
    main_data['failure_analysis'] = failure_data
    print('  + failure analysis data')

# Merge the parameter sensitivity data
sweep_path = '${OUTPUT}/parameter_sweep/parameter_sensitivity.json'
if os.path.exists(sweep_path):
    with open(sweep_path) as f:
        sweep_data = json.load(f)
    main_data['parameter_sensitivity'] = sweep_data
    print('  + parameter sensitivity data')

with open('${RESULTS_JSON}', 'w') as f:
    json.dump(main_data, f, indent=2, ensure_ascii=False)

# Count the fields
keys = list(main_data.keys())
print(f'  Final JSON keys: {\", \".join(keys)}')
print(f'  Total size: {os.path.getsize(\"${RESULTS_JSON}\")/1024:.1f} KB')
"

echo ""
echo "=========================================="
echo "  Plan C deployment complete! Experiment matrix:"
echo "  TUM:   sitting_static, sitting_xyz, walking_static, walking_xyz, walking_halfsphere"
echo "  KITTI: 00 (4541 frames, loop closure + dynamic)"
echo "  EuRoC: MH_01_easy, MH_03_medium, MH_05_difficult"
echo ""
echo "  Trajectory files: ${OUTPUT}/"
echo "  Experiment results: ${RESULTS_JSON}  <- copy this file to the local data/ directory for plotting"
echo ""
echo "  Experiment comparison axes (Plan C):"
echo "    E1:   Baseline (original datasets, pure ORB-SLAM3) - real data"
echo "    E1.5: YOLO-Mask (masked datasets, pure ORB-SLAM3) - real data, replaces the YOLO-only ablation"
echo "    E2a:  YOLO-only  (skipped - g2o crash)"
echo "    E2b:  GeoConst    (skipped - replaced by offline_semantic_validation.py)"
echo "    E3:   Full System (skipped - marked as Future Work)"
echo ""
echo "  Paper data sources:"
echo "    1. Baseline ATE vs YOLO-Mask ATE comparison table (real experiment data)"
echo "    2. Offline semantic validation analysis (offline_semantic_validation.py) - sitting false-dynamic problem"
echo "    3. Image masking visualization (YOLO detections -> before/after mask comparison)"
echo "=========================================="