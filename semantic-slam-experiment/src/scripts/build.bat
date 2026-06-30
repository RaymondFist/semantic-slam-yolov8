@echo off
REM =============================================================================
REM Windows Build Script for SemanticSLAM-YOLOv8
REM =============================================================================
REM Prerequisites:
REM   1. Visual Studio 2019/2022 with C++ CMake tools
REM   2. vcpkg or manual installation of: OpenCV, Eigen3, Pangolin, DBoW2, g2o
REM   3. ORB-SLAM3 source built and installed
REM   4. CUDA Toolkit 11.8+ and TensorRT 8.6+
REM =============================================================================
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set BUILD_DIR=%SCRIPT_DIR%..\build

if not defined ORB_SLAM3_ROOT set ORB_SLAM3_ROOT=C:\libs\ORB_SLAM3

echo ===================================================
echo SemanticSLAM-YOLOv8 Build (Windows)
echo ===================================================
echo ORB_SLAM3: %ORB_SLAM3_ROOT%

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DORB_SLAM3_ROOT="%ORB_SLAM3_ROOT%" ^
    -DUSE_TENSORRT=ON ^
    -DENABLE_BENCHMARKS=ON

if %errorlevel% neq 0 (
    echo CMake configuration failed.
    exit /b 1
)

cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

echo ===================================================
echo Build complete.
echo   %BUILD_DIR%\Release\SemanticSLAM.dll
echo ===================================================