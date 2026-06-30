cmake_minimum_required(VERSION 3.10)
project(SemanticSLAM-YOLOv8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Build Options
# ============================================================================
option(USE_TENSORRT      "Build with TensorRT support"             ON)
option(USE_OPENCV_DNN    "Use OpenCV DNN as fallback backend"      ON)
option(ENABLE_BENCHMARKS "Build benchmark/evaluation executables"  ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings"              OFF)

# ============================================================================
# External Dependencies
# ============================================================================
find_package(OpenCV 4.2 REQUIRED COMPONENTS core imgproc imgcodecs
             video dnn calib3d features2d highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")

find_package(Pangolin REQUIRED)

# ============================================================================
# ORB-SLAM3 + Thirdparty (DBoW2, g2o)
# ============================================================================
set(ORB_SLAM3_ROOT "$ENV{ORB_SLAM3_ROOT}" CACHE PATH
    "Path to ORB-SLAM3 source tree or installation")

if(ORB_SLAM3_ROOT AND EXISTS "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so")
     message(STATUS "Using ORB-SLAM3 Thirdparty from source tree: ${ORB_SLAM3_ROOT}")
     include_directories(
         ${ORB_SLAM3_ROOT}/include
         ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2
         ${ORB_SLAM3_ROOT}/Thirdparty/g2o
     )
     set(DBoW2_LIBS ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so)
     set(g2o_LIBS    ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so)
     set(ORB_SLAM3_LIBS ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so)
 elseif(ORB_SLAM3_ROOT)
     message(STATUS "Using ORB-SLAM3 from install: ${ORB_SLAM3_ROOT}")
     include_directories(${ORB_SLAM3_ROOT}/include)
     link_directories(${ORB_SLAM3_ROOT}/lib)
     set(DBoW2_LIBS DBoW2)
     set(g2o_LIBS    g2o)
     set(ORB_SLAM3_LIBS ORB_SLAM3)
 else()
     set(DBoW2_LIBS DBoW2)
     set(g2o_LIBS    g2o)
     set(ORB_SLAM3_LIBS ORB_SLAM3)
 endif()

# ============================================================================
# TensorRT (optional)
# ============================================================================
if(USE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS /usr/include/aarch64-linux-gnu
              /usr/include/x86_64-linux-gnu
              $ENV{TENSORRT_DIR}/include
              ${CMAKE_SOURCE_DIR}/third_party/TensorRT/include)
    find_library(TENSORRT_LIBRARY nvinfer
        PATHS /usr/lib/aarch64-linux-gnu
              /usr/lib/x86_64-linux-gnu
              $ENV{TENSORRT_DIR}/lib)
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
        add_definitions(-DHAS_TENSORRT)
    else()
        set(USE_TENSORRT OFF)
        message(WARNING "TensorRT not found. Using OpenCV DNN fallback.")
    endif()
endif()

# ============================================================================
# Source Files
# ============================================================================
set(SEMANTIC_SOURCES
    src/YoloDetector.cc
    src/DynamicFeatureFilter.cc
    src/SemanticWeights.cc
    src/SemanticSLAM.cc
)

set(SEMANTIC_HEADERS
    include/YoloDetector.h
    include/DynamicFeatureFilter.h
    include/SemanticWeights.h
    include/SemanticSLAM.h
    include/ORBSLAM3Patch.h
)

# ============================================================================
# Library: libSemanticSLAM.so
# ============================================================================
add_library(SemanticSLAM SHARED ${SEMANTIC_SOURCES})

target_include_directories(SemanticSLAM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

if(ORB_SLAM3_ROOT)
    target_include_directories(SemanticSLAM PUBLIC
        ${ORB_SLAM3_ROOT}/include
        ${ORB_SLAM3_ROOT}
        ${ORB_SLAM3_ROOT}/include/CameraModels
        ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2
        ${ORB_SLAM3_ROOT}/Thirdparty/g2o
        ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
    )
endif()

if(USE_TENSORRT)
    target_include_directories(SemanticSLAM PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(SemanticSLAM ${TENSORRT_LIBRARY})
endif()

target_link_libraries(SemanticSLAM
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${DBoW2_LIBS}
    ${g2o_LIBS}
)

target_compile_options(SemanticSLAM PRIVATE
    -Wall -Wextra -O3 -march=native
)

# ============================================================================
# Executables
# ============================================================================
if(ENABLE_BENCHMARKS)
    # Common settings for all benchmark targets
    set(BENCHMARK_LIBS
        SemanticSLAM
        ${ORB_SLAM3_LIBS}
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        ${DBoW2_LIBS}
        ${g2o_LIBS}
    )

    # Unified Benchmark (KITTI + EuRoC + TUM + Ablation)
    add_executable(semantic_slam_benchmark
        benchmark/benchmark_main.cc
        benchmark/kitti_benchmark.cc
        benchmark/euroc_benchmark.cc
        benchmark/tum_benchmark.cc
        benchmark/ablation_runner.cc
    )
    target_link_libraries(semantic_slam_benchmark ${BENCHMARK_LIBS})

    # Suppress warnings from ORB-SLAM3 headers (deprecated Eigen APIs, etc.)
    set(BENCHMARK_COMPILE_OPTIONS
        -Wno-deprecated-declarations
        -Wno-maybe-uninitialized
        -Wno-reorder
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-unused-local-typedefs
        -Wno-dangling-else
    )
    foreach(tgt semantic_slam_benchmark)
        target_compile_options(${tgt} PRIVATE ${BENCHMARK_COMPILE_OPTIONS})
        if(ORB_SLAM3_ROOT)
            target_include_directories(${tgt} PRIVATE
                ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
                ${ORB_SLAM3_ROOT}/include/CameraModels
            )
        endif()
    endforeach()
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS SemanticSLAM
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES ${SEMANTIC_HEADERS}
    DESTINATION include/semantic_slam
)

# ============================================================================
# Tests
# ============================================================================
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "===========================================")
message(STATUS "SemanticSLAM-YOLOv8 Configuration Summary")
message(STATUS "  TensorRT:    ${USE_TENSORRT}")
message(STATUS "  Benchmarks:  ${ENABLE_BENCHMARKS}")
message(STATUS "  ORB-SLAM3:   ${ORB_SLAM3_ROOT}")
message(STATUS "===========================================")