cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Google Test
# ============================================================================
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs
             video calib3d features2d)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ============================================================================
# SemanticSLAM shared library (if not built from parent)
# ============================================================================
set(SEMANTIC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/YoloDetector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/DynamicFeatureFilter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/SemanticWeights.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/SemanticSLAM.cc
)

set(SEMANTIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/YoloDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/DynamicFeatureFilter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SemanticWeights.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/SemanticSLAM.h
)

add_library(SemanticSLAM_obj OBJECT ${SEMANTIC_SOURCES})

target_include_directories(SemanticSLAM_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(SemanticSLAM_obj
    ${OpenCV_LIBS}
)

# ============================================================================
# Test Targets
# ============================================================================
set(TEST_LIBS
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)

function(add_semantic_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME}
        SemanticSLAM_obj
        ${TEST_LIBS}
    )
    target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -g)
    gtest_discover_tests(${TEST_NAME}
        TEST_PREFIX "${TEST_NAME}:"
        DISCOVERY_TIMEOUT 30
    )
endfunction()

add_semantic_test(test_yolo_detector     test_yolo_detector.cc)
add_semantic_test(test_dynamic_filter    test_dynamic_filter.cc)
add_semantic_test(test_semantic_weights  test_semantic_weights.cc)
add_semantic_test(test_integration       test_integration.cc)

# ============================================================================
# Code Coverage (GCC/Clang)
# ============================================================================
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(SemanticSLAM_obj PRIVATE -O0 -fprofile-arcs -ftest-coverage)
        target_link_options(SemanticSLAM_obj INTERFACE --coverage)
        add_custom_target(coverage
            COMMAND lcov --capture --directory . --output-file coverage.info
            COMMAND lcov --remove coverage.info '/usr/*' '*/tests/*' '*/third_party/*' --output-file coverage_filtered.info
            COMMAND genhtml coverage_filtered.info --output-directory coverage_report
            COMMENT "Generating code coverage report..."
        )
        message(STATUS "Coverage target: make coverage")
    endif()
endif()

message(STATUS "===========================================")
message(STATUS "Test Configuration Summary")
message(STATUS "  YoloDetector:       test_yolo_detector")
message(STATUS "  DynamicFilter:      test_dynamic_filter")
message(STATUS "  SemanticWeights:    test_semantic_weights")
message(STATUS "  Integration:        test_integration")
message(STATUS "===========================================")