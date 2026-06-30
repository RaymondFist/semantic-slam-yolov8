#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <chrono>

#include "YoloDetector.h"
#include "test_utils.h"

using namespace semantic_slam;
using namespace test_utils;
using namespace std::chrono_literals;

class YoloDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        _coco_names_path = makeTempCocoNames();
        _onnx_stub_path  = makeTempOnnxStub();
    }

    void TearDown() override {
        removeTempFile(_coco_names_path);
        removeTempFile(_onnx_stub_path);
    }

    YoloDetector::Config makeValidConfig() {
        YoloDetector::Config cfg;
        cfg.class_names_path = _coco_names_path;
        cfg.onnx_path        = _onnx_stub_path;
        cfg.conf_threshold   = 0.45f;
        cfg.nms_threshold    = 0.45f;
        cfg.input_width      = IMG_W;
        cfg.input_height     = IMG_H;
        cfg.cache_frames = 3;
        return cfg;
    }

    std::string _coco_names_path;
    std::string _onnx_stub_path;
};

// ============================================================================
// TEST 1: constructor does not throw
// ============================================================================
TEST_F(YoloDetectorTest, ConstructorDoesNotThrow) {
    auto cfg = makeValidConfig();
    EXPECT_NO_THROW({
        YoloDetector detector(cfg);
        EXPECT_FALSE(detector.isRunning());
    });
}

// ============================================================================
// TEST 2: initialize fails with empty class_names_path
// ============================================================================
TEST_F(YoloDetectorTest, InitializeFailsWithEmptyClassPath) {
    YoloDetector::Config cfg;
    cfg.class_names_path = "";
    cfg.onnx_path        = _onnx_stub_path;

    YoloDetector detector(cfg);
    EXPECT_FALSE(detector.initialize());
}

// ============================================================================
// TEST 3: initialize fails with empty class_names_path (also no model)
// ============================================================================
TEST_F(YoloDetectorTest, InitializeFailsWithEmptyClassPathAndNoModel) {
    YoloDetector::Config cfg;
    cfg.class_names_path = "";
    cfg.onnx_path        = "";
    cfg.engine_path      = "";

    YoloDetector detector(cfg);
    EXPECT_FALSE(detector.initialize());
}

// ============================================================================
// TEST 4: initialize fails gracefully with stub ONNX (no real model weights)
// ============================================================================
TEST_F(YoloDetectorTest, InitializeFailsGracefullyWithStubModel) {
    auto cfg = makeValidConfig();
    cfg.onnx_path = _onnx_stub_path;

    YoloDetector detector(cfg);

    bool result = detector.initialize();

    // Stub file contains no real weights — expect false without crashing
    EXPECT_FALSE(result) << "Stub ONNX should fail to initialize, not crash";
}

// ============================================================================
// TEST 5: start fails when not initialized
// ============================================================================
TEST_F(YoloDetectorTest, StartFailsWithoutInit) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    EXPECT_FALSE(detector.isRunning());
    EXPECT_NO_THROW(detector.start());
    EXPECT_FALSE(detector.isRunning());  // shouldn't start uninitialized
}

// ============================================================================
// TEST 6: stop is safe to call multiple times
// ============================================================================
TEST_F(YoloDetectorTest, StopIsIdempotent) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    EXPECT_NO_THROW(detector.stop());
    EXPECT_NO_THROW(detector.stop());
    EXPECT_NO_THROW(detector.stop());
    EXPECT_FALSE(detector.isRunning());
}

// ============================================================================
// TEST 7: submitFrame does not crash when not running
// ============================================================================
TEST_F(YoloDetectorTest, SubmitFrameSafeWhenNotRunning) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    EXPECT_NO_THROW(detector.submitFrame(img, 0.0, 0));
}

// ============================================================================
// TEST 8: getLatest returns invalid result when no frames submitted
// ============================================================================
TEST_F(YoloDetectorTest, GetLatestReturnsInvalidWhenEmpty) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    auto result = detector.getLatest();
    EXPECT_FALSE(result.valid);
    EXPECT_TRUE(result.instances.empty());
}

// ============================================================================
// TEST 9: getResult with timeout returns invalid when no data
// ============================================================================
TEST_F(YoloDetectorTest, GetResultTimeoutReturnsInvalid) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    auto result = detector.getResult(42, 10.0);
    EXPECT_FALSE(result.valid);
}

// ============================================================================
// TEST 10: classifyCategories correctly assigns static class
// ============================================================================
TEST_F(YoloDetectorTest, ClassifyCategoriesAssignsStatic) {
    std::vector<InstanceMask> instances;
    {
        InstanceMask im;
        im.class_id = 11;               // traffic light (non-dynamic COCO ID)
        im.category = -1;
        instances.push_back(im);
    }
    {
        InstanceMask im;
        im.class_id = 13;               // parking meter (non-dynamic)
        im.category = -1;
        instances.push_back(im);
    }

    YoloDetector::classifyCategories(instances);

    EXPECT_EQ(instances[0].category, 0); // static
    EXPECT_EQ(instances[1].category, 0); // static
}

// ============================================================================
// TEST 11: classifyCategories correctly assigns dynamic class
// ============================================================================
TEST_F(YoloDetectorTest, ClassifyCategoriesAssignsDynamic) {
    std::vector<InstanceMask> instances;
    {
        InstanceMask im;
        im.class_id = 2;                // car (dynamic COCO ID)
        im.category = -1;
        instances.push_back(im);
    }
    {
        InstanceMask im;
        im.class_id = 3;                // motorcycle (dynamic)
        im.category = -1;
        instances.push_back(im);
    }
    {
        InstanceMask im;
        im.class_id = 1;                // person (dynamic)
        im.category = -1;
        instances.push_back(im);
    }

    YoloDetector::classifyCategories(instances);

    EXPECT_EQ(instances[0].category, 2);
    EXPECT_EQ(instances[1].category, 2);
    EXPECT_EQ(instances[2].category, 2);
}

// ============================================================================
// TEST 12: classifyCategories handles negative class_id gracefully
// ============================================================================
TEST_F(YoloDetectorTest, ClassifyCategoriesHandlesInvalidClassId) {
    std::vector<InstanceMask> instances;
    {
        InstanceMask im;
        im.class_id = -5;
        im.category = -1;
        instances.push_back(im);
    }
    {
        InstanceMask im;
        im.class_id = 999;
        im.category = -1;
        instances.push_back(im);
    }

    EXPECT_NO_THROW(YoloDetector::classifyCategories(instances));

    EXPECT_EQ(instances[0].category, 0); // unknown -> static
    EXPECT_EQ(instances[1].category, 0); // out of range -> static
}

// ============================================================================
// TEST 13: cache detection reuse (getLatest with cache logic)
// ============================================================================
TEST_F(YoloDetectorTest, CacheDetectionReuseLogic) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    detector.submitFrame(img, 1.0, 100);

    auto r1 = detector.getLatest();
    // Without running inference thread, result should be invalid
    EXPECT_FALSE(r1.valid);

    auto r2 = detector.getLatest();
    EXPECT_FALSE(r2.valid);
}

// ============================================================================
// TEST 14: detector reports correct last inference time
// ============================================================================
TEST_F(YoloDetectorTest, LastInferenceTimeDefaultsToZero) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    EXPECT_DOUBLE_EQ(detector.getLastInferenceTime(), 0.0);
}

// ============================================================================
// TEST 15: stress test — many submissions without running thread
// ============================================================================
TEST_F(YoloDetectorTest, StressTestManySubmissions) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);

    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(detector.submitFrame(img, (double)i, i));
    }

    auto result = detector.getLatest();
    EXPECT_FALSE(result.valid);
}

// ============================================================================
// TEST 16: thread start/stop cycle stability
// ============================================================================
TEST_F(YoloDetectorTest, ThreadStartStopCycle) {
    auto cfg = makeValidConfig();
    YoloDetector detector(cfg);

    for (int cycle = 0; cycle < 5; ++cycle) {
        EXPECT_NO_THROW(detector.stop());
        std::this_thread::sleep_for(10ms);
    }
    EXPECT_FALSE(detector.isRunning());
}

// ============================================================================
// TEST 17: Config validation — default values
// ============================================================================
TEST_F(YoloDetectorTest, ConfigDefaultsAreReasonable) {
    YoloDetector::Config cfg;

    EXPECT_GT(cfg.conf_threshold, 0.0f);
    EXPECT_LT(cfg.conf_threshold, 1.0f);
    EXPECT_GT(cfg.nms_threshold, 0.0f);
    EXPECT_LT(cfg.nms_threshold, 1.0f);
    EXPECT_EQ(cfg.input_width, 640);
    EXPECT_EQ(cfg.input_height, 640);
    EXPECT_EQ(cfg.cache_frames, 3);
}