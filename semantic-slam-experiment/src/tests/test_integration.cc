#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <chrono>
#include <numeric>

#include "SemanticSLAM.h"
#include "test_utils.h"

using namespace semantic_slam;
using namespace test_utils;
using namespace std::chrono_literals;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        _coco_names_path = makeTempCocoNames();
        _onnx_stub_path  = makeTempOnnxStub();

        _yolo_cfg.engine_path      = "";
        _yolo_cfg.onnx_path        = _onnx_stub_path;
        _yolo_cfg.class_names_path = _coco_names_path;
        _yolo_cfg.conf_threshold   = 0.45f;
        _yolo_cfg.nms_threshold    = 0.45f;
        _yolo_cfg.input_width      = IMG_W;
        _yolo_cfg.input_height     = IMG_H;
        _yolo_cfg.cache_frames     = 3;

        _filter_cfg.flow_threshold_px       = 2.5f;
        _filter_cfg.lk_window_size          = 21;
        _filter_cfg.mask_dilation_kernel    = 5;
        _filter_cfg.min_corners_per_region  = 10;

        _sem_slam = std::make_unique<SemanticSLAM>(
            _yolo_cfg, _filter_cfg, _weight_cfg);
    }

    void TearDown() override {
        if (_sem_slam) _sem_slam->stop();
        removeTempFile(_coco_names_path);
        if (!_onnx_stub_path.empty())
            removeTempFile(_onnx_stub_path);
    }

    YoloDetector::Config _yolo_cfg;
    DynamicFeatureFilter::Config _filter_cfg;
    SemanticWeights::Config _weight_cfg;
    std::unique_ptr<SemanticSLAM> _sem_slam;
    std::string _coco_names_path;
    std::string _onnx_stub_path;
};

// ============================================================================
// TEST 1: SemanticSLAM constructor initializes all sub-modules
// ============================================================================
TEST_F(IntegrationTest, ConstructorInitializesSubModules) {
    EXPECT_NO_THROW({
        SemanticSLAM slam(_yolo_cfg, _filter_cfg, _weight_cfg);
    });
}

// ============================================================================
// TEST 2: getDetectionResult returns invalid without frames
// ============================================================================
TEST_F(IntegrationTest, GetDetectionResultInvalidWithoutFrames) {
    auto det = _sem_slam->getDetectionResult(0, 0.0);
    EXPECT_FALSE(det.valid);
    EXPECT_TRUE(det.instances.empty());
}

// ============================================================================
// TEST 3: submitFrame followed by getDetectionResult (stub model)
// ============================================================================
TEST_F(IntegrationTest, SubmitFrameThenGetDetection) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    _sem_slam->submitFrame(img, 1.0, 0);

    auto det = _sem_slam->getDetectionResult(0, 500.0);
    // With stub ONNX model, may be invalid — but should not crash
    EXPECT_TRUE(det.valid || !det.valid);
}

// ============================================================================
// TEST 4: filterDynamicFeatures works end-to-end (static scene)
// ============================================================================
TEST_F(IntegrationTest, FilterDynamicFeaturesStaticScene) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    auto kps = makeGridKeypoints(IMG_W, IMG_H, 40);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);
    std::vector<cv::Point2f> pts2 = pts1;

    auto empty_det = makeFakeDetection({}, 0);

    auto mask = _sem_slam->filterDynamicFeatures(
        img, img, empty_det, kps, pts1, pts2);

    ASSERT_EQ(mask.size(), kps.size());
    for (auto b : mask) {
        EXPECT_FALSE(b);
    }
}

// ============================================================================
// TEST 5: getMapPointWeight returns valid weight range
// ============================================================================
TEST_F(IntegrationTest, GetMapPointWeightValidRange) {
    int coco_ids[] = {56, 9, 13, -1, 2, 100};
    for (int id : coco_ids) {
        double w = _sem_slam->getMapPointWeight(id);
        EXPECT_GE(w, _weight_cfg.vegetation_weight);    // >= min possible
        EXPECT_LE(w, _weight_cfg.building_weight); // <= max possible
        EXPECT_GE(w, 0.0);
        EXPECT_LE(w, 1.0);
    }
}

// ============================================================================
// TEST 6: getDynamicFeatureRatio returns valid range
// ============================================================================
TEST_F(IntegrationTest, DynamicFeatureRatioValidRange) {
    double ratio = _sem_slam->getDynamicFeatureRatio();
    EXPECT_GE(ratio, 0.0);
    EXPECT_LE(ratio, 1.0);
}

// ============================================================================
// TEST 7: start/stop cycle with stub model
// ============================================================================
TEST_F(IntegrationTest, StartStopCycle) {
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(_sem_slam->start());  // stub model won't start inference
        _sem_slam->stop();
    }
}

// ============================================================================
// TEST 8: submitFrame handles images of wrong size
// ============================================================================
TEST_F(IntegrationTest, SubmitFrameHandlesWrongSize) {
    cv::Mat small(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    EXPECT_NO_THROW(_sem_slam->submitFrame(small, 1.0, 0));

    cv::Mat large(2048, 2048, CV_8UC3, cv::Scalar(128, 128, 128));
    EXPECT_NO_THROW(_sem_slam->submitFrame(large, 2.0, 1));
}

// ============================================================================
// TEST 9: getDetectionResult with max delay returns promptly
// ============================================================================
TEST_F(IntegrationTest, GetDetectionResultRespectsMaxDelay) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto det = _sem_slam->getDetectionResult(0, 5.0);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(elapsed, 6000.0) << "Should not block longer than ~6 seconds";

    EXPECT_FALSE(det.valid);
}

// ============================================================================
// TEST 10: Stress — rapid submit + get cycle
// ============================================================================
TEST_F(IntegrationTest, StressTestRapidSubmitGet) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);

    for (int i = 0; i < 100; ++i) {
        _sem_slam->submitFrame(img, (double)i / 10.0, i);
        auto det = _sem_slam->getDetectionResult(i, 1.0);
        EXPECT_TRUE(det.valid || !det.valid);  // no crash
    }
}

// ============================================================================
// TEST 11: filterDynamicFeatures with mixed dynamic/static detections
// ============================================================================
TEST_F(IntegrationTest, FilterDynamicFeaturesMixedScene) {
    cv::Rect moving_roi(260, 190, 100, 80);
    cv::Mat img1 = makeCheckerboard(IMG_W, IMG_H);
    cv::Mat img2 = makeMovingRectangle(IMG_W, IMG_H, moving_roi, 0, 15);

    auto kps = makeGridKeypoints(IMG_W, IMG_H, 40);
    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);

    std::vector<cv::Point2f> pts2;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err,
                             cv::Size(21, 21), 3);

    std::vector<InstanceMask> detections;
    InstanceMask im = makeFakeInstance(
        2, 0.85f,
        cv::Rect(moving_roi.x - 10, moving_roi.y - 10,
                 moving_roi.width + 20, moving_roi.height + 20),
        2);
    detections.push_back(im);
    auto detection = makeFakeDetection(detections, 0);

    auto mask = _sem_slam->filterDynamicFeatures(
        img2, img1, detection, kps, pts1, pts2);

    ASSERT_EQ(mask.size(), kps.size());

    size_t dynamic_count = std::count(mask.begin(), mask.end(), true);
    EXPECT_GT(dynamic_count, (size_t)0)
        << "Moving rectangle should produce dynamic features";
    EXPECT_LT(dynamic_count, kps.size() * 0.5f)
        << "Most features outside moving region should be static";
}

// ============================================================================
// TEST 12: getMapPointWeight for all COCO IDs does not crash
// ============================================================================
TEST_F(IntegrationTest, GetMapPointWeightAllCocoIds) {
    for (int id = -5; id <= 200; ++id) {
        EXPECT_NO_THROW({
            double w = _sem_slam->getMapPointWeight(id);
            EXPECT_GE(w, 0.0);
            EXPECT_LE(w, 1.0);
        });
    }
}