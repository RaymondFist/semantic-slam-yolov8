#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <numeric>

#include "DynamicFeatureFilter.h"
#include "test_utils.h"

using namespace semantic_slam;
using namespace test_utils;

class DynamicFeatureFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        _filter_cfg.flow_threshold_px  = 2.5f;
        _filter_cfg.lk_window_size     = 21;
        _filter_cfg.lk_max_level       = 3;
        _filter_cfg.min_corners_per_region = 10;
        _filter_cfg.mask_dilation_kernel = 5;

        _filter = std::make_unique<DynamicFeatureFilter>(_filter_cfg);
    }

    DynamicFeatureFilter::Config _filter_cfg;
    std::unique_ptr<DynamicFeatureFilter> _filter;

    cv::Mat _prev_img;
    cv::Mat _curr_img;
};

// ============================================================================
// TEST 1: Config defaults are reasonable
// ============================================================================
TEST_F(DynamicFeatureFilterTest, ConfigDefaultsReasonable) {
    DynamicFeatureFilter::Config cfg;
    EXPECT_GT(cfg.flow_threshold_px, 0.0f);
    EXPECT_GT(cfg.lk_window_size, 0);
    EXPECT_GT(cfg.lk_max_level, 0);
    EXPECT_GT(cfg.min_corners_per_region, 0);
    EXPECT_GT(cfg.mask_dilation_kernel, 0);
}

// ============================================================================
// TEST 2: computeGlobalFlow returns near-zero for identical images
// ============================================================================
TEST_F(DynamicFeatureFilterTest, ComputeGlobalFlowIdenticalImages) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    auto kps1 = makeGridKeypoints(IMG_W, IMG_H, 40);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps1, pts1);

    std::vector<cv::Point2f> pts2;
    std::vector<unsigned char> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(img, img, pts1, pts2, status, err,
                             cv::Size(21, 21), 3);

    cv::Point2f gf = _filter->computeGlobalFlow(pts1, pts2, status);

    // global flow should be near (0,0) for identical images
    EXPECT_NEAR(gf.x, 0.0f, 0.5f);
    EXPECT_NEAR(gf.y, 0.0f, 0.5f);
}

// ============================================================================
// TEST 3: computeGlobalFlow captures uniform translation
// ============================================================================
TEST_F(DynamicFeatureFilterTest, ComputeGlobalFlowCapturesTranslation) {
    cv::Mat img1 = makeCheckerboard(IMG_W, IMG_H);
    cv::Mat img2 = makeShiftedImage(img1, 5.0f, 0.0f);

    auto kps = makeGridKeypoints(IMG_W, IMG_H, 40);
    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);

    std::vector<cv::Point2f> pts2;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err,
                             cv::Size(21, 21), 3);

    cv::Point2f gf = _filter->computeGlobalFlow(pts1, pts2, status);

    EXPECT_NEAR(gf.x, 5.0f, 1.0f);
    EXPECT_NEAR(gf.y, 0.0f, 1.0f);
}

// ============================================================================
// TEST 4: processFrameMasked returns all false for static scene
// ============================================================================
TEST_F(DynamicFeatureFilterTest, ProcessFrameMaskedStaticScene) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    auto kps = makeGridKeypoints(IMG_W, IMG_H, 40);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);

    std::vector<cv::Point2f> pts2 = pts1;

    auto dummy_det = makeFakeDetection({}, 0);

    auto mask = _filter->processFrameMasked(
        img, img, dummy_det, kps, pts1, pts2);

    ASSERT_EQ(mask.size(), kps.size());
    for (size_t i = 0; i < mask.size(); ++i) {
        EXPECT_FALSE(mask[i]) << "Feature " << i << " incorrectly flagged dynamic";
    }
}

// ============================================================================
// TEST 5: processFrameMasked flags moving region with detections
// ============================================================================
TEST_F(DynamicFeatureFilterTest, ProcessFrameMaskedFlagsMovingRegion) {
    cv::Rect roi(250, 180, 100, 80);

    cv::Mat img1 = makeCheckerboard(IMG_W, IMG_H);
    cv::Mat img2 = makeMovingRectangle(IMG_W, IMG_H, roi, 5, 8);

    auto kps_all = makeGridKeypoints(IMG_W, IMG_H, 40);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps_all, pts1);

    std::vector<cv::Point2f> pts2;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err,
                             cv::Size(21, 21), 3);

    std::vector<InstanceMask> detections;
    {
        InstanceMask im = makeFakeInstance(
            2, 0.9f, cv::Rect(roi.x - 10, roi.y - 10,
                              roi.width + 20, roi.height + 20), 2);
        detections.push_back(im);
    }
    auto detection = makeFakeDetection(detections, 0);

    auto mask = _filter->processFrameMasked(
        img2, img1, detection, kps_all, pts1, pts2);

    ASSERT_EQ(mask.size(), kps_all.size());

    size_t dynamic_count = std::count(mask.begin(), mask.end(), true);
    EXPECT_GT(dynamic_count, (size_t)0)
        << "Expected some features to be flagged as dynamic";

    // Most features outside the moving region should be static
    EXPECT_LT(dynamic_count, kps_all.size() * 0.6f)
        << "Too many features flagged as dynamic";
}

// ============================================================================
// TEST 6: empty detection list produces all static mask
// ============================================================================
TEST_F(DynamicFeatureFilterTest, EmptyDetectionsAllStatic) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    auto kps = makeGridKeypoints(IMG_W, IMG_H, 40);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);
    std::vector<cv::Point2f> pts2 = pts1;

    auto empty_det = makeFakeDetection({}, 0);

    auto mask = _filter->processFrameMasked(
        img, img, empty_det, kps, pts1, pts2);

    ASSERT_EQ(mask.size(), kps.size());
    for (auto b : mask) EXPECT_FALSE(b);
}

// ============================================================================
// TEST 7: isRegionFlowing returns false for insufficient corners
// ============================================================================
TEST_F(DynamicFeatureFilterTest, IsRegionFlowingInsufficientCorners) {
    std::vector<cv::Point2f> few_pts_prev = { {10, 10}, {20, 20} };
    std::vector<cv::Point2f> few_pts_curr = { {10, 10}, {20, 20} };
    std::vector<unsigned char> status = { 1, 1 };
    cv::Point2f gf(0, 0);

    bool flowing = _filter->isRegionFlowing(
        few_pts_prev, few_pts_curr, status, gf);

    EXPECT_FALSE(flowing)
        << "Should return false when corners below threshold";
}

// ============================================================================
// TEST 8: isRegionFlowing returns true for significant motion
// ============================================================================
TEST_F(DynamicFeatureFilterTest, IsRegionFlowingWithSignificantMotion) {
    std::vector<cv::Point2f> pts_prev;
    std::vector<cv::Point2f> pts_curr;
    std::vector<unsigned char> status;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> pos(100.0f, 300.0f);
    std::uniform_real_distribution<float> motion(8.0f, 15.0f); // > 2.5 px

    for (int i = 0; i < 30; ++i) {
        float x = pos(rng), y = pos(rng);
        pts_prev.push_back({x, y});
        pts_curr.push_back({x + motion(rng), y + motion(rng)});
        status.push_back(1);
    }

    cv::Point2f gf(0, 0); // no global flow

    bool flowing = _filter->isRegionFlowing(pts_prev, pts_curr, status, gf);
    EXPECT_TRUE(flowing) << "Should detect significant motion";
}

// ============================================================================
// TEST 9：dilateMask expands mask region
// ============================================================================
TEST_F(DynamicFeatureFilterTest, DilateMaskExpandsRegion) {
    cv::Mat mask = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC1);
    cv::Rect roi(300, 200, 40, 30);
    mask(roi) = 255;

    cv::Mat dilated = DynamicFeatureFilter::dilateMask(mask, 5);

    int original_pixels = cv::countNonZero(mask);
    int dilated_pixels  = cv::countNonZero(dilated);

    EXPECT_GT(dilated_pixels, original_pixels)
        << "Dilated mask should have more non-zero pixels";
    EXPECT_GT(dilated_pixels, original_pixels * 1.5)
        << "Dilated area should be significantly larger";

    // Original region should still be filled
    cv::Mat original_region = dilated(roi);
    int filled_in_dilated = cv::countNonZero(original_region);
    EXPECT_EQ(filled_in_dilated, roi.area());
}

// ============================================================================
// TEST 10: processFrameMasked handles empty keypoints
// ============================================================================
TEST_F(DynamicFeatureFilterTest, ProcessFrameMaskedEmptyKeypoints) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    std::vector<cv::KeyPoint> empty_kps;
    std::vector<cv::Point2f> empty_pts_prev, empty_pts_curr;

    auto empty_det = makeFakeDetection({}, 0);

    std::vector<bool> mask;
    EXPECT_NO_THROW(
        mask = _filter->processFrameMasked(
            img, img, empty_det, empty_kps, empty_pts_prev, empty_pts_curr)
    );
    EXPECT_TRUE(mask.empty());
}

// ============================================================================
// TEST 11: processFrameMasked with invalid detection still safe
// ============================================================================
TEST_F(DynamicFeatureFilterTest, ProcessFrameMaskedInvalidDetection) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    auto kps = makeGridKeypoints(IMG_W, IMG_H, 40);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);
    std::vector<cv::Point2f> pts2 = pts1;

    DetectionResult invalid_det;
    invalid_det.valid = false;

    EXPECT_NO_THROW({
        auto mask = _filter->processFrameMasked(
            img, img, invalid_det, kps, pts1, pts2);
        ASSERT_EQ(mask.size(), kps.size());
        for (auto b : mask) EXPECT_FALSE(b);
    });
}

// ============================================================================
// TEST 12: getDynamicFeatureCount returns zero before processing
// ============================================================================
TEST_F(DynamicFeatureFilterTest, DynamicFeatureCountDefaultsToZero) {
    EXPECT_EQ(_filter->getDynamicFeatureCount(), 0u);
}

// ============================================================================
// TEST 13: stress test with 1000 keypoints
// ============================================================================
TEST_F(DynamicFeatureFilterTest, StressTestLargeKeypoints) {
    cv::Mat img1 = makeCheckerboard(IMG_W, IMG_H);
    cv::Mat img2 = makeCheckerboard(IMG_W, IMG_H);

    auto kps = makeGridKeypoints(IMG_W, IMG_H, 20);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);
    std::vector<cv::Point2f> pts2 = pts1;

    auto empty_det = makeFakeDetection({}, 0);

    auto mask = _filter->processFrameMasked(
        img2, img1, empty_det, kps, pts1, pts2);

    EXPECT_EQ(mask.size(), kps.size());
    for (auto b : mask) EXPECT_FALSE(b);
}

// ============================================================================
// TEST 14: temporal consistency — repeated static frames remain static
// ============================================================================
TEST_F(DynamicFeatureFilterTest, TemporalConsistencyStaticScene) {
    cv::Mat img = makeCheckerboard(IMG_W, IMG_H);
    auto kps = makeGridKeypoints(IMG_W, IMG_H, 40);

    std::vector<cv::Point2f> pts1;
    cv::KeyPoint::convert(kps, pts1);
    std::vector<cv::Point2f> pts2 = pts1;

    auto empty_det = makeFakeDetection({}, 0);

    for (int frame = 0; frame < 5; ++frame) {
        auto mask = _filter->processFrameMasked(
            img, img, empty_det, kps, pts1, pts2);

        ASSERT_EQ(mask.size(), kps.size());
        for (auto b : mask) EXPECT_FALSE(b);
    }

    EXPECT_EQ(_filter->getDynamicFeatureCount(), 0u);
}