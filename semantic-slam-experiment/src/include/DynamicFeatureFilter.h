#ifndef DYNAMICFEATUREFILTER_H
#define DYNAMICFEATUREFILTER_H

#include "YoloDetector.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace semantic_slam {

class DynamicFeatureFilter {
public:
    struct Config {
        float flow_threshold_px = 2.5f;      // optical flow deviation threshold (pixels)
        int   lk_window_size = 21;           // Lucas-Kanade window size
        int   lk_max_level = 3;              // pyramid levels for LK
        int   min_corners_per_region = 10;   // min corners to trust flow estimate
        int   mask_dilation_kernel = 5;      // morphological dilation for YOLO masks
    };

    DynamicFeatureFilter(const Config& config);
    ~DynamicFeatureFilter();

    std::vector<bool> processFrameMasked(
        const cv::Mat& current_image,
        const cv::Mat& previous_image,
        const DetectionResult& detection,
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<cv::Point2f>& tracked_points_prev,
        const std::vector<cv::Point2f>& tracked_points_curr);

    

    cv::Point2f computeGlobalFlow(
        const std::vector<cv::Point2f>& pts_prev,
        const std::vector<cv::Point2f>& pts_curr,
        const std::vector<unsigned char>& status);

    bool isRegionFlowing(
        const std::vector<cv::Point2f>& region_pts_prev,
        const std::vector<cv::Point2f>& region_pts_curr,
        const std::vector<unsigned char>& status,
        const cv::Point2f& global_flow);

    double getLastProcessingTime() const { return _last_processing_ms; }

    size_t getDynamicFeatureCount() const {
        return _dynamic_frame_dynamic_count;
    }

    static cv::Mat dilateMask(const cv::Mat& mask, int kernel_size);

private:
    Config _config;
    double _last_processing_ms;

    size_t _dynamic_frame_dynamic_count = 0;
};

} // namespace semantic_slam

#endif // DYNAMICFEATUREFILTER_H