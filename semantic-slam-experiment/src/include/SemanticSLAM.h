#ifndef SEMANTICSLAM_H
#define SEMANTICSLAM_H

#include "YoloDetector.h"
#include "DynamicFeatureFilter.h"
#include "SemanticWeights.h"

#include <memory>
#include <mutex>
#include <opencv2/core.hpp>

namespace semantic_slam {

class SemanticSLAM {
public:
    SemanticSLAM(
        const YoloDetector::Config& yolo_cfg,
        const DynamicFeatureFilter::Config& filter_cfg,
        const SemanticWeights::Config& weight_cfg);

    ~SemanticSLAM();

    bool initialize();

    bool start();
    void stop();

    void submitFrame(const cv::Mat& image, double timestamp, int64_t frame_id);

    DetectionResult getDetectionResult(int64_t frame_id, double max_delay_ms = 100.0);

    std::vector<bool> filterDynamicFeatures(
        const cv::Mat& current_image,
        const cv::Mat& previous_image,
        const DetectionResult& detection,
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<cv::Point2f>& curr_pts);

    double getMapPointWeight(int coco_class_id) const;

    SemanticWeights& getWeights() { return *_weights; }
    const SemanticWeights& getWeights() const { return *_weights; }

    YoloDetector& getDetector() { return *_detector; }
    DynamicFeatureFilter& getFilter() { return *_filter; }

    bool hasValidDetection() const;

    bool isDynamicFeature(size_t keypoint_idx) const;

    size_t countDynamicFeatures() const;

    double getDynamicFeatureRatio() const;

private:
    std::unique_ptr<YoloDetector>          _detector;
    std::unique_ptr<DynamicFeatureFilter>  _filter;
    std::unique_ptr<SemanticWeights>       _weights;

    mutable std::mutex _dynamic_mutex;
    std::vector<bool> _dynamic_feature_mask;
};

} // namespace semantic_slam

#endif // SEMANTICSLAM_H