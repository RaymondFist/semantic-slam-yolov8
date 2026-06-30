#include "SemanticSLAM.h"
#include <cstdio>
#include <iostream>

namespace semantic_slam {

SemanticSLAM::SemanticSLAM(
    const YoloDetector::Config& yolo_cfg,
    const DynamicFeatureFilter::Config& filter_cfg,
    const SemanticWeights::Config& weight_cfg) {

    _detector.reset(new YoloDetector(yolo_cfg));
    _filter.reset(new DynamicFeatureFilter(filter_cfg));
    _weights.reset(new SemanticWeights(weight_cfg));
}

SemanticSLAM::~SemanticSLAM() {
    stop();
}

bool SemanticSLAM::initialize() {
    if (!_detector->initialize()) {
        std::cerr << "[SemanticSLAM] Failed to initialize YOLO detector." << std::endl;
        return false;
    }
    std::cout << "[SemanticSLAM] All modules initialized." << std::endl;
    return true;
}

bool SemanticSLAM::start() {
    if (!_detector) return false;
    _detector->start();
    std::cout << "[SemanticSLAM] System started." << std::endl;
    return true;
}

void SemanticSLAM::stop() {
    if (_detector) _detector->stop();
    std::cout << "[SemanticSLAM] System stopped." << std::endl;
}

void SemanticSLAM::submitFrame(const cv::Mat& image, double timestamp, int64_t frame_id) {
    if (_detector && _detector->isRunning()) {
        _detector->submitFrame(image, timestamp, frame_id);
    }
}

DetectionResult SemanticSLAM::getDetectionResult(int64_t frame_id, double max_delay_ms) {
    if (!_detector) {
        DetectionResult empty; empty.valid = false; return empty;
    }

    if (max_delay_ms > 0.0) {
        return _detector->getResult(frame_id, max_delay_ms);
    } else {
        return _detector->getLatest();
    }
}

std::vector<bool> SemanticSLAM::filterDynamicFeatures(
    const cv::Mat& current_image,
    const cv::Mat& previous_image,
    const DetectionResult& detection,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<cv::Point2f>& curr_pts) {

    fprintf(stderr, "[DEBUG_SLAM] filterDynamicFeatures: calling processFrameMasked "
            "(kp=%zu, det_inst=%zu)\n",
            keypoints.size(), detection.instances.size());

    auto mask = _filter->processFrameMasked(
        current_image, previous_image, detection, keypoints, prev_pts, curr_pts);

    fprintf(stderr, "[DEBUG_SLAM] filterDynamicFeatures: processFrameMasked returned "
            "(mask_size=%zu)\n", mask.size());

    {
        std::lock_guard<std::mutex> lock(_dynamic_mutex);
        _dynamic_feature_mask = mask;
    }

    return mask;
}

double SemanticSLAM::getMapPointWeight(int coco_class_id) const {
    return _weights->getWeight(coco_class_id);
}

bool SemanticSLAM::hasValidDetection() const {
    if (!_detector) return false;
    return _detector->getLatest().valid;
}

bool SemanticSLAM::isDynamicFeature(size_t keypoint_idx) const {
    std::lock_guard<std::mutex> lock(_dynamic_mutex);
    if (keypoint_idx >= _dynamic_feature_mask.size()) return false;
    return _dynamic_feature_mask[keypoint_idx];
}

size_t SemanticSLAM::countDynamicFeatures() const {
    std::lock_guard<std::mutex> lock(_dynamic_mutex);
    return std::count(_dynamic_feature_mask.begin(), _dynamic_feature_mask.end(), true);
}

double SemanticSLAM::getDynamicFeatureRatio() const {
    std::lock_guard<std::mutex> lock(_dynamic_mutex);
    if (_dynamic_feature_mask.empty()) return 0.0;
    return (double)std::count(_dynamic_feature_mask.begin(),
                              _dynamic_feature_mask.end(), true)
           / (double)_dynamic_feature_mask.size();
}

} // namespace semantic_slam