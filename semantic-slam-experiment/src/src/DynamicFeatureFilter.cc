#include "DynamicFeatureFilter.h"
#include <chrono>
#include <cstdio>
#include <iostream>

namespace semantic_slam {

DynamicFeatureFilter::DynamicFeatureFilter(const Config& config) : _config(config) {}

DynamicFeatureFilter::~DynamicFeatureFilter() {}



cv::Mat DynamicFeatureFilter::dilateMask(const cv::Mat& mask, int kernel_size) {
    if (mask.empty()) return cv::Mat();
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(2 * kernel_size + 1, 2 * kernel_size + 1));
    cv::Mat dilated;
    cv::dilate(mask, dilated, kernel);
    return dilated;
}

cv::Point2f DynamicFeatureFilter::computeGlobalFlow(
    const std::vector<cv::Point2f>& pts_prev,
    const std::vector<cv::Point2f>& pts_curr,
    const std::vector<unsigned char>& status) {

    std::vector<double> dx, dy;
    dx.reserve(pts_prev.size());
    dy.reserve(pts_prev.size());

    for (size_t i = 0; i < pts_prev.size(); ++i) {
        // SEMANTIC_SLAM_FIX: Bounds check on status and pts_curr
        if (i >= status.size() || i >= pts_curr.size()) break;
        if (!status[i]) continue;
        dx.push_back((double)(pts_curr[i].x - pts_prev[i].x));
        dy.push_back((double)(pts_curr[i].y - pts_prev[i].y));
    }

    if (dx.empty()) return {0.f, 0.f};

    size_t mid = dx.size() / 2;
    std::nth_element(dx.begin(), dx.begin() + mid, dx.end());
    std::nth_element(dy.begin(), dy.begin() + mid, dy.end());

    return {(float)dx[mid], (float)dy[mid]};
}

bool DynamicFeatureFilter::isRegionFlowing(
    const std::vector<cv::Point2f>& region_pts_prev,
    const std::vector<cv::Point2f>& region_pts_curr,
    const std::vector<unsigned char>& status,
    const cv::Point2f& global_flow) {

    if (region_pts_prev.size() < (size_t)_config.min_corners_per_region) {
        return false;
    }

    std::vector<double> deviations;
    deviations.reserve(region_pts_prev.size());
    size_t valid = 0;

    for (size_t i = 0; i < region_pts_prev.size(); ++i) {
        // SEMANTIC_SLAM_FIX: Bounds check on status and region_pts_curr
        if (i >= status.size() || i >= region_pts_curr.size()) break;
        if (!status[i]) continue;
        float dx = region_pts_curr[i].x - region_pts_prev[i].x - global_flow.x;
        float dy = region_pts_curr[i].y - region_pts_prev[i].y - global_flow.y;
        deviations.push_back(dx * dx + dy * dy);
        valid++;
    }

    if (valid < (size_t)_config.min_corners_per_region) return false;

    // SEMANTIC_SLAM_FIX: Guard against empty deviations (e.g. all optical flow
    // points failed). Accessing deviations[0] on an empty vector is UB and can
    // cause SIGSEGV.
    if (deviations.empty()) return false;

    size_t mid = deviations.size() / 2;
    std::nth_element(deviations.begin(), deviations.begin() + mid, deviations.end());

    double median_sq = deviations[mid];
    double threshold_sq = _config.flow_threshold_px * _config.flow_threshold_px;

    return median_sq > threshold_sq;
}

static bool isInsideMask(int x, int y, const cv::Rect& bbox, const cv::Mat& mask) {
    // If pixel-level mask is available, use it
    if (!mask.empty()) {
        if (x < 0 || x >= mask.cols || y < 0 || y >= mask.rows) return false;
        return mask.at<uchar>(y, x) > 128;
    }
    // Fallback: use bounding box
    return bbox.contains(cv::Point(x, y));
}

std::vector<bool> DynamicFeatureFilter::processFrameMasked(
    const cv::Mat& current_image,
    const cv::Mat& previous_image,
    const DetectionResult& detection,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::Point2f>& tracked_points_prev,
    const std::vector<cv::Point2f>& tracked_points_curr) {

    fprintf(stderr, "[DEBUG_DFF] processFrameMasked entry: kp=%zu det_inst=%zu flow_th=%f "
            "curr=%dx%d prev=%dx%d\n",
            keypoints.size(), detection.instances.size(), _config.flow_threshold_px,
            current_image.cols, current_image.rows,
            previous_image.cols, previous_image.rows);

    // SEMANTIC_SLAM_FIX: Defensive guard — empty keypoints or invalid images
    if (keypoints.empty()) {
        fprintf(stderr, "[DEBUG_DFF] processFrameMasked: empty keypoints, returning\n");
        return std::vector<bool>();
    }
    if (current_image.empty() || previous_image.empty()) {
        fprintf(stderr, "[DEBUG_DFF] processFrameMasked: empty images, returning\n");
        return std::vector<bool>(keypoints.size(), false);
    }
    if (!detection.valid || detection.instances.empty()) {
        fprintf(stderr, "[DEBUG_DFF] processFrameMasked: no valid detections, returning\n");
        return std::vector<bool>(keypoints.size(), false);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // SEMANTIC_SLAM_FIX: YOLO-only fast path — when flow_threshold_px == 0,
    // skip optical flow entirely and use only YOLO category for dynamic detection.
    // This avoids unnecessary/costly optical flow computation and prevents
    // potential crashes from lk_max_level=0 edge cases.
    if (_config.flow_threshold_px == 0.0f) {
        fprintf(stderr, "[DEBUG_DFF] YOLO-only fast path (flow_th=%f)\n", _config.flow_threshold_px);
        std::vector<bool> dynamic(keypoints.size(), false);
        for (const auto& det : detection.instances) {
            if (det.category == 0) continue;
            if (det.bbox.x < 0 || det.bbox.y < 0 ||
                det.bbox.x + det.bbox.width > current_image.cols ||
                det.bbox.y + det.bbox.height > current_image.rows) {
                continue;
            }
            for (size_t i = 0; i < keypoints.size(); ++i) {
                if (isInsideMask((int)keypoints[i].pt.x, (int)keypoints[i].pt.y,
                                 det.bbox, det.mask)) {
                    // YOLO-only: mark as dynamic if category is dynamic (person=0, vehicle=2, etc.)
                    // or if the category is anything other than static background
                    if (det.category == 2 || det.category == 3) {
                        dynamic[i] = true;
                    }
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        _last_processing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        _dynamic_frame_dynamic_count = std::count(dynamic.begin(), dynamic.end(), true);
        fprintf(stderr, "[DEBUG_DFF] YOLO-only done: dynamic_count=%zu time=%.2fms\n",
                _dynamic_frame_dynamic_count, _last_processing_ms);
        return dynamic;
    }

    fprintf(stderr, "[DEBUG_DFF] Optical flow path (flow_th=%f > 0)\n", _config.flow_threshold_px);

    cv::Mat curr_gray, prev_gray;
    if (current_image.channels() == 3)
        cv::cvtColor(current_image, curr_gray, cv::COLOR_BGR2GRAY);
    else
        curr_gray = current_image.clone();
    if (previous_image.channels() == 3)
        cv::cvtColor(previous_image, prev_gray, cv::COLOR_BGR2GRAY);
    else
        prev_gray = previous_image.clone();

    // SEMANTIC_SLAM_FIX: Guard against mismatched image sizes (e.g. resolution change)
    if (curr_gray.size() != prev_gray.size()) {
        return std::vector<bool>(keypoints.size(), false);
    }

    // Initialize tracking points from current frame's keypoints
    std::vector<cv::Point2f> pts_curr;  // positions in current frame
    std::vector<cv::Point2f> pts_prev;  // positions in previous frame (to be computed)

    if (!tracked_points_prev.empty() && !tracked_points_curr.empty() &&
        tracked_points_prev.size() == tracked_points_curr.size()) {
        // Caller provided pre-tracked points: prev positions -> curr positions
        pts_prev = tracked_points_prev;
        pts_curr = tracked_points_curr;
    } else {
        // No pre-tracked points: use backward optical flow
        // Track current keypoints backward to find their positions in the previous frame
        pts_curr.resize(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            pts_curr[i] = keypoints[i].pt;
        }
        pts_prev = pts_curr;  // initial guess for previous positions
    }

    std::vector<unsigned char> lk_status;
    std::vector<float> lk_err;

    if (!tracked_points_prev.empty() && !tracked_points_curr.empty() &&
        tracked_points_prev.size() == tracked_points_curr.size()) {
        // Caller provided pre-tracked points — skip optical flow computation
        lk_status.resize(pts_curr.size(), 1);
    } else {
        // SEMANTIC_SLAM_FIX: Boundary check — ensure all keypoints are within image bounds
        // before calling calcOpticalFlowPyrLK, which can crash on out-of-bounds points
        for (size_t i = 0; i < pts_curr.size(); ++i) {
            if (pts_curr[i].x < 0 || pts_curr[i].x >= curr_gray.cols ||
                pts_curr[i].y < 0 || pts_curr[i].y >= curr_gray.rows) {
                return std::vector<bool>(keypoints.size(), false);
            }
        }

        // Compute optical flow: backward from current to previous frame
        // SEMANTIC_SLAM_FIX: Use square window (lk_window_size x lk_window_size),
        // NOT (lk_window_size x lk_max_level). The old code accidentally used
        // lk_max_level as the window height, which created invalid windows:
        //   - YOLO-only (lk_max_level=0): Size(21,0) → SIGSEGV
        //   - Semantic   (lk_max_level=3): Size(21,3) → abnormally narrow
        try {
            cv::calcOpticalFlowPyrLK(curr_gray, prev_gray, pts_curr, pts_prev,
                                     lk_status, lk_err,
                                     cv::Size(_config.lk_window_size, _config.lk_window_size),
                                     _config.lk_max_level);
        } catch (const cv::Exception& e) {
            std::cerr << "[DynamicFeatureFilter] Optical flow failed: "
                      << e.what() << " — returning all static" << std::endl;
            return std::vector<bool>(keypoints.size(), false);
        }
    }

    // SEMANTIC_SLAM_FIX: Verify optical flow output sizes are consistent.
    // If calcOpticalFlowPyrLK silently failed (e.g., empty output), skip filtering.
    if (lk_status.size() != pts_curr.size() || pts_prev.size() != pts_curr.size()) {
        std::cerr << "[DynamicFeatureFilter] Optical flow output size mismatch: "
                  << "status=" << lk_status.size() << " prev=" << pts_prev.size()
                  << " curr=" << pts_curr.size() << " — returning all static" << std::endl;
        return std::vector<bool>(keypoints.size(), false);
    }

    // Global flow = median(pts_curr - pts_prev) = forward displacement
    cv::Point2f global_flow = computeGlobalFlow(pts_prev, pts_curr, lk_status);

    std::vector<bool> dynamic(keypoints.size(), false);

    for (const auto& det : detection.instances) {
        if (det.category == 0) continue;
        // SEMANTIC_SLAM_FIX: Skip detections with invalid (out-of-bounds) bbox
        if (det.bbox.x < 0 || det.bbox.y < 0 ||
            det.bbox.x + det.bbox.width > curr_gray.cols ||
            det.bbox.y + det.bbox.height > curr_gray.rows) {
            continue;
        }

        std::vector<cv::Point2f> region_prev, region_curr;
        std::vector<unsigned char> region_stat;
        std::vector<size_t> indices;

        for (size_t i = 0; i < keypoints.size(); ++i) {
            // SEMANTIC_SLAM_FIX: Bounds-check against pts_prev/pts_curr/lk_status
            // to prevent out-of-bounds access if vector sizes are inconsistent.
            if (i >= pts_prev.size() || i >= pts_curr.size() || i >= lk_status.size()) break;
            if (isInsideMask((int)keypoints[i].pt.x, (int)keypoints[i].pt.y, det.bbox, det.mask)) {
                region_prev.push_back(pts_prev[i]);
                region_curr.push_back(pts_curr[i]);
                region_stat.push_back(lk_status[i]);
                indices.push_back(i);
            }
        }

        if (region_prev.empty()) continue;

        bool flowing = isRegionFlowing(region_prev, region_curr, region_stat, global_flow);
        if (flowing || det.category == 2) {
            for (auto idx : indices) dynamic[idx] = true;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    _last_processing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    _dynamic_frame_dynamic_count = std::count(
        dynamic.begin(), dynamic.end(), true);

    return dynamic;
}

} // namespace semantic_slam