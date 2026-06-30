#include "YoloDetector.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace semantic_slam {

// NOTE: Keep in sync with scripts/yolov8_offline_inference.py: DYNAMIC_COCO_IDS
const std::array<int, 8> DYNAMIC_COCO_IDS   = {0, 1, 2, 3, 5, 7, 16, 17};  // person, bicycle, car, motorcycle, bus, truck, dog, cat
const std::array<int, 3>  POTENTIAL_COCO_IDS = {58, 62, 26};  // potted plant, tv, umbrella

static bool inArray(int id, const int* arr, size_t n) {
    for (size_t i = 0; i < n; ++i) if (arr[i] == id) return true;
    return false;
}

// FIX: Deep-clone a DetectionResult so that returned cv::Mat masks are
// independent of the internally cached result. Prevents use-after-free
// when the YOLO thread overwrites _latest_result while the main thread
// is still reading the previously returned DetectionResult.
static DetectionResult cloneDetectionResult(const DetectionResult& src) {
    DetectionResult dst = src;
    for (auto& inst : dst.instances) {
        if (!inst.mask.empty()) {
            inst.mask = inst.mask.clone();
        }
    }
    return dst;
}

YoloDetector::YoloDetector(const Config& config) : _config(config) {}

YoloDetector::~YoloDetector() { stop(); }

bool YoloDetector::initialize() {
    if (_config.class_names_path.empty()) {
        std::cerr << "[YoloDetector] class_names_path is required." << std::endl;
        return false;
    }

    // Offline JSON mode: skip model loading entirely
    if (!_config.detection_dir.empty()) {
        _offline_mode = true;
        std::cout << "[YoloDetector] Offline JSON mode: " << _config.detection_dir << std::endl;
        _initialized.store(true);
        std::cout << "[YoloDetector] Initialized (offline)." << std::endl;
        return true;
    }

    try {
        cv::dnn::Net net;

        if (!_config.engine_path.empty()) {
            net = cv::dnn::readNet(_config.engine_path);
            std::cout << "[YoloDetector] Loaded TensorRT engine: " << _config.engine_path << std::endl;
        } else if (!_config.onnx_path.empty()) {
            net = cv::dnn::readNetFromONNX(_config.onnx_path);
            std::cout << "[YoloDetector] Loaded ONNX model: " << _config.onnx_path << std::endl;
        } else {
            std::cerr << "[YoloDetector] No model path provided." << std::endl;
            return false;
        }

        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        _net = net;
    } catch (const cv::Exception& e) {
        std::cerr << "[YoloDetector] CUDA backend failed, trying CPU: " << e.what() << std::endl;
        try {
            cv::dnn::Net net;
            if (!_config.onnx_path.empty()) {
                net = cv::dnn::readNetFromONNX(_config.onnx_path);
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                _net = net;
                std::cout << "[YoloDetector] Fallback to CPU OK." << std::endl;
            } else {
                return false;
            }
        } catch (const cv::Exception& e2) {
            std::cerr << "[YoloDetector] Failed to load model: " << e2.what() << std::endl;
            return false;
        }
    }

    _initialized.store(true);
    std::cout << "[YoloDetector] Initialized (live inference)." << std::endl;
    return true;
}

void YoloDetector::start() {
    if (!_initialized.load()) { std::cerr << "[YoloDetector] Not initialized." << std::endl; return; }
    _running.store(true);
    _thread.reset(new std::thread(&YoloDetector::inferenceLoop, this));
}

void YoloDetector::stop() {
    _running.store(false);
    _input_cv.notify_all();
    if (_thread && _thread->joinable()) {
        try { _thread->join(); } catch (...) {}
    }
    _thread.reset();
}

void YoloDetector::submitFrame(const cv::Mat& image, double timestamp, int64_t frame_id) {
    std::lock_guard<std::mutex> lock(_input_mutex);
    image.copyTo(_current_image);
    _current_timestamp = timestamp;
    _current_frame_id = frame_id;
    if (_offline_mode) {
        _offline_frame_counter++;
    }
    _has_new_frame = true;
    _input_cv.notify_one();
}

void YoloDetector::preProcess(const cv::Mat& image, cv::Mat& blob) {
    static const cv::Scalar MEAN(0.0, 0.0, 0.0);
    static const float SCALE = 1.0f / 255.0f;
    cv::dnn::blobFromImage(image, blob, SCALE,
                           cv::Size(_config.input_width, _config.input_height),
                           MEAN, true, false, CV_32F);
}

static std::vector<InstanceMask> processMaskOutput(
    const cv::Mat& proto, const cv::Mat& mask_coeff, const cv::Rect& bbox,
    const cv::Size& orig_size, int class_id, float conf)
{
    std::vector<InstanceMask> results;

    if (proto.empty() || mask_coeff.total() < 32) return results;

    cv::Mat coeff = mask_coeff.reshape(1, 1);
    cv::Mat mask_proto = proto.reshape(1, {proto.size[1], proto.size[0] * proto.size[2]});

    cv::Mat mask = coeff * mask_proto;
    mask = mask.reshape(1, {proto.size[1], proto.size[0]});

    cv::Mat mask_float;
    cv::exp(-mask, mask_float);
    mask_float = 1.0f / (1.0f + mask_float);

    cv::resize(mask_float, mask_float, orig_size, 0, 0, cv::INTER_LINEAR);

    cv::Mat binary;
    cv::threshold(mask_float, binary, 0.5, 255, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1);

    cv::Mat roi = binary(bbox & cv::Rect(0, 0, orig_size.width, orig_size.height));
    if (cv::countNonZero(roi) < 20) return results;

    InstanceMask im;
    im.class_id = class_id;
    im.confidence = conf;
    im.bbox = bbox;
    im.mask = binary.clone();
    results.push_back(im);
    return results;
}

DetectionResult YoloDetector::getLatest(double /*max_age_ms*/) {
    std::lock_guard<std::mutex> lock(_output_mutex);

    if (_cached_frames_remaining > 0) {
        _cached_frames_remaining--;
        DetectionResult cached = _cached_result;
        cached.frame_id = _latest_result.frame_id;
        cached.timestamp = _latest_result.timestamp;
        return cloneDetectionResult(cached);
    }

    return cloneDetectionResult(_latest_result);
}

DetectionResult YoloDetector::getResult(int64_t frame_id, double max_delay_ms) {
    using namespace std::chrono;
    auto deadline = steady_clock::now() + duration<double, std::milli>(max_delay_ms);

    while (steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(_output_mutex);
            if (_latest_result.frame_id >= frame_id && _latest_result.valid) {
                return cloneDetectionResult(_latest_result);
            }
        }
        std::this_thread::sleep_for(milliseconds(1));
    }

    DetectionResult empty;
    empty.valid = false;
    return empty;
}

void YoloDetector::classifyCategories(std::vector<InstanceMask>& instances) {
    for (auto& im : instances) {
        if (im.class_id < 0 || (size_t)im.class_id >= (size_t)80) {
            im.category = 0; // unknown -> treat as static
            continue;
        }

        if (inArray(im.class_id, DYNAMIC_COCO_IDS.data(), DYNAMIC_COCO_IDS.size())) {
            im.category = 2;
        } else if (inArray(im.class_id, POTENTIAL_COCO_IDS.data(), POTENTIAL_COCO_IDS.size())) {
            im.category = 1;
        } else {
            im.category = 0;
        }
    }
}

void YoloDetector::inferenceLoop() {
    // Offline mode: load detections from JSON files
    if (_offline_mode) {
        std::cout << "[YoloDetector] Offline inference thread started." << std::endl;

        while (_running.load()) {
            cv::Mat image;
            double timestamp;
            int64_t frame_id;
            int64_t seq_idx = -1;

            {
                std::unique_lock<std::mutex> lock(_input_mutex);
                _input_cv.wait_for(lock, std::chrono::milliseconds(33),
                                  [this]() { return _has_new_frame || !_running.load(); });

                if (!_running.load()) break;
                if (!_has_new_frame) continue;

                _current_image.copyTo(image);
                timestamp = _current_timestamp;
                frame_id = _current_frame_id;
                // Capture sequential index under lock to avoid race:
                // submitFrame may be called again before we reach loadOfflineDetection,
                // which would increment _offline_frame_counter. Capturing it here
                // ensures we use the correct index for *this* frame.
                seq_idx = _offline_frame_counter - 1;
                _has_new_frame = false;
            }

            // Load detection from JSON
            DetectionResult result = loadOfflineDetection(frame_id, timestamp, seq_idx);

            {
                std::lock_guard<std::mutex> lock(_output_mutex);
                _latest_result = result;
            }
        }

        std::cout << "[YoloDetector] Offline inference thread stopped." << std::endl;
        return;
    }

    std::cout << "[YoloDetector] Inference thread started." << std::endl;

    while (_running.load()) {
        cv::Mat image;
        double timestamp;
        int64_t frame_id;

        {
            std::unique_lock<std::mutex> lock(_input_mutex);
            _input_cv.wait_for(lock, std::chrono::milliseconds(33),
                              [this]() { return _has_new_frame || !_running.load(); });

            if (!_running.load()) break;
            if (!_has_new_frame) continue;

            _current_image.copyTo(image);
            timestamp = _current_timestamp;
            frame_id = _current_frame_id;
            _has_new_frame = false;
        }

        DetectionResult result;
        result.timestamp = timestamp;
        result.frame_id = frame_id;
        result.valid = false;

        auto t0 = std::chrono::high_resolution_clock::now();

        try {
            cv::Mat blob;
            preProcess(image, blob);
            _net.setInput(blob);

            std::vector<cv::Mat> outputs;
            _net.forward(outputs, _net.getUnconnectedOutLayersNames());

            if (outputs.size() >= 2) {
                cv::Mat det_output = outputs[0];
                cv::Mat proto_output = outputs[1];

                cv::Mat det_reshaped = det_output.reshape(1, det_output.size[1]);

                std::vector<cv::Rect> boxes;
                std::vector<float> confidences;
                std::vector<int> class_ids;
                std::vector<cv::Mat> mask_coeffs;

                for (int i = 0; i < det_reshaped.rows; ++i) {
                    float* row = det_reshaped.ptr<float>(i);
                    float obj_conf = row[4];

                    if (obj_conf < _config.conf_threshold) continue;

                    cv::Mat class_scores(1, 80, CV_32F, row + 5);
                    cv::Point max_loc;
                    double max_score;
                    cv::minMaxLoc(class_scores, nullptr, &max_score, nullptr, &max_loc);

                    float final_conf = obj_conf * (float)max_score;
                    if (final_conf < _config.conf_threshold) continue;

                    float cx = row[0], cy = row[1], w = row[2], h = row[3];
                    int left   = (int)((cx - w / 2.0f) * image.cols);
                    int top    = (int)((cy - h / 2.0f) * image.rows);
                    int width  = (int)(w * image.cols);
                    int height = (int)(h * image.rows);

                    left = std::max(0, left); top = std::max(0, top);
                    width = std::min(width, image.cols - left);
                    height = std::min(height, image.rows - top);

                    if (width < 10 || height < 10) continue;

                    cv::Mat mask_coeff(1, 32, CV_32F, row + 6);
                    mask_coeffs.push_back(mask_coeff.clone());

                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(final_conf);
                    class_ids.push_back(max_loc.x);
                }

                std::vector<int> nms_indices;
                cv::dnn::NMSBoxes(boxes, confidences, _config.conf_threshold,
                                 _config.nms_threshold, nms_indices);

                for (int idx : nms_indices) {
                    auto masks = processMaskOutput(proto_output, mask_coeffs[idx],
                                                   boxes[idx], image.size(),
                                                   class_ids[idx], confidences[idx]);
                    if (!masks.empty()) {
                        masks[0].class_id = class_ids[idx];
                        result.instances.insert(result.instances.end(), masks.begin(), masks.end());
                    }
                }
                result.valid = true;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[YoloDetector] Inference error: " << e.what() << std::endl;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        _last_inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        classifyCategories(result.instances);

        {
            std::lock_guard<std::mutex> lock(_output_mutex);
            _latest_result = result;
            _cached_result = result;
            _cached_frames_remaining = _config.cache_frames;
        }
    }

    std::cout << "[YoloDetector] Inference thread stopped." << std::endl;
}

DetectionResult YoloDetector::loadOfflineDetection(int64_t frame_id, double timestamp,
                                                    int64_t seq_idx) {
    DetectionResult result;
    result.timestamp = timestamp;
    result.frame_id = frame_id;
    result.valid = false;

    // Build JSON file path using the caller-provided sequential index.
    // This value was captured under _input_mutex by the inference loop,
    // so it's immune to races from concurrent submitFrame calls.
    char json_path[512];
    snprintf(json_path, sizeof(json_path), "%s/%06lld.json",
             _config.detection_dir.c_str(), (long long)seq_idx);

    std::ifstream f(json_path);
    if (!f.is_open()) {
        // Fallback: try using frame_id (mnId) directly for backward compatibility
        char fallback_path[512];
        snprintf(fallback_path, sizeof(fallback_path), "%s/%06lld.json",
                 _config.detection_dir.c_str(), (long long)frame_id);
        f.open(fallback_path);
        if (!f.is_open()) {
            // No detection file for this frame — return empty but valid result
            result.valid = true;
            return result;
        }
        snprintf(json_path, sizeof(json_path), "%s", fallback_path);
    }

    // Simple JSON parsing (no external library dependency)
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Parse detections array from JSON
    // Using minimal string-based parsing to avoid jsoncpp dependency
    size_t det_start = content.find("\"detections\"");
    if (det_start == std::string::npos) {
        result.valid = true;
        return result;
    }

    size_t arr_start = content.find('[', det_start);
    // Find the matching closing ']' for the detections array (not a nested one like bbox)
    // Use bracket depth counting to handle nested arrays such as "bbox": [x1, y1, x2, y2]
    size_t arr_end = std::string::npos;
    if (arr_start != std::string::npos) {
        int depth = 0;
        for (size_t i = arr_start; i < content.size(); ++i) {
            if (content[i] == '[') depth++;
            else if (content[i] == ']') {
                depth--;
                if (depth == 0) { arr_end = i; break; }
            }
        }
    }
    if (arr_start == std::string::npos || arr_end == std::string::npos) {
        result.valid = true;
        return result;
    }

    std::string det_array = content.substr(arr_start, arr_end - arr_start + 1);

    // Parse each detection object
    size_t pos = 0;
    while ((pos = det_array.find("\"class_id\"", pos)) != std::string::npos) {
        try {
            InstanceMask im;
            im.category = 0;

            // Extract class_id
            size_t cid_pos = det_array.find(':', pos + 10);
            if (cid_pos != std::string::npos) {
                im.class_id = std::stoi(det_array.substr(cid_pos + 1));
            }

            // Extract confidence
            size_t conf_pos = det_array.find("\"confidence\"", pos);
            if (conf_pos != std::string::npos && conf_pos < det_array.size()) {
                size_t cv = det_array.find(':', conf_pos + 12);
                if (cv != std::string::npos) {
                    im.confidence = std::stof(det_array.substr(cv + 1));
                }
            }

            // Extract bbox [x1, y1, x2, y2]
            size_t bbox_pos = det_array.find("\"bbox\"", pos);
            if (bbox_pos != std::string::npos) {
                size_t bstart = det_array.find('[', bbox_pos + 6);
                size_t bend = det_array.find(']', bstart);
                if (bstart != std::string::npos && bend != std::string::npos) {
                    std::string bbox_str = det_array.substr(bstart + 1, bend - bstart - 1);
                    float bbox_vals[4] = {};
                    int bi = 0;
                    std::istringstream bss(bbox_str);
                    std::string val;
                    while (bi < 4 && std::getline(bss, val, ',')) {
                        bbox_vals[bi++] = std::stof(val);
                    }
                    im.bbox = cv::Rect(
                        (int)bbox_vals[0], (int)bbox_vals[1],
                        (int)(bbox_vals[2] - bbox_vals[0]),
                        (int)(bbox_vals[3] - bbox_vals[1]));
                }
            }

            result.instances.push_back(im);
        } catch (const std::exception& e) {
            std::cerr << "[YoloDetector] JSON parse error in " << json_path
                      << ": " << e.what() << " — skipping entry" << std::endl;
        }
        size_t next_brace = det_array.find('}', pos);
        if (next_brace == std::string::npos) break;
        pos = next_brace + 1;
    }

    result.valid = true;
    classifyCategories(result.instances);
    return result;
}

} // namespace semantic_slam