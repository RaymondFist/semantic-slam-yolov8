#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace semantic_slam {

struct InstanceMask {
    int class_id = -1;
    float confidence = 0.0f;
    cv::Rect bbox;
    cv::Mat mask;           // binary mask, CV_8UC1
    int category = 0;       // 0=static, 1=potentially_dynamic, 2=dynamic
};

struct DetectionResult {
    double timestamp = 0.0;
    int64_t frame_id = -1;
    std::vector<InstanceMask> instances;
    bool valid = false;
};

class YoloDetector {
public:
    struct Config {
        std::string engine_path;          // TensorRT serialized engine
        std::string onnx_path;            // ONNX model path (fallback)
        std::string class_names_path;     // COCO class names file
        std::string detection_dir;        // Offline JSON detection dir (skips live inference)
        float conf_threshold = 0.45f;
        float nms_threshold = 0.45f;
        int input_width = 640;
        int input_height = 640;
        int cache_frames = 3;             // re-use detections for N frames
    };

    YoloDetector(const Config& config);
    ~YoloDetector();

    bool initialize();

    void start();
    void stop();

    void submitFrame(const cv::Mat& image, double timestamp, int64_t frame_id);

    DetectionResult getLatest(double max_age_ms = 100.0);

    DetectionResult getResult(int64_t frame_id, double max_delay_ms = 200.0);

    bool isRunning() const { return _running.load(); }

    double getLastInferenceTime() const { return _last_inference_ms; }

    static void classifyCategories(std::vector<InstanceMask>& instances);

private:
    void inferenceLoop();

    void preProcess(const cv::Mat& image, cv::Mat& blob);

    DetectionResult loadOfflineDetection(int64_t frame_id, double timestamp,
                                         int64_t seq_idx);

    Config _config;

    cv::dnn::Net _net;

    std::unique_ptr<std::thread> _thread;
    std::atomic<bool> _running{false};
    std::atomic<bool> _initialized{false};
    bool _offline_mode{false};

    std::mutex _input_mutex;
    cv::Mat _current_image;
    double _current_timestamp{0.0};
    int64_t _current_frame_id{-1};
    bool _has_new_frame{false};
    std::condition_variable _input_cv;

    std::mutex _output_mutex;
    DetectionResult _latest_result;
    DetectionResult _cached_result;
    int _cached_frames_remaining{0};

    double _last_inference_ms{0.0};

    // Sequential frame counter for offline mode — matches inference script's
    // sequential idx (0,1,2,...) instead of ORB-SLAM3's mnId which may not
    // be contiguous if extra Frame objects are created internally.
    int64_t _offline_frame_counter{0};
};

} // namespace semantic_slam

#endif // YOLODETECTOR_H