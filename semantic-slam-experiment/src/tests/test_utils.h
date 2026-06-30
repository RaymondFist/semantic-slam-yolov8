#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <filesystem>

#include "YoloDetector.h"
#include "DynamicFeatureFilter.h"

namespace test_utils {

const int IMG_W = 640;
const int IMG_H = 480;

inline cv::Mat makeSolidImage(int w, int h, cv::Scalar color = cv::Scalar(128, 128, 128)) {
    cv::Mat img(h, w, CV_8UC3);
    img = color;
    return img;
}

inline cv::Mat makeCheckerboard(int w, int h, int cell_size = 40) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; y += cell_size) {
        for (int x = 0; x < w; x += cell_size) {
            uchar val = ((x / cell_size) + (y / cell_size)) % 2 == 0 ? 255 : 50;
            cv::Rect roi(x, y,
                         std::min(cell_size, w - x),
                         std::min(cell_size, h - y));
            img(roi) = cv::Scalar(val, val, val);
        }
    }
    return img;
}

inline cv::Mat makeMovingRectangle(int w, int h, cv::Rect rect, int frame, int velocity_x) {
    cv::Mat img = makeCheckerboard(w, h);

    cv::Rect moving(rect.x + frame * velocity_x, rect.y,
                    rect.width, rect.height);

    if (moving.x + moving.width > w) moving.x = w - moving.width;
    if (moving.x < 0) moving.x = 0;

    cv::rectangle(img, moving, cv::Scalar(0, 255, 0), -1);
    return img;
}

inline cv::Rect makeDefaultROI() {
    return cv::Rect(200, 150, 120, 100);
}

inline std::vector<cv::KeyPoint> makeGridKeypoints(int w, int h, int spacing = 30) {
    std::vector<cv::KeyPoint> kps;
    for (int y = spacing; y < h - spacing; y += spacing) {
        for (int x = spacing; x < w - spacing; x += spacing) {
            kps.push_back(cv::KeyPoint((float)x, (float)y, 10.0f));
        }
    }
    return kps;
}

inline std::vector<cv::KeyPoint> makeKeypointsInROI(const cv::Rect& roi, int count = 20) {
    std::vector<cv::KeyPoint> kps;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dx(0, (float)roi.width);
    std::uniform_real_distribution<float> dy(0, (float)roi.height);
    for (int i = 0; i < count; ++i) {
        kps.push_back(cv::KeyPoint(
            roi.x + dx(rng), roi.y + dy(rng), 8.0f));
    }
    return kps;
}

inline cv::Mat makeBinaryMask(const cv::Size& size, const cv::Rect& roi) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    cv::Rect clamped(
        std::max(0, roi.x), std::max(0, roi.y),
        std::min(roi.width, size.width - roi.x),
        std::min(roi.height, size.height - roi.y));
    mask(clamped) = 255;
    return mask;
}

inline semantic_slam::InstanceMask makeFakeInstance(
    int class_id, float conf, const cv::Rect& roi, int category) {
    semantic_slam::InstanceMask im;
    im.class_id   = class_id;
    im.confidence = conf;
    im.bbox       = roi;
    im.category   = category;
    im.mask       = makeBinaryMask(cv::Size(IMG_W, IMG_H), roi);
    return im;
}

inline semantic_slam::DetectionResult makeFakeDetection(
    const std::vector<semantic_slam::InstanceMask>& instances,
    int64_t frame_id = 0) {
    semantic_slam::DetectionResult dr;
    dr.timestamp  = 0.0;
    dr.frame_id   = frame_id;
    dr.instances  = instances;
    dr.valid      = !instances.empty();
    return dr;
}

inline cv::Mat makeShiftedImage(const cv::Mat& src, float dx, float dy) {
    cv::Mat shifted;
    cv::Mat affine = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(src, shifted, affine, src.size(),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return shifted;
}

inline void writeTempFile(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
    f.close();
}

inline void removeTempFile(const std::string& path) {
    std::filesystem::remove(path);
}

inline std::string makeTempCocoNames() {
    const std::string path = "./test_coco.names";
    std::ofstream f(path);
    for (int i = 0; i < 80; ++i) {
        f << "class_" << i << "\n";
    }
    f.close();
    return path;
}

inline std::string makeTempOnnxStub() {
    // This creates a minimal ONNX stub; real ONNX model needed for full tests.
    // Unit tests that need real inference should be tagged [Slow] or [RequiresModel].
    const std::string path = "./test_yolov8_stub.onnx";
    std::ofstream f(path, std::ios::binary);

    const unsigned char stub[] = {
        // Minimal ONNX file header (IR version 7, opset 12)
        0x08, 0x07, 0x12, 0x0c, 0x62, 0x61, 0x63, 0x6b,
        0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74,
        0x3a, 0x0a, 0x2a, 0x0a, 0x0a, 0x08, 0x08, 0x01,
        0x12, 0x04, 0x69, 0x6e, 0x66, 0x6f
    };
    f.write((const char*)stub, sizeof(stub));
    f.close();
    return path;
}

inline double approxEqual(double a, double b, double eps = 0.001) {
    return std::abs(a - b) < eps;
}

inline bool pointInside(const cv::Point2f& pt, const cv::Size& size) {
    return pt.x >= 0 && pt.x < size.width && pt.y >= 0 && pt.y < size.height;
}

} // namespace test_utils

#endif // TEST_UTILS_H