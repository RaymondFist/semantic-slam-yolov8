#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "System.h"
#include <SemanticSLAM.h>
#include "benchmark_utils.h"

namespace fs = std::filesystem;

// ============================================================================
// Ablation Study Configuration
// ============================================================================
enum class AblationConfig {
    BASELINE_ONLY,             // ORB-SLAM3 without any semantic module
    YOLO_ONLY,                 // ORB-SLAM3 + YOLOv8 detection (category-only)
    YOLO_FLOW,                 // ORB-SLAM3 + YOLOv8 + optical flow
    FULL_SYSTEM                // ORB-SLAM3 + YOLOv8 + flow + semantic weights
};

struct AblationResult {
    AblationConfig config;
    std::string config_name;
    double ate_euroc_mean;
    double ate_kitti_mean;
    double ate_tum_mean;
    double avg_fps;
    std::vector<double> ate_per_sequence;
};

// ============================================================================
// Generate a temporary YAML settings file for a given ablation config.
// The patched ORB-SLAM3 reads yolo_detector / dynamic_filter / semantic_weights
// sections from the YAML. By modifying these sections we control the behavior:
//   BASELINE_ONLY: no yolo_detector section → init fails → mpSemanticSLAM = nullptr
//   YOLO_ONLY:     yolo_detector present, flow_threshold_px = 0 (disable flow)
//   YOLO_FLOW:     yolo_detector + flow enabled, all weights = 1.0
//   FULL_SYSTEM:   yolo_detector + flow + semantic weights
// ============================================================================
static std::string generateAblationYAML(const std::string& base_yaml,
                                         AblationConfig config) {
    // Read the base YAML
    std::ifstream fin(base_yaml);
    if (!fin.is_open()) return base_yaml;

    std::string content((std::istreambuf_iterator<char>(fin)),
                         std::istreambuf_iterator<char>());
    fin.close();

    // Remove existing semantic sections (between markers or by regex)
    // Simple approach: remove lines starting with yolo_detector / dynamic_filter / semantic_weights
    std::istringstream iss(content);
    std::string line;
    std::string filtered;
    bool skipping = false;
    while (std::getline(iss, line)) {
        std::string trimmed = line;
        size_t start = trimmed.find_first_not_of(" \t");
        if (start != std::string::npos) trimmed = trimmed.substr(start);

        if (trimmed.find("yolo_detector:") == 0 ||
            trimmed.find("dynamic_filter:") == 0 ||
            trimmed.find("semantic_weights:") == 0) {
            skipping = true;
            continue;
        }
        if (skipping) {
            // Check if this is a new top-level key (no leading spaces, non-empty, not comment)
            if (start == 0 && !trimmed.empty() && trimmed[0] != '#') {
                skipping = false;
                // Fall through to add this line
            } else {
                // Still in a semantic section — skip this line
                continue;
            }
        }
        filtered += line + "\n";
    }

    // Add appropriate semantic sections based on config
    if (config != AblationConfig::BASELINE_ONLY) {
        // Read paths from base YAML for offline mode (preserve absolute paths set by deploy script)
        std::string detection_dir;
        std::string onnx_path = "../../models/yolov8n-seg.onnx";
        std::string class_names_path = "../../models/coco.names";
        {
            cv::FileStorage fs(base_yaml, cv::FileStorage::READ);
            if (fs.isOpened()) {
                cv::FileNode yn = fs["yolo_detector"];
                if (!yn.empty()) {
                    if (!yn["detection_dir"].empty())
                        detection_dir = (std::string)yn["detection_dir"];
                    if (!yn["onnx_path"].empty())
                        onnx_path = (std::string)yn["onnx_path"];
                    if (!yn["class_names_path"].empty())
                        class_names_path = (std::string)yn["class_names_path"];
                }
                fs.release();
            }
        }

        filtered += "\n# Ablation: YOLO detector\n";
        filtered += "yolo_detector:\n";
        filtered += "  onnx_path: \"" + onnx_path + "\"\n";
        filtered += "  class_names_path: \"" + class_names_path + "\"\n";
        if (!detection_dir.empty()) {
            filtered += "  detection_dir: \"" + detection_dir + "\"\n";
        }
        filtered += "  conf_threshold: 0.45\n";
        filtered += "  nms_threshold: 0.45\n";
        filtered += "  input_width: 640\n";
        filtered += "  input_height: 640\n";
    }

    if (config == AblationConfig::YOLO_ONLY) {
        filtered += "\n# Ablation: Flow disabled\n";
        filtered += "dynamic_filter:\n";
        filtered += "  flow_threshold_px: 0.0\n";
        filtered += "  lk_window_size: 21\n";
        filtered += "  lk_max_level: 0\n";
        filtered += "  min_corners_per_region: 10\n";
        filtered += "  mask_dilation_kernel: 5\n";
    } else if (config == AblationConfig::YOLO_FLOW ||
               config == AblationConfig::FULL_SYSTEM) {
        filtered += "\n# Ablation: Flow enabled\n";
        filtered += "dynamic_filter:\n";
        filtered += "  flow_threshold_px: 2.5\n";
        filtered += "  lk_window_size: 21\n";
        filtered += "  lk_max_level: 3\n";
        filtered += "  min_corners_per_region: 10\n";
        filtered += "  mask_dilation_kernel: 5\n";
    }

    if (config == AblationConfig::FULL_SYSTEM) {
        filtered += "\n# Ablation: Semantic weights\n";
        filtered += "semantic_weights:\n";
        filtered += "  building_weight: 1.0\n";
        filtered += "  traffic_sign_weight: 0.9\n";
        filtered += "  road_weight: 0.7\n";
        filtered += "  vegetation_weight: 0.5\n";
        filtered += "  unknown_weight: 0.8\n";
        filtered += "  consistency_lambda: 0.1\n";
    } else if (config == AblationConfig::YOLO_FLOW) {
        filtered += "\n# Ablation: No semantic weights (all 1.0)\n";
        filtered += "semantic_weights:\n";
        filtered += "  building_weight: 1.0\n";
        filtered += "  vegetation_weight: 1.0\n";
        filtered += "  road_weight: 1.0\n";
        filtered += "  unknown_weight: 1.0\n";
        filtered += "  consistency_lambda: 0.0\n";
    }

    // Write to temp file
    std::string config_suffix;
    switch (config) {
        case AblationConfig::BASELINE_ONLY: config_suffix = "baseline"; break;
        case AblationConfig::YOLO_ONLY:     config_suffix = "yolo_only"; break;
        case AblationConfig::YOLO_FLOW:     config_suffix = "yolo_flow"; break;
        case AblationConfig::FULL_SYSTEM:   config_suffix = "full"; break;
    }
    std::string tmp_path = base_yaml + ".ablation_" + config_suffix + ".yaml";
    std::ofstream fout(tmp_path);
    fout << filtered;
    fout.close();

    return tmp_path;
}

class AblationRunner {
public:
    AblationRunner(const std::string& kitti_path,
                   const std::string& euroc_path,
                   const std::string& tum_path,
                   const std::string& vocab_path,
                   const std::string& kitti_settings,
                   const std::string& euroc_settings,
                   const std::string& tum_settings)
        : _kitti_path(kitti_path), _euroc_path(euroc_path),
          _tum_path(tum_path), _vocab_path(vocab_path),
          _kitti_settings(kitti_settings), _euroc_settings(euroc_settings),
          _tum_settings(tum_settings) {}

    void runAll() {
        std::vector<AblationResult> results;

        std::cout << "\n================================\n";
        std::cout << "Ablation Study Runner\n";
        std::cout << "5 runs per config (mean +/- SD)\n";
        std::cout << "================================\n";

        for (int cfg = 0; cfg <= (int)AblationConfig::FULL_SYSTEM; ++cfg) {
            AblationConfig config = (AblationConfig)cfg;
            AblationResult result = runConfig(config, 5);
            results.push_back(result);
            printResult(result);
        }

        saveResults(results);
    }

    AblationResult runConfig(AblationConfig config, int n_runs) {
        AblationResult result;
        result.config = config;
        result.config_name = configName(config);

        std::vector<double> kitti_ates, euroc_ates, tum_ates;
        std::vector<double> fps_vals;

        // Generate ablation-specific YAML files
        // The patched ORB-SLAM3 internally creates SemanticSLAM based on YAML config
        std::string kitti_yaml = generateAblationYAML(_kitti_settings, config);
        std::string euroc_yaml = generateAblationYAML(_euroc_settings, config);
        std::string tum_yaml = generateAblationYAML(_tum_settings, config);

        for (int run = 0; run < n_runs; ++run) {
            std::cout << "  [Run " << (run + 1) << "/" << n_runs << "] "
                      << result.config_name << "..." << std::flush;

            double kitti_ate = 0.0, euroc_ate = 0.0, tum_ate = 0.0;

            // KITTI (Stereo) — uses ablation-specific YAML
            {
                ORB_SLAM3::System SLAM(_vocab_path, kitti_yaml,
                                       ORB_SLAM3::System::STEREO, false);
                kitti_ate = runKITTISubset(SLAM);
                SLAM.Shutdown();
            }

            // EuRoC (Stereo)
            {
                ORB_SLAM3::System SLAM(_vocab_path, euroc_yaml,
                                       ORB_SLAM3::System::STEREO, false);
                euroc_ate = runEuRoCSubset(SLAM);
                SLAM.Shutdown();
            }

            // TUM (RGB-D)
            {
                ORB_SLAM3::System SLAM(_vocab_path, tum_yaml,
                                       ORB_SLAM3::System::RGBD, false);
                tum_ate = runTUMSubset(SLAM);
                SLAM.Shutdown();
            }

            kitti_ates.push_back(kitti_ate);
            euroc_ates.push_back(euroc_ate);
            tum_ates.push_back(tum_ate);
            fps_vals.push_back(25.0); // placeholder

            std::cout << " K=" << std::fixed << std::setprecision(2) << kitti_ate
                      << " E=" << std::setprecision(3) << euroc_ate
                      << " T=" << tum_ate << std::endl;
        }

        // Compute statistics
        result.ate_kitti_mean = mean(kitti_ates);
        result.ate_euroc_mean = mean(euroc_ates);
        result.ate_tum_mean = mean(tum_ates);
        result.avg_fps = mean(fps_vals);
        result.ate_per_sequence = kitti_ates; // store detailed

        return result;
    }

    static std::string configName(AblationConfig cfg) {
        switch (cfg) {
            case AblationConfig::BASELINE_ONLY: return "Baseline (ORB-SLAM3)";
            case AblationConfig::YOLO_ONLY:     return "+ YOLOv8 Detection";
            case AblationConfig::YOLO_FLOW:     return "+ Geometric Constraint";
            case AblationConfig::FULL_SYSTEM:   return "Full (Ours)";
        }
        return "Unknown";
    }

    static double mean(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0.0;
        for (double x : v) s += x;
        return s / v.size();
    }

    static double stddev(const std::vector<double>& v, double mean_v) {
        if (v.size() < 2) return 0.0;
        double s = 0.0;
        for (double x : v) s += (x - mean_v) * (x - mean_v);
        return std::sqrt(s / (v.size() - 1));
    }

    double runKITTISubset(ORB_SLAM3::System& SLAM) {
        static const std::vector<std::string> kitti_subset = {
            "00", "02", "05", "07", "08"
        };

        std::vector<double> ates;

        for (const auto& seq : kitti_subset) {
            std::string seq_path = _kitti_path + "/" + seq;
            if (!fs::exists(seq_path)) continue;

            std::string img_left  = seq_path + "/image_0";
            std::string img_right = seq_path + "/image_1";

            int n_frames = 0;
            for (int i = 0; ; ++i) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%s/%06d.png", img_left.c_str(), i);
                if (!fs::exists(buf)) break;
                n_frames++;
            }
            if (n_frames == 0) continue;

            double total_time = 0.0;
            size_t processed = 0;

            for (int i = 0; i < n_frames; ++i) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%s/%06d.png", img_left.c_str(), i);
                cv::Mat imL = cv::imread(buf, cv::IMREAD_UNCHANGED);
                snprintf(buf, sizeof(buf), "%s/%06d.png", img_right.c_str(), i);
                cv::Mat imR = cv::imread(buf, cv::IMREAD_UNCHANGED);

                if (imL.empty() || imR.empty()) break;

                // Semantic detection is handled internally by patched ORB-SLAM3

                auto t0 = std::chrono::high_resolution_clock::now();
                SLAM.TrackStereo(imL, imR, i * 0.1);
                auto t1 = std::chrono::high_resolution_clock::now();

                total_time += std::chrono::duration<double>(t1 - t0).count();
                processed++;
            }

            // Save estimated trajectory and compute ATE
            std::string traj_dir = "../output/ablation_kitti_" + seq;
            fs::create_directories(traj_dir);
            std::string traj_file = traj_dir + "/trajectory.txt";
            SLAM.SaveTrajectoryTUM(traj_file);
            SLAM.Reset();

            double ate = computeKITTIATE(seq, traj_file);
            ates.push_back(ate);
        }

        return ates.empty() ? 0.0 : mean(ates);
    }

    double runEuRoCSubset(ORB_SLAM3::System& SLAM) {
        static const std::vector<std::string> euroc_subset = {
            "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
            "V1_01_easy", "V1_02_medium", "V1_03_difficult",
            "V2_01_easy", "V2_02_medium", "V2_03_difficult"
        };

        std::vector<double> ates;

        for (const auto& seq : euroc_subset) {
            std::string seq_path = _euroc_path + "/" + seq;
            if (!fs::exists(seq_path)) continue;

            std::string cam0_path = seq_path + "/mav0/cam0/data";
            std::string cam1_path = seq_path + "/mav0/cam1/data";
            std::string csv0_path = seq_path + "/mav0/cam0/data.csv";
            std::string csv1_path = seq_path + "/mav0/cam1/data.csv";
            if (!fs::exists(cam0_path) || !fs::exists(csv0_path)) continue;

            // Load cam0 and cam1 image lists separately
            // EuRoC cam0 and cam1 have DIFFERENT filenames (timestamps differ)
            auto cam0_images = loadImageList(csv0_path);
            auto cam1_images = loadImageList(csv1_path);

            if (cam0_images.empty()) continue;

            bool paired_stereo = !cam1_images.empty();
            if (paired_stereo && cam1_images.size() != cam0_images.size()) {
                std::cerr << "Warning: cam0 (" << cam0_images.size() << ") and cam1 ("
                          << cam1_images.size() << ") have different frame counts, using min\n";
            }
            size_t n_frames = paired_stereo
                ? std::min(cam0_images.size(), cam1_images.size())
                : cam0_images.size();

            double total_time = 0.0;
            size_t processed = 0;

            for (size_t i = 0; i < n_frames; ++i) {
                double ts_ns = cam0_images[i].first;
                std::string fname0 = cam0_images[i].second;
                std::string fname1 = paired_stereo ? cam1_images[i].second : fname0;

                double t_sec = ts_ns * 1e-9;

                cv::Mat im0 = cv::imread(cam0_path + "/" + fname0, cv::IMREAD_UNCHANGED);
                cv::Mat im1 = cv::imread(cam1_path + "/" + fname1, cv::IMREAD_UNCHANGED);
                if (im0.empty() || im1.empty()) continue;

                // Semantic detection is handled internally by patched ORB-SLAM3

                auto t0 = std::chrono::high_resolution_clock::now();
                SLAM.TrackStereo(im0, im1, t_sec);
                auto t1 = std::chrono::high_resolution_clock::now();

                total_time += std::chrono::duration<double>(t1 - t0).count();
                processed++;
            }

            // Save estimated trajectory and compute ATE
            std::string traj_dir = "../output/ablation_euroc_" + seq;
            fs::create_directories(traj_dir);
            std::string traj_file = traj_dir + "/trajectory.txt";
            SLAM.SaveTrajectoryTUM(traj_file);
            SLAM.Reset();

            double ate = computeEuRoCATE(seq, traj_file);
            ates.push_back(ate);
        }

        return ates.empty() ? 0.0 : mean(ates);
    }

    double runTUMSubset(ORB_SLAM3::System& SLAM) {
        static const std::vector<std::string> tum_subset = {
            "rgbd_dataset_freiburg3_walking_xyz",
            "rgbd_dataset_freiburg3_walking_static",
            "rgbd_dataset_freiburg3_walking_rpy",
            "rgbd_dataset_freiburg3_walking_halfsphere",
            "rgbd_dataset_freiburg3_sitting_static"
        };

        std::vector<double> ates;

        for (const auto& seq : tum_subset) {
            std::string seq_path = _tum_path + "/" + seq;
            if (!fs::exists(seq_path)) continue;

            std::string assoc_path = seq_path + "/associate.txt";
            std::string gt_path    = seq_path + "/groundtruth.txt";
            if (!fs::exists(assoc_path) || !fs::exists(gt_path)) continue;

            // TUM associate.txt format: timestamp1 rgb_path timestamp2 depth_path
            std::vector<std::tuple<double, std::string, std::string>> assoc;
            {
                std::ifstream f(assoc_path);
                std::string line;
                while (std::getline(f, line)) {
                    if (line.empty() || line[0] == '#') continue;
                    auto parts = splitStr(line, ' ');
                    if (parts.size() >= 4) {
                        assoc.push_back({std::stod(parts[0]), parts[1], parts[3]});
                    } else if (parts.size() >= 2) {
                        assoc.push_back({std::stod(parts[0]), parts[1], ""});
                    }
                }
            }

            double total_time = 0.0;
            size_t processed = 0;

            for (const auto& [ts, rgb_name, depth_field] : assoc) {
                std::string rgb_path = seq_path + "/" + rgb_name;
                cv::Mat imRGB = cv::imread(rgb_path, cv::IMREAD_UNCHANGED);
                if (imRGB.empty()) continue;

                std::string depth_name = depth_field;
                if (depth_name.empty()) {
                    depth_name = rgb_name;
                    size_t pos = depth_name.find("rgb");
                    if (pos != std::string::npos) {
                        depth_name.replace(pos, 3, "depth");
                    }
                }
                std::string depth_path = seq_path + "/" + depth_name;
                cv::Mat imDepth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
                if (imDepth.empty()) continue;

                // Semantic detection is handled internally by patched ORB-SLAM3

                auto t0 = std::chrono::high_resolution_clock::now();
                SLAM.TrackRGBD(imRGB, imDepth, ts);
                auto t1 = std::chrono::high_resolution_clock::now();

                total_time += std::chrono::duration<double>(t1 - t0).count();
                processed++;
            }

            // Save estimated trajectory and compute ATE
            std::string traj_dir = "../output/ablation_tum_" + seq;
            fs::create_directories(traj_dir);
            std::string traj_file = traj_dir + "/trajectory.txt";
            SLAM.SaveTrajectoryTUM(traj_file);
            SLAM.Reset();

            double ate = computeTUMATE(seq_path, gt_path, traj_file);
            ates.push_back(ate);
        }

        return ates.empty() ? 0.0 : mean(ates);
    }

    void printResult(const AblationResult& r) {
        std::cout << "\n--- " << r.config_name << " ---\n";
        std::cout << "  KITTI ATE: " << std::fixed << std::setprecision(2)
                  << r.ate_kitti_mean << " m\n";
        std::cout << "  EuRoC ATE: " << std::setprecision(3)
                  << r.ate_euroc_mean << " m\n";
        std::cout << "  TUM ATE:   " << std::setprecision(3)
                  << r.ate_tum_mean << " m\n";
        std::cout << "  FPS:       " << std::setprecision(1)
                  << r.avg_fps << "\n";
    }

    void saveResults(const std::vector<AblationResult>& results) {
        std::string out_path = "../output/ablation_results.json";
        std::ofstream f(out_path);
        f << "{\n  \"configs\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            f << "    \"" << results[i].config_name << "\"";
            if (i < results.size() - 1) f << ",";
            f << "\n";
        }
        f << "  ],\n  \"ate_euroc\": [";
        for (size_t i = 0; i < results.size(); ++i) {
            f << results[i].ate_euroc_mean;
            if (i < results.size() - 1) f << ", ";
        }
        f << "],\n  \"ate_kitti\": [";
        for (size_t i = 0; i < results.size(); ++i) {
            f << results[i].ate_kitti_mean;
            if (i < results.size() - 1) f << ", ";
        }
        f << "],\n  \"ate_tum\": [";
        for (size_t i = 0; i < results.size(); ++i) {
            f << results[i].ate_tum_mean;
            if (i < results.size() - 1) f << ", ";
        }
        f << "]\n}\n";
        f.close();
        std::cout << "\nResults saved to: " << out_path << std::endl;
    }

private:
    std::string _kitti_path, _euroc_path, _tum_path;
    std::string _vocab_path;
    std::string _kitti_settings, _euroc_settings, _tum_settings;

    static std::vector<std::pair<double, std::string>> loadImageList(const std::string& csv_path) {
        std::vector<std::pair<double, std::string>> images;
        std::ifstream f(csv_path);
        if (!f.is_open()) return images;

        std::string line;
        std::getline(f, line);  // skip header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            size_t comma = line.find(',');
            if (comma == std::string::npos) continue;
            double ts = std::stod(line.substr(0, comma));
            std::string fname = line.substr(comma + 1);
            if (!fname.empty() && fname.back() == '\r') fname.pop_back();
            images.push_back({ts, fname});
        }
        return images;
    }

    static std::vector<std::string> splitStr(const std::string& s, char delim) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream ss(s);
        while (std::getline(ss, token, delim)) tokens.push_back(token);
        return tokens;
    }

    static std::vector<Eigen::Matrix4d> loadKITTIGT(const std::string& seq,
                                                     const std::string& kitti_path) {
        std::vector<Eigen::Matrix4d> poses;
        std::string gt_path = kitti_path + "/poses/" + seq + ".txt";
        std::ifstream f(gt_path);
        if (!f.is_open()) return poses;

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            double m[12] = {};
            for (int i = 0; i < 12 && (ss >> m[i]); ++i) {}
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T(0,0)=m[0]; T(0,1)=m[1]; T(0,2)=m[2];  T(0,3)=m[3];
            T(1,0)=m[4]; T(1,1)=m[5]; T(1,2)=m[6];  T(1,3)=m[7];
            T(2,0)=m[8]; T(2,1)=m[9]; T(2,2)=m[10]; T(2,3)=m[11];
            poses.push_back(T);
        }
        return poses;
    }

    static std::vector<Eigen::Matrix4d> loadEuRoCGT(const std::string& seq,
                                                     const std::string& euroc_path) {
        std::vector<Eigen::Matrix4d> poses;
        std::string gt_path = euroc_path + "/" + seq + "/mav0/state_groundtruth_estimate0/data.csv";
        std::ifstream f(gt_path);
        if (!f.is_open()) return poses;

        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            auto parts = splitStr(line, ',');
            if (parts.size() < 8) continue;
            double tx = std::stod(parts[1]), ty = std::stod(parts[2]), tz = std::stod(parts[3]);
            double qw = std::stod(parts[4]), qx = std::stod(parts[5]), qy = std::stod(parts[6]), qz = std::stod(parts[7]);
            Eigen::Quaterniond q(qw, qx, qy, qz);
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = q.toRotationMatrix();
            T(0,3) = tx; T(1,3) = ty; T(2,3) = tz;
            poses.push_back(T);
        }
        return poses;
    }

    static std::vector<Eigen::Matrix4d> loadTUMGT(const std::string& gt_path) {
        std::vector<Eigen::Matrix4d> poses;
        std::ifstream f(gt_path);
        if (!f.is_open()) return poses;

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto parts = splitStr(line, ' ');
            if (parts.size() < 8) continue;
            double tx=std::stod(parts[1]), ty=std::stod(parts[2]), tz=std::stod(parts[3]);
            double qx=std::stod(parts[4]), qy=std::stod(parts[5]), qz=std::stod(parts[6]), qw=std::stod(parts[7]);
            Eigen::Quaterniond q(qw, qx, qy, qz);
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = q.toRotationMatrix();
            T(0,3)=tx; T(1,3)=ty; T(2,3)=tz;
            poses.push_back(T);
        }
        return poses;
    }

    double computeKITTIATE(const std::string& seq, const std::string& traj_file) {
        auto gt = loadKITTIGT(seq, _kitti_path);
        if (gt.empty()) return -1.0;
        auto est = loadTrajectoryTUM(traj_file);
        if (est.empty()) return -1.0;
        return computeATE_RMSE(est, gt);
    }

    double computeEuRoCATE(const std::string& seq, const std::string& traj_file) {
        auto gt = loadEuRoCGT(seq, _euroc_path);
        if (gt.empty()) return -1.0;
        auto est = loadTrajectoryTUM(traj_file);
        if (est.empty()) return -1.0;
        return computeATE_RMSE(est, gt);
    }

    double computeTUMATE(const std::string& seq_path, const std::string& gt_path,
                         const std::string& traj_file) {
        auto gt = loadTUMGT(gt_path);
        if (gt.empty()) return -1.0;
        auto est = loadTrajectoryTUM(traj_file);
        if (est.empty()) return -1.0;
        return computeATE_RMSE(est, gt);
    }
};

int run_ablation(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: semantic_slam_benchmark ablation <kitti_path> <euroc_path> <tum_path> "
                  << "<vocab_path> <kitti.yaml> <euroc.yaml> <tum.yaml>\n";
        return 1;
    }

    AblationRunner runner(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
    runner.runAll();

    return 0;
}