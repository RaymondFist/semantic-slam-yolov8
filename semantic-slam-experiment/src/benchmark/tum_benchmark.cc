#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "System.h"
#include <SemanticSLAM.h>
#include "benchmark_utils.h"
// SemanticSLAM is now integrated inside ORB-SLAM3 (accessed via GetSemanticSLAM())

namespace fs = std::filesystem;

struct SequenceResult {
    std::string name;
    double ate_rmse;
    double rpe_rmse;
    double rpe_rmse_rot;
    double avg_fps;
    size_t total_frames;
    size_t dynamic_frames;
    double tracking_lost_ratio;
};

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream ss(s);
    while (std::getline(ss, token, delim)) tokens.push_back(token);
    return tokens;
}

static std::vector<std::tuple<double, std::string, std::string>> loadAssocFile(const std::string& path) {
    std::vector<std::tuple<double, std::string, std::string>> entries;
    std::ifstream f(path);
    if (!f.is_open()) return entries;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto parts = split(line, ' ');
        // TUM associate.txt format: timestamp1 rgb_path timestamp2 depth_path
        if (parts.size() >= 4) {
            entries.push_back({std::stod(parts[0]), parts[1], parts[3]});
        } else if (parts.size() >= 2) {
            entries.push_back({std::stod(parts[0]), parts[1], ""});
        }
    }
    return entries;
}

static std::vector<Eigen::Matrix4d> loadTUMGroundTruth(const std::string& gt_path) {
    std::vector<Eigen::Matrix4d> poses;
    std::ifstream f(gt_path);
    if (!f.is_open()) return poses;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto parts = split(line, ' ');
        if (parts.size() < 8) continue;
        double tx = std::stod(parts[1]), ty = std::stod(parts[2]), tz = std::stod(parts[3]);
        double qx = std::stod(parts[4]), qy = std::stod(parts[5]), qz = std::stod(parts[6]), qw = std::stod(parts[7]);
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = q.toRotationMatrix();
        T(0,3) = tx; T(1,3) = ty; T(2,3) = tz;
        poses.push_back(T);
    }
    return poses;
}

class TUMBenchmark {
public:
    TUMBenchmark(const std::string& dataset_path,
                 const std::string& vocab_path,
                 const std::string& settings_path,
                 const std::string& output_dir)
        : _dataset_path(dataset_path)
        , _vocab_path(vocab_path)
        , _settings_path(settings_path)
        , _output_dir(output_dir) {}

    SequenceResult runSequence(const std::string& seq_name) {
        SequenceResult result;
        result.name = seq_name;

        std::string rgb_file   = _dataset_path + "/rgb.txt";
        std::string depth_file = _dataset_path + "/depth.txt";
        std::string assoc_file = _dataset_path + "/associate.txt";
        std::string gt_file    = _dataset_path + "/groundtruth.txt";

        if (!fs::exists(assoc_file)) {
            std::cerr << "associate.txt not found. Run: python associate.py rgb.txt depth.txt > associate.txt\n";
            return result;
        }

        fs::create_directories(_output_dir);
        std::string traj_file = _output_dir + "/" + seq_name + "_trajectory.txt";

        // NOTE: After patching, ORB-SLAM3 internally creates and uses
        // SemanticSLAM. We don't need a separate instance here.
        // Statistics are accessed via SLAM.GetSemanticSLAM().

        ORB_SLAM3::System SLAM(_vocab_path, _settings_path,
                               ORB_SLAM3::System::RGBD, true); // enable viewer (xvfb provides virtual display)

        auto assoc = loadAssocFile(assoc_file);
        result.total_frames = assoc.size();

        double total_time = 0.0;
        size_t processed = 0;

        for (size_t i = 0; i < assoc.size(); ++i) {
            double ts = std::get<0>(assoc[i]);
            std::string rgb_name = std::get<1>(assoc[i]);
            std::string depth_name = std::get<2>(assoc[i]);

            cv::Mat imRGB   = cv::imread(_dataset_path + "/" + rgb_name, cv::IMREAD_UNCHANGED);
            // If depth path not in associate.txt, derive from rgb path
            if (depth_name.empty()) {
                depth_name = rgb_name;
                size_t pos = depth_name.find("rgb");
                if (pos != std::string::npos) depth_name.replace(pos, 3, "depth");
            }
            cv::Mat imDepth = cv::imread(_dataset_path + "/" + depth_name, cv::IMREAD_UNCHANGED);

            if (imRGB.empty() || imDepth.empty()) {
                std::cerr << "Failed to load frame " << i << std::endl;
                continue;
            }

            // Semantic detection is now handled internally by patched ORB-SLAM3
            // No need to call submitFrame manually

            auto t0 = std::chrono::high_resolution_clock::now();
            if (i < 5 || i % 100 == 0) {
                std::cout << "[DEBUG] Frame " << i << "/" << assoc.size()
                          << " size=" << imRGB.cols << "x" << imRGB.rows
                          << " calling TrackRGBD..." << std::flush;
            }
            Sophus::SE3f Tcw = SLAM.TrackRGBD(imRGB, imDepth, ts);
            if (i < 5 || i % 100 == 0) {
                std::cout << " OK" << std::endl;
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            total_time += std::chrono::duration<double>(t1 - t0).count();
            processed++;

            // Get dynamic frame statistics from ORB-SLAM3's internal SemanticSLAM
            auto* pSemSlam = SLAM.GetSemanticSLAM();
            if (pSemSlam && pSemSlam->hasValidDetection()) {
                if (pSemSlam->countDynamicFeatures() > 0) {
                    result.dynamic_frames++;
                }
            }
        }

        // Save trajectory in TUM format (compatible with evo_ape tum)
        SLAM.SaveTrajectoryTUM(traj_file);
        SLAM.Shutdown();
        // No need to stop semSlam — it's managed by ORB-SLAM3 internally

        result.avg_fps = processed > 0 ? processed / total_time : 0.0;

        auto gt_poses = loadTUMGroundTruth(gt_file);
        auto est_poses = loadTrajectoryTUM(traj_file);

        result.ate_rmse = computeATE_RMSE(est_poses, gt_poses);
        return result;
    }

private:
    std::string _dataset_path;
    std::string _vocab_path;
    std::string _settings_path;
    std::string _output_dir;
};

int run_tum_benchmark(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: semantic_slam_benchmark tum <dataset_path> <vocab_path> <settings_path> [output_dir]\n";
        std::cerr << "Example: semantic_slam_benchmark tum /datasets/TUM/fr1_xyz ../models/ORBvoc.txt ./config/TUM1.yaml ./results\n";
        return 1;
    }

    std::string dataset  = argv[1];
    std::string vocab    = argv[2];
    std::string settings = argv[3];
    std::string output   = argc > 4 ? argv[4] : "../../output";

    std::cout << "TUM Benchmark\n";
    std::cout << "  Dataset:  " << dataset << "\n";
    std::cout << "  Vocab:    " << vocab << "\n";
    std::cout << "  Settings: " << settings << "\n";
    std::cout << "  Output:   " << output << "\n\n";

    TUMBenchmark bench(dataset, vocab, settings, output);
    auto result = bench.runSequence(fs::path(dataset).filename().string());

    std::cout << "\n--- Result ---\n";
    std::cout << "  Sequence:    " << result.name << "\n";
    std::cout << "  Frames:      " << result.total_frames << "\n";
    std::cout << "  ATE RMSE:    " << std::fixed << std::setprecision(4)
              << result.ate_rmse << " m\n";
    std::cout << "  Avg FPS:     " << std::fixed << std::setprecision(1)
              << result.avg_fps << "\n";
    std::cout << "  Dyn Frames:  " << result.dynamic_frames << "\n";

    return 0;
}