#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "System.h"
#include <SemanticSLAM.h>
#include "benchmark_utils.h"
// SemanticSLAM is now integrated inside ORB-SLAM3 (accessed via GetSemanticSLAM())

namespace fs = std::filesystem;

static const std::vector<std::string> EUROC_SEQUENCES = {
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult"
};

struct EuRoCResult {
    std::string name;
    double ate_rmse;
    double rpe_rmse;
    double avg_fps;
    size_t total_frames;
    size_t dynamic_frames;
    double tracking_lost_ratio;
};

class EuRoCBenchmark {
public:
    EuRoCBenchmark(const std::string& dataset_path,
                   const std::string& vocab_path,
                   const std::string& settings_path,
                   const std::string& output_dir = "../../output")
        : _dataset_path(dataset_path)
        , _vocab_path(vocab_path)
        , _settings_path(settings_path)
        , _output_dir(output_dir) {}

    void runAll() {
        std::cout << "\n============================\n";
        std::cout << "EuRoC MAV Benchmark\n";
        std::cout << "============================\n\n";

        std::vector<EuRoCResult> results;

        for (const auto& seq : EUROC_SEQUENCES) {
            std::string seq_path = _dataset_path + "/" + seq;
            if (!fs::exists(seq_path)) {
                std::cout << "  Seq " << seq << ": SKIP (not found)\n";
                continue;
            }

            std::cout << "  Seq " << seq << ": running... " << std::flush;
            auto result = runSequence(seq);
            results.push_back(result);
            std::cout << "ATE=" << std::fixed << std::setprecision(3)
                      << result.ate_rmse << "m  " << result.avg_fps << "FPS\n";
        }

        printSummary(results);
    }

    EuRoCResult runSequence(const std::string& seq) {
        EuRoCResult result;
        result.name = seq;
        result.dynamic_frames = 0;

        std::string seq_path = _dataset_path + "/" + seq;
        std::string cam0_path = seq_path + "/mav0/cam0/data";
        std::string cam1_path = seq_path + "/mav0/cam1/data";
        std::string csv0_path = seq_path + "/mav0/cam0/data.csv";
        std::string csv1_path = seq_path + "/mav0/cam1/data.csv";
        std::string gt_path   = seq_path + "/mav0/state_groundtruth_estimate0/data.csv";

        if (!fs::exists(cam0_path) || !fs::exists(csv0_path)) {
            std::cerr << "cam0 data not found\n";
            return result;
        }

        // Load cam0 and cam1 image lists separately
        // EuRoC cam0 and cam1 have DIFFERENT filenames (timestamps differ)
        auto cam0_images = loadImageList(csv0_path);
        auto cam1_images = loadImageList(csv1_path);

        if (cam0_images.empty()) {
            std::cerr << "No images found in cam0 CSV\n";
            return result;
        }

        // If cam1 CSV exists, use paired loading; otherwise fallback to same filename
        bool paired_stereo = !cam1_images.empty();
        if (paired_stereo && cam1_images.size() != cam0_images.size()) {
            std::cerr << "Warning: cam0 (" << cam0_images.size() << ") and cam1 ("
                      << cam1_images.size() << ") have different frame counts, using min\n";
        }
        size_t n_frames = paired_stereo
            ? std::min(cam0_images.size(), cam1_images.size())
            : cam0_images.size();

        // NOTE: After patching, ORB-SLAM3 internally creates and uses
        // SemanticSLAM. We don't need a separate instance here.

        ORB_SLAM3::System SLAM(
            _vocab_path,
            _settings_path,
            ORB_SLAM3::System::STEREO,
            true); // enable viewer (xvfb provides virtual display)

        result.total_frames = n_frames;

        double total_time = 0.0;
        size_t processed = 0;

        for (size_t i = 0; i < n_frames; ++i) {
            double ts_ns = cam0_images[i].first;
            std::string fname0 = cam0_images[i].second;
            std::string fname1 = paired_stereo ? cam1_images[i].second : fname0;

            double ts_sec = ts_ns * 1e-9;

            cv::Mat im0 = cv::imread(cam0_path + "/" + fname0, cv::IMREAD_UNCHANGED);
            cv::Mat im1 = cv::imread(cam1_path + "/" + fname1, cv::IMREAD_UNCHANGED);

            if (im0.empty() || im1.empty()) {
                std::cerr << "Failed to load frame " << i << " (" << fname0 << " / " << fname1 << ")\n";
                continue;
            }

            // Semantic detection is handled internally by patched ORB-SLAM3

            auto t0 = std::chrono::high_resolution_clock::now();
            if (i < 5 || i % 100 == 0) {
                std::cout << "[DEBUG] Frame " << i << "/" << n_frames
                          << " size=" << im0.cols << "x" << im0.rows
                          << " calling TrackStereo..." << std::flush;
            }
            Sophus::SE3f Tcw = SLAM.TrackStereo(im0, im1, ts_sec);
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

        fs::create_directories(_output_dir);
        SLAM.SaveTrajectoryTUM(_output_dir + "/euroc_" + seq + "_trajectory.txt");
        SLAM.Shutdown();
        // SemanticSLAM is managed by ORB-SLAM3 internally

        result.avg_fps = processed > 0 ? processed / total_time : 0.0;

        auto gt_poses = loadEuRoCGroundTruth(gt_path);
        auto est_poses = loadTrajectoryTUM(_output_dir + "/euroc_" + seq + "_trajectory.txt");
        result.ate_rmse = computeATE_RMSE(est_poses, gt_poses);

        return result;
    }

    std::vector<std::pair<double, std::string>> loadImageList(const std::string& csv_path) {
        std::vector<std::pair<double, std::string>> images;
        std::ifstream f(csv_path);
        if (!f.is_open()) return images;

        std::string line;
        std::getline(f, line);
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

    std::vector<Eigen::Matrix4d> loadEuRoCGroundTruth(const std::string& gt_path) {
        std::vector<Eigen::Matrix4d> poses;
        std::ifstream f(gt_path);
        if (!f.is_open()) return poses;

        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto parts = split(line, ',');
            if (parts.size() < 8) continue;

            double tx = std::stod(parts[1]);
            double ty = std::stod(parts[2]);
            double tz = std::stod(parts[3]);
            double qw = std::stod(parts[4]);
            double qx = std::stod(parts[5]);
            double qy = std::stod(parts[6]);
            double qz = std::stod(parts[7]);

            Eigen::Quaterniond q(qw, qx, qy, qz);
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = q.toRotationMatrix();
            T(0,3) = tx; T(1,3) = ty; T(2,3) = tz;
            poses.push_back(T);
        }
        return poses;
    }

    static std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream ss(s);
        while (std::getline(ss, token, delim)) tokens.push_back(token);
        return tokens;
    }

    void printSummary(const std::vector<EuRoCResult>& results) {
        std::cout << "\n--- EuRoC Summary ---\n";
        std::cout << std::left
                  << std::setw(20) << "Sequence"
                  << std::setw(12) << "ATE(m)"
                  << std::setw(10) << "FPS"
                  << std::setw(12) << "DynFrames"
                  << "\n";
        std::cout << std::string(54, '-') << "\n";

        double total_ate = 0.0;
        double total_fps = 0.0;
        int count = 0;
        for (const auto& r : results) {
            std::cout << std::left
                      << std::setw(20) << r.name
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.ate_rmse
                      << std::setw(10) << std::fixed << std::setprecision(1) << r.avg_fps
                      << std::setw(12) << r.dynamic_frames
                      << "\n";
            total_ate += r.ate_rmse;
            total_fps += r.avg_fps;
            count++;
        }
        std::cout << std::string(54, '-') << "\n";
        if (count > 0) {
            std::cout << std::left
                      << std::setw(20) << "Mean"
                      << std::setw(12) << std::fixed << std::setprecision(3) << (total_ate / count)
                      << std::setw(10) << std::fixed << std::setprecision(1) << (total_fps / count)
                      << "\n";
        }
    }

private:
    std::string _dataset_path;
    std::string _vocab_path;
    std::string _settings_path;
    std::string _output_dir;
};

int run_euroc_benchmark(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: semantic_slam_benchmark euroc <dataset_path> <vocab_path> <settings.yaml> [sequence_name] [output_dir]\n";
        std::cerr << "Example: semantic_slam_benchmark euroc /data/EuRoC ./ORBvoc.txt ./config/EuRoC.yaml MH_01_easy ./output\n";
        return 1;
    }

    std::string dataset  = argv[1];
    std::string vocab    = argv[2];
    std::string settings = argv[3];
    std::string seq_name = argc >= 5 ? argv[4] : "";
    std::string output   = argc >= 6 ? argv[5] : "../../output";

    EuRoCBenchmark benchmark(dataset, vocab, settings, output);

    if (!seq_name.empty()) {
        std::string seq_path = dataset + "/" + seq_name;
        if (!fs::exists(seq_path)) {
            std::cerr << "Sequence not found: " << seq_path << "\n";
            return 1;
        }
        std::cout << "Running single sequence: " << seq_name << "\n";
        auto result = benchmark.runSequence(seq_name);
        std::cout << "\n--- Result ---\n"
                  << "  Sequence:    " << result.name << "\n"
                  << "  ATE RMSE:    " << std::fixed << std::setprecision(4) << result.ate_rmse << " m\n"
                  << "  Avg FPS:     " << std::fixed << std::setprecision(1) << result.avg_fps << "\n"
                  << "  Dyn Frames:  " << result.dynamic_frames << "\n";
    } else {
        benchmark.runAll();
    }

    return 0;
}