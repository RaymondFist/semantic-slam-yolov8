#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "System.h"
#include <SemanticSLAM.h>
#include "benchmark_utils.h"
// SemanticSLAM is now integrated inside ORB-SLAM3 (accessed via GetSemanticSLAM())

namespace fs = std::filesystem;

static const std::vector<std::string> KITTI_SEQUENCES = {
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"
};

struct SequenceResult {
    std::string name;
    double ate_rmse;
    double rpe_rmse;
    double avg_fps;
    size_t total_frames;
    size_t dynamic_frames;
    double tracking_lost_ratio;
};

class KITTIBenchmark {
public:
    KITTIBenchmark(const std::string& dataset_path,
                   const std::string& vocab_path,
                   const std::string& settings_path,
                   const std::string& output_dir = "../../output")
        : _dataset_path(dataset_path)
        , _vocab_path(vocab_path)
        , _settings_path(settings_path)
        , _output_dir(output_dir) {}

    void runAll() {
        std::cout << "\n============================\n";
        std::cout << "KITTI Odometry Benchmark\n";
        std::cout << "============================\n\n";

        std::vector<SequenceResult> results;

        for (const auto& seq : KITTI_SEQUENCES) {
            std::string seq_path = _dataset_path + "/" + seq;
            if (!fs::exists(seq_path)) {
                std::cout << "  Seq " << seq << ": SKIP (not found)\n";
                continue;
            }

            std::cout << "  Seq " << seq << ": running... " << std::flush;
            auto result = runSequence(seq);
            results.push_back(result);
            std::cout << "ATE=" << std::fixed << std::setprecision(2)
                      << result.ate_rmse << "m  " << result.avg_fps << "FPS\n";
        }

        printSummary(results);
    }

    SequenceResult runSequence(const std::string& seq) {
        SequenceResult result;
        result.name = seq;

        // NOTE: After patching, ORB-SLAM3 internally creates and uses
        // SemanticSLAM. We don't need a separate instance here.

        ORB_SLAM3::System SLAM(
            _vocab_path,
            _settings_path,
            ORB_SLAM3::System::STEREO,
            true); // enable viewer (xvfb provides virtual display)

        std::string img_left_path  = _dataset_path + "/" + seq + "/image_0";
        std::string img_right_path = _dataset_path + "/" + seq + "/image_1";
        std::string timestamp_path = _dataset_path + "/" + seq + "/times.txt";

        std::vector<double> timestamps = loadTimestamps(timestamp_path);
        if (timestamps.empty()) {
            for (int i = 0; ; ++i) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%s/%06d.png", img_left_path.c_str(), i);
                if (!fs::exists(buf)) break;
                timestamps.push_back(i * 0.1);
            }
        }

        result.total_frames = timestamps.size();
        result.dynamic_frames = 0;

        double total_time = 0.0;
        size_t processed = 0;

        for (size_t i = 0; i < timestamps.size(); ++i) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s/%06zu.png", img_left_path.c_str(), i);
            cv::Mat imLeft = cv::imread(buf, cv::IMREAD_UNCHANGED);
            snprintf(buf, sizeof(buf), "%s/%06zu.png", img_right_path.c_str(), i);
            cv::Mat imRight = cv::imread(buf, cv::IMREAD_UNCHANGED);

            if (imLeft.empty() || imRight.empty()) {
                std::cerr << "Failed to load frame " << i << std::endl;
                break;
            }

            // Semantic detection is handled internally by patched ORB-SLAM3

            auto t0 = std::chrono::high_resolution_clock::now();

            // ORB-SLAM3 tracking
            if (i < 5 || i % 100 == 0) {
                std::cout << "[DEBUG] Frame " << i << "/" << timestamps.size()
                          << " size=" << imLeft.cols << "x" << imLeft.rows
                          << " calling TrackStereo..." << std::flush;
            }
            Sophus::SE3f Tcw = SLAM.TrackStereo(imLeft, imRight, timestamps[i]);
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
        SLAM.SaveTrajectoryTUM(_output_dir + "/kitti_" + seq + "_trajectory.txt");
        SLAM.Shutdown();
        // SemanticSLAM is managed by ORB-SLAM3 internally

        result.avg_fps = processed / total_time;
        result.ate_rmse = computeATEFromFile(seq, _output_dir + "/kitti_" + seq + "_trajectory.txt");

        return result;
    }

    std::vector<double> loadTimestamps(const std::string& path) {
        std::vector<double> stamps;
        std::ifstream f(path);
        if (!f.is_open()) return stamps;
        double t;
        while (f >> t) stamps.push_back(t);
        return stamps;
    }

    double computeATEFromFile(const std::string& seq, const std::string& traj_file) {
        auto gt_poses = loadKITTIGroundTruth(seq);
        if (gt_poses.empty()) return -1.0;
        auto est_poses = loadTrajectoryTUM(traj_file);
        if (est_poses.empty()) return -1.0;
        return computeATE_RMSE(est_poses, gt_poses);
    }

    std::vector<Eigen::Matrix4d> loadKITTIGroundTruth(const std::string& seq) {
        std::vector<Eigen::Matrix4d> poses;
        // Try multiple GT paths: ORB-SLAM3 default, our deploy layout, and absolute
        std::vector<std::string> gt_paths = {
            _dataset_path + "/poses/" + seq + ".txt",
            _dataset_path + "/../dataset/poses/" + seq + ".txt",
            std::string(getenv("HOME") ? getenv("HOME") : "/root") + "/datasets/KITTI/poses/" + seq + ".txt",
        };
        std::ifstream f;
        for (const auto& p : gt_paths) {
            f.open(p);
            if (f.is_open()) break;
        }
        if (!f.is_open()) return poses;

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            double m[12];
            for (int i = 0; i < 12; ++i) {
                if (!(ss >> m[i])) break;
            }
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T(0,0) = m[0]; T(0,1) = m[1]; T(0,2) = m[2];  T(0,3) = m[3];
            T(1,0) = m[4]; T(1,1) = m[5]; T(1,2) = m[6];  T(1,3) = m[7];
            T(2,0) = m[8]; T(2,1) = m[9]; T(2,2) = m[10]; T(2,3) = m[11];
            poses.push_back(T);
        }
        return poses;
    }

    void printSummary(const std::vector<SequenceResult>& results) {
        std::cout << "\n--- KITTI Summary ---\n";
        std::cout << std::left
                  << std::setw(8)  << "Seq"
                  << std::setw(12) << "ATE(m)"
                  << std::setw(10) << "FPS"
                  << std::setw(12) << "DynFrames"
                  << "\n";
        std::cout << std::string(42, '-') << "\n";

        double total_ate = 0.0;
        double total_fps = 0.0;
        for (const auto& r : results) {
            std::cout << std::left
                      << std::setw(8)  << r.name
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.ate_rmse
                      << std::setw(10) << std::fixed << std::setprecision(1) << r.avg_fps
                      << std::setw(12) << r.dynamic_frames
                      << "\n";
            total_ate += r.ate_rmse;
            total_fps += r.avg_fps;
        }
        std::cout << std::string(42, '-') << "\n";
        std::cout << std::left
                  << std::setw(8)  << "Mean"
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << (total_ate / results.size())
                  << std::setw(10) << std::fixed << std::setprecision(1)
                  << (total_fps / results.size())
                  << "\n";
    }

private:
    std::string _dataset_path;
    std::string _vocab_path;
    std::string _settings_path;
    std::string _output_dir;
};

int run_kitti_benchmark(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: semantic_slam_benchmark kitti <dataset_path> <vocab_path> <settings.yaml> [output_dir]\n";
        std::cerr << "Example: semantic_slam_benchmark kitti /data/KITTI ./ORBvoc.txt ./KITTI00.yaml ./output\n";
        return 1;
    }

    std::string dataset  = argv[1];
    std::string vocab    = argv[2];
    std::string settings = argv[3];
    std::string output   = argc >= 5 ? argv[4] : "../../output";

    KITTIBenchmark benchmark(dataset, vocab, settings, output);
    benchmark.runAll();

    return 0;
}