#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<Eigen::Matrix4d> loadTrajectoryTUM(const std::string& path) {
    std::vector<Eigen::Matrix4d> poses;
    std::ifstream f(path);
    if (!f.is_open()) return poses;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double ts, tx, ty, tz, qx, qy, qz, qw;
        if (!(ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = q.toRotationMatrix();
        T(0,3) = tx; T(1,3) = ty; T(2,3) = tz;
        poses.push_back(T);
    }
    return poses;
}

inline double computeATE_RMSE(const std::vector<Eigen::Matrix4d>& est,
                              const std::vector<Eigen::Matrix4d>& gt) {
    size_t n = std::min(est.size(), gt.size());
    if (n == 0) return -1.0;

    Eigen::Vector3d mean_est = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gt  = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < n; ++i) {
        mean_est += est[i].block<3,1>(0,3);
        mean_gt  += gt[i].block<3,1>(0,3);
    }
    mean_est /= n;
    mean_gt  /= n;

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(3, 3);
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3d pe = est[i].block<3,1>(0,3) - mean_est;
        Eigen::Vector3d pg = gt[i].block<3,1>(0,3)  - mean_gt;
        W += pe * pg.transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;
        R = V * svd.matrixU().transpose();
    }
    Eigen::Vector3d t = mean_gt - R * mean_est;

    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3d p_aligned = R * est[i].block<3,1>(0,3) + t;
        Eigen::Vector3d diff = p_aligned - gt[i].block<3,1>(0,3);
        sum_sq += diff.squaredNorm();
    }
    return std::sqrt(sum_sq / n);
}

#endif // BENCHMARK_UTILS_H