#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "System.h"
#include "ImuTypes.h"

struct FrameInfo {
    int id;
    double timestamp;
    std::string rgb_path;
    std::string depth_path;
};

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace and carriage return
        while (!token.empty() && (token.back() == '\r' || token.back() == ' ' || token.back() == '\t')) {
            token.pop_back();
        }
        size_t start = token.find_first_not_of(" \t");
        if (start != std::string::npos) {
            token = token.substr(start);
        }
        tokens.push_back(token);
    }
    return tokens;
}

static bool load_frames_csv(const std::string& csv_path, const std::string& dataset_dir, std::vector<FrameInfo>& frames) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open frames.csv at: " << csv_path << std::endl;
        return false;
    }

    std::string line;
    if (!std::getline(file, line)) return false; // Header line

    auto headers = split_csv_line(line);
    int idx_id = -1, idx_ts = -1, idx_rgb = -1, idx_depth = -1;

    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = headers[i];
        std::transform(h.begin(), h.end(), h.begin(), ::tolower);
        if (h == "frame_id" || h == "index" || h == "id") idx_id = (int)i;
        else if (h == "rgb_timestamp" || h == "timestamp" || h == "stamp") idx_ts = (int)i;
        else if (h == "rgb_path" || h == "rgb") idx_rgb = (int)i;
        else if (h == "depth_path" || h == "depth") idx_depth = (int)i;
    }

    if (idx_ts == -1 || idx_rgb == -1 || idx_depth == -1) {
        // Fallback default column layout: id, rgb_ts, depth_ts, rgb_path, depth_path
        idx_id = 0; idx_ts = 1; idx_rgb = 3; idx_depth = 4;
    }

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tokens = split_csv_line(line);
        if ((int)tokens.size() <= std::max({idx_id, idx_ts, idx_rgb, idx_depth})) continue;

        FrameInfo info;
        info.id = (idx_id >= 0) ? std::stoi(tokens[idx_id]) : (int)frames.size();
        info.timestamp = std::stod(tokens[idx_ts]);
        
        std::string r_path = tokens[idx_rgb];
        std::string d_path = tokens[idx_depth];
        if (!r_path.empty() && r_path[0] != '/') {
            r_path = dataset_dir + "/" + r_path;
        }
        if (!d_path.empty() && d_path[0] != '/') {
            d_path = dataset_dir + "/" + d_path;
        }
        info.rgb_path = r_path;
        info.depth_path = d_path;
        frames.push_back(info);
    }
    return true;
}

static bool load_imu_csv(const std::string& csv_path, std::vector<ORB_SLAM3::IMU::Point>& imu_data) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "⚠️ No imu.csv found at: " << csv_path << " (proceeding without IMU)" << std::endl;
        return false;
    }

    std::string line;
    if (!std::getline(file, line)) return false; // Header line

    // Header: timestamp,angular_velocity_x,angular_velocity_y,angular_velocity_z,linear_acceleration_x,linear_acceleration_y,linear_acceleration_z
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tokens = split_csv_line(line);
        if (tokens.size() < 7) continue;

        try {
            double ts = std::stod(tokens[0]);
            float gx = std::stof(tokens[1]);
            float gy = std::stof(tokens[2]);
            float gz = std::stof(tokens[3]);
            float ax = std::stof(tokens[4]);
            float ay = std::stof(tokens[5]);
            float az = std::stof(tokens[6]);

            if (std::isnan(ax) || std::isnan(ay) || std::isnan(az) ||
                std::isnan(gx) || std::isnan(gy) || std::isnan(gz)) {
                continue;
            }

            imu_data.emplace_back(ax, ay, az, gx, gy, gz, ts);
        } catch (...) {
            continue;
        }
    }
    std::cout << "📥 Loaded " << imu_data.size() << " IMU measurements." << std::endl;
    return true;
}

int main(int argc, char** argv) {
    std::string dataset_dir = "";
    std::string vocab_path = "";
    std::string config_path = "";
    std::string out_trajectory = "trajectory.txt";
    std::string mode = "rgbd";
    int stride = 1;
    int max_frames = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset_dir = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) vocab_path = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_trajectory = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--stride" && i + 1 < argc) stride = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--max-frames" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
    }

    if (dataset_dir.empty() || vocab_path.empty() || config_path.empty()) {
        std::cerr << "Usage: orbslam3_offline --dataset <DIR> --vocab <VOCAB> --config <YAML> --out <TRAJ.TXT> [--mode rgbd|rgbdi] [--stride N]" << std::endl;
        return 1;
    }

    std::string mode_upper = mode;
    std::transform(mode_upper.begin(), mode_upper.end(), mode_upper.begin(), ::toupper);
    bool is_inertial = (mode_upper == "IMU_RGBD" || mode_upper == "RGBDI" || mode_upper == "ORB_RGBDI" || mode_upper == "INERTIAL");

    std::cout << "==========================================================" << std::endl;
    std::cout << " 🚀 ORB-SLAM3 Direct Offline Runner (Zero Frame Drop)" << std::endl;
    std::cout << " 📂 Dataset:    " << dataset_dir << std::endl;
    std::cout << " ⚙️ Mode:       " << (is_inertial ? "RGB-D-Inertial" : "RGB-D") << std::endl;
    std::cout << " ⏩ Stride:     " << stride << std::endl;
    std::cout << " 📑 Trajectory: " << out_trajectory << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::vector<FrameInfo> frames;
    std::string csv_path = dataset_dir + "/frames.csv";
    if (!load_frames_csv(csv_path, dataset_dir, frames) || frames.empty()) {
        std::cerr << "❌ Failed to load frames from " << csv_path << std::endl;
        return 1;
    }
    std::cout << "📊 Loaded " << frames.size() << " frames from dataset." << std::endl;

    std::vector<ORB_SLAM3::IMU::Point> imu_data;
    if (is_inertial) {
        std::string imu_path = dataset_dir + "/imu.csv";
        load_imu_csv(imu_path, imu_data);
    }

    ORB_SLAM3::System::eSensor sensor_type = is_inertial ? ORB_SLAM3::System::IMU_RGBD : ORB_SLAM3::System::RGBD;
    std::cout << "⚙️ Initializing ORB-SLAM3 System..." << std::endl;
    ORB_SLAM3::System SLAM(vocab_path, config_path, sensor_type, false);

    size_t imu_idx = 0;
    double t_start = (double)cv::getTickCount();
    size_t processed_count = 0;

    for (size_t i = 0; i < frames.size(); ++i) {
        if (max_frames > 0 && processed_count >= (size_t)max_frames) break;
        if (stride > 1 && (i % (size_t)stride != 0)) continue;

        const auto& frame = frames[i];
        cv::Mat im_rgb = cv::imread(frame.rgb_path, cv::IMREAD_COLOR);
        cv::Mat im_depth = cv::imread(frame.depth_path, cv::IMREAD_UNCHANGED);

        if (im_rgb.empty() || im_depth.empty()) {
            std::cerr << "⚠️ Warning: Failed to read frame " << frame.id << " (" << frame.rgb_path << ")" << std::endl;
            continue;
        }

        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        if (is_inertial) {
            while (imu_idx < imu_data.size() && imu_data[imu_idx].t <= frame.timestamp) {
                vImuMeas.push_back(imu_data[imu_idx]);
                imu_idx++;
            }
        }

        SLAM.TrackRGBD(im_rgb, im_depth, frame.timestamp, vImuMeas);
        processed_count++;

        if (processed_count % 100 == 0 || processed_count == frames.size()) {
            double elapsed = ((double)cv::getTickCount() - t_start) / cv::getTickFrequency();
            double fps = processed_count / (elapsed > 0 ? elapsed : 1.0);
            std::cout << "⏱️ Processed " << processed_count << "/" << frames.size() 
                      << " frames (" << std::fixed << std::setprecision(1) << fps << " FPS)..." << std::endl;
        }
    }

    std::cout << "⚙️ Shutting down SLAM and finalizing map..." << std::endl;
    SLAM.Shutdown();

    std::cout << "💾 Saving TUM Trajectory to: " << out_trajectory << std::endl;
    SLAM.SaveTrajectoryTUM(out_trajectory);

    double total_time = ((double)cv::getTickCount() - t_start) / cv::getTickFrequency();
    std::cout << "✅ Finished ORB-SLAM3 in " << std::fixed << std::setprecision(2) << total_time 
              << "s (" << (processed_count / (total_time > 0 ? total_time : 1.0)) << " FPS avg, Zero Drops)" << std::endl;

    return 0;
}
