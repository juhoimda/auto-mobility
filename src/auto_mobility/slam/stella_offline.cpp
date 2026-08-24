#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <stella_vslam/system.h>
#include <stella_vslam/config.h>

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
    if (!std::getline(file, line)) return false;

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

int main(int argc, char** argv) {
    std::string dataset_dir = "";
    std::string vocab_path = "";
    std::string config_path = "";
    std::string out_trajectory = "trajectory.txt";
    int stride = 1;
    int max_frames = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset_dir = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) vocab_path = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_trajectory = argv[++i];
        else if (arg == "--stride" && i + 1 < argc) stride = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--max-frames" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
    }

    if (dataset_dir.empty() || vocab_path.empty() || config_path.empty()) {
        std::cerr << "Usage: stella_offline --dataset <DIR> --vocab <FBOW> --config <YAML> --out <TRAJ.TXT> [--stride N]" << std::endl;
        return 1;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << " 🚀 stella_vslam Direct Offline Runner (Zero Frame Drop)" << std::endl;
    std::cout << " 📂 Dataset:    " << dataset_dir << std::endl;
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

    auto cfg = std::make_shared<stella_vslam::config>(config_path);
    auto SLAM = std::make_unique<stella_vslam::system>(cfg, vocab_path);

    std::cout << "⚙️ Starting stella_vslam System..." << std::endl;
    SLAM->startup();

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

        SLAM->feed_RGBD_frame(im_rgb, im_depth, frame.timestamp);
        processed_count++;

        if (processed_count % 100 == 0 || processed_count == frames.size()) {
            double elapsed = ((double)cv::getTickCount() - t_start) / cv::getTickFrequency();
            double fps = processed_count / (elapsed > 0 ? elapsed : 1.0);
            std::cout << "⏱️ Processed " << processed_count << "/" << frames.size() 
                      << " frames (" << std::fixed << std::setprecision(1) << fps << " FPS)..." << std::endl;
        }
    }

    std::cout << "⚙️ Shutting down stella_vslam System..." << std::endl;
    SLAM->shutdown();

    std::cout << "💾 Saving TUM Trajectory to: " << out_trajectory << std::endl;
    SLAM->save_frame_trajectory(out_trajectory, "TUM");

    double total_time = ((double)cv::getTickCount() - t_start) / cv::getTickFrequency();
    std::cout << "✅ Finished stella_vslam in " << std::fixed << std::setprecision(2) << total_time 
              << "s (" << (processed_count / (total_time > 0 ? total_time : 1.0)) << " FPS avg, Zero Drops)" << std::endl;

    return 0;
}
