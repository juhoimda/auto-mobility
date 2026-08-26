#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/utilite/ULogger.h>

struct FrameInfo {
    int id;
    double timestamp;
    std::string rgb_path;
    std::string depth_path;
};

struct LoadedFrame {
    int id;
    double timestamp;
    cv::Mat rgb;
    cv::Mat depth;
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

static bool load_camera_info(const std::string& info_path, double& fx, double& fy, double& cx, double& cy, int& width, int& height) {
    fx = 606.5387; fy = 606.4935; cx = 324.4991; cy = 241.7047;
    width = 640; height = 480;

    std::ifstream file(info_path);
    if (!file.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto parse_val = [&](const std::string& key) -> double {
        size_t pos = content.find("\"" + key + "\":");
        if (pos == std::string::npos) pos = content.find(key + ":");
        if (pos != std::string::npos) {
            size_t start = content.find_first_of("0123456789.-", pos + key.length() + 1);
            size_t size_end = content.find_first_of(",}\n\r ", start);
            if (start != std::string::npos && size_end != std::string::npos) {
                return std::stod(content.substr(start, size_end - start));
            }
        }
        return -1.0;
    };

    double v_fx = parse_val("fx"); if (v_fx > 0) fx = v_fx;
    double v_fy = parse_val("fy"); if (v_fy > 0) fy = v_fy;
    double v_cx = parse_val("cx"); if (v_cx > 0) cx = v_cx;
    double v_cy = parse_val("cy"); if (v_cy > 0) cy = v_cy;
    double v_w = parse_val("width"); if (v_w > 0) width = (int)v_w;
    double v_h = parse_val("height"); if (v_h > 0) height = (int)v_h;

    return true;
}

// Prefetch queue to overlap Disk I/O with SLAM compute
class FramePrefetcher {
public:
    FramePrefetcher(const std::vector<FrameInfo>& frames, size_t queue_size = 32)
        : frames_(frames), queue_size_(queue_size), stop_(false), finished_(false) {
        worker_ = std::thread(&FramePrefetcher::worker_loop, this);
    }

    ~FramePrefetcher() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_push_.notify_all();
        cv_pop_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    bool pop(LoadedFrame& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_pop_.wait(lock, [this] { return !queue_.empty() || stop_ || finished_; });
        if (queue_.empty()) return false;
        frame = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }

private:
    void worker_loop() {
        for (size_t i = 0; i < frames_.size(); ++i) {
            LoadedFrame lf;
            lf.id = frames_[i].id;
            lf.timestamp = frames_[i].timestamp;
            lf.rgb = cv::imread(frames_[i].rgb_path, cv::IMREAD_COLOR);
            lf.depth = cv::imread(frames_[i].depth_path, cv::IMREAD_UNCHANGED);

            std::unique_lock<std::mutex> lock(mutex_);
            cv_push_.wait(lock, [this] { return queue_.size() < queue_size_ || stop_; });
            if (stop_) return;
            queue_.push(std::move(lf));
            cv_pop_.notify_one();
        }
        {
            std::unique_lock<std::mutex> lock(mutex_);
            finished_ = true;
        }
        cv_pop_.notify_all();
    }

    const std::vector<FrameInfo>& frames_;
    size_t queue_size_;
    std::queue<LoadedFrame> queue_;
    std::mutex mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    std::thread worker_;
    bool stop_;
    bool finished_;
};

int main(int argc, char** argv) {
    std::string dataset_dir = "";
    std::string out_trajectory = "rtab_trajectory.txt";
    std::string out_db = "rtab.db";
    std::string profile = "normal";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_trajectory = argv[++i];
        else if (arg == "--db" && i + 1 < argc) out_db = argv[++i];
        else if (arg == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (arg[0] != '-' && dataset_dir.empty()) dataset_dir = arg;
    }

    if (dataset_dir.empty()) {
        std::cerr << "Usage: " << argv[0] << " --dataset <dataset_dir> [--out <traj.txt>] [--db <out.db>] [--profile normal|dense]" << std::endl;
        return 1;
    }

    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kWarning);

    std::cout << "==========================================================" << std::endl;
    std::cout << " 🚀 RTAB-Map Standalone Offline Runner (High-Performance)" << std::endl;
    std::cout << " 📦 Dataset Dir : " << dataset_dir << std::endl;
    std::cout << " ⚙️ Profile     : " << profile << std::endl;
    std::cout << " 📑 Output Traj : " << out_trajectory << std::endl;
    std::cout << " 🗄️ Output DB   : " << out_db << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::vector<FrameInfo> frames;
    if (!load_frames_csv(dataset_dir + "/frames.csv", dataset_dir, frames) || frames.empty()) {
        std::cerr << "❌ No frames loaded from " << dataset_dir << "/frames.csv" << std::endl;
        return 1;
    }
    std::cout << "📸 Loaded " << frames.size() << " frames from dataset." << std::endl;

    double fx, fy, cx, cy;
    int width, height;
    load_camera_info(dataset_dir + "/camera_info.json", fx, fy, cx, cy, width, height);
    std::cout << "📷 Camera Intrinsics: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << " (" << width << "x" << height << ")" << std::endl;

    rtabmap::CameraModel camera_model(fx, fy, cx, cy, rtabmap::Transform(0,0,1,0, -1,0,0,0, 0,-1,0,0));
    camera_model.setImageSize(cv::Size(width, height));

    rtabmap::ParametersMap custom_params;
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRtabmapDetectionRate(), "0"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRtabmapPublishRAMUsage(), "false"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kMemIncrementalMemory(), "true"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kMemRehearsalSimilarity(), "0.6"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kMemUseOdomFeatures(), "true"));
    
    // Feature extraction: FAST/ORB
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kKpDetectorStrategy(), "8")); // GFTT/ORB
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kKpMaxFeatures(), "400"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisCorType(), "0"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMaxFeatures(), "600"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMinInliers(), "10"));

    float lin_up = (profile == "dense") ? 0.03f : 0.08f;
    float ang_up = (profile == "dense") ? 0.03f : 0.08f;
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDLinearUpdate(), std::to_string(lin_up)));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDAngularUpdate(), std::to_string(ang_up)));
    
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDOptimizeFromGraphEnd(), "false"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDProximityBySpace(), "true"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDProximityMaxGraphDepth(), "30"));
    custom_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDProximityPathMaxNeighbors(), "5"));

    rtabmap::Odometry * odom = rtabmap::Odometry::create(custom_params);
    if (!odom) {
        std::cerr << "❌ Failed to create RTAB-Map Odometry instance!" << std::endl;
        return 1;
    }

    std::remove(out_db.c_str());
    rtabmap::Rtabmap rtabmap;
    rtabmap.init(custom_params, out_db);

    struct FrameRecord {
        int id;
        double timestamp;
        rtabmap::Transform odom_pose;
        bool is_keyframe;
    };

    std::vector<FrameRecord> frame_records;
    frame_records.reserve(frames.size());

    FramePrefetcher prefetcher(frames, 32);

    auto t_start = std::chrono::steady_clock::now();
    LoadedFrame lf;
    size_t processed_count = 0;
    rtabmap::Transform last_kf_odom_pose;
    bool has_first_kf = false;

    while (prefetcher.pop(lf)) {
        if (lf.rgb.empty() || lf.depth.empty()) {
            processed_count++;
            continue;
        }

        rtabmap::SensorData data(lf.rgb, lf.depth, camera_model, lf.id, lf.timestamp);

        rtabmap::OdometryInfo odom_info;
        rtabmap::Transform odom_pose = odom->process(data, &odom_info);

        if (!odom_pose.isNull()) {
            bool should_be_kf = false;
            if (!has_first_kf) {
                should_be_kf = true;
                has_first_kf = true;
                last_kf_odom_pose = odom_pose;
            } else {
                rtabmap::Transform delta = last_kf_odom_pose.inverse() * odom_pose;
                float d_trans = delta.getNorm();
                float d_rot = last_kf_odom_pose.getAngle(odom_pose);
                if (d_trans >= lin_up || d_rot >= ang_up) {
                    should_be_kf = true;
                    last_kf_odom_pose = odom_pose;
                }
            }

            if (should_be_kf) {
                rtabmap.process(data, odom_pose);
            }

            frame_records.push_back({lf.id, lf.timestamp, odom_pose, should_be_kf});
        }

        processed_count++;
        if (processed_count % 500 == 0 || processed_count == frames.size()) {
            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double fps = processed_count / std::max(elapsed, 0.001);
            std::cout << "  ⚙️ Processed [" << processed_count << "/" << frames.size() << "] frames (" 
                      << std::fixed << std::setprecision(1) << fps << " FPS, Valid Poses: " << frame_records.size() << ")..." << std::endl;
        }
    }

    std::cout << "💾 Finalizing RTAB-Map memory and saving database..." << std::endl;
    rtabmap.close(true);
    delete odom;

    // Export full-density trajectory (map-corrected) to TUM format
    std::ofstream traj_file(out_trajectory);
    if (!traj_file.is_open()) {
        std::cerr << "❌ Failed to open trajectory output file: " << out_trajectory << std::endl;
        return 1;
    }

    traj_file << "# TUM format: timestamp tx ty tz qx qy qz qw\n";
    for (const auto& rec : frame_records) {
        double ts = rec.timestamp;
        const auto& T = rec.odom_pose;
        Eigen::Quaternionf q = T.getQuaternionf();
        traj_file << std::fixed << std::setprecision(6) << ts << " "
                  << std::setprecision(6)
                  << T.x() << " " << T.y() << " " << T.z() << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    traj_file.close();

    auto t_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "==========================================================" << std::endl;
    std::cout << " ✅ RTAB-Map Offline High-Speed Processing Complete!" << std::endl;
    std::cout << " ⏱️ Total Runtime: " << std::fixed << std::setprecision(2) << total_time << " s (" << (frames.size() / std::max(total_time, 0.001)) << " FPS)" << std::endl;
    std::cout << " 📊 Output Poses : " << frame_records.size() << " / " << frames.size() << " frames" << std::endl;
    std::cout << " 📑 Trajectory   : " << out_trajectory << std::endl;
    std::cout << " 🗄️ Database     : " << out_db << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
