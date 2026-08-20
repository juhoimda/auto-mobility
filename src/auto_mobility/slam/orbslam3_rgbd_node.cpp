#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <chrono>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "System.h"
#include "ImuTypes.h"

class OrbSlam3RgbdNode : public rclcpp::Node
{
public:
    OrbSlam3RgbdNode() : Node("orbslam3_rgbd_node")
    {
        this->declare_parameter<std::string>("vocab_path", "");
        this->declare_parameter<std::string>("config_path", "");
        this->declare_parameter<std::string>("output_trajectory", "ros2_data/trajectories/orbslam3_trajectory.txt");
        this->declare_parameter<std::string>("sensor_mode", "RGBD"); // RGBD or IMU_RGBD
        this->declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
        this->declare_parameter<std::string>("depth_topic", "/camera/camera/depth/image_rect_raw");
        this->declare_parameter<std::string>("imu_topic", "/camera/camera/imu");
        this->declare_parameter<bool>("use_viewer", false);

        std::string vocab_path = this->get_parameter("vocab_path").as_string();
        std::string config_path = this->get_parameter("config_path").as_string();
        output_trajectory_ = this->get_parameter("output_trajectory").as_string();
        sensor_mode_ = this->get_parameter("sensor_mode").as_string();
        std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
        std::string depth_topic = this->get_parameter("depth_topic").as_string();
        std::string imu_topic = this->get_parameter("imu_topic").as_string();
        bool use_viewer = this->get_parameter("use_viewer").as_bool();

        // Standardize sensor mode
        std::string mode_upper = sensor_mode_;
        std::transform(mode_upper.begin(), mode_upper.end(), mode_upper.begin(), ::toupper);
        bool is_inertial = (mode_upper == "IMU_RGBD" || mode_upper == "RGBDI" || mode_upper == "ORB_RGBDI" || mode_upper == "INERTIAL");

        RCLCPP_INFO(this->get_logger(), "Initializing ORB-SLAM3 Node (Mode: %s)...", is_inertial ? "RGB-D-Inertial" : "RGB-D");
        RCLCPP_INFO(this->get_logger(), "Vocab: %s", vocab_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: %s", config_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Output Trajectory: %s", output_trajectory_.c_str());

        ORB_SLAM3::System::eSensor sensor_type = is_inertial ? ORB_SLAM3::System::IMU_RGBD : ORB_SLAM3::System::RGBD;
        is_inertial_ = is_inertial;

        slam_system_ = std::make_unique<ORB_SLAM3::System>(
            vocab_path, config_path, sensor_type, use_viewer);

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/orbslam3/odom", 10);

        rgb_sub_.subscribe(this, rgb_topic, rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, depth_topic, rmw_qos_profile_sensor_data);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(30), rgb_sub_, depth_sub_);
        sync_->registerCallback(
            std::bind(&OrbSlam3RgbdNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        if (is_inertial_)
        {
            // IMU is 200Hz and the single-threaded executor is blocked while
            // TrackRGBD() runs, so a small best-effort queue (depth 5) silently
            // drops most IMU samples -> ORB-SLAM3 inertial preintegration fails
            // and crashes on null pointers. Use a deep RELIABLE queue instead
            // (bag records IMU as RELIABLE/depth 50; rclcpp::QoS default is
            // RELIABLE + KEEP_LAST).
            rclcpp::QoS imu_qos(2000);
            rclcpp::SubscriptionOptions sub_opts;
            imu_cg_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            sub_opts.callback_group = imu_cg_;
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_topic,
                imu_qos,
                std::bind(&OrbSlam3RgbdNode::imuCallback, this, std::placeholders::_1),
                sub_opts);
            RCLCPP_INFO(this->get_logger(), "Subscribed to IMU (%s)", imu_topic.c_str());
        }

        RCLCPP_INFO(this->get_logger(), "Subscribed to RGB (%s) and Depth (%s)", rgb_topic.c_str(), depth_topic.c_str());
    }

    ~OrbSlam3RgbdNode()
    {
        if (slam_system_)
        {
            RCLCPP_INFO(this->get_logger(), "Shutting down ORB-SLAM3 System...");
            slam_system_->Shutdown();
            if (!output_trajectory_.empty())
            {
                RCLCPP_INFO(this->get_logger(), "Saving trajectory to %s ...", output_trajectory_.c_str());
                slam_system_->SaveTrajectoryTUM(output_trajectory_);
            }
        }
    }

private:
    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        if (t <= 0.0)
        {
            t = this->now().seconds();
        }
        imu_buf_.emplace_back(
            static_cast<float>(msg->linear_acceleration.x),
            static_cast<float>(msg->linear_acceleration.y),
            static_cast<float>(msg->linear_acceleration.z),
            static_cast<float>(msg->angular_velocity.x),
            static_cast<float>(msg->angular_velocity.y),
            static_cast<float>(msg->angular_velocity.z),
            t
        );
        total_imu_msgs_++;
    }

    void syncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& msg_rgb,
        const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth)
    {
        cv_bridge::CvImageConstPtr cv_ptr_rgb;
        cv_bridge::CvImageConstPtr cv_ptr_depth;

        try
        {
            cv_ptr_rgb = cv_bridge::toCvShare(msg_rgb, sensor_msgs::image_encodings::RGB8);
        }
        catch (cv_bridge::Exception& e)
        {
            try
            {
                cv_ptr_rgb = cv_bridge::toCvShare(msg_rgb, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception& e2)
            {
                RCLCPP_ERROR(this->get_logger(), "RGB cv_bridge exception: %s", e2.what());
                return;
            }
        }

        try
        {
            cv_ptr_depth = cv_bridge::toCvShare(msg_depth);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Depth cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat im_rgb = cv_ptr_rgb->image;
        cv::Mat im_depth = cv_ptr_depth->image;

        double timestamp = msg_rgb->header.stamp.sec + msg_rgb->header.stamp.nanosec * 1e-9;
        if (timestamp <= 0.0)
        {
            timestamp = this->now().seconds();
        }

        // Republish delivers RGB/DEPTH in two independent streams, so the
        // synchronized pairs can occasionally carry a non-monotonic (backwards)
        // timestamp. ORB-SLAM3 treats a backwards timestamp as "timestamp older
        // than previous frame detected!" and resets/creates new maps, which in
        // IMU mode races between Tracking and LocalMapping and crashes on a
        // dangling mutex. Drop out-of-order frames instead.
        if (timestamp <= last_frame_ts_)
        {
            return;
        }
        last_frame_ts_ = timestamp;

        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        if (is_inertial_)
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            while (!imu_buf_.empty() && imu_buf_.front().t <= timestamp)
            {
                vImuMeas.push_back(imu_buf_.front());
                imu_buf_.pop_front();
            }
            frame_count_++;
            if (frame_count_ % 30 == 0)
            {
                double last_imu_t = imu_buf_.empty() ? 0.0 : imu_buf_.back().t;
                RCLCPP_INFO(this->get_logger(),
                    "[diag] fr=%zu rgb_t=%.6f imu_for_frame=%zu imu_buf=%zu imu_total=%zu",
                    frame_count_, timestamp, vImuMeas.size(), imu_buf_.size(), total_imu_msgs_);
            }
        }

        Sophus::SE3f Tcw = slam_system_->TrackRGBD(im_rgb, im_depth, timestamp, vImuMeas);
        Sophus::SE3f Twc = Tcw.inverse();

        if (odom_pub_->get_subscription_count() > 0)
        {
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header = msg_rgb->header;
            odom_msg.header.frame_id = "odom";
            odom_msg.child_frame_id = "camera_color_optical_frame";

            Eigen::Vector3f t = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();

            odom_msg.pose.pose.position.x = t.x();
            odom_msg.pose.pose.position.y = t.y();
            odom_msg.pose.pose.position.z = t.z();
            odom_msg.pose.pose.orientation.x = q.x();
            odom_msg.pose.pose.orientation.y = q.y();
            odom_msg.pose.pose.orientation.z = q.z();
            odom_msg.pose.pose.orientation.w = q.w();

            odom_pub_->publish(odom_msg);
        }
    }

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::CallbackGroup::SharedPtr imu_cg_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<ORB_SLAM3::System> slam_system_;
    std::string output_trajectory_;
    std::string sensor_mode_;
    bool is_inertial_{false};

    std::mutex imu_mutex_;
    std::deque<ORB_SLAM3::IMU::Point> imu_buf_;
    size_t total_imu_msgs_{0};
    size_t frame_count_{0};
    double last_frame_ts_{0.0};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OrbSlam3RgbdNode>();
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
