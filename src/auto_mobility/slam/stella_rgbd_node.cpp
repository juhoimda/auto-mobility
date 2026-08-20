#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <stella_vslam/system.h>
#include <stella_vslam/config.h>

class StellaVslamRgbdNode : public rclcpp::Node
{
public:
    StellaVslamRgbdNode() : Node("stella_vslam_rgbd_node")
    {
        this->declare_parameter<std::string>("vocab_path", "");
        this->declare_parameter<std::string>("config_path", "");
        this->declare_parameter<std::string>("output_trajectory", "ros2_data/trajectories/stella_trajectory.txt");
        this->declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
        this->declare_parameter<std::string>("depth_topic", "/camera/camera/depth/image_rect_raw");

        std::string vocab_path = this->get_parameter("vocab_path").as_string();
        std::string config_path = this->get_parameter("config_path").as_string();
        output_trajectory_ = this->get_parameter("output_trajectory").as_string();
        std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
        std::string depth_topic = this->get_parameter("depth_topic").as_string();

        RCLCPP_INFO(this->get_logger(), "Initializing stella_vslam RGB-D Node...");
        RCLCPP_INFO(this->get_logger(), "Vocab: %s", vocab_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: %s", config_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Output Trajectory: %s", output_trajectory_.c_str());

        auto cfg = std::make_shared<stella_vslam::config>(config_path);
        slam_system_ = std::make_unique<stella_vslam::system>(cfg, vocab_path);
        slam_system_->startup();

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/stella_vslam/odom", 10);

        rgb_sub_.subscribe(this, rgb_topic, rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, depth_topic, rmw_qos_profile_sensor_data);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(30), rgb_sub_, depth_sub_);
        sync_->registerCallback(
            std::bind(&StellaVslamRgbdNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Subscribed to RGB (%s) and Depth (%s)", rgb_topic.c_str(), depth_topic.c_str());
    }

    ~StellaVslamRgbdNode()
    {
        if (slam_system_)
        {
            RCLCPP_INFO(this->get_logger(), "Shutting down stella_vslam System...");
            slam_system_->shutdown();
            if (!output_trajectory_.empty())
            {
                RCLCPP_INFO(this->get_logger(), "Saving trajectory to %s (TUM format)...", output_trajectory_.c_str());
                slam_system_->save_frame_trajectory(output_trajectory_, "TUM");
            }
        }
    }

private:
    void syncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& msg_rgb,
        const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth)
    {
        cv_bridge::CvImageConstPtr cv_ptr_rgb;
        cv_bridge::CvImageConstPtr cv_ptr_depth;

        try
        {
            cv_ptr_rgb = cv_bridge::toCvShare(msg_rgb, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            try
            {
                cv_ptr_rgb = cv_bridge::toCvShare(msg_rgb, sensor_msgs::image_encodings::RGB8);
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

        if (timestamp <= last_frame_ts_)
        {
            return;
        }
        last_frame_ts_ = timestamp;

        try
        {
            auto cam_pose_wc = slam_system_->feed_RGBD_frame(im_rgb, im_depth, timestamp);

            if (cam_pose_wc && odom_pub_->get_subscription_count() > 0)
            {
                nav_msgs::msg::Odometry odom_msg;
                odom_msg.header = msg_rgb->header;
                odom_msg.header.frame_id = "odom";
                odom_msg.child_frame_id = "camera_color_optical_frame";

                const auto& pose = *cam_pose_wc;
                Eigen::Matrix3d R = pose.block<3, 3>(0, 0).cast<double>();
                Eigen::Vector3d t = pose.block<3, 1>(0, 3).cast<double>();
                Eigen::Quaterniond q(R);

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
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "feed_RGBD_frame exception: %s", e.what());
        }
    }

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<stella_vslam::system> slam_system_;
    std::string output_trajectory_;
    double last_frame_ts_{0.0};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StellaVslamRgbdNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
