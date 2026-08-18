#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "System.h"

class OrbSlam3RgbdNode : public rclcpp::Node
{
public:
    OrbSlam3RgbdNode() : Node("orbslam3_rgbd_node")
    {
        this->declare_parameter<std::string>("vocab_path", "");
        this->declare_parameter<std::string>("config_path", "");
        this->declare_parameter<std::string>("output_trajectory", "ros2_data/trajectories/orbslam3_trajectory.txt");
        this->declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
        this->declare_parameter<std::string>("depth_topic", "/camera/camera/depth/image_rect_raw");
        this->declare_parameter<bool>("use_viewer", false);

        std::string vocab_path = this->get_parameter("vocab_path").as_string();
        std::string config_path = this->get_parameter("config_path").as_string();
        output_trajectory_ = this->get_parameter("output_trajectory").as_string();
        std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
        std::string depth_topic = this->get_parameter("depth_topic").as_string();
        bool use_viewer = this->get_parameter("use_viewer").as_bool();

        RCLCPP_INFO(this->get_logger(), "Initializing ORB-SLAM3 RGB-D Node...");
        RCLCPP_INFO(this->get_logger(), "Vocab: %s", vocab_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Config: %s", config_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Output Trajectory: %s", output_trajectory_.c_str());

        slam_system_ = std::make_unique<ORB_SLAM3::System>(
            vocab_path, config_path, ORB_SLAM3::System::RGBD, use_viewer);

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/orbslam3/odom", 10);

        rgb_sub_.subscribe(this, rgb_topic, rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, depth_topic, rmw_qos_profile_sensor_data);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(30), rgb_sub_, depth_sub_);
        sync_->registerCallback(
            std::bind(&OrbSlam3RgbdNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

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

        Sophus::SE3f Tcw = slam_system_->TrackRGBD(im_rgb, im_depth, timestamp);
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

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<ORB_SLAM3::System> slam_system_;
    std::string output_trajectory_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OrbSlam3RgbdNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
