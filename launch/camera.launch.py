import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[{
                'enable_sync': True,
                'align_depth.enable': True,
                'depth_module.depth_profile': '1280x720x30',
                'rgb_camera.color_profile': '1280x720x30',
                'pointcloud.enable': False,
                'enable_accel': True,
                'enable_gyro': True,
                'unite_imu_method': 1,
                'gyro_fps': 200,
                'accel_fps': 63
            }]
        )
    ])

