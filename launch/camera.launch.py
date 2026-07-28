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
            ros_arguments=[
                '-p', 'enable_sync:=false',
                '-p', 'align_depth.enable:=true',
                '-p', 'depth_module.profile:=640x480x15',
                '-p', 'depth_module.depth_profile:=640x480x15',
                '-p', 'rgb_camera.profile:=640x480x15',
                '-p', 'rgb_camera.color_profile:=640x480x15',
                '-p', 'pointcloud.enable:=false',
                '-p', 'enable_accel:=true',
                '-p', 'enable_gyro:=true',
                '-p', 'unite_imu_method:=1',
                '-p', 'gyro_fps:=200',
                '-p', 'accel_fps:=100'
            ]
        )
    ])

