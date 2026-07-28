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
                'initial_reset': True,
                'enable_sync': False,
                'enable_infra': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'align_depth': {
                    'enable': False
                },
                'depth_module': {
                    'depth_profile': '640x480x15',
                    'infra_profile': '640x480x15'
                },
                'rgb_camera': {
                    'color_profile': '640x480x15'
                },
                'pointcloud': {
                    'enable': False
                },
                'enable_accel': True,
                'enable_gyro': True,
                'unite_imu_method': 1,
                'gyro_fps': 200,
                'accel_fps': 100
            }]
        )
    ])

