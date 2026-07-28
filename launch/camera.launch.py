from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    depth_profile_arg = DeclareLaunchArgument('depth_profile', default_value='640x480x15')
    rgb_profile_arg = DeclareLaunchArgument('rgb_profile', default_value='640x480x15')

    return LaunchDescription([
        depth_profile_arg,
        rgb_profile_arg,
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[{
                'enable_sync': False,
                'align_depth.enable': True,
                'depth_module.profile': LaunchConfiguration('depth_profile'),
                'depth_module.depth_profile': LaunchConfiguration('depth_profile'),
                'rgb_camera.profile': LaunchConfiguration('rgb_profile'),
                'rgb_camera.color_profile': LaunchConfiguration('rgb_profile'),
                'rgb_camera.color.profile': LaunchConfiguration('rgb_profile'),
                'pointcloud.enable': False,
                'enable_accel': True,
                'enable_gyro': True,
                'unite_imu_method': 1,
                'gyro_fps': 200,
                'accel_fps': 100
            }]
        )
    ])

