import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rtabmap_launch_dir = get_package_share_directory('rtabmap_launch')
    
    database_path_arg = DeclareLaunchArgument(
        'database_path',
        default_value='./ros2_data/databases/rtabmap_bag.db',
        description='Path to rtabmap.db'
    )

    rgb_transport_arg = DeclareLaunchArgument(
        'rgb_transport',
        default_value='compressed',
        description='Transport type for RGB image (e.g. raw, compressed)'
    )

    depth_transport_arg = DeclareLaunchArgument(
        'depth_transport',
        default_value='compressedDepth',
        description='Transport type for Depth image (e.g. raw, compressedDepth)'
    )

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir, 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'rgb_transport': LaunchConfiguration('rgb_transport'),
            'depth_transport': LaunchConfiguration('depth_transport'),
            'frame_id': 'camera_link',
            'approx_sync': 'true',
            'visual_odometry': 'true',
            'use_sim_time': 'true',
            'rviz': 'true',
            'rtabmap_viz': 'true',
            'database_path': LaunchConfiguration('database_path')
        }.items()
    )

    return LaunchDescription([
        database_path_arg,
        rgb_transport_arg,
        depth_transport_arg,
        rtabmap_launch
    ])
