import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition

def generate_launch_description():
    rtabmap_launch_dir = get_package_share_directory('rtabmap_launch')
    
    database_path_arg = DeclareLaunchArgument(
        'database_path',
        default_value='./ros2_data/databases/rtabmap_bag.db',
        description='Path to rtabmap.db'
    )

    use_compressed_arg = DeclareLaunchArgument(
        'use_compressed',
        default_value='true',
        description='Whether the bag was recorded with compressed topics'
    )

    use_imu_arg = DeclareLaunchArgument(
        'use_imu',
        default_value='true',
        description='Whether to use IMU (requires imu_filter_madgwick to compute orientation)'
    )

    depth_compressed_topic_arg = DeclareLaunchArgument(
        'depth_compressed_topic',
        default_value='/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
        description='Name of compressed depth topic in bag'
    )

    # 1. RGB & Depth 압축 해제 노드 (Best Effort QoS 지원 및 C++ plugin 오류 해결)
    republish_compressed_node = Node(
        package='auto_mobility',
        executable='republish_compressed.py',
        name='republish_compressed',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'depth_compressed_topic': LaunchConfiguration('depth_compressed_topic')
        }],
        condition=IfCondition(LaunchConfiguration('use_compressed'))
    )

    # 4. IMU Orientation 계산 노드 (imu_filter_madgwick)
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
            'use_sim_time': True,
        }],
        remappings=[
            ('imu/data_raw', '/camera/camera/imu'),
            ('imu/data', '/camera/camera/imu/filtered')
        ],
        condition=IfCondition(LaunchConfiguration('use_imu'))
    )

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir, 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'imu_topic': '/camera/camera/imu/filtered',
            'subscribe_imu': LaunchConfiguration('use_imu'),
            'qos_imu': '2',
            'qos_image': '2',
            'qos_depth': '2',
            'qos_camera_info': '2',
            'frame_id': 'camera_link',
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.15',
            'topic_queue_size': '30',
            'always_process_most_recent_frame': 'false',
            'visual_odometry': 'true',
            'use_sim_time': 'true',
            'rviz': 'true',
            'rtabmap_viz': 'false',
            'database_path': LaunchConfiguration('database_path'),
            'rtabmap_args': '--Vis/MinInliers 10'
        }.items()
    )

    return LaunchDescription([
        database_path_arg,
        use_compressed_arg,
        depth_compressed_topic_arg,
        use_imu_arg,
        republish_compressed_node,
        imu_filter_node,
        rtabmap_launch
    ])
