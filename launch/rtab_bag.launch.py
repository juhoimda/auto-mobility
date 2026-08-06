import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from auto_mobility.launch_common import RTABMAP_ARGS, create_republish_node, create_imu_filter_node

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
        default_value='/camera/camera/depth/image_rect_raw/compressedDepth',
        description='Name of compressed depth topic in bag'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera/camera/depth/image_rect_raw',
        description='Name of raw depth topic'
    )

    # 공통 노드 생성 (Bag 환경: use_sim_time=True)
    republish_compressed_node = create_republish_node(
        use_sim_time=True,
        depth_compressed_topic=LaunchConfiguration('depth_compressed_topic')
    )
    imu_filter_node = create_imu_filter_node(use_sim_time=True)

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir, 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': '/camera/camera/color/camera_info',
            'imu_topic': '/camera/camera/imu/filtered',
            'subscribe_imu': LaunchConfiguration('use_imu'),

            # QoS profile = [0: system default, 1: Reliable, 2: Best Effort]
            'qos_imu': '2',
            'qos_image': '2',
            'qos_depth': '2',
            'qos_camera_info': '2',

            'frame_id': 'camera_link',

            # Synchronization
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.15',
            'topic_queue_size': '30',

            # Frame processing
            'always_process_most_recent_frame': 'false',

            # RTAB-Map options
            'visual_odometry': 'true',
            'use_sim_time': 'true',
            'rviz': 'true',

            # Database
            'rtabmap_viz': 'false',
            'database_path': LaunchConfiguration('database_path'),

            # Additional RTAB-Map parameters
            'rtabmap_args': RTABMAP_ARGS
        }.items()
    )

    return LaunchDescription([
        database_path_arg,
        use_compressed_arg,
        depth_compressed_topic_arg,
        depth_topic_arg,
        use_imu_arg,
        republish_compressed_node,
        imu_filter_node,
        rtabmap_launch
    ])
