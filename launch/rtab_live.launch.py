import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from auto_mobility.launch_common import RTABMAP_ARGS

def generate_launch_description():
    rtabmap_launch_dir = get_package_share_directory('rtabmap_launch')
    
    database_path_arg = DeclareLaunchArgument(
        'database_path',
        default_value='./ros2_data/databases/rtabmap.db',
        description='Path to rtabmap.db'
    )

    use_compressed_arg = DeclareLaunchArgument(
        'use_compressed',
        default_value='false',
        description='Whether to subscribe to compressed camera topics and decompress locally (Set true only if camera is running remotely)'
    )

    use_imu_arg = DeclareLaunchArgument(
        'use_imu',
        default_value='true',
        description='Whether to use IMU (RealSense D435i IMU enabled & filtered via imu_filter_madgwick)'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera/camera/depth/image_rect_raw',
        description='Depth topic name (e.g. /camera/camera/depth/image_rect_raw or /camera/camera/aligned_depth_to_color/image_raw)'
    )

    # 1. RGB & Depth 압축 해제 노드 (Best Effort QoS 지원, Depth 비손실 16UC1 복원)
    republish_compressed_node = Node(
        package='auto_mobility',
        executable='republish.py',
        name='republish_compressed',
        output='screen',
        parameters=[{
            'use_sim_time': False
        }],
        condition=IfCondition(LaunchConfiguration('use_compressed'))
    )

    # 2. IMU Orientation 계산 노드 (imu_filter_madgwick)
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
            'use_sim_time': False,
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
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': '/camera/camera/color/camera_info',
            'imu_topic': '/camera/camera/imu/filtered',
            'subscribe_imu': LaunchConfiguration('use_imu'),
            'always_process_most_recent_frame': 'false', # false: 프레임을 최대한 순서대로 처리
            
            # QoS profile = [0: system default, 1: Reliable, 2: Best Effort]
            'qos_image': '0',
            'qos_depth': '0',
            'qos_camera_info': '0',
            'qos_imu': '0',

            'frame_id': 'camera_link',

            # Synchronization
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.15',
            'topic_queue_size': '30',

            'visual_odometry': 'true',
            'rviz': 'true',
            'rviz_cfg': os.path.join(get_package_share_directory('auto_mobility'), 'config', 'rtabmap_vmware.rviz'),
            'rtabmap_viz': 'false',
            
            'database_path': LaunchConfiguration('database_path'),
            'rtabmap_args': RTABMAP_ARGS
        }.items()
    )

    return LaunchDescription([
        database_path_arg,
        use_compressed_arg,
        use_imu_arg,
        depth_topic_arg,
        republish_compressed_node,
        imu_filter_node,
        rtabmap_launch
    ])

