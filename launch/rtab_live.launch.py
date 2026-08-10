import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from auto_mobility.launch.launch_common import RTABMAP_ARGS, create_republish_node, create_imu_filter_node, create_cloud_throttle_node

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
        description='Depth topic name (e.g. /camera/camera/depth/image_rect_raw)'
    )

    # 공통 노드 생성 (Live 환경: use_sim_time=False)
    republish_compressed_node = create_republish_node(use_sim_time=False)
    imu_filter_node = create_imu_filter_node(use_sim_time=False)
    # RViz 소프트웨어 렌더링 부하 절감용 cloud_map 2Hz 중계 (SLAM에는 영향 없음)
    cloud_throttle_node = create_cloud_throttle_node(use_sim_time=False, max_rate=2.0)

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
            # true: odometry가 카메라보다 느려도 최신 프레임만 처리 → 지연/큐 백로그 누적 차단
            # (false는 rosbag 오프라인 재생용. live에서 false면 위상 지연이 계속 쌓임)
            'odom_always_process_most_recent_frame': 'true',
            
            # QoS profile = [0: system default, 1: Reliable, 2: Best Effort]
            'qos_image': '2',
            'qos_depth': '2',
            'qos_camera_info': '2',
            'qos_imu': '2',

            'frame_id': 'camera_link',

            # Synchronization
            # 0.15s: VO delay(~70ms, always_process_most_recent_frame 특성) 흡수하여
            # rtabmap의 간헐적 "no odometry" 스킵(맵 구멍) 방지
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.15',
            'topic_queue_size': '50',
            'wait_for_transform': '0.5',

            'visual_odometry': 'true',
            'rviz': 'true',
            'rviz_cfg': os.path.join(get_package_share_directory('auto_mobility'), 'config', 'rviz', 'rtabmap_vmware.rviz'),
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
        cloud_throttle_node,
        rtabmap_launch
    ])

