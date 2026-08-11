import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from auto_mobility.launch.launch_common import (
    RTABMAP_ARGS,
    RTAB_LIVE_TOPIC_QUEUE_SIZE,
    get_rtabmap_base_args,
    create_republish_node,
    create_imu_filter_node,
    create_cloud_throttle_node,
)
from auto_mobility.config import CAMERA_DEPTH_TOPIC

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
        default_value=CAMERA_DEPTH_TOPIC,
        description=f'Depth topic name (default: {CAMERA_DEPTH_TOPIC})'
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
            **get_rtabmap_base_args(),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'subscribe_imu': LaunchConfiguration('use_imu'),
            # true: odometry가 카메라보다 느려도 최신 프레임만 처리 → 지연/큐 백로그 누적 차단
            # (false는 rosbag 오프라인 재생용. live에서 false면 위상 지연이 계속 쌓임)
            'odom_always_process_most_recent_frame': 'true',
            'topic_queue_size': RTAB_LIVE_TOPIC_QUEUE_SIZE,
            'wait_for_transform': '0.5',
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
