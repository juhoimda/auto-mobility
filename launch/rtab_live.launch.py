import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from auto_mobility.launch.launch_common import (
    RTABMAP_ARGS,
    ODOM_ARGS,
    RTAB_LIVE_TOPIC_QUEUE_SIZE,
    get_rtabmap_base_args,
    create_republish_node,
    create_imu_filter_node,
    create_cloud_throttle_node,
    create_point_cloud_node,
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

    # Windows 원격 카메라 대응 (2026-08-13 추가):
    # 로컬 WSL(usbipd)에선 realsense2_camera_node 가 camera_link→camera_*_optical_frame TF 를
    # 직접 발행하지만, Windows 네이티브 realsense_pub.py 는 TF 를 발행하지 않는다.
    # RTAB-Map 의 wait_for_transform 이 실패하지 않도록 원격 모드에서만 static TF 를 공급한다.
    #   - depth 는 Windows에서 color 로 align 되므로 두 optical frame 을 동일(identity) 처리.
    #   - 실제 카메라 외부 파라미터와 정확히 일치하진 않지만, RTAB-Map SLAM 정확도에는 영향 없음
    #     (camera_link 를 기준으로 odom/map 이 상대 좌표로 구성되기 때문).
    publish_camera_tf_arg = DeclareLaunchArgument(
        'publish_camera_tf',
        default_value='true',
        description='Windows 원격 카메라 모드에서 camera optical frame static TF 발행'
    )

    camera_tf_pub_node = Node(
        package='auto_mobility',
        executable='camera_tf_pub.py',
        name='camera_tf_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_compressed'))
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
            # false: 프레임 드랍 없이 순차 처리 → Odometry 유실 및 'no odometry provided' 방지
            'odom_always_process_most_recent_frame': 'false',
            'topic_queue_size': RTAB_LIVE_TOPIC_QUEUE_SIZE,
            'wait_for_transform': '0.2',
            'database_path': LaunchConfiguration('database_path'),
            'rtabmap_args': RTABMAP_ARGS,
            # Odom/OdomF2M 계열은 rtabmap 메인이 선언하지 않아 크래시 → rgbd_odometry에 odom_args로 전달
            'odom_args': ODOM_ARGS
        }.items()
    )

    return LaunchDescription([
        database_path_arg,
        use_compressed_arg,
        use_imu_arg,
        depth_topic_arg,
        publish_camera_tf_arg,
        camera_tf_pub_node,
        republish_compressed_node,
        imu_filter_node,
        cloud_throttle_node,
        rtabmap_launch
    ])
