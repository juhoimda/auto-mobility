import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from auto_mobility.launch.launch_common import (
    RTABMAP_ARGS,
    ODOM_ARGS,
    RTAB_BAG_TOPIC_QUEUE_SIZE,
    get_rtabmap_base_args,
    create_republish_node,
    create_imu_filter_node,
    create_cloud_throttle_node,
    create_point_cloud_node,
)
from auto_mobility.config import CAMERA_DEPTH_TOPIC, CAMERA_DEPTH_COMPRESSED_TOPIC

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
        default_value=CAMERA_DEPTH_COMPRESSED_TOPIC,
        description='Name of compressed depth topic in bag'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value=CAMERA_DEPTH_TOPIC,
        description='Name of raw depth topic'
    )

    # Windows 원격 카메라 또는 Bag 재생 시 static TF 보장 (camera_link -> optical frames)
    camera_tf_pub_node = Node(
        package='auto_mobility',
        executable='camera_tf_pub.py',
        name='camera_tf_publisher_bag',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # 공통 노드 생성 (Bag 환경: use_sim_time=True)
    republish_compressed_node = create_republish_node(
        use_sim_time=True,
        depth_compressed_topic=LaunchConfiguration('depth_compressed_topic')
    )
    imu_filter_node = create_imu_filter_node(use_sim_time=True)
    cloud_throttle_node = create_cloud_throttle_node(use_sim_time=True, max_rate=2.0)
    point_cloud_node = create_point_cloud_node(use_sim_time=True)

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir, 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            **get_rtabmap_base_args(),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'subscribe_imu': LaunchConfiguration('use_imu'),
            # Frame processing (모든 프레임 순서 처리 → map 밀도 확보)
            'odom_always_process_most_recent_frame': 'false',
            'topic_queue_size': RTAB_BAG_TOPIC_QUEUE_SIZE,
            'use_sim_time': 'true',
            'database_path': LaunchConfiguration('database_path'),
            'rtabmap_args': RTABMAP_ARGS,
            # Odom/OdomF2M 계열은 rtabmap 메인이 선언하지 않아 크래시 → rgbd_odometry에 odom_args로 전달
            'odom_args': ODOM_ARGS
        }.items()
    )

    return LaunchDescription([
        database_path_arg,
        use_compressed_arg,
        depth_compressed_topic_arg,
        depth_topic_arg,
        use_imu_arg,
        camera_tf_pub_node,
        republish_compressed_node,
        imu_filter_node,
        cloud_throttle_node,
        point_cloud_node,
        rtabmap_launch
    ])

