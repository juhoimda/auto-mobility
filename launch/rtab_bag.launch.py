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

    # 압축 RGB 이미지 자동 해제 노드 (use_compressed가 true일 때만 실행)
    republish_rgb_node = Node(
        package='image_transport',
        executable='republish',
        name='republish_rgb',
        arguments=['compressed', 'raw'],
        remappings=[
            ('in/compressed', '/camera/camera/color/image_raw/compressed'),
            ('out', '/camera/camera/color/image_raw')
        ],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_compressed'))
    )

    # 압축 Depth 이미지 자동 해제 노드 (/camera/camera/aligned_depth_to_color/image_raw/compressedDepth)
    republish_depth_node = Node(
        package='image_transport',
        executable='republish',
        name='republish_depth',
        arguments=['compressedDepth', 'raw'],
        remappings=[
            ('in/compressedDepth', '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth'),
            ('out', '/camera/camera/depth/image_rect_raw')
        ],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_compressed'))
    )

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir, 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/depth/image_rect_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'frame_id': 'camera_link',
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.15',
            'topic_queue_size': '30',
            'qos_image': '1',
            'qos_depth': '1',
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
        republish_rgb_node,
        republish_depth_node,
        rtabmap_launch
    ])
