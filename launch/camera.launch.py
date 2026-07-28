from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Intel RealSense D435i 최소 핵심 런치 파일
    - RGB (640x480@15fps)
    - Depth (640x480@15fps)
    - IMU (/camera/camera/imu 가속도+자이로 통합 토픽)
    """
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[{
                'depth_module.depth_profile': '640x480x15',
                'rgb_camera.color_profile': '640x480x15',
                'enable_accel': True,
                'enable_gyro': True,
                'unite_imu_method': 1,
            }]
        )
    ])
