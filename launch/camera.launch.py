from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Intel RealSense D435i 최소 핵심 런치 파일 (RTAB-Map SLAM 표준 640x480)
    - RGB (640x480@15fps, RGB8 - RTAB-Map SLAM 정밀 추적)
    - Depth (640x480@15fps, Z16)
    - IMU (/camera/camera/imu 가속도+자이로 통합 ~185Hz)
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
                'rgb_camera.color_format': 'RGB8',
                'align_depth.enable': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_infra': False,
                'enable_accel': True,
                'enable_gyro': True,
                'unite_imu_method': 1,
                'rgb_camera.auto_exposure_priority': False,
            }]
        )
    ])
