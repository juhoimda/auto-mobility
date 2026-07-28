from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Intel RealSense D435i 초경량 가상머신(VMware) 최적화 런치 파일
    - RGB (424x240@15fps, YUYV - 가상머신 USB 대역폭 부담 75% 감소)
    - Depth (424x240@15fps - D435i 초경량 네이티브 해상도)
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
                'depth_module.depth_profile': '424x240x15',
                'rgb_camera.color_profile': '424x240x15',
                'rgb_camera.color_format': 'YUYV',
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
