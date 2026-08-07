from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Intel RealSense D435i VMware 가상화 최적 런치 파일
    - RGB (640x480@30fps, MJPEG 하드웨어 압축 -> USB 대역폭 90% 절감)
    - Depth (640x480@30fps, Z16)
    - align_depth.enable: False (vCPU 픽셀 정렬 병목 제거)
    - IMU (/camera/camera/imu 통합 ~185Hz, unite_imu_method='copy')
    - FastDDS SHM + SENSOR_DATA QoS 적용
    """
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[{
                'depth_module.depth_profile': '848x480x30',
                'rgb_camera.color_profile': '848x480x30',
                'rgb_camera.color_format': 'RGB8',
                'align_depth.enable': True,   # Align RGB & Depth pixels for exact color mapping
                'enable_infra1': False,
                'enable_infra2': False,
                'depth_module.emitter_enabled': True,  # IR Laser Projector (무늬 없는 흰 벽면 특징점 강제 생성)
                'enable_accel': True,
                'enable_gyro': True,
                'unite_imu_method': 1,  # 1: copy mode (gyro + accel merged into /camera/camera/imu)
                'enable_metadata': False,
                'global_time_enabled': False,
                'initial_reset': False,
                'rgb_camera.auto_exposure_priority': False,
                'color_qos': 'SENSOR_DATA',
                'color_info_qos': 'SENSOR_DATA',
                'depth_qos': 'SENSOR_DATA',
                'depth_info_qos': 'SENSOR_DATA',
                'filters': 'spatial,temporal,hole_filling',
                'spatial_filter.spatial_alpha': 0.5,
                'spatial_filter.spatial_delta': 20,
                'temporal_filter.temporal_alpha': 0.4,
                'temporal_filter.temporal_delta': 20,
                'hole_filling_filter.holes_fill': 1,
            }]
        )
    ])
