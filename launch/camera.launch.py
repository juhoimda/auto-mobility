from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Intel RealSense D435i VMware 가상화 최적 런치 파일
    - RGB (640x480@30fps, RGB8)  [벤치마크 2026-08-10 확정: 30.1Hz / depth 29.1Hz 안정]
    - Depth (640x480@30fps, Z16)
    - 848x480 depth는 19.4Hz로 드랍 (RGB8+Z16=61MB/s > VM USB 한계 ~50MB/s) → 미채택
    - align_depth.enable: False (vCPU 픽셀 정렬 병목 제거, RTAB-Map이 자체 정렬)
    - IMU (/camera/camera/imu 통합, unite_imu_method=1 copy)
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
                'depth_module.depth_profile': '640x480x30',
                'rgb_camera.color_profile': '640x480x30',
                'rgb_camera.color_format': 'RGB8',
                'align_depth.enable': False,   # vCPU 정렬 병목 제거 (RTAB-Map 자체 정렬 사용)
                'enable_infra1': False,
                'enable_infra2': False,
                'depth_module.emitter_enabled': 1,  # IR Laser Projector (1: enabled)
                'enable_accel': True,
                'enable_gyro': True,
                'enable_sync': True,
                'unite_imu_method': 1,  # 1: copy mode (CPU 절감, 벤치마크 검증)
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
