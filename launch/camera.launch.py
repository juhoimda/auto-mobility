from launch import LaunchDescription
from launch_ros.actions import Node
from auto_mobility.config import CAMERA_PARAMS

def generate_launch_description():
    """
    Intel RealSense D435i VMware 가상화 최적 런치 파일
    - RGB (640x480@30fps, RGB8)  [벤치마크 2026-08-10 확정: 30.1Hz / depth 29.1Hz 안정]
    - Depth (640x480@30fps, Z16)
    - 848x480 depth는 19.4Hz로 드랍 (RGB8+Z16=61MB/s > VM USB 한계 ~50MB/s) → 미채택
    - align_depth.enable: False (vCPU 픽셀 정렬 병목 제거, RTAB-Map이 자체 정렬)
    - IMU (/camera/camera/imu 통합, unite_imu_method=1 copy)
    - FastDDS SHM + SENSOR_DATA QoS 적용
    - 카메라 파라미터 단일 소스: src/auto_mobility/config.py 의 CAMERA_PARAMS
    """
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[dict(CAMERA_PARAMS)]
        )
    ])
