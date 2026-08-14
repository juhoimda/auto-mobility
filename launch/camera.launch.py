from launch import LaunchDescription
from launch_ros.actions import Node
from auto_mobility.config import CAMERA_PARAMS, validate_camera_params

def generate_launch_description():
    """
    Intel RealSense D435i 최적 런치 파일 (WSL2 기준, 2026-08-12 실측)
    - RGB (640x480@30fps, RGB8) + Depth (640x480@30fps, Z16) — WSL2 USB 패스스루에서 30fps 안정
    - 848x480 이상은 USB 패스스루에서 프레임 손상 발생 → 640x480 고정
    - align_depth.enable: False (CPU 정렬 병목 제거, RTAB-Map이 자체 정렬)
    - QoS 파라미터(color_qos 등)는 4.58.3에서 FPS 급락 유발(실측) → 기본 RELIABLE 사용
    - IMU: udev 룰 적용됨. 단 WSL2에서는 "No HID info" 이슈로 별도 대응 필요 (진행 중)
    - 카메라 파라미터 단일 소스: src/auto_mobility/config.py 의 CAMERA_PARAMS
    """
    issues = validate_camera_params(CAMERA_PARAMS)
    if issues:
        print("\n".join(["[camera.launch] 카메라 파라미터 문제:"] + ["  - " + i for i in issues]))
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
