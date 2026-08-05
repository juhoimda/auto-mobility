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
                'depth_module.depth_profile': '640x480x30',
                'rgb_camera.color_profile': '640x480x30',
                'rgb_camera.color_format': 'MJPEG',
                'align_depth.enable': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_accel': True,
                'enable_gyro': True,
                'motion_module.enable_accel': True,
                'motion_module.enable_gyro': True,
                'unite_imu_method': 1,
                'motion_module.unite_imu_method': 1,
                'enable_metadata': False,
                'rgb_camera.auto_exposure_priority': False,
                'color_qos': 'SENSOR_DATA',
                'color_info_qos': 'SENSOR_DATA',
                'depth_qos': 'SENSOR_DATA',
                'depth_info_qos': 'SENSOR_DATA',
                'pointcloud.pointcloud_qos': 'SENSOR_DATA',
            }]
        )
    ])
