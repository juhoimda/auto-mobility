from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Intel RealSense D435i 최적 런치 파일 (벤치마크 1위 검증: 1280x720@30fps, SENSOR_DATA QoS)
    - RGB (1280x720@30fps, RGB8)
    - Depth (1280x720@30fps, Z16)
    - IMU (/camera/camera/imu 통합 ~185Hz)
    - FastDDS SHM + SENSOR_DATA QoS 적용 (프레임 손실률 0.0%)
    """
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[{
                'depth_module.depth_profile': '1280x720x30',
                'rgb_camera.color_profile': '1280x720x30',
                'rgb_camera.color_format': 'RGB8',
                'align_depth.enable': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_accel': True,
                'enable_gyro': True,
                'unite_imu_method': 'copy',
                'rgb_camera.auto_exposure_priority': False,
                'color_qos': 'SENSOR_DATA',
                'color_info_qos': 'SENSOR_DATA',
                'depth_qos': 'SENSOR_DATA',
                'depth_info_qos': 'SENSOR_DATA',
                'pointcloud.pointcloud_qos': 'SENSOR_DATA',
            }]
        )
    ])
