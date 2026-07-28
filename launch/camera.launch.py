import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    realsense_launch_dir = get_package_share_directory('realsense2_camera')
    
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(realsense_launch_dir, 'launch', 'rs_launch.py')
            ),
            launch_arguments={
                'enable_sync': 'true',
                'align_depth.enable': 'true',
                'depth_module.depth_profile': '1280x720x15',
                'rgb_camera.color_profile': '1280x720x15',
                'depth_module.profile': '1280x720x15',
                'rgb_camera.profile': '1280x720x15',
                'pointcloud.enable': 'false',
                'enable_accel': 'true',
                'enable_gyro': 'true',
                'unite_imu_method': '1',
                'gyro_fps': '200',
                'accel_fps': '63'
            }.items()
        )
    ])
