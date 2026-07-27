#!/bin/bash

set -e

source /opt/ros/$ROS_DISTRO/setup.bash

WORKSPACE="$HOME/ros2_ws"
DATA_DIR="$HOME/ros2_data"

BAG_DIR="$DATA_DIR/bags"
DB_DIR="$DATA_DIR/databases"
POINTCLOUD_DIR="$DATA_DIR/pointclouds"
MESH_DIR="$DATA_DIR/meshes"
ISAAC_DIR="$DATA_DIR/isaac_sim"
LOG_DIR="$DATA_DIR/logs"

RGB_TOPIC="/camera/camera/color/image_raw"
DEPTH_TOPIC="/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC="/camera/camera/color/camera_info"
POINTS_TOPIC="/camera/camera/depth/color/points"
IMU_TOPIC="/camera/camera/imu"

mkdir -p "$BAG_DIR" "$DB_DIR" "$POINTCLOUD_DIR" "$MESH_DIR" "$ISAAC_DIR" "$LOG_DIR"

if [ -f "$WORKSPACE/install/setup.bash" ]; then
    source "$WORKSPACE/install/setup.bash"
fi
