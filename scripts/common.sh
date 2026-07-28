#!/bin/bash

set -e

source /opt/ros/$ROS_DISTRO/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKSPACE="${WORKSPACE:-$PROJECT_DIR}"
DATA_DIR="$PROJECT_DIR/ros2_data"

BAG_DIR="$DATA_DIR/bags"
DB_DIR="$DATA_DIR/databases"
POINTCLOUD_DIR="$DATA_DIR/pointclouds"
MESH_DIR="$DATA_DIR/meshes"
ISAAC_DIR="$DATA_DIR/isaac_sim"
LOG_DIR="$DATA_DIR/logs"

RAM_BAG_DIR="/dev/shm/ros2_bags"
STORAGE_FORMAT="mcap"

RGB_TOPIC="/camera/camera/color/image_raw"
RGB_COMPRESSED_TOPIC="/camera/camera/color/image_raw/compressed"
DEPTH_TOPIC="/camera/camera/depth/image_rect_raw"
DEPTH_COMPRESSED_TOPIC="/camera/camera/aligned_depth_to_color/image_raw/compressedDepth"
CAMERA_INFO_TOPIC="/camera/camera/color/camera_info"
IMU_TOPIC="/camera/camera/imu"

mkdir -p "$BAG_DIR" "$DB_DIR" "$POINTCLOUD_DIR" "$MESH_DIR" "$ISAAC_DIR" "$LOG_DIR" "$RAM_BAG_DIR"

if [ -f "$PROJECT_DIR/install/setup.bash" ]; then
    source "$PROJECT_DIR/install/setup.bash"
elif [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    source "$HOME/ros2_ws/install/setup.bash"
fi
