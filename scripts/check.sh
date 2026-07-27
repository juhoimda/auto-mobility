#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

check_topic() {
    if ros2 topic list | grep -qx "$1"; then
        echo "[O] $1"
    else
        echo "[X] $1"
    fi
}

echo "=== ROS2 Topic Health Check ==="
check_topic "$RGB_TOPIC"
check_topic "$DEPTH_TOPIC"
check_topic "$CAMERA_INFO_TOPIC"
check_topic "$POINTS_TOPIC"
check_topic "$IMU_TOPIC"
check_topic "/tf"
check_topic "/tf_static"
